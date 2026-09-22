//! Exercise static MoRI routing through real worker HTTP endpoints, without ZMQ discovery.
mod common;

use axum::{
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::{sync::Notify, task::JoinHandle};
use vllm_router_rs::{
    config::{KvConnector, PolicyConfig, RouterConfig, RoutingMode},
    protocols::spec::ChatCompletionRequest,
    routers::RouterFactory,
};

#[derive(Default)]
struct Exchange {
    requests: Mutex<Vec<(String, HeaderMap, Value)>>,
    prefill_finished: Mutex<bool>,
    decode_started: Notify,
}

struct Worker {
    url: String,
    task: JoinHandle<()>,
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn worker(role: &str, mode: &str, dp_size: usize, exchange: Arc<Exchange>) -> Worker {
    let metadata = json!({
        "type": role,
        // Static configuration, not this advertised URL, must determine the HTTP destination.
        "http_address": "unreachable.invalid:1234",
        "zmq_address": format!("host:{role},handshake:6301,notify:61001"),
        "tp_size": if role == "P" { 2 } else { 4 },
        "dp_size": dp_size,
        "transfer_mode": mode,
    });
    let role = role.to_string();
    let mode = mode.to_string();
    let app = Router::new()
        .route("/health", get(|| async { StatusCode::OK }))
        .route(
            "/v1/moriio/metadata",
            get(move |headers: HeaderMap| async move {
                if headers.get("authorization").and_then(|v| v.to_str().ok())
                    != Some("Bearer test-key")
                {
                    return StatusCode::UNAUTHORIZED.into_response();
                }
                Json(metadata).into_response()
            }),
        )
        .route(
            "/v1/chat/completions",
            post(move |headers: HeaderMap, Json(body): Json<Value>| {
                let exchange = exchange.clone();
                let role = role.clone();
                let mode = mode.clone();
                async move {
                    exchange
                        .requests
                        .lock()
                        .unwrap()
                        .push((role.clone(), headers, body));
                    if role == "P" {
                        if mode == "WRITE" {
                            // A sequential router deadlocks here; only concurrent dispatch succeeds.
                            exchange.decode_started.notified().await;
                        }
                        *exchange.prefill_finished.lock().unwrap() = true;
                        Json(json!({
                            "choices": [],
                            "kv_transfer_params": {"remote_engine_id": "prefill-result"}
                        }))
                    } else {
                        if mode == "READ" {
                            assert!(*exchange.prefill_finished.lock().unwrap());
                        }
                        exchange.decode_started.notify_one();
                        Json(json!({"choices": [{"message": {"content": "done"}}]}))
                    }
                }
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    Worker { url, task }
}

fn config(prefill: &Worker, decode: &Worker) -> RouterConfig {
    RouterConfig {
        mode: RoutingMode::VllmPrefillDecode {
            prefill_urls: vec![(prefill.url.clone(), None)],
            decode_urls: vec![decode.url.clone()],
            prefill_policy: None,
            decode_policy: None,
            discovery_address: None,
        },
        kv_connector: KvConnector::MoriIO,
        policy: PolicyConfig::Random,
        api_key: Some("test-key".to_string()),
        worker_startup_timeout_secs: 2,
        worker_startup_check_interval_secs: 1,
        disable_retries: true,
        ..RouterConfig::default()
    }
}

async fn check_dispatch(mode: &str) {
    let exchange = Arc::new(Exchange::default());
    let prefill = worker("P", mode, 1, exchange.clone()).await;
    let decode = worker("D", mode, 1, exchange.clone()).await;
    let context = common::create_test_context(config(&prefill, &decode));
    let router = RouterFactory::create_router(&context).await.unwrap();
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test", "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 10, "stream": false
    }))
    .unwrap();
    let response = tokio::time::timeout(
        Duration::from_secs(3),
        router.route_chat(None, &request, None),
    )
    .await
    .expect("MoRI dispatch stalled; WRITE must send decode before prefill completes");
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let requests = exchange.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    let (_, p_headers, p_body) = requests.iter().find(|(role, _, _)| role == "P").unwrap();
    let (_, d_headers, d_body) = requests.iter().find(|(role, _, _)| role == "D").unwrap();
    assert_eq!(p_headers["authorization"], "Bearer test-key");
    assert_eq!(d_headers["authorization"], "Bearer test-key");
    let request_id = p_headers["x-request-id"].to_str().unwrap();
    assert_eq!(request_id, d_headers["x-request-id"].to_str().unwrap());
    assert!(request_id.contains("host:P,handshake:6301,notify:61001"));
    assert!(request_id.contains("host:D,handshake:6301,notify:61001"));
    assert_eq!(p_body["max_tokens"], 1);
    assert_eq!(d_body["max_tokens"], 10);
    if mode == "READ" {
        assert_eq!(requests[0].0, "P");
        assert_eq!(
            d_body["kv_transfer_params"]["remote_engine_id"],
            "prefill-result"
        );
    } else {
        assert_eq!(p_body["kv_transfer_params"]["remote_tp_size"], 4);
        assert_eq!(d_body["kv_transfer_params"]["remote_tp_size"], 2);
        assert_eq!(
            p_body["kv_transfer_params"]["transfer_id"],
            d_body["kv_transfer_params"]["transfer_id"]
        );
    }
}

#[tokio::test]
async fn static_read_passes_prefill_result_to_decode() {
    check_dispatch("READ").await;
}

#[tokio::test]
async fn static_write_dispatches_prefill_and_decode_concurrently() {
    check_dispatch("WRITE").await;
}

#[tokio::test]
async fn static_metadata_rejects_incompatible_workers() {
    for (decode_role, decode_mode, decode_dp, expected) in [
        ("D", "WRITE", 1, "mode"),
        ("P", "READ", 1, "role"),
        ("D", "READ", 2, "dp_size"),
    ] {
        let exchange = Arc::new(Exchange::default());
        let prefill = worker("P", "READ", 1, exchange.clone()).await;
        let decode = worker(decode_role, decode_mode, decode_dp, exchange).await;
        let context = common::create_test_context(config(&prefill, &decode));
        let error = RouterFactory::create_router(&context).await.unwrap_err();
        assert!(
            error.contains(expected),
            "Expected {expected} error, got: {error}"
        );
    }
}
