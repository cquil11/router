//! Static MoRI-IO routing: READ mode through worker URLs, without ZMQ discovery.
mod common;

use axum::{http::HeaderMap, routing::get, routing::post, Json, Router};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use vllm_router_rs::{
    config::{KvConnector, PolicyConfig, RouterConfig, RoutingMode},
    protocols::spec::ChatCompletionRequest,
    routers::RouterFactory,
};

type Requests = Arc<Mutex<Vec<(&'static str, HeaderMap, Value)>>>;

/// What a MoRI-IO READ producer's `request_finished` returns (vLLM MoRIIOConnectorScheduler).
fn prefill_kv_transfer_params() -> Value {
    json!({
        "do_remote_prefill": true,
        "do_remote_decode": false,
        "remote_block_ids": [[1, 2, 3]],
        "remote_engine_id": "10.0.0.1:6301",
        "remote_host": "10.0.0.1",
        "remote_handshake_port": 6301,
        "remote_notify_port": 61005,
        "remote_dp_rank": 1,
        "remote_dp_rank_override": true,
        // Differs from the router's intra-node DP size (1): prefill's value must win.
        "remote_dp_size": 2,
        "remote_dp_size_local": 2,
        "tp_size": 4,
        "transfer_id": "tx-from-prefill",
    })
}

/// Mock vLLM worker; prefill answers with MoRI-IO producer kv_transfer_params.
async fn worker(role: &'static str, requests: Requests) -> (String, tokio::task::JoinHandle<()>) {
    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route(
            "/v1/chat/completions",
            post(
                move |headers: HeaderMap, Json(body): Json<Value>| async move {
                    requests.lock().unwrap().push((role, headers, body));
                    let mut response = json!({"choices": [{"message": {"content": "done"}}]});
                    if role == "P" {
                        response["kv_transfer_params"] = prefill_kv_transfer_params();
                    }
                    Json(response)
                },
            ),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    (
        url,
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() }),
    )
}

#[tokio::test]
async fn static_read_forwards_prefill_addresses_to_decode() {
    let requests: Requests = Arc::default();
    let (prefill_url, prefill) = worker("P", requests.clone()).await;
    let (decode_url, decode) = worker("D", requests.clone()).await;
    let config = RouterConfig {
        mode: RoutingMode::VllmPrefillDecode {
            prefill_urls: vec![(prefill_url, None)],
            decode_urls: vec![decode_url],
            prefill_policy: None,
            decode_policy: None,
            discovery_address: None,
        },
        kv_connector: KvConnector::MoriIO,
        policy: PolicyConfig::Random,
        worker_startup_timeout_secs: 2,
        worker_startup_check_interval_secs: 1,
        disable_retries: true,
        ..RouterConfig::default()
    };
    let router = RouterFactory::create_router(&common::create_test_context(config))
        .await
        .unwrap();
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test", "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 10, "stream": false
    }))
    .unwrap();
    let response = router.route_chat(None, &request, None).await;
    assert!(response.status().is_success(), "{}", response.status());

    let requests = requests.lock().unwrap();
    let roles: Vec<_> = requests.iter().map(|(role, _, _)| *role).collect();
    assert_eq!(roles, ["P", "D"], "READ dispatches prefill before decode");
    let (_, p_headers, p_body) = &requests[0];
    let (_, d_headers, d_body) = &requests[1];

    // A plain request ID: the connector would try to parse embedded addresses otherwise.
    let request_id = p_headers["x-request-id"].to_str().unwrap();
    assert_eq!(request_id, d_headers["x-request-id"].to_str().unwrap());
    assert!(!request_id.contains("___prefill_addr_"), "{request_id}");

    assert_eq!(p_body["kv_transfer_params"]["do_remote_decode"], true);

    // Decode gets prefill's MoRI-IO addresses and DP layout unchanged.
    assert_eq!(d_body["kv_transfer_params"], prefill_kv_transfer_params());

    prefill.abort();
    decode.abort();
}
