# Static MoRI-IO workers (PoC)

Use the usual worker URLs with MoRI-IO, without a ZMQ registration listener:

```bash
vllm-router --vllm-pd-disaggregation --kv-connector moriio \
  --prefill http://prefill-host:8000 \
  --decode http://decode-host:8000
```

This requires the companion vLLM PoC exposing `GET /v1/moriio/metadata`.
Start the workers before starting the router. For example, on separate hosts:

```bash
# Prefill host; replace MODEL and PREFILL_IP.
vllm serve MODEL --host 0.0.0.0 --port 8000 \
  --kv-transfer-config '{"kv_connector":"MoRIIOConnector","kv_role":"kv_producer","kv_connector_extra_config":{"host_ip":"PREFILL_IP","read_mode":true}}'

# Decode host; replace MODEL and DECODE_IP.
vllm serve MODEL --host 0.0.0.0 --port 8000 \
  --kv-transfer-config '{"kv_connector":"MoRIIOConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"host_ip":"DECODE_IP","read_mode":true}}'
```

The workers do not need `proxy_ip`, `proxy_ping_port`, or a duplicate `http_port`
inside the connector configuration. Control ports remain worker settings;
the router reads them, the role, TP size, and transfer mode over HTTP at startup.
The configured URLs remain authoritative for HTTP routing. The existing static
worker health checks and load-balancing policies select workers.

Set `read_mode` to `false` on both workers to use WRITE. READ uses the existing
sequential dispatch path; WRITE uses the existing concurrent dispatch path.
The router rejects mixed modes and incorrect P/D roles. When workers require
an API key, set the router's `--api-key` to that same key.

## Scope

- HTTP worker origins only, with DP=1, PP=1 and single-node `uni`/`mp` execution
  per worker. Local tensor parallelism is supported; prefill and decode can be
  on different hosts. Multi-DP and remote executors remain on the discovery path.
- Use a peer-reachable `host_ip` on machines with multiple interfaces. Workers
  sharing a host must use distinct handshake and notification port ranges.
- Metadata is cached at startup. Restart the router when membership, ports,
  mode, or topology changes. Live metadata refresh is outside this PoC.
- Load accounting reserves both selected workers for the request, including
  the sequential READ path; it does not track each READ stage separately.
- The HTTP checks exercise router coordination only; actual KV transfer and
  performance still need validation on AMD GPUs.

Existing `--vllm-discovery-address` deployments continue to use worker
registration. This PoC does not introduce a new transfer protocol or router flags.
