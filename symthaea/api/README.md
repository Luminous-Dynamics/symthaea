# API

HTTP API server for external integration with Symthaea.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/v1/health` | Health check (v1 alias) |
| POST | `/v1/submit` | Submit model for spectral connectivity (lambda2) evaluation |
| GET | `/v1/results/{submission_id}` | Get evaluation results |
| GET | `/v1/leaderboard` | Public leaderboard |
| GET | `/v1/leaderboard/topologies` | Topology rankings |
| GET | `/v1/datasets` | List available datasets |
| GET | `/v1/datasets/{dataset_id}` | Dataset details |
| POST | `/v1/compare` | Compare two models |
| POST | `/v1/dimensional-sweep` | Queue a dimensional sweep |

## Running

Note: API fields labeled "phi" currently report spectral connectivity (lambda2), not IIT Phi.
Supported `topology_type` values: `ring`, `torus`, `hypercube`, `star`, `random`, `small_world`, `dense`, `custom`.
For `custom`, set `topology_type=custom` and provide exactly one of `adjacency_matrix`, `edge_list`, or `node_representations`.
`n_nodes` is ignored for `torus` (fixed 3x3) and `hypercube` (dimension selects size).

Run the API server with:

```bash
cargo run --bin symthaea-api --features api_module
```

Set `SYMTHAEA_API_ADDR` to override the bind address (default `0.0.0.0:8080`).

## Configuration

Configuration is handled by the hosting binary; there is no `api/config.toml` in this repo.

## Authentication

Authentication is not enforced in the current API module. The `SYMTHAEA_API_TOKEN`
environment variable is reserved for future use.
