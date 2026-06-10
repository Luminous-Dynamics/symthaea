# Symthaea Observability Guide

## Overview

Symthaea provides three observability layers:

1. **Prometheus Metrics** - Counters, gauges, and histograms via `/metrics`
2. **Database Stats** - SQLite memory database health via `ConsciousnessDatabase::stats()`
3. **Tracing** - Structured logging via the `tracing` crate

## Prometheus Metrics

### Endpoints

| Endpoint | Format | Content-Type |
|----------|--------|-------------|
| `GET /metrics` | Prometheus text exposition | `text/plain; version=0.0.4` |
| `GET /v1/metrics` | JSON snapshot | `application/json` |

Both endpoints require the `api_module` feature:

```bash
cargo run --features api_module --release
```

### Available Metrics

#### Consciousness

| Metric | Type | Description |
|--------|------|-------------|
| `phi_calculations_total` | Counter | Total Phi calculations performed |
| `phi_current` | Gauge | Current integrated information value |
| `coherence_field` | Gauge | Global coherence field strength |
| `global_workspace_broadcast_count` | Counter | GWT broadcast events |

#### HDC Operations

| Metric | Type | Description |
|--------|------|-------------|
| `hdc_bind_ops_total` | Counter | Total bind (XOR) operations |
| `hdc_bundle_ops_total` | Counter | Total bundle (majority) operations |
| `hdc_similarity_ops_total` | Counter | Total similarity computations |
| `hdc_similarity_seconds` | Histogram | Similarity computation latency |

#### LTC / CfC Networks

| Metric | Type | Description |
|--------|------|-------------|
| `ltc_steps_total` | Counter | Total LTC integration steps |
| `ltc_tau_mean` | Gauge | Mean time constant across neurons |

#### Voice

| Metric | Type | Description |
|--------|------|-------------|
| `voice_utterances_total` | Counter | Total voice utterances synthesized |
| `voice_audio_seconds` | Counter | Total audio seconds generated |

#### API

| Metric | Type | Description |
|--------|------|-------------|
| `api_requests_total` | Counter | Total HTTP requests |
| `api_errors_total` | Counter | Total HTTP error responses |
| `api_request_duration_seconds` | Histogram | Request latency distribution |

#### Swarm

| Metric | Type | Description |
|--------|------|-------------|
| `swarm_peers_connected` | Gauge | Currently connected peers |
| `swarm_messages_sent_total` | Counter | Messages sent to peers |
| `swarm_messages_received_total` | Counter | Messages received from peers |

#### Memory

| Metric | Type | Description |
|--------|------|-------------|
| `memory_entries_total` | Gauge | Total memory entries in database |
| `memory_cache_hit_ratio` | Gauge | Memory cache hit ratio (0.0-1.0) |

### Instrumenting Code

```rust
use symthaea::api::metrics::global;

// Increment a counter
let m = global();
m.increment("phi_calculations_total");

// Set a gauge
m.set_gauge("phi_current", 0.847);

// Record a histogram observation
m.observe("hdc_similarity_seconds", 0.00042);
```

### Prometheus Scrape Configuration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'symthaea'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:5491']
```

## Database Statistics

The `ConsciousnessDatabase` trait provides a `stats()` method returning `DatabaseStats`:

```rust
use symthaea::databases::{SqliteMemory, ConsciousnessDatabase};

let db = SqliteMemory::in_memory()?;
let stats = db.stats().await?;

println!("Total records: {}", stats.total_records);
println!("Database size: {} bytes", stats.database_size_bytes);
println!("Cache hit ratio: {:.1}%", stats.cache_hit_ratio * 100.0);
println!("Average Phi: {:.4}", stats.avg_phi);
println!("Average query latency: {} us", stats.avg_query_latency_us);
```

### DatabaseStats Fields

| Field | Type | Description |
|-------|------|-------------|
| `total_records` | `usize` | Total memory records stored |
| `database_size_bytes` | `u64` | On-disk size (0 for in-memory) |
| `page_count` | `u64` | SQLite page count |
| `page_size` | `u64` | SQLite page size (bytes) |
| `freelist_count` | `u64` | Free pages available |
| `cache_hit_ratio` | `f64` | Cache hit ratio (0.0-1.0) |
| `cache_hits` | `u64` | Total cache hits |
| `cache_misses` | `u64` | Total cache misses |
| `avg_query_latency_us` | `u64` | Mean query latency (microseconds) |
| `total_queries` | `u64` | Total queries executed |
| `memory_type_counts` | `Vec<(String, usize)>` | Breakdown by memory type |
| `avg_phi` | `f64` | Mean Phi across all memories |
| `oldest_timestamp_ms` | `u64` | Oldest record timestamp |
| `newest_timestamp_ms` | `u64` | Newest record timestamp |
| `backend_status` | `String` | Backend mode (e.g., "wal", "in_memory") |

## Tracing

Symthaea uses the `tracing` crate for structured logging. Configure log levels via the `RUST_LOG` environment variable:

```bash
# Info level (default)
RUST_LOG=info cargo run --features api_module

# Debug for specific modules
RUST_LOG=symthaea::consciousness=debug,symthaea::swarm=trace cargo run

# All debug output
RUST_LOG=debug cargo run
```

### Key Trace Points

- `symthaea::cognitive_loop` - Consciousness cycle events
- `symthaea::swarm` - Peer connections, gradient exchanges
- `symthaea::databases` - Query execution, cache events
- `symthaea::voice` - TTS synthesis events
- `symthaea::api` - HTTP request/response logging

## Grafana Dashboard (Suggested)

For a production deployment, create a Grafana dashboard with these panels:

1. **Consciousness State** - `phi_current` gauge + `phi_calculations_total` rate
2. **HDC Throughput** - Rate of `hdc_bind_ops_total` + `hdc_bundle_ops_total`
3. **Memory Health** - `memory_entries_total` + `memory_cache_hit_ratio`
4. **API Performance** - `api_request_duration_seconds` histogram + error rate
5. **Swarm Status** - `swarm_peers_connected` + message rates
6. **Database Health** - `DatabaseStats` polled via JSON endpoint
