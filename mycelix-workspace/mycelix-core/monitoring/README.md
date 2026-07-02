# Mycelix FL Monitoring and Alerting

Comprehensive monitoring, alerting, and observability stack for Mycelix Federated Learning systems.

## Overview

This monitoring stack provides:

- **Prometheus** - Metrics collection and storage
- **Grafana** - Visualization dashboards
- **Alertmanager** - Alert routing and notification
- **Node Exporter** - System metrics
- **cAdvisor** - Container metrics
- **Custom Exporters** - FL-specific metrics

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.9+ (for custom exporters)
- Network access to Mycelix services

### Start the Monitoring Stack

```bash
cd /home/tstoltz/Luminous-Dynamics/Mycelix-Core/monitoring

# Create secrets directory with credentials
mkdir -p alertmanager/secrets
echo "your-slack-webhook-url" > alertmanager/secrets/slack_webhook_url
echo "your-smtp-password" > alertmanager/secrets/smtp_password
echo "your-pagerduty-key" > alertmanager/secrets/pagerduty_ops_key

# Set environment variables
export GRAFANA_ADMIN_PASSWORD=your-secure-password
export POSTGRES_PASSWORD=your-postgres-password

# Start the stack
docker-compose up -d

# Verify services are running
docker-compose ps
```

### Access Points

| Service | URL | Default Credentials |
|---------|-----|---------------------|
| Prometheus | http://localhost:9090 | N/A |
| Grafana | http://localhost:3000 | admin / $GRAFANA_ADMIN_PASSWORD |
| Alertmanager | http://localhost:9093 | N/A |

## Architecture

```
                    +------------------+
                    |    Grafana       |
                    |   (Dashboards)   |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
     +--------v--------+          +---------v--------+
     |   Prometheus    |          |   Alertmanager   |
     | (Metrics Store) |          |  (Alert Routing) |
     +--------+--------+          +------------------+
              |                            |
    +---------+---------+         +--------v--------+
    |         |         |         |  Notification   |
    v         v         v         |  Channels       |
+-------+ +-------+ +-------+     | (Slack, Email,  |
| FL    | | Holo- | | Stor- |     |  PagerDuty)     |
| Coord | | chain | | age   |     +-----------------+
+-------+ +-------+ +-------+
```

## Dashboards

### 1. FL Training Progress

**Location:** Grafana > Mycelix FL > Training Progress

Monitors federated learning training health:

- **Training Overview** - Total rounds, active nodes, Byzantine ratio
- **Training Throughput** - Rounds/minute, gradients/second
- **Round Latency** - p50, p95, p99 latency distributions
- **Gradient Flow** - Received, healed, rejected gradients
- **Node Status** - Active vs healthy nodes over time
- **Error Tracking** - Errors by type

**Key Metrics:**
- `mycelix_fl_rounds_total` - Total training rounds
- `mycelix_fl_round_latency_ms` - Round completion time
- `mycelix_fl_byzantine_ratio` - Current Byzantine ratio

### 2. Byzantine Detection

**Location:** Grafana > Mycelix FL > Byzantine Detection

Security-focused dashboard for attack detection:

- **Security Status** - Real-time threat level indicator
- **Detection Timeline** - Byzantine detections over time
- **Attack Type Distribution** - Pie chart of attack types
- **Detector Performance** - Multi-layer, Shapley, Hyperfeel scores
- **Detection Latency** - SLA compliance (<100ms target)
- **Node Reputation** - Timeline of reputation scores

**Key Metrics:**
- `mycelix_fl_byzantine_detected_total` - Detection count by attack type
- `mycelix_fl_detection_latency_ms` - Detection speed
- `mycelix_fl_node_reputation` - Per-node reputation scores

### 3. Network Health

**Location:** Grafana > Mycelix FL > Network Health

Network and connectivity monitoring:

- **Service Health** - FL Coordinator, ZeroTrustML, Holochain status
- **Node Connectivity** - Connected/disconnected nodes
- **Network Latency** - Inter-service latency percentiles
- **Holochain DHT** - Entry count, I/O latency
- **Hybrid Bridge** - Sync status and backlog
- **Message Queue** - Queue depth and wait times

**Key Metrics:**
- `up{job="..."}` - Service availability
- `mycelix_network_latency_seconds` - Network latency
- `hybrid_bridge_pending` - Sync backlog size

### 4. Resource Usage

**Location:** Grafana > Mycelix FL > Resource Usage

Infrastructure and resource monitoring:

- **System Overview** - CPU, memory, disk usage gauges
- **Service Resources** - Per-service CPU and memory
- **Storage Performance** - Operations/sec, latency by backend
- **Database Health** - PostgreSQL connections, transactions
- **Disk I/O** - Read/write throughput
- **Network I/O** - Receive/transmit rates

**Key Metrics:**
- `process_cpu_seconds_total` - CPU usage
- `process_resident_memory_bytes` - Memory usage
- `mycelix_storage_operation_duration_seconds` - Storage latency

## Alert Rules

### Critical Alerts (Immediate Response Required)

| Alert | Description | Threshold | Response |
|-------|-------------|-----------|----------|
| `ByzantineAttackDetected` | Active attack in progress | >0.5/sec detections | Check Byzantine dashboard, enable strict mode |
| `SybilAttackSuspected` | Coordinated Sybil attack | >1/sec + new nodes | Pause node registration, investigate |
| `NodeDown` | Critical service unavailable | Down >1 min | Check service logs, restart if needed |
| `HybridBridgeIntegrityError` | Data integrity violation | Any occurrence | Immediate investigation, check audit trail |
| `PrivacyViolationDetected` | Privacy breach | Any occurrence | Compliance team notification, investigate |
| `PostgreSQLDown` | Database unavailable | Down >1 min | Failover or restart database |

### Warning Alerts (Investigate Within 1 Hour)

| Alert | Description | Threshold | Response |
|-------|-------------|-----------|----------|
| `HighByzantineRatio` | Approaching tolerance limit | >35% | Review node quality, increase validation |
| `TrainingStalled` | No rounds completing | 0 rounds/15min | Check coordinator logs, node connectivity |
| `ByzantineDetectionDegraded` | Slow detection | >500ms p95 | Optimize detection pipeline |
| `HighNetworkLatency` | Network performance degraded | >1s p95 | Check network, optimize routing |
| `DiskSpaceLow` | Running out of disk | <15% available | Clean up or expand storage |

### Tuning Alert Thresholds

Alert thresholds can be customized in `prometheus/rules/mycelix_alerts.yml`:

```yaml
# Example: Adjust Byzantine ratio threshold
- alert: HighByzantineRatio
  expr: mycelix_fl_byzantine_ratio > 0.35  # Change from 0.35 to 0.40
  for: 5m
  labels:
    severity: warning
```

After changes:

```bash
# Validate configuration
docker exec mycelix-prometheus promtool check rules /etc/prometheus/rules/*.yml

# Reload configuration
curl -X POST http://localhost:9090/-/reload
```

## Alertmanager Configuration

### Notification Channels

Alerts are routed based on severity and category:

| Severity | Category | Channels |
|----------|----------|----------|
| critical | security | PagerDuty, Slack #security-alerts, Email security@ |
| critical | availability | PagerDuty, Slack #alerts-critical |
| critical | compliance | PagerDuty, Slack, Email compliance@ |
| warning | any | Slack #alerts-warnings |
| any | byzantine | Slack #byzantine-detection |

### Adding Notification Channels

Edit `alertmanager/alertmanager.yml`:

```yaml
receivers:
  - name: 'my-new-channel'
    slack_configs:
      - channel: '#my-channel'
        api_url_file: '/etc/alertmanager/secrets/slack_webhook_url'
```

### Silencing Alerts

**Via UI:** Navigate to http://localhost:9093/#/silences

**Via API:**
```bash
# Silence an alert for 4 hours
curl -X POST http://localhost:9093/api/v2/silences \
  -H "Content-Type: application/json" \
  -d '{
    "matchers": [{"name": "alertname", "value": "HighNetworkLatency"}],
    "startsAt": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'",
    "endsAt": "'$(date -u -d "+4 hours" +%Y-%m-%dT%H:%M:%SZ)'",
    "createdBy": "operator",
    "comment": "Investigating network issue"
  }'
```

## Adding Metrics to Your Service

### Python Integration

```python
from monitoring.src.metrics_exporter import MetricsExporter, metrics

# Start metrics server
exporter = MetricsExporter(port=9100)
await exporter.start()

# Record metrics
metrics.rounds_total.inc()
metrics.round_latency_ms.observe(150.5)
metrics.byzantine_ratio.set(0.15)
metrics.detector_score.set(0.95, {"detector": "multi_layer"})

# Use timing decorator
@timed(metrics.detection_latency_ms)
async def detect_byzantine(gradients):
    # detection logic
    pass
```

### Rust Integration (via prometheus crate)

```rust
use prometheus::{Counter, Gauge, Histogram, register_counter, register_gauge};

lazy_static! {
    static ref ROUNDS_TOTAL: Counter = register_counter!(
        "mycelix_fl_rounds_total",
        "Total FL training rounds completed"
    ).unwrap();

    static ref BYZANTINE_RATIO: Gauge = register_gauge!(
        "mycelix_fl_byzantine_ratio",
        "Current Byzantine node ratio"
    ).unwrap();
}

// Record metrics
ROUNDS_TOTAL.inc();
BYZANTINE_RATIO.set(0.15);
```

### Adding New Scrape Targets

Edit `prometheus/prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'my-new-service'
    static_configs:
      - targets: ['my-service:9100']
    metrics_path: /metrics
    scrape_interval: 15s
```

For dynamic discovery, create `prometheus/sd/my-service.yml`:

```yaml
- targets:
    - 'my-service-1:9100'
    - 'my-service-2:9100'
  labels:
    service: my-service
    environment: production
```

## Operations

### Scaling

**Prometheus:**
- Increase retention: `--storage.tsdb.retention.time=60d`
- Increase storage: `--storage.tsdb.retention.size=100GB`
- Enable remote write for long-term storage

**Alertmanager:**
- Deploy in cluster mode for HA
- Configure `--cluster.peer` for each instance

**Grafana:**
- Use external database (PostgreSQL) for multi-instance
- Configure session storage in Redis

### Backup

```bash
# Backup Prometheus data
docker exec mycelix-prometheus tar czf /tmp/prometheus-backup.tar.gz /prometheus
docker cp mycelix-prometheus:/tmp/prometheus-backup.tar.gz ./backups/

# Backup Grafana dashboards and config
docker exec mycelix-grafana tar czf /tmp/grafana-backup.tar.gz /var/lib/grafana
docker cp mycelix-grafana:/tmp/grafana-backup.tar.gz ./backups/

# Backup Alertmanager silences
curl -s http://localhost:9093/api/v2/silences > ./backups/alertmanager-silences.json
```

### Troubleshooting

**Prometheus not scraping targets:**
```bash
# Check target status
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health}'

# Check configuration
docker exec mycelix-prometheus promtool check config /etc/prometheus/prometheus.yml
```

**Alerts not firing:**
```bash
# Check alert rules status
curl http://localhost:9090/api/v1/rules | jq '.data.groups[].rules[] | {name: .name, state: .state}'

# Test alert expression
curl 'http://localhost:9090/api/v1/query?query=mycelix_fl_byzantine_ratio>0.35'
```

**Grafana dashboard not loading:**
```bash
# Check datasource connectivity
curl http://localhost:3000/api/datasources/proxy/1/api/v1/query?query=up

# Reload dashboards
docker restart mycelix-grafana
```

### Health Checks

```bash
# Check all services
curl -s http://localhost:9090/-/healthy && echo "Prometheus: OK"
curl -s http://localhost:9093/-/healthy && echo "Alertmanager: OK"
curl -s http://localhost:3000/api/health | jq -r '.database' && echo "Grafana: OK"

# Check metrics collection
curl -s http://localhost:9090/api/v1/query?query=up | jq '.data.result | length'
```

## Recording Rules

Pre-computed metrics for dashboard efficiency are defined in `prometheus/rules/recording_rules.yml`:

| Rule | Description | Use Case |
|------|-------------|----------|
| `mycelix_fl:rounds_per_minute` | Training velocity | Dashboard overview |
| `mycelix_fl:byzantine_detection_rate` | Normalized detection rate | Trend analysis |
| `mycelix_fl:detection_latency_p95_ms` | 95th percentile latency | SLA monitoring |
| `mycelix_fl:healthy_node_percentage` | Cluster health | Availability tracking |
| `mycelix_storage:cache_hit_rate` | Cache effectiveness | Performance tuning |

## Best Practices

### Dashboard Design

1. **Use consistent time ranges** - Default to 1 hour for operational dashboards
2. **Add thresholds** - Visual indicators for warning/critical levels
3. **Include runbook links** - Every alert panel should link to documentation
4. **Group related metrics** - Use collapsible rows for organization

### Alert Design

1. **Set meaningful `for` durations** - Avoid alert fatigue from transient spikes
2. **Include resolution steps** - Every alert should have actionable guidance
3. **Use inhibition rules** - Prevent cascading alerts
4. **Test alerts in staging** - Validate before production deployment

### Metric Naming

Follow Prometheus naming conventions:
- Use `snake_case`
- Include unit in name (`_seconds`, `_bytes`, `_total`)
- Use `_total` suffix for counters
- Prefix with application name (`mycelix_fl_`)

## Support

For issues with monitoring:

1. Check logs: `docker-compose logs -f prometheus grafana alertmanager`
2. Review this documentation
3. Check Prometheus targets: http://localhost:9090/targets
4. Contact: ops@mycelix.io

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-08 | Initial comprehensive monitoring setup |
