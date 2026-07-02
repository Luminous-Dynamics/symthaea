# Metrics and Monitoring

Observability guide for the Knowledge hApp.

## Overview

Effective monitoring ensures Knowledge operates reliably. This guide covers:
- Key metrics to track
- Monitoring setup
- Alerting thresholds
- Dashboard design
- Incident response

## Key Metrics

### System Health

| Metric | Description | Target | Alert |
|--------|-------------|--------|-------|
| `node_up` | Agent node is running | 1 | 0 |
| `dht_peers` | Connected DHT peers | >10 | <5 |
| `memory_usage_mb` | Agent memory | <512MB | >800MB |
| `cpu_usage_percent` | Agent CPU | <50% | >80% |
| `disk_usage_percent` | Storage used | <70% | >85% |

### Application Metrics

| Metric | Description | Target | Alert |
|--------|-------------|--------|-------|
| `claims_total` | Total claims in system | Growing | Sudden drop |
| `claims_created_rate` | Claims/minute | Varies | >100x normal |
| `queries_per_second` | Query throughput | >50 | <10 |
| `fact_checks_per_minute` | Fact-check rate | Varies | >10x normal |
| `cache_hit_rate` | Cache effectiveness | >80% | <50% |

### Latency Metrics

| Metric | Description | P50 | P99 | Alert |
|--------|-------------|-----|-----|-------|
| `query_latency_ms` | Search latency | 100ms | 500ms | P99>1s |
| `create_latency_ms` | Write latency | 50ms | 200ms | P99>500ms |
| `propagation_latency_ms` | Belief propagation | 1s | 5s | P99>10s |
| `factcheck_latency_ms` | Fact-check time | 500ms | 2s | P99>5s |

### Quality Metrics

| Metric | Description | Target | Alert |
|--------|-------------|--------|-------|
| `verification_rate` | % claims verified | >30% | <10% |
| `contradiction_rate` | % claims with contradictions | <20% | >40% |
| `flag_rate` | Flags per 1000 claims | <50 | >100 |
| `author_reputation_avg` | Average author score | >0.6 | <0.4 |

## Metric Collection

### Built-in Metrics

```typescript
// Enable metrics in SDK
const knowledge = new KnowledgeClient(client, {
  metrics: {
    enabled: true,
    endpoint: 'http://localhost:9090/metrics', // Prometheus
    prefix: 'knowledge_',
    labels: {
      environment: 'production',
      region: 'us-east',
    },
  },
});
```

### Custom Metrics

```typescript
import { Counter, Histogram, Gauge } from 'prom-client';

// Counter for claims
const claimsCreated = new Counter({
  name: 'knowledge_claims_created_total',
  help: 'Total claims created',
  labelNames: ['domain', 'status'],
});

// Histogram for latency
const queryLatency = new Histogram({
  name: 'knowledge_query_latency_seconds',
  help: 'Query latency in seconds',
  labelNames: ['query_type'],
  buckets: [0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
});

// Gauge for current state
const activeClaims = new Gauge({
  name: 'knowledge_active_claims',
  help: 'Number of active claims',
  labelNames: ['domain'],
});

// Usage
async function createClaim(input: CreateClaimInput) {
  const timer = queryLatency.startTimer({ query_type: 'create' });
  try {
    const result = await knowledge.claims.createClaim(input);
    claimsCreated.inc({ domain: input.domain, status: 'success' });
    return result;
  } catch (error) {
    claimsCreated.inc({ domain: input.domain, status: 'error' });
    throw error;
  } finally {
    timer();
  }
}
```

## Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'knowledge'
    static_configs:
      - targets: ['localhost:9090']
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance
        regex: '(.+):\d+'
        replacement: '${1}'

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

rule_files:
  - 'knowledge_alerts.yml'
```

## Alert Rules

```yaml
# knowledge_alerts.yml
groups:
  - name: knowledge
    rules:
      # Node health
      - alert: KnowledgeNodeDown
        expr: knowledge_node_up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Knowledge node is down"
          description: "Node {{ $labels.instance }} has been down for more than 1 minute."

      # Latency
      - alert: HighQueryLatency
        expr: histogram_quantile(0.99, rate(knowledge_query_latency_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High query latency"
          description: "P99 query latency is {{ $value }}s (threshold: 1s)"

      # Error rate
      - alert: HighErrorRate
        expr: rate(knowledge_errors_total[5m]) / rate(knowledge_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate"
          description: "Error rate is {{ $value | humanizePercentage }}"

      # Cache
      - alert: LowCacheHitRate
        expr: knowledge_cache_hit_rate < 0.5
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Low cache hit rate"
          description: "Cache hit rate is {{ $value | humanizePercentage }}"

      # DHT health
      - alert: LowPeerCount
        expr: knowledge_dht_peers < 5
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Low DHT peer count"
          description: "Only {{ $value }} DHT peers connected"
```

## Grafana Dashboard

### Dashboard JSON

```json
{
  "title": "Knowledge hApp",
  "panels": [
    {
      "title": "Claims Created",
      "type": "graph",
      "targets": [
        {
          "expr": "rate(knowledge_claims_created_total[5m])",
          "legendFormat": "{{domain}}"
        }
      ]
    },
    {
      "title": "Query Latency",
      "type": "heatmap",
      "targets": [
        {
          "expr": "rate(knowledge_query_latency_seconds_bucket[5m])",
          "format": "heatmap"
        }
      ]
    },
    {
      "title": "Cache Hit Rate",
      "type": "gauge",
      "targets": [
        {
          "expr": "knowledge_cache_hit_rate"
        }
      ],
      "options": {
        "thresholds": [
          { "value": 0, "color": "red" },
          { "value": 0.5, "color": "yellow" },
          { "value": 0.8, "color": "green" }
        ]
      }
    },
    {
      "title": "DHT Peers",
      "type": "stat",
      "targets": [
        {
          "expr": "knowledge_dht_peers"
        }
      ]
    }
  ]
}
```

### Key Dashboard Panels

1. **Overview Row**
   - Total claims (stat)
   - Active users (stat)
   - Error rate (gauge)
   - Uptime (stat)

2. **Performance Row**
   - Query latency (heatmap)
   - Write latency (graph)
   - Cache hit rate (gauge)
   - Throughput (graph)

3. **Quality Row**
   - Verification rate (graph)
   - Contradiction rate (graph)
   - Flag rate (graph)
   - Avg credibility (graph)

4. **Infrastructure Row**
   - CPU/Memory (graph)
   - DHT peers (graph)
   - Disk usage (gauge)
   - Network I/O (graph)

## Logging

### Structured Logging

```typescript
import { Logger } from '@mycelix/knowledge-sdk';

const logger = new Logger({
  level: 'info',
  format: 'json',
  output: 'stdout',
});

// Structured log entry
logger.info('Claim created', {
  claimId: hash.toString(),
  domain: input.domain,
  author: agentPubKey,
  durationMs: Date.now() - startTime,
});

// Output:
// {"level":"info","message":"Claim created","claimId":"uhCkk...","domain":"climate","durationMs":45,"timestamp":"2024-01-15T10:30:00Z"}
```

### Log Levels

| Level | Use Case | Example |
|-------|----------|---------|
| `error` | Failures requiring attention | "DHT write failed" |
| `warn` | Potential issues | "High latency detected" |
| `info` | Normal operations | "Claim created" |
| `debug` | Detailed debugging | "Cache lookup: hit" |
| `trace` | Very detailed | "DHT message received" |

### Log Aggregation

```yaml
# Fluentd configuration
<source>
  @type tail
  path /var/log/knowledge/*.log
  pos_file /var/log/td-agent/knowledge.log.pos
  tag knowledge
  <parse>
    @type json
  </parse>
</source>

<filter knowledge>
  @type record_transformer
  <record>
    service knowledge
    environment ${ENV}
  </record>
</filter>

<match knowledge>
  @type elasticsearch
  host elasticsearch
  port 9200
  index_name knowledge-logs
</match>
```

## Health Checks

### Endpoint

```typescript
// Health check endpoint
app.get('/health', async (req, res) => {
  const health = await checkHealth();

  if (health.healthy) {
    res.status(200).json(health);
  } else {
    res.status(503).json(health);
  }
});

async function checkHealth(): Promise<HealthStatus> {
  const checks = await Promise.all([
    checkDHTConnection(),
    checkCacheHealth(),
    checkDiskSpace(),
  ]);

  const healthy = checks.every(c => c.healthy);

  return {
    healthy,
    timestamp: new Date().toISOString(),
    checks: {
      dht: checks[0],
      cache: checks[1],
      disk: checks[2],
    },
  };
}
```

### Health Response

```json
{
  "healthy": true,
  "timestamp": "2024-01-15T10:30:00Z",
  "checks": {
    "dht": {
      "healthy": true,
      "peers": 15,
      "latencyMs": 45
    },
    "cache": {
      "healthy": true,
      "hitRate": 0.85,
      "size": 8500
    },
    "disk": {
      "healthy": true,
      "usedPercent": 45
    }
  }
}
```

## Incident Response

### Severity Levels

| Level | Description | Response Time | Example |
|-------|-------------|---------------|---------|
| P1 | Service down | 15 minutes | All nodes down |
| P2 | Major degradation | 1 hour | >50% errors |
| P3 | Minor issue | 4 hours | Slow queries |
| P4 | Low priority | 24 hours | Dashboard bug |

### Runbooks

#### High Error Rate

1. Check error logs: `grep "error" /var/log/knowledge/*.log | tail -100`
2. Identify error type: most common error message
3. Check DHT health: `knowledge_dht_peers` metric
4. Check recent deployments: rollback if needed
5. Scale up if load-related

#### High Latency

1. Check cache hit rate: if low, investigate cache
2. Check DHT peer count: network issues?
3. Check query patterns: new expensive queries?
4. Check resource usage: CPU/memory pressure?
5. Enable query profiling for slow queries

#### Node Down

1. Check node status: `systemctl status knowledge`
2. Check logs: `journalctl -u knowledge -n 100`
3. Check resources: memory, disk, CPU
4. Restart if needed: `systemctl restart knowledge`
5. Investigate root cause after recovery

## SLIs and SLOs

### Service Level Indicators

| SLI | Measurement |
|-----|-------------|
| Availability | % of successful health checks |
| Latency | P99 query latency |
| Error Rate | % of requests returning errors |
| Throughput | Requests per second |

### Service Level Objectives

| Metric | SLO | Error Budget |
|--------|-----|--------------|
| Availability | 99.9% | 43.8 min/month |
| P99 Latency | <1s | - |
| Error Rate | <1% | 1% of requests |

### Error Budget Policy

- **>50% budget remaining**: Normal operations
- **25-50% budget remaining**: Prioritize reliability work
- **<25% budget remaining**: Feature freeze, fix reliability
- **0% budget**: Incident review required

---

## Related Documentation

- [Performance](./PERFORMANCE.md) - Performance optimization
- [Security](./SECURITY.md) - Security considerations

---

*Observe, measure, improve.*
