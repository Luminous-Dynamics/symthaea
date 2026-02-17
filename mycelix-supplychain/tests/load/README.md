# Load Testing Suite

Performance and load testing for Mycelix Supply Chain API using K6.

## Prerequisites

Install K6:

```bash
# macOS
brew install k6

# Ubuntu/Debian
sudo gpg -k
sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
sudo apt-get update
sudo apt-get install k6

# Docker
docker pull grafana/k6
```

## Test Scenarios

### 1. Smoke Test (Quick Sanity Check)

Verify the system works under minimal load.

```bash
k6 run smoke-test.js
```

**Profile**:
- Duration: 30 seconds
- VUs: 1-5
- Goal: Verify basic functionality

---

### 2. Load Test (Normal Operations)

Simulate normal production load.

```bash
k6 run load-test.js
```

**Profile**:
- Duration: 10 minutes
- Ramp-up: 0 → 100 VUs over 2 min
- Sustained: 100 VUs for 6 min
- Ramp-down: 100 → 0 VUs over 2 min
- Target: 50 req/s sustained

**Thresholds**:
- p95 latency < 100ms
- p99 latency < 200ms
- Error rate < 1%
- Success rate > 99%

---

### 3. Stress Test (Find Breaking Point)

Push the system to its limits to find the breaking point.

```bash
k6 run stress-test.js
```

**Profile**:
- Duration: 15 minutes
- Ramp-up: 0 → 500 VUs over 5 min
- Peak: 500 VUs for 5 min
- Ramp-down: 500 → 0 VUs over 5 min
- Target: Discover maximum capacity

**Goals**:
- Find maximum throughput
- Identify resource bottlenecks
- Test graceful degradation

---

### 4. Spike Test (Traffic Surges)

Test response to sudden traffic spikes.

```bash
k6 run spike-test.js
```

**Profile**:
- Baseline: 10 VUs
- Spike: 0 → 300 VUs in 10 seconds
- Duration: 2 minutes at peak
- Recovery: 300 → 10 VUs in 10 seconds

**Goals**:
- Verify auto-scaling works
- Test rate limiting
- Validate circuit breakers

---

### 5. Soak Test (Endurance)

Test system stability over extended period.

```bash
k6 run soak-test.js
```

**Profile**:
- Duration: 2 hours
- VUs: 50 sustained
- Target: Detect memory leaks, degradation

**Thresholds**:
- No increase in error rate over time
- Stable memory usage
- No connection pool exhaustion

---

## Running Tests

### Local Development

```bash
# Start service
cargo run --release

# Run smoke test
k6 run smoke-test.js

# Run with custom target
k6 run load-test.js -e BASE_URL=http://localhost:8080
```

### Docker

```bash
docker run -i grafana/k6 run - <load-test.js
```

### CI/CD Integration

```bash
# GitHub Actions
k6 run --out json=results.json load-test.js

# Fail pipeline if thresholds not met
k6 run --no-color load-test.js || exit 1
```

## Metrics

K6 tracks these metrics:

### HTTP Metrics
- `http_reqs`: Total HTTP requests
- `http_req_duration`: Request duration (p50, p95, p99)
- `http_req_failed`: Failed requests percentage
- `http_req_receiving`: Time to receive response
- `http_req_sending`: Time to send request
- `http_req_waiting`: Time waiting for response (TTFB)

### Custom Metrics
- `events_created`: Total events created
- `claims_verified`: Total claims verified
- `lineage_queries`: Total lineage queries

### System Metrics (if monitoring enabled)
- CPU usage
- Memory usage
- Database connections
- Request queue depth

## Interpreting Results

### Good Performance

```
✓ http_req_duration..............: avg=45ms   min=10ms med=42ms max=150ms p(90)=70ms p(95)=85ms
✓ http_req_failed................: 0.12%
✓ http_reqs......................: 30000 (100/s)
✓ iterations.....................: 30000 (100/s)
```

### Performance Issues

```
✗ http_req_duration..............: avg=850ms  min=10ms med=750ms max=5s p(95)=1.2s p(99)=2s
✗ http_req_failed................: 5.23%
✗ http_reqs......................: 5000 (16.6/s)  # Below target
```

## Optimization Tips

If tests reveal performance issues:

1. **High Latency (p95 > 200ms)**
   - Enable database connection pooling
   - Add caching layer
   - Review slow queries
   - Scale horizontally

2. **High Error Rate (> 1%)**
   - Check database connection limits
   - Review timeout settings
   - Verify rate limiting thresholds
   - Check for resource exhaustion

3. **Low Throughput (< 50 req/s)**
   - Profile CPU usage
   - Check for blocking operations
   - Review async/await patterns
   - Consider load balancer

4. **Memory Growth (in soak test)**
   - Check for connection leaks
   - Review in-memory caching
   - Monitor goroutine/thread count

## Performance Targets

### Minimum (Development)
- Throughput: 50 req/s
- p95 latency: < 200ms
- p99 latency: < 500ms
- Error rate: < 5%

### Production (Recommended)
- Throughput: 200 req/s
- p95 latency: < 100ms
- p99 latency: < 200ms
- Error rate: < 1%

### High-Scale (Enterprise)
- Throughput: 1000+ req/s
- p95 latency: < 50ms
- p99 latency: < 100ms
- Error rate: < 0.1%

## Monitoring During Tests

### Prometheus Metrics

```bash
# Watch metrics in real-time
watch -n 1 'curl -s http://localhost:8080/metrics | grep supplychain_events'
```

### Database Connections

```bash
# Monitor SQLite
sqlite3 data/claims.db "SELECT COUNT(*) FROM claims;"

# Monitor PostgreSQL
psql -c "SELECT count(*) FROM pg_stat_activity;"
```

### System Resources

```bash
# CPU and Memory
docker stats supplychain

# Network
iftop -i eth0
```

## Continuous Performance Testing

Add to CI/CD pipeline:

```yaml
# .github/workflows/performance.yml
name: Performance Tests
on:
  pull_request:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  load-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Start service
        run: docker-compose up -d
      - name: Run load test
        run: k6 run --out json=results.json tests/load/load-test.js
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: load-test-results
          path: results.json
```

## Troubleshooting

### Connection Refused

```bash
# Check service is running
curl http://localhost:8080/health

# Check firewall
sudo ufw status
```

### Out of Memory

```bash
# Increase container memory
docker run --memory="2g" ...

# Or adjust VU count
k6 run --vus 50 load-test.js  # Reduce VUs
```

### Timeout Errors

```bash
# Increase timeout in K6
export K6_HTTP_TIMEOUT=30s
k6 run load-test.js
```

---

## Next Steps

1. Run smoke test to establish baseline
2. Run load test and optimize until targets met
3. Run stress test to find capacity limits
4. Set up continuous performance testing in CI/CD
5. Monitor production metrics via Prometheus + Grafana

---

**Version**: 1.0.0
**Last Updated**: 2025-11-15
