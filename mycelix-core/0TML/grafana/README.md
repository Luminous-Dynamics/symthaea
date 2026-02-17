# BFT Monitoring with Grafana + Prometheus

Comprehensive monitoring stack for Byzantine Fault Tolerance detection metrics.

## Quick Start

### 1. Start the monitoring stack

```bash
cd grafana
docker-compose up -d
```

### 2. Access Grafana

Open http://localhost:3000

- **Username**: admin (or set via `GRAFANA_ADMIN_USER`)
- **Password**: admin (or set via `GRAFANA_ADMIN_PASSWORD`)

### 3. View Dashboard

The "Byzantine Fault Tolerance Monitoring" dashboard will be automatically loaded.

## What's Monitored

### Key Metrics

1. **False Positive Rate** - Percentage of honest nodes incorrectly flagged
   - Target: ≤5% for label skew
   - Alert threshold: >10% (warning)

2. **Detection Rate** - Percentage of Byzantine nodes caught
   - Target: ≥95%
   - Alert threshold: <90% (critical)

3. **Regression Success** - Whether CI tests pass
   - Alert: immediate on failure (critical)

4. **Honest Reputation Trend** - Average reputation of honest nodes
   - Indicator of reputation collapse

### Dashboard Panels

- **Time Series**: FP rate and detection rate over time by distribution
- **Stat Panels**: Current label skew FP rate and regression status
- **Gauges**: IID baseline health and attack success rate
- **Heatmap**: FP rate detail by round
- **Table**: Recent test runs with color-coded metrics

## Alert Configuration

### Built-in Alerts

1. **High False Positive Rate** (Warning)
   - Trigger: FP rate >10% for 5 minutes
   - Action: Slack/Email notification

2. **Low Detection Rate** (Critical)
   - Trigger: Detection <90% for 5 minutes
   - Action: Immediate Slack/Email

3. **Regression Failure** (Critical)
   - Trigger: `matl_regression_success == 0` for 1 minute
   - Action: Immediate notification

4. **FP Rate Trending Up** (Warning)
   - Trigger: FP increased by >5% in last hour
   - Action: Slack/Email notification

5. **Honest Reputation Collapse** (Critical)
   - Trigger: Average FP >50% for 5 minutes
   - Action: Immediate notification

### Configuring Notifications

Edit `.env` file:

```bash
# Slack webhook for alerts
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# Email addresses for alerts (comma-separated)
ALERT_EMAIL_ADDRESSES=dev-team@example.com,ops@example.com
```

Then restart:

```bash
docker-compose down
docker-compose up -d
```

## Updating Metrics

### Manual Update

After running tests:

```bash
# Export metrics
python scripts/export_bft_metrics.py \
  --matrix tests/results/bft_attack_matrix.json \
  --output artifacts/matl_metrics.prom

# Copy to Grafana artifacts directory
cp artifacts/matl_metrics.prom grafana/artifacts/

# Prometheus will auto-detect in ~1 minute
```

### CI/CD Integration

The nightly regression workflow (`matl-regression.yml`) automatically:
1. Runs tests
2. Exports metrics to `artifacts/matl_metrics.prom`
3. Uploads artifacts

To integrate with Grafana:

**Option A: Manual sync**
```bash
# Download from GitHub Actions
gh run download <run-id> -n matl_metrics

# Copy to monitoring stack
cp matl_metrics.prom grafana/artifacts/
```

**Option B: Automated sync**
```bash
# Add to cron
0 3 * * * gh run download --repo Luminous-Dynamics/Mycelix-Core latest -n matl_metrics && \
          cp matl_metrics.prom /path/to/grafana/artifacts/
```

## Customization

### Add Custom Panels

1. Open Grafana UI
2. Click "+" → "Dashboard" → "Add Panel"
3. Query: `matl_false_positive_rate_percent{your_filter}`
4. Save dashboard
5. Export JSON to `dashboards/bft_monitoring.json`

### Add Custom Alerts

Edit `provisioning/alerting/bft_alerts.yml`:

```yaml
- uid: custom_alert
  title: My Custom Alert
  condition: A
  data:
    - refId: A
      model:
        expr: your_prometheus_query > threshold
  for: 5m
  annotations:
    description: Your alert description
  labels:
    severity: warning
```

Restart Grafana:

```bash
docker-compose restart grafana
```

## Architecture

```
┌─────────────────────┐
│  CI/CD Pipeline     │
│  (GitHub Actions)   │
└──────────┬──────────┘
           │ Exports metrics
           ▼
┌─────────────────────┐
│  matl_metrics.prom  │
│  (Prometheus format)│
└──────────┬──────────┘
           │ Scraped by
           ▼
┌─────────────────────┐
│   Prometheus        │
│   (Time series DB)  │
└──────────┬──────────┘
           │ Queried by
           ▼
┌─────────────────────┐
│     Grafana         │
│   (Visualization)   │
└──────────┬──────────┘
           │ Sends alerts
           ▼
┌─────────────────────┐
│  Slack / Email      │
│  (Notifications)    │
└─────────────────────┘
```

## Metrics Format

Prometheus metrics exported by `export_bft_metrics.py`:

```prometheus
# Detection rate percentage
matl_detection_rate_percent{attack="noise_0.5",distribution="iid",ratio="30"} 100.0

# False positive rate percentage
matl_false_positive_rate_percent{attack="noise_0.5",distribution="label_skew",ratio="30",alpha="0.2"} 42.9

# Regression success flag (0 or 1)
matl_regression_success{attack="noise_0.5",distribution="iid",ratio="30"} 1
```

## Troubleshooting

### Dashboard not showing data

**Check Prometheus is scraping:**
```bash
# View Prometheus targets
curl http://localhost:9090/api/v1/targets

# Check if metrics file exists
ls -lh grafana/artifacts/matl_metrics.prom
```

**Verify metrics are loaded:**
```bash
# Query Prometheus directly
curl 'http://localhost:9090/api/v1/query?query=matl_detection_rate_percent'
```

### Alerts not firing

**Check alert rules:**
```bash
# View active alerts
curl http://localhost:9090/api/v1/alerts

# View Grafana alert status
# UI: http://localhost:3000/alerting/list
```

**Verify environment variables:**
```bash
docker-compose config | grep -A5 grafana
```

### Permission issues with volumes

```bash
# Fix ownership
sudo chown -R 472:472 grafana/grafana-data
sudo chown -R 65534:65534 grafana/prometheus-data
```

## Production Deployment

### Use persistent volumes

Edit `docker-compose.yml`:

```yaml
volumes:
  prometheus-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /data/prometheus

  grafana-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /data/grafana
```

### Secure with HTTPS

Add nginx reverse proxy:

```nginx
server {
    listen 443 ssl;
    server_name metrics.example.com;

    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Set strong credentials

```bash
# Generate secure password
export GRAFANA_ADMIN_PASSWORD=$(openssl rand -base64 32)

# Save to .env
echo "GRAFANA_ADMIN_PASSWORD=$GRAFANA_ADMIN_PASSWORD" >> .env

# Restart
docker-compose up -d
```

## Resources

- [Grafana Documentation](https://grafana.com/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [PromQL Tutorial](https://prometheus.io/docs/prometheus/latest/querying/basics/)

## Support

For issues with the monitoring stack:
1. Check logs: `docker-compose logs grafana` or `docker-compose logs prometheus`
2. Verify connectivity: `docker-compose ps`
3. Review configuration files in `provisioning/`
