# Mycelix Bootstrap Node Operator Guide

This guide explains how to run your own bootstrap node for the Mycelix network, contributing to network resilience and decentralization.

## Overview

Bootstrap nodes help new peers discover other nodes in the network. The Mycelix hierarchical bootstrap system eliminates single points of failure by supporting multiple tiers of bootstrap servers:

1. **Primary** - Official Holo + Mycelix-operated servers
2. **Secondary** - Community-operated servers (like yours!)
3. **Tertiary** - Development/experimental servers
4. **Local Cache** - Previously discovered peers
5. **DHT Discovery** - Gossip-based peer finding

By running a secondary bootstrap node, you help make the network more resilient and decentralized.

## Requirements

### Hardware
- **CPU**: 2+ cores
- **RAM**: 2GB minimum (4GB recommended)
- **Storage**: 10GB SSD
- **Network**: Stable internet connection, public IP or reverse proxy

### Software
- Linux server (Ubuntu 22.04 or newer recommended)
- Docker and Docker Compose (recommended) OR
- Node.js 18+ OR
- Rust toolchain (latest stable)

## Quick Start with Docker

The easiest way to run a bootstrap node is with Docker.

### 1. Clone the Repository

```bash
git clone https://github.com/Mycelix/bootstrap-server.git
cd bootstrap-server
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```bash
# Server configuration
BOOTSTRAP_PORT=8888
BOOTSTRAP_HOST=0.0.0.0

# TLS (recommended for production)
TLS_ENABLED=true
TLS_CERT_PATH=/etc/letsencrypt/live/yourdomain/fullchain.pem
TLS_KEY_PATH=/etc/letsencrypt/live/yourdomain/privkey.pem

# Logging
LOG_LEVEL=info

# Peer storage
PEER_EXPIRY_HOURS=24
MAX_PEERS_PER_SPACE=10000

# Rate limiting
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000
```

### 3. Start the Server

```bash
docker-compose up -d
```

### 4. Verify It's Running

```bash
curl http://localhost:8888/health
# Should return: {"status": "healthy", "peers": 0}
```

## Manual Installation

### Option A: Node.js Implementation

```bash
# Install dependencies
npm install

# Start server
npm start
```

### Option B: Rust Implementation (Higher Performance)

```bash
# Build
cargo build --release

# Run
./target/release/mycelix-bootstrap-server
```

## Configuration

### Bootstrap Server Configuration

Create `bootstrap-server-config.yaml`:

```yaml
server:
  host: 0.0.0.0
  port: 8888

tls:
  enabled: true
  cert_path: /etc/letsencrypt/live/yourdomain/fullchain.pem
  key_path: /etc/letsencrypt/live/yourdomain/privkey.pem

storage:
  type: memory  # or "sqlite" for persistence
  sqlite_path: ./peers.db

peers:
  expiry_hours: 24
  max_per_space: 10000
  cleanup_interval_minutes: 30

rate_limiting:
  enabled: true
  requests_per_minute: 60
  requests_per_hour: 1000

metrics:
  enabled: true
  port: 9090

logging:
  level: info
  format: json
```

## Setting Up TLS (Required for Production)

### Using Let's Encrypt

```bash
# Install certbot
sudo apt install certbot

# Get certificate
sudo certbot certonly --standalone -d bootstrap.yourdomain.com

# Auto-renewal (add to crontab)
0 0 * * * /usr/bin/certbot renew --quiet
```

### Using Caddy as Reverse Proxy

```
# Caddyfile
bootstrap.yourdomain.com {
    reverse_proxy localhost:8888
}
```

## API Reference

### Health Check

```
GET /health
```

Response:
```json
{
  "status": "healthy",
  "peers": 1234,
  "uptime_seconds": 86400,
  "version": "1.0.0"
}
```

### Bootstrap Request

```
POST /bootstrap
Content-Type: application/json

{
  "space": "uhC0kxxxxxxxxx",
  "limit": 16
}
```

Response:
```json
{
  "peers": [
    {
      "agent_pub_key": "uhCAkxxxxxxxxx",
      "urls": ["wss://1.2.3.4:8888"],
      "signed_at_ms": 1704067200000,
      "expires_at_ms": 1704153600000
    }
  ]
}
```

### Register Peer

```
POST /peer
Content-Type: application/json

{
  "space": "uhC0kxxxxxxxxx",
  "agent_pub_key": "uhCAkxxxxxxxxx",
  "urls": ["wss://1.2.3.4:8888"],
  "signature": "base64-signature"
}
```

## Registering Your Bootstrap Node

Once your node is running and accessible, you can register it with the Mycelix network.

### 1. Verify Accessibility

```bash
# From a different machine
curl https://bootstrap.yourdomain.com/health
```

### 2. Submit for Inclusion

Open an issue at https://github.com/Mycelix/bootstrap-registry with:

- Your bootstrap URL
- Geographic region
- Your contact information
- SLA commitment (uptime target)

### 3. Add to Local Config

While waiting for official inclusion, you can add your server to your local config:

Edit `~/.mycelix/bootstrap_config.yaml`:

```yaml
bootstrap:
  secondary:
    - url: https://bootstrap.yourdomain.com
      weight: 70
      region: your-region
      operator: your-org-name
```

## Monitoring

### Prometheus Metrics

The server exposes metrics at `/metrics`:

```
# HELP bootstrap_peers_total Total peers registered
# TYPE bootstrap_peers_total counter
bootstrap_peers_total{space="uhC0k..."} 1234

# HELP bootstrap_requests_total Total bootstrap requests
# TYPE bootstrap_requests_total counter
bootstrap_requests_total{status="success"} 5678

# HELP bootstrap_request_latency_seconds Request latency
# TYPE bootstrap_request_latency_seconds histogram
bootstrap_request_latency_seconds_bucket{le="0.1"} 1000
```

### Grafana Dashboard

Import the provided dashboard from `monitoring/grafana-dashboard.json`.

### Alerting

Recommended alerts:

```yaml
# prometheus-alerts.yaml
groups:
  - name: bootstrap
    rules:
      - alert: BootstrapServerDown
        expr: up{job="bootstrap"} == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: Bootstrap server is down

      - alert: HighErrorRate
        expr: rate(bootstrap_requests_total{status="error"}[5m]) > 0.1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: High error rate on bootstrap server
```

## Security Best Practices

### 1. Network Security

```bash
# Allow only HTTPS
sudo ufw allow 443/tcp
sudo ufw enable
```

### 2. Rate Limiting

Always enable rate limiting to prevent abuse:

```yaml
rate_limiting:
  enabled: true
  requests_per_minute: 60
  requests_per_hour: 1000
  ban_duration_minutes: 60
```

### 3. Signature Verification

Enable signature verification to prevent spam:

```yaml
security:
  require_signatures: true
  signature_algorithm: ed25519
```

### 4. Regular Updates

```bash
# Pull latest updates
git pull
docker-compose pull
docker-compose up -d
```

## Troubleshooting

### Server Won't Start

```bash
# Check logs
docker-compose logs -f

# Check port availability
sudo lsof -i :8888
```

### Peers Not Being Discovered

1. Check your firewall allows incoming connections
2. Verify TLS certificate is valid
3. Ensure DNS is properly configured

### High Memory Usage

```bash
# Check peer count
curl http://localhost:8888/health | jq .peers

# If too high, reduce max_per_space
```

### Connection Timeouts

1. Check network connectivity
2. Verify no ISP blocking
3. Consider using a CDN for DDoS protection

## Community Support

- Discord: https://discord.gg/mycelix
- Forum: https://forum.mycelix.net
- GitHub Issues: https://github.com/Mycelix/bootstrap-server/issues

## Operator Checklist

Before going live, verify:

- [ ] TLS enabled with valid certificate
- [ ] Rate limiting configured
- [ ] Monitoring set up
- [ ] Backups configured (if using persistent storage)
- [ ] Auto-restart on failure (systemd or Docker restart policy)
- [ ] Firewall configured
- [ ] Health endpoint accessible
- [ ] Tested with real Mycelix clients

## Thank You

By running a bootstrap node, you're contributing to a more resilient and decentralized Mycelix network. Your contribution helps ensure that users can always connect, even if primary servers experience issues.

For questions or support, reach out on Discord or open a GitHub issue.
