# Mycelix Mail Deployment Guide

This guide covers deploying Mycelix Mail for production use, including infrastructure requirements, configuration, and operational best practices.

## Deployment Options

### 1. Self-Hosted (Full Control)

Run your own Holochain nodes and optional gateway server.

### 2. Hybrid (Gateway + P2P)

Use a hosted gateway API while maintaining P2P for direct communications.

### 3. Pure P2P (Desktop/Mobile)

Embedded Holochain conductor in client applications.

## Infrastructure Requirements

### Holochain Node

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 2 GB | 8 GB |
| Storage | 20 GB SSD | 100 GB NVMe |
| Network | 10 Mbps | 100+ Mbps |
| OS | Linux (Ubuntu 22.04+) | NixOS |

### Gateway Server (Optional)

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 8+ cores |
| RAM | 4 GB | 16 GB |
| Storage | 50 GB SSD | 200 GB NVMe |
| Network | 100 Mbps | 1 Gbps |

## Docker Deployment

### Quick Start with Docker Compose

```bash
cd /path/to/mycelix-mail
docker-compose -f docker/docker-compose.yml up -d
```

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  holochain:
    image: mycelix/holochain-conductor:latest
    container_name: mycelix-holochain
    restart: unless-stopped
    volumes:
      - holochain_data:/var/lib/holochain
      - ./holochain/conductor-config.yaml:/etc/holochain/conductor-config.yaml:ro
    ports:
      - "8888:8888"  # Admin API
      - "8889:8889"  # App API
    environment:
      - RUST_LOG=info
      - HOLOCHAIN_ADMIN_PORT=8888
      - HOLOCHAIN_APP_PORT=8889
    networks:
      - mycelix

  gateway:
    image: mycelix/gateway:latest
    container_name: mycelix-gateway
    restart: unless-stopped
    depends_on:
      - holochain
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - HOLOCHAIN_URL=ws://holochain:8889
      - API_KEY_SECRET=${API_KEY_SECRET}
      - DATABASE_URL=postgres://user:pass@postgres:5432/mycelix
    networks:
      - mycelix

  postgres:
    image: postgres:15-alpine
    container_name: mycelix-postgres
    restart: unless-stopped
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=mycelix
    networks:
      - mycelix

  redis:
    image: redis:7-alpine
    container_name: mycelix-redis
    restart: unless-stopped
    volumes:
      - redis_data:/data
    networks:
      - mycelix

volumes:
  holochain_data:
  postgres_data:
  redis_data:

networks:
  mycelix:
    driver: bridge
```

### Build Docker Images

```bash
# Build all images
make docker

# Build specific image
docker build -f docker/Dockerfile.holochain -t mycelix/holochain-conductor .
docker build -f docker/Dockerfile.gateway -t mycelix/gateway .
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.25+)
- kubectl configured
- Helm 3.x
- Ingress controller (nginx/traefik)

### Deploy with Helm

```bash
# Add Mycelix Helm repository
helm repo add mycelix https://charts.mycelix.mail
helm repo update

# Install with custom values
helm install mycelix-mail mycelix/mycelix-mail \
  --namespace mycelix \
  --create-namespace \
  --values values.production.yaml
```

### Custom Values

```yaml
# values.production.yaml
holochain:
  replicas: 3
  resources:
    requests:
      memory: "2Gi"
      cpu: "1000m"
    limits:
      memory: "8Gi"
      cpu: "4000m"
  persistence:
    enabled: true
    size: 100Gi
    storageClass: fast-ssd

gateway:
  replicas: 2
  resources:
    requests:
      memory: "512Mi"
      cpu: "500m"
    limits:
      memory: "2Gi"
      cpu: "2000m"
  ingress:
    enabled: true
    className: nginx
    hosts:
      - host: api.mycelix.example.com
        paths:
          - path: /
            pathType: Prefix
    tls:
      - secretName: mycelix-tls
        hosts:
          - api.mycelix.example.com

postgres:
  enabled: true
  auth:
    existingSecret: mycelix-postgres-secret
  primary:
    persistence:
      size: 50Gi

redis:
  enabled: true
  auth:
    enabled: true
    existingSecret: mycelix-redis-secret
```

## Conductor Configuration

### Production Conductor Config

```yaml
# conductor-config.yaml
environment_path: /var/lib/holochain
keystore:
  type: lair_server_in_proc

admin_interfaces:
  - driver:
      type: websocket
      port: 8888
      allowed_origins:
        - "localhost"

network:
  bootstrap_service: https://bootstrap.holo.host
  transport_pool:
    - type: webrtc
      signal_url: wss://signal.holo.host

tuning_params:
  gossip_loop_iteration_delay_ms: 100
  gossip_peer_on_success_next_gossip_delay_ms: 500
  gossip_peer_on_error_next_gossip_delay_ms: 5000
  gossip_round_redundant_check_interval_ms: 10000
  gossip_timeout_ms: 60000
```

### Installing hApps

```bash
# Install the Mycelix Mail hApp
hc sandbox call admin-port 8888 install-app \
  --app-id mycelix-mail \
  --bundle-path ./mycelix-mail.happ \
  --membrane-proofs '{}'

# Enable the app
hc sandbox call admin-port 8888 enable-app \
  --app-id mycelix-mail
```

## Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| `HOLOCHAIN_ADMIN_PORT` | Holochain admin API port |
| `HOLOCHAIN_APP_PORT` | Holochain app API port |
| `API_KEY_SECRET` | Secret for API key generation |
| `DATABASE_URL` | PostgreSQL connection string |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `RUST_LOG` | `info` | Rust logging level |
| `NODE_ENV` | `development` | Node environment |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection |
| `MAX_UPLOAD_SIZE` | `25MB` | Maximum attachment size |
| `TRUST_CACHE_TTL` | `3600` | Trust score cache TTL (seconds) |

## TLS/SSL Configuration

### Using Let's Encrypt

```bash
# Install certbot
apt install certbot python3-certbot-nginx

# Obtain certificate
certbot --nginx -d api.mycelix.example.com

# Auto-renewal
systemctl enable certbot.timer
```

### Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/mycelix
server {
    listen 443 ssl http2;
    server_name api.mycelix.example.com;

    ssl_certificate /etc/letsencrypt/live/api.mycelix.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.mycelix.example.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;

    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket for Holochain
    location /holochain {
        proxy_pass http://127.0.0.1:8889;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
```

## Monitoring & Observability

### Prometheus Metrics

The gateway exposes Prometheus metrics at `/metrics`:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'mycelix-gateway'
    static_configs:
      - targets: ['gateway:3000']
    metrics_path: /metrics

  - job_name: 'mycelix-holochain'
    static_configs:
      - targets: ['holochain:9090']
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| `mycelix_emails_sent_total` | Total emails sent |
| `mycelix_emails_received_total` | Total emails received |
| `mycelix_trust_attestations_total` | Total trust attestations |
| `mycelix_api_request_duration_seconds` | API request latency |
| `holochain_dht_ops_total` | DHT operations count |

### Grafana Dashboards

Import pre-built dashboards from `monitoring/grafana/`:

```bash
# Copy dashboards
cp monitoring/grafana/dashboards/*.json /var/lib/grafana/dashboards/
```

### Logging

Configure structured logging:

```yaml
# logging.yaml
level: info
format: json
outputs:
  - type: stdout
  - type: file
    path: /var/log/mycelix/gateway.log
    rotate:
      max_size: 100MB
      max_age: 30d
      compress: true
```

## Backup & Recovery

### Automated Backups

```bash
# Create backup script
cat > /opt/mycelix/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR=/backups/mycelix
DATE=$(date +%Y%m%d_%H%M%S)

# Backup Holochain data
tar -czf $BACKUP_DIR/holochain_$DATE.tar.gz /var/lib/holochain

# Backup PostgreSQL
pg_dump -h localhost -U mycelix mycelix | gzip > $BACKUP_DIR/postgres_$DATE.sql.gz

# Upload to S3
aws s3 cp $BACKUP_DIR/holochain_$DATE.tar.gz s3://mycelix-backups/
aws s3 cp $BACKUP_DIR/postgres_$DATE.sql.gz s3://mycelix-backups/

# Cleanup old local backups (keep 7 days)
find $BACKUP_DIR -mtime +7 -delete
EOF

chmod +x /opt/mycelix/backup.sh

# Schedule daily backup
echo "0 2 * * * /opt/mycelix/backup.sh" | crontab -
```

### Disaster Recovery

1. **Deploy fresh infrastructure** using Terraform/Kubernetes
2. **Restore Holochain data** from latest backup
3. **Restore PostgreSQL** from SQL dump
4. **Verify DHT sync** with other nodes
5. **Run health checks** before routing traffic

## Security Hardening

### Firewall Rules

```bash
# UFW configuration
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 443/tcp       # HTTPS
ufw allow 8888/tcp      # Holochain admin (local only)
ufw allow 8889/tcp      # Holochain app
ufw enable
```

### Rate Limiting

Configure rate limiting in the gateway:

```yaml
# rate-limit.yaml
global:
  requests_per_minute: 1000
  burst: 100

endpoints:
  /v1/emails/send:
    requests_per_minute: 100
    burst: 10
  /v1/trust/attestations:
    requests_per_minute: 50
    burst: 5
```

### Security Headers

```nginx
# Security headers
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Content-Security-Policy "default-src 'self'" always;
```

## Scaling

### Horizontal Scaling

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mycelix-gateway-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mycelix-gateway
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

### Load Balancing

- Use consistent hashing for Holochain agent affinity
- WebSocket sticky sessions for real-time connections
- Geographic distribution for latency optimization

## Troubleshooting

### Common Issues

**Holochain not syncing:**
```bash
# Check peer connectivity
hc sandbox call admin-port 8888 list-dnas
hc sandbox call admin-port 8888 agent-info

# Check network status
curl -s http://localhost:9090/metrics | grep holochain_network
```

**High memory usage:**
```bash
# Increase swap
fallocate -l 4G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile

# Tune conductor GC
export RUST_MEMORY_LIMIT=4096
```

**API errors:**
```bash
# Check gateway logs
docker logs mycelix-gateway --tail 100

# Test Holochain connection
websocat ws://localhost:8889
```

## Next Steps

- [Getting Started](./getting-started.md) - Development setup
- [Architecture](./architecture.md) - System design
- [API Reference](/docs/api/) - Complete API documentation
