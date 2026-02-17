# Production Deployment Guide

Step-by-step guide for deploying Mycelix Supply Chain to production.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Deployment Options](#deployment-options)
  - [Docker (Standalone)](#docker-standalone)
  - [Docker Compose](#docker-compose)
  - [Kubernetes](#kubernetes)
  - [Cloud Platforms](#cloud-platforms)
- [Configuration](#configuration)
- [Database Setup](#database-setup)
- [TLS/SSL](#tlsssl)
- [Reverse Proxy](#reverse-proxy)
- [Monitoring](#monitoring)
- [Security](#security)
- [Scaling](#scaling)
- [Backup & Recovery](#backup--recovery)
- [Troubleshooting](#troubleshooting)
- [Production Checklist](#production-checklist)

## Prerequisites

### Required
- Docker 24.0+ (for containerized deployment)
- 2GB RAM minimum (4GB+ recommended)
- 20GB disk space (for database and logs)
- Linux/macOS/Windows with WSL2

### Optional
- Kubernetes 1.28+ (for K8s deployment)
- PostgreSQL 14+ (alternative to SQLite)
- Prometheus & Grafana (for monitoring)
- Let's Encrypt (for free TLS certificates)

## Environment Setup

### 1. Create Data Directory

```bash
mkdir -p /var/lib/mycelix-supplychain/{data,logs,backups}
chmod 750 /var/lib/mycelix-supplychain
```

### 2. Set Environment Variables

Create `.env` file:

```bash
# Service Configuration
PORT=8080
HOST=0.0.0.0
LOG_LEVEL=info

# Database
DATABASE_URL=sqlite:///var/lib/mycelix-supplychain/data/claims.db
# Or for PostgreSQL:
# DATABASE_URL=postgres://user:password@localhost:5432/supplychain

# Cryptography
KEYPAIR_SEED=<64-character-hex-seed>  # Generate with: supplychain keygen

# Optional: Future features
# API_KEY_SALT=<random-string>
# WEBHOOK_SECRET=<random-string>
```

### 3. Generate Keypair

```bash
./supplychain keygen --output /var/lib/mycelix-supplychain/keypair.json
```

Extract seed from `keypair.json` and add to `.env`.

## Deployment Options

### Docker (Standalone)

#### 1. Build Image

```bash
cd rust/service
docker build -t mycelix-supplychain:latest .
```

#### 2. Run Container

```bash
docker run -d \
  --name supplychain \
  -p 8080:8080 \
  -v /var/lib/mycelix-supplychain/data:/data \
  -v /var/lib/mycelix-supplychain/logs:/logs \
  --env-file .env \
  --restart unless-stopped \
  mycelix-supplychain:latest
```

#### 3. Verify

```bash
curl http://localhost:8080/health
```

### Docker Compose

#### 1. Create `docker-compose.prod.yml`

```yaml
version: '3.8'

services:
  supplychain:
    image: mycelix-supplychain:latest
    build:
      context: .
      dockerfile: rust/service/Dockerfile
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=postgres://supplychain:${DB_PASSWORD}@postgres:5432/supplychain
      - PORT=8080
      - HOST=0.0.0.0
      - LOG_LEVEL=info
    volumes:
      - ./data:/data
      - ./logs:/logs
    depends_on:
      - postgres
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_DB=supplychain
      - POSTGRES_USER=supplychain
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U supplychain"]
      interval: 10s
      timeout: 5s
      retries: 5

  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on:
      - supplychain
    restart: unless-stopped

volumes:
  postgres_data:
```

#### 2. Deploy

```bash
export DB_PASSWORD=$(openssl rand -hex 32)
docker-compose -f docker-compose.prod.yml up -d
```

#### 3. Check Logs

```bash
docker-compose -f docker-compose.prod.yml logs -f supplychain
```

### Kubernetes

#### 1. Create Namespace

```bash
kubectl create namespace mycelix-supplychain
```

#### 2. Create Secrets

```bash
kubectl create secret generic supplychain-secrets \
  --from-literal=database-url="postgres://..." \
  --from-literal=keypair-seed="<seed>" \
  -n mycelix-supplychain
```

#### 3. Deploy with Manifests

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: supplychain
  namespace: mycelix-supplychain
spec:
  replicas: 3
  selector:
    matchLabels:
      app: supplychain
  template:
    metadata:
      labels:
        app: supplychain
    spec:
      containers:
      - name: supplychain
        image: mycelix-supplychain:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: supplychain-secrets
              key: database-url
        - name: PORT
          value: "8080"
        - name: LOG_LEVEL
          value: "info"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10

---
apiVersion: v1
kind: Service
metadata:
  name: supplychain
  namespace: mycelix-supplychain
spec:
  selector:
    app: supplychain
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: supplychain
  namespace: mycelix-supplychain
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: supplychain
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

#### 4. Apply

```bash
kubectl apply -f deployment.yaml
```

#### 5. Verify

```bash
kubectl get pods -n mycelix-supplychain
kubectl get svc -n mycelix-supplychain
```

### Cloud Platforms

#### AWS ECS

```bash
# 1. Create ECR repository
aws ecr create-repository --repository-name mycelix-supplychain

# 2. Build and push image
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

docker build -t mycelix-supplychain .
docker tag mycelix-supplychain:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/mycelix-supplychain:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/mycelix-supplychain:latest

# 3. Create ECS task definition and service via AWS Console or Terraform
```

#### Google Cloud Run

```bash
# 1. Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/mycelix-supplychain

# 2. Deploy
gcloud run deploy supplychain \
  --image gcr.io/PROJECT_ID/mycelix-supplychain \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars DATABASE_URL="<url>"
```

#### Azure Container Instances

```bash
# 1. Create container registry
az acr create --resource-group myResourceGroup --name mycelixRegistry --sku Basic

# 2. Build and push
az acr build --registry mycelixRegistry --image mycelix-supplychain:latest .

# 3. Deploy
az container create \
  --resource-group myResourceGroup \
  --name supplychain \
  --image mycelixRegistry.azurecr.io/mycelix-supplychain:latest \
  --dns-name-label mycelix-supplychain \
  --ports 8080 \
  --environment-variables DATABASE_URL="<url>"
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8080 | HTTP server port |
| `HOST` | 0.0.0.0 | Bind address |
| `DATABASE_URL` | sqlite://data/claims.db | Database connection string |
| `LOG_LEVEL` | info | Logging level (trace, debug, info, warn, error) |
| `RUST_LOG` | info | Rust logging (overrides LOG_LEVEL) |
| `KEYPAIR_SEED` | (generated) | Ed25519 keypair seed (64 hex chars) |

### Logging

Configure structured logging:

```bash
export RUST_LOG=info,supplychain=debug,sqlx=warn
```

Log to file:

```bash
export LOG_FILE=/var/log/supplychain/service.log
```

## Database Setup

### SQLite (Development/Small Scale)

```bash
# 1. Create database directory
mkdir -p /var/lib/mycelix-supplychain/data

# 2. Set DATABASE_URL
export DATABASE_URL=sqlite:///var/lib/mycelix-supplychain/data/claims.db

# 3. Run migrations
./supplychain db migrate

# 4. Verify
./supplychain db check
```

**Pros**: Simple, no separate server, perfect for <10k events/day
**Cons**: No concurrent writes, single server only

### PostgreSQL (Production/High Scale)

```bash
# 1. Install PostgreSQL
sudo apt-get install postgresql-16

# 2. Create database and user
sudo -u postgres psql
CREATE DATABASE supplychain;
CREATE USER supplychain WITH ENCRYPTED PASSWORD 'secure-password';
GRANT ALL PRIVILEGES ON DATABASE supplychain TO supplychain;

# 3. Configure connection
export DATABASE_URL=postgres://supplychain:secure-password@localhost:5432/supplychain

# 4. Run migrations
./supplychain db migrate
```

**Pros**: Concurrent writes, replication, better performance at scale
**Cons**: More complex setup, requires separate server

### Database Tuning (PostgreSQL)

```sql
-- postgresql.conf optimizations
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
work_mem = 4MB
min_wal_size = 1GB
max_wal_size = 4GB
```

## TLS/SSL

### Option 1: Let's Encrypt (Recommended)

```bash
# 1. Install certbot
sudo apt-get install certbot

# 2. Generate certificate
sudo certbot certonly --standalone -d api.example.com

# 3. Certificates will be in /etc/letsencrypt/live/api.example.com/
```

### Option 2: Self-Signed (Development Only)

```bash
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout key.pem -out cert.pem -days 365 \
  -subj "/CN=localhost"
```

### Configure Nginx with TLS

```nginx
server {
    listen 443 ssl http2;
    server_name api.example.com;

    ssl_certificate /etc/letsencrypt/live/api.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.example.com/privkey.pem;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

server {
    listen 80;
    server_name api.example.com;
    return 301 https://$server_name$request_uri;
}
```

## Reverse Proxy

### Nginx Configuration

```nginx
upstream supplychain_backend {
    least_conn;
    server 127.0.0.1:8080 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8081 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8082 max_fails=3 fail_timeout=30s;
}

server {
    listen 443 ssl http2;
    server_name api.example.com;

    # TLS configuration
    ssl_certificate /etc/letsencrypt/live/api.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.example.com/privkey.pem;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req zone=api_limit burst=20 nodelay;

    # Timeouts
    proxy_connect_timeout 60s;
    proxy_send_timeout 60s;
    proxy_read_timeout 60s;

    location / {
        proxy_pass http://supplychain_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    # Health check endpoint (don't rate limit)
    location /health {
        proxy_pass http://supplychain_backend;
        limit_req off;
    }
}
```

### Caddy Configuration (Alternative)

```caddy
api.example.com {
    reverse_proxy localhost:8080 localhost:8081 localhost:8082 {
        lb_policy least_conn
        health_uri /health
        health_interval 30s
    }

    @api {
        path /v1/*
    }

    rate_limit @api {
        zone api_limit 10MB
        rate 100r/m
    }

    encode gzip

    header {
        X-Frame-Options DENY
        X-Content-Type-Options nosniff
        Strict-Transport-Security "max-age=31536000"
    }
}
```

## Monitoring

### Health Checks

```bash
# Kubernetes
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 30

# Docker Compose
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

### Prometheus Metrics (Future Feature)

Once metrics are implemented:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'supplychain'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Grafana Dashboard (Future)

Import dashboard template: `docs/grafana-dashboard.json`

### Logging Best Practices

```bash
# Structured JSON logging
export LOG_FORMAT=json

# Log rotation (logrotate config)
/var/log/supplychain/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0640 supplychain supplychain
    sharedscripts
    postrotate
        systemctl reload supplychain
    endscript
}
```

## Security

### Network Security

```bash
# Firewall (ufw)
sudo ufw allow 443/tcp
sudo ufw allow 80/tcp
sudo ufw deny 8080/tcp  # Only allow internal access
sudo ufw enable

# iptables
iptables -A INPUT -p tcp --dport 443 -j ACCEPT
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 8080 -s 127.0.0.1 -j ACCEPT
iptables -A INPUT -p tcp --dport 8080 -j DROP
```

### Secrets Management

#### Option 1: Environment Variables

```bash
# .env (never commit!)
DATABASE_URL=postgres://user:$(cat /run/secrets/db_password)@db:5432/supplychain
```

#### Option 2: Docker Secrets

```yaml
secrets:
  db_password:
    external: true

services:
  supplychain:
    secrets:
      - db_password
    environment:
      - DATABASE_PASSWORD_FILE=/run/secrets/db_password
```

#### Option 3: Kubernetes Secrets

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: supplychain-secrets
type: Opaque
data:
  database-url: <base64-encoded-url>
```

### Regular Updates

```bash
# Update base image monthly
docker pull rust:1.78-alpine
docker build --pull -t mycelix-supplychain:latest .

# Apply security patches
apt-get update && apt-get upgrade -y
```

## Scaling

### Horizontal Scaling

Run multiple instances behind a load balancer:

```bash
# Start 3 instances
for port in 8080 8081 8082; do
  PORT=$port DATABASE_URL="<shared-db>" ./provenance-service &
done
```

Requirements:
- Shared database (PostgreSQL, not SQLite)
- Stateless application (✅ Already stateless)
- Load balancer (Nginx, HAProxy, ALB)

### Vertical Scaling

Increase resources per instance:

```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "1000m"
  limits:
    memory: "2Gi"
    cpu: "2000m"
```

### Database Scaling

#### Read Replicas

```yaml
# PostgreSQL streaming replication
PRIMARY_DB=postgres://user:pass@primary:5432/db
REPLICA_DB=postgres://user:pass@replica:5432/db

# Route reads to replica (future feature)
```

#### Connection Pooling

```bash
# PgBouncer
[databases]
supplychain = host=localhost dbname=supplychain

[pgbouncer]
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 20
```

## Backup & Recovery

### SQLite Backups

```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR=/var/lib/mycelix-supplychain/backups
DATE=$(date +%Y%m%d_%H%M%S)

sqlite3 /var/lib/mycelix-supplychain/data/claims.db ".backup $BACKUP_DIR/claims_$DATE.db"
gzip $BACKUP_DIR/claims_$DATE.db

# Keep last 30 days
find $BACKUP_DIR -name "claims_*.db.gz" -mtime +30 -delete
```

Add to crontab:
```bash
0 2 * * * /usr/local/bin/backup-supplychain.sh
```

### PostgreSQL Backups

```bash
# pg_dump backup
pg_dump -U supplychain -h localhost -F c -f /backups/supplychain_$(date +%Y%m%d).dump supplychain

# Restore
pg_restore -U supplychain -h localhost -d supplychain /backups/supplychain_20251115.dump
```

### Backup to S3

```bash
# Upload to S3
aws s3 cp /backups/claims_latest.db.gz s3://mycelix-backups/$(date +%Y/%m/%d)/

# Lifecycle policy (delete after 90 days)
```

## Troubleshooting

### Service Won't Start

```bash
# Check logs
journalctl -u supplychain -n 100 --no-pager

# Check port availability
sudo lsof -i :8080

# Verify database connection
./supplychain db check
```

### High Memory Usage

```bash
# Check container stats
docker stats supplychain

# Reduce connection pool size
export SQLX_MAX_CONNECTIONS=10

# Enable memory limits
docker run --memory="512m" --memory-swap="1g" ...
```

### Slow API Responses

```bash
# Check database performance
./supplychain db stats

# Enable query logging
export RUST_LOG=sqlx::query=debug

# Add database indexes (already included in migrations)
```

### Database Locked (SQLite)

```bash
# Check for long-running transactions
fuser /var/lib/mycelix-supplychain/data/claims.db

# Increase timeout
export SQLITE_BUSY_TIMEOUT=30000  # 30 seconds
```

## Production Checklist

### Pre-Deployment

- [ ] Environment variables configured
- [ ] Database migrations run
- [ ] Keypair generated and secured
- [ ] TLS certificates obtained
- [ ] Firewall rules configured
- [ ] Backup strategy implemented
- [ ] Monitoring configured
- [ ] Load testing completed

### Post-Deployment

- [ ] Health check passing
- [ ] Metrics being collected
- [ ] Logs being aggregated
- [ ] Backups running successfully
- [ ] SSL/TLS configured correctly
- [ ] Rate limiting working
- [ ] Documentation updated

### Monthly Maintenance

- [ ] Review logs for errors
- [ ] Check disk space
- [ ] Verify backups
- [ ] Update dependencies
- [ ] Review security patches
- [ ] Test disaster recovery

---

## Next Steps

1. Review [API Usage Guide](api-guide.md) for integration details
2. Set up [monitoring](../examples/monitoring/) (when available)
3. Configure [CI/CD](.github/workflows/) for automated deployments
4. Join our [Community Discord](https://discord.gg/mycelix) for support

---

**Version**: 1.0.0
**Last Updated**: 2025-11-15
**Maintainer**: Luminous Dynamics DevOps Team
