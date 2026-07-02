# Production Deployment Guide

Complete guide for deploying Mycelix-DeSci in production environments.

## Table of Contents

- [Deployment Options](#deployment-options)
- [Quick Start (Docker)](#quick-start-docker)
- [NixOS Deployment](#nixos-deployment)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Cloud Platform Deployment](#cloud-platform-deployment)
- [Security Hardening](#security-hardening)
- [Monitoring & Observability](#monitoring--observability)
- [Backup & Recovery](#backup--recovery)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)

---

## Deployment Options

Mycelix-DeSci supports multiple deployment strategies:

| Option | Complexity | Reproducibility | Best For |
|--------|-----------|-----------------|----------|
| **Docker Compose** | Low | Medium | Development, small deployments |
| **NixOS** | Medium | **Highest** | Scientific reproducibility, production |
| **Kubernetes** | High | Medium | Large scale, high availability |
| **Cloud VMs** | Low | Low | Quick cloud deployment |

**Recommendation:** Use **NixOS** for maximum reproducibility (essential for science) or **Kubernetes** for scalability.

---

## Quick Start (Docker)

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 2GB RAM minimum
- 10GB disk space

### 1. Clone Repository

```bash
git clone https://github.com/Luminous-Dynamics/mycelix-desci
cd mycelix-desci
```

### 2. Start Services

```bash
docker-compose up -d
```

### 3. Verify Deployment

```bash
# Check health
curl http://localhost:8080/health

# View logs
docker-compose logs -f

# Check status
docker-compose ps
```

### 4. Access API

- **API Base:** `http://localhost:8080/api/v1`
- **Health Check:** `http://localhost:8080/health`
- **API Docs:** `http://localhost:8080/docs`

---

## NixOS Deployment

### Why NixOS for Science?

NixOS provides **bit-for-bit reproducibility**, essential for scientific research:

- 🔒 **Reproducible builds** - Same inputs → same outputs, always
- 📦 **Declarative configuration** - Infrastructure as code
- ⏪ **Atomic rollbacks** - Safe deployments
- 🔐 **Strong isolation** - Security by default

See [NIX.md](NIX.md) for complete NixOS guide.

### Quick NixOS Deployment

#### 1. Add to Configuration

Edit `/etc/nixos/configuration.nix`:

```nix
{
  # Import Mycelix-DeSci flake
  inputs.mycelix-desci.url = "github:Luminous-Dynamics/mycelix-desci";

  # Add to configuration
  imports = [
    inputs.mycelix-desci.nixosModules.default
  ];

  # Enable and configure service
  services.mycelix-desci = {
    enable = true;
    port = 8080;
    host = "0.0.0.0";
    logLevel = "info";
    corsOrigins = "*";
    openFirewall = true;
  };
}
```

#### 2. Rebuild System

```bash
sudo nixos-rebuild switch
```

#### 3. Verify Service

```bash
# Check service status
sudo systemctl status mycelix-api

# View logs
sudo journalctl -u mycelix-api -f

# Test API
curl http://localhost:8080/health
```

### NixOS Configuration Options

```nix
services.mycelix-desci = {
  enable = true;                  # Enable the service

  # Network settings
  port = 8080;                    # API port (default: 8080)
  host = "0.0.0.0";               # Bind address (default: 0.0.0.0)
  openFirewall = true;            # Open firewall port automatically

  # Logging
  logLevel = "info";              # trace|debug|info|warn|error

  # CORS
  corsOrigins = "*";              # Allowed origins (use specific domains in prod)

  # Service management
  user = "mycelix";               # Service user (default: mycelix)
  group = "mycelix";              # Service group (default: mycelix)
  dataDir = "/var/lib/mycelix";   # Data directory

  # Additional environment
  extraEnvironment = {
    CUSTOM_VAR = "value";
  };
};
```

### NixOS Service Management

```bash
# Start/stop/restart
sudo systemctl start mycelix-api
sudo systemctl stop mycelix-api
sudo systemctl restart mycelix-api

# Enable at boot
sudo systemctl enable mycelix-api

# View status
sudo systemctl status mycelix-api

# View logs
sudo journalctl -u mycelix-api -f        # Follow logs
sudo journalctl -u mycelix-api --since today  # Today's logs
sudo journalctl -u mycelix-api -n 100    # Last 100 lines
```

---

## Docker Deployment

### Production Docker Compose

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  api:
    image: mycelix-api:latest
    build:
      context: .
      dockerfile: Dockerfile
    restart: unless-stopped
    ports:
      - "8080:8080"
    environment:
      - PORT=8080
      - RUST_LOG=mycelix_api=info
      - CORS_ORIGINS=https://your-domain.com
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### Deploy with Production Compose

```bash
# Build and start
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Scale if needed
docker-compose -f docker-compose.prod.yml up -d --scale api=3

# Update deployment
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d
```

### Docker with Reverse Proxy (Nginx)

Add to `docker-compose.prod.yml`:

```yaml
  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
```

Create `nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream api {
        server api:8080;
    }

    server {
        listen 80;
        server_name your-domain.com;

        # Redirect to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        # SSL certificates
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        # SSL configuration
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;
        ssl_prefer_server_ciphers on;

        # Proxy to API
        location / {
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }
    }
}
```

---

## Kubernetes Deployment

### Prerequisites

- Kubernetes 1.20+
- kubectl configured
- Persistent storage provisioner

### 1. Create Namespace

```bash
kubectl create namespace mycelix-desci
```

### 2. Deployment Manifest

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mycelix-api
  namespace: mycelix-desci
  labels:
    app: mycelix-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mycelix-api
  template:
    metadata:
      labels:
        app: mycelix-api
    spec:
      containers:
      - name: api
        image: mycelix-api:latest
        ports:
        - containerPort: 8080
          name: http
        env:
        - name: PORT
          value: "8080"
        - name: RUST_LOG
          value: "mycelix_api=info"
        - name: CORS_ORIGINS
          value: "*"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
```

### 3. Service Manifest

Create `k8s/service.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: mycelix-api
  namespace: mycelix-desci
spec:
  selector:
    app: mycelix-api
  ports:
  - port: 80
    targetPort: 8080
    name: http
  type: LoadBalancer
```

### 4. Ingress (Optional)

Create `k8s/ingress.yaml`:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mycelix-api
  namespace: mycelix-desci
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.your-domain.com
    secretName: mycelix-tls
  rules:
  - host: api.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: mycelix-api
            port:
              number: 80
```

### 5. Deploy to Kubernetes

```bash
# Apply manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# Check status
kubectl get pods -n mycelix-desci
kubectl get svc -n mycelix-desci
kubectl get ing -n mycelix-desci

# View logs
kubectl logs -f -n mycelix-desci deployment/mycelix-api

# Scale deployment
kubectl scale deployment mycelix-api -n mycelix-desci --replicas=5
```

### 6. Auto-scaling (HPA)

Create `k8s/hpa.yaml`:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mycelix-api
  namespace: mycelix-desci
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mycelix-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

Apply:

```bash
kubectl apply -f k8s/hpa.yaml
```

---

## Cloud Platform Deployment

### AWS EC2

#### 1. Launch EC2 Instance

```bash
# Launch Ubuntu 22.04 instance
# Instance type: t3.medium (2 vCPU, 4 GB RAM)
# Security group: Allow inbound TCP 8080, 22
```

#### 2. SSH and Install Dependencies

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker ubuntu

# Clone repository
git clone https://github.com/Luminous-Dynamics/mycelix-desci
cd mycelix-desci

# Start with Docker Compose
docker-compose up -d
```

#### 3. Configure Security Group

- Allow inbound TCP port 8080
- Optionally use Elastic IP for static address

### Google Cloud Platform (GCP)

#### 1. Create Compute Engine Instance

```bash
gcloud compute instances create mycelix-api \
  --machine-type=e2-medium \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --tags=http-server,https-server \
  --metadata-from-file startup-script=startup.sh
```

Create `startup.sh`:

```bash
#!/bin/bash
curl -fsSL https://get.docker.com | sh
git clone https://github.com/Luminous-Dynamics/mycelix-desci /opt/mycelix-desci
cd /opt/mycelix-desci
docker-compose up -d
```

#### 2. Create Firewall Rule

```bash
gcloud compute firewall-rules create allow-mycelix-api \
  --allow tcp:8080 \
  --target-tags http-server
```

### Azure VM

#### 1. Create Resource Group

```bash
az group create --name mycelix-rg --location eastus
```

#### 2. Create VM

```bash
az vm create \
  --resource-group mycelix-rg \
  --name mycelix-vm \
  --image Ubuntu2204 \
  --size Standard_B2s \
  --admin-username azureuser \
  --generate-ssh-keys
```

#### 3. Open Port

```bash
az vm open-port \
  --resource-group mycelix-rg \
  --name mycelix-vm \
  --port 8080
```

#### 4. Install and Run

```bash
# SSH into VM
ssh azureuser@<vm-ip>

# Install Docker and deploy
curl -fsSL https://get.docker.com | sh
git clone https://github.com/Luminous-Dynamics/mycelix-desci
cd mycelix-desci
docker-compose up -d
```

---

## Security Hardening

### TLS/SSL Configuration

#### Using Let's Encrypt with Certbot

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo certbot renew --dry-run
```

#### Manual Certificate Setup

```bash
# Generate self-signed certificate (development only)
openssl req -x509 -newkey rsa:4096 \
  -keyout key.pem -out cert.pem \
  -days 365 -nodes

# Use in nginx configuration
```

### Firewall Configuration

#### UFW (Ubuntu)

```bash
# Enable UFW
sudo ufw enable

# Allow SSH
sudo ufw allow 22/tcp

# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow API (if direct access needed)
sudo ufw allow 8080/tcp

# Check status
sudo ufw status
```

#### iptables

```bash
# Allow established connections
sudo iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# Allow SSH
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# Allow HTTP/HTTPS
sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Allow API
sudo iptables -A INPUT -p tcp --dport 8080 -j ACCEPT

# Drop all other inbound
sudo iptables -P INPUT DROP

# Save rules
sudo iptables-save > /etc/iptables/rules.v4
```

### Environment Variables Security

Never commit secrets to git. Use environment files:

Create `.env.production`:

```bash
PORT=8080
RUST_LOG=mycelix_api=info
CORS_ORIGINS=https://your-domain.com
# Add other secrets here
```

```bash
# Load environment
set -a
source .env.production
set +a

# Or use with Docker
docker-compose --env-file .env.production up -d
```

### Input Validation

The API already includes:
- ✅ Request size limits (10MB max)
- ✅ Timeout protection (30s)
- ✅ JSON schema validation
- ✅ Parameter sanitization

### Security Headers

Add to nginx configuration:

```nginx
# Security headers
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "no-referrer-when-downgrade" always;
add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
```

---

## Monitoring & Observability

### Health Checks

The API provides built-in health endpoints:

```bash
# Basic health
curl http://localhost:8080/health

# System metrics
curl http://localhost:8080/api/v1/system/metrics

# Version info
curl http://localhost:8080/api/v1/system/version
```

### Logging

#### Structured Logging

The API uses `tracing` for structured logging:

```bash
# Set log level via environment
export RUST_LOG=mycelix_api=debug,tower_http=debug

# Log to file
cargo run --release 2>&1 | tee mycelix-api.log
```

#### Log Aggregation with Loki

Docker Compose with Loki:

```yaml
version: '3.8'

services:
  api:
    # ... existing config ...
    logging:
      driver: loki
      options:
        loki-url: "http://localhost:3100/loki/api/v1/push"

  loki:
    image: grafana/loki:latest
    ports:
      - "3100:3100"
    volumes:
      - loki-data:/loki

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana

volumes:
  loki-data:
  grafana-data:
```

### Metrics with Prometheus

**Note:** Prometheus metrics endpoint is planned for Phase 7.

Future metrics will include:
- Request rate (requests/sec)
- Response time (p50, p95, p99)
- Error rate
- Claim creation rate
- Query execution time
- Trust score updates

Prometheus configuration example:

```yaml
scrape_configs:
  - job_name: 'mycelix-api'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Alerting

Example Prometheus alert rules:

```yaml
groups:
  - name: mycelix_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"

      - alert: APIDown
        expr: up{job="mycelix-api"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Mycelix API is down"

      - alert: HighResponseTime
        expr: http_request_duration_seconds{quantile="0.95"} > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time (p95 > 1s)"
```

---

## Backup & Recovery

### Data Backup Strategy

Currently, Mycelix-DeSci uses in-memory storage. For production:

**Planned for Phase 6:**
- Persistent storage backends (IPFS, Arweave)
- Database snapshots
- Incremental backups

### Configuration Backup

Backup critical files:

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/mycelix-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup configuration
cp docker-compose.yml "$BACKUP_DIR/"
cp .env.production "$BACKUP_DIR/"

# Backup nginx config
cp nginx.conf "$BACKUP_DIR/"

# Backup SSL certificates
cp -r ssl "$BACKUP_DIR/"

# Create tarball
tar -czf "$BACKUP_DIR.tar.gz" "$BACKUP_DIR"
rm -rf "$BACKUP_DIR"

echo "Backup created: $BACKUP_DIR.tar.gz"
```

### Disaster Recovery Plan

#### RTO/RPO Targets

- **Recovery Time Objective (RTO):** < 1 hour
- **Recovery Point Objective (RPO):** < 15 minutes

#### Recovery Procedure

1. **Restore configuration:**
   ```bash
   tar -xzf backup.tar.gz
   cd backup-*/
   ```

2. **Redeploy service:**
   ```bash
   docker-compose up -d
   # or
   sudo nixos-rebuild switch
   ```

3. **Verify health:**
   ```bash
   curl http://localhost:8080/health
   ```

---

## Performance Tuning

### Resource Allocation

**Minimum Requirements:**
- CPU: 1 core
- RAM: 1 GB
- Disk: 10 GB

**Recommended (Production):**
- CPU: 2-4 cores
- RAM: 4-8 GB
- Disk: 50 GB SSD

### Docker Resource Limits

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### Rust Optimization

Build with maximum optimization:

```bash
cargo build --release

# With additional optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

### Concurrency Tuning

Tokio runtime configuration (handled automatically):

- Default worker threads: Number of CPU cores
- Max blocking threads: 512

For custom tuning, set environment:

```bash
# Set worker threads explicitly
TOKIO_WORKER_THREADS=4 cargo run --release
```

### Database Connection Pooling

**Planned for Phase 6** when persistent storage is added:

- Connection pool size: 10-20
- Idle timeout: 10 minutes
- Connection timeout: 30 seconds

---

## Troubleshooting

### Common Issues

#### Port Already in Use

```bash
# Find process using port 8080
sudo lsof -i :8080

# Kill process
sudo kill -9 <PID>

# Or change port
PORT=8081 cargo run --release
```

#### Docker Container Won't Start

```bash
# Check logs
docker-compose logs api

# Remove and recreate
docker-compose down
docker-compose up -d --force-recreate

# Check Docker daemon
sudo systemctl status docker
```

#### High Memory Usage

```bash
# Check memory usage
docker stats

# Limit container memory
docker run -m 2g mycelix-api

# Or in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 2G
```

#### Slow Response Times

```bash
# Check system resources
top
htop

# Check API metrics
curl http://localhost:8080/api/v1/system/metrics

# Enable debug logging
RUST_LOG=debug cargo run --release
```

### Debug Mode

Enable verbose logging:

```bash
# Maximum verbosity
RUST_LOG=trace cargo run --release

# Specific module debugging
RUST_LOG=mycelix_api=debug,tower_http=trace cargo run --release
```

### Health Check Failures

```bash
# Check if service is running
curl http://localhost:8080/health

# Check Docker container
docker ps
docker logs mycelix-api

# Check systemd service
sudo systemctl status mycelix-api
sudo journalctl -u mycelix-api -n 100
```

---

## Next Steps

After successful deployment:

1. ✅ **Monitor** - Set up monitoring and alerting
2. ✅ **Backup** - Implement backup strategy
3. ✅ **Scale** - Add more replicas if needed
4. ✅ **Secure** - Enable HTTPS, configure firewall
5. ✅ **Optimize** - Tune based on metrics

## Support

- **Documentation:** [docs/](.)
- **Issues:** [GitHub Issues](https://github.com/Luminous-Dynamics/mycelix-desci/issues)
- **Discussions:** [GitHub Discussions](https://github.com/Luminous-Dynamics/mycelix-desci/discussions)

---

**Your Mycelix-DeSci deployment is production-ready!** 🚀✨
