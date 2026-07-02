# Mycelix Supply Chain Provenance Service - Deployment Guide

Complete guide for deploying the Provenance Service to production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Configuration](#configuration)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Monitoring and Observability](#monitoring-and-observability)
6. [Security Hardening](#security-hardening)
7. [Performance Tuning](#performance-tuning)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required

- **Container Runtime**: Docker 24.0+ or Kubernetes 1.27+
- **Database**: PostgreSQL 14+ (production) or SQLite 3.35+ (development)
- **Resources**: Minimum 128MB RAM, 100m CPU per instance

### Recommended

- **TLS Certificates**: Let's Encrypt or organizational CA
- **Monitoring**: Prometheus + Grafana
- **Secrets Management**: HashiCorp Vault, AWS Secrets Manager, or equivalent

---

## Configuration

### Environment Variables

All configuration is managed via environment variables following 12-factor app principles.

Copy `.env.example` to `.env` and customize:

```bash
cp rust/service/.env.example rust/service/.env
```

#### Critical Production Settings

```bash
# Database (REQUIRED - use PostgreSQL in production)
DATABASE_URL=postgresql://user:password@host:5432/dbname

# Security (REQUIRED)
ALLOWED_ORIGINS=https://app.example.com,https://dashboard.example.com
PERMISSIVE_CORS=false

# Observability (RECOMMENDED)
RUST_LOG=info
JSON_LOGS=true
RUST_ENV=production
```

See `.env.example` for complete configuration options.

---

## Docker Deployment

### Quick Start

#### 1. Build the Image

From the `rust` directory:

```bash
cd rust
docker build -f service/Dockerfile -t mycelix/provenance-service:latest .
```

#### 2. Run with Docker Compose

```bash
cd rust/service
docker-compose up -d
```

This starts:
- Provenance service on port 8080
- PostgreSQL database on port 5432

#### 3. Verify Deployment

```bash
# Check health
curl http://localhost:8080/health

# Check liveness
curl http://localhost:8080/health/live

# Check readiness
curl http://localhost:8080/health/ready
```

### Production Docker Deployment

#### 1. Use PostgreSQL (not SQLite)

Update `docker-compose.yml` or set environment variable:

```yaml
environment:
  - DATABASE_URL=postgresql://user:password@postgres:5432/provenance
```

#### 2. Enable Health Checks

Health checks are already configured in the Dockerfile:

```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health/live || exit 1
```

#### 3. Configure Resource Limits

```yaml
services:
  provenance-service:
    # ... other config
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.1'
          memory: 128M
```

#### 4. Enable Logging

```yaml
environment:
  - JSON_LOGS=true
  - RUST_LOG=info
```

---

## Kubernetes Deployment

### Architecture Overview

```
┌─────────────────────────────────────────────┐
│              Ingress (TLS)                  │
│         api.mycelix.example.com             │
└──────────────────┬──────────────────────────┘
                   │
         ┌─────────▼─────────┐
         │   Service (L4)    │
         └─────────┬─────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
┌───▼───┐     ┌───▼───┐     ┌───▼───┐
│ Pod 1 │     │ Pod 2 │     │ Pod 3 │
│       │     │       │     │       │
│ App + │     │ App + │     │ App + │
│Health │     │Health │     │Health │
└───┬───┘     └───┬───┘     └───┬───┘
    │             │             │
    └─────────────┼─────────────┘
                  │
         ┌────────▼────────┐
         │   PostgreSQL    │
         └─────────────────┘
```

### Step-by-Step Deployment

#### 1. Create Namespace

```bash
kubectl create namespace mycelix
```

#### 2. Create Secret for Database Credentials

**Option A: Using kubectl (development)**

```bash
kubectl create secret generic provenance-service-secret \
  --from-literal=DATABASE_URL='postgresql://user:pass@host:5432/db' \
  -n mycelix
```

**Option B: Using Sealed Secrets (production)**

```bash
# Install sealed-secrets controller
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.24.0/controller.yaml

# Create sealed secret
echo -n 'postgresql://user:pass@host:5432/db' | \
  kubeseal --raw --from-file=/dev/stdin --scope strict \
  --name provenance-service-secret --namespace mycelix
```

#### 3. Apply ConfigMap

```bash
kubectl apply -f rust/service/k8s/configmap.yaml
```

Verify:

```bash
kubectl get configmap provenance-service-config -n mycelix -o yaml
```

#### 4. Deploy the Application

```bash
kubectl apply -f rust/service/k8s/deployment.yaml
kubectl apply -f rust/service/k8s/service.yaml
```

#### 5. Verify Deployment

```bash
# Check pods
kubectl get pods -n mycelix

# Check logs
kubectl logs -n mycelix -l app=provenance-service -f

# Check health
kubectl exec -n mycelix -it <pod-name> -- curl localhost:8080/health
```

#### 6. Configure Ingress (Optional)

Update `k8s/ingress.yaml` with your domain:

```yaml
spec:
  tls:
  - hosts:
    - api.yourdomain.com  # Change this
    secretName: provenance-service-tls
  rules:
  - host: api.yourdomain.com  # Change this
```

Apply:

```bash
kubectl apply -f rust/service/k8s/ingress.yaml
```

### Kubernetes Features

#### Zero-Downtime Deployments

The deployment is configured for rolling updates:

```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxSurge: 1
    maxUnavailable: 0  # Zero-downtime
```

#### Health Checks

Three types of probes are configured:

1. **Liveness Probe** (`/health/live`) - Restart unhealthy pods
2. **Readiness Probe** (`/health/ready`) - Route traffic only to healthy pods
3. **Startup Probe** - Allow 60s for slow starts

#### Graceful Shutdown

```yaml
terminationGracePeriodSeconds: 30
```

This allows in-flight requests to complete before pod termination.

#### Resource Management

```yaml
resources:
  requests:
    cpu: 100m
    memory: 128Mi
  limits:
    cpu: 500m
    memory: 512Mi
```

Adjust based on load testing results.

---

## Monitoring and Observability

### Health Endpoints

| Endpoint | Purpose | Kubernetes Use |
|----------|---------|----------------|
| `/health` | Detailed health status | Manual debugging |
| `/health/live` | Liveness check | Liveness probe |
| `/health/ready` | Readiness check | Readiness probe |
| `/metrics` | Prometheus metrics | Monitoring |

### Prometheus Metrics

The `/metrics` endpoint exposes:

- Request counts and latencies
- Database connection pool stats
- Rate limiting metrics
- Error rates

Example scrape configuration:

```yaml
scrape_configs:
  - job_name: 'provenance-service'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - mycelix
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
```

### Logging

#### Structured JSON Logging

Enable for production:

```bash
JSON_LOGS=true
```

Example log entry:

```json
{
  "timestamp": "2025-11-16T12:00:00Z",
  "level": "info",
  "message": "Request completed",
  "request_id": "abc123",
  "method": "POST",
  "path": "/v1/events",
  "status": 200,
  "duration_ms": 15
}
```

#### Log Aggregation

Recommended setup:

```
Application (JSON logs)
    ↓
Fluent Bit / Fluentd
    ↓
Elasticsearch / Loki
    ↓
Kibana / Grafana
```

---

## Security Hardening

### Production Security Checklist

- [ ] **Database**: Use PostgreSQL with TLS connections
- [ ] **CORS**: Set specific `ALLOWED_ORIGINS` (no wildcards)
- [ ] **Rate Limiting**: Configure `RATE_LIMIT_RPS` appropriately
- [ ] **TLS**: Enable HTTPS with valid certificates
- [ ] **Secrets**: Use secrets manager (Vault, AWS Secrets Manager)
- [ ] **Network Policies**: Restrict pod-to-pod communication
- [ ] **Non-root User**: Already configured in Dockerfile
- [ ] **Read-only Filesystem**: Configured in Kubernetes manifest
- [ ] **Security Scanning**: Scan images with Trivy or Snyk

### Network Policies

Example policy to restrict traffic:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: provenance-service-netpol
  namespace: mycelix
spec:
  podSelector:
    matchLabels:
      app: provenance-service
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:  # DNS
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: UDP
      port: 53
```

---

## Performance Tuning

### Database Optimization

#### Connection Pooling

```bash
# Tune based on expected concurrency
DB_MAX_CONNECTIONS=20  # Start conservative

# Formula: (CPU cores × 2) + effective_spindle_count
# For 2 CPU cores: (2 × 2) + 1 = 5-10 connections
```

#### Query Performance

The service includes optimized indexes (see `migrations/20251116000002_performance_indexes.sql`):

- Composite indexes for multi-filter queries
- Expected improvements: ~10-20x for filtered queries

### Rate Limiting

Adjust based on load testing:

```bash
# High traffic
RATE_LIMIT_RPS=500
RATE_LIMIT_BURST=100

# Medium traffic (default)
RATE_LIMIT_RPS=100
RATE_LIMIT_BURST=20

# Low traffic
RATE_LIMIT_RPS=50
RATE_LIMIT_BURST=10
```

### Horizontal Scaling

Kubernetes deployment supports horizontal scaling:

```bash
# Manual scaling
kubectl scale deployment provenance-service --replicas=5 -n mycelix

# Auto-scaling (HPA)
kubectl autoscale deployment provenance-service \
  --cpu-percent=70 \
  --min=3 \
  --max=10 \
  -n mycelix
```

---

## Troubleshooting

### Common Issues

#### 1. Pod CrashLoopBackOff

**Symptoms**: Pods continuously restart

**Diagnosis**:
```bash
kubectl logs -n mycelix <pod-name> --previous
kubectl describe pod -n mycelix <pod-name>
```

**Common causes**:
- Database connection failure (check `DATABASE_URL`)
- Invalid configuration (check ConfigMap/Secret)
- Resource limits too low

#### 2. Readiness Probe Failing

**Symptoms**: Pods running but not receiving traffic

**Diagnosis**:
```bash
kubectl exec -n mycelix <pod-name> -- curl localhost:8080/health/ready
```

**Common causes**:
- Database not accessible
- Health check timeout too short
- Service initialization slow

#### 3. High Latency

**Symptoms**: Slow response times

**Diagnosis**:
```bash
# Check metrics
curl http://<service>/metrics | grep request_duration

# Check database
kubectl exec -n mycelix <pod-name> -- curl localhost:8080/health
```

**Solutions**:
- Increase database connection pool
- Add more replicas
- Check database indexes
- Review rate limiting settings

#### 4. Rate Limiting Issues

**Symptoms**: Getting 429 Too Many Requests

**Diagnosis**:
```bash
# Check current rate limit config
kubectl get configmap provenance-service-config -n mycelix -o yaml | grep RATE_LIMIT
```

**Solutions**:
- Increase `RATE_LIMIT_RPS` and `RATE_LIMIT_BURST`
- Implement client-side retry with exponential backoff
- Consider per-client rate limiting

### Debug Mode

Enable debug logging temporarily:

```bash
kubectl set env deployment/provenance-service RUST_LOG=debug -n mycelix
```

Remember to revert:

```bash
kubectl set env deployment/provenance-service RUST_LOG=info -n mycelix
```

---

## Deployment Checklist

### Pre-Deployment

- [ ] Review and update configuration (`.env` or ConfigMap)
- [ ] Set up database with migrations
- [ ] Configure secrets management
- [ ] Review security settings (CORS, rate limits)
- [ ] Set up monitoring and alerts
- [ ] Perform load testing
- [ ] Document rollback procedures

### Deployment

- [ ] Build and tag Docker image
- [ ] Push to container registry
- [ ] Apply Kubernetes manifests (if using K8s)
- [ ] Verify health checks pass
- [ ] Monitor logs for errors
- [ ] Test API endpoints
- [ ] Verify metrics collection

### Post-Deployment

- [ ] Monitor error rates and latencies
- [ ] Verify database connections stable
- [ ] Check resource utilization
- [ ] Review security logs
- [ ] Update documentation
- [ ] Notify stakeholders

---

## Support and Resources

- **Documentation**: `/docs`
- **Health Check**: `GET /health`
- **Metrics**: `GET /metrics`
- **API Docs**: See `API.md`

For issues and questions:
- GitHub Issues: https://github.com/Luminous-Dynamics/mycelix-supplychain/issues
- Team Contact: dev@luminous-dynamics.dev
