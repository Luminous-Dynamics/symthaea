# Mycelix Mail Deployment Guide

## Quick Start

### Docker (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/mycelix/mail.git
cd mail/deploy/docker
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

3. Generate secrets:
```bash
# Generate secure passwords
openssl rand -base64 32  # For DB_PASSWORD
openssl rand -base64 64  # For JWT_SECRET
openssl rand -base64 32  # For MEILISEARCH_KEY
```

4. Start services:
```bash
docker compose up -d
```

5. Access at `http://localhost:3000`

### With AI Features

```bash
docker compose --profile ai up -d
```

### With Monitoring

```bash
docker compose --profile monitoring up -d
```

## Kubernetes

### Prerequisites
- Kubernetes 1.25+
- Helm 3.10+
- kubectl configured

### Installation

1. Add Helm repository:
```bash
helm repo add mycelix https://charts.mycelix.dev
helm repo update
```

2. Create namespace:
```bash
kubectl create namespace mycelix-mail
```

3. Create secrets:
```bash
kubectl create secret generic mycelix-secrets \
  --from-literal=jwt-secret=$(openssl rand -base64 64) \
  --from-literal=db-password=$(openssl rand -base64 32) \
  -n mycelix-mail
```

4. Install:
```bash
helm install mycelix-mail mycelix/mycelix-mail \
  --namespace mycelix-mail \
  --set global.domain=mail.example.com \
  --set api.replicaCount=3 \
  -f custom-values.yaml
```

### Upgrade

```bash
helm upgrade mycelix-mail mycelix/mycelix-mail \
  --namespace mycelix-mail \
  -f custom-values.yaml
```

## NixOS

### Flake-based Installation

1. Add to flake inputs:
```nix
{
  inputs.mycelix-mail.url = "github:mycelix/mail";
}
```

2. Import module:
```nix
{
  imports = [
    inputs.mycelix-mail.nixosModules.default
  ];

  services.mycelix-mail = {
    enable = true;
    domain = "mail.example.com";

    ssl = {
      enable = true;
      acme = true;
      email = "admin@example.com";
    };

    ollama.enable = true;  # For AI features

    backup = {
      enable = true;
      schedule = "daily";
      location = "/backup/mycelix";
    };
  };
}
```

3. Create secrets file at `/run/secrets/mycelix-mail`:
```
JWT_SECRET=your-jwt-secret
MEILISEARCH_KEY=your-meilisearch-key
```

4. Apply configuration:
```bash
sudo nixos-rebuild switch
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABASE_URL` | PostgreSQL connection string | Yes |
| `REDIS_URL` | Redis connection string | Yes |
| `JWT_SECRET` | Secret for JWT signing | Yes |
| `MEILISEARCH_URL` | Meilisearch endpoint | No |
| `MEILISEARCH_KEY` | Meilisearch API key | No |
| `OLLAMA_URL` | Ollama endpoint for AI | No |
| `SMTP_HOST` | SMTP server for sending | No |
| `DOMAIN` | Public domain | Yes |

### Database

PostgreSQL 14+ required. Create database:
```sql
CREATE DATABASE mycelix;
CREATE USER mycelix WITH PASSWORD 'secure-password';
GRANT ALL PRIVILEGES ON DATABASE mycelix TO mycelix;
```

Run migrations:
```bash
mycelix-api migrate
```

### SSL/TLS

For production, always use HTTPS. Options:

1. **Caddy (automatic HTTPS)**
```
mail.example.com {
    reverse_proxy localhost:3000
    reverse_proxy /api/* localhost:8080
}
```

2. **nginx + Let's Encrypt**
```bash
certbot --nginx -d mail.example.com
```

3. **Kubernetes Ingress**
```yaml
annotations:
  cert-manager.io/cluster-issuer: letsencrypt-prod
```

## Scaling

### Horizontal Scaling

API servers are stateless and can be horizontally scaled:
```yaml
# Kubernetes
api:
  replicaCount: 5
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 20
```

### Database Scaling

For high availability:
1. Use PostgreSQL with streaming replication
2. Consider Patroni for automatic failover
3. Use PgBouncer for connection pooling

### Redis Scaling

For high availability:
1. Redis Sentinel for automatic failover
2. Redis Cluster for horizontal scaling

## Backup & Recovery

### Automated Backups

Docker:
```bash
# Add to crontab
0 2 * * * docker exec mycelix-postgres pg_dump -U mycelix mycelix | gzip > /backup/mycelix-$(date +%Y%m%d).sql.gz
```

Kubernetes:
```yaml
backup:
  enabled: true
  schedule: "0 2 * * *"
  s3:
    bucket: mycelix-backups
    endpoint: s3.amazonaws.com
```

### Recovery

```bash
# Restore database
gunzip -c backup.sql.gz | docker exec -i mycelix-postgres psql -U mycelix mycelix

# Restore attachments
tar -xzf attachments-backup.tar.gz -C /data/attachments
```

## Monitoring

### Health Checks

- **Liveness:** `GET /health/liveness`
- **Readiness:** `GET /health/readiness`
- **Metrics:** `GET /metrics` (Prometheus format)

### Prometheus

```yaml
scrape_configs:
  - job_name: 'mycelix-mail'
    static_configs:
      - targets: ['mycelix-api:8080']
```

### Alerting

Key metrics to alert on:
- `mycelix_http_request_duration_seconds` > 1s
- `mycelix_sync_errors_total` increasing
- `mycelix_pending_sync_items` > 1000
- Health check failures

## Security

### Checklist

- [ ] Use strong passwords for all services
- [ ] Enable SSL/TLS
- [ ] Configure firewall (only expose 80/443)
- [ ] Regular security updates
- [ ] Enable audit logging
- [ ] Configure rate limiting
- [ ] Set up intrusion detection

### Firewall Rules

```bash
# UFW example
ufw allow 80/tcp
ufw allow 443/tcp
ufw deny 8080/tcp  # Block direct API access
ufw deny 5432/tcp  # Block direct DB access
```

## Troubleshooting

### Common Issues

**Can't connect to database**
```bash
# Check PostgreSQL is running
docker logs mycelix-postgres

# Test connection
docker exec -it mycelix-postgres psql -U mycelix -c "SELECT 1"
```

**Sync not working**
```bash
# Check sync worker logs
docker logs mycelix-sync

# Check Redis connection
docker exec -it mycelix-redis redis-cli ping
```

**Search not working**
```bash
# Check Meilisearch
curl http://localhost:7700/health

# Trigger reindex
curl -X POST http://localhost:8080/api/admin/reindex
```

### Logs

```bash
# All logs
docker compose logs -f

# Specific service
docker compose logs -f api

# Kubernetes
kubectl logs -f deployment/mycelix-api -n mycelix-mail
```

## Upgrade Guide

### Minor Updates

```bash
# Docker
docker compose pull
docker compose up -d

# Kubernetes
helm upgrade mycelix-mail mycelix/mycelix-mail
```

### Major Updates

1. Read release notes
2. Backup database
3. Test in staging
4. Apply migrations
5. Rolling update

```bash
# Run migrations first
docker exec mycelix-api ./mycelix-api migrate

# Then update
docker compose up -d
```
