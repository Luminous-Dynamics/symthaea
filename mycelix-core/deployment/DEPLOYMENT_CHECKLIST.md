# Mycelix Testnet Validator Deployment Checklist

This document provides a comprehensive checklist for deploying a Mycelix testnet validator node. Follow each section in order to ensure a successful deployment.

---

## Table of Contents

1. [Pre-Deployment Requirements](#pre-deployment-requirements)
2. [Secrets Management Security](#secrets-management-security)
3. [Step-by-Step Deployment](#step-by-step-deployment)
4. [Verification Steps](#verification-steps)
5. [Post-Deployment Configuration](#post-deployment-configuration)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [Maintenance Procedures](#maintenance-procedures)

---

## Pre-Deployment Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8+ GB |
| Storage | 50 GB SSD | 100+ GB NVMe SSD |
| Network | 100 Mbps | 1 Gbps |
| OS | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |

### Software Requirements

- [ ] Docker Engine 24.0+ installed
- [ ] Docker Compose v2.20+ installed
- [ ] Git 2.40+ installed
- [ ] curl installed
- [ ] jq installed (for JSON parsing)

### Network Requirements

- [ ] Static IP address or DNS hostname
- [ ] Firewall ports configured (see below)
- [ ] Outbound HTTPS (443) access to bootstrap servers
- [ ] Outbound WSS access to signal servers

### Required Ports

| Port | Protocol | Purpose | Inbound/Outbound |
|------|----------|---------|------------------|
| 39329 | TCP | Holochain Admin | Inbound (local) |
| 9998 | TCP | Holochain App | Inbound |
| 9090 | TCP | Metrics (Prometheus) | Inbound (local) |
| 8080 | TCP | Health Check | Inbound |
| 3000 | TCP | Grafana Dashboard | Inbound (local) |
| 443 | TCP | Bootstrap HTTPS | Outbound |
| Various | UDP | WebRTC | Inbound/Outbound |

### Pre-Deployment Checklist

- [ ] Hardware meets minimum requirements
- [ ] Operating system is up to date
- [ ] Docker and Docker Compose installed and working
- [ ] Firewall rules configured
- [ ] DNS/IP address ready
- [ ] Backup strategy planned
- [ ] Monitoring alerts configured (optional)

---

## Secrets Management Security

Proper secrets management is **critical** for production deployments. This section covers security requirements, tools, and procedures for handling sensitive credentials.

### Security Requirements Checklist

Before deployment, ensure you have addressed ALL of the following:

- [ ] **Secrets Manager Selected** - Choose and configure one of:
  - [ ] HashiCorp Vault (recommended for enterprise)
  - [ ] AWS Secrets Manager (for AWS deployments)
  - [ ] GCP Secret Manager (for GCP deployments)
  - [ ] Azure Key Vault (for Azure deployments)
  - [ ] Bitnami Sealed Secrets (for Kubernetes)
  - [ ] Doppler (for multi-cloud/hybrid)

- [ ] **No Plaintext Secrets** - Verify:
  - [ ] `.env` file does NOT contain production secrets
  - [ ] `secrets.yaml` is NOT committed to version control
  - [ ] No secrets in Docker Compose files
  - [ ] No secrets in CI/CD pipeline logs
  - [ ] `.gitignore` includes `*.env`, `secrets.yaml`, `*-secrets.yaml`

- [ ] **Access Controls Configured**:
  - [ ] Secrets manager access restricted to required services only
  - [ ] Human access requires MFA/2FA
  - [ ] Audit logging enabled for all secret access
  - [ ] Service accounts have minimal required permissions

### Required Secrets Inventory

| Secret | Description | Min Length | Rotation Period |
|--------|-------------|------------|-----------------|
| `POSTGRES_PASSWORD` | Database password | 32 chars | 90 days |
| `GRAFANA_ADMIN_PASSWORD` | Dashboard admin password | 16 chars | 90 days |
| `BOOTSTRAP_PRIVATE_KEY` | Node identity key (hex) | 64 chars | On compromise only |
| `JWT_SECRET` | API authentication secret | 64 chars | 90 days |

### Secret Generation Commands

Generate cryptographically secure secrets using these commands:

```bash
# PostgreSQL password (32+ characters)
openssl rand -base64 32

# Grafana password (24+ characters)
openssl rand -base64 24

# Bootstrap private key (256-bit hex)
openssl rand -hex 32

# JWT secret (64+ characters)
openssl rand -base64 64
```

### Kubernetes Deployments - Sealed Secrets Setup

For Kubernetes deployments, use Bitnami Sealed Secrets to safely store encrypted secrets in version control.

#### Step 1: Install Sealed Secrets Controller

```bash
# Add Helm repository
helm repo add sealed-secrets https://bitnami-labs.github.io/sealed-secrets

# Install controller
helm install sealed-secrets sealed-secrets/sealed-secrets \
  --namespace kube-system \
  --set-string fullnameOverride=sealed-secrets-controller
```

#### Step 2: Install kubeseal CLI

```bash
# macOS
brew install kubeseal

# Linux (x86_64)
KUBESEAL_VERSION="0.24.5"
wget "https://github.com/bitnami-labs/sealed-secrets/releases/download/v${KUBESEAL_VERSION}/kubeseal-${KUBESEAL_VERSION}-linux-amd64.tar.gz"
tar -xvzf kubeseal-${KUBESEAL_VERSION}-linux-amd64.tar.gz
sudo install -m 755 kubeseal /usr/local/bin/kubeseal
```

#### Step 3: Create and Seal Secrets

```bash
# Generate secrets
export POSTGRES_PASSWORD=$(openssl rand -base64 32)
export GRAFANA_PASSWORD=$(openssl rand -base64 24)
export BOOTSTRAP_PRIVATE_KEY=$(openssl rand -hex 32)
export JWT_SECRET=$(openssl rand -base64 64)

# Create raw secret (temporary)
kubectl create secret generic mycelix-secrets \
  --namespace=mycelix-testnet \
  --from-literal=POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
  --from-literal=GRAFANA_PASSWORD="$GRAFANA_PASSWORD" \
  --from-literal=BOOTSTRAP_PRIVATE_KEY="$BOOTSTRAP_PRIVATE_KEY" \
  --from-literal=JWT_SECRET="$JWT_SECRET" \
  --dry-run=client -o yaml > /tmp/mycelix-secrets-raw.yaml

# Seal the secret
kubeseal --format=yaml \
  --controller-name=sealed-secrets-controller \
  --controller-namespace=kube-system \
  < /tmp/mycelix-secrets-raw.yaml \
  > testnet/kubernetes/sealed-mycelix-secrets.yaml

# CRITICAL: Securely delete raw secret
shred -u /tmp/mycelix-secrets-raw.yaml

# Apply sealed secret
kubectl apply -f testnet/kubernetes/sealed-mycelix-secrets.yaml
```

See `testnet/kubernetes/sealed-secrets-template.yaml` for detailed templates and examples.

### Docker Compose Deployments - Secret Injection

For non-Kubernetes deployments, inject secrets at runtime:

#### Option A: Environment File (Development/Testing Only)

```bash
# Create .env from template
cp .env.production.template .env

# Edit .env with secrets (NEVER commit this file!)
nano .env

# Ensure .env is in .gitignore
echo ".env" >> .gitignore
```

#### Option B: HashiCorp Vault Agent (Recommended for Production)

```bash
# Install Vault agent
# Configure template to generate .env at runtime
vault agent -config=vault-agent.hcl

# In vault-agent.hcl:
template {
  source      = ".env.production.template"
  destination = "/run/secrets/mycelix.env"
  perms       = 0640
}
```

#### Option C: AWS Secrets Manager

```bash
# Fetch secrets at startup
aws secretsmanager get-secret-value \
  --secret-id mycelix/production/all \
  --query SecretString --output text | jq -r 'to_entries|.[]|.key+"="+.value' > .env
```

### Secret Rotation Procedures

#### Scheduled Rotation (Every 90 Days)

1. **Generate new secrets** using commands above
2. **Update secrets manager** with new values
3. **For Kubernetes**: Create new SealedSecret, apply to cluster
4. **Restart affected services**:
   ```bash
   # Docker Compose
   docker compose restart validator postgres grafana

   # Kubernetes
   kubectl rollout restart deployment/mycelix-validator -n mycelix-testnet
   kubectl rollout restart deployment/postgres -n mycelix-testnet
   kubectl rollout restart deployment/grafana -n mycelix-testnet
   ```
5. **Verify services are healthy** after rotation
6. **Document rotation** in security log

#### Emergency Rotation (Security Incident)

If a secret is compromised:

1. **Immediately rotate ALL secrets** (assume all are compromised)
2. **Revoke old secrets** in secrets manager
3. **Review audit logs** for unauthorized access
4. **Check for unauthorized changes** in affected systems
5. **For Kubernetes**: Rotate Sealed Secrets controller key:
   ```bash
   kubectl delete secret -n kube-system -l sealedsecrets.bitnami.com/sealed-secrets-key
   ```
6. **Re-seal ALL SealedSecrets** with new controller key
7. **Document incident** following security incident procedure
8. **Notify stakeholders** per incident response plan

### Backup and Recovery

#### Backup Sealed Secrets Master Key (CRITICAL)

```bash
# Export master key (store securely offline!)
kubectl get secret -n kube-system \
  -l sealedsecrets.bitnami.com/sealed-secrets-key \
  -o yaml > sealed-secrets-master-key.yaml

# Store in:
# - Hardware Security Module (HSM)
# - Secure offline storage
# - Encrypted cloud backup with separate access controls
```

**WARNING**: Without this key, you CANNOT decrypt SealedSecrets if the cluster is lost!

#### Test Recovery Procedure

Quarterly, test your secret recovery procedure:

1. Export current sealed secrets
2. Restore master key to test cluster
3. Verify secrets can be decrypted
4. Document any issues found

### Security Audit Checklist

Perform these checks before every production deployment:

- [ ] **No secrets in version control**
  ```bash
  git log -p | grep -i "password\|secret\|private_key\|jwt" | head -20
  ```

- [ ] **Secrets have sufficient entropy**
  - Passwords: 32+ characters
  - Keys: 256-bit minimum

- [ ] **Access logs are enabled**
  ```bash
  # Vault
  vault audit list

  # AWS
  aws cloudtrail lookup-events --lookup-attributes AttributeKey=ResourceName,AttributeValue=mycelix
  ```

- [ ] **TLS enabled for secrets manager connections**

- [ ] **Network policies restrict secret access**

- [ ] **Emergency contacts documented for security incidents**

---

## Step-by-Step Deployment

### Step 1: Clone the Repository

```bash
# Clone Mycelix Core repository
git clone https://github.com/luminous-dynamics/mycelix-core.git
cd mycelix-core/deployment
```

### Step 2: Create Data Directories

```bash
# Create persistent storage directories
mkdir -p ./data/{validator,keystore,logs,cache,prometheus,grafana}

# Set appropriate permissions
chmod -R 755 ./data
```

### Step 3: Configure Environment

```bash
# Copy example configuration
cp .env.example .env

# Edit configuration (use your preferred editor)
nano .env
```

**Required Configuration Changes:**

```bash
# MUST CHANGE: Set unique node identifier
VALIDATOR_NODE_ID=validator-your-unique-name

# MUST CHANGE: Set your region
VALIDATOR_REGION=na  # Options: na, eu, asia, oceania, sa, africa, global

# MUST CHANGE: Set operator name
VALIDATOR_OPERATOR=your-organization-name

# RECOMMENDED: Change Grafana password
GRAFANA_ADMIN_PASSWORD=your-secure-password
```

### Step 4: Validate Configuration

```bash
# Check docker-compose configuration is valid
docker compose config --quiet && echo "Configuration valid!"

# Verify environment variables
docker compose config | grep -E "VALIDATOR_NODE_ID|VALIDATOR_REGION"
```

### Step 5: Pull Docker Images

```bash
# Pull all required images
docker compose pull

# Build validator image
docker compose build validator
```

### Step 6: Start Services

```bash
# Start all services in detached mode
docker compose up -d

# View startup logs
docker compose logs -f --tail=100
```

### Step 7: Verify Startup

```bash
# Check all containers are running
docker compose ps

# Expected output:
# NAME                    STATUS
# mycelix-validator       Up (healthy)
# mycelix-prometheus      Up (healthy)
# mycelix-grafana         Up (healthy)
```

---

## Verification Steps

### 1. Health Check Verification

```bash
# Check validator health endpoint
curl -s http://localhost:8080/health | jq .

# Expected response:
# {
#   "healthy": true,
#   "ready": true,
#   "checks": {
#     "coordinator": true,
#     "holochain": true,
#     "metrics": true
#   }
# }
```

### 2. Metrics Verification

```bash
# Check Prometheus metrics endpoint
curl -s http://localhost:9090/metrics | head -20

# Verify specific metrics exist
curl -s http://localhost:9090/metrics | grep mycelix_active_nodes
```

### 3. Grafana Dashboard Verification

```bash
# Check Grafana is accessible
curl -s http://localhost:3000/api/health

# Login to Grafana
# URL: http://localhost:3000
# Username: admin (or GRAFANA_ADMIN_USER value)
# Password: (GRAFANA_ADMIN_PASSWORD value)
```

### 4. Network Connectivity Verification

```bash
# Test bootstrap server connectivity
curl -s https://bootstrap.mycelix.net/health

# Test from within container
docker compose exec validator curl -s https://bootstrap.mycelix.net/health
```

### 5. Log Verification

```bash
# Check for errors in logs
docker compose logs validator --tail=50 | grep -i error

# Check for successful startup messages
docker compose logs validator | grep -i "ready"
```

### 6. FL Coordinator Verification

```bash
# Check coordinator status
docker compose exec validator python3 -c "
from mycelix_fl.distributed.coordinator import FLCoordinator
print('FL Coordinator module loaded successfully')
"
```

---

## Post-Deployment Configuration

### Configure Alerting (Optional)

1. Start alertmanager:
   ```bash
   docker compose --profile alerting up -d alertmanager
   ```

2. Configure alert receivers in `alertmanager.yml`

### Enable Host Monitoring (Optional)

```bash
# Start node exporter for host metrics
docker compose --profile monitoring up -d node-exporter
```

### Set Up Log Rotation

Logs are automatically rotated by Docker, but verify settings:

```bash
# Check log settings
docker inspect mycelix-validator | jq '.[0].HostConfig.LogConfig'
```

### Configure Backups

```bash
# Example backup script (add to cron)
#!/bin/bash
BACKUP_DIR="/backup/mycelix/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR
docker compose exec -T validator tar czf - /data > $BACKUP_DIR/validator-data.tar.gz
```

---

## Troubleshooting Guide

### Issue: Container Fails to Start

**Symptoms:** Container immediately exits or restarts

**Diagnosis:**
```bash
docker compose logs validator --tail=100
docker compose ps -a
```

**Common Causes & Solutions:**

1. **Port conflict:**
   ```bash
   # Check if ports are in use
   sudo lsof -i :9090 -i :8080 -i :39329

   # Solution: Change port in .env or stop conflicting service
   ```

2. **Insufficient permissions:**
   ```bash
   # Fix data directory permissions
   sudo chown -R 1000:1000 ./data
   ```

3. **Out of memory:**
   ```bash
   # Check memory usage
   docker stats --no-stream

   # Solution: Increase MEMORY_LIMIT in .env
   ```

### Issue: Health Check Failing

**Symptoms:** Container shows "unhealthy" status

**Diagnosis:**
```bash
curl -v http://localhost:8080/health
docker compose logs validator | grep -i health
```

**Solutions:**

1. **Service not ready yet:**
   - Wait 60 seconds for startup period
   - Check logs for initialization progress

2. **Coordinator failed to start:**
   ```bash
   docker compose exec validator python3 -c "import mycelix_fl"
   ```

3. **Network issues:**
   ```bash
   docker compose exec validator ping -c 3 bootstrap.mycelix.net
   ```

### Issue: Cannot Connect to Network

**Symptoms:** No peers discovered, bootstrap failures

**Diagnosis:**
```bash
docker compose logs validator | grep -i bootstrap
docker compose logs validator | grep -i "peer"
```

**Solutions:**

1. **Firewall blocking outbound:**
   ```bash
   # Test HTTPS connectivity
   docker compose exec validator curl -v https://bootstrap.mycelix.net/health
   ```

2. **DNS resolution failure:**
   ```bash
   docker compose exec validator nslookup bootstrap.mycelix.net
   ```

3. **Wrong bootstrap URL:**
   - Verify BOOTSTRAP_URL in .env matches testnet configuration

### Issue: High Resource Usage

**Symptoms:** CPU/memory usage exceeds limits

**Diagnosis:**
```bash
docker stats mycelix-validator
```

**Solutions:**

1. **Reduce worker count:**
   ```bash
   # In .env
   MAX_WORKERS=2
   ```

2. **Lower HV dimension:**
   ```bash
   # In .env (reduces accuracy slightly)
   MYCELIX_HV_DIMENSION=1024
   ```

### Issue: Metrics Not Appearing in Grafana

**Symptoms:** Dashboard shows "No Data"

**Diagnosis:**
```bash
# Check Prometheus is scraping
curl http://localhost:9091/targets

# Check metrics endpoint
curl http://localhost:9090/metrics
```

**Solutions:**

1. **Prometheus not scraping:**
   - Verify prometheus.yml configuration
   - Check container networking

2. **Wrong datasource:**
   - Verify Grafana datasource points to prometheus:9090

### Issue: Persistent Data Loss

**Symptoms:** Data lost after container restart

**Diagnosis:**
```bash
docker volume ls | grep mycelix
docker inspect mycelix-validator | jq '.[0].Mounts'
```

**Solutions:**

1. **Volumes not bound:**
   - Verify DATA_PATH in .env
   - Ensure directories exist on host

2. **Permission issues:**
   ```bash
   sudo chown -R 1000:1000 ./data
   ```

---

## Maintenance Procedures

### Upgrading the Validator

```bash
# 1. Pull latest code
git pull origin main

# 2. Backup current data
./scripts/backup.sh

# 3. Stop services
docker compose down

# 4. Rebuild images
docker compose build --no-cache validator

# 5. Start services
docker compose up -d

# 6. Verify health
curl http://localhost:8080/health | jq .
```

### Viewing Logs

```bash
# Real-time logs
docker compose logs -f validator

# Last 100 lines
docker compose logs --tail=100 validator

# Logs with timestamps
docker compose logs -t validator
```

### Stopping the Validator

```bash
# Graceful shutdown
docker compose down

# Force stop (if unresponsive)
docker compose kill
docker compose down
```

### Resetting State

```bash
# WARNING: This deletes all data!
docker compose down -v
rm -rf ./data/*
```

---

## Quick Reference

### Common Commands

| Action | Command |
|--------|---------|
| Start all services | `docker compose up -d` |
| Stop all services | `docker compose down` |
| View logs | `docker compose logs -f validator` |
| Check status | `docker compose ps` |
| Restart validator | `docker compose restart validator` |
| Health check | `curl http://localhost:8080/health` |
| View metrics | `curl http://localhost:9090/metrics` |
| Enter container | `docker compose exec validator bash` |

### Important Files

| File | Purpose |
|------|---------|
| `.env` | Environment configuration |
| `docker-compose.yml` | Service definitions |
| `prometheus.yml` | Metrics collection config |
| `grafana/dashboards/` | Dashboard definitions |
| `data/` | Persistent storage |

### Support Resources

- Documentation: https://docs.mycelix.net
- Discord: https://discord.gg/mycelix
- GitHub Issues: https://github.com/luminous-dynamics/mycelix-core/issues

---

*Last updated: January 2026*
*Version: 1.0.0*
