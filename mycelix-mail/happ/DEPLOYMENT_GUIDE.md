# 🚀 Mycelix Mail - Deployment Guide

**Quick Deploy**: Production-ready Holochain DNA with trust-based spam filtering

**Time to Deploy**: 15-30 minutes
**Requirements**: Holochain conductor, PostgreSQL, Python 3.8+

---

## 📋 Pre-Deployment Checklist

Before deploying, ensure you have:
- ✅ DNA validated: `hc dna hash dna/mycelix_mail.dna`
- ✅ Integration tests passing: `python3 tests/integration_test_suite.py`
- ✅ Pre-commit hook active: `.git/hooks/pre-commit`
- ✅ Documentation reviewed: `README_START_HERE.md`

---

## 🎯 Deployment Options

### Option 1: Quick Deploy (Holochain Only)
**Time**: 5 minutes | **Features**: Basic email without DID/MATL

```bash
# 1. Copy DNA to conductor
scp dna/mycelix_mail.dna conductor@host:/apps/

# 2. Install on conductor
ssh conductor@host "hc app install /apps/mycelix_mail.dna"

# 3. Verify installation
ssh conductor@host "hc app list"
```

**What you get**: Core email functionality on Holochain DHT

**What's missing**: DID resolution, trust-based spam filtering

---

### Option 2: Full Stack (Recommended)
**Time**: 30 minutes | **Features**: Complete L1+L5+L6 integration

#### Step 1: Deploy DID Registry (Layer 5)

```bash
# Setup PostgreSQL database
psql -U postgres << EOF
CREATE DATABASE mycelix_mail_did;
CREATE USER mycelix_mail WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE mycelix_mail_did TO mycelix_mail;
EOF

# Initialize schema
psql -U mycelix_mail mycelix_mail_did < did-registry/schema.sql

# Configure DID service
cd did-registry
cp .env.example .env
# Edit .env with your database credentials

# Install dependencies
pip install -r requirements.txt

# Start DID resolution service
python3 did_resolver.py &
```

**Test DID Registry**:
```bash
curl http://localhost:5000/health
# Should return: {"status": "ok"}
```

#### Step 2: Deploy MATL Bridge (Layer 6)

```bash
# Configure MATL bridge
cd matl-bridge
cp .env.example .env
# Edit .env with MATL API credentials

# Install dependencies
pip install -r requirements.txt

# Start MATL sync daemon
python3 matl_sync.py &
```

**Test MATL Bridge**:
```bash
# Check sync status
curl http://localhost:5001/sync/status
```

#### Step 3: Deploy Holochain DNA (Layer 1)

```bash
# Copy DNA to conductor
scp dna/mycelix_mail.dna conductor@host:/apps/

# Install on conductor with DID/MATL integration
ssh conductor@host << 'EOF'
  export DID_REGISTRY_URL="http://did-service:5000"
  export MATL_BRIDGE_URL="http://matl-service:5001"
  hc app install /apps/mycelix_mail.dna
EOF

# Verify installation
ssh conductor@host "hc app list"
```

---

### Option 3: Docker Compose (Easiest)
**Time**: 10 minutes | **Features**: Everything, automated

**Create**: `docker-compose.yml`

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: mycelix_mail_did
      POSTGRES_USER: mycelix_mail
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - ./did-registry/schema.sql:/docker-entrypoint-initdb.d/schema.sql
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  did-registry:
    build: ./did-registry
    environment:
      DATABASE_URL: postgresql://mycelix_mail:${DB_PASSWORD}@postgres:5432/mycelix_mail_did
    depends_on:
      - postgres
    ports:
      - "5000:5000"

  matl-bridge:
    build: ./matl-bridge
    environment:
      MATL_API_KEY: ${MATL_API_KEY}
      DID_REGISTRY_URL: http://did-registry:5000
    depends_on:
      - did-registry
    ports:
      - "5001:5001"

  holochain:
    image: holochain/holochain:latest
    environment:
      DID_REGISTRY_URL: http://did-registry:5000
      MATL_BRIDGE_URL: http://matl-bridge:5001
    volumes:
      - ./dna/mycelix_mail.dna:/app/mycelix_mail.dna
    ports:
      - "8888:8888"
    command: >
      sh -c "hc app install /app/mycelix_mail.dna &&
             hc sandbox run -p 8888"

volumes:
  postgres_data:
```

**Deploy**:
```bash
# Set environment variables
echo "DB_PASSWORD=your_secure_password" > .env
echo "MATL_API_KEY=your_matl_api_key" >> .env

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

---

## 🔍 Post-Deployment Verification

### 1. Check DNA Installation

```bash
# On conductor host
hc app list
# Should show: mycelix_mail

# Verify DNA hash
hc app info mycelix_mail
# Hash should match: uhC0kV_byY-EylKlDHg-AeGab0xNhFCIkEFAk2Nr9EDd7mV17oU_U
```

### 2. Test DID Resolution

```bash
# Register test DID
curl -X POST http://localhost:5000/register \
  -H "Content-Type: application/json" \
  -d '{"did": "did:mycelix:test", "agent_pub_key": "uhCAkTest123"}'

# Resolve DID
curl http://localhost:5000/resolve/did:mycelix:test
# Should return: {"agent_pub_key": "uhCAkTest123"}
```

### 3. Test MATL Bridge

```bash
# Set trust score
curl -X POST http://localhost:5001/trust/set \
  -H "Content-Type: application/json" \
  -d '{"did": "did:mycelix:test", "score": 0.85}'

# Sync to Holochain
curl -X POST http://localhost:5001/sync/did:mycelix:test

# Verify sync
curl http://localhost:5001/sync/status/did:mycelix:test
```

### 4. End-to-End Test

```bash
# Send test message (via Holochain conductor)
hc call mycelix_mail mail_messages send_message '{
  "from_did": "did:mycelix:alice",
  "to_did": "did:mycelix:bob",
  "subject_encrypted": [1,2,3,4,5],
  "body_cid": "QmTestCID123",
  "thread_id": null,
  "epistemic_tier": "Tier2PrivatelyVerifiable"
}'

# List received messages
hc call mycelix_mail mail_messages list_inbox '{}'
```

---

## 🔧 Configuration

### Environment Variables

**DID Registry** (`.env`):
```bash
DATABASE_URL=postgresql://user:pass@localhost:5432/mycelix_mail_did
API_PORT=5000
LOG_LEVEL=INFO
```

**MATL Bridge** (`.env`):
```bash
MATL_API_KEY=your_api_key_here
MATL_API_URL=https://matl.mycelix.net/api
DID_REGISTRY_URL=http://localhost:5000
HOLOCHAIN_CONDUCTOR_URL=http://localhost:8888
SYNC_INTERVAL=300  # 5 minutes
API_PORT=5001
LOG_LEVEL=INFO
```

**Holochain Conductor**:
```yaml
# conductor-config.yaml
admin_interfaces:
  - driver:
      type: websocket
      port: 8888

app_interfaces:
  - driver:
      type: websocket
      port: 9999

environment_variables:
  DID_REGISTRY_URL: http://localhost:5000
  MATL_BRIDGE_URL: http://localhost:5001
```

---

## 🚨 Troubleshooting

### DNA Installation Fails

**Issue**: `hc app install` returns error

**Solutions**:
```bash
# Check DNA integrity
hc dna hash dna/mycelix_mail.dna

# Verify conductor is running
ps aux | grep holochain

# Check conductor logs
journalctl -u holochain -f

# Restart conductor
systemctl restart holochain
```

### DID Registry Not Responding

**Issue**: `curl http://localhost:5000/health` fails

**Solutions**:
```bash
# Check service status
ps aux | grep did_resolver

# Check PostgreSQL connection
psql -U mycelix_mail mycelix_mail_did -c "SELECT 1;"

# Check logs
tail -f /var/log/mycelix-mail/did-registry.log

# Restart service
pkill -f did_resolver.py
python3 did-registry/did_resolver.py &
```

### MATL Bridge Sync Failing

**Issue**: Trust scores not appearing in Holochain

**Solutions**:
```bash
# Verify MATL API key
curl -H "Authorization: Bearer $MATL_API_KEY" https://matl.mycelix.net/api/health

# Check bridge logs
tail -f /var/log/mycelix-mail/matl-bridge.log

# Manual sync test
python3 -c "from matl_sync import sync_trust_score; sync_trust_score('did:mycelix:test')"

# Restart bridge
pkill -f matl_sync.py
python3 matl-bridge/matl_sync.py &
```

---

## 📊 Monitoring

### Health Checks

**DID Registry**:
```bash
curl http://localhost:5000/health
# Expected: {"status": "ok", "version": "1.0.0"}
```

**MATL Bridge**:
```bash
curl http://localhost:5001/health
# Expected: {"status": "ok", "synced_dids": 123}
```

**Holochain Conductor**:
```bash
hc admin list-apps
# Expected: List of installed apps including mycelix_mail
```

### Metrics

**Message Volume**:
```bash
hc call mycelix_mail mail_messages get_stats '{}'
# Returns: message counts, average delivery time
```

**Trust Score Coverage**:
```bash
curl http://localhost:5001/metrics
# Returns: synced DIDs, average score, sync success rate
```

---

## 🔐 Security Considerations

### Before Production

1. **Change Default Passwords**:
   ```bash
   # PostgreSQL
   psql -U postgres -c "ALTER USER mycelix_mail PASSWORD 'strong_random_password';"
   ```

2. **Enable SSL/TLS**:
   - Configure PostgreSQL SSL
   - Use HTTPS for DID registry and MATL bridge
   - Configure Holochain conductor SSL

3. **Firewall Configuration**:
   ```bash
   # Allow only necessary ports
   ufw allow 5432/tcp  # PostgreSQL (localhost only)
   ufw allow 5000/tcp  # DID Registry (internal)
   ufw allow 5001/tcp  # MATL Bridge (internal)
   ufw allow 8888/tcp  # Holochain (public)
   ```

4. **Set Up Monitoring**:
   - Application logs → Centralized logging (ELK/Loki)
   - Metrics → Prometheus + Grafana
   - Alerts → PagerDuty/Slack

---

## 📈 Scaling

### Horizontal Scaling

**DID Registry** (Read Replicas):
```yaml
# docker-compose-scale.yml
services:
  did-registry:
    deploy:
      replicas: 3

  nginx:
    image: nginx
    volumes:
      - ./nginx-did-lb.conf:/etc/nginx/nginx.conf
    depends_on:
      - did-registry
```

**MATL Bridge** (Sharded by DID):
```bash
# Start multiple bridges with shard assignment
python3 matl_sync.py --shard 0 --total-shards 4 &
python3 matl_sync.py --shard 1 --total-shards 4 &
python3 matl_sync.py --shard 2 --total-shards 4 &
python3 matl_sync.py --shard 3 --total-shards 4 &
```

---

## 🎯 Success Criteria

Deployment is successful when:
- ✅ DNA hash verified on conductor
- ✅ DID registry responding to health checks
- ✅ MATL bridge syncing trust scores
- ✅ End-to-end message flow working
- ✅ Spam filtering active (low-trust messages filtered)
- ✅ All health checks passing
- ✅ Logs showing no errors

---

## 📞 Support

**Issues**: See `TROUBLESHOOTING_GUIDE.md`
**Questions**: See `QUICK_REF.md` and `PROJECT_SUMMARY.md`
**Contact**: tristan.stoltz@evolvingresonantcocreationism.com

---

**Deployment Status**: Ready for production
**Estimated Deployment Time**: 15-30 minutes
**Difficulty**: Intermediate (Docker Compose) to Advanced (Manual)

🍄 **Deploy with confidence - DNA is production-ready!** 🍄
