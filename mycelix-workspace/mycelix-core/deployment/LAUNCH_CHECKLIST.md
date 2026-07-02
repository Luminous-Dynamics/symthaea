# Mycelix Validator Node Launch Checklist - Sepolia Testnet

This document provides a comprehensive checklist for launching a Mycelix validator node on the Ethereum Sepolia testnet.

## Table of Contents

1. [Pre-Launch Verification](#pre-launch-verification)
2. [Environment Variables](#environment-variables)
3. [Docker Commands](#docker-commands)
4. [Health Check Commands](#health-check-commands)
5. [Monitoring Setup](#monitoring-setup)
6. [Troubleshooting Guide](#troubleshooting-guide)

---

## Pre-Launch Verification

### 1. System Requirements

- [ ] **Hardware Requirements**
  - CPU: 4+ cores recommended
  - RAM: 4GB minimum, 8GB recommended
  - Storage: 50GB+ SSD recommended
  - Network: Stable internet connection with open ports

- [ ] **Software Requirements**
  - Docker Engine 20.10+
  - Docker Compose v2.0+
  - curl (for health checks)
  - jq (optional, for JSON parsing)

```bash
# Verify Docker installation
docker --version
docker compose version
```

### 2. Network Connectivity

- [ ] **Sepolia RPC Access**
  - Verify RPC endpoint is reachable
  - Test with: `curl -X POST https://sepolia.drpc.org -H "Content-Type: application/json" -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'`

- [ ] **Required Ports**
  | Port  | Service              | Direction |
  |-------|---------------------|-----------|
  | 39329 | Holochain Admin     | Inbound   |
  | 9998  | Holochain App       | Inbound   |
  | 9090  | Metrics             | Internal  |
  | 8080  | Health Check        | Internal  |
  | 50051 | FL gRPC (optional)  | Inbound   |
  | 3000  | Grafana             | Inbound   |
  | 9091  | Prometheus          | Internal  |

### 3. Contract Verification

- [ ] **Verify Sepolia Contract Addresses**

  | Contract          | Address                                      | Verified |
  |-------------------|----------------------------------------------|----------|
  | MycelixRegistry   | `0x69411e4c9D99814952EBA9a35028c12c2bbD03cd` | [ ]      |
  | ReputationAnchor  | `0x042ee96Ce1CaFFF1e84d6da0585CD440d73DAF99` | [ ]      |
  | PaymentRouter     | `0x3BD80003d27f9786bd4ffeaA5ffA7CC83365da1b` | [ ]      |

```bash
# Verify contracts exist on Sepolia
for addr in 0x69411e4c9D99814952EBA9a35028c12c2bbD03cd 0x042ee96Ce1CaFFF1e84d6da0585CD440d73DAF99 0x3BD80003d27f9786bd4ffeaA5ffA7CC83365da1b; do
  echo "Checking $addr..."
  curl -s -X POST https://sepolia.drpc.org \
    -H "Content-Type: application/json" \
    -d "{\"jsonrpc\":\"2.0\",\"method\":\"eth_getCode\",\"params\":[\"$addr\",\"latest\"],\"id\":1}" | jq -r '.result | if . == "0x" then "NO CONTRACT" else "CONTRACT EXISTS" end'
done
```

### 4. Wallet Configuration

- [ ] **Validator Wallet Setup**
  - Generate or import a wallet for the validator
  - Fund with Sepolia ETH (use faucet: https://sepoliafaucet.com)
  - Recommended: 0.5+ Sepolia ETH for gas fees
  - **NEVER use mainnet private keys on testnet!**

```bash
# Check wallet balance (replace with your address)
curl -s -X POST https://sepolia.drpc.org \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"eth_getBalance","params":["YOUR_ADDRESS","latest"],"id":1}' | jq -r '.result' | xargs printf "%d\n" | awk '{print $1/1e18 " ETH"}'
```

### 5. Data Directory Setup

- [ ] **Create Required Directories**

```bash
# Create data directories with proper permissions
mkdir -p ./data/{validator,keystore,logs,cache,prometheus,grafana}
chmod -R 755 ./data
```

---

## Environment Variables

### Required Variables

Copy `.env.sepolia.example` to `.env` and configure:

```bash
cp .env.sepolia.example .env
```

| Variable                    | Required | Description                           | Example                                      |
|-----------------------------|----------|---------------------------------------|----------------------------------------------|
| `VALIDATOR_NODE_ID`         | Yes      | Unique node identifier                | `validator-001`                              |
| `ETH_PRIVATE_KEY`           | Yes      | Validator wallet private key          | `0x...` (64 hex chars)                       |
| `GRAFANA_ADMIN_PASSWORD`    | Yes      | Grafana admin password                | `secure-password-here`                       |
| `ETH_RPC_URL`               | No       | Custom RPC URL                        | `https://sepolia.drpc.org`                   |
| `MYCELIX_REGISTRY_ADDRESS`  | No       | Override registry contract            | `0x69411e4c9D99814952EBA9a35028c12c2bbD03cd` |
| `REPUTATION_ANCHOR_ADDRESS` | No       | Override reputation contract          | `0x042ee96Ce1CaFFF1e84d6da0585CD440d73DAF99` |
| `PAYMENT_ROUTER_ADDRESS`    | No       | Override payment contract             | `0x3BD80003d27f9786bd4ffeaA5ffA7CC83365da1b` |

### Security Notes

- **NEVER commit `.env` to version control**
- Use strong, unique passwords for Grafana
- Consider using Docker secrets for production
- Private keys should be stored securely (consider hardware wallets for mainnet)

---

## Docker Commands

### Starting the Stack

```bash
# Navigate to deployment directory
cd /home/tstoltz/Luminous-Dynamics/Mycelix-Core/deployment

# Validate configuration before starting
./launch-sepolia.sh --validate

# Start all services
docker compose up -d

# Start with full monitoring stack
docker compose --profile monitoring --profile alerting up -d

# View logs
docker compose logs -f validator
```

### Stopping the Stack

```bash
# Stop all services
docker compose down

# Stop and remove volumes (CAUTION: deletes data)
docker compose down -v
```

### Updating Services

```bash
# Pull latest images
docker compose pull

# Restart with new images
docker compose up -d --force-recreate
```

### Service Management

```bash
# Restart specific service
docker compose restart validator

# View service status
docker compose ps

# Scale FL nodes (if applicable)
docker compose up -d --scale fl-node=5
```

---

## Health Check Commands

### Quick Health Check

```bash
# All-in-one health check
./launch-sepolia.sh --health-check

# Or manually:
curl -s http://localhost:8080/health | jq
```

### Individual Service Checks

```bash
# Validator health
curl -s http://localhost:8080/health | jq '.status'

# Prometheus health
curl -s http://localhost:9091/-/healthy

# Grafana health
curl -s http://localhost:3000/api/health | jq

# Holochain admin API
curl -s http://localhost:39329/health || echo "Admin API check requires websocket"
```

### Ethereum Connection Check

```bash
# Check Sepolia connection via validator
docker exec mycelix-validator python3 -c "
from web3 import Web3
w3 = Web3(Web3.HTTPProvider('https://sepolia.drpc.org'))
print(f'Connected: {w3.is_connected()}')
print(f'Block: {w3.eth.block_number}')
print(f'Chain ID: {w3.eth.chain_id}')
"
```

### Metrics Check

```bash
# View key metrics
curl -s http://localhost:9090/metrics | grep -E "^(mycelix_|fl_)"
```

---

## Monitoring Setup

### Grafana Access

1. Open http://localhost:3000 in your browser
2. Login with admin credentials from `.env`
3. Pre-configured dashboards are available under "Dashboards"

### Key Dashboards

| Dashboard               | Description                           |
|-------------------------|---------------------------------------|
| Mycelix Overview        | Node health, peer count, FL rounds    |
| Byzantine Detection     | 0TML detection metrics, false positives|
| Ethereum Integration    | Contract calls, gas usage, tx status  |
| Holochain DHT           | DHT operations, peer connectivity     |

### Alert Configuration

Default alerts are configured in `alertmanager.yml`:

| Alert                   | Threshold        | Severity |
|-------------------------|------------------|----------|
| NodeDown                | > 5 min          | Critical |
| HighByzantineRate       | > 10%            | Warning  |
| LowPeerCount            | < 3 peers        | Warning  |
| HighMemoryUsage         | > 90%            | Warning  |
| EthereumRPCFailure      | 3 consecutive    | Critical |

### Custom Prometheus Queries

```promql
# FL round success rate (last 1 hour)
sum(rate(mycelix_fl_rounds_completed_total[1h])) / sum(rate(mycelix_fl_rounds_total[1h]))

# Byzantine detection rate
rate(mycelix_byzantine_detected_total[5m])

# Ethereum transaction success rate
sum(rate(mycelix_eth_tx_success_total[1h])) / sum(rate(mycelix_eth_tx_total[1h]))
```

---

## Troubleshooting Guide

### Common Issues

#### 1. Container Fails to Start

**Symptoms**: Container exits immediately or enters restart loop

```bash
# Check container logs
docker compose logs validator --tail=100

# Check for resource constraints
docker stats --no-stream

# Verify data directory permissions
ls -la ./data/
```

**Solutions**:
- Ensure data directories exist and are writable
- Verify sufficient disk space: `df -h`
- Check if ports are in use: `netstat -tlnp | grep -E '(8080|9090|39329)'`

#### 2. Ethereum Connection Issues

**Symptoms**: `Failed to connect to RPC`, `Connection refused`

```bash
# Test RPC directly
curl -X POST https://sepolia.drpc.org \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}'
```

**Solutions**:
- Try alternative RPC endpoints:
  - `https://rpc.sepolia.org`
  - `https://ethereum-sepolia-rpc.publicnode.com`
  - `https://sepolia.infura.io/v3/YOUR_PROJECT_ID`
- Check firewall rules for outbound HTTPS
- Verify DNS resolution: `nslookup sepolia.drpc.org`

#### 3. Transaction Failures

**Symptoms**: `Transaction failed`, `Insufficient funds`, `Nonce too low`

```bash
# Check wallet balance
docker exec mycelix-validator python3 -c "
from web3 import Web3
w3 = Web3(Web3.HTTPProvider('$ETH_RPC_URL'))
print(w3.eth.get_balance('YOUR_ADDRESS') / 1e18, 'ETH')
"
```

**Solutions**:
- Fund wallet with more Sepolia ETH
- Reset nonce if stuck: restart validator
- Check gas price settings in `.env`

#### 4. Holochain DHT Issues

**Symptoms**: `No peers found`, `DHT bootstrap failed`

```bash
# Check Holochain logs
docker compose logs validator | grep -i holochain

# Verify signal server connectivity
curl -s https://signal.holo.host/health
```

**Solutions**:
- Ensure SIGNAL_URL is correct
- Check BOOTSTRAP_URL configuration
- Verify firewall allows WebSocket connections

#### 5. High Memory Usage

**Symptoms**: OOM kills, slow performance

```bash
# Check memory usage
docker stats mycelix-validator

# Adjust resource limits in .env
MEMORY_LIMIT=8G
MAX_MEMORY_MB=6144
```

#### 6. Metrics Not Appearing

**Symptoms**: Grafana dashboards empty, Prometheus targets down

```bash
# Check Prometheus targets
curl -s http://localhost:9091/api/v1/targets | jq '.data.activeTargets[] | {job: .job, health: .health}'

# Verify metrics endpoint
curl -s http://localhost:9090/metrics | head -20
```

**Solutions**:
- Verify METRICS_ENABLED=true in `.env`
- Check prometheus.yml configuration
- Restart Prometheus: `docker compose restart prometheus`

### Log Analysis

```bash
# Search for errors
docker compose logs validator 2>&1 | grep -i error

# Follow logs with timestamp
docker compose logs -f --timestamps validator

# Export logs for analysis
docker compose logs validator > validator_logs.txt 2>&1
```

### Recovery Procedures

#### Soft Reset (preserve data)

```bash
docker compose down
docker compose up -d
```

#### Hard Reset (fresh start)

```bash
docker compose down -v
rm -rf ./data/*
mkdir -p ./data/{validator,keystore,logs,cache,prometheus,grafana}
docker compose up -d
```

#### Backup Data

```bash
# Backup keystore (CRITICAL)
tar -czvf keystore_backup_$(date +%Y%m%d).tar.gz ./data/keystore

# Backup all data
tar -czvf mycelix_backup_$(date +%Y%m%d).tar.gz ./data
```

---

## Quick Reference

### Essential Commands

```bash
# Start
./launch-sepolia.sh

# Stop
docker compose down

# Logs
docker compose logs -f validator

# Health
curl localhost:8080/health

# Metrics
curl localhost:9090/metrics | grep mycelix_
```

### Important URLs

| Service    | URL                          |
|------------|------------------------------|
| Health     | http://localhost:8080/health |
| Metrics    | http://localhost:9090/metrics|
| Prometheus | http://localhost:9091        |
| Grafana    | http://localhost:3000        |

### Contract Links (Sepolia Etherscan)

- [MycelixRegistry](https://sepolia.etherscan.io/address/0x69411e4c9D99814952EBA9a35028c12c2bbD03cd)
- [ReputationAnchor](https://sepolia.etherscan.io/address/0x042ee96Ce1CaFFF1e84d6da0585CD440d73DAF99)
- [PaymentRouter](https://sepolia.etherscan.io/address/0x3BD80003d27f9786bd4ffeaA5ffA7CC83365da1b)

---

*Last Updated: 2026-01-08*
*Version: Sepolia Testnet v1.0*
