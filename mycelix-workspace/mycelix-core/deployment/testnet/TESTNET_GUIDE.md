# Mycelix Public Testnet Guide

Welcome to the Mycelix Public Testnet! This guide will help you join the network, participate in federated learning, and contribute to the ecosystem.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Joining as a Node Operator](#joining-as-a-node-operator)
- [Submitting Gradients](#submitting-gradients)
- [Monitoring Your Node](#monitoring-your-node)
- [Getting Testnet Tokens](#getting-testnet-tokens)
- [API Reference](#api-reference)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Security Requirements](#security-requirements)
- [Reporting Issues](#reporting-issues)

## Overview

The Mycelix Testnet is a public testing environment for the Mycelix federated learning network. It allows developers and researchers to:

- Run nodes and participate in decentralized federated learning
- Test gradient submission and aggregation
- Experiment with the tokenomics system
- Develop and test applications on the network

### Network Information

| Property | Value |
|----------|-------|
| Network Name | `mycelix-testnet` |
| Chain ID | `mycelix-testnet-v1` |
| Token Symbol | `MYC` |
| Block Time | ~5 seconds |
| FL Round Duration | ~5 minutes |
| Minimum Participants | 3 |

### Bootstrap Nodes

```
testnet.mycelix.network:9000
testnet-bootstrap-1.mycelix.network:9000
testnet-seed-1.mycelix.network:9000
testnet-seed-2.mycelix.network:9000
```

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/mycelix/core.git
cd core/deployment/testnet

# Run the join script
./scripts/join-testnet.sh --docker --name my-node
```

### Option 2: Manual Docker Run

```bash
docker run -d \
  --name mycelix-node \
  -p 9000:9000 \
  -p 9001:9001 \
  -v ~/.mycelix:/data \
  -e MYCELIX_NETWORK=testnet \
  -e BOOTSTRAP_NODES=testnet.mycelix.network:9000 \
  mycelix/fl-node:testnet-latest
```

### Option 3: Native Binary

```bash
# Download the latest release
curl -L https://github.com/mycelix/core/releases/latest/download/mycelix-linux-amd64 -o mycelix
chmod +x mycelix

# Run the node
./mycelix node start \
  --network testnet \
  --bootstrap testnet.mycelix.network:9000
```

## Joining as a Node Operator

### Prerequisites

- **Hardware Requirements:**
  - CPU: 2+ cores
  - RAM: 4GB minimum, 8GB recommended
  - Disk: 20GB+ SSD
  - Network: Stable internet connection, 10Mbps+

- **Software Requirements:**
  - Docker 20.10+ (recommended) or
  - Go 1.21+ (for building from source)
  - Linux, macOS, or Windows with WSL2

### Step-by-Step Setup

1. **Create a data directory:**
   ```bash
   mkdir -p ~/.mycelix
   ```

2. **Generate node identity:**
   ```bash
   # The join script handles this automatically
   # Or manually:
   openssl rand -hex 32 > ~/.mycelix/node.key
   ```

3. **Download genesis configuration:**
   ```bash
   curl -o ~/.mycelix/genesis.json \
     https://testnet.mycelix.network/genesis.json
   ```

4. **Run the join script:**
   ```bash
   ./scripts/join-testnet.sh \
     --name my-awesome-node \
     --type participant \
     --p2p-port 9000 \
     --api-port 9001
   ```

### Node Types

| Type | Description | Requirements |
|------|-------------|--------------|
| `participant` | Participates in FL rounds, submits gradients | Default, no stake required |
| `seed` | Helps with peer discovery and network stability | Higher uptime requirements |
| `validator` | Validates blocks and FL rounds | Stake required (future) |

### Configuration Options

Create a `~/.mycelix/config.json` for advanced configuration:

```json
{
  "node": {
    "id": "my-node",
    "type": "participant"
  },
  "network": {
    "listen_address": "0.0.0.0",
    "p2p_port": 9000,
    "api_port": 9001,
    "max_peers": 50
  },
  "federated_learning": {
    "enabled": true,
    "auto_participate": true,
    "model_path": "/data/models"
  },
  "logging": {
    "level": "info",
    "format": "json"
  }
}
```

## Submitting Gradients

### Automatic Participation

By default, nodes automatically participate in FL rounds when they have compatible data:

```bash
# Enable auto-participation (default)
docker exec mycelix-node mycelix fl enable-auto
```

### Manual Gradient Submission

For more control, you can submit gradients manually:

```bash
# Get current round info
curl http://localhost:9001/fl/status

# Submit gradients
curl -X POST http://localhost:9001/fl/gradients \
  -H "Content-Type: application/json" \
  -d '{
    "round_id": 123,
    "gradient_data": "<base64-encoded-gradient>",
    "model_version": "v1.0.0"
  }'
```

### Using the Python SDK

```python
from mycelix import Client, FederatedLearning

# Connect to your local node
client = Client("http://localhost:9001")

# Get FL instance
fl = client.federated_learning()

# Wait for a round
round_info = fl.wait_for_round()

# Train your model locally and compute gradients
# ... your training code ...

# Submit gradients
result = fl.submit_gradient(
    round_id=round_info.round_id,
    gradient=my_gradient,
    metadata={"epochs": 10, "samples": 1000}
)

print(f"Gradient submitted! TX: {result.transaction_hash}")
```

### Gradient Requirements

- **Format:** Compressed numpy arrays (msgpack or protobuf)
- **Max Size:** 10MB per submission
- **Validation:** Gradients are validated for:
  - Correct dimensions
  - Numeric stability (no NaN/Inf)
  - Differential privacy compliance
  - Byzantine resistance checks

## Monitoring Your Node

### Using the Monitor Script

```bash
# Show overall status
./scripts/monitor-testnet.sh status

# Watch live dashboard
./scripts/monitor-testnet.sh watch

# Check health
./scripts/monitor-testnet.sh health

# View FL status
./scripts/monitor-testnet.sh fl

# Check logs
./scripts/monitor-testnet.sh logs
```

### Web Dashboards

| Dashboard | URL | Description |
|-----------|-----|-------------|
| Grafana | http://localhost:3000 | Metrics visualization |
| Node API | http://localhost:9001 | Node status and API |
| Network Explorer | https://explorer.testnet.mycelix.network | Block explorer |

Default Grafana credentials:
- Username: `admin`
- Password: Check your `.env` file or `GRAFANA_PASSWORD`

### Key Metrics to Watch

| Metric | Description | Healthy Range |
|--------|-------------|---------------|
| `mycelix_peers_connected` | Number of connected peers | 3-50 |
| `mycelix_block_time_seconds` | Average block time | 4-6 seconds |
| `mycelix_fl_active_participants` | FL participants | 3+ |
| `mycelix_fl_gradients_submitted_total` | Gradients submitted | Increasing |
| `mycelix_reputation_score` | Your reputation | 80-100 |

### API Endpoints

```bash
# Node health
curl http://localhost:9001/health

# Network status
curl http://localhost:9001/network/info

# Peer list
curl http://localhost:9001/network/peers

# FL status
curl http://localhost:9001/fl/status

# Your node stats
curl http://localhost:9001/node/stats
```

## Getting Testnet Tokens

### Faucet

Request free testnet tokens from the faucet:

**Web Interface:**
https://faucet.testnet.mycelix.network

**API:**
```bash
curl -X POST https://faucet.testnet.mycelix.network/api/request \
  -H "Content-Type: application/json" \
  -d '{"address": "YOUR_NODE_ADDRESS"}'
```

**Limits:**
- 1000 MYC per request
- 24-hour cooldown per IP/address
- Maximum 10 requests per address

### Earning Tokens

You can earn testnet tokens by:

1. **Submitting valid gradients:** 1 MYC per accepted gradient
2. **Validating blocks:** 0.5 MYC per validated block
3. **Model improvements:** Up to 10 MYC for significant improvements

## API Reference

### Node API (port 9001)

#### Health Check
```
GET /health
Response: { "status": "healthy", "version": "1.0.0" }
```

#### Network Info
```
GET /network/info
Response: {
  "chain_id": "mycelix-testnet-v1",
  "block_height": 12345,
  "peer_count": 15,
  "synced": true
}
```

#### FL Status
```
GET /fl/status
Response: {
  "current_round": 100,
  "round_status": "active",
  "participants_count": 5,
  "model_version": "v1.0.0"
}
```

#### Submit Gradient
```
POST /fl/gradients
Content-Type: application/json

{
  "round_id": 100,
  "gradient_data": "<base64>",
  "model_version": "v1.0.0",
  "metadata": {}
}

Response: {
  "accepted": true,
  "gradient_hash": "0x...",
  "transaction_hash": "0x..."
}
```

## Known Limitations

### Current Testnet Limitations

1. **Performance:**
   - Block time may vary under high load
   - FL round duration is fixed at 5 minutes
   - Maximum 50 participants per FL round

2. **Features:**
   - Staking/slashing not yet implemented
   - Governance features limited
   - Some privacy features in beta

3. **Stability:**
   - Network may be reset periodically
   - Data persistence not guaranteed
   - Breaking changes possible between versions

4. **Security:**
   - Do NOT use real private keys or sensitive data
   - Testnet tokens have no value
   - Smart contract audits pending

### Upcoming Improvements

- [ ] Horizontal scaling for FL aggregation
- [ ] Enhanced privacy with secure aggregation
- [ ] Validator staking mechanism
- [ ] Cross-shard communication
- [ ] Mobile node support

## Troubleshooting

### Common Issues

#### Node won't connect to peers

```bash
# Check if ports are open
sudo netstat -tlnp | grep 9000

# Check firewall
sudo ufw status

# Allow ports if needed
sudo ufw allow 9000/tcp
sudo ufw allow 9001/tcp
```

#### Out of sync

```bash
# Check sync status
curl http://localhost:9001/network/info

# Restart node with fresh sync
docker stop mycelix-node
docker rm mycelix-node
rm -rf ~/.mycelix/db
./scripts/join-testnet.sh
```

#### Gradients rejected

```bash
# Check gradient status
curl http://localhost:9001/fl/gradients/status

# Common reasons:
# - Wrong model version
# - Invalid gradient format
# - Duplicate submission
# - Rate limited
```

#### High memory usage

```bash
# Reduce peer connections
docker exec mycelix-node mycelix config set max_peers 20

# Enable garbage collection
docker exec mycelix-node mycelix gc run
```

### Getting Logs

```bash
# Docker logs
docker logs -f mycelix-node

# With timestamps
docker logs -f --timestamps mycelix-node

# Last 100 lines
docker logs --tail 100 mycelix-node

# Export logs
docker logs mycelix-node > node.log 2>&1
```

## Security Requirements

### CRITICAL: Never Deploy with Default Credentials

Before deploying any node to the testnet, you MUST configure proper security credentials.

### Required Environment Variables

All passwords MUST be set before deployment. The following environment variables are required:

```bash
# Required - deployment will fail without these
POSTGRES_PASSWORD=your_secure_password_here
GRAFANA_PASSWORD=your_secure_password_here

# Recommended for production deployments
JWT_SECRET=your_256_bit_secret_key
REDIS_PASSWORD=your_secure_password_here
```

### Password Requirements

All passwords must meet these minimum requirements:

1. **Minimum Length:** 16 characters
2. **Character Mix:** Must include uppercase, lowercase, numbers, and special characters
3. **Uniqueness:** Each service should have a unique password
4. **Not in Weak List:** Must NOT be any of the following:
   - `admin`, `password`, `changeme`, `secret`
   - `postgres`, `grafana`, `mycelix`, `test`
   - `minioadmin`, `root`, `testnet_secret`
   - Any variation of the above

### Setting Up Secrets

#### Option 1: Generate Secrets Automatically (Recommended)

```bash
# From the repository root
cd deployment/testnet

# Generate secure secrets
./scripts/generate-secrets.sh

# This creates a .env file with secure random passwords
```

#### Option 2: Manual Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Generate secure passwords:
   ```bash
   # Generate a secure password
   openssl rand -base64 32

   # Generate a 256-bit key for JWT
   openssl rand -hex 32
   ```

3. Replace ALL placeholder values in `.env`

4. Verify your configuration:
   ```bash
   # Run the password validation script
   ../../scripts/validate-passwords.sh .env
   ```

### Pre-deployment Security Check

Before starting any deployment, run the pre-flight check:

```bash
# Full security check
./scripts/deployment-preflight.sh --env .env

# Or from repository root
./scripts/deployment-preflight.sh --compose deployment/testnet/
```

The pre-flight check validates:
- All required secrets are set
- No default/weak passwords are in use
- Docker-compose files don't contain hardcoded secrets
- Environment files have proper permissions

**Deployment will be BLOCKED if any critical security issues are found.**

### Docker-Compose Security

All docker-compose files use environment variable references with required validation:

```yaml
environment:
  # This syntax requires the variable to be set - deployment fails otherwise
  POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?POSTGRES_PASSWORD must be set}
```

### Grafana Dashboard Access

After deployment, Grafana dashboards are available at `http://localhost:3000`.

**Default username:** `admin`
**Password:** Must be set via `GRAFANA_PASSWORD` environment variable

To change the password after deployment:
```bash
docker exec mycelix-grafana grafana-cli admin reset-admin-password NEW_PASSWORD
```

### Security Best Practices

1. **Never commit secrets to git**
   - Add `.env` to `.gitignore`
   - Use `.env.example` as a template

2. **Rotate passwords regularly**
   - Especially after personnel changes
   - At least quarterly for production

3. **Use secret management**
   - Consider HashiCorp Vault, AWS Secrets Manager, or similar
   - See `scripts/bws-*` for Bitwarden Secrets Manager integration

4. **Monitor for unauthorized access**
   - Check Grafana login attempts
   - Monitor database connection logs

5. **Restrict network access**
   - Use firewalls to limit database access
   - Only expose necessary ports

### Reporting Security Issues

If you discover a security vulnerability, please report it privately:
- Email: security@mycelix.network
- Do NOT create public GitHub issues for security vulnerabilities

## Reporting Issues

### Before Reporting

1. Check existing issues: https://github.com/mycelix/core/issues
2. Search the documentation
3. Try the troubleshooting steps above
4. Collect relevant logs and information

### How to Report

**GitHub Issues:**
https://github.com/mycelix/core/issues/new

**Include:**
- Node version: `docker exec mycelix-node mycelix version`
- Operating system and version
- Docker version (if applicable)
- Relevant logs (last 100 lines)
- Steps to reproduce
- Expected vs actual behavior

**Discord:**
Join our Discord for real-time help: https://discord.gg/mycelix

**Template:**
```markdown
### Description
[Describe the issue]

### Environment
- OS: [e.g., Ubuntu 22.04]
- Node Version: [e.g., 1.0.0]
- Deployment: [Docker/Native/Kubernetes]

### Steps to Reproduce
1. ...
2. ...

### Expected Behavior
[What should happen]

### Actual Behavior
[What actually happens]

### Logs
```
[Paste relevant logs here]
```

### Additional Context
[Any other information]
```

## Community

- **Website:** https://mycelix.network
- **Documentation:** https://docs.mycelix.network
- **GitHub:** https://github.com/mycelix
- **Discord:** https://discord.gg/mycelix
- **Twitter:** https://twitter.com/mycelix

## License

This project is licensed under the MIT License. See [LICENSE](../../LICENSE) for details.

---

Thank you for participating in the Mycelix Testnet! Your contributions help build a more decentralized and privacy-preserving future for machine learning.
