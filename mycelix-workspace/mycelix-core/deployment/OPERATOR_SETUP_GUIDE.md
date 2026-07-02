# Mycelix Testnet Validator Operator Setup Guide

Welcome to the Mycelix community validator program! This guide will help you set up and operate a validator node on the Mycelix testnet.

---

## Table of Contents

1. [Introduction](#introduction)
2. [What is a Validator?](#what-is-a-validator)
3. [Requirements](#requirements)
4. [Quick Start (5 Minutes)](#quick-start-5-minutes)
5. [Detailed Setup](#detailed-setup)
6. [Configuration Options](#configuration-options)
7. [Monitoring Your Node](#monitoring-your-node)
8. [Best Practices](#best-practices)
9. [FAQ](#faq)
10. [Getting Help](#getting-help)

---

## Introduction

Mycelix is a decentralized federated learning network that enables privacy-preserving machine learning across distributed participants. Validators play a crucial role in:

- **Aggregating model updates** from FL participants
- **Detecting Byzantine behavior** using our breakthrough 0TML detection system
- **Maintaining network consensus** through the Holochain DHT
- **Ensuring data integrity** without accessing raw training data

By running a validator, you contribute to the security and decentralization of the Mycelix network while earning participation rewards (on mainnet).

---

## What is a Validator?

A Mycelix validator node consists of several components:

### FL Coordinator (0TML)
The core federated learning coordinator that:
- Receives gradient updates from participants
- Applies Byzantine-resilient aggregation (45% fault tolerance)
- Computes Shapley values for contribution tracking
- Heals corrupted gradients when possible

### Holochain Conductor
Manages the distributed hash table (DHT) for:
- Peer discovery and networking
- State synchronization
- Proof-of-Gradient-Quality (PoGQ) validation

### Metrics Exporter
Provides real-time monitoring data:
- Active node counts
- Round completion rates
- Detection statistics
- Latency measurements

### Health Server
Exposes endpoints for:
- Liveness checks (`/live`)
- Readiness checks (`/ready`)
- Detailed health status (`/health`)

---

## Requirements

### Minimum Hardware

| Component | Requirement |
|-----------|-------------|
| CPU | 2 cores x86_64 |
| RAM | 4 GB |
| Storage | 50 GB SSD |
| Network | 100 Mbps, stable connection |

### Recommended Hardware

| Component | Requirement |
|-----------|-------------|
| CPU | 4+ cores x86_64 |
| RAM | 8+ GB |
| Storage | 100+ GB NVMe SSD |
| Network | 1 Gbps, low latency |

### Software

- **Operating System:** Ubuntu 22.04 LTS (recommended), Debian 12, or any Linux with Docker support
- **Docker:** Version 24.0 or higher
- **Docker Compose:** Version 2.20 or higher

### Network

- Static IP address or DNS hostname
- Open outbound ports 443 (HTTPS) and UDP for WebRTC
- Inbound ports can be firewalled except for peer connections

---

## Quick Start (5 Minutes)

For experienced operators who want to get started quickly:

```bash
# 1. Clone and enter deployment directory
git clone https://github.com/luminous-dynamics/mycelix-core.git
cd mycelix-core/deployment

# 2. Create data directories
mkdir -p ./data/{validator,keystore,logs,cache,prometheus,grafana}

# 3. Configure (change VALIDATOR_NODE_ID and VALIDATOR_OPERATOR)
cp .env.example .env
sed -i 's/validator-001/validator-'$(hostname)'/g' .env
sed -i 's/community/my-organization/g' .env

# 4. Start
docker compose up -d

# 5. Verify
curl http://localhost:8080/health | jq .
```

For detailed setup instructions, continue reading below.

---

## Detailed Setup

### Step 1: Prepare Your Server

First, ensure your server meets the requirements and has Docker installed:

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Docker (if not already installed)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Install Docker Compose (if not already installed)
sudo apt install docker-compose-plugin -y

# Log out and back in for group membership to take effect
# Then verify installation
docker --version
docker compose version
```

### Step 2: Clone the Repository

```bash
# Clone the Mycelix Core repository
git clone https://github.com/luminous-dynamics/mycelix-core.git
cd mycelix-core/deployment
```

### Step 3: Create Data Directories

```bash
# Create directories for persistent data
mkdir -p ./data/validator
mkdir -p ./data/keystore
mkdir -p ./data/logs
mkdir -p ./data/cache
mkdir -p ./data/prometheus
mkdir -p ./data/grafana

# Set permissions
chmod -R 755 ./data
```

### Step 4: Configure Your Validator

Copy and edit the configuration file:

```bash
cp .env.example .env
```

Edit `.env` with your preferred text editor. **You MUST change these values:**

```bash
# Your unique validator identifier
# Format: alphanumeric with hyphens, no spaces
# Example: validator-us-east-alice, validator-tokyo-bob
VALIDATOR_NODE_ID=validator-your-unique-name

# Your geographic region (helps with peer optimization)
# Options: na (North America), eu (Europe), asia, oceania, sa (South America), africa, global
VALIDATOR_REGION=na

# Your organization or username (for network statistics)
VALIDATOR_OPERATOR=your-organization-name

# Change the Grafana password!
GRAFANA_ADMIN_PASSWORD=your-secure-password-here
```

### Step 5: Start Your Validator

```bash
# Start all services
docker compose up -d

# Watch the logs to ensure proper startup
docker compose logs -f validator
```

You should see output like:

```
mycelix-validator  | Mycelix Validator Node starting
mycelix-validator  | version=1.0.0 mode=validator node_id=validator-your-unique-name
mycelix-validator  | Initializing FL Coordinator...
mycelix-validator  | FL Coordinator initialized successfully
mycelix-validator  | Health server started port=8080
mycelix-validator  | Metrics server started port=9090
mycelix-validator  | Validator node is ready node_id=validator-your-unique-name
```

### Step 6: Verify Your Setup

```bash
# Check health endpoint
curl http://localhost:8080/health | jq .

# Expected output:
# {
#   "healthy": true,
#   "ready": true,
#   "checks": {
#     "coordinator": true,
#     "holochain": true,
#     "metrics": true
#   }
# }

# Check all containers are running
docker compose ps
```

---

## Configuration Options

### Essential Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `VALIDATOR_NODE_ID` | Unique identifier for your node | `validator-001` |
| `VALIDATOR_REGION` | Geographic region | `global` |
| `VALIDATOR_OPERATOR` | Your organization name | `community` |
| `NETWORK_TYPE` | Network to join | `testnet` |

### FL Coordinator Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `FL_MIN_NODES` | Minimum nodes for a round | `3` |
| `FL_MAX_NODES` | Maximum nodes per round | `100` |
| `FL_ROUND_TIMEOUT_SECONDS` | Round timeout | `120` |
| `FL_BYZANTINE_THRESHOLD` | Byzantine tolerance | `0.45` |

### Detection Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `MYCELIX_HV_DIMENSION` | Hypervector dimension (1024/2048/4096) | `2048` |
| `MYCELIX_USE_RUST` | Use Rust backend | `true` |

### Resource Limits

| Variable | Description | Default |
|----------|-------------|---------|
| `CPU_LIMIT` | Max CPU cores | `4` |
| `MEMORY_LIMIT` | Max memory | `4G` |
| `MAX_WORKERS` | Parallel workers | `4` |

---

## Monitoring Your Node

### Grafana Dashboard

Access your monitoring dashboard at `http://your-server-ip:3000`

Default credentials:
- Username: `admin`
- Password: (value of `GRAFANA_ADMIN_PASSWORD` in your `.env`)

The dashboard shows:
- **Active Nodes:** Current participants in FL rounds
- **Rounds Completed:** Total FL rounds processed
- **Byzantine Detection Rate:** Percentage of attacks detected
- **Aggregation Latency:** Time to complete aggregation
- **Model Convergence:** Current model accuracy metric

### Key Metrics to Watch

| Metric | Healthy Range | Alert Threshold |
|--------|--------------|-----------------|
| Detection Rate | >95% | <90% |
| Aggregation Latency | <100ms | >500ms |
| Active Nodes | >3 | <3 |
| Byzantine Ratio | <45% | >40% |

### Command-Line Monitoring

```bash
# Real-time logs
docker compose logs -f validator

# Resource usage
docker stats mycelix-validator

# Health status
curl -s http://localhost:8080/health | jq .

# Prometheus metrics
curl -s http://localhost:9090/metrics | grep mycelix
```

---

## Best Practices

### Security

1. **Change default passwords** in `.env`
2. **Use firewall rules** to restrict access:
   ```bash
   # Allow only necessary ports
   sudo ufw allow 9998/tcp  # Holochain app
   sudo ufw allow 8080/tcp  # Health check (optional, for load balancers)
   ```
3. **Keep Docker updated** for security patches
4. **Monitor logs** for suspicious activity

### Reliability

1. **Use persistent storage** on SSD/NVMe
2. **Configure automatic restarts** (already set in docker-compose)
3. **Set up monitoring alerts** via Grafana or external tools
4. **Plan for maintenance windows**

### Performance

1. **Place validator close to other nodes** in your region
2. **Use low-latency network connections**
3. **Avoid overcommitting resources** on shared hosts
4. **Monitor resource usage** and scale as needed

### Backups

```bash
# Create a backup of your validator data
docker compose stop validator
tar -czvf backup-$(date +%Y%m%d).tar.gz ./data
docker compose start validator
```

---

## FAQ

### Q: Do I need to stake tokens to run a validator?

**A:** On testnet, no staking is required. Mainnet will have staking requirements.

### Q: How much bandwidth does a validator use?

**A:** Typical usage is 1-5 GB/day depending on network activity and node count.

### Q: Can I run multiple validators on one server?

**A:** Yes, but use different ports and data directories. Each validator should have a unique `VALIDATOR_NODE_ID`.

### Q: How do I update my validator?

**A:**
```bash
git pull origin main
docker compose build --no-cache validator
docker compose up -d
```

### Q: What happens if my validator goes offline?

**A:** The network continues without you. When you come back online, your node will resync automatically.

### Q: How do I safely stop my validator?

**A:**
```bash
docker compose down
```

### Q: Can I change my VALIDATOR_NODE_ID?

**A:** Yes, but your node will appear as a new validator to the network.

### Q: What's the difference between testnet and mainnet?

**A:** Testnet is for testing and development. Mainnet (coming later) will handle real workloads and include economic incentives.

---

## Getting Help

### Documentation

- Full documentation: https://docs.mycelix.net
- Technical specifications: https://github.com/luminous-dynamics/mycelix-core/docs

### Community

- Discord: https://discord.gg/mycelix
- Forum: https://forum.mycelix.net
- Telegram: https://t.me/mycelix

### Bug Reports

- GitHub Issues: https://github.com/luminous-dynamics/mycelix-core/issues

### Commercial Support

For enterprise deployments, contact: enterprise@luminousdynamics.com

---

## Appendix: Network Diagram

```
                                 +-------------------+
                                 |   Bootstrap       |
                                 |   Servers         |
                                 +--------+----------+
                                          |
                    +---------------------+----------------------+
                    |                     |                      |
           +--------v-------+    +--------v-------+     +--------v-------+
           |  Your          |    |  Other         |     |  Other         |
           |  Validator     |    |  Validators    |     |  Validators    |
           |                |    |                |     |                |
           | +------------+ |    | +------------+ |     | +------------+ |
           | |FL Coord    | |    | |FL Coord    | |     | |FL Coord    | |
           | +------------+ |    | +------------+ |     | +------------+ |
           | |Holochain   | |    | |Holochain   | |     | |Holochain   | |
           | +------------+ |    | +------------+ |     | +------------+ |
           +--------+-------+    +--------+-------+     +--------+-------+
                    |                     |                      |
                    +---------------------+----------------------+
                                          |
                                 +--------v----------+
                                 |   FL              |
                                 |   Participants    |
                                 |   (Training)      |
                                 +-------------------+
```

---

## Changelog

### Version 1.0.0 (January 2026)

- Initial release of operator guide
- Support for testnet deployment
- Grafana dashboard included
- Prometheus metrics collection

---

*Thank you for contributing to the Mycelix network!*

*Luminous Dynamics Team*
