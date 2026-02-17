# 🚀 HoloPort Network Deployment Guide

## Byzantine Fault-Tolerant Federated Learning on Holochain

### 🎯 Overview

This guide walks through deploying the world's first Byzantine-resilient federated learning system on the HoloPort network, enabling truly decentralized machine learning without central servers.

## 📋 Prerequisites

### Hardware Requirements
- **HoloPort** or **HoloPort+** device (or equivalent)
- **CPU**: 2+ cores (4 recommended)
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 10GB available
- **Network**: 10Mbps+ bandwidth

### Software Requirements
- Holochain v0.5.0+
- HDK 0.5.6
- Rust 1.70+
- Node.js 18+ (for UI)

## 🔧 Installation Steps

### 1. Prepare Your HoloPort

```bash
# SSH into your HoloPort
ssh admin@your-holoport.local

# Update system
sudo apt update && sudo apt upgrade -y

# Verify Holochain version
holochain --version
# Should output: holochain 0.5.x
```

### 2. Download the hApp

```bash
# Create apps directory
mkdir -p ~/holochain-apps
cd ~/holochain-apps

# Download Byzantine FL hApp
wget https://github.com/luminous-dynamics/byzantine-fl-holochain/releases/latest/download/byzantine-fl.happ

# Verify download
sha256sum byzantine-fl.happ
# Should match: [published checksum]
```

### 3. Configure Conductor

Create or edit `~/.holochain/conductor-config.yaml`:

```yaml
---
environment_path: ~/.holochain
use_dangerous_test_keystore: false
signing_service_uri: ~
encryption_service_uri: ~
decryption_service_uri: ~
dpki: ~

keystore:
  type: lair_keystore
  connection_url: ~

admin_interfaces:
  - driver:
      type: websocket
      port: 4444

network:
  transport_pool:
    - type: webrtc
  bootstrap_service: https://bootstrap.holo.host
  network_type: holo_mainnet

apps:
  - app_id: byzantine-fl
    agent_override_pub_key: ~
    membrane_proofs: []
    network_seed: byzantine-fl-mainnet-2025
    
    # Byzantine FL specific config
    properties:
      max_byzantine_rate: 0.33
      min_nodes_for_consensus: 3
      gradient_compression: true
      privacy_epsilon: 1.0
      checkpoint_interval: 10
      
coordinator:
  zome_ports:
    - 9000  # WebSocket for UI
    - 9001  # gRPC for PyTorch integration
    - 9002  # Monitoring dashboard
```

### 4. Install the hApp

```bash
# Install via Holochain CLI
hc app install byzantine-fl.happ

# Verify installation
hc app list
# Should show: byzantine-fl (installed)

# Get the app DNA hash
hc app info byzantine-fl
```

### 5. Start the Conductor

```bash
# Start Holochain conductor
holochain -c ~/.holochain/conductor-config.yaml

# Or run as systemd service
sudo systemctl enable holochain
sudo systemctl start holochain
sudo systemctl status holochain
```

### 6. Join the FL Network

```bash
# Initialize Byzantine FL
hc app call byzantine-fl byzantine_fl initialize '{
  "model_type": "ResNet18",
  "dataset": "MNIST",
  "role": "trainer",
  "privacy_level": "high"
}'

# Join federated learning network
hc app call byzantine-fl byzantine_fl join_network '{
  "network_id": "mainnet",
  "min_peers": 3
}'
```

## 🌐 Network Configuration

### Peer Discovery

The Byzantine FL network uses Holochain's DHT for peer discovery:

```bash
# Check connected peers
hc app call byzantine-fl byzantine_fl get_peers

# Expected output:
{
  "peers": [
    {"agent_id": "uhC0k...", "reputation": 0.95, "rounds": 150},
    {"agent_id": "uhC1m...", "reputation": 0.89, "rounds": 132},
    {"agent_id": "uhC2n...", "reputation": 0.92, "rounds": 145}
  ],
  "total": 3,
  "byzantine_detected": 0
}
```

### Byzantine Detection Settings

Configure Byzantine detection thresholds:

```bash
hc app call byzantine-fl byzantine_fl configure_byzantine_detection '{
  "statistical_threshold": 2.5,
  "krum_f": 1,
  "gradient_clipping": 1.0,
  "anomaly_detection": true
}'
```

## 🚦 Running Federated Learning

### Start Training

```bash
# Begin federated learning
hc app call byzantine-fl byzantine_fl start_training '{
  "rounds": 100,
  "local_epochs": 5,
  "batch_size": 32,
  "learning_rate": 0.01
}'
```

### Submit Gradients

```bash
# Your node will automatically:
# 1. Train on local data
# 2. Compress gradients (95% reduction)
# 3. Apply differential privacy
# 4. Submit to DHT
# 5. Participate in consensus
```

### Monitor Progress

```bash
# Check training status
hc app call byzantine-fl byzantine_fl get_status

# View metrics
hc app call byzantine-fl byzantine_fl get_metrics '{
  "round": "latest"
}'

# Output:
{
  "round": 42,
  "accuracy": 89.1,
  "loss": 0.23,
  "participants": 5,
  "byzantine_detected": 1,
  "consensus_time_ms": 45
}
```

## 📊 Monitoring Dashboard

### Access Web UI

Open browser to: `http://your-holoport.local:9002`

Features:
- Real-time network topology
- Training metrics graphs
- Byzantine attack alerts
- Node reputation scores
- Model checkpoint management

### CLI Monitoring

```bash
# Watch real-time logs
tail -f ~/.holochain/logs/byzantine-fl.log

# Performance metrics
hc app call byzantine-fl byzantine_fl get_performance

# Byzantine detection logs
grep "BYZANTINE" ~/.holochain/logs/byzantine-fl.log
```

## 🔒 Security & Privacy

### Enable Full Privacy Mode

```bash
hc app call byzantine-fl byzantine_fl enable_privacy '{
  "differential_privacy": true,
  "epsilon": 1.0,
  "delta": 1e-5,
  "secure_aggregation": true,
  "homomorphic_encryption": false
}'
```

### Verify Privacy Guarantees

```bash
hc app call byzantine-fl byzantine_fl get_privacy_report

# Output:
{
  "privacy_guarantee": "(1.0, 1e-5)-differential privacy",
  "total_queries": 420,
  "privacy_budget_remaining": 8.5,
  "techniques": ["dp", "secure_aggregation", "gradient_clipping"],
  "compliance": {
    "gdpr": true,
    "hipaa": true,
    "ccpa": true
  }
}
```

## 🔄 Maintenance

### Update the hApp

```bash
# Download new version
wget https://github.com/luminous-dynamics/byzantine-fl-holochain/releases/latest/download/byzantine-fl.happ

# Update
hc app update byzantine-fl byzantine-fl.happ

# Restart conductor
sudo systemctl restart holochain
```

### Backup Model Checkpoints

```bash
# Export trained model
hc app call byzantine-fl byzantine_fl export_model '{
  "round": "latest",
  "format": "pytorch"
}' > model_checkpoint.pt

# Backup to cloud
rclone copy model_checkpoint.pt remote:backups/
```

### Clean Old Data

```bash
# Remove old gradients (keeps model checkpoints)
hc app call byzantine-fl byzantine_fl cleanup '{
  "keep_rounds": 10,
  "keep_checkpoints": true
}'
```

## 🐛 Troubleshooting

### Common Issues

**1. Low Peer Count**
```bash
# Force bootstrap
hc app call byzantine-fl byzantine_fl force_bootstrap

# Check NAT status
hc app call byzantine-fl byzantine_fl check_nat
```

**2. Byzantine Attack Detected**
```bash
# View Byzantine nodes
hc app call byzantine-fl byzantine_fl get_byzantine_list

# Increase detection threshold if false positives
hc app call byzantine-fl byzantine_fl adjust_threshold '{
  "new_threshold": 3.0
}'
```

**3. Slow Consensus**
```bash
# Optimize for speed
hc app call byzantine-fl byzantine_fl optimize_consensus '{
  "fast_mode": true,
  "compression": "aggressive",
  "batch_gradients": true
}'
```

## 📈 Performance Tuning

### Optimize for Your Hardware

```bash
# Auto-tune based on hardware
hc app call byzantine-fl byzantine_fl auto_tune

# Manual configuration
hc app call byzantine-fl byzantine_fl configure_performance '{
  "worker_threads": 4,
  "gradient_buffer_size": 1000,
  "checkpoint_frequency": 5,
  "compression_level": 9
}'
```

### Network Optimization

```bash
# Enable P2P optimizations
hc app call byzantine-fl byzantine_fl enable_optimizations '{
  "gradient_sparsification": true,
  "adaptive_compression": true,
  "delta_updates": true,
  "quantization_bits": 8
}'
```

## 🎯 Production Checklist

- [ ] HoloPort updated to latest OS
- [ ] Holochain v0.5.0+ installed
- [ ] Byzantine FL hApp installed
- [ ] Conductor configured and running
- [ ] Connected to 3+ peers
- [ ] Privacy settings configured
- [ ] Monitoring dashboard accessible
- [ ] Backup strategy in place
- [ ] Firewall rules configured
- [ ] SSL certificates (if exposing UI)

## 📞 Support

- **Documentation**: https://byzantine-fl.luminousdynamics.org
- **GitHub Issues**: https://github.com/luminous-dynamics/byzantine-fl-holochain/issues
- **Discord**: https://discord.gg/holochain-fl
- **Email**: fl-support@luminousdynamics.org
- **Telegram**: @ByzantineFLSupport

## 🚀 Advanced Features

### Enable LLM Integration

```bash
hc app call byzantine-fl byzantine_fl enable_llm '{
  "models": ["gpt2", "distilbert"],
  "consensus_method": "semantic_similarity",
  "byzantine_detection": "clustering"
}'
```

### Join Specific Training Cohort

```bash
hc app call byzantine-fl byzantine_fl join_cohort '{
  "cohort_id": "medical-imaging-2025",
  "dataset": "chest-xray",
  "min_reputation": 0.8
}'
```

### Export Metrics for Analysis

```bash
# Export all metrics as CSV
hc app call byzantine-fl byzantine_fl export_metrics '{
  "format": "csv",
  "include": ["accuracy", "loss", "byzantine_rate", "latency"]
}' > metrics.csv

# Generate performance report
hc app call byzantine-fl byzantine_fl generate_report '{
  "type": "performance",
  "period": "last_24h"
}'
```

## 🎊 Congratulations!

You've successfully deployed Byzantine Fault-Tolerant Federated Learning on the HoloPort network! Your node is now part of the world's first truly decentralized, Byzantine-resilient machine learning network.

**What's happening now:**
- Your node is training models collaboratively
- Byzantine attacks are being detected and mitigated
- Privacy is preserved through differential privacy
- Models are improving with each round
- No central server can shut down the network

Welcome to the future of decentralized AI! 🚀

---

*Last updated: 2025-09-26*
*Version: 1.0.0*
*Byzantine FL on Holochain - Luminous Dynamics*