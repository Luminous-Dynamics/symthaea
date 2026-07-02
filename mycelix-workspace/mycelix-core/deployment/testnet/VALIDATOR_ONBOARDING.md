# Mycelix Validator Onboarding Checklist

## Overview

This document provides a step-by-step checklist for becoming a Mycelix validator
for the mainnet launch. Validators are critical infrastructure providers who
participate in consensus and FL aggregation.

## Prerequisites

### Hardware Requirements

| Tier | CPU | RAM | Storage | Network | Role |
|------|-----|-----|---------|---------|------|
| Minimum | 4 cores | 8 GB | 100 GB NVMe | 100 Mbps | Observer |
| Recommended | 8 cores | 16 GB | 250 GB NVMe | 500 Mbps | Validator |
| High Performance | 16+ cores | 32 GB | 500 GB NVMe | 1 Gbps | Genesis Validator |

### Software Requirements

- [ ] Linux (Ubuntu 22.04+ recommended) or macOS 12+
- [ ] Docker 24.0+ with Docker Compose v2
- [ ] Nix (optional, for reproducible builds)
- [ ] Git

### Network Requirements

- [ ] Static public IP address
- [ ] Open ports: 9000 (P2P), 9090 (Metrics), 8080 (Health)
- [ ] Firewall configured per security recommendations
- [ ] DDoS protection (recommended for genesis validators)

## Onboarding Steps

### Phase 1: Registration (Week 1)

- [ ] **Fill out validator application**
  - Organization name
  - Technical contact
  - Geographic region
  - Hardware specifications
  - Expected uptime commitment

- [ ] **Sign validator agreement**
  - Review terms of service
  - Understand slashing conditions
  - Acknowledge responsibilities

- [ ] **Join validator communication channels**
  - Discord: #validators
  - Telegram: @mycelix_validators
  - Mailing list: validators@mycelix.net

### Phase 2: Setup (Week 2)

- [ ] **Provision infrastructure**
  ```bash
  # Clone repository
  git clone https://github.com/mycelix/core.git
  cd core/deployment/testnet

  # Copy environment template
  cp .env.example .env

  # Generate node identity
  ./scripts/generate-identity.sh
  ```

- [ ] **Configure node**
  - Edit `.env` with your settings
  - Set `VALIDATOR_NODE_ID` (unique identifier)
  - Set `VALIDATOR_REGION` (na, eu, asia, etc.)
  - Set `VALIDATOR_OPERATOR` (your organization)

- [ ] **Set up monitoring**
  ```bash
  # Enable monitoring profile
  docker compose --profile monitoring up -d
  ```

- [ ] **Configure backups**
  - Set up automated backups
  - Test backup restoration
  - Document recovery procedure

### Phase 3: Testing (Week 3-4)

- [ ] **Join testnet**
  ```bash
  docker compose up -d
  ```

- [ ] **Verify node health**
  ```bash
  # Check node status
  curl http://localhost:8080/health

  # Check peer connections
  curl http://localhost:8080/peers

  # Check FL participation
  curl http://localhost:8080/fl/status
  ```

- [ ] **Complete test scenarios**
  - [ ] Node successfully syncs
  - [ ] Node participates in FL rounds
  - [ ] Node survives restart
  - [ ] Node handles network partition
  - [ ] Monitoring alerts fire correctly

- [ ] **Submit verification proof**
  ```bash
  ./scripts/submit-verification.sh
  ```

### Phase 4: Genesis Preparation (Week 5)

- [ ] **Generate genesis keys**
  ```bash
  ./scripts/generate-genesis-keys.sh
  ```

- [ ] **Submit genesis parameters**
  - Public key
  - Initial stake amount
  - Commission rate
  - Node URL

- [ ] **Verify genesis inclusion**
  - Check genesis.json includes your validator
  - Verify stake amount
  - Confirm commission rate

### Phase 5: Mainnet Launch (Week 6+)

- [ ] **Prepare for launch**
  - Clear testnet data
  - Apply mainnet configuration
  - Final infrastructure review

- [ ] **Launch sequence**
  ```bash
  # Stop testnet
  docker compose down

  # Switch to mainnet config
  cp .env.mainnet .env

  # Start mainnet node
  docker compose up -d
  ```

- [ ] **Post-launch verification**
  - Node is producing blocks
  - FL rounds are completing
  - Monitoring is operational
  - Alerts are configured

## Validator Responsibilities

### Uptime Requirements

| Tier | Required Uptime | Grace Period | Slashing |
|------|-----------------|--------------|----------|
| Genesis | 99.5% | 1 hour | 0.5% stake |
| Active | 99.0% | 4 hours | 0.25% stake |
| Inactive | 95.0% | 24 hours | Warning |

### Security Requirements

- [ ] Run behind firewall
- [ ] Use dedicated hardware (no shared hosting)
- [ ] Implement key management with HSM or air-gapped
- [ ] Enable intrusion detection
- [ ] Regular security updates

### Communication Requirements

- [ ] Respond to security alerts within 2 hours
- [ ] Participate in upgrade coordination
- [ ] Report issues promptly
- [ ] Attend monthly validator calls

## Troubleshooting

### Node Won't Start

```bash
# Check logs
docker compose logs -f validator

# Verify configuration
./scripts/verify-config.sh

# Reset state (WARNING: deletes data)
./scripts/reset-node.sh
```

### Not Participating in FL

```bash
# Check FL coordinator connection
curl http://localhost:8080/fl/coordinator

# Verify trust score
curl http://localhost:8080/trust/score

# Force rejoin
./scripts/rejoin-fl.sh
```

### Low Trust Score

1. Check gradient quality
2. Verify network connectivity
3. Review recent FL participation
4. Contact support if persistent

## Support

- **Documentation**: https://docs.mycelix.io/validators
- **Discord**: https://discord.gg/mycelix
- **Email**: validators@mycelix.net
- **Emergency**: Use PagerDuty (genesis validators only)

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-16 | Platform Team | Initial version |
