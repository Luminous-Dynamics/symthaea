# Mycelix Testnet Validator Deployment

This directory contains production-ready deployment configurations for running a Mycelix testnet validator node.

## Quick Start

```bash
# Initialize environment
./scripts/init.sh

# Edit configuration
nano .env

# Start validator
docker compose up -d

# View dashboard
open http://localhost:3000
```

## Directory Structure

```
deployment/
├── docker-compose.yml          # Main service definitions
├── Dockerfile                  # Validator container build
├── .env.example               # Configuration template
├── prometheus.yml             # Metrics collection config
├── alertmanager.yml           # Alert routing config
├── DEPLOYMENT_CHECKLIST.md    # Step-by-step deployment guide
├── OPERATOR_SETUP_GUIDE.md    # Community operator documentation
├── grafana/
│   ├── dashboards/
│   │   └── mycelix-validator.json    # Main monitoring dashboard
│   └── provisioning/
│       ├── dashboards/dashboards.yml
│       └── datasources/prometheus.yml
├── scripts/
│   ├── init.sh                # Environment initialization
│   └── validator_entrypoint.py # Container entrypoint
└── testnet/                   # Full testnet infrastructure
    ├── kubernetes/            # K8s manifests
    └── terraform/             # Cloud infrastructure
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| Validator | 9090, 8080 | FL Coordinator + Health |
| Prometheus | 9091 | Metrics collection |
| Grafana | 3000 | Dashboards |
| Holochain | 39329, 9998 | DHT conductor |

## Documentation

- [Deployment Checklist](DEPLOYMENT_CHECKLIST.md) - Complete deployment instructions
- [Operator Setup Guide](OPERATOR_SETUP_GUIDE.md) - For community validators

## Requirements

- Docker 24.0+
- Docker Compose v2.20+
- 4 GB RAM minimum
- 50 GB SSD storage

## Support

- Documentation: https://docs.mycelix.net
- Discord: https://discord.gg/mycelix
- Issues: https://github.com/luminous-dynamics/mycelix-core/issues
