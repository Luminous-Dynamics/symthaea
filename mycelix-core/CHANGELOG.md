# Changelog

All notable changes to the Mycelix ecosystem are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Infrastructure & Deployment (January 2026)

- **Terraform Infrastructure** (`deployment/terraform/`)
  - Complete AWS infrastructure as code for testnet deployment
  - VPC module with 3-tier subnet architecture and NAT gateways
  - EKS cluster with managed node groups (validators, FL nodes)
  - RDS PostgreSQL with Multi-AZ support
  - ElastiCache Redis cluster with encryption
  - S3 buckets for models, backups, and logs
  - Comprehensive security groups per component
  - Support for dev, staging, and production workspaces

- **Kubernetes Manifests** (`deployment/kubernetes/`)
  - Namespace definitions with Pod Security Standards enforcement
  - Validator StatefulSet with persistent storage and HPA
  - FL Aggregator and Coordinator Deployments
  - ConfigMaps for application and Prometheus configuration
  - External Secrets integration with AWS Secrets Manager
  - NetworkPolicies for traffic isolation
  - Ingress with TLS via cert-manager
  - PodDisruptionBudgets and PriorityClasses
  - ResourceQuotas per namespace

- **CI/CD Pipeline** (`.github/workflows/testnet-deploy.yml`)
  - Multi-stage Docker builds with layer caching
  - Security scanning (Trivy, cargo-audit, npm audit)
  - SBOM generation with Syft
  - Container image signing with Sigstore/cosign
  - Automated testnet deployment on main branch
  - Manual workflow dispatch for deploy/rollback/restart/status
  - Slack notifications on success/failure
  - Rollback capability with state preservation

- **Docker Images** (`deployment/docker/`)
  - `Dockerfile.validator` - Multi-stage build for RB-BFT validators
  - `Dockerfile.fl-aggregator` - FL aggregation service with privacy
  - `Dockerfile.fl-coordinator` - Round coordination service
  - Non-root user execution
  - Optimized layer caching
  - Health check endpoints

- **Deployment Scripts** (`scripts/deploy/`)
  - `terraform-deploy.sh` - Full infrastructure deployment automation
    - Workspace management (dev/staging/prod)
    - Pre-deployment validation
    - Automatic backup of tfstate
    - Rollback on failure
  - `kubernetes-deploy.sh` - Application deployment automation
    - Namespace verification
    - Secret synchronization
    - Rolling update with health checks
    - Component scaling controls
    - Rollback capability

#### Configuration & Monitoring

- **Application Configs** (`deployment/config/`)
  - `validator.toml` - Consensus parameters, reputation thresholds
  - `fl-aggregator.toml` - Byzantine detection, privacy epsilon, ZK settings
  - `fl-coordinator.toml` - Round scheduling, participant requirements

- **Grafana Dashboards** (`deployment/grafana/dashboards/`)
  - `validator-overview.json` - Block production, reputation, consensus timing
  - `fl-pipeline.json` - Submissions, Byzantine detection, ZK proof metrics

- **Prometheus Configuration** (`deployment/prometheus/`)
  - `prometheus-dev.yml` - Local development scrape configs

#### Documentation

- **Validator Program** (`docs/validators/`)
  - `GENESIS_VALIDATOR_PROGRAM.md` - Requirements, benefits, selection process
  - `APPLICATION_TEMPLATE.md` - Structured validator application form
  - `VALIDATOR_ECONOMICS.md` - Token distribution, rewards, slashing
  - `OUTREACH_TEMPLATES.md` - Community outreach email templates

- **Operations** (`docs/operations/`)
  - `DEPLOYMENT_RUNBOOK.md` - Step-by-step deployment procedures
  - `MAINNET_ROADMAP.md` - Q1-Q4 2026 milestones to mainnet

- **Project Documentation**
  - `PROJECT_STATUS.md` - Comprehensive ecosystem status (65-70% complete)
  - `CONTRIBUTING.md` - Updated developer onboarding guide

#### Development Environment

- **Pre-commit Hooks** (`.pre-commit-config.yaml`)
  - Rust: cargo fmt, clippy, test
  - Python: ruff linting and formatting, mypy type checking
  - Solidity: forge fmt
  - Shell: shellcheck
  - YAML/JSON/TOML validation
  - Terraform: fmt and validate
  - Dockerfile: hadolint
  - Security: detect-secrets, detect-private-key
  - Commit message: commitizen format enforcement

- **Developer Setup** (`scripts/setup-dev.sh`)
  - Prerequisite checking (rust, python, docker, git)
  - Rust component installation (clippy, rustfmt)
  - Pre-commit hook installation
  - Secrets baseline creation
  - Local environment file generation
  - Optional library building
  - Local services startup

- **Local Development Stack** (`deployment/docker-compose.dev.yml`)
  - PostgreSQL 15 with initialization script
  - Redis 7 for caching
  - Prometheus (monitoring profile)
  - Grafana (monitoring profile)
  - LocalStack (aws profile) for S3/SecretsManager/DynamoDB
  - Jaeger (tracing profile) for distributed tracing

- **Database Schema** (`deployment/init-db.sql`)
  - Schemas: mycelix, fl, consensus
  - Tables: validators, consensus.rounds, fl.rounds, fl.submissions
  - Byzantine detection tracking
  - Trust history with PoGQ/TCDM/Entropy components
  - Metrics storage
  - Read-only and application user roles

### Security

- Identified and documented credential exposure issues for remediation:
  - Supabase keys in `.env.local`
  - JWT secret in `.env.development`
  - Age identity key exposure
- Added security scanning to CI pipeline
- Implemented container image signing
- Created secrets baseline for pre-commit

### Performance

- Documented ZK proof optimization target (18s -> <10s)
- GPU acceleration roadmap (CUDA/Metal backends)
- Configured autoscaling policies for validators and FL nodes

## [0.1.0] - 2025-12-XX

### Core Libraries

Initial release of production-ready core libraries:

- **kvector-zkp** (1,031 LOC) - STARK-based zero-knowledge proofs for trust vectors
- **feldman-dkg** (2,169 LOC) - Distributed key generation with verifiable secret sharing
- **matl-bridge** (1,863 LOC) - Multi-Agent Trust Layer with PoGQ, TCDM, Entropy
- **rb-bft-consensus** (2,184 LOC) - Reputation-based BFT with 45% Byzantine tolerance
- **fl-aggregator** (22,396 LOC) - Federated learning with Byzantine detection
- **differential-privacy** (2,058 LOC) - DP mechanisms for gradient privacy
- **homomorphic-encryption** (1,516 LOC) - Secure aggregation primitives
- **mycelix-core-types** (2,048 LOC) - Shared type definitions

### Holochain Zomes

- **pogq_validation** (818 LOC) - Proof of Gradient Quality validation
- **federated_learning** (5,934 LOC) - FL coordinator and participant logic
- **bridge** (1,554 LOC) - Cross-chain communication
- **agents** (532 LOC) - Agent lifecycle management

### Smart Contracts

- **MycelixRegistry.sol** (621 LOC) - Core registry for validators
- **ReputationAnchor.sol** (409 LOC) - On-chain reputation anchoring
- **ContributionRegistry.sol** (743 LOC) - Contribution tracking
- **ModelRegistry.sol** (718 LOC) - ML model registration
- **PaymentRouter.sol** (644 LOC) - Payment distribution

### Symthaea Integration

- **mycelix_bridge.rs** (1,114 LOC) - Consciousness-gated governance
- **GIS v4.0** (10,849 LOC) - Graceful Ignorance System
- **mapper.rs** (252 LOC) - Phi to Economic/Neural/Mythic mapping
- **kosmic_song.rs** (966 LOC) - Identity harmonics
- **network.rs** (485 LOC) - Network K-Vector computation

### Application Modules

- **mycelix-governance** (9,518 LOC) - Governance with 6 zomes
- **mycelix-finance** (9,044 LOC) - DeFi primitives with 9 zomes
- **mycelix-property** (2,670 LOC) - Property rights with 5 zomes
- **mycelix-praxis** (~220,000 LOC) - Educational platform with 10 zomes
- **mycelix-knowledge** (10,730 LOC) - Knowledge graphs with 7 zomes
- **mycelix-energy** (2,955 LOC) - Energy markets with 4 zomes
- **mycelix-identity** (~2,000 LOC) - Identity management with 7 zomes
- **mycelix-justice** (~2,500 LOC) - Dispute resolution with 5 zomes
- **mycelix-media** (~2,000 LOC) - Content distribution with 5 zomes

---

## Roadmap

### Q1 2026 - Foundation
- [x] Core library completion
- [x] Infrastructure as code
- [x] CI/CD pipeline
- [ ] Security audit preparation
- [ ] Testnet infrastructure deployment

### Q2 2026 - Security & Performance
- [ ] External security audits (Trail of Bits, OpenZeppelin)
- [ ] ZK proof optimization
- [ ] GPU acceleration
- [ ] Bug bounty program launch

### Q3 2026 - Testnet Operations
- [ ] Public testnet launch
- [ ] Validator onboarding (21+ genesis validators)
- [ ] Stress testing and optimization
- [ ] Documentation finalization

### Q4 2026 - Mainnet
- [ ] Final security review
- [ ] Genesis ceremony
- [ ] Mainnet launch
- [ ] Post-launch monitoring (72+ hours)

---

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development setup and contribution guidelines.

## License

[License information]
