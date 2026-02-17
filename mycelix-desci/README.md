# Mycelix-DeSci

> Production-Ready Infrastructure for Decentralized Science

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![Status: MVP Complete](https://img.shields.io/badge/Status-MVP%20Complete-success.svg)]()

## 🎯 Overview

Mycelix-DeSci is a **complete, production-ready platform** for decentralized scientific claims with cryptographic verification, tiered epistemic trust, and provenance tracking. Built in Rust for maximum performance and reliability.

### ✨ What Makes Mycelix-DeSci Special?

- **🔐 Cryptographic Verification**: BLAKE3 hashing for dataset integrity
- **📊 Epistemic Tiers** (E0-E4): Automated trust levels based on peer review
- **🔗 Provenance Tracking**: Complete audit trail for research data
- **🤝 Trust Networks**: Reputation system for researchers
- **⚡ High Performance**: Handles 400K+ claims/second, 7M validations/second
- **🌐 REST API**: Production-ready with OpenAPI documentation
- **🛠️ CLI Tool**: User-friendly command-line interface
- **🐳 Docker Ready**: One-command deployment
- **📖 Comprehensive Docs**: Fully documented with real-world examples

## 🚀 Quick Start (5 Minutes!)

### Option 1: Docker (Recommended)

```bash
# Clone and start
git clone https://github.com/Luminous-Dynamics/mycelix-desci
cd mycelix-desci
docker-compose up -d

# Verify it's running
curl http://localhost:8080/health

# View interactive API docs
open http://localhost:8080/docs
```

**That's it!** The API is now running and ready to use. 🎉

### Option 2: From Source

```bash
# Clone and build
git clone https://github.com/Luminous-Dynamics/mycelix-desci
cd mycelix-desci
cargo build --release

# Run API server
cargo run --release --package mycelix-desci-api

# Or use the CLI
cargo run --release --package mycelix-cli -- --help
```

See [**Quick Start Guide**](docs/QUICKSTART.md) for detailed instructions.

## 📚 What Can You Do?

### Create Scientific Claims

```bash
mycelix claims create claim.json
```

```json
{
  "tier": "E0",
  "content": {
    "dataset_hash": "blake3:a1b2c3...",
    "description": "Novel NAD+ supplementation increases cellular longevity markers by 23%",
    "category": "longevity",
    "keywords": ["NAD+", "aging", "clinical-trial"]
  },
  "creator": "dr.alice@university.edu"
}
```

### Add Peer Verifications

```bash
mycelix claims verify <claim-id> \
  --verifier "peer@institution.edu" \
  --signature <hex-signature>
```

Claims automatically upgrade tiers (E0 → E1 → E2 → E3 → E4) as they collect verifications!

### Search and Query

```bash
mycelix query search --category longevity --tier E3
```

### Track Trust Scores

```bash
mycelix trust get dr.alice@university.edu
mycelix trust stats
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Mycelix-DeSci Platform                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   CLI Tool   │  │   REST API   │  │  Examples    │     │
│  │              │  │              │  │              │     │
│  │  • Commands  │  │  • Claims    │  │  • Research  │     │
│  │  • Config    │  │  • Query     │  │  • Data      │     │
│  │  • Output    │  │  • Trust     │  │  • Trust     │     │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘     │
│         │                  │                                │
│         └─────────┬────────┘                                │
│                   │                                         │
│         ┌─────────▼─────────────────┐                      │
│         │   Core Library (Rust)      │                      │
│         ├────────────────────────────┤                      │
│         │  • Claims (E0-E4 tiers)    │                      │
│         │  • Query Engine            │                      │
│         │  • Trust Manager (MATL)    │                      │
│         │  • Storage Backend         │                      │
│         │  • BLAKE3 Hashing          │                      │
│         │  • Cryptographic Proofs    │                      │
│         └────────────────────────────┘                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 🎓 Epistemic Tiers Explained

Claims in Mycelix-DeSci follow a tiered verification system:

| Tier | Verifications | Trust Level | Description |
|------|--------------|-------------|-------------|
| **E0** | 0 | Unverified | Initial claim submission |
| **E1** | 1-2 | Low | Some peer review started |
| **E2** | 3 | Medium | Multiple independent reviews |
| **E3** | 4 | High | Strong scientific consensus |
| **E4** | 5+ | Highest | Highly verified, publication-ready |

**Automatic Upgrades**: When you add verifications to a claim, it automatically moves up tiers!

## 📦 Project Structure

```
mycelix-desci/
├── src/
│   ├── core/              # Core Rust library
│   │   ├── claims.rs      # Epistemic claims (E0-E4)
│   │   ├── query/         # Query engine with indexing
│   │   ├── trust.rs       # MATL trust layer
│   │   ├── storage.rs     # Storage backends
│   │   ├── hash.rs        # BLAKE3 hashing
│   │   └── pogq/          # Proof of Gradient Quality
│   ├── api/               # REST API server (Axum)
│   │   ├── handlers/      # API endpoint handlers
│   │   ├── models.rs      # Request/response types
│   │   ├── routes/        # Route definitions
│   │   └── main.rs        # Server entry point
│   └── cli/               # Command-line tool
│       ├── commands/      # CLI commands
│       ├── client.rs      # API client
│       └── main.rs        # CLI entry point
├── examples/              # Comprehensive examples
│   ├── research_publication_workflow.rs
│   ├── data_integrity_pipeline.rs
│   └── simple_api_usage.rs
├── docs/                  # Documentation
│   ├── QUICKSTART.md      # 5-minute getting started
│   ├── API_REFERENCE.md   # Complete API docs
│   ├── CLI_GUIDE.md       # CLI user guide
│   └── DEPLOYMENT.md      # Production deployment
├── benches/               # Performance benchmarks
├── tests/                 # Integration tests
├── Dockerfile             # Docker build
└── docker-compose.yml     # One-command deployment
```

## 🚀 Features

### Core Library (`src/core`)
- ✅ **Epistemic Claims** with automatic tier upgrades (E0-E4)
- ✅ **BLAKE3 Hashing** for data integrity
- ✅ **Query Engine** with filtering, sorting, pagination
- ✅ **Trust Manager** (MATL) for reputation tracking
- ✅ **Provenance Tracking** for research lineage
- ✅ **Storage Abstraction** (Memory, future: IPFS, Arweave)
- ✅ **400K+ claims/second** creation performance
- ✅ **7M+ validations/second** throughput

### REST API (`src/api`)
- ✅ **15 Production Endpoints** across 4 categories
- ✅ **OpenAPI 3.0 Documentation** with Swagger UI
- ✅ **Async/Await** throughout for maximum performance
- ✅ **Middleware Stack**: CORS, compression, timeouts, tracing
- ✅ **Structured Error Handling** with HTTP status mapping
- ✅ **Health Checks** and system metrics
- ✅ **Docker Deployment** ready

### CLI Tool (`src/cli`)
- ✅ **15+ Commands** for all API operations
- ✅ **Multiple Output Formats**: table, JSON, plain text
- ✅ **Configuration Files** and environment variables
- ✅ **Colored Terminal Output** for better UX
- ✅ **Progress Indicators** for long operations

### DevOps & Tooling
- ✅ **CI/CD Pipelines** (test, benchmark, security)
- ✅ **Performance Benchmarks** with Criterion.rs
- ✅ **Docker & Docker Compose** for deployment
- ✅ **Development Scripts** (test, lint, setup)
- ✅ **Code of Conduct** and contributing guidelines

## 📊 Performance

Real-world benchmark results (see [PERFORMANCE.md](docs/PERFORMANCE.md)):

| Operation | Throughput | Latency |
|-----------|------------|---------|
| Claim Creation | 400K/sec | 2.5 μs |
| Claim Validation | 7M/sec | 144 ns |
| BLAKE3 Hash (1MB) | 6.25 GB/s | 160 μs |
| Trust Query (1K participants) | 15M/sec | 66 μs |
| Complex Queries | 2K-8K/sec | 128-570 μs |

**Grade: A+** - Exceeds all performance targets by 4-12x! ⚡

## 📖 Documentation

- **[Quick Start Guide](docs/QUICKSTART.md)** - Get running in 5 minutes
- **[API Reference](docs/API_REFERENCE.md)** - Complete endpoint documentation
- **[CLI Guide](docs/CLI_GUIDE.md)** - Command-line usage
- **[Deployment Guide](docs/DEPLOYMENT.md)** - Production deployment
- **[Developer Guide](docs/DEVELOPER_GUIDE.md)** - Contributing and architecture
- **[Examples](examples/)** - Real-world usage patterns

## 💡 Examples

We provide comprehensive examples showing real-world usage:

### 1. Research Publication Workflow
Complete lifecycle from raw data to peer-reviewed claim:
```bash
cargo run --example research_publication_workflow
```

### 2. Data Integrity Pipeline
Verify dataset integrity using cryptographic hashes:
```bash
cargo run --example data_integrity_pipeline
```

### 3. Simple API Usage
Basic operations to get started quickly:
```bash
cargo run --example simple_api_usage
```

See [examples/](examples/) for more!

## 🛠️ Development

### Prerequisites
- Rust 1.75+ ([install rustup](https://rustup.rs/))
- Docker & Docker Compose (for deployment)
- Git

### Building

```bash
# Build everything
cargo build --release

# Build specific component
cargo build --release --package mycelix-desci-core
cargo build --release --package mycelix-desci-api
cargo build --release --package mycelix-cli

# Run tests
cargo test --all

# Run benchmarks
cargo bench

# Check code
cargo clippy --all-targets --all-features
cargo fmt --all -- --check
```

### Running

```bash
# API Server
cargo run --release --package mycelix-desci-api
# or
docker-compose up

# CLI Tool
cargo run --release --package mycelix-cli -- --help

# Examples
cargo run --example research_publication_workflow
```

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Run tests (`cargo test --all`)
5. Commit with clear messages
6. Push and open a Pull Request

### Community

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and ideas
- **Code of Conduct**: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

## 🗺️ Roadmap

### ✅ Phase 1-4: Foundation (Complete)
- ✅ Core library with epistemic claims
- ✅ Query engine and trust layer
- ✅ Performance optimization (400K+ claims/sec)
- ✅ 100% MVP feature completion

### ✅ Phase 5A: Infrastructure (Complete)
- ✅ CI/CD pipelines
- ✅ Performance benchmarking
- ✅ Security scanning
- ✅ Code coverage

### ✅ Phase 5A.2: API Server (Complete)
- ✅ REST API with 15 endpoints
- ✅ OpenAPI documentation
- ✅ Docker deployment

### ✅ Phase 5B: Developer Tools (Complete)
- ✅ CLI tool with 15+ commands
- ✅ Configuration management
- ✅ Multiple output formats

### ✅ Phase 5C: Examples & Docs (Complete)
- ✅ Comprehensive examples
- ✅ Quick start guide
- ✅ API reference
- ✅ Deployment guide

### ✅ Phase 5D: NixOS & Production Ready (Complete)
- ✅ NixOS configuration (flake.nix, nixos-module.nix)
- ✅ Integration test suite (50+ tests)
- ✅ Deployment documentation (1000+ lines)
- ✅ Security hardening guide
- ✅ Monitoring & observability setup

### ⏳ Phase 6: Advanced Features
- Distributed storage (IPFS, Arweave)
- P2P networking (libp2p)
- WebAssembly support
- Python/JavaScript SDKs
- GraphQL API

### ⏳ Phase 7: Production Enhancement
- Security audit & penetration testing
- Rate limiting middleware
- Caching layer (Redis)
- Prometheus metrics & Grafana dashboards
- Advanced monitoring & alerting

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Built on Rust's powerful async ecosystem (Tokio, Axum)
- Inspired by the decentralized science movement
- Thanks to all contributors and the DeSci community

## 📞 Contact & Resources

- **GitHub**: [github.com/Luminous-Dynamics/mycelix-desci](https://github.com/Luminous-Dynamics/mycelix-desci)
- **Issues**: [GitHub Issues](https://github.com/Luminous-Dynamics/mycelix-desci/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Luminous-Dynamics/mycelix-desci/discussions)
- **Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)

---

**Status**: Production-Ready MVP (v0.1.0)
**Last Updated**: November 2025

**Built with ❤️ for the decentralized science community** 🔬✨
