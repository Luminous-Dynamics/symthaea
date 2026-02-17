# Contributing to Mycelix

Welcome to Mycelix! This guide will help you get started contributing to the project.

**Last Updated**: January 2026
**Project Status**: Testnet Ready

---

## Quick Start

### Prerequisites

```bash
# Rust (1.75+)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable
rustup component add clippy rustfmt

# Node.js (20+) - for contract development
curl -fsSL https://fnm.vercel.app/install | bash
fnm use 20

# Docker - for local services
# Install from https://docs.docker.com/get-docker/

# Holochain (optional, for zome development)
nix-shell https://holochain.love
```

### Clone and Build

```bash
# Clone the repository
git clone https://github.com/mycelix/core.git
cd core

# Build core libraries
cd libs/kvector-zkp && cargo build --release && cd ../..
cd libs/feldman-dkg && cargo build --release && cd ../..
cd libs/matl-bridge && cargo build --release && cd ../..
cd libs/rb-bft-consensus && cargo build --release && cd ../..

# Run tests
cd libs/kvector-zkp && cargo test && cd ../..
```

---

## Repository Layout

```
Mycelix-Core/
├── libs/                    # Core Rust libraries (primary development)
│   ├── kvector-zkp/         # ZK proofs for K-Vector trust metrics
│   ├── feldman-dkg/         # Distributed key generation
│   ├── matl-bridge/         # Multi-Agent Trust Layer
│   ├── rb-bft-consensus/    # 45% Byzantine fault tolerant consensus
│   ├── fl-aggregator/       # Federated learning aggregation
│   ├── differential-privacy/# DP mechanisms
│   └── homomorphic-encryption/# HE operations
├── zomes/                   # Holochain zomes
├── contracts/               # Solidity smart contracts
├── deployment/              # Infrastructure (Terraform, K8s, Docker)
├── tests/                   # Integration and security tests
└── docs/                    # Documentation
```

---

## Component Status (January 2026)

| Component | Location | Status | Test Coverage |
|-----------|----------|--------|---------------|
| K-Vector ZK Proofs | `libs/kvector-zkp/` | Production | 85% |
| Feldman DKG | `libs/feldman-dkg/` | Production | 90% |
| MATL Bridge | `libs/matl-bridge/` | Production | 82% |
| RB-BFT Consensus | `libs/rb-bft-consensus/` | Production | 88% |
| FL Aggregator | `libs/fl-aggregator/` | Production | 82% |
| Smart Contracts | `contracts/` | Production | 92% |
| Infrastructure | `deployment/` | Production | N/A |

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for detailed status.

---

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name    # New features
git checkout -b fix/issue-description        # Bug fixes
git checkout -b docs/what-you-document       # Documentation
```

### 2. Make Changes

```bash
# Format code
cargo fmt

# Run linter
cargo clippy --all-targets --all-features -- -D warnings

# Run tests
cargo test

# Run benchmarks (if applicable)
cargo bench
```

### 3. Commit Using Conventional Commits

```bash
git commit -m "feat(kvector): add batch proof generation"
git commit -m "fix(consensus): handle view change timeout"
git commit -m "docs(readme): update build instructions"
git commit -m "test(fl): add Byzantine detection tests"
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

### 4. Submit Pull Request

PR should include:
- Description of changes
- Related issue numbers
- Test results
- Documentation updates (if applicable)

---

## Coding Standards

### Rust

```rust
// Document public APIs
/// Generates a ZK proof for the given K-Vector.
///
/// # Arguments
/// * `k_vector` - The K-Vector to prove
/// * `mode` - Security level (Fast, Standard, High)
///
/// # Returns
/// A `Result` containing the proof or an error
pub fn generate_proof(k_vector: &KVector, mode: SecurityLevel) -> Result<Proof, Error> {
    // Implementation
}

// Use meaningful constants
const MAX_BYZANTINE_TOLERANCE: f64 = 0.45;

// Handle errors explicitly
let result = operation().map_err(|e| MyError::OperationFailed(e))?;
```

### Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_happy_path() {
        // Arrange
        let input = create_test_input();

        // Act
        let result = function_under_test(input);

        // Assert
        assert!(result.is_ok());
    }
}
```

---

## Key Components

### K-Vector ZK Proofs
Generates STARK proofs for K-Vector range validation.
- Key files: `prover.rs`, `verifier.rs`, `air.rs`
- Target: <10s proof generation

### MATL Bridge
Trust scoring with formula: T = 0.4×PoGQ + 0.3×TCDM + 0.3×Entropy
- Key files: `pogq.rs`, `tcdm.rs`, `entropy.rs`

### RB-BFT Consensus
Reputation-weighted Byzantine consensus with 45% tolerance.
- Key files: `consensus.rs`, `reputation.rs`, `byzantine.rs`

---

## Testing

```bash
# Unit tests
cargo test

# Integration tests
cd tests/integration && cargo test

# Python integration tests
cd tests/integration && pytest -v

# Security tests
cd tests/security && cargo test
```

---

## Deployment

```bash
# Deploy testnet infrastructure
./scripts/deploy/terraform-deploy.sh testnet apply

# Deploy Kubernetes workloads
./scripts/deploy/kubernetes-deploy.sh testnet deploy
```

See [Deployment Runbook](docs/operations/DEPLOYMENT_RUNBOOK.md).

---

## Getting Help

- **Slack**: #dev-help
- **GitHub Issues**: Bugs and feature requests
- **Docs**: `cargo doc --open` for API documentation
- **Architecture**: `docs/architecture/`

---

## Security

Report vulnerabilities to: security@luminous-dynamics.org

- Never commit secrets or credentials
- Don't disable security checks
- Security-sensitive code requires additional review

---

## Code Review Checklist

- [ ] Code compiles without warnings
- [ ] All tests pass
- [ ] Code formatted (`cargo fmt`)
- [ ] Clippy passes
- [ ] Documentation updated
- [ ] No secrets in code
- [ ] New code has tests

---

Thank you for contributing to Mycelix!
