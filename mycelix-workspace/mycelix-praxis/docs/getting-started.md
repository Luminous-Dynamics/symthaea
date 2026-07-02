# Getting Started with Praxis

Welcome to Mycelix Praxis! This guide will help you set up your development environment and make your first contribution.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Your First Contribution](#your-first-contribution)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)

---

## Prerequisites

### Required

- **Rust** (latest stable)
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  rustc --version  # Should be 1.70+
  ```

- **Node.js** (>= 18)
  ```bash
  # Using nvm (recommended)
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
  nvm install 20
  nvm use 20
  node --version  # Should be v18+
  ```

- **Git**
  ```bash
  git --version  # Should be 2.0+
  ```

### Optional (for full Holochain development)

- **Holochain dev tools** (will be needed in v0.2+)
  ```bash
  cargo install holochain_cli --version 0.3.0-beta-dev.39
  cargo install lair_keystore --version 0.4.4
  hc --version
  ```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Luminous-Dynamics/mycelix-praxis.git
cd mycelix-praxis
```

### 2. Build the Project

```bash
# Build all components (Rust + Web)
make build
```

**Expected output:**
```
▶ Building Rust workspace...
   Compiling praxis-core v0.1.0
   Compiling praxis-agg v0.1.0
   ...
   Finished `dev` profile

▶ Building web application...
   Installing dependencies...
   Building for production...
   ✓ Build complete
```

**Time**: ~2-5 minutes on first build (downloads dependencies)

### 3. Run Tests

```bash
make test
```

**Expected output:**
```
▶ Running Rust tests...
   running 14 tests
   test result: ok. 14 passed; 0 failed

▶ Running web tests...
   PASS  src/App.test.tsx
   ✓ All tests passed
```

---

## Project Structure

```
mycelix-praxis/
├── apps/web/              # React web client
│   ├── src/               # Source code
│   ├── public/            # Static assets
│   └── package.json       # Dependencies
├── crates/                # Rust libraries
│   ├── praxis-core/       # Core types, crypto, provenance
│   └── praxis-agg/        # Robust aggregation
├── zomes/                 # Holochain zomes (future HDK integration)
│   ├── learning_zome/     # Courses, progress
│   ├── fl_zome/           # Federated learning
│   ├── credential_zome/   # Verifiable Credentials
│   └── dao_zome/          # Governance
├── schemas/               # W3C VC + DHT entry schemas
├── docs/                  # Documentation
├── examples/              # Example data
├── scripts/               # Dev tooling
└── Makefile               # Build commands
```

**Key files:**
- `Cargo.toml`: Rust workspace configuration
- `README.md`: Project overview
- `CONTRIBUTING.md`: Contribution guidelines
- `docs/protocol.md`: FL protocol specification
- `docs/architecture.md`: System architecture

---

## Development Workflow

### Start Development Environment

```bash
make dev
```

This will:
1. Build Rust workspace
2. Install web dependencies
3. Start web dev server at `http://localhost:3000`

**Note**: Holochain conductor integration coming in v0.2

### Code Format

```bash
make fmt
```

Runs:
- `cargo fmt --all` (Rust)
- `npm run format` (Web - Prettier)

### Lint

```bash
make lint
```

Runs:
- `cargo clippy --all` (Rust)
- `npm run lint` (Web - ESLint)

### Watch Mode (Auto-rebuild)

```bash
# Rust (in one terminal)
cargo watch -x build

# Web (in another terminal)
cd apps/web && npm run dev
```

---

## Your First Contribution

### Step 1: Find an Issue

Browse [good first issues](https://github.com/Luminous-Dynamics/mycelix-praxis/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) or create a new one.

### Step 2: Create a Branch

```bash
git checkout -b feat/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

**Branch naming:**
- `feat/*`: New features
- `fix/*`: Bug fixes
- `docs/*`: Documentation updates
- `chore/*`: Maintenance tasks

### Step 3: Make Changes

**Example: Add a new aggregation method**

1. Edit `crates/praxis-agg/src/methods.rs`:
   ```rust
   /// Compute geometric mean of vectors
   pub fn geometric_mean(updates: &[Vec<f32>]) -> Result<Vec<f32>> {
       // Your implementation
   }
   ```

2. Add tests:
   ```rust
   #[test]
   fn test_geometric_mean() {
       let updates = vec![vec![1.0, 4.0], vec![4.0, 16.0]];
       let result = geometric_mean(&updates).unwrap();
       assert_eq!(result, vec![2.0, 8.0]);
   }
   ```

3. Run tests:
   ```bash
   cargo test -p praxis-agg
   ```

### Step 4: Commit

```bash
git add .
git commit -m "feat(agg): add geometric mean aggregation

Implements geometric mean as an alternative to trimmed mean
for scenarios where multiplicative relationships matter.

Closes #123"
```

**Commit format**: [Conventional Commits](https://www.conventionalcommits.org/)
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Tests
- `chore:` Maintenance

### Step 5: Push and Create PR

```bash
git push origin feat/your-feature-name
```

Then open a PR on GitHub. See [CONTRIBUTING.md](../CONTRIBUTING.md) for PR checklist.

---

## Testing

### Rust Tests

```bash
# All tests
cargo test --workspace

# Specific crate
cargo test -p praxis-core

# Specific test
cargo test test_trimmed_mean

# With output
cargo test -- --nocapture
```

### Web Tests

```bash
cd apps/web

# All tests
npm test

# Watch mode
npm test -- --watch

# Coverage
npm test -- --coverage
```

### Integration Tests (Future)

```bash
# End-to-end FL round
cargo test --test integration_fl_round
```

---

## Troubleshooting

### "error: could not compile `praxis-core`"

**Cause**: Missing Rust toolchain or dependencies

**Fix**:
```bash
rustup update stable
cargo clean
cargo build --workspace
```

### "npm ERR! ERESOLVE unable to resolve dependency tree"

**Cause**: Node version mismatch or conflicting dependencies

**Fix**:
```bash
cd apps/web
rm -rf node_modules package-lock.json
npm install
```

### "make: command not found"

**Cause**: Make not installed (common on Windows)

**Fix**:
```bash
# Windows (use WSL or Git Bash)
# Or run commands manually:
cargo build --workspace
cd apps/web && npm install && npm run build
```

### Slow Rust compilation

**Fix**: Use `sccache` to cache compilations
```bash
cargo install sccache
export RUSTC_WRAPPER=sccache
```

### "DHT entry not found"

**Cause**: Holochain conductor not running or DNA not installed

**Fix** (when Holochain integration is ready):
```bash
./scripts/hc-reset.sh  # Reset conductor state
make dev                # Restart
```

---

## Next Steps

### Learn the Codebase

1. **Read the docs**:
   - [Protocol Specification](protocol.md)
   - [Architecture Overview](architecture.md)
   - [Threat Model](threat-model.md)

2. **Explore examples**:
   ```bash
   cat examples/credentials/valid-achievement.json
   cat examples/fl-rounds/round-001-discover.json
   ```

3. **Run the web client**:
   ```bash
   make dev
   # Open http://localhost:3000
   ```

### Join the Community

- **GitHub Discussions**: Ask questions, share ideas
- **Issues**: Report bugs, request features
- **Pull Requests**: Contribute code

### Pick a Task

**Beginner-friendly:**
- Add more tests to `praxis-agg`
- Improve web UI components
- Write more examples
- Fix typos in documentation

**Intermediate:**
- Implement missing zome functions
- Add DAO governance logic
- Improve error handling

**Advanced:**
- Implement Holochain DNA
- Add differential privacy
- Optimize FL aggregation
- External security audit

---

## Quick Reference

### Common Commands

```bash
# Build
make build          # Build everything
make rust           # Build Rust only
make web            # Build web only

# Test
make test           # Run all tests
cargo test -p <pkg> # Test specific package

# Dev
make dev            # Start dev environment
make fmt            # Format code
make lint           # Lint code
make clean          # Clean build artifacts

# Reset
make reset          # Reset Holochain state (future)
```

### File Locations

- **Rust code**: `crates/*/src/`, `zomes/*/src/`
- **Web code**: `apps/web/src/`
- **Tests**: `**/tests/`, `**/*_test.rs`, `**/*.test.tsx`
- **Docs**: `docs/`, `README.md`
- **Examples**: `examples/`
- **Scripts**: `scripts/`

### Getting Help

- **Documentation**: `docs/`
- **FAQ**: [faq.md](faq.md)
- **Issues**: [GitHub Issues](https://github.com/Luminous-Dynamics/mycelix-praxis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Luminous-Dynamics/mycelix-praxis/discussions)

---

**Welcome to the team! 🎉**

We're excited to have you contributing to a more equitable, privacy-preserving future for education.
