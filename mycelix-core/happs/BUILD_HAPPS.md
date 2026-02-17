# Building Mycelix-Core Holochain hApps

This document provides comprehensive instructions for building, packaging, and testing the Holochain hApps in the Mycelix-Core ecosystem.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Building Individual Zomes](#building-individual-zomes)
- [Packaging DNAs](#packaging-dnas)
- [Packaging hApps](#packaging-happs)
- [Testing with Tryorama](#testing-with-tryorama)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

#### 1. Rust Toolchain

Install Rust via rustup with the required target:

```bash
# Install rustup if not present
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Add WASM target for Holochain zomes
rustup target add wasm32-unknown-unknown

# Verify installation
rustc --version
cargo --version
```

**Minimum versions:**
- Rust: 1.75.0 or later
- Cargo: 1.75.0 or later

#### 2. Holochain CLI (hc)

Install the Holochain developer tools:

```bash
# Using Holonix (recommended)
nix-shell https://holochain.love

# Or install directly via cargo
cargo install holochain_cli --version 0.3.0

# Verify installation
hc --version
```

**Required Holochain version:** 0.3.x

#### 3. Node.js and npm (for Tryorama testing)

```bash
# Install Node.js 18+ (LTS recommended)
# Using nvm:
nvm install 18
nvm use 18

# Verify installation
node --version
npm --version
```

#### 4. Additional Dependencies

```bash
# Linux (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libssl-dev

# macOS (with Homebrew)
brew install openssl pkg-config

# Nix (cross-platform, recommended)
nix-shell -p openssl pkg-config
```

### Environment Setup

Create a `.env` file in the `happs/` directory:

```bash
# Holochain configuration
HOLOCHAIN_VERSION=0.3.0
HDK_VERSION=0.3.0

# Build configuration
CARGO_TARGET_DIR=./target
WASM_OPT_LEVEL=3

# Testing configuration
TRYORAMA_LOG_LEVEL=info
CONDUCTOR_TIMEOUT_MS=60000
```

---

## Project Structure

```
happs/
├── BUILD_HAPPS.md              # This file
├── HAPP_REGISTRY.md            # Registry of all hApps
├── build-happs.sh              # Automated build script
│
├── zerotrustml-identity.happ.yaml
├── zerotrustml-reputation.happ.yaml
├── zerotrustml-governance.happ.yaml
├── mycelix-mail.happ.yaml
├── mycelix-marketplace.happ.yaml
│
├── zomes/                      # Zome source code
│   ├── identity_core/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── lib.rs
│   ├── capability_tokens/
│   ├── reputation_ledger/
│   ├── behavioral_proofs/
│   ├── stake_registry/
│   ├── governance_core/
│   ├── policy_registry/
│   ├── prediction_markets/
│   ├── model_validation/
│   ├── mail_core/
│   ├── channels/
│   ├── attachments/
│   ├── notifications/
│   ├── marketplace_core/
│   ├── escrow/
│   ├── model_registry/
│   ├── reviews/
│   └── analytics/
│
├── dnas/                       # DNA manifest files
│   ├── zerotrustml-identity.yaml
│   ├── capability-tokens.yaml
│   ├── zerotrustml-reputation.yaml
│   ├── behavioral-proofs.yaml
│   ├── stake-registry.yaml
│   ├── zerotrustml-governance.yaml
│   ├── policy-registry.yaml
│   ├── prediction-markets.yaml
│   ├── model-validation.yaml
│   ├── mycelix-mail.yaml
│   ├── mail-channels.yaml
│   ├── mail-attachments.yaml
│   ├── mail-notifications.yaml
│   ├── mycelix-marketplace.yaml
│   ├── marketplace-escrow.yaml
│   ├── model-registry.yaml
│   ├── marketplace-reviews.yaml
│   └── marketplace-analytics.yaml
│
├── tests/                      # Tryorama test suites
│   ├── package.json
│   ├── tsconfig.json
│   └── src/
│       ├── identity.test.ts
│       ├── reputation.test.ts
│       ├── governance.test.ts
│       ├── mail.test.ts
│       └── marketplace.test.ts
│
└── dist/                       # Build outputs
    ├── zomes/                  # Compiled WASM zomes
    ├── dnas/                   # Packaged DNAs
    └── happs/                  # Final hApp bundles
```

---

## Building Individual Zomes

### Manual Build Process

Each zome is a Rust crate that compiles to WASM:

```bash
# Navigate to a zome directory
cd happs/zomes/identity_core

# Build the zome
cargo build --release --target wasm32-unknown-unknown

# The output will be at:
# target/wasm32-unknown-unknown/release/identity_core.wasm
```

### Zome Cargo.toml Template

Each zome requires a properly configured `Cargo.toml`:

```toml
[package]
name = "identity_core"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]
name = "identity_core"

[dependencies]
hdk = "0.3"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

[profile.release]
opt-level = "z"
lto = true
codegen-units = 1
panic = "abort"
```

### Building All Zomes

```bash
# From the happs directory
for zome_dir in zomes/*/; do
    echo "Building $(basename $zome_dir)..."
    (cd "$zome_dir" && cargo build --release --target wasm32-unknown-unknown)
done
```

### WASM Optimization

Optimize WASM files for smaller bundle sizes:

```bash
# Install wasm-opt (part of binaryen)
# Ubuntu/Debian
sudo apt-get install binaryen

# macOS
brew install binaryen

# Optimize a zome
wasm-opt -O3 -o optimized.wasm input.wasm
```

---

## Packaging DNAs

### DNA Manifest Structure

Each DNA requires a manifest file (`dna.yaml`):

```yaml
---
manifest_version: "1"
name: zerotrustml-identity
integrity:
  origin_time: 2024-01-01T00:00:00.000000Z
  network_seed: ~
  properties: ~
  zomes:
    - name: identity_core
      bundled: ../dist/zomes/identity_core.wasm
coordinator:
  zomes:
    - name: identity_core_coordinator
      bundled: ../dist/zomes/identity_core_coordinator.wasm
      dependencies:
        - name: identity_core
```

### Packaging a DNA

```bash
# Using hc CLI
hc dna pack ./dnas/zerotrustml-identity.yaml -o ./dist/dnas/zerotrustml-identity.dna

# Verify the DNA
hc dna info ./dist/dnas/zerotrustml-identity.dna
```

### DNA Hash Verification

After packaging, verify the DNA hash:

```bash
# Get DNA hash
hc dna hash ./dist/dnas/zerotrustml-identity.dna

# Output example:
# uhC0k...abc123 (base64 hash)
```

---

## Packaging hApps

### hApp Manifest Structure

The `.happ.yaml` files in this directory define complete hApp bundles:

```yaml
manifest_version: "1"
name: zerotrustml-identity
roles:
  - id: identity_core
    provisioning:
      strategy: create
      deferred: false
    dna:
      bundled: ../dist/dnas/zerotrustml-identity.dna
```

### Building a Complete hApp

```bash
# Package the hApp
hc app pack ./zerotrustml-identity.happ.yaml -o ./dist/happs/zerotrustml-identity.happ

# Verify the hApp bundle
hc app info ./dist/happs/zerotrustml-identity.happ
```

### Build Order (Dependency Resolution)

Due to inter-hApp dependencies, build in this order:

1. **zerotrustml-identity** (no dependencies)
2. **zerotrustml-reputation** (depends on identity)
3. **zerotrustml-governance** (depends on identity, reputation)
4. **mycelix-mail** (depends on identity)
5. **mycelix-marketplace** (depends on identity, reputation)

---

## Testing with Tryorama

### Setting Up Tryorama

```bash
cd happs/tests

# Initialize npm project (if not exists)
npm init -y

# Install dependencies
npm install @holochain/tryorama @holochain/client typescript ts-node
npm install --save-dev @types/node jest ts-jest
```

### Test Configuration

Create `tryorama.config.ts`:

```typescript
import { Config } from '@holochain/tryorama';

export const config: Config = {
  timeout: 60000,
  logger: {
    level: 'info',
  },
};
```

### Example Test Suite

```typescript
// tests/src/identity.test.ts
import { runScenario, pause, CallableCell } from '@holochain/tryorama';
import { ActionHash, Record } from '@holochain/client';
import path from 'path';

const hAppPath = path.join(__dirname, '../../dist/happs/zerotrustml-identity.happ');

describe('ZeroTrustML Identity', () => {
  it('creates an agent identity', async () => {
    await runScenario(async scenario => {
      // Set up conductor
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: hAppPath } }
      ]);

      // Create identity
      const identity: Record = await alice.cells[0].callZome({
        zome_name: 'identity_core',
        fn_name: 'create_agent_identity',
        payload: {
          agent_type: 'ml_model',
          metadata: {
            name: 'TestAgent',
            version: '1.0.0',
          },
        },
      });

      expect(identity).toBeDefined();
      expect(identity.signed_action).toBeDefined();
    });
  });

  it('verifies agent identity', async () => {
    await runScenario(async scenario => {
      const [alice, bob] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: hAppPath } },
        { appBundleSource: { path: hAppPath } },
      ]);

      // Alice creates identity
      const identity = await alice.cells[0].callZome({
        zome_name: 'identity_core',
        fn_name: 'create_agent_identity',
        payload: { agent_type: 'ml_model', metadata: {} },
      });

      await pause(500);

      // Bob verifies Alice's identity
      const verification = await bob.cells[0].callZome({
        zome_name: 'identity_core',
        fn_name: 'verify_agent_identity',
        payload: { agent_id: alice.agentPubKey },
      });

      expect(verification.valid).toBe(true);
    });
  });
});
```

### Running Tests

```bash
# Run all tests
npm test

# Run specific test suite
npm test -- --testPathPattern=identity

# Run with verbose output
npm test -- --verbose

# Run in watch mode
npm test -- --watch
```

### Test Coverage

```bash
# Generate coverage report
npm test -- --coverage

# View coverage report
open coverage/lcov-report/index.html
```

---

## Troubleshooting

### Common Build Issues

#### 1. WASM Target Not Installed

```
error[E0463]: can't find crate for `std`
```

**Solution:**
```bash
rustup target add wasm32-unknown-unknown
```

#### 2. HDK Version Mismatch

```
error: HDK version incompatible with Holochain
```

**Solution:** Ensure `hdk` version in Cargo.toml matches your Holochain installation:
```toml
hdk = "0.3"  # Must match Holochain 0.3.x
```

#### 3. DNA Packaging Fails

```
error: Failed to read zome WASM
```

**Solution:** Ensure zomes are built before packaging DNAs:
```bash
./build-happs.sh --zomes-only
./build-happs.sh --dnas-only
```

#### 4. Test Timeout

```
TimeoutError: Scenario timed out
```

**Solution:** Increase timeout in test configuration:
```typescript
await runScenario(async scenario => {
  // tests...
}, { timeout: 120000 });
```

### Logging and Debugging

Enable verbose logging:

```bash
# Build with debug output
RUST_LOG=debug cargo build --release --target wasm32-unknown-unknown

# Run tests with Holochain logs
RUST_LOG=holochain=debug npm test
```

### Getting Help

- Holochain Developer Forum: https://forum.holochain.org
- Holochain Discord: https://chat.holochain.org
- Mycelix-Core Issues: https://github.com/Luminous-Dynamics/Mycelix-Core/issues

---

## Continuous Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/build-happs.yml
name: Build hApps

on:
  push:
    paths:
      - 'happs/**'
  pull_request:
    paths:
      - 'happs/**'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Nix
        uses: cachix/install-nix-action@v24

      - name: Setup Holonix
        run: nix-shell https://holochain.love --run "hc --version"

      - name: Build hApps
        run: |
          cd happs
          ./build-happs.sh

      - name: Run Tests
        run: |
          cd happs/tests
          npm ci
          npm test

      - name: Upload Artifacts
        uses: actions/upload-artifact@v4
        with:
          name: happ-bundles
          path: happs/dist/happs/*.happ
```

---

## Version History

| Version | Date       | Changes                           |
|---------|------------|-----------------------------------|
| 0.1.0   | 2024-01-15 | Initial hApp packaging structure  |

---

*Last updated: 2024-01-15*
*Maintained by the Mycelix-Core Team*
