# Phase 2.5 Holochain DHT Setup Guide

**Status**: Implementation Complete, Testing Pending
**Date**: November 13, 2025

---

## Overview

This guide walks through setting up and testing the Holochain DHT-backed client registry for Phase 2.5 post-quantum authenticated federated learning.

### What Was Built

1. **pogq_zome_dilithium** - Extended Holochain zome with Dilithium authentication
2. **holochain_client_registry.py** - Python WebSocket connector
3. **zerotrustml.yaml** - DNA manifest for packaging
4. **conductor-config.yaml** - Conductor configuration for development

---

## Prerequisites

### System Requirements

- **Holochain**: v0.4.4+ (install from https://developer.holochain.org/install/)
- **Rust**: 1.70+ with `wasm32-unknown-unknown` target
- **Python**: 3.11+ (for client)
- **Node.js**: 16+ (for Holochain CLI tools)

### Install Holochain

```bash
# Option 1: Using Nix (recommended for NixOS)
nix-shell -p holochain

# Option 2: Using Cargo
cargo install holochain

# Option 3: Download binary
curl -L https://github.com/holochain/holochain/releases/download/holochain-0.4.4/holochain-x86_64-unknown-linux-gnu.tar.gz | tar xz
```

### Install Rust WASM Target

```bash
rustup target add wasm32-unknown-unknown
```

---

## Step 1: Compile the Zome to WASM

```bash
cd /srv/luminous-dynamics/Mycelix-Core/0TML/holochain/zomes/pogq_zome_dilithium

# Clean previous builds (optional)
cargo clean

# Build for WASM (this takes 5-10 minutes first time)
cargo build --target wasm32-unknown-unknown --release

# Verify output
ls -lh target/wasm32-unknown-unknown/release/pogq_zome_dilithium.wasm
```

**Expected Output**:
```
-rw-r--r-- 1 user user 1.2M Nov 13 pogq_zome_dilithium.wasm
```

---

## Step 2: Package the DNA

```bash
cd /srv/luminous-dynamics/Mycelix-Core/0TML/holochain

# Package DNA with hc (Holochain CLI)
hc dna pack dnas/zerotrustml.yaml

# Expected output: zerotrustml.dna
ls -lh zerotrustml.dna
```

**If `hc` is not available**:
```bash
# Install Holochain CLI
cargo install holochain_cli
```

---

## Step 3: Start the Holochain Conductor

### Create Data Directory

```bash
mkdir -p /tmp/holochain-zerotrustml/keystore
```

### Start Conductor

```bash
cd /srv/luminous-dynamics/Mycelix-Core/0TML/holochain

# Start conductor with config
holochain -c conductor-config.yaml

# Expected output:
# [INFO] Conductor running on ws://127.0.0.1:9000 (admin)
# [INFO] App interface running on ws://127.0.0.1:8888
```

**Keep this terminal open!** The conductor needs to keep running.

---

## Step 4: Install the ZeroTrustML App

In a **new terminal**:

```bash
# Install app using admin interface
hc app install \
  --app-id zerotrustml \
  --agent-key "$(hc keygen)" \
  --dna zerotrustml.dna \
  --conductor-url ws://localhost:9000

# Activate app
hc app enable zerotrustml
```

---

## Step 5: Test Python Client Connection

```bash
cd /srv/luminous-dynamics/Mycelix-Core/0TML

# Install Python dependencies (if not already installed)
pip install websocket-client

# Test connection
python src/zerotrustml/gen7/holochain_client_registry.py
```

**Expected Output**:
```
Phase 2.5 Week 2: Holochain Client Registry Example
============================================================
⚠️  This example requires a running Holochain conductor with zerotrustml app installed
⚠️  Start conductor with: holochain -c conductor-config.yaml

✅ Connected to Holochain conductor

📝 Registering client: 3a7f2e1d9c4b...
✅ Client registered: 3a7f2e1d9c4b...

📊 Submitting authenticated proof...
✅ Proof accepted: Proof published successfully

✅ Connection closed
```

---

## Step 6: Integration with E7.5 Federated Learning

### Update AuthenticatedGradientCoordinator

```python
from zerotrustml.gen7.holochain_client_registry import HolochainClientRegistry

coordinator = AuthenticatedGradientCoordinator(
    backend="holochain",
    conductor_url="ws://localhost:8888",
    app_id="zerotrustml",
)
```

### Run E7.5 Experiment

```bash
cd /srv/luminous-dynamics/Mycelix-Core/0TML

# Run federated learning with Holochain backend
python experiments/run_e7_5_with_holochain.py --rounds 5 --clients 10
```

---

## Troubleshooting

### Error: "Can't connect to conductor"

**Cause**: Conductor not running or wrong port.

**Fix**:
```bash
# Check if conductor is running
ps aux | grep holochain

# Check ports are available
lsof -i :8888
lsof -i :9000

# Restart conductor
holochain -c conductor-config.yaml
```

### Error: "Nonce already used"

**Cause**: Replay attack prevention working correctly.

**Fix**: Use a fresh nonce for each proof submission.

```python
import os
nonce = os.urandom(32)  # Generate new nonce each time
```

### Error: "Client not registered"

**Cause**: Client needs to register before submitting proofs.

**Fix**:
```python
# Register client first
client_id = registry.register_client(dilithium_pubkey)

# Then submit proofs
registry.submit_proof(client_id=client_id, ...)
```

### Error: "Timestamp out of bounds"

**Cause**: System clocks out of sync or proof too old/future.

**Fix**:
```bash
# Sync system clock
sudo ntpdate pool.ntp.org

# Or use current time in Python
import time
timestamp = int(time.time() * 1_000_000)
```

### Compilation Error: "can't find crate for `core`"

**Cause**: WASM target not installed.

**Fix**:
```bash
rustup target add wasm32-unknown-unknown
```

---

## Performance Testing

### Measure Proof Submission Latency

```python
import time
from zerotrustml.gen7.holochain_client_registry import HolochainClientRegistry

registry = HolochainClientRegistry()

# Measure 100 proof submissions
latencies = []
for i in range(100):
    start = time.time()
    is_valid, error = registry.submit_proof(...)
    latency = time.time() - start
    latencies.append(latency)

print(f"Average latency: {sum(latencies) / len(latencies):.3f}s")
print(f"P50 latency: {sorted(latencies)[50]:.3f}s")
print(f"P95 latency: {sorted(latencies)[95]:.3f}s")
```

### Measure DHT Query Performance

```python
# Measure client info lookups
start = time.time()
client_info = registry.get_client_info(client_id)
lookup_time = time.time() - start
print(f"Client info lookup: {lookup_time:.3f}s")

# Measure participation stats aggregation
start = time.time()
stats = registry.get_participation_stats(client_id)
stats_time = time.time() - start
print(f"Stats aggregation: {stats_time:.3f}s")
```

---

## Next Steps

### Week 2 Remaining Tasks (Nov 14-20)

1. ✅ **Compile zome to WASM** - In progress
2. ✅ **Add missing zome functions** - Complete
3. ✅ **Create DNA manifest** - Complete
4. ✅ **Create conductor config** - Complete
5. ⏳ **Setup Holochain conductor** - Pending user execution
6. ⏳ **Test Python ↔ Holochain** - Pending conductor setup
7. ⏳ **Integration testing** - Pending
8. ⏳ **Performance optimization** - Pending
9. ⏳ **Update coordinator** - Pending
10. ⏳ **End-to-end E7.5 test** - Pending

### Production Considerations

1. **Dilithium Verification**: Currently placeholder (`Ok(true)`). Options:
   - External verification service (coordinator verifies before DHT)
   - Wait for WASM-compatible Dilithium library
   - Use lighter post-quantum signature (Falcon, Sphincs+)

2. **Scalability**: Current `get_participation_stats()` iterates through rounds 0-1000. For production:
   - Add client_id index to PoGQProofEntry links
   - Use range queries for bounded lookups
   - Cache aggregated stats

3. **Monitoring**: Add metrics collection:
   - Proof submission rate
   - DHT query latency
   - Nonce collision rate
   - Client registration count

---

## Reference

### Zome Functions Exposed

| Function | Description | Input | Output |
|----------|-------------|-------|--------|
| `register_client()` | Register Dilithium public key | `dilithium_pubkey: Vec<u8>` | `ActionHash` |
| `publish_pogq_proof()` | Submit authenticated proof | `PoGQProofEntry` | `EntryHash` |
| `verify_pogq_proof()` | Verify proof validity | `proof_hash: EntryHash` | `bool` |
| `publish_gradient()` | Submit gradient entry | `GradientEntry` | `EntryHash` |
| `get_round_gradients()` | Query round gradients | `round: u64` | `Vec<(GradientEntry, bool)>` |
| `get_client_info_public()` | Get client registration | `client_id: Vec<u8>` | `Option<ClientRegistration>` |
| `get_participation_stats()` | Aggregate client stats | `client_id: Vec<u8>` | `ParticipationStats` |
| `list_all_clients()` | List all registered clients | - | `Vec<(Vec<u8>, ClientRegistration)>` |
| `health_check()` | Conductor health status | - | `HealthStatus` |

### Entry Types

- **PoGQProofEntry**: zkSTARK proof + Dilithium signature + metadata
- **NonceEntry**: Replay attack prevention
- **GradientEntry**: Gradient data with validation metadata
- **ClientRegistration**: Dilithium public key + reputation
- **ParticipationRecord**: Client participation tracking

---

**End of Setup Guide**

For questions or issues, see:
- `PHASE_2_5_HOLOCHAIN_DHT_DESIGN.md` - Architecture documentation
- `PHASE_2_5_WEEK2_SESSION_SUMMARY_Nov13.md` - Implementation details
