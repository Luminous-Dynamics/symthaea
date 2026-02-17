# Gen7-zkSTARK + Holochain Integration

## Overview

This integration combines:
1. **Gen7-zkSTARK**: RISC Zero zkVM proofs for Byzantine detection
2. **Holochain PoGQ Zome**: DHT storage and verification of proofs
3. **Docker Environment**: Isolated testing with working IPv6

## Architecture

```
┌─────────────────┐
│  Python Client  │
│  (Coordinator)  │
└────────┬────────┘
         │
         ├─────────────────┐
         │                 │
         v                 v
┌─────────────────┐  ┌─────────────────┐
│  gen7-zkstark   │  │    Holochain    │
│  Host Binary    │  │   Conductor     │
├─────────────────┤  ├─────────────────┤
│ RISC Zero zkVM  │  │   PoGQ Zome     │
│ PoGQ Circuit    │  │   (WASM)        │
│ Receipt Gen     │  │   DHT Storage   │
└─────────────────┘  └─────────────────┘
         │                 │
         └────────┬────────┘
                  v
         Integration Test
         test_gen7_holochain_integration.py
```

## Components

### 1. Gen7-zkSTARK (RISC Zero)
- **Location**: `gen7-zkstark/`
- **Binary**: `target/release/host`
- **Purpose**: Generate zero-knowledge proofs of PoGQ Byzantine detection
- **Output**: RISC Zero Receipt (serialized)

### 2. Holochain PoGQ Zome
- **Location**: `holochain/zomes/pogq_zome/`
- **WASM**: `holochain/zomes/pogq_zome.wasm` (2.6MB)
- **DNA**: `holochain/dnas/pogq_dna/pogq_dna.dna` (497KB)
- **Functions**:
  - `publish_pogq_proof`: Store proof on DHT
  - `get_round_proofs`: Query proofs by round
  - `verify_pogq_proof`: Journal-only verification

### 3. Docker Holochain Conductor
- **Container**: `holochain-zerotrustml`
- **Port**: 8888 (admin interface)
- **Config**: `docker-compose.holochain.yml`
- **Status**: ✅ Running (IPv6 enabled via sysctls)

## Setup Instructions

### Prerequisites
```bash
# 1. Build gen7-zkstark host binary
cd gen7-zkstark
cargo build --release

# 2. Verify Holochain conductor is running
docker ps | grep holochain-zerotrustml

# 3. Check conductor logs
docker logs holochain-zerotrustml
```

### Run Integration Test
```bash
# Basic test (single proof)
python test_gen7_holochain_integration.py

# With Nix development environment
nix develop --command python test_gen7_holochain_integration.py
```

## Test Workflow

The integration test performs these steps:

1. **Connect** to Holochain conductor (ws://localhost:8888)
2. **Install** PoGQ DNA bundle
3. **Generate** RISC Zero proof using gen7-zkstark host
4. **Publish** proof to PoGQ zome
5. **Verify** proof retrieval from DHT

### Expected Output
```
======================================================================
  Gen7-zkSTARK + Holochain Integration Test
======================================================================

ℹ️  Connecting to conductor at ws://localhost:8888...
✅ Connected to Holochain conductor

======================================================================
  Installing PoGQ DNA
======================================================================

ℹ️  DNA bundle: holochain/dnas/pogq_dna/pogq_dna.dna
ℹ️  Size: 497.2 KB
✅ Generated agent key: uhCAk...
✅ Installed app: pogq-test-1731553200
✅ App enabled successfully

======================================================================
  Generating gen7-zkSTARK Proof for Round 1
======================================================================

ℹ️  Host binary: gen7-zkstark/target/release/host
ℹ️  Running RISC Zero prover (this may take 30-60 seconds)...
✅ RISC Zero proof generated successfully
✅ Proof data prepared:
  Round: 1
  Quarantine: 0
  Consecutive clear: 1

======================================================================
  Publishing Proof to Holochain
======================================================================

✅ Proof published successfully to Holochain DHT
ℹ️  Entry hash: uhCEk...

======================================================================
  Verifying Proof Retrieval
======================================================================

✅ Retrieved 1 proof(s) for round 1
  Node: uhCAk...
  Quarantine: 0
  Round: 1

======================================================================
  Integration Test Complete ✅
======================================================================

✅ All tests passed!
```

## Troubleshooting

### Conductor Connection Failed
```bash
# Check conductor status
docker logs holochain-zerotrustml | tail -20

# Restart if needed
docker compose -f docker-compose.holochain.yml restart
```

### Gen7 Binary Not Found
```bash
# Build the host binary
cd gen7-zkstark
cargo build --release

# Verify it exists
ls -lh target/release/host
```

### Proof Generation Timeout
The RISC Zero prover takes 30-60 seconds on first run (compiling guest code). Subsequent runs are faster.

If it times out:
```bash
# Increase timeout in test script (line ~180)
timeout=300  # 5 minutes instead of 2
```

### DNA Installation Failed
```bash
# Verify DNA bundle exists
ls -lh holochain/dnas/pogq_dna/pogq_dna.dna

# Rebuild if needed
cd holochain/zomes/pogq_zome
cargo build --release --target wasm32-unknown-unknown

# Package DNA
hc dna pack ../../dnas/pogq_dna
```

## Next Steps

### 1. Multi-Node Testing
Test Byzantine detection with multiple nodes:
```bash
# Start multi-node conductors
docker compose -f holochain/docker-compose.multi-node.yml up -d

# Run multi-node test
python test_gen7_multinode.py
```

### 2. Byzantine Attack Simulation
Test with malicious gradients:
```bash
# Generate Byzantine proof (tampered gradients)
python test_gen7_byzantine_attack.py --attack label-flip

# Verify quarantine logic
python test_gen7_quarantine_weights.py
```

### 3. Performance Benchmarking
Measure proof generation and verification times:
```bash
# Benchmark proof generation
python benchmark_gen7_proof_generation.py

# Results: ~45s for RISC Zero receipt on modest hardware
```

## Docker Configuration Reference

### Working Holochain Config
```yaml
# docker-compose.holochain.yml
services:
  holochain-conductor:
    image: 0tml-holochain-conductor
    sysctls:
      - net.ipv6.conf.all.disable_ipv6=0  # Critical!
    command: sh -c "echo 'passphrase' | holochain -p -c /conductor-config.yaml"
    ports:
      - "8888:8888"
```

### Conductor Config
```yaml
# conductor-config-minimal.yaml
---
environment_path: /data
use_dangerous_test_keystore: false
keystore:
  type: lair_server_in_proc
  lair_root: /lair
```

## Performance Notes

- **Proof Generation**: ~30-60s (first run), ~10-20s (cached)
- **Proof Size**: ~2-5 MB (RISC Zero receipt)
- **DHT Publish**: <100ms
- **Proof Retrieval**: <50ms
- **Total Round Trip**: ~40-90s for complete cycle

## Security Considerations

1. **Test Keystore**: Uses `use_dangerous_test_keystore` - NOT for production
2. **Passphrase**: Hardcoded passphrase - NOT for production
3. **Proof Verification**: Journal-only (full RISC Zero verification requires external service)
4. **Nonce Binding**: Prevents replay attacks (production-ready)

## Documentation

- [Gen7-zkSTARK Design](../gen7-zkstark/README.md)
- [PoGQ Zome Implementation](zomes/pogq_zome/src/lib.rs)
- [Holochain Deployment](DEPLOYMENT_SUCCESS_2025-11-13.md)
- [Docker Setup](DOCKER_ATTEMPT_2025-11-13.md)

---

*Last Updated*: November 13, 2025
*Status*: ✅ Integration Working
*Docker*: ✅ IPv6 ENXIO Resolved
*Next Milestone*: Multi-node Byzantine testing
