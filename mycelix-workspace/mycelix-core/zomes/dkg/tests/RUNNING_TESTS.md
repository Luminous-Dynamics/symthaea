# Running DKG Synapse Simulation Tests

## Current Status

The test infrastructure is complete, but there are version compatibility issues between the Holochain tools.

### What Works
- ✅ WASM compilation (hdi 0.7, hdk 0.6)
- ✅ DNA packaging with `hc dna pack`
- ✅ hApp packaging with `hc app pack`
- ✅ Tryorama test framework setup
- ✅ Vitest configuration
- ✅ Test code written for 4 simulation scenarios

### Current Issue
The `@holochain/client` v0.20.0 and `@holochain/tryorama` v0.19.0 are having deserialization issues when communicating with the Holochain conductor. This appears to be a protocol version mismatch.

**Error Message:**
```
deserialization: Failed to deserialize request
at AdminWebsocket.installApp
```

## Version Matrix

| Component | Version Used | Expected |
|-----------|--------------|----------|
| hdi | 0.7 | Compatible with Holochain 0.6.x |
| hdk | 0.6 | Compatible with Holochain 0.6.x |
| holochain | 0.6.1-rc.0 | Should be stable 0.6.x |
| @holochain/client | 0.20.0 | Designed for 0.6.x |
| @holochain/tryorama | 0.19.0 | Designed for 0.6.x |

## Recommended Solution

Use the official Holochain scaffolding to set up a compatible environment:

```bash
# Option 1: Use hc-scaffold to create a properly configured project
nix shell github:holochain/holonix/main-0.6#hc-scaffold --accept-flake-config
hc scaffold web-app test_project --setup-nix yes

# Then copy our zomes and tests into the scaffold
```

Or wait for the official Holochain 0.6.x stable release which should have better compatibility.

## Running Tests (When Compatibility is Fixed)

```bash
cd zomes/dkg

# 1. Build the WASM
cargo build --release --target wasm32-unknown-unknown

# 2. Enter holonix environment
nix shell github:holochain/holonix/main-0.6#{holochain,hc,lair-keystore,bootstrap-srv} --accept-flake-config

# 3. Pack the hApp
cd workdir
hc dna pack . -o dkg.dna
hc app pack . -o dkg.happ

# 4. Run tests
cd ../tests
npm install
npm test
```

## Test Scenarios

The test suite includes 4 scenarios:

1. **The Survival of Truth** - Byzantine fault tolerance simulation
   - 10 agents: 1 Historian, 6 Crowd, 3 Liars
   - Truth should rise (>0.9 confidence)
   - Lies should decay (<0.4 confidence)

2. **Epistemic Classification** - E/N/M constraint enforcement
   - Validates epistemic axis constraints
   - E4 requires ≥0.5 confidence
   - N3 requires E3/E4 evidence

3. **Asymptotic Reinforcement** - MAX_CONFIDENCE limit test
   - Confidence approaches but never exceeds 0.9999
   - Tests reinforcement dynamics

4. **Self-Attestation Prevention** - Sybil protection
   - Agents cannot attest to their own claims

## Files

- `package.json` - Test dependencies
- `vitest.config.ts` - Test configuration
- `src/epistemic_simulation.test.ts` - Main test file
- `../workdir/dna.yaml` - DNA manifest (manifest_version: "0")
- `../workdir/happ.yaml` - hApp manifest

## Next Steps

1. Monitor Holochain releases for stable 0.6.x
2. Update tryorama/client versions when compatible release is available
3. Consider using Holochain's official test templates
4. Run tests once version compatibility is resolved
