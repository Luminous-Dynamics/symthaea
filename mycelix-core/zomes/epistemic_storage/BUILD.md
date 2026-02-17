# Epistemic Storage - Build Instructions

## Current Status

- **WASM Built**: ✅ Coordinator (3.0 MB) + Integrity (1.9 MB)
- **TypeScript SDK**: ✅ DHTBackend integrated with EpistemicStorageImpl
- **Tests**: ✅ 309 passing (9 DHT integration tests)
- **DNA/hApp**: ⏳ Requires `hc` tool from nix environment

## Quick Start

```bash
# 1. Enter Holochain environment
cd /srv/luminous-dynamics/mycelix-workspace
nix develop .#holochain

# 2. Navigate to zome directory
cd happs/core/zomes/epistemic_storage

# 3. Run packaging script (builds DNA + hApp)
./package-happ.sh
```

## Prerequisites

Enter the Holochain development environment (first build takes ~20 minutes):
```bash
cd /srv/luminous-dynamics/mycelix-workspace
nix develop .#holochain
```

## Build Steps

### 1. Build WASM zomes (DONE)
```bash
cd happs/core/zomes/epistemic_storage
cargo build --release --target wasm32-unknown-unknown
```

WASM files location:
- `target/wasm32-unknown-unknown/release/epistemic_storage_integrity.wasm` (1.9 MB)
- `target/wasm32-unknown-unknown/release/epistemic_storage_coordinator.wasm` (3.0 MB)

### 2. Package DNA + hApp (requires nix environment)
```bash
# Use the convenience script:
./package-happ.sh

# Or manually:
hc dna pack . -o workdir/epistemic_storage.dna
hc app pack . -o workdir/epistemic_storage.happ
```

### 3. Run with conductor (for testing)
```bash
hc sandbox generate workdir/epistemic_storage.happ --run 8888
```

### 4. Run SDK integration tests
```bash
cd /srv/luminous-dynamics/mycelix-workspace/sdk-ts

# Unit tests (no conductor needed)
npm test tests/storage/epistemic-dht-integration.test.ts

# Full integration tests (with conductor)
CONDUCTOR_AVAILABLE=true npm test tests/storage/dht.test.ts
```

## SDK Integration

```typescript
import { AppWebsocket } from '@holochain/client';
import { createEpistemicStorage, MaterialityLevel } from '@mycelix/sdk/storage';

// Connect to conductor
const client = await AppWebsocket.connect('ws://localhost:8888');

// Create storage with DHT backend
const storage = createEpistemicStorage({
  agentId: 'agent:alice',
  dhtBackend: {
    client,
    zomeName: 'epistemic_storage',
    dnaRole: 'epistemic_storage',
  },
});

// M2/M3 data routes to DHT automatically
await storage.store('profile:alice', { name: 'Alice' }, {
  empirical: EmpiricalLevel.E3_Cryptographic,
  normative: NormativeLevel.N2_Network,
  materiality: MaterialityLevel.M2_Persistent,  // → DHT
}, { schema: { id: 'profile', version: '1.0.0' } });
```

See: `sdk-ts/examples/dht-storage-example.ts` for more examples.

## Manifest Files

- `dna.yaml` - DNA configuration with E/N/M properties
- `happ.yaml` - hApp bundle configuration

## Zome API

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `store_epistemic_entry` | `StoreInput` | `ActionHash` | Store entry with E/N/M classification |
| `get_epistemic_entry` | `GetInput` | `Option<EpistemicEntry>` | Retrieve by key |
| `has_epistemic_entry` | `GetInput` | `bool` | Check existence |
| `delete_epistemic_entry` | `GetInput` | `bool` | Soft delete (tombstone) |
| `list_epistemic_keys` | `ListKeysInput` | `Vec<String>` | List keys with pattern matching |
| `get_storage_stats` | `()` | `StorageStats` | Get storage statistics |
| `clear_epistemic_storage` | `()` | `u64` | Clear all entries |
| `get_by_cid` | `GetByCidInput` | `Option<EpistemicEntry>` | Retrieve by content ID |
| `get_replication_status` | `GetInput` | `ReplicationStatus` | Check DHT replication |
| `ensure_replication` | `ReplicationInput` | `bool` | Request replication |

## Validation Rules

- Keys: max 256 chars, non-empty
- CIDs: must start with "cid:"
- E-axis: 0-4 (E3+ entries are immutable)
- N-axis: 0-3
- M-axis: 0-3
- Schema ID and version: required
