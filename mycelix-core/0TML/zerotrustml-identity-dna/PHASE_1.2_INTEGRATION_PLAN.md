# Phase 1.2: Python-Holochain Integration Plan

**Prerequisites**: ✅ HDK 0.6.0-rc.0 migration complete  
**Status**: Ready to begin  
**Estimated Duration**: 2-3 days

## Overview
Integrate the Python coordinator with the upgraded Holochain conductor, replacing mock DHT operations with real distributed storage.

## Tasks

### 1. Conductor Configuration ⏭️
- [ ] Create conductor configuration file
- [ ] Set up DNA installation config
- [ ] Configure networking (bootstrap servers)
- [ ] Set up lair keystore for agent keys

### 2. Python Client Integration ⏭️
- [ ] Install Holochain Python client library
- [ ] Create conductor connection wrapper
- [ ] Implement zome call interface
- [ ] Add error handling and retry logic

### 3. DID Operations Implementation ⏭️
Replace mock operations in Python coordinator:
- [ ] `create_did()` → Call `did_registry::create_did_document`
- [ ] `resolve_did()` → Call `did_registry::resolve_did`
- [ ] `update_did()` → Call `did_registry::update_did_document`
- [ ] `deactivate_did()` → Call `did_registry::deactivate_did`

### 4. Reputation Operations ⏭️
- [ ] `add_reputation()` → Call `reputation_sync::add_reputation`
- [ ] `get_reputation()` → Call `reputation_sync::get_reputation_by_did`
- [ ] `aggregate_reputation()` → Call `reputation_sync::aggregate_reputation`

### 5. Guardian Graph Operations ⏭️
- [ ] `add_guardian()` → Call `guardian_graph::add_guardian_relationship`
- [ ] `remove_guardian()` → Call `guardian_graph::remove_guardian_relationship`
- [ ] `get_guardians()` → Call `guardian_graph::get_guardians`

### 6. Integration Testing ⏭️
- [ ] Test DID lifecycle (create → update → resolve → deactivate)
- [ ] Test multi-node reputation sync
- [ ] Test guardian relationship queries
- [ ] Performance benchmarks (latency, throughput)

### 7. Documentation ⏭️
- [ ] Python-Holochain integration guide
- [ ] API mapping documentation
- [ ] Troubleshooting guide
- [ ] Performance tuning guide

## Technical Details

### Conductor Setup
```bash
# Install Holochain (already in nix-shell)
hc --version

# Create conductor config
cat > conductor-config.yaml << 'YAML'
---
environment_path: conductors
use_dangerous_test_keystore: true
dpki: null
keystore_path: null
passphrase_service: null
network:
  network_type: quic_bootstrap
  transport_pool:
    - type: quic
  bootstrap_service: https://bootstrap.holo.host
  tuning_params:
    gossip_loop_iteration_delay_ms: 100
YAML
```

### Python Client Example
```python
from holochain_client import HolochainClient, ZomeCallUnsignedInput

class HolochainDHT:
    def __init__(self, app_port=8888):
        self.client = HolochainClient(f"http://localhost:{app_port}")
    
    async def create_did(self, did_document):
        """Create DID document in Holochain"""
        result = await self.client.call_zome(
            cell_id=self.cell_id,
            zome_name="did_registry",
            fn_name="create_did_document",
            payload=did_document
        )
        return result
```

## Success Criteria
- ✅ All DID operations working against real Holochain DHT
- ✅ Multi-node federation (3+ nodes) successfully syncing
- ✅ <500ms latency for local operations
- ✅ <2s latency for network-wide queries
- ✅ 100% test coverage for integration layer

## Dependencies
- Holochain conductor (v0.6.0+)
- Python holochain-client library
- PostgreSQL (for Byzantine detection)

## Risks & Mitigations
| Risk | Mitigation |
|------|-----------|
| Network connectivity issues | Implement robust retry logic |
| Conductor crashes | Add health checks + auto-restart |
| Key management complexity | Use test keystore initially |
| Performance bottlenecks | Benchmark early, optimize incrementally |

## Next Steps After Completion
1. Phase 1.3: Byzantine attack testing
2. Phase 1.4: Multi-region deployment
3. Phase 2.0: Production hardening
