# Zero-TrustML Identity DNA

**Status**: Phase 1.1 Complete ✅
**DNA Hash**: `uhC0kp9yIePPFvhqZU2Q2jCv0SnMJytqgdmPuMBfnlQyilNBa0NfI`

Byzantine-resistant decentralized identity and governance system for Zero-TrustML federated learning.

## Quick Start

```bash
# Enter development environment
nix-shell

# Build all zomes
cargo build --release --target wasm32-unknown-unknown

# Pack DNA bundle
hc dna pack .

# Verify DNA
hc dna hash zerotrustml_identity.dna
```

## Zomes (5 total)

1. **did_registry** - W3C DID document management
2. **identity_store** - Identity factors, credentials, signals
3. **reputation_sync** - Multi-network reputation aggregation
4. **guardian_graph** - Guardian relationships & cartel detection
5. **governance_record** - Quadratic voting & proposal execution

## Documentation

- **[Phase 1.1 Completion Report](PHASE_1_1_HOLOCHAIN_INTEGRATION_COMPLETE.md)** - Full migration details
- **[HDK 0.4.4 Quick Reference](HDK_0_4_4_QUICK_REFERENCE.md)** - Developer guide
- **[DNA Configuration](dna.yaml)** - Bundle manifest

## Architecture

```
Identity DNA (Holochain)
├── DID Registry (W3C DIDs)
├── Identity Store (Multi-factor identity)
├── Reputation Sync (Cross-network aggregation)
├── Guardian Graph (Trust relationships)
└── Governance (Quadratic voting + PoGQ)
        ↓
Zero-TrustML Python Client
├── Coordinator (Byzantine consensus)
├── Worker Nodes (Federated learning)
└── MATL (Reputation oracle integration)
```

## Key Features

- **W3C DID Compliance**: Full decentralized identifier support
- **Multi-Network Identity**: Aggregate reputation from multiple sources
- **Byzantine Resistance**: 45% BFT tolerance via PoGQ oracle
- **Quadratic Voting**: Democratic governance for protocol decisions
- **Cartel Detection**: Graph-based guardian relationship analysis

## Development

### Build Requirements

- NixOS or Nix package manager
- Rust 1.91.1+ with wasm32-unknown-unknown target
- Holochain CLI (`hc`) 0.5.6+

### Testing

```bash
# Create test sandbox
hc sandbox create --in-process-lair -d test

# Install DNA (future step)
hc sandbox call install-app zerotrustml_identity.dna
```

## Integration Status

| Component | Status | Notes |
|-----------|--------|-------|
| Zome Compilation | ✅ Complete | All 5 zomes compile to WASM |
| DNA Bundle | ✅ Complete | 4.0MB compressed bundle |
| Python Client | ⏳ Pending | Phase 1.2 |
| Integration Tests | ⏳ Pending | Phase 1.3 |
| Performance Baseline | ⏳ Pending | Phase 1.3 |

## Technical Stack

- **HDK**: 0.4.4
- **Rust**: 1.91.1
- **Holochain**: 0.5.6
- **WASM Target**: wasm32-unknown-unknown
- **Build System**: Nix + Cargo

## License

Apache 2.0 - See [LICENSE](../../LICENSE)

## References

- **Zero-TrustML Paper**: [Link to paper]
- **Holochain Docs**: https://docs.holochain.org/
- **W3C DID Spec**: https://www.w3.org/TR/did-core/
- **HDK Documentation**: https://docs.rs/hdk/0.4.4/

---

**Last Updated**: November 11, 2025
**Maintainer**: Zero-TrustML Team
