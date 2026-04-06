# Mycelix Praxis - Claude Context

**Version**: v0.2.0-dev (Holochain 0.6)
**Status**: Phase 10 (Live at praxis.mycelix.net), 210+ tests passing
**Features**: 29 learning science frameworks, Spore consciousness integration, Craft credential pipeline
**Live**: https://praxis.mycelix.net (Cloudflare Tunnel → :8107)
**Credential Pipeline**: Praxis issues (PoL + BKT mastery) → Craft publishes (living credentials + guild context)

---

## Quick Commands

```bash
nix develop                                    # Enter environment
cargo build --workspace                        # Build all
cargo test --all                               # Run tests
cargo build --target wasm32-unknown-unknown --release  # Build WASM

# Package and run
hc dna pack dna/ -o dna/praxis.dna
hc app pack happ/ -o happ/mycelix-praxis.happ
echo "" | hc sandbox --piped generate -a 9999 happ/mycelix-praxis.happ --run=8888
```

---

## Project Structure

```
mycelix-praxis/
├── crates/
│   └── praxis-core/       # Core types, crypto, provenance
├── zomes/
│   ├── learning_zome/     # Courses, progress (31 tests)
│   ├── fl_zome/           # Federated learning (27 tests)
│   ├── credential_zome/   # W3C Verifiable Credentials (31 tests)
│   ├── dao_zome/          # Governance (27 tests)
│   ├── srs_zome/          # Spaced Repetition SM-2 (10 tests)
│   ├── gamification_zome/ # XP, badges, streaks (5 tests)
│   ├── adaptive_zome/     # BKT, ZPD, VARK (75 tests)
│   └── integration_zome/  # Cross-zome orchestration (22 tests)
└── apps/web/              # React + TypeScript client
```

---

## Version Compatibility

| Component | Version |
|-----------|---------|
| holochain | 0.6.0 |
| hdk | 0.6.0 |
| hdi | 0.7.0 |
| @holochain/client | 0.20.0 |

**Critical**: Use `getrandom_backend="custom"` in `.cargo/config.toml` for WASM.

---

## Roadmap Status

| Phase | Status | Deliverable |
|-------|--------|-------------|
| 1-5 | ✅ Complete | DNA + 4 core zomes |
| 6 | ✅ Complete | Web client integration |
| 7 | ✅ Complete | 29 learning science features |
| 8 | ✅ Complete | WASM packaging |
| 9 | 🚧 In Progress | E2E testing |

---

## Learning Science Stack (29 Features)

Core: SRS (SM-2), Gamification, BKT/ZPD/VARK, Flow State, Knowledge Graph

Advanced: Bloom's Taxonomy, Transfer of Learning, Desirable Difficulties, Dual Coding, Testing Effect, Productive Failure, Growth Mindset, Critical Thinking, Socratic Dialogue, UDL, CASEL, Deliberate Practice

See `docs/dev/DIFFERENTIATION_FEATURES_COMPLETE.md` for full details.

---

## Key Concepts

### Privacy Guarantees
- Gradient clipping (L2 norm)
- Differential privacy (ε-δ)
- No raw data sharing
- Secure aggregation with commitments

### DAO Governance
- Three-tier proposals: Fast/Normal/Slow
- Quorum-based voting
- Community-driven curriculum

---

## Current Phase: Web Client (Phase 9)

**Done**:
- Zome names fixed in client
- TypeScript types match Rust structs
- Conductor connection verified
- create_course/list_courses working

**Remaining**:
- Differentiation zome wrappers
- Full user flow testing
- React UI connected to real data

---

## Documentation

| Purpose | File |
|---------|------|
| Implementation plan | `docs/dev/V0_2_IMPLEMENTATION_PLAN.md` |
| Phase 8 report | `docs/dev/PHASE_8_WASM_PACKAGING_COMPLETE.md` |
| Differentiation features | `docs/dev/DIFFERENTIATION_FEATURES_COMPLETE.md` |
| FL protocol | `docs/FL_PROTOCOL.md` |

---

## Success Criteria (v0.2.0)

- [ ] DNA packages with `hc dna pack`
- [ ] All 8 zomes compile to WASM
- [ ] Web client connects via WebSocket
- [ ] Zome call latency <500ms (p99)
- [ ] 10+ integration tests passing

---

*Privacy-preserving decentralized education* 🎓
