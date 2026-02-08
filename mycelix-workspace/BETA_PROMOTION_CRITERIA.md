# hApp Beta Promotion Criteria

**Last Updated**: 2026-02-08

This document defines the criteria for promoting a Mycelix hApp from **scaffold** to **beta** status.

---

## Status Definitions

| Status | Definition |
|--------|------------|
| **Stub** | Minimal files, no functional code |
| **Scaffold** | Types/structure exist, compiles to WASM, core logic incomplete |
| **Beta** | Core functionality works, tests pass, ready for limited use |
| **Production** | Tests pass, benchmarks validated, actively used |

---

## Beta Promotion Checklist

### 1. Core Requirements (Must Have)

- [ ] **Entry Types**: All entry types in `schema.rs` have create/read/update/delete handlers
- [ ] **Link Types**: All link types have corresponding functions
- [ ] **No Stubs**: No `unimplemented!()`, `todo!()`, or `panic!()` in main code paths
- [ ] **Error Handling**: All error paths return proper `ExternResult<T>` errors
- [ ] **DNA Manifest**: Valid `dna.yaml` with manifest_version "0"
- [ ] **hApp Manifest**: Valid `happ.yaml` with manifest_version "0"
- [ ] **Bundle Builds**: `hc dna pack` and `hc app pack` succeed

### 2. Test Requirements (Must Have)

- [ ] **Zome Unit Tests**: Minimum 3 tests per entry type
- [ ] **Test Coverage**: 60%+ line coverage (use `cargo tarpaulin`)
- [ ] **Integration Tests**: 5+ tests that exercise zome-to-zome calls
- [ ] **Error Path Tests**: All error conditions have test coverage
- [ ] **Sweettest**: At least 3 sweettest scenarios pass

### 3. SDK Integration (Must Have)

- [ ] **TypeScript Client**: Client module in `sdk-ts/src/clients/`
- [ ] **Client Tests**: 10+ integration tests for SDK client
- [ ] **Type Exports**: All entry types exported from SDK
- [ ] **Documentation**: JSDoc comments on all public functions

### 4. Cross-hApp Integration (Must Have)

- [ ] **Bridge Zome**: If depends on other hApps, bridge zome implemented
- [ ] **Integration Test**: At least 1 cross-hApp workflow tested
- [ ] **Dependency Graph**: Dependencies documented in README

### 5. Documentation (Must Have)

- [ ] **README.md**: Overview, installation, usage examples
- [ ] **Zome Overview**: Each zome has a doc comment explaining purpose
- [ ] **API Reference**: Entry types and functions documented
- [ ] **Migration Guide**: If upgrading from previous version

### 6. Quality Gates (Must Pass)

- [ ] **Clippy**: `cargo clippy -- -D warnings` passes with 0 warnings
- [ ] **Formatting**: `cargo fmt --check` passes
- [ ] **CI Green**: All CI jobs pass for this hApp
- [ ] **Security Audit**: `cargo audit` shows no critical vulnerabilities

---

## Current hApp Status

| hApp | Core | Tests | SDK | Cross-hApp | Docs | Quality | Status |
|------|------|-------|-----|------------|------|---------|--------|
| **identity** | 70% | 40% | 60% | 20% | 50% | 80% | 4-5 weeks to beta |
| **governance** | 75% | 50% | 70% | 30% | 60% | 80% | 3-4 weeks to beta |
| **justice** | 40% | 20% | 30% | 0% | 30% | 70% | 6-8 weeks to beta |
| **knowledge** | 50% | 30% | 40% | 10% | 40% | 70% | 6-8 weeks to beta |
| **finance** | 60% | 40% | 50% | 20% | 40% | 80% | 5-6 weeks to beta |
| **property** | 65% | 45% | 60% | 20% | 50% | 80% | 4-5 weeks to beta |
| **energy** | 55% | 35% | 40% | 10% | 40% | 75% | 5-6 weeks to beta |
| **media** | 45% | 25% | 35% | 0% | 35% | 70% | 6-8 weeks to beta |
| **health** | 50% | 30% | 40% | 10% | 45% | 75% | 6-8 weeks to beta |
| **space** | 35% | 15% | 20% | 0% | 25% | 70% | 8-10 weeks to beta |
| **care** | 30% | 10% | 15% | 0% | 20% | 70% | 8-10 weeks to beta |
| **emergency** | 30% | 10% | 15% | 0% | 20% | 70% | 8-10 weeks to beta |
| **water** | 25% | 10% | 15% | 0% | 20% | 70% | 8-10 weeks to beta |
| **housing** | 30% | 15% | 20% | 0% | 25% | 70% | 8-10 weeks to beta |

---

## Priority Promotion Path

### Phase 1: Foundation (Weeks 1-5)
Focus on **Identity** and **Governance** - these are foundational for all other hApps.

**Identity** (8 zomes):
| Zome | Current | Target | Gap |
|------|---------|--------|-----|
| did_registry | 90% | 100% | Add revocation tests |
| credentials | 80% | 100% | Schema validation tests |
| recovery | 60% | 100% | Threshold logic tests |
| bridge | 70% | 100% | Cross-hApp integration |
| schema | 50% | 100% | Validation handlers |
| revocation | 40% | 100% | Core revocation logic |
| trust | 55% | 100% | Trust scoring algorithm |
| education | 30% | 100% | Credential verification |

**Governance** (7 zomes):
| Zome | Current | Target | Gap |
|------|---------|--------|-----|
| proposals | 85% | 100% | Amendment handlers |
| voting | 80% | 100% | Weighted voting tests |
| councils | 70% | 100% | Election logic |
| delegation | 60% | 100% | Revocation chain |
| constitutional | 50% | 100% | Amendment process |
| bridge | 65% | 100% | Identity integration |
| analytics | 40% | 100% | Reporting functions |

### Phase 2: Economics (Weeks 6-10)
**Finance** and **Property** enable economic activity.

### Phase 3: Domain-Specific (Weeks 11+)
Energy, Health, Knowledge, etc.

---

## Promotion Process

1. **Self-Assessment**: hApp owner fills out checklist
2. **Peer Review**: Another team member reviews
3. **CI Validation**: All automated checks pass
4. **Integration Test**: Cross-hApp test with Identity + Governance
5. **Documentation Review**: README and API docs complete
6. **Merge**: Update status in ECOSYSTEM_STATUS.md

---

## Metrics Tracking

Track these metrics weekly:

```bash
# Test coverage
cd mycelix-workspace/happs/[happ]
cargo tarpaulin --out Html --output-dir coverage/

# Lines of code vs tests
tokei --type Rust zomes/
tokei --type Rust tests/

# Clippy warnings
cargo clippy 2>&1 | grep -c "warning:"
```

---

## Exceptions

Some hApps may have reduced requirements:

- **Bridge hApps**: May skip SDK integration if only used internally
- **Utility hApps**: May have fewer entry types
- **Experimental hApps**: Marked as such, reduced test requirements

Document any exceptions in the hApp's README.

---

*This document is the single source of truth for beta promotion criteria.*
