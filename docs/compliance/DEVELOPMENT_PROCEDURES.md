# Development Procedures — ISO 42001 A.4.5

Classification: Internal | Version: 1.0 | Date: 2026-03-08
Owner: Tristan Stoltz, Luminous Dynamics

---

## Purpose

This document formalizes Symthaea's development procedures per ISO 42001 A.4.5 (AI system development processes). It complements `SDLC.md` (lifecycle phases) and `GOVERNANCE_CHARTER.md` (governance) with day-to-day operational procedures.

---

## 1. Development Environment

### 1.1 Required toolchain

All development occurs within a Nix flake environment:

```bash
nix develop    # Enter development shell (mandatory before any build)
```

**Rule**: Fix the flake, don't work around it. No ad-hoc `cargo install` or system-level package modifications.

### 1.2 Platform requirements

| Requirement | Specification |
|-------------|--------------|
| OS | NixOS 25.11 (reference), Linux (supported) |
| Rust | 1.92.0+ (managed by flake) |
| Target dir | Default `target/` (no custom CARGO_TARGET_DIR) |
| Cache | sccache handles compilation caching |
| Editor | Any (Neovim reference); no IDE-specific config committed |

### 1.3 Session discipline

- **One session per workspace**: Do not run multiple editing sessions on the same workspace simultaneously
- Concurrent sessions cause cargo lock contention, edit conflicts, and 10-50x slower builds
- Use `git worktree` for parallel work on separate branches
- Multiple sessions waiting on the same cargo lock is acceptable (incremental build)

---

## 2. Code Change Procedures

### 2.1 Change classification

Per `SDLC.md` and `GOVERNANCE_CHARTER.md`:

| Class | Scope | Examples | Review Required |
|-------|-------|---------|----------------|
| **A** (Safety-Critical) | Safety, ethics, consciousness thresholds | `safety/agent.rs`, `ethics_engine.rs`, `thresholds.rs` | ADR + full test evidence |
| **B** (Core Pipeline) | Cognitive loop, HDC, LTC, Phi | `cycle.rs`, `cycle_phase_*.rs`, `consciousness_engine.rs` | Integration test evidence |
| **C** (Supporting) | Docs, CI, tooling, tests | `docs/`, `.github/`, `tests/` | Basic review |

### 2.2 Commit governance

**Pre-commit hooks**:
1. **Secrets scan**: Blocks commits containing credentials, API keys, or tokens
2. **Governance hook**: Validates commit message format for safety-critical changes

**Commit message format**: Conventional Commits
```
type(scope): description

[optional body with rationale]

Co-Authored-By: [if AI-assisted]
```

Valid types: `feat`, `fix`, `refactor`, `test`, `docs`, `safety`, `perf`, `ci`

**Safety-critical commits** (Class A files): Must use `safety:` prefix or include safety rationale in body.

### 2.3 Branch strategy

- **main**: Primary development branch, always green CI
- **Feature branches**: For multi-commit changes; merge via PR
- **No force-push to main**: Ever
- **Git worktrees**: Preferred for parallel development over multiple sessions

---

## 3. Threshold Change Protocol

`src/cognitive_loop/thresholds.rs` contains 119+ scientifically-cited constants. Changes to this file follow a strict protocol:

### 3.1 Requirements for threshold changes

1. **Scientific citation**: Every constant must cite published literature (author, year)
2. **Rationale documentation**: Why the value was chosen, not just what it is
3. **Ordering validation**: `validate()` function checks cross-constant invariants
4. **Test evidence**: At least one test exercises the threshold's behavioral effect
5. **Property-based coverage**: Proptests verify stability under perturbation (`proptest_threshold_sensitivity.rs`)

### 3.2 Change process

1. Identify the constant and its current citation
2. Document the new value and updated citation
3. Verify `validate()` still passes
4. Run relevant proptest suite to confirm stability
5. Commit as Class A change with scientific rationale

### 3.3 Ordering invariants (enforced programmatically)

```
CONSCIOUSNESS_RED < CONSCIOUSNESS_YELLOW < CONSCIOUSNESS_GREEN
SUBSYSTEM_LR_FACTOR_MIN < 1.0 < SUBSYSTEM_LR_FACTOR_MAX
CROSS_MODULE_AGREEMENT_LOW < CROSS_MODULE_AGREEMENT_HIGH
```

---

## 4. Feature Flag Discipline

### 4.1 Flag management

Symthaea has 88 feature flags. Rules:

- **Default features = []** (empty): All features opt-in
- **No `default` feature bloat**: Only add to default if universally required
- **Feature gating**: Safety-critical code must work WITHOUT optional features
- **CI matrix**: 39 feature combinations tested in CI

### 4.2 Adding new features

1. Declare in `Cargo.toml` with descriptive name
2. Gate all new code with `#[cfg(feature = "name")]`
3. Add to CI feature matrix if non-trivial
4. Document in code comments what the feature enables
5. Ensure `cargo test --lib` (no features) still passes

### 4.3 Feature dependency chains

`config.rs::resolve_dependencies()` auto-enables upstream features:
- `reasoning_engine` → enables `identity` + `neural-bridge`
- `full_consciousness` → enables consciousness subsystems
- etc.

---

## 5. Testing Procedures

### 5.1 Test hierarchy

| Level | Command | When | Count |
|-------|---------|------|-------|
| Unit tests | `cargo test --lib` | Every change | 4,067+ |
| Integration tests | `cargo test --test <name>` | Relevant changes | 167+ |
| Property tests | `cargo test --test proptest_*` | Threshold/stability changes | 30+ |
| Soak tests | Included in integration | Safety changes | 15+ |
| Compliance dashboard | `bash scripts/compliance-dashboard.sh` | Pre-release, CI | 14 suites |
| Sub-crate tests | `cargo test -p <crate>` | Sub-crate changes | Varies |

### 5.2 Test quality rules

- **Test what exists**: No aspirational tests for unimplemented features
- **Assert behavior, not implementation**: Test outcomes, not internal state
- **Stochastic systems**: Assert finiteness and bounds, not specific values
- **EMA convergence**: Allow ~120 cycles for alpha=0.05 EMA to converge
- **Soak tests**: ≥500 cycles for stability verification

### 5.3 Test evidence for compliance

All test results are CI-reproducible:
```bash
# Full compliance evidence generation
bash scripts/compliance-dashboard.sh --json > compliance-report.json
```

CI retains compliance artifacts for 90 days.

---

## 6. CI Pipeline

### 6.1 Pipeline structure (`.github/workflows/ci.yml`)

| Job | Purpose | Failure Action |
|-----|---------|---------------|
| `fmt` | Code formatting | Block merge |
| `clippy` | Lint analysis (0 warnings) | Block merge |
| `test` | 39-feature matrix test | Block merge |
| `docs` | Documentation generation | Block merge |
| `compliance` | Dashboard (14 suites + 15 docs) | Block merge |

### 6.2 CI quality gates

- **Clippy**: Zero warnings policy (no `#[allow]` without justification)
- **Tests**: 100% pass rate required
- **Compliance**: All suites pass, all docs present
- **Artifacts**: Compliance report uploaded (90-day retention)

### 6.3 Weekly jobs

- **Psych-bench regression**: Weekly behavioral baseline check
- **Dependency audit**: Weekly `cargo audit` for security advisories

---

## 7. Incident Response Integration

When a CI failure or production incident occurs:

1. **Classify severity** per `INCIDENT_RUNBOOK.md` (SEV-1 through SEV-4)
2. **SEV-1/2**: Immediate response, Class A change procedures
3. **SEV-3/4**: Normal change procedures, tracked in commit history
4. **Post-incident**: Update risk register if new risk identified
5. **Serious incidents**: Generate `SeriousIncidentReport` (Article 73)

---

## 8. Documentation Standards

### 8.1 Code documentation

- **Public API**: Doc comments required on all `pub` items
- **Thresholds**: Scientific citation required for every constant
- **Magic numbers**: Must be named constants in `thresholds.rs`
- **Unsafe code**: Justification comment required (none currently exists)

### 8.2 Compliance documentation

- **Version and date**: Every compliance doc has version and date header
- **Cross-references**: Use relative paths to other docs
- **Living documents**: Update when system changes materially affect coverage
- **Dashboard verification**: All docs checked for existence in CI

### 8.3 Architecture Decision Records (ADRs)

For Class A changes or significant architectural decisions:
- Template: `docs/compliance/adr/ADR-TEMPLATE.md`
- Numbering: Sequential (ADR-001, ADR-002, ...)
- Status: Proposed → Accepted → Superseded

---

## 9. Credential Management

- **BWS (Bitwarden Secrets)**: Only approved credential manager
- **No hardcoded credentials**: Pre-commit hook blocks secrets
- **No `.env` files committed**: gitignored
- **Access**: `bws get secret-name` for runtime credentials

---

## 10. Review and Update

| Activity | Frequency | Owner |
|----------|-----------|-------|
| Procedure review | Quarterly | Development lead |
| Threshold audit | Quarterly | Development lead |
| CI pipeline review | Monthly | Development lead |
| Feature flag cleanup | Quarterly | Development lead |
| Documentation currency check | Monthly (via CI) | Automated |

---

*These procedures are mandatory for all Symthaea development. Deviations require documented rationale.*
