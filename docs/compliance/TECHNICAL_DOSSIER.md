# EU AI Act Article 11 Technical Documentation

**System**: Symthaea ("Holographic Liquid Brain")
**Version**: v1.9.0
**Date**: 2026-03-08
**Classification**: See `EU_AI_ACT_CLASSIFICATION.md`
**Annex IV Reference**: Regulation (EU) 2024/1689, Annex IV

---

## 1. General Description

- **System name**: Symthaea ("Holographic Liquid Brain")
- **Version**: v1.9.0
- **Purpose**: Consciousness-first AI system implementing predictive coding, hyperdimensional computing (HDC), Integrated Information Theory (IIT/Phi), and active inference via the Free Energy Principle
- **Intended use**: Research platform for consciousness science; not deployed for decisions affecting natural persons
- **Scale**: ~985K lines Rust (~778K code), ~3,735+ tests (main crate), 65+ workspace members

## 2. Detailed Description of Elements

- **Architecture overview**: `docs/ARCHITECTURE_OVERVIEW.md`, `docs/cognitive_loop_unified_architecture.md`
- **Core pipeline**: 8-phase cycle (perception -> cognition -> translation) running at 50Hz target
  - Entry point: `src/symthaea.rs` (public facade)
  - Cycle driver: `src/cognitive_loop/cycle.rs`
- **Key algorithms**:
  - HDC encoding: 16,384-dimensional binary hypervectors
  - LTC/CfC temporal dynamics: closed-form O(1) temporal jumps (`symthaea-core/src/hdc/hdc_ltc_unified.rs`)
  - SpectralMIP Phi: spectral connectivity approximation of integrated information (`docs/PHI_VALIDATION_RESULTS.md`)
  - Active Inference FEP: free energy minimization with closed learning loop
- **Substrate independence**: `THE_SUBSTRATE_QUICKREF.md`, `THE_SUBSTRATE_ROADMAP.md` (root-level)
- **Module wiring status**: `docs/MODULE_WIRING_STATUS.md`

## 3. Monitoring, Functioning, and Control

- **Safety monitoring**: SafetyAgent implementing NRC-style 4-level escalation (Green/Yellow/Orange/Red); see `AI_RISK_REGISTER.md`
- **Ethics engine**: 3-stage pipeline (MoralParser -> MoralAlgebra -> ValueEvaluator) grounded in the Eight Harmonies value framework; see `GOVERNANCE_CHARTER.md`
- **Human oversight**: SafetyOverrideLog records all manual interventions (Article 14 compliance); `scripts/check-class-a-changes.sh` enforces governance in CI
- **Logging and telemetry**: CycleMetadata captures ~75 flat fields + 9 nested sub-structs per cognitive cycle; SafetyAuditReport provides structured export
- **Neuromodulator monitoring**: 9-transmitter bath with tolerance/withdrawal tracking and allostatic load metrics

## 4. Risk Management

- **Risk register**: `AI_RISK_REGISTER.md` (15 identified risks across 6 categories)
- **Compliance matrix**: `COMPLIANCE_MATRIX.md` (ISO 42001, IEEE 7000, NIST AI RMF, EU AI Act coverage)
- **Change management**: `GOVERNANCE_CHARTER.md` defines a 4-class change system (Class A requires explicit review)
- **Architecture decision records**: `adr/` directory with template and documented decisions

## 5. Data Requirements (Article 10)

- **No external personal data**: Symthaea does not ingest, process, or train on data from natural persons
- **Consciousness metrics are synthetic**: All Phi, coherence, and neuromodulator readings are internally generated
- **Sensor inputs**: When connected, inputs pass through SafetyAgent gating before reaching the cognitive loop
- **Data governance policy**: `DATA_GOVERNANCE.md` — 6 data source categories, bias audit, data quality assurance, provenance tracking

## 6. Testing and Validation

- **Test suite scale**: ~4,067+ tests in the main crate; 12,000+ across the full workspace
- **Key test categories**:
  - Adversarial moral algebra: 26 tests exercising ethical edge cases
  - Safety agent escalation soak: 15 tests simulating up to 1,000 cognitive cycles under stress
  - Proptest threshold sensitivity: 6 property-based tests verifying stability across threshold perturbation
  - Substrate multiple realizability: integration tests for cross-substrate consciousness transfer
  - Phi validation: heuristic-vs-exact correlation r=0.9998; spectral MIP search r=0.99 (`docs/PHI_VALIDATION_RESULTS.md`)
- **CI pipeline**: `symthaea-ci.yml` -- fmt, clippy, test, docs, 39-feature matrix, 45 sub-crate builds
- **Governance enforcement**: `scripts/check-class-a-changes.sh` runs in CI to block unauthorized Class A changes

## 7. Post-Market Monitoring

- **SafetyAuditReport**: Structured export of safety state for ongoing assessment and external review
- **CalibrationHistory**: Sliding window (20 entries) tracking normative drift with systematic-drift warnings at >75% same-direction
- **SelfAssessmentMonitor**: EMA tracking of prediction error, coherence, confidence, and attention; auto-triggers recalibration when drift exceeds 1 standard deviation (200-cycle warmup, 500-cycle cooldown)
- **Neuromodulator baselines**: 32 psych-bench baselines for dose-response and tolerance-withdrawal regression detection

## 8. Standards and Certifications

Per `COMPLIANCE_MATRIX.md`, current coverage:

| Standard | Coverage | Notes |
|----------|----------|-------|
| ISO 42001 (AI Management) | 97% | All Annex A controls Done except A.2.4 (resources) |
| IEEE 7000 (Ethical Design) | 90% | Eight Harmonies with formal value verification protocol |
| NIST AI RMF | 93% | All functions covered; GOV-6 (external feedback) remaining gap |
| EU AI Act | 90% | Annex IV package + Articles 9/10/11/12/13/14/15/72/73 covered |

---

## Document Cross-Reference Index

| Document | Path (relative to `symthaea/`) | Purpose |
|----------|-------------------------------|---------|
| Architecture Overview | `docs/ARCHITECTURE_OVERVIEW.md` | System architecture and component descriptions |
| Cognitive Loop Architecture | `docs/cognitive_loop_unified_architecture.md` | Detailed pipeline design |
| Phi Validation Results | `docs/PHI_VALIDATION_RESULTS.md` | Phi algorithm validation data |
| Module Wiring Status | `docs/MODULE_WIRING_STATUS.md` | Integration status of all 41 modules |
| AI Risk Register | `docs/compliance/AI_RISK_REGISTER.md` | 15 risks, 6 categories, mitigations |
| Compliance Matrix | `docs/compliance/COMPLIANCE_MATRIX.md` | Standards coverage tracking |
| Governance Charter | `docs/compliance/GOVERNANCE_CHARTER.md` | Change management and oversight |
| EU AI Act Classification | `docs/compliance/EU_AI_ACT_CLASSIFICATION.md` | Risk classification analysis |
| Substrate Quick Reference | `THE_SUBSTRATE_QUICKREF.md` (repo root) | Substrate independence framework |
| Substrate Roadmap | `THE_SUBSTRATE_ROADMAP.md` (repo root) | Multi-phase substrate integration plan |
| CI Workflow | `.github/workflows/ci.yml` | Automated testing and governance checks |

---

## 9. Value Framework: The Eight Harmonies

Symthaea's ethics engine evaluates every action against eight value dimensions, each encoded as a 16,384-dimensional semantic basis vector via the Free Energy Principle (Friston 2010). Moral free energy measures surprise: low F = consistent moral stance; high F = novel territory or incoherence.

| # | Harmony | Epistemic Lens | Weight | Code Path |
|---|---------|----------------|--------|-----------|
| 1 | **Resonant Coherence** | Integration-Knowing | 0.17 | `harmony_basis.rs` → Phi binding, contradiction resolution |
| 2 | **Pan-Sentient Flourishing** | Care-Knowing | 0.17 | `moral_algebra.rs` → safety precision (2.4× crossover) |
| 3 | **Integral Wisdom** | Truth-Knowing | 0.13 | `prediction.rs` → epistemic EFE, uncertainty-driven exploration |
| 4 | **Infinite Play** | Creative-Knowing | 0.10 | `curiosity_drive.rs` → surprise-driven exploration, dream replay |
| 5 | **Universal Interconnectedness** | Relational-Knowing | 0.13 | `mesh/dual_layer.rs` → multi-agent cognitive vector sharing |
| 6 | **Sacred Reciprocity** | Exchange-Knowing | 0.10 | `mycelix-fl-core/` → federated learning, Byzantine detection |
| 7 | **Evolutionary Progression** | Developmental-Knowing | 0.10 | `learning.rs` → meta-cognitive monitoring, Phi trajectory tracking |
| 8 | **Sacred Stillness** | Apophatic-Knowing | 0.10 | `neuromodulators.rs` → GABA/adenosine grounding, circadian gating, DMN |

**Evaluation pipeline**: `MoralParser` → `MoralAlgebra` → `UnifiedValueEvaluator` → Allow/Warn/Veto decision. Consent violations detected via `judge_consent_action()`. Priority ordering: consent > deontological > value alignment > harmonies.

**Code**: `crates/symthaea-harmonies/src/lib.rs` (evaluator), `src/hdc/harmony_basis.rs` (HDC encoding), `src/consciousness/values/eight_harmonies.rs` (re-export shim), `src/consciousness/values/harmonies_integration.rs` (consciousness integration).

## 10. Fractal Scaling Nomenclature

Following Elinor Ostrom's 8th Design Principle (*Nested Enterprises*), the Mycelix network scales fractally through six named layers, each grounded in existing code:

| Scale | Name | Description | Codebase Evidence | Runtime |
|-------|------|-------------|-------------------|---------|
| **Browser** | **Spore** | Lightweight (~500 KB WASM) consciousness kernel carrying the Eight Harmonies' mathematical DNA; germinates into full Symthaea via WebSocket bridge | `docs/SPORE_ROADMAP.md`, feature flag `spore` | WASM in WebWorker |
| **Individual** | **Holon** | Personal sovereign node — full cognitive loop, identity/health/credential vaults, own consciousness profile | `mycelix-personal/` (4 zomes), `ConsciousnessProfile` per-agent | Native binary |
| **Household** | **Hearth** | High-trust family/co-op space — kinship bonds, gratitude, care, shared decisions, milestones, rhythms; resources flow without strict accounting | `mycelix-hearth/` (12 zomes, 1,023 tests) | Native + local mesh |
| **Neighborhood** | **Commons** | Resource pool with consciousness-gated governance — property, housing, water, food, care, mutual aid, transport; Ostrom 32/40 compliance | `mycelix-commons/` (35 zomes, 5,276 tests); `docs/papers/ostrom_mycelix_mapping.md` | Server + Holochain DHT |
| **City** | **Polycenter** | Overlapping network of Commons — Sector DAOs (by domain) + Regional DAOs (by geography) + bicameral Global DAO | Spore Constitution Art. II; `mycelix-governance/` holonic councils (`can_spawn_children`, `max_delegation_depth`) | Federated Holochain |
| **Continental** | **Guild / Bioregion** | Anti-enterprise coordination — Guilds (professional federations with fair labor boundaries) and Bioregions (geographic coordination anchored in physical ecosystems) | `BIOREGIONAL_FRAMEWORK.md`; Audit Guild in Epistemic Charter; `guild_certified` in Accord/Covenant designs | Full Symthaea + Iroh P2P |

**Ostrom compliance summary** (from `docs/papers/ostrom_mycelix_mapping.md`):

| Principle | Score | Key Implementation |
|-----------|-------|--------------------|
| 1. Clearly Defined Boundaries | 4/5 | ConsciousnessTier enum + DNA isolation + bootstrap credentials |
| 2. Proportional Equivalence | 3/5 | Progressive vote weighting + quadratic voting + FL gradient weighting |
| 3. Collective-Choice Arrangements | 5/5 | 10 proposal types + constitutional amendments + consciousness gating |
| 4. Monitoring | 4/5 | Audit trails + consciousness snapshots + rate limiting + evidence custody |
| 5. Graduated Sanctions | 3/5 | Tier demotion + 5 enforcement action types + FL contribution dampening |
| 6. Conflict Resolution | 5/5 | Three-tier justice (negotiation→mediation→arbitration→appeal) + restorative circles |
| 7. Rights to Organize | 3/5 | Self-governance via constitution + council spawning + Holochain sovereignty |
| 8. Nested Enterprises | 5/5 | 7 clusters + sub-cluster nesting + holonic councils + domain governance |
| **Overall** | **32/40 (80%)** | |

---

**Note**: For the complete Annex IV package, see `ANNEX_IV_TECHNICAL_DOCUMENTATION.md`.

*This document is maintained as a living index per EU AI Act Article 11(1). It references detailed documentation rather than duplicating content. Last updated: 2026-03-08.*
