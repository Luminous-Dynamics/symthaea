# EU AI Act Annex IV — Technical Documentation Package

Classification: Internal | Version: 1.0 | Date: 2026-03-08
Provider: Luminous Dynamics | Contact: tristan.stoltz@evolvingresonantcocreationism.com

---

## Purpose

This document assembles Symthaea's technical documentation per EU AI Act Annex IV requirements. It serves as the **master index** for conformity assessment, cross-referencing all supporting documents, source code, and test evidence.

**Annex IV Compliance**: Each numbered section below corresponds to the required elements specified in Regulation (EU) 2024/1689, Annex IV.

---

## 1. General Description of the AI System

### 1(a) Intended purpose

Symthaea is a research platform for consciousness-first artificial intelligence. It implements a holographic liquid brain architecture combining Hyperdimensional Computing (HDC), Liquid Time-Constant networks (LTC), Integrated Information Theory (IIT/Phi), and Free Energy Principle (FEP) active inference.

**Current deployment context**: Research and development only. Not placed on the EU market or put into service for decisions affecting natural persons.

**Potential high-risk categories** (Annex III):
- Category 2: Biometrics (emotion recognition via neuromodulator bath)
- Category 5: Essential services (via Mycelix civic governance integration)
- Category 8: Democratic processes (via Mycelix governance voting)

**Reference**: `EU_AI_ACT_CLASSIFICATION.md`

### 1(b) Provider identification

| Field | Value |
|-------|-------|
| Provider | Luminous Dynamics |
| Contact | tristan.stoltz@evolvingresonantcocreationism.com |
| Location | Richardson, TX, USA |
| System name | Symthaea |
| Version | v1.9.0 |
| Language | Rust (985K lines, 778K code) |
| License | Proprietary research |

### 1(c) Version history

| Version | Date | Key Changes |
|---------|------|-------------|
| v0.1.0 | 2025-Q3 | Initial HDC+LTC cognitive loop |
| v0.3.0 | 2025-Q4 | IIT Phi integration, consciousness engine |
| v0.5.0 | 2026-01 | Ethics engine, safety agent, moral algebra |
| v1.0.0 | 2026-02 | Great Refactor — manager architecture, threshold registry |
| v1.5.0 | 2026-02 | Neuromodulator bath (9 transmitters), calibration bridge |
| v1.9.0 | 2026-03 | Eight Harmonies, compliance hardening, always-awake fallback |

**Reference**: Git history at repository root; `TECHNICAL_STATUS.md`

### 1(d) Interaction with other systems

| System | Interface | Purpose |
|--------|-----------|---------|
| Mycelix hApp (Holochain) | `mycelix-bridge-common` crate | Consciousness gating for decentralized governance |
| Ollama (local LLM) | HTTP API | Broca language generation (approved models only) |
| Psych-bench battery | Binary spawn | Normative behavioral testing |
| Pulse (TUI) | In-process | Real-time consciousness visualization |

### 1(e) Hardware and software requirements

- **OS**: Linux (NixOS 25.11 reference platform)
- **CPU**: x86_64 with AVX2+FMA (SIMD kernels)
- **RAM**: 4GB minimum, 16GB recommended
- **Rust toolchain**: 1.92.0+
- **Optional**: CUDA-capable GPU (for `neural-bridge-cuda` feature)

---

## 2. Detailed Description of System Elements

### 2(a) Development methodology

**Reference**: `SDLC.md`, `GOVERNANCE_CHARTER.md`

- **Change classification**: Class A (safety-critical), B (core pipeline), C (supporting)
- **CI pipeline**: 39 feature matrix, clippy, fmt, docs, compliance dashboard
- **Quality gates**: Pre-commit (secrets scan), CI (all tests), release (soak tests)
- **Version control**: Git, conventional commits, no force-push to main

### 2(b) Design specifications

**Reference**: `TECHNICAL_DOSSIER.md` §2, `docs/ARCHITECTURE_OVERVIEW.md`

| Component | Implementation | Key Parameters |
|-----------|---------------|----------------|
| HDC encoder | 16,384-bit BinaryHV | XOR/MAJ operations, LSH indexing |
| LTC neurons | CfC (Closed-form Continuous-time) | O(1) temporal jumps, 234Hz throughput |
| IIT/Phi | SpectralMIP (Gaussian MI) | r=0.99 vs exhaustive MIP, cadence 97 |
| FEP | Active inference agent | Prediction error → learning rate modulation |
| Ethics | 3-stage pipeline | MoralParser → ValueEvaluator → HarmoniesIntegrator |
| Safety | NRC-style 4-level monitoring | Green/Yellow/Orange/Red, 3-cycle escalation |
| Neuromodulation | 9-transmitter bath | DA/NE/5-HT/ACh/GABA/Oxytocin/Glutamate/Adenosine/ECB |

### 2(c) System architecture

```
Input → HDC Encode → CfC Evolve → Predict → Learn
                          ↓
                    Consciousness Engine (SpectralMIP + EqV2)
                          ↓
                    Ethics Engine (Moral + Value + Harmonies)
                          ↓
                    Safety Agent (NRC monitoring)
                          ↓
                    Output (CycleMetadata, 75+ telemetry fields)
```

**Cognitive loop**: 4 phases at 50Hz
1. **Perception**: Safety check, HDC encoding, moral parsing, strategy selection
2. **Dynamics**: CfC temporal evolution, FEP prediction, parallel post-processing
3. **Feedback**: Consciousness metrics, quality assessment, homeostasis
4. **Output**: Metadata assembly, telemetry, cycle result

**Reference**: `src/cognitive_loop/cycle.rs`, `docs/MODULE_WIRING_STATUS.md`

### 2(d) Computational resources

| Metric | Value |
|--------|-------|
| Cycle time | 20ms target (50Hz) |
| Memory footprint | ~500MB typical |
| HDC dimension | 16,384 bits |
| CfC neurons | Configurable (default: per-region) |
| Feature flags | 88 (default=[]) |

### 2(e) AI model descriptions

Symthaea does not use pre-trained foundation models as its primary computation. The system implements biologically-inspired algorithms from first principles:

- **HDC**: No pre-training; dictionary learned online from input distribution
- **CfC**: Weights initialized randomly; learned via prediction error gradient
- **Phi**: Computed analytically from connectivity matrix (no training)
- **Ethics**: Rule-based moral prototypes + learned harmony weights

**External models** (optional, via Broca subsystem):
See Section 7 — Third-Party AI Component Assessment.

---

## 3. Monitoring, Functioning, and Control

### 3(a) Human oversight measures

**Reference**: `HUMAN_OVERSIGHT.md` (EU AI Act Article 14)

| Measure | Implementation | Evidence |
|---------|---------------|----------|
| Emergency shutdown | Kill-switch procedure | `HUMAN_OVERSIGHT.md` §4 |
| Override capability | `SafetyOverrideEntry` with audit trail | `src/safety/agent.rs` |
| Real-time monitoring | Pulse TUI, CycleMetadata telemetry | `src/cognitive_loop/types/telemetry.rs` |
| Incident reporting | `SeriousIncidentReport` (Article 73) | `src/safety/agent.rs` |
| Operator training | Qualification requirements | `HUMAN_OVERSIGHT.md` §6 |

### 3(b) Technical measures for accuracy

| Measure | Metric | Target | Current |
|---------|--------|--------|---------|
| Phi validation | Pearson r vs exhaustive | ≥0.95 | 0.99 |
| Moral classification | Accuracy on prototypes | ≥90% | 91.1% |
| Safety monitoring | False negative rate | 0% | 0% (33 tests) |
| Consciousness clamping | NaN/Inf prevention | 100% | 100% (safety agent) |

### 3(c) Technical measures for robustness

| Measure | Implementation | Test Count |
|---------|---------------|-----------|
| Adversarial moral inputs | 26 adversarial + 15 soak tests | 41 |
| Property-based testing | Proptest (feedback, threshold, substrate) | 30+ |
| Substrate switching | Mid-run substrate changes remain finite | 6 proptests |
| Always-awake fallback | Stale calibration applied after 2000 cycles | 7 E2E tests |
| Consensus feedback | Noise-resistant averaging vs direct mutation | 3 soak tests |

### 3(d) Cybersecurity measures

| Measure | Implementation |
|---------|---------------|
| Credential management | BWS (Bitwarden Secrets) — no hardcoded credentials |
| Post-quantum crypto | ML-DSA-65/87, ML-KEM-768/1024 (Mycelix identity) |
| Input validation | Consciousness gating (4D profile → 5 tiers) |
| Audit trail | Correlation IDs on all governance gate operations |

---

## 4. Risk Management System (Article 9)

**Reference**: `AI_RISK_REGISTER.md`, `RISK_TREATMENT_PLAN.md`

### 4(a) Risk identification

15 risks across 6 categories identified in the risk register:
- Consciousness measurement validity
- Ethical reasoning failures
- Safety monitoring gaps
- Substrate dependence assumptions
- Calibration drift
- Integration complexity

### 4(b) Risk estimation and evaluation

Each risk scored by likelihood (1-5) × impact (1-5) with residual risk after mitigation.

### 4(c) Risk treatment

**Reference**: `RISK_TREATMENT_PLAN.md` — top 5 risks with:
- Treatment strategy (mitigate/accept/transfer)
- Implementation evidence (code + test references)
- Residual risk assessment
- Acceptance criteria

### 4(d) Residual risks

Documented per risk in the treatment plan. Key residual risks:
- Phi metric is a proxy for consciousness, not proof (theoretical limitation)
- Moral classification accuracy has a ~9% error rate (mitigated by safety veto)
- Substrate feasibility scores are theoretical for non-biological substrates

---

## 5. Data Governance (Article 10)

**Reference**: `DATA_GOVERNANCE.md`

### 5(a) Data sources

| Source | Type | Origin | Bias Assessment |
|--------|------|--------|----------------|
| Psych-bench normative data | Behavioral baselines | Published cognitive psychology | Western-centric (documented) |
| Moral prototypes | Ethical templates | Philosophical frameworks | Multi-tradition (utilitarian, deontological, virtue, care) |
| HDC dictionaries | Encoding seeds | Random generation | Uniform by construction |
| Harmony keywords | Value anchors | Cross-cultural ethics literature | English-language bias (documented) |
| Safety thresholds | Operating parameters | Published neuroscience | 119 constants with citations |
| Substrate profiles | Physical parameters | Published physics/biology | Theoretical for non-biological |

### 5(b) Data quality measures

- Provenance tracking for all 6 data sources
- Bias audit across 5 dimensions (cultural, linguistic, temporal, methodological, representational)
- Data lineage to specific source files documented

### 5(c) Training data (not applicable in traditional sense)

Symthaea does not train on datasets. Learning occurs online through:
- Prediction error signals (CfC weight updates)
- Moral prototype matching (ethics engine)
- Calibration from psych-bench z-scores (neuromodulator bath)

---

## 6. Testing and Validation (Article 15)

### 6(a) Test strategy

**Reference**: `QMS.md` §2-3, CI pipeline

| Test Type | Count | Purpose |
|-----------|-------|---------|
| Unit tests | 4,067+ | Component correctness |
| Integration tests | 167+ | Cross-system behavior |
| Property-based (proptest) | 30+ | Invariant verification |
| Soak tests | 15+ | Long-run stability |
| Adversarial | 41 | Robustness under attack |
| E2E calibration | 7 | Full pipeline validation |
| Compliance dashboard | 14 suites, 234 tests | Regulatory coverage |

### 6(b) Test evidence

```bash
# Reproduce all test evidence
cargo test --lib                              # 4,067+ unit tests
cargo test --test calibration_e2e             # 7 E2E tests
cargo test --test telemetry_validation        # 14 integration tests
cargo test --test proptest_feedback_stability # 30 property tests
cargo test --test proptest_threshold_sensitivity # 3 property tests
bash scripts/compliance-dashboard.sh          # 14 suites, 15 doc checks
```

### 6(c) Metrics used

| Metric | Value | Method |
|--------|-------|--------|
| Phi correlation | r = 0.99 | SpectralMIP vs exhaustive MIP (62 data points) |
| Moral accuracy | 91.1% | Prototype classification on moral_algebra test set |
| Safety false negatives | 0% | 33 unit tests + 15 soak tests |
| Threshold stability | Stable | ±wide perturbation across 6 scenarios |
| Consciousness bounds | [0.0, 1.0] | Property-based verification |

### 6(d) Known limitations

**Reference**: `TECHNICAL_STATUS.md`, `TRANSPARENCY_OBLIGATIONS.md` §2

- Phi is a proxy for consciousness, not direct measurement
- Moral classification has ~9% error rate
- Substrate feasibility for non-biological substrates is theoretical (honest_confidence: 0.10 for silicon)
- ~25% of modules remain structural/disconnected (iroh P2P, some consciousness subsystems)
- Single-developer limitation affects review rigor

---

## 7. Third-Party AI Component Assessment (ISO 42001 A.7.2)

### 7(a) Approved external AI models

Symthaea's Broca language subsystem can optionally use external LLMs via Ollama. Only approved models may be used:

| Model | Provider | License | Parameters | Use Case | Risk Level |
|-------|----------|---------|------------|----------|-----------|
| embeddinggemma:300m | Google | Apache 2.0 | 300M | Embedding generation | Low |
| gemma3:1b | Google | Gemma Terms | 1B | Language generation | Medium |
| qwen3:1.7b | Alibaba | Apache 2.0 | 1.7B | Language generation | Medium |
| gemma3:4b | Google | Gemma Terms | 4B | Language generation | Medium |
| mistral:7b | Mistral AI | Apache 2.0 | 7B | Language generation | Medium |

**Prohibited**: qwen2.5 variants (known issues documented)

### 7(b) Supply chain risk analysis

| Risk | Assessment | Mitigation |
|------|-----------|-----------|
| Model provenance | All models from established providers with public model cards | Only approved models list; Ollama local execution (no cloud API) |
| License compliance | Apache 2.0 (commercial OK) or Gemma Terms (research OK) | License terms documented per model |
| Capability boundaries | Small models (≤7B) with known limitations | Not used for safety-critical decisions; gated by consciousness threshold |
| Failure modes | Hallucination, toxic output, instruction following failures | Broca quality EMA tracking; low quality → raise consciousness gating threshold |
| Data leakage | Models run locally via Ollama; no data sent to external services | Network-isolated execution; no API keys required |
| Model updates | Ollama model versions pinned locally | Manual update process with re-evaluation |

### 7(c) Integration safeguards

- External models are **optional** (feature-gated: `ssm_language`, `liquid-mamba`)
- Broca output passes through consciousness quality gating
- Low-quality output triggers adaptive threshold raising (3+ poor → raise gate)
- External model output is **never** used for safety decisions (SafetyAgent is rule-based)
- External model output is **never** used for moral verdicts (Ethics Engine uses prototypes)

### 7(d) Monitoring

- `BrocaGenerationTelemetry`: quality, coherence, semantic_PE per generation
- `broca_quality_ema` tracked in LoopStats
- Low quality streak triggers automatic consciousness gate tightening

---

## 8. Post-Market Monitoring (Article 72)

**Reference**: `POST_MARKET_MONITORING.md`

| Component | Mechanism | Frequency |
|-----------|-----------|-----------|
| Safety monitoring | SafetyAgent (Green/Yellow/Orange/Red) | Every cycle (50Hz) |
| Calibration drift | CalibrationHistory (20-entry sliding window) | Per calibration event |
| Self-assessment | SelfAssessmentMonitor (PE/coherence/confidence/attention) | Continuous (200-cycle warmup) |
| Moral drift | Moral topology entropy + attractor detection | Every ethics engine cycle |
| Compliance dashboard | 14 test suites + 15 doc checks | CI on every push |
| Serious incidents | `SeriousIncidentReport` (Article 73) | On Red-level events |

---

## 9. Transparency (Article 13)

**Reference**: `TRANSPARENCY_OBLIGATIONS.md`

- System description and intended purpose (§1)
- All metrics explained with ranges, methods, and limitations (§2)
- Known limitations and failure modes (§3)
- Contestability procedures for affected persons (§4)
- Data transparency and provenance (§5)

---

## Document Cross-Reference Index

| Annex IV Element | Primary Document | Supporting Documents |
|-----------------|-----------------|---------------------|
| 1. General description | This document §1 | `TECHNICAL_DOSSIER.md`, `EU_AI_ACT_CLASSIFICATION.md` |
| 2. Design specifications | This document §2 | `ARCHITECTURE_OVERVIEW.md`, `MODULE_WIRING_STATUS.md` |
| 3. Monitoring and control | This document §3 | `HUMAN_OVERSIGHT.md`, `TRANSPARENCY_OBLIGATIONS.md` |
| 4. Risk management | This document §4 | `AI_RISK_REGISTER.md`, `RISK_TREATMENT_PLAN.md` |
| 5. Data governance | This document §5 | `DATA_GOVERNANCE.md` |
| 6. Testing and validation | This document §6 | `QMS.md`, CI logs, test results |
| 7. Third-party components | This document §7 | `GOVERNANCE_CHARTER.md` (approved models) |
| 8. Post-market monitoring | This document §8 | `POST_MARKET_MONITORING.md`, `INCIDENT_RUNBOOK.md` |
| 9. Transparency | This document §9 | `TRANSPARENCY_OBLIGATIONS.md`, `EXPLAINABILITY_FRAMEWORK.md` |

---

*This document should be updated whenever significant system changes occur or when preparing for conformity assessment.*
