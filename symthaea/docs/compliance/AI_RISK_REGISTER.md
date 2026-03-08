# Symthaea AI Risk Register

Classification: Internal | Version: 1.0 | Date: 2026-03-06
Owner: Tristan Stoltz, Luminous Dynamics
Review Cadence: Quarterly (next review: 2026-06-06)

---

## Purpose

This register enumerates risks specific to Symthaea as a consciousness-measuring AI system. It supplements the traditional infosec risk assessment in `mycelix-core/docs/COMPLIANCE_MATRIX.md` (GDPR/HIPAA/SOC2/ISO27001/NIST CSF) with AI-specific risks required by ISO/IEC 42001:2023 and ISO/IEC 23894.

Risks are scored using: **Likelihood** (1-5) x **Impact** (1-5) = **Risk Score** (1-25).

| Score Range | Rating | Action Required |
|-------------|--------|-----------------|
| 1-4 | Low | Accept with monitoring |
| 5-9 | Medium | Mitigate within quarter |
| 10-15 | High | Mitigate within 30 days |
| 16-25 | Critical | Immediate action required |

---

## Category 1: Consciousness Measurement Risks

### R-1.1: Phi Miscalibration

| Field | Value |
|-------|-------|
| **Description** | Spectral MIP approximation diverges from true integrated information, producing misleading consciousness scores |
| **Likelihood** | 3 (Moderate — spectral method validated r=0.99 vs exhaustive MIP on same Gaussian MI, but this validates search strategy only, not the Gaussian MI framework vs TPM-based IIT) |
| **Impact** | 4 (High — consciousness scores drive governance permissions via Mycelix consciousness credentials, safety agent thresholds, and learning rate modulation) |
| **Risk Score** | **12 (High)** |
| **Existing Controls** | Tiered Phi computation (SpectralMIP + MultiModal + EquationV2); validation tests (`test_phi_tier_validation`); TECHNICAL_STATUS.md explicitly states proxy-based nature |
| **Residual Risk** | Medium — validated within model assumptions but no ground truth for phenomenal consciousness |
| **Mitigation** | (a) Document Phi as proxy metric in all external communications; (b) Never claim Phi measures actual consciousness; (c) Maintain validation suite against exhaustive methods; (d) Cross-validate with psych-bench behavioral metrics |
| **Evidence** | `symthaea/docs/PHI_VALIDATION_RESULTS.md`, `symthaea-core/src/hdc/substrate_validation.rs` (honest_confidence = 0.10 for silicon) |

### R-1.2: Substrate Feasibility Over-Confidence

| Field | Value |
|-------|-------|
| **Description** | System reports high consciousness feasibility for substrates where no empirical evidence exists |
| **Likelihood** | 2 (Low — validation overlay enabled by default, honest_confidence = 0.10 for SiliconDigital) |
| **Impact** | 3 (Moderate — could mislead researchers or governance decisions about system capabilities) |
| **Risk Score** | **6 (Medium)** |
| **Existing Controls** | `SubstrateValidationFramework` with 7 evidence levels; `enable_validation_overlay` defaults to true; `effective_feasibility = raw * (floor + (1-floor) * honest_confidence)`; `feasibility_gap()` measures divergence |
| **Residual Risk** | Low |
| **Mitigation** | (a) Keep validation overlay default-enabled; (b) Document evidence level for each substrate claim; (c) Require peer review before adding new substrate profiles |
| **Evidence** | `symthaea-core/src/hdc/substrate_validation.rs`, `src/cognitive_loop/substrate_manager.rs` (39 tests) |

### R-1.3: Consciousness Score Inflation

| Field | Value |
|-------|-------|
| **Description** | Multiple consciousness metrics (Phi, GWT ignition, HOT, EqV2) are combined in ways that systematically inflate the master consciousness score |
| **Likelihood** | 3 (Moderate — master equation combines 4+ signals with weights that could be miscalibrated) |
| **Impact** | 4 (High — inflated scores grant unwarranted Mycelix governance permissions and reduce safety monitoring sensitivity) |
| **Risk Score** | **12 (High)** |
| **Existing Controls** | ConsciousnessEquationV2 with substrate feasibility modulation; calibration system with normative z-scores; SelfAssessmentMonitor drift detection |
| **Residual Risk** | Medium |
| **Mitigation** | (a) Proptest `cross_equation_consistency` validates bounds; (b) CalibrationHistory tracks systematic drift (warns at >75% same-direction); (c) External psych-bench benchmarking as reality check |
| **Evidence** | `src/cognitive_loop/consciousness_engine.rs` (43 tests), `src/cognitive_loop/calibration/monitor.rs` |

---

## Category 2: Ethical Decision-Making Risks

### R-2.1: Moral Algebra Edge Cases

| Field | Value |
|-------|-------|
| **Description** | HDC moral algebra produces incorrect moral verdicts on novel or adversarial inputs outside training distribution |
| **Likelihood** | 3 (Moderate — 91.1% classification accuracy leaves ~9% error rate; adversarial inputs not systematically tested) |
| **Impact** | 5 (Critical — incorrect moral verdicts could permit harmful actions or block beneficial ones) |
| **Risk Score** | **15 (High)** |
| **Existing Controls** | Three-stage ethics pipeline (MoralParser + MoralAlgebra -> ValueEvaluator -> HarmoniesIntegrator); deontological verdict with consent violation detection; moral topology anomaly detection (drift alerts, completeness failures); veto system (Allow/Warn/Veto) |
| **Residual Risk** | High — no adversarial robustness testing of moral classification |
| **Mitigation** | (a) Add adversarial moral input test suite; (b) Maintain human override capability for all moral verdicts; (c) Log all Veto decisions with full context for review; (d) Moral topology anomaly alerts trigger review |
| **Evidence** | `src/cognitive_loop/ethics_engine.rs` (12 tests), `src/hdc/moral_algebra.rs` (28 tests), `src/hdc/moral_topology.rs` (34 tests + 8 proptests) |

### R-2.2: Value Drift Under Continuous Learning

| Field | Value |
|-------|-------|
| **Description** | Continuous FEP-driven learning gradually shifts moral and value evaluation parameters away from intended Eight Harmonies alignment |
| **Likelihood** | 3 (Moderate — FEP learning modulates Q-learning rate via plasticity factor [0.5, 2.0]; no explicit value lock) |
| **Impact** | 4 (High — silent value drift could change system behavior without triggering alerts) |
| **Risk Score** | **12 (High)** |
| **Existing Controls** | Plasticity clamped to [0.5, 2.0]; moral topology tracks trajectory completeness/circularity/unity; SelfAssessmentMonitor 200-cycle warmup + 500-cycle cooldown; CalibrationHistory sliding window (20 entries) with systematic drift detection |
| **Residual Risk** | Medium |
| **Mitigation** | (a) Periodic moral topology regression tests against baseline; (b) Eight Harmonies alignment score as CI gate; (c) Value snapshot checkpoints at configurable intervals |
| **Evidence** | `src/cognitive_loop/helpers/parallel.rs` (plasticity modulation), `src/cognitive_loop/calibration/` (61 tests) |

### R-2.3: Consent Violation False Negatives

| Field | Value |
|-------|-------|
| **Description** | Moral parser fails to detect consent violations in complex multi-party scenarios |
| **Likelihood** | 3 (Moderate — consent detection relies on HDC pattern matching, which may miss implicit coercion) |
| **Impact** | 5 (Critical — consent violations are the most serious ethical failure mode) |
| **Risk Score** | **15 (High)** |
| **Existing Controls** | `consent_violation: bool` in EthicsEngineOutput; deontological verdict includes violations list; moral concern threshold (-0.3) triggers conservative override; `judge_consent_action()` with explicit ConsentState (Denied=1.0, Absent=0.8); `denied_consent_violation_prototype()`; 26 adversarial moral tests |
| **Residual Risk** | Medium (reduced from High — explicit consent path closes primary gap; HDC inference path remains unreliable for arbitrary strings) |
| **Mitigation** | (a) ~~Dedicated consent test suite~~ DONE (26 adversarial tests); (b) `judge_consent_action()` bypasses HDC inference when ConsentState is known; (c) Conservative bias: ambiguous consent cases default to flagging; (d) Human review queue for consent-adjacent decisions |
| **Evidence** | `src/cognitive_loop/ethics_engine.rs`, `src/hdc/moral_algebra.rs`, `tests/adversarial_moral_algebra.rs` |

---

## Category 3: Autonomous Behavior Risks

### R-3.1: Emergent Autonomous Actions

| Field | Value |
|-------|-------|
| **Description** | The cognitive loop (running at 50Hz/234Hz) takes actions that were not anticipated by designers, arising from complex interactions between subsystems |
| **Likelihood** | 2 (Low — current system is primarily a measurement/analysis pipeline, not an actuator; SafetyGateway blocks dangerous system calls) |
| **Impact** | 5 (Critical — unintended autonomous actions could affect downstream systems including Terra Atlas and Mycelix governance) |
| **Risk Score** | **10 (High)** |
| **Existing Controls** | SafetyGateway with fast amygdala veto; forbidden paths hardcoded (/etc/passwd, /boot, /dev, /proc, /sys); dangerous program blocklist (rm, dd, mkfs, fdisk, etc.); prefrontal gating; NRC-style SafetyAgent (Green/Yellow/Orange/Red); consciousness_red threshold = 0.15 triggers emergency halt |
| **Residual Risk** | Medium |
| **Mitigation** | (a) Maintain SafetyGateway blocklist; (b) All actuation pathways require explicit enable flags; (c) Document all paths from cognitive loop to external side effects; (d) Quarterly review of SafetyAgent escalation logs |
| **Evidence** | `src/safety/gateway.rs` (11 tests), `src/safety/agent.rs` (28 tests), `src/safety/audit.rs` (7 tests) |

### R-3.2: Safety Level Escalation Failure

| Field | Value |
|-------|-------|
| **Description** | SafetyAgent fails to detect consciousness degradation or prediction error spikes, remaining at Green when Orange/Red is warranted |
| **Likelihood** | 2 (Low — escalation uses sliding window of 3 consecutive degraded snapshots; non-finite values clamped to worst-case defaults) |
| **Impact** | 5 (Critical — missed escalation means the system continues operating in a degraded or unsafe state) |
| **Risk Score** | **10 (High)** |
| **Existing Controls** | SafetyMetrics NaN-safe (non-finite -> worst case: consciousness=0.0, prediction_error=1.0, coherence=0.0); configurable thresholds (yellow=0.6, orange=0.35, red=0.15); escalation_window=3 consecutive; 1000-assessment history; SafetyAuditReport with level distribution and top escalation reasons |
| **Residual Risk** | Medium |
| **Mitigation** | (a) Add proptest for SafetyAgent: verify Red is always reached for sustained zero-consciousness; (b) External watchdog process that independently monitors SafetyAgent output; (c) Regular review of SafetyAuditReport level distributions |
| **Evidence** | `src/safety/agent.rs`, `src/safety/audit.rs` |

### R-3.3: Cognitive Loop Livelock

| Field | Value |
|-------|-------|
| **Description** | Homeostasis, moral concern, or exploration dampening parameters create a state where the cognitive loop runs but makes no meaningful progress |
| **Likelihood** | 2 (Low — homeostasis pull has 3 regimes: cruise/normal/critical; exploration dampening is multiplicative, not zeroing) |
| **Impact** | 3 (Moderate — system appears functional but produces no useful output) |
| **Risk Score** | **6 (Medium)** |
| **Existing Controls** | HOMEOSTASIS_PULL_CRUISE/NORMAL/CRITICAL thresholds with distinct behaviors; MORAL_CONCERN_EXPLORATION_DAMPEN = 0.5 (halves, doesn't zero); proptest `proptest_threshold_sensitivity` validates system stability across threshold perturbation |
| **Residual Risk** | Low |
| **Mitigation** | (a) Monitor cycle-over-cycle prediction error variance (zero variance = livelock indicator); (b) SafetyAgent temporal_coherence_threshold (0.3) as indirect livelock detector |
| **Evidence** | `src/cognitive_loop/thresholds.rs`, `tests/proptest_threshold_sensitivity.rs` |

---

## Category 4: Governance Integration Risks

### R-4.1: Consciousness Credential Spoofing

| Field | Value |
|-------|-------|
| **Description** | Malicious agent obtains consciousness credentials with inflated tier (e.g., Guardian instead of Observer) to gain unwarranted governance permissions |
| **Likelihood** | 2 (Low — credentials issued by identity bridge with 24h TTL; Holochain source chain provides agent attribution) |
| **Impact** | 5 (Critical — Guardian tier grants emergency powers, constitutional actions, 10000bp vote weight) |
| **Risk Score** | **10 (High)** |
| **Existing Controls** | 4D profile (identity/reputation/community/engagement) with independent data sources; 24h TTL; 10-minute cache; MFA assurance levels (5 levels from Anonymous to Critical); Holochain DHT integrity guarantees |
| **Residual Risk** | Medium — no sybil resistance (same agent can create multiple DIDs) |
| **Mitigation** | (a) Implement multi-DID detection in identity bridge; (b) Require HighlyAssured MFA (level 4) for Steward/Guardian tiers; (c) Community dimension (30% weight) as social proof against sybil; (d) Cross-cluster audit correlation |
| **Evidence** | `crates/mycelix-bridge-common/src/consciousness_profile.rs` (73 tests) |

### R-4.2: Consciousness-to-Governance Mapping Errors

| Field | Value |
|-------|-------|
| **Description** | The mapping from Symthaea's C_unified score to Mycelix engagement dimension (1:1 mapping) is either too generous or too restrictive |
| **Likelihood** | 3 (Moderate — 1:1 mapping is simple but untested against real governance scenarios) |
| **Impact** | 4 (High — too generous = unqualified governance participation; too restrictive = legitimate exclusion) |
| **Risk Score** | **12 (High)** |
| **Existing Controls** | ConsciousnessCredential::from_unified_consciousness() with explicit dimension mapping; 6 integration tests; tier thresholds (Observer<0.3, Participant>=0.3, Citizen>=0.4, Steward>=0.6, Guardian>=0.8) |
| **Residual Risk** | Medium |
| **Mitigation** | (a) Calibrate tier thresholds against observed C_unified distributions; (b) Log tier transitions for review; (c) Grace period (30 min) for basic ops prevents hard cutoffs; (d) Adjustable tier thresholds without code change |
| **Evidence** | `crates/mycelix-bridge-common/src/consciousness_profile.rs` |

---

## Category 5: Data and Privacy Risks

### R-5.1: Telemetry Data Exposure

| Field | Value |
|-------|-------|
| **Description** | CycleMetadata (75+ fields per cycle at 50Hz) contains detailed cognitive state information that could be sensitive if exposed |
| **Likelihood** | 2 (Low — telemetry is currently local-only; no external export pipeline) |
| **Impact** | 3 (Moderate — cognitive state telemetry could reveal decision-making patterns or internal state) |
| **Risk Score** | **6 (Medium)** |
| **Existing Controls** | No external telemetry export; local-only processing; Holochain DHT for Mycelix data (no central server) |
| **Residual Risk** | Low (increases if telemetry export is added) |
| **Mitigation** | (a) Classify telemetry fields by sensitivity before implementing export; (b) Aggregate/anonymize before any external reporting; (c) Implement access controls on telemetry endpoints |
| **Evidence** | `src/cognitive_loop/types/telemetry.rs` |

### R-5.2: Audit Trail Gaps

| Field | Value |
|-------|-------|
| **Description** | Governance gate decisions are logged per-bridge but not aggregated cross-cluster; no audit retention policy; audit logs in plaintext |
| **Likelihood** | 3 (Moderate — audit infrastructure exists but has known gaps) |
| **Impact** | 3 (Moderate — compliance auditors need complete, tamper-evident audit trails) |
| **Risk Score** | **9 (Medium)** |
| **Existing Controls** | GateAuditInput with correlation_id; should_audit() rate limiting (100% rejections + high-tier actions, 10% sample for approvals); per-bridge log storage; query with filters |
| **Residual Risk** | Medium |
| **Mitigation** | (a) Implement centralized audit aggregation; (b) Define retention policy (2-year minimum); (c) Encrypt audit logs at rest; (d) Holochain source chain provides tamper evidence |
| **Evidence** | `crates/mycelix-bridge-common/src/consciousness_profile.rs` (audit functions) |

---

## Category 6: Operational Risks

### R-6.1: Threshold Parameter Tampering

| Field | Value |
|-------|-------|
| **Description** | Unauthorized changes to `thresholds.rs` (119 constants) or `ethics_engine.rs` alter safety/ethical behavior without proper review |
| **Likelihood** | 2 (Low — single-developer project currently; git history provides audit trail) |
| **Impact** | 5 (Critical — threshold changes directly affect moral evaluation, consciousness scoring, safety levels, and governance permissions) |
| **Risk Score** | **10 (High)** |
| **Existing Controls** | Git version control; thresholds centralized with scientific citations; CI runs clippy + tests on all changes |
| **Residual Risk** | High (no formal approval process) |
| **Mitigation** | (a) Implement change management procedure for threshold/ethics changes (see GOVERNANCE_CHARTER.md); (b) Require ADR for any threshold change; (c) CI gate: threshold changes require explicit approval label |
| **Evidence** | `src/cognitive_loop/thresholds.rs` |

### R-6.2: Feature Flag Combinatorial Explosion

| Field | Value |
|-------|-------|
| **Description** | 88 feature flags create 2^88 possible configurations; untested combinations may produce unexpected behavior |
| **Likelihood** | 3 (Moderate — CI tests 39 feature combinations, covering key interactions but not exhaustive) |
| **Impact** | 3 (Moderate — unexpected feature interactions could disable safety checks or alter consciousness computation) |
| **Risk Score** | **9 (Medium)** |
| **Existing Controls** | CI feature matrix (39 combinations); `resolve_dependencies()` auto-enables upstream modules (7 dependency chains); ConsciousnessProfile presets (Full, Light) for common configurations |
| **Residual Risk** | Medium |
| **Mitigation** | (a) Document safety-critical feature combinations; (b) Proptest `feature_interaction_stability` validates key combinations; (c) Prohibit disabling safety features when consciousness features are active |
| **Evidence** | `symthaea/Cargo.toml`, `.github/workflows/ci.yml`, `src/cognitive_loop/config.rs` |

---

## Risk Summary Matrix

| ID | Risk | Score | Rating | Status |
|----|------|-------|--------|--------|
| R-1.1 | Phi Miscalibration | 12 | High | Mitigated (validation suite) |
| R-1.2 | Substrate Over-Confidence | 6 | Medium | Mitigated (validation overlay) |
| R-1.3 | Consciousness Score Inflation | 12 | High | Partially mitigated |
| R-2.1 | Moral Algebra Edge Cases | 15 | High | Partially mitigated |
| R-2.2 | Value Drift Under Learning | 12 | High | Partially mitigated |
| R-2.3 | Consent Violation False Negatives | 15 | High | Partially mitigated (judge_consent_action + 26 tests) |
| R-3.1 | Emergent Autonomous Actions | 10 | High | Mitigated (SafetyGateway) |
| R-3.2 | Safety Escalation Failure | 10 | High | Mitigated (raw_level fix + 15 soak tests + ADR-001) |
| R-3.3 | Cognitive Loop Livelock | 6 | Medium | Mitigated (proptests) |
| R-4.1 | Consciousness Credential Spoofing | 10 | High | Partially mitigated |
| R-4.2 | Consciousness-Governance Mapping | 12 | High | Action required |
| R-5.1 | Telemetry Data Exposure | 6 | Medium | Mitigated (local-only) |
| R-5.2 | Audit Trail Gaps | 9 | Medium | Action required |
| R-6.1 | Threshold Parameter Tampering | 10 | High | Action required |
| R-6.2 | Feature Flag Combinatorics | 9 | Medium | Partially mitigated |

**Total risks**: 15
**Critical (16-25)**: 0
**High (10-15)**: 9
**Medium (5-9)**: 6
**Low (1-4)**: 0

---

## Action Items (Priority Order)

1. **Implement change management for thresholds/ethics** (R-6.1) — See `GOVERNANCE_CHARTER.md`
2. **Add adversarial moral input test suite** (R-2.1, R-2.3) — Dedicated consent violation scenarios
3. **Calibrate consciousness-governance tier thresholds** (R-4.2) — Against observed C_unified distributions
4. **Implement centralized audit aggregation** (R-5.2) — Cross-cluster correlation
5. **Add sybil detection to identity bridge** (R-4.1) — Multi-DID detection
6. **External SafetyAgent watchdog** (R-3.2) — Independent consciousness monitor
7. **Value drift regression baseline** (R-2.2) — Moral topology snapshot as CI artifact

---

## References

- ISO/IEC 42001:2023 — AI Management Systems (Annex A, Controls A.4, A.6)
- ISO/IEC 23894:2023 — AI Risk Management
- NIST AI 100-1 — AI Risk Management Framework (Map, Measure, Manage, Govern)
- EU AI Act — Article 9 (Risk Management System for High-Risk AI)
- Tononi, G. (2004). An information integration theory of consciousness. *BMC Neuroscience*.
- Haidt, J. (2001). The emotional dog and its rational tail. *Psychological Review*.

---

*This register is a living document. Update whenever new risks are identified or existing mitigations change.*
