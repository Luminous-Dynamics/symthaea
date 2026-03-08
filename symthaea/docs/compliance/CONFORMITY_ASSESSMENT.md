# Conformity Assessment Preparation — EU AI Act Article 43

This document prepares Symthaea for third-party conformity assessment under the EU AI Act. It identifies the assessment pathway, applicable standards, potential notified bodies, and current readiness gaps.

---

## 1. Classification Determination

Symthaea is potentially **High-Risk** under EU AI Act Annex III in the following categories:

- **Category 2** — Biometrics (if deployed for emotion recognition or behavioral analysis)
- **Category 5** — Access to essential services (if used in housing, credit, or public benefit decisions via Mycelix integration)
- **Category 8** — AI systems intended to influence democratic processes (if civic-bridge governance features are deployed publicly)

**Reference**: `EU_AI_ACT_CLASSIFICATION.md` for the full classification analysis.

**Current status**: Symthaea is a **research platform** and is not yet deployed for decisions affecting natural persons. The consciousness engine, cognitive loop, and ethical reasoning subsystems operate in experimental contexts only.

**If deployed commercially**: Symthaea would require conformity assessment per Article 43(1) before being placed on the EU market or put into service.

---

## 2. Conformity Assessment Pathway

Article 43(1) provides two pathways:

1. **Internal control** (Annex VI) — Self-assessment by the provider
2. **Third-party assessment** (Annex VII) — Assessment by a notified body

### Selection criteria

- For AI systems under **Annex III category 1 (biometrics for identification)**: third-party assessment is **mandatory**.
- For **other Annex III categories** (including 2, 5, 8): internal control is sufficient **if** harmonized standards have been applied and correctly followed.
- If no harmonized standards exist or are not followed: third-party assessment is required.

### Recommendation

Pursue **internal control (Annex VI)** initially, as Symthaea does not fall under category 1. Prepare documentation and processes to a level that would satisfy third-party scrutiny, so that escalation to Annex VII is straightforward if:

- Scope expands to include biometric identification
- Harmonized standards are not yet available or cannot be fully applied
- Market or customer requirements demand third-party certification

---

## 3. Applicable Harmonized Standards

| Standard | Status | Relevance |
|----------|--------|-----------|
| ISO/IEC 42001:2023 | Published | AI Management System — primary framework for organizational AI governance |
| ISO/IEC 23894:2023 | Published | AI Risk Management — alignment with our risk register and FRIA methodology |
| ISO/IEC 42005:2025 | Published | AI Impact Assessment — methodology for Fundamental Rights Impact Assessment |
| IEEE 7000-2021 | Published | Value-Based Design — direct alignment with Symthaea's Eight Harmonies framework |
| ISO/IEC 25059 | Published | AI Quality Model — quality metrics for AI system evaluation |
| CEN-CENELEC JTC 21 | In progress | EU AI Act harmonized standards (expected 2025-2026) |

### Notes on harmonized standards

- CEN-CENELEC JTC 21 is developing the official harmonized standards referenced by the AI Act. Until these are published in the Official Journal of the EU, voluntary standards (ISO/IEC, IEEE) serve as the best available reference.
- ISO/IEC 42001 is the strongest candidate for presumption of conformity once harmonized standard references are finalized.
- IEEE 7000 is not an EU harmonized standard but provides direct methodological support for Symthaea's value-based architecture.

---

## 4. Notified Bodies (Potential)

The following organizations have AI certification programs or related capabilities and are candidates for third-party conformity assessment:

| Organization | Country | Relevant Programs |
|-------------|---------|-------------------|
| TUV SUD | Germany | AI certification programs, established test lab infrastructure |
| Bureau Veritas | France | Digital trust services, AI assurance |
| BSI Group | UK | AI standards and certification (may apply for mutual recognition post-Brexit) |
| DNV | Norway | AI assurance framework, risk-based certification |

**Important**: The official list of notified bodies designated under the EU AI Act has **not yet been published** (expected 2025-2026). Member States must designate notified bodies and notify the Commission. The organizations listed above are prospective candidates based on their existing capabilities and stated intentions.

Monitor the EU AI Act notified body database (NANDO) for official designations as they become available.

---

## 5. Documentation Requirements (Annex IV)

Article 11 requires technical documentation drawn up before the AI system is placed on the market. Annex IV specifies the contents. Cross-reference to `TECHNICAL_DOSSIER.md` for the following required elements:

1. **General system description** — Purpose, intended use, version history, hardware/software dependencies
2. **Design specifications and development methodology** — Architecture overview, cognitive loop design, HDC/LTC/IIT pipeline
3. **Risk management documentation** — Risk register, mitigation measures, residual risk acceptance criteria
4. **Data governance documentation** — Training data provenance, bias assessment, data quality metrics (see `DATA_GOVERNANCE.md`)
5. **Testing and validation results** — Test suites (`cargo test --lib`, `cargo test --all-features`), property-based tests (`tests/proptest_feedback_stability.rs`, `tests/proptest_threshold_sensitivity.rs`), soak tests, E2E integration tests
6. **Post-market monitoring plan** — Telemetry collection via `CycleMetadata`, `SafetyAuditReport`, anomaly detection procedures

### Key test commands for evidence gathering

```bash
# Core test suite (~3,735+ tests)
cargo test --lib

# Full feature matrix
cargo test --all-features

# Property-based stability tests
cargo test --test proptest_feedback_stability
cargo test --test proptest_threshold_sensitivity

# Substrate independence validation
cargo test -p symthaea-core --lib substrate_independence
cargo test -p symthaea-core --lib substrate_validation

# Calibration and telemetry
cargo test --test calibration_e2e
cargo test --test telemetry_validation
```

---

## 6. Current Readiness Assessment

| Requirement | Article | Status | Gap | Priority |
|-------------|---------|--------|-----|----------|
| Technical documentation | Art. 11 | 90% | `ANNEX_IV_TECHNICAL_DOCUMENTATION.md` — master index covering all 9 Annex IV elements with cross-references | Low |
| Risk management | Art. 9 | 75% | Risk register + treatment plan in place; formalize residual risk acceptance criteria | Medium |
| Data governance | Art. 10 | 85% | `DATA_GOVERNANCE.md` with training data provenance, bias audit, and data quality assurance | Low |
| Human oversight | Art. 14 | 90% | `HUMAN_OVERSIGHT.md` — override procedures, kill-switch, operator roles, `SeriousIncidentReport` | Low |
| Accuracy and robustness | Art. 15 | 80% | Soak tests, proptests, adversarial tests in place; formal accuracy metrics documented in Annex IV §6 | Low |
| Quality management | Art. 17 | 90% | `QMS.md` + `DEVELOPMENT_PROCEDURES.md` — quality gates, CI pipeline, threshold protocol | Low |
| Logging | Art. 12 | 85% | `SafetyAuditReport` + `CycleMetadata` + `SeriousIncidentReport` (Article 73); retention policy | Low |
| Transparency | Art. 13 | 90% | `TRANSPARENCY_OBLIGATIONS.md` + Annex IV §9 — system description, metrics, limitations, contestability | Low |

### Priority actions

1. ~~**(High)** Assemble Annex IV technical documentation~~ — **Done**: `ANNEX_IV_TECHNICAL_DOCUMENTATION.md`
2. ~~**(High)** Complete human oversight mechanisms~~ — **Done**: `HUMAN_OVERSIGHT.md`
3. ~~**(Medium)** Formalize QMS and development procedures~~ — **Done**: `QMS.md` + `DEVELOPMENT_PROCEDURES.md`
4. **(Medium)** Formal data quality framework — psych-bench baselines need formal quality metrics
5. **(Medium)** External stakeholder feedback loop — needed for NIST GOV-6 and IEEE 7000 value validation

---

## 7. Timeline and Milestones

| Phase | Period | Objective | Status |
|-------|--------|-----------|--------|
| Phase 1 | Q1 2026 | Compliance documentation framework | Complete |
| Phase 2 | Q1 2026 | Close High-priority gaps (Annex IV, human oversight, development procedures, value verification) | **Complete** |
| Phase 3 | Q3 2026 | Internal conformity assessment dry run | Planned |
| Phase 4 | Q4 2026 | Submit for third-party assessment if required | Contingent |

### Critical date

**August 2, 2026**: EU AI Act enforcement date for High-Risk AI system provisions (Article 113). Any High-Risk AI system placed on the EU market or put into service after this date must comply.

### Phase 2 deliverables (Q2 2026)

- Annex IV technical documentation package
- Human oversight procedure document
- Formal residual risk acceptance register
- User-facing transparency documentation

### Phase 3 activities (Q3 2026)

- Internal audit against Annex VI checklist
- Gap remediation from audit findings
- Mock assessment with external AI governance consultant (optional)
- Final documentation review

---

## 8. Cost Estimate

| Activity | Estimated Cost | Notes |
|----------|---------------|-------|
| Internal conformity assessment (Annex VI) | Engineering time only | ~2-3 person-weeks for documentation assembly and gap closure |
| Third-party assessment (Annex VII) | EUR 15,000 - EUR 50,000 | Based on comparable AI certification programs; varies by scope and notified body |
| External AI governance consultant | EUR 5,000 - EUR 15,000 | Optional; recommended for Phase 3 dry run |
| Ongoing compliance maintenance | ~10% of engineering time | Documentation updates, test maintenance, monitoring |

### Cost drivers for third-party assessment

- System complexity (Symthaea's multi-subsystem architecture increases assessment scope)
- Number of Annex III categories claimed (each category may require separate evaluation)
- Availability of harmonized standards (absence increases assessment effort)
- Geographic scope of deployment (multiple EU member states may increase cost)

---

## References

- **EU AI Act**: Regulation (EU) 2024/1689 of the European Parliament and of the Council
- **Article 43**: Conformity assessment
- **Annex III**: High-risk AI systems
- **Annex IV**: Technical documentation
- **Annex VI**: Internal control conformity assessment procedure
- **Annex VII**: Conformity assessment based on assessment of quality management system and technical documentation
- **Related project files**:
  - `symthaea/docs/compliance/EU_AI_ACT_CLASSIFICATION.md` — Classification analysis
  - `symthaea/docs/compliance/TECHNICAL_DOSSIER.md` — Technical documentation package
  - `symthaea/docs/compliance/DATA_GOVERNANCE.md` — Data governance framework
  - `symthaea/src/cognitive_loop/types/telemetry.rs` — CycleMetadata and telemetry structures
  - `symthaea/src/cognitive_loop/thresholds.rs` — Centralized safety thresholds
  - `symthaea/.github/workflows/ci.yml` — CI pipeline (quality assurance evidence)
