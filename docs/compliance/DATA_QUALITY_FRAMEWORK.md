# Data Quality Framework — ISO 42001 A.5.3

Classification: Internal | Version: 1.0 | Date: 2026-03-08
Owner: Tristan Stoltz, Luminous Dynamics

---

## Purpose

This document defines the data quality framework for Symthaea's internal data sources per ISO 42001 A.5.3 and EU AI Act Article 10(3). Symthaea does not ingest external personal data — all data sources are synthetic, derived from published science, or generated at runtime.

---

## 1. Data Quality Dimensions

Each data source is assessed across 6 quality dimensions:

| Dimension | Definition | Measurement |
|-----------|-----------|-------------|
| **Accuracy** | Data correctly represents intended values | Validation tests, cross-reference to published source |
| **Completeness** | No missing values or coverage gaps | Enumeration checks, domain coverage analysis |
| **Consistency** | No contradictions within or across sources | Cross-source invariant checks, ordering validation |
| **Timeliness** | Data remains current for its purpose | Publication date tracking, update frequency |
| **Provenance** | Origin and transformation chain documented | Source file references in `DATA_GOVERNANCE.md` §4 |
| **Relevance** | Data serves its intended purpose | Alignment with scientific citations |

---

## 2. Per-Source Quality Assessment

### 2.1 Psych-Bench Normative Data

| Dimension | Score | Evidence |
|-----------|-------|---------|
| Accuracy | High | z-scores derived from published cognitive psychology (Stroop, N-back, CPT, etc.) |
| Completeness | Medium | 14 benchmark domains; some domains (e.g., social cognition) underrepresented |
| Consistency | High | `NormativeReport` struct enforces sign conventions; `from_normative_z_scores()` un-corrects directionality |
| Timeliness | High | Based on contemporary (2010-2024) normative datasets |
| Provenance | High | Each benchmark cites source publication in code comments |
| Relevance | High | Direct calibration of 9-transmitter neuromodulator bath |

**Quality controls**:
- 633 unit tests in `crates/symthaea-psych-bench/`
- 52 benchmarks with validation bounds
- Weekly regression job in CI detects drift
- Tolerance gating: `tolerance_adjusted_factor()` prevents withdrawal artifacts

**Known limitations**:
- Western-centric normative baselines (documented in `DATA_GOVERNANCE.md`)
- Some benchmarks (UG/RME for oxytocin) rely on behavioral economics which has replication concerns

### 2.2 Moral Prototypes

| Dimension | Score | Evidence |
|-----------|-------|---------|
| Accuracy | High | Multi-tradition coverage (utilitarian, deontological, virtue, care) |
| Completeness | Medium | English-language prototypes only; cultural coverage gaps |
| Consistency | High | 28 moral_algebra tests + 26 adversarial tests verify classification |
| Timeliness | High | Based on established ethical frameworks (not trend-dependent) |
| Provenance | High | Prototypes trace to named philosophical traditions |
| Relevance | High | Direct input to ethics engine 3-stage pipeline |

**Quality controls**:
- 91.1% classification accuracy on test set
- 26 adversarial inputs test edge cases
- Moral topology anomaly detection (entropy + attractor monitoring)

**Known limitations**:
- ~9% error rate on classification
- English-language bias in keyword matching

### 2.3 Safety Thresholds (`thresholds.rs`)

| Dimension | Score | Evidence |
|-----------|-------|---------|
| Accuracy | High | 119+ constants, each citing published neuroscience/psychology |
| Completeness | High | All safety-critical parameters named and documented |
| Consistency | High | `validate()` function checks ordering invariants programmatically |
| Timeliness | High | Citations range from foundational (Horn 1989) to recent (2020s) |
| Provenance | High | Author + year citation on every constant |
| Relevance | High | Direct control of cognitive loop behavior |

**Quality controls**:
- `validate()` enforces cross-constant ordering invariants
- 3 proptest suites verify stability under ±wide perturbation
- Class A change protocol for modifications (ADR required)

**Known limitations**:
- Some thresholds are engineering choices informed by (but not derived from) citations
- Single-developer review limitation

### 2.4 HDC Dictionaries

| Dimension | Score | Evidence |
|-----------|-------|---------|
| Accuracy | N/A | Random generation — no "correct" value |
| Completeness | High | Full 16,384-bit coverage |
| Consistency | High | Quasi-orthogonality guaranteed by high dimensionality (Johnson-Lindenstrauss) |
| Timeliness | N/A | Generated at runtime |
| Provenance | High | Deterministic seeding with documented algorithm |
| Relevance | High | Foundation for all HDC encoding operations |

**Quality controls**:
- xorshift64 seed-0 fix: `state = seed ^ 0x9E3779B97F4A7C15`
- Hamming distance tests verify quasi-orthogonality
- LSH cache auto-built for ≥500 records

### 2.5 Harmony Keywords

| Dimension | Score | Evidence |
|-----------|-------|---------|
| Accuracy | Medium | Keywords selected from cross-cultural ethics literature |
| Completeness | Medium | 8 harmonies covered; keyword lists manually curated |
| Consistency | High | No contradictions between harmony keyword sets |
| Timeliness | Medium | Based on established ethical traditions |
| Provenance | High | Documented in `VALUE_VERIFICATION.md` |
| Relevance | High | Input to HarmoniesIntegrator evaluation |

**Known limitations**:
- English-language bias (documented)
- Keyword matching is substring-based, not semantic

### 2.6 Substrate Profiles

| Dimension | Score | Evidence |
|-----------|-------|---------|
| Accuracy | Variable | Biological: high (empirical). Others: theoretical |
| Completeness | High | 8 substrate types with 9-dimensional profiles |
| Consistency | High | `SubstrateRequirements` struct enforces [0,1] bounds |
| Timeliness | Medium | Some profiles based on speculative physics |
| Provenance | High | Each substrate profile cites relevant literature |
| Relevance | High | Controls substrate_feasibility in consciousness equation |

**Quality controls**:
- `honest_confidence` explicitly acknowledges evidence level per substrate
- `feasibility_gap()` measures divergence between theoretical feasibility and evidence
- 37 substrate tests + 6 proptests

**Known limitations**:
- Non-biological substrate profiles are theoretical (honest_confidence: 0.10 for silicon)
- Exotic substrates (plasma, BZ reactions) are speculative

---

## 3. Data Quality Monitoring

### 3.1 Automated monitoring

| Monitor | Frequency | Trigger |
|---------|-----------|---------|
| `validate()` in thresholds.rs | Every CI run | Ordering invariant violation |
| Psych-bench regression | Weekly | Normative drift > 2σ |
| CalibrationHistory | Per calibration | Systematic drift > 75% same-direction |
| SelfAssessmentMonitor | Continuous (50Hz) | Drift > 1σ from baseline |
| Moral topology entropy | Per ethics cycle | Entropy outside expected bounds |
| Compliance dashboard | Every CI push | Any test failure |

### 3.2 Manual review

| Activity | Frequency | Scope |
|----------|-----------|-------|
| Threshold citation audit | Quarterly | Verify citations still represent scientific consensus |
| Moral prototype review | Annually | Cross-cultural coverage assessment |
| Normative baseline review | Annually | Check for updated normative data |
| Substrate profile review | Annually | Update based on new experimental evidence |

---

## 4. Non-Conformance Handling

When data quality issues are detected:

1. **Classify severity**: Critical (safety impact) / Major (accuracy impact) / Minor (documentation)
2. **Immediate action**: Critical → Class A change protocol; Major → Class B; Minor → Class C
3. **Root cause analysis**: Document in ADR if systemic
4. **Corrective action**: Update data source, tests, and quality assessment
5. **Verification**: Re-run relevant test suite to confirm resolution

---

## 5. Data Quality Metrics Summary

| Source | Overall Quality | Confidence | Monitoring |
|--------|----------------|-----------|-----------|
| Psych-bench normative | High | Validated (0.95) | Weekly regression + EMA tracking |
| Moral prototypes | High | Validated (0.91 accuracy) | 54 tests + topology monitoring |
| Safety thresholds | High | Validated (119 cited) | CI invariant checks + proptests |
| HDC dictionaries | High | By construction | Runtime orthogonality checks |
| Harmony keywords | Medium | Expert-curated | Value verification protocol |
| Substrate profiles | Variable | Theoretical (0.10-0.95) | honest_confidence + feasibility_gap |

---

*Review annually or when significant data source changes occur.*
