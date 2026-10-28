# Risk-Aware Memory Escalation — synthetic run

**Protocol:** `BROCA-RISK-ESCALATION-V1`  
**Run date:** 2026-10-28  
**Classification:** synthetic research only; no LLM, Symthaea runtime, live API or external benchmark was executed.

## Run and verification

- 3,000 development cases split by complete groups into 1,500 training and 1,500 calibration cases.
- Five held-out regimes × 2,000 cases = 10,000 test cases.
- 15 policies, including controls, six calibrated risk budgets and one evaluator-only oracle.
- 150,000 per-case/policy outcome rows.
- 14 automated tests passed.
- Independent streaming verifier passed row/matrix checks, raw output digest, config digest, source digest and recomputed summary aggregates.
- Negative controls reject digest tampering, dropped policy rows and outcome tampering even when altered artifact and summary digests are recomputed.

## IID results

All rates here come from the hand-authored synthetic generator and do not describe real Broca behavior.

| Policy | Task success | Harmful answers | Assumed tokens/query | Assumed net utility |
|---|---:|---:|---:|---:|
| Always answer | 54.85% | 45.15% | 450 | -0.4355 |
| Evidence threshold | 59.75% | 3.05% | 272 | +0.1623 |
| Point-risk-budgeted | 35.70% | 0.85% | 264 | -0.1901 |
| Calibrated 5% risk control | 70.65% | 2.80% | 691 | +0.2648 |
| Calibrated 7.5% risk control | 79.30% | 6.50% | 747 | +0.3413 |
| Learned expected utility | 90.90% | 9.10% | 883 | +0.4660 |
| Oracle diagnostic (not deployable) | 94.15% | 0.00% | 536 | +0.7079 |

The 5% calibrated policy improved task success by **10.90 percentage points** over evidence-threshold gating (95% episode-group bootstrap CI **[+9.30, +12.55] pp**) and utility by **+0.1024** (CI **[+0.0664, +0.1376]**). It used about 419 more tokens per query, so this is not a matched-cost win.

## Distribution shift

| Test regime | Calibrated 5% task success | Harmful rate | Mean tokens | Evidence-threshold success | Evidence-threshold harmful rate |
|---|---:|---:|---:|---:|---:|
| IID | 70.65% | 2.80% | 691 | 59.75% | 3.05% |
| High-similarity distractors | 52.95% | 2.45% | 525 | 44.65% | 2.15% |
| Stale revisions | 64.90% | 3.90% | 657 | 44.30% | 1.35% |
| Scope shift | 54.10% | 1.95% | 532 | 45.65% | 3.60% |
| Compound shift | 51.60% | 2.80% | 507 | 37.40% | 8.05% |

The calibrated 5% policy improved task success across the five generated regimes, but its harm rate was 2.55 pp higher than evidence gating on stale revisions and 0.30 pp higher on high-similarity distractors. It reduced harm on scope and compound shift. Governance outcomes remain a separate promotion gate.

The more permissive 7.5% policy reached 79.30% IID task success and 6.50% harmful rate; harm rose to 8.60% under high-similarity distractors, 10.95% under scope shift, and 11.25% under compound shift. A calibration target is not an out-of-distribution guarantee.

## Calibration failure is itself a result

With 1,500 calibration cases and the current risk score, the 1%, 2% and 3.5% budgets admitted **no action subset** meeting the selected Wilson upper-bound criterion; the corresponding policies fell back to always-abstain. The 5% budget admitted 966 escalation cases with 3.62% observed calibration harm and a 5.00% Wilson upper endpoint.

This is evidence that the calibration set/model combination is too data-inefficient for very strict thresholds—not a reason to lower the gate silently.

## Limits

- Features, latent classes, correctness probabilities, utility and token costs are set by the synthetic generator.
- The learner is a light ridge model, not an LLM or Symthaea/HDC memory implementation.
- Confidence intervals quantify synthetic episode-group variation only.
- The oracle sees evaluator-only outcomes.
- No real-world generalization, measured token savings, latency gain or SOTA improvement is claimed.

## Artifact pins

- Raw outcome JSONL SHA-256: `f8b2aaee93569621d316decdc94b04ba7a0281b1fd1f7de6dd3c3d541c40f2be` (local-only, 150,000 rows).
- Configuration SHA-256: `3f1ac8ba5c80885d5b744e8dfe354f52006db2c5e8a0d7486638af37d5483de7`.
- Corrected local source SHA-256: `ec0bf7a910e54dd22c34cf65f284509482307bfaf6233c8226d63cbb74c94ee0`.
- Local machine-readable summary, report, receipt and frontier CSVs are packaged with the full artifact bundle.
