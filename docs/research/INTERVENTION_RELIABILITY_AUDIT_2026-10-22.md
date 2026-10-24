# Intervention Reliability Audit — 2026-10-22

**Status:** exact local reference experiment; embedded self-test passed and 10/10 unit tests passed locally. This is a falsification-oriented diagnostic pilot, not an ACDB/CausalArena reproduction, not a qualified discovery system, and not an SOTA claim.

## Research question

Causal-discovery systems often score an action as if the requested intervention is exactly realized. In real experiments, interventions can fail, be weak, or produce off-target effects. Does a perfect-hard-do learner confuse noncompliance with an impossible observation, and does optimizing information within a restricted candidate set also produce useful out-of-distribution (OOD) falsification?

This experiment isolates compliance as one failure mode. Mechanism noise is known to the learner in this audit; the separate selective-discovery pilot tests noise misspecification. The candidate support is a chain and fork. The out-of-class world is an XOR collider not present in that candidate support.

## Exact model and protocol

Implementation: validation/intervention_reliability_audit.py; tests: validation/test_intervention_reliability_audit.py.

- Three binary variables, two adaptive intervention trials, and exhaustive enumeration of all possible binary observations per trial.
- For an assigned action on node i, compliance probability q clamps the node to its requested value. With probability 1-q, that node follows its ordinary noisy structural mechanism. The compliance flag is latent; the realized node value is observed.
- Actual compliance q ∈ {0.50, 0.75, 1.00}; mechanism noise p ∈ {0.10, 0.25}.
- Learner arms: (a) predeclared reference sequence do(X0=0), do(X1=0) with q known; (b) greedy expected information gain (EIG) but assumes q=1; (c) greedy EIG with actual q supplied.
- OOD score: cumulative predictive surprisal, -log2 P(observation | current candidate mixture), over the adaptive path. The reported AUC is exact weighted ROC AUC separating the XOR world from the equal-prior mixture of in-class chain and fork worlds. Zero-probability predictions retain infinite surprise; they are not clipped.
- Graph-information metric: one initial bit of chain-versus-fork entropy minus expected final posterior graph entropy in an equal-prior mixture of the two in-class worlds.
- The fixed sequence is a reference control, **not** an optimized or exhaustive best-fixed-action baseline. The suite has only three hand-designed SCMs and is not a broad discovery benchmark.

## Results

AUC 0.5 is chance ranking. Zero support is the in-class probability mass on paths assigned zero probability by a learner that assumes perfect compliance.

### Mechanism noise = 0.10

| Actual q | Policy / learner assumption | Expected graph information (bits) | OOD surprise AUC | In-class zero-support mass |
|---:|---|---:|---:|---:|
| 0.50 | Reference, q-aware | 0.1030 | 0.5066 | 0.0000 |
| 0.50 | Greedy EIG, assumes q=1 | 0.2776 | 0.5169 | 0.4395 |
| 0.50 | Greedy EIG, q-aware | 0.3026 | 0.5199 | 0.0000 |
| 0.75 | Reference, q-aware | 0.1030 | 0.5005 | 0.0000 |
| 0.75 | Greedy EIG, assumes q=1 | 0.3898 | 0.5132 | 0.2249 |
| 0.75 | Greedy EIG, q-aware | 0.4024 | 0.5153 | 0.0000 |
| 1.00 | Reference, q-aware | 0.1030 | 0.5000 | 0.0000 |
| 1.00 | Greedy EIG, q=1 / q-aware | 0.5022 | 0.5137 | 0.0000 |

### Mechanism noise = 0.25

| Actual q | Policy / learner assumption | Expected graph information (bits) | OOD surprise AUC | In-class zero-support mass |
|---:|---|---:|---:|---:|
| 0.50 | Reference, q-aware | 0.0915 | 0.5079 | 0.0000 |
| 0.50 | Greedy EIG, assumes q=1 | 0.1122 | 0.5310 | 0.3984 |
| 0.50 | Greedy EIG, q-aware | 0.1358 | 0.7019 | 0.0000 |
| 0.75 | Reference, q-aware | 0.0915 | 0.5004 | 0.0000 |
| 0.75 | Greedy EIG, assumes q=1 | 0.1458 | 0.5284 | 0.2090 |
| 0.75 | Greedy EIG, q-aware | 0.1580 | 0.7171 | 0.0000 |
| 1.00 | Reference, q-aware | 0.0915 | 0.5005 | 0.0000 |
| 1.00 | Greedy EIG, q=1 / q-aware | 0.1802 | 0.5310 | 0.0000 |

## Interpretation

1. **A wrong intervention model can make valid in-class evidence impossible.** At p=0.10, assuming perfect compliance assigns zero probability to 43.95% of in-class path mass at q=0.50, and 22.49% at q=0.75. At p=0.25, those values are 39.84% and 20.90%. This is not evidence, by itself, that the graph is wrong; the actuator model is wrong.
2. **Graph-identification utility and OOD falsification are different objectives.** In these toy worlds, greedy EIG often reduces chain/fork uncertainty more than the chosen reference schedule, while surprise-AUC remains near chance for all policies at noise 0.10. These small, hand-designed cases do not establish a general policy ranking; the reference is deliberately not the best fixed baseline.
3. **Knowing compliance removes zero support, not all model uncertainty.** The q-aware learner has no in-class zero-probability path mass in this fixture. Its OOD AUC varies substantially with noise (near chance at 0.10; around 0.70 for q-aware EIG at 0.25). The suite is too small to interpret that variation as a stable trend.
4. **Use separate diagnoses, not one confidence score.** Maintain distinct state for graph uncertainty, model-set inadequacy, actuator/compliance uncertainty, and mechanism/noise shift. On impossible evidence, do not coerce a posterior; return an explicit ACTION_OR_MODEL_MISMATCH / NOT_IDENTIFIED state and request a control or alternative probe.

## Paradigm shift: interventions as uncertain instruments

Change the action interface from “request do(X=x) and trust it” to an auditable action record:

- **Intent:** requested target, value, dose, timing and constraints.
- **Realization:** measured target state and, when available, independent compliance sensor/negative control.
- **Causal response:** outcomes conditional on actual realization, not only the command sent.
- **Validity status:** confirmed, partial/soft, failed, off-target, or unknown.

Experiment design should optimize at least two axes: (a) expected reduction in graph uncertainty *within* the current support and (b) power to falsify the support or identify failed-action models. These must be reported as a Pareto surface, not silently collapsed into a single EIG score. When action compliance is unknown, include it as a nuisance parameter or use control actions to estimate it on disjoint data.

## Qualification targets (proposed, not achieved)

- Test at least 1,000 independently seeded SCMs across topology- and mechanism-held-out cells before a broad ranking claim.
- Separate perfect, weak, off-target, delayed, and unknown-target interventions; include latent confounding, hidden selection, and non-faithful mechanisms.
- Report graph skeleton F1, directed-edge F1, SHD, intervention count/cost, zero-support rate, OOD detection AUC, false-alarm rate at 5% nominal, power at multiple OOD severities, and risk–coverage curves.
- Estimate compliance from separate controls and test calibration under compliance shift. Do not calibrate and evaluate on the same worlds.
- Use paired intervals clustered by world, report failures and abstentions, and compare action policies at matched intervention and observation budgets.
- For a practical target, require at least 10% improvement on a preregistered primary metric with confidence intervals excluding zero **without** degrading the paired safety/adequacy endpoint; treat this as a proposed gate, not a result.

## Industry and research comparisons

- [ACDB (September 2026)](https://arxiv.org/abs/2609.31675) separates skeleton recovery, directed recovery, and intervention efficiency under budgeted hard interventions. Its reported linear-Gaussian results are not numerically comparable to this Boolean audit.
- [CausalArena (September 2026)](https://arxiv.org/abs/2609.11897) motivates broad topology/mechanism coverage, semantic and formula-grounded SCMs, real-world checks, and careful separation of train/test environments. Three hand-designed worlds fall far short of this breadth.
- [Falsifying Causal Graphs With Outlier Events (UAI 2026)](https://proceedings.mlr.press/v337/orchard26a.html) is a relevant reference for graph falsification with explicit false-positive control and power analysis under stated assumptions.
- [Since Faithfulness Fails (ICML 2025)](https://proceedings.mlr.press/v267/olko25a.html) reports that faithfulness violations can substantially limit discovery performance; robustness should be measured across violations rather than a single synthetic distribution.

These sources are methodological comparators, not baselines executed in this experiment. No ACDB/CausalArena score, external ranking, or SOTA claim is made.

## Reproduction and provenance

From the repository root, run:

    python3 validation/intervention_reliability_audit.py --self-test
    python3 -m unittest -v validation/test_intervention_reliability_audit.py
    python3 validation/intervention_reliability_audit.py --source-git-blob f0f0a684891707171c886737a536ef1d96174052 --output /tmp/intervention-reliability.json
    sha256sum /tmp/intervention-reliability.json

- Source Git blob: f0f0a684891707171c886737a536ef1d96174052; source SHA-256: df9414f40ac0fcdc39043548f29a96e5b596bd7e061350084195f613b499975d.
- Test Git blob: cc7a0dffa4907861a5d050a8853b2e44fcf5d2ef; test SHA-256: 5ae9a7f5cd6853ed347fe8f6e2de0ee40a336c822e9845ba3d6d7e3b2bd73033.
- Embedded self-test passed; 10/10 unit tests passed locally against the exact source/test bytes represented by those Git blobs.
- Machine-readable artifact SHA-256: 14bfad7f19b167e6ee740f564a219235a975f9df15c1b82780f0f71212fc3170, 6,630 bytes; the output binds the source blob above.
- Local validation is not a GitHub Actions pass. Remote skipped checks are not passes.
