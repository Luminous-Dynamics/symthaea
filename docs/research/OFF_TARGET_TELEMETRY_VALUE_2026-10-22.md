# Value of Intervention-Realization Telemetry — 2026-10-22

**Status:** exact local enumeration; embedded self-test and 6/6 unit tests passed. This is a matched-budget small-SCM pilot, not an external benchmark or SOTA result.

## Research question

After making intervention reliability a latent variable, should a causal-discovery system infer actuation from downstream outcomes alone, or directly observe which variables were actually affected? This experiment compares outcome-only learning with a noisy spill sensor and an ideal sensor exposing the realized target subset.

The ideal sensor is an upper bound, not an assumption that perfect intervention telemetry is readily available in practice.

## Protocol

Implementation: validation/off_target_telemetry_value.py; tests: validation/test_off_target_telemetry_value.py.

- Candidate SCMs: three-node Boolean chain and fork.
- Latent intervention parameters: target compliance q ∈ {0.50, 0.75, 1.00}, and conditional collateral-effect probability r ∈ {0.00, 0.25}.
- The action kernel has three possible realized target sets: no variables clamped; requested target only; or requested target plus its cyclic successor. This is deliberately narrower than general action-dependent target-set models.
- Four true regimes: clean q=.75/r=0 at noise .10; off-target q=.75/r=.25 at noise .10; weak q=.50/r=.25 at noise .10; and noisy q=.75/r=.25 at noise .25.
- Two out-of-class worlds: collider XOR and reversed chain.
- Observation channels: outcome vector only; outcome plus a noisy spill flag (90% sensitivity and specificity); or outcome plus perfect realized-set telemetry.
- Acquisition policies: graph EIG and a balanced graph + compliance + spill-information objective. All policies share the same two-distinct-intervention budget.
- Full grid: 4 regimes × 2 held-out worlds × 3 observation channels × 2 acquisition policies = 48 exact rows.

For each channel, the likelihood is the exact joint distribution of downstream outcome and the extra telemetry. Bayesian updating marginalizes over the declared graph × q × r hypothesis space. Adaptive paths are exhaustively enumerated.

## Selected results

AUC 0.50 is chance-level ranking of held-out-world trajectories against the equal-prior in-class chain/fork trajectories. Information quantities are expected entropy reductions, averaged over the two held-out worlds per regime/policy.

### Graph-EIG acquisition

| Regime | Observation channel | OOD AUC | Graph information (bits) | Compliance information (bits) | Spill information (bits) |
|---|---|---:|---:|---:|---:|
| Clean, noise .10 | Outcome only | .8796 | .3300 | .3142 | .0093 |
| Clean, noise .10 | Noisy spill sensor | .8424 | .3766 | .2479 | .0466 |
| Clean, noise .10 | Perfect target-set telemetry | .8271 | .4024 | .3877 | .0378 |
| Matched off-target, noise .10 | Outcome only | .8519 | .3019 | .3146 | .0088 |
| Matched off-target, noise .10 | Noisy spill sensor | .8342 | .3032 | .2715 | .1156 |
| Matched off-target, noise .10 | Perfect target-set telemetry | .7689 | .3114 | .3877 | .3621 |
| Weak off-target, noise .10 | Outcome only | .7727 | .2324 | .4752 | .0062 |
| Weak off-target, noise .10 | Noisy spill sensor | .7615 | .2350 | .4285 | .0822 |
| Weak off-target, noise .10 | Perfect target-set telemetry | .6650 | .2414 | .5640 | .2480 |
| Noisy off-target, noise .25 | Outcome only | .6784 | .1238 | .2277 | .0089 |
| Noisy off-target, noise .25 | Noisy spill sensor | .6470 | .1231 | .2345 | .1148 |
| Noisy off-target, noise .25 | Perfect target-set telemetry | .5843 | .1253 | .3877 | .3621 |

The balanced policy likewise increases q/r information but can reduce the surprisal AUC more strongly in telemetry-enabled conditions. The full artifact contains every row and both policies.

## Interpretation

1. **A spill sensor exposes what outcome-only data barely identifies.** In the matched off-target regime, r-information increases from .0088 bits to .1156 with the noisy sensor and .3621 with ideal target-set telemetry under graph-EIG. In the noisy regime it rises from .0089 to .1148 and .3621, respectively.
2. **Better diagnosis of actuation does not maximize a graph-OOD score.** Surprise AUC falls when telemetry is available. This need not be a negative engineering outcome: telemetry lets the learner explain surprising downstream outcomes through the action channel rather than wrongly attributing them to a novel causal graph. OOD graph detection and actuator diagnosis must be scored separately.
3. **The sensor trades graph information against action-model information.** Perfect target-set telemetry adds some graph information but much more q/r information in off-target regimes. Under two probes, it does not remove graph uncertainty; conditional graph MAP error remains material in these tiny SCMs.
4. **The deployment design should be instrumentation-aware.** A direct measurement of the realized intervention target set may be more cost-effective than repeating interventions to infer a hidden action kernel. Its value depends on sensor error, cost, and the scientific objective. A real benchmark must price the sensor and test imperfect telemetry under distribution shift.

## Paradigm shift

Represent every causal experiment as two linked channels: the intended intervention and an uncertain realization channel. Whenever possible, measure or independently instrument the realized target set. Then assess graph recovery, intervention-kernel calibration, and model-set falsification separately. A confident graph posterior is not enough; a high surprisal is not itself proof that the graph is wrong.

## External research and comparison

- **Causal Discovery under Off-Target Interventions (AISTATS 2024)** models an attempted action as drawing an affected-variable subset from an action-dependent distribution. This is more general than our three-subset cyclic kernel and is an important comparator: https://proceedings.mlr.press/v238/choo24a.html
- **MetaCaDI (UAI 2026)** addresses unknown intervention-target discovery from multiple environments with few-shot adaptation: https://proceedings.mlr.press/v337/ong26a.html
- **SCOUT (ICML 2026)** addresses cyclic causal discovery under soft interventions with unknown targets: https://proceedings.mlr.press/v306/turkoglu26a.html
- **ACDB (September 2026)** separates graph skeleton recovery, directed recovery and intervention efficiency under its own perfect-intervention linear-Gaussian protocol. Its scores cannot be numerically compared to this experiment: https://arxiv.org/abs/2609.31675
- **CausalArena (September 2026)** evaluates a broader set of synthetic, semantic-operational and formula-grounded SCMs with cross-family ranking analysis: https://arxiv.org/abs/2609.11897

No external baseline was run. This is not a leaderboard comparison or a claim of SOTA.

## Proposed qualification gates (not achieved)

- Compare outcome-only, noisy spill sensors with multiple error rates, perfect realized-set telemetry, and repeated probes at the same total data/action cost.
- Replace cyclic spill with action-dependent distributions over target subsets on small graphs; then include hidden target, multiple off-target effects, delay and dose.
- Report graph SHD, skeleton/directed precision-recall, q/r/target-set log score or Brier score, OOD AUROC/AUPRC, calibrated false alarms, detection power, and cost-normalized information gain.
- Use 1,000+ seeded SCMs with fully held-out topology, mechanism and actuation-kernel families, separate calibration worlds and paired intervals clustered by SCM.
- A claimed improvement must preserve validity/calibration endpoints while improving a preregistered primary endpoint at equal overall experiment cost.

## Reproduction and provenance

    python3 validation/off_target_telemetry_value.py --self-test
    python3 -m unittest -v validation/test_off_target_telemetry_value.py
    python3 validation/off_target_telemetry_value.py --source-git-blob 0f811ee1264816b4afa159285bdfa834f5ad0fbb --output /tmp/off-target-telemetry.json

- Implementation Git blob: 0f811ee1264816b4afa159285bdfa834f5ad0fbb; SHA-256 c6b6140b25636e7dc9f364b7b0db07527632590b05bee2a01c34d602f8ab3ff6; 12,545 bytes.
- Test Git blob: 3e734c85ba273c1f3ed78aac62eec573444f01a8; SHA-256 5e63188b16363a692c71aee35ee59a63d22efc33ccce4ab4c93a9009f082c3ce; six tests.
- Embedded self-test and 6/6 tests passed locally against these exact source/test blobs.
- Full exact result: 23,589 bytes; SHA-256 e601f4683e3335b4aa978ca8a5d1997c95558a7ea3006cdffcd8714be2dca79d; artifact binds the implementation blob above.
- Local results are independent of GitHub Actions; no remote pass is inferred.
