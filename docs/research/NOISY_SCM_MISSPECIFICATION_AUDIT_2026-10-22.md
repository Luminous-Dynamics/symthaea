# Out-of-Class Noisy SCM Audit — 2026-10-22

**Status:** exploratory exact-enumeration falsification fixture. This is not a general causal-discovery benchmark, a direct ACDB/CausalArena comparison, or evidence of SOTA.

## Why this experiment exists

The existing noisy active-probe pilot generates outcomes and computes the learner's Bayesian updates inside the same finite candidate-SCM class and noise model. Its exact oracle is therefore exact relative to that assumed class. It does not establish that entropy reduction remains meaningful when the true graph or noise rate is outside the learner's assumptions.

This audit separates the data-generating SCM from the candidate models used for intervention selection and posterior updates. The true system is an XOR collider, \(X_0 \to X_2 \leftarrow X_1\). The learner is only offered a chain and a fork; the true graph is explicitly absent from the candidate class. Hard interventions are applied to each node/value pair, with two interventions per run. Outcome probabilities are enumerated exactly under the true SCM; posterior updates and policy decisions use the learner's assumed models and noise rate.

## Policies and conditions

Policies: fixed order, seeded random order (eight deterministic priority schedules averaged), greedy expected information gain, and a two-step dynamic-programming oracle **under the learner's assumed model**. The oracle is not omniscient about the true graph.

Actual mechanism-noise rates: 0.05, 0.20, 0.35. Assumed learner noise rates: 0.10 and 0.25. The report retains separate values for residual graph entropy, mean MAP-candidate-graph posterior, cumulative prequential log loss (bits over two observations), and the fraction of exact outcome paths ending with MAP candidate-graph posterior at least 0.9 despite the true graph being out of support.

## Preliminary result

A useful counterexample occurs at actual noise 0.05 and assumed noise 0.10:

| Policy | Expected residual graph entropy (bits; lower looks better) | Mean MAP candidate-graph posterior | Expected cumulative prequential log loss (bits; lower is better) | False high-confidence path rate |
|---|---:|---:|---:|---:|
| Fixed order | 0.5154 | 0.8642 | 4.9872 | 90.71% |
| Random | 0.7389 | 0.6963 | 2.1262 | 47.17% |
| Greedy EIG | 0.4805 | 0.8830 | 2.0501 | 90.29% |
| Assumed-model oracle | 0.4805 | 0.8830 | 2.0501 | 94.55% |

This is an adversarial fixture, not a policy ranking to generalize. The key observation is definitional: the model-based oracle lowers its own expected posterior entropy, but in 94.55% of the exact truth-world paths it ends at >= 0.9 posterior for one of the wrong candidate graphs. The true graph has zero posterior support by construction. Entropy reduction is not a graph-correctness metric under candidate-set misspecification.

At actual noise 0.20 / assumed noise 0.10, false high-confidence path rates remain 74.88% for the assumed-model oracle and 66.08% for greedy EIG. When the learner assumes noise 0.25, those rates are much lower in this particular fixture, showing that noise misspecification materially changes calibration. Do not interpret this as a generally optimal conservative noise setting.

The difference between policies' expected graph entropy and their prequential log loss is itself useful: acquisition should be judged against future predictive validity, not only the uncertainty measure the acquisition policy optimizes.

## Exact reproduction and provenance

Run from the repository root:

~~~sh
python3 validation/noisy_scm_misspecification_audit.py --self-test
python3 -m unittest -v validation.test_noisy_scm_misspecification_audit
python3 validation/noisy_scm_misspecification_audit.py \
  --source-sha da35147a8b3fb93bda12b2fa4e459fff720a57bc \
  --output validation/NOISY_SCM_MISSPECIFICATION_AUDIT_2026-10-22.json
~~~

Exact executed source matches the Git blob da35147a8b3fb93bda12b2fa4e459fff720a57bc. Exact test source matches Git blob dd9004554561dea80524ce51627f32f817bae82e. The embedded self-test passed and all four committed unit tests passed locally against those exact bytes. The generated JSON was 9,250 bytes, SHA-256 552cff7619714f7abf5896a118c706ccb2125e19b6ece39ffa6161bbd341bd2e. This is a local exact-source execution receipt; it is not a GitHub Actions result and does not qualify the older benchmark source whose own source-byte mismatch remains unresolved.

## Limitations and next gates

This is only a three-node hand-authored system, two candidate graphs, one held-out XOR-collider graph, two interventions, and two assumed-noise settings. The eight random schedules are not an inferential confidence interval. There is no latent-confounder generator, intervention failure, adaptive noise estimation, large-scale data, real-task utility, external benchmark adapter, or comparison under ACDB's scoring contract. The fixture has been chosen to falsify a strong interpretation, not to estimate typical performance.

Next qualification steps:

1. Extend to seeded held-out topology **and** mechanism families, with explicit truth-out-of-support cases and graph-recovery metrics (skeleton/direction precision-recall and SHD).
2. Add a rejection/abstention decision based on prequential log score and posterior predictive checks; score coverage and selective risk, not only posterior sharpness.
3. Treat true noise, inferred noise, and learner-assumed noise as separate fields. Add off-target/failed interventions and missing-variable / latent-confounding stressors.
4. Compare all policies on paired seeds and identical actions/budgets against the known-truth oracle, the assumed-model oracle, random and low-cost baselines; report paired uncertainty intervals and cost/error frontiers.
5. Implement a contract-matched ACDB lane and family-held-out CausalArena checks before publishing any external ranking.

Prior-art context: [ACDB (September 2026)](https://arxiv.org/abs/2609.31675) evaluates budgeted interventional graph recovery; [CausalArena (September 2026)](https://arxiv.org/abs/2609.11897) emphasizes mechanism-family and protocol shifts; [TCD-Arena (ICLR 2026)](https://tcd-arena.github.io/) stresses time-series assumption violations. The present fixture is not numerically comparable with those benchmarks.
