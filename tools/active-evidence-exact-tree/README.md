# Active-evidence exact-tree microbenchmark

**Status:** synthetic research fixture, version 0.1. No production selector, real-data result, or state-of-the-art claim.

## Purpose

This small, dependency-free Rust program exhaustively evaluates a sequential evidence-acquisition decision tree. It is a local testbed for failure modes the larger v2.0 benchmark must test:

- information gain can differ from downstream decision value;
- acquisition policies can behave differently across measurement-model shifts;
- a one-step robust acquisition rule can stop early and pay a robustness/conservatism tax;
- an ambiguity-set guarantee is conditional on that set and does not transport to an explicitly out-of-set control.

The latent hypothesis prior and loss matrix remain fixed. The shift axes here alter measurement outcome probabilities, not class priors or the task distribution. It is a deliberately narrow mechanism-isolation fixture.

## Run

From the repository root:

    cargo test --manifest-path tools/active-evidence-exact-tree/Cargo.toml
    cargo run --quiet --manifest-path tools/active-evidence-exact-tree/Cargo.toml > /tmp/active-evidence.csv
    python3 tools/active-evidence-exact-tree/reference_check.py /tmp/active-evidence.csv

No third-party Rust crates or Python packages are required. The exact Rust results have not been qualified until those commands run on the checked-out source. The independent Python checker is included to compare the emitted CSV against a separate reference calculation.

Filter to one regime using --scenario=iid, --scenario=compound_shift, or another name printed by --help.

## Policies and meaning

- random: uniform random choice over remaining queries; the evaluator integrates over the random choice exactly rather than sampling seeds.
- entropy: greedy expected entropy reduction per declared query cost.
- plugin_evsi: greedy one-step expected downstream decision-risk reduction per cost under nominal likelihoods.
- robust_evsi: greedy one-step risk reduction under the worst model in a frozen, finite ambiguity set; posterior updates still use nominal likelihoods. This is intentionally **not** a robust posterior or a robust finite-horizon planner.
- known_channel_evsi_reference: the same greedy EVSI objective with the true scenario likelihoods supplied. This is a diagnostic reference, not a global horizon oracle, and is not a deployable baseline.

The evaluator integrates over all hypotheses and possible binary outcome sequences up to three queries. Its output is exact for this finite synthetic model; it has no Monte Carlo standard error. This exactness does not imply external validity.

## Frozen fixture

Hypotheses have prior [0.6, 0.3, 0.1]. Losses (rows = true hypothesis; columns = terminal decision) are:

| True hypothesis | Decision 0 | Decision 1 | Decision 2 |
|---|---:|---:|---:|
| 0 | 0 | 4 | 7 |
| 1 | 5 | 0 | 3 |
| 2 | 8 | 2 | 0 |

Query costs are [1, 1, 2, 1]. The binary-channel likelihood table stores P(outcome=1 | hypothesis, query). The nominal table and each shifted table are frozen as constants in src/main.rs. The robust ambiguity set contains IID, the q0 conditional shift, the q1 asymmetric-error shift, and the compound shift. out_of_set_flip is deliberately excluded.

Regret is terminal decision loss minus the minimum loss available with the true hypothesis known. Expected query cost and count are reported separately; cost is not silently folded into the decision-loss matrix.

## Independent reference findings

The standard-library Python reference calculation, executed locally before Rust compilation was available, produced the following preliminary values. The reference_check.py script must reproduce every emitted Rust row before these numbers are treated as Rust-qualified:

| Scenario | Policy | Query cap | Expected regret | Expected query cost |
|---|---|---:|---:|---:|
| IID | No query (all policies) | 0 | 2.3000 | 0.000 |
| IID | Random | 3 | 1.1708 | 3.750 |
| IID | Entropy | 3 | 1.1218 | 3.000 |
| IID | Plug-in EVSI | 3 | 1.0916 | 3.874 |
| IID | Robust EVSI | 3 | 1.1480 | 1.000 |
| IID | Known-channel EVSI reference | 3 | 1.0916 | 3.874 |
| Compound shift | Random | 3 | 2.2000 | 3.750 |
| Compound shift | Entropy | 3 | 1.7050 | 3.000 |
| Compound shift | Plug-in EVSI | 3 | 1.9718 | 3.872 |
| Compound shift | Robust EVSI | 3 | 1.7480 | 1.000 |
| Out-of-set flip | No query (all policies) | 0 | 2.3000 | 0.000 |
| Out-of-set flip | Entropy | 1 | 3.7700 | 1.000 |
| Out-of-set flip | Plug-in EVSI | 1 | 3.7700 | 1.000 |
| Out-of-set flip | Robust EVSI | 1 | 3.7700 | 1.000 |

### What the pilot says—and does not say

1. Under IID conditions, this particular one-step robust policy stops after one query, while plug-in EVSI continues. Its lower acquisition cost comes with somewhat higher terminal regret. That is a concrete robustness-tax / premature-stopping failure mode to investigate, not evidence robust selection is generally too conservative.
2. Under compound shift, the robust policy has lower regret than plug-in EVSI in this finite fixture, but it does not dominate entropy sampling. The preferred simple baseline changes by regime.
3. Under the out-of-set flip, the robust policy can be harmed just like the nominal policies because the actual channel is outside the declared ambiguity set. This is an explicit counterexample to any claim of arbitrary-shift protection.
4. A better next candidate is not simply a wider ambiguity set. It is a **shift-aware sequential policy** that has a separately measured model-adequacy signal, re-calibrates its ambiguity set from permitted evidence, and may abstain when the signal is insufficient. Its false-alarm and missed-shift costs must be measured.

These are analytical results for five hand-defined measurement matrices and one prior/loss model. They are not independent replications, real sensor calibration, empirical deployment performance, or SOTA comparisons.

## Hard limits

- Three hypotheses, four binary queries, a fixed prior and a fixed loss matrix.
- Only conditional measurement-channel shifts are modeled.
- The robust ambiguity set is handcrafted and small.
- Policies are greedy one-step selectors; no globally optimal finite-horizon acquisition algorithm is implemented.
- There is no training model, external dataset, human annotation workflow, statistical inference over a task population, or production Symthaea integration.
