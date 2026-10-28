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
    cargo run --quiet --manifest-path tools/active-evidence-exact-tree/Cargo.toml --bin active-evidence-exact-tree > /tmp/active-evidence.csv
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


## Follow-up: exact finite-horizon robust policy-tree optimizer

A second Rust binary, src/bin/robust_tree.rs, enumerates complete observation-contingent acquisition policy trees rather than choosing each query by a greedy one-step score. For three query rounds and four available queries, it enumerates 3,137 candidate trees, then chooses one fixed tree minimizing the worst expected terminal decision loss over the four declared ambiguity models. Equal worst-case loss is broken by lower worst-case expected query cost. The same observation-contingent tree is evaluated under every model; the adversarial model is not allowed to switch independently at every branch.

Commands:

    cargo test --manifest-path tools/active-evidence-exact-tree/Cargo.toml
    cargo run --quiet --manifest-path tools/active-evidence-exact-tree/Cargo.toml --bin robust_tree > /tmp/robust-tree.csv
    python3 tools/active-evidence-exact-tree/reference_check_robust_tree.py --self-test
    python3 tools/active-evidence-exact-tree/reference_check_robust_tree.py /tmp/robust-tree.csv

Independent Python reference results at query cap 3:
- Greedy robust EVSI: worst regret across the four ambiguity models = 1.7780; IID regret = 1.1480 at expected cost 1.000.
- Finite-horizon robust tree: worst regret across the same four models = 1.6090; IID regret = 1.0752 at expected cost 2.632.
- Greedy plug-in EVSI: IID regret = 1.0916 at expected cost 3.8737.
- Under the out-of-set channel flip, the finite-horizon tree still has regret 3.5133, above the no-query baseline 2.3000.

The result is a promising algorithmic hypothesis, not a superiority claim. The optimizer is exact only for this finite 3-hypothesis/4-query fixture and four-model ambiguity set. It minimizes worst-case loss first, then expected cost; it is not constrained to a fixed cumulative-cost budget. The comparison therefore reports both query cap and realized expected costs, and must not be described as a matched-cost SOTA comparison. The selected tree still uses nominal posterior updates and nominal terminal actions, so it is not a fully robust decision/inference system.

The Rust binary and its Rust tests have not yet been compiled or run in this environment because Cargo/rustc/rustfmt are unavailable. The independent Python prototype exhaustively evaluated the same finite problem and reproduced the listed reference witnesses; the Rust-versus-Python CSV check remains pending exact-source execution.


## Follow-up: cost-constrained minimax frontier

The robust-tree binary now emits two views:
1. a depth/query-cap view (the best worst-case tree for depth 0–3);
2. an expected-cost-constrained view over budgets 0, 0.5, 1, 1.5, 2, 2.5, 2.632, 2.75, and 3 cost units.

For the cost-constrained view, a candidate is feasible only if its **expected query cost is at or below the budget under every declared ambiguity model**. Among feasible plans, selection minimizes worst-case expected terminal loss; ties prefer lower worst-case expected cost. This is a bound on expectation under the finite model set, not a per-trajectory hard cap.

Independent Python reference results for selected budgets:
| Maximum expected cost across ambiguity models | Worst-case expected regret |
|---:|---:|
| 0.0 | 2.300000 |
| 1.0 | 1.778000 |
| 1.5 | 1.735898 |
| 2.0 | 1.706826 |
| 2.5 | 1.648986 |
| 2.632 | 1.609023 |

This provides a more informative frontier than the prior risk-first tie-break, but it still does not establish a win over entropy/diversity/SOTA at matched budgets. The next fair comparison is to take the best baseline result feasible under the same **worst-case expected-cost cap**, report the full risk-cost frontier and its dominated segments, then expand to task/source and shift-family replication.

The robust_tree.rs output schema is now mode,limit,scenario,...; the independent checker also verifies the additional worst_case_ambiguity_cost field and rejects a mutated row.


### Matched expected-cost comparison against the implemented greedy arms

Using the same four ambiguity scenarios and selecting each greedy arm's best available query cap subject to its maximum expected cost over those scenarios staying within the budget gives this preliminary frontier:

| Max expected-cost budget | Best implemented greedy arm | Greedy worst-case regret | Exact-tree worst-case regret | Exact-tree max expected cost |
|---:|---|---:|---:|---:|
| 1.0 | Entropy / plug-in EVSI / robust greedy | 1.778000 | 1.778000 | 1.000 |
| 1.5 | Entropy / plug-in EVSI / robust greedy | 1.778000 | 1.735898 | 1.483 |
| 2.0 | Entropy / plug-in EVSI / robust greedy | 1.778000 | 1.706826 | 1.621 |
| 2.5 | Entropy / plug-in EVSI / robust greedy | 1.778000 | 1.648986 | 2.457 |
| 2.632 | Entropy / plug-in EVSI / robust greedy | 1.778000 | 1.609023 | 2.632 |
| 3.0 | Entropy | 1.735898 | 1.609023 | 2.632 |

This is a **matched worst-case expected-cost comparison only against the small set of implemented arms** (random, entropy, plug-in EVSI, and one-step robust EVSI), over the same four hand-defined ambiguity scenarios. The best greedy arm is chosen among the four based on these same scenarios, so this is exploratory rather than a sealed held-out comparison. It does not include Uncertainty Herding or R-IDeA, does not satisfy a per-trajectory hard cost cap, and does not establish SOTA superiority. The key testable hypothesis is narrower: explicit multi-step planning can improve the worst-case decision-regret/cost frontier versus one-step greedy selection in this toy model.


The cross-policy matched-budget report can be regenerated after both CSVs are produced:

    python3 tools/active-evidence-exact-tree/compare_cost_frontiers.py /tmp/active-evidence.csv /tmp/robust-tree.csv

The comparator checks completeness of both input inventories, excludes the known-channel reference from deployable-baseline selection, computes the best feasible implemented greedy arm at each shared worst-case expected-cost budget, checks the tree's stated risk/cost aggregates against its scenario rows, and enforces frozen witness values. It does not treat the hindsight/known-channel arm as a deployable method or claim external validity.


When the dedicated GitHub Actions qualification workflow completes, it uploads an exact-head artifact containing both deterministic result CSVs, the matched-cost frontier CSV, and a provenance receipt with the source commit/tree, toolchain versions, runner identity, and SHA-256 digests. This is designed to make the generated benchmark report inspectable without assuming the result of any queued workflow.


## Follow-up: ambiguity set derived from calibration uncertainty

A third Rust binary, src/bin/calibrated_box.rs, replaces the hand-coded ambiguity set with simultaneous confidence intervals for the 12 binary channel parameters (4 queries × 3 true hypotheses). For each calibration size n, the fixture sets the synthetic success count to round(n × nominal_probability) and uses a two-sided Hoeffding interval with familywise alpha 0.05 allocated over 12 cells:

    radius = sqrt(log(2 * 12 / 0.05) / (2 * n))

**Important distinction:** these are deterministic synthetic calibration counts centered around the nominal probabilities. They are a sample-size / interval-width sensitivity experiment, not observed sensor calibration and not an empirical coverage estimate. Under independent fixed-sample Bernoulli calibration for each cell and stationarity, Hoeffding plus the union bound gives at least 95% simultaneous coverage; deployment shift can still put the true channel outside the box.

### Exact optimization over the calibration box

For a fixed query tree, conditional trajectory risk for a given true hypothesis h is multi-affine in the four channel probabilities associated with that h, provided a query is never repeated on a path. Its maximum over the row's four-dimensional interval box therefore occurs at one of 16 endpoint combinations. Since the uncertainty box factorizes by hypothesis, worst-case expected loss and expected cost can be computed by maximizing each of the three class-conditional rows separately: 48 row-vertex evaluations per tree, rather than a full 2^12 = 4,096 matrix-vertex sweep. A Rust regression test and an independent Python self-test compare the factorized maximum with exhaustive 4,096-vertex enumeration for a shallow tree. This shortcut is specific to a product box and no-query-repeat trees; correlated confidence regions require a different treatment.

### Exploratory reference values

The standard-library Python prototype selected these trees over the fixed calibration-size sweep:

| Synthetic calibration observations per query/hypothesis cell | Confidence radius | Worst-case expected loss over box | Worst-case expected query cost | Policy outcome |
|---:|---:|---:|---:|---|
| 20 | 0.39287 | 2.30000 | 0.000 | Abstain / no query |
| 50 | 0.24847 | 2.26012 | 1.000 | One query |
| 100 | 0.17570 | 1.93863 | 1.000 | One query |
| 200 | 0.12424 | 1.70706 | 1.000 | One query |
| 500 | 0.07857 | 1.50158 | 1.000 | One query |
| 1,000 | 0.05556 | 1.38936 | 3.188 | Deeper tree |

For n = 20 the interval is so broad that even the selected full tree is no better in worst-case risk than stopping. At n = 50–500, the robust optimum is a one-query tree. At n = 1,000 it can justify a deeper tree. This is the behavior the calibration-sized experiment is meant to expose; it does not imply these are minimum sample sizes in any real application.

The calibrated-box policy also remains vulnerable outside its confidence set. In the deliberately out-of-box channel-flip control, expected regret is 3.77 for n = 50–500, and 2.553 at n = 1,000, versus 2.300 with no query. Do not interpret the interval guarantee as deployment-shift protection.

### Run and qualification

    cargo run --locked --quiet --manifest-path tools/active-evidence-exact-tree/Cargo.toml --bin calibrated_box > /tmp/calibrated-box.csv
    python3 tools/active-evidence-exact-tree/reference_check_calibrated_box.py --self-test
    python3 tools/active-evidence-exact-tree/reference_check_calibrated_box.py /tmp/calibrated-box.csv

The dedicated workflow now includes this binary, deterministic output comparison, independent row-by-row reference verification, a mutated-row negative control, and the generated CSV in the exact-head artifact/receipt. The Rust binary and checker have not been compiled/executed together in this environment; these reference values are from a separate local Python prototype pending exact-head CI.
