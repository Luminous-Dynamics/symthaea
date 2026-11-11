# ACED-REPAIR-ADAPTER-001

**Status:** pinned upstream inventory and metadata qualifier; no model trials or world payload execution  
**Owner:** [Symthaea #11598](https://github.com/Luminous-Dynamics/symthaea/issues/11598)  
**Parent:** [IG-CAUSAL-PRIMITIVES-001 #11505](https://github.com/Luminous-Dynamics/symthaea/issues/11505)

## Comparative targets

The cross-domain research comparison and proposed SOTA target ladder are maintained in [INDUSTRY_BENCHMARK_CROSSWALK_V0_1.md](INDUSTRY_BENCHMARK_CROSSWALK_V0_1.md). It compares ACED, ACDB, AgentAbstain, Agentic Abstention, AuthGateBench, AuthorizationBench, AgentGovBench and DelibSim without treating their metrics as interchangeable. All Luminous numeric targets are explicitly proposed, not results.

## Research target

Integrate the official MIT-licensed [ACED-Bench](https://github.com/Rose-STL-Lab/ACED-Bench) as an external active causal investigation comparator.

Pinned upstream revision: `998dc692bfcf5574e5319d67f1cea9c170569d80`  
Paper: [ACED v2, arXiv:2610.14381](https://arxiv.org/abs/2610.14381), revision dated 8 November 2026.

ACED-Repair tests whether an agent can investigate a hidden causal system, select useful interventions under a budget, identify working levers and doses, and recommend a fix that is independently re-simulated against the true objective. Its published protocol uses 15 experiments and 32 turns per world.

The pinned inventory contains:
- **540 test worlds** across nine causal archetypes (60 each).
- **54 evaluation worlds** (six per archetype).
- Ten domains across clinical, fermentation, agronomy, aquaculture, battery, bioprocess, catalysis, datacenter, semiconductor and water treatment.

The evaluation index is a **selected subset of the test index**, not an independent disjoint dataset. The qualifier verifies every eval row is an exact file/domain/archetype/seed match in test. Do not pool these sets as 594 independent worlds or use test/eval overlap as independent statistical evidence.

## What is pinned here

The `upstream/` directory contains only:
- `repair_test_index.json`: 540 world filename/domain/archetype/seed records.
- `repair_eval_index.json`: 54 matching selected records.
- `SHA256SUMS`: the upstream hash ledger for the repository's data files.

The source manifest also pins the raw SHA-256 and Git blob identity of the two index files and checksum ledger, upstream commit, MIT license, expected distributions and split semantics. The upstream license is preserved in `ACED-UPSTREAM-LICENSE`; attribution is in `UPSTREAM_NOTICE.md`. The closed `ACED_REPAIR_INDEX_V0_1.schema.json` contract separately defines each metadata-only row (file, domain, archetype, seed); the Rust qualifier enforces split-specific counts and distribution constraints beyond that general row schema.

**No compressed world payloads, structural equations, ground-truth interventions, oracle fixes, scoring batteries, hidden roles or grader state are copied into this adapter.** The checksum ledger exposes file paths and digests, not the payload bytes.

## ACED-Query inventory and split identity

The ACED-Query metadata-only extension pins four upstream tracks from the same v2 commit:
- `core_struct`: 10 structural worlds;
- `core_decision`: 60 decision worlds;
- `guess_struct`: 30 structural worlds;
- `guess_decision`: 48 decision worlds.

The inventory records full track-relative path, Git blob SHA-1, byte size, SHA-256 from upstream `data/SHA256SUMS`, domain, task family, seed and the filename-declared graph/sample sizes. It does not copy any world JSON or make a world file candidate-visible.

**Split identity is content-addressed, not filename-addressed.** The 10 Core structural payloads are exact duplicates in `guess_struct`. There are also two same-basename pairs in `core_decision` and `guess_decision`, but the files differ in byte size and both Git blob/SHA-256 digest; they are distinct payloads. Across 148 track entries there are 138 distinct payload digests. Preserve the full track path; never infer content identity from a filename.

The Query README describes four-turn Active, Active+Code and Modular agents plus a guess-only baseline. A future Query trial must reproduce the upstream task/agent/tool/turn budgets and scoring unchanged before adding Luminous metrics. The guess and core tracks are not automatically independent datasets.

## ACED-Query metadata inventory

The repository also pins the four ACED-Query inventory tracks in `upstream/aced_query_world_inventory.json`, with a separate v0.2 source manifest and closed schemas. Only metadata is copied: track path, source blob/digest, size, domain, task family, scenario subtype, seed and filename-declared graph/sample sizes. No Query world JSON or hidden evaluator content is stored in this adapter.

The inventory distinguishes track membership from content identity:
- 10 exact structural-world payloads appear in both `core_struct` and `guess_struct`;
- 2 decision-world basenames collide between `core_decision` and `guess_decision`, but the files have different Git blob and SHA-256 digests;
- 148 track entries contain 138 unique payload digests.

Decision-task domains are parsed separately from task family and scenario subtype. For the Core decision track, `invalid_premise` has six `valid_proposed_intervention` and four `wrong_side_intervention` subtypes. They remain one family and their actual domains remain distinct; they are not mislabeled as extra domains.

The library exposes `query_inventory::qualify_query_inventory` and `query_protocol::qualify_query_comparison_protocol`; the `aced-query-index-qualifier` binary validates exact track counts, distributions, path safety, the inventory and source-ledger digests, all 148 path/digest entries, payload overlap, filename collisions, the metadata-only candidate boundary, paper task denominators, parser identity, published baseline and resource-regime contract, and the closed comparison schema. It does not download or verify payload bytes, execute the four-turn Query protocol, or produce any agent score.

## ACED-Query comparison protocol

`ACED_QUERY_COMPARISON_PROTOCOL_V0_1.json` and its closed schema define the future Core/Qual model comparison. A critical distinction is that the four metadata tracks count **world files**, not scored tasks:

| Track | World files | Scored task projection |
|---|---:|---|
| Core structural | 10 | 10 Core list queries + 50 Qual questions |
| Core decision | 60 | 60 Core decisions |
| Guess structural | 30 | 30 Core list queries + 150 Qual questions |
| Guess decision | 48 | 48 Core decisions |

Thus the paper's active agents are evaluated on 70 Core tasks and 50 Qual questions. The Guess-only baseline is evaluated on its separate 78 Core / 150 Qual task set. Do not infer task denominators by summing world files or compare the Guess baseline as though it used exactly the same task set.

The v2 paper reports Claude Opus 5.5 with the Modular agent at 0.94 Core / 0.98 Qual, and a question-targeted active-learning procedure at 0.96 / 1.00. The latter is a serious classical baseline that Symthaea must reproduce, not a result to dismiss because it is not an LLM. The paper also reports that, under the separate Samp5k regime, Active+Code improves decision-task accuracy in 8 of 9 model cells by an average of 7.3 percentage points. These are external paper results, not Symthaea measurements. See [the full ACED v2 paper](https://arxiv.org/abs/2610.14381).

The protocol therefore keeps two resource regimes separate: the official four-outer-turn comparison and a 5,000-sampled-row budget without the turn limit. It pins the request-parser model, preserves raw and parsed answers separately, and adds a deterministic parser diagnostic without replacing the official LLM-extractor score.

## Reproduction

From the Symthaea repository root:

```bash
cargo fmt --manifest-path research/aced-adapter-001/qualifier/Cargo.toml --check
cargo test --manifest-path research/aced-adapter-001/qualifier/Cargo.toml
cargo run --manifest-path research/aced-adapter-001/qualifier/Cargo.toml --bin aced-repair-index-qualifier -- \
  research/aced-adapter-001/upstream/repair_test_index.json \
  research/aced-adapter-001/upstream/repair_eval_index.json \
  research/aced-adapter-001/ACED_REPAIR_SOURCE_MANIFEST_V0_1.json \
  research/aced-adapter-001/upstream/SHA256SUMS
cargo run --manifest-path research/aced-adapter-001/qualifier/Cargo.toml --bin aced-query-index-qualifier -- \
  research/aced-adapter-001/upstream/aced_query_world_inventory.json \
  research/aced-adapter-001/ACED_QUERY_SOURCE_MANIFEST_V0_1.json \
  research/aced-adapter-001/upstream/SHA256SUMS \
  research/aced-adapter-001/ACED_QUERY_COMPARISON_PROTOCOL_V0_1.json \
  research/aced-adapter-001/ACED_QUERY_COMPARISON_PROTOCOL_V0_1.schema.json
```

A passing inventory qualifier checks:
1. Exact raw index and checksum-manifest SHA-256.
2. Pinned upstream repository, commit, paper revision and license identity.
3. Exact inventory sizes, unique filenames/seeds and path safety.
4. Domain and archetype distributions.
5. Exact evaluation-subset membership and metadata equality.
6. Presence of all test/evaluation world payload paths in the upstream hash ledger.
7. Checksum-ledger entries for both index files match their pinned raw bytes.

It does **not** download or verify the compressed world payload bytes. For an official upstream reproduction, checkout the pinned upstream commit and run its `prepare_dataset.sh` so the original `data/SHA256SUMS` is checked against the actual payloads. Keep upstream data and source attribution intact.

## Security boundary for future model trials

The upstream repository explicitly warns that its Repair code sandbox is a restricted process safeguard, not a security boundary; it also warns that the world files contain answers and should not be mounted into a candidate process for a public leaderboard. Its Query agents may execute code without restrictions.

Luminous must not simply run the upstream agent code sandbox on the host and call it secure. Before model trials:
- Keep compressed worlds, hidden SCM truth, oracle fixes, objective functions, scoring batteries and grader state in a separate evaluator-only store/process.
- Give candidate agents only the public scenario, permitted measurement/intervention responses, visible evidence ledger, tool schema and remaining budgets.
- Execute candidate code in an independently controlled Firecracker/gVisor or equivalently isolated microVM boundary with no network, cloud metadata, credentials, host mounts or evaluator IPC.
- Separate tool request, authorization decision, dispatch, tool response, independent state observation and outcome grading in their receipts.
- An unknown tool outcome must block subsequent state mutation until independently reconciled.
- Preserve every timeout, invalid result, incomplete world and failed trial in the denominator; never silently drop them.

## Comparative scorecard

Preserve official ACED metrics as their own lane:
- accepted rate;
- utility threshold pass rate (`part_a_utility_ok`);
- mechanism/battery score (`battery_fraction`);
- benefit recovered versus oracle fix;
- recommended utility versus oracle utility.

Add independent Symthaea research measures without altering the official score:
- intervention regret and objective improvement;
- true-lever discovery and unsupported-lever uptake;
- dose/combination selection accuracy;
- experiment efficiency and wasted-experiment rate;
- confounder, collider, instrument-only and surrogate-trap failures;
- subgroup-specific policy quality and worst-subgroup utility;
- evidence-ledger provenance and unsupported mechanism claims;
- experiment/turn/token/tool/compute budget and p50/p95 cost;
- timeout, invalid, incomplete and indeterminate denominators;
- worst-archetype and worst-domain results.

No single aggregate causal score is authorized. Report the complete vector, per-world paired comparisons and uncertainty intervals.

## Experiment selection and optimal stopping

`ACED_REPAIR_EXPERIMENT_SELECTION_PROTOCOL_V0_1.json` specifies a separate research comparison that preserves the upstream score. It compares:
- official ACED agent;
- random valid intervention;
- greedy expected information gain;
- expected value of sample information (EVSI);
- risk-adjusted EVSI;
- adaptive evidence-gated experiment selection.

The scientific distinction is important: an experiment can reduce uncertainty without changing the best available decision. Conversely, a low-entropy observation can reverse the optimal intervention. We therefore measure generic information gain separately from decision-relevant value.

The protocol also pre-registers stopping comparisons: premature-stop regret when a useful experiment remains, versus unnecessary experiment cost when further evidence cannot materially improve the decision. Predicted experiment value must be recorded before the outcome is observed; an agent's post-hoc explanation is not a substitute.

The 54-world upstream evaluation split is useful for an initial matched reproduction, but its domain counts are only five or six worlds in most groups. It is not automatically powered for all subgroup claims. Confirmatory comparisons require a preregistered power analysis and world-clustered uncertainty; the test/eval parent/subset relationship must not be treated as independent samples.

## Decision-value experiment selector

The Rust qualifier now includes a small executable reference selector in `qualifier/src/decision_value.rs`. It implements three distinct policies over an explicit finite hypothesis distribution, downstream action-utility table and likelihood model for each experiment:

- **EIG:** maximize expected entropy reduction minus experiment cost.
- **EVSI:** maximize expected improvement in the best downstream decision minus experiment cost.
- **Risk-adjusted EVSI:** subtract a declared experiment-risk penalty as well.

This deliberately separates *learning about the world* from *learning what would change the decision*. In the synthetic counterexample, a high-entropy probe separates two hypotheses that recommend the same action; it reduces entropy but has effectively zero decision value. A lower-entropy signal can reverse the optimal action and therefore has positive EVSI. A third probe reveals the world perfectly but has high declared intervention risk, so ordinary EVSI chooses it while risk-adjusted EVSI chooses the safer decision-relevant signal.

The module includes negative controls for malformed probability models, deterministic tie-breaking and stopping when every experiment has negative net value. These are synthetic algorithmic tests, not ACED model-run results.

The selector is a reference baseline, not a claim of general optimality. In a real trial, posterior probabilities, likelihood calibration, utility functions and risk weights must be supplied from information available to the agent before observing the outcome. The hidden ACED SCM, oracle fix and grader must never be used to fit or backfill those values.

The plan now requires a numeric sample design, not free-text power assumptions. The reusable `qualifier/src/power_design.rs` calculator checks a two-sided, equal-allocation, independent-means normal approximation. The fixture uses standardized effect size 0.5, alpha 0.05 and target power 0.80, requiring 63 units per arm (126 total); the previous 40-unit design is rejected. This is **not** a general power solver: do not apply it to clustered, paired, binary, ordinal, survival, sequential/adaptive or multiplicity-adjusted designs without a design-specific method.

The plan's primary endpoint is explicitly continuous (`primary_objective`) with a two-sided independent-means estimand. The positive/negative forecast labels are a separate preregistered threshold taxonomy; they are not the outcome scale used in the sample-size formula. The validator rejects a categorical primary endpoint paired with the continuous-mean power method. The Rust validator recomputes the minimum required total and refuses a plan below it. The checked-in synthetic plan's 126 units satisfy the declared approximation; a 40-unit counterfactual is a negative control and fails qualification. The power calculator's normal approximation is intentionally limited to independent equal-allocation continuous outcomes; clustered, paired, binary, ordinal, survival, sequential/adaptive, repeated-look and multiplicity-adjusted designs require another method.

Plan v0.2 also requires one of two seeded randomized allocation methods (`seeded_randomized_pair_assignment` or `blocked_seeded_randomization`) and an explicit matched no-intervention, active or sham control. It rejects `uncontrolled_total_effect` and `control.kind = none`. These are the supported controlled designs for this version; observational, crossover, adaptive or cluster-randomized studies need separate schemas and power models rather than being squeezed into the equal-allocation mean approximation.

The experiment-plan artifact is separately defined by `ACED_EXPERIMENT_PLAN_V0_2.schema.json`, with a synthetic fixture at `qualifier/tests/fixtures/aced_experiment_plan_v0_2.json`. The Rust library exposes `validate_experiment_plan` and `validate_trace_with_plans(trace, plan_artifacts)`: it validates the closed nested plan fields, recomputes the canonical SHA-256 from the supplied plan object, resolves every proposal digest to exactly one supplied plan, and requires `experiment_id == plan_id`. Substituted controls/interventions, a missing plan artifact or a mismatched plan ID fail closed. This validates the supplied artifact and trace relationship; it does not prove that the proposal was timestamped before execution or that the external runner executed the plan faithfully.

## Robust EVSI across plausible decision models

The selector library now has a separate `robust_decision_value` module. It evaluates the same candidate experiment against a **predeclared set of plausible decision models**, where each scenario supplies its own prior, action-utility table and experiment likelihoods.

The selected experiment maximizes its **worst-case net EVSI** over the declared set:

`minimum_scenario(EVSI - experiment_cost - risk_penalty)`

This addresses a failure mode of plug-in EVSI: a probe may look valuable only under one fragile prior/likelihood/utility specification. The robust selector refuses to let a high mean or best-case value hide a non-positive worst case. It reports minimum/maximum EVSI, the minimum/maximum scenario net value, the descriptive scenario mean and the number of scenarios. It does not assign probabilities to scenarios or present the arithmetic mean as an expectation.

The module requires at least two distinct scenario identities, an identical candidate experiment set, and invariant declared cost/risk across scenarios. It fails closed on duplicate identities, set mismatch, cost/risk drift, invalid probabilities and invalid remaining budgets; it returns no experiment when no candidate has positive worst-case net value.

**Evidence requirement:** the uncertainty set must be fixed before the outcome is observed. Build it from independently calibrated model alternatives, predeclared parameter bounds or a held-out uncertainty analysis. Do not let the candidate invent favorable scenarios after seeing the result. Report scenario coverage and sensitivity to set construction. Worst-case robustness can be conservative; compare it with nominal EVSI and EIG as separate arms. The module is a finite-set robust decision rule, not a Bayesian posterior over models, not distributionally robust optimization over an unbounded ambiguity set, and not a multi-stage POMDP solver.

The executable tests use synthetic decision tables to show how the lower-envelope rule prefers an experiment that remains valuable across model alternatives, even when a competing experiment has higher best-case value. They are algorithm controls only, not ACED world runs or model results.

## Prospective experiment-value trace

The reusable Rust library now also exports `experiment_trace`. Its trace contract records three event types:

1. `experiment_proposal`: exact belief-state digest, selector, outcome probabilities, predicted information gain (bits), expected value of sample information (utility), risk, risk aversion, cost (utility) and selector-specific predicted selection score. EIG is kept in bits; EIG minus utility cost is deliberately not computed.
2. `experiment_observation`: a later event referencing the proposal event ID, with observed / failed / unknown status, an evaluator-resolved outcome label (when observed), an observation digest and actual cost. Failed/unknown attempts remain in the trace but are not scored as ordinary outcomes.
3. `stop`: the final evidence-state digest and reason. A stop is rejected if any proposed experiment lacks an observed, failed or unknown reconciliation event.

Each event carries a sequence number, previous-event digest and SHA-256 over a canonical JSON envelope. The validator rejects altered forecasts, hash-chain breaks, unknown/future proposal references, duplicate observations, malformed outcome distributions, inconsistent selector arithmetic, and proposals after stop.

Every proposal requires `experiment_plan_sha256`. The digest must bind a separately retained immutable plan specifying the target/hypothesis, intervention and controls, randomization, measurement variables/timing, outcome taxonomy, sample size and power assumptions, budget/cost, safety limits, and stop/replicate/conflict rules. A change to the plan requires a new proposal and digest; do not overwrite the earlier plan. The current validator checks digest format and chain linkage only. A future runner must retrieve the plan artifact and recompute its hash before dispatch, and independently compare the execution receipt with the plan. This does not yet provide trusted timestamping or prove that the experiment was executed as planned.

The trace also computes proper predictive scores: multi-class Brier score and log loss for an evaluator-resolved observed outcome. Brier is a proper score but **not a pure calibration measure**: it mixes reliability/calibration, resolution and outcome uncertainty. Report per-outcome reliability diagrams or a decomposition when the sample supports it, with bin counts and uncertainty; include sharpness and outcome-frequency baselines. See the [scikit-learn calibration guidance](https://scikit-learn.org/stable/modules/calibration.html) and the 2026 [review of proper scoring rules](https://doi.org/10.1146/annurev-statistics-042424-050626). Do not use ECE or one Brier aggregate as the only calibration metric.

The schema is `EXPERIMENT_VALUE_TRACE_V0_1.schema.json`; the Rust implementation is `qualifier/src/experiment_trace.rs`. The validator computes multi-class Brier score and log loss for observed outcomes using the proposal's frozen forecast and the evaluator-resolved outcome class. Zero-probability observed outcomes are flagged as infinite log loss rather than silently clipped. Failed/unknown observations are retained but excluded from proper-score denominators. The trace is not yet wired into an ACED model runner. Its `predicted_selection_score` follows selector units: EIG bits; EVSI utility minus cost; risk-adjusted EVSI utility minus cost and the declared risk penalty. It is invalid to compare EIG score magnitudes directly against EVSI utility scores.

**Important limit:** a hash chain alone does not prove when an event was created. To make a forecast genuinely prospective, the event digest must be anchored outside the agent process at proposal time (for example, by a trusted timestamping/signing service or independently controlled append-only recorder). The current module validates structure and internal hash consistency only; it does not provide trusted time, signer identity or WORM storage.

## Sample-size-aware categorical shift diagnostics

The ACED-Query Active agent currently summarizes interventional-versus-observational category-frequency differences using a fixed two-percentage-point display threshold, and its cross-condition summary uses a five-point maximum-delta threshold. These are descriptive heuristics, not inferential tests: the same absolute shift has different uncertainty at (n=10) and (n=1{,}000), and examining many variables/categories increases the chance of false alarms.

The Rust library now includes `categorical_shift.rs`, a diagnostic comparator that:
- requires identical declared category sets and valid, non-empty sample counts;
- reports each category's reference and intervention proportion and signed difference;
- reports total-variation distance and maximum absolute category shift;
- calculates Wilson score intervals for each proportion and a Newcombe interval for the difference;
- keeps the minimum practical effect threshold separate from whether the supplied interval excludes zero.

The caller supplies a **predeclared familywise critical z value**. This module does not infer a multiple-testing policy from the observed data or pretend that unadjusted 95% category intervals control familywise error. Choose the critical value from a preregistered comparison family (for example, a conservative Bonferroni design), record that family size and retain all tested variables/categories in the result receipt. For repeated/adaptive looks, use a sequentially valid method rather than reusing a fixed-look critical value.

Synthetic controls test that a 30-percentage-point shift can remain uncertain at (n=10) while the same shift is detected at (n=1{,}000), and that effect size and interval evidence remain separately reported. This is a statistical diagnostic module, not an intervention-effect estimator, proof of randomization, or ACED agent result. A distribution shift is not automatically a causal effect; it requires a valid assignment/intervention design, comparable measurement, and a correct counterfactual contrast.

## Baselines and ablations

Reproduce the official scripted/mock controls and official evaluation protocol where licensing and execution contracts allow. Then compare under matched model snapshot, prompts, tools, 15-experiment/32-turn budgets and compute accounting:
- single-agent investigation;
- independent-sample/self-consistency;
- standard multi-agent debate;
- evidence-ledger / revision-aware agent;
- active experiment-selection agent;
- full evidence-gated Symthaea system.

Ablate active experiment selection, causal structure support, evidence ledger, uncertainty calibration, dose/combination search, subgroup policy handling and stopping rules one at a time.

Proposed success condition: demonstrate a preregistered, statistically supported improvement over the strongest matched baseline in official acceptance and/or utility regret **without worsening worst-archetype or worst-domain outcomes**. Do not declare a percentage-point superiority target before establishing a reproduced baseline and power analysis.

## Claim ceiling

A PASS currently qualifies only the pinned source metadata, exact inventory, checksum-ledger coverage and evaluation-subset contract. It does not qualify compressed world payload integrity, model behavior, official grader reproduction, sandbox security, causal discovery or causal repair competence. SOTA or general capability claims require executed trials, matched baselines, independent outcome evaluation, multiple model/harness configurations and uncertainty estimates.
