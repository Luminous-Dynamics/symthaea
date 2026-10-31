# AR-5B4 Event Tape v0.1

**Status:** implementation candidate; not yet compiled or executed on an exact repository head.  
**Module:** `symthaea-architecture-engineering::event_tape`  
**Tracking:** [Issue #9362](https://github.com/Luminous-Dynamics/symthaea/issues/9362)

## Purpose

The event tape is the evaluator-owned record of what a search policy selected, what the evaluator/verifier returned, what the policy was allowed to observe, and what resources the action consumed. It provides a common event contract for mechanism-matched policy comparisons and a replay boundary for policy adapters.

It is not a benchmark result, a digital signature, proof of runner identity, or evidence that the evaluator itself is honest.

## Frozen identities

The header binds the campaign, exact source commit, runtime/toolchain/dependency/hardware-manifest digest, policy ID and algorithm ID, canonical JSON policy configuration and its SHA-256 digest, task recipe, comparison profile, evaluator, verifier, proposal source, initial canonical direction IDs, execution mode, worker count, declared seats per round, claim caps, normalized-cost model ID and digest, and total normalized budget ceiling.

A mechanism-matched tape must include a shared proposal-stream digest. The source commit must be a full 40- or 64-character Git object ID, and the environment digest must cover the toolchain, dependency lock, runtime, and hardware manifest used for the run. A best-practical-system tape may use a different proposal source, but must identify that source. These comparison modes must not be merged into one result table.

## Event contents

Every event contains a unique event ID, policy ID, worker and round, optional batch ID, the last event sequence visible to the policy, the digest of that exact visible prefix, an optional retry reference, the selected action intent, the evaluator/verifier outcome, and a complete resource receipt.

Candidate evaluations bind candidate, parent, semantic-delta, task-recipe, comparison-profile, and evaluator digests. Every candidate outcome binds both evaluator and independent-verifier build digests plus a verifier receipt, including rejected, crashed, and timed-out work. A verified result additionally requires a finite score and evidence-artifact digest. Rejected, inconclusive, crashed, and timed-out actions remain in the tape and remain charged.

Direction proposals bind a descriptor digest and semantic delta. Only the verifier may admit a canonical direction ID. A duplicate must map to an already registered ID and cannot mint another arm. An admitted ID must be genuinely new to the current registry.

## Resource accounting

Each action records host CPU milliseconds, wall milliseconds, accelerator milliseconds, host peak memory bytes, accelerator peak memory bytes, energy millijoules, input/output tokens, tool calls, evaluator calls, verifier calls, and normalized budget charge in microunits.

Normalized charges are meaningful only relative to the embedded, executable budget model and its digest; the same microunit value under two different models is not automatically comparable. The model is either `ActionUnit` (an explicitly synthetic fixed charge per action) or `WeightedResources` (integer fixed-point weights over measured resource dimensions). Every event's reported charge is recomputed from the raw receipt and must match exactly. An unknown input needed by a weighted model makes the charge uncomputable; an unsupported charge is rejected rather than trusted. The formula itself is serialized in the tape header, hashed, and included in cross-policy contract reconciliation. A reading is either `Measured(value)` or `Unknown(reason)`. Measured zero is valid; unknown is never converted to zero. `validate_integrity` preserves structurally valid but unmetered tapes for diagnosis. `validate` is the qualification gate and rejects any unknown measurement or unknown charge. Retries are separate events and consume separate charges. The running normalized charge may not exceed the header budget ceiling.

## Sequential and batched visibility

Sequential events must use worker 0, have no batch ID, advance one round per event, and see the entire completed prefix.

Batched events use a common batch ID and round. All events in the batch must have the same pre-batch visibility cutoff and the same visible-history digest. Each worker may contribute at most one action to a batch, and sibling results cannot become visible until a later batch. The declared seats-per-round count must be met before a batch can close or the tape can qualify; a partial final batch remains structurally inspectable but fails qualification. Worker count, seats per round, and claim caps are part of the frozen header. An empty tape also fails qualification: a campaign must contain executed actions before it can be reported as an experiment.

## Integrity and replay

The tape is append-only at the API boundary. The first event's prior hash is the deterministic digest of the frozen header; each later event hash binds its sequence, prior event hash, and serialized payload. The header itself is read-only through the API. Validation checks sequence continuity, the hash chain, the policy/config identity, visible-prefix digest, action/outcome agreement, canonical direction registration, retry identity, resource receipts, worker/batch semantics, claim caps, and budget ceiling.

A `PolicyReplay` adapter must identify the same policy ID, algorithm/build ID, and frozen config digest, then receives only the event prefix declared visible for a decision plus that decision's worker/round context. It must reproduce whether the policy selected a registered direction or requested a new direction. Candidate-generation and evaluator correctness remain separate responsibilities.

The first event links to a digest of the frozen header, so changing the budget, worker count, policy snapshot or other header field invalidates the chain. The SHA-256 chain is tamper-evident only when the tape is retained and compared against a trusted digest. It does not authenticate who produced the tape. Signing, remote append-only storage, independent evaluator deployment, and cross-policy campaign reconciliation remain follow-on work.

## Qualification limits

Passing the tape validator establishes structural integrity and accounting completeness under the declared contract. It does not establish that a scientific claim is novel, that the evaluator's labels are correct, or that one policy is superior. Confirmatory comparisons still require paired landscapes, independent templates, pre-registered endpoints, sealed evaluation, external baseline reproduction, independent replay, and held-out transfer.


## Cross-policy campaign reconciliation

The `matched_campaign` module accepts two or more per-policy tapes for one campaign/seed/budget. It requires a common source commit, execution-environment digest, task recipe, comparison profile, evaluator/verifier identities, proposal-source and proposal-stream digests, budget-model ID/digest, budget ceiling, initial direction registry, and identical execution/seat contract. Policy IDs, algorithms and configurations are intentionally allowed to differ.

For every candidate digest appearing in multiple tapes, direction, parent, semantic delta, and full evaluator/verifier outcome must agree. The same rule applies to repeated proposal descriptor digests and verifier outcomes. The summary reports candidate/proposal identity overlap, per-policy event counts and charges, batch completion, strict resource qualification, and oracle-policy IDs separately.

This reconciler establishes common-contract consistency and cross-policy outcome consistency for identities that overlap. It does **not** prove that the evaluator's complete offered-choice set was materialized or that a shared stream digest faithfully represents every possible option. A future campaign-input artifact must carry the actual offer-set manifest and digest, with independent verification that every policy received the same manifest. It also does not establish scientific superiority; paired statistical analysis and held-out transfer remain separate gates.


## Coral four-seat execution adapter

The separate `coral_execution` module wraps the formula-level scheduler baseline without changing its allocation equations. It rejects modified scheduler parameters for the reference profile: W=32, prior_n=4, infancy_k=3, κ=0.05, four agents, first-round per-direction cap=1, and later per-direction cap=2.

For each round, it freezes the pre-round board digest, the exact advisory allocation, four worker IDs, their actual arm claims, and a round digest. The event-tape batch ID is derived from that round digest, so the four action events can refer to the precise seat plan. The four actual claims must cover workers 0–3 exactly once, obey the direction caps, and map one-for-one to four events in the matching event-tape batch. Candidate-evaluation events must match the claimed direction; proposal events correspond to NEW.

The recommendation remains advisory. The adapter records deviation between suggested and actual seat allocation instead of forcing a worker to follow a board recommendation. This reflects Coral Discovery's documented workflow in which agents finish a line of work, consult the direction board, and choose; duplicate/crowding checks are separate from the board's advisory allocation. The adapter is not presented as a complete reproduction of the multi-agent runtime or as proof of Coral's empirical result.

The event-batch reconciliation requires strict event-tape qualification, including measured resource receipts. Synthetic data with unknown resource dimensions cannot pass this gate. Each round-plan artifact should be stored next to its event tape; the content-derived batch ID creates a deterministic cross-reference, while the plan digest remains independently auditable.


## Content-addressed offer-set manifests

The `offer_set` module closes a gap that a shared proposal-stream digest cannot close. A per-policy manifest records, for each sequential decision or batched round, the visible-prefix cutoff, sorted canonical direction IDs actually available, whether NEW was permitted, and the digest of the evaluator-owned pre-decision state from which those choices were derived.

Manifest validation requires exact coverage of all and only the event tape's decision boundaries. Every candidate evaluation must target a direction in that boundary's offered set; every direction proposal requires NEW permission. Every snapshot and the manifest root are content-addressed. Cross-policy comparison requires identical snapshot digests for each round/batch key, not merely the same declared random seed or generator name.

The state digest is still a commitment, not the state artifact itself. The campaign evidence bundle must retain the underlying evaluator-owned state/offer-set artifact, and an independent auditor must recompute `source_state_sha256`. Without that artifact, the manifest establishes cross-policy digest consistency but not the truth of the offered choices. The current API intentionally does not claim cryptographic identity of the evaluator process or a remote timestamp.

The `matched_campaign` summary also aggregates full per-policy resource vectors (host CPU time, summed action wall-time/worker-time, accelerator time, energy, tokens, tools, evaluator/verifier calls and normalized charges) rather than exposing only a scalar. Per-action host and accelerator peak memory aggregate by maximum, not by sum; these are not a substitute for campaign-wide concurrent-memory telemetry. Summed action wall duration is worker-time, not elapsed campaign latency. The summary separately reports whether normalized total spend is equal across policies. Equal budget ceilings are not equal realized spend; a campaign can reconcile its common contract while explicitly failing the equal-spend comparison gate.


## Landscape-clustered paired inference

The `paired_inference` module implements a preregistered primary-endpoint comparison. It accepts exactly two policy identities (baseline and challenger), one metric/direction, a practically meaningful effect threshold, alpha, bootstrap/randomization replicate counts, deterministic analysis seed, and minimum landscape/paired-seed counts.

The frozen plan also binds the sorted event-tape set digest used to produce the observations, and every individual metric observation binds the digest of its archived source artifact. The receipt therefore commits to both the campaign evidence set and the exact metric artifacts, rather than merely storing scalar values. An independent analysis runner must recompute the event-tape-set digest and artifact digests from the archived objects before treating the result as an auditable scientific comparison. Every `(landscape, seed)` must contain exactly one observation per policy. Within each landscape, matched seed deltas are averaged first; landscapes—not runs/seeds—are then the independent units. The confidence interval resamples landscapes with replacement. The paired sign-flip randomization test operates on those landscape means, using exhaustive sign assignments up to the configured exact limit and seeded Monte Carlo beyond it. Its null calibration assumes that, under the sharp no-effect null, paired landscape differences are exchangeable under sign reversal; this assumption must be stated in the preregistration and is not a distribution-free test of a mean. The cluster-bootstrap interval and per-landscape win/tie/loss pattern are reported alongside it as complementary evidence. The receipt includes per-landscape deltas, landscape and seed win/tie/loss counts, CI, p-value, plan/data digests and an inference digest.

A policy is labelled superior only when the entire two-sided cluster-bootstrap interval exceeds the preregistered minimum practical effect and the paired sign-flip p-value is at most alpha. An analogous rule labels inferiority. Otherwise the result is `NoDemonstratedDifference`; it is not an equivalence claim. This module handles one primary endpoint at a time. If several endpoints or subgroups are promoted to confirmatory claims, the campaign must pre-register a familywise/multiplicity procedure (for example Holm correction) outside this v0.1 primitive.

This method intentionally prevents a large number of correlated seeds from one landscape from being counted as a large number of independent samples. For a serious confirmatory comparison, target at least 30 independent landscape templates and 100 paired seeds per landscape where feasible. More seeds improve within-landscape measurement precision; more independent landscape templates improve external validity. The values are recommended campaign targets, not a claim that every existing synthetic campaign has met them.

## Research scoreboards and benchmark positioning

Do not collapse all evidence into one leaderboard. Publish three separate scoreboards:

1. **Mechanism-matched policy quality.** Same materialized offer-set, evaluator, verifier, source/runtime, batch semantics, normalized-cost model and pre-registered budget. Use paired landscapes, seeds and the clustered inference receipt. This answers whether one allocation/search mechanism is better.
2. **Best-practical-system performance.** Each complete system may use native proposal generation, orchestration, retries, batching and model/tool choices. Charge all work, including failed attempts and verifier/evaluator overhead. This answers which deployable system gets the best outcome per real cost.
3. **Scientific integrity and transfer.** Evaluate reproduction of known findings, claim/evidence consistency, mechanism/insight quality, held-out prediction/ranking and transfer to independent task families. This answers whether an apparent optimization win is actually useful science.

External benchmark mapping:

| Benchmark/system | What it adds to AR-5B4 | How to use it |
|---|---|---|
| [Coral Discovery](https://discovery.coral-labs.ai/research/) | Direction-board mechanism; four-agent allocation; duplicates and direction crowding | Reproduce all runs and task conditions; treat its published 6/1/2 best-of-three result as a reference, not a guaranteed expected win. Report cost and margins, not just win count. |
| [ShinkaEvolve](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7886b9bafe76c52fd568db10ff9772df-Abstract-Conference.html) | Parent selection, novelty rejection and bandit selection of model ensembles | Add as evolution-search controls and ablate each operator; the paper reports a 150-sample circle-packing discovery, but compare under a matched task/model/budget. |
| [CodeEvolve](https://aclanthology.org/2026.findings-emnlp.1345/) | Island evolution, inspiration crossover, CVT-MAP-Elites and model ensembles | Include as a stronger evolutionary-system baseline. Its 2026 Findings paper reports matched-condition advantages on several AlphaEvolve benchmark tasks; this makes it an important competitor, not an optional citation. |
| [FormulaCode](https://proceedings.mlr.press/v306/sehgal26a.html) | Repository-scale performance diagnosis and correctness-constrained multi-objective optimization | Use external tasks when eligible; add Symthaea/Rust repository bottlenecks as a separate domain-specific transfer suite with expert-reviewed patches/workloads. |
| [FIRE-Bench](https://proceedings.mlr.press/v306/wang26d.html) | Full-cycle rediscovery, experimental design, execution and evidence-backed claims | Use as the scientific-integrity track; inspect all run-level variance and claim-level precision/recall, not only aggregate F1. |
| [SciAgentArena](https://sciagentarena.github.io/) | Approximately 200 interactive, stepwise-verified tasks across scientific domains | Use for cross-domain interactive transfer and per-step failure diagnosis. |
| [Into the Unknown](https://arxiv.org/abs/2610.19159) | Scientist-authored open research conditions and integrity verification | Use as a hard scientific-integrity stress test; no system should pass by producing a polished report unsupported by its tape. |
| [Autonomous Science Bench 0.1](https://www.autonomous-science-bench.ai/) | Six pilot real-lab workflows, budgeted rounds, discovery plus ranking of unmeasured designs | Use as a pilot/adapter target, not a stable broad leaderboard. Its own scoring separates discovery from learning/ranking unseen designs. |
| [EurekaBench](https://arxiv.org/abs/2610.00492) | Long-horizon mechanisms, predictive accuracy and scientific insight | Use to test whether policies discover explanatory mechanisms, not only high-scoring candidates. |

## Metrics to publish

**Primary efficiency metrics:** area under best independently verified quality versus cumulative normalized spend; cost to first independently verified improvement; cost per verified improvement; probability of reaching a pre-registered target before budget exhaustion; and simple regret at fixed budget.

**Search-mechanism diagnostics:** unique directions admitted; duplicate proposal rate; false novelty rate from blinded verifier audit; direction entropy and Herfindahl concentration; candidate/offer-set coverage; fraction of budget spent on exploitation versus new directions; and time/budget to first useful direction.

**Reliability metrics:** exact tape replay rate; missing/corrupt artifact rate; false-positive/false-negative verifier rates from independently adjudicated controls; unmetered action rate; budget reconciliation error; and claim-to-evidence contradiction rate. Failures, retries and verifier costs remain charged.

**Transfer metrics:** held-out task-family performance; ranking quality on candidates never evaluated by the search policy; performance retention after domain shift; mechanism/insight rubric scores; and degradation under misleading, noisy, nonstationary or adversarial verifier feedback.

**Pareto metrics:** quality, latency, host CPU, accelerator time, host/accelerator peak memory, energy, input/output tokens and tool/evaluator/verifier calls, reported as vectors. Also report dominated fraction, hypervolume under a frozen reference point and the cost-quality frontier. Never collapse this to one scalar unless the conversion weights were pre-registered.

## Synthetic stress landscapes

The eight-policy direction campaign should not be evaluated only on stationary independent-arm payoffs. Build frozen task families covering: deceptive local optima; heavy-tailed and heteroscedastic noise; correlated directions; nonstationarity and change points; delayed feedback; duplicate/near-duplicate proposals; variable evaluator costs; verifier false positives and false negatives; censored/time-out observations; sparse rewards; changing offer sets; multi-objective quality/cost trade-offs; and capability cliffs requiring a new operator or abstraction.

For each family include a null control, a positive-control policy/landscape, an oracle upper bound reported separately, and at least one adversarial integrity condition. Freeze the generator and seed manifest before policy evaluation; do not retune a policy after inspecting the confirmatory holdout.

## What would count as beating the state of the art

Do not define SOTA as beating one paper's best run or winning more tasks on a best-of-k table. First reproduce Coral, ShinkaEvolve, CodeEvolve and appropriate repository/science-agent baselines under disclosed conditions. Then seek a pre-registered practical improvement in at least two independent task families, with a cluster-level interval above the minimum effect, no material regression in critical families, full replayable evidence, and lower or equal total normalized spend. Finally repeat on held-out landscapes and one external domain suite.

A particularly strong result would be: **same independently verified target quality with substantially lower total cost, while retaining the advantage under held-out landscapes and external task families**. A second, distinct breakthrough would be greater scientific insight/transfer at equal budget, even where raw best score is unchanged. These are stronger claims than simply obtaining more discoveries per fixed number of synthetic actions.

### Additional 2026 benchmark context

- **Autonomous Science Bench 0.1** provides six pilot laboratory campaigns with fixed rounds, batch sizes and budgets. Its leaderboard separates discovery of good designs from learning to rank designs never measured; the maintainers describe 0.1 as a pilot, so use it as an adapter and diagnostic suite rather than a mature broad leaderboard. https://www.autonomous-science-bench.ai/ https://www.autonomous-science-bench.ai/run
- **OpenScience's September 2026 results** report strong numbers on Terminal-Bench Science and BiomniBench-DA, but the authors' 1 October update acknowledges benchmark-scoring comparability issues and weaker engineering performance, and says reruns are underway. Treat those figures as provisional, not a settled SOTA claim. https://www.openscience.sh/benchmark


The weighted model supports host CPU, action wall-time, accelerator time, energy, input/output tokens, tool calls, evaluator calls, verifier calls, and memory-time proxy terms. The memory-time terms currently multiply a measured action peak by action wall duration; this is a declared conservative proxy, not an integral of the actual memory time-series. A campaign making memory-efficiency claims must collect true memory telemetry (preferably sampled/integrated over the execution interval) and should version a model using that signal. The synthetic direction campaign uses the ActionUnit model and intentionally remains unqualified for real compute-efficiency claims because its hardware resource meters are Unknown.

## Unified confirmatory comparison gate

The `matched_campaign::qualify_mechanism_matched_campaign` entry point consumes all policy tapes and all corresponding offer-set manifests. It first reconciles the common experiment and executable budget model, then verifies each manifest against its tape and requires identical options/state digests at every policy decision boundary. It returns an explicit qualification decision with blockers instead of treating a common header hash as a matched comparison.

The gate qualifies a campaign for *confirmatory mechanism comparison* only when: at least two distinct policies have a common experiment contract; all decision-boundary offer sets match; all sequential/batched execution schedules are complete; every required resource meter and normalized charge is known and internally recomputed; and all policies have equal realized normalized spend. Equal ceilings alone are insufficient. An unmetered synthetic campaign remains structurally useful but is blocked from resource-efficiency claims.

This is an execution-fairness gate, not a scientific-discovery verdict. It does not prove novelty, human usefulness, causal mechanism, external replication, or SOTA. Those require independently controlled evaluator/verifier processes, sealed test landscapes, registered hypotheses/endpoints, held-out transfer, and independent replay.

## Current claim ceiling

The AR-5B4 work now has implementation candidates for the event tape, executable cost models, cross-policy reconciliation, content-addressed offer-set manifests, and four-seat Coral round accounting. The source-state digest in an offer-set manifest remains a commitment until the underlying evaluator-owned state artifact is retained and independently rehashed. The Coral four-seat adapter preserves the published scheduler as advisory and verifies worker claims; it is not a reproduction of the complete Coral autonomous-agent runtime or a reproduction of the Coral paper's empirical results.

No result is claimed as a verified compile/test pass or a benchmark win until exact-head validation and independent reproduction are available.
