# AR-5B4 — matched-budget direction-search and scientific qualification protocol v0.1

**Status:** protocol and two new policy controls authored; implementation is a sequential synthetic harness only. No experiment results or SOTA claims.  
**Tracking:** [#9362](https://github.com/Luminous-Dynamics/symthaea/issues/9362)  
**Stack base:** [workspace/orphan repair #9361](https://github.com/Luminous-Dynamics/symthaea/pull/9361)  
**Research sources:** Coral Discovery (23 Oct 2026), FIRE-Bench (ICML 2026), Into the Unknown (13 Oct 2026), FormulaCode (ICML 2026), SciAgentArena, AutoResearchBench, ShinkaEvolve, CodeEvolve.

## 1. Research shift: optimize the evidence-producing process

The relevant target is not merely the highest score produced by an agent or the lowest objective value reached by a candidate. It is the **lowest fully charged cost to independently verified, reproducible, transferable improvement**.

A search system should therefore be decomposed into five independently observable capacities:

1. **Proposal quality:** generates a diverse, testable candidate or research direction.
2. **Allocation quality:** spends the next unit of work where it has the highest defensible expected value, without prematurely collapsing onto one approach.
3. **Experimental quality:** designs and executes a valid experiment that isolates the claimed mechanism.
4. **Evidence quality:** binds claims to actual inputs, artifacts, outputs, evaluator versions and negative controls.
5. **Transfer quality:** retains improvement under held-out tasks, perturbations, environment changes and independent replay.

This decomposes the research lifecycle into separate failure surfaces. A high proposal rate cannot hide invalid experiments; an optimizer improvement cannot hide a missing build or reproduction; a correct result on one benchmark cannot substitute for held-out transfer.

## 2. What the current implementation does — and does not — establish

The existing sequential direction campaign has four deterministic payoff families:

- delayed breakthrough;
- no-new-direction null;
- new-operator breakthrough;
- rank reversal.

Before policy-controlled actions, all policies receive the same three initial direction evaluations (A, B, C). Each subsequent action is either one direction evaluation or one NEW-direction proposal. Each action costs one synthetic unit. The verifier is currently a deterministic fixture that owns the payoff schedule and admits the one configured new operator direction.

The current policy set contains eight entries:

1. score-only winner-takes-all;
2. fixed round-robin;
3. seeded random;
4. global UCB;
5. Symthaea verifier-grounded DirectionBoard;
6. discounted UCB;
7. Beta-Bernoulli Thompson sampling;
8. oracle upper bound (uses hidden payoff information and is never a deployable baseline).

Two new controls were added in AR-5B4:

### Discounted UCB

For observed rewards r(i,t), the implementation forms an exponentially weighted mean:

    mu_hat(i,t) = sum(delta^j * r(i,t-j)) / sum(delta^j)

and adds an effective-sample exploration bonus. The fixed benchmark discount parameter is delta=0.85. This is a declared nonstationary control, not a claim to reproduce every discounted-UCB algorithm in the literature. The NEW arm's historical proposal dispositions are also discounted; an admission is a success, while duplicate/rejected/inconclusive proposals are failures. Its prior is explicit in the policy-config snapshot.

### Thompson sampling

The baseline samples from a Beta posterior per direction. Every direction evaluation becomes a Bernoulli success only when it reaches the fixed campaign target (currently 0.90); all other completed evaluations count as failures. The NEW arm earns a success only when the verifier admits a genuinely new direction. A duplicate, rejection or inconclusive proposal counts as failure. Priors are Beta(1,1), and the sampler is a seeded Marsaglia–Tsang gamma implementation.

This is a probability-of-target-hit baseline, not a general continuous-reward Thompson sampler. It should be compared on target-hit probability and time-to-target, and must not be presented as an optimized universal policy.

Every run now carries a typed frozen policy-parameter snapshot and its SHA-256 digest. This is useful configuration evidence, but it is not a signature and does not authenticate the runner or the result artifact.

The common event-tape contract is implemented in `campaign_tape.rs` and exported by the architecture-engineering crate. It provides domain-separated event/tape digests, hash chaining, evaluator-assigned admission/duplicate/rejection decisions, round/worker visibility cutoffs, claim caps, multi-worker action slots, unknown-is-not-zero resource aggregation, conservative within-round peak-memory aggregation, hard-cap enforcement and matched-header validation. The tape also has a `NovelCandidateEvaluation` event for systems such as Coral in which one NEW-arm worker action both evaluates a candidate and receives the verifier's admission/duplicate/rejection decision. A mapped duplicate remains charged to the NEW arm; it is not silently reclassified as a direction assignment. The current tape module has 27 authored tests.

The sequential DirectionBoard campaign reconstructs a tape for each run, stores its digest in `DirectionCampaignRun`, and validates the reconstructed digest during aggregation and paired comparison. The three shared bootstrap evaluations are encoded as one policy-invisible batch; policy-selected actions occupy subsequent sequential rounds. The direction campaign contains tests for per-policy tape reconstruction, matched headers, canonical NEW admission and stale-receipt rejection.

A four-seat Coral scheduler adapter is now implemented in `coral_batch_benchmark.rs`. It preserves the reference configuration (W=32, prior_n=4, k=3, kappa=0.05, four seats, first-round cap one, steady-state cap two), fixes all four allocations from one pre-batch snapshot, records idle seats rather than reassigning blocked claims, and captures the NEW direction's opening global best before processing any result from its admitting batch. Its seven authored tests cover parameter identity, batch visibility, caps, scored/rejected NEW attempts, and opening-best semantics.

**Limitations that remain:** the DirectionBoard runner is still sequential, unit-cost, synthetic and evaluator-known; the Coral adapter is also synthetic and uses the same payoff fixture primitives, not real candidate generation or measured runtime telemetry. Resource dimensions other than action units remain unknown. The event tape provides hash integrity but no signature, trusted key registry, out-of-process verifier, external proposal generator, sealed split, or independently controlled replay process. The Coral adapter does not reproduce the complete Coral Discovery multi-agent execution/verifier/orchestration system. A direct Coral-vs-DirectionBoard score table is still invalid until all policies consume the same fixed candidate stream and share the same four-seat worker/batch semantics.

## 3. Common action tape required for fair policy comparison

AR-5B4 defines a versioned, append-only event tape owned by the evaluator/verifier. A single paid action can create one of these events:

### Candidate evaluation

Required fields:

- action/event ID and sequence number;
- candidate digest, parent digest and semantic-delta digest;
- task concept, recipe, evaluator and comparison-profile identities;
- canonical verifier-assigned direction ID;
- worker ID, round ID and policy-visible information cutoff;
- disposition: verified, rejected, inconclusive, crashed or timed out;
- score and uncertainty only when the verifier permits them;
- complete resource receipt, including unknown dimensions.

### Direction proposal

Required fields:

- proposal ID and content digest;
- canonical descriptor request;
- parent/inspiration lineage;
- cost and tool-call receipt;
- verifier decision: admitted, duplicate(existing_direction_id), rejected or inconclusive;
- reason-class and verifier receipt.

For schedulers where a NEW worker action is itself a scored candidate attempt (the Coral seat semantics), use `NovelCandidateEvaluation`: it binds candidate, parent, semantic delta, proposed descriptor, verifier decision and optional score into one event and one charge. Admitted/duplicate outcomes require a finite score; rejected/inconclusive outcomes cannot publish a score. The proposal-only `DirectionProposal` event remains available for systems that actually spend a separate proposal action before later evaluation. These forms must not be substituted for one another in budget comparisons.

A candidate generator cannot authoritatively declare its own direction novel. A duplicate proposal maps to an already registered arm; it does not create a new arm or receive an admission reward. A rejected or inconclusive proposal still consumes its full charged cost.

### Resource and replay rules

- Unknown resource usage remains unknown, never coerced to zero.
- Retry, timeout, crash, evaluator call and verifier call costs are recorded.
- A batch records its full action set before any member's result is revealed.
- The tape records which observations each policy was allowed to see at each decision.
- The evaluator rejects duplicate event IDs, nonmonotonic sequence numbers, mismatched task/profile IDs, missing digests and cost arithmetic that does not reconcile.
- A replay engine reconstructs each scheduler's state from the same tape and must reproduce its recommendation exactly.

This action tape is a prerequisite to comparing all policy families, not a cosmetic logging feature.

## 4. Preserve algorithm identity and concurrency semantics

Do not call all direction boards "Coral" or silently change the paper's parameters.

The published Coral Discovery scheduler uses W=32, prior_n=4, first-k infancy k=3, kappa=0.05, a historical maximum-yield exploration scale, largest-remainder seat allocation and first-round/steady-state direction claim caps. Its four-agent configuration is an allocation of simultaneous advisory seats, not four independent sequential choices. The formula baseline in #9336 reproduces the scheduler equations and caps. The new `coral_batch_benchmark.rs` adapter feeds the reference configuration through four-seat pre-batch assignment and the common event tape, but it remains a deterministic synthetic scheduler adapter—not the complete Coral agent/verifier/orchestration system.

For an exact-formula comparison:

- preserve Coral's published per-evaluation credit semantics and defaults;
- treat all four advisory agent seats and their claims explicitly;
- apply the same worker count and batch boundary to every baseline in the mechanism-matched experiment;
- do not charge Coral four seats while charging another policy one worker, or compare one policy's action to another policy's whole batch;
- record the paper configuration and implementation digest in each run.

A single-seat Coral adapter or a sequentialized Coral formula is a useful ablation, but it must have a distinct policy name and cannot be described as the paper's four-agent reference configuration.

Publish two comparison tables:

1. **Mechanism-matched:** identical candidate stream, task/evaluator, action tape, batch boundary, worker count and charged budget. This isolates allocation/search policy.
2. **Best-practical-system:** native proposal generation, orchestration and agent collaboration are allowed, but every resource is normalized and fully charged. This compares useful complete systems, not an isolated scheduler.

## 5. Policy matrix

The confirmatory matrix should include:

| Family | Required control | Main question |
|---|---|---|
| Greedy | Score-only | How costly is following the current leader? |
| Fixed | Equal share / round-robin | Does diversity alone help? |
| Stochastic | Seeded random | Is a policy better than chance? |
| Stationary bandit | Global UCB | Does global reward plus exploration suffice? |
| Direction-aware | Symthaea DirectionBoard | Does direction-local credit preserve promising alternatives? |
| Reference method | Exact Coral scheduler | Does the published direction allocation replicate? |
| Nonstationary | Discounted UCB | Can recency weighting recover after regime changes? |
| Bayesian | Beta-Bernoulli Thompson | Does posterior sampling improve target discovery probability? |
| Quality-diversity | MAP-Elites / CVT-MAP-Elites | Do behavior archives improve feasible diversity and transfer? |
| Evolutionary | ShinkaEvolve / CodeEvolve-style parent, novelty and island controls | Is proposal generation the actual source of gains? |
| Upper bound | Oracle | Is the landscape and budget capable of reaching the target? |

MAP-Elites is relevant only when behavior descriptors have a defensible interpretation and archive coverage can be measured; do not manufacture arbitrary descriptors to create a MAP-Elites arm. Similarly, LLM-parent/novelty methods belong in the end-to-end proposal comparison unless all schedulers see the same fixed candidate stream.

## 6. Landscape design

Keep the original four landscapes as regression fixtures. Expand to at least six independent landscape families before confirmatory claims:

1. delayed breakthrough and early leader saturation;
2. true null (no NEW direction is better);
3. out-of-grammar primitive discovery;
4. rank reversal / changing regimes;
5. deceptive proxy and correlated directions;
6. heteroscedastic noise and unequal evaluation variance;
7. multi-objective improvement with resource trade-offs;
8. heterogeneous costs, hard caps and unknown resource estimates;
9. duplicate semantic aliases and verifier disagreement;
10. negative controls where all candidate families are equivalent or every improvement is illusory.

The six-family criterion is a floor, not a sample-size guarantee. Use at least 100 paired landscape instances/seeds per family for the initial confirmatory campaign, then set final sample size using pilot variance, a predeclared minimum effect and multiplicity control. Seeds generated from the same payoff template are not independent landscapes; uncertainty must be clustered at the independent landscape/template level.

Use two separate tracks:

- **Known-label policy controls:** evaluator supplies correct arm labels to isolate allocator arithmetic.
- **Discovered-label end-to-end controls:** the verifier must infer/assign direction labels from candidate deltas, with seeded alias/semantic-merge errors and a measurable confusion matrix.

A policy must not get oracle labels just because its implementation expects a direction ID. The independent verifier owns that mapping.

## 7. Primary metrics and success thresholds

Use fixed total charged resource, not evaluation count alone, as the primary denominator in real experiments. Until heterogeneous resources are integrated, report the sequential synthetic score as a narrow algorithmic control.

### Primary endpoints

- held-out verified score / objective at fixed charged budget;
- simple regret against a declared feasible optimum or lower bound;
- probability of hitting a predeclared target before budget exhaustion;
- charged resources to first verified improvement and to target;
- worst-family performance and transfer retention.

### Secondary endpoints

- unique verifier-admitted directions and valid direction coverage;
- false-NEW, duplicate and rejection rates;
- wasted proposal/evaluation fraction;
- allocation entropy / HHI and direction starvation;
- Pareto hypervolume where objectives are genuinely multi-objective;
- resource Pareto frontier (wall, CPU, GPU, host/GPU peak memory, energy, tokens, tool calls, evaluator calls);
- verifier disagreement, leakage-control failures and independent replay success.

### Proposed targets (not achieved results)

- exact charged-budget conservation: **100%**;
- reproducible policy replay for fixed seed/tape/config: **100%**;
- hidden-oracle access in non-oracle arms: **0**;
- false novel-direction admissions on seeded duplicate controls: **0**;
- verified receipt coverage for every result used in a reported metric: **100%**;
- all policies compared on the same evaluator/tape/budget: **100%**;
- positive paired held-out improvement in a predeclared majority of independent landscape families, with no unacceptable worst-family regression;
- superiority claim only after confidence intervals, independent replication, sealed transfer and baseline reproduction support it.

These are acceptance targets, not claims about the current implementation.

## 8. Scientific evidence ladder — do not optimize appearance

The end-to-end scientific agent must pass separate gates before a result can be called a discovery:

1. **Execution:** the experiment actually ran; command, environment, commit, seed and exit status are captured.
2. **Provenance:** source, config, task, recipe, data, evaluator, dependencies and outputs are content-addressed.
3. **Claim binding:** every reported numeric claim points to a real artifact and a deterministic derivation.
4. **Controls:** negative controls, leakage probes, baseline controls and evaluator integrity checks behave as expected.
5. **Independent replay:** a separately controlled verifier can repeat the result from an evidence packet without trusting the proposer.
6. **Robustness:** perturbations, resource changes, seeds and held-out tasks preserve the effect within predeclared tolerance.
7. **Transfer:** the mechanism improves a genuinely held-out task or regime, not merely another sample from the same synthetic payoff generator.
8. **Scientific review:** domain experts adjudicate interpretation, novelty, relevance and any claim not reducible to deterministic evaluation.

The sealed evaluator must withhold hidden payoff schedules, split membership, labels and final rubrics from proposal generators. Signing a receipt authenticates control of a signing key; it does not establish that a signer was independent, honest, competent or scientifically correct.

## 9. Competitive reference points (2026)

- **Coral Discovery** reports 6 wins, 1 tie and 2 losses against its strongest open autonomous multi-agent comparison on nine tasks. That is evidence for direction-aware division of cognitive labor on those reported tasks, not universal superiority. Its central distinction—direction-local credit plus a priced NEW option—is the relevant scheduler baseline. https://arxiv.org/abs/2610.33281
- **FIRE-Bench** evaluates rediscovery of verifiable scientific findings through experiment design, execution and evidence-backed conclusions, and reports that even the strongest agents remain below 50 F1 with high variance. This argues for evidence quality as a first-class endpoint. https://proceedings.mlr.press/v306/wang26d.html
- **Into the Unknown** tests open scientific follow-up directions contributed by DOE-lab scientists. None of 135 agent submissions satisfied every domain condition; 114 failed integrity checks, often because claimed work was not done or numbers contradicted recorded evidence. This strongly motivates execution/claim binding, not paper-style scoring alone. https://arxiv.org/abs/2610.19159
- **FormulaCode** evaluates 957 repository-level performance bottlenecks with correctness constraints and multiple performance workloads. It is a strong external anchor for whole-repository optimization and performance/correctness trade-offs. https://proceedings.mlr.press/v306/sehgal26a.html
- **SciAgentArena** provides approximately 200 scientific tasks with stepwise verification in multiple domains. Use it as an external scientific-agent benchmark where reproducible task access permits. https://sciagentarena.github.io/
- **AutoResearchBench** evaluates deep and wide scientific literature search; its reported results show that even strong browsing systems struggle on research-oriented discovery tasks. It measures upstream evidence retrieval, not experimental discovery. https://arxiv.org/abs/2604.25256
- **ShinkaEvolve** is a relevant parent-selection, novelty-rejection and model-bandit baseline. https://proceedings.iclr.cc/paper_files/paper/2026/hash/7886b9bafe76c52fd568db10ff9772df-Abstract-Conference.html
- **CodeEvolve** adds island-based evolution, inspiration crossover, depth refinement, CVT-MAP-Elites and weighted model ensembles; its reported AlphaEvolve-suite results make it a useful end-to-end coding-optimization competitor. https://aclanthology.org/2026.findings-emnlp.1345/

Do not compare reported scores from different task sets, models, budgets, judge access, or scoring protocols as if they were one leaderboard. Pin code, model snapshots, prompts, hardware and task/evaluator versions before reproducing any result.

## 10. Next implementation sequence

1. Port the remaining baseline policies into four-seat adapters that use the same batch visibility, claim caps and common evaluator-owned candidate stream; retain separate sequential ablations.
2. Implement candidate-stream parity and verifier canonicalization for fixed-proposal mechanism comparisons.
4. Add the six-plus landscape suite, including nulls, correlated directions and noisy/nonstationary payoffs.
5. Add matched-seed statistics with cluster bootstrap at the landscape/template level, pre-registered primary endpoints and multiplicity treatment.
6. Connect full resource receipts from the single-worker Pareto allocator; only then add atomic multi-worker reservations.
7. Connect the sealed AR-5D evaluator and separately controlled replay verifier.
8. Add MAP-Elites and evolutionary proposal baselines after defining non-arbitrary behavior descriptors and fixed-proposal vs native-proposal tracks.
9. Run at least 100 paired landscape instances per family and publish the complete event tape, receipts, analysis script and exact commit identifiers.

## 11. Claim ceiling

The current policy harness demonstrates deterministic policy discrimination on four handcrafted payoff schedules. The two added policies are controls authored for the next comparison, not demonstrated improvements. The authored unit tests have not been compiled or executed yet.

Even after the harness passes, only a matched synthetic campaign can support statements about scheduler behavior on those synthetic landscapes. A claim that Symthaea beats SOTA in scientific discovery or code optimization requires independent, held-out, budget-matched reproduction against the external systems and separate scientific integrity qualification.
