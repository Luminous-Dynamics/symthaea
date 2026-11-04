# Deterministic query-disambiguation observation planning V1

**Tracking:** [#9987 — value of next evidence / scientific review prioritization](https://github.com/Luminous-Dynamics/symthaea/issues/9987)  
**Related:** [#9344 — query robustness across plausible reconstructions](https://github.com/Luminous-Dynamics/symthaea/issues/9344), [#9255 — query-scoped reliability certificates](https://github.com/Luminous-Dynamics/symthaea/issues/9255), [#10502 — exact query-flipping witnesses](https://github.com/Luminous-Dynamics/symthaea/pull/10502).  
**Tool:** connectome-query-disambiguation-audit.  
**Status:** standalone Rust exact planner, schema, known-answer fixtures, independent combinations/set-cover oracle and isolated CI authored. This is deterministic discrimination over a submitted finite world/action model, not probabilistic value of information.

## 1. Question

Given a frozen set of plausible connectome reconstruction worlds and a scientific query whose decision differs across some worlds:

> What is the minimum declared expert effort needed to select observations that distinguish every pair of submitted worlds that disagree about the query decision?

This deliberately narrows the next-evidence problem. It does not maximize predictive entropy, compute expected information gain, infer probabilities or claim that a measurement will be biologically decisive. It first asks whether the submitted actions can distinguish the conclusion-changing alternatives at all.

## 2. Formal problem

Let worlds be \(W=\{w_1,\ldots,w_m\}\), each with a frozen query decision \(d(w)\in\{0,1\}\). Define the decision-conflict pairs:

\[
P=\{(w_i,w_j):d(w_i)\ne d(w_j)\}.
\]

Each candidate observation action \(a_k\) has:
- a positive integer effort cost \(c_k\);
- a stable action identity and kind;
- source-evidence and observation-model digests;
- explicit shared-ancestry/dependence group labels;
- exactly one declared deterministic categorical outcome \(o_k(w)\) for each submitted world.

An action separates pair \((w_i,w_j)\) exactly when \(o_k(w_i)\ne o_k(w_j)\). A selected action set \(A'\) is a complete disambiguation plan iff:

\[
\forall (w_i,w_j)\in P,\quad
\exists a_k\in A': o_k(w_i)\ne o_k(w_j).
\]

V1 solves:

\[
A^*\in\arg\min_{A'\subseteq A}
\sum_{a_k\in A'}c_k
\]

subject to complete separation of all decision-conflict pairs. This is a minimum set-cover problem over the pair universe. V1 enumerates all non-empty subsets of at most 20 actions (1,048,575 subsets) and caps submitted worlds at 24 (276 possible unordered pairs).

## 3. Result states

- MINIMUM_COST_DISAMBIGUATION_PLAN: every decision-conflict pair is separated and the exact minimum action set is within the declared effort budget.
- MINIMUM_COST_DISAMBIGUATION_EXCEEDS_BUDGET: a complete minimum plan exists, but its effort is above the declared budget. The plan remains visible for review.
- NO_COMPLETE_DISAMBIGUATION_WITHIN_DECLARED_ACTIONS: at least one opposing-decision pair has identical declared outcomes under every candidate action.
- NO_DECISION_DISAGREEMENT_WITHIN_DECLARED_SCENARIOS: every submitted world yields the same query decision. No additional action is needed to resolve disagreement within this world set; this does not prove the world set is complete.

Undiscriminable pairs are always returned explicitly. The result is never called a posterior, a confidence level, an expected value of information or proof of biological identifiability outside the supplied worlds.

## 4. Observation-model and dependence contract

Every action must define exactly one non-empty outcome category for every submitted scenario. These outcomes are deterministic declarations from a frozen observation model, measurement protocol or adjudication rubric. V1 does not admit likelihoods, uncertainty distributions or noisy outcome models. A noisy measurement requires a separately justified probability/measurement model and a different estimator.

Actions bind source-evidence and observation-model SHA-256 digests. Shared ancestry group IDs are retained so reviewers can see when actions reuse a volume, detector, reviewer panel or annotation lineage. The planner does not assume that actions are independent; it also does not quantify dependence or prove that the outcome mappings are accurate. A digest identifies supplied bytes; it does not establish that an acquisition or reviewer actually followed the protocol.

## 5. Synthetic known-answer fixture

The primary fixture declares four worlds:
- alpha: query decision met;
- beta: query decision not met;
- gamma: query decision not met;
- delta: query decision met.

There are four decision-conflict pairs: alpha-beta, alpha-gamma, beta-delta and gamma-delta.

Five observation actions are supplied:
- inspect-synapse, effort 5, separates all four pairs;
- inspect-morphology, effort 2, separates alpha-beta and gamma-delta;
- trace-morphology-review, effort 2, covers the same two pairs under a distinct declared observation method;
- reviewer-panel, effort 2, separates alpha-gamma and beta-delta;
- image-sharpness-decoy, effort 1, gives the same outcome in all worlds.

The minimum complete plans are:
- inspect-morphology + reviewer-panel; or
- trace-morphology-review + reviewer-panel.

Expected result: four conflicting pairs, all individually discriminable, two tied minimum plans, minimum effort 4, both within the four-minute budget. The single inspect-synapse action is a valid complete discriminator at cost 5, but it is not minimal.

The example deliberately records a common raw-volume ancestry label for multiple actions. The planner does not turn that into an independence claim.

## 6. Negative control: no action distinguishes a conflicting pair

The second fixture contains one world supporting the query and one refuting it. Its only candidate action returns the same “indeterminate” category for both worlds. The correct status is NO_COMPLETE_DISAMBIGUATION_WITHIN_DECLARED_ACTIONS, with the pair explicitly listed and no minimum plan.

This prevents the planner from claiming that every uncertainty can be resolved simply because an observation budget exists.

## 7. Independent verification

The Python standard-library oracle:
- recomputes the canonical scenario-set and action-set hashes;
- independently reconstructs all decision-conflict pairs;
- computes each action's covered pairs;
- enumerates action subsets using itertools.combinations rather than Rust bit-mask/bitset set-cover search;
- recomputes minimum costs, tied plans, budget disposition and undiscriminable pairs;
- rejects deliberately mutated costs, action membership, coverage counts and no-plan dispositions.

The oracle is test-only. It is not imported into production Rust code or used to produce the plan.

## 8. Exact optimality oracle versus greedy set cover

The report includes a deterministic weighted-greedy baseline that repeatedly selects the remaining action with the highest marginal number of newly separated decision-conflict pairs per effort unit. Ties prefer greater marginal coverage, then stable lexical action ID. This is a comparator only; the exact subset search remains authoritative.

For a complete weighted set-cover instance, this standard greedy rule has the classical harmonic approximation guarantee (H_d), where (d) is the maximum number of universe elements covered by any one action (Chvátal's weighted greedy set cover; see [Young's survey](https://www.cs.ucr.edu/~neal/publication/Young16SetCover.pdf)). In V1 there are at most 24 worlds, and at most 12 positive × 12 negative = 144 opposing-decision pairs, so (d le 144) and the worst-case bound is (H_{144} approx 5.55). This is deliberately conservative; exact-vs-greedy regret is measured on each fixture and generated benchmark. No complete-cover approximation guarantee applies when the submitted action universe cannot distinguish every opposing-decision pair.

The primary fixture deliberately demonstrates approximation regret:
- a three-unit broad scan separates three of four conflict pairs;
- a two-unit morphology review separates the remaining pair;
- greedy total effort is 5;
- the exact optimum uses two complementary two-unit actions, total effort 4;
- greedy regret is therefore 1 effort unit, while both plans are complete.

The oracle independently recomputes this greedy trace and regret. Future benchmarks should also compare greedy set cover, cost-aware greedy variants, beam search, branch-and-bound and ILP/SMT against the exact solver over generated worlds/action matrices. Report plan completion, pair coverage, cost regret, action count, execution time and memory. Do not promote greedy as optimal or interpret its partial plan as complete.

## 9. Benchmarks and promotion criteria (proposed, not results)

| Measure | Proposed target |
|---|---:|
| Exact minimum-cost plan recovery on enumerated cases (≤20 actions) | 100% |
| Regret against independent set-cover oracle | 0 |
| Missed decision-conflict pairs in a returned plan | 0 |
| False pair separation when all declared action outcomes agree | 0 |
| Undiscriminable-pair reporting | 100% |
| Tied optimal plans counted | 100% |
| Deterministic output under world/action/outcome input reordering | 100% |
| Probability/entropy claims for deterministic unweighted worlds | 0 |

For larger action sets, compare branch-and-bound, ILP/SMT, beam search and greedy set-cover approximations against the exact 20-action oracle. Report optimality gaps, runtime, memory, number of action evaluations, fraction of decision-conflict pairs separated, and cost under identical outcome mappings.

For biological promotion, additionally measure whether the declared outcome mapping predicts actual independent review or acquisition results, whether action provenance is complete, whether shared ancestry changes reliability, and whether a blinded expert confirms that the chosen observation can genuinely discriminate the competing reconstructions.

## 10. Integration boundaries

- #9987: this tool supplies deterministic minimum-cost discrimination plans without requiring probability assumptions.
- #9344: worlds and query disagreement must come from a frozen reconstruction set and query result.
- #10502: a minimal query-flipping witness can generate observation hypotheses, but the planner needs explicit outcome mappings for every world.
- #9255: a future certificate may bind the scenario/action set hashes, exact plan receipt, verifier identity and scope. A plan does not itself certify a scientific claim.
- #10406, #10435 and #10453: evidence dependence, frame closure and execution-route observability remain separate conjunctive gates.
- Probability-based VOI remains a separate extension. It needs calibrated outcome likelihoods and a declared scientific decision-loss function. Do not multiply a finite unweighted world list into a probability distribution.

## 11. Non-claims

The tool does not prove that an observation will be acquired, that an outcome mapping is scientifically correct, that a candidate action is independent of other evidence, that submitted worlds exhaust biological possibilities, or that any action has positive expected utility. It ranks only deterministic separation coverage under supplied mappings and effort costs. No biological observation or correction is executed.
