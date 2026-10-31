# PITB probabilistic dependency-aware support landscape (2026-10-31)

**Status:** research implementation, trial schema, and focused workflow. This is an extension to the deterministic support-antichain pilot, not a replacement for the primary transfer endpoint. It has no real Lean rollout ledger or model result yet.

## Why deterministic ablation is not enough

A theorem prover driven by an LLM or adaptive search policy is stochastic. A single failed attempt with a context does not establish that the context is insufficient; it may be a search failure, unlucky sampling, or budget exhaustion. Conversely, one successful proof does not show that a context is reliably sufficient.

There is a second issue: Lean local contexts are ordered. The official Lean API describes LocalContext as an ordered set of local declarations, and LocalDecl.index records a declaration's position. Later declarations can depend on earlier declarations. Arbitrary subsets are therefore not all valid proof states. [Lean.LocalContext API](https://lean-lang.org/doc/api/Lean/LocalContext.html). Lean's MVarId.cleanup removes declarations based on dependency relevance to the target, preserved declarations, and let values; that is a useful context-pruning baseline, but not an empirical measure of a bounded prover's success probability. [Lean cleanup API](https://lean-lang.org/doc/api/Lean/Meta/Tactic/Cleanup.html).

The new landscape therefore separates:

1. **Context validity:** does the requested set form a well-formed local context under the pinned telescope/dependency graph?
2. **Operational proof reliability:** across independent paired seeds, how often does the frozen prover complete the exact target, with kernel and semantic-gate acceptance?
3. **Efficiency:** what is the cost of each reliable context in tokens, latency, calls and spend?

An invalid context is not a failed mathematical proof. It is excluded from support sufficiency and reported separately.

## Executable artifacts

- Analyzer: scripts/research/pitb_support_landscape_v1.py
- Trial schema: data/benchmarks/proof-insight-transfer-v0/support-landscape-trial.schema.json
- Focused workflow: .github/workflows/pitb-support-landscape.yml

The schema describes one row per requested context and replicate. The analyzer adds cross-row invariants that JSON Schema cannot express.

### Frozen identity and context contract

Every group binds the exact theorem statement, target, model ID/version, checker version, Lean environment/toolchain, fixed context, solver policy, resource budget and replicate count. The declaration manifest records:
- a mutable declaration universe;
- fixed declarations in the non-mutated context;
- the original telescope order;
- direct prerequisites for each mutable declaration;
- the exact requested subset.

For selected set S, fixed set F, and prerequisite relation D(x), the requested context is valid only when:

\[
\forall x \in S,\quad D(x) \subseteq S \cup F
\]

The dependency graph must respect telescope order: mutable prerequisites precede the declaration that refers to them. The complete powerset is still accounted for, but dependency-invalid subsets are explicitly recorded as invalid_context, with no solver attempt and zero solver/checker spend. They do not count as mathematical non-solves.

### Repeated-seed reliability

Every requested subset is run under the same frozen replicate schedule. Replicate seed digests must match across contexts, so each context comparison is paired by replicate index. The analyzer requires the complete cross-product of all \(2^{|D|}\) requested subsets and all replicate indices exactly once.

A valid context's observed success probability is:

\[
\hat p(S)=\frac{\text{kernel + semantic verified successes}}{\text{all scheduled attempts}}
\]

Timeouts and ordinary non-solves remain in the denominator. Missing trials, unknown status, tool/checker unavailability, digest changes, or out-of-budget runs reject the ledger rather than being silently reclassified. A verified_success requires both trusted-kernel and independent semantic-gate success.

### Simultaneous reliability qualification

The reliability threshold is not based on one success or a per-context unadjusted interval. For each well-formed context, the analyzer computes a one-sided Wilson lower bound with a Bonferroni adjustment over **all well-formed contexts in the frozen lattice**. A context qualifies only when:
- the number of replicates is at least the frozen minimum; and
- its simultaneous lower bound is at least the preregistered reliability floor.

The default reliability floor is 0.80, the default minimum is 20 attempts per context, and the default simultaneous familywise alpha is 0.05. Twenty attempts is a pilot floor, not a recommended final sample size; 50–100 paired seeds per context is a more useful calibration range for the eight-variable pilot.

The **reliable minimal-support antichain** is the set of inclusion-minimal contexts that pass this simultaneous reliability gate, conditional on the full context itself passing the same gate. The output separately reports:
- empirical success rate and ordinary two-sided Wilson interval;
- Bonferroni simultaneous one-sided lower bound;
- invalid-context count;
- timeouts and non-solves;
- observed non-monotonicity across nested contexts;
- Pareto-efficient reliable contexts across declaration count, tokens, latency and micro-USD spend;
- cost per verified success.

The Pareto frontier matters: the smallest reliable context may not be the cheapest context. Keep those questions separate.

### No unjustified pass@k extrapolation

The report gives direct repeated-seed frequencies and their intervals. It does not extrapolate to pass@k beyond observed replicates. The September 2026 analysis *What Fixed-Rollout pass@k Evaluations Can Identify* shows that fixed-n success counts do not nonparametrically identify generic pass@k for k>n; such forecasts require explicit additional assumptions and must be labelled model-conditioned forecasts, not measurements. [Paper](https://arxiv.org/abs/2609.09245).

## Paradigm shifts and research hypotheses

1. **From a premise list to a valid proof-state manifold.** Contexts are not arbitrary bags of hypotheses; they are ordered telescopes with dependency-closed subcontexts. Learn and search over well-formed states, not impossible ablations.
2. **From a binary support label to a reliability surface.** A route may work 98% of the time, 50% of the time, or only under a particular resource budget. Report the whole observed probability/cost surface.
3. **From one proof route to a risk-aware strategy portfolio.** Alternative routes form an antichain, but their reliability and cost can differ. The controller should prefer robust low-cost routes and retain an alternate when the primary route is fragile.
4. **From static retrieval to adaptive proof search.** A 2026 context-dependent Lean search study reports +12.8 percentage points average pass rate and 21.9% fewer LLM calls than pass@k baselines within a pass@32 budget on seven miniCTX-v2 projects. This is a strong comparator for seed scheduling, diversity and current-best refinement—not a direct PITB result. [Paper](https://arxiv.org/abs/2608.18084).
5. **From rebuilding proofs to localized repair.** EditableSketch supports incremental edits while preserving previously proved subgoals and reports 76.0% pass rate on FormalMath-Lite in its setup. BlueprintRepair separately compares typed local edits, source patches and full rewrites on 142 controlled Lean blueprint failures. PITB should compare these as repair systems, while keeping repair success distinct from support reliability. [EditableSketch](https://proceedings.mlr.press/v306/xiao26q.html) · [BlueprintRepair](https://aclanthology.org/2026.mathnlp-main.3/).

## Competitive evaluation plan

Run the same context lattice, seed schedule, target, prover, checker and budgets across:
- full context;
- Lean MVarId.cleanup baseline;
- direct type/value dependency closure;
- minimal support antichain search;
- typed dependency-graph retrieval;
- HDC/LTC-ranked context selection;
- random valid-context selection;
- oracle-minimal support as an upper-bound diagnostic (never a deployable baseline).

Report reliability/cost Pareto curves and the simultaneous reliable-support antichain. Then test whether selecting from the predicted landscape improves exact-target solve probability or reduces cost against the strongest validation-selected policy.

A useful target for the prediction system is to recover a reliable context with at least 0.80 simultaneous lower-bound success while minimizing context size and cost. A useful target for the controller is at least 10% lower spend per verified success than the best fixed-context baseline without reducing the reliability floor. These are proposed qualification goals, not achieved results.

The model must be evaluated on held-out audited components. The 372-family / 961-path artifact split is not yet a qualified independent mathematical holdout because real cross-family dependency and reviewer-coverage evidence remains incomplete.

## Synthetic controls and limitations

The analyzer self-test includes:
- disjunctive support: [{h}, {hEq}];
- conjunctive support: [{h1,h2}];
- dependency invalidation: removing base while retaining dependent yields an invalid context, not a proof failure;
- repeated-seed 48/50 reliability and 45/50 below-floor examples;
- incomplete lattice, seed mismatch, changed fixed context, out-of-budget attempts, semantic pass without kernel pass, and false dependency-closure controls.

These are synthetic contract tests. Receipt digests are syntax-checked and row-bound, but receipt bytes still require independent authentication. The dependency graph must be generated from the actual Lean elaborated context (including types and available let-values), not manually guessed. No real support landscape, Lean build, model prediction, transfer gain, or SOTA win is claimed.
