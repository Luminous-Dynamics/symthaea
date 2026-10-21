# Task-Conditioned Counterfactual Integration Profile (TCCIP), v0

**Status:** Candidate research proposal. This is not an established metric, not evidence of novelty, not an IIT substitute, and not a consciousness measure. Its definitions and predictions are deliberately falsifiable. Do not promote a single scalar to production until the fixture suite discriminates it from simpler baselines.

## Executive proposal

Symthaea should investigate a measure aimed at a different question from IIT Φ:

> Given a declared task and environment distribution, do cross-component causal interactions make a measurable, reproducible contribution to competent behavior—and how does that contribution trade off against fragility and recovery?

IIT asks a theory-specific question about intrinsic cause-effect structure. Entropy measures uncertainty; mutual information measures dependence under a distribution; graph/spectral measures describe connectivity. None should be relabeled as task competence. This candidate makes **task performance under counterfactual interventions** the outcome and reports integration, baseline capability, robustness, and recovery as a profile.

The working name is **Task-Conditioned Counterfactual Integration Profile (TCCIP)**. The name is provisional and makes no priority or novelty claim. The research contribution, if it survives tests, would be the operational definition plus matched intervention controls and the evidence protocol—not merely attaching a new label to edge ablation.

## Why investigate this alongside Φ?

A high integration value need not imply that a system performs a useful task well. Conversely, a modular system with redundancy may lose little utility after one cross-component link is cut, despite relying on rich coordination under harder interventions. Those are different questions.

We should retain formal IIT calculations as a separate comparison where relevant. A task-oriented metric must not claim to replace IIT on IIT's own theoretical target. It can be better for an engineering question only if the benchmark demonstrates that it predicts or diagnoses the engineering outcome better.

## Operational setup

Before evaluating a candidate system, specify and version:

- a causal transition model or an independently supported intervention interface;
- components and cross-component links, with the partition chosen independently of evaluation outcomes or frozen on a discovery split;
- task/environment distribution \(\mathcal D\), initial-state distribution, random seeds, and evaluation horizon;
- bounded task utility \(J\in[0,1]\), or a documented normalization for another utility scale;
- admissible edge/component interventions and their costs;
- a matched-control generator for sham/random interventions;
- a held-out environment split and an uncertainty procedure.

If the causal graph or intervention semantics are unknown, report that limitation and do not silently treat correlation or ordinary feature masking as a causal intervention.

## Candidate quantity: cross-component counterfactual task contribution

Let \(M\) be the system, \(\mathcal D\) the frozen task/environment distribution, and \(J(M;\mathcal D)\) its expected task utility under a declared rollout protocol.

For an intervention set \(C\), define the task loss:

\[
L(C)=J(M;\mathcal D)-J(M\mid do(C);\mathcal D).
\]

Here \(do(C)\) must name an executable intervention—for example, replace selected cross-component messages by a predeclared null signal while keeping all other execution rules fixed. “Remove an edge” is insufficient unless the affected transition probabilities, timing, and downstream behavior are specified.

For cut budget / intervention stratum \(k\), draw interventions from two frozen distributions:

- \(\mathcal C^{cross}_k\): cross-component causal cuts;
- \(\mathcal C^{matched}_k\): sham or control interventions matched on intervention count/cost, node/edge degree where possible, target interface, and intervention mechanism, but not selected for task outcomes.

Define the candidate contrast:

\[
\operatorname{TCCIP}_k =
\mathbb E_{C\sim\mathcal C^{cross}_k}[L(C)]
-
\mathbb E_{C'\sim\mathcal C^{matched}_k}[L(C')].
\]

With utility bounded to [0,1], this contrast lies in [-1, 1]. Positive values mean the chosen cross-component cuts damage task utility more than matched controls; near zero means no detected excess effect under this design; negative values mean those cuts are less harmful than the controls or even improve performance. Negative values are valid results and must not be clamped to zero.

**Important:** this is a contrast in task effect, not a pure property of topology. It is conditional on task, environment, intervention semantics, partition, rollout horizon, and control distribution. Report those alongside the contrast. If matching is poor, report the mismatch and treat the estimate as exploratory.

## Do not reduce the profile to one number

The proposed output vector should include:

1. **Intact competence:** \(J_0=J(M;\mathcal D)\). Prevents “integrated but useless” systems from ranking well.
2. **Counterfactual contribution curve:** \(\operatorname{TCCIP}_k\) for multiple intervention budgets/cut strata.
3. **Worst-case utility:** minimum or a declared lower quantile of task utility over admissible interventions at cost \(k\), with the search procedure reported.
4. **Redundancy/recoverability:** task utility after one or more independent paths are removed, and utility after an explicit repair/recovery protocol.
5. **Distribution-shift stability:** the same profile on held-out environments, state priors, noise levels, and task variants.
6. **Uncertainty and execution cost:** confidence interval / bootstrap procedure, number of independent seeds and environments, runtime, memory, and intervention-evaluation budget.

These dimensions can conflict. A tightly coupled controller can have a strong positive TCCIP curve and poor fault tolerance. A modular redundant system may have low single-cut loss and higher multi-cut resilience. Neither should be declared universally superior. Publish the vector and Pareto frontier; use a composite only for a named decision with pre-registered weights and sensitivity analysis.

## Hypotheses that would make this worth keeping

- **H1: task relevance.** Cross-component interactions that are irrelevant to the evaluated task yield TCCIP near zero after matched controls, even if graph connectivity is high.
- **H2: useful coordination.** When a task genuinely requires coordination across a frozen component boundary, cross-component cuts cause larger utility loss than matched controls, giving positive TCCIP.
- **H3: synergy.** Tasks with XOR-like or other joint-input requirements exhibit a different cut-response profile from systems whose task inputs act independently. We must compare against well-defined PID/SURD or conditional-information baselines rather than infer synergy from a positive cut score alone.
- **H4: confounding resistance.** An observational common-driver fixture does not receive false cross-component causal credit when interventions sever only a non-causal association.
- **H5: redundancy.** Duplicate, independently sufficient paths can preserve utility after single cuts, while coordinated multi-cut interventions may reveal dependence. The profile should show this difference without equating resilience with absence of integration.
- **H6: harmful coordination.** If cross-component interactions degrade task utility, TCCIP can be negative. This is a feature, not a failure of sign convention.
- **H7: transfer.** A profile estimated on the discovery environments predicts intervention-induced utility degradation in frozen held-out environments better than graph density, spectral connectivity, entropy, raw MI, and pairwise-only baselines.

A result that fails any hypothesis should narrow the claim and possibly reject this candidate. Do not adjust definitions after inspecting held-out results.

## Minimal adversarial fixture suite

| Fixture | What it tests | Required result/property |
|---|---|---|
| Independent noise / irrelevant cross-links | Connected graph versus task relevance | Graph density may rise while task-conditioned contribution remains near zero |
| Cross-component XOR task | Joint dependence | Cross-boundary cuts should damage utility more than properly matched controls |
| Common driver / collider | Correlation versus intervention | Avoid attributing the driver's shared variation to a direct cross-component edge |
| Useful coordination | Task-relevant communication | Positive contribution on the predeclared task distribution |
| Harmful coupling | Bad coordination | Negative contribution is retained and correctly interpreted |
| Redundant dual path | Resilience versus dependence | One-cut utility remains high; multi-cut curve exposes joint reliance |
| Edge-count/degree-matched sham cuts | Generic damage control | Contrast is not driven only by cutting more/high-degree edges |
| Prior and environment shift | Generalization | Report the full degradation curve and uncertainty; no evaluation-set retuning |
| No-op intervention | Instrumentation validity | Zero effect up to numerical tolerance |
| Randomized/disconnected negative control | False-positive control | No unexplained positive task-causal credit |
| Node/label permutation | Representation dependence | Invariant results remain invariant, if the intervention/task mapping is permuted consistently |
| Unequal rollout/runtime budget | Measurement confound | Reject comparisons with materially unequal budget or normalize prospectively |

For stochastic systems, repeat across independently generated environment/rollout seeds. Do not count timesteps from one trajectory as independent experimental replicates.

## Baselines that must be beaten or complemented

1. **Task-only baseline:** intact task utility and task utility after generic resource degradation. TCCIP must explain something beyond task success itself.
2. **Edge ablation / causal effect baseline:** unadjusted cross-cut task loss. The matched-control contrast must improve interpretation, not merely add complexity.
3. **Graph density, degree, spectral connectivity:** structural baselines.
4. **Entropy, mutual information, conditional mutual information, transfer entropy:** information/predictive baselines.
5. **PID / SURD or another explicitly specified information decomposition:** where the fixture supports a well-defined comparison.
6. **Effective information / causal-emergence baselines:** where transition models, interventions, and state distributions satisfy their assumptions.
7. **PyPhi under a pinned named IIT formalism:** a separate reference track, not a task-utility gold standard.
8. **Ablation-matched randomized control:** tests whether the proposed quantity adds anything over a simpler intervention experiment.

Do not compare numbers across incompatible units and call the largest the winner. Compare each method on the outcome it claims to predict, using held-out systems and preregistered evaluation.

## Qualification gates

**Gate 0 — executable semantics**
- A complete public definition of each intervention and task rollout.
- Utility normalization, intervention/control sampling, partition, prior, and horizon frozen in a versioned manifest.
- Reproducible results from fixed seeds; exact model, task and intervention digests in receipts.

**Gate 1 — implementation correctness**
- A deterministic toy fixture with an analytically calculable task-utility difference.
- No-op invariance and consistent node-label permutation.
- Matched-control tests with deliberately mismatched controls rejected or marked unqualified.
- Uncertainty intervals verified on simulated nulls; false-positive rate assessed over independent generated systems, not just one handpicked topology.

**Gate 2 — construct validity**
- H1–H6 tested on a preregistered synthetic fixture set.
- TCCIP compared with the simpler unadjusted cut-loss and all baselines above that apply.
- Ablation and null controls show that positive values do not arise merely from dense graphs, high-degree components, common drivers, or generic damage.

**Gate 3 — held-out predictive validity**
- Freeze the method and thresholds using the discovery split.
- Test the association between the precomputed profile and realized utility degradation on unseen dynamics and environment families.
- Report performance with uncertainty, failed/ambiguous cases and calibration. A correlation is not enough; compare predictive errors and calibration against baselines.

**Gate 4 — useful deployment**
- On a real Symthaea control or planning task, show that the profile helps select a more reliable architecture, identify harmful coupling, or allocate testing/repair effort compared with simpler baselines.
- Measure task performance, compute overhead, fault tolerance and transfer under a fixed budget.
- Until this gate passes, the metric remains research-only and must not affect production control or any consciousness-related output.

## Why it may fail

- Task utility may be the right engineering target, but a poor substitute for intrinsic integration.
- TCCIP depends on task selection; adversarial reward shaping can reverse rankings.
- Partitions and matched controls may be ambiguous or impossible to construct fairly.
- Edge-cut results can be artifacts of the chosen null-signal or simulator, not a stable system property.
- Redundancy may hide integration under single cuts and appear only under compound interventions.
- A complicated matched-control protocol may not outperform the simpler causal-ablation curve enough to justify its costs.

These are testable reasons to reject or narrow the proposal, not caveats to bury after favorable results.

## Relationship to prior work and novelty boundary

Relevant prior families already include effective information and causal emergence, information-geometric causal integration (including Φ_CIS/Φ_CII), partial information decomposition, and SURD's synergistic/unique/redundant decomposition of causality. Recent state-aware causal-inference work also emphasizes that causal influence can depend on the specific system state.

This proposal is therefore **not claimed to be the first causal integration metric**. Its candidate distinction is the task-utility estimand plus matched counterfactual cut controls, explicit resilience/recovery dimensions, and a fail-closed benchmark receipt. A literature review and implementation comparison must establish whether that combination adds a measurable capability before claiming novelty.

References for the first research pass:
- Tononi & Sporns, “Measuring information integration” (2003): https://pmc.ncbi.nlm.nih.gov/articles/PMC331407/
- Langer & Ay, “Complexity as Causal Information Integration” (2020): https://arxiv.org/abs/2008.11430
- Martínez-Sánchez et al., “Decomposing causality into its synergistic, unique, and redundant components” (SURD, 2024): https://www.nature.com/articles/s41467-024-53373-4
- “Observational causality by states and interaction type for scientific discovery” (2025): https://www.nature.com/articles/s42005-025-02447-w
- PyPhi formalism differences: https://pyphi.readthedocs.io/en/latest/theory/formalism-versions.html

## Prototype status (2026-10-21)

Added `examples/task_conditioned_counterfactual_profile.rs`, a dependency-free Rust scoring kernel that consumes precomputed intact/cross-cut/matched-control utilities and emits a budget-wise profile. Seven test definitions cover positive, zero, and negative contrasts, cumulative budgets, missing-control fail-closed behavior, invalid utility, duplicate IDs, and budget ordering.

This is intentionally only the **summary/scoring layer**. It does not run a causal simulator, generate matched controls, establish that a provided cut is a valid intervention, estimate uncertainty, or demonstrate usefulness. Example data in `main` are illustrative values only, not empirical or simulation results. Rust is unavailable in the current environment, so the example and tests have not been compiled or executed. The next step remains an end-to-end finite-state toy evaluator that produces raw utility outcomes from explicit model/task/intervention fixtures.

## Exact finite-state evaluator: reproduction commands

On a machine with the repository's Rust toolchain:

```bash
git rev-parse HEAD
cargo test --example task_conditioned_finite_state_eval
mkdir -p results
SYMTHAEA_SOURCE_SHA="$(git rev-parse HEAD)" \
  cargo run --release --example task_conditioned_finite_state_eval \
  > results/tccip_exact_finite_state_v1.tsv
sha256sum results/tccip_exact_finite_state_v1.tsv
```

The output includes canonical fixture definitions and one row per intervention plus a profile summary. If `SYMTHAEA_SOURCE_SHA` is absent, the artifact labels the source revision `UNBOUND` and is exploratory only. Preserve the complete output, exact source SHA, test output, and artifact digest together. The run is an exact two-input reference, not a large-scale benchmark.

## First exact finite-state evaluator prototype (2026-10-21)

Added `examples/task_conditioned_finite_state_eval.rs`. Unlike the earlier scorer, this evaluator computes intact and intervention utility by exhaustive enumeration of all two-bit exogenous inputs and both outcomes of the fair-coin replacement signal. The cut is applied to a declared edge in an acyclic Boolean rule graph; whether it crosses the declared two-component partition is derived from endpoint labels, not supplied by the score caller. Raw intervention outcomes and aggregate profile rows are emitted separately.

The first fixtures test three deliberately different situations:

| Fixture | Intact utility | Cross-cut utility | Matched-control utility | Expected TCCIP |
|---|---:|---:|---:|---:|
| XOR coordination | 1.0 | 0.5 | 1.0 | +0.5 |
| Irrelevant cross-component branch | 1.0 | 1.0 | 1.0 | 0.0 |
| Harmful constant-bias coupling | 0.0 | 0.5 | 0.0 | -0.5 |

These exact expectations were independently checked with a separate, small enumerator. **That is not a Rust compilation or test pass.** Eight Rust test definitions now cover the three expected contrasts, edge/partition semantics, fail-closed missing strata, malformed edges/topology, repeatability/bounds, and invalid IDs/costs. Rust/cargo are unavailable in the current execution environment, so this example remains uncompiled and unexecuted.

These are conformance fixtures, not evidence of empirical usefulness. The controls are deliberately simple matched edge-signal replacements; they do not yet solve all practical matching problems involving degree, delay, bandwidth, component role, or network centrality. The harmful-coupling fixture also begins with zero task accuracy, so it demonstrates the required sign behavior, not a competitive architecture. A real evaluation must report intact capability and matched-control quality alongside the profile and use held-out systems.

### New result: partition choice can reverse the score

The evaluator now enumerates every nontrivial bipartition of each small network, fixing node 0 to side A to avoid counting complementary partitions twice. For each partition, the same predeclared edge interventions are reclassified by their endpoint labels. Partitions with no cross-cut or no internal-control observations are explicitly counted as skipped rather than receiving fabricated scores.

A separate exhaustive Python enumerator (used as a cross-check, **not a Rust test run**) gives:

| Fixture | Nontrivial partitions | With both strata | Skipped | TCCIP range across usable partitions |
|---|---:|---:|---:|---:|
| XOR coordination | 127 | 112 | 15 | [-0.5, +0.5] |
| Irrelevant branch | 15 | 8 | 7 | [0.0, 0.0] |
| Harmful coupling | 511 | 448 | 63 | [-0.5, +0.5] |

This is an important warning, not a result to hide: for the same fixed toy causal model and intervention outcome table, the contrast can reverse sign with the partition. The declared partitions recover the expected +0.5, 0.0 and -0.5 fixture scores, but these score ranges show that the contrast is **partition-relative**, not an intrinsic scalar property of the network. This is why the metric must expose the full partition profile and freeze partition selection before held-out evaluation. It also makes independent control selection and matched-stratum quality central requirements. The toy sweep still does not prove that TCCIP is useful or better than existing methods.

Twelve Rust test definitions are now authored, including exact expected sign reversals and the invariant that the declared partition appears in the exhaustive sweep. The source remains uncompiled and unexecuted in this environment.

## Literature-aware scope and a stronger next paradigm

The initial literature check confirms that the broad territory is populated: information-geometric integrated-information measures such as Φ_CIS/Φ_CII, causal emergence/effective information, and SURD already address related questions about causal integration, scale, and synergistic/unique/redundant contributions. SURD is designed to decompose future information contributions, including hidden-variable "causality leak"; see [Martínez-Sánchez et al. (2024)](https://www.nature.com/articles/s41467-024-53373-4). Information geometry already defines integrated information as a divergence from a constrained causally disconnected model; see [Oizumi et al. (2016)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5187746/). Recent work on causal emergence also argues that a single maximizing scale omits the distribution of causal contribution across multiple scales; see [A reframed landscape of causal emergence (2025)](https://doi.org/10.1016/j.patter.2025.101476).

Therefore, the strongest prospective direction is not just task-conditioned edge ablation. It is an **interventional task-relevance landscape across multiple partitions, states, and budgets**, with:
- a profile over multiple candidate partitions / coarse-grainings rather than a single chosen cut;
- separate discovery and held-out partitions to reduce selection bias;
- matched interventions and explicit intervention-support diagnostics;
- state-conditional profile curves to find mechanisms that help in some regimes and harm in others;
- synergy/redundancy controls compared directly against relevant PID/SURD outputs when assumptions permit;
- task utility and fault recovery retained as outcomes, not re-labeled as intrinsic Φ.

This only becomes a contribution if those additions improve held-out prediction or engineering decisions beyond raw causal-ablation curves and existing multiscale / information-theoretic baselines. TCCIP is a provisional engineering estimand, not a replacement for the target of IIT or a measure of consciousness.

## Immediate implementation plan

1. Extend the now-added exact evaluator to a declarative fixture manifest with separate model, task, partition, and intervention identities.
2. Sweep predeclared alternate partitions / coarse-grainings and retain the full multiscale profile; do not select the best partition on the evaluation set.
3. Add state-conditioned and controlled prior-shift fixtures, plus degree/delay/bandwidth-matched intervention controls where meaningful.
4. Compare against raw task ablation, graph/spectral measures, entropy/MI/transfer entropy, and relevant PID/SURD/effective-information baselines.
5. Test the preregistered adversarial matrix, then connect a real Symthaea task only after exact synthetic gates pass.
6. Keep the feature research-only until external review and evidence support a concrete engineering use.

**Current evidence status:** specification only. No implementation accuracy, novel-metric claim, IIT correspondence, consciousness inference, SOTA result, or passing test is asserted.
