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

Added `examples/task_conditioned_finite_state_eval.rs`. Unlike the earlier scorer, this evaluator computes intact and intervention utility by exhaustive enumeration of all two-bit exogenous inputs and both outcomes of the fair-coin replacement signal. The expected fixture parameters, input-state weights, intervention edges, and oracle outputs are recorded in schema-v3 `validation/fixtures/tccip_toy_expectations.tsv`; both the Rust evaluator and independent Python oracle parse it. The cut is applied to a declared edge in an acyclic Boolean rule graph; whether it crosses the declared two-component partition is derived from endpoint labels, not supplied by the score caller. Raw intervention outcomes and aggregate profile rows are emitted separately.

The first fixtures test three deliberately different situations:

| Fixture | Intact utility | Cross-cut utility | Matched-control utility | Expected TCCIP |
|---|---:|---:|---:|---:|
| XOR coordination | 1.0 | 0.5 | 1.0 | +0.5 |
| Irrelevant cross-component branch | 1.0 | 1.0 | 1.0 | 0.0 |
| Harmful constant-bias coupling | 0.0 | 0.5 | 0.0 | -0.5 |

These exact expectations were independently checked with a separate, small enumerator. **That is not a Rust compilation or test pass.** The Rust example now has 19 test definitions covering exact toy contrasts, partition sensitivity/sign reversal, prior-dependent score changes, a higher-order XOR-information counterexample, malformed interventions, fail-closed strata, determinism, bounds, and manifest parity. Rust/cargo are unavailable in the current execution environment, so this example remains uncompiled and unexecuted.

These first three fixtures are conformance/sign tests, not evidence of empirical usefulness. In particular, the XOR and harmful-coupling cases originally compared task-output edges with controls on an unused side branch; that is not a fair role-matched effect estimate. They are retained as negative/sign-convention controls only. The new AND prior pair improves this design because both tested edges enter the same task-output gate, but it still does not solve all matching issues involving degree, path redundancy, latency, bandwidth, or network centrality. A real evaluation must report intact capability and per-edge effects alongside the profile and use held-out systems.

### New result: partition choice can reverse the score

The evaluator now enumerates every nontrivial bipartition of each small network, fixing node 0 to side A to avoid counting complementary partitions twice. For each partition, the same predeclared edge interventions are reclassified by their endpoint labels. Partitions with no cross-cut or no internal-control observations are explicitly counted as skipped rather than receiving fabricated scores.

A separate exhaustive Python enumerator (used as a cross-check, **not a Rust test run**) gives:

| Fixture | Nontrivial partitions | With both strata | Skipped | TCCIP range across usable partitions |
|---|---:|---:|---:|---:|
| XOR coordination | 127 | 112 | 15 | [-0.5, +0.5] |
| Irrelevant branch | 15 | 8 | 7 | [0.0, 0.0] |
| Harmful coupling | 511 | 448 | 63 | [-0.5, +0.5] |
| Role-matched AND, high \(x_1\) activity prior | 15 | 8 | 7 | [-0.3, +0.3] |
| Same AND system, reversed input prior | 15 | 8 | 7 | [-0.3, +0.3] |
| Role-matched XOR synergy control | 15 | 8 | 7 | [0.0, 0.0] |

This is an important warning, not a result to hide: for the same fixed toy causal model and intervention outcome table, the contrast can reverse sign with the partition. The declared partitions recover the expected +0.5, 0.0 and -0.5 fixture scores, but these score ranges show that the contrast is **partition-relative**, not an intrinsic scalar property of the network. This is why the metric must expose the full partition profile and freeze partition selection before held-out evaluation. It also makes independent control selection and matched-stratum quality central requirements. The toy sweep still does not prove that TCCIP is useful or better than existing methods.

The 19 authored Rust test definitions include partition sign reversals, input-prior effects, XOR-information baselines, and schema-v3 manifest parity. The independent Python oracle has 16 authored unittest definitions. Neither committed suite has been executed here; Cargo/rustc are unavailable, and the committed Python module/tests have not themselves been run. A separate ad-hoc enumerator independently checked all six manifest fixture utilities, TCCIP values, partition ranges, and the XOR information calculation, but this is not a test run of either committed implementation.

## State-prior conditioning with role-matched task-gate cuts (new experiment)

The previous positive fixture compared a task-relevant edge against an edge into an unused branch. That contrast was too easy to game: different task roles were being compared. The manifest now adds a stronger pair of fixtures in which the cross-partition edge and within-partition control edge both enter the same AND gate that directly computes the task output. Both edges have the same target node, gate type, intervention operation, and cost. The only changed input is the declared input-state distribution.

Input states use bit 0 for input 0 and bit 1 for input 1; integer weights are listed in binary-state order 00, 01, 10, 11 only if explicitly specified by that schema. In schema v3, weights are listed by least-significant-bit state index (00, 10, 01, 11). Thus weights `4,1,16,4` give \(P(x_0=1)=0.2\) and \(P(x_1=1)=0.8\); reverse weights `4,16,1,4` give \(P(x_0=1)=0.8\) and \(P(x_1=1)=0.2\).

| Same causal graph and task, different input-state weights | Cross-cut mean loss | Internal-control mean loss | TCCIP |
|---|---:|---:|---:|
| (P(x_0=1)=0.2, P(x_1=1)=0.8) | 0.4 | 0.1 | +0.3 |
| (P(x_0=1)=0.8, P(x_1=1)=0.2) | 0.1 | 0.4 | −0.3 |

The intact system performs perfectly in both cases. The underlying rules, task, component labels, and intervention edges are identical; only the state prior changes. The cross-cut term has greater expected task influence under the first prior and lower expected influence under the second. This is a useful positive result for a **task- and distribution-conditioned engineering profile**, but a disqualifying result if anyone tries to describe TCCIP as an intrinsic, prior-free integration scalar.

This does not show that TCCIP is better than a direct conditional intervention-effect curve. In this simple example, the main effect is already understandable from the marginal activity of the other AND input. The next requirement is to show added predictive or diagnostic value on harder held-out systems after controlling for such simpler explanations.

## A stronger negative control: balanced XOR synergy is invisible to TCCIP

Added `role_matched_xor_synergy_uniform_prior` to the schema-v3 manifest. Both inputs of an XOR task are equally important; the tested cross-partition and within-partition edges enter the same XOR gate. Under the uniform input prior:

| Measurement | Result |
|---|---:|
| Intact task accuracy | 1.0 |
| Accuracy after cutting either edge | 0.5 |
| Cross-cut mean utility loss | 0.5 |
| Matched internal-edge mean loss | 0.5 |
| TCCIP contrast | 0.0 |
| Target entropy \(H(Y)\) | 1 bit |
| Individual MI \(I(X_0;Y)\) | 0 bits |
| Individual MI \(I(X_1;Y)\) | 0 bits |
| Joint MI \(I((X_0,X_1);Y)\) | 1 bit |

The zero TCCIP is correct for the *difference of edge-loss means*: the cross-cut and internal control edges have equal task effects. But if we marketed TCCIP as a universal measure of integration or higher-order synergy, this would be a serious false negative. The XOR target contains one bit of information jointly available from both inputs even though neither input alone carries information about it.

**Resulting paradigm change:** TCCIP cannot be the headline scalar. Keep the raw per-intervention response curve, task utility, partition profile, and input-state-conditioned effects as primary outputs; add explicit higher-order information diagnostics and, where estimators' assumptions apply, compare against PID/SURD or effective-information baselines. A task-performance metric and a synergy/integration metric should remain distinct unless an independently validated bridge between them is demonstrated.

This fixture is deliberately a falsification test, not a victory claim. The additional exact mutual-information baseline is computed from the same declared state weights and emitted by the independent Python oracle; the Rust evaluator also reports \(H(Y)\), individual-input MI values, and joint-input MI in its output. These implementations and their 15 Rust / 11 Python authored tests remain unexecuted.

## Second-order counterfactual utility interactions (new diagnostic; not a synergy metric)

To capture interactions between failures rather than merely compare classes of edges, the Rust evaluator and independent Python reference now calculate a second-order finite difference of task utility for every pair of declared edge interventions:

\[
\Delta^2 J_{ij}=J(\varnothing)-J(\{i\})-J(\{j\})+J(\{i,j\}).
\]

Each simultaneous cut replaces its own edge signal with an independent fair coin. Positive values mean the two single-cut losses overlap or saturate (the combined loss is less than their sum); negative values mean compound failure amplifies utility loss beyond the additive prediction; zero means the interaction contrast is additive under this intervention definition. These signs concern *utility response to cuts*, not positive/negative information synergy.

The canonical pairwise oracle is `validation/fixtures/tccip_pairwise_utility_interactions.tsv` (16 intervention pairs across the six fixtures). A Rust test and Python test compare every individual and joint intervention outcome against this file. The independent exhaustive reference calculation produced this illustrative subset:

| Fixture / pair | \(J_0\) | \(J_i\) | \(J_j\) | \(J_{ij}\) | \(\Delta^2J\) |
|---|---:|---:|---:|---:|---:|
| XOR: cut both task edges | 1.0 | 0.5 | 0.5 | 0.5 | +0.5 |
| Harmful bias: cut correct path and bias path | 0.0 | 0.5 | 0.5 | 0.5 | −0.5 |
| AND task, prior A: cut both task inputs | 1.0 | 0.6 | 0.9 | 0.67 | +0.17 |
| AND task, prior B: same cuts, reversed prior | 1.0 | 0.9 | 0.6 | 0.67 | +0.17 |

In the XOR case, the positive value means cutting either input already destroys half the task accuracy; cutting both does not double the damage. It is a *non-additivity/saturation* result. The system's informational synergy is captured by a different baseline: H(Y)=1 bit, I(X0;Y)=I(X1;Y)=0, I((X0,X1);Y)=1 bit. Do not relabel \(\Delta^2J\) as PID synergy, PEID, or IIT \(\Phi\).

This is an established class of interaction-effect calculation in mathematical form; our contribution is to make it an exact, manifest-driven, task-utility diagnostic in this benchmark and test it against prior art—not a claim to have invented finite differences or causal interaction effects. The next comparison must include simple second-order utility-response baselines and, when assumptions permit, PID/SURD and the 2026 preprint *Partial Effective Information Decomposition for Synergistic Causality* (PEID, https://arxiv.org/abs/2605.03267).

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

## Reproduction commands for the manifest and independent oracle

```bash
python -m unittest validation.test_tccip_reference_oracle -v
python validation/tccip_reference_oracle.py \
  --source-sha "$(git rev-parse HEAD)" \
  --output results/tccip_python_reference.tsv
sha256sum results/tccip_python_reference.tsv
cargo test --example task_conditioned_finite_state_eval
SYMTHAEA_SOURCE_SHA="$(git rev-parse HEAD)" \
  cargo run --release --example task_conditioned_finite_state_eval \
  > results/tccip_rust_reference.tsv
sha256sum results/tccip_rust_reference.tsv
```

These commands intentionally produce separately labeled Python and Rust artifacts; do not relabel one as the other. Both consume the same manifest. Compare raw intervention values and every usable partition, not just one aggregate. Record the exact source SHA and artifact hash together.

## Immediate implementation plan

1. The model/task/input-prior/partition/intervention definitions now come from schema-v3 TSV; a second TSV lists expected pairwise cut outcomes. Next, version the intervention *semantics* explicitly, rather than only encoding an implicit fair-coin edge replacement in the evaluator.
2. Sweep predeclared alternate partitions / coarse-grainings and retain the full multiscale profile; do not select the best partition on the evaluation set.
3. Add state-conditioned and controlled prior-shift fixtures, plus degree/delay/bandwidth-matched intervention controls where meaningful.
4. Compare against raw task ablation, graph/spectral measures, entropy/MI/transfer entropy, and relevant PID/SURD/effective-information baselines.
5. Test the preregistered adversarial matrix, then connect a real Symthaea task only after exact synthetic gates pass.
6. Keep the feature research-only until external review and evidence support a concrete engineering use.

**Current evidence status:** research specification plus a dependency-free exact toy evaluator, exhaustive partition-sensitivity code, an expected-output TSV, and an independent reference calculation have been authored. The Rust executable and its 15 tests have not been compiled or run; the committed independent Python oracle's 16 tests have also not been run. There is no demonstrated empirical usefulness, no validated advantage over established baselines, no novelty/SOTA claim, no IIT correspondence, and no consciousness inference.
