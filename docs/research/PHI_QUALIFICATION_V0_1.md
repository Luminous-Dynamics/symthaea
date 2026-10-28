# Symthaea Phi Qualification v0.1

**Status:** Draft research protocol; not an implementation claim or consciousness test  
**Version:** 0.1.0  
**Date:** 2026-10-28  
**Scope:** Formal IIT calculations, IIT-inspired proxies, and task utility signals must be evaluated as separate products.

## Executive decision

The next improvement is not to raise a Phi score or increase its numerical precision. It is to make the score's *identity, provenance, uncertainty, and permitted interpretation machine-checkable*.

Symthaea currently contains multiple quantities called or exposed as Phi. They are not interchangeable:

- The spectral/algebraic-connectivity family measures a graph property such as the Laplacian spectral gap.
- The HDC integrated-information path estimates information-like differences from hypervector similarities and partitions.
- The `IIT4Calculator` path uses binned distributions, mutual-information-like quantities, and proxy cause/effect fields; its `analyze()` method delegates the system-level field to `TruePhiCalculator`.
- Attention and cognitive-loop code may consume these values operationally.

This is a useful research substrate, but naming a result `big_phi`, `IIT4Result`, or `ConsciousnessState` is not enough to establish conformance to IIT 4.0. In the inspected `iit4.rs` path, the API does not receive an explicit state-indexed transition probability model, intervention semantics, or the full candidate-system/causal-repertoire specification needed to independently establish a formal IIT 4.0 computation. Downstream implementations and other modules must be audited before any broader claim is made.

**Qualification principle:** the implementation must earn a claim by passing tests for the specific formalism it says it computes. An IIT-inspired estimator remains valuable if it is named and evaluated as an estimator.

## 1. Claim levels

Every emitted measurement and report must be assigned exactly one level. Levels describe evidence about the *computation*, not evidence that a system is conscious.

| Level | Name | Required evidence | Permitted statement |
|---|---|---|---|
| Q0 | Unclassified | Legacy scalar without sufficient identity/provenance | “Legacy score”; no IIT or consciousness interpretation |
| Q1 | Operational proxy | Explicit metric definition, units, input features, seed and build identity | “This proxy measures [specified property] under [specified assumptions]” |
| Q2 | Reference-qualified approximation | Q1 plus a preregistered comparison against appropriate small-system references, error and uncertainty reporting, adversarial controls and held-out evaluation | “This approximation agrees with the named reference on the tested domain, within reported error” |
| Q3 | Formal-computation qualified | Formalism-specific input model, full required algorithm for that version/quantity, independently checked reference cases, exhaustive/bounded search declaration, and reproducible results | “This implementation computes [exact quantity] for the stated finite model and assumptions” |

There is deliberately **no “consciousness certified” level**. Passing a computational specification does not establish phenomenal experience, and a proxy-validation result does not convert the proxy into IIT.

## 2. Required measurement contract

Each result must carry the fields below either directly or through an immutable manifest referenced by digest.

### Identity and provenance

- `metric_id`: stable, specific name such as `spectral_laplacian_lambda2`, `hdc_mutual_information_partition_proxy`, or `iit4_system_phi_s`.
- `metric_version`, implementation source revision, executable/build digest, and configuration digest.
- `claim_level`: Q0–Q3.
- `theory` and `formalism_profile`: e.g. `none`, `IIT_3_0`, `IIT_4_0_2023`, `IIT_4_0_2026`, or another explicitly versioned framework. “IIT 4.0” alone is insufficient for a reference claim.
- `quantity`: distinguish mechanism-level small phi (`φ`), system integrated information (`φ_s`), and structure integrated information (`Φ` / the Φ-structure), as well as any proxy or task-utility quantity. PyPhi's `Analysis.phi`, `Analysis.ces`, and `Analysis.big_phi` are distinct outputs: `phi` is system integrated information, `ces` is the cause–effect structure, and `big_phi` is structure integrated information. Do not map a legacy `big_phi` field to formal `Φ` without checking its implementation; do not use a bare `phi` as the only identifier.

### Model identity

- causal units, state encoding, state at which the result is evaluated, and time grain;
- state-indexed transition model (or the explicit reason a formal causal model is unavailable);
- intervention/noise semantics and how the transition model was estimated;
- candidate system boundary and excluded units;
- partition family searched, search completeness, pruning/heuristics, and whether the answer is exact, bounded, sampled, or approximate;
- the entropy estimator, discretization/binning, sample size, random seed, estimator calibration, and missing-data policy where applicable.

If a field is not applicable, record an explicit `not_applicable` plus a reason. An omitted field is not equivalent to a justified non-applicability.

### Result and uncertainty

- raw result, unit/normalization, numeric status, and intermediate stage outputs;
- reference result and reference implementation/version when available;
- absolute and relative error where meaningful; rank metrics only where ranking is the preregistered claim;
- uncertainty interval and method, or `not_estimated` with a reason;
- elapsed time, peak memory, unit count, state count, partition evaluations, and early-stop/pruning flags;
- explicit allowed interpretation and disallowed interpretation.

NaN, infinity, silently clamped inputs, invalid causal models, incomplete searches, and timeouts must never be silently converted to successful zero scores.

## 3. Three independent tracks

### Track A — Formal reference computation

Implement or wrap a pinned and independently checked reference implementation. Keep `IIT_3_0`, `IIT_4_0_2023` (Albantakis et al. 2023), and `IIT_4_0_2026` (the updated intrinsic-information requirement described by Mayner et al. 2026) separate: their quantities, partition rules, and cause–effect formalisms are not interchangeable. Current PyPhi configuration docs expose these as distinct formalism profiles and note that a preset selects compatible measures and partition schemes, not merely a version string. The PyPhi 2.0 documentation also says the current release is a release candidate and a bare `pip install pyphi` installs 1.2 rather than 2.0; therefore pin the exact package version, artifact hash/source revision, formalism profile, and preset. IIT 4.0 qualification must state whether it computes system integrated information, a cause–effect structure, distinctions/relations, or only a subset. A partial calculation must be labeled partial; it must not be renamed “full IIT 4.0”.

Start with binary, finite, explicit transition-probability models small enough for exhaustive work. Archive model bytes, exact state, expected values, reference version, and digests. Include degenerate and analytically tractable known-answer cases. The reference implementation is a comparator, not an oracle whose name removes the need to inspect model equivalence.

### Track B — Scalable proxy suite

Keep each proxy under its own name. Candidate measurements include:

- graph connectivity / Laplacian λ₂;
- observational mutual information and conditional mutual information;
- perturbation-based causal influence or intervention response;
- recurrence, effective dimensionality, state differentiation, and dynamical recovery;
- calibrated estimates or bounds, where the assumptions support them.

Do not combine these into one unlabelled scalar. A composite is allowed only with a frozen formula, declared units/normalization, nonnegative/finite input constraints, ablations, and evidence that its added complexity improves held-out prediction over simpler baselines.

For every scalable estimator, compare it with the correct exact/reference quantity on overlapping small cases. Report error distribution, worst-case error, rank reversals, subgroup/family performance, sensitivity to bins/noise/state/grain/boundary, runtime and memory scaling. Correlation alone is insufficient: a high correlation can coexist with unacceptable absolute errors or systematic failure on an important topology family.

### Track C — Cognitive utility

Test whether metrics improve an engineering outcome, such as memory binding, temporal reasoning, robust context retention, recovery after perturbation, or selective attention.

Use paired tasks and seeds, held-out task families, baselines, confidence intervals, cost accounting, and ablations. Compare against:
1. the same architecture without the metric;
2. a simple heuristic using the same input budget;
3. a parameter-/compute-matched learned or fixed baseline, where applicable;
4. a random or shuffled-score control.

The score earns operational influence only if it improves the specified task under a preregistered acceptance rule. If it does not, remove or reduce its control authority. Never interpret a cognitive-utility win as proof of consciousness.

## 4. Adversarial qualification suite

The fixture manifest at `benches/18-consciousness/fixtures/phi-qualification-cases-v0.1.json` is a case catalogue, not precomputed evidence. Every case requires an explicit model generator, stable seed, model digest, and assertions before it can count as evaluated.

| Family | Adversarial question | Required checks |
|---|---|---|
| Disconnected and independent units | Does the metric award integration when the causal model contains none? | Formal reference known answer; no false positive beyond declared numerical tolerance |
| Feed-forward chain | Does connectivity get mistaken for irreducible recurrent causation? | Distinguish reachability/topology from formal result |
| Recurrent ring and modular networks | Does the result change predictably as cross-module coupling is swept? | Paired sweep; boundaries and state fixed; full curve reported |
| Common-driver synchrony | Can correlation/synchrony masquerade as direct causal integration? | Same/near-same observational association with different intervention structures |
| Bottleneck / bridge deletion | Does an intervention have the reported effect on the actual causal model? | Pre/post model digest; expected direction specified per metric and formalism |
| Unit relabeling | Does a consistent permutation of causal-unit labels change the result? | Invariance within numerical tolerance |
| Hypervector-coordinate relabeling | Does a pure coordinate permutation change a purported representation-invariant metric? | Invariance, unless the metric explicitly depends on coordinate identity |
| Duplicated or near-duplicate units | Does scale or redundancy inflate the proxy while task utility falls? | Report score, redundancy, size and task outcomes separately; do not assume a universal IIT direction |
| Noise/sample-size sweep | Is the estimator stable and calibrated? | Fixed underlying model, paired seeds, error-vs-sample-size curve |
| Boundary / spectator unit | Is the result sensitive to the chosen system boundary? | Report boundary variants; do not hide exclusion search limits |
| Time-grain / coarse-graining | Does changing update windows alter the model or quantity? | Declare grain; compare only like-for-like results |
| Model misspecification | Does observational fit conceal incorrect interventional behavior? | Observationally matched but interventional-distinct controls |
| Invalid/underspecified input | Can the evaluator fabricate a score from missing transition data, mismatched lengths or NaNs? | Fail closed with typed error / unscorable outcome |
| Task-negative control | Does increasing the proxy harm memory/reasoning/attention under matched resources? | Metric loses control authority when preregistered utility gate fails |

Expected direction must be specified **per metric and formalism**, not copied from an intuition such as “a star should have more Phi than a random graph.” Different metrics answer different questions; topology rankings are not universal IIT ground truth.

## 5. Metrics and release gates

The following gates are targets for qualification, not claims that they have already passed.

### Formal track

- 100% agreement with frozen, independently checked known-answer cases within documented numeric tolerances.
- 100% correct status classification for exact / bounded / approximate / timeout / invalid-model fixtures.
- No silent substitution of a heuristic or fallback for the requested formal quantity.
- Unit relabeling and equivalent causal-model encodings preserve the result within numerical tolerance.
- Report exhaustive-search completeness or a mathematically justified bound; “finished” does not mean “exact” if pruning changes the definition.
- Reproduce reference results from pinned source/model artifacts and publish all disagreements.

### Proxy track

Report MAE/RMSE (where scale-compatible), bias, worst-case and quantile error, Spearman ρ and Kendall τ, pairwise ranking reversals, calibration/coverage if intervals are claimed, and performance stratified by topology and generator family. Freeze the primary metric and acceptance thresholds before unblinding. Split by **generative family**, not merely random rows, so near-duplicate topologies do not leak across train/test.

No universal “r > 0.95” gate is appropriate until the target quantity, domain and baseline are specified. If ranking is the use case, report pairwise accuracy and rank reversals with confidence intervals; if values are the use case, report scale-aware error and calibration.

### Cognitive utility track

Use paired comparisons and cluster-aware uncertainty for tasks drawn from the same environment/repository. Include negative controls and a compute/memory-matched baseline. Report task success, failure severity, abstention, latency, wall-clock cost, energy if measured, and cost per independently verified successful outcome. Preregister a minimum useful effect and do not count missing/unscorable results as successes.

### Fail-closed release rule

Do not issue Q2 or Q3 qualification if any required field is missing, source/model digests disagree, a critical negative control false-passes, the formalism/reference pair is mismatched, or the exact tested revision cannot be identified. Mark a blocked or incomplete run as such; never infer a pass from queued CI, a partial fixture, a prior head, or a successful build alone.

## 6. Comparison with the field

This proposal should benchmark against reference implementations and established methods, not an invented “industry Phi leaderboard.”

- **PyPhi / IIT reference implementations:** appropriate for the exact formalism profile and quantities they actually implement. Pin the package/source revision, `IIT_3_0`, `IIT_4_0_2023` or `IIT_4_0_2026` profile, preset, partition scheme and model. Never conflate `Analysis.phi` (`φ_s`, system integration), `Analysis.ces` (the Φ-structure) and `Analysis.big_phi` (`Φ`, structure integration). The official complexity guide documents severe combinatorial limits for IIT 4.0 cause–effect structures, with practical ceilings dependent on stage, topology and options. This is a reason for carefully validated proxies—not a reason to call a proxy exact. References: https://pyphi.readthedocs.io/en/latest/reference/configuration.html, https://pyphi.readthedocs.io/en/latest/reference/_autosummary/pyphi.html, and https://pyphi.readthedocs.io/en/latest/theory/computational-complexity.html
- **Spectral graph measures:** valid baselines for connectivity and diffusion properties, not formal IIT scores.
- **Information-theoretic and perturbational baselines:** compare only when the causal/data assumptions match; clearly separate observational measures from intervention-based ones.
- **Cognitive baselines:** compare with simple heuristics, ablated Symthaea, and compute-matched alternatives on held-out tasks.
- **Consciousness-theory evidence:** maintain a separate literature/evidence table for IIT, Global Neuronal Workspace, Recurrent Processing and other theories. A preregistered 2025 adversarial collaboration found both supporting and challenging results for IIT and GNWT, underscoring that formal implementation correctness is not theory confirmation.

Winning means demonstrable improvement on a named task: lower approximation error at equal cost, more exact cases solved within fixed resources, fewer false positives, tighter valid bounds, or higher held-out cognitive utility. It does **not** mean a larger score or a declaration that Symthaea has exceeded “SOTA consciousness.”

## 7. First experiments (no runner dependency)

Run these sequentially from a clean checkout, recording commit, command, environment, exit code and output digest for each stage:

1. **Static claim inventory:** identify every exported field called `phi`, `big_phi`, `small_phi`, `spectral_phi`, `phi_hdc` and every production control consumer. Classify the measure and permitted interpretation.
2. **Known-answer controls:** build 2–6 unit finite causal systems with explicit TPMs; validate the reference path against pinned independent cases.
3. **Representation invariance:** relabel units and hypervector coordinates where the stated metric claims invariance; record every violation.
4. **Causal confound pair:** construct observationally similar cases with distinct intervention behavior; verify that observational proxies are not described as causal estimates.
5. **Approximation matrix:** run every proxy on the exact/reference cases and report worst-case error and ranking reversals by generator family.
6. **Utility ablation:** use frozen Symthaea tasks to compare no metric, each candidate signal, and a shuffled-score control under matched resource budgets.

Stop and document any failed controls before scaling to larger systems.

## 8. Immediate source-language cleanup

Until qualification is complete, user-facing descriptions and internal comments should say:

- “integration proxy” or a specific metric name, not “consciousness measured by Phi”;
- “IIT-inspired” or “partial IIT 4.0 measure” where formal requirements are not demonstrated;
- “task utility signal” when a metric guides attention or action;
- “unvalidated/unknown” when the implementation or reference comparison has not been independently tested.

In particular, attention should not assign higher weight because a value is asserted to indicate greater consciousness. It may use a qualified integration feature as one learned input, but should be evaluated against task utility, preserve a baseline without that input, and avoid locking the controller into a metric that fails negative controls.

## References

- Albantakis et al. (2023), “Integrated Information Theory (IIT) 4.0: Formulating the Properties of Phenomenal Existence in Physical Terms”: https://doi.org/10.1371/journal.pcbi.1011465
- PyPhi configuration reference: distinguishes `IIT_3_0`, `IIT_4_0_2023`, and `IIT_4_0_2026`: https://pyphi.readthedocs.io/en/latest/reference/configuration.html
- PyPhi API reference: distinguishes `Analysis.phi` (system integrated information), `Analysis.ces` (Φ-structure), and `Analysis.big_phi` (structure integrated information): https://pyphi.readthedocs.io/en/latest/reference/_autosummary/pyphi.html
- PyPhi installation notes: pin 2.0 exactly; the bare install resolves to 1.2 according to current docs: https://pyphi.readthedocs.io/en/latest/
- PyPhi, “Computational complexity” (IIT 4.0 and IIT 3.0 stage-by-stage limits): https://pyphi.readthedocs.io/en/latest/theory/computational-complexity.html
- Cogitate Consortium et al. (2025), “Adversarial testing of global neuronal workspace and integrated information theories of consciousness,” Nature: https://doi.org/10.1038/s41586-025-08888-1

## Claim boundary for this draft

This file is a qualification protocol and source-informed research proposal. It does not add an IIT 4.0 reference implementation, execute the case catalogue, repair existing production behavior, establish a proxy's accuracy, or provide evidence for or against consciousness in Symthaea.
