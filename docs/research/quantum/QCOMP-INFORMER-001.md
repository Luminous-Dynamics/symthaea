# QCOMP-INFORMER-001 — Value of information in quantum-informed MIS reduction

**Status:** Research contract; not implemented or qualified.  
**Parent workstream:** QCOMP-PORTFOLIO-001  
**Implementation issue:** [#8320](https://github.com/Luminous-Dynamics/symthaea/issues/8320)

## 1. Scientific question

For a maximum-independent-set (MIS) instance, does an information source improve the final solution produced by a fixed classical reducer enough to pay for the cost of acquiring and using that information? If it does, does a quantum-device signal add value beyond the strongest matched-budget classical informer?

This is not a proposal to recreate qReduMIS under a new name. qReduMIS is an existing published and open-source quantum-informed reduction algorithm. Its architectural idea is the starting point; the proposed contribution is a rigorous matched-budget *informer ablation* that separates useful information from source, reducer, postprocessing, and overhead effects.

## 2. Prior art and competitive context

- Schuetz et al., *Quantum-informed reduction algorithm for the maximum independent set problem*, Physical Review Research 8, 033296, published 10 September 2026, DOI: https://doi.org/10.1103/nh3b-1wv5. The paper describes classical kernelization driven by a quantum coprocessor's information about vertices likely to appear in large independent sets, with experiments using up to 231 qubits on Rydberg hardware.
- Reference implementation (Apache-2.0): https://github.com/jpmorganchase/qReduMIS.
- QOBLIB benchmark suite and live track record: https://zib-aopt.github.io/QOBLIB/.
- QOBLIB source repository: https://github.com/ZIB-AOPT/QOBLIB.
- Symthaea workload portfolio protocol: [QCOMP-PORTFOLIO-001 / PR #8294](https://github.com/Luminous-Dynamics/symthaea/pull/8294).
- Symthaea independent MIS graph/solution verification prototype: [PR #8296](https://github.com/Luminous-Dynamics/symthaea/pull/8296).

A published upstream classical ParallelILS submission reports best-known-or-better objectives across all 50 QOBLIB MIS instances, with two new best-known values. On the 186-variable es60fst02 instance it reports objective 88 in 0.0006 seconds, whereas published QPU-containing records for the same instance also report objective 88 with substantially different total runtimes. Those records are not a controlled head-to-head comparison, so use them to motivate careful accounting, not to infer a universal hardware speed ratio. The tiny exact solver in PR #8296 is a correctness oracle only.

### Two cohorts, two distinct claims

Keep two benchmark cohorts separate:

1. **Paper-reproduction cohort:** first reproduce the published qReduMIS experiment contract on its four reported 137-vertex instances, with hardness parameters approximately 1.478, 14.12, 125.5, and 1,435. The paper reports QAA success probabilities 0.35(2), 0.008(5), 0.007(4), and 0.0 respectively, while qReduMIS in-set and out-set strategies report 1.0 for those four cases (1,000 shots per QPU call in the paper's described setup). These are *paper-reported results*, not our replication. Preserve the original definitions, sampling protocol, and hardware profile; if original data or parameters are unavailable, mark the corresponding reproduction as incomplete rather than reconstructing hidden inputs.
2. **External-generalization cohort:** a source-frozen QOBLIB MIS sample including the stronger published classical records. This tests whether informer value generalizes outside the paper's selected graph family. It cannot be conflated with a faithful paper reproduction.

The official [qReduMIS package README](https://github.com/jpmorganchase/qReduMIS/blob/main/qReduMIS_package/README.md) explicitly notes that its example problem is fully reducible and does not need to call the quantum backend. Therefore, a demo or passing unit test is not backend evidence: instrumentation must confirm whether a backend call actually occurred, count calls/shots, and label no-call runs as classical-only execution.

## 3. Paradigm shift: compute the value of information

Do not optimize the quantum subroutine in isolation. Treat the informer as a budgeted signal source and measure its incremental effect on the downstream reducer.

The key quantity is:

**Incremental value of information = downstream improvement caused by a signal − cost of generating, transforming, and applying that signal.**

This should be measured against a no-informer baseline, cheap structural signals, and the strongest eligible classical source. An informative-looking sample is not a win if it causes poor reductions or costs more than the saved search.

The study should distinguish four kinds of evidence:

1. Synthetic signal fixtures used to validate implementation.
2. Classical signal generated from graph structure, heuristics, local-search populations, or ensembles.
3. Replayed, source-identified quantum result records.
4. New physical-device observations with exact backend and execution metadata.

None of these categories can silently be promoted into another.

## 4. Matched ablation matrix

Every arm uses the same source graph, downstream reducer, output verifier, maximum decision count, stopping rule, and declared total budget.

| Arm | Informer signal | Purpose |
|---|---|---|
| A0 | No informer; classical reducer only | Baseline benefit of reduction without extra information |
| A1 | Seeded random vertex choice | Negative/control baseline |
| A2 | Structural score: degree, neighborhood, local graph features | Cheap classical information |
| A3 | Frequency from a population of valid local-search solutions | Classical solution-distribution signal |
| A4 | Ensemble agreement and disagreement from independent classical solvers | Stronger classical informer |
| A5 | Scrambled/permuted signal preserving the original score histogram | Tests whether vertex-specific information matters beyond generic score distribution |
| A6 | Exact-oracle guidance on tiny graphs only | Positive control for plumbing and metric sensitivity; never a scalable competitor |
| A7 | qReduMIS algorithm / quantum informer | Established reference method; include only with a clearly identified backend or source dataset |
| A8 | Physical quantum informer | Prospective hardware claim only when actual execution evidence and end-to-end costs are available |

Where feasible, include a degree-stratified permutation control so an informer cannot appear useful merely because its output is correlated with an easy degree heuristic. Add a random signal with the same selection distribution and a signal that intentionally reverses the scores as negative controls.

The quantum-informed method must not receive more candidate evaluations, more restarts, a larger reducer budget, or privileged test information without those resources being counted or matched.

## 5. First local work that does not require a QPU

The first run should be a deterministic, source-controlled experiment on tiny hand-checkable and seeded synthetic graphs. It validates the interface and the experiment protocol, not quantum advantage.

1. Freeze each input graph, source generator, seed, and its identity.
2. Use an independent checker for every resulting independent set.
3. Use exact enumeration only on the deliberately small correctness cohort to measure optimum and objective regret.
4. Run each signal arm through the identical reducer and decision budget.
5. Record every decision and every intermediate residual graph, not only the final score.
6. Replay the same manifest and verify normalized outputs and traces are identical.
7. Test malformed, stale, wrong-graph, missing, tied, all-in, all-out, and contradictory signals.
8. Demonstrate a counterexample where misleading information worsens the result; ensure that the harness reports it instead of hiding it.

A synthetic exact-oracle informer is a plumbing positive control, not a feasible source of information on large real instances. Never train the test-time selector on exact labels from held-out graphs.

## 6. Benchmark cohort

Reuse QOBLIB rather than fork or rename its suite. At cohort-freeze time, record the exact upstream commit, instance IDs, data-file hashes, formulation, and optimal/best-known/unknown status.

Select instances by a declared stratification policy spanning graph families, graph sizes, density/hardness profiles, and current best-known gaps. The rule must be fixed before any candidate arms are run. Hold out full instance families or graph-generation families, not just alternate random seeds from the same base instance.

For tiny exact fixtures, enumerate every subset to establish optimality independently. For QOBLIB instances with only best-known values, report gap-to-best-known and do not call it an optimality gap.

## 7. Metrics

### Final application outcomes

- Independently verified feasibility rate.
- Final objective and gap to a proven optimum only when proven.
- Gap to best-known, clearly labelled, when optimality is unresolved.
- Success probability to a preregistered target.
- Verified quality versus the total-cost budget curve.
- Wall-clock time-to-target with failed runs, retries, and timeouts preserved.

### Reduction dynamics

- Residual vertex and edge counts after each reduction step.
- Exact reduction rules triggered and number of forced vertices.
- Incumbent quality over time.
- Number of informer calls, selected decisions, and total acquired samples.
- Fraction of decisions that improve the later incumbent, trigger useful reductions, or lower final search effort.
- Invalid or contradictory signals rejected by validation.

A smaller residual graph is a diagnostic, not an application win by itself. If reduction removes vertices that belonged to a high-quality solution, the final objective can worsen; report both.

### End-to-end cost

Record preprocessing, classical inference, data conversion, QPU queueing, calibration, shots, repeats, postprocessing, verification, and hardware execution separately. Track elapsed time, CPU/GPU allocation, direct monetary cost, and energy only with an explicit measured/modelled boundary. Unknown energy remains unknown.

### Incremental information value

Use paired per-instance differences between every informer and A0, A2, and A4. Report:
- objective / time-to-target uplift at matched total budget;
- marginal cost per additional verified objective unit or per successful target run;
- regret against the best *measured eligible arm* for that instance;
- signal calibration and abstention behavior;
- performance by held-out graph family;
- uncertainty intervals and all unsuccessful runs.

A single average score cannot hide a material regression on one declared family.

## 8. Statistical and anti-leakage protocol

- Freeze hypotheses, cohort, primary outcome, budgets, run count, and analysis before the confirmatory run.
- Use paired comparisons on the same instance and budget; report paired distributions, not only aggregate means.
- Separate tuning and confirmation data by instance family.
- Estimate required repetitions from pilot variance and a predeclared practical effect rather than treating 30 runs as a magic number.
- Control multiplicity when comparing many informer variants.
- Use bootstrap intervals or an appropriate paired test and publish raw per-instance outcomes.
- If confidence intervals cannot distinguish the best classical informer from the quantum-informed arm, report inconclusive; do not claim a win.
- Treat published QOBLIB / qReduMIS records as external evidence inputs, not as experiments executed by this repo.

## 9. Proposed research targets — not current results

### Gate A: correct harness
100% of outputs are independently verified; malformed or stale signals fail closed; tiny exact fixtures agree with independent enumeration; fixed manifests replay deterministically.

### Gate B: demonstrate information value
At least one informer arm shows a prespecified, reproducible improvement over A0 on held-out instances at the same total budget. A result that only beats random but not a cheap structural heuristic does not establish a meaningful advantage.

### Gate C: outperform the best classical informer
A preregistered held-out family shows materially better verified quality or time-to-target than A2 and A4 under matched total budgets, with uncertainty, full costs, and independent replay.

### Gate D: quantum incremental value
Actual quantum-derived signals improve the final workload outcome relative to the strongest eligible classical informer and the no-informer baseline, after charging all acquisition and workflow overhead. The comparison uses the same graph instances, reducer, selection budget, stopping rule, and output verifier.

A suggested practical target is at least 2× lower end-to-end cost-to-target at non-inferior verified quality, or a statistically supported quality increase within equal budgets. This is an aspirational target, not a claim about existing results or a universal scientific threshold.

## 10. Project responsibilities and implementation order

1. **Symthaea:** verify the existing MIS verifier; implement signal/provenance types and tiny deterministic informer fixture runner; then build matched reducer ablations and paired reporting.
2. **Portfolio track:** freeze the QOBLIB MIS cohort and upstream solver records; add classical local-search/ensemble baselines before any QPU comparison.
3. **qReduMIS integration:** inspect and cite the public implementation and relevant paper; avoid a parallel reimplementation unless a specific tested gap exists.
4. **Symtropy:** later model resource topology, queueing, transfer, and thermal/energy budgets, with simulation explicitly labelled as modelling rather than observation.
5. **Mycelix:** optionally bind manifests and raw artifacts into verifiable receipts for collaborators. A receipt does not upgrade the claim's evidence class.

## Nonclaims

This spec reports no new measurement and no successful quantum-advantage result. It does not claim that Symthaea has executed qReduMIS or a physical QPU. Its immediate contribution is an experiment that can distinguish genuine signal value from algorithmic overhead and from strong classical substitutes, without waiting for quantum hardware access or external CI completion.
