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
- Symthaea independent MIS graph/solution verification prototype: [PR #8324](https://github.com/Luminous-Dynamics/symthaea/pull/8324).

A published upstream classical ParallelILS submission reports best-known-or-better objectives across all 50 QOBLIB MIS instances, with two new best-known values. On the 186-variable es60fst02 instance it reports objective 88 in 0.0006 seconds, whereas published QPU-containing records for the same instance also report objective 88 with substantially different total runtimes. Those records are not a controlled head-to-head comparison, so use them to motivate careful accounting, not to infer a universal hardware speed ratio. The tiny exact solver in PR #8324 is a correctness oracle only.

### Two cohorts, two distinct claims

Keep two benchmark cohorts separate:

1. **Paper-reproduction cohort:** first reproduce the published qReduMIS experiment contract on its four reported 137-vertex instances, with hardness parameters approximately 1.478, 14.12, 125.5, and 1,435. The paper reports QAA success probabilities 0.35(2), 0.008(5), 0.007(4), and 0.0 respectively, while qReduMIS in-set and out-set strategies report 1.0 for those four cases (1,000 shots per QPU call in the paper's described setup). These are *paper-reported results*, not our replication. Preserve the original definitions, sampling protocol, and hardware profile; if original data or parameters are unavailable, mark the corresponding reproduction as incomplete rather than reconstructing hidden inputs.
2. **External-generalization cohort:** a source-frozen QOBLIB MIS sample including the stronger published classical records. This tests whether informer value generalizes outside the paper's selected graph family. It cannot be conflated with a faithful paper reproduction.

The official [qReduMIS package README](https://github.com/jpmorganchase/qReduMIS/blob/main/qReduMIS_package/README.md) explicitly notes that its example problem is fully reducible and does not need to call the quantum backend. Therefore, a demo or passing unit test is not backend evidence: instrumentation must confirm whether a backend call actually occurred, count calls/shots, and label no-call runs as classical-only execution.

### Additional contemporary comparator: fixed-angle QAOA-informed greedy

Wybo et al., *A scalable quantum-enhanced greedy algorithm for maximum independent set problems* (preprint https://arxiv.org/abs/2601.21923; accepted by Physical Review Research on 19 October 2026, https://doi.org/10.1103/ds1n-wsdm), is another distinct prior-art architecture. It uses fixed QAOA angles derived from regular-tree neighborhoods to score local vertex inclusion expectations, then chooses among greedy candidates and incrementally updates affected neighborhoods. It is not the same approach as qReduMIS kernelization.

The paper reports random 3-regular graph experiments for N = 50, 100, 200, 500, 1,000, 2,000, and 5,000; 200 graph instances per size for QGreedy analyses; tensor-network local-expectation calculations through p = 4; and a 20-qubit IQM Garnet experiment. Its reported p = 4 improvement against the linear-prioritized-search baseline applies to its stated random-regular cohort and through N <= 2,000. The authors also describe cases where p = 3 light cones exceed the device's 20-qubit capacity. Treat these as paper-reported results, not as Symthaea measurements or as evidence of wins on arbitrary graph families.

A companion reproduction/stress-test workstream is tracked in [QCOMP-QGREEDY-001 / issue #8332](https://github.com/Luminous-Dynamics/symthaea/issues/8332). Keep its random-regular reproduction cohort and greedy algorithm costs separate from the qReduMIS paper cohort and QOBLIB generalization cohort below. For the fixed-reducer informer study in this document, QGreedy is a separate algorithm-family comparator, not a drop-in substitute for one arm of the matched reducer matrix.

### Broader solver context: separate MIS workload classes

The QGreedy paper's reported "SOTA" comparison is scoped to its random fixed-degree regular-graph cohort and the specialized linear-prioritized search algorithm. Do not interpret it as a win over every maximum-independent-set solver on arbitrary graph families.

For broader general-graph evaluation, add a separate cohort with reproducible runs of [KaMIS](https://github.com/KarlsruheMIS/KaMIS) (including ReduMIS where applicable), [CHILS](https://github.com/KarlsruheMIS/CHILS), and [LearnAndReduce](https://github.com/KarlsruheMIS/LearnAndReduce), plus frozen [QOBLIB MIS instances](https://zib-aopt.github.io/QOBLIB/). SATLIB results are supplementary only when underlying instance identities, objectives, compute environment and budget can be recovered and frozen. Pin every solver version, build configuration, threads/accelerators, restarts, seed, timeout, and output verifier.

Keep maximum independent set (maximize cardinality) distinct from maximal independent set (cannot add another compatible vertex). For example, the 2026 cuMIS framework is described as a high-throughput **maximal** independent set solver; its runtime/scale results are useful systems context, but they are not directly comparable as a maximum-set objective without a different experiment contract. See https://doi.org/10.1145/3797905.3800529.

Report paired, per-instance objective and cost results by graph family. Do not average random 3-regular ratios together with QOBLIB, SATLIB, or real-world sparse graphs and then label the combined mean a general SOTA win.

### Hardness-stratified workload selection

A separate workstream ([QCOMP-HARDNESS-001 / issue #8358](https://github.com/Luminous-Dynamics/symthaea/issues/8358)) constructs a challenge cohort whose operational hardness is calibrated against a frozen classical solver portfolio before confirmatory testing. This follows the lesson from Cazals et al., *Identifying hard native instances for the maximum-independent-set problem on neutral-atom quantum processors* (Phys. Rev. Applied, 26 March 2026, https://doi.org/10.1103/8gjv-ll1f): graph structure matters, and tuning density/treewidth-related factors can materially change classical difficulty; the paper reports no advantage for the small quantum cases it studied. Related work finds that graph density, interaction range, and vertex weights alter classical reducibility for native weighted Rydberg-array graphs (https://arxiv.org/abs/2605.07952).

Accordingly, do not let an algorithm pick its own favorable test instances. Freeze graph-family and generator strata, record exact reduction ratios/kernel sizes plus a labelled treewidth proxy, calibrate time-to-target and timeout distributions using strong eligible classical solvers, and hold out entire parameter regions for confirmation. Treat operational hardness as solver-portfolio- and budget-relative, not as a universal or theoretical hardness label. Only compare a quantum arm on the same graphs and task objective, and include physical placement/embedding, calibration, shots, preprocessing, verification, and total wall/cash/energy cost.

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
