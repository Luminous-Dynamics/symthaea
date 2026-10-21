# ODE-ROBERTSON-STIFF-001 — benchmark protocol

**State:** protocol proposal; implementation and numerical results are not yet qualified.  
**Scope:** stiff IVP work–precision, invariant preservation, and solver failure characterization.  
**Parent work:** ODE verification 001, Dopri5 comparison 001, and decay Dopri5 comparison 001.

## Why this is the next paradigm shift

A solver should not be judged by one easy equation, one tolerance, or wall-clock time from a shared runner. The benchmark must distinguish four questions:

1. **Did it finish?** Record timeout, step-budget exhaustion, nonlinear/convergence errors, NaN/Inf, and early termination separately.
2. **Was the answer accurate?** Compare at common output times against a reference whose own uncertainty is bounded by a second tighter solve and an independent solver family.
3. **Did it preserve the model's structure?** Score conservation and non-negativity separately from trajectory error; never clip states or hide failures.
4. **How much work did it take?** Report RHS calls, rejected steps, Jacobian/linear-solver work when available, memory, and repeated timing only at matched achieved error.

This is a diagnostic benchmark, not an assumed win for either solver. The existing Symthaea explicit Cash–Karp method may perform poorly on a deliberately stiff problem; that outcome would be useful evidence for a stiffness-aware method-selection policy, not grounds to weaken the gate.

## Frozen mathematical fixture

Use the classic Robertson autocatalytic kinetics system:

\[
\begin{aligned}
y_1' &= -0.04y_1 + 10^4y_2y_3,\\
y_2' &= 0.04y_1 - 10^4y_2y_3 - 3\times10^7y_2^2,\\
y_3' &= 3\times10^7y_2^2,
\end{aligned}
\]

with initial state \(y(0)=(1,0,0)\) and time interval \([0,10^{11}]\). The rate constants create widely separated time scales. The total concentration is invariant in exact arithmetic:

\[
y_1(t)+y_2(t)+y_3(t)=1.
\]

Do not rescale the whole state vector: these components have physically distinct magnitudes, so scalar state scaling would conceal the issue that component-wise absolute tolerances are designed to solve.

### Common output times

Record states at \(t=0\), then at \(10^{-8},10^{-6},10^{-4},10^{-2},1\), followed by every decade through \(10^{11}\). Also retain solver-provided diagnostic points separately; they must not replace the common, predeclared output grid. The exact grid is part of the report schema and must be identical for every solver.

## Reference ladder

### Primary reference

Use SUNDIALS **CVODE 7.9.0**, BDF mode with Newton iteration and a dense linear solver. Pin the release source, archive digest, compiler, build flags, container/base-image digest, and complete dependency inventory. Record the complete CVODE option set and all counters. Do not report the reference as trusted merely because CVODE returns success.

Construct candidate references at two increasingly strict component-wise tolerance profiles. A reference is eligible only when:
- both solves complete across the full time interval;
- both preserve the invariant within the reference error budget;
- their common-grid trajectories agree within a stricter envelope than the benchmark candidate's acceptance envelope;
- both remain finite and their observed negative excursions are bounded by the chosen per-component absolute tolerance.

If these checks fail, the reference is unresolved and all downstream candidate accuracy scores remain UNQUALIFIED.

The reference ladder needs an explicit quantitative agreement gate. Compare the two tighter candidate-reference trajectories at the protocol's common output grid using the strictest candidate profile denominator for each component, \`atol_i + rtol * |y_i|\`. Require the maximum normalized cross-reference discrepancy to be \\(\\le 10^{-2}\\). This reserves at most one percent of the strictest candidate tolerance envelope for reference disagreement, while the candidate error gate is 10, leaving a three-order-of-magnitude gap between reference disagreement and the qualification threshold. If the reference pair fails this gate, the result is UNQUALIFIED; do not average the references or relax the threshold after seeing candidate outcomes. This is a reference-agreement criterion, not a mathematical proof of reference error.

### Independent cross-check

Before freezing golden reference bytes, compare against a separately implemented stiff method (preferably a pinned Rust BDF implementation after API and license review, or another independently built solver). Different solvers are not interchangeable oracles by brand name: save their raw state tables and compare their common-grid results. If independent solutions disagree beyond the declared reference budget, investigate the discrepancy rather than averaging it away.

The reference file must include the input equations/parameters, output grid, full tolerance vector, solver version/build identities, raw state samples, invariant defect, positivity defect, counters, and SHA-256 digest. It is not generated from the Symthaea candidate output.

## Candidate tolerance matrix

Cross these relative tolerances:
- \(rtol\in\{10^{-4},10^{-6},10^{-8}\}\)

with the following component-wise absolute tolerance profiles:
- **balanced:** \((10^{-10},10^{-14},10^{-10})\)
- **strict:** \((10^{-12},10^{-16},10^{-12})\)

These are initial benchmark profiles, not universal engineering tolerances. A calibration run may reveal they are inappropriate for a particular implementation, but any change must be documented as a versioned protocol change before looking at comparative score outcomes. Reference-generation tolerances must be materially tighter than the candidate profiles and validated through the reference ladder.

Do not compare the solvers using their internally reported error estimates as if those numbers were identical. Error norms and controller designs differ. Compare achieved error on the common output grid and report each solver's own diagnostics.

## Scorecard and gates

Every solver/profile observation records:

- status: completed, rhs_budget_exceeded, timeout, solver_error, non_finite, or invalid_output; exactly one terminal status.
- maximum component-normalized common-grid error against the qualified reference, with denominator \(atol_i + rtol\,|y_{i,ref}|\);
- maximum mass/invariant defect \(|y_1+y_2+y_3-1|\);
- minimum accepted/output value per component and maximum negative excursion;
- accepted/rejected steps, RHS evaluations, Jacobian evaluations, linear solves, nonlinear iterations, and error-test failures where exposed;
- wall-clock integration time as a separate repeated-measurement dataset—not in the deterministic numerical receipt;
- exact source SHA, checkout SHA, lock/build manifest digests, solver executable digest, input/config digest, output digest and raw-log digest.

A numerical candidate may qualify only if it completes, all values are finite, the common-grid normalized error is at most 10 for every component and time, the conservation defect is within \(10 \cdot (atol_1 + atol_2 + atol_3) + 100 \cdot machine\_epsilon\), and no accepted/output state is below \(-10 \cdot atol_i\). Do not clamp or project a failed state back onto the simplex. Record every failed case and preserve its artifacts.

Start with a protective cap of two million accepted-plus-rejected step attempts. Reaching the cap is a named failure mode rather than permission to raise the cap after observing which solver benefits. Change the cap only by publishing a new protocol revision with the resource rationale.

The initial thresholds qualify only this fixture and tolerance grid; they are not a claim of general correctness or physical validation.

## First implementation increment: bounded explicit diagnostic

The bounded first probe lives at `research/ode-verify-001/examples/robertson_explicit_probe.rs`. It invokes Symthaea's current Cash–Karp controller over the frozen 17-point output grid, with a 100,000-reported-attempt budget. It preserves named outcomes, accepted-endpoint invariant/positivity diagnostics, work counters, source/checkout SHAs, and deterministic output samples; CI runs the receipt twice and compares the bytes.

This is a **stress diagnostic**, not a qualified solver benchmark. It does not use the exploratory SciPy samples as an oracle, does not silently clip negative concentrations, and explicitly labels every receipt `UNQUALIFIED_DIAGNOSTIC_ONLY`. Its 100,000-attempt cap is a quick diagnostic budget, not the 2,000,000-attempt budget proposed for the full qualification matrix.

One API limitation is now made visible: `Rk45Controller::try_step` returns attempted/rejected-step counters for successful steps, but a failed call returns only `SimulationError` without the partial work counters or last attempted timestep. The probe wraps the ODE in an atomic derivative-callback counter, compares the wrapper count against each successful `Rk45StepReport`, and reports the total RHS callback count exactly even when the last call errors. The receipt still marks accepted/rejected internal-step totals incomplete on an error. A follow-up solver API should return a failure receipt containing last accepted state/time, attempted/rejected internal steps, RHS evaluations, minimum step reached, and the typed failure reason.

## Work–precision and runtime protocol

1. Run correctness gates before timing.
2. For each solver/profile, collect timing on a pinned, otherwise idle machine after documented warm-up runs. Use at least 21 timed repetitions and report median, interquartile range, minimum, and all raw measurements.
3. Separate cold-start/initialization time from integration time. Report compile time separately from both.
4. Plot error versus RHS evaluations and error versus wall time. A failed or unfinished case is not plotted as a successful fast point.
5. Compare only at achieved error levels supported by both methods. If no overlap exists, report that the comparison is inconclusive at matched accuracy.
6. Do not pool CI timings from heterogeneous or shared runners into an SOTA claim.

## Required negative controls

The verifier suite must reject deliberately mutated reference/config/output records, including:
- changed rate constant or initial condition with an unchanged fixture ID;
- missing, duplicated, reordered, or non-finite time/state samples;
- a wrong output grid or a stale reference digest;
- a solver status that says success while a completion flag or terminal time disagrees;
- negative concentrations concealed by clipping;
- a conservation defect hidden by rounded display values;
- a missing component-wise tolerance, or a changed tolerance vector with the same configuration digest;
- source/checkout/build/lock digests that do not match the run's raw artifacts.

Test these with synthetic fixtures before any solver comparison is considered qualified.

## Exit criteria and claim ceiling

The first milestone is **not** “beat CVODE.” It is to produce a deterministic, independently inspectable answer to:
- Where does the explicit method cease to complete within the fixed budget?
- At which tolerances does each solver satisfy common-grid accuracy, positivity, and conservation?
- What work does the explicit method perform before failure or qualification?
- Is the reference itself stable under tighter tolerances and independent cross-checking?

Only after that should the project consider an implicit BDF/Rosenbrock path, stiffness detection, or an automatic explicit-to-implicit switch. Any such method-selection claim should be evaluated against the same locked problem corpus with the decision logic included in the receipt.

**No qualified reference run, solver-comparison result, SOTA ranking, real-world chemical validation, or safety claim is asserted by this protocol document.**

## References

- SUNDIALS CVODE official documentation: https://computing.llnl.gov/projects/sundials/cvode
- SUNDIALS user notes on nonphysical negative solutions in Robertson kinetics: https://computing.llnl.gov/projects/sundials/usage-notes
- SciML solver benchmarking methodology: https://docs.sciml.ai/SciMLBenchmarksOutput/stable/
- Robertson model and stiff-benchmark overview: https://pydiffsol.readthedocs.io/en/latest/bench/robertson_ode.html
