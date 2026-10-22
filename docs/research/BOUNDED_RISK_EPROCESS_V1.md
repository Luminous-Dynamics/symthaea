# Anytime-Valid Bounded-Risk E-Process V1

Status: isolated Rust research prototype and paired synthetic benchmark. Exact-head qualification is required before this branch is called qualified. Nothing here changes the live Symthaea control loop.

## Why add a second sequential detector?

The CUSUM branch accumulates residual excess above a fixed allowance and calibrates its alarm threshold on a nominal synthetic panel. It is a useful change detector, but its false-alarm behavior depends on the calibration/test protocol and assumptions about the nominal residual sequence.

This branch adds a different object: a nonnegative e-process for a bounded risk signal. The key hypothesis is explicit. For losses \(L_t \in [0,1]\), suppose the safe-system null claims
\[
\mathbb{E}[L_t \mid \mathcal{F}_{t-1}] \leq \epsilon
\]
for the history before time \(t\). For a fixed predictable bet \(\lambda > 0\) satisfying \(\lambda\epsilon < 1\), the factor
\[
E_t = 1 + \lambda (L_t - \epsilon)
\]
is nonnegative and has conditional expectation at most one under that null. The product \(M_t=\prod_{s=1}^{t}E_s\) is then a nonnegative supermartingale. Ville's inequality bounds the chance that it ever crosses \(1/\delta\) by \(\delta\), under the stated conditional-risk assumption.

That assumption is distinct from exchangeability-based conformal calibration. A nominal calibration set alone does not establish a conditional risk bound under arbitrary distribution shift. This implementation makes the assumption, risk bound, bet and familywise allocation explicit instead of calling an empirical threshold “anytime valid.”

## Implementation

- tools/allostatic-control-bench/src/bounded_risk_eprocess.rs implements a bounded-loss e-process with fixed predictable bet, validation of all parameters and inputs, latched crossing, reset and a Bonferroni allocation over parallel monitors.
- Evidence accumulates in log space, avoiding silent floating-point underflow/overflow when a stream is long. Alarm threshold in log space is \(\log(K/\delta)\), where \(K\) is the number of parallel operating-point monitors and \(\delta\) is the total familywise budget.
- tools/allostatic-control-bench/src/bin/allostatic-risk-eprocess-bench.rs reads the exact raw 48,000-step input CSV from the CUSUM benchmark. It preserves the first 14 input columns byte-for-byte and appends normalized bounded loss, log e-value, latched alarm and the fixed risk parameters. It rejects a different schema, incomplete episode or mismatched full-episode flags.
- The qualifier runs the e-process benchmark twice, compares output bytes, checks the recurrence independently, verifies the original source trace is unchanged, and compares nominal/stealth alarms for paired seeds.

## Fixed synthetic design

- The input is the same 5 operating-point × 4 condition × 300 episode × 8 step panel as the CUSUM benchmark.
- The bounded loss is \(L_t=\min(r_t/0.05,1)\), where \(r_t\) is the task-residual score. The declared null risk bound is \(\epsilon=0.15\), the fixed bet is \(\lambda=0.5/\epsilon\), the global false-alarm budget is \(\delta=0.05\), and five operating-point monitors use threshold \(K/\delta=100\) (or \(\log 100\) in log space).
- These values are frozen for this fixture, not tuned on test episodes. The bounded-risk interpretation requires that nominal expected bounded loss be no greater than \(\epsilon\) conditional on past observations. The synthetic run checks basic behavior but cannot prove that assumption for a real system.
- The stealth control reuses exactly the nominal residual scores, while the scenario label records an unobserved dynamics shift. Neither the pointwise, CUSUM nor e-process monitor should appear to detect information that is absent from the monitored signal.

## Qualification gates

- Byte-identical repeated raw and summary outputs, exact 48,000 raw rows and 20 summary cells.
- Raw e-process output must retain the exact first 14 columns of the CUSUM input.
- Each bounded loss must equal the frozen score normalization; log e-values must match the multiplicative recurrence recomputed independently.
- The global threshold must equal \(\log(5/0.05)\), and the recorded risk bound, bet and familywise delta must equal their frozen configuration.
- Nominal false alarms no greater than 10% per operating-point cell; gradual-shift detection at least 80% even when all residuals remain below the pointwise threshold; abrupt-shift detection at least 90%.
- Stealth shift must produce the same e-process alarm counts as the paired nominal streams, with most latent shifts missed. That is a negative control, not a failure to exploit evidence that does not exist.
- Source and artifact SHA-256 hashes and row counts must be written to the provenance receipt.

## Comparison to current research

The conceptual reference is Timans, Verma, Nalisnick and Naesseth, “On Continuous Monitoring of Risk Violations under Unknown Shift,” UAI 2025, which uses testing-by-betting to monitor bounded risk violations in evolving streams. Their implementation also compares several e-process constructions; this code is a small fixed-bet bounded-risk baseline, not a reproduction of their full method or benchmark. [Paper](https://proceedings.mlr.press/v286/timans25a.html) · [Reference implementation](https://github.com/alextimans/risk-monitor).

For runtime control, also preserve:
- [SODA-MPC (L4DC 2025)](https://proceedings.mlr.press/v283/contreras25a.html): conformalized ensemble OOD monitoring and a reachability fallback.
- [Safety Beyond the Training Data: Robust OOD MPC via Conformalized SLS (L4DC 2026)](https://proceedings.mlr.press/v331/srinivasan26a.html): state/control-dependent weighted conformal model-error bounds integrated into robust MPC and forward reachable sets.

Those methods include physical or high-fidelity control demonstrations beyond this scalar synthetic benchmark. The current e-process implementation should not be described as matching or beating them.

## What would justify a stronger claim?

1. Reproduce at least one published runtime-assurance benchmark with its pinned dynamics, prediction model and original baseline.
2. Keep training, bet selection, risk-bound selection, calibration, and final shift families disjoint.
3. Match false-alarm budgets explicitly: e-process under a declared conditional-risk null, CUSUM under a frozen calibration protocol, and conformal OOD monitor under its exchangeability assumptions.
4. Report detection delay, false-safe exposure before fallback, reward/task success, no-certified-action rate, runtime, per-state false alarms and confidence intervals.
5. Test gradual drift, abrupt shifts, autocorrelated disturbances, delayed/missing telemetry, sensor bias and sensor faults—not just one score distribution.

A useful result would be fewer task-critical unsafe episodes at a matched nominal false-alarm budget without unacceptable reward or latency loss on held-out disturbances. This branch currently makes that comparison executable; it does not establish real-world safety or state of the art.
