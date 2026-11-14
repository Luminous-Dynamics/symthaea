# Adaptive U(1)-TEBD: step-doubling and error-budget policy

## Purpose

The fixed-step Strang integrator is extended with a controller that separates two numerical error sources:

1. **MPS compression error** from the charge-block Schmidt projections.
2. **Trotter time-discretization error** from replacing continuous evolution by the second-order even/odd/even Strang circuit.

It does not combine both into one unlabelled residual.

## Step-doubling method

For requested interval \(h\), compute:

- one coarse step \(S_h(\psi)\);
- two fine steps \(S_{h/2}(S_{h/2}(\psi))\).

Measure the state-vector distance \(\delta\) between the two results by MPS norm/overlap contractions; no dense state-vector expansion is used.

Let \(e_c\) be the globally qualified compression bound for the coarse path and \(e_f\) the propagated sum of the two fine-path compression bounds. The first half-step compression error is multiplied by an operator-norm upper bound for the second half-step gate product before adding the second half-step error.

The exact discrete-circuit coarse/fine difference is therefore enclosed by:

\[
\max(0,\delta-e_c-e_f)
\leq
\|\psi_h^{\mathrm{ideal}}-\psi_{h/2,h/2}^{\mathrm{ideal}}\|_2
\leq
\delta+e_c+e_f.
\]

For a second-order Strang method in its asymptotic regime, the fine path's local Trotter defect is estimated by one third of the coarse/fine difference. The API reports the corresponding lower and upper **estimates** after applying the compression uncertainty.

This Richardson factor is an asymptotic numerical estimator, not a rigorous global Trotter theorem. The controller's compression components are only called globally qualified when every constituent TEBD path has passed the canonical-gauge receipt checks.

## Acceptance and rejection

A step is accepted only if both conditions hold:

- fine-path compression upper bound <= configured compression budget;
- upper end of the asymptotic Trotter-error estimate <= configured Trotter budget.

On acceptance, the result returns the two-half-step state and advances by the requested interval. On rejection, it returns no accepted state and recommends a smaller interval based on a cubic controller law, with safety factor and configured min/max scale clamps.

If any coarse or fine result lacks a global canonical-gauge compression receipt, the controller fails closed with \`AdaptiveGlobalErrorUnqualified\`. It does not silently revert to a local Frobenius residual or an empirical-only estimate.

## Explicit limits

- The Trotter estimate assumes the tested interval is in the asymptotic second-order regime. A refinement ladder should verify ratios near 4 before relying on quantitative controller behavior.
- The global compression bound refers to the discrete gate paths and is separate from the Trotter defect.
- Floating-point/model/parameter uncertainty is not covered by these two budgets.
- The first version is a single-attempt decision function. The caller owns retry scheduling, minimum-step termination, and total simulated-time accounting.
- No adaptive order selection, commutator-based estimator, Krylov reference, or higher-order integrator is claimed.

## Qualification targets

- Zero Hamiltonian: zero coarse/fine distance, accept, preserve norm and state.
- Noncommuting three-site hopping: step doubling detects nonzero error; an intentionally tight Trotter budget rejects and recommends a smaller step.
- Compression-heavy state: report coarse and fine compression bounds separately from the Trotter estimate.
- Missing global gauge receipt: fail closed before returning an accepted state.
- Over a refinement ladder, verify the measured second-order regime before interpreting the factor-1/3 estimate quantitatively.
- Compare adaptive and fixed-step runs at matched dense-oracle error and matched observable error; report total local gate applications and wall time.

## Research opportunity

The longer-term paradigm is an **error-budget allocator**: assign separate user budgets to Trotter error, MPS compression, floating-point residuals, and model uncertainty; then let the controller allocate work to time-step refinement, bond growth, and stronger precision. The current API implements only the first two budgets and keeps their receipts distinct.
