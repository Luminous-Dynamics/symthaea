# Approximate-symmetry finite-horizon error envelope v0.1

First executable reference for [SYMM-ERROR-ENVELOPE-001 #12830](https://github.com/Luminous-Dynamics/symthaea/issues/12830).

## Why this is a distinct research layer

An approximate symmetry is not an equivalence relation. It cannot be substituted for exact group closure or used for exact quotient pruning. Its legitimate role is bounded comparison: quantify how far two represented trajectories, objective values, and constraint residuals may diverge over a finite horizon.

Every step-map Lipschitz bound and one-step defect bound carries its own non-empty certificate identifier. Objective/constraint Lipschitz and direct-defect premises have separate identifiers as well. The receipt binds these IDs together with all numeric assumptions; it does not verify the cited external certificates.

The reference propagates the recurrence

`e_{k+1} ≤ L_k e_k + δ_k`

using nonnegative fixed-point integer bounds, rational Lipschitz constants, checked multiplication/addition, and outward ceiling division. Defect bounds may vary per step.

## Domain guard

Propagation is enabled only when the caller supplies a non-empty external domain-invariance certificate identifier and a boundary-margin sequence for t=0 through the requested horizon. The engine does not verify that certificate; it binds the claim into the receipt.

If domain evidence is missing or contradicted, it abstains without propagation. If the accumulated state-error bound exceeds the reference orbit's boundary margin, it records that first crossing and stops. It does not extrapolate a Lipschitz bound outside the domain where it was assumed valid.

## Objective and constraint envelopes

For an objective with declared Lipschitz constant (L_J) and direct equivariance/measurement defect (delta_J), the objective error is bounded by

`E_J = ceil(L_J e_H) + δ_J`.

A nominal objective gap is certified stable only when it is strictly greater than (2E_J). Equality is unresolved.

For a constraint with Lipschitz constant (L_C) and direct defect (delta_C),

`E_C = ceil(L_C e_H) + δ_C`.

A positive nominal feasibility slack is certified to remain positive only when it is strictly greater than (E_C). Equality is unresolved.

If the domain guard stops early, objective and constraint assessments are not made.

## Tests

The reference includes:
- contractive (L<1), neutral (L=1), and expansive (L>1) exact integer recurrence oracles;
- horizons 1, 2, 4, 8, 16, 32 and 64;
- zero, constant and step-varying defect envelopes;
- initial and later domain-boundary crossings;
- strict objective-gap and constraint-slack thresholds;
- missing/contradicted domain evidence;
- arithmetic overflow and receipt-tampering controls.

## Reproduction

    cargo fmt --manifest-path research/symm-error-envelope-001/Cargo.toml --all -- --check
    cargo test --locked --manifest-path research/symm-error-envelope-001/Cargo.toml
    cargo clippy --locked --manifest-path research/symm-error-envelope-001/Cargo.toml --all-targets -- -D warnings

## Connection to Symtropy

The dense-coordinate force-free RATTLE one-step reference in [Symtropy PR #5870](https://github.com/Luminous-Dynamics/symtropy/pull/5870) is a future physical source for defect bounds and transformed-trajectory comparisons. It is not itself an approximate-symmetry certificate. First establish independently justified one-step defect bounds and a common invariant domain; only then feed them into this recurrence.

## Benchmark targets

Prospective targets, not results:
- zero false-safe classifications on exact toy oracles;
- 100% replay agreement for every receipt;
- 100% abstention when domain or Lipschitz premises are missing;
- bound ratio (certified upper / exact observed error) reported, with target median ≤2 on in-domain exact toy maps;
- report certified coverage, abstention, state-error envelope, objective/constraint classification, and receipt overhead separately.

## Claim boundary

This reference propagates supplied assumptions; it does not prove that the one-step defect, Lipschitz, boundary-margin, objective or constraint assumptions hold. It does not prove approximate group structure, infer confidence intervals, certify a physical system, authorize safety-critical actions, or emit hard-prune permission. A bound above tolerance means **no within-tolerance certificate was obtained**; it does not prove that the actual error exceeds tolerance.
