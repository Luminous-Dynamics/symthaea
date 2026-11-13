# MACE-PI-001 protocol

Date: 2026-11-10. Status: deterministic finite-cohort research reference.

## Question

When an agent learns from an observed action and outcome, what effect range is compatible with the observed finite cohort and the assumptions actually declared? Can that range justify a thresholded decision, or must the system return NOT_IDENTIFIED?

## Estimand and assumptions

For binary treatment A_i and binary outcome Y_i, let Y_i(1), Y_i(0) be the potential outcomes for unit i. The finite-cohort average treatment effect is:

ATE_N = (1/N) Σ_i [Y_i(1) - Y_i(0)].

The current reference requires the cohort to be fully enumerated and declares consistency, no interference, and a well-defined binary treatment. Those assumptions are required input declarations, not externally authenticated facts. The observed assignment fixes exactly one potential outcome per unit; the other binary potential outcome is unknown and enumerated.

No exchangeability or randomization is presumed.

## Monotone-treatment-response sensitivity

MTR states Y_i(1) ≥ Y_i(0) for each unit. Rather than silently imposing it, the reference accepts a maximum number k of individual MTR violations:
- k=0: MTR imposed;
- k=N: no MTR restriction;
- intermediate k: finite-population sensitivity analysis.

The dynamic programme minimizes/maximizes the total finite-cohort effect for each exact violation count, then forms the cumulative sharp interval for at most k violations. It uses exact rational arithmetic.

## Decision rule

For threshold τ and identified interval [L,U]:
- ROBUST_INTERVENE if L > τ;
- ROBUST_MAINTAIN if U ≤ τ;
- NOT_IDENTIFIED otherwise.

Equality at the threshold is deliberately not promoted to intervention. The observed treated-minus-control difference is reported separately as association, not as the causal estimand.

## Scope, exclusions and safeguards

- Exact regime match only; units from other regimes are excluded and reported, never pooled as if transportability were established.
- Duplicate identities, omitted treatment/outcome values, unknown JSON fields, multiple JSON payloads, undeclared assumptions and invalid sensitivity budgets fail closed.
- The reference has a deterministic maximum cohort size of 4,096.
- Dataset digest is content identity only. It does not authenticate issuers, prove randomization, or establish data truth.
- Exact finite-cohort bounds have no sampling uncertainty and are not population confidence intervals.
- A declared assumption can be false. Sensitivity analysis helps expose consequences; it does not verify assumptions.

## Local execution

Go 1.23.2, standard library only:
- go test -count=1 ./...
- go test -race -count=1 ./...
- go vet ./...
- go test -cover ./...
- go run . --input fixtures/example.json
- go run . --input fixtures/example.json --mtr

Latest local result: 13/13 top-level tests; 256 generated cohorts match an independent exhaustive counterfactual completion oracle for every MTR violation budget; race/vet/gofmt clean; 77.4% statement coverage.

## Canonical illustrative fixture

Observed association +2/3. Unrestricted sharp finite-cohort ATE bounds [-1/4,+3/4]. MTR-only bounds [0,+3/4]. One allowed MTR violation restores the lower bound to -1/4. At τ=+1/10, result is NOT_IDENTIFIED under both unrestricted and MTR-only arms.

This is a four-unit exact illustration, not an empirical effect or population claim.
