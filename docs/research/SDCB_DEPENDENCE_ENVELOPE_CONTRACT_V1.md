# SDCB Dependence-Regime Envelope Contract v1

**Status:** exact-source Rust 1.96 workflow passed on code head `0708fb7a5e20d0c0321aa8f5c3a07e04a15a4031`: Rustfmt check passed; `rustc -D warnings --test` compiled and all **10/10 negative-control tests passed** ([workflow run](https://github.com/Luminous-Dynamics/symthaea/actions/runs/46004270255)). This increment is stacked on #13990; it does not change production SDCB assessment.

## Motivation

The Gaussian and standardized Student-t(5) regime-shift experiments in #13986 and #13990 show that unchanged one-point marginals do not preserve the distribution of block-maximum residual scores when within-block dependence changes. The production decision must therefore not inherit a positive calibration result merely because marginal goodness-of-fit still passes.

This reference turns the research implication into a fail-closed state contract.

## Contract

The frozen `ValidatedDependenceEnvelopeV1` binds:
- exact scope;
- marginal family and its supported dependence-model pairing;
- dependence model;
- score definition;
- block length;
- nominal alpha;
- an inclusive lower/upper rho envelope;
- minimum diagnostic sample count;
- digest of the independent validation receipt;
- digest of the held-out regime set used to validate the envelope.

A runtime `DependenceObservationV1` binds the same scope/model/score/block/alpha metadata to a conservative rho interval, diagnostic sample count, declared model-adequacy state and evidence digest.

The classifier implements three outcomes:

| Outcome | Required condition |
|---|---|
| `QualifiedWithinEnvelope` | Exact metadata match; supported marginal/model pair; model assessment Adequate; sufficient diagnostic samples; canonical evidence digest; entire observed rho interval is inside the frozen bounds |
| `NotQualified` | Invalid envelope, scope/model/marginal/score/block/alpha mismatch, rejected dependence model, or observed interval entirely outside the frozen bounds |
| `Inconclusive` | Model assessment indeterminate, insufficient samples, invalid observation digest/interval, or interval overlaps an envelope boundary without being wholly inside |

Envelope bounds are inclusive. The whole uncertainty interval—not its midpoint or point estimate—must be contained for a positive result.

## Negative controls

The standalone Rust 2024 reference tests:
1. wholly-contained interval qualifies;
2. exact envelope boundaries are inclusive;
3. disjoint interval is NotQualified;
4. partial boundary overlap is Inconclusive;
5. block-length and marginal-family mismatch cannot inherit calibration;
6. rejected/inconclusive model diagnostics do not qualify;
7. insufficient samples or malformed observation digest are inconclusive;
8. reversed envelope bounds and non-finite observed intervals fail closed;
9. unsupported marginal-family/model pairing is an invalid envelope;
10. scope, model and alpha mismatches are rejected.

The focused workflow compiles with `-D warnings`, runs the exact source test suite and checks Rustfmt. It does not rely on the full workspace runner.

## Trust and scope boundaries

This is a **decision-contract reference**, not a complete statistical detector or cryptographic authorization system.

- The envelope's validation-receipt and held-out-regime hashes are shape-checked, not cryptographically authenticated here.
- The caller must establish that the validation receipt was produced by an independent evaluator, that the held-out regimes were truly excluded from fitting, and that the envelope itself was approved by an authorized policy owner.
- The observation's `ModelAssessmentV1` is an input from a separate diagnostic. This module does not prove that a stationary AR(1) model fits the data.
- A scalar rho interval does not capture arbitrary multivariate, nonlinear, lag-varying or latent common-cause dependence.
- The enum pairing currently supports only Gaussian AR(1) and standardized Student-t(5) common-scale AR(1), matching the tested research models. This does not imply they span real-world data.
- `QualifiedWithinEnvelope` means only "inside this exact frozen, declared and independently validated model/score envelope." It is not a theorem of conformal coverage or a guarantee of physical adequacy.
- Calibration episode custody, test episode separation, root/source evidence and revocation remain separate requirements; this module does not replace #13946 or the existing SDCB custody contract.

## Production integration gates

Before wiring this state machine into a production adequacy assessor:
1. Replace caller-supplied validation claims with a verified attestation/authorization boundary and immutable validation receipts.
2. Bind calibration and observation receipts to exact source episode IDs, time windows, block construction and score implementation digests.
3. Validate model adequacy with held-out diagnostics that include residual dependence beyond first-order rho.
4. Extend to multivariate dependence, cross-channel common causes, negative rho, heavy tails, drift, abrupt transitions and adversarial diagnostic failure.
5. Establish detection power and false-qualified rates on preregistered held-out regimes; report confidence intervals and compute cost.
6. Require `NotQualified` or `Inconclusive` to prevent any positive SDCB adequacy or downstream authorization.
7. Add independent integration tests showing that an out-of-envelope or indeterminate state cannot authorize a payment, safety claim, device operation or irreversible action.

The contract is intentionally small. The paradigm shift is to make the validity domain an explicit, reviewable input to every calibration-derived decision, not an implicit assumption attached to a confidence interval.
