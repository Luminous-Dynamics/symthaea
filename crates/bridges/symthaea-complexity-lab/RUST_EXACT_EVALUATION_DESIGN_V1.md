# ETH-027 — Rust exact evaluation design v1

**Purpose:** specify the next Rust implementation boundary before introducing uncompiled production code. This document is a design, not an implementation or build claim.

## 1. Reuse the existing algebra boundary

The complexity-lab crate already exposes ExactSemiring in src/semiring.rs and a generic exact factor-graph eliminator. Existing exact scalar/count paths are bounded (CheckedCount(u128) and WeightProfile with checked i128 coefficients). The latter explicitly describes itself as a bounded reference representation, not arbitrary precision.

The new projected circuit evaluator should reuse the generic algebra idea but should not silently route Q or Q(i) through i128 or a floating-point profile.

The checked-in lockfile already contains:
- num-bigint 0.4.6
- num-rational 0.4.2
- num-traits 0.2.19

The implementation should add these as explicit direct dependencies of the complexity-lab crate if needed, preserving the versions already resolved in the repository lockfile.

## 2. Coefficient types

Use distinct semantic types rather than one untagged numeric enum:

- BigNatural: arbitrary-precision nonnegative integer for #SAT/PMC support counts.
- ExactRational: normalized BigRational for WMC/PWMC over Q.
- ExactGaussianRational: (real: BigRational, imaginary: BigRational) for complex AMC over Q(i).

ExactGaussianRational multiplication is (a+bi)(c+di) = (ac-bd) + (ad+bc)i. Do not coerce Q to Q(i) implicitly at an artifact boundary; any embedding must be declared in the contract and serialized explicitly.

Canonical wire values are reduced numerator/positive-denominator decimal strings. Gaussian values serialize both rational components independently. Zero is canonicalized as 0/1.

## 3. Circuit IR and trust boundary

Initial IR nodes:
- False, True
- Literal { variable, polarity }
- And { children }
- Or { children, determinism_receipt }

Every node has a stable ID, a declared/computed sorted scope and a representation digest. DAG sharing is allowed. Cycles, missing references, duplicate node IDs, invalid literal variables and malformed topological order fail closed.

Structural validation is performed independently of arithmetic:

- AND children must have pairwise disjoint scopes (decomposability).
- OR children must be pairwise model-disjoint. This is not decidable from the gate shape alone in general; require a checked certificate/receipt from the selected proof adapter. Never accept a producer-supplied Boolean deterministic=true flag as proof.
- The OR parent scope is the union of child scopes. Smoothing need not be materialized in the DAG if the evaluator performs explicit scope completion.
- The root is completed to the contract's declared output universe (visible variables for PMC/PWMC).

The SCPOG adapter must bind source CNF bytes, visible-variable partition, projected output bytes, proof bytes and exact checker identity/binary digest. A CertifiedProjection receipt is created only after the independent checker reports acceptance for those exact digests. The arithmetic evaluator does not promote a projection proof into a weighted-value proof.

## 4. Bottom-up exact evaluation

For a deterministic OR with parent scope S and child scope S_i:

OR_value = sum_i(child_value * product_{v in S \\ S_i}(w_v(false)+w_v(true))).

For a decomposable AND:

AND_value = product_i(child_value).

After evaluating the root, complete variables in the contract's declared universe that do not occur in the root scope by the same weight-sum factor.

For PMC, the visible literal weights are (1,1), so each missing visible variable contributes 2 to support cardinality. For PWMC use exactly the declared visible rational weights. Hidden variables are absent from the projected circuit's weight map. Full WMC and AMC use the full assignment universe and their corresponding coefficient domain.

Maintain two independent values for projected contracts:
1. support cardinality in BigNatural;
2. arithmetic value in ExactRational or ExactGaussianRational.

An exact arithmetic zero never clears nonempty support.

## 5. Cache and reuse model

Separate structural caches from arithmetic caches:

- Scope/decomposability metadata is keyed by circuit digest and IR version.
- The projection certificate is keyed by source digest, parser profile, visible/hidden partition digest, target representation digest, proof digest and checker identity.
- Arithmetic memoization is keyed by circuit digest, coefficient-domain ID/version and weight-map digest.

A changed weight map may reuse the certified support circuit and structural metadata, but must not reuse arithmetic node values from a different weight-map digest. Each final scalar result has its own result digest and replay receipt.

## 6. Independent replay

The first Rust evaluator should be bottom-up and DAG-memoized. The independent arithmetic replay should be a structurally separate implementation (for example, a recursive/reference evaluator over the same immutable DAG with separate memoization and different traversal). Agreement between two functions sharing the same gate-value routine is not independent verification.

The source-to-projected-support claim remains delegated to the checked SCPOG certificate. The arithmetic replay establishes only the scalar evaluation of that already-certified representation under the exact weight-map digest.

## 7. Required Rust tests

- Arbitrary-precision growth beyond i128 for #SAT, Q and Q(i).
- Rational sign normalization, gcd reduction, denominator positivity, and canonical zero.
- Gaussian multiplication, addition and complex cancellation.
- Decomposable AND acceptance/rejection, including overlapping child scopes.
- Deterministic OR only with a valid verifier receipt; reject missing, stale or wrong-output receipts.
- Non-smooth OR completion for multiple missing variables.
- Root completion for variables absent from the root scope.
- Shared DAG child evaluated once per digest/domain/weight-map key.
- Same support reused across several Q maps and Q(i) maps without arithmetic-cache contamination.
- Exact zero with nonempty support.
- Cycle, malformed reference, duplicate ID and out-of-universe literal rejection.
- Mutation controls that corrupt one proof digest, scope, weight, child edge or result byte.

## 8. Acceptance gates

1. The crate compiles and tests locally on the pinned toolchain before opening a non-draft implementation PR.
2. A bounded exhaustive Rust oracle agrees with the existing Python reference on a generated corpus, while a second Rust replay agrees independently.
3. No production path uses f64 or unchecked overflow for these coefficient domains.
4. The SCPOG adapter reproduces at least one pinned upstream example before any claim of certified projected compilation.
5. The complete receipt chain identifies exact source, parser, representation, certificate, checker binary, domain, weight map and result digests.

This v1 design intentionally avoids a premature Rust implementation because the current execution environment has no Rust compiler/Cargo available for local validation. The next code change should be made only where the pinned compiler and dependencies can be run; CI skipped is not a substitute.
