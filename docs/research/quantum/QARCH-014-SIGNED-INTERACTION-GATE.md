# QARCH-014 — Validation-bounded signed interaction gate

**Issue:** [#9103](https://github.com/Luminous-Dynamics/symthaea/issues/9103)  
**Related proof audit:** [QARCH-012 #9092](https://github.com/Luminous-Dynamics/symthaea/issues/9092)  
**Related paper-equation reproduction:** [QARCH-011 #9088 / PR #9096](https://github.com/Luminous-Dynamics/symthaea/pull/9096)  
**Code:** `crates/domains/symthaea-quantum-comp/src/signed_interaction_gate.rs`  
**Fixtures:** `docs/research/quantum/benchmarks/QARCH-014-SIGNED-INTERACTION-GATE.json`

## Why a different method is needed

The QARCH-012 audit found that the printed R-IDeA absolute-gap gate does not identify the sign of the squared-error interaction. Theorem 2's Appendix C subset relation (iii), as printed, converts a difference of absolute values into a signed product without proving sign alignment.

QARCH-014 is a new conservative gate. It does not change the paper equations or claim the paper's theorem. It asks a different question: **given independently verified uncertainty bounds on both signed factors, can we prove the sign of their product at a particular design point?**

## Inputs and interval derivation

From Proposition 1, define:

[
E=\hat f-\bar f,\qquad M=\bar f-f^\star,\qquad A=E M.
]

The squared-error identity is:

[
(\hat f-f^\star)^2=(\bar f-f^\star)^2+(\hat f-\bar f)^2+2A.
]

Inputs are the current prediction \(\hat f(x)\), proxy prediction \(g(x)\), a separately verified uniform proxy bound \(|g-\bar f|\le\tau_2\), and a separate validation interval \(f^\star(x)\in[L,U]\).

Therefore:

[
\bar f\in[g-\tau_2,g+\tau_2]
]

[
E\in[\hat f-g-\tau_2,\hat f-g+\tau_2]
]

[
M\in[g-\tau_2-U,g+\tau_2-L].
]

The signed product interval is the minimum and maximum of the four endpoint products. This is conservative for interval-valued E and M, without assuming independence between them.

A signed direction is reported only when the entire A interval is below -margin or above +margin. If it crosses either boundary, the status is ambiguous. If it lies inside the margin, the interaction is neutral within the declared tolerance.

## Coverage and evidence

The proxy certificate and true-function validation interval have separate coverage lower bounds \(c_g\) and \(c_f\). The implementation reports the union-bound joint lower bound:

[
c_{joint}=\max(0,c_g+c_f-1).
]

No independence assumption is required. If joint coverage misses the preregistered minimum, status is INSUFFICIENT_COVERAGE even if the numerical interval has a clear sign.

Four immutable artifact IDs and four split IDs are required and must each be distinct: proxy fit, proxy-bound validation, true-function validation and evaluator. Distinct digests alone do not prove row-level separation; the upstream provenance verifier must resolve the source manifests and confirm disjoint partitions, artifact content, target domain, model version, interval construction and coverage statements. The Rust function accepts verification flags from that upstream verifier; it does not independently establish the validity of a certificate.

Importantly, the deployable API takes a **true-function validation interval**, never an evaluator-only true-function point value. The evaluator artifact is recorded for downstream scoring only.

## Frozen examples

The fixture covers:
- exact offsetting and reinforcing signs with identical form of inputs;
- a wide interval crossing zero, which must be ambiguous;
- adequate pointwise bounds but insufficient joint coverage, which must not certify;
- an unverified validation interval, which must not certify.

These are synthetic analytic inputs for the interval arithmetic and gate logic. They do not provide proxy certificates, physical validation intervals, or photonic hardware evidence.

## Acceptance gates

- Zero false signed certifications against exhaustive interval endpoint oracles.
- 100% agreement with known interval-product and union-bound fixtures.
- A wider interval may reduce certification rate but must not increase false-certification rate.
- Report certified-offsetting, certified-reinforcing, neutral, ambiguous, insufficient-coverage and evidence-unverified rates separately.
- Compare against magnitude-only DeA and measure the abstention/decision-regret cost using QARCH-009.
- Keep the new gate separate from published R-IDeA reproduction until the paper's sign proof is corrected or its assumptions amended.

## Current implementation boundary

The module implements conservative interval arithmetic, distinct-artifact checks, evidence-verification gating, union-bound coverage composition and nine unit tests. It does not build the proxy model, estimate the true-function interval, prove the proxy sup-norm bound, or guarantee lower global test risk. Rust compilation/tests have not been run in this environment because the Rust toolchain is unavailable.