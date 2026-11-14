# AIRROAD-ONE-SIDED-SET-INTERPOLATION-001 — exact endpoint semantics

**Status:** V0.1 exact-rational set-interpolation reference  
**Tracking issue:** [#11973](https://github.com/Luminous-Dynamics/symthaea/issues/11973)  
**Parent paper-transcription work:** [#11970](https://github.com/Luminous-Dynamics/symthaea/issues/11970)

> This reference validates a generic one-sided set operator on synthetic rational polytopes. It does not implement the complete sampled-FRF IQC method.

## 1. Mathematical contract

For adjacent frequency knots (\omega_k,\omega_{k+1}), the paper's one-sided interpolation uses the outgoing endpoint set at the left knot and the incoming endpoint set at the right knot:

[
\mathcal C_k(\lambda) =
(1-\lambda)\operatorname{co}(\Gamma_{S,k}^{+})
+\lambda\operatorname{co}(\Gamma_{S,k+1}^{-}),
\qquad \lambda\in[0,1].
]

The plus/minus endpoint labels are semantic. At an internal knot (k), the incoming set (\Gamma_{S,k}^{-}) and outgoing set (\Gamma_{S,k}^{+}) may be different. The operator does not impose continuity or insert an interpolation segment between those two sets.

For finite generator sets (A=\{a_i\}) and (B=\{b_j\}):

[
(1-\lambda)\operatorname{co}(A)+\lambda\operatorname{co}(B)
=
\operatorname{co}\{(1-\lambda)a_i+\lambda b_j\}_{i,j}.
]

The right side is a finite convex hull of all pairwise affine combinations. V0.1 checks this identity using exact rational 2D coordinates and a deterministic monotone-chain convex-hull oracle.

## 2. Frozen two-interval discriminator

Three frequency knots are declared at normalized frequencies 0, 1 and 2. The middle knot intentionally has distinct one-sided sets:

- incoming/left limit (\Gamma_{S,1}^{-}): the vertical segment from (4,0) to (4,2);
- outgoing/right limit (\Gamma_{S,1}^{+}): the horizontal segment from (10,10) to (12,10).

These are separate declarations, not a data gap. The incoming set terminates interval 0; the outgoing set begins interval 1.

Interval 0 uses:
- left endpoint: (\Gamma_{S,0}^{+}), segment (0,0)–(2,0);
- right endpoint: (\Gamma_{S,1}^{-}), segment (4,0)–(4,2).

Interval 1 uses:
- left endpoint: (\Gamma_{S,1}^{+}), segment (10,10)–(12,10);
- right endpoint: (\Gamma_{S,2}^{-}), segment (14,10)–(14,12).

At (\lambda=1/2), interval 0 produces the rectangle with vertices (2,0), (3,0), (3,1), (2,1). Interval 1 produces the rectangle with vertices (12,10), (13,10), (13,11), (12,11). These exact hulls make endpoint sign errors visible.

## 3. Required verifier checks

- Frequency ordering and interval adjacency.
- Same metadata identity across each interval's endpoints.
- Left endpoint resolves only to outgoing (+).
- Right endpoint resolves only to incoming (-).
- Pairwise generator combinations and exact convex hull at \(\lambda=0,1/2,1\).
- Inclusion of every pairwise affine combination.
- Equality of output under endpoint-generator permutation.
- Distinct incoming/outgoing sets at the middle knot.
- Rejection of invalid \(\lambda\), missing endpoint generators, wrong sign labels, stale identities and statistical/hard-set promotion.

The verifier is a research fixture oracle. It does not prove that a physical controller's RCF trajectory obeys the supplied endpoint polytopes.

## 4. Claim ceiling

V0.1 is an exact-rational, synthetic 2D set-interpolation reference only. It does not reproduce the paper's generalized-plant/RCF construction, form \(\Gamma_{M,k}^{\pm}\), solve an IQC, qualify an FRF, or establish physical/aircraft robust stability.
