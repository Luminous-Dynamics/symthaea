# Phi Metric Verification Audit

> Created: 2026-02-17
> Status: COMPLETE — Critical findings documented
> Audited by: Automated deep audit of phi-lab/ and symthaea/ codebases

---

## Executive Summary

Multiple implementations across phi-lab and symthaea compute **different metrics** but label them all "Phi" or claim they measure IIT integrated information. The metrics have **fundamentally different topology preferences** and **weak negative correlation** (Pearson r = -0.14, Spearman rho = -0.59) between SpectralConnectivity (lambda2) and ExhaustivePartition (Exact IIT Phi). Note: the production SpectralMIPFinder (MI Laplacian + Fiedler + MIP sweep on ContinuousHV covariance) is a distinct algorithm whose correlation with Exact remains unknown.

**P1 (HAI)**: SAFE — honestly labels lambda2 as a proxy, documents limitations.
**P7 (Master Equation)**: AFFECTED — claims IIT Phi but uses lambda2.
**P12 (Temporal Topology)**: CONFIRMED AFFECTED — claims IIT Phi, makes false PyPhi validation claim.
**All other papers**: See matrix below.

---

## Implementations Found

### 1. `phi-lab/src/hdc/phi_real.rs` — Algebraic Connectivity (lambda2)

| Property | Value |
|----------|-------|
| **Metric** | 2nd smallest eigenvalue of normalized Laplacian |
| **Formula** | L_norm = I - D^(-1/2) A D^(-1/2), then eigendecompose |
| **Complexity** | O(n^3) |
| **Max tractable n** | ~256 |
| **Code header says** | "Phi (Integrated Information) Calculation for RealHV" |
| **Algorithm section says** | "Calculate algebraic connectivity (2nd smallest eigenvalue)" |
| **Is this IIT Phi?** | **NO** — lambda2 measures graph mixing time, IIT Phi measures irreducible information |
| **Used by** | ALL 260 measurements in temporal topology paper, P7 Master Equation |

### 2. `phi-lab/src/hdc/tiered_phi.rs` — 4-Tier Approximation

| Tier | Complexity | What it computes | Is it IIT? |
|------|-----------|-----------------|------------|
| 0 (Mock) | O(1) | Deterministic formula: phi = 0.1*n + 0.3 | **NO** |
| 1 (Heuristic) | O(n) | phi ≈ 1 - avg_pairwise_similarity | **NO** |
| 2 (Spectral) | O(n^2) | Same as phi_real.rs (lambda2) | **NO** |
| 3 (Exact) | O(2^n) | Exhaustive minimum information partition | **YES** (but intractable for n>8) |

Default: Tier 1 (Heuristic) — a made-up metric, not IIT.

### 3. `symthaea-core/src/hdc/iit_exact.rs` — True IIT 3.0

| Property | Value |
|----------|-------|
| **Metric** | True IIT 3.0 via TPM → cause-effect repertoires → EMD → MIP |
| **Complexity** | O(2^(2^n)) for full MICS, O(2^n) for MIP per mechanism |
| **Max tractable n** | 8 nodes |
| **PyPhi validated?** | YES — passes 6/6 theory tests |
| **Used by default?** | **NO** — never invoked by default due to intractability |

### 4. `symthaea-core/src/phi_engine/mod.rs` — PhiEngine (Runtime Selection)

| Property | Value |
|----------|-------|
| **Default method** | SpectralConnectivity (lambda2) |
| **Code comment** | "Previously named 'Continuous'; renamed since this is NOT IIT Phi but Fiedler value (lambda2)" |
| **Auto selection** | n <= 256: SpectralConnectivity, n > 256: Resonator |
| **iit_exact ever used?** | Available via Tiered(Exact) but never selected by Auto |

---

## Empirical Divergence

From `phi-lab/papers/temporal_topology_consciousness/CRITICAL_FINDINGS.md`:

| Comparison Metric | Value | Meaning |
|-------------------|-------|---------|
| Pearson r (SpectralConnectivity lambda2 vs ExhaustivePartition Exact) | **-0.14** | Weak negative linear correlation |
| Spearman rho (SpectralConnectivity lambda2 vs ExhaustivePartition Exact) | **-0.59** | Moderate negative rank correlation — rankings substantially divergent |
| SampledPartition (Heuristic) Pearson r vs Exact | **0.9998** | Near-perfect linear correlation |
| SampledPartition (Heuristic) Spearman rho vs Exact | **0.9985** | Near-perfect rank correlation |

> **Note**: The earlier reported r = 0.0972 / rho = 0.0070 was from a prior methodology and was
> attributed to the wrong algorithm tier. The corrected values above distinguish SpectralConnectivity
> (lambda2, which measures graph mixing time, not IIT integration) from SampledPartition (Heuristic,
> which closely tracks Exact IIT Phi). The production SpectralMIPFinder (MI Laplacian + Fiedler +
> MIP sweep on ContinuousHV covariance) is a distinct algorithm whose correlation with Exact is UNKNOWN.

Example rank divergences:
- Line graph: lambda2 rank 19, Phi rank 6
- Klein Bottle: lambda2 rank 5, Phi rank 18
- 4 topologies (Lattice, Torus, Klein Bottle, Hypercube 4D): IIT Phi = 0.000, lambda2 > 0.5

**Conclusion**: lambda2 and true IIT Phi measure fundamentally different properties. Using one as a proxy for the other is scientifically invalid.

---

## Paper Impact Matrix

| Paper | What it claims | What it computes | Honest? | Action |
|-------|---------------|------------------|---------|--------|
| **P1 (HAI)** | "lambda2 as proxy for IIT Phi... intractable beyond 12 nodes" | lambda2 | **YES** | No action needed |
| **P7 (Master Equation)** | "C = f(Phi, B, W, A, R)" where Phi is IIT integration | lambda2 via phi_real.rs | **NO** | MUST reframe: rename Phi to lambda2, retarget venue |
| **P12 (Temporal Topology)** | "measure Integrated Information (Phi)", "r=0.994 with PyPhi" | lambda2 via phi_real.rs | **NO** | MUST reframe as spectral topology, remove PyPhi claim |
| **HK papers** | K-Index (different metric entirely) | K-Index | N/A | Safe — not affected |
| **Kosmic papers 1-5** | K-Index | K-Index | N/A | Safe — not affected |
| **P4 (K-Index Framework)** | Phi as one of 5 theory scores | Internal computation | Needs check | Verify what P4's "IIT score" computes |
| **Phi-Lab satellites (02-15)** | All depend on P7's Master Equation | Likely lambda2 | **Blocked** | Cannot proceed until P7 is reframed |

---

## False Validation Claim

**Location**: `phi-lab/papers/temporal_topology_consciousness/main.md`, line 88
**Claim**: "All Phi calculations validated against PyPhi... r = 0.994"

**Evidence this is FALSE**:
1. PyPhi validation feature is gated (`#[cfg(feature = "pyphi")]`)
2. Feature appears never compiled by default
3. No results file exists in repository
4. Actual correlation when tested: SpectralConnectivity (lambda2) vs Exact has Pearson r = -0.14 (not 0.994). Lambda2 measures graph mixing time, not IIT integration.

**Severity**: Unsubstantiated claim. Must be removed before submission.

---

## Recommendations

### For P7 (Master Equation) — BLOCKED

**Option A (RECOMMENDED)**: Reframe as spectral-cognitive metric
- Rename Phi component to "lambda2 (algebraic connectivity)" throughout
- Rewrite abstract: focus on "spectral connectivity as a component of consciousness measurement"
- Retarget: PLoS Comp Bio (instead of Nature Neuroscience)
- Core contribution (five-component equation) remains novel and publishable

**Option B**: Use true IIT Phi from iit_exact.rs
- Limited to n <= 8 nodes — severely constrains the paper
- Would take weeks of recomputation
- May not produce useful results (Phi degenerates to 0 for weighted matrices)

### For P12 (Temporal Topology) — CONFIRMED AFFECTED

- Reframe as "spectral topology" paper (already drafted: `letter_reframed.md`)
- Remove ALL IIT/Tononi claims
- Remove PyPhi validation claim (r = 0.994)
- Add honest statement: "SpectralConnectivity (lambda2) shows weak negative correlation (Pearson r = -0.14, Spearman rho = -0.59) with ExhaustivePartition (Exact IIT Phi); lambda2 measures graph mixing time, not IIT integration"
- Core finding (99.2% optimal 3D small-world, lambda2 → 0.5 asymptotically) remains novel
- Retarget: Network Neuroscience or Journal of Complex Networks

### For P1 (HAI) — NO ACTION NEEDED

Already honest. Lines 475, 508, 513, 517 explicitly document lambda2 as proxy with limitations.

### For Phi-Lab Satellites (02-15) — WAIT

All 15 depend on P7. Cannot proceed until P7 is reframed and accepted.

---

## What Remains Valid (Despite the Mismatch)

1. **Topology matters** for both metrics (confirmed empirically)
2. **3D structures perform well** across both lambda2 and IIT Phi
3. **lambda2 → 0.5 asymptotically with dimension** (mathematically provable, independent of IIT)
4. **Energy efficiency claims** are unrelated to Phi metric choice
5. **Symthaea's Phi proxy framing** is scientifically appropriate
6. **iit_exact.rs** in symthaea-core is a correct IIT 3.0 implementation (just intractable at scale)

---

## Documentation References

- Lambda2 vs IIT divergence data: `phi-lab/papers/temporal_topology_consciousness/CRITICAL_FINDINGS.md`
- Reframed version of P12: `phi-lab/papers/temporal_topology_consciousness/letter_reframed.md`
- True IIT implementation: `symthaea/symthaea-core/src/hdc/iit_exact.rs`
- PhiEngine architecture: `symthaea/symthaea-core/src/phi_engine/mod.rs`
- HAI paper honest treatment: `symthaea/papers/latex/hai_paper.tex` (lines 475-520)

---

*Audit complete. The honest path forward is reframing, not recomputing.*
