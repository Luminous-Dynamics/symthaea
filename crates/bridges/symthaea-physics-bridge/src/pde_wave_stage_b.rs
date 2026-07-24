// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! PDE discovery, Stage B (verification milestone): true Noether currents.
//!
//! Stage A (`pde_wave_stage_a.rs`) finds a single *global* conserved scalar
//! `E = Σᵢ ρᵢ` for a discretized field. That's a materially weaker claim than
//! a genuine Noether current: a local density `ρᵢ` and flux `J_{i+1/2}`
//! satisfying the discrete continuity equation
//!
//! ```text
//! dρᵢ/dt = J_{i+1/2} − J_{i−1/2}     for every free grid point i, at every time
//! ```
//!
//! This module builds and verifies the hand-derived `(ρ, J)` pair for the
//! wave chain, against
//! [`symthaea_core::hdc::conjecture_engine::discrete_continuity_residual`]
//! (M0/M1), and attempts GP discovery of `J` given the known `ρ`
//! (M2, `discover_flux_given_density`).
//!
//! **M0/M1 status**: done. A negative control (a wrong flux) fails the
//! checker, confirming it's actually checking something.
//!
//! **M2 status**: infrastructure built and working (gauge-fixing,
//! cross-validation, symbolic-closeness probing), but the search itself is
//! an **honest negative** at the budgets tried so far. M2.1 diagnosed *why*
//! (see `flux_discovery::tests::reachability_estimate_for_the_needed_
//! structural_motifs` and `component_fitness_landscape_around_true_flux`
//! below): the fitness landscape offers no partial credit toward the
//! answer (every near-miss scores worse than the trivial `0` baseline),
//! and the needed structural motif appears in only 0.022% of randomly
//! generated trees. M2.2 tested whether decomposing the search into two
//! independently-evolved factors (`discover_flux_factorized`, `J = A*B`)
//! solves the structural credit-assignment problem -- also an honest
//! negative (0/5 seeds), though with a real qualitative signal: some
//! factorized runs' best candidates contain genuine multi-variable cross
//! terms that almost never appeared in the blind single-tree search. M2.3
//! restricted the same factorized search to a strict polynomial grammar
//! (matching `j_truth()`'s actual grammar exactly) to test whether trig/
//! log/div/pow dilution was the dominant obstacle -- also 0/5, no
//! meaningful improvement over M2.2 at the identical budget/seeds, though
//! with a striking asymmetry: factor `B` (the velocity side) repeatedly
//! converged to something algebraically equal to `v_c+v_r` across several
//! seeds, while factor `A` (the position side) never once resembled
//! `u_r-u_c` in any printed candidate.
//!
//! **M2.4a** ran a three-arm causal comparison (no dedup / direct-vector
//! semantic dedup / HDC semantic dedup) via
//! `discover_flux_factorized_with_dedup`. Still 0/5 recovered in every arm
//! (see `m24a_dedup_three_arm_comparison_n3`), but with a genuine, important
//! **methodological finding**: Arm 2 (direct-vector) and Arm 3 (HDC)
//! produced byte-identical results on every seed -- not an "HDC provides no
//! advantage" empirical result, but a mathematical necessity. The HDC
//! encoding used (`hdc_fingerprint`, mirroring `DimensionalEncoder`'s
//! basis-scaled-bundle-then-normalize pattern) embeds via an *orthonormal*
//! basis (`ContinuousHV::orthogonal_set` does Gram-Schmidt + unit-normalize
//! every vector), which makes the embedding a linear isometry: cosine
//! similarity is exactly preserved between the raw fingerprint and its HDC
//! encoding. The comparison as built could not have found a difference
//! between Arms 2 and 3 regardless of whether one exists -- a real design
//! flaw, caught empirically (five seeds agreeing to the exact integer is
//! what surfaced it), not glossed over. A genuine HDC-vs-vector test needs
//! a *non-isometric* encoding (binary/quantized hypervectors, a redundant/
//! overcomplete random projection, or a nonlinear encoding step). Dedup
//! itself (Arms 2/3 vs Arm 1) gave a modest improvement in mean best
//! training residual (0.643 vs 0.705) but no recovery, alongside a striking
//! ~39% false-merge rate among rejected duplicates at threshold 0.98 -- a
//! real signal that the threshold may be too loose, discarding candidates
//! whose actual fitness differs meaningfully.
//!
//! **M2.4a-corrected** (`m24a_corrected_hdc_arm_n3`) re-ran the HDC arm with
//! a genuinely non-isometric encoding: `DedupMode::HdcQuantized` lossily
//! buckets the raw fingerprint (`quantize_fingerprint`, bucket width 0.25)
//! *before* HDC encoding, so two behaviorally close-but-distinct candidates
//! can now collapse to the same bucketed fingerprint even though their raw
//! cosine similarity differs -- verified non-isometric in isolation by
//! `quantized_hdc_similarity_can_differ_from_raw_vector_similarity`. This
//! time Arm 2 (vector) and Arm 3' (HDC-quantized) produced genuinely
//! different numbers, confirming the isometry bug is actually fixed:
//! duplicates_rejected 55021 vs 56240, false_merges 21556 vs 20775,
//! mean_best_train_residual 0.643 vs 0.681. **Still an honest negative on
//! every axis that matters**: 0/5 recovered in both arms (matching Arm2's
//! original M2.4a run exactly, as expected from identical seeds/config --
//! a useful sanity check that nothing else regressed), and the
//! HDC-quantized arm's mean best residual is *worse* than the plain vector
//! arm's, not better. There is no evidence in this experiment that a
//! non-isometric HDC encoding offers an advantage over direct-vector
//! cosine similarity for this dedup task; if anything the direct-vector
//! arm performed marginally better. This closes the HDC-vs-vector question
//! for M2.4a's dedup mechanism as investigated -- a genuine negative result,
//! not a methodological artifact.
//!
//! **M2.5** tested target-blind fitness shaping: instead of the raw
//! discrete-continuity residual (which M2.1 diagnosed as offering no
//! partial credit -- every near-miss scores worse than trivial `0`),
//! score candidates via `shape_calibrated_residual` (see
//! `symthaea_core::hdc::conjecture_engine::shape_calibrated_residual`'s
//! docs) -- shape alignment with an analytically-fit optimal scale,
//! algebraically `1 - alignment²`, never referencing the known-correct
//! flux. A frozen landscape audit
//! (`m25_landscape_audit_shape_calibrated_residual`) confirmed the new
//! score genuinely distinguishes controlled near-misses (real partial
//! credit exists, though weak -- alignment magnitude under 0.04 -- for
//! single-factor-only candidates specifically), clearing the predeclared
//! gate to run the real comparison.
//!
//! **The paired 10-seed evolutionary comparison
//! (`m25_shaped_vs_raw_residual_n3`) is an unambiguous, fully-diagnosed
//! honest negative -- and not merely "no improvement": every one of the
//! four predeclared go/no-go criteria failed, and on the two continuous
//! metrics that showed any difference, the shaped arm was actively
//! *worse* than the raw-residual baseline.** Real numbers (10/10 seeds
//! each arm): recovered 0/10 both arms (tied). Mean best training
//! residual: raw 0.621 vs shaped 1.235 (shaped is ~2x worse). Mean
//! held-out residual: raw 0.748 vs shaped 1.414 (~1.9x worse). Train-val
//! gap: raw 0.127 vs shaped 0.179 (shaped generalizes slightly worse
//! too). Mean complexity: raw 13.5 vs shaped 18.5 (shaped candidates are
//! more bloated for a worse result). Structural-motif onset: 0/10 seeds
//! in *either* arm ever showed the full two-factor motif simultaneously
//! (tied at zero, no improvement).
//!
//! The most diagnostic single number: the shaped arm's own
//! final-generation mean alignment -- the exact quantity its fitness
//! directly selects for across 100 generations -- is 0.0241, essentially
//! zero. Meanwhile the raw-residual arm, which never optimizes for
//! alignment at all, incidentally reaches a much stronger mean alignment
//! magnitude of 0.4218 as a side effect of its harsher, all-or-nothing
//! fitness. The mechanism this suggests: smoothing the landscape for
//! partial credit also *reduces selection pressure* -- mediocre-but-
//! tolerable candidates now score "good enough" to survive and
//! reproduce, crowding out the rare genuinely-good structural motifs
//! (M2.1: 0.022% reachability) that a harsher, more punishing fitness
//! culls more decisively toward. Partial credit was the intended fix for
//! M2.1's diagnosed problem; empirically, for this factorized-product
//! search, it made the problem worse, not better.
//!
//! Per the predeclared stopping rule, this closes M2 as fully diagnosed
//! rather than escalating to a larger seed count (escalation was
//! conditioned on the shaped arm showing meaningful improvement; it
//! showed meaningful *regression* instead). Summary of the whole M2 arc:
//! conventional tree evolution could not recover the local flux under
//! the tested budgets, and neither factorization (M2.2), grammar
//! restriction (M2.3), semantic deduplication (M2.4a), HDC-quantized
//! deduplication (M2.4a-corrected), nor target-blind fitness shaping
//! (M2.5) resolved the structural credit-assignment problem M2.1
//! diagnosed. A genuine fix would need to address structural
//! reachability itself (e.g. constrained/typed motif primitives), not
//! another fitness or search-mechanism variant.
//!
//! **M3 Phase 1 (2026-07-16): that specific hypothesis was tested directly, and it holds.**
//! `flux_discovery::random_motif_expr` -- a constructive generator biased toward
//! difference-of-two-variables/sum-of-two-variables/product-of-two-subexpressions motifs,
//! generic over *which* variable pair fills each slot (not answer-shaped seeding) -- was run
//! through the identical M2.1 reachability measurement (same `contains_pattern` checks, same
//! N=50,000). Result: the "contains both needed motifs" rate rose from M2.1's measured 0.022%
//! to **2.398%** (1199/50,000) -- a ~109x improvement, comfortably clearing the predeclared 1%
//! SUPPORTED threshold. Individual motif rates: displacement 1.394%->14.332%, velocity-sum
//! 1.304%->14.432%. This is a pure reachability measurement (no evolutionary search run), the
//! same cheap methodology M2.1 itself used -- see `flux_discovery.rs`'s
//! `motif_constrained_generator_reachability_estimate` test for the exact numbers and
//! `random_motif_expr`'s doc comment for the generator design (plus a real LCG-low-bits bug
//! found and fixed while calibrating it: this crate's `lcg_step` has a provably short-period
//! low bit cycle, and naively taking `% 4` of raw output against a 4-variable pool silently
//! collapsed reachable variable pairs from 6 to 2 -- fixed via `rand_index_high_bits`, which
//! draws from the state's high 32 bits instead).
//!
//! **M3 Phase 2 (2026-07-16): wired the motif generator into an actual evolutionary search --
//! reachability improvement does NOT translate into recovery at this budget. Honest negative.**
//! New `flux_discovery::discover_flux_given_density_seeded` (a parallel, generator-injectable
//! variant of `discover_flux_given_density`, via a new `breed_generation_seeded` rather than
//! modifying the established `breed_generation`) let motif-seeded and unrestricted-seeded
//! evolution run through an identical harness (population_size=200, generations=100, seed=42),
//! isolating the generator as the only variable. Result, single seed, this test's own
//! `motif_seeded_vs_unrestricted_seeded_evolutionary_comparison_n3`: **neither condition
//! cross-validated on held-out trajectories at all** -- the same non-recovery tier the
//! original 400x150 baseline test itself anticipated as a real possibility. Motif-seeded did
//! show a directionally lower best training residual (0.538 vs baseline's 0.713, ~1.3x) but
//! nowhere near the predeclared 10x SUPPORTED threshold. **Interpretation**: Phase 1's ~109x
//! reachability improvement measures whether the target *shape* is constructible at all;
//! Phase 2 shows that alone isn't sufficient for evolution to actually find and refine it at
//! this budget -- the credit-assignment problem M2.1 originally diagnosed (partial structures
//! aren't rewarded until the whole product assembles) may still bind even with a
//! motif-friendly generator, or a larger budget than this bounded single-seed test used might
//! close the gap. Neither investigated further here, per this arc's scope discipline (a single
//! frozen comparison, not open-ended escalation chasing a positive result).
//!
//! **M3 Phase 3 (2026-07-17): attempted the originally-scoped blind joint (ρ,J) co-evolution --
//! no recovery. Honest negative, closing M3's most direct investigation for now.** Unlike every
//! prior M2/M3 experiment, `rho` was NOT handed to the search here either -- both `rho` and `J`
//! were discovered together via a new `flux_discovery::discover_joint_density_and_flux`, a
//! two-population co-evolution modeled on `discover_flux_factorized`'s "pair against the other
//! side's current best" mechanism. This needed real new design work, not a parameter tweak:
//! with both sides free, the trivial pair `rho=c1, J=c2` (any constants) satisfies the discrete
//! continuity equation *exactly* (`0=0`) -- a free, perfect-residual escape hatch the ρ-fixed
//! setting never had -- so a new hard nontriviality guard (`is_nontrivial`, variance-based,
//! threshold matching `continuity.rs`'s own `MIN_SHAPE_VARIANCE`) was mathematically required
//! on every candidate, and the co-evolution's initial "best" pairing had to be seeded from the
//! populations' own first draws rather than `discover_flux_factorized`'s `Const(1.0)/Const(1.0)`
//! (degenerate here). Also found: `rho_truth()` is a *sum of squared* terms (kinetic + two bond
//! terms) -- a shape neither `random_motif_expr` nor any other existing generator can reach by
//! chance (`Mul(X,X)` needs two independently-drawn subtrees to coincide) -- so a new
//! `random_density_motif_expr` (dedicated `Square`-of-motif leaf, `Add`-weighted recursion,
//! mirroring `typed_generation.rs`'s `Combinator::Square` design) was added rather than
//! retrofitting the historical `random_motif_expr` (Phase 1/2's results stay untouched).
//!
//! Real result (single attempt, seed 42, `population_size=300, generations=150` -- a harder
//! 2-unknown problem given more budget than Phase 2's single-unknown case, still one bounded
//! seed): **no candidate pair cross-validated on held-out trajectories at all.** Only one
//! candidate survived to reporting (train_residual 0.331, `rho ≈ -0.5·u_l`, far from
//! `rho_truth()`'s real shape) -- markedly sparser than Phase 2's 5-10 surviving candidates per
//! condition, consistent with the nontriviality guard (now required on *both* sides
//! simultaneously) making this landscape genuinely harder to navigate, not just a bigger search
//! space. **This closes M3's three-phase investigation with a complete, honest picture**:
//! Phase 1 showed the target *shape* is constructible (~109x reachability gain); Phase 2 showed
//! that alone doesn't yield evolutionary recovery even in the easier single-unknown case; Phase
//! 3 shows the harder, originally-scoped joint problem doesn't succeed either at a bounded
//! budget. None of this proves joint discovery is impossible -- a larger budget or a refined
//! grammar remain open, explicitly unstarted, options -- but M3 as scoped has now been given a
//! genuine, good-faith attempt at every level it was ever framed at.
//!
//! ## FPU-alpha chain: an independently re-derived conserved current for a fresh problem
//! (2026-07-17)
//!
//! With M3's wave-chain investigation complete, this extends the same M0/M1 verification
//! discipline to a genuinely different physical system: the Fermi-Pasta-Ulam-Tsingou alpha
//! chain (FPU-alpha) -- same 1D nearest-neighbor topology, with a cubic nonlinear term added to
//! the inter-particle potential. **This is an independently re-derived result for this
//! codebase, not a scientific discovery** -- the FPU-alpha energy current is established
//! physics (the system is historically significant in its own right as the 1955 FPU recurrence
//! paradox).
//!
//! Hamiltonian: `H = Σᵢ 0.5·vᵢ² + Σ_bonds [0.5·r² + (α/3)·r³]`, `r = u_{i+1}-uᵢ`. EOM
//! (`v̇ᵢ = -∂H/∂uᵢ`, same fixed boundaries as the wave-chain): `v̇ᵢ = (u_{i+1}-2uᵢ+u_{i-1}) +
//! α·[(u_{i+1}-uᵢ)² - (uᵢ-u_{i-1})²]` (`fpu_rhs_n3_with_alpha`, `pde_wave_stage_a.rs`). Density
//! (bond energy split equally between endpoints, the *exact same method* used for `rho_truth()`
//! above): `ρᵢ = 0.5·vᵢ² + 0.25·(uᵢ-u_{i-1})² + 0.25·(u_{i+1}-uᵢ)² + (α/6)·(uᵢ-u_{i-1})³ +
//! (α/6)·(u_{i+1}-uᵢ)³` (`rho_fpu_truth`). Flux (differentiate `ρᵢ` along the flow, regroup by
//! bond): `J_{i+1/2} = 0.5·(u_{i+1}-uᵢ)·(vᵢ+v_{i+1}) + 0.5·α·(u_{i+1}-uᵢ)²·(vᵢ+v_{i+1})`
//! (`j_fpu_truth`). At `α=0` both reduce exactly to `rho_truth()`/`j_truth()` -- a **tested**
//! regression (`fpu_reduces_to_wave_chain_at_alpha_zero`), not just an algebraic observation.
//!
//! **Verification discipline, deliberately more thorough than a single continuity-residual
//! check** (this arc's own standing rule: algebra alone, even independently re-derived twice --
//! once here, once in an external review that also caught real design gaps -- is not sufficient
//! trust): (1) `fpu_local_flow_with_alpha` and `fpu_rhs_n3_with_alpha` are independently-written
//! functions that must agree on `v̇_c`, tested directly rather than assumed; (2) the `α=0`
//! reduction is tested as an actual equality, not just claimed; (3) the negative control omits
//! the nonlinear flux term entirely (a realistic "missing physics" bug) rather than merely
//! sign-flipping it, and the assertion is a residual *ratio*, not an absolute threshold; (4) an
//! independent Hamiltonian-drift check (unrelated to the continuity algebra) confirms the new
//! RHS and the existing RK4 integrator actually behave physically together. `α=0.05` is frozen
//! from a closed-form safety argument (see `FPU_ALPHA`'s doc comment in `pde_wave_stage_a.rs`
//! for the full derivation) *before* any implementation or test run, not tuned empirically
//! after the fact -- the FPU-alpha cubic potential is not globally bounded below, so this
//! matters. `symthaea-core` stays physics-agnostic throughout:
//! `discrete_continuity_residual_with_flow` (`continuity.rs`) takes an injectable `local_flow`
//! closure; all FPU-specific code (including `α` itself) lives here in
//! `symthaea-physics-bridge`.
//!
//! **Scope**: this is M0/M1 only -- establish and verify the ground truth. No discovery search
//! is attempted in this pass; a motif-seeded search for `J` given `ρ=rho_fpu_truth(FPU_ALPHA)`
//! fixed (the natural next, "M2-equivalent" step) is a flagged, explicitly not-yet-started
//! follow-up.
//!
//! ## The physics
//!
//! For free grid points `i = 1..n` (boundaries `u0 = u_{n+1} = 0`,
//! `v0 = v_{n+1} = 0`), splitting each bond's energy equally between its two
//! endpoints gives a natural local density:
//!
//! ```text
//! ρᵢ = 1/2·vᵢ² + 1/4·(uᵢ − u_{i−1})² + 1/4·(u_{i+1} − uᵢ)²
//! ```
//!
//! Differentiating along the wave-chain flow (`dvᵢ/dt = u_{i+1} − 2uᵢ + u_{i−1}`)
//! and regrouping by bond gives an exact discrete continuity equation with
//! flux
//!
//! ```text
//! J_{i+1/2} = 1/2·(u_{i+1} − uᵢ)·(vᵢ + v_{i+1})
//! ```
//!
//! i.e. `dρᵢ/dt = J_{i+1/2} − J_{i−1/2}` exactly. Both depend only on a
//! *local stencil* -- the same functional form at every `i` (translation
//! invariance), which is what makes this a field-theoretic statement rather
//! than `n` separate point-particle facts. Derived by hand; verified
//! numerically below (that's the entire point of Milestone 0).

use symthaea_core::hdc::conjecture_engine::{BinOp, Expr};

/// Hand-derived local energy density, over the point-stencil names
/// `u_l, u_c, u_r, v_c` (see `discrete_continuity_residual`'s module docs).
pub fn rho_truth() -> Expr {
    let var = |n: &str| Expr::Var(n.to_string());
    let pow2 = |n: &str| Expr::BinOp(BinOp::Pow, Box::new(var(n)), Box::new(Expr::Const(2.0)));
    let diff = |a: &str, b: &str| Expr::BinOp(BinOp::Sub, Box::new(var(a)), Box::new(var(b)));
    let sq = |e: Expr| Expr::BinOp(BinOp::Pow, Box::new(e), Box::new(Expr::Const(2.0)));
    let add = |a: Expr, b: Expr| Expr::BinOp(BinOp::Add, Box::new(a), Box::new(b));
    let scale = |c: f64, e: Expr| Expr::BinOp(BinOp::Mul, Box::new(Expr::Const(c)), Box::new(e));

    let kinetic = scale(0.5, pow2("v_c"));
    let left_bond = scale(0.25, sq(diff("u_c", "u_l")));
    let right_bond = scale(0.25, sq(diff("u_r", "u_c")));
    add(add(kinetic, left_bond), right_bond)
}

/// Hand-derived flux, over the bond-stencil names `u_c, u_r, v_c, v_r`
/// (left/right endpoints of one bond).
pub fn j_truth() -> Expr {
    let var = |n: &str| Expr::Var(n.to_string());
    let diff = |a: &str, b: &str| Expr::BinOp(BinOp::Sub, Box::new(var(a)), Box::new(var(b)));
    let add = |a: Expr, b: Expr| Expr::BinOp(BinOp::Add, Box::new(a), Box::new(b));
    let mul = |a: Expr, b: Expr| Expr::BinOp(BinOp::Mul, Box::new(a), Box::new(b));
    let scale = |c: f64, e: Expr| Expr::BinOp(BinOp::Mul, Box::new(Expr::Const(c)), Box::new(e));

    let stretch = diff("u_r", "u_c");
    let vel_sum = add(var("v_c"), var("v_r"));
    scale(0.5, mul(stretch, vel_sum))
}

/// A deliberately wrong flux (sign-flipped) -- the negative control. Must
/// fail [`discrete_continuity_residual`] with a large residual, or the
/// checker can't be trusted to actually be checking anything.
pub fn j_wrong_sign() -> Expr {
    Expr::BinOp(BinOp::Mul, Box::new(Expr::Const(-1.0)), Box::new(j_truth()))
}

// ---------------------------------------------------------------------
// FPU-alpha chain extension (2026-07-17) -- see this module's doc comment
// for the full derivation and verification discipline. `symthaea-core`
// stays physics-agnostic: all FPU-specific code, including `alpha` itself,
// lives here rather than in `continuity.rs`.
// ---------------------------------------------------------------------

/// FPU-alpha local flow for [`discrete_continuity_residual_with_flow`],
/// parameterized by `alpha` (a closure capturing a specific value is passed
/// at the call site, so `symthaea-core` never needs to know about FPU or
/// `alpha`). Mirrors the wave-chain's own `wave_chain_local_flow` shape and
/// its "two unused slots" caveat exactly -- must independently agree with
/// [`crate::pde_wave_stage_a::fpu_rhs_n3_with_alpha`]'s `v̇_c` term, tested
/// directly rather than assumed (see the `fpu_local_flow_agrees_with_rhs`
/// test).
pub fn fpu_local_flow_with_alpha(
    alpha: f64,
    u_l: f64,
    u_c: f64,
    u_r: f64,
    v_l: f64,
    v_c: f64,
    v_r: f64,
) -> [f64; 6] {
    let r_left = u_c - u_l;
    let r_right = u_r - u_c;
    [
        v_l,
        v_c,
        v_r,
        0.0,
        (u_r - 2.0 * u_c + u_l) + alpha * (r_right * r_right - r_left * r_left),
        0.0,
    ]
}

/// Hand-derived FPU-alpha local energy density, over the same point-stencil
/// names as [`rho_truth`] (`u_l, u_c, u_r, v_c`) -- a strict generalization:
/// `rho_fpu_truth(0.0)` evaluates identically to `rho_truth()` (tested, see
/// `fpu_reduces_to_wave_chain_at_alpha_zero`). Bond potential energy
/// `0.5*r² + (alpha/3)*r³` split equally between each bond's two endpoints,
/// the *exact same method* `rho_truth()`'s own doc comment describes.
pub fn rho_fpu_truth(alpha: f64) -> Expr {
    let var = |n: &str| Expr::Var(n.to_string());
    let pow2 = |n: &str| Expr::BinOp(BinOp::Pow, Box::new(var(n)), Box::new(Expr::Const(2.0)));
    let diff = |a: &str, b: &str| Expr::BinOp(BinOp::Sub, Box::new(var(a)), Box::new(var(b)));
    let sq = |e: Expr| Expr::BinOp(BinOp::Pow, Box::new(e), Box::new(Expr::Const(2.0)));
    let cube = |e: Expr| Expr::BinOp(BinOp::Pow, Box::new(e), Box::new(Expr::Const(3.0)));
    let add = |a: Expr, b: Expr| Expr::BinOp(BinOp::Add, Box::new(a), Box::new(b));
    let scale = |c: f64, e: Expr| Expr::BinOp(BinOp::Mul, Box::new(Expr::Const(c)), Box::new(e));

    let kinetic = scale(0.5, pow2("v_c"));
    let left_bond_sq = scale(0.25, sq(diff("u_c", "u_l")));
    let right_bond_sq = scale(0.25, sq(diff("u_r", "u_c")));
    let left_bond_cube = scale(alpha / 6.0, cube(diff("u_c", "u_l")));
    let right_bond_cube = scale(alpha / 6.0, cube(diff("u_r", "u_c")));
    add(
        add(add(kinetic, left_bond_sq), right_bond_sq),
        add(left_bond_cube, right_bond_cube),
    )
}

/// Hand-derived FPU-alpha flux, over the same bond-stencil names as
/// [`j_truth`] (`u_c, u_r, v_c, v_r`) -- derived by differentiating
/// [`rho_fpu_truth`] along the flow and regrouping by bond (verified
/// independently twice: once during planning, once in an external review).
/// At `alpha=0` reduces exactly to `j_truth()` (tested).
pub fn j_fpu_truth(alpha: f64) -> Expr {
    let var = |n: &str| Expr::Var(n.to_string());
    let diff = |a: &str, b: &str| Expr::BinOp(BinOp::Sub, Box::new(var(a)), Box::new(var(b)));
    let add = |a: Expr, b: Expr| Expr::BinOp(BinOp::Add, Box::new(a), Box::new(b));
    let mul = |a: Expr, b: Expr| Expr::BinOp(BinOp::Mul, Box::new(a), Box::new(b));
    let scale = |c: f64, e: Expr| Expr::BinOp(BinOp::Mul, Box::new(Expr::Const(c)), Box::new(e));
    let sq = |e: Expr| Expr::BinOp(BinOp::Pow, Box::new(e), Box::new(Expr::Const(2.0)));

    let stretch = diff("u_r", "u_c");
    let vel_sum = add(var("v_c"), var("v_r"));
    let linear = scale(0.5, mul(stretch.clone(), vel_sum.clone()));
    let nonlinear = scale(0.5 * alpha, mul(sq(stretch), vel_sum));
    add(linear, nonlinear)
}

/// The negative control: [`j_fpu_truth`] with the nonlinear term omitted
/// entirely (not sign-flipped) -- a more realistic "correct but missing
/// physics" failure mode than [`j_wrong_sign`]'s sign flip. Must fail
/// [`discrete_continuity_residual_with_flow`] with a residual far larger
/// than [`j_fpu_truth`]'s own (asserted as a ratio, not an absolute
/// threshold -- see `fpu_negative_control_by_omission`).
pub fn j_fpu_linear_only(alpha: f64) -> Expr {
    let _ = alpha; // kept for signature symmetry with j_fpu_truth; the omitted term is what's under test
    let var = |n: &str| Expr::Var(n.to_string());
    let diff = |a: &str, b: &str| Expr::BinOp(BinOp::Sub, Box::new(var(a)), Box::new(var(b)));
    let add = |a: Expr, b: Expr| Expr::BinOp(BinOp::Add, Box::new(a), Box::new(b));
    let mul = |a: Expr, b: Expr| Expr::BinOp(BinOp::Mul, Box::new(a), Box::new(b));
    let scale = |c: f64, e: Expr| Expr::BinOp(BinOp::Mul, Box::new(Expr::Const(c)), Box::new(e));

    let stretch = diff("u_r", "u_c");
    let vel_sum = add(var("v_c"), var("v_r"));
    scale(0.5, mul(stretch, vel_sum))
}

/// Total FPU-alpha Hamiltonian for an `n=3` state `[u1,u2,u3,v1,v2,v3]` --
/// `H = Σᵢ 0.5·vᵢ² + Σ_bonds [0.5·r² + (alpha/3)·r³]`. Used only for the
/// Hamiltonian-drift sanity test (`fpu_hamiltonian_drift_is_small`), an
/// independent check that the new RHS + the existing RK4 integrator behave
/// physically together -- unrelated to the continuity-equation algebra.
fn fpu_hamiltonian(state: &[f64], alpha: f64) -> f64 {
    let (u1, u2, u3, v1, v2, v3) = (state[0], state[1], state[2], state[3], state[4], state[5]);
    let kinetic = 0.5 * (v1 * v1 + v2 * v2 + v3 * v3);
    let bonds = [u1, u2 - u1, u3 - u2, -u3]; // r0..r3, boundaries u0=u4=0
    let potential: f64 = bonds
        .iter()
        .map(|r| 0.5 * r * r + (alpha / 3.0) * r * r * r)
        .sum();
    kinetic + potential
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pde_wave_stage_a::{
        FPU_ALPHA, fpu_rhs_n3, fpu_rhs_n3_with_alpha, fpu_trajectory_n3, wave_rhs_n3,
        wave_trajectory_n3, wave_trajectory_n4,
    };
    use symthaea_core::hdc::ContinuousHV;
    use symthaea_core::hdc::conjecture_engine::{
        DedupMode, FluxDiscoveryResult, GenerationSnapshot, RegressorConfig,
        discover_flux_factorized, discover_flux_factorized_shaped,
        discover_flux_factorized_with_dedup, discover_flux_factorized_with_snapshots,
        discover_flux_given_density, discover_flux_given_density_seeded,
        discover_joint_density_and_flux, discrete_continuity_residual,
        discrete_continuity_residual_with_flow, gauge_fix_flux, hdc_probe_basis,
        random_fpu_flux_motif_expr, random_motif_expr, shape_calibrated_residual,
    };

    #[test]
    fn continuity_holds_pointwise_on_a_real_trajectory_n3() {
        let traj = wave_trajectory_n3([1.0, -0.5, 0.3, 0.2, 0.3, -0.1], 200, 0.01);
        let residual = discrete_continuity_residual(&rho_truth(), &j_truth(), &traj, 3);
        assert!(
            residual.is_finite() && residual < 1e-6,
            "M0: hand-derived (rho, J) should satisfy the discrete continuity \
             equation pointwise on a real wave_rhs_n3 trajectory; residual={residual:e}"
        );
    }

    #[test]
    fn continuity_holds_pointwise_on_a_real_trajectory_n4() {
        let traj = wave_trajectory_n4([1.0, -0.5, 0.3, -0.2, 0.2, 0.3, -0.1, 0.4], 200, 0.01);
        let residual = discrete_continuity_residual(&rho_truth(), &j_truth(), &traj, 4);
        assert!(
            residual.is_finite() && residual < 1e-6,
            "M0: hand-derived (rho, J) should satisfy the discrete continuity \
             equation pointwise on a real wave_rhs_n4 trajectory; residual={residual:e}"
        );
    }

    #[test]
    fn continuity_holds_from_multiple_initial_conditions_n3() {
        for ic in [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.5, -0.5, 0.2],
            [2.0, -1.0, 0.5, 0.1, 0.1, -0.3],
        ] {
            let traj = wave_trajectory_n3(ic, 150, 0.01);
            let residual = discrete_continuity_residual(&rho_truth(), &j_truth(), &traj, 3);
            assert!(
                residual.is_finite() && residual < 1e-6,
                "IC {ic:?}: residual={residual:e}"
            );
        }
    }

    #[test]
    fn wrong_sign_flux_fails_the_checker_n3() {
        // Negative control: if this passed, discrete_continuity_residual
        // wouldn't actually be checking anything meaningful.
        let traj = wave_trajectory_n3([1.0, -0.5, 0.3, 0.2, 0.3, -0.1], 200, 0.01);
        let residual = discrete_continuity_residual(&rho_truth(), &j_wrong_sign(), &traj, 3);
        assert!(
            residual.is_finite() && residual > 0.1,
            "sign-flipped flux should badly fail continuity; residual={residual:e}"
        );
    }

    // -------------------------------------------------------------------
    // FPU-alpha chain: six independent checks, per the module doc's
    // verification discipline. Test 4 (fpu_continuity_holds...) is the
    // main gate; the rest catch failure modes a single continuity-residual
    // pass/fail could miss on its own.
    // -------------------------------------------------------------------

    /// Check 1: `fpu_local_flow_with_alpha` and `fpu_rhs_n3_with_alpha` are
    /// two independently-written functions that must agree on `v̇_c`, or the
    /// main continuity test below could pass while being self-consistently
    /// wrong (both sharing the same bug rather than checked against each
    /// other). Verified directly, at several hand-picked states, not
    /// inferred from the continuity residual alone.
    #[test]
    fn fpu_local_flow_agrees_with_rhs() {
        let states: [[f64; 6]; 4] = [
            [1.0, -0.5, 0.3, 0.2, 0.3, -0.1],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, -1.0, 0.5, 0.1, 0.1, -0.3],
            [-0.7, 0.4, -0.2, 0.05, -0.05, 0.2],
        ];
        for s in states {
            let rhs = fpu_rhs_n3_with_alpha(FPU_ALPHA, &s, 0.0);
            let (u1, u2, u3, v1, v2, v3) = (s[0], s[1], s[2], s[3], s[4], s[5]);
            // Center site (i=2): u_l=u1, u_c=u2, u_r=u3, v_l=v1, v_c=v2, v_r=v3.
            let flow = fpu_local_flow_with_alpha(FPU_ALPHA, u1, u2, u3, v1, v2, v3);
            assert!(
                (flow[4] - rhs[4]).abs() < 1e-12,
                "fpu_local_flow_with_alpha's v\u{0307}_c ({}) must match \
                 fpu_rhs_n3_with_alpha's v2 entry ({}) at state {s:?}",
                flow[4],
                rhs[4]
            );
        }
    }

    /// Check 2: `fpu_rhs_n3_with_alpha(0.0, ...)` must equal `wave_rhs_n3`
    /// exactly -- a tested fact, not an algebraic aside.
    #[test]
    fn fpu_rhs_reduces_to_wave_chain_at_alpha_zero() {
        let states: [[f64; 6]; 3] = [
            [1.0, -0.5, 0.3, 0.2, 0.3, -0.1],
            [2.0, -1.0, 0.5, 0.1, 0.1, -0.3],
            [-0.7, 0.4, -0.2, 0.05, -0.05, 0.2],
        ];
        for s in states {
            let fpu = fpu_rhs_n3_with_alpha(0.0, &s, 0.0);
            let wave = wave_rhs_n3(&s, 0.0);
            assert_eq!(
                fpu, wave,
                "alpha=0 FPU RHS must equal wave_rhs_n3 exactly at {s:?}"
            );
        }
    }

    /// Check 3: `rho_fpu_truth(0.0)`/`j_fpu_truth(0.0)` must evaluate
    /// identically to `rho_truth()`/`j_truth()` at several probe points --
    /// comparing evaluated values (not `Expr` tree structure, which can
    /// differ syntactically while being semantically identical).
    #[test]
    fn fpu_reduces_to_wave_chain_at_alpha_zero() {
        let rho_stencil = ["u_l", "u_c", "u_r", "v_c"];
        let bond_stencil = ["u_c", "u_r", "v_c", "v_r"];
        let rho_probes: [[f64; 4]; 3] = [
            [0.3, -0.7, 0.2, -0.4],
            [1.0, 1.0, 0.0, 0.0],
            [-0.5, 0.5, 0.9, -0.9],
        ];
        let j_probes: [[f64; 4]; 3] = [
            [0.3, -0.7, 0.2, -0.4],
            [1.0, 1.0, 0.0, 0.0],
            [-0.5, 0.5, 0.9, -0.9],
        ];

        let (rho_fpu0, rho_wave) = (rho_fpu_truth(0.0), rho_truth());
        for probe in rho_probes {
            let bindings: Vec<(&str, f64)> = rho_stencil.iter().copied().zip(probe).collect();
            let (a, b) = (rho_fpu0.eval(&bindings), rho_wave.eval(&bindings));
            assert!(
                (a - b).abs() < 1e-12,
                "rho_fpu_truth(0.0) vs rho_truth() at {probe:?}: {a} vs {b}"
            );
        }

        let (j_fpu0, j_wave) = (j_fpu_truth(0.0), j_truth());
        for probe in j_probes {
            let bindings: Vec<(&str, f64)> = bond_stencil.iter().copied().zip(probe).collect();
            let (a, b) = (j_fpu0.eval(&bindings), j_wave.eval(&bindings));
            assert!(
                (a - b).abs() < 1e-12,
                "j_fpu_truth(0.0) vs j_truth() at {probe:?}: {a} vs {b}"
            );
        }
    }

    /// Check 4 (the main gate): the hand-derived, independently re-derived
    /// FPU-alpha `(rho, J)` satisfies the discrete continuity equation
    /// pointwise on a real `fpu_trajectory_n3` trajectory, at the frozen
    /// `FPU_ALPHA`. If this fails, re-derive the algebra honestly -- do not
    /// tune the formula to make this pass.
    #[test]
    fn fpu_continuity_holds_pointwise_on_a_real_trajectory_n3() {
        let traj = fpu_trajectory_n3([1.0, -0.5, 0.3, 0.2, 0.3, -0.1], 200, 0.01);
        let residual = discrete_continuity_residual_with_flow(
            &rho_fpu_truth(FPU_ALPHA),
            &j_fpu_truth(FPU_ALPHA),
            &traj,
            3,
            |u_l, u_c, u_r, v_l, v_c, v_r| {
                fpu_local_flow_with_alpha(FPU_ALPHA, u_l, u_c, u_r, v_l, v_c, v_r)
            },
        );
        assert!(
            residual.is_finite() && residual < 1e-6,
            "FPU-alpha M0: hand-derived (rho, J) should satisfy the discrete continuity \
             equation pointwise on a real fpu_trajectory_n3 trajectory; residual={residual:e}"
        );
    }

    /// Check 5: negative control by *omission* (the nonlinear flux term
    /// entirely missing, not sign-flipped -- a more realistic "correct but
    /// missing physics" bug than `j_wrong_sign`'s sign flip). Assert the
    /// wrong residual exceeds the correct residual by a large *ratio*, not
    /// just an absolute threshold.
    #[test]
    fn fpu_negative_control_by_omission() {
        let traj = fpu_trajectory_n3([1.0, -0.5, 0.3, 0.2, 0.3, -0.1], 200, 0.01);
        let flow = |u_l: f64, u_c: f64, u_r: f64, v_l: f64, v_c: f64, v_r: f64| {
            fpu_local_flow_with_alpha(FPU_ALPHA, u_l, u_c, u_r, v_l, v_c, v_r)
        };
        let correct_residual = discrete_continuity_residual_with_flow(
            &rho_fpu_truth(FPU_ALPHA),
            &j_fpu_truth(FPU_ALPHA),
            &traj,
            3,
            flow,
        );
        let wrong_residual = discrete_continuity_residual_with_flow(
            &rho_fpu_truth(FPU_ALPHA),
            &j_fpu_linear_only(FPU_ALPHA),
            &traj,
            3,
            flow,
        );
        assert!(
            correct_residual.is_finite() && correct_residual < 1e-6,
            "sanity: correct FPU residual should itself be tiny; got {correct_residual:e}"
        );
        assert!(
            wrong_residual.is_finite() && wrong_residual > 1000.0 * correct_residual.max(1e-12),
            "omitting the nonlinear flux term should badly fail continuity relative to the \
             correct case; correct={correct_residual:e}, wrong={wrong_residual:e}, \
             ratio={:e}",
            wrong_residual / correct_residual.max(1e-12)
        );
    }

    /// Check 6: an independent sanity check, structurally unrelated to the
    /// continuity-equation algebra -- does the new RHS + the existing RK4
    /// integrator actually behave physically together? RK4 isn't
    /// symplectic, so *some* energy drift is expected over the sampled
    /// window, but it should stay small/bounded, not run away.
    #[test]
    fn fpu_hamiltonian_drift_is_small() {
        let traj = fpu_trajectory_n3([1.0, -0.5, 0.3, 0.2, 0.3, -0.1], 2000, 0.01);
        let h0 = fpu_hamiltonian(&traj[0], FPU_ALPHA);
        let h_end = fpu_hamiltonian(traj.last().unwrap(), FPU_ALPHA);
        let relative_drift = (h_end - h0).abs() / h0.abs();
        assert!(
            relative_drift < 0.01,
            "FPU Hamiltonian should stay approximately constant over a short RK4 trajectory \
             (some drift expected, RK4 isn't symplectic); h0={h0}, h_end={h_end}, \
             relative_drift={relative_drift:e}"
        );
    }

    /// M2.5 landscape audit -- run BEFORE any evolutionary search, per the
    /// frozen M2.5 design: does `shape_calibrated_residual` (target-blind
    /// fitness shaping, see `symthaea_core::hdc::conjecture_engine::
    /// shape_calibrated_residual`'s docs for the math) actually distinguish
    /// a frozen set of controlled candidates, or does it fail to create the
    /// partial-credit gradient M2.1 diagnosed as missing from the raw
    /// residual? If this test can't tell these near-misses apart, the
    /// evolutionary comparison must not be launched.
    ///
    /// All candidates are gauge-fixed before scoring, matching exactly how
    /// `discover_flux_factorized_shaped`'s `pair_score` treats every
    /// candidate during search -- an audit against un-gauge-fixed
    /// candidates wouldn't predict real search behavior.
    #[test]
    fn m25_landscape_audit_shape_calibrated_residual() {
        let var = |n: &str| Expr::Var(n.to_string());
        let add = |a: Expr, b: Expr| Expr::BinOp(BinOp::Add, Box::new(a), Box::new(b));
        let sub = |a: Expr, b: Expr| Expr::BinOp(BinOp::Sub, Box::new(a), Box::new(b));
        let mul = |a: Expr, b: Expr| Expr::BinOp(BinOp::Mul, Box::new(a), Box::new(b));
        let scale = |c: f64, e: Expr| mul(Expr::Const(c), e);

        // -- controlled candidate set (frozen for this audit) --
        let exact = j_truth();
        // truth with perturbed coefficient (0.5 -> 0.85, i.e. 1.7x)
        let perturbed_coefficient = scale(1.7, j_truth());
        // truth with wrong sign (the existing M0/M1 negative control)
        let wrong_sign = j_wrong_sign();
        // correct product with one factor missing entirely
        let stretch_only = sub(var("u_r"), var("u_c"));
        let vel_sum_only = add(var("v_c"), var("v_r"));
        // correct variables, but the "wrong" combination (sum where truth
        // has difference, difference where truth has sum)
        let wrong_cross_product = mul(add(var("u_c"), var("u_r")), sub(var("v_c"), var("v_r")));
        // a single bond-stencil variable alone
        let single_var = var("u_c");
        // constants: must be rejected outright (z(J) == 0 identically, no
        // shape information at all -- the negligible-variance guard)
        let constant_flux = Expr::Const(2.0);

        let candidates: [(&str, Expr); 8] = [
            ("exact", exact.clone()),
            ("perturbed_coefficient", perturbed_coefficient),
            ("wrong_sign", wrong_sign),
            ("stretch_only", stretch_only),
            ("vel_sum_only", vel_sum_only),
            ("wrong_cross_product", wrong_cross_product),
            ("single_var", single_var),
            ("constant_flux", constant_flux),
        ];

        let traj = wave_trajectory_n3([1.0, -0.5, 0.3, 0.2, 0.3, -0.1], 1000, 0.01);
        let rho = rho_truth();

        let mut results: Vec<(&str, Option<f64>, Option<f64>)> = Vec::new();
        for (label, cand) in &candidates {
            let fixed = gauge_fix_flux(cand);
            let cal = shape_calibrated_residual(&rho, &fixed, &traj, 3);
            let (residual, alignment) = match cal {
                Some(c) => (Some(c.calibrated_residual), Some(c.alignment)),
                None => (None, None),
            };
            println!("[{label}] calibrated_residual={residual:?}, alignment={alignment:?}");
            results.push((label, residual, alignment));
        }

        let get =
            |label: &str| -> Option<f64> { results.iter().find(|(l, ..)| *l == label).unwrap().1 };

        // The "perfect cluster": exact truth, and any nonzero scalar
        // multiple of it (positive *or* negative -- alpha* absorbs sign
        // freely, same math as the coefficient case). This is a real,
        // disclosed property of scale/sign-calibrated fitness, not an
        // oversight: it means the calibration genuinely cannot distinguish
        // "right shape, wrong sign" from "right shape, wrong amplitude" --
        // both are recoverable by a single scalar. Only a shape that's
        // NOT any real multiple of the truth produces a nonzero residual.
        for label in ["exact", "perturbed_coefficient", "wrong_sign"] {
            let r = get(label).unwrap_or_else(|| panic!("{label} must score, not be rejected"));
            assert!(
                r < 1e-9,
                "{label} is a scalar multiple of the true flux; calibration should \
                 recover it as perfect regardless of scale/sign, got {r:e}"
            );
        }

        // Degenerate candidate: must be rejected, not silently scored.
        assert!(
            get("constant_flux").is_none(),
            "a constant flux has z(J)=0 identically and must be rejected by the \
             negligible-variance guard, not scored"
        );

        // Structurally-incomplete-but-real candidates: must score
        // meaningfully worse than the perfect cluster (this is not a
        // trivial 0), but must still be a valid, non-degenerate score --
        // this IS the partial-credit gradient M2.1 found missing from the
        // raw residual (every near-miss there scored no better than doing
        // nothing at all).
        for label in [
            "stretch_only",
            "vel_sum_only",
            "wrong_cross_product",
            "single_var",
        ] {
            let r = get(label).unwrap_or_else(|| panic!("{label} must score, not be rejected"));
            assert!(
                r > 1e-6,
                "{label} is not a scalar multiple of the truth; should score \
                 measurably worse than the perfect cluster, got {r:e}"
            );
            assert!(
                r < 1.0 - 1e-6,
                "{label} should retain SOME shape correlation with the target \
                 (calibrated_residual < 1.0, i.e. nonzero |alignment|), got {r:e} \
                 -- if this fails, the audit has NOT demonstrated a partial-credit \
                 gradient and the evolutionary run must not proceed"
            );
        }

        println!(
            "M2.5 landscape audit PASSED: the calibrated residual distinguishes the \
             perfect cluster (exact/scale/sign-perturbed truth, residual<1e-9) from \
             structurally-incomplete-but-real candidates (residual in (1e-6, 1.0)) \
             from a degenerate constant (rejected outright). A real partial-credit \
             gradient exists -- proceeding to the evolutionary comparison is justified."
        );
    }

    #[test]
    fn mismatched_chain_length_returns_max() {
        // n doesn't match the trajectory's actual state width -- must fail
        // loudly (f64::MAX), not silently evaluate garbage.
        let traj = wave_trajectory_n3([1.0, -0.5, 0.3, 0.2, 0.3, -0.1], 50, 0.01);
        let residual = discrete_continuity_residual(&rho_truth(), &j_truth(), &traj, 4);
        assert_eq!(residual, f64::MAX);
    }

    #[test]
    fn empty_trajectory_returns_max() {
        let residual = discrete_continuity_residual(&rho_truth(), &j_truth(), &[], 3);
        assert_eq!(residual, f64::MAX);
    }

    // -------------------------------------------------------------
    // M2 canonicalization: gauge-fixing. See continuity.rs's
    // gauge_fix_flux docs for why this is needed before any GP search
    // over J -- without it, J_truth + c scores identically to J_truth
    // for any constant c, so the search is underdetermined.
    // -------------------------------------------------------------

    #[test]
    fn gauge_fixing_is_a_noop_on_an_already_zeroed_flux() {
        // j_truth() already vanishes at the all-zero stencil state
        // (algebraically 0.5*(0-0)*(0+0)=0), so gauge-fixing it should be
        // a true no-op: same value everywhere it's evaluated.
        let fixed = gauge_fix_flux(&j_truth());
        let bond_names = ["u_c", "u_r", "v_c", "v_r"];
        for probe in [
            [0.3, -0.7, 0.2, -0.4],
            [1.0, 1.0, 0.0, 0.0],
            [-0.5, 0.5, 0.9, -0.9],
        ] {
            let bindings: Vec<(&str, f64)> = bond_names.iter().copied().zip(probe).collect();
            let truth_val = j_truth().eval(&bindings);
            let fixed_val = fixed.eval(&bindings);
            assert!(
                (truth_val - fixed_val).abs() < 1e-12,
                "probe {probe:?}: truth={truth_val}, gauge-fixed={fixed_val}"
            );
        }
    }

    #[test]
    fn gauge_fixing_removes_a_constant_shift() {
        // The actual point of gauge-fixing: J_truth + c must gauge-fix
        // back to (numerically) J_truth, recovering the same physical
        // answer despite the shift.
        let shifted = Expr::BinOp(BinOp::Add, Box::new(j_truth()), Box::new(Expr::Const(7.3)));
        let fixed = gauge_fix_flux(&shifted);
        let bond_names = ["u_c", "u_r", "v_c", "v_r"];
        for probe in [[0.3, -0.7, 0.2, -0.4], [1.0, 1.0, 0.0, 0.0]] {
            let bindings: Vec<(&str, f64)> = bond_names.iter().copied().zip(probe).collect();
            let truth_val = j_truth().eval(&bindings);
            let fixed_val = fixed.eval(&bindings);
            assert!(
                (truth_val - fixed_val).abs() < 1e-12,
                "probe {probe:?}: truth={truth_val}, gauge-fixed(shifted)={fixed_val}"
            );
        }
    }

    #[test]
    fn gauge_fixed_wrong_sign_flux_still_fails_the_checker() {
        // Sanity: gauge-fixing doesn't accidentally "fix" a genuinely
        // wrong flux into passing -- it only removes the constant-shift
        // degeneracy, not real physical wrongness.
        let traj = wave_trajectory_n3([1.0, -0.5, 0.3, 0.2, 0.3, -0.1], 200, 0.01);
        let fixed_wrong = gauge_fix_flux(&j_wrong_sign());
        let residual = discrete_continuity_residual(&rho_truth(), &fixed_wrong, &traj, 3);
        assert!(
            residual.is_finite() && residual > 0.1,
            "gauge-fixed sign-flipped flux should still badly fail; residual={residual:e}"
        );
    }

    #[test]
    #[ignore] // GP search: slow, run explicitly.
    fn discover_flux_given_known_density_n3() {
        // M2: rho is FIXED (the M0/M1-verified truth) -- only J is
        // unknown. Deliberately no seed template shaped like j_truth();
        // the search must find it (or a gauge-equivalent form) from
        // generic random expressions + mutation/crossover, matching the
        // Stage A N=3-unseeded precedent for what counts as a genuine
        // discovery test rather than a re-confirmation.
        //
        // Honest expectation, stated before running: this may not
        // succeed at this budget -- same risk class as Stage A's
        // N=3-unseeded result. Report pass or fail either way, do not
        // force a result.
        let config = RegressorConfig {
            population_size: 400,
            generations: 150,
            max_depth: 4,
            max_complexity: 24,
            seed: 42,
            ..RegressorConfig::for_autonomous_discovery()
        };
        let train_ic = [1.0, -0.5, 0.3, 0.2, 0.3, -0.1];
        let holdout_ics = [
            [0.4, 1.1, -0.6, 0.5, -0.3, 0.2],
            [-0.8, 0.2, 0.9, -0.1, 0.4, -0.5],
        ];

        let candidates = discover_flux_given_density(
            &rho_truth(),
            wave_rhs_n3,
            &train_ic,
            3,
            &config,
            10.0,
            0.01,
        );
        assert!(
            !candidates.is_empty(),
            "search should produce at least one finite-scoring candidate"
        );

        let bond_names = ["u_c", "u_r", "v_c", "v_r"];
        let probe_points: [[f64; 4]; 4] = [
            [0.3, -0.7, 0.2, -0.4],
            [1.0, 1.0, 0.0, 0.0],
            [-0.5, 0.5, 0.9, -0.9],
            [0.6, -0.2, -0.3, 0.7],
        ];

        let mut cross_validated = None;
        for cand in &candidates {
            let holdout_residuals: Vec<f64> = holdout_ics
                .iter()
                .map(|ic| {
                    let traj = wave_trajectory_n3(*ic, 1000, 0.01);
                    discrete_continuity_residual(&rho_truth(), &cand.formula, &traj, 3)
                })
                .collect();
            let holdout_ok = holdout_residuals.iter().all(|r| r.is_finite() && *r < 1e-4);

            let max_probe_diff = probe_points
                .iter()
                .map(|probe| {
                    let bindings: Vec<(&str, f64)> =
                        bond_names.iter().copied().zip(*probe).collect();
                    (j_truth().eval(&bindings) - cand.formula.eval(&bindings)).abs()
                })
                .fold(0.0_f64, f64::max);

            println!(
                "candidate: {} (train_residual={:.3e}, holdout_ok={}, max_probe_diff_vs_truth={:.3e})",
                cand.formula_str, cand.residual, holdout_ok, max_probe_diff
            );

            if holdout_ok && cross_validated.is_none() {
                cross_validated = Some((cand, max_probe_diff));
            }
        }

        match cross_validated {
            Some((cand, diff)) if diff < 1e-6 => println!(
                "RESULT: flux recovered and symbolically close to j_truth() (max probe diff \
                 {diff:e}): {} -- genuine discovery, matching the known answer's equivalence \
                 class.",
                cand.formula_str
            ),
            Some((cand, diff)) => println!(
                "RESULT: found a candidate that cross-validates on held-out trajectories but \
                 differs numerically from j_truth() at probe points (max diff {diff:e}): {} \
                 -- either a genuinely different (still-valid, gauge-inequivalent-looking) \
                 flux, or a subtler match that the probe-point check doesn't recognize as \
                 equivalent. Worth manual inspection before calling this confirmed.",
                cand.formula_str
            ),
            None => println!(
                "RESULT: no candidate cross-validated against held-out trajectories at this \
                 budget. Honest finding, same risk class as Stage A's N=3-unseeded result --\
                 not evidence of a bug in rho_truth(), discrete_continuity_residual, or \
                 gauge_fix_flux (all independently verified in M0/M1). A larger budget, a \
                 deeper max_depth, or accepting this as M2's honest scaling/difficulty limit \
                 are the real next options."
            ),
        }
    }

    /// M3 Phase 2 (see the module-level doc's M3 note): does seeding/mutating the GP population
    /// with `random_motif_expr` instead of the unrestricted `fresh_random_bond_expr` actually
    /// let evolution *recover* the flux, at a budget where the M2 arc's six unrestricted-
    /// generator sub-experiments all failed? Phase 1 showed motif-constrained generation raises
    /// raw reachability ~109x; this tests whether that translates into real discovery success.
    /// Isolates the generator as the only variable: both conditions run through the identical
    /// `discover_flux_given_density_seeded` harness, identical config/trajectories/seed --
    /// reusing `discover_flux_given_known_density_n3`'s exact recovery methodology (held-out
    /// cross-validation, then probe-point comparison against `j_truth()`) for both.
    ///
    /// Predeclared threshold, frozen before running: motif is SUPPORTED if its best comparable
    /// residual (best held-out-cross-validated residual, or best training residual if neither
    /// condition cross-validates at all) is more than 10x lower than baseline's. Single seed
    /// per condition (matches this budget class's established precedent -- a full 200x100 GP
    /// run is too expensive to casually multiply by seed count); an ambiguous result is a
    /// legitimate, reportable outcome on its own, not grounds to keep re-seeding.
    #[test]
    #[ignore] // GP search: slow, run explicitly.
    fn motif_seeded_vs_unrestricted_seeded_evolutionary_comparison_n3() {
        let config = RegressorConfig {
            population_size: 200,
            generations: 100,
            max_depth: 3,
            max_complexity: 24,
            seed: 42,
            ..RegressorConfig::for_autonomous_discovery()
        };
        let train_ic = [1.0, -0.5, 0.3, 0.2, 0.3, -0.1];
        let holdout_ics = [
            [0.4, 1.1, -0.6, 0.5, -0.3, 0.2],
            [-0.8, 0.2, 0.9, -0.1, 0.4, -0.5],
        ];
        let bond_names = ["u_c", "u_r", "v_c", "v_r"];
        let probe_points: [[f64; 4]; 4] = [
            [0.3, -0.7, 0.2, -0.4],
            [1.0, 1.0, 0.0, 0.0],
            [-0.5, 0.5, 0.9, -0.9],
            [0.6, -0.2, -0.3, 0.7],
        ];

        // (label, best_comparable_residual, tier_description)
        let run_condition = |label: &str, candidates: Vec<FluxDiscoveryResult>| -> (String, f64) {
            let mut best_cross_validated: Option<(f64, f64)> = None; // (residual, max_probe_diff)
            let mut best_training_residual = f64::MAX;
            for cand in &candidates {
                best_training_residual = best_training_residual.min(cand.residual);
                let holdout_residuals: Vec<f64> = holdout_ics
                    .iter()
                    .map(|ic| {
                        let traj = wave_trajectory_n3(*ic, 1000, 0.01);
                        discrete_continuity_residual(&rho_truth(), &cand.formula, &traj, 3)
                    })
                    .collect();
                let holdout_ok = holdout_residuals.iter().all(|r| r.is_finite() && *r < 1e-4);
                if !holdout_ok {
                    continue;
                }
                let max_probe_diff = probe_points
                    .iter()
                    .map(|probe| {
                        let bindings: Vec<(&str, f64)> =
                            bond_names.iter().copied().zip(*probe).collect();
                        (j_truth().eval(&bindings) - cand.formula.eval(&bindings)).abs()
                    })
                    .fold(0.0_f64, f64::max);
                if best_cross_validated.is_none_or(|(r, _)| cand.residual < r) {
                    best_cross_validated = Some((cand.residual, max_probe_diff));
                }
            }

            let (tier, comparable_residual) = match best_cross_validated {
                Some((r, diff)) if diff < 1e-6 => (
                    format!(
                        "GENUINE RECOVERY (cross-validated, max_probe_diff={diff:e} vs j_truth())"
                    ),
                    r,
                ),
                Some((r, diff)) => (
                    format!("cross-validates but differs from j_truth() (max_probe_diff={diff:e})"),
                    r,
                ),
                None => (
                    "no candidate cross-validated on held-out trajectories".to_string(),
                    best_training_residual,
                ),
            };
            println!(
                "[{label}] {} candidates, {tier}, best_comparable_residual={comparable_residual:e}",
                candidates.len()
            );
            (tier, comparable_residual)
        };

        // Baseline: the unchanged `discover_flux_given_density` wrapper (unrestricted
        // `fresh_random_bond_expr` generator) -- the exact same call
        // `discover_flux_given_known_density_n3` makes, just at this test's smaller
        // 200x100 budget instead of that test's 400x150.
        let baseline_candidates = discover_flux_given_density(
            &rho_truth(),
            wave_rhs_n3,
            &train_ic,
            3,
            &config,
            10.0,
            0.01,
        );
        let (_baseline_tier, baseline_residual) =
            run_condition("baseline (unrestricted)", baseline_candidates);

        let motif_candidates = discover_flux_given_density_seeded(
            &rho_truth(),
            wave_rhs_n3,
            &train_ic,
            3,
            &config,
            10.0,
            0.01,
            |rng| random_motif_expr(rng, &bond_names, 3),
        );
        let (_motif_tier, motif_residual) = run_condition("motif-seeded", motif_candidates);

        let verdict = if motif_residual < baseline_residual / 10.0 {
            "SUPPORTED"
        } else {
            "NEGATIVE"
        };
        println!(
            "M3 Phase 2 verdict: baseline_best={baseline_residual:e}, motif_best={motif_residual:e}, \
             predeclared threshold: motif < baseline/10 -> {verdict}"
        );
    }

    /// FPU-alpha flux discovery: the real GP evolutionary search for `J` given
    /// `rho=rho_fpu_truth(FPU_ALPHA)` fixed -- the actual "M2-equivalent" step for FPU, run after
    /// the square-aware generator (`random_fpu_flux_motif_expr`) fixed structural reachability
    /// (0.028% -> 1.588% co-occurrence, SUPPORTED). Directly mirrors
    /// [`motif_seeded_vs_unrestricted_seeded_evolutionary_comparison_n3`]'s own design: a frozen,
    /// predeclared, single-seed, two-condition comparison (baseline unrestricted generator vs.
    /// motif-seeded), same harness, same held-out cross-validation (`residual < 1e-4`) +
    /// probe-point-vs-ground-truth (`max_probe_diff < 1e-6` = genuine recovery) methodology, same
    /// predeclared verdict shape (motif SUPPORTED if its best comparable residual is more than
    /// 10x lower than baseline's).
    ///
    /// **One deliberate deviation from Phase 2's exact config**: `max_complexity` raised from 24
    /// to 48. `j_fpu_truth(FPU_ALPHA)`'s own node-count (via `Expr::complexity()`) is 21 -- a cap
    /// of 24 would leave only 3 nodes of margin over the *exact minimal* target shape, repeating
    /// the same mistake class the Stage A "recall gap" fix already corrected once in this
    /// codebase's history (`max_complexity: 16` silently killed an already-correct seed of
    /// complexity 21). 48 matches Stage A's own N=3 case's choice, giving >2x headroom.
    /// `population_size`/`generations`/`max_depth`/`seed` all stay frozen at Phase 2's exact
    /// values for direct comparability. `train_ic`/`holdout_ics`/`probe_points` are reused
    /// verbatim from Phase 2 -- their bond displacements (max |r|: 1.5, 1.7, 1.0 respectively)
    /// all sit comfortably below the `1/FPU_ALPHA=20` potential-barrier margin `FPU_ALPHA` was
    /// originally frozen against (>11x margin even for the least-safe IC), so no new safety
    /// argument is needed.
    #[test]
    #[ignore] // GP search: slow, run explicitly.
    fn fpu_motif_seeded_vs_unrestricted_evolutionary_comparison_n3() {
        let config = RegressorConfig {
            population_size: 200,
            generations: 100,
            max_depth: 3,
            max_complexity: 48,
            seed: 42,
            ..RegressorConfig::for_autonomous_discovery()
        };
        let train_ic = [1.0, -0.5, 0.3, 0.2, 0.3, -0.1];
        let holdout_ics = [
            [0.4, 1.1, -0.6, 0.5, -0.3, 0.2],
            [-0.8, 0.2, 0.9, -0.1, 0.4, -0.5],
        ];
        let bond_names = ["u_c", "u_r", "v_c", "v_r"];
        let probe_points: [[f64; 4]; 4] = [
            [0.3, -0.7, 0.2, -0.4],
            [1.0, 1.0, 0.0, 0.0],
            [-0.5, 0.5, 0.9, -0.9],
            [0.6, -0.2, -0.3, 0.7],
        ];

        let rho = rho_fpu_truth(FPU_ALPHA);
        let j_truth_fpu = j_fpu_truth(FPU_ALPHA);

        // (label, best_comparable_residual, tier_description)
        let run_condition = |label: &str, candidates: Vec<FluxDiscoveryResult>| -> (String, f64) {
            let mut best_cross_validated: Option<(f64, f64)> = None; // (residual, max_probe_diff)
            let mut best_training_residual = f64::MAX;
            for cand in &candidates {
                best_training_residual = best_training_residual.min(cand.residual);
                let holdout_residuals: Vec<f64> = holdout_ics
                    .iter()
                    .map(|ic| {
                        let traj = fpu_trajectory_n3(*ic, 1000, 0.01);
                        discrete_continuity_residual(&rho, &cand.formula, &traj, 3)
                    })
                    .collect();
                let holdout_ok = holdout_residuals.iter().all(|r| r.is_finite() && *r < 1e-4);
                if !holdout_ok {
                    continue;
                }
                let max_probe_diff = probe_points
                    .iter()
                    .map(|probe| {
                        let bindings: Vec<(&str, f64)> =
                            bond_names.iter().copied().zip(*probe).collect();
                        (j_truth_fpu.eval(&bindings) - cand.formula.eval(&bindings)).abs()
                    })
                    .fold(0.0_f64, f64::max);
                if best_cross_validated.is_none_or(|(r, _)| cand.residual < r) {
                    best_cross_validated = Some((cand.residual, max_probe_diff));
                }
            }

            let (tier, comparable_residual) = match best_cross_validated {
                Some((r, diff)) if diff < 1e-6 => (
                    format!(
                        "GENUINE RECOVERY (cross-validated, max_probe_diff={diff:e} vs j_fpu_truth())"
                    ),
                    r,
                ),
                Some((r, diff)) => (
                    format!(
                        "cross-validates but differs from j_fpu_truth() (max_probe_diff={diff:e})"
                    ),
                    r,
                ),
                None => (
                    "no candidate cross-validated on held-out trajectories".to_string(),
                    best_training_residual,
                ),
            };
            println!(
                "[{label}] {} candidates, {tier}, best_comparable_residual={comparable_residual:e}",
                candidates.len()
            );
            (tier, comparable_residual)
        };

        // Baseline: the unchanged `discover_flux_given_density` wrapper (unrestricted
        // `fresh_random_bond_expr` generator), same call shape as Phase 2's baseline, just with
        // FPU physics (rho, rhs) substituted in.
        let baseline_candidates =
            discover_flux_given_density(&rho, fpu_rhs_n3, &train_ic, 3, &config, 10.0, 0.01);
        let (_baseline_tier, baseline_residual) =
            run_condition("baseline (unrestricted)", baseline_candidates);

        let motif_candidates = discover_flux_given_density_seeded(
            &rho,
            fpu_rhs_n3,
            &train_ic,
            3,
            &config,
            10.0,
            0.01,
            |rng| random_fpu_flux_motif_expr(rng, &bond_names, 3),
        );
        let (_motif_tier, motif_residual) =
            run_condition("FPU square-aware motif-seeded", motif_candidates);

        let verdict = if motif_residual < baseline_residual / 10.0 {
            "SUPPORTED"
        } else {
            "NEGATIVE"
        };
        println!(
            "FPU flux GP search verdict: baseline_best={baseline_residual:e}, \
             motif_best={motif_residual:e}, predeclared threshold: motif < baseline/10 -> {verdict}"
        );
    }

    /// M3 Phase 3 (see the module-level doc's M3 note): blind joint discovery of BOTH `rho` and
    /// `J` together -- neither is handed to the search, unlike every M2 sub-experiment and M3
    /// Phase 1/2. This is the problem M3 was originally scoped for. Not a paired ablation like
    /// Phase 1/2 -- a single discovery attempt, matching Stage A's N=3-unseeded precedent and
    /// M2's own baseline test's honesty-over-threshold reporting convention (genuine recovery /
    /// different-but-valid / no-recovery), applied to both `rho` and `J` jointly. See
    /// `discover_joint_density_and_flux`'s doc comment for the nontriviality-guard rationale
    /// (mathematically required, not optional -- the unconstrained `rho=c1,J=c2` pair satisfies
    /// continuity exactly) and the explicitly-acknowledged gauge-freedom limitation (only
    /// `J`'s constant-shift gauge is fixed; the larger joint gauge group is not).
    #[test]
    #[ignore] // GP search: slow, run explicitly.
    fn blind_joint_density_and_flux_discovery_n3() {
        let config = RegressorConfig {
            population_size: 300,
            generations: 150,
            max_depth: 3,
            max_complexity: 24,
            seed: 42,
            ..RegressorConfig::for_autonomous_discovery()
        };
        let train_ic = [1.0, -0.5, 0.3, 0.2, 0.3, -0.1];
        let holdout_ics = [
            [0.4, 1.1, -0.6, 0.5, -0.3, 0.2],
            [-0.8, 0.2, 0.9, -0.1, 0.4, -0.5],
        ];
        let rho_stencil_names = ["u_l", "u_c", "u_r", "v_c"];
        let bond_names = ["u_c", "u_r", "v_c", "v_r"];
        let rho_probe_points: [[f64; 4]; 4] = [
            [0.3, -0.7, 0.2, -0.4],
            [1.0, 1.0, 0.0, 0.0],
            [-0.5, 0.5, 0.9, -0.9],
            [0.6, -0.2, -0.3, 0.7],
        ];
        let j_probe_points: [[f64; 4]; 4] = [
            [0.3, -0.7, 0.2, -0.4],
            [1.0, 1.0, 0.0, 0.0],
            [-0.5, 0.5, 0.9, -0.9],
            [0.6, -0.2, -0.3, 0.7],
        ];

        let candidates =
            discover_joint_density_and_flux(wave_rhs_n3, &train_ic, 3, &config, 10.0, 0.01);
        assert!(
            !candidates.is_empty(),
            "search should produce at least one finite-scoring, nontrivial candidate pair"
        );

        let mut cross_validated = None;
        for cand in &candidates {
            let holdout_residuals: Vec<f64> = holdout_ics
                .iter()
                .map(|ic| {
                    let traj = wave_trajectory_n3(*ic, 1000, 0.01);
                    discrete_continuity_residual(&cand.rho, &cand.j, &traj, 3)
                })
                .collect();
            let holdout_ok = holdout_residuals.iter().all(|r| r.is_finite() && *r < 1e-4);

            let max_rho_diff = rho_probe_points
                .iter()
                .map(|probe| {
                    let bindings: Vec<(&str, f64)> =
                        rho_stencil_names.iter().copied().zip(*probe).collect();
                    (rho_truth().eval(&bindings) - cand.rho.eval(&bindings)).abs()
                })
                .fold(0.0_f64, f64::max);
            let max_j_diff = j_probe_points
                .iter()
                .map(|probe| {
                    let bindings: Vec<(&str, f64)> =
                        bond_names.iter().copied().zip(*probe).collect();
                    (j_truth().eval(&bindings) - cand.j.eval(&bindings)).abs()
                })
                .fold(0.0_f64, f64::max);
            let max_diff = max_rho_diff.max(max_j_diff);

            println!(
                "candidate: rho=({}) j=({}) (train_residual={:.3e}, holdout_ok={holdout_ok}, \
                 max_rho_diff_vs_truth={max_rho_diff:.3e}, max_j_diff_vs_truth={max_j_diff:.3e})",
                cand.rho, cand.j, cand.residual
            );

            if holdout_ok && cross_validated.is_none() {
                cross_validated = Some((cand, max_diff));
            }
        }

        match cross_validated {
            Some((cand, diff)) if diff < 1e-6 => println!(
                "RESULT: joint (rho, J) recovered and symbolically close to the known answer \
                 (max diff {diff:e}): rho=({}), j=({}) -- genuine discovery, matching the known \
                 answer's equivalence class.",
                cand.rho, cand.j
            ),
            Some((cand, diff)) => println!(
                "RESULT: found a candidate pair that cross-validates on held-out trajectories \
                 but differs numerically from rho_truth()/j_truth() at probe points (max diff \
                 {diff:e}): rho=({}), j=({}) -- either a genuinely different (still-valid, \
                 gauge-inequivalent-looking) conserved current, or a subtler match this probe \
                 check doesn't recognize as equivalent. This is a legitimate outcome here \
                 specifically because the joint gauge freedom is larger than the rho-fixed \
                 case's -- worth manual inspection, not automatically a failure.",
                cand.rho, cand.j
            ),
            None => println!(
                "RESULT: no candidate pair cross-validated against held-out trajectories at \
                 this budget. Honest finding -- blind joint discovery is a genuinely harder \
                 problem than the rho-fixed case M2/M3 Phase 1/2 explored (2 free unknowns, a \
                 larger gauge group, and a nontriviality guard that must be satisfied by BOTH \
                 sides simultaneously). Not evidence of a bug in discrete_continuity_residual, \
                 gauge_fix_flux, or is_nontrivial (all independently exercised elsewhere). A \
                 larger budget or a refined density/flux grammar are the real next options, \
                 explicitly not pursued in this same turn."
            ),
        }
    }

    /// M2.1 diagnostic (not pass/fail -- reports the fitness landscape):
    /// does `discrete_continuity_residual` offer partial credit for the
    /// two structural factors `j_truth()` needs (the displacement
    /// `u_r-u_c` and the velocity sum `v_c+v_r`) individually, or does a
    /// candidate need both simultaneously before scoring any better than
    /// noise? If partial structures aren't rewarded, mutation/crossover
    /// has to assemble the whole product near-simultaneously -- a much
    /// harder search than incremental improvement toward the answer.
    #[test]
    fn component_fitness_landscape_around_true_flux() {
        let traj = wave_trajectory_n3([1.0, -0.5, 0.3, 0.2, 0.3, -0.1], 200, 0.01);
        let var = |n: &str| Expr::Var(n.to_string());
        let sub = |a: Expr, b: Expr| Expr::BinOp(BinOp::Sub, Box::new(a), Box::new(b));
        let add = |a: Expr, b: Expr| Expr::BinOp(BinOp::Add, Box::new(a), Box::new(b));
        let mul = |a: Expr, b: Expr| Expr::BinOp(BinOp::Mul, Box::new(a), Box::new(b));
        let scale =
            |c: f64, e: Expr| Expr::BinOp(BinOp::Mul, Box::new(Expr::Const(c)), Box::new(e));

        let candidates: Vec<(&str, Expr)> = vec![
            ("displacement alone: u_r - u_c", sub(var("u_r"), var("u_c"))),
            ("velocity sum alone: v_c + v_r", add(var("v_c"), var("v_r"))),
            (
                "wrong product: (u_r+u_c)(v_c+v_r)",
                mul(add(var("u_r"), var("u_c")), add(var("v_c"), var("v_r"))),
            ),
            (
                "wrong product: (u_r-u_c)(v_c-v_r)",
                mul(sub(var("u_r"), var("u_c")), sub(var("v_c"), var("v_r"))),
            ),
            ("correct product, wrong sign", j_wrong_sign()),
            (
                "correct + 0.1*u_c perturbation",
                add(j_truth(), scale(0.1, var("u_c"))),
            ),
            ("j_truth() itself (reference)", j_truth()),
            ("constant 0 (reference)", Expr::Const(0.0)),
        ];

        println!("--- M2.1 component-fitness landscape (train_residual, lower=better) ---");
        for (label, cand) in &candidates {
            let fixed = gauge_fix_flux(cand);
            let residual = discrete_continuity_residual(&rho_truth(), &fixed, &traj, 3);
            assert!(residual.is_finite(), "{label}: non-finite residual");
            println!("{label}: residual={residual:.4e}");
        }
    }

    /// Shared driver for the factorized-search seed sweeps (M2.2, M2.3):
    /// run `discover_flux_factorized` across `seeds`, cross-validate every
    /// returned candidate on held-out trajectories, print per-seed detail,
    /// return the recovery count. Same budget/seeds across conditions
    /// (unchanged, per the user's explicit instruction) so results are
    /// directly comparable.
    fn run_factorized_search_seeds(polynomial_only: bool, seeds: &[u64], label: &str) -> usize {
        let train_ic = [1.0, -0.5, 0.3, 0.2, 0.3, -0.1];
        let holdout_ics = [
            [0.4, 1.1, -0.6, 0.5, -0.3, 0.2],
            [-0.8, 0.2, 0.9, -0.1, 0.4, -0.5],
        ];
        let mut recovered = 0;

        for &seed in seeds {
            let config = RegressorConfig {
                population_size: 200,
                generations: 100,
                max_depth: 3,
                max_complexity: 24,
                seed,
                ..RegressorConfig::for_autonomous_discovery()
            };
            let candidates = discover_flux_factorized(
                &rho_truth(),
                wave_rhs_n3,
                &train_ic,
                3,
                &config,
                10.0,
                0.01,
                polynomial_only,
            );

            let mut seed_recovered = false;
            for cand in &candidates {
                let product = Expr::BinOp(
                    BinOp::Mul,
                    Box::new(cand.factor_a.clone()),
                    Box::new(cand.factor_b.clone()),
                );
                let fixed = gauge_fix_flux(&product);
                let holdout_ok = holdout_ics.iter().all(|ic| {
                    let traj = wave_trajectory_n3(*ic, 1000, 0.01);
                    let r = discrete_continuity_residual(&rho_truth(), &fixed, &traj, 3);
                    r.is_finite() && r < 1e-4
                });
                println!(
                    "[{label}] seed {seed}: {} (train_residual={:.3e}, holdout_ok={holdout_ok})",
                    cand.product_str, cand.residual
                );
                if holdout_ok && !seed_recovered {
                    seed_recovered = true;
                    recovered += 1;
                }
            }
            if !seed_recovered {
                println!("[{label}] seed {seed}: no candidate cross-validated");
            }
        }
        recovered
    }

    #[test]
    #[ignore] // GP search x5 seeds: slow, run explicitly.
    fn discover_flux_factorized_n3() {
        // M2.2: does decomposing the search into two independently-evolved
        // factors (J = A*B, see discover_flux_factorized's docs) solve the
        // structural credit-assignment problem M2.1's diagnostics found in
        // the single-tree search (0.022% reachability, no partial-credit
        // gradient toward the answer)? Run across several seeds -- a
        // single run's outcome isn't meaningful on its own; report a
        // recovery rate.
        let seeds: [u64; 5] = [1, 2, 3, 4, 5];
        let recovered = run_factorized_search_seeds(false, &seeds, "unrestricted");
        println!(
            "RESULT (M2.2, unrestricted grammar): recovered (cross-validated on held-out \
             trajectories) in {recovered}/{} seeded runs via factorized search. Compare to the \
             single-tree search's 0/1 at a comparable budget -- if this is meaningfully higher, \
             the bottleneck was structural credit assignment in the single-tree search \
             topology, not the verifier or data.",
            seeds.len()
        );
    }

    #[test]
    #[ignore] // GP search x5 seeds: slow, run explicitly.
    fn discover_flux_factorized_polynomial_n3() {
        // M2.3: same factorized search, same budget, same 5 seeds as M2.2
        // -- the only change is the grammar. j_truth() is exactly
        // Var/Const/Add/Sub/Mul (no Div/Pow/Func); this restricts the
        // search to that grammar to test whether trig/log/div/pow
        // "dilution" of the unrestricted grammar was the dominant
        // obstacle. Not unfair answer-leakage: the hypothesis is "a
        // physics-compatible polynomial language improves discovery
        // efficiency for polynomial dynamics," not a seed shaped like the
        // specific answer.
        let seeds: [u64; 5] = [1, 2, 3, 4, 5];
        let recovered = run_factorized_search_seeds(true, &seeds, "polynomial-only");
        println!(
            "RESULT (M2.3, polynomial-only grammar): recovered {recovered}/{} seeded runs. \
             Compare directly to M2.2's unrestricted-grammar result at the identical budget and \
             seeds -- if this is meaningfully higher, grammar dilution was the dominant \
             obstacle; if similar, the no-partial-credit landscape problem dominates regardless \
             of grammar size.",
            seeds.len()
        );
    }

    // -------------------------------------------------------------
    // M2.4a: HDC-guided semantic duplicate suppression, three-arm causal
    // comparison. Design frozen in the session's plan doc before any of
    // this was implemented: (1) no suppression -- M2.2/M2.3's baseline,
    // (2) direct-vector cosine similarity on a behavioral fingerprint,
    // (3) the same fingerprint encoded via ContinuousHV (DimensionalEncoder
    // pattern). Preregistered claim: behavioral duplicate suppression may
    // improve search efficiency by preventing repeated population
    // occupation by semantically equivalent expressions; HDC is compared
    // against a direct-vector control to determine whether any gain is
    // HDC-specific or just "any deduplication."
    // -------------------------------------------------------------

    /// Frozen probe bank for M2.4a's behavioral fingerprints: 24 states,
    /// generated once (fixed seed, one third-party trajectory distinct
    /// from every fitness/holdout IC used elsewhere in this file), never
    /// tuned after seeing which candidates survive. 8 uniform-random
    /// points, 4 points sampled from a genuinely different wave-chain
    /// trajectory, 4 sign-reversed variants, 4 left/right-swapped variants
    /// (probes the antisymmetry J should have under u_c<->u_r, v_c<->v_r),
    /// 4 sparse one-hot states.
    fn probe_bank() -> Vec<[f64; 4]> {
        let mut rng: u64 = 0xFEED_5EED_0000_0001;
        let next = |rng: &mut u64| -> f64 {
            *rng ^= *rng << 13;
            *rng ^= *rng >> 7;
            *rng ^= *rng << 17;
            let u = (*rng >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
            (u * 3.0) - 1.5 // [-1.5, 1.5]
        };

        let mut probes = Vec::with_capacity(24);
        let mut random_points: Vec<[f64; 4]> = Vec::with_capacity(8);
        for _ in 0..8 {
            let p = [
                next(&mut rng),
                next(&mut rng),
                next(&mut rng),
                next(&mut rng),
            ];
            random_points.push(p);
            probes.push(p);
        }

        // Third-party trajectory, distinct from train_ic and both
        // holdout_ics used everywhere else in this file -- probes must
        // never overlap physics-fitness evaluation data.
        let probe_ic = [0.6, -0.3, 0.7, -0.5, 0.2, -0.6];
        let probe_traj = wave_trajectory_n3(probe_ic, 400, 0.01);
        for i in [0usize, 100, 200, 300] {
            let s = &probe_traj[i.min(probe_traj.len() - 1)];
            probes.push([s[0], s[1], s[3], s[4]]); // bond (u1, u2, v1, v2)
        }

        for p in &random_points[..4] {
            probes.push([-p[0], -p[1], -p[2], -p[3]]);
        }
        for p in &random_points[..4] {
            probes.push([p[1], p[0], p[3], p[2]]); // u_c<->u_r, v_c<->v_r
        }

        probes.push([1.0, 0.0, 0.0, 0.0]);
        probes.push([0.0, 1.0, 0.0, 0.0]);
        probes.push([0.0, 0.0, 1.0, 0.0]);
        probes.push([0.0, 0.0, 0.0, 1.0]);

        probes
    }

    /// M2.4a result (real run, ~33 min, 15 seed-runs): 0/5 recovered in
    /// every arm. **Headline finding is methodological, not the recovery
    /// rate**: Arm2 (direct-vector) and Arm3 (HDC) produced *byte-identical*
    /// output on all 5 seeds (`total_duplicates_rejected=55021`,
    /// `total_false_merges=21556`, identical to the integer, both arms).
    /// This isn't "HDC provides no advantage" -- it's that
    /// `hdc_fingerprint`'s encoding (orthonormal basis via
    /// `ContinuousHV::orthogonal_set`, which does Gram-Schmidt +
    /// unit-normalize) is a linear *isometry*: cosine similarity of the
    /// encoded vectors exactly equals cosine similarity of the raw
    /// fingerprints, by construction. The comparison could not have found a
    /// difference between Arms 2/3 regardless of whether one exists in
    /// principle. A real HDC-vs-vector test needs a non-isometric encoding
    /// (binary/quantized hypervectors, an overcomplete/redundant
    /// projection, or a nonlinear step) -- not attempted here; flagged as
    /// the concrete next step if this line of investigation continues.
    ///
    /// Separately real: deduplication itself (Arms 2/3 vs Arm 1) gave a
    /// modest improvement in mean best training residual (0.643 vs 0.705)
    /// but didn't unlock recovery, alongside a ~39% false-merge rate among
    /// rejected duplicates at threshold 0.98 (21556/55021) -- a genuine
    /// signal the threshold may be too loose, discarding candidates whose
    /// actual fitness differs meaningfully under the banner of "same
    /// behavior."
    /// Run one dedup arm across `seeds`, print per-seed detail, print an
    /// aggregate summary line. Shared by `m24a_dedup_three_arm_comparison_n3`
    /// (original 3-arm run) and `m24a_corrected_hdc_arm_n3` (Arm 2 vs. the
    /// non-isometric Arm 3' only, avoiding redundant recompute of arms
    /// already characterized).
    fn run_dedup_arm(
        label: &str,
        mode: DedupMode,
        probes: &[[f64; 4]],
        basis: &[ContinuousHV],
        seeds: &[u64],
        train_ic: [f64; 6],
        holdout_ics: &[[f64; 6]],
    ) {
        let mut recovered = 0;
        let mut best_residuals: Vec<f64> = Vec::new();
        let (mut total_dup, mut total_unique, mut total_false_merge) = (0usize, 0usize, 0usize);

        for &seed in seeds {
            let config = RegressorConfig {
                population_size: 200,
                generations: 100,
                max_depth: 3,
                max_complexity: 24,
                seed,
                ..RegressorConfig::for_autonomous_discovery()
            };
            let (candidates, metrics_a, metrics_b) = discover_flux_factorized_with_dedup(
                &rho_truth(),
                wave_rhs_n3,
                &train_ic,
                3,
                &config,
                10.0,
                0.01,
                false,
                mode,
                probes,
                basis,
            );
            total_dup += metrics_a.duplicates_rejected + metrics_b.duplicates_rejected;
            total_unique +=
                metrics_a.unique_candidates_accepted + metrics_b.unique_candidates_accepted;
            total_false_merge += metrics_a.false_merges + metrics_b.false_merges;

            let mut seed_recovered = false;
            let mut seed_best = f64::INFINITY;
            for cand in &candidates {
                let product = Expr::BinOp(
                    BinOp::Mul,
                    Box::new(cand.factor_a.clone()),
                    Box::new(cand.factor_b.clone()),
                );
                let fixed = gauge_fix_flux(&product);
                let holdout_ok = holdout_ics.iter().all(|ic| {
                    let traj = wave_trajectory_n3(*ic, 1000, 0.01);
                    let r = discrete_continuity_residual(&rho_truth(), &fixed, &traj, 3);
                    r.is_finite() && r < 1e-4
                });
                seed_best = seed_best.min(cand.residual);
                if holdout_ok && !seed_recovered {
                    seed_recovered = true;
                    recovered += 1;
                }
            }
            if seed_best.is_finite() {
                best_residuals.push(seed_best);
            }
            println!(
                "[{label}] seed {seed}: best_train_residual={seed_best:.3e}, \
                 recovered={seed_recovered}, dup_rejected(A+B)={}, \
                 unique_accepted(A+B)={}, false_merges(A+B)={}",
                metrics_a.duplicates_rejected + metrics_b.duplicates_rejected,
                metrics_a.unique_candidates_accepted + metrics_b.unique_candidates_accepted,
                metrics_a.false_merges + metrics_b.false_merges,
            );
        }

        let mean_best = if best_residuals.is_empty() {
            f64::NAN
        } else {
            best_residuals.iter().sum::<f64>() / best_residuals.len() as f64
        };
        println!(
            "=== {label}: recovered {recovered}/{} seeds, mean_best_train_residual={mean_best:.3e}, \
             total_duplicates_rejected={total_dup}, total_unique_accepted={total_unique}, \
             total_false_merges={total_false_merge} ===",
            seeds.len()
        );
    }

    #[test]
    #[ignore] // GP search x3 arms x5 seeds: very slow (~33 min), run explicitly.
    fn m24a_dedup_three_arm_comparison_n3() {
        let probes = probe_bank();
        let basis = hdc_probe_basis(probes.len(), 0xBA51_5EED);

        let train_ic = [1.0, -0.5, 0.3, 0.2, 0.3, -0.1];
        let holdout_ics = [
            [0.4, 1.1, -0.6, 0.5, -0.3, 0.2],
            [-0.8, 0.2, 0.9, -0.1, 0.4, -0.5],
        ];
        let seeds: [u64; 5] = [1, 2, 3, 4, 5];
        let threshold = 0.98;
        let arms: [(&str, DedupMode); 3] = [
            ("Arm1-none", DedupMode::None),
            ("Arm2-vector", DedupMode::Vector { threshold }),
            ("Arm3-hdc", DedupMode::Hdc { threshold }),
        ];

        for (label, mode) in arms {
            run_dedup_arm(label, mode, &probes, &basis, &seeds, train_ic, &holdout_ics);
        }

        println!(
            "M2.4a NOTE: this run reports the core preregistered comparison (recovery, best \
             residual, duplicates rejected, unique candidates accepted, false merges). It does \
             NOT implement every metric from the frozen design doc -- population behavioral \
             diversity (mean pairwise similarity), per-generation motif incidence tracking, \
             duplicate-miss counting, and the threshold sensitivity sweep were scoped out for \
             this pass and are honest simplifications, not silent omissions."
        );
    }

    /// M2.4a-corrected: re-run the HDC-vs-vector comparison with a
    /// genuinely non-isometric HDC arm. `DedupMode::Hdc` (Arm 3 above) was
    /// proven byte-identical to `DedupMode::Vector` (Arm 2) because
    /// `hdc_fingerprint`'s orthonormal-basis encoding is a linear isometry
    /// -- see the doc comment on `m24a_dedup_three_arm_comparison_n3` and
    /// `feedback_hdc_orthonormal_encoding_is_isometry.md`. `DedupMode::HdcQuantized`
    /// breaks the isometry by lossily bucketing the raw fingerprint
    /// (`quantize_fingerprint`) *before* HDC encoding, so two behaviorally
    /// close-but-distinct candidates can collapse to the same bucketed
    /// fingerprint even though their raw cosine similarity differs --
    /// verified as genuinely non-isometric by
    /// `quantized_hdc_similarity_can_differ_from_raw_vector_similarity` in
    /// `flux_discovery.rs`. Only Arm1 (none) is skipped here -- its result
    /// is already recorded above and doesn't bear on the HDC-vs-vector
    /// question -- to avoid ~11 minutes of redundant recompute.
    ///
    /// **Real result (5/5 seeds, ~17 min)**: Arm2 and Arm3' are no longer
    /// byte-identical -- duplicates_rejected 55021 vs 56240, false_merges
    /// 21556 vs 20775, mean_best_train_residual 0.643 vs 0.681 -- confirming
    /// the encoding is genuinely non-isometric this time. Still 0/5
    /// recovered in both arms, and Arm3' (HDC-quantized)'s mean best
    /// residual is *worse* than Arm2 (vector)'s, not better. No evidence
    /// that non-isometric HDC dedup outperforms direct-vector dedup here;
    /// closes the HDC-vs-vector question for this mechanism as an honest
    /// negative, not a methodological artifact.
    #[test]
    #[ignore] // GP search x2 arms x5 seeds: slow (~20 min), run explicitly.
    fn m24a_corrected_hdc_arm_n3() {
        let probes = probe_bank();
        let basis = hdc_probe_basis(probes.len(), 0xBA51_5EED);

        let train_ic = [1.0, -0.5, 0.3, 0.2, 0.3, -0.1];
        let holdout_ics = [
            [0.4, 1.1, -0.6, 0.5, -0.3, 0.2],
            [-0.8, 0.2, 0.9, -0.1, 0.4, -0.5],
        ];
        let seeds: [u64; 5] = [1, 2, 3, 4, 5];
        let threshold = 0.98;
        let bucket_width = 0.25;
        let arms: [(&str, DedupMode); 2] = [
            ("Arm2-vector", DedupMode::Vector { threshold }),
            (
                "Arm3prime-hdc-quantized",
                DedupMode::HdcQuantized {
                    threshold,
                    bucket_width,
                },
            ),
        ];

        for (label, mode) in arms {
            run_dedup_arm(label, mode, &probes, &basis, &seeds, train_ic, &holdout_ics);
        }

        println!(
            "M2.4a-corrected NOTE: same scope limitations as the original M2.4a run apply \
             (see that test's trailing note) -- this pass only settles whether a genuinely \
             non-isometric HDC encoding behaves differently from direct-vector dedup, not the \
             full frozen metric set."
        );
    }

    /// True iff `snap` shows the *full* structural motif (both factors
    /// simultaneously correct in kind, not just one side) present in the
    /// current generation's best pair.
    fn snapshot_has_full_motif(snap: &GenerationSnapshot) -> bool {
        snap.best_a_has_displacement_motif && snap.best_b_has_velocity_motif
    }

    /// First generation index at which the full motif appears, and what
    /// fraction of the *remaining* generations (from first appearance to
    /// the end) continued to show it -- a simple persistence proxy. `None`
    /// first-appearance implies `0.0` persistence (nothing to persist).
    fn motif_onset_and_persistence(snapshots: &[GenerationSnapshot]) -> (Option<usize>, f64) {
        let Some(first) = snapshots.iter().position(snapshot_has_full_motif) else {
            return (None, 0.0);
        };
        let remaining = &snapshots[first..];
        let held = remaining
            .iter()
            .filter(|s| snapshot_has_full_motif(s))
            .count();
        (Some(first), held as f64 / remaining.len() as f64)
    }

    /// M2.5: paired evolutionary comparison of the existing raw-residual
    /// fitness (`discover_flux_factorized_with_snapshots`, selection
    /// criterion unchanged from M2.2/M2.3/M2.4a) against the target-blind
    /// shape-calibrated fitness (`discover_flux_factorized_shaped`).
    /// Everything else held fixed per the frozen M2.5 design: grammar
    /// (general, `polynomial_only=false`), factorized representation,
    /// population/generation budget (200/100), mutation/crossover/
    /// selection mechanics (shared `breed_generation`), gauge-fixing
    /// canonicalization, training trajectory (`train_ic`), held-out
    /// trajectories (`holdout_ics`) -- all identical to every prior M2.x
    /// comparison in this file. No HDC, novelty selection, larger
    /// populations, or curriculum learning are introduced here.
    ///
    /// 10 paired seeds (the frozen initial-gate size) x 2 arms. Predeclared
    /// stopping rule (per the M2.5 design): continue M2 only if the shaped
    /// arm shows validated recovery, a major held-out-residual reduction,
    /// clearly increased structural-motif persistence, or a materially
    /// earlier/more frequent motif onset than the raw arm. Otherwise close
    /// M2 as a fully-diagnosed honest negative.
    ///
    /// **Real result (10/10 seeds, ~45 min GP search on top of a ~105 min
    /// LTO release compile under heavy concurrent-session load): all four
    /// stopping-rule criteria fail, and not just as "no improvement" --
    /// the shaped arm actively regresses on every continuous metric.**
    /// recovered 0/10 both arms. mean_best_train_residual: raw=0.621,
    /// shaped=1.235 (~2x worse). mean_holdout_residual: raw=0.748,
    /// shaped=1.414 (~1.9x worse). train_val_gap: raw=0.127, shaped=0.179.
    /// mean_complexity: raw=13.5, shaped=18.5 (more bloat for a worse
    /// result). motif_onset: 0/10 in both arms (tied). Most diagnostic:
    /// mean_final_alignment -- the exact quantity the shaped fitness
    /// directly selects for -- is only 0.0241 for the shaped arm itself,
    /// vs. -0.4218 (much stronger magnitude) for the raw arm, which never
    /// optimizes for alignment at all. See the module-level doc comment
    /// for the full writeup and the proposed mechanism (partial credit
    /// reduces selection pressure, letting mediocre candidates crowd out
    /// the rare good ones). Closes M2 per the predeclared rule -- no
    /// escalation to 20-30 seeds, since escalation was conditioned on
    /// meaningful improvement, not regression.
    #[test]
    #[ignore] // GP search x2 arms x10 seeds: slow, run explicitly.
    fn m25_shaped_vs_raw_residual_n3() {
        let train_ic = [1.0_f64, -0.5, 0.3, 0.2, 0.3, -0.1];
        let holdout_ics = [
            [0.4_f64, 1.1, -0.6, 0.5, -0.3, 0.2],
            [-0.8_f64, 0.2, 0.9, -0.1, 0.4, -0.5],
        ];
        let seeds: [u64; 10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let train_traj = wave_trajectory_n3(train_ic, 1000, 0.01);
        let holdout_trajs: Vec<Vec<Vec<f64>>> = holdout_ics
            .iter()
            .map(|ic| wave_trajectory_n3(*ic, 1000, 0.01))
            .collect();

        for &use_shaped in &[false, true] {
            let label = if use_shaped { "M2.5-shaped" } else { "M2-raw" };
            let mut recovered = 0;
            let mut train_residuals: Vec<f64> = Vec::new();
            let mut holdout_residuals: Vec<f64> = Vec::new();
            let mut complexities: Vec<usize> = Vec::new();
            let mut onset_gens: Vec<usize> = Vec::new();
            let mut persistences: Vec<f64> = Vec::new();
            let mut final_alignments: Vec<f64> = Vec::new();

            for &seed in &seeds {
                let config = RegressorConfig {
                    population_size: 200,
                    generations: 100,
                    max_depth: 3,
                    max_complexity: 24,
                    seed,
                    ..RegressorConfig::for_autonomous_discovery()
                };

                // `finals`: gauge-fixed (and, for the shaped arm,
                // alpha-scaled) candidate expressions, sorted best-first by
                // each arm's OWN native fitness -- used identically by both
                // arms below for classic (apples-to-apples) recovery/
                // residual/complexity reporting.
                let (finals, snapshots): (Vec<Expr>, Vec<GenerationSnapshot>) = if use_shaped {
                    let (results, snaps) = discover_flux_factorized_shaped(
                        &rho_truth(),
                        wave_rhs_n3,
                        &train_ic,
                        3,
                        &config,
                        10.0,
                        0.01,
                        false,
                    );
                    (
                        results.into_iter().map(|r| r.scaled_product).collect(),
                        snaps,
                    )
                } else {
                    let (results, snaps) = discover_flux_factorized_with_snapshots(
                        &rho_truth(),
                        wave_rhs_n3,
                        &train_ic,
                        3,
                        &config,
                        10.0,
                        0.01,
                        false,
                    );
                    let finals = results
                        .into_iter()
                        .map(|r| {
                            let product =
                                Expr::BinOp(BinOp::Mul, Box::new(r.factor_a), Box::new(r.factor_b));
                            gauge_fix_flux(&product)
                        })
                        .collect();
                    (finals, snaps)
                };

                let mut seed_recovered = false;
                let mut best_train: Option<f64> = None;
                let mut best_train_holdout_mean: Option<f64> = None;
                let mut best_complexity: Option<usize> = None;
                for cand in &finals {
                    let train_r = discrete_continuity_residual(&rho_truth(), cand, &train_traj, 3);
                    if !train_r.is_finite() {
                        continue;
                    }
                    if best_train.is_none_or(|b| train_r < b) {
                        best_train = Some(train_r);
                        best_complexity = Some(cand.complexity());
                        let mean_holdout: f64 = holdout_trajs
                            .iter()
                            .map(|traj| discrete_continuity_residual(&rho_truth(), cand, traj, 3))
                            .sum::<f64>()
                            / holdout_trajs.len() as f64;
                        best_train_holdout_mean = Some(mean_holdout);
                    }
                    let holdout_ok = holdout_trajs.iter().all(|traj| {
                        let r = discrete_continuity_residual(&rho_truth(), cand, traj, 3);
                        r.is_finite() && r < 1e-4
                    });
                    if holdout_ok {
                        seed_recovered = true;
                    }
                }
                if seed_recovered {
                    recovered += 1;
                }
                if let Some(tr) = best_train {
                    train_residuals.push(tr);
                }
                if let Some(hr) = best_train_holdout_mean {
                    holdout_residuals.push(hr);
                }
                if let Some(c) = best_complexity {
                    complexities.push(c);
                }

                let (onset, persistence) = motif_onset_and_persistence(&snapshots);
                if let Some(g) = onset {
                    onset_gens.push(g);
                }
                persistences.push(persistence);
                final_alignments.push(snapshots.last().map(|s| s.best_alignment).unwrap_or(0.0));

                println!(
                    "[{label}] seed {seed}: recovered={seed_recovered}, \
                     best_train_residual={best_train:?}, \
                     motif_onset_gen={onset:?}, persistence={persistence:.2}"
                );
            }

            let mean = |v: &[f64]| -> f64 {
                if v.is_empty() {
                    f64::NAN
                } else {
                    v.iter().sum::<f64>() / v.len() as f64
                }
            };
            let mean_usize = |v: &[usize]| -> f64 {
                if v.is_empty() {
                    f64::NAN
                } else {
                    v.iter().sum::<usize>() as f64 / v.len() as f64
                }
            };
            let mean_train = mean(&train_residuals);
            let mean_holdout = mean(&holdout_residuals);

            println!(
                "=== {label}: recovered {recovered}/{} seeds | \
                 mean_best_train_residual={mean_train:.4e} | \
                 mean_holdout_residual={mean_holdout:.4e} | \
                 train_val_gap={:.4e} | \
                 mean_complexity={:.1} | \
                 motif_onset: {}/{} seeds ever showed it, mean_onset_gen={:.1} | \
                 mean_persistence={:.3} | \
                 mean_final_alignment={:.4} ===",
                seeds.len(),
                (mean_train - mean_holdout).abs(),
                mean_usize(&complexities),
                onset_gens.len(),
                seeds.len(),
                mean_usize(&onset_gens),
                mean(&persistences),
                mean(&final_alignments),
            );
        }

        println!(
            "M2.5 NOTE: 10-seed initial gate per the frozen design. Escalation to 20-30 \
             seeds is deliberately NOT automatic here -- report this gate's result first \
             and decide whether escalation or closure is warranted, per the predeclared \
             stopping rule."
        );
    }
}
