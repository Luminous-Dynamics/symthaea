// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Stage B M2: constrained discovery of the flux `J` given a known-correct
//! density `ρ`.
//!
//! Fixing `ρ` (verified in `continuity.rs`'s M0/M1 work) turns "discover a
//! Noether current from scratch" into a more tractable single-unknown
//! search: find `J` such that the discrete continuity equation
//! (`discrete_continuity_residual`, see `continuity.rs`) holds. Every
//! candidate is gauge-fixed ([`super::gauge_fix_flux`]) before scoring --
//! without that, `J_truth + c` for any constant `c` scores identically to
//! `J_truth`, and the search is underdetermined.
//!
//! Deliberately seeded with only *generic* random expressions over the
//! bond-stencil names (`u_c, u_r, v_c, v_r`) -- no template shaped like the
//! true answer, matching Stage A's N=3-unseeded precedent: this tests
//! genuine discovery, not whether a hand-fed answer survives.

use super::*;
use crate::hdc::{ContinuousHV, HDC_DIMENSION};

/// A candidate flux surviving [`discover_flux_given_density`]'s search,
/// already gauge-fixed. `residual` is scored against the *training*
/// trajectory only -- callers should cross-validate against held-out
/// trajectories/initial conditions before trusting a result, same as Stage
/// A's `discover_invariants_autonomous` callers do.
#[derive(Debug, Clone)]
pub struct FluxDiscoveryResult {
    pub formula: Expr,
    pub formula_str: String,
    pub residual: f64,
}

const BOND_VAR_NAMES: [&str; 4] = ["u_c", "u_r", "v_c", "v_r"];

/// Fraction of each generation's non-elite population replaced with fresh
/// random individuals rather than elite-bred offspring. Rejected candidates
/// never enter the scalar fitness domain (`score` returns `None`, not a
/// sentinel value like `f64::MAX`) -- sentinels tend to leak into sorting/
/// aggregation exactly the way one did in an earlier version of this
/// function (a `f64::MAX`-scored candidate passed a bare `is_finite()`
/// filter and leaked into the results list; fixed by removing the sentinel
/// convention entirely here rather than adding another bound to check for
/// it).
const RANDOM_INJECTION_FRACTION: f64 = 0.10;

/// `None` if `expr` is out of the complexity band or fails to produce a
/// finite discrete-continuity residual after gauge-fixing; `Some(residual)`
/// otherwise ("small is good", matching [`discrete_continuity_residual`]'s
/// convention for the value it wraps).
fn score(
    rho: &Expr,
    expr: &Expr,
    trajectory: &[Vec<f64>],
    n: usize,
    max_complexity: usize,
) -> Option<f64> {
    let c = expr.complexity();
    if !(2..=max_complexity).contains(&c) {
        return None;
    }
    let fixed = gauge_fix_flux(expr);
    let residual = discrete_continuity_residual(rho, &fixed, trajectory, n);
    if residual.is_finite() && residual < 1e10 {
        Some(residual)
    } else {
        None
    }
}

fn cmp_scores(a: Option<f64>, b: Option<f64>) -> std::cmp::Ordering {
    match (a, b) {
        (Some(x), Some(y)) => x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    }
}

fn fresh_random_bond_expr(rng: &mut u64, max_depth: usize, exclude_trig: bool) -> Expr {
    *rng = lcg_step(*rng);
    random_expr_multivar(rng, max_depth, &BOND_VAR_NAMES, exclude_trig)
}

/// Strictly polynomial random generator: `Var`/`Const`/`Add`/`Sub`/`Mul`
/// only -- no `Div`, no `Pow`, no `Func` (so no `Sqrt`/`Log`/`Exp`/`Sin`/
/// `Cos` either). Matches `j_truth()`'s actual grammar exactly (M2.3: tests
/// whether trig/log/div/pow "dilution" of the unrestricted grammar was the
/// dominant obstacle, not just diversity or structural credit assignment).
fn fresh_random_polynomial_bond_expr(rng: &mut u64, max_depth: usize) -> Expr {
    *rng = lcg_step(*rng);
    if max_depth == 0 || (*rng % 3 == 0 && max_depth < 3) {
        *rng = lcg_step(*rng);
        if *rng % 3 == 0 {
            *rng = lcg_step(*rng);
            Expr::Var(BOND_VAR_NAMES[*rng as usize % BOND_VAR_NAMES.len()].to_string())
        } else {
            let constants = [0.0, 0.5, 1.0, 2.0, 3.0, -1.0, -0.5];
            *rng = lcg_step(*rng);
            Expr::Const(constants[*rng as usize % constants.len()])
        }
    } else {
        let ops = [BinOp::Add, BinOp::Sub, BinOp::Mul];
        *rng = lcg_step(*rng);
        let op = ops[*rng as usize % ops.len()];
        Expr::BinOp(
            op,
            Box::new(fresh_random_polynomial_bond_expr(rng, max_depth - 1)),
            Box::new(fresh_random_polynomial_bond_expr(rng, max_depth - 1)),
        )
    }
}

/// `true` if `expr` uses only `Var`/`Const`/`Add`/`Sub`/`Mul` -- the
/// grammar [`fresh_random_polynomial_bond_expr`] generates. Needed because
/// `Expr::mutate`'s internal subtree-replacement fallback
/// (`expressions::random_expr`) is *not* grammar-aware -- it can introduce
/// `Div`/`Pow`/`Func` nodes (and an unrelated hardcoded `"n"` variable)
/// regardless of what grammar seeded the population. Without this check, a
/// "polynomial-only" run would silently drift away from that grammar after
/// a few generations of mutation.
fn is_polynomial(expr: &Expr) -> bool {
    match expr {
        Expr::Var(_) | Expr::Const(_) => true,
        Expr::BinOp(op, l, r) => {
            matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul)
                && is_polynomial(l)
                && is_polynomial(r)
        }
        Expr::Func(_, _) | Expr::Sum(_, _) => false,
    }
}

/// `lcg_step`-advance `rng`, then draw a uniform index in `0..n` from the state's **high** 32
/// bits, not the low bits. Real bug found while calibrating this module (see
/// `debug_print_sample_motif_trees` in this file's history / the M3 memory writeup): this
/// crate's `lcg_step` is a standard full-period-2^64 multiplicative LCG, and for *any* such
/// LCG the low-order bits have a provably short cycle (bit k has period at most `2^(k+1)`,
/// independent of the rest of the state) -- a classical, well-documented LCG pitfall
/// ("Numerical Recipes" explicitly warns against it). `var_names.len() == 4`, a power of 2, so
/// `*rng % 4` was extracting exactly the two weakest bits: an empirical debug dump showed only
/// 2 of the 6 possible variable pairs from [`two_distinct_vars`] ever appearing across many
/// draws, which silently made the specific target motifs unreachable regardless of the
/// generator's intended bias. The high 32 bits of a full-period LCG don't have this problem.
/// Scoped to this module's own new code only -- not touching the established (and already
/// experimentally closed) M2 generators elsewhere in this file, which use the same raw
/// low-bit pattern but are out of scope for this plan to re-audit.
fn rand_index_high_bits(rng: &mut u64, n: usize) -> usize {
    *rng = lcg_step(*rng);
    ((*rng >> 32) as usize) % n
}

/// Pick two *distinct* variable names uniformly at random from `var_names`.
/// Panics if `var_names` has fewer than 2 entries (a caller bug, not a
/// runtime condition this module needs to handle gracefully -- `BOND_VAR_NAMES`
/// always has 4).
fn two_distinct_vars(rng: &mut u64, var_names: &[&str]) -> (String, String) {
    let i = rand_index_high_bits(rng, var_names.len());
    let mut j = i;
    while j == i {
        j = rand_index_high_bits(rng, var_names.len());
    }
    (var_names[i].to_string(), var_names[j].to_string())
}

/// M3 Phase 1: constructive generator biased toward physically-meaningful
/// compositional *motifs* -- difference-of-two-variables, sum-of-two-
/// variables, product-of-two-subexpressions -- instead of
/// [`fresh_random_bond_expr`]'s unbiased assembly of `Var`/`Const`/`BinOp`/
/// `Func` nodes. Generic over **which** variable pair fills each
/// difference/sum slot (drawn uniformly from `var_names` via
/// [`two_distinct_vars`], never hand-fed a specific pair) -- this biases
/// toward the *class* of structures conserved currents tend to be built
/// from, not toward any one target formula, matching the non-leakage
/// standard `discover_flux_factorized`'s own docs hold M2.2/M2.3 to.
///
/// See the module-level plan doc (or `pde_wave_stage_b.rs`'s M3 note) for
/// why this exists: M2.1 measured that only 0.022% of trees from
/// [`fresh_random_bond_expr`] contain both structural motifs the true flux
/// needs; this generator is a direct test of whether a constructive,
/// motif-biased generator -- the same technique `typed_generation.rs`
/// (`symthaea-physics-bridge`) used to fix an analogous dimensional-
/// reachability gap -- closes that gap here too. See
/// `motif_constrained_generator_reachability_estimate` for the measured
/// result. `pub` (not just `pub(crate)`) so M3 Phase 2 (`pde_wave_stage_b.rs`, a different
/// crate) can seed an actual evolutionary search with it via
/// [`discover_flux_given_density_seeded`], not just measure raw reachability.
pub fn random_motif_expr(rng: &mut u64, var_names: &[&str], depth: usize) -> Expr {
    const MOTIF_CONSTANTS: [f64; 5] = [0.5, 1.0, 2.0, -1.0, -0.5];
    // Stop into a motif leaf either at the depth floor or, above it, about
    // 1-in-3 of the time -- biases toward at least one level of `Mul`
    // recursion (needed to reach a product-of-two-motifs form) while still
    // allowing shallow/single-motif trees for diversity.
    let leafy = depth == 0 || rand_index_high_bits(rng, 3) == 0;
    if leafy {
        match rand_index_high_bits(rng, 4) {
            0 => {
                let (a, b) = two_distinct_vars(rng, var_names);
                Expr::BinOp(BinOp::Sub, Box::new(Expr::Var(a)), Box::new(Expr::Var(b)))
            }
            1 => {
                let (a, b) = two_distinct_vars(rng, var_names);
                Expr::BinOp(BinOp::Add, Box::new(Expr::Var(a)), Box::new(Expr::Var(b)))
            }
            2 => {
                let v = var_names[rand_index_high_bits(rng, var_names.len())];
                let c = MOTIF_CONSTANTS[rand_index_high_bits(rng, MOTIF_CONSTANTS.len())];
                Expr::BinOp(
                    BinOp::Mul,
                    Box::new(Expr::Const(c)),
                    Box::new(Expr::Var(v.to_string())),
                )
            }
            _ => Expr::Const(MOTIF_CONSTANTS[rand_index_high_bits(rng, MOTIF_CONSTANTS.len())]),
        }
    } else {
        let op = match rand_index_high_bits(rng, 3) {
            0 | 1 => BinOp::Mul, // double-weighted: the target is a product of two motifs
            _ => {
                if rand_index_high_bits(rng, 2) == 0 {
                    BinOp::Add
                } else {
                    BinOp::Sub
                }
            }
        };
        Expr::BinOp(
            op,
            Box::new(random_motif_expr(rng, var_names, depth - 1)),
            Box::new(random_motif_expr(rng, var_names, depth - 1)),
        )
    }
}

/// M3 Phase 3: like [`random_motif_expr`], but for density-shaped targets. `rho_truth()`
/// (`pde_wave_stage_b.rs`) is a *sum* of *squared* terms (kinetic `0.5*v_c^2` plus two bond
/// terms `0.25*(u-u)^2`), a structurally different shape from the flux's product-of-two-motifs
/// -- neither `random_motif_expr` nor any other generator in this file can reach a "square of a
/// specific subexpression" motif by chance (`Mul(X,X)` needs two independently-drawn subtrees
/// to be structurally identical, vanishingly unlikely). Deliberately a **separate** function,
/// not a modification of `random_motif_expr` -- that generator's Phase 1/2 results are already
/// historical and shouldn't retroactively change. Leaf motifs: `c*(a-b)^2`, `c*v^2` (covers the
/// kinetic-term shape directly), `a-b`, `a+b`, or a constant -- generic over which variable pair
/// fills each slot, same non-leakage standard as `random_motif_expr`. Recursive combination is
/// `Add`/`Sub`-only (no `Mul`), matching `rho_truth()`'s own additive structure.
pub fn random_density_motif_expr(rng: &mut u64, var_names: &[&str], depth: usize) -> Expr {
    const MOTIF_CONSTANTS: [f64; 5] = [0.5, 1.0, 2.0, -1.0, -0.5];
    let leafy = depth == 0 || rand_index_high_bits(rng, 3) == 0;
    if leafy {
        match rand_index_high_bits(rng, 5) {
            0 => {
                let (a, b) = two_distinct_vars(rng, var_names);
                let c = MOTIF_CONSTANTS[rand_index_high_bits(rng, MOTIF_CONSTANTS.len())];
                let base = Expr::BinOp(BinOp::Sub, Box::new(Expr::Var(a)), Box::new(Expr::Var(b)));
                let squared = Expr::BinOp(BinOp::Pow, Box::new(base), Box::new(Expr::Const(2.0)));
                Expr::BinOp(BinOp::Mul, Box::new(Expr::Const(c)), Box::new(squared))
            }
            1 => {
                let v = var_names[rand_index_high_bits(rng, var_names.len())];
                let c = MOTIF_CONSTANTS[rand_index_high_bits(rng, MOTIF_CONSTANTS.len())];
                let squared = Expr::BinOp(
                    BinOp::Pow,
                    Box::new(Expr::Var(v.to_string())),
                    Box::new(Expr::Const(2.0)),
                );
                Expr::BinOp(BinOp::Mul, Box::new(Expr::Const(c)), Box::new(squared))
            }
            2 => {
                let (a, b) = two_distinct_vars(rng, var_names);
                Expr::BinOp(BinOp::Sub, Box::new(Expr::Var(a)), Box::new(Expr::Var(b)))
            }
            3 => {
                let (a, b) = two_distinct_vars(rng, var_names);
                Expr::BinOp(BinOp::Add, Box::new(Expr::Var(a)), Box::new(Expr::Var(b)))
            }
            _ => Expr::Const(MOTIF_CONSTANTS[rand_index_high_bits(rng, MOTIF_CONSTANTS.len())]),
        }
    } else {
        let op = if rand_index_high_bits(rng, 3) == 0 {
            BinOp::Sub
        } else {
            BinOp::Add // double-weighted: rho_truth() is a SUM of squared terms
        };
        Expr::BinOp(
            op,
            Box::new(random_density_motif_expr(rng, var_names, depth - 1)),
            Box::new(random_density_motif_expr(rng, var_names, depth - 1)),
        )
    }
}

/// FPU-alpha flux generator: like [`random_motif_expr`], but with one extra leaf option so it
/// can actually *construct* a squared-displacement shape, not just stumble onto one by
/// `Mul(X,X)` coincidence. `j_fpu_truth(alpha)` (`pde_wave_stage_b.rs`) needs `Pow(Sub(u_r,u_c),
/// Const(2.0))` as one multiplicative factor alongside a `(v_c+v_r)`-shaped factor -- a
/// reachability diagnostic measured that `random_motif_expr` (no `Pow`/square combinator at all)
/// only reaches that shape via two independently-drawn identical `Sub(u_r,u_c)` subtrees
/// coincidentally multiplying together, at a rate (0.028% co-occurrence with the velocity-sum
/// motif) statistically indistinguishable from the wholly-unbiased M2.1 baseline (0.022%) that
/// `random_motif_expr` itself was built to improve on for the *linear* motifs.
///
/// Deliberately a **separate** function, not a modification of `random_motif_expr` -- matching
/// [`random_density_motif_expr`]'s own stated precedent for exactly this situation (an
/// established generator's historical Phase 1/2 results are already on record and shouldn't
/// retroactively change). Reuses `random_motif_expr`'s exact leafy-stop condition and recursive
/// `Mul`(double-weighted)/`Add`/`Sub` combination logic unchanged -- the only difference is the
/// leaf selection widens from 4 to 5 options, with the new 5th arm building an **unscaled**
/// squared-diff leaf `Pow(Sub(a,b), Const(2.0))` (deliberately without
/// [`random_density_motif_expr`]'s leading `Const*` multiplier on the leaf itself -- unlike
/// `rho_truth()`'s squared terms, `j_fpu_truth`'s squared term carries no per-leaf scale; the
/// overall `0.5*alpha` scale sits at the outer `Mul` node, which the existing `Const*Var` leaf +
/// recursive `Mul` combinator can already supply without any help from this new leaf).
pub fn random_fpu_flux_motif_expr(rng: &mut u64, var_names: &[&str], depth: usize) -> Expr {
    const MOTIF_CONSTANTS: [f64; 5] = [0.5, 1.0, 2.0, -1.0, -0.5];
    let leafy = depth == 0 || rand_index_high_bits(rng, 3) == 0;
    if leafy {
        match rand_index_high_bits(rng, 5) {
            0 => {
                let (a, b) = two_distinct_vars(rng, var_names);
                Expr::BinOp(BinOp::Sub, Box::new(Expr::Var(a)), Box::new(Expr::Var(b)))
            }
            1 => {
                let (a, b) = two_distinct_vars(rng, var_names);
                Expr::BinOp(BinOp::Add, Box::new(Expr::Var(a)), Box::new(Expr::Var(b)))
            }
            2 => {
                let v = var_names[rand_index_high_bits(rng, var_names.len())];
                let c = MOTIF_CONSTANTS[rand_index_high_bits(rng, MOTIF_CONSTANTS.len())];
                Expr::BinOp(
                    BinOp::Mul,
                    Box::new(Expr::Const(c)),
                    Box::new(Expr::Var(v.to_string())),
                )
            }
            3 => {
                let (a, b) = two_distinct_vars(rng, var_names);
                let base = Expr::BinOp(BinOp::Sub, Box::new(Expr::Var(a)), Box::new(Expr::Var(b)));
                Expr::BinOp(BinOp::Pow, Box::new(base), Box::new(Expr::Const(2.0)))
            }
            _ => Expr::Const(MOTIF_CONSTANTS[rand_index_high_bits(rng, MOTIF_CONSTANTS.len())]),
        }
    } else {
        let op = match rand_index_high_bits(rng, 3) {
            0 | 1 => BinOp::Mul, // double-weighted: the target is a product of two motifs
            _ => {
                if rand_index_high_bits(rng, 2) == 0 {
                    BinOp::Add
                } else {
                    BinOp::Sub
                }
            }
        };
        Expr::BinOp(
            op,
            Box::new(random_fpu_flux_motif_expr(rng, var_names, depth - 1)),
            Box::new(random_fpu_flux_motif_expr(rng, var_names, depth - 1)),
        )
    }
}

/// M3 Phase 3: true iff `expr`'s response varies meaningfully across `probes` -- i.e. it isn't
/// (numerically) a constant. **Mathematically required, not optional**, for blind joint
/// density/flux discovery: with both `rho` and `J` free (unlike every M2 sub-experiment and M3
/// Phase 1/2, which fixed `rho`), the pair `rho=c1, J=c2` (any constants) satisfies the discrete
/// continuity equation *exactly* -- `dρ/dt=0` for a constant, and `J_{i+1/2}-J_{i-1/2}=c2-c2=0`,
/// so `0=0` -- a free, perfect-residual escape hatch that never existed when `rho` was fixed.
/// Without this gate, evolution would trivially "solve" the problem by collapsing to constants.
/// Threshold (`1e-9`) matches `continuity.rs`'s own `MIN_SHAPE_VARIANCE`, this codebase's
/// established convention for "how much variance counts as nondegenerate."
fn is_nontrivial(expr: &Expr, var_names: &[&str], probes: &[Vec<f64>]) -> bool {
    let vals: Option<Vec<f64>> = probes
        .iter()
        .map(|p| {
            let bindings: Vec<(&str, f64)> =
                var_names.iter().copied().zip(p.iter().copied()).collect();
            let v = expr.eval(&bindings);
            v.is_finite().then_some(v)
        })
        .collect();
    let Some(vals) = vals else {
        return false;
    };
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
    variance > 1e-9
}

/// A candidate joint `(rho, J)` pair surviving M3 Phase 3's blind co-evolution search -- unlike
/// every M2/earlier-M3 result, `rho` was NOT handed to the search; both sides were discovered
/// together. `residual` is scored against the *training* trajectory only, same caveat as
/// [`FluxDiscoveryResult`]; both `rho` and `j` pass [`is_nontrivial`] by construction.
#[derive(Debug, Clone)]
pub struct JointDiscoveryResult {
    pub rho: Expr,
    pub j: Expr,
    pub residual: f64,
}

/// Point-stencil variable names for density candidates (`rho_truth()`'s own convention, see
/// `continuity.rs`'s module docs): `u_l, u_c, u_r` (left/center/right position), `v_c` (center
/// velocity).
const RHO_STENCIL_NAMES: [&str; 4] = ["u_l", "u_c", "u_r", "v_c"];

/// M3 Phase 3: blind joint discovery of BOTH the local density `rho` and its matching flux `J`
/// -- neither is handed to the search (contrast every M2 sub-experiment and M3 Phase 1/2, which
/// all fixed `rho = rho_truth()`). This is the problem M3 was originally named for.
///
/// Co-evolves two populations (`pop_rho` over [`RHO_STENCIL_NAMES`], `pop_j` over
/// [`BOND_VAR_NAMES`]) using the same "pair against the other side's current best" mechanism
/// [`discover_flux_factorized`] already uses for its two multiplicative factors -- reused here
/// for `rho` vs. `J` instead of factor `A` vs. factor `B`. Each side is bred via
/// [`breed_generation_seeded`] (not the raw `breed_generation`, which is hardcoded to
/// [`BOND_VAR_NAMES`]-only generators -- wrong variable set for the `rho` population entirely).
///
/// **Cannot** reuse [`discover_flux_factorized`]'s `Const(1.0)/Const(1.0)` initial seed: with
/// `rho` free, that pairing scores a *perfect* residual (`0 = 0`) by construction (see
/// [`is_nontrivial`]'s doc comment for the algebra), so every candidate must pass a hard
/// nontriviality gate, and the initial "best" pairing is seeded from the populations' own first
/// (guaranteed-nontrivial-by-construction) draws instead.
///
/// Explicitly does **not** attempt full gauge-fixing for the larger joint gauge group (`rho' =
/// rho + Ġ, J' = J - flux(G)` for arbitrary `G`) -- only [`gauge_fix_flux`]'s constant-shift
/// removal on `J` is applied, an honest, stated limitation. A discovered pair that cross-
/// validates but doesn't numerically match `rho_truth()`/`j_truth()` may be a genuinely
/// different, still-valid solution rather than a failure -- report accordingly, don't assume it
/// must equal the known answer.
pub fn discover_joint_density_and_flux(
    rhs: fn(&[f64], f64) -> Vec<f64>,
    initial_state: &[f64],
    n: usize,
    config: &RegressorConfig,
    t_max: f64,
    dt: f64,
) -> Vec<JointDiscoveryResult> {
    let Some(trajectory) = sampled_trajectory(rhs, initial_state, t_max, dt) else {
        return Vec::new();
    };

    let pop_size = config.population_size.max(8);
    let generations = config.generations;
    let max_depth = config.max_depth.min(3);
    let max_complexity = config.max_complexity;
    let mut rng = config.seed;

    // Nontriviality probes: a small, fixed set of stencil-value points, deliberately
    // independent of the training trajectory -- this guard measures a candidate's own
    // functional shape, not an artifact of one specific trajectory's numeric range.
    let mut make_probes = |names_len: usize| -> Vec<Vec<f64>> {
        (0..6)
            .map(|_| {
                (0..names_len)
                    .map(|_| {
                        rng = lcg_step(rng);
                        -2.0 + 4.0 * ((rng >> 32) as f64 / u32::MAX as f64)
                    })
                    .collect()
            })
            .collect()
    };
    let rho_probes = make_probes(RHO_STENCIL_NAMES.len());
    let j_probes = make_probes(BOND_VAR_NAMES.len());

    let pair_score = |rho: &Expr, j: &Expr| -> Option<f64> {
        let c = rho.complexity() + j.complexity();
        if !(2..=max_complexity).contains(&c) {
            return None;
        }
        if !is_nontrivial(rho, &RHO_STENCIL_NAMES, &rho_probes) {
            return None;
        }
        if !is_nontrivial(j, &BOND_VAR_NAMES, &j_probes) {
            return None;
        }
        let fixed_j = gauge_fix_flux(j);
        let residual = discrete_continuity_residual(rho, &fixed_j, &trajectory, n);
        if residual.is_finite() && residual < 1e10 {
            Some(residual)
        } else {
            None
        }
    };

    let fresh_rho = |rng: &mut u64| random_density_motif_expr(rng, &RHO_STENCIL_NAMES, max_depth);
    let fresh_j = |rng: &mut u64| random_motif_expr(rng, &BOND_VAR_NAMES, max_depth);

    let mut pop_rho: Vec<Expr> = (0..pop_size).map(|_| fresh_rho(&mut rng)).collect();
    let mut pop_j: Vec<Expr> = (0..pop_size).map(|_| fresh_j(&mut rng)).collect();
    let mut best_rho = pop_rho[0].clone();
    let mut best_j = pop_j[0].clone();
    let mut best_pair_residual = pair_score(&best_rho, &best_j).unwrap_or(f64::INFINITY);

    for _gen in 0..generations {
        let fit_rho: Vec<(usize, Option<f64>)> = pop_rho
            .iter()
            .enumerate()
            .map(|(i, r)| (i, pair_score(r, &best_j)))
            .collect();
        if let Some(&(idx, Some(r))) = fit_rho
            .iter()
            .filter(|(_, s)| s.is_some())
            .min_by(|x, y| cmp_scores(x.1, y.1))
            .filter(|&&(_, r_opt)| r_opt.is_some_and(|val| val < best_pair_residual))
        {
            best_rho = pop_rho[idx].clone();
            best_pair_residual = r;
        }

        let fit_j: Vec<(usize, Option<f64>)> = pop_j
            .iter()
            .enumerate()
            .map(|(i, j)| (i, pair_score(&best_rho, j)))
            .collect();
        if let Some(&(idx, Some(r))) = fit_j
            .iter()
            .filter(|(_, s)| s.is_some())
            .min_by(|x, y| cmp_scores(x.1, y.1))
            .filter(|&&(_, r_opt)| r_opt.is_some_and(|val| val < best_pair_residual))
        {
            best_j = pop_j[idx].clone();
            best_pair_residual = r;
        }

        pop_rho = breed_generation_seeded(&pop_rho, &fit_rho, pop_size, fresh_rho, &mut rng);
        pop_j = breed_generation_seeded(&pop_j, &fit_j, pop_size, fresh_j, &mut rng);
    }

    let mut candidates: Vec<(Expr, Expr, f64)> = Vec::new();
    for r in pop_rho.iter().take(20) {
        if let Some(residual) = pair_score(r, &best_j) {
            candidates.push((r.clone(), best_j.clone(), residual));
        }
    }
    for j in pop_j.iter().take(20) {
        if let Some(residual) = pair_score(&best_rho, j) {
            candidates.push((best_rho.clone(), j.clone(), residual));
        }
    }
    if let Some(residual) = pair_score(&best_rho, &best_j) {
        candidates.push((best_rho.clone(), best_j.clone(), residual));
    }
    candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut results = Vec::new();
    for (rho, j, residual) in candidates {
        let key = format!("{rho} | {j}");
        if !seen.insert(key) {
            continue;
        }
        results.push(JointDiscoveryResult { rho, j, residual });
        if results.len() >= 10 {
            break;
        }
    }
    results
}

// ---------------------------------------------------------------------
// M2.4a: behavioral duplicate suppression. See the module-level plan doc
// (or `pde_wave_stage_b.rs`'s M2.4a tests) for the full three-arm design.
// `flux_discovery.rs` stays generic over the physical system (matching
// `discover_flux_factorized`'s own `rhs`/`initial_state` parameters) -- the
// probe bank is supplied by the caller, not hardcoded here, since building
// it (trajectory-sampled points) needs `symthaea-physics-bridge`'s
// `wave_trajectory_n3`, which depends on this crate, not the other way
// around.
// ---------------------------------------------------------------------

/// Evaluate `expr` at every state in `probes` (each `[u_c, u_r, v_c, v_r]`),
/// returning the raw response vector -- a "behavioral fingerprint" for
/// semantic duplicate detection. `None` if any probe evaluates to a
/// non-finite value, so such a candidate is excluded from the semantic
/// archive entirely (it's still handled normally by the physics-fitness
/// `score()`/`None` path).
pub fn behavioral_fingerprint(expr: &Expr, probes: &[[f64; 4]]) -> Option<Vec<f32>> {
    let mut out = Vec::with_capacity(probes.len());
    for probe in probes {
        let bindings: Vec<(&str, f64)> = BOND_VAR_NAMES.iter().copied().zip(*probe).collect();
        let v = expr.eval(&bindings);
        if !v.is_finite() {
            return None;
        }
        out.push(v as f32);
    }
    Some(out)
}

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

/// Cosine similarity of two behavioral fingerprints, after independently
/// L2-normalizing both -- the "direct-vector" arm of M2.4a's duplicate-
/// suppression comparison (Arm 2). Both fingerprints must be the same
/// length (same probe bank).
pub fn vector_similarity(a: &[f32], b: &[f32]) -> f32 {
    let na = l2_normalize(a);
    let nb = l2_normalize(b);
    na.iter().zip(nb.iter()).map(|(x, y)| x * y).sum()
}

/// Encode a behavioral fingerprint into a [`ContinuousHV`], using the same
/// basis-scaled-bundle-then-normalize pattern as `symthaea-physics-bridge`'s
/// `DimensionalEncoder` (`dimensional.rs`): one orthogonal basis vector per
/// probe, scaled by the candidate's response at that probe, summed, then
/// L2-normalized. `basis` must have `fingerprint.len()` orthogonal vectors,
/// built once via [`hdc_probe_basis`] and reused across every candidate in
/// a run (the encoding is only comparable if every candidate uses the same
/// basis).
///
/// **Important**: this encoding, applied directly to a raw fingerprint, is
/// a linear *isometry* -- `ContinuousHV::orthogonal_set`'s basis is
/// genuinely orthonormal (Gram-Schmidt + unit-normalize), so cosine
/// similarity of two encodings *exactly* equals cosine similarity of the
/// two raw fingerprints (see [`vector_similarity`]). Comparing this
/// directly against `vector_similarity` is a null test by construction --
/// confirmed empirically in the M2.4a run (`pde_wave_stage_b.rs`'s
/// `m24a_dedup_three_arm_comparison_n3`: Arms 2 and 3 produced byte-
/// identical results on all 5 seeds). See
/// `feedback_hdc_orthonormal_encoding_is_isometry.md` in the session
/// memory. For a genuinely different (non-isometric) representation, feed
/// this a *quantized* fingerprint via [`quantize_fingerprint`] instead of
/// the raw one -- quantization is lossy, so the isometry no longer holds.
pub fn hdc_fingerprint(fingerprint: &[f32], basis: &[ContinuousHV]) -> ContinuousHV {
    assert_eq!(
        fingerprint.len(),
        basis.len(),
        "fingerprint/basis length mismatch"
    );
    let dim = basis.first().map(|b| b.dim()).unwrap_or(HDC_DIMENSION);
    let mut result = ContinuousHV::zero(dim);
    for (value, basis_vec) in fingerprint.iter().zip(basis.iter()) {
        result = result.add(&basis_vec.scale(*value));
    }
    result.normalize()
}

/// Build the orthogonal basis [`hdc_fingerprint`] needs, one vector per
/// probe. `seed` must be fixed and reused for the whole comparison run (a
/// different seed per candidate would make encodings incomparable).
pub fn hdc_probe_basis(probe_count: usize, seed: u64) -> Vec<ContinuousHV> {
    ContinuousHV::orthogonal_set(HDC_DIMENSION, probe_count, seed)
}

/// Quantize a fingerprint to a coarse grid of width `bucket_width` --
/// a genuinely lossy, non-isometric preprocessing step. Two distinct raw
/// fingerprints can quantize to the *same* bucketed values (information is
/// destroyed), so feeding this into [`hdc_fingerprint`] no longer preserves
/// [`vector_similarity`] exactly, unlike encoding the raw fingerprint
/// directly (see that function's docs for why the naive version was a null
/// test). Coarser `bucket_width` means more information loss (more
/// candidates collapse to the same bucketed fingerprint) but also more risk
/// of merging genuinely different candidates -- a real tradeoff, not tuned
/// away here.
pub fn quantize_fingerprint(fingerprint: &[f32], bucket_width: f32) -> Vec<f32> {
    fingerprint
        .iter()
        .map(|&v| (v / bucket_width).round() * bucket_width)
        .collect()
}

/// Elite selection + breeding for one generation, shared by
/// [`discover_flux_given_density`] and [`discover_flux_factorized`]. Elites
/// are only ever drawn from validly (`Some`) scored candidates -- if too
/// few exist (common in early generations), falls back to fresh random
/// material rather than breeding from a rejected individual.
/// `RANDOM_INJECTION_FRACTION` of the tail is always fresh random rather
/// than elite-bred offspring, to counter the genealogical collapse an
/// elites-only scheme is prone to.
///
/// `polynomial_only`: if set, every offspring is checked with
/// [`is_polynomial`] and regenerated from the polynomial-only grammar on
/// violation (see that function's docs for why this check is needed, not
/// just grammar restriction at population-init time).
fn breed_generation(
    population: &[Expr],
    fitnesses: &[(usize, Option<f64>)],
    pop_size: usize,
    max_depth: usize,
    exclude_trig: bool,
    polynomial_only: bool,
    rng: &mut u64,
) -> Vec<Expr> {
    let fresh = |rng: &mut u64| -> Expr {
        if polynomial_only {
            fresh_random_polynomial_bond_expr(rng, max_depth)
        } else {
            fresh_random_bond_expr(rng, max_depth, exclude_trig)
        }
    };

    let mut sorted = fitnesses.to_vec();
    sorted.sort_by(|a, b| cmp_scores(a.1, b.1));

    let elite_count = (pop_size / 5).max(2).min(pop_size);
    let mut elites: Vec<Expr> = sorted
        .iter()
        .filter(|(_, s)| s.is_some())
        .take(elite_count)
        .map(|(i, _)| population[*i].clone())
        .collect();
    while elites.len() < elite_count {
        elites.push(fresh(rng));
    }

    let injection_count = ((pop_size as f64) * RANDOM_INJECTION_FRACTION).round() as usize;
    let mut next_gen: Vec<Expr> = elites.clone();
    while next_gen.len() < pop_size {
        if pop_size - next_gen.len() <= injection_count {
            next_gen.push(fresh(rng));
            continue;
        }
        *rng = lcg_step(*rng);
        let a = &elites[*rng as usize % elites.len()];
        *rng = lcg_step(*rng);
        let child = if *rng % 2 == 0 {
            *rng = lcg_step(*rng);
            let b = &elites[*rng as usize % elites.len()];
            crossover(a, b, rng)
        } else {
            a.mutate(rng, 0)
        };
        let child = if polynomial_only && !is_polynomial(&child) {
            fresh(rng)
        } else {
            child
        };
        next_gen.push(child);
    }
    next_gen
}

fn sampled_trajectory(
    rhs: fn(&[f64], f64) -> Vec<f64>,
    initial_state: &[f64],
    t_max: f64,
    dt: f64,
) -> Option<Vec<Vec<f64>>> {
    let (_t, states) = rk45_trajectory(rhs, initial_state, t_max, dt);
    let n_samples = 200.min(states.len());
    if n_samples < 20 {
        return None;
    }
    let step = (states.len() / n_samples.max(1)).max(1);
    Some(
        states
            .iter()
            .step_by(step)
            .take(n_samples)
            .cloned()
            .collect(),
    )
}

/// GP search for a flux `J` completing the discrete continuity equation for
/// a *fixed, known* density `rho`. Reuses `RegressorConfig` for search
/// parameters (population/generations/depth/complexity/seed/exclude_trig),
/// matching Stage A's calling convention.
///
/// Each generation, `RANDOM_INJECTION_FRACTION` of the non-elite population
/// is replaced with fresh random individuals rather than pure elite-bred
/// offspring, to counter the genealogical collapse an elites-only breeding
/// scheme is prone to (converge around a few constants/unary forms early,
/// then never generate the material a multi-variable product needs).
pub fn discover_flux_given_density(
    rho: &Expr,
    rhs: fn(&[f64], f64) -> Vec<f64>,
    initial_state: &[f64],
    n: usize,
    config: &RegressorConfig,
    t_max: f64,
    dt: f64,
) -> Vec<FluxDiscoveryResult> {
    let max_depth = config.max_depth.min(3);
    let exclude_trig = config.exclude_trig;
    discover_flux_given_density_seeded(rho, rhs, initial_state, n, config, t_max, dt, move |rng| {
        fresh_random_bond_expr(rng, max_depth, exclude_trig)
    })
}

/// Like [`breed_generation`], but takes an injectable `fresh` generator instead of branching
/// on `polynomial_only` -- lets [`discover_flux_given_density_seeded`] swap in
/// [`random_motif_expr`] (or any other generator) for M3 Phase 2 without touching the
/// established `breed_generation`/M2 code at all. No polynomial-specific post-crossover check
/// (that's specific to the `polynomial_only` path this function doesn't serve). Uses
/// [`rand_index_high_bits`] throughout for its own index picks, per this module's own
/// LCG-low-bits lesson (see that function's doc comment) -- new code, not a retrofit of
/// `breed_generation`'s existing raw low-bit picks, which stay untouched.
fn breed_generation_seeded(
    population: &[Expr],
    fitnesses: &[(usize, Option<f64>)],
    pop_size: usize,
    fresh: impl Fn(&mut u64) -> Expr + Copy,
    rng: &mut u64,
) -> Vec<Expr> {
    let mut sorted = fitnesses.to_vec();
    sorted.sort_by(|a, b| cmp_scores(a.1, b.1));

    let elite_count = (pop_size / 5).max(2).min(pop_size);
    let mut elites: Vec<Expr> = sorted
        .iter()
        .filter(|(_, s)| s.is_some())
        .take(elite_count)
        .map(|(i, _)| population[*i].clone())
        .collect();
    while elites.len() < elite_count {
        elites.push(fresh(rng));
    }

    let injection_count = ((pop_size as f64) * RANDOM_INJECTION_FRACTION).round() as usize;
    let mut next_gen: Vec<Expr> = elites.clone();
    while next_gen.len() < pop_size {
        if pop_size - next_gen.len() <= injection_count {
            next_gen.push(fresh(rng));
            continue;
        }
        let a = &elites[rand_index_high_bits(rng, elites.len())];
        let child = if rand_index_high_bits(rng, 2) == 0 {
            let b = &elites[rand_index_high_bits(rng, elites.len())];
            crossover(a, b, rng)
        } else {
            a.mutate(rng, 0)
        };
        next_gen.push(child);
    }
    next_gen
}

/// [`discover_flux_given_density`], generalized to take an injectable generator `fresh`
/// instead of hardcoding [`fresh_random_bond_expr`] -- lets M3 Phase 2 test whether a
/// different generator (e.g. [`random_motif_expr`]) changes actual evolutionary discovery
/// success, not just raw reachability (Phase 1). `discover_flux_given_density` itself is now
/// a thin wrapper around this with the original generator, so its behavior is unchanged --
/// verified by the existing `flux_discovery` test suite staying green after this refactor,
/// not just by inspection.
pub fn discover_flux_given_density_seeded(
    rho: &Expr,
    rhs: fn(&[f64], f64) -> Vec<f64>,
    initial_state: &[f64],
    n: usize,
    config: &RegressorConfig,
    t_max: f64,
    dt: f64,
    fresh: impl Fn(&mut u64) -> Expr + Copy,
) -> Vec<FluxDiscoveryResult> {
    let Some(trajectory) = sampled_trajectory(rhs, initial_state, t_max, dt) else {
        return Vec::new();
    };

    let pop_size = config.population_size.max(8);
    let generations = config.generations;
    let max_complexity = config.max_complexity;
    let mut rng = config.seed;

    let mut population: Vec<Expr> = (0..pop_size).map(|_| fresh(&mut rng)).collect();

    for _gen in 0..generations {
        let fitnesses: Vec<(usize, Option<f64>)> = population
            .iter()
            .enumerate()
            .map(|(i, e)| (i, score(rho, e, &trajectory, n, max_complexity)))
            .collect();
        population = breed_generation_seeded(&population, &fitnesses, pop_size, fresh, &mut rng);
    }

    let mut scored: Vec<(Expr, f64)> = population
        .into_iter()
        .filter_map(|e| score(rho, &e, &trajectory, n, max_complexity).map(|s| (e, s)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut results = Vec::new();
    for (expr, residual) in scored {
        let fixed = gauge_fix_flux(&expr);
        let formula_str = format!("{fixed}");
        if !seen.insert(formula_str.clone()) {
            continue;
        }
        results.push(FluxDiscoveryResult {
            formula: fixed,
            formula_str,
            residual,
        });
        if results.len() >= 10 {
            break;
        }
    }
    results
}

/// A candidate factorized flux `J = A * B` surviving
/// [`discover_flux_factorized`]'s search. `residual` is scored against the
/// *training* trajectory only, same caveat as [`FluxDiscoveryResult`].
#[derive(Debug, Clone)]
pub struct FactorizedFluxResult {
    pub factor_a: Expr,
    pub factor_b: Expr,
    pub product_str: String,
    pub residual: f64,
}

/// M2.2: instead of one population searching the full 4-variable product
/// tree at once (M2.1 measured this to have almost no reachable path to the
/// answer -- 0.022% of random trees contain both needed motifs, and every
/// single-factor/partial candidate scores worse than doing nothing), evolve
/// two SEPARATE populations -- factor `A` and factor `B` -- and score each
/// side by pairing against the other side's current best representative.
/// `J = A * B`. Tests whether decomposing the search into two independently
/// -evolved factors solves the structural credit-assignment problem the
/// single-tree search couldn't, rather than testing whether raw compute or
/// diversity alone can.
///
/// Deliberately NOT told which stencil variables belong in which factor --
/// both `A` and `B` search over the full bond-stencil names. A success here
/// is a real result about search topology, not a hand-fed decomposition of
/// the answer.
///
/// `polynomial_only`: M2.3 -- restrict both factors' grammar to
/// `Var`/`Const`/`Add`/`Sub`/`Mul` (matches `j_truth()`'s actual grammar
/// exactly, see [`fresh_random_polynomial_bond_expr`]), testing whether
/// trig/log/div/pow "dilution" of the unrestricted grammar was the
/// dominant obstacle. This is not unfair answer-leakage: the hypothesis is
/// "a physics-compatible polynomial language improves discovery efficiency
/// when the target system's dynamics are polynomial," not a seed shaped
/// like the answer's specific structure.
pub fn discover_flux_factorized(
    rho: &Expr,
    rhs: fn(&[f64], f64) -> Vec<f64>,
    initial_state: &[f64],
    n: usize,
    config: &RegressorConfig,
    t_max: f64,
    dt: f64,
    polynomial_only: bool,
) -> Vec<FactorizedFluxResult> {
    let Some(trajectory) = sampled_trajectory(rhs, initial_state, t_max, dt) else {
        return Vec::new();
    };

    let pop_size = config.population_size.max(8);
    let generations = config.generations;
    let max_depth = config.max_depth.min(3);
    let max_complexity = config.max_complexity;
    let exclude_trig = config.exclude_trig;
    let mut rng = config.seed;

    let fresh = |rng: &mut u64| -> Expr {
        if polynomial_only {
            fresh_random_polynomial_bond_expr(rng, max_depth)
        } else {
            fresh_random_bond_expr(rng, max_depth, exclude_trig)
        }
    };

    let pair_score = |a: &Expr, b: &Expr| -> Option<f64> {
        let product = Expr::BinOp(BinOp::Mul, Box::new(a.clone()), Box::new(b.clone()));
        let c = product.complexity();
        if !(2..=max_complexity).contains(&c) {
            return None;
        }
        let fixed = gauge_fix_flux(&product);
        let residual = discrete_continuity_residual(rho, &fixed, &trajectory, n);
        if residual.is_finite() && residual < 1e10 {
            Some(residual)
        } else {
            None
        }
    };

    let mut pop_a: Vec<Expr> = (0..pop_size).map(|_| fresh(&mut rng)).collect();
    let mut pop_b: Vec<Expr> = (0..pop_size).map(|_| fresh(&mut rng)).collect();
    // Init to Const(1.0): early pairing is well-defined (product = the
    // other factor alone), giving each side an initial gradient to climb.
    let mut best_a = Expr::Const(1.0);
    let mut best_b = Expr::Const(1.0);
    let mut best_pair_residual = f64::INFINITY;

    for _gen in 0..generations {
        let fit_a: Vec<(usize, Option<f64>)> = pop_a
            .iter()
            .enumerate()
            .map(|(i, a)| (i, pair_score(a, &best_b)))
            .collect();
        if let Some(&(idx, Some(r))) = fit_a
            .iter()
            .filter(|(_, s)| s.is_some())
            .min_by(|x, y| cmp_scores(x.1, y.1))
            .filter(|&&(_, r_opt)| r_opt.is_some_and(|val| val < best_pair_residual))
        {
            best_a = pop_a[idx].clone();
            best_pair_residual = r;
        }

        let fit_b: Vec<(usize, Option<f64>)> = pop_b
            .iter()
            .enumerate()
            .map(|(i, b)| (i, pair_score(&best_a, b)))
            .collect();
        if let Some(&(idx, Some(r))) = fit_b
            .iter()
            .filter(|(_, s)| s.is_some())
            .min_by(|x, y| cmp_scores(x.1, y.1))
            .filter(|&&(_, r_opt)| r_opt.is_some_and(|val| val < best_pair_residual))
        {
            best_b = pop_b[idx].clone();
            best_pair_residual = r;
        }

        pop_a = breed_generation(
            &pop_a,
            &fit_a,
            pop_size,
            max_depth,
            exclude_trig,
            polynomial_only,
            &mut rng,
        );
        pop_b = breed_generation(
            &pop_b,
            &fit_b,
            pop_size,
            max_depth,
            exclude_trig,
            polynomial_only,
            &mut rng,
        );
    }

    let mut candidates: Vec<(Expr, Expr, f64)> = Vec::new();
    for a in pop_a.iter().take(20) {
        if let Some(r) = pair_score(a, &best_b) {
            candidates.push((a.clone(), best_b.clone(), r));
        }
    }
    for b in pop_b.iter().take(20) {
        if let Some(r) = pair_score(&best_a, b) {
            candidates.push((best_a.clone(), b.clone(), r));
        }
    }
    if let Some(r) = pair_score(&best_a, &best_b) {
        candidates.push((best_a.clone(), best_b.clone(), r));
    }
    candidates.sort_by(|x, y| x.2.partial_cmp(&y.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut results = Vec::new();
    for (a, b, residual) in candidates {
        let product_str = format!("({a}) * ({b})");
        if !seen.insert(product_str.clone()) {
            continue;
        }
        results.push(FactorizedFluxResult {
            factor_a: a,
            factor_b: b,
            product_str,
            residual,
        });
        if results.len() >= 10 {
            break;
        }
    }
    results
}

// ---------------------------------------------------------------------
// M2.5: target-blind fitness shaping. See `shape_calibrated_residual`
// (continuity.rs) for the scoring itself; this section is the search
// driver built on top of it, kept as its own function (not a modification
// of `discover_flux_factorized`) following this arc's established pattern
// of never risking a proven driver -- every M2.x variant gets its own
// function that reuses shared helpers (`breed_generation`) instead.
// ---------------------------------------------------------------------

/// A candidate factorized flux surviving [`discover_flux_factorized_shaped`],
/// with the analytically-fit scale already absorbed in. `factor_a`/
/// `factor_b` are the *raw*, unscaled evolved factors (what the search
/// actually explored); `scaled_product` is `alpha * factor_a * factor_b`,
/// gauge-fixed -- the expression callers should cross-validate against
/// held-out trajectories (via [`discrete_continuity_residual`]), since the
/// calibrated fitness only ever measures shape+scale against the *training*
/// trajectory.
#[derive(Debug, Clone)]
pub struct ShapedFluxResult {
    pub factor_a: Expr,
    pub factor_b: Expr,
    pub alpha: f64,
    pub alignment: f64,
    pub calibrated_residual: f64,
    pub scaled_product: Expr,
    pub scaled_product_str: String,
}

/// Per-generation telemetry from [`discover_flux_factorized_shaped`], for
/// tracking whether the shaped fitness changes *when* and *how durably*
/// genuine structural motifs (matching `contains_pattern`'s check in the
/// test module) survive in the population -- continuity with M2.1/M2.3's
/// own motif-incidence observations, not a targeted reward (the fitness
/// itself never inspects for these patterns; this is read-only telemetry
/// computed after the fact from whichever factors the generation already
/// selected as best).
#[derive(Debug, Clone, Copy)]
pub struct GenerationSnapshot {
    pub generation: usize,
    pub best_pair_fitness: f64,
    pub best_alignment: f64,
    /// `true` if `best_a` (this generation's) is algebraically the
    /// displacement pattern `u_r - u_c` (or `u_c - u_r`).
    pub best_a_has_displacement_motif: bool,
    /// `true` if `best_b` (this generation's) is algebraically the
    /// velocity-sum pattern `v_c + v_r`.
    pub best_b_has_velocity_motif: bool,
}

fn is_displacement_motif(e: &Expr) -> bool {
    matches!(
        e,
        Expr::BinOp(BinOp::Sub, l, r)
            if matches!((&**l, &**r), (Expr::Var(a), Expr::Var(b))
                if (a == "u_r" && b == "u_c") || (a == "u_c" && b == "u_r"))
    )
}

fn is_velocity_sum_motif(e: &Expr) -> bool {
    matches!(
        e,
        Expr::BinOp(BinOp::Add, l, r)
            if matches!((&**l, &**r), (Expr::Var(a), Expr::Var(b))
                if (a == "v_c" && b == "v_r") || (a == "v_r" && b == "v_c"))
    )
}

/// M2.5: identical search architecture to [`discover_flux_factorized`]
/// (two independently-evolved factors `A`/`B`, `J = A*B`, paired against
/// the partner side's current best, `breed_generation`'s exact
/// elite/injection/crossover/mutation mechanics) with **only the fitness
/// function changed**: pair scoring uses
/// [`shape_calibrated_residual`](super::shape_calibrated_residual) instead
/// of the raw [`discrete_continuity_residual`]. Grammar, factorization,
/// population/generation budget, mutation/crossover/selection,
/// canonicalization (gauge-fixing), and training-trajectory sampling are
/// all identical to `discover_flux_factorized` -- the isolation this arc's
/// prior comparisons (M2.4a) established as necessary to attribute any
/// difference to the fitness function alone.
///
/// **Target-blind**: `shape_calibrated_residual` never references
/// `j_truth` or any other known-correct flux -- it only uses the
/// physically observed continuity target from `rho` and the system's own
/// dynamics. No distance-to-known-answer term is used in this search.
///
/// Returns `(results, snapshots)`: `results` mirrors
/// `discover_flux_factorized`'s output shape (top candidates, deduped,
/// capped at 10) but as [`ShapedFluxResult`] (scale absorbed, ready for
/// held-out cross-validation); `snapshots` is one [`GenerationSnapshot`]
/// per generation, for the "first generation with a genuine cross-term" /
/// "persistence" / "alignment over generations" tracking the M2.5 design
/// calls for.
#[allow(clippy::too_many_arguments)]
pub fn discover_flux_factorized_shaped(
    rho: &Expr,
    rhs: fn(&[f64], f64) -> Vec<f64>,
    initial_state: &[f64],
    n: usize,
    config: &RegressorConfig,
    t_max: f64,
    dt: f64,
    polynomial_only: bool,
) -> (Vec<ShapedFluxResult>, Vec<GenerationSnapshot>) {
    let Some(trajectory) = sampled_trajectory(rhs, initial_state, t_max, dt) else {
        return (Vec::new(), Vec::new());
    };

    let pop_size = config.population_size.max(8);
    let generations = config.generations;
    let max_depth = config.max_depth.min(3);
    let max_complexity = config.max_complexity;
    let exclude_trig = config.exclude_trig;
    let mut rng = config.seed;

    let fresh = |rng: &mut u64| -> Expr {
        if polynomial_only {
            fresh_random_polynomial_bond_expr(rng, max_depth)
        } else {
            fresh_random_bond_expr(rng, max_depth, exclude_trig)
        }
    };

    // "Small is good", matching every other fitness in this module: reject
    // out-of-band complexity on the raw (unscaled) product -- amplitude
    // scale is analytically calibrated below, so it must never gate on
    // tree-shape complexity, only shape does.
    let pair_score = |a: &Expr, b: &Expr| -> Option<(f64, ShapeCalibration)> {
        let product = Expr::BinOp(BinOp::Mul, Box::new(a.clone()), Box::new(b.clone()));
        let c = product.complexity();
        if !(2..=max_complexity).contains(&c) {
            return None;
        }
        let fixed = gauge_fix_flux(&product);
        let cal = shape_calibrated_residual(rho, &fixed, &trajectory, n)?;
        Some((cal.calibrated_residual, cal))
    };

    let mut pop_a: Vec<Expr> = (0..pop_size).map(|_| fresh(&mut rng)).collect();
    let mut pop_b: Vec<Expr> = (0..pop_size).map(|_| fresh(&mut rng)).collect();
    let mut best_a = Expr::Const(1.0);
    let mut best_b = Expr::Const(1.0);
    let mut best_pair_fitness = f64::INFINITY;
    let mut best_cal: Option<ShapeCalibration> = None;
    let mut snapshots = Vec::with_capacity(generations);

    for generation_idx in 0..generations {
        let fit_a: Vec<(usize, Option<f64>)> = pop_a
            .iter()
            .enumerate()
            .map(|(i, a)| (i, pair_score(a, &best_b).map(|(f, _)| f)))
            .collect();
        if let Some(&(idx, Some(f))) = fit_a
            .iter()
            .filter(|(_, s)| s.is_some())
            .min_by(|x, y| cmp_scores(x.1, y.1))
            .filter(|&&(_, f_opt)| f_opt.is_some_and(|val| val < best_pair_fitness))
        {
            best_a = pop_a[idx].clone();
            best_pair_fitness = f;
            best_cal = pair_score(&best_a, &best_b).map(|(_, c)| c);
        }

        let fit_b: Vec<(usize, Option<f64>)> = pop_b
            .iter()
            .enumerate()
            .map(|(i, b)| (i, pair_score(&best_a, b).map(|(f, _)| f)))
            .collect();
        if let Some(&(idx, Some(f))) = fit_b
            .iter()
            .filter(|(_, s)| s.is_some())
            .min_by(|x, y| cmp_scores(x.1, y.1))
            .filter(|&&(_, f_opt)| f_opt.is_some_and(|val| val < best_pair_fitness))
        {
            best_b = pop_b[idx].clone();
            best_pair_fitness = f;
            best_cal = pair_score(&best_a, &best_b).map(|(_, c)| c);
        }

        snapshots.push(GenerationSnapshot {
            generation: generation_idx,
            best_pair_fitness,
            best_alignment: best_cal.map(|c| c.alignment).unwrap_or(0.0),
            best_a_has_displacement_motif: is_displacement_motif(&best_a),
            best_b_has_velocity_motif: is_velocity_sum_motif(&best_b),
        });

        pop_a = breed_generation(
            &pop_a,
            &fit_a,
            pop_size,
            max_depth,
            exclude_trig,
            polynomial_only,
            &mut rng,
        );
        pop_b = breed_generation(
            &pop_b,
            &fit_b,
            pop_size,
            max_depth,
            exclude_trig,
            polynomial_only,
            &mut rng,
        );
    }

    let finalize = |a: &Expr, b: &Expr| -> Option<ShapedFluxResult> {
        let (_, cal) = pair_score(a, b)?;
        let product = Expr::BinOp(BinOp::Mul, Box::new(a.clone()), Box::new(b.clone()));
        let scaled = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Const(cal.alpha)),
            Box::new(product),
        );
        let scaled_fixed = gauge_fix_flux(&scaled);
        Some(ShapedFluxResult {
            factor_a: a.clone(),
            factor_b: b.clone(),
            alpha: cal.alpha,
            alignment: cal.alignment,
            calibrated_residual: cal.calibrated_residual,
            scaled_product_str: format!("{scaled_fixed}"),
            scaled_product: scaled_fixed,
        })
    };

    let mut candidates: Vec<ShapedFluxResult> = Vec::new();
    for a in pop_a.iter().take(20) {
        if let Some(r) = finalize(a, &best_b) {
            candidates.push(r);
        }
    }
    for b in pop_b.iter().take(20) {
        if let Some(r) = finalize(&best_a, b) {
            candidates.push(r);
        }
    }
    if let Some(r) = finalize(&best_a, &best_b) {
        candidates.push(r);
    }
    candidates.sort_by(|x, y| {
        x.calibrated_residual
            .partial_cmp(&y.calibrated_residual)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut results = Vec::new();
    for cand in candidates {
        if !seen.insert(cand.scaled_product_str.clone()) {
            continue;
        }
        results.push(cand);
        if results.len() >= 10 {
            break;
        }
    }
    (results, snapshots)
}

/// Identical search to [`discover_flux_factorized`] -- same selection
/// criterion (raw [`discrete_continuity_residual`]), same everything --
/// except it also records a [`GenerationSnapshot`] per generation, for a
/// fair per-generation comparison against
/// [`discover_flux_factorized_shaped`] (M2.5's paired evolutionary
/// comparison needs both arms' motif-incidence/alignment trajectories, not
/// just final recovery).
///
/// The `alignment` field on each snapshot is a **pure diagnostic**: it's
/// computed via `shape_calibrated_residual` purely for telemetry, exactly
/// like the shaped arm reports it, but it plays no role in this function's
/// actual selection criterion (still the raw residual) -- so comparing the
/// two arms' `snapshots` isolates the effect of the fitness function on
/// what evolves, without changing what's being measured.
#[allow(clippy::too_many_arguments)]
pub fn discover_flux_factorized_with_snapshots(
    rho: &Expr,
    rhs: fn(&[f64], f64) -> Vec<f64>,
    initial_state: &[f64],
    n: usize,
    config: &RegressorConfig,
    t_max: f64,
    dt: f64,
    polynomial_only: bool,
) -> (Vec<FactorizedFluxResult>, Vec<GenerationSnapshot>) {
    let Some(trajectory) = sampled_trajectory(rhs, initial_state, t_max, dt) else {
        return (Vec::new(), Vec::new());
    };

    let pop_size = config.population_size.max(8);
    let generations = config.generations;
    let max_depth = config.max_depth.min(3);
    let max_complexity = config.max_complexity;
    let exclude_trig = config.exclude_trig;
    let mut rng = config.seed;

    let fresh = |rng: &mut u64| -> Expr {
        if polynomial_only {
            fresh_random_polynomial_bond_expr(rng, max_depth)
        } else {
            fresh_random_bond_expr(rng, max_depth, exclude_trig)
        }
    };

    let pair_score = |a: &Expr, b: &Expr| -> Option<f64> {
        let product = Expr::BinOp(BinOp::Mul, Box::new(a.clone()), Box::new(b.clone()));
        let c = product.complexity();
        if !(2..=max_complexity).contains(&c) {
            return None;
        }
        let fixed = gauge_fix_flux(&product);
        let residual = discrete_continuity_residual(rho, &fixed, &trajectory, n);
        if residual.is_finite() && residual < 1e10 {
            Some(residual)
        } else {
            None
        }
    };

    let diagnostic_alignment = |a: &Expr, b: &Expr| -> f64 {
        let product = Expr::BinOp(BinOp::Mul, Box::new(a.clone()), Box::new(b.clone()));
        let fixed = gauge_fix_flux(&product);
        shape_calibrated_residual(rho, &fixed, &trajectory, n)
            .map(|c| c.alignment)
            .unwrap_or(0.0)
    };

    let mut pop_a: Vec<Expr> = (0..pop_size).map(|_| fresh(&mut rng)).collect();
    let mut pop_b: Vec<Expr> = (0..pop_size).map(|_| fresh(&mut rng)).collect();
    let mut best_a = Expr::Const(1.0);
    let mut best_b = Expr::Const(1.0);
    let mut best_pair_residual = f64::INFINITY;
    let mut snapshots = Vec::with_capacity(generations);

    for generation_idx in 0..generations {
        let fit_a: Vec<(usize, Option<f64>)> = pop_a
            .iter()
            .enumerate()
            .map(|(i, a)| (i, pair_score(a, &best_b)))
            .collect();
        if let Some(&(idx, Some(r))) = fit_a
            .iter()
            .filter(|(_, s)| s.is_some())
            .min_by(|x, y| cmp_scores(x.1, y.1))
            .filter(|&&(_, r_opt)| r_opt.is_some_and(|val| val < best_pair_residual))
        {
            best_a = pop_a[idx].clone();
            best_pair_residual = r;
        }

        let fit_b: Vec<(usize, Option<f64>)> = pop_b
            .iter()
            .enumerate()
            .map(|(i, b)| (i, pair_score(&best_a, b)))
            .collect();
        if let Some(&(idx, Some(r))) = fit_b
            .iter()
            .filter(|(_, s)| s.is_some())
            .min_by(|x, y| cmp_scores(x.1, y.1))
            .filter(|&&(_, r_opt)| r_opt.is_some_and(|val| val < best_pair_residual))
        {
            best_b = pop_b[idx].clone();
            best_pair_residual = r;
        }

        snapshots.push(GenerationSnapshot {
            generation: generation_idx,
            best_pair_fitness: best_pair_residual,
            best_alignment: diagnostic_alignment(&best_a, &best_b),
            best_a_has_displacement_motif: is_displacement_motif(&best_a),
            best_b_has_velocity_motif: is_velocity_sum_motif(&best_b),
        });

        pop_a = breed_generation(
            &pop_a,
            &fit_a,
            pop_size,
            max_depth,
            exclude_trig,
            polynomial_only,
            &mut rng,
        );
        pop_b = breed_generation(
            &pop_b,
            &fit_b,
            pop_size,
            max_depth,
            exclude_trig,
            polynomial_only,
            &mut rng,
        );
    }

    let mut candidates: Vec<(Expr, Expr, f64)> = Vec::new();
    for a in pop_a.iter().take(20) {
        if let Some(r) = pair_score(a, &best_b) {
            candidates.push((a.clone(), best_b.clone(), r));
        }
    }
    for b in pop_b.iter().take(20) {
        if let Some(r) = pair_score(&best_a, b) {
            candidates.push((best_a.clone(), b.clone(), r));
        }
    }
    if let Some(r) = pair_score(&best_a, &best_b) {
        candidates.push((best_a.clone(), best_b.clone(), r));
    }
    candidates.sort_by(|x, y| x.2.partial_cmp(&y.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut results = Vec::new();
    for (a, b, residual) in candidates {
        let product_str = format!("({a}) * ({b})");
        if !seen.insert(product_str.clone()) {
            continue;
        }
        results.push(FactorizedFluxResult {
            factor_a: a,
            factor_b: b,
            product_str,
            residual,
        });
        if results.len() >= 10 {
            break;
        }
    }
    (results, snapshots)
}

/// M2.4a duplicate-suppression mode. `threshold` is a cosine-similarity
/// cutoff in `[-1, 1]`; two candidates at or above it are treated as
/// behavioral duplicates.
#[derive(Debug, Clone, Copy)]
pub enum DedupMode {
    /// Arm 1: no semantic suppression (matches `discover_flux_factorized`'s
    /// existing behavior exactly).
    None,
    /// Arm 2: direct-vector cosine similarity on the raw (L2-normalized)
    /// behavioral fingerprint.
    Vector { threshold: f32 },
    /// Arm 3 (original, flawed): `ContinuousHV`-encoded cosine similarity
    /// on the *raw* fingerprint. Kept for reproducibility of the original
    /// M2.4a run, but see [`hdc_fingerprint`]'s docs -- this is a linear
    /// isometry of Arm 2, so it is mathematically guaranteed to produce
    /// identical results to `Vector` at the same threshold. Use
    /// [`DedupMode::HdcQuantized`] for a genuine HDC-vs-vector test.
    Hdc { threshold: f32 },
    /// Arm 3' (corrected): `ContinuousHV`-encoded cosine similarity on a
    /// *quantized* fingerprint (see [`quantize_fingerprint`]) -- lossy, so
    /// no longer guaranteed to agree with `Vector`. This is the arm that
    /// actually tests whether HDC's representation differs from a raw
    /// feature vector, since `Hdc` cannot by construction.
    HdcQuantized { threshold: f32, bucket_width: f32 },
}

/// Metrics collected by [`discover_flux_factorized_with_dedup`] for one run
/// (one population -- factor A or factor B get their own).
#[derive(Debug, Clone, Default)]
pub struct DedupRunMetrics {
    /// Candidates accepted into the population across all generations
    /// (each counted once, whenever it first occupies a slot).
    pub unique_candidates_accepted: usize,
    /// Candidates rejected as behavioral duplicates of an already-accepted
    /// candidate (Arm 1 always reports 0 -- it never rejects on this
    /// basis).
    pub duplicates_rejected: usize,
    /// Among duplicate-rejections where the *new* candidate replaced the
    /// existing one, how many had residuals differing by >10% relative --
    /// a proxy for over-merging (two candidates treated as behaviorally
    /// equivalent whose actual fitness meaningfully differs). Only
    /// meaningful for Arms 2/3.
    pub false_merges: usize,
}

/// M2.4a: like [`discover_flux_factorized`], but every generation
/// overproduces a buffer of `2 * population_size` candidates (via the
/// exact same [`breed_generation`] mechanics `discover_flux_factorized`
/// uses -- this function does not modify that proven driver, it calls the
/// same shared helper with a larger output size) and filters it down to
/// `population_size` per `dedup_mode`. A behaviorally-duplicate candidate
/// replaces its match only if it scores better (lower residual) or is
/// simpler when scores tie/are both invalid.
///
/// Causal-parity note: all three arms share *identical* initial
/// populations and identical breeding mechanics: the RNG stream driving
/// crossover/mutation is only identical byte-for-byte across arms at
/// generation 0 (shared starting population). From generation 1 onward,
/// arms' populations naturally diverge because Arms 2/3 accept different
/// survivors than Arm 1 -- breeding then proceeds from each arm's own
/// (already-diverged) population, which is the correct and only coherent
/// behavior for a multi-generation stochastic search (you cannot hold a
/// population's *content* fixed across arms and still let it evolve). What
/// stays identical across arms throughout is the breeding *mechanism*
/// (same overproduction size, same elite/injection/crossover/mutation
/// logic, same RNG advancement pattern given each arm's own inputs) --
/// only the acceptance rule for entering the scored population differs.
///
/// `probes`: the frozen probe bank (see module docs -- built by the
/// caller, e.g. `pde_wave_stage_b.rs`, since building it needs
/// system-specific trajectory sampling this crate doesn't have). `basis`:
/// only used when `dedup_mode` is `Hdc`; must have `probes.len()` vectors
/// (see [`hdc_probe_basis`]).
#[allow(clippy::too_many_arguments)]
pub fn discover_flux_factorized_with_dedup(
    rho: &Expr,
    rhs: fn(&[f64], f64) -> Vec<f64>,
    initial_state: &[f64],
    n: usize,
    config: &RegressorConfig,
    t_max: f64,
    dt: f64,
    polynomial_only: bool,
    dedup_mode: DedupMode,
    probes: &[[f64; 4]],
    basis: &[ContinuousHV],
) -> (Vec<FactorizedFluxResult>, DedupRunMetrics, DedupRunMetrics) {
    let Some(trajectory) = sampled_trajectory(rhs, initial_state, t_max, dt) else {
        return (
            Vec::new(),
            DedupRunMetrics::default(),
            DedupRunMetrics::default(),
        );
    };

    let pop_size = config.population_size.max(8);
    let generations = config.generations;
    let max_depth = config.max_depth.min(3);
    let max_complexity = config.max_complexity;
    let exclude_trig = config.exclude_trig;
    let mut rng = config.seed;

    let fresh = |rng: &mut u64| -> Expr {
        if polynomial_only {
            fresh_random_polynomial_bond_expr(rng, max_depth)
        } else {
            fresh_random_bond_expr(rng, max_depth, exclude_trig)
        }
    };

    let pair_score = |a: &Expr, b: &Expr| -> Option<f64> {
        let product = Expr::BinOp(BinOp::Mul, Box::new(a.clone()), Box::new(b.clone()));
        let c = product.complexity();
        if !(2..=max_complexity).contains(&c) {
            return None;
        }
        let fixed = gauge_fix_flux(&product);
        let residual = discrete_continuity_residual(rho, &fixed, &trajectory, n);
        if residual.is_finite() && residual < 1e10 {
            Some(residual)
        } else {
            None
        }
    };

    let mut pop_a: Vec<Expr> = (0..pop_size).map(|_| fresh(&mut rng)).collect();
    let mut pop_b: Vec<Expr> = (0..pop_size).map(|_| fresh(&mut rng)).collect();
    let mut best_a = Expr::Const(1.0);
    let mut best_b = Expr::Const(1.0);
    let mut best_pair_residual = f64::INFINITY;
    let mut metrics_a = DedupRunMetrics::default();
    let mut metrics_b = DedupRunMetrics::default();

    let overproduce = pop_size * 2;

    for _gen in 0..generations {
        let fit_a: Vec<(usize, Option<f64>)> = pop_a
            .iter()
            .enumerate()
            .map(|(i, a)| (i, pair_score(a, &best_b)))
            .collect();
        if let Some(&(idx, Some(r))) = fit_a
            .iter()
            .filter(|(_, s)| s.is_some())
            .min_by(|x, y| cmp_scores(x.1, y.1))
            .filter(|&&(_, r_opt)| r_opt.is_some_and(|val| val < best_pair_residual))
        {
            best_a = pop_a[idx].clone();
            best_pair_residual = r;
        }

        let fit_b: Vec<(usize, Option<f64>)> = pop_b
            .iter()
            .enumerate()
            .map(|(i, b)| (i, pair_score(&best_a, b)))
            .collect();
        if let Some(&(idx, Some(r))) = fit_b
            .iter()
            .filter(|(_, s)| s.is_some())
            .min_by(|x, y| cmp_scores(x.1, y.1))
            .filter(|&&(_, r_opt)| r_opt.is_some_and(|val| val < best_pair_residual))
        {
            best_b = pop_b[idx].clone();
            best_pair_residual = r;
        }

        let buffer_a = breed_generation(
            &pop_a,
            &fit_a,
            overproduce,
            max_depth,
            exclude_trig,
            polynomial_only,
            &mut rng,
        );
        let buffer_b = breed_generation(
            &pop_b,
            &fit_b,
            overproduce,
            max_depth,
            exclude_trig,
            polynomial_only,
            &mut rng,
        );

        let score_a = |e: &Expr| pair_score(e, &best_b);
        pop_a = filter_buffer_for_population(
            buffer_a,
            pop_size,
            score_a,
            dedup_mode,
            probes,
            basis,
            &mut metrics_a,
        );
        let score_b = |e: &Expr| pair_score(&best_a, e);
        pop_b = filter_buffer_for_population(
            buffer_b,
            pop_size,
            score_b,
            dedup_mode,
            probes,
            basis,
            &mut metrics_b,
        );
    }

    let mut candidates: Vec<(Expr, Expr, f64)> = Vec::new();
    for a in pop_a.iter().take(20) {
        if let Some(r) = pair_score(a, &best_b) {
            candidates.push((a.clone(), best_b.clone(), r));
        }
    }
    for b in pop_b.iter().take(20) {
        if let Some(r) = pair_score(&best_a, b) {
            candidates.push((best_a.clone(), b.clone(), r));
        }
    }
    if let Some(r) = pair_score(&best_a, &best_b) {
        candidates.push((best_a.clone(), best_b.clone(), r));
    }
    candidates.sort_by(|x, y| x.2.partial_cmp(&y.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut results = Vec::new();
    for (a, b, residual) in candidates {
        let product_str = format!("({a}) * ({b})");
        if !seen.insert(product_str.clone()) {
            continue;
        }
        results.push(FactorizedFluxResult {
            factor_a: a,
            factor_b: b,
            product_str,
            residual,
        });
        if results.len() >= 10 {
            break;
        }
    }
    (results, metrics_a, metrics_b)
}

/// Filters `buffer` (an overproduced set of candidates from
/// [`breed_generation`]) down to `pop_size`, per `dedup_mode`. Behaviorally
/// duplicate candidates (fingerprint similarity >= threshold) replace their
/// match only when they score better (lower `score_fn` residual) or are
/// simpler on a tie/invalid-score. Two exactly-identical expressions
/// automatically have identical fingerprints and thus similarity 1.0, so
/// this subsumes exact-syntactic-duplicate detection for Arms 2/3 without a
/// separate string-based pre-filter.
fn filter_buffer_for_population(
    buffer: Vec<Expr>,
    pop_size: usize,
    score_fn: impl Fn(&Expr) -> Option<f64>,
    dedup_mode: DedupMode,
    probes: &[[f64; 4]],
    basis: &[ContinuousHV],
    metrics: &mut DedupRunMetrics,
) -> Vec<Expr> {
    let DedupMode::None = dedup_mode else {
        let threshold = match dedup_mode {
            DedupMode::Vector { threshold }
            | DedupMode::Hdc { threshold }
            | DedupMode::HdcQuantized { threshold, .. } => threshold,
            DedupMode::None => unreachable!(),
        };
        let mut accepted: Vec<Expr> = Vec::with_capacity(pop_size);
        let mut accepted_fps: Vec<Option<Vec<f32>>> = Vec::with_capacity(pop_size);
        let mut accepted_hvs: Vec<Option<ContinuousHV>> = Vec::with_capacity(pop_size);

        for cand in buffer {
            if accepted.len() >= pop_size {
                break;
            }
            let fp = behavioral_fingerprint(&cand, probes);
            let cand_hv = match (dedup_mode, &fp) {
                (DedupMode::Hdc { .. }, Some(f)) => Some(hdc_fingerprint(f, basis)),
                (DedupMode::HdcQuantized { bucket_width, .. }, Some(f)) => Some(hdc_fingerprint(
                    &quantize_fingerprint(f, bucket_width),
                    basis,
                )),
                _ => None,
            };

            let mut dup_index = None;
            if let Some(ref fp_vec) = fp {
                for i in 0..accepted.len() {
                    let sim = match dedup_mode {
                        DedupMode::Vector { .. } => accepted_fps[i]
                            .as_ref()
                            .map(|e| vector_similarity(e, fp_vec)),
                        DedupMode::Hdc { .. } | DedupMode::HdcQuantized { .. } => accepted_hvs[i]
                            .as_ref()
                            .zip(cand_hv.as_ref())
                            .map(|(e, c)| e.similarity(c)),
                        DedupMode::None => None,
                    };
                    if sim.is_some_and(|s| s >= threshold) {
                        dup_index = Some(i);
                        break;
                    }
                }
            }

            match dup_index {
                Some(i) => {
                    metrics.duplicates_rejected += 1;
                    let existing_score = score_fn(&accepted[i]);
                    let new_score = score_fn(&cand);
                    let existing_wins = match (existing_score, new_score) {
                        (Some(e), Some(n)) => e <= n,
                        (Some(_), None) => true,
                        (None, Some(_)) => false,
                        (None, None) => accepted[i].complexity() <= cand.complexity(),
                    };
                    if let (Some(e), Some(n)) = (existing_score, new_score) {
                        let denom = e.max(n).max(1e-9);
                        if (e - n).abs() / denom > 0.10 {
                            metrics.false_merges += 1;
                        }
                    }
                    if !existing_wins {
                        accepted[i] = cand;
                        accepted_fps[i] = fp;
                        accepted_hvs[i] = cand_hv;
                    }
                }
                None => {
                    accepted.push(cand);
                    accepted_fps.push(fp);
                    accepted_hvs.push(cand_hv);
                    metrics.unique_candidates_accepted += 1;
                }
            }
        }
        return accepted;
    };
    metrics.unique_candidates_accepted += pop_size.min(buffer.len());
    buffer.into_iter().take(pop_size).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// True if `expr` contains, anywhere in its tree, a `BinOp` of `op`
    /// whose two direct operands are exactly `Var(a)` and `Var(b)` (in
    /// either order).
    fn contains_pattern(expr: &Expr, op: BinOp, a: &str, b: &str) -> bool {
        match expr {
            Expr::BinOp(o, l, r) => {
                let direct = *o == op
                    && matches!(
                        (&**l, &**r),
                        (Expr::Var(x), Expr::Var(y))
                            if (x == a && y == b) || (x == b && y == a)
                    );
                direct || contains_pattern(l, op, a, b) || contains_pattern(r, op, a, b)
            }
            Expr::Func(_, arg) => contains_pattern(arg, op, a, b),
            _ => false,
        }
    }

    /// M2.1 diagnostic (not a pass/fail test -- reports reachability
    /// statistics): even with a generous `max_complexity`, how often does
    /// the random-tree generator [`random_expr_multivar`] actually produce
    /// a subtree shaped like `u_r - u_c` (the displacement `j_truth()`
    /// needs), one shaped like `v_c + v_r` (the velocity sum it needs), or
    /// both? "Representable under the complexity cap" and "reachable by
    /// the generator/mutation distribution" are different claims -- this
    /// measures the second directly instead of inferring it from a failed
    /// search. Pure syntactic pattern matching (does NOT catch semantically
    /// equivalent but differently-shaped subtrees, e.g. `0 - (u_c - u_r)`
    /// or a `Div`-based rescaling -- an honest limitation of this specific
    /// diagnostic, not a claim about semantic reachability).
    #[test]
    fn reachability_estimate_for_the_needed_structural_motifs() {
        let bond_names = BOND_VAR_NAMES;
        let n_samples: u64 = 50_000;
        let mut rng: u64 = 0xC0FFEE_u64;
        let (mut has_displacement, mut has_vel_sum, mut has_both) = (0u64, 0u64, 0u64);

        for _ in 0..n_samples {
            rng = lcg_step(rng);
            let expr = random_expr_multivar(&mut rng, 4, &bond_names, true);
            let d = contains_pattern(&expr, BinOp::Sub, "u_r", "u_c");
            let v = contains_pattern(&expr, BinOp::Add, "v_c", "v_r");
            if d {
                has_displacement += 1;
            }
            if v {
                has_vel_sum += 1;
            }
            if d && v {
                has_both += 1;
            }
        }

        println!(
            "--- M2.1 reachability estimate over {n_samples} random trees (max_depth=4) ---\n\
             contains a (u_r-u_c)-shaped subtree: {has_displacement} ({:.4}%)\n\
             contains a (v_c+v_r)-shaped subtree: {has_vel_sum} ({:.4}%)\n\
             contains BOTH (necessary, not sufficient, for the correct product): \
             {has_both} ({:.5}%)",
            100.0 * has_displacement as f64 / n_samples as f64,
            100.0 * has_vel_sum as f64 / n_samples as f64,
            100.0 * has_both as f64 / n_samples as f64,
        );
    }

    /// M3 Phase 1 (see the module-level plan doc / `pde_wave_stage_b.rs`'s M3 note): the
    /// identical M2.1 measurement above, run against [`random_motif_expr`] instead of
    /// [`random_expr_multivar`]. Predeclared threshold, frozen before running: the "contains
    /// BOTH" rate must exceed 1% (a ~45x improvement over M2.1's measured 0.022% baseline) to
    /// count as SUPPORTED. This is a diagnostic reporting test, not a hard pass/fail gate --
    /// matching `reachability_estimate_for_the_needed_structural_motifs` above and this arc's
    /// practice throughout of reporting the real number regardless of verdict.
    #[test]
    fn motif_constrained_generator_reachability_estimate() {
        let bond_names = BOND_VAR_NAMES;
        let n_samples: u64 = 50_000;
        let mut rng: u64 = 0xC0FFEE_u64;
        let (mut has_displacement, mut has_vel_sum, mut has_both) = (0u64, 0u64, 0u64);

        for _ in 0..n_samples {
            rng = lcg_step(rng);
            let expr = random_motif_expr(&mut rng, &bond_names, 3);
            let d = contains_pattern(&expr, BinOp::Sub, "u_r", "u_c");
            let v = contains_pattern(&expr, BinOp::Add, "v_c", "v_r");
            if d {
                has_displacement += 1;
            }
            if v {
                has_vel_sum += 1;
            }
            if d && v {
                has_both += 1;
            }
        }

        let both_rate = has_both as f64 / n_samples as f64;
        let verdict = if both_rate > 0.01 {
            "SUPPORTED"
        } else {
            "NEGATIVE"
        };
        println!(
            "--- M3 Phase 1: motif-constrained generator reachability over {n_samples} trees (depth=3) ---\n\
             contains a (u_r-u_c)-shaped subtree: {has_displacement} ({:.4}%)\n\
             contains a (v_c+v_r)-shaped subtree: {has_vel_sum} ({:.4}%)\n\
             contains BOTH: {has_both} ({:.4}%)\n\
             vs. M2.1 unrestricted-generator baseline: 0.02200% (11/50_000)\n\
             predeclared threshold: >1.0000% -> {verdict}",
            100.0 * has_displacement as f64 / n_samples as f64,
            100.0 * has_vel_sum as f64 / n_samples as f64,
            100.0 * both_rate,
        );
    }

    /// True if `expr` contains, anywhere in its tree, a `Mul` node whose two direct children are
    /// *both* exactly the pattern `Sub(Var(a), Var(b))` (in either order, matching
    /// [`contains_pattern`]'s own order-insensitive convention) -- i.e. `diff * diff`, the only
    /// way [`random_motif_expr`] can construct a squared-displacement shape, since it has no
    /// `Pow`/square combinator at all (contrast [`random_density_motif_expr`], which was given
    /// a dedicated `Pow`-based square leaf for exactly this reason -- see that function's doc
    /// comment).
    fn contains_squared_diff_pattern(expr: &Expr, a: &str, b: &str) -> bool {
        let is_diff_leaf = |e: &Expr| -> bool {
            matches!(
                e,
                Expr::BinOp(BinOp::Sub, l, r)
                    if matches!((&**l, &**r), (Expr::Var(x), Expr::Var(y))
                        if (x == a && y == b) || (x == b && y == a))
            )
        };
        match expr {
            Expr::BinOp(op, l, r) => {
                let direct = *op == BinOp::Mul && is_diff_leaf(l) && is_diff_leaf(r);
                direct
                    || contains_squared_diff_pattern(l, a, b)
                    || contains_squared_diff_pattern(r, a, b)
            }
            Expr::Func(_, arg) => contains_squared_diff_pattern(arg, a, b),
            _ => false,
        }
    }

    /// FPU-alpha reachability diagnostic (see the plan doc): `j_fpu_truth(alpha)`
    /// (`pde_wave_stage_b.rs`) needs a squared-displacement term `(u_r-u_c)^2` on top of the
    /// linear flux's displacement/velocity-sum motifs. [`random_motif_expr`] was built and
    /// validated (M3 Phase 1/2) with no square/power combinator, so the only way it can
    /// construct that shape is the `diff * diff` coincidence [`contains_squared_diff_pattern`]
    /// checks for -- two independently-drawn recursive branches happening to produce the
    /// identical `Sub(u_r, u_c)` leaf. Same settings as
    /// [`motif_constrained_generator_reachability_estimate`] (`n_samples=50_000`,
    /// `rng=0xC0FFEE_u64`, `random_motif_expr(&mut rng, &bond_names, 3)`) for a directly
    /// comparable report. **Purely descriptive, no predeclared pass/fail threshold** -- unlike
    /// the M3 Phase 1 diagnostic above, there is no prior baseline here to compare a claimed
    /// improvement against; this measures a fresh question, not a repeat of an already-answered
    /// one. Scope: diagnostic only -- does not evolve, does not modify `random_motif_expr`, and
    /// does not itself decide whether a square-aware generator extension or a GP search for
    /// FPU's flux is warranted (see the plan doc's final Verification step for that call).
    #[test]
    fn motif_generator_reachability_for_fpu_flux_shape() {
        let bond_names = BOND_VAR_NAMES;
        let n_samples: u64 = 50_000;
        let mut rng: u64 = 0xC0FFEE_u64;
        let (mut has_displacement, mut has_vel_sum, mut has_squared_diff, mut has_squared_and_vel) =
            (0u64, 0u64, 0u64, 0u64);

        for _ in 0..n_samples {
            rng = lcg_step(rng);
            let expr = random_motif_expr(&mut rng, &bond_names, 3);
            let d = contains_pattern(&expr, BinOp::Sub, "u_r", "u_c");
            let v = contains_pattern(&expr, BinOp::Add, "v_c", "v_r");
            let sq = contains_squared_diff_pattern(&expr, "u_r", "u_c");
            if d {
                has_displacement += 1;
            }
            if v {
                has_vel_sum += 1;
            }
            if sq {
                has_squared_diff += 1;
            }
            if sq && v {
                has_squared_and_vel += 1;
            }
        }

        println!(
            "--- FPU-alpha flux reachability over {n_samples} random_motif_expr trees (depth=3) ---\n\
             contains a (u_r-u_c)-shaped subtree: {has_displacement} ({:.4}%)\n\
             contains a (v_c+v_r)-shaped subtree: {has_vel_sum} ({:.4}%)\n\
             contains a (u_r-u_c)*(u_r-u_c)-shaped subtree (the FPU-specific question, since \
             random_motif_expr has no Pow/square combinator): {has_squared_diff} ({:.4}%)\n\
             contains BOTH the squared-diff motif AND the velocity-sum motif (what the \
             nonlinear flux term jointly needs): {has_squared_and_vel} ({:.5}%)\n\
             (purely descriptive -- no predeclared threshold, no prior baseline for this \
             specific question)",
            100.0 * has_displacement as f64 / n_samples as f64,
            100.0 * has_vel_sum as f64 / n_samples as f64,
            100.0 * has_squared_diff as f64 / n_samples as f64,
            100.0 * has_squared_and_vel as f64 / n_samples as f64,
        );
    }

    /// True if `expr` contains, anywhere in its tree, a `Pow` node whose base is exactly
    /// `Sub(Var(a), Var(b))` (or `Sub(Var(b), Var(a))`, order-insensitive, matching
    /// [`contains_pattern`]'s convention) and whose exponent is `Const(2.0)` -- the shape
    /// [`random_fpu_flux_motif_expr`]'s new leaf actually constructs. A separate helper from
    /// [`contains_squared_diff_pattern`], which checks the older `Mul(diff, diff)` coincidence
    /// shape [`random_motif_expr`] can only reach by accident -- the two check genuinely
    /// different tree shapes, and both stay useful (the old one for the historical
    /// `random_motif_expr` measurement, this one for the new generator).
    fn contains_squared_diff_pow_pattern(expr: &Expr, a: &str, b: &str) -> bool {
        let is_diff_leaf = |e: &Expr| -> bool {
            matches!(
                e,
                Expr::BinOp(BinOp::Sub, l, r)
                    if matches!((&**l, &**r), (Expr::Var(x), Expr::Var(y))
                        if (x == a && y == b) || (x == b && y == a))
            )
        };
        match expr {
            Expr::BinOp(BinOp::Pow, base, exp) => {
                let direct = is_diff_leaf(base) && matches!(&**exp, Expr::Const(c) if *c == 2.0);
                direct
                    || contains_squared_diff_pow_pattern(base, a, b)
                    || contains_squared_diff_pow_pattern(exp, a, b)
            }
            Expr::BinOp(_, l, r) => {
                contains_squared_diff_pow_pattern(l, a, b)
                    || contains_squared_diff_pow_pattern(r, a, b)
            }
            Expr::Func(_, arg) => contains_squared_diff_pow_pattern(arg, a, b),
            _ => false,
        }
    }

    /// Re-measures [`motif_generator_reachability_for_fpu_flux_shape`]'s question against
    /// [`random_fpu_flux_motif_expr`] (the square-aware generator built in response to that
    /// diagnostic's negative result) instead of the unmodified `random_motif_expr`. Same
    /// `n_samples=50_000`, `rng=0xC0FFEE_u64` seed, and draw depth (3) for a directly comparable
    /// report. **Predeclared threshold this time** (unlike the prior diagnostic, which had no
    /// baseline): SUPPORTED if the squared-diff-AND-velocity-sum co-occurrence rate exceeds 1%
    /// -- the same bar M3 Phase 1 itself used for its own analogous "does a targeted generator
    /// change beat the previous rate" question, frozen before running, chosen because this run
    /// *does* have a real prior number to beat (0.028%, the unmodified generator's measured
    /// rate).
    #[test]
    fn motif_generator_with_squares_reachability_for_fpu_flux_shape() {
        let bond_names = BOND_VAR_NAMES;
        let n_samples: u64 = 50_000;
        let mut rng: u64 = 0xC0FFEE_u64;
        let (mut has_displacement, mut has_vel_sum, mut has_squared_diff, mut has_squared_and_vel) =
            (0u64, 0u64, 0u64, 0u64);

        for _ in 0..n_samples {
            rng = lcg_step(rng);
            let expr = random_fpu_flux_motif_expr(&mut rng, &bond_names, 3);
            let d = contains_pattern(&expr, BinOp::Sub, "u_r", "u_c");
            let v = contains_pattern(&expr, BinOp::Add, "v_c", "v_r");
            let sq = contains_squared_diff_pow_pattern(&expr, "u_r", "u_c");
            if d {
                has_displacement += 1;
            }
            if v {
                has_vel_sum += 1;
            }
            if sq {
                has_squared_diff += 1;
            }
            if sq && v {
                has_squared_and_vel += 1;
            }
        }

        let both_rate = has_squared_and_vel as f64 / n_samples as f64;
        let verdict = if both_rate > 0.01 {
            "SUPPORTED"
        } else {
            "NEGATIVE"
        };
        println!(
            "--- Square-aware FPU flux generator reachability over {n_samples} \
             random_fpu_flux_motif_expr trees (depth=3) ---\n\
             contains a (u_r-u_c)-shaped subtree: {has_displacement} ({:.4}%)\n\
             contains a (v_c+v_r)-shaped subtree: {has_vel_sum} ({:.4}%)\n\
             contains a Pow(u_r-u_c, 2)-shaped subtree (the FPU-specific question): \
             {has_squared_diff} ({:.4}%)\n\
             contains BOTH the squared-diff motif AND the velocity-sum motif: \
             {has_squared_and_vel} ({:.4}%)\n\
             vs. unmodified random_motif_expr baseline: 0.02800% (14/50_000)\n\
             predeclared threshold: >1.0000% -> {verdict}",
            100.0 * has_displacement as f64 / n_samples as f64,
            100.0 * has_vel_sum as f64 / n_samples as f64,
            100.0 * has_squared_diff as f64 / n_samples as f64,
            100.0 * both_rate,
        );
    }

    /// Regression test for a real bug found while calibrating this module: `two_distinct_vars`
    /// originally drew indices via `*rng % var_names.len()` directly on `lcg_step`'s raw
    /// output. `var_names.len() == 4` is a power of 2, and the low-order bits of any full-
    /// period-2^64 LCG have a provably short cycle independent of the rest of the state -- an
    /// empirical debug dump showed only 2 of the 6 possible variable pairs ever appearing
    /// across many draws, silently making the target motifs unreachable regardless of the
    /// generator's intended bias. Fixed via `rand_index_high_bits` (draws from the state's
    /// high 32 bits instead). This test asserts the fix: over enough draws, `two_distinct_vars`
    /// should reach all `C(4,2) = 6` unordered pairs, not collapse to 1-2.
    #[test]
    fn two_distinct_vars_reaches_all_unordered_pairs() {
        let bond_names = BOND_VAR_NAMES;
        let mut rng: u64 = 0xC0FFEE_u64;
        let mut seen: std::collections::HashSet<(String, String)> =
            std::collections::HashSet::new();
        for _ in 0..500 {
            let (a, b) = two_distinct_vars(&mut rng, &bond_names);
            let key = if a < b { (a, b) } else { (b, a) };
            seen.insert(key);
        }
        assert_eq!(
            seen.len(),
            6,
            "expected all 6 unordered pairs from 4 variables to be reachable within 500 \
             draws, got only {}: {seen:?} -- possible regression of the LCG low-bit pitfall \
             this test guards against",
            seen.len()
        );
    }

    #[test]
    fn polynomial_generator_never_produces_non_polynomial_trees() {
        let mut rng: u64 = 12345;
        for _ in 0..5_000 {
            let expr = fresh_random_polynomial_bond_expr(&mut rng, 4);
            assert!(is_polynomial(&expr), "non-polynomial tree: {expr}");
        }
    }

    #[test]
    fn is_polynomial_rejects_div_pow_and_func() {
        let u = || Expr::Var("u_c".to_string());
        assert!(!is_polynomial(&Expr::BinOp(
            BinOp::Div,
            Box::new(u()),
            Box::new(Expr::Const(2.0))
        )));
        assert!(!is_polynomial(&Expr::BinOp(
            BinOp::Pow,
            Box::new(u()),
            Box::new(Expr::Const(2.0))
        )));
        assert!(!is_polynomial(&Expr::Func(UnaryFn::Sin, Box::new(u()))));
        assert!(is_polynomial(&Expr::BinOp(
            BinOp::Add,
            Box::new(u()),
            Box::new(Expr::Const(1.0))
        )));
    }

    fn tiny_probe_bank() -> Vec<[f64; 4]> {
        vec![
            [0.3, -0.7, 0.2, -0.4],
            [1.0, 1.0, 0.0, 0.0],
            [-0.5, 0.5, 0.9, -0.9],
            [0.0, 0.0, 0.0, 0.0],
        ]
    }

    #[test]
    fn behavioral_fingerprint_rejects_non_finite_probes() {
        let probes = tiny_probe_bank();
        // 1/(u_c - u_c) = 1/0 -> NaN at every probe (u_c - u_c is always 0).
        let bad = Expr::BinOp(
            BinOp::Div,
            Box::new(Expr::Const(1.0)),
            Box::new(Expr::BinOp(
                BinOp::Sub,
                Box::new(Expr::Var("u_c".to_string())),
                Box::new(Expr::Var("u_c".to_string())),
            )),
        );
        assert!(behavioral_fingerprint(&bad, &probes).is_none());

        let good = Expr::Var("u_c".to_string());
        let fp = behavioral_fingerprint(&good, &probes);
        assert!(fp.is_some());
        assert_eq!(fp.unwrap().len(), probes.len());
    }

    #[test]
    fn vector_similarity_self_is_one_and_orthogonal_is_low() {
        let probes = tiny_probe_bank();
        let a = behavioral_fingerprint(&j_truth_for_test(), &probes).unwrap();
        let self_sim = vector_similarity(&a, &a);
        assert!(
            (self_sim - 1.0).abs() < 1e-5,
            "self-similarity should be ~1.0, got {self_sim}"
        );

        // A candidate that's a pure function of a DIFFERENT variable than
        // j_truth()'s dominant behavior should be markedly less similar
        // than self-similarity, on this small probe bank.
        let other = Expr::Var("v_c".to_string());
        let b = behavioral_fingerprint(&other, &probes).unwrap();
        let cross_sim = vector_similarity(&a, &b);
        assert!(
            cross_sim < 0.99,
            "expected a meaningfully lower similarity than self-similarity, got {cross_sim}"
        );
    }

    #[test]
    fn hdc_fingerprint_self_similarity_is_high() {
        let probes = tiny_probe_bank();
        let basis = hdc_probe_basis(probes.len(), 42);
        let a = behavioral_fingerprint(&j_truth_for_test(), &probes).unwrap();
        let hv_a = hdc_fingerprint(&a, &basis);
        let hv_a_again = hdc_fingerprint(&a, &basis);
        let self_sim = hv_a.similarity(&hv_a_again);
        assert!(
            (self_sim - 1.0).abs() < 1e-4,
            "identical fingerprints should encode near-identically, got {self_sim}"
        );
    }

    #[test]
    fn hdc_and_vector_similarity_agree_on_ordering() {
        // Not a claim that HDC and direct-vector similarity are numerically
        // identical (they aren't -- different representations) -- just that
        // for a small, clean probe bank they should agree on *ordering*:
        // a self-comparison should score higher than a comparison against a
        // clearly different candidate, under both encodings.
        let probes = tiny_probe_bank();
        let basis = hdc_probe_basis(probes.len(), 42);
        let a = behavioral_fingerprint(&j_truth_for_test(), &probes).unwrap();
        let b = behavioral_fingerprint(&Expr::Var("v_c".to_string()), &probes).unwrap();

        let vec_self = vector_similarity(&a, &a);
        let vec_cross = vector_similarity(&a, &b);
        let hdc_self = hdc_fingerprint(&a, &basis).similarity(&hdc_fingerprint(&a, &basis));
        let hdc_cross = hdc_fingerprint(&a, &basis).similarity(&hdc_fingerprint(&b, &basis));

        assert!(vec_self > vec_cross);
        assert!(hdc_self > hdc_cross);
    }

    /// Local stand-in for `pde_wave_stage_b::j_truth()` (that crate depends
    /// on this one, not the other way around) -- just needs to be *some*
    /// nontrivial multi-variable expression for these encoding sanity
    /// tests, not the actual physical answer.
    fn j_truth_for_test() -> Expr {
        Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::BinOp(
                BinOp::Sub,
                Box::new(Expr::Var("u_r".to_string())),
                Box::new(Expr::Var("u_c".to_string())),
            )),
            Box::new(Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::Var("v_c".to_string())),
                Box::new(Expr::Var("v_r".to_string())),
            )),
        )
    }

    #[test]
    fn quantize_fingerprint_is_genuinely_lossy() {
        // Two distinct raw fingerprints, close enough to land in the same
        // buckets at a coarse bucket_width -- the actual information-loss
        // mechanism that makes the quantized+HDC arm non-isometric.
        let a = vec![0.10_f32, -0.30, 1.02, -0.98];
        let b = vec![0.14_f32, -0.34, 0.97, -1.03];
        assert_ne!(a, b, "sanity: the two raw fingerprints must differ");
        assert!(
            vector_similarity(&a, &b) < 0.999999,
            "sanity: raw fingerprints should not be perfectly similar"
        );

        let qa = quantize_fingerprint(&a, 0.5);
        let qb = quantize_fingerprint(&b, 0.5);
        assert_eq!(
            qa, qb,
            "at bucket_width=0.5 these should collide -- that collision IS \
             the information loss that breaks the isometry"
        );
    }

    #[test]
    fn quantized_hdc_similarity_can_differ_from_raw_vector_similarity() {
        // The actual empirical proof of non-isometry: construct a pair
        // whose RAW cosine similarity is clearly below 1.0, but whose
        // QUANTIZED-then-HDC-encoded similarity is exactly 1.0 (because
        // quantization collapses them to the same bucketed fingerprint).
        // This could never happen with the unquantized `hdc_fingerprint`
        // (that path is proven-identical to `vector_similarity` by
        // construction -- see `hdc_fingerprint`'s docs) -- it can only
        // happen once a genuinely lossy step is introduced.
        // Chosen so raw cosine similarity is clearly below 1.0 (~0.724,
        // verified by hand) while both still round to the same bucketed
        // fingerprint [0.5, 0.5, -0.5, -0.5] at bucket_width=0.5.
        let a = vec![0.7_f32, 0.3, -0.3, -0.7];
        let b = vec![0.3_f32, 0.7, -0.7, -0.3];
        let raw_sim = vector_similarity(&a, &b);
        assert!(
            raw_sim < 0.9,
            "raw similarity should be meaningfully imperfect, got {raw_sim}"
        );

        let basis = hdc_probe_basis(a.len(), 7);
        let bucket_width = 0.5;
        let hv_a = hdc_fingerprint(&quantize_fingerprint(&a, bucket_width), &basis);
        let hv_b = hdc_fingerprint(&quantize_fingerprint(&b, bucket_width), &basis);
        let quantized_hdc_sim = hv_a.similarity(&hv_b);

        assert!(
            (quantized_hdc_sim - 1.0).abs() < 1e-4,
            "quantized fingerprints collided, so their HDC encodings should \
             be ~identical, got {quantized_hdc_sim}"
        );
        assert!(
            (quantized_hdc_sim - raw_sim).abs() > 0.01,
            "quantized-HDC similarity ({quantized_hdc_sim}) should differ \
             from raw-vector similarity ({raw_sim}) -- this is the concrete \
             proof this encoding is NOT an isometry of vector_similarity, \
             unlike the unquantized hdc_fingerprint path"
        );
    }
}
