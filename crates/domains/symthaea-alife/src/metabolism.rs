// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Metabolism: thermodynamic grounding, per `ALIFE_PLAN_2026-07-08.md` Phase 3.
//!
//! Turns homeostasis from a soft FEP *preference* into a hard *energy-budget constraint* --
//! Schrödinger's "negative entropy" framing: an organism must import free energy and export
//! entropy to maintain internal order, or it dies.
//!
//! Ports the actual physics from `src/cognitive_loop/thermodynamic_physics_bridge.rs` (the main
//! crate's `ThermodynamicPhysicsBridge`) -- the Landauer and Prigogine *formulas*, not the
//! `ConsciousnessPhase`/`ThermodynamicRegime` coupling those live inside there (per the plan's
//! explicit instruction: "port the math, not the coupling").
//!
//! **On units**: like the main crate's own `K_CONSCIOUSNESS_BOLTZMANN = 0.01`
//! (`thresholds/consciousness.rs`), [`K_ALIFE_BOLTZMANN`] is a dimensionless analog scaled to
//! this crate's `[0, 1]` energy/observation units, **not** the true physical Boltzmann constant
//! (1.380649×10⁻²³ J/K). Landauer's bound is real physics; the specific numeric constant here is
//! a deliberate, documented rescaling to a dimensionless system, matching how the main crate
//! already handles this rather than inventing a new convention.

/// Dimensionless Boltzmann-constant analog for alife energy units. See module docs. Chosen (with
/// `OrganismConfig::dissipation_rate`) so the combined Landauer+Prigogine floor is a real,
/// measurable addition to existing per-tick costs without swamping them -- a traced diagnostic
/// at the original 0.01 found mean physical_cost≈0.0104/tick, roughly *doubling*
/// `OrganismConfig::default().metabolic_cost` (0.01) outright, which crashed every Phase 0/1/2
/// organism's energy to zero within ~10 ticks. Scaled down ~10x to land around ~10% of that
/// baseline instead.
pub const K_ALIFE_BOLTZMANN: f64 = 0.0005;

/// Landauer's bound (1961): erasing/writing `bits` of information costs at least
/// `k_B * T * ln(2)` per bit. This is a real physical lower bound, not a heuristic -- any
/// physically-realizable irreversible computation must dissipate at least this much energy.
///
/// `bits` should be non-negative (irreversible information written/erased this step); negative
/// or NaN inputs are clamped to 0. `temperature` is floored away from zero to avoid a divide/
/// blowup at T=0 (which real physics disallows -- absolute zero is unreachable).
pub fn landauer_minimum(bits: f64, temperature: f64) -> f64 {
    bits.max(0.0) * K_ALIFE_BOLTZMANN * temperature.max(1e-6) * std::f64::consts::LN_2
}

/// Shannon entropy, in bits, of a probability distribution. Used to quantify how many bits of
/// uncertainty are *resolved* (erased, in Landauer's sense) when an agent commits to one action
/// out of a distribution over several -- `H(p) = -Σ p_i log2(p_i)`, zero contribution from any
/// `p_i <= 0`. Bounded above by `log2(n)` for `n` equiprobable outcomes.
pub fn shannon_entropy_bits(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.log2())
        .sum::<f64>()
        .max(0.0)
}

/// Prigogine (1947/1977): a dissipative structure maintaining order far from equilibrium must
/// continuously export entropy production, at a rate that grows with how much order is being
/// maintained. Standard near-equilibrium linear-response form: entropy production
/// `σ = J·X` with flux `J` proportional to thermodynamic force `X` near equilibrium, and `X`
/// itself proportional to the deviation from the disordered (zero-order) state -- giving a
/// quadratic-in-order dissipation cost, scaled by temperature and a dissipation-rate constant
/// analogous to a transport coefficient.
///
/// `order_maintained` is expected in `[0, 1]` (0 = no order kept, 1 = maximal); values outside
/// that range are not clamped here (callers are expected to pass a genuine `[0,1]` quantity --
/// see `Organism::tick`'s use, which derives it from homeostatic deficit).
pub fn prigogine_dissipation_cost(
    order_maintained: f64,
    temperature: f64,
    dissipation_rate: f64,
) -> f64 {
    dissipation_rate.max(0.0) * order_maintained.powi(2) * temperature.max(1e-6)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn landauer_minimum_is_zero_for_zero_bits() {
        assert_eq!(landauer_minimum(0.0, 1.0), 0.0);
    }

    #[test]
    fn landauer_minimum_scales_linearly_with_bits() {
        let one_bit = landauer_minimum(1.0, 1.0);
        let ten_bits = landauer_minimum(10.0, 1.0);
        assert!((ten_bits - one_bit * 10.0).abs() < 1e-9);
    }

    #[test]
    fn landauer_minimum_never_negative_for_negative_bits_input() {
        // Negative "bits" isn't physically meaningful (can't un-erase information) -- clamp,
        // don't let it produce a negative "cost" that could offset real charges elsewhere.
        assert_eq!(landauer_minimum(-5.0, 1.0), 0.0);
    }

    #[test]
    fn shannon_entropy_zero_for_certain_outcome() {
        assert_eq!(shannon_entropy_bits(&[1.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn shannon_entropy_one_bit_for_fair_coin() {
        let h = shannon_entropy_bits(&[0.5, 0.5]);
        assert!((h - 1.0).abs() < 1e-9, "expected 1 bit, got {h}");
    }

    #[test]
    fn shannon_entropy_bounded_by_log2_n() {
        // 4 equiprobable outcomes: max entropy = log2(4) = 2 bits.
        let h = shannon_entropy_bits(&[0.25, 0.25, 0.25, 0.25]);
        assert!((h - 2.0).abs() < 1e-9, "expected 2 bits, got {h}");
    }

    #[test]
    fn prigogine_cost_grows_with_order_maintained() {
        let low = prigogine_dissipation_cost(0.1, 1.0, 1.0);
        let high = prigogine_dissipation_cost(0.9, 1.0, 1.0);
        assert!(
            high > low,
            "maintaining more order should cost more, not less"
        );
    }

    #[test]
    fn prigogine_cost_zero_when_no_order_maintained() {
        assert_eq!(prigogine_dissipation_cost(0.0, 1.0, 1.0), 0.0);
    }
}
