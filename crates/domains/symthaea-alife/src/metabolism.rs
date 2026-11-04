// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Metabolism: dimensionless resource-cost policies for ALife simulations.
//!
//! This module applies Landauer- and Prigogine-inspired formula shapes as
//! configurable costs in the simulated organism energy budget. These are
//! model rules, not SI thermodynamic measurements or evidence that physical
//! memory erasure / entropy production has been measured.
//!
//! The coefficient used below is deliberately dimensionless and scaled to
//! this crate's resource units. Consequently the returned values are ALife
//! model costs, not joules and not physical lower bounds. Keep the legacy API
//! names for compatibility while preserving this claim boundary.

/// Dimensionless coefficient for the ALife information/order cost policy.
/// It was selected to make the added model cost affect simulated energy
/// budgets without overwhelming existing costs. It is not a rescaled SI
/// Boltzmann constant or a measured biological/physical parameter.
pub const K_ALIFE_BOLTZMANN: f64 = 0.0005;

/// Legacy Landauer-inspired model cost for a nominal information quantity
/// and a dimensionless effective temperature.
///
/// This is not an SI Landauer lower bound: the coefficient is dimensionless,
/// and the input is supplied by simulation telemetry rather than a measured
/// count of physical bit erasures. The result is an ALife resource cost, not
/// joules or a thermodynamic impossibility test. The legacy function name is
/// retained for source compatibility.
///
/// Negative or NaN bit counts are handled by the existing model policy and
/// the effective temperature is floored at 1e-6. These are simulation
/// safeguards, not statements about absolute-zero thermodynamics.
pub fn landauer_minimum(bits: f64, temperature: f64) -> f64 {
    bits.max(0.0) * K_ALIFE_BOLTZMANN * temperature.max(1e-6) * std::f64::consts::LN_2
}

/// Shannon entropy, in bits, of a probability distribution:
/// H(p) = -Σ p_i log2(p_i). It measures uncertainty in the modeled action
/// distribution; it does not establish that this many physical memory bits
/// were irreversibly erased when the agent selected an action. Using this
/// quantity in the ALife cost policy is a modeling choice. Terms with
/// p_i <= 0 contribute zero; the maximum is log2(n) for n equiprobable
/// outcomes.
pub fn shannon_entropy_bits(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.log2())
        .sum::<f64>()
        .max(0.0)
}

/// Prigogine-inspired quadratic model penalty for maintaining simulated order.
///
/// The current expression is dissipation_rate * order_maintained^2 *
/// temperature. It is an ALife scenario rule, not a derived entropy-production
/// measurement or a universal statement that far-from-equilibrium systems
/// minimize entropy production. The minimum-entropy-production result has
/// restricted near-equilibrium assumptions; this model does not estimate
/// conjugate fluxes and forces here.
///
/// order_maintained is expected in [0, 1]; this function does not clamp it.
pub fn prigogine_dissipation_cost(
    order_maintained: f64,
    temperature: f64,
    dissipation_rate: f64,
) -> f64 {
    dissipation_rate.max(0.0) * order_maintained.powi(2) * temperature.max(1e-6)
}

/// Modelled resolution capacity for a [0,1] observation with bucket width
/// grain: log2(1/grain). This estimates the number of distinguishable
/// buckets, not the number of physical bits erased by a memory-reset
/// operation. The value feeds the ALife dimensionless information-cost policy;
/// it is not a measured physical energy cost. grain is clamped away from zero.
pub fn perceptual_resolution_bits(grain: f64) -> f64 {
    (1.0 / grain.max(1e-6)).log2().max(0.0)
}

/// Quantize a `[0,1]` observation to the nearest multiple of `grain` -- the coarse-graining a
/// perceptual strategy actually applies to what it observes, before belief update ever sees it.
/// Companion to [`perceptual_resolution_bits`]: the same `grain` drives both how distorted the
/// observation is and how much it cost to have resolved that much of it.
pub fn quantize_to_grain(value: f64, grain: f64) -> f64 {
    let grain = grain.max(1e-6);
    ((value / grain).round() * grain).clamp(0.0, 1.0)
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

    #[test]
    fn resolution_bits_grows_as_grain_shrinks() {
        let coarse = perceptual_resolution_bits(0.5); // 2 buckets, 1 bit
        let fine = perceptual_resolution_bits(0.02); // 50 buckets, ~5.64 bits
        assert!(
            fine > coarse,
            "finer grain should cost more bits: coarse={coarse}, fine={fine}"
        );
        assert!(
            (coarse - 1.0).abs() < 1e-9,
            "log2(1/0.5) should be exactly 1 bit, got {coarse}"
        );
    }

    #[test]
    fn resolution_bits_never_negative() {
        assert!(perceptual_resolution_bits(2.0) >= 0.0); // grain > 1 -- fewer than 1 "bucket"
        assert!(perceptual_resolution_bits(1.0) >= 0.0);
    }

    #[test]
    fn quantize_stays_in_unit_range() {
        for grain in [0.02, 0.1, 0.4, 0.9] {
            for i in 0..=20 {
                let value = i as f64 / 20.0;
                let q = quantize_to_grain(value, grain);
                assert!(
                    (0.0..=1.0).contains(&q),
                    "grain={grain} value={value} -> {q}"
                );
            }
        }
    }

    #[test]
    fn coarse_quantization_collapses_distinct_values() {
        // A wide bucket should make two nearby values indistinguishable, where a narrow one
        // keeps them apart -- the actual coarse-graining Hoffman's "interface" strategies apply.
        let (a, b) = (0.40, 0.45);
        assert_eq!(
            quantize_to_grain(a, 0.5),
            quantize_to_grain(b, 0.5),
            "grain=0.5 should collapse 0.40 and 0.45 into the same bucket"
        );
        assert_ne!(
            quantize_to_grain(a, 0.02),
            quantize_to_grain(b, 0.02),
            "grain=0.02 should keep 0.40 and 0.45 distinguishable"
        );
    }
}
