// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Ancient DNA damage model based on established molecular biology.
//!
//! Models the four primary damage pathways in ancient/degraded DNA:
//! - **Deamination** (C->T / G->A): Most common, ~5.7e-6 per base per year at 15C
//! - **Depurination**: Loss of purine bases, ~4.0e-7 per base per year at 15C
//! - **Strand breaks**: Backbone breakage, ~1.0e-7 per base per year at 15C
//! - **Crosslinks**: Interstrand crosslinks, ~2.0e-8 per base per year at 15C
//!
//! All rates follow Arrhenius kinetics: rate = A * exp(-Ea / (R * T))
//! where T is absolute temperature (Kelvin).

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::types::{Base, DamageEvent, DamageType, DnaSequence};

/// Gas constant in eV/K (Boltzmann constant in eV units).
const KB_EV: f64 = 8.617e-5;

/// Reference temperature for baseline rates (15C = 288.15K).
const REFERENCE_TEMP_K: f64 = 288.15;

/// Activation energies (eV) for each damage pathway.
/// These are approximate values from aDNA literature.
const EA_DEAMINATION: f64 = 1.17;
const EA_DEPURINATION: f64 = 1.30;
const EA_STRAND_BREAK: f64 = 1.40;
const EA_CROSSLINK: f64 = 1.50;

/// Baseline rates at reference temperature (per base per year).
const RATE_DEAMINATION_REF: f64 = 5.7e-6;
const RATE_DEPURINATION_REF: f64 = 4.0e-7;
const RATE_STRAND_BREAK_REF: f64 = 1.0e-7;
const RATE_CROSSLINK_REF: f64 = 2.0e-8;

/// Ancient DNA damage model based on Arrhenius kinetics.
///
/// Models C->T deamination (most common), depurination, strand breaks, crosslinks.
/// Rates scale with temperature following the Arrhenius equation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DamageModel {
    /// Deamination rate (per base per year).
    pub deamination_rate: f64,
    /// Depurination rate (per base per year).
    pub depurination_rate: f64,
    /// Strand break rate (per base per year).
    pub strand_break_rate: f64,
    /// Crosslink rate (per base per year).
    pub crosslink_rate: f64,
}

impl DamageModel {
    /// Create a new damage model calibrated to the given temperature.
    ///
    /// Uses Arrhenius kinetics to scale rates from the reference temperature (15C).
    /// Higher temperatures dramatically increase all damage rates.
    pub fn new(temperature_celsius: f64) -> Self {
        let temp_k = temperature_celsius + 273.15;

        Self {
            deamination_rate: arrhenius_scale(RATE_DEAMINATION_REF, EA_DEAMINATION, temp_k),
            depurination_rate: arrhenius_scale(RATE_DEPURINATION_REF, EA_DEPURINATION, temp_k),
            strand_break_rate: arrhenius_scale(RATE_STRAND_BREAK_REF, EA_STRAND_BREAK, temp_k),
            crosslink_rate: arrhenius_scale(RATE_CROSSLINK_REF, EA_CROSSLINK, temp_k),
        }
    }

    /// Predict stochastic damage events for a sequence at a given age.
    ///
    /// Each base is independently tested against each damage pathway.
    /// Returns a list of damage events sorted by position.
    pub fn predict_damage(
        &self,
        sequence: &DnaSequence,
        age_years: f64,
        rng: &mut impl Rng,
    ) -> Vec<DamageEvent> {
        let mut events = Vec::new();

        for (pos, &base) in sequence.bases.iter().enumerate() {
            // Deamination: only affects C (->T) and G (->A on complementary strand)
            if base == Base::C || base == Base::G {
                let p_deamination = cumulative_probability(self.deamination_rate, age_years);
                if rng.r#gen::<f64>() < p_deamination {
                    events.push(DamageEvent {
                        position: pos,
                        damage_type: DamageType::Deamination,
                        severity: p_deamination.min(1.0),
                    });
                }
            }

            // Depurination: only affects purines (A, G)
            if base == Base::A || base == Base::G {
                let p_depurination = cumulative_probability(self.depurination_rate, age_years);
                if rng.r#gen::<f64>() < p_depurination {
                    events.push(DamageEvent {
                        position: pos,
                        damage_type: DamageType::Depurination,
                        severity: p_depurination.min(1.0),
                    });
                }
            }

            // Strand break: affects any base
            let p_break = cumulative_probability(self.strand_break_rate, age_years);
            if rng.r#gen::<f64>() < p_break {
                events.push(DamageEvent {
                    position: pos,
                    damage_type: DamageType::StrandBreak,
                    severity: p_break.min(1.0),
                });
            }

            // Crosslink: affects any base
            let p_crosslink = cumulative_probability(self.crosslink_rate, age_years);
            if rng.r#gen::<f64>() < p_crosslink {
                events.push(DamageEvent {
                    position: pos,
                    damage_type: DamageType::Crosslink,
                    severity: p_crosslink.min(1.0),
                });
            }
        }

        events.sort_by_key(|e| e.position);
        events
    }

    /// Expected fraction of C/G bases that will be deaminated at a given age.
    ///
    /// Uses the cumulative probability: P = 1 - exp(-rate * age).
    pub fn expected_deamination_fraction(&self, age_years: f64) -> f64 {
        cumulative_probability(self.deamination_rate, age_years)
    }

    /// Detect damage patterns from a set of reads.
    ///
    /// Looks for C->T enrichment at fragment ends (the classic aDNA signature).
    /// Also detects elevated rates of other substitution patterns.
    pub fn detect_damage_pattern(reads: &[DnaSequence]) -> Vec<DamageEvent> {
        let mut events = Vec::new();

        if reads.is_empty() {
            return events;
        }

        // Scan each read for the characteristic C->T pattern at ends
        // In real aDNA, C->T transitions are enriched in the first ~10 bases
        // and G->A transitions at the last ~10 bases.
        let end_window = 10;

        for read in reads {
            let len = read.len();
            if len < 2 {
                continue;
            }

            // Check 5' end for C->T substitutions (deamination signature)
            let check_len = end_window.min(len);
            for pos in 0..check_len {
                if read.bases[pos] == Base::T {
                    // This T could be a deaminated C. The probability increases
                    // as we get closer to the end.
                    let distance_from_end = pos as f64;
                    let severity = 1.0 / (1.0 + distance_from_end);
                    if severity > 0.3 {
                        events.push(DamageEvent {
                            position: pos,
                            damage_type: DamageType::Deamination,
                            severity,
                        });
                    }
                }
            }

            // Check 3' end for G->A substitutions (complementary deamination)
            let start = len.saturating_sub(end_window);
            for pos in start..len {
                if read.bases[pos] == Base::A {
                    let distance_from_end = (len - 1 - pos) as f64;
                    let severity = 1.0 / (1.0 + distance_from_end);
                    if severity > 0.3 {
                        events.push(DamageEvent {
                            position: pos,
                            damage_type: DamageType::Deamination,
                            severity,
                        });
                    }
                }
            }

            // Check for N bases (possible depurination/abasic sites)
            for (pos, &base) in read.bases.iter().enumerate() {
                if base == Base::N {
                    events.push(DamageEvent {
                        position: pos,
                        damage_type: DamageType::Depurination,
                        severity: 0.8,
                    });
                }
            }
        }

        events
    }
}

/// Compute Arrhenius-scaled rate from reference temperature.
///
/// rate(T) = rate_ref * exp(-Ea/kB * (1/T - 1/T_ref))
fn arrhenius_scale(rate_ref: f64, ea_ev: f64, temp_k: f64) -> f64 {
    let exponent = -(ea_ev / KB_EV) * (1.0 / temp_k - 1.0 / REFERENCE_TEMP_K);
    rate_ref * exponent.exp()
}

/// Cumulative probability of damage: P = 1 - exp(-rate * time).
fn cumulative_probability(rate: f64, time_years: f64) -> f64 {
    1.0 - (-rate * time_years).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_damage_model_at_reference_temp() {
        let model = DamageModel::new(15.0);
        // At reference temperature, rates should be close to reference values
        assert!(
            (model.deamination_rate - RATE_DEAMINATION_REF).abs() / RATE_DEAMINATION_REF < 0.01,
            "Deamination rate at 15C should match reference: {}",
            model.deamination_rate
        );
    }

    #[test]
    fn test_temperature_dependence() {
        let cold = DamageModel::new(5.0);
        let warm = DamageModel::new(25.0);

        // Higher temperature should increase all damage rates
        assert!(
            warm.deamination_rate > cold.deamination_rate,
            "Deamination should increase with temperature"
        );
        assert!(
            warm.depurination_rate > cold.depurination_rate,
            "Depurination should increase with temperature"
        );
        assert!(
            warm.strand_break_rate > cold.strand_break_rate,
            "Strand breaks should increase with temperature"
        );
        assert!(
            warm.crosslink_rate > cold.crosslink_rate,
            "Crosslinks should increase with temperature"
        );
    }

    #[test]
    fn test_damage_increases_with_age() {
        let model = DamageModel::new(15.0);
        let seq = DnaSequence::from_str("ATGCATGCATGCATGCATGCATGCATGCATGC");
        let mut rng = rand::thread_rng();

        // Run multiple trials for statistical reliability
        let mut young_count = 0;
        let mut old_count = 0;
        let trials = 50;

        for _ in 0..trials {
            let young_damage = model.predict_damage(&seq, 100.0, &mut rng);
            let old_damage = model.predict_damage(&seq, 100_000.0, &mut rng);
            young_count += young_damage.len();
            old_count += old_damage.len();
        }

        assert!(
            old_count > young_count,
            "Older samples should have more damage: young={}, old={}",
            young_count,
            old_count
        );
    }

    #[test]
    fn test_deamination_dominates() {
        let model = DamageModel::new(15.0);
        // Deamination should be the most common damage type
        assert!(model.deamination_rate > model.depurination_rate);
        assert!(model.deamination_rate > model.strand_break_rate);
        assert!(model.deamination_rate > model.crosslink_rate);
    }

    #[test]
    fn test_expected_deamination_fraction() {
        let model = DamageModel::new(15.0);

        let young = model.expected_deamination_fraction(100.0);
        let old = model.expected_deamination_fraction(50_000.0);

        assert!(young < old, "Older samples should have more deamination");
        assert!(young >= 0.0 && young <= 1.0, "Fraction must be in [0,1]");
        assert!(old >= 0.0 && old <= 1.0, "Fraction must be in [0,1]");
    }

    #[test]
    fn test_deamination_fraction_monotonic() {
        let model = DamageModel::new(15.0);
        let ages = [0.0, 100.0, 1_000.0, 10_000.0, 100_000.0];
        let fractions: Vec<f64> = ages
            .iter()
            .map(|&a| model.expected_deamination_fraction(a))
            .collect();

        for i in 1..fractions.len() {
            assert!(
                fractions[i] >= fractions[i - 1],
                "Deamination fraction should be monotonically increasing"
            );
        }
    }

    #[test]
    fn test_deamination_fraction_zero_at_age_zero() {
        let model = DamageModel::new(15.0);
        let frac = model.expected_deamination_fraction(0.0);
        assert!(
            frac.abs() < 1e-10,
            "At age 0, deamination fraction should be 0, got {}",
            frac
        );
    }

    #[test]
    fn test_detect_damage_pattern() {
        // Create reads with typical aDNA damage signature (T at 5' end)
        let reads = vec![
            DnaSequence::from_str("TAGCATGC"),
            DnaSequence::from_str("TTGCATGC"),
            DnaSequence::from_str("ATGCATGA"),
        ];
        let events = DamageModel::detect_damage_pattern(&reads);
        assert!(
            !events.is_empty(),
            "Should detect damage patterns in reads with end substitutions"
        );

        // Verify deamination is detected
        let deamination_events: Vec<_> = events
            .iter()
            .filter(|e| e.damage_type == DamageType::Deamination)
            .collect();
        assert!(
            !deamination_events.is_empty(),
            "Should detect deamination events"
        );
    }

    #[test]
    fn test_detect_damage_pattern_empty() {
        let events = DamageModel::detect_damage_pattern(&[]);
        assert!(events.is_empty());
    }

    #[test]
    fn test_detect_n_bases_as_depurination() {
        let reads = vec![DnaSequence::from_str("ATNCNATGC")];
        let events = DamageModel::detect_damage_pattern(&reads);
        let depurination_count = events
            .iter()
            .filter(|e| e.damage_type == DamageType::Depurination)
            .count();
        assert_eq!(
            depurination_count, 2,
            "Each N base should be detected as depurination"
        );
    }

    #[test]
    fn test_arrhenius_scale_identity() {
        // At reference temp, scaling should be ~1.0
        let scaled = arrhenius_scale(1.0, EA_DEAMINATION, REFERENCE_TEMP_K);
        assert!(
            (scaled - 1.0).abs() < 0.01,
            "At reference temp, scale should be 1.0, got {}",
            scaled
        );
    }

    #[test]
    fn test_cumulative_probability_bounds() {
        assert!((cumulative_probability(0.001, 0.0)).abs() < 1e-10);
        let p = cumulative_probability(0.001, 10_000.0);
        assert!(
            p >= 0.0 && p <= 1.0,
            "Probability must be in [0,1], got {}",
            p
        );
    }

    #[test]
    fn test_predict_damage_only_cg_deaminate() {
        let model = DamageModel::new(15.0);
        // Sequence with only A and T - should not produce deamination events
        let seq = DnaSequence::from_str("AATTAATT");
        let mut rng = rand::thread_rng();

        let mut deamination_count = 0;
        for _ in 0..100 {
            let events = model.predict_damage(&seq, 10_000.0, &mut rng);
            deamination_count += events
                .iter()
                .filter(|e| e.damage_type == DamageType::Deamination)
                .count();
        }
        assert_eq!(
            deamination_count, 0,
            "AT-only sequence should have no deamination events"
        );
    }
}
