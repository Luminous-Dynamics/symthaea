// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Dependency-free biodiversity accounting.
//!
//! These metrics summarize a non-negative abundance vector without inferring
//! species interactions or ecological mechanisms. Hill numbers make richness,
//! Shannon diversity, and inverse Simpson diversity comparable on an effective-
//! species scale.

use crate::error::{ModelError, require_finite, require_non_negative};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiodiversitySummary {
    pub total_abundance: f64,
    pub observed_richness: usize,
    pub shannon_entropy: f64,
    pub simpson_concentration: f64,
    pub simpson_diversity: f64,
    pub hill_q0: f64,
    pub hill_q1: f64,
    pub hill_q2: f64,
    pub pielou_evenness: Option<f64>,
    pub berger_parker_dominance: f64,
}

pub fn biodiversity_summary(abundances: &[f64]) -> Result<BiodiversitySummary, ModelError> {
    if abundances.is_empty() {
        return Err(ModelError::EmptySeries {
            series: "abundance vector",
        });
    }
    let mut total_abundance = 0.0;
    let mut observed_richness = 0usize;
    let mut maximum_abundance: f64 = 0.0;
    for &abundance in abundances {
        require_non_negative("abundance", abundance)?;
        total_abundance += abundance;
        require_finite("total_abundance", total_abundance)?;
        if abundance > 0.0 {
            observed_richness += 1;
            maximum_abundance = maximum_abundance.max(abundance);
        }
    }
    if total_abundance <= 0.0 {
        return Err(ModelError::NonPositive {
            parameter: "total_abundance",
            value: total_abundance,
        });
    }

    let mut shannon_entropy = 0.0;
    let mut simpson_concentration = 0.0;
    for &abundance in abundances {
        if abundance == 0.0 {
            continue;
        }
        let proportion = abundance / total_abundance;
        shannon_entropy -= proportion * proportion.ln();
        simpson_concentration += proportion * proportion;
    }
    simpson_concentration = simpson_concentration.clamp(0.0, 1.0);
    let hill_q0 = observed_richness as f64;
    let hill_q1 = shannon_entropy.exp();
    let hill_q2 = 1.0 / simpson_concentration;
    let pielou_evenness = if observed_richness > 1 {
        Some(shannon_entropy / hill_q0.ln())
    } else {
        None
    };
    Ok(BiodiversitySummary {
        total_abundance,
        observed_richness,
        shannon_entropy,
        simpson_concentration,
        simpson_diversity: 1.0 - simpson_concentration,
        hill_q0,
        hill_q1,
        hill_q2,
        pielou_evenness,
        berger_parker_dominance: maximum_abundance / total_abundance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_four_species_have_four_effective_species() {
        let summary = biodiversity_summary(&[10.0, 10.0, 10.0, 10.0]).unwrap();
        assert_eq!(summary.observed_richness, 4);
        assert!((summary.shannon_entropy - 4.0_f64.ln()).abs() < 1.0e-12);
        assert!((summary.hill_q1 - 4.0).abs() < 1.0e-12);
        assert!((summary.hill_q2 - 4.0).abs() < 1.0e-12);
        assert!((summary.pielou_evenness.unwrap() - 1.0).abs() < 1.0e-12);
        assert!((summary.berger_parker_dominance - 0.25).abs() < 1.0e-12);
    }

    #[test]
    fn metrics_are_invariant_to_common_abundance_scaling() {
        let first = biodiversity_summary(&[1.0, 2.0, 7.0]).unwrap();
        let second = biodiversity_summary(&[10.0, 20.0, 70.0]).unwrap();
        assert!((first.shannon_entropy - second.shannon_entropy).abs() < 1.0e-12);
        assert!((first.hill_q2 - second.hill_q2).abs() < 1.0e-12);
        assert!((first.berger_parker_dominance - second.berger_parker_dominance).abs() < 1.0e-12);
    }

    #[test]
    fn absent_species_do_not_inflate_richness() {
        let summary = biodiversity_summary(&[10.0, 0.0, 0.0]).unwrap();
        assert_eq!(summary.observed_richness, 1);
        assert_eq!(summary.hill_q0, 1.0);
        assert_eq!(summary.hill_q1, 1.0);
        assert_eq!(summary.hill_q2, 1.0);
        assert_eq!(summary.pielou_evenness, None);
    }

    #[test]
    fn zero_total_or_invalid_abundance_fails_closed() {
        assert!(biodiversity_summary(&[0.0, 0.0]).is_err());
        assert!(biodiversity_summary(&[1.0, -1.0]).is_err());
        assert!(biodiversity_summary(&[1.0, f64::NAN]).is_err());
    }
}
