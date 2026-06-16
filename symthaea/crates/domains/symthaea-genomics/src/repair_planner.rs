// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Repair planning for degraded DNA samples.
//!
//! Given damage analysis and quality assessment, the planner determines which
//! regions can be repaired and which strategies to use. The confidence of
//! each repair inversely correlates with damage severity.

use crate::types::{
    DamageEvent, DamageType, DegradedSample, GenomeQuality, RepairPlan, RepairRegion,
    RepairStrategy,
};

/// Repair planner that determines strategies for fixing damaged DNA.
#[derive(Debug, Clone)]
pub struct RepairPlanner {
    /// Maximum gap size (in bases) that can be filled synthetically (default 100).
    pub max_gap_for_synthetic: usize,
    /// Minimum confidence to attempt a repair (default 0.6).
    pub min_confidence: f64,
}

impl Default for RepairPlanner {
    fn default() -> Self {
        Self {
            max_gap_for_synthetic: 100,
            min_confidence: 0.6,
        }
    }
}

impl RepairPlanner {
    /// Create a new planner with custom parameters.
    pub fn new(max_gap_for_synthetic: usize, min_confidence: f64) -> Self {
        Self {
            max_gap_for_synthetic,
            min_confidence,
        }
    }

    /// Plan repairs for a degraded sample.
    ///
    /// Examines each damage event, determines whether it is repairable, selects
    /// a strategy, and computes confidence. Larger damage events and strand
    /// breaks with no coverage are marked as irreparable gaps.
    pub fn plan_repairs(&self, sample: &DegradedSample, quality: &GenomeQuality) -> RepairPlan {
        let mut repairable_regions = Vec::new();
        let mut irreparable_gaps: Vec<(usize, usize)> = Vec::new();
        let mut synthetic_fill_needed = 0usize;

        for event in &sample.damage_map {
            let repairable = self.is_repairable(event);

            if !repairable {
                // Determine gap extent. For strand breaks, the gap is typically
                // a few bases around the break site. For others, just the single position.
                let gap_size = match event.damage_type {
                    DamageType::StrandBreak => 5, // Typical gap around a break
                    DamageType::Crosslink => 3,
                    _ => 1,
                };
                let end = (event.position + gap_size).min(sample.sequence.len());
                irreparable_gaps.push((event.position, end));
                continue;
            }

            let strategy = self.select_strategy(event, quality);
            let confidence = self.compute_confidence(event, quality);

            if confidence < self.min_confidence {
                let end = (event.position + 1).min(sample.sequence.len());
                irreparable_gaps.push((event.position, end));
                continue;
            }

            let region_size = match event.damage_type {
                DamageType::Deamination => 1,
                DamageType::Depurination => 1,
                DamageType::StrandBreak => 5,
                DamageType::Crosslink => 3,
                DamageType::Oxidation => 1,
            };
            let end = (event.position + region_size).min(sample.sequence.len());

            if strategy == RepairStrategy::SyntheticFillIn {
                let fill_size = end - event.position;
                if fill_size > self.max_gap_for_synthetic {
                    irreparable_gaps.push((event.position, end));
                    continue;
                }
                synthetic_fill_needed += fill_size;
            }

            repairable_regions.push(RepairRegion {
                start: event.position,
                end,
                damage_type: event.damage_type,
                repair_strategy: strategy,
                confidence,
            });
        }

        // Merge overlapping irreparable gaps
        irreparable_gaps.sort_by_key(|g| g.0);
        let merged_gaps = Self::merge_gaps(&irreparable_gaps);

        // Overall confidence: geometric mean of individual repair confidences,
        // penalized by the fraction of irreparable damage.
        let overall_confidence = if repairable_regions.is_empty() {
            if sample.damage_map.is_empty() {
                1.0 // No damage = perfect confidence
            } else {
                0.0 // All damage is irreparable
            }
        } else {
            let product: f64 = repairable_regions.iter().map(|r| r.confidence).product();
            let geo_mean = product.powf(1.0 / repairable_regions.len() as f64);
            let repair_fraction = repairable_regions.len() as f64
                / (repairable_regions.len() + merged_gaps.len()).max(1) as f64;
            geo_mean * repair_fraction
        };

        RepairPlan {
            repairable_regions,
            irreparable_gaps: merged_gaps,
            synthetic_fill_needed,
            confidence: overall_confidence.clamp(0.0, 1.0),
        }
    }

    /// Determine whether a damage event is repairable.
    ///
    /// - Deamination is always repairable (simple base reversion).
    /// - Oxidation is usually repairable.
    /// - Depurination is repairable if severity < 0.8.
    /// - Strand breaks are repairable if severity < 0.5.
    /// - Crosslinks with severity > 0.7 are irreparable.
    pub fn is_repairable(&self, damage: &DamageEvent) -> bool {
        match damage.damage_type {
            DamageType::Deamination => true,
            DamageType::Oxidation => damage.severity < 0.9,
            DamageType::Depurination => damage.severity < 0.8,
            DamageType::StrandBreak => damage.severity < 0.5,
            DamageType::Crosslink => damage.severity < 0.7,
        }
    }

    /// Select the best repair strategy for a damage event.
    fn select_strategy(&self, event: &DamageEvent, quality: &GenomeQuality) -> RepairStrategy {
        match event.damage_type {
            DamageType::Deamination => RepairStrategy::DeaminationRevert,
            DamageType::Depurination => {
                if quality.mean_coverage >= 3.0 {
                    RepairStrategy::ConsensusVoting
                } else {
                    RepairStrategy::ReferenceGuided
                }
            }
            DamageType::StrandBreak => {
                if quality.mean_coverage >= 5.0 {
                    RepairStrategy::ConsensusVoting
                } else {
                    RepairStrategy::SyntheticFillIn
                }
            }
            DamageType::Crosslink => RepairStrategy::ReferenceGuided,
            DamageType::Oxidation => {
                if quality.mean_coverage >= 3.0 {
                    RepairStrategy::ConsensusVoting
                } else {
                    RepairStrategy::ReferenceGuided
                }
            }
        }
    }

    /// Compute confidence for a repair, based on damage severity and assembly quality.
    fn compute_confidence(&self, event: &DamageEvent, quality: &GenomeQuality) -> f64 {
        // Base confidence inversely proportional to severity
        let severity_factor = 1.0 - event.severity;

        // Coverage factor: more coverage = higher confidence
        let coverage_factor = (quality.mean_coverage / 10.0).min(1.0);

        // Type-specific bonus
        let type_bonus = match event.damage_type {
            DamageType::Deamination => 0.2, // Well-understood damage
            DamageType::Oxidation => 0.1,
            DamageType::Depurination => 0.0,
            DamageType::StrandBreak => -0.1,
            DamageType::Crosslink => -0.2,
        };

        (severity_factor * 0.5 + coverage_factor * 0.3 + 0.2 + type_bonus).clamp(0.0, 1.0)
    }

    /// Merge overlapping gap intervals.
    fn merge_gaps(gaps: &[(usize, usize)]) -> Vec<(usize, usize)> {
        if gaps.is_empty() {
            return Vec::new();
        }

        let mut merged = vec![gaps[0]];
        for &(start, end) in &gaps[1..] {
            let last = merged.last_mut().expect("merged initialized with gaps[0]");
            if start <= last.1 {
                last.1 = last.1.max(end);
            } else {
                merged.push((start, end));
            }
        }
        merged
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DnaSequence;

    fn sample_quality() -> GenomeQuality {
        GenomeQuality {
            n50: 500,
            total_length: 1000,
            num_contigs: 2,
            mean_coverage: 10.0,
            completeness: 0.95,
            error_rate: 0.01,
        }
    }

    fn make_sample(damage: Vec<DamageEvent>) -> DegradedSample {
        DegradedSample {
            sequence: DnaSequence::from_str(&"A".repeat(1000)),
            estimated_age_years: 5000.0,
            storage_temp_celsius: 15.0,
            contamination_estimate: 0.05,
            damage_map: damage,
        }
    }

    #[test]
    fn test_deamination_always_repairable() {
        let planner = RepairPlanner::default();
        let event = DamageEvent {
            position: 50,
            damage_type: DamageType::Deamination,
            severity: 0.99,
        };
        assert!(
            planner.is_repairable(&event),
            "Deamination should always be repairable"
        );
    }

    #[test]
    fn test_severe_strand_break_not_repairable() {
        let planner = RepairPlanner::default();
        let event = DamageEvent {
            position: 50,
            damage_type: DamageType::StrandBreak,
            severity: 0.9,
        };
        assert!(
            !planner.is_repairable(&event),
            "Severe strand break should not be repairable"
        );
    }

    #[test]
    fn test_severe_crosslink_not_repairable() {
        let planner = RepairPlanner::default();
        let event = DamageEvent {
            position: 50,
            damage_type: DamageType::Crosslink,
            severity: 0.8,
        };
        assert!(
            !planner.is_repairable(&event),
            "Severe crosslink should not be repairable"
        );
    }

    #[test]
    fn test_plan_repairs_deamination() {
        let planner = RepairPlanner::default();
        let sample = make_sample(vec![DamageEvent {
            position: 50,
            damage_type: DamageType::Deamination,
            severity: 0.3,
        }]);
        let quality = sample_quality();

        let plan = planner.plan_repairs(&sample, &quality);

        assert_eq!(plan.repairable_regions.len(), 1);
        assert_eq!(
            plan.repairable_regions[0].repair_strategy,
            RepairStrategy::DeaminationRevert
        );
        assert!(plan.irreparable_gaps.is_empty());
        assert!(plan.confidence > 0.5);
    }

    #[test]
    fn test_plan_repairs_irreparable_gap() {
        let planner = RepairPlanner::default();
        let sample = make_sample(vec![DamageEvent {
            position: 50,
            damage_type: DamageType::StrandBreak,
            severity: 0.9, // Very severe
        }]);
        let quality = sample_quality();

        let plan = planner.plan_repairs(&sample, &quality);

        assert!(
            !plan.irreparable_gaps.is_empty(),
            "Severe strand break should produce irreparable gap"
        );
    }

    #[test]
    fn test_plan_repairs_no_damage() {
        let planner = RepairPlanner::default();
        let sample = make_sample(vec![]);
        let quality = sample_quality();

        let plan = planner.plan_repairs(&sample, &quality);

        assert!(plan.repairable_regions.is_empty());
        assert!(plan.irreparable_gaps.is_empty());
        assert!(
            (plan.confidence - 1.0).abs() < 0.01,
            "No damage = full confidence"
        );
    }

    #[test]
    fn test_confidence_tracks_severity() {
        let planner = RepairPlanner::default();
        let quality = sample_quality();

        let mild = make_sample(vec![DamageEvent {
            position: 50,
            damage_type: DamageType::Deamination,
            severity: 0.1,
        }]);
        let severe = make_sample(vec![DamageEvent {
            position: 50,
            damage_type: DamageType::Deamination,
            severity: 0.9,
        }]);

        let mild_plan = planner.plan_repairs(&mild, &quality);
        let severe_plan = planner.plan_repairs(&severe, &quality);

        assert!(
            mild_plan.confidence > severe_plan.confidence,
            "Mild damage should have higher confidence: mild={}, severe={}",
            mild_plan.confidence,
            severe_plan.confidence
        );
    }

    #[test]
    fn test_merge_gaps() {
        let gaps = vec![(1, 5), (3, 8), (10, 15), (14, 20)];
        let merged = RepairPlanner::merge_gaps(&gaps);
        assert_eq!(merged, vec![(1, 8), (10, 20)]);
    }

    #[test]
    fn test_merge_gaps_empty() {
        let merged = RepairPlanner::merge_gaps(&[]);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_merge_gaps_no_overlap() {
        let gaps = vec![(1, 3), (5, 7), (10, 12)];
        let merged = RepairPlanner::merge_gaps(&gaps);
        assert_eq!(merged.len(), 3);
    }

    #[test]
    fn test_low_confidence_becomes_irreparable() {
        let planner = RepairPlanner::new(100, 0.99); // Very high threshold

        let quality = GenomeQuality {
            n50: 100,
            total_length: 200,
            num_contigs: 5,
            mean_coverage: 1.0, // Low coverage
            completeness: 0.5,
            error_rate: 0.3,
        };

        let sample = make_sample(vec![DamageEvent {
            position: 50,
            damage_type: DamageType::Depurination,
            severity: 0.7,
        }]);

        let plan = planner.plan_repairs(&sample, &quality);

        // With low coverage and high severity, confidence should be below threshold
        assert!(
            !plan.irreparable_gaps.is_empty() || plan.repairable_regions.is_empty(),
            "Low confidence repair should be rejected"
        );
    }
}
