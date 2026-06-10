// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Staged microbial colonization schedule for the artificial womb.
//!
//! In natural birth, the gut microbiome is seeded during vaginal delivery.
//! Ectogenesis requires deliberate staged introduction of commensal bacteria
//! to establish a healthy microbiome before term.

use serde::{Deserialize, Serialize};

use crate::types::GestationalWeek;

/// A single colonization stage: which bacteria and how many.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColonizationStage {
    /// Gestational week at which this stage begins.
    pub week: GestationalWeek,
    /// Bacterial species/strains to introduce.
    pub bacteria: Vec<String>,
    /// Target colony-forming units (CFU).
    pub colony_forming_units: f64,
    /// Description of the colonization goal.
    pub purpose: String,
}

/// Evidence-based microbial colonization schedule.
///
/// Provides a 4-stage introduction of commensal bacteria timed to
/// gut development milestones.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicrobiomeSchedule {
    /// Ordered colonization stages.
    pub stages: Vec<ColonizationStage>,
}

impl MicrobiomeSchedule {
    /// Create the standard evidence-based colonization schedule.
    ///
    /// Stage 1 (week 28): Pioneer species for initial gut colonization.
    /// Stage 2 (week 32): Diversification with short-chain fatty acid producers.
    /// Stage 3 (week 36): Immune-priming species.
    /// Stage 4 (week 38): Full community establishment pre-birth.
    pub fn standard() -> Self {
        Self {
            stages: vec![
                ColonizationStage {
                    week: GestationalWeek::new(28),
                    bacteria: vec![
                        "Bifidobacterium longum".into(),
                        "Lactobacillus rhamnosus".into(),
                    ],
                    colony_forming_units: 1e6,
                    purpose: "Pioneer colonization: establish initial commensal community".into(),
                },
                ColonizationStage {
                    week: GestationalWeek::new(32),
                    bacteria: vec![
                        "Bifidobacterium breve".into(),
                        "Lactobacillus reuteri".into(),
                        "Bacteroides fragilis".into(),
                    ],
                    colony_forming_units: 1e7,
                    purpose: "Diversification: SCFA production and mucosal barrier".into(),
                },
                ColonizationStage {
                    week: GestationalWeek::new(36),
                    bacteria: vec![
                        "Lactobacillus acidophilus".into(),
                        "Bifidobacterium infantis".into(),
                        "Faecalibacterium prausnitzii".into(),
                    ],
                    colony_forming_units: 1e8,
                    purpose: "Immune priming: Treg induction and IgA stimulation".into(),
                },
                ColonizationStage {
                    week: GestationalWeek::new(38),
                    bacteria: vec![
                        "Escherichia coli Nissle 1917".into(),
                        "Akkermansia muciniphila".into(),
                        "Roseburia intestinalis".into(),
                    ],
                    colony_forming_units: 1e9,
                    purpose: "Full community: mature microbiome for post-natal transition".into(),
                },
            ],
        }
    }

    /// Get the colonization stage appropriate for the given week.
    ///
    /// Returns the most recent stage whose start week is <= the given week.
    pub fn stage_for_week(&self, week: GestationalWeek) -> Option<&ColonizationStage> {
        self.stages.iter().rev().find(|s| s.week <= week)
    }

    /// Whether all stages have been reached.
    pub fn all_stages_reached(&self, current_week: GestationalWeek) -> bool {
        self.stages
            .last()
            .map(|s| current_week >= s.week)
            .unwrap_or(true)
    }

    /// Total number of unique species across all stages.
    pub fn total_species(&self) -> usize {
        let mut all: Vec<&str> = self
            .stages
            .iter()
            .flat_map(|s| s.bacteria.iter().map(|b| b.as_str()))
            .collect();
        all.sort();
        all.dedup();
        all.len()
    }
}

impl Default for MicrobiomeSchedule {
    fn default() -> Self {
        Self::standard()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_schedule_has_4_stages() {
        let schedule = MicrobiomeSchedule::standard();
        assert_eq!(schedule.stages.len(), 4);
    }

    #[test]
    fn test_stages_chronologically_ordered() {
        let schedule = MicrobiomeSchedule::standard();
        for i in 1..schedule.stages.len() {
            assert!(
                schedule.stages[i].week > schedule.stages[i - 1].week,
                "Stages must be chronologically ordered"
            );
        }
    }

    #[test]
    fn test_stage_for_week_before_first() {
        let schedule = MicrobiomeSchedule::standard();
        let stage = schedule.stage_for_week(GestationalWeek::new(20));
        assert!(stage.is_none(), "No stage before first colonization");
    }

    #[test]
    fn test_stage_for_week_at_stage() {
        let schedule = MicrobiomeSchedule::standard();
        let stage = schedule.stage_for_week(GestationalWeek::new(28));
        assert!(stage.is_some());
        assert_eq!(stage.unwrap().week, GestationalWeek::new(28));
    }

    #[test]
    fn test_stage_for_week_between_stages() {
        let schedule = MicrobiomeSchedule::standard();
        let stage = schedule.stage_for_week(GestationalWeek::new(30));
        assert!(stage.is_some());
        // Should return stage at week 28 (most recent)
        assert_eq!(stage.unwrap().week, GestationalWeek::new(28));
    }

    #[test]
    fn test_stage_for_week_at_last() {
        let schedule = MicrobiomeSchedule::standard();
        let stage = schedule.stage_for_week(GestationalWeek::new(40));
        assert!(stage.is_some());
        assert_eq!(stage.unwrap().week, GestationalWeek::new(38));
    }

    #[test]
    fn test_all_stages_reached() {
        let schedule = MicrobiomeSchedule::standard();
        assert!(!schedule.all_stages_reached(GestationalWeek::new(30)));
        assert!(schedule.all_stages_reached(GestationalWeek::new(38)));
        assert!(schedule.all_stages_reached(GestationalWeek::new(40)));
    }

    #[test]
    fn test_total_species_unique() {
        let schedule = MicrobiomeSchedule::standard();
        let total = schedule.total_species();
        assert!(
            total >= 8,
            "Should have at least 8 unique species, got {total}"
        );
    }

    #[test]
    fn test_cfu_increases_across_stages() {
        let schedule = MicrobiomeSchedule::standard();
        for i in 1..schedule.stages.len() {
            assert!(
                schedule.stages[i].colony_forming_units
                    > schedule.stages[i - 1].colony_forming_units,
                "CFU should increase across stages"
            );
        }
    }

    #[test]
    fn test_colonization_stage_serde_roundtrip() {
        let schedule = MicrobiomeSchedule::standard();
        let json = serde_json::to_string(&schedule).unwrap();
        let deser: MicrobiomeSchedule = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.stages.len(), schedule.stages.len());
        assert_eq!(
            deser.stages[0].bacteria.len(),
            schedule.stages[0].bacteria.len()
        );
    }
}
