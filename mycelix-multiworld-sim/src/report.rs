// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Final civilization report generation: summary statistics, trajectories, and
//! human-readable output for the 150-year simulation.

use crate::config::TICKS_PER_YEAR;
use crate::epoch::EpochSnapshot;
use crate::events::{CivEvent, CivEventType};

use serde::{Deserialize, Serialize};

/// Names of the Eight Harmonies.
const HARMONY_NAMES: [&str; 8] = [
    "Resonant Coherence",
    "Pan-Sentient Flourishing",
    "Evolving Resonant Co-Creationism",
    "Radical Translucency",
    "Sacred Reciprocity",
    "Temporal Wholeness",
    "Emergent Ethical Wisdom",
    "Sacred Stillness",
];

/// Final output of a simulation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CivilizationReport {
    pub scenario_name: String,
    pub total_ticks: u32,
    pub duration_years: f64,
    pub seed: u64,

    // Final state
    pub final_population: usize,
    pub final_worlds: usize,
    pub final_cvs: f64,
    pub survived: bool,

    // Trajectories
    pub epoch_snapshots: Vec<EpochSnapshot>,
    pub checkpoint_results: Vec<(u32, String, bool, Vec<String>)>,
    pub checkpoints_passed: usize,
    pub checkpoints_failed: usize,

    // Component scores
    pub final_genetic_diversity: f64,
    pub final_economic_sustainability: f64,
    pub final_harmony_scores: [f64; 8],
    pub final_love_coherence: f64,
    pub final_max_oppression: f64,
    pub final_collective_phi: f64,

    // Psychological needs
    pub final_mean_allostatic_load: f64,
    pub final_mean_engagement: f64,
    pub thrill_incidents_total: usize,
    pub biohack_incidents_total: usize,

    // Key events
    pub total_events: usize,
    pub first_birth_tick: Option<u32>,
    pub first_constitution_tick: Option<u32>,
    pub first_trade_tick: Option<u32>,
    pub constitutional_crises: usize,
    pub civil_wars: usize,
    pub secessions: usize,
    pub turchin_transitions: usize,
    pub epidemics: usize,
    pub breakthroughs: usize,
    pub worlds_founded: usize,
    pub total_migrants: u64,
    pub faction_cycles: usize,
    pub peak_faction_count: usize,
    pub max_elite_persistence: f64,
    pub max_innovation_stagnation: f64,
    pub phi_trend_at_end: String,
    pub max_trauma: f64,
    pub speciation_events: usize,
    pub constitutional_calcification_events: usize,

    // Disaster statistics
    pub total_disasters: u64,
    pub carrington_events: u32,
    pub tech_milestones_achieved: usize,

    /// Phase 2c: Mycelix-specific attack-surface resilience, computed from
    /// end-of-sim agent state. `None` when no adversaries were injected
    /// (saves old reports from carrying irrelevant fields). See
    /// `red_team::MycelixResilience`.
    #[serde(default)]
    pub mycelix_resilience: Option<crate::red_team::MycelixResilience>,

    /// Geometric-mean CVS — "weakest link" variant from kosmic-lab K-index.
    /// Always computed alongside `final_cvs` (arithmetic). Strictly ≤
    /// arithmetic for the same inputs. A civilization with one collapsed
    /// dimension scores near 0 here, regardless of other strengths.
    #[serde(default)]
    pub final_cvs_geometric: f64,
}

impl CivilizationReport {
    /// Generate a human-readable multi-line summary of the simulation results.
    pub fn summary(&self) -> String {
        let mut s = String::with_capacity(2048);

        s.push_str("=== 150-YEAR CIVILIZATION REPORT ===\n");
        s.push_str(&format!(
            "Scenario: {} (seed: {})\n",
            self.scenario_name, self.seed
        ));
        s.push_str(&format!(
            "Duration: {:.1} years ({} ticks)\n",
            self.duration_years, self.total_ticks
        ));
        s.push('\n');

        let result_str = if self.survived {
            "SURVIVED"
        } else {
            "COLLAPSED"
        };
        s.push_str(&format!(
            "RESULT: {} (CVS: {:.3})\n",
            result_str, self.final_cvs
        ));
        s.push('\n');

        s.push_str(&format!(
            "Population: {} across {} worlds\n",
            self.final_population, self.final_worlds
        ));
        let total_checkpoints = self.checkpoints_passed + self.checkpoints_failed;
        s.push_str(&format!(
            "Checkpoints: {}/{} passed\n",
            self.checkpoints_passed, total_checkpoints
        ));
        s.push('\n');

        // Milestones
        let birth_str = self
            .first_birth_tick
            .map(|t| format!("Year {:.1}", t as f64 / TICKS_PER_YEAR as f64))
            .unwrap_or_else(|| "Never".into());
        let const_str = self
            .first_constitution_tick
            .map(|t| format!("Year {:.1}", t as f64 / TICKS_PER_YEAR as f64))
            .unwrap_or_else(|| "Never".into());
        let trade_str = self
            .first_trade_tick
            .map(|t| format!("Year {:.1}", t as f64 / TICKS_PER_YEAR as f64))
            .unwrap_or_else(|| "Never".into());
        s.push_str(&format!(
            "First birth: {} | Constitution: {} | First trade: {}\n",
            birth_str, const_str, trade_str
        ));
        s.push('\n');

        // Harmony scores
        s.push_str("HARMONY SCORES (Eight Harmonies):\n");
        for (i, name) in HARMONY_NAMES.iter().enumerate() {
            s.push_str(&format!(
                "  {}. {:<35} {:.3}\n",
                i + 1,
                name,
                self.final_harmony_scores[i]
            ));
        }
        s.push_str(&format!(
            "Love Coherence: {:.3}\n",
            self.final_love_coherence
        ));
        s.push('\n');

        // Viability components
        s.push_str("VIABILITY COMPONENTS:\n");
        s.push_str(&format!(
            "  Genetic diversity:  {:.3}\n",
            self.final_genetic_diversity
        ));
        s.push_str(&format!(
            "  Economic sustain:   {:.3}\n",
            self.final_economic_sustainability
        ));
        let harmony_mean: f64 = self.final_harmony_scores.iter().sum::<f64>() / 8.0;
        s.push_str(&format!("  Harmony mean:       {:.3}\n", harmony_mean));
        s.push_str(&format!(
            "  Oppression (inv):   {:.3}\n",
            1.0 - self.final_max_oppression
        ));
        s.push_str(&format!(
            "  Collective Phi:     {:.3}\n",
            self.final_collective_phi
        ));
        s.push_str(&format!("  CVS:                {:.3}\n", self.final_cvs));
        s.push('\n');

        // Psychological needs
        s.push_str("PSYCHOLOGICAL NEEDS:\n");
        s.push_str(&format!(
            "  Allostatic load:    {:.3}\n",
            self.final_mean_allostatic_load
        ));
        s.push_str(&format!(
            "  Engagement:         {:.3}\n",
            self.final_mean_engagement
        ));
        s.push_str(&format!(
            "  Thrill incidents:   {}\n",
            self.thrill_incidents_total
        ));
        s.push_str(&format!(
            "  Biohack incidents:  {}\n",
            self.biohack_incidents_total
        ));
        s.push('\n');

        // Key events
        s.push_str("KEY EVENTS:\n");
        s.push_str(&format!(
            "  Constitutional crises: {}\n",
            self.constitutional_crises
        ));
        s.push_str(&format!("  Civil wars: {}\n", self.civil_wars));
        s.push_str(&format!("  Secessions: {}\n", self.secessions));
        s.push_str(&format!(
            "  Turchin transitions: {}\n",
            self.turchin_transitions
        ));
        s.push_str(&format!("  Epidemics: {}\n", self.epidemics));
        s.push_str(&format!("  Breakthroughs: {}\n", self.breakthroughs));
        s.push_str(&format!("  Worlds founded: {}\n", self.worlds_founded));
        s.push_str(&format!("  Total migrants: {}\n", self.total_migrants));

        s
    }

    /// Extended summary for 777+ year reports with faction analysis.
    pub fn summary_extended(&self) -> String {
        let mut s = self.summary();
        if self.duration_years >= 300.0 {
            s.push_str(
                "
FACTION ANALYSIS:
",
            );
            s.push_str(&format!(
                "  Faction cycles:         {}
",
                self.faction_cycles
            ));
            s.push_str(&format!(
                "  Peak faction count:     {}
",
                self.peak_faction_count
            ));
            s.push_str(&format!(
                "  Max elite persistence:  {:.3}
",
                self.max_elite_persistence
            ));
            s.push_str(&format!(
                "  Max innovation stag:    {:.3}
",
                self.max_innovation_stagnation
            ));
            s.push_str(&format!(
                "  Phi trend at end:       {}
",
                self.phi_trend_at_end
            ));
            s.push_str(&format!(
                "  Max trauma:             {:.3}
",
                self.max_trauma
            ));
            s.push_str(&format!(
                "  Speciation events:      {}
",
                self.speciation_events
            ));
            s.push_str(&format!(
                "  Calcification events:   {}
",
                self.constitutional_calcification_events
            ));
            s.push_str("\nDISASTER ENGINE:\n");
            s.push_str(&format!(
                "  Total disasters:        {}\n",
                self.total_disasters
            ));
            s.push_str(&format!(
                "  Carrington events:      {}\n",
                self.carrington_events
            ));
            s.push_str(&format!(
                "  Tech milestones:        {}\n",
                self.tech_milestones_achieved
            ));
        }
        s
    }

    /// Build a report from simulation state.
    pub fn build(
        scenario_name: &str,
        seed: u64,
        total_ticks: u32,
        final_population: usize,
        final_worlds: usize,
        final_cvs: f64,
        epoch_snapshots: Vec<EpochSnapshot>,
        checkpoint_results: Vec<(u32, String, bool, Vec<String>)>,
        events: &[CivEvent],
        genetic_diversity: f64,
        economic_sustainability: f64,
        harmony_scores: [f64; 8],
        love_coherence: f64,
        max_oppression: f64,
        collective_phi: f64,
        first_birth_tick: Option<u32>,
        first_constitution_tick: Option<u32>,
        first_trade_tick: Option<u32>,
        mean_allostatic_load: f64,
        mean_engagement: f64,
    ) -> Self {
        let checkpoints_passed = checkpoint_results.iter().filter(|(_, _, p, _)| *p).count();
        let checkpoints_failed = checkpoint_results.iter().filter(|(_, _, p, _)| !*p).count();

        let constitutional_crises = events
            .iter()
            .filter(|e| e.event_type == CivEventType::ConstitutionalAmendment)
            .count();
        let civil_wars = events
            .iter()
            .filter(|e| e.description.contains("CIVIL WAR"))
            .count();
        let secessions = events
            .iter()
            .filter(|e| e.description.contains("SECESSION"))
            .count();
        let turchin_transitions = events
            .iter()
            .filter(|e| e.description.contains("Turchin cycle"))
            .count();
        let epidemics = events
            .iter()
            .filter(|e| e.event_type == CivEventType::EpidemicStart)
            .count();
        let breakthroughs = events
            .iter()
            .filter(|e| e.event_type == CivEventType::InnovationBreakthrough)
            .count();
        let worlds_founded = events
            .iter()
            .filter(|e| e.event_type == CivEventType::WorldFounded)
            .count();
        let total_migrants = events
            .iter()
            .filter(|e| e.event_type == CivEventType::Migration)
            .count() as u64;
        let thrill_incidents_total = events
            .iter()
            .filter(|e| e.event_type == CivEventType::ThrillIncident)
            .count();
        let biohack_incidents_total = events
            .iter()
            .filter(|e| e.event_type == CivEventType::BiohackIncident)
            .count();

        Self {
            scenario_name: scenario_name.into(),
            total_ticks,
            duration_years: total_ticks as f64 / TICKS_PER_YEAR as f64,
            seed,
            final_population,
            final_worlds,
            final_cvs,
            survived: final_cvs > 0.3,
            epoch_snapshots,
            checkpoint_results,
            checkpoints_passed,
            checkpoints_failed,
            final_genetic_diversity: genetic_diversity,
            final_economic_sustainability: economic_sustainability,
            final_harmony_scores: harmony_scores,
            final_love_coherence: love_coherence,
            final_max_oppression: max_oppression,
            final_collective_phi: collective_phi,
            final_mean_allostatic_load: mean_allostatic_load,
            final_mean_engagement: mean_engagement,
            thrill_incidents_total,
            biohack_incidents_total,
            total_events: events.len(),
            first_birth_tick,
            first_constitution_tick,
            first_trade_tick,
            constitutional_crises,
            civil_wars,
            secessions,
            turchin_transitions,
            epidemics,
            breakthroughs,
            worlds_founded,
            total_migrants,
            faction_cycles: events
                .iter()
                .filter(|e| e.event_type == CivEventType::FactionDissolved)
                .count(),
            peak_faction_count: events
                .iter()
                .filter(|e| e.event_type == CivEventType::FactionEmerged)
                .count(),
            max_elite_persistence: 0.0,
            max_innovation_stagnation: 0.0,
            phi_trend_at_end: String::new(),
            max_trauma: 0.0,
            speciation_events: events
                .iter()
                .filter(|e| e.event_type == CivEventType::GeneticSpeciation)
                .count(),
            constitutional_calcification_events: events
                .iter()
                .filter(|e| e.event_type == CivEventType::ConstitutionalCalcification)
                .count(),
            total_disasters: 0,
            carrington_events: 0,
            tech_milestones_achieved: 0,
            mycelix_resilience: None,
            final_cvs_geometric: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epoch::EpochSnapshot;

    fn make_test_report(cvs: f64) -> CivilizationReport {
        CivilizationReport {
            scenario_name: "Test Scenario".into(),
            total_ticks: 1800,
            duration_years: 150.0,
            seed: 42,
            final_population: 50_000,
            final_worlds: 3,
            final_cvs: cvs,
            survived: cvs > 0.3,
            epoch_snapshots: vec![EpochSnapshot {
                epoch_id: 5,
                tick: 1800,
                total_population: 50_000,
                off_earth_population: 5_000,
                world_count: 3,
                mean_self_sufficiency: 0.85,
                mean_phi: 0.45,
                mean_love_coherence: 0.4,
                genetic_diversity: 0.7,
                max_oppression_index: 0.1,
                mean_tech_level: 0.8,
                civilization_viability_score: cvs,
                mean_allostatic_load: 0.15,
                mean_engagement: 0.75,
                faction_count: 0,
                elite_persistence: 0.0,
                amendment_rate: 0.0,
                innovation_stagnation: 0.0,
                inter_world_divergence: 0.0,
                consciousness_gini: 0.0,
                phi_trend: "Rising".into(),
                recovery_count: 0,
                trauma_level: 0.0,
                speciation_index: 0.0,
                worlds: Vec::new(),
                ethics_mean: [0.4; 4],
                ethics_std: [0.1; 4],
                ethics_diversity: 0.5,
            }],
            checkpoint_results: vec![
                (60, "Base assembled".into(), true, vec![]),
                (168, "Seeds end".into(), false, vec!["Pop too low".into()]),
            ],
            checkpoints_passed: 1,
            checkpoints_failed: 1,
            final_genetic_diversity: 0.7,
            final_economic_sustainability: 0.85,
            final_harmony_scores: [0.5; 8],
            final_love_coherence: 0.4,
            final_max_oppression: 0.1,
            final_collective_phi: 0.45,
            final_mean_allostatic_load: 0.15,
            final_mean_engagement: 0.75,
            thrill_incidents_total: 5,
            biohack_incidents_total: 0,
            total_events: 10_000,
            first_birth_tick: Some(200),
            first_constitution_tick: Some(300),
            first_trade_tick: Some(540),
            constitutional_crises: 2,
            epidemics: 3,
            breakthroughs: 5,
            worlds_founded: 3,
            total_migrants: 150,
            faction_cycles: 2,
            peak_faction_count: 4,
            max_elite_persistence: 0.1,
            max_innovation_stagnation: 0.2,
            phi_trend_at_end: "Rising".into(),
            max_trauma: 0.15,
            speciation_events: 0,
            constitutional_calcification_events: 1,
            total_disasters: 42,
            carrington_events: 0,
            tech_milestones_achieved: 3,
            civil_wars: 0,
            secessions: 0,
            turchin_transitions: 0,
            mycelix_resilience: None,
            final_cvs_geometric: 0.0,
        }
    }

    #[test]
    fn test_summary_contains_key_data() {
        let report = make_test_report(0.65);
        let summary = report.summary();

        assert!(
            summary.contains("Test Scenario"),
            "Should contain scenario name"
        );
        assert!(summary.contains("seed: 42"), "Should contain seed");
        assert!(
            summary.contains("SURVIVED"),
            "CVS > 0.3 should show SURVIVED"
        );
        assert!(summary.contains("50000"), "Should contain population");
        assert!(summary.contains("3 worlds"), "Should contain world count");
        assert!(
            summary.contains("1/2 passed"),
            "Should show checkpoint ratio"
        );
        assert!(
            summary.contains("Resonant Coherence"),
            "Should contain harmony names"
        );
        assert!(
            summary.contains("Love Coherence"),
            "Should contain love coherence"
        );
        assert!(summary.contains("CVS"), "Should contain CVS label");
        assert!(
            summary.contains("Epidemics: 3"),
            "Should contain epidemic count"
        );
    }

    #[test]
    fn test_cvs_threshold_determines_survival() {
        let survived = make_test_report(0.5);
        assert!(survived.survived, "CVS 0.5 should survive");

        let collapsed = make_test_report(0.2);
        assert!(!collapsed.survived, "CVS 0.2 should collapse");

        let borderline = make_test_report(0.3);
        assert!(
            !borderline.survived,
            "CVS 0.3 exactly should collapse (> not >=)"
        );

        let just_above = make_test_report(0.301);
        assert!(just_above.survived, "CVS 0.301 should survive");
    }

    #[test]
    fn test_report_build_from_events() {
        let events = vec![
            CivEvent::new(0, Some(0), CivEventType::WorldFounded, "Earth founded"),
            CivEvent::new(100, Some(0), CivEventType::Birth, "Agent born"),
            CivEvent::new(200, Some(0), CivEventType::EpidemicStart, "Flu outbreak"),
            CivEvent::new(300, None, CivEventType::Migration, "Migration event"),
            CivEvent::new(
                400,
                Some(0),
                CivEventType::InnovationBreakthrough,
                "New tech",
            ),
            CivEvent::new(480, Some(1), CivEventType::WorldFounded, "Mars founded"),
        ];

        let report = CivilizationReport::build(
            "Build Test",
            42,
            1800,
            1000,
            2,
            0.55,
            vec![],
            vec![(60, "Test".into(), true, vec![])],
            &events,
            0.7,
            0.8,
            [0.5; 8],
            0.4,
            0.1,
            0.3,
            Some(100),
            Some(250),
            Some(540),
            0.15,
            0.75,
        );

        assert_eq!(report.worlds_founded, 2);
        assert_eq!(report.epidemics, 1);
        assert_eq!(report.breakthroughs, 1);
        assert_eq!(report.total_migrants, 1);
        assert_eq!(report.checkpoints_passed, 1);
        assert_eq!(report.checkpoints_failed, 0);
        assert!(report.survived);
        assert_eq!(report.total_events, 6);
    }

    #[test]
    fn test_summary_shows_collapsed_for_low_cvs() {
        let report = make_test_report(0.1);
        let summary = report.summary();
        assert!(
            summary.contains("COLLAPSED"),
            "Low CVS should show COLLAPSED"
        );
    }

    #[test]
    fn test_milestone_year_display() {
        let report = make_test_report(0.5);
        let summary = report.summary();
        // First birth at tick 200 = year 16.7
        assert!(
            summary.contains("Year 16.7"),
            "Should display first birth year"
        );
    }
}
