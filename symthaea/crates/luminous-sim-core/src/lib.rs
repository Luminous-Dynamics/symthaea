// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Core traits and types for Luminous Dynamics simulation modules.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod module_trait;
pub mod report;

/// Unified simulation configuration (TOML-driven).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedConfig {
    #[serde(default)]
    pub simulation: SimulationSection,
    #[serde(default)]
    pub policy: PolicySection,
    #[serde(default)]
    pub params: ParamsSection,
    #[serde(default)]
    pub feedback_loops: FeedbackLoopsSection,
}

impl Default for UnifiedConfig {
    fn default() -> Self {
        Self {
            simulation: SimulationSection::default(),
            policy: PolicySection::default(),
            params: ParamsSection::default(),
            feedback_loops: FeedbackLoopsSection::default(),
        }
    }
}

impl UnifiedConfig {
    pub fn from_toml(s: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(s)
    }

    pub fn to_toml(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationSection {
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_years")]
    pub years: u32,
}

fn default_seed() -> u64 {
    42
}
fn default_years() -> u32 {
    150
}

impl Default for SimulationSection {
    fn default() -> Self {
        Self {
            seed: default_seed(),
            years: default_years(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PolicySection {
    pub trust_weighted_governance: Option<bool>,
    pub turchin_cycles_enabled: Option<bool>,
    pub fep_immiseration_enabled: Option<bool>,
    pub sacred_stillness_enabled: Option<bool>,
    pub disasters_enabled: Option<bool>,
    pub faction_enabled: Option<bool>,
    pub education_enabled: Option<bool>,
    pub migration_enabled: Option<bool>,
    pub hybrid_earth: Option<bool>,
    pub disasters_solar_enabled: Option<bool>,
    pub disasters_impact_enabled: Option<bool>,
    pub disasters_planetary_enabled: Option<bool>,
    pub disasters_eclss_enabled: Option<bool>,
    pub disasters_psychological_enabled: Option<bool>,
    pub disasters_technology_enabled: Option<bool>,
    pub disasters_civilization_enabled: Option<bool>,
    pub pair_bond_rate: Option<f64>,
    pub care_effectiveness: Option<f64>,
    pub trade_openness: Option<f64>,
    pub defense_spending: Option<f64>,
    pub exploration_investment: Option<f64>,
    pub birth_policy: Option<String>,
    pub project_strategy: Option<String>,
    pub resource_priority: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParamsSection {
    pub stress_baseline: Option<f64>,
    pub stress_isolation: Option<f64>,
    pub stress_overwork: Option<f64>,
    pub stress_decay: Option<f64>,
    pub burnout_threshold: Option<f64>,
    pub trauma_decay_base: Option<f64>,
    pub trauma_disaster_scale: Option<f64>,
    pub radiation_mars_sv_month: Option<f64>,
    pub radiation_moon_sv_month: Option<f64>,
    pub radiation_career_limit_sv: Option<f64>,
    pub spoilage_food: Option<f64>,
    pub spoilage_energy: Option<f64>,
    pub spoilage_water: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackLoopsSection {
    #[serde(default = "default_true")]
    pub disaster_trauma: bool,
    #[serde(default = "default_true")]
    pub harmony_policy: bool,
    #[serde(default = "default_true")]
    pub affect_behavior: bool,
    #[serde(default = "default_true")]
    pub consciousness_governance: bool,
    #[serde(default = "default_true")]
    pub cultural_memory: bool,
}

fn default_true() -> bool {
    true
}

impl Default for FeedbackLoopsSection {
    fn default() -> Self {
        Self {
            disaster_trauma: true,
            harmony_policy: true,
            affect_behavior: true,
            consciousness_governance: true,
            cultural_memory: true,
        }
    }
}

/// Standardized simulation report with JSON/Markdown export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardizedReport {
    pub executive_summary: report::ExecutiveSummary,
    pub module_reports: Vec<module_trait::ReportSection>,
    pub honest_assessment: HonestAssessment,
    pub reproducibility: report::ReproducibilityInfo,
}

impl StandardizedReport {
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn to_markdown(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("# {}\n\n", self.executive_summary.title));
        s.push_str(&format!(
            "**Outcome**: {}\n\n",
            self.executive_summary.outcome
        ));
        s.push_str("## Key Findings\n\n");
        for finding in &self.executive_summary.key_findings {
            s.push_str(&format!("- {}\n", finding));
        }
        s.push_str("\n## Honest Assessment\n\n");
        for lim in &self.honest_assessment.limitations {
            s.push_str(&format!("- {}\n", lim));
        }
        s
    }
}

/// Honest limitations and disclaimers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HonestAssessment {
    pub limitations: Vec<String>,
    pub cannot_predict: Vec<String>,
    pub confidence_level: String,
}
