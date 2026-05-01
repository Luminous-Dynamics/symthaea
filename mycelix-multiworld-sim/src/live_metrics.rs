// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Live metrics: per-tick dynamic probability output for the toggle matrix UI.
//!
//! Each tick, the simulator computes real-time probabilities and state variables
//! that reflect the current risk landscape. When mechanisms are toggled off,
//! their probabilities change — making the impact of each mechanism visible.

use crate::cliodynamics::{SecularCyclePhase, SecularCycleState};
use crate::config::PolicyConfig;
use serde::{Deserialize, Serialize};

/// Per-tick snapshot of dynamic probabilities and key state variables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveMetrics {
    pub tick: u32,
    pub year: f64,

    // === Turchin secular cycle state ===
    pub secular_phase: String,
    pub psi: f64,             // Elite overproduction [0, 5]
    pub immiseration: f64,    // W [0, 1]
    pub state_stability: f64, // S [0, 1]

    // === Dynamic probabilities (per tick) ===
    pub p_civil_war: f64,
    pub p_secession: f64,
    pub p_carrington: f64,
    pub p_eclss_failure: f64,
    pub p_kessler: f64,
    pub p_epidemic: f64,

    // === Aggregate civilization state ===
    pub cvs: f64,
    pub total_population: usize,
    pub collective_phi: f64,
    pub gini: f64,
    pub mean_allostatic_load: f64,
    pub infrastructure_mean: f64,

    // === Toggle state (which mechanisms are active) ===
    pub active_toggles: Vec<String>,
    pub toggle_count: usize,
}

impl LiveMetrics {
    /// Compute live metrics from current simulator state.
    pub fn compute(
        tick: u32,
        policy: &PolicyConfig,
        secular: Option<&SecularCycleState>,
        cvs: f64,
        total_population: usize,
        collective_phi: f64,
        gini: f64,
        mean_allostatic_load: f64,
        infrastructure_mean: f64,
        solar_cycle_max: bool,
        kessler_density: f64,
    ) -> Self {
        let year = tick as f64 / 12.0;

        // Extract Turchin state if available
        let (phase, psi, immiseration, stability) = match secular {
            Some(s) => (
                format!("{:?}", s.phase),
                s.psi,
                s.immiseration,
                s.state_stability,
            ),
            None => ("N/A".to_string(), 0.0, 0.0, 1.0),
        };

        // Compute dynamic probabilities
        let p_civil_war = if policy.turchin_cycles_enabled {
            if let Some(s) = secular {
                if s.phase == SecularCyclePhase::Crisis && s.state_stability < 0.3 {
                    0.05 // 5% per tick during crisis
                } else if s.phase == SecularCyclePhase::Crisis {
                    0.01 // 1% during non-severe crisis
                } else {
                    0.0
                }
            } else {
                0.0
            }
        } else {
            0.0
        };

        let p_secession = if policy.turchin_cycles_enabled {
            if let Some(s) = secular {
                if s.psi > 0.8 && s.state_stability < 0.4 {
                    0.01
                } else {
                    0.0
                }
            } else {
                0.0
            }
        } else {
            0.0
        };

        let p_carrington = if policy.disasters_enabled && policy.disasters_solar_enabled {
            if solar_cycle_max {
                0.002
            } else {
                0.0005
            }
        } else {
            0.0
        };

        let p_eclss_failure = if policy.disasters_enabled && policy.disasters_eclss_enabled {
            let base = 0.01; // ~1% base per tick
            let infra_factor = 1.0 + (1.0 - infrastructure_mean).max(0.0) * 2.0;
            base * infra_factor
        } else {
            0.0
        };

        let p_kessler = if policy.disasters_enabled && policy.disasters_civilization_enabled {
            0.001 * (kessler_density / 1.0).max(1.0).ln().max(0.0)
        } else {
            0.0
        };

        let p_epidemic = if policy.disasters_enabled && policy.disasters_psychological_enabled {
            if total_population > 100 {
                0.005
            } else {
                0.001
            }
        } else {
            0.0
        };

        // Collect active toggles
        let mut active = Vec::new();
        if policy.trust_weighted_governance {
            active.push("trust_weighting".to_string());
        }
        if policy.turchin_cycles_enabled {
            active.push("turchin_cycles".to_string());
        }
        if policy.fep_immiseration_enabled {
            active.push("fep_immiseration".to_string());
        }
        if policy.sacred_stillness_enabled {
            active.push("sacred_stillness".to_string());
        }
        if policy.disasters_enabled {
            active.push("disasters".to_string());
        }
        if policy.faction_enabled {
            active.push("factions".to_string());
        }
        if policy.education_enabled {
            active.push("education".to_string());
        }
        if policy.migration_enabled {
            active.push("migration".to_string());
        }
        if policy.amendment_enabled {
            active.push("amendments".to_string());
        }
        if policy.stoichiometry_enabled {
            active.push("stoichiometry".to_string());
        }
        if policy.epistemic_decay_enabled {
            active.push("epistemic_decay".to_string());
        }
        if policy.age_demographics_enabled {
            active.push("age_demographics".to_string());
        }
        let toggle_count = active.len();

        Self {
            tick,
            year,
            secular_phase: phase,
            psi,
            immiseration,
            state_stability: stability,
            p_civil_war,
            p_secession,
            p_carrington,
            p_eclss_failure,
            p_kessler,
            p_epidemic,
            cvs,
            total_population,
            collective_phi,
            gini,
            mean_allostatic_load,
            infrastructure_mean,
            active_toggles: active,
            toggle_count,
        }
    }

    /// Format as a compact one-line summary for logging.
    pub fn one_line(&self) -> String {
        format!(
            "Yr{:>6.1} | Pop {:>6} | CVS {:.3} | Φ {:.3} | Gini {:.2} | {:?} Ψ={:.2} W={:.2} S={:.2} | P(war)={:.3} P(sec)={:.3} | Toggles: {}",
            self.year, self.total_population, self.cvs, self.collective_phi,
            self.gini, self.secular_phase, self.psi, self.immiseration,
            self.state_stability, self.p_civil_war, self.p_secession,
            self.toggle_count,
        )
    }

    /// Append this metric as a single JSON line to a JSONL file.
    ///
    /// JSONL (JSON Lines) format: one JSON object per line.
    /// Ingestible by: `pd.read_json('file.jsonl', lines=True)` in Python,
    /// `d3.json()` per line in D3.js, or any line-oriented JSON parser.
    pub fn append_jsonl(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        let json = serde_json::to_string(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        writeln!(file, "{}", json)?;
        Ok(())
    }

    /// Risk level: 0=green, 1=yellow, 2=red based on aggregate threat.
    pub fn risk_level(&self) -> u8 {
        let total_risk = self.p_civil_war + self.p_secession + self.p_carrington + self.p_kessler;
        if total_risk > 0.05 {
            2
        }
        // Red
        else if total_risk > 0.01 {
            1
        }
        // Yellow
        else {
            0
        } // Green
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_metrics() {
        let policy = PolicyConfig::default();
        let metrics = LiveMetrics::compute(
            120, &policy, None, 0.7, 1000, 0.3, 0.25, 0.15, 0.8, false, 1.0,
        );
        assert_eq!(metrics.tick, 120);
        assert!((metrics.year - 10.0).abs() < 0.01);
        assert!(metrics.toggle_count >= 8);
        assert_eq!(metrics.p_civil_war, 0.0); // No secular state → no civil war prob
    }

    #[test]
    fn crisis_state_shows_war_probability() {
        let policy = PolicyConfig::default();
        let mut secular = SecularCycleState::default();
        secular.phase = SecularCyclePhase::Crisis;
        secular.state_stability = 0.2;
        secular.psi = 1.5;
        secular.immiseration = 0.7;

        let metrics = LiveMetrics::compute(
            600,
            &policy,
            Some(&secular),
            0.5,
            5000,
            0.2,
            0.5,
            0.6,
            0.5,
            false,
            1.0,
        );
        assert!(
            metrics.p_civil_war > 0.04,
            "Crisis should show war prob: {}",
            metrics.p_civil_war
        );
        assert_eq!(metrics.risk_level(), 2); // Red
    }

    #[test]
    fn legacy_mode_fewer_toggles() {
        let legacy = PolicyConfig::legacy_mode();
        let default = PolicyConfig::default();

        let m_legacy = LiveMetrics::compute(
            100, &legacy, None, 0.5, 1000, 0.2, 0.3, 0.2, 0.7, false, 1.0,
        );
        let m_default = LiveMetrics::compute(
            100, &default, None, 0.5, 1000, 0.2, 0.3, 0.2, 0.7, false, 1.0,
        );
        assert!(
            m_legacy.toggle_count < m_default.toggle_count,
            "Legacy should have fewer toggles: {} vs {}",
            m_legacy.toggle_count,
            m_default.toggle_count
        );
    }

    #[test]
    fn utopia_mode_no_disasters() {
        let utopia = PolicyConfig::utopia_mode();
        let metrics = LiveMetrics::compute(
            100, &utopia, None, 0.8, 1000, 0.5, 0.15, 0.1, 0.9, true, 5.0,
        );
        assert_eq!(metrics.p_carrington, 0.0);
        assert_eq!(metrics.p_eclss_failure, 0.0);
        assert_eq!(metrics.p_civil_war, 0.0);
    }

    #[test]
    fn one_line_format() {
        let policy = PolicyConfig::default();
        let metrics = LiveMetrics::compute(
            1200, &policy, None, 0.72, 50000, 0.35, 0.28, 0.12, 0.85, false, 1.0,
        );
        let line = metrics.one_line();
        assert!(line.contains("Yr"));
        assert!(line.contains("CVS"));
        assert!(line.contains("Toggles"));
    }

    #[test]
    fn jsonl_append_creates_valid_output() {
        let policy = PolicyConfig::default();
        let path = std::path::Path::new("/tmp/test_metrics.jsonl");
        let _ = std::fs::remove_file(path); // Clean up

        // Write 3 ticks
        for tick in 0..3 {
            let m = LiveMetrics::compute(
                tick * 12,
                &policy,
                None,
                0.7,
                1000 + tick as usize * 10,
                0.3,
                0.25,
                0.15,
                0.8,
                false,
                1.0,
            );
            m.append_jsonl(path).expect("Should write JSONL");
        }

        // Verify: 3 lines, each valid JSON
        let content = std::fs::read_to_string(path).expect("Should read file");
        let lines: Vec<&str> = content.trim().split('\n').collect();
        assert_eq!(lines.len(), 3, "Should have 3 lines");
        for line in &lines {
            let _: serde_json::Value =
                serde_json::from_str(line).expect("Each line should be valid JSON");
        }

        // Verify first line has expected fields
        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert!(first.get("tick").is_some());
        assert!(first.get("cvs").is_some());
        assert!(first.get("p_civil_war").is_some());
        assert!(first.get("active_toggles").is_some());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn risk_level_green_when_safe() {
        let policy = PolicyConfig::default();
        let metrics = LiveMetrics::compute(
            100, &policy, None, 0.8, 1000, 0.5, 0.15, 0.1, 0.9, false, 1.0,
        );
        assert_eq!(metrics.risk_level(), 0); // Green
    }
}
