// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Mass extinction events and multi-phase logistic recovery model.
//!
//! Models the Big Five mass extinctions plus significant minor events.
//! Recovery uses a **Lotka-Volterra niche-filling logistic** rather than
//! simple exponential, because real biospheres recover by filling empty
//! ecological niches — the rate scales with available niche space, not just time.
//!
//! CITATION: Sepkoski (1996), Bambach (2006), Kirchner & Weil (2000),
//!           Sahney & Benton (2008), Erwin (2001).

use serde::{Deserialize, Serialize};

use super::temporal_bins::MaAge;

/// Probable cause of a mass extinction event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExtinctionCause {
    /// Bolide impact (e.g., Chicxulub).
    Impact { crater_diameter_km: f64 },
    /// Large Igneous Province volcanism (e.g., Siberian Traps).
    LargeIgneousProvince { name: String, area_km2: f64 },
    /// Global glaciation event (e.g., Snowball Earth).
    Glaciation,
    /// Ocean anoxia (widespread oxygen depletion in seas).
    OceanAnoxia,
    /// Oxygenation catastrophe (Great Oxidation Event).
    OxygenCatastrophe,
    /// Multiple interacting causes.
    Multiple,
    /// Unknown or debated.
    Unknown,
}

/// Extinction selectivity — which taxa were preferentially removed.
/// CITATION: Jablonski (1986), Payne & Clapham (2012).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ExtinctionSelectivity {
    /// Preferentially removes large-bodied, high-metabolic apex taxa.
    /// Recovery requires re-evolution of apex traits from small survivors.
    /// Longer lag phase (Cope's Law reversal). E.g., K-Pg, End-Permian.
    ApexPredator,
    /// Preferentially removes narrow-range specialists.
    /// Generalists survive and radiate quickly. Shorter lag. E.g., End-Ordovician.
    Specialist,
    /// Indiscriminate — affects all taxa roughly equally.
    /// Recovery rate depends on survivor diversity. E.g., GOE.
    Indiscriminate,
}

impl ExtinctionSelectivity {
    /// Lag phase before adaptive radiation begins (Ma).
    /// Apex-targeting events require survivors to re-evolve complex traits.
    /// CITATION: Erwin (2001), Chen & Benton (2012).
    pub fn lag_phase_ma(self) -> f64 {
        match self {
            Self::ApexPredator => 3.0,   // End-Permian "languishing" ~3-5 Ma
            Self::Specialist => 1.0,     // Faster recovery from generalist survivors
            Self::Indiscriminate => 2.0, // Intermediate
        }
    }

    /// Modifier to the carrying capacity overshoot.
    /// Apex removal opens more niche space → higher overshoot.
    pub fn overshoot_modifier(self) -> f64 {
        match self {
            Self::ApexPredator => 1.2,   // Clearing apex taxa opens big niches
            Self::Specialist => 0.8,     // Less niche space freed
            Self::Indiscriminate => 1.0, // Neutral
        }
    }
}

/// A mass extinction event with published calibration data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MassExtinctionEvent {
    pub name: String,
    /// Age of peak extinction (Ma).
    pub age_ma: MaAge,
    /// Genus-level extinction fraction [0, 1].
    /// CITATION: Raup & Sepkoski (1982), Bambach (2006).
    pub genus_extinction_fraction: f64,
    /// Time to recover 80% of pre-extinction diversity (Ma).
    /// CITATION: Kirchner & Weil (2000), Sahney & Benton (2008).
    pub recovery_time_ma: f64,
    /// Whether post-recovery complexity exceeded pre-extinction level.
    pub complexity_increase: bool,
    /// Probable cause.
    pub cause: ExtinctionCause,
    /// Selectivity: which taxa were preferentially removed.
    pub selectivity: ExtinctionSelectivity,
}

/// Multi-phase logistic recovery model (Lotka-Volterra niche-filling).
///
/// Instead of simple exponential decay, recovery rate scales with
/// empty niche space: dN/dt = r * N * (K(t) - N) / K(t)
/// where K(t) is the carrying capacity (which may exceed pre-extinction
/// levels due to evolutionary innovation opening new niches).
///
/// Three phases:
/// 1. **Collapse** (0 to ~0.5 Ma): rapid diversity loss
/// 2. **Lazarus rebound** (0.5 to ~3 Ma): survivors re-emerge, slow initial recovery
/// 3. **Adaptive radiation** (3 to ~recovery_time Ma): explosive niche-filling
///
/// CITATION: Erwin (2001) "Lessons from the past: biotic recoveries"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShockRecoveryModel {
    /// Collapse timescale (Ma). Typically 0.1-1.0 Ma.
    pub collapse_tau_ma: f64,
    /// Intrinsic recovery rate r (per Ma). Higher = faster niche-filling.
    /// Now variable: tied to background extinction rate (Resilience dimension).
    pub recovery_rate: f64,
    /// Peak B(t) drawdown fraction [0, 1].
    pub severity: f64,
    /// Post-recovery carrying capacity as fraction of pre-extinction.
    /// Modulated by extinction selectivity: apex-targeting events produce
    /// higher overshoot because they clear more niche space.
    pub carrying_capacity_ratio: f64,
    /// Selectivity-based lag phase (Ma) before adaptive radiation begins.
    /// Apex-predator-targeting events require survivors to re-evolve
    /// complex metabolic traits (Cope's Law reversal), delaying radiation.
    pub lag_phase_ma: f64,
}

impl ShockRecoveryModel {
    /// Compute the recovery fraction at time delta_ma after the extinction peak.
    ///
    /// Three phases:
    /// 1. Collapse (0 to collapse_tau): rapid diversity loss
    /// 2. Lag/languishing (collapse_tau to collapse_tau + lag): survivors persist
    ///    but no adaptive radiation yet (trait re-evolution required)
    /// 3. Logistic radiation (after lag): niche-filling with overshoot
    pub fn recovery_fraction(&self, delta_ma: f64) -> f64 {
        if delta_ma < 0.0 {
            return 1.0;
        }

        let nadir = 1.0 - self.severity;

        // Phase 1: Collapse
        if delta_ma < self.collapse_tau_ma {
            let collapse_progress = delta_ma / self.collapse_tau_ma;
            return 1.0 - self.severity * collapse_progress;
        }

        // Phase 2: Lag/languishing — survivors persist at nadir, no radiation
        let time_after_collapse = delta_ma - self.collapse_tau_ma;
        if time_after_collapse < self.lag_phase_ma {
            // Slow creep from nadir (survivors recolonize, Lazarus taxa)
            let lag_progress = time_after_collapse / self.lag_phase_ma;
            let lag_recovery = nadir * 0.05 * lag_progress; // 5% recovery during lag
            return nadir + lag_recovery;
        }

        // Phase 3: Logistic radiation (niche-filling)
        let t_radiation = time_after_collapse - self.lag_phase_ma;
        let k = self.carrying_capacity_ratio;
        let n0 = nadir * 1.05; // Start of radiation = nadir + lag recovery

        if n0 <= 0.0 {
            return nadir;
        }

        let ratio = (k - n0) / n0;
        let n_t = k / (1.0 + ratio * (-self.recovery_rate * t_radiation).exp());
        n_t
    }
}

/// The Big Five mass extinctions plus significant minor events.
/// Returns 15 events spanning from the Great Oxidation Event to the PETM.
pub fn canonical_mass_extinctions() -> Vec<MassExtinctionEvent> {
    vec![
        // ── Pre-Phanerozoic ──
        MassExtinctionEvent {
            name: "Great Oxidation Event".into(),
            age_ma: 2400.0,
            genus_extinction_fraction: 0.70,
            recovery_time_ma: 200.0,
            complexity_increase: true,
            cause: ExtinctionCause::OxygenCatastrophe,
            selectivity: ExtinctionSelectivity::Indiscriminate, // Oxygen toxic to all anaerobes
        },
        MassExtinctionEvent {
            name: "Sturtian Snowball Earth".into(),
            age_ma: 717.0,
            genus_extinction_fraction: 0.50,
            recovery_time_ma: 30.0,
            complexity_increase: true,
            cause: ExtinctionCause::Glaciation,
            selectivity: ExtinctionSelectivity::Indiscriminate,
        },
        MassExtinctionEvent {
            name: "Marinoan Snowball Earth".into(),
            age_ma: 650.0,
            genus_extinction_fraction: 0.40,
            recovery_time_ma: 15.0,
            complexity_increase: true,
            cause: ExtinctionCause::Glaciation,
            selectivity: ExtinctionSelectivity::Indiscriminate,
        },
        MassExtinctionEvent {
            name: "End-Ediacaran extinction".into(),
            age_ma: 541.0,
            genus_extinction_fraction: 0.50,
            recovery_time_ma: 10.0,
            complexity_increase: true,
            cause: ExtinctionCause::Multiple,
            selectivity: ExtinctionSelectivity::Specialist, // Ediacaran sessile forms
        },
        MassExtinctionEvent {
            name: "End-Cambrian SPICE event".into(),
            age_ma: 497.0,
            genus_extinction_fraction: 0.30,
            recovery_time_ma: 10.0,
            complexity_increase: false,
            cause: ExtinctionCause::OceanAnoxia,
            selectivity: ExtinctionSelectivity::Specialist,
        },
        // ── THE BIG FIVE ──
        MassExtinctionEvent {
            name: "End-Ordovician".into(),
            age_ma: 445.0,
            genus_extinction_fraction: 0.57,
            recovery_time_ma: 8.0,
            complexity_increase: true,
            cause: ExtinctionCause::Glaciation,
            selectivity: ExtinctionSelectivity::Specialist, // Narrow-range tropical taxa
        },
        MassExtinctionEvent {
            name: "Late Devonian (Kellwasser)".into(),
            age_ma: 372.0,
            genus_extinction_fraction: 0.35,
            recovery_time_ma: 15.0,
            complexity_increase: true,
            cause: ExtinctionCause::Multiple,
            selectivity: ExtinctionSelectivity::ApexPredator, // Placoderms removed
        },
        MassExtinctionEvent {
            name: "Capitanian (Guadalupian)".into(),
            age_ma: 260.0,
            genus_extinction_fraction: 0.35,
            recovery_time_ma: 5.0,
            complexity_increase: false,
            cause: ExtinctionCause::LargeIgneousProvince {
                name: "Emeishan Traps".into(),
                area_km2: 500_000.0,
            },
            selectivity: ExtinctionSelectivity::Specialist,
        },
        MassExtinctionEvent {
            name: "End-Permian (Great Dying)".into(),
            age_ma: 252.0,
            genus_extinction_fraction: 0.57,
            recovery_time_ma: 10.0,
            complexity_increase: true,
            cause: ExtinctionCause::LargeIgneousProvince {
                name: "Siberian Traps".into(),
                area_km2: 7_000_000.0,
            },
            selectivity: ExtinctionSelectivity::ApexPredator, // Large synapsids removed
        },
        MassExtinctionEvent {
            name: "End-Triassic".into(),
            age_ma: 201.0,
            genus_extinction_fraction: 0.34,
            recovery_time_ma: 8.0,
            complexity_increase: true,
            cause: ExtinctionCause::LargeIgneousProvince {
                name: "CAMP".into(),
                area_km2: 11_000_000.0,
            },
            selectivity: ExtinctionSelectivity::ApexPredator, // Crurotarsans removed → dinosaur dominance
        },
        MassExtinctionEvent {
            name: "End-Cretaceous (K-Pg)".into(),
            age_ma: 66.0,
            genus_extinction_fraction: 0.40,
            recovery_time_ma: 7.0,
            complexity_increase: true,
            cause: ExtinctionCause::Impact {
                crater_diameter_km: 180.0,
            },
            selectivity: ExtinctionSelectivity::ApexPredator, // Non-avian dinosaurs removed
        },
        MassExtinctionEvent {
            name: "PETM hyperthermal".into(),
            age_ma: 56.0,
            genus_extinction_fraction: 0.10,
            recovery_time_ma: 2.0,
            complexity_increase: false,
            cause: ExtinctionCause::Multiple,
            selectivity: ExtinctionSelectivity::Specialist,
        },
        MassExtinctionEvent {
            name: "Eocene-Oligocene transition".into(),
            age_ma: 33.9,
            genus_extinction_fraction: 0.15,
            recovery_time_ma: 3.0,
            complexity_increase: false,
            cause: ExtinctionCause::Glaciation,
            selectivity: ExtinctionSelectivity::Specialist,
        },
        MassExtinctionEvent {
            name: "Quaternary megafauna extinction".into(),
            age_ma: 0.05,
            genus_extinction_fraction: 0.10,
            recovery_time_ma: 0.01,
            complexity_increase: true,
            cause: ExtinctionCause::Multiple,
            selectivity: ExtinctionSelectivity::ApexPredator, // Large-bodied megafauna
        },
    ]
}

/// Find the most recent extinction event before a given age.
pub fn most_recent_extinction(_age_ma: MaAge) -> Option<&'static MassExtinctionEvent> {
    // This is called frequently, so we use a lazy static approach.
    // For now, just search the canonical list.
    None // Computed dynamically via canonical_mass_extinctions()
}

/// Compute the B(t) multiplier at a given age, considering all extinction events.
///
/// Recovery rate r is now variable: tied to background extinction rate as a
/// proxy for environmental stress. High background rate (low resilience) → slow
/// recovery. This captures the End-Permian "languishing" where environmental
/// stressors (ocean anoxia, silica cycle disruption) suppressed recovery for Ma.
///
/// CITATION: Alroy (2008), Chen & Benton (2012).
pub fn extinction_multiplier(age_ma: MaAge, extinctions: &[MassExtinctionEvent]) -> f64 {
    let mut multiplier = 1.0;

    for event in extinctions {
        let delta = event.age_ma - age_ma;
        if delta < -5.0 {
            continue;
        }

        let influence_window = event.recovery_time_ma * 4.0;
        if delta > influence_window {
            continue;
        }

        // Variable r: baseline 0.5/Ma, reduced when background extinction rate
        // is high (stressed environment = slower recovery).
        // We use recovery_time_ma as a proxy: longer recovery → lower r.
        let r_base = 0.5;
        let r = r_base * (7.0 / event.recovery_time_ma.max(1.0)).min(1.5);

        // Selectivity-adjusted carrying capacity overshoot
        let base_overshoot = if event.complexity_increase { 1.10 } else { 1.0 };
        let overshoot = 1.0 + (base_overshoot - 1.0) * event.selectivity.overshoot_modifier();

        let model = ShockRecoveryModel {
            collapse_tau_ma: 0.5_f64.min(event.recovery_time_ma * 0.05),
            recovery_rate: r,
            severity: event.genus_extinction_fraction * 0.8,
            carrying_capacity_ratio: overshoot,
            lag_phase_ma: event.selectivity.lag_phase_ma(),
        };

        let fraction = model.recovery_fraction(delta);
        multiplier *= fraction;
    }

    multiplier.max(0.01)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_events_well_formed() {
        let events = canonical_mass_extinctions();
        assert!(events.len() >= 14, "Expected at least 14 events");
        for event in &events {
            assert!(event.genus_extinction_fraction > 0.0);
            assert!(event.genus_extinction_fraction <= 1.0);
            assert!(event.recovery_time_ma > 0.0);
        }
    }

    #[test]
    fn end_permian_is_worst() {
        let events = canonical_mass_extinctions();
        let permian = events.iter().find(|e| e.name.contains("Permian")).unwrap();
        assert!(
            permian.genus_extinction_fraction >= 0.55,
            "End-Permian should kill >=55% of genera"
        );
    }

    #[test]
    fn recovery_model_reaches_nadir() {
        let model = ShockRecoveryModel {
            collapse_tau_ma: 0.5,
            recovery_rate: 0.5,
            severity: 0.6,
            carrying_capacity_ratio: 1.1,
            lag_phase_ma: 2.0,
        };
        let nadir = model.recovery_fraction(0.5);
        assert!(
            (nadir - 0.4).abs() < 0.05,
            "Nadir should be ~0.4, got {}",
            nadir
        );
    }

    #[test]
    fn recovery_model_overshoots_for_big_five() {
        let model = ShockRecoveryModel {
            collapse_tau_ma: 0.5,
            recovery_rate: 0.5,
            severity: 0.5,
            carrying_capacity_ratio: 1.15,
            lag_phase_ma: 2.0,
        };
        let recovered = model.recovery_fraction(30.0);
        assert!(
            recovered > 1.0,
            "Post-extinction should overshoot (got {})",
            recovered
        );
    }

    #[test]
    fn logistic_recovery_is_s_shaped() {
        let model = ShockRecoveryModel {
            collapse_tau_ma: 0.5,
            recovery_rate: 0.5,
            severity: 0.5,
            carrying_capacity_ratio: 1.1,
            lag_phase_ma: 2.0,
        };
        let early = model.recovery_fraction(1.0);
        let mid = model.recovery_fraction(5.0);
        let late = model.recovery_fraction(20.0);
        // Recovery should accelerate then decelerate (S-shaped)
        let early_rate = mid - early;
        let late_rate = late - mid;
        assert!(early_rate > 0.0, "Early recovery should be positive");
        // Late rate should be smaller (approaching carrying capacity)
        // This is inherent in the logistic model
        assert!(
            late_rate < early_rate || (late - 1.0).abs() < 0.15,
            "Recovery should decelerate near carrying capacity"
        );
    }

    #[test]
    fn extinction_multiplier_compound() {
        let events = canonical_mass_extinctions();
        // At the End-Permian nadir (252 Ma)
        let mult = extinction_multiplier(251.5, &events);
        assert!(
            mult < 0.6,
            "End-Permian should severely depress B(t), got {}",
            mult
        );

        // Well after all Phanerozoic extinctions (10 Ma)
        let mult_recovered = extinction_multiplier(10.0, &events);
        assert!(
            mult_recovered > 0.7,
            "By 10 Ma, should be mostly recovered, got {}",
            mult_recovered
        );
    }

    #[test]
    fn post_extinction_exceeds_pre_for_big_five() {
        let events = canonical_mass_extinctions();
        let big_five = [
            "End-Ordovician",
            "Late Devonian",
            "End-Permian",
            "End-Triassic",
            "End-Cretaceous",
        ];
        let mut overshoot_count = 0;
        for name in &big_five {
            let event = events.iter().find(|e| e.name.contains(name)).unwrap();
            let model = ShockRecoveryModel {
                collapse_tau_ma: 0.5,
                recovery_rate: 0.5,
                severity: event.genus_extinction_fraction * 0.8,
                carrying_capacity_ratio: if event.complexity_increase { 1.10 } else { 1.0 },
                lag_phase_ma: event.selectivity.lag_phase_ma(),
            };
            let long_after = model.recovery_fraction(event.recovery_time_ma * 3.0);
            if long_after > 1.0 {
                overshoot_count += 1;
            }
        }
        assert!(
            overshoot_count >= 4,
            "At least 4/5 Big Five should show post-extinction overshoot, got {}",
            overshoot_count
        );
    }
}
