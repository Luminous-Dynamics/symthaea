// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Narrative engine: transforms simulation events and affect data into
//! structured story arcs. Each significant event gets a narrative frame
//! describing the crisis, the collective response, and the outcome.
//!
//! The engine tracks "memorable events" — moments where the civilization's
//! trajectory meaningfully changed. These create the emergent history of the sim.

use serde::{Deserialize, Serialize};

/// A memorable narrative event — a moment worth remembering in civilization history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeEvent {
    /// Simulation tick when this occurred.
    pub tick: u32,
    /// Year (tick / 12.0).
    pub year: f64,
    /// Which world this primarily affects (None = civilization-wide).
    pub world: Option<String>,
    /// The crisis or catalyst that triggered this moment.
    pub crisis: String,
    /// How the collective responded (derived from affect state).
    pub response: String,
    /// What happened as a result.
    pub outcome: String,
    /// Narrative severity: 1 = notable, 2 = significant, 3 = pivotal, 4 = epoch-defining.
    pub severity: u8,
    /// Affect snapshot at the moment (collective).
    pub joy: f64,
    pub sadness: f64,
    pub desire: f64,
    pub care: f64,
}

/// The narrative engine accumulates memorable events across the simulation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NarrativeEngine {
    /// Memorable events (kept to ~100 most significant).
    pub events: Vec<NarrativeEvent>,
    /// Previous tick's per-world population (for detecting crashes).
    prev_populations: Vec<(String, usize)>,
    /// Previous tick's CVS (for detecting viability drops).
    prev_cvs: f64,
    /// Tech milestones already narrated (to avoid duplicates).
    narrated_milestones: Vec<String>,
}

impl NarrativeEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate narrative events from current simulation state.
    /// Called once per tick. Only generates events for significant moments.
    pub fn tick(
        &mut self,
        tick: u32,
        worlds: &[(String, String, usize, f64, f64, f64, f64, f64)],
        // (name, location, pop, mean_joy, mean_sadness, mean_desire, mean_care, self_sufficiency)
        cvs: f64,
        tech_milestones_achieved: &[String],
        active_disaster_count: u32,
        kessler_active: bool,
        excursion_active: bool,
    ) {
        let year = tick as f64 / 12.0;

        // 1. Population crash detection (>10% drop in any world)
        for (name, _loc, pop, _joy, sadness, desire, care, _ss) in worlds {
            if let Some((_, prev_pop)) = self.prev_populations.iter().find(|(n, _)| n == name) {
                if *prev_pop > 20 && *pop < *prev_pop {
                    let drop_frac = 1.0 - (*pop as f64 / *prev_pop as f64);
                    if drop_frac > 0.10 {
                        let severity = if drop_frac > 0.30 { 4 } else if drop_frac > 0.20 { 3 } else { 2 };
                        let response = if *care > 0.5 {
                            format!("The survivors rallied with extraordinary mutual aid (care: {:.2})", care)
                        } else if *desire > 0.6 {
                            format!("Desperate striving gripped the colony (desire: {:.2})", desire)
                        } else if *sadness > 0.5 {
                            format!("A pall of grief settled over the colony (sadness: {:.2})", sadness)
                        } else {
                            "The colony struggled to process the loss".into()
                        };
                        let outcome = if *care > *sadness {
                            "Community bonds held — reconstruction began within days"
                        } else {
                            "Social fabric frayed — faction tensions rose"
                        };
                        self.events.push(NarrativeEvent {
                            tick, year,
                            world: Some(name.clone()),
                            crisis: format!("{} lost {:.0}% of its population ({} → {})",
                                name, drop_frac * 100.0, prev_pop, pop),
                            response,
                            outcome: outcome.into(),
                            severity,
                            joy: 0.0, sadness: *sadness, desire: *desire, care: *care,
                        });
                    }
                }
            }
        }

        // 2. CVS drops (civilization viability declining)
        if self.prev_cvs > 0.0 && cvs < self.prev_cvs - 0.03 {
            self.events.push(NarrativeEvent {
                tick, year,
                world: None,
                crisis: format!("Civilization viability dropped from {:.3} to {:.3}", self.prev_cvs, cvs),
                response: "Inter-world councils convened emergency sessions".into(),
                outcome: if cvs > 0.5 {
                    "Viability remained above critical threshold — reforms initiated".into()
                } else {
                    "WARNING: Approaching collapse threshold".into()
                },
                severity: if cvs < 0.4 { 4 } else { 2 },
                joy: 0.0, sadness: 0.0, desire: 0.0, care: 0.0,
            });
        }

        // 3. Tech milestone achievements
        for milestone in tech_milestones_achieved {
            if !self.narrated_milestones.contains(milestone) {
                let (crisis, response) = match milestone.as_str() {
                    "Fission Surface Power" => (
                        "Decades of energy poverty on the frontier",
                        "Nuclear fission reactors activated — the outer system colonies can finally power their own survival",
                    ),
                    "Radiation Hardening" => (
                        "Jupiter's radiation belt ravaged Europa's electronics for years",
                        "New rad-hardened circuits deployed — Europa can now sustain surface operations",
                    ),
                    "Cryogenic Materials" => (
                        "Titan's -179°C temperatures cracked seals and shattered standard materials",
                        "Austenitic steel alloys and aerogel composites conquered the cold",
                    ),
                    "Fusion Grid Scale" => (
                        "Energy scarcity constrained every colony's growth",
                        "Fusion power went online — the thermodynamic ceiling lifted for the first time",
                    ),
                    "Fusion Drive" => (
                        "26-month transfer windows left colonies stranded for years",
                        "Continuous-thrust fusion drives made interplanetary trade permanent — transfer windows are history",
                    ),
                    "Bioregenerative Agriculture" => (
                        "Hydroponics couldn't keep up with growing populations",
                        "Self-sustaining bioregenerative farms achieved — food security at last",
                    ),
                    "Closed-Loop ECLSS" => (
                        "Water and oxygen recycling losses drained reserves every month",
                        "98% closed-loop life support achieved — colonies became self-sustaining",
                    ),
                    "Terraforming Precursor" => (
                        "Centuries of survival in sealed habitats tested human endurance",
                        "The first atmospheric modification experiments began — a new chapter in planetary engineering",
                    ),
                    "Genetic Engineering" => (
                        "Small colonies faced genetic bottleneck — inbreeding threatened future generations",
                        "CRISPR gene therapy and Yamanaka factor reprogramming eliminated hereditary defects — the genetic trap was broken",
                    ),
                    "Interstellar Probe" => (
                        "A thousand years of solar system civilization reached its culmination",
                        "The first probe departed for Alpha Centauri — humanity's reach extended beyond the Sun",
                    ),
                    _ => (
                        "Research breakthrough achieved",
                        "New capabilities unlocked for the civilization",
                    ),
                };
                self.events.push(NarrativeEvent {
                    tick, year,
                    world: None,
                    crisis: crisis.into(),
                    response: response.into(),
                    outcome: format!("MILESTONE: {}", milestone),
                    severity: 3,
                    joy: 0.0, sadness: 0.0, desire: 0.0, care: 0.0,
                });
                self.narrated_milestones.push(milestone.clone());
            }
        }

        // 4. Kessler cascade onset
        if kessler_active && !self.narrated_milestones.contains(&"Kessler".into()) {
            self.events.push(NarrativeEvent {
                tick, year,
                world: Some("Earth".into()),
                crisis: "Earth's governance collapse triggered a catastrophic orbital debris cascade".into(),
                response: "The LEO Barricade sealed Earth behind a wall of hypervelocity shrapnel".into(),
                outcome: "Outer colonies faced sudden isolation — the question of self-sufficiency became existential".into(),
                severity: 4,
                joy: 0.0, sadness: 0.8, desire: 0.9, care: 0.0,
            });
            self.narrated_milestones.push("Kessler".into());
        }

        // 5. Magnetosphere excursion
        if excursion_active && !self.narrated_milestones.contains(&"Laschamp".into()) {
            self.events.push(NarrativeEvent {
                tick, year,
                world: Some("Earth".into()),
                crisis: "Earth's magnetic field collapsed to 5% — a Laschamp-type excursion began".into(),
                response: "Solar storms now struck Earth with 4× their normal fury. The ozone layer thinned.".into(),
                outcome: "Earth was no longer the safe harbor. For the first time, all worlds faced equal danger.".into(),
                severity: 4,
                joy: 0.0, sadness: 0.6, desire: 0.7, care: 0.0,
            });
            self.narrated_milestones.push("Laschamp".into());
        }

        // 6. Refugee crisis (from affect-driven migration)
        // Already handled by CivEvent in tick_interworld — we detect it from population shifts

        // Update state for next tick
        self.prev_populations = worlds.iter()
            .map(|(name, _, pop, _, _, _, _, _)| (name.clone(), *pop))
            .collect();
        self.prev_cvs = cvs;

        // Trim to most significant 100 events
        if self.events.len() > 120 {
            self.events.sort_by(|a, b| b.severity.cmp(&a.severity)
                .then_with(|| b.tick.cmp(&a.tick)));
            self.events.truncate(100);
        }
    }

    /// Format the narrative as a readable history.
    pub fn format_history(&self) -> String {
        let mut sorted = self.events.clone();
        sorted.sort_by_key(|e| e.tick);

        let mut out = String::from("=== CIVILIZATION NARRATIVE ===\n\n");
        for event in &sorted {
            let world_str = event.world.as_deref().unwrap_or("All Worlds");
            let severity_marker = match event.severity {
                4 => "████",
                3 => " ███",
                2 => "  ██",
                _ => "   █",
            };
            out.push_str(&format!(
                "{} Year {:.0} [{}]\n  {}\n  → {}\n  ⇒ {}\n\n",
                severity_marker, event.year, world_str,
                event.crisis, event.response, event.outcome
            ));
        }
        out
    }
}
