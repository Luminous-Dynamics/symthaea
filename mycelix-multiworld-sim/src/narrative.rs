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
    /// Named character involved (generated for pivotal events).
    pub character: Option<String>,
}

/// Generate a plausible name for a colony character.
/// Uses deterministic selection from agent pool characteristics.
pub fn generate_character_name(world_name: &str, role: &str, generation: u16) -> String {
    // Deterministic name generation from world + role + generation
    let first_names = [
        "Kenji", "Amara", "Chen", "Fatima", "Dmitri", "Yuki", "Kofi", "Leila", "Ravi", "Ingrid",
        "Marcus", "Zara", "Tomás", "Aisha", "Sven", "Priya", "Oleg", "Mei", "Hassan", "Luna",
    ];
    let last_names = [
        "Watanabe",
        "Okafor",
        "Zhang",
        "Al-Rashid",
        "Petrov",
        "Tanaka",
        "Mensah",
        "Nazari",
        "Patel",
        "Lindqvist",
        "Santos",
        "Kim",
        "Reyes",
        "Mwangi",
        "Johansson",
        "Sharma",
        "Volkov",
        "Liu",
        "Ibrahim",
        "Estrada",
    ];

    let seed = world_name.len() + role.len() + generation as usize;
    let first = first_names[seed % first_names.len()];
    let last = last_names[(seed * 7 + 3) % last_names.len()];
    format!("{} {} (Gen {})", first, last, generation)
}

/// The narrative engine accumulates memorable events across the simulation.
/// #7: Cultural memory — civilizations learn from their past.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CulturalMemory {
    /// Name of the event that created this memory.
    pub name: String,
    /// Which world this memory belongs to.
    pub world: Option<String>,
    /// Tick when the event occurred.
    pub origin_tick: u32,
    /// Lesson learned: modifies future behavior.
    pub lesson: CulturalLesson,
    /// Strength [0, 1]. Decays over generations (half-life ~100 years).
    pub strength: f64,
}

/// What the civilization learned from a major event.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct CulturalLesson {
    /// Preparedness bonus for similar disasters [0, 0.3].
    pub preparedness: f64,
    /// Risk tolerance shift [-0.2, +0.2]. Negative = more cautious.
    pub risk_shift: f64,
    /// Community cohesion bonus [0, 0.2].
    pub cohesion: f64,
}

pub struct NarrativeEngine {
    /// Memorable events (kept to ~200 most significant).
    pub events: Vec<NarrativeEvent>,
    /// Previous tick's per-world population (for detecting crashes).
    prev_populations: Vec<(String, usize)>,
    /// Previous tick's CVS (for detecting viability drops).
    prev_cvs: f64,
    /// Tech milestones already narrated (to avoid duplicates).
    narrated_milestones: Vec<String>,
    /// #7: Cultural memories — lessons from major events.
    pub cultural_memories: Vec<CulturalMemory>,
}

impl NarrativeEngine {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            prev_populations: Vec::new(),
            prev_cvs: 0.0,
            narrated_milestones: Vec::new(),
            cultural_memories: Vec::new(),
        }
    }

    /// Create a cultural memory from a major event.
    pub fn remember(
        &mut self,
        name: String,
        world: Option<String>,
        tick: u32,
        lesson: CulturalLesson,
    ) {
        self.cultural_memories.push(CulturalMemory {
            name,
            world,
            origin_tick: tick,
            lesson,
            strength: 1.0,
        });
    }

    /// Decay all cultural memories. Half-life ~100 years (1200 ticks).
    pub fn decay_memories(&mut self) {
        let decay = 0.5_f64.powf(1.0 / 1200.0); // Per-tick decay for 100yr half-life
        for mem in &mut self.cultural_memories {
            mem.strength *= decay;
        }
        // Remove memories below perception threshold
        self.cultural_memories.retain(|m| m.strength > 0.01);
    }

    /// Get the aggregate cultural lesson for a world (sum of all active memories).
    pub fn cultural_lesson_for(&self, world: Option<&str>) -> CulturalLesson {
        let mut lesson = CulturalLesson::default();
        for mem in &self.cultural_memories {
            let applies = match (&mem.world, world) {
                (None, _) => true,            // Civilization-wide memory
                (Some(w), Some(q)) => w == q, // World-specific
                _ => false,
            };
            if applies {
                lesson.preparedness += mem.lesson.preparedness * mem.strength;
                lesson.risk_shift += mem.lesson.risk_shift * mem.strength;
                lesson.cohesion += mem.lesson.cohesion * mem.strength;
            }
        }
        lesson.preparedness = lesson.preparedness.min(0.5);
        lesson.cohesion = lesson.cohesion.min(0.3);
        lesson
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
                if *prev_pop > 10 && *pop < *prev_pop {
                    let drop_frac = 1.0 - (*pop as f64 / *prev_pop as f64);
                    // Lower threshold for small colonies (5% for <100 pop, 10% for larger)
                    let threshold = if *prev_pop < 100 { 0.05 } else { 0.10 };
                    if drop_frac > threshold {
                        let severity = if drop_frac > 0.30 {
                            4
                        } else if drop_frac > 0.20 {
                            3
                        } else {
                            2
                        };
                        // P3: Response diversity — crisis count matters
                        let prior_crises = self
                            .events
                            .iter()
                            .filter(|e| e.world.as_deref() == Some(name) && e.severity >= 2)
                            .count();
                        let response = if *care > 0.5 {
                            format!(
                                "The survivors rallied with extraordinary mutual aid (care: {:.2})",
                                care
                            )
                        } else if *desire > 0.6 && prior_crises < 5 {
                            format!(
                                "Desperate striving gripped the colony (desire: {:.2})",
                                desire
                            )
                        } else if *sadness > 0.5 && prior_crises < 3 {
                            format!(
                                "A pall of grief settled over the colony (sadness: {:.2})",
                                sadness
                            )
                        } else if prior_crises > 10 {
                            "A grim, practiced efficiency took over — they had done this before"
                                .into()
                        } else if prior_crises > 5 {
                            "The colony absorbed the blow with the numbness of experience".into()
                        } else {
                            "Emergency protocols activated — the colony braced for aftermath".into()
                        };
                        let outcome = if *care > *sadness {
                            "Community bonds held — reconstruction began within days"
                        } else if prior_crises > 8 {
                            "The colony endured — scarred but unbroken"
                        } else if *desire > 0.7 {
                            "Determination hardened — the colony would not let this define them"
                        } else if *pop < 50 {
                            "Every remaining colonist felt the absence — the community was intimate enough to grieve each loss by name"
                        } else {
                            "Social fabric strained — trust eroded between factions"
                        };
                        // Generate a named character for pivotal events
                        let character = if severity >= 3 {
                            let role = if *care > 0.5 {
                                "medic"
                            } else if *desire > 0.6 {
                                "engineer"
                            } else {
                                "leader"
                            };
                            Some(generate_character_name(name, role, (year / 30.0) as u16))
                        } else {
                            None
                        };
                        let crisis_text = if let Some(ref ch) = character {
                            format!("{} lost {:.0}% of its population ({} → {}). {} was among the casualties.",
                                name, drop_frac * 100.0, prev_pop, pop, ch)
                        } else {
                            format!(
                                "{} lost {:.0}% of its population ({} → {})",
                                name,
                                drop_frac * 100.0,
                                prev_pop,
                                pop
                            )
                        };
                        self.events.push(NarrativeEvent {
                            tick,
                            year,
                            world: Some(name.clone()),
                            crisis: crisis_text,
                            response,
                            outcome: outcome.into(),
                            severity,
                            joy: 0.0,
                            sadness: *sadness,
                            desire: *desire,
                            care: *care,
                            character,
                        });
                    }
                }
            }
        }

        // 2. CVS drops (civilization viability declining)
        if self.prev_cvs > 0.0 && cvs < self.prev_cvs - 0.03 {
            self.events.push(NarrativeEvent {
                tick,
                year,
                world: None,
                crisis: format!(
                    "Civilization viability dropped from {:.3} to {:.3}",
                    self.prev_cvs, cvs
                ),
                response: "Inter-world councils convened emergency sessions".into(),
                outcome: if cvs > 0.5 {
                    "Viability remained above critical threshold — reforms initiated".into()
                } else {
                    "WARNING: Approaching collapse threshold".into()
                },
                severity: if cvs < 0.4 { 4 } else { 2 },
                joy: 0.0,
                sadness: 0.0,
                desire: 0.0,
                care: 0.0,
                character: None,
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
                    tick,
                    year,
                    world: None,
                    crisis: crisis.into(),
                    response: response.into(),
                    outcome: format!("MILESTONE: {}", milestone),
                    severity: 3,
                    joy: 0.0,
                    sadness: 0.0,
                    desire: 0.0,
                    care: 0.0,
                    character: None,
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
                character: None,
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
                character: None,
            });
            self.narrated_milestones.push("Laschamp".into());
        }

        // 6. Population milestones (first time a world crosses 1000, 5000, 10000)
        for (name, _loc, pop, joy, _sadness, _desire, care, _ss) in worlds {
            for threshold in [100, 500, 1000, 5000, 10000] {
                let key = format!("pop_{}_{}", name, threshold);
                if *pop >= threshold && !self.narrated_milestones.contains(&key) {
                    self.events.push(NarrativeEvent {
                        tick,
                        year,
                        world: Some(name.clone()),
                        crisis: format!("{} reached {} population", name, threshold),
                        response: if *joy > 0.5 {
                            "Celebration marked the milestone — the colony felt its own permanence"
                                .into()
                        } else {
                            "The milestone passed quietly — survival remained the focus".into()
                        },
                        outcome: if *care > 0.5 {
                            "Community bonds strengthened with scale".into()
                        } else {
                            "Growing pains tested institutional capacity".into()
                        },
                        severity: if threshold >= 5000 { 3 } else { 1 },
                        joy: *joy,
                        sadness: 0.0,
                        desire: 0.0,
                        care: *care,
                        character: None,
                    });
                    self.narrated_milestones.push(key);
                }
            }
        }

        // 7. Active disaster overload (5+ simultaneous disasters)
        if active_disaster_count >= 5 {
            let key = format!("overload_{}", tick / 120); // Once per 10 years max
            if !self.narrated_milestones.contains(&key) {
                self.events.push(NarrativeEvent {
                    tick, year,
                    world: None,
                    crisis: format!("{} simultaneous disasters overwhelming civilization response", active_disaster_count),
                    response: "Emergency coordinators struggled to triage across multiple crises".into(),
                    outcome: "Resource allocation became zero-sum — saving one colony meant neglecting another".into(),
                    severity: 3,
                    joy: 0.0, sadness: 0.0, desire: 0.0, care: 0.0,
                character: None,
                });
                self.narrated_milestones.push(key);
            }
        }

        // 8. Refugee crisis detection

        // Update state for next tick
        self.prev_populations = worlds
            .iter()
            .map(|(name, _, pop, _, _, _, _, _)| (name.clone(), *pop))
            .collect();
        self.prev_cvs = cvs;

        // Trim to most significant 200 events (raised from 100 for richer chronicles)
        if self.events.len() > 250 {
            self.events.sort_by(|a, b| {
                b.severity
                    .cmp(&a.severity)
                    .then_with(|| b.tick.cmp(&a.tick))
            });
            self.events.truncate(200);
        }
    }

    /// P0: Ingest CivEvents into the narrative.
    /// Bridges the two event systems so projects, independence movements,
    /// explorations, Dunbar transitions, and supply chain disruptions
    /// appear in the chronicle.
    pub fn ingest_civ_events(&mut self, events: &[crate::events::CivEvent]) {
        for event in events {
            let desc = &event.description;
            let tick = event.tick;
            let year = tick as f64 / 12.0;

            // Determine severity and narrative content from CivEvent markers
            let (severity, crisis, response, outcome) = if desc.contains("PROJECT COMPLETE") {
                let response = if desc.contains("Fission Reactor") {
                    "The hum of the reactor filled the habitat — power at last"
                } else if desc.contains("Greenhouse") {
                    "Fresh green growth under artificial light — the colony could feed itself"
                } else if desc.contains("Habitat Expansion") {
                    "New modules pressurized — families moved in within days"
                } else if desc.contains("Medical") {
                    "The colony's first surgery suite opened — no more improvised medicine"
                } else if desc.contains("Centrifuge") {
                    "The centrifuge spun up — children could be born here now"
                } else if desc.contains("Fabrication") {
                    "Local manufacturing online — the supply chain dependency loosened"
                } else if desc.contains("Water Extraction") {
                    "Water flowed from local extraction — the colony drank its own land"
                } else if desc.contains("Radiation Shelter") {
                    "The shelter doors sealed — Jupiter's wrath could be weathered"
                } else {
                    "Construction crews celebrated as the structure came online"
                };
                let outcome = if desc.contains("Centrifuge") {
                    "Reproduction became possible — the colony's future secured"
                } else if desc.contains("Fission") {
                    "Energy independence achieved — everything else became possible"
                } else {
                    "Colony capability expanded — a step closer to self-sufficiency"
                };
                (2, desc.clone(), response.to_string(), outcome.to_string())
            } else if desc.contains("INDEPENDENCE MOVEMENT") {
                (
                    3,
                    desc.clone(),
                    "Political assemblies debated sovereignty with unprecedented fervor".into(),
                    "The relationship between colony and homeworld entered a new chapter"
                        .to_string(),
                )
            } else if desc.contains("EXPLORATION SUCCESS") {
                (2,
                 desc.clone(),
                 "Survey teams returned with samples and data that would reshape the colony's future".into(),
                 "New resources mapped — the frontier expanded".to_string())
            } else if desc.contains("DUNBAR TRANSITION") {
                (
                    2,
                    desc.clone(),
                    "The old ways of knowing everyone by name gave way to formal institutions"
                        .into(),
                    "Governance adapted to scale — or tried to".to_string(),
                )
            } else if desc.contains("FISSION DELIVERY") {
                (
                    3,
                    desc.clone(),
                    "Nuclear reactors powered up for the first time on alien soil".into(),
                    "The energy equation changed forever".to_string(),
                )
            } else if desc.contains("KESSLER CASCADE") {
                (
                    4,
                    desc.clone(),
                    "Debris clouds sealed orbit — every launch window became a gamble".into(),
                    "Earth's connection to its colonies hung by a thread".to_string(),
                )
            } else if desc.contains("ROBOT COMMISSIONED") {
                (
                    1,
                    desc.clone(),
                    "The workshop hummed with new purpose as the machine powered up".into(),
                    "Another pair of tireless hands joined the colony's workforce".to_string(),
                )
            } else if desc.contains("BROWNOUT CASCADE") {
                (3,
                 desc.clone(),
                 "Lights flickered. Robots froze mid-task. The colony held its breath".into(),
                 "Humans stepped into the hazardous modules their machines had protected them from".to_string())
            } else if desc.contains("MORAL DILEMMA") {
                (
                    3,
                    desc.clone(),
                    "The colony gathered in emergency session — two valid paths, one community"
                        .into(),
                    "The weight of the choice would echo through generations".to_string(),
                )
            } else if desc.contains("ETHICS SHIFT") {
                (
                    2,
                    desc.clone(),
                    "Old certainties gave way as the community's values quietly transformed".into(),
                    "The colony that emerged was not the one that had been founded".to_string(),
                )
            } else if desc.contains("ETHICAL SYNTHESIS") {
                (3,
                 desc.clone(),
                 "From the ashes of conflict, a new moral understanding crystallized".into(),
                 "Former adversaries found they had built something neither could have imagined alone".to_string())
            } else if desc.contains("ETHICAL FORK") {
                (4,
                 desc.clone(),
                 "The fracture ran too deep — two visions of the good life could not share one home".into(),
                 "A new colony was born from conviction, not convenience".to_string())
            } else if desc.contains("ETHICAL CRISIS") {
                (3,
                 desc.clone(),
                 "Competing moral frameworks tore at the colony's consensus — governance ground to a halt".into(),
                 "The question was no longer what to do, but who we are".to_string())
            } else if desc.contains("MORAL REVIVAL") {
                (3,
                 desc.clone(),
                 "The grief and anger that had long simmered finally broke through — the community could no longer ignore what it had become".into(),
                 "Something old and necessary returned: the sense that how we treat each other is not negotiable".to_string())
            } else {
                continue; // Skip events we don't narrate
            };

            // Deduplicate: don't narrate the same event description twice
            if self.events.iter().any(|e| e.crisis == crisis) {
                continue;
            }

            let world = event.world_id.map(|_| {
                // Extract world name from description if present
                desc.split(':').next().unwrap_or("Unknown").to_string()
            });

            // Generate named characters for significant events
            let character = if severity >= 2 {
                let world_name = event
                    .world_id
                    .map(|_| desc.split(':').next().unwrap_or("Colony"))
                    .unwrap_or("Colony");
                let role = if desc.contains("PROJECT") {
                    "engineer"
                } else if desc.contains("EXPLORATION") {
                    "scientist"
                } else if desc.contains("INDEPENDENCE") {
                    "leader"
                } else if desc.contains("MORAL DILEMMA")
                    || desc.contains("ETHICAL")
                    || desc.contains("MORAL REVIVAL")
                {
                    "ethicist"
                } else {
                    "citizen"
                };
                Some(generate_character_name(
                    world_name,
                    role,
                    (year / 25.0) as u16,
                ))
            } else {
                None
            };

            self.events.push(NarrativeEvent {
                tick,
                year,
                world,
                crisis,
                response,
                outcome,
                severity,
                joy: 0.0,
                sadness: 0.0,
                desire: 0.0,
                care: 0.0,
                character,
            });
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
                severity_marker, event.year, world_str, event.crisis, event.response, event.outcome
            ));
        }
        out
    }

    /// Generate a connected prose chronicle from the event history.
    /// Groups events into eras, weaves named characters, and produces
    /// paragraphs that read like a history book rather than a log.
    pub fn format_chronicle(&self) -> String {
        let mut sorted = self.events.clone();
        sorted.sort_by_key(|e| e.tick);
        if sorted.is_empty() {
            return String::from("No events recorded.\n");
        }

        let mut out = String::new();
        out.push_str("╔══════════════════════════════════════════════════════════╗\n");
        out.push_str("║   A CHRONICLE OF THE EXPANSION: 1000 YEARS IN SPACE    ║\n");
        out.push_str("╚══════════════════════════════════════════════════════════╝\n\n");

        // Group events into eras (50-year periods)
        let eras = [
            (
                0.0,
                50.0,
                "THE FOUNDING (Years 1-50)",
                "The first generation. Everything was fragile.",
            ),
            (
                50.0,
                150.0,
                "THE ESTABLISHMENT (Years 50-150)",
                "Technology unlocked survival. Populations grew.",
            ),
            (
                150.0,
                300.0,
                "THE EXPANSION (Years 150-300)",
                "Outer system colonies found their footing.",
            ),
            (
                300.0,
                500.0,
                "THE TRIALS (Years 300-500)",
                "Distance bred divergence. Crises tested the bonds.",
            ),
            (
                500.0,
                750.0,
                "THE MATURATION (Years 500-750)",
                "Civilizations within the civilization.",
            ),
            (
                750.0,
                1001.0,
                "THE MILLENNIUM (Years 750-1000)",
                "A species spanning worlds.",
            ),
        ];

        for (start, end, title, intro) in &eras {
            let era_events: Vec<&NarrativeEvent> = sorted
                .iter()
                .filter(|e| e.year >= *start && e.year < *end)
                .collect();
            if era_events.is_empty() {
                continue;
            }

            out.push_str(&format!("── {} ──\n\n", title));
            out.push_str(&format!("{}\n\n", intro));

            for event in &era_events {
                let world_str = event.world.as_deref().unwrap_or("across all worlds");

                // Affect-informed tone
                let tone = if event.sadness > 0.6 {
                    "The grief was palpable."
                } else if event.joy > 0.5 {
                    "Hope surged through the colony."
                } else if event.desire > 0.7 {
                    "Desperate determination drove the response."
                } else if event.care > 0.5 {
                    "The community closed ranks."
                } else {
                    "The response was measured."
                };

                // Character integration
                let char_clause = if let Some(ref ch) = event.character {
                    format!(" {} became the face of this moment.", ch)
                } else {
                    String::new()
                };

                // Severity determines paragraph weight
                match event.severity {
                    4 => {
                        out.push_str(&format!(
                            "  In Year {:.0}, {}: {}. {} {}{}\n\n",
                            event.year, world_str, event.crisis, event.response, tone, char_clause
                        ));
                        out.push_str(&format!("  The consequences: {}\n\n", event.outcome));
                    }
                    3 => {
                        out.push_str(&format!(
                            "  Year {:.0} — {}: {}. {}{}\n  Result: {}\n\n",
                            event.year,
                            world_str,
                            event.crisis,
                            event.response,
                            char_clause,
                            event.outcome
                        ));
                    }
                    2 => {
                        out.push_str(&format!(
                            "  Year {:.0}: {}. {}\n\n",
                            event.year, event.crisis, event.outcome
                        ));
                    }
                    _ => {
                        out.push_str(&format!("  Year {:.0}: {}\n", event.year, event.crisis));
                    }
                }
            }
            out.push('\n');
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_character_name_is_deterministic() {
        let a = generate_character_name("Mars", "engineer", 3);
        let b = generate_character_name("Mars", "engineer", 3);
        assert_eq!(a, b, "Same inputs should produce same name");
    }

    #[test]
    fn generate_character_name_varies_by_input() {
        let a = generate_character_name("Mars", "engineer", 1);
        let b = generate_character_name("Europa", "medic", 2);
        assert_ne!(a, b, "Different inputs should produce different names");
    }

    #[test]
    fn narrative_engine_new_is_empty() {
        let engine = NarrativeEngine::new();
        assert!(engine.events.is_empty());
        assert!(engine.cultural_memories.is_empty());
    }

    #[test]
    fn remember_stores_cultural_memory() {
        let mut engine = NarrativeEngine::new();
        engine.remember(
            "Great Quake".into(),
            Some("Moon".into()),
            100,
            CulturalLesson {
                preparedness: 0.2,
                risk_shift: -0.1,
                cohesion: 0.15,
            },
        );
        assert_eq!(engine.cultural_memories.len(), 1);
        assert_eq!(engine.cultural_memories[0].name, "Great Quake");
        assert!((engine.cultural_memories[0].strength - 1.0).abs() < 1e-10);
    }

    #[test]
    fn decay_memories_reduces_strength() {
        let mut engine = NarrativeEngine::new();
        engine.remember("Event".into(), None, 0, CulturalLesson::default());
        let initial = engine.cultural_memories[0].strength;
        for _ in 0..100 {
            engine.decay_memories();
        }
        assert!(
            engine.cultural_memories[0].strength < initial,
            "Strength should decay over 100 ticks"
        );
        assert!(
            engine.cultural_memories[0].strength > 0.01,
            "Should not be pruned after only 100 ticks"
        );
    }

    #[test]
    fn decay_memories_prunes_weak_memories() {
        let mut engine = NarrativeEngine::new();
        engine.remember("Ancient".into(), None, 0, CulturalLesson::default());
        // Force strength below threshold
        engine.cultural_memories[0].strength = 0.005;
        engine.decay_memories();
        assert!(
            engine.cultural_memories.is_empty(),
            "Memory below 0.01 threshold should be pruned"
        );
    }

    #[test]
    fn cultural_lesson_for_filters_by_world() {
        let mut engine = NarrativeEngine::new();
        engine.remember(
            "Mars Quake".into(),
            Some("Mars".into()),
            10,
            CulturalLesson {
                preparedness: 0.2,
                risk_shift: 0.0,
                cohesion: 0.0,
            },
        );
        engine.remember(
            "Global Pandemic".into(),
            None,
            20,
            CulturalLesson {
                preparedness: 0.0,
                risk_shift: -0.1,
                cohesion: 0.0,
            },
        );

        let mars_lesson = engine.cultural_lesson_for(Some("Mars"));
        assert!(
            mars_lesson.preparedness > 0.0,
            "Mars should get its own memory"
        );
        assert!(
            mars_lesson.risk_shift < 0.0,
            "Mars should also get global memory"
        );

        let moon_lesson = engine.cultural_lesson_for(Some("Moon"));
        assert!(
            (moon_lesson.preparedness - 0.0).abs() < 1e-10,
            "Moon should not get Mars-specific memory"
        );
        assert!(
            moon_lesson.risk_shift < 0.0,
            "Moon should get global memory"
        );
    }

    #[test]
    fn format_history_produces_output() {
        let mut engine = NarrativeEngine::new();
        engine.events.push(NarrativeEvent {
            tick: 10,
            year: 10.0 / 12.0,
            world: Some("Luna".into()),
            crisis: "Moonquake magnitude 5.2".into(),
            response: "Emergency shelters activated".into(),
            outcome: "Minor structural damage".into(),
            severity: 2,
            joy: 0.1,
            sadness: 0.6,
            desire: 0.3,
            care: 0.5,
            character: None,
        });
        let history = engine.format_history();
        assert!(
            !history.is_empty(),
            "History should produce non-empty output"
        );
        assert!(
            history.contains("Moonquake"),
            "History should contain crisis description"
        );
    }
}
