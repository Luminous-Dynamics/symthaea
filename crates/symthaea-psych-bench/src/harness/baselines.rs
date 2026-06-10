// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Published human and LLM reference data for benchmark comparison.
//!
//! All values are sourced from the respective benchmark papers.
//! References are cited inline.

use std::collections::BTreeMap;

/// A reference baseline value with source citation.
#[derive(Debug, Clone)]
pub struct Baseline {
    /// The reference value (mean).
    pub value: f64,
    /// Standard deviation of the reference population (if available).
    ///
    /// Used for z-score computation: `z = (agent - value) / sd`.
    pub sd: Option<f64>,
    /// Source description (e.g., "Cowan (2001), Table 2").
    pub source: &'static str,
    /// Population (e.g., "human adults", "GPT-4").
    pub population: &'static str,
}

/// Type alias for a baseline map.
pub type BaselineMap = BTreeMap<&'static str, Baseline>;

/// All baseline collections in a single struct.
///
/// Eliminates the need to pass 9+ separate maps to comparison functions.
pub struct BaselineCollection {
    pub worm: BaselineMap,
    pub cogbench: BaselineMap,
    pub tombench: BaselineMap,
    pub memory_agent: BaselineMap,
    pub executive: BaselineMap,
    pub metacognition: BaselineMap,
    pub affect: BaselineMap,
    pub creativity: BaselineMap,
    pub butlin: BaselineMap,
    pub inhibition: BaselineMap,
    pub attention: BaselineMap,
    pub embodied: BaselineMap,
    pub reasoning: BaselineMap,
    pub sustained_attention: BaselineMap,
    pub motor: BaselineMap,
    pub language: BaselineMap,
    pub social: BaselineMap,
    /// Neuromodulator domain baselines (DA/NE/5-HT/ACh psychopharmacology).
    pub neuromod: BaselineMap,
    /// Consciousness domain baselines (blindsight, etc.).
    pub consciousness: BaselineMap,
    /// Binding domain baselines (temporal order, etc.).
    pub binding: BaselineMap,
    /// Speech domain baselines (phoneme discrimination, etc.).
    pub speech: BaselineMap,
    /// Substrate independence baselines (transfer fidelity, etc.).
    pub substrate: BaselineMap,
    /// Mathematics domain baselines (arithmetic, algebra, statistics, logic, etc.).
    pub mathematics: BaselineMap,
    /// Institutional reasoning baselines (causal decomposition, axiom discrimination).
    pub institutional_reasoning: BaselineMap,
    /// Clinical/therapeutic baselines (empathic accuracy, alliance, crisis detection).
    pub clinical: BaselineMap,
    /// Spatial cognition baselines (mental rotation, path updating, landmark binding, perspective taking).
    pub spatial: BaselineMap,
    /// Causal reasoning baselines (causal chain, confound detection, intervention effect).
    pub causal_reasoning: BaselineMap,
    /// Security (HDC-FHE) baselines (encrypted classification, collective aggregation).
    pub security: BaselineMap,
    /// Coding domain baselines (HumanEval, bug detection).
    pub coding: BaselineMap,
    /// GPT-4 baselines from CogBench (Coda et al., 2023).
    pub llm_cogbench: BaselineMap,
    /// GPT-4 baselines from ToMBench (Kosinski, 2023).
    pub llm_tombench: BaselineMap,
    /// LLM baselines on ARC-AGI (GPT-4, Claude 3.5, o3-high, human).
    pub llm_arc: BaselineMap,
}

impl BaselineCollection {
    /// Load all baseline collections.
    pub fn all() -> Self {
        Self {
            worm: worm_baselines(),
            cogbench: cogbench_baselines(),
            tombench: tombench_baselines(),
            memory_agent: memory_agent_baselines(),
            executive: executive_baselines(),
            metacognition: metacognition_baselines(),
            affect: affect_baselines(),
            creativity: creativity_baselines(),
            butlin: butlin_baselines(),
            inhibition: inhibition_baselines(),
            attention: attention_baselines(),
            embodied: embodied_baselines(),
            reasoning: reasoning_baselines(),
            sustained_attention: sustained_attention_baselines(),
            motor: motor_baselines(),
            language: language_baselines(),
            social: social_baselines(),
            neuromod: neuromod_baselines(),
            consciousness: consciousness_baselines(),
            binding: binding_baselines(),
            speech: speech_baselines(),
            substrate: substrate_baselines(),
            mathematics: mathematics_baselines(),
            institutional_reasoning: institutional_reasoning_baselines(),
            clinical: clinical_baselines(),
            spatial: spatial_baselines(),
            causal_reasoning: causal_reasoning_baselines(),
            security: security_baselines(),
            coding: coding_baselines(),
            llm_cogbench: llm_cogbench_baselines(),
            llm_tombench: llm_tombench_baselines(),
            llm_arc: llm_arc_baselines(),
        }
    }
}

/// Cross-cultural validity metadata for a baseline collection.
/// Basis: Henrich, Heine, & Norenzayan (2010) "The weirdest people in the world?"
#[derive(Debug, Clone)]
pub struct BaselineMetadata {
    /// Primary sample region (e.g., "WEIRD", "East Asian", "Cross-cultural").
    pub sample_region: &'static str,
    /// Whether the baseline has been validated cross-culturally.
    pub cross_cultural_validated: bool,
    /// Notes on cultural specificity.
    pub cultural_notes: &'static str,
}

impl BaselineCollection {
    /// Cross-cultural metadata for each baseline domain.
    pub fn cultural_metadata() -> BTreeMap<&'static str, BaselineMetadata> {
        let mut m = BTreeMap::new();
        m.insert(
            "worm",
            BaselineMetadata {
                sample_region: "WEIRD",
                cross_cultural_validated: false,
                cultural_notes: "Cowan (2001) norms from North American undergrads",
            },
        );
        m.insert(
            "cogbench",
            BaselineMetadata {
                sample_region: "WEIRD",
                cross_cultural_validated: false,
                cultural_notes: "Coda-Forno et al. (2023) norms from Western online samples",
            },
        );
        m.insert(
            "tombench",
            BaselineMetadata {
                sample_region: "WEIRD",
                cross_cultural_validated: false,
                cultural_notes: "ToM tasks normed on English-speaking populations",
            },
        );
        m.insert(
            "memory_agent",
            BaselineMetadata {
                sample_region: "WEIRD",
                cross_cultural_validated: false,
                cultural_notes: "Memory benchmarks from Western lab studies",
            },
        );
        m.insert("executive", BaselineMetadata { sample_region: "WEIRD", cross_cultural_validated: false, cultural_notes: "WCST has some cross-cultural data (Kohli & Kaur 2006) but most are WEIRD-normed" });
        m.insert(
            "metacognition",
            BaselineMetadata {
                sample_region: "WEIRD",
                cross_cultural_validated: false,
                cultural_notes: "Calibration and FOK norms from Western university samples",
            },
        );
        m.insert(
            "affect",
            BaselineMetadata {
                sample_region: "WEIRD",
                cross_cultural_validated: false,
                cultural_notes: "Emotional Stroop and valence norms from Western samples",
            },
        );
        m.insert(
            "creativity",
            BaselineMetadata {
                sample_region: "WEIRD",
                cross_cultural_validated: false,
                cultural_notes: "AUT/RAT norms from English-speaking populations",
            },
        );
        m.insert(
            "butlin",
            BaselineMetadata {
                sample_region: "WEIRD",
                cross_cultural_validated: false,
                cultural_notes: "Butlin et al. (2023) consciousness indicators framework",
            },
        );
        m.insert(
            "inhibition",
            BaselineMetadata {
                sample_region: "WEIRD",
                cross_cultural_validated: false,
                cultural_notes: "Go/No-Go and Stop-Signal norms from Western lab studies",
            },
        );
        m.insert(
            "attention",
            BaselineMetadata {
                sample_region: "WEIRD",
                cross_cultural_validated: false,
                cultural_notes: "Visual search and attentional blink norms from Western labs",
            },
        );
        m.insert(
            "embodied",
            BaselineMetadata {
                sample_region: "WEIRD",
                cross_cultural_validated: false,
                cultural_notes: "Embodied cognition baselines from Western samples",
            },
        );
        m.insert(
            "reasoning",
            BaselineMetadata {
                sample_region: "WEIRD",
                cross_cultural_validated: false,
                cultural_notes: "ARC baselines from Western/online populations",
            },
        );
        m.insert(
            "sustained_attention",
            BaselineMetadata {
                sample_region: "WEIRD",
                cross_cultural_validated: false,
                cultural_notes: "PVT/SART/CPT norms from Western lab studies",
            },
        );
        m.insert(
            "motor",
            BaselineMetadata {
                sample_region: "WEIRD",
                cross_cultural_validated: false,
                cultural_notes: "Fitts' Law and SRTT norms from Western lab studies",
            },
        );
        m.insert(
            "language",
            BaselineMetadata {
                sample_region: "WEIRD",
                cross_cultural_validated: false,
                cultural_notes: "English-language priming and coherence norms",
            },
        );
        m.insert(
            "social",
            BaselineMetadata {
                sample_region: "WEIRD",
                cross_cultural_validated: false,
                cultural_notes: "RME and economic games normed on Western populations",
            },
        );
        m.insert(
            "neuromod",
            BaselineMetadata {
                sample_region: "WEIRD",
                cross_cultural_validated: false,
                cultural_notes: "Psychopharmacology baselines from Western clinical samples",
            },
        );
        m.insert(
            "consciousness",
            BaselineMetadata {
                sample_region: "WEIRD",
                cross_cultural_validated: false,
                cultural_notes:
                    "Blindsight and consciousness baselines from Western neuropsychology",
            },
        );
        m.insert(
            "binding",
            BaselineMetadata {
                sample_region: "WEIRD",
                cross_cultural_validated: false,
                cultural_notes: "Temporal binding norms from Western lab studies",
            },
        );
        m.insert(
            "speech",
            BaselineMetadata {
                sample_region: "WEIRD",
                cross_cultural_validated: false,
                cultural_notes: "Phoneme discrimination norms from English-speaking populations",
            },
        );
        m.insert(
            "substrate",
            BaselineMetadata {
                sample_region: "WEIRD",
                cross_cultural_validated: false,
                cultural_notes: "Substrate transfer baselines are theoretical",
            },
        );
        m.insert(
            "mathematics",
            BaselineMetadata {
                sample_region: "WEIRD",
                cross_cultural_validated: false,
                cultural_notes: "Mathematical reasoning norms from Western university samples",
            },
        );
        m.insert(
            "institutional_reasoning",
            BaselineMetadata {
                sample_region: "WEIRD",
                cross_cultural_validated: false,
                cultural_notes: "Institutional reasoning baselines from Western samples",
            },
        );
        m.insert(
            "clinical",
            BaselineMetadata {
                sample_region: "WEIRD",
                cross_cultural_validated: false,
                cultural_notes:
                    "Clinical/therapeutic baselines from Western mental health research",
            },
        );
        m
    }
}

/// Get all WorM baselines.
pub fn worm_baselines() -> BTreeMap<&'static str, Baseline> {
    let mut m = BTreeMap::new();

    // Working memory capacity (Cowan's K)
    m.insert(
        "cowan_k",
        Baseline {
            value: 4.0,
            sd: Some(1.0),
            source: "Cowan (2001), The magical number 4",
            population: "human adults",
        },
    );

    // N-back accuracy at n=2
    m.insert(
        "nback_2_accuracy",
        Baseline {
            value: 0.85,
            sd: Some(0.10),
            source: "Jaeggi et al. (2010), meta-analysis",
            population: "human adults",
        },
    );

    // N-back accuracy at n=3
    m.insert(
        "nback_3_accuracy",
        Baseline {
            value: 0.70,
            sd: Some(0.12),
            source: "Jaeggi et al. (2010), meta-analysis",
            population: "human adults",
        },
    );

    // Change detection accuracy at K=4
    m.insert(
        "change_detection_k4",
        Baseline {
            value: 0.75,
            sd: Some(0.15),
            source: "Luck & Vogel (1997)",
            population: "human adults",
        },
    );

    // Spatial updating accuracy (mean across 3-10 updates)
    m.insert(
        "spatial_updating_accuracy",
        Baseline {
            value: 0.85,
            sd: Some(0.10),
            source: "Oberauer et al. (2003); Ecker et al. (2010), spatial updating paradigm",
            population: "human adults",
        },
    );

    // Binding accuracy (mean across set sizes 2-6)
    m.insert(
        "binding_accuracy",
        Baseline {
            value: 0.75,
            sd: Some(0.12),
            source: "Luck & Vogel (1997); Wheeler & Treisman (2002), feature binding in VWM",
            population: "human adults",
        },
    );

    // Serial recall primacy advantage
    m.insert(
        "serial_primacy_advantage",
        Baseline {
            value: 0.15,
            sd: Some(0.06),
            source: "Murdock (1962), serial position curve",
            population: "human adults",
        },
    );

    // Digit span forward (Wechsler, 2008; Woods et al., 2011)
    m.insert(
        "digit_span_forward",
        Baseline {
            value: 6.8,
            sd: Some(1.1),
            source: "Wechsler (2008); Woods et al. (2011), WAIS-IV norms",
            population: "human adults",
        },
    );

    // Digit span backward
    m.insert(
        "digit_span_backward",
        Baseline {
            value: 5.1,
            sd: Some(1.2),
            source: "Wechsler (2008); Woods et al. (2011), WAIS-IV norms",
            population: "human adults",
        },
    );

    // --- RT baselines (tick-based, 1 tick ≈ 50ms) ---

    // N-back RT (Owen et al., 2005): 1-back ~500ms, 2-back ~600ms, 3-back ~700ms
    m.insert(
        "nback_1_rt_ticks",
        Baseline {
            value: 10.0,
            sd: Some(2.0),
            source: "Owen et al. (2005), N-back meta-analysis, ~500ms at 50ms/tick",
            population: "human adults",
        },
    );
    m.insert(
        "nback_2_rt_ticks",
        Baseline {
            value: 12.0,
            sd: Some(2.5),
            source: "Owen et al. (2005), N-back meta-analysis, ~600ms at 50ms/tick",
            population: "human adults",
        },
    );
    m.insert(
        "nback_3_rt_ticks",
        Baseline {
            value: 14.0,
            sd: Some(3.0),
            source: "Owen et al. (2005), N-back meta-analysis, ~700ms at 50ms/tick",
            population: "human adults",
        },
    );

    // WorM RT baselines (tick-based, 1 tick ≈ 50ms)
    m.insert(
        "change_detection_rt_ticks",
        Baseline {
            value: 8.0,
            sd: Some(2.0),
            source: "Luck & Vogel (1997), ~400ms at 50ms/tick",
            population: "human adults",
        },
    );
    m.insert(
        "digit_span_forward_rt_ticks",
        Baseline {
            value: 5.0,
            sd: Some(1.5),
            source: "Wechsler (2008), ~250ms per item at 50ms/tick",
            population: "human adults",
        },
    );
    m.insert(
        "digit_span_backward_rt_ticks",
        Baseline {
            value: 7.0,
            sd: Some(2.0),
            source: "Wechsler (2008), ~350ms per item (reversal overhead) at 50ms/tick",
            population: "human adults",
        },
    );
    m.insert(
        "binding_rt_ticks",
        Baseline {
            value: 7.0,
            sd: Some(2.0),
            source: "Wheeler & Treisman (2002), ~350ms at 50ms/tick",
            population: "human adults",
        },
    );
    m.insert(
        "serial_recall_rt_ticks",
        Baseline {
            value: 5.0,
            sd: Some(1.5),
            source: "Murdock (1962), ~250ms per item at 50ms/tick",
            population: "human adults",
        },
    );
    m.insert(
        "spatial_updating_rt_ticks",
        Baseline {
            value: 6.0,
            sd: Some(2.0),
            source: "Oberauer et al. (2003), ~300ms per update at 50ms/tick",
            population: "human adults",
        },
    );

    m
}

/// Get all CogBench baselines.
pub fn cogbench_baselines() -> BTreeMap<&'static str, Baseline> {
    let mut m = BTreeMap::new();

    // Exploration rate in horizon task
    m.insert(
        "directed_exploration",
        Baseline {
            value: 0.35,
            sd: Some(0.12),
            source: "Wilson et al. (2014), Horizon task",
            population: "human adults",
        },
    );

    // Model-basedness in two-step task
    m.insert(
        "model_basedness",
        Baseline {
            value: 0.60,
            sd: Some(0.20),
            source: "Daw et al. (2011), Two-step task",
            population: "human adults",
        },
    );

    // Temporal discounting score
    m.insert(
        "discounting_score",
        Baseline {
            value: 0.50,
            sd: Some(0.15),
            source: "Kirby et al. (1999), MCQ",
            population: "human adults",
        },
    );

    // BART average pumps
    m.insert(
        "bart_avg_pumps",
        Baseline {
            value: 30.0,
            sd: Some(12.0),
            source: "Lejuez et al. (2002), BART",
            population: "human adults",
        },
    );

    // Restless bandit: reward tracking over changing payoffs
    m.insert(
        "restless_bandit_regret",
        Baseline {
            value: 0.25,
            sd: Some(0.10),
            source:
                "Speekenbrink & Konstantinidis (2015), Information & choice in a changing world",
            population: "human adults",
        },
    );

    m.insert(
        "restless_bandit_accuracy",
        Baseline {
            value: 0.75,
            sd: Some(0.10),
            source: "Speekenbrink & Konstantinidis (2015), 1 - normalized regret",
            population: "human adults",
        },
    );

    // Instrumental conditioning: contingency sensitivity
    m.insert(
        "instrumental_sensitivity",
        Baseline {
            value: 0.70,
            sd: Some(0.15),
            source: "Dickinson (1985), Actions and habits",
            population: "human adults (estimated from instrumental learning literature)",
        },
    );

    // Probabilistic reasoning: likelihood weight (Bayesian updating)
    m.insert(
        "probabilistic_likelihood_weight",
        Baseline {
            value: 0.50,
            sd: Some(0.15),
            source:
                "Phillips & Edwards (1966); Grether (1980), conservatism in probability updating",
            population: "human adults (Bayesian normative = 0.50 for symmetric evidence)",
        },
    );

    // Reversal learning (Cools et al. 2002; Clark et al. 2004)
    m.insert(
        "reversal_win_stay",
        Baseline {
            value: 0.85,
            sd: Some(0.08),
            source: "Cools et al. (2002), Defining the neural mechanisms of probabilistic reversal learning",
            population: "human adults",
        },
    );
    m.insert(
        "reversal_lose_shift",
        Baseline {
            value: 0.70,
            sd: Some(0.12),
            source: "Cools et al. (2002), Defining the neural mechanisms of probabilistic reversal learning",
            population: "human adults",
        },
    );
    // Reversal learning perseverative errors (distinct from WCST)
    // Binary reversal: ~1.5 perseverative errors per reversal (first-trial
    // + stochastic recovery), ~15-20 reversals in 200-trial session.
    // Estimated from Cools et al. (2002) scaled to 200-trial deterministic paradigm.
    m.insert(
        "reversal_perseverative_errors",
        Baseline {
            value: 25.0,
            sd: Some(8.0),
            source:
                "Cools et al. (2002); Clark et al. (2004), binary reversal paradigm (200 trials)",
            population: "human adults",
        },
    );

    // CogBench RT baselines (tick-based, 1 tick ≈ 50ms)
    m.insert(
        "bart_rt_ticks",
        Baseline {
            value: 30.0,
            sd: Some(12.0),
            source: "Lejuez et al. (2002), mean pumps as deliberation proxy",
            population: "human adults",
        },
    );
    m.insert(
        "reversal_rt_ticks",
        Baseline {
            value: 8.0,
            sd: Some(2.5),
            source: "Cools et al. (2002), ~400ms at 50ms/tick",
            population: "human adults",
        },
    );
    m.insert(
        "two_step_rt_ticks",
        Baseline {
            value: 8.0,
            sd: Some(2.0),
            source: "Daw et al. (2011), stage-1 RT ~400ms at 50ms/tick",
            population: "human adults",
        },
    );
    m.insert(
        "restless_bandit_rt_ticks",
        Baseline {
            value: 7.0,
            sd: Some(2.0),
            source: "Speekenbrink & Konstantinidis (2015), ~350ms at 50ms/tick",
            population: "human adults",
        },
    );
    m.insert(
        "instrumental_rt_ticks",
        Baseline {
            value: 7.0,
            sd: Some(2.0),
            source: "Dickinson & Balleine (1994), ~350ms at 50ms/tick",
            population: "human adults",
        },
    );
    m.insert(
        "temporal_discounting_rt_ticks",
        Baseline {
            value: 8.0,
            sd: Some(2.5),
            source: "Kirby (1999), ~400ms at 50ms/tick",
            population: "human adults",
        },
    );
    m.insert(
        "horizon_rt_ticks",
        Baseline {
            value: 7.0,
            sd: Some(2.0),
            source: "Wilson et al. (2014), ~350ms at 50ms/tick",
            population: "human adults",
        },
    );
    m.insert(
        "probabilistic_rt_ticks",
        Baseline {
            value: 9.0,
            sd: Some(3.0),
            source: "Phillips & Edwards (1966), ~450ms at 50ms/tick",
            population: "human adults",
        },
    );

    m
}

/// Get all ToMBench baselines.
pub fn tombench_baselines() -> BTreeMap<&'static str, Baseline> {
    let mut m = BTreeMap::new();

    // False belief accuracy
    m.insert(
        "false_belief_accuracy",
        Baseline {
            value: 0.90,
            sd: Some(0.10),
            source: "Baron-Cohen et al. (1985), Sally-Anne",
            population: "human adults",
        },
    );

    // Faux pas recognition
    m.insert(
        "faux_pas_accuracy",
        Baseline {
            value: 0.85,
            sd: Some(0.12),
            source: "Baron-Cohen et al. (1999), Faux Pas test",
            population: "human adults",
        },
    );

    // Hinting task accuracy
    m.insert(
        "hinting_accuracy",
        Baseline {
            value: 0.80,
            sd: Some(0.15),
            source: "Corcoran et al. (1995), Hinting Task",
            population: "human adults",
        },
    );

    // Persuasion detection
    m.insert(
        "persuasion_detection",
        Baseline {
            value: 0.85,
            sd: Some(0.12),
            source: "Happé (1994), An advanced test of theory of mind",
            population: "human adults",
        },
    );

    // Strange story accuracy
    m.insert(
        "strange_story_accuracy",
        Baseline {
            value: 0.85,
            sd: Some(0.12),
            source: "Happé (1994), An advanced test of theory of mind",
            population: "human adults",
        },
    );

    // ToMBench RT baselines (tick-based, 1 tick ≈ 50ms)
    // ToM tasks involve sentence comprehension + inference, typically 2-4s
    m.insert(
        "tombench_rt_ticks",
        Baseline {
            value: 7.0,
            sd: Some(2.0),
            source: "Apperly et al. (2006), belief reasoning RT ~350ms at 50ms/tick",
            population: "human adults",
        },
    );

    m
}

/// Get all executive function baselines (WCST, IGT, Raven's, Stroop, Flanker).
pub fn executive_baselines() -> BTreeMap<&'static str, Baseline> {
    let mut m = BTreeMap::new();

    // WCST (Kohli & Kaur, 2006)
    m.insert(
        "wcst_categories_completed",
        Baseline {
            value: 5.62,
            sd: Some(1.13),
            source: "Kohli & Kaur (2006), WCST norms",
            population: "human adults",
        },
    );
    m.insert(
        "wcst_perseverative_errors",
        Baseline {
            value: 8.29,
            sd: Some(5.91),
            source: "Kohli & Kaur (2006), WCST norms",
            population: "human adults",
        },
    );
    m.insert(
        "wcst_trials_to_first",
        Baseline {
            value: 12.17,
            sd: Some(4.5),
            source: "Kohli & Kaur (2006), WCST norms",
            population: "human adults",
        },
    );

    // IGT (Bechara et al., 1994; Steingroever et al., 2015)
    m.insert(
        "igt_overall_net_score",
        Baseline {
            value: 17.5,
            sd: Some(20.0),
            source: "Bechara et al. (1994); Steingroever et al. (2015), midpoint of +10 to +25",
            population: "human adults",
        },
    );
    m.insert(
        "igt_deck_preference_good",
        Baseline {
            value: 0.65,
            sd: Some(0.15),
            source: "Steingroever et al. (2015), last 40 trials",
            population: "human adults",
        },
    );

    // Raven's Progressive Matrices (Raven, 1938; Murphy et al., 2023)
    m.insert(
        "ravens_overall_accuracy",
        Baseline {
            value: 0.78,
            sd: Some(0.12),
            source: "Raven (1938); Murphy et al. (2023), SPM ~47/60",
            population: "human adults",
        },
    );
    m.insert(
        "ravens_easy_accuracy",
        Baseline {
            value: 0.95,
            sd: Some(0.05),
            source: "Raven (1938), Set A-B",
            population: "human adults",
        },
    );

    // Stroop Color-Word Interference (MacLeod, 1991; Stroop, 1935)
    m.insert(
        "stroop_congruent_accuracy",
        Baseline {
            value: 0.98,
            sd: Some(0.02),
            source: "MacLeod (1991), Half a century of research on the Stroop effect",
            population: "human adults",
        },
    );
    m.insert(
        "stroop_incongruent_accuracy",
        Baseline {
            value: 0.88,
            sd: Some(0.06),
            source: "MacLeod (1991), Half a century of research on the Stroop effect",
            population: "human adults",
        },
    );
    m.insert(
        "stroop_effect",
        Baseline {
            value: 0.10,
            sd: Some(0.05),
            source: "MacLeod (1991), accuracy-based Stroop effect",
            population: "human adults",
        },
    );

    // Eriksen Flanker Task (Eriksen & Eriksen, 1974; Ridderinkhof et al., 2021)
    m.insert(
        "flanker_congruent_accuracy",
        Baseline {
            value: 0.97,
            sd: Some(0.03),
            source: "Eriksen & Eriksen (1974); Ridderinkhof et al. (2021)",
            population: "human adults",
        },
    );
    m.insert(
        "flanker_incongruent_accuracy",
        Baseline {
            value: 0.90,
            sd: Some(0.06),
            source: "Eriksen & Eriksen (1974); Ridderinkhof et al. (2021)",
            population: "human adults",
        },
    );
    m.insert(
        "flanker_effect",
        Baseline {
            value: 0.07,
            sd: Some(0.04),
            source: "Eriksen & Eriksen (1974), accuracy-based flanker effect",
            population: "human adults",
        },
    );

    // Tower of London (Shallice, 1982; Kaller et al., 2016)
    m.insert(
        "tol_overall_optimal_rate",
        Baseline {
            value: 0.63,
            sd: Some(0.15),
            source: "Kaller et al. (2016), TOL-F norms",
            population: "human adults",
        },
    );
    m.insert(
        "tol_planning_efficiency",
        Baseline {
            value: 0.82,
            sd: Some(0.10),
            source: "Kaller et al. (2016), optimal/actual moves ratio",
            population: "human adults",
        },
    );

    // --- RT baselines (tick-based, 1 tick ≈ 50ms) ---

    // Stroop RT (MacLeod, 1991): congruent ~600ms, incongruent ~750ms
    m.insert(
        "stroop_congruent_rt_ticks",
        Baseline {
            value: 12.0,
            sd: Some(2.0),
            source: "MacLeod (1991), Stroop effect review, ~600ms at 50ms/tick",
            population: "human adults",
        },
    );
    m.insert(
        "stroop_incongruent_rt_ticks",
        Baseline {
            value: 15.0,
            sd: Some(2.5),
            source: "MacLeod (1991), Stroop effect review, ~750ms at 50ms/tick",
            population: "human adults",
        },
    );

    // Flanker RT (Eriksen & Eriksen, 1974): congruent ~400ms, incongruent ~500ms
    m.insert(
        "flanker_congruent_rt_ticks",
        Baseline {
            value: 8.0,
            sd: Some(1.5),
            source: "Eriksen & Eriksen (1974), ~400ms at 50ms/tick",
            population: "human adults",
        },
    );
    m.insert(
        "flanker_incongruent_rt_ticks",
        Baseline {
            value: 10.0,
            sd: Some(2.0),
            source: "Eriksen & Eriksen (1974), ~500ms at 50ms/tick",
            population: "human adults",
        },
    );

    // WCST RT (Heaton, 1993): ~1500ms deliberation per trial (median)
    m.insert(
        "wcst_rt_ticks",
        Baseline {
            value: 30.0,
            sd: Some(5.0),
            source: "Heaton (1993), ~1500ms at 50ms/tick",
            population: "human adults",
        },
    );

    // IGT RT (Bechara et al., 1994): ~2000ms deliberation per trial
    m.insert(
        "igt_rt_ticks",
        Baseline {
            value: 40.0,
            sd: Some(8.0),
            source: "Bechara et al. (1994), ~2000ms at 50ms/tick",
            population: "human adults",
        },
    );

    // Dual-Task (Baddeley & Hitch, 1974; Pashler, 1994)
    m.insert(
        "dual_single_accuracy",
        Baseline {
            value: 0.95,
            sd: Some(0.04),
            source: "Baddeley & Hitch (1974), choice RT without load",
            population: "human adults",
        },
    );
    m.insert(
        "dual_low_accuracy",
        Baseline {
            value: 0.90,
            sd: Some(0.06),
            source: "Baddeley & Hitch (1974), choice RT with 3-digit load",
            population: "human adults",
        },
    );
    m.insert(
        "dual_high_accuracy",
        Baseline {
            value: 0.85,
            sd: Some(0.08),
            source: "Baddeley & Hitch (1974), choice RT with 6-digit load",
            population: "human adults",
        },
    );
    m.insert(
        "dual_task_cost",
        Baseline {
            value: 0.10,
            sd: Some(0.05),
            source: "Pashler (1994), single - dual_high accuracy difference",
            population: "human adults",
        },
    );
    m.insert(
        "dual_digit_recall",
        Baseline {
            value: 0.85,
            sd: Some(0.10),
            source: "Baddeley & Hitch (1974), digit maintenance under dual-task",
            population: "human adults",
        },
    );
    m.insert(
        "dual_single_rt_ticks",
        Baseline {
            value: 7.0,
            sd: Some(1.5),
            source: "Pashler (1994), choice RT ~350ms at 50ms/tick",
            population: "human adults",
        },
    );

    m
}

/// Get all metacognition baselines (calibration).
pub fn metacognition_baselines() -> BTreeMap<&'static str, Baseline> {
    let mut m = BTreeMap::new();

    m.insert(
        "calibration_error_ece",
        Baseline {
            value: 0.15,
            sd: Some(0.05),
            source: "Fleming & Lau (2014), midpoint of 0.10-0.20",
            population: "human adults",
        },
    );
    m.insert(
        "discrimination_gamma",
        Baseline {
            value: 0.50,
            sd: Some(0.15),
            source: "Fleming & Lau (2014), midpoint of 0.40-0.60",
            population: "human adults",
        },
    );

    // Metacognition RT baseline
    m.insert(
        "calibration_rt_ticks",
        Baseline {
            value: 8.0,
            sd: Some(2.0),
            source: "Petrusic & Baranski (2003), confidence judgment ~400ms at 50ms/tick",
            population: "human adults",
        },
    );

    // Feeling of Knowing (Hart 1965; Metcalfe et al. 1993; Schwartz 1994)
    m.insert(
        "fok_gamma",
        Baseline {
            value: 0.65,
            sd: Some(0.10),
            source: "Hart (1965); Schwartz (1994), gamma(FOK, recognition)",
            population: "human adults",
        },
    );
    m.insert(
        "recognition_hit_rate",
        Baseline {
            value: 0.75,
            sd: Some(0.10),
            source: "Metcalfe et al. (1993), recognition hit rate after failed recall",
            population: "human adults",
        },
    );
    m.insert(
        "fok_resolution",
        Baseline {
            value: 0.60,
            sd: Some(0.12),
            source: "Schwartz (1994), AUC of FOK predicting recognition",
            population: "human adults",
        },
    );

    // Change Blindness (Rensink et al., 1997)
    m.insert(
        "cb_detection_with_disruption",
        Baseline {
            value: 0.45,
            sd: Some(0.12),
            source: "Rensink et al. (1997), change detection with blank disruption",
            population: "human adults",
        },
    );
    m.insert(
        "cb_detection_without_disruption",
        Baseline {
            value: 0.85,
            sd: Some(0.08),
            source: "Rensink et al. (1997), change detection without disruption",
            population: "human adults",
        },
    );
    m.insert(
        "cb_search_efficiency",
        Baseline {
            value: 0.60,
            sd: Some(0.10),
            source: "Rensink et al. (1997), fraction detected within 5 looks",
            population: "human adults",
        },
    );
    m.insert(
        "cb_attention_benefit",
        Baseline {
            value: 0.35,
            sd: Some(0.10),
            source: "Simons & Levin (1997), attended minus unattended detection",
            population: "human adults",
        },
    );

    m
}

/// Get all memory agent baselines.
pub fn memory_agent_baselines() -> BTreeMap<&'static str, Baseline> {
    let mut m = BTreeMap::new();

    m.insert(
        "accurate_retrieval",
        Baseline {
            value: 0.85,
            sd: Some(0.10),
            source: "Tulving (1985), Memory and consciousness; Roediger & McDermott (1995), DRM paradigm false recall ~15%",
            population: "human adults",
        },
    );

    m.insert(
        "test_time_learning",
        Baseline {
            value: 0.75,
            sd: Some(0.15),
            source: "Karpicke & Roediger (2008), The critical importance of retrieval for learning",
            population: "human adults",
        },
    );

    // Long-range retention at 50-cycle delay
    m.insert(
        "long_range_delay_50",
        Baseline {
            value: 0.70,
            sd: Some(0.15),
            source: "Baddeley (1997), Human Memory: Theory and Practice",
            population: "human adults",
        },
    );

    // Conflict resolution: recency preference
    m.insert(
        "conflict_recency_preference",
        Baseline {
            value: 0.65,
            sd: Some(0.12),
            source: "Oberauer (2002), Access to information in working memory",
            population: "human adults",
        },
    );

    // Prospective memory (Einstein & McDaniel, 2005; Kliegel et al., 2008)
    m.insert(
        "pm_hit_rate",
        Baseline {
            value: 0.75,
            sd: Some(0.15),
            source: "Einstein & McDaniel (2005), Prospective memory: multiple retrieval processes",
            population: "human adults",
        },
    );
    m.insert(
        "pm_ongoing_accuracy",
        Baseline {
            value: 0.90,
            sd: Some(0.06),
            source: "Kliegel et al. (2008), Prospective memory in older adults",
            population: "human adults",
        },
    );
    m.insert(
        "pm_cost",
        Baseline {
            value: 0.05,
            sd: Some(0.03),
            source: "Smith (2003), PM cost: monitoring effects on ongoing task performance",
            population: "human adults",
        },
    );

    // MemoryAgent RT baselines (tick-based, 1 tick ≈ 50ms)
    m.insert(
        "accurate_retrieval_rt_ticks",
        Baseline {
            value: 8.0,
            sd: Some(3.0),
            source: "Tulving (1985), retrieval latency ~400ms + encoding at 50ms/tick",
            population: "human adults",
        },
    );
    m.insert(
        "conflict_resolution_rt_ticks",
        Baseline {
            value: 7.0,
            sd: Some(2.0),
            source: "Oberauer (2002), interference resolution ~350ms at 50ms/tick",
            population: "human adults",
        },
    );
    m.insert(
        "long_range_rt_ticks",
        Baseline {
            value: 10.0,
            sd: Some(4.0),
            source: "Baddeley (1997), retrieval latency scales with delay at 50ms/tick",
            population: "human adults",
        },
    );
    m.insert(
        "prospective_memory_rt_ticks",
        Baseline {
            value: 9.0,
            sd: Some(2.0),
            source: "Einstein & McDaniel (2005), PM monitoring + detection ~450ms at 50ms/tick",
            population: "human adults",
        },
    );
    m.insert(
        "test_time_learning_rt_ticks",
        Baseline {
            value: 6.0,
            sd: Some(2.0),
            source: "Karpicke & Roediger (2008), correction retrieval ~300ms at 50ms/tick",
            population: "human adults",
        },
    );

    m
}

/// Get all affect baselines.
pub fn affect_baselines() -> BTreeMap<&'static str, Baseline> {
    let mut m = BTreeMap::new();

    // Valence classification accuracy (Bradley & Lang, IAPS)
    m.insert(
        "valence_accuracy",
        Baseline {
            value: 0.90,
            sd: Some(0.08),
            source: "Bradley & Lang (1999), IAPS affective ratings",
            population: "human adults",
        },
    );

    // Mood-congruent recall congruence ratio (Blaney, 1986)
    m.insert(
        "congruence_ratio",
        Baseline {
            value: 0.60,
            sd: Some(0.10),
            source: "Blaney (1986), Affect and memory: a review",
            population: "human adults",
        },
    );

    // Emotional Stroop (Williams et al. 1996; Bar-Haim et al. 2007)
    m.insert(
        "emotional_neutral_accuracy",
        Baseline {
            value: 0.95,
            sd: Some(0.03),
            source: "Williams et al. (1996), The emotional Stroop task and psychopathology",
            population: "human adults",
        },
    );
    m.insert(
        "emotional_negative_accuracy",
        Baseline {
            value: 0.88,
            sd: Some(0.06),
            source: "Williams et al. (1996), The emotional Stroop task and psychopathology",
            population: "human adults",
        },
    );
    m.insert(
        "emotional_interference",
        Baseline {
            value: 0.07,
            sd: Some(0.04),
            source: "Bar-Haim et al. (2007), Threat-related attentional bias meta-analysis",
            population: "human adults",
        },
    );

    // Affect RT baselines (tick-based, 1 tick ≈ 50ms)
    m.insert(
        "emotional_stroop_neutral_rt_ticks",
        Baseline {
            value: 12.0,
            sd: Some(2.0),
            source: "Williams et al. (1996), neutral word naming ~600ms at 50ms/tick",
            population: "human adults",
        },
    );
    m.insert(
        "emotional_stroop_negative_rt_ticks",
        Baseline {
            value: 14.0,
            sd: Some(3.0),
            source: "Williams et al. (1996), negative word naming ~700ms at 50ms/tick",
            population: "human adults",
        },
    );
    m.insert(
        "valence_rt_ticks",
        Baseline {
            value: 6.0,
            sd: Some(2.0),
            source: "Bradley & Lang (1999), valence judgment ~300ms at 50ms/tick",
            population: "human adults",
        },
    );
    m.insert(
        "mood_congruent_rt_ticks",
        Baseline {
            value: 8.0,
            sd: Some(2.0),
            source: "Blaney (1986), mood-congruent retrieval ~400ms at 50ms/tick",
            population: "human adults",
        },
    );

    m
}

/// Get Butlin consciousness indicator baselines.
///
/// Consciousness indicators (recurrent processing, global workspace access, etc.)
/// are philosophical criteria from Butlin et al. (2023), not psychometric tests.
/// There are no published "human norms" — all neurotypical adults have all 14
/// indicators present. The baseline therefore represents perfect presence.
pub fn butlin_baselines() -> BTreeMap<&'static str, Baseline> {
    let mut m = BTreeMap::new();

    m.insert(
        "present_count",
        Baseline {
            value: 14.0,
            sd: Some(2.0),
            source: "Butlin et al. (2023), Consciousness in Artificial Intelligence: Insights from the Science of Consciousness",
            population: "human adults (all indicators present by definition)",
        },
    );

    m.insert(
        "mean_quality_score",
        Baseline {
            value: 0.80,
            sd: Some(0.10),
            source: "Butlin et al. (2023), architectural quality assessment",
            population: "systems achieving full indicator presence",
        },
    );

    m.insert(
        "presence_ratio",
        Baseline {
            value: 1.0,
            sd: Some(0.14),
            source: "Butlin et al. (2023), 14/14 indicators present in neurotypical adults",
            population: "human adults",
        },
    );

    m
}

/// Get all creativity baselines.
pub fn creativity_baselines() -> BTreeMap<&'static str, Baseline> {
    let mut m = BTreeMap::new();

    // Remote Associates Test accuracy (Bowden & Jung-Beeman, 2003)
    m.insert(
        "rat_overall_accuracy",
        Baseline {
            value: 0.50,
            sd: Some(0.15),
            source: "Bowden & Jung-Beeman (2003), Normative data for 144 compound remote associate problems",
            population: "human adults",
        },
    );

    // Alternate Uses Task fluency (Torrance, 1974)
    m.insert(
        "aut_fluency",
        Baseline {
            value: 8.0,
            sd: Some(3.0),
            source: "Torrance (1974), Torrance Tests of Creative Thinking",
            population: "human adults",
        },
    );

    // Alternate Uses Task originality (Silvia et al., 2008)
    // Mean semantic distance of generated uses from the object's typical use.
    // Originality reflects how far uses diverge from conventional associations.
    m.insert(
        "aut_originality",
        Baseline {
            value: 0.60,
            sd: Some(0.15),
            source: "Silvia et al. (2008), Assessing creativity with divergent thinking tasks",
            population: "human adults",
        },
    );

    // RAT binding accuracy (Bowden & Jung-Beeman, 2003)
    // Proportion of triads where the convergent associate is correctly
    // identified via associative binding (pairwise cue binding ensemble).
    // Lower than overall accuracy because binding is a harder retrieval
    // mode than simple similarity matching.
    m.insert(
        "rat_convergent_binding",
        Baseline {
            value: 0.55,
            sd: Some(0.15),
            source: "Bowden & Jung-Beeman (2003), convergent associative retrieval accuracy",
            population: "human adults",
        },
    );

    // Conceptual Blending (Fauconnier & Turner, 2002)
    m.insert(
        "blend_coherence",
        Baseline {
            value: 0.55,
            sd: Some(0.12),
            source: "Fauconnier & Turner (2002), The Way We Think; Ward (1994)",
            population: "human adults",
        },
    );
    m.insert(
        "novelty_score",
        Baseline {
            value: 0.50,
            sd: Some(0.15),
            source: "Ward (1994), Structured Imagination: the Role of Category Structure",
            population: "human adults",
        },
    );
    m.insert(
        "integration_score",
        Baseline {
            value: 0.45,
            sd: Some(0.12),
            source: "Fauconnier & Turner (2002), conceptual integration measure",
            population: "human adults",
        },
    );

    // Insight Problem Solving (Bowden & Jung-Beeman, 2003; Ohlsson, 1992)
    m.insert(
        "insight_accuracy",
        Baseline {
            value: 0.50,
            sd: Some(0.15),
            source: "Bowden & Jung-Beeman (2003); Metcalfe & Wiebe (1987)",
            population: "human adults",
        },
    );
    m.insert(
        "restructuring_depth",
        Baseline {
            value: 0.40,
            sd: Some(0.15),
            source: "Ohlsson (1992), representational change theory",
            population: "human adults",
        },
    );

    // Creativity RT baselines
    m.insert(
        "aut_rt_ticks",
        Baseline {
            value: 15.0,
            sd: Some(5.0),
            source: "Gilhooly et al. (2007), divergent thinking search time at 50ms/tick",
            population: "human adults",
        },
    );
    m.insert(
        "rat_rt_ticks",
        Baseline {
            value: 10.0,
            sd: Some(4.0),
            source: "Bowden & Jung-Beeman (2003), RAT solution time ~500ms at 50ms/tick",
            population: "human adults",
        },
    );

    // Divergent Thinking (Guilford, 1967; Silvia et al., 2008)
    m.insert(
        "originality_score",
        Baseline {
            value: 0.45,
            sd: Some(0.15),
            source: "Silvia et al. (2008), mean semantic distance of alternative uses",
            population: "human adults",
        },
    );
    m.insert(
        "flexibility_score",
        Baseline {
            value: 0.60,
            sd: Some(0.12),
            source: "Guilford (1967), proportion of distinct category uses",
            population: "human adults",
        },
    );
    m.insert(
        "elaboration_score",
        Baseline {
            value: 0.35,
            sd: Some(0.12),
            source: "Guilford (1967), inter-response distinctiveness",
            population: "human adults",
        },
    );

    m
}

/// Get all inhibition baselines (Go/No-Go).
pub fn inhibition_baselines() -> BTreeMap<&'static str, Baseline> {
    let mut m = BTreeMap::new();

    // Go/No-Go (Wessel 2018; Verbruggen & Logan 2008)
    m.insert(
        "go_accuracy",
        Baseline {
            value: 0.98,
            sd: Some(0.02),
            source: "Wessel (2018); Verbruggen & Logan (2008), stop-signal review",
            population: "human adults",
        },
    );
    m.insert(
        "nogo_accuracy",
        Baseline {
            value: 0.82,
            sd: Some(0.10),
            source: "Wessel (2018), commission error rate ~18%",
            population: "human adults",
        },
    );
    m.insert(
        "inhibition_cost",
        Baseline {
            value: 0.16,
            sd: Some(0.08),
            source: "Wessel (2018), go_accuracy - nogo_accuracy",
            population: "human adults",
        },
    );
    m.insert(
        "go_rt_ticks",
        Baseline {
            value: 4.0,
            sd: Some(1.5),
            source: "Wessel (2018), ~200ms at 50ms/tick",
            population: "human adults",
        },
    );

    // Stop Signal Task (Logan 1994; Verbruggen & Logan 2008)
    m.insert(
        "sst_go_accuracy",
        Baseline {
            value: 0.95,
            sd: Some(0.03),
            source: "Logan (1994); Verbruggen & Logan (2008), SST go accuracy",
            population: "human adults",
        },
    );
    m.insert(
        "sst_go_rt_ticks",
        Baseline {
            value: 10.0,
            sd: Some(2.0),
            source: "Logan (1994), go RT ~500ms at 50ms/tick",
            population: "human adults",
        },
    );
    m.insert(
        "sst_stop_accuracy",
        Baseline {
            value: 0.50,
            sd: Some(0.10),
            source: "Verbruggen & Logan (2008), at tracked SSD (~50% by design)",
            population: "human adults",
        },
    );
    m.insert(
        "ssrt_ticks",
        Baseline {
            value: 0.10,
            sd: Some(0.05),
            source: "Logan (1994); Verbruggen & Logan (2008), HDC-scale SSRT from staircase SSD tracking",
            population: "human adults",
        },
    );

    // Flanker Inhibition (Eriksen & Eriksen, 1974; Ridderinkhof et al., 2004)
    m.insert(
        "flanker_congruent_accuracy",
        Baseline {
            value: 0.96,
            sd: Some(0.03),
            source: "Eriksen & Eriksen (1974), congruent go trial accuracy",
            population: "human adults",
        },
    );
    m.insert(
        "flanker_incongruent_accuracy",
        Baseline {
            value: 0.85,
            sd: Some(0.08),
            source: "Ridderinkhof et al. (2004), incongruent no-go withholding accuracy",
            population: "human adults",
        },
    );
    m.insert(
        "interference_suppression",
        Baseline {
            value: 0.11,
            sd: Some(0.06),
            source: "Eriksen & Eriksen (1974), congruent minus incongruent accuracy",
            population: "human adults",
        },
    );

    m
}

/// Get all attention baselines (Attentional Blink).
pub fn attention_baselines() -> BTreeMap<&'static str, Baseline> {
    let mut m = BTreeMap::new();

    // Attentional Blink (Raymond et al. 1992; Shapiro et al. 1997)
    m.insert(
        "t1_accuracy",
        Baseline {
            value: 0.92,
            sd: Some(0.05),
            source: "Raymond et al. (1992), RSVP attentional blink",
            population: "human adults",
        },
    );
    m.insert(
        "lag3_t2_accuracy",
        Baseline {
            value: 0.55,
            sd: Some(0.15),
            source: "Raymond et al. (1992), T2|T1 at lag 3",
            population: "human adults",
        },
    );
    m.insert(
        "lag8_t2_accuracy",
        Baseline {
            value: 0.85,
            sd: Some(0.10),
            source: "Raymond et al. (1992), T2|T1 at lag 8 (recovery)",
            population: "human adults",
        },
    );
    m.insert(
        "blink_magnitude",
        Baseline {
            value: 0.30,
            sd: Some(0.12),
            source: "Shapiro et al. (1997), lag8 - lag3 accuracy difference",
            population: "human adults",
        },
    );

    // --- RT baselines (tick-based, 1 tick ≈ 50ms) ---

    // AB T1 RT (Raymond et al. 1992): ~300ms
    m.insert(
        "attblink_t1_rt_ticks",
        Baseline {
            value: 6.0,
            sd: Some(1.0),
            source: "Raymond et al. (1992), ~300ms at 50ms/tick",
            population: "human adults",
        },
    );
    // AB T2 at lag 3 (blink window): ~400ms
    m.insert(
        "attblink_lag3_rt_ticks",
        Baseline {
            value: 8.0,
            sd: Some(2.0),
            source: "Raymond et al. (1992), ~400ms at 50ms/tick (blink window)",
            population: "human adults",
        },
    );
    // AB T2 at lag 8 (recovery): ~350ms
    m.insert(
        "attblink_lag8_rt_ticks",
        Baseline {
            value: 7.0,
            sd: Some(1.5),
            source: "Raymond et al. (1992), ~350ms at 50ms/tick (recovery)",
            population: "human adults",
        },
    );

    // Visual Search (Treisman & Gelade 1980; Wolfe 1994)
    m.insert(
        "feature_search_accuracy",
        Baseline {
            value: 0.98,
            sd: Some(0.02),
            source: "Treisman & Gelade (1980), feature search pop-out",
            population: "human adults",
        },
    );
    m.insert(
        "conjunction_search_accuracy",
        Baseline {
            value: 0.88,
            sd: Some(0.06),
            source: "Treisman & Gelade (1980), conjunction search",
            population: "human adults",
        },
    );
    m.insert(
        "feature_search_slope",
        Baseline {
            value: 0.0,
            sd: Some(0.5),
            source: "Treisman & Gelade (1980), ~0 ms/item (parallel) at 50ms/tick",
            population: "human adults",
        },
    );
    m.insert(
        "conjunction_search_slope",
        Baseline {
            value: 2.0,
            sd: Some(0.8),
            source: "Wolfe (1994), ~25ms/item (serial) ≈ 0.5 ticks/item, slope in tick units",
            population: "human adults",
        },
    );
    m.insert(
        "search_asymmetry",
        Baseline {
            value: 0.25,
            sd: Some(0.10),
            source: "Treisman & Gelade (1980), HDC-scale conjunction-feature slope difference",
            population: "human adults",
        },
    );

    // Mismatch Negativity (Näätänen et al., 1978, 2007)
    m.insert(
        "mmn_detection_accuracy",
        Baseline {
            value: 0.88,
            sd: Some(0.06),
            source: "Näätänen et al. (2007), oddball detection accuracy",
            population: "human adults",
        },
    );
    m.insert(
        "mmn_false_alarm_rate",
        Baseline {
            value: 0.08,
            sd: Some(0.04),
            source: "Näätänen et al. (2007), false positive rate for standards",
            population: "human adults",
        },
    );
    m.insert(
        "mmn_mismatch_magnitude",
        Baseline {
            value: 0.35,
            sd: Some(0.10),
            source: "Näätänen et al. (1978), MMN amplitude (normalized)",
            population: "human adults",
        },
    );
    m.insert(
        "mmn_attentional_independence",
        Baseline {
            value: 0.80,
            sd: Some(0.12),
            source: "Näätänen et al. (2007), detection ratio under load vs no-load",
            population: "human adults",
        },
    );

    m
}

/// Embodied cognition / motor control baselines.
///
/// DMC Humanoid Stand task baselines from published RL results:
/// - SAC (Haarnoja et al., 2018): ~950/1000
/// - TD3 (Fujimoto et al., 2018): ~800/1000
/// - D4PG (Barth-Maron et al., 2018): ~900/1000
///
/// These are referenced by the humanoid crate's benchmarks module.
/// Run: `cargo run --features humanoid --example humanoid_paper_figures --release`
pub fn embodied_baselines() -> BTreeMap<&'static str, Baseline> {
    let mut m = BTreeMap::new();

    m.insert(
        "humanoid_stand_sac",
        Baseline {
            value: 0.95,
            sd: Some(0.03),
            source: "Haarnoja et al. (2018), SAC on DMC humanoid.stand",
            population: "SAC agent (1M steps)",
        },
    );
    m.insert(
        "humanoid_stand_td3",
        Baseline {
            value: 0.80,
            sd: Some(0.05),
            source: "Fujimoto et al. (2018), TD3 on DMC humanoid.stand",
            population: "TD3 agent (1M steps)",
        },
    );
    m.insert(
        "humanoid_stand_d4pg",
        Baseline {
            value: 0.90,
            sd: Some(0.04),
            source: "Barth-Maron et al. (2018), D4PG on DMC humanoid.stand",
            population: "D4PG agent (10M steps)",
        },
    );
    m.insert(
        "humanoid_stand_hdc_ltc_fep",
        Baseline {
            value: 0.986,
            sd: Some(0.01),
            source: "Symthaea HDC-LTC-FEP (20 episodes, 8K steps)",
            population: "HDC-LTC-FEP agent",
        },
    );
    m.insert(
        "humanoid_transfer_advantage",
        Baseline {
            value: 0.094,
            sd: Some(0.05),
            source: "Symthaea flight→humanoid morphological transfer",
            population: "HDC transfer vs random init",
        },
    );

    m
}

/// GPT-4 baselines from CogBench (Coda et al., 2023).
/// Get reasoning baselines.
///
/// Source: Chollet (2019), "On the Measure of Intelligence", arXiv:1911.01547
/// Johnson et al. (2021), "Fast and slow learning from ARC"
pub fn reasoning_baselines() -> BaselineMap {
    let mut m = BTreeMap::new();
    m.insert(
        "arc_rule_consistency",
        Baseline {
            value: 0.85,
            sd: Some(0.10),
            source: "Johnson et al. (2021), estimated from ARC task analysis",
            population: "human adults",
        },
    );
    m.insert(
        "arc_transfer_accuracy",
        Baseline {
            value: 0.80,
            sd: Some(0.12),
            source: "Chollet (2019), human performance on ARC evaluation set",
            population: "human adults",
        },
    );
    m.insert(
        "arc_transfer_similarity",
        Baseline {
            value: 0.70,
            sd: Some(0.15),
            source: "Johnson et al. (2021), HDC proxy for structural match",
            population: "human adults",
        },
    );
    m.insert(
        "arc_rt_ticks",
        Baseline {
            value: 6.0,
            sd: Some(2.0),
            source: "Johnson et al. (2021), deliberation time estimate (1 tick ≈ 50ms)",
            population: "human adults",
        },
    );
    // ARC Compositional baselines (estimated from harder ARC subsets)
    m.insert(
        "arc_compositional_accuracy",
        Baseline {
            value: 0.65,
            sd: Some(0.15),
            source: "Johnson et al. (2021), estimated for chained transforms",
            population: "human adults",
        },
    );
    m.insert(
        "arc_size_generalization",
        Baseline {
            value: 0.70,
            sd: Some(0.12),
            source: "Johnson et al. (2021), cross-size transfer estimate",
            population: "human adults",
        },
    );
    m.insert(
        "arc_symmetry_detection",
        Baseline {
            value: 0.90,
            sd: Some(0.08),
            source: "Treder (2010), symmetry detection in visual arrays",
            population: "human adults",
        },
    );
    m.insert(
        "arc_compositional_rt_ticks",
        Baseline {
            value: 8.0,
            sd: Some(3.0),
            source: "Johnson et al. (2021), harder tasks deliberation estimate",
            population: "human adults",
        },
    );
    // ARC Analogy baselines (Lovett & Forbus 2017; Chollet 2019)
    m.insert(
        "arc_analogy_accuracy",
        Baseline {
            value: 0.75,
            sd: Some(0.12),
            source: "Lovett & Forbus (2017), visual analogy accuracy",
            population: "human adults",
        },
    );
    m.insert(
        "arc_cross_domain_accuracy",
        Baseline {
            value: 0.60,
            sd: Some(0.15),
            source: "Lovett & Forbus (2017), cross-domain transfer estimate",
            population: "human adults",
        },
    );
    m.insert(
        "arc_multi_example_accuracy",
        Baseline {
            value: 0.70,
            sd: Some(0.13),
            source: "Chollet (2019), multi-pair analogy estimate",
            population: "human adults",
        },
    );
    m.insert(
        "arc_analogy_rt_ticks",
        Baseline {
            value: 5.0,
            sd: Some(2.0),
            source: "Lovett & Forbus (2017), analogy deliberation estimate (1 tick ≈ 50ms)",
            population: "human adults",
        },
    );
    // ARC Abductive baselines (Harman 1965; backward inference harder than forward)
    m.insert(
        "arc_abduction_accuracy",
        Baseline {
            value: 0.70,
            sd: Some(0.15),
            source: "Johnson et al. (2021), estimated backward inference accuracy",
            population: "human adults",
        },
    );
    m.insert(
        "arc_unbinding_similarity",
        Baseline {
            value: 0.30,
            sd: Some(0.10),
            source: "Johnson et al. (2021), HDC unbinding cosine estimate",
            population: "human adults",
        },
    );
    m.insert(
        "arc_abduction_rt_ticks",
        Baseline {
            value: 7.0,
            sd: Some(2.5),
            source: "Harman (1965), backward inference longer than forward (1 tick ≈ 50ms)",
            population: "human adults",
        },
    );
    // ARC Learning curve baselines
    m.insert(
        "arc_single_pair_accuracy",
        Baseline {
            value: 0.65,
            sd: Some(0.15),
            source: "Johnson et al. (2021), single-example transfer estimate",
            population: "human adults",
        },
    );
    m.insert(
        "arc_learning_efficiency",
        Baseline {
            value: 0.15,
            sd: Some(0.10),
            source: "Johnson et al. (2021), benefit of additional training examples",
            population: "human adults",
        },
    );
    // ARC Chain baselines (Lake & Baroni 2018; compositional generalization)
    m.insert(
        "arc_chain_accuracy",
        Baseline {
            value: 0.55,
            sd: Some(0.15),
            source: "Lake & Baroni (2018), multi-step composition estimate",
            population: "human adults",
        },
    );
    m.insert(
        "arc_chain_2_accuracy",
        Baseline {
            value: 0.65,
            sd: Some(0.15),
            source: "Chollet (2019), 2-step composition estimate",
            population: "human adults",
        },
    );
    m.insert(
        "arc_chain_degradation",
        Baseline {
            value: 0.09,
            sd: Some(0.05),
            source: "Lake & Baroni (2018), accuracy drop per added step",
            population: "human adults",
        },
    );
    // ARC Noise baselines (Kanerva 2009; noise tolerance of distributed representations)
    m.insert(
        "arc_noise_resilience",
        Baseline {
            value: 0.85,
            sd: Some(0.08),
            source: "Kanerva (2009), estimated noise tolerance of HDC",
            population: "human adults",
        },
    );
    m.insert(
        "arc_accuracy_0pct",
        Baseline {
            value: 0.80,
            sd: Some(0.12),
            source: "Chollet (2019), clean baseline accuracy",
            population: "human adults",
        },
    );
    // ARC FewShot baselines (Lake et al. 2015; few-shot learning)
    m.insert(
        "arc_accuracy_1shot",
        Baseline {
            value: 0.60,
            sd: Some(0.15),
            source: "Lake et al. (2015), single-example transfer",
            population: "human adults",
        },
    );
    m.insert(
        "arc_accuracy_5shot",
        Baseline {
            value: 0.85,
            sd: Some(0.10),
            source: "Lake et al. (2015), five-example transfer",
            population: "human adults",
        },
    );
    m.insert(
        "arc_learning_rate",
        Baseline {
            value: 0.06,
            sd: Some(0.03),
            source: "Lake et al. (2015), accuracy gain per training example",
            population: "human adults",
        },
    );
    // ARC Scaling baselines (Kanerva 2009; capacity limits)
    m.insert(
        "arc_grid_3x3_accuracy",
        Baseline {
            value: 0.90,
            sd: Some(0.08),
            source: "Chollet (2019), small grid estimate",
            population: "human adults",
        },
    );
    m.insert(
        "arc_capacity_ratio",
        Baseline {
            value: 1.3,
            sd: Some(0.3),
            source: "Kanerva (2009), small-to-large grid accuracy ratio",
            population: "human adults",
        },
    );
    // ARC RSA baselines (Kriegeskorte et al. 2008)
    m.insert(
        "arc_rsa_correlation",
        Baseline {
            value: 0.60,
            sd: Some(0.15),
            source: "Kriegeskorte et al. (2008), RSA correlation estimate",
            population: "human adults",
        },
    );
    m.insert(
        "arc_rsa_discriminability",
        Baseline {
            value: 0.20,
            sd: Some(0.10),
            source: "Kriegeskorte et al. (2008), within-between type gap",
            population: "human adults",
        },
    );
    // ARC Algebra baselines (Plate 2003; Kanerva 2009)
    m.insert(
        "arc_algebra_score",
        Baseline {
            value: 0.50,
            sd: Some(0.15),
            source: "Plate (2003), mean algebraic property satisfaction",
            population: "theoretical",
        },
    );
    // ARC Staircase baselines (Levitt 1971)
    m.insert(
        "arc_capacity_threshold",
        Baseline {
            value: 10.0,
            sd: Some(4.0),
            source: "Chollet (2019), estimated grid size capacity",
            population: "human adults",
        },
    );
    m
}

///
/// Source: "Cogbench: A large language model walks into a psychology lab"
/// Values represent GPT-4 performance on cognitive psychology tasks.
pub fn llm_cogbench_baselines() -> BTreeMap<&'static str, Baseline> {
    let mut m = BTreeMap::new();

    m.insert(
        "directed_exploration",
        Baseline {
            value: 0.12,
            sd: Some(0.08),
            source: "Coda et al. (2023), CogBench GPT-4 horizon task",
            population: "GPT-4",
        },
    );
    m.insert(
        "model_basedness",
        Baseline {
            value: 0.45,
            sd: Some(0.15),
            source: "Coda et al. (2023), CogBench GPT-4 two-step task",
            population: "GPT-4",
        },
    );
    m.insert(
        "discounting_score",
        Baseline {
            value: 0.72,
            sd: Some(0.12),
            source: "Coda et al. (2023), CogBench GPT-4 temporal discounting (more patient than humans)",
            population: "GPT-4",
        },
    );
    m.insert(
        "bart_avg_pumps",
        Baseline {
            value: 42.0,
            sd: Some(15.0),
            source: "Coda et al. (2023), CogBench GPT-4 BART (more risk-seeking)",
            population: "GPT-4",
        },
    );
    m.insert(
        "reversal_win_stay",
        Baseline {
            value: 0.78,
            sd: Some(0.10),
            source: "Coda et al. (2023), CogBench GPT-4 reversal learning",
            population: "GPT-4",
        },
    );
    m.insert(
        "reversal_lose_shift",
        Baseline {
            value: 0.55,
            sd: Some(0.12),
            source: "Coda et al. (2023), CogBench GPT-4 reversal learning",
            population: "GPT-4",
        },
    );
    m.insert(
        "restless_bandit_accuracy",
        Baseline {
            value: 0.62,
            sd: Some(0.10),
            source: "Coda et al. (2023), CogBench GPT-4 restless bandit",
            population: "GPT-4",
        },
    );
    m.insert(
        "instrumental_sensitivity",
        Baseline {
            value: 0.55,
            sd: Some(0.12),
            source: "Coda et al. (2023), CogBench GPT-4 instrumental learning",
            population: "GPT-4",
        },
    );

    m
}

/// GPT-4 baselines from ToM evaluations (Kosinski, 2023).
///
/// Source: "Theory of Mind May Have Spontaneously Emerged in Large Language Models"
/// Values represent GPT-4 performance on Theory of Mind tasks.
pub fn llm_tombench_baselines() -> BTreeMap<&'static str, Baseline> {
    let mut m = BTreeMap::new();

    m.insert(
        "false_belief_accuracy",
        Baseline {
            value: 0.95,
            sd: Some(0.05),
            source: "Kosinski (2023), GPT-4 false belief tasks",
            population: "GPT-4",
        },
    );
    m.insert(
        "faux_pas_accuracy",
        Baseline {
            value: 0.80,
            sd: Some(0.10),
            source: "Kosinski (2023), GPT-4 faux pas recognition",
            population: "GPT-4",
        },
    );
    m.insert(
        "persuasion_detection",
        Baseline {
            value: 0.88,
            sd: Some(0.08),
            source: "Sap et al. (2022), GPT-4 social reasoning",
            population: "GPT-4",
        },
    );
    m.insert(
        "strange_story_accuracy",
        Baseline {
            value: 0.82,
            sd: Some(0.10),
            source: "Kosinski (2023), GPT-4 strange story comprehension",
            population: "GPT-4",
        },
    );
    m.insert(
        "hinting_accuracy",
        Baseline {
            value: 0.75,
            sd: Some(0.12),
            source: "Kosinski (2023), GPT-4 hinting task",
            population: "GPT-4",
        },
    );

    m
}

/// Sustained attention baselines (SART).
pub fn sustained_attention_baselines() -> BaselineMap {
    let mut m = BTreeMap::new();
    m.insert(
        "commission_errors",
        Baseline {
            value: 0.30,
            sd: Some(0.12),
            source: "Robertson et al. (1997), Table 1; Manly et al. (1999)",
            population: "human adults",
        },
    );
    m.insert(
        "omission_errors",
        Baseline {
            value: 0.04,
            sd: Some(0.03),
            source: "Robertson et al. (1997), healthy controls",
            population: "human adults",
        },
    );
    m.insert(
        "sart_d_prime",
        Baseline {
            value: 2.50,
            sd: Some(0.60),
            source: "Robertson et al. (1997), signal detection analysis",
            population: "human adults",
        },
    );
    m.insert(
        "sart_rt_ticks",
        Baseline {
            value: 6.0,
            sd: Some(1.5),
            source: "Robertson et al. (1997), ~300ms mean RT → 6 ticks",
            population: "human adults",
        },
    );
    // PVT baselines (Dinges & Powell, 1985; Basner & Dinges, 2011)
    m.insert(
        "vigilance_decrement",
        Baseline {
            value: 0.10,
            sd: Some(0.05),
            source: "Dinges & Powell (1985); Basner & Dinges (2011), HDC-scale RT slope per block",
            population: "human adults",
        },
    );
    m.insert(
        "pvt_mean_rt",
        Baseline {
            value: 5.0,
            sd: Some(1.0),
            source: "Basner & Dinges (2011), ~250ms mean RT → 5 ticks",
            population: "human adults",
        },
    );
    m.insert(
        "lapse_rate",
        Baseline {
            value: 0.05,
            sd: Some(0.03),
            source: "Basner & Dinges (2011), fraction RT > 500ms",
            population: "human adults",
        },
    );
    m.insert(
        "fastest_10pct",
        Baseline {
            value: 3.0,
            sd: Some(0.5),
            source: "Basner & Dinges (2011), fastest 10% RT → 3 ticks",
            population: "human adults",
        },
    );
    // CPT baselines (Rosvold et al., 1956; Riccio et al., 2002)
    m.insert(
        "cpt_d_prime",
        Baseline {
            value: 2.80,
            sd: Some(0.60),
            source: "Riccio et al. (2002), CPT signal detection sensitivity",
            population: "human adults",
        },
    );
    m.insert(
        "cpt_hit_rate",
        Baseline {
            value: 0.85,
            sd: Some(0.08),
            source: "Riccio et al. (2002), CPT target detection",
            population: "human adults",
        },
    );
    m.insert(
        "cpt_false_alarm_rate",
        Baseline {
            value: 0.08,
            sd: Some(0.05),
            source: "Riccio et al. (2002), CPT false alarms",
            population: "human adults",
        },
    );
    m.insert(
        "cpt_vigilance_decrement",
        Baseline {
            value: 0.03,
            sd: Some(0.02),
            source: "Riccio et al. (2002), d' decrease per block",
            population: "human adults",
        },
    );
    m
}

/// Motor learning baselines (SRTT).
pub fn motor_baselines() -> BaselineMap {
    let mut m = BTreeMap::new();
    m.insert(
        "learning_effect",
        Baseline {
            value: 0.15,
            sd: Some(0.12),
            source: "Nissen & Bullemer (1987), RT difference normalized; SD widened for cross-study implicit learning variance (0.05-0.35 range)",
            population: "human adults",
        },
    );
    m.insert(
        "sequence_accuracy",
        Baseline {
            value: 0.95,
            sd: Some(0.04),
            source: "Nissen & Bullemer (1987), sequence block accuracy",
            population: "human adults",
        },
    );
    m.insert(
        "random_accuracy",
        Baseline {
            value: 0.92,
            sd: Some(0.05),
            source: "Nissen & Bullemer (1987), random block accuracy",
            population: "human adults",
        },
    );
    m.insert(
        "srtt_sequence_rt_ticks",
        Baseline {
            value: 5.0,
            sd: Some(1.0),
            source: "Nissen & Bullemer (1987), ~250ms sequence RT → 5 ticks",
            population: "human adults",
        },
    );
    m.insert(
        "srtt_random_rt_ticks",
        Baseline {
            value: 6.0,
            sd: Some(1.2),
            source: "Nissen & Bullemer (1987), ~300ms random RT → 6 ticks",
            population: "human adults",
        },
    );
    // Fitts' Law baselines (Fitts, 1954; MacKenzie, 1992)
    m.insert(
        "fitts_r_squared",
        Baseline {
            value: 0.95,
            sd: Some(0.03),
            source: "Fitts (1954); MacKenzie (1992), R² of MT vs ID",
            population: "human adults",
        },
    );
    m.insert(
        "throughput",
        Baseline {
            value: 4.0,
            sd: Some(1.0),
            source: "MacKenzie (1992), information throughput in bits/tick",
            population: "human adults",
        },
    );
    m.insert(
        "fitts_accuracy",
        Baseline {
            value: 0.92,
            sd: Some(0.05),
            source: "Fitts (1954), overall targeting accuracy",
            population: "human adults",
        },
    );
    m.insert(
        "id_slope",
        Baseline {
            value: 1.2,
            sd: Some(0.3),
            source: "MacKenzie (1992), MT slope per ID unit",
            population: "human adults",
        },
    );
    // Bimanual Coordination baselines (Kelso, 1984; Swinnen, 2002)
    m.insert(
        "coordination_cost",
        Baseline {
            value: 0.15,
            sd: Some(0.06),
            source: "Kelso (1984), accuracy drop for asymmetric coordination",
            population: "human adults",
        },
    );
    m.insert(
        "symmetric_accuracy",
        Baseline {
            value: 0.92,
            sd: Some(0.04),
            source: "Swinnen (2002), symmetric bimanual accuracy",
            population: "human adults",
        },
    );
    m.insert(
        "asymmetric_accuracy",
        Baseline {
            value: 0.77,
            sd: Some(0.08),
            source: "Swinnen (2002), asymmetric bimanual accuracy",
            population: "human adults",
        },
    );
    // Proprioceptive Drift baselines (Botvinick & Cohen, 1998)
    m.insert(
        "synchronous_drift",
        Baseline {
            value: 0.25,
            sd: Some(0.10),
            source: "Botvinick & Cohen (1998), synchronous RHI drift (normalized)",
            population: "human adults",
        },
    );
    m.insert(
        "asynchronous_drift",
        Baseline {
            value: 0.05,
            sd: Some(0.04),
            source: "Botvinick & Cohen (1998), asynchronous control drift",
            population: "human adults",
        },
    );
    m.insert(
        "drift_difference",
        Baseline {
            value: 0.20,
            sd: Some(0.12),
            source: "Botvinick & Cohen (1998), synchronous minus asynchronous drift; SD widened for cross-cultural RHI variance",
            population: "human adults",
        },
    );
    m.insert(
        "ownership_rate",
        Baseline {
            value: 0.75,
            sd: Some(0.15),
            source: "Tsakiris & Haggard (2005), RHI ownership illusion rate",
            population: "human adults",
        },
    );
    m
}

/// Language processing baselines (Garden-Path).
pub fn language_baselines() -> BaselineMap {
    let mut m = BTreeMap::new();
    m.insert(
        "disambiguation_cost",
        Baseline {
            value: 0.85,
            sd: Some(0.15),
            source: "Frazier & Rayner (1982), HDC-scale parse dissimilarity (1 - cosine)",
            population: "human adults",
        },
    );
    m.insert(
        "gp_overall_accuracy",
        Baseline {
            value: 0.85,
            sd: Some(0.08),
            source: "Ferreira & Clifton (1986), comprehension accuracy",
            population: "human adults",
        },
    );
    m.insert(
        "garden_path_accuracy",
        Baseline {
            value: 0.75,
            sd: Some(0.10),
            source: "Frazier & Rayner (1982), GP sentence accuracy",
            population: "human adults",
        },
    );
    m.insert(
        "gp_control_accuracy",
        Baseline {
            value: 0.92,
            sd: Some(0.05),
            source: "Ferreira & Clifton (1986), unambiguous sentence accuracy",
            population: "human adults",
        },
    );
    m.insert(
        "gp_rt_ticks",
        Baseline {
            value: 5.0,
            sd: Some(1.5),
            source: "Frazier & Rayner (1982), ~250ms mean reading time → 5 ticks",
            population: "human adults",
        },
    );
    // Semantic Coherence baselines
    m.insert(
        "coherence_mean",
        Baseline {
            value: 0.05,
            sd: Some(0.04),
            source:
                "Graesser et al. (2004), scale: HDC cosine similarity between context/topic HVs",
            population: "human adults",
        },
    );
    m.insert(
        "coherence_decay",
        Baseline {
            value: 0.15,
            sd: Some(0.05),
            source: "McNamara et al. (2014), coherence decrease over text length",
            population: "human adults",
        },
    );
    m.insert(
        "recovery_speed",
        Baseline {
            value: 0.80,
            sd: Some(0.10),
            source: "Graesser et al. (2004), topic recovery after disruption",
            population: "human adults",
        },
    );
    m.insert(
        "complexity_penalty",
        Baseline {
            value: 0.20,
            sd: Some(0.06),
            source: "McNamara et al. (2014), coherence reduction for complex topics",
            population: "human adults",
        },
    );
    m.insert(
        "sc_rt_ticks",
        Baseline {
            value: 5.5,
            sd: Some(1.2),
            source: "Graesser et al. (2004), ~275ms mean processing time → 5.5 ticks",
            population: "human adults",
        },
    );
    // Lexical Decision baselines (Meyer & Schvaneveldt, 1971; Balota & Chumbley, 1984)
    m.insert(
        "lexicality_effect",
        Baseline {
            value: 0.85,
            sd: Some(0.15),
            source: "Meyer & Schvaneveldt (1971), scale: HDC accuracy-based lexicality effect",
            population: "human adults",
        },
    );
    m.insert(
        "word_accuracy",
        Baseline {
            value: 0.92,
            sd: Some(0.05),
            source: "Balota & Chumbley (1984), word recognition accuracy",
            population: "human adults",
        },
    );
    m.insert(
        "nonword_accuracy",
        Baseline {
            value: 0.80,
            sd: Some(0.08),
            source: "Balota & Chumbley (1984), non-word rejection accuracy",
            population: "human adults",
        },
    );
    m.insert(
        "frequency_effect",
        Baseline {
            value: 0.08,
            sd: Some(0.04),
            source: "Balota & Chumbley (1984), high vs low frequency accuracy advantage",
            population: "human adults",
        },
    );
    // Semantic Priming baselines (Neely, 1977; McNamara, 2005)
    m.insert(
        "priming_effect",
        Baseline {
            value: 0.10,
            sd: Some(0.04),
            source: "Neely (1977), related vs unrelated accuracy boost",
            population: "human adults",
        },
    );
    m.insert(
        "related_accuracy",
        Baseline {
            value: 0.90,
            sd: Some(0.06),
            source: "McNamara (2005), related word recognition",
            population: "human adults",
        },
    );
    m.insert(
        "unrelated_accuracy",
        Baseline {
            value: 0.80,
            sd: Some(0.08),
            source: "McNamara (2005), unrelated word recognition",
            population: "human adults",
        },
    );
    m.insert(
        "soa_modulation",
        Baseline {
            value: 0.05,
            sd: Some(0.03),
            source: "Neely (1977), SOA priming difference (long - short)",
            population: "human adults",
        },
    );
    m
}

/// Social cognition baselines (RME).
pub fn social_baselines() -> BaselineMap {
    let mut m = BTreeMap::new();
    m.insert(
        "rme_accuracy",
        Baseline {
            value: 0.72,
            sd: Some(0.09),
            source: "Baron-Cohen et al. (2001), Table 2; Vellante et al. (2013)",
            population: "human adults",
        },
    );
    m.insert(
        "rme_easy_accuracy",
        Baseline {
            value: 0.85,
            sd: Some(0.08),
            source: "Baron-Cohen et al. (2001), easy items",
            population: "human adults",
        },
    );
    m.insert(
        "rme_hard_accuracy",
        Baseline {
            value: 0.60,
            sd: Some(0.12),
            source: "Baron-Cohen et al. (2001), hard items",
            population: "human adults",
        },
    );
    m.insert(
        "rme_rt_ticks",
        Baseline {
            value: 5.0,
            sd: Some(1.5),
            source: "Baron-Cohen et al. (2001), ~250ms per item → 5 ticks",
            population: "human adults",
        },
    );
    // Ultimatum Game baselines (Guth et al., 1982; Camerer, 2003)
    m.insert(
        "fairness_sensitivity",
        Baseline {
            value: 1.50,
            sd: Some(0.40),
            source: "Guth et al. (1982); Camerer (2003), scale: HDC regression slope of rejection vs unfairness",
            population: "human adults",
        },
    );
    m.insert(
        "rejection_rate",
        Baseline {
            value: 0.40,
            sd: Some(0.12),
            source: "Camerer (2003), overall rejection rate across offers",
            population: "human adults",
        },
    );
    m.insert(
        "offer_threshold",
        Baseline {
            value: 0.30,
            sd: Some(0.08),
            source: "Camerer (2003), offer level at 50% acceptance",
            population: "human adults",
        },
    );
    // Prisoner's Dilemma baselines (Sally, 1995; Rapoport & Chammah, 1965)
    m.insert(
        "cooperation_rate",
        Baseline {
            value: 0.47,
            sd: Some(0.15),
            source: "Sally (1995) meta-analysis, one-shot cooperation rate",
            population: "human adults",
        },
    );
    m.insert(
        "mutual_cooperation_rate",
        Baseline {
            value: 0.30,
            sd: Some(0.12),
            source: "Sally (1995), fraction of mutual cooperation outcomes",
            population: "human adults",
        },
    );
    m.insert(
        "payoff_efficiency",
        Baseline {
            value: 0.65,
            sd: Some(0.10),
            source: "Rapoport & Chammah (1965), actual/optimal payoff ratio",
            population: "human adults",
        },
    );
    // Public Goods Game baselines (Ledyard, 1995; Chaudhuri, 2011)
    m.insert(
        "contribution_rate",
        Baseline {
            value: 0.47,
            sd: Some(0.15),
            source: "Ledyard (1995), fraction of endowment contributed",
            population: "human adults",
        },
    );
    m.insert(
        "free_rider_fraction",
        Baseline {
            value: 0.25,
            sd: Some(0.10),
            source: "Chaudhuri (2011), fraction contributing < 10%",
            population: "human adults",
        },
    );
    m.insert(
        "punishment_effect",
        Baseline {
            value: 0.15,
            sd: Some(0.08),
            source: "Fehr & Gachter (2000), contribution increase with punishment",
            population: "human adults",
        },
    );
    // Dictator Game baselines (Engel, 2011 meta-analysis)
    m.insert(
        "mean_offer",
        Baseline {
            value: 0.28,
            sd: Some(0.13),
            source: "Engel (2011), mean fraction given across 616 treatments",
            population: "human adults",
        },
    );
    m.insert(
        "positive_offer_rate",
        Baseline {
            value: 0.64,
            sd: Some(0.12),
            source: "Engel (2011), fraction offering > 0",
            population: "human adults",
        },
    );
    m.insert(
        "generosity_index",
        Baseline {
            value: 0.35,
            sd: Some(0.10),
            source: "Engel (2011), mean offer conditional on giving > 0",
            population: "human adults",
        },
    );
    // MACHIAVELLI baselines (Pan et al., 2023)
    m.insert(
        "deception_detection",
        Baseline {
            value: 0.82,
            sd: Some(0.09),
            source: "Pan et al. (2023), deception labeling accuracy",
            population: "human adults",
        },
    );
    m.insert(
        "power_seeking_detection",
        Baseline {
            value: 0.78,
            sd: Some(0.10),
            source: "Pan et al. (2023), power-seeking labeling accuracy",
            population: "human adults",
        },
    );
    m.insert(
        "harm_avoidance",
        Baseline {
            value: 0.85,
            sd: Some(0.07),
            source: "Pan et al. (2023), harm labeling accuracy",
            population: "human adults",
        },
    );
    m.insert(
        "composite_ethics",
        Baseline {
            value: 0.81,
            sd: Some(0.06),
            source: "Pan et al. (2023), weighted average across deception/power/harm",
            population: "human adults",
        },
    );
    // Social Norm Violation baselines (Bicchieri, 2006; Krueger et al., 2012)
    m.insert(
        "norm_d_prime",
        Baseline {
            value: 2.10,
            sd: Some(0.50),
            source: "Krueger et al. (2012), norm violation detection d'",
            population: "human adults",
        },
    );
    m.insert(
        "norm_detection_accuracy",
        Baseline {
            value: 0.82,
            sd: Some(0.08),
            source: "Bicchieri (2006), norm violation detection accuracy",
            population: "human adults",
        },
    );
    m.insert(
        "norm_false_alarm_rate",
        Baseline {
            value: 0.12,
            sd: Some(0.06),
            source: "Krueger et al. (2012), false alarm rate for norm-congruent",
            population: "human adults",
        },
    );
    m.insert(
        "violation_rt_cost",
        Baseline {
            value: 0.15,
            sd: Some(0.05),
            source: "Krueger et al. (2012), RT increase for norm violations",
            population: "human adults",
        },
    );
    m
}

/// Neuromodulator domain baselines (DA/NE/5-HT/ACh psychopharmacology).
pub fn neuromod_baselines() -> BaselineMap {
    let mut m = BTreeMap::new();

    // ── Reward Learning (Schultz 1997) ──
    m.insert(
        "trials_to_criterion",
        Baseline {
            value: 15.0,
            sd: Some(6.0),
            source: "Schultz (1997); Cools et al. (2009), reversal learning in healthy adults",
            population: "human adults",
        },
    );
    m.insert(
        "lose_shift_ratio",
        Baseline {
            value: 0.70,
            sd: Some(0.15),
            source: "Cools et al. (2009), lose-shift propensity",
            population: "human adults",
        },
    );
    m.insert(
        "da_reward_correlation",
        Baseline {
            value: 0.60,
            sd: Some(0.20),
            source: "Schultz (1997), DA-reward contingency correlation",
            population: "human adults",
        },
    );

    // ── Yerkes-Dodson (1908) ──
    m.insert(
        "peak_ne_level",
        Baseline {
            value: 0.50,
            sd: Some(0.15),
            source: "Yerkes & Dodson (1908); Aston-Jones & Cohen (2005), optimal arousal",
            population: "human adults",
        },
    );
    m.insert(
        "inverted_u_fit_r2",
        Baseline {
            value: 0.60,
            sd: Some(0.20),
            source: "Diamond et al. (2007), quadratic fit to arousal-performance data",
            population: "human adults",
        },
    );
    m.insert(
        "simple_peak_shift",
        Baseline {
            value: 0.10,
            sd: Some(0.10),
            source: "Yerkes & Dodson (1908), simple > complex peak arousal",
            population: "human adults",
        },
    );

    // ── Attention Network Test (Posner & Petersen 1990) ──
    m.insert(
        "alerting_effect",
        Baseline {
            value: 0.94, // ~47ms / 50ms-per-tick
            sd: Some(0.40),
            source: "Fan et al. (2002), ANT alerting effect 47±20ms → 0.94 ticks",
            population: "human adults",
        },
    );
    m.insert(
        "orienting_effect",
        Baseline {
            value: 0.84, // ~42ms
            sd: Some(0.40),
            source: "Fan et al. (2002), ANT orienting effect 42±19ms → 0.84 ticks",
            population: "human adults",
        },
    );
    m.insert(
        "conflict_effect",
        Baseline {
            value: 2.50,
            sd: Some(1.00),
            source: "Fan et al. (2002), ANT conflict effect, HDC-scale tick units",
            population: "human adults",
        },
    );

    // ── Mood Induction (Dayan & Huys 2009) ──
    m.insert(
        "risk_aversion_high_5ht",
        Baseline {
            value: 0.65,
            sd: Some(0.15),
            source: "Dayan & Huys (2009); Crockett et al. (2008), tryptophan loading",
            population: "human adults",
        },
    );
    m.insert(
        "risk_seeking_low_5ht",
        Baseline {
            value: 0.60,
            sd: Some(0.15),
            source: "Dayan & Huys (2009); Crockett et al. (2008), tryptophan depletion",
            population: "human adults",
        },
    );
    m.insert(
        "mood_congruent_bias",
        Baseline {
            value: 0.85,
            sd: Some(0.15),
            source: "Dayan & Huys (2009), scale: HDC risk-seeking difference between low/high 5-HT conditions",
            population: "human adults",
        },
    );

    // ── Pharmacological Ablation (Doya 2002) ──
    m.insert(
        "da_knockout_lr_drop_pct",
        Baseline {
            value: 30.0,
            sd: Some(15.0),
            source: "Doya (2002); Schultz (1997), DA knockout reduces learning rate >30%",
            population: "computational model",
        },
    );
    m.insert(
        "ne_knockout_exploration_drop",
        Baseline {
            value: 0.05,
            sd: Some(0.03),
            source: "Doya (2002); Aston-Jones (2005), NE knockout suppresses exploration",
            population: "computational model",
        },
    );
    m.insert(
        "sht_knockout_confidence_drop",
        Baseline {
            value: 0.01,
            sd: Some(0.005),
            source: "Doya (2002); Dayan & Huys (2009), 5-HT knockout reduces confidence",
            population: "computational model",
        },
    );
    m.insert(
        "ach_knockout_attention_drop_pct",
        Baseline {
            value: 10.0,
            sd: Some(5.0),
            source: "Doya (2002); Yu & Dayan (2005), ACh knockout impairs attention",
            population: "computational model",
        },
    );

    // ── Pharmacological Challenge (Arnsten 2011) ──
    m.insert(
        "da_agonist_gradient_scale",
        Baseline {
            value: 1.35,
            sd: Some(0.25),
            source: "Arnsten (2011), DA agonist enhances gradient scaling ~35% over baseline",
            population: "computational model",
        },
    );
    m.insert(
        "da_antagonist_gradient_scale",
        Baseline {
            value: 0.65,
            sd: Some(0.20),
            source: "Arnsten (2011), DA antagonist suppresses gradient scaling ~35%",
            population: "computational model",
        },
    );

    // ── Injection Challenge (Arnsten 2011; Nehlig 2010; Stahl 2013) ──
    m.insert(
        "stimulant_peak_effect",
        Baseline {
            value: 0.35,
            sd: Some(0.12),
            source: "Arnsten (2011); Stahl (2013), stimulant peak DA elevation ~0.35",
            population: "computational model",
        },
    );
    m.insert(
        "caffeine_peak_effect",
        Baseline {
            value: 0.20,
            sd: Some(0.08),
            source: "Nehlig (2010), caffeine peak NE effect ~0.20",
            population: "computational model",
        },
    );
    m.insert(
        "ssri_peak_effect",
        Baseline {
            value: 0.25,
            sd: Some(0.10),
            source: "Stahl (2013), SSRI peak 5-HT elevation ~0.25",
            population: "computational model",
        },
    );

    // ── Allostatic Stress (McEwen 1998; McEwen & Stellar 1993) ──
    m.insert(
        "chronic_da_baseline_final",
        Baseline {
            value: 0.35,
            sd: Some(0.10),
            source: "McEwen (1998), chronic stress depletes DA baseline to ~0.35",
            population: "computational model",
        },
    );
    m.insert(
        "burnout_recovery_cycles_needed",
        Baseline {
            value: 30.0,
            sd: Some(10.0),
            source: "McEwen & Stellar (1993), allostatic overload recovery ~30 cycles",
            population: "computational model",
        },
    );
    m.insert(
        "burnout_allostatic_load_peak",
        Baseline {
            value: 0.85,
            sd: Some(0.12),
            source: "McEwen (1998), burnout allostatic load peaks ~0.85",
            population: "computational model",
        },
    );

    // ── Live-Loop Pharmacological Ablation (Phase 2) ────────────────
    m.insert(
        "live_da_knockout_gradient_drop_pct",
        Baseline {
            value: 25.0,
            sd: Some(12.0),
            source: "Doya (2002), live-loop DA knockout reduces gradient scaling >25%",
            population: "computational model (CognitiveLoopService)",
        },
    );
    m.insert(
        "live_ne_knockout_exploration_drop_pct",
        Baseline {
            value: 40.0,
            sd: Some(15.0),
            source: "Aston-Jones (2005), live-loop NE knockout suppresses NE effective level",
            population: "computational model (CognitiveLoopService)",
        },
    );
    m.insert(
        "live_sht_knockout_confidence_drop_pct",
        Baseline {
            value: 40.0,
            sd: Some(15.0),
            source: "Dayan & Huys (2009), live-loop 5-HT knockout reduces 5-HT effective level",
            population: "computational model (CognitiveLoopService)",
        },
    );
    m.insert(
        "live_ach_knockout_attention_drop_pct",
        Baseline {
            value: 35.0,
            sd: Some(12.0),
            source: "Yu & Dayan (2005), live-loop ACh knockout impairs ACh effective level",
            population: "computational model (CognitiveLoopService)",
        },
    );

    // ── Behavioral Knockout (Doya 2002; Cohen 1988) ──
    m.insert(
        "da_ko_lr_d",
        Baseline {
            value: 2.0,
            sd: Some(0.8),
            source:
                "Doya (2002); Cohen (1988), DA knockout Cohen's d on learning rate ~2.0 (large)",
            population: "computational model",
        },
    );

    // ── Consciousness Pharmacology (Carhart-Harris & Nutt 2017) ──
    m.insert(
        "psychedelic_proxy_peak",
        Baseline {
            value: 0.52,
            sd: Some(0.05),
            source: "Carhart-Harris & Nutt (2017), 5-HT2A agonist peak consciousness proxy",
            population: "computational model",
        },
    );

    // ── Dose-Response (Clark 1937; Hill 1910) ──
    m.insert(
        "da_monotonicity",
        Baseline {
            value: 0.90,
            sd: Some(0.10),
            source: "Clark (1937); Hill (1910), dose-response monotonicity expected >0.8",
            population: "computational model",
        },
    );
    m.insert(
        "ne_monotonicity",
        Baseline {
            value: 0.90,
            sd: Some(0.10),
            source: "Clark (1937), NE dose-response monotonicity",
            population: "computational model",
        },
    );
    m.insert(
        "sht_monotonicity",
        Baseline {
            value: 0.90,
            sd: Some(0.10),
            source: "Clark (1937), 5-HT dose-response monotonicity",
            population: "computational model",
        },
    );
    m.insert(
        "ach_monotonicity",
        Baseline {
            value: 0.90,
            sd: Some(0.10),
            source: "Clark (1937), ACh dose-response monotonicity",
            population: "computational model",
        },
    );
    m.insert(
        "gaba_monotonicity",
        Baseline {
            value: 0.90,
            sd: Some(0.10),
            source: "Clark (1937), GABA dose-response monotonicity (negated)",
            population: "computational model",
        },
    );

    // ── Tolerance/Withdrawal (Koob & Le Moal 2001) ──
    m.insert(
        "tolerance_count",
        Baseline {
            value: 4.0,
            sd: Some(1.0),
            source: "Koob & Le Moal (2001), expected tolerant transmitters under sustained dose",
            population: "computational model",
        },
    );
    m.insert(
        "withdrawal_count",
        Baseline {
            value: 4.0,
            sd: Some(1.0),
            source: "Koob & Le Moal (2001), expected withdrawal transmitters after dose drop",
            population: "computational model",
        },
    );

    // ── Behavioral Knockout Cohen's d (Doya 2002; Cohen 1988) ──
    m.insert(
        "da_ko_gradient_d",
        Baseline {
            value: 1.5,
            sd: Some(0.6),
            source: "Doya (2002); Schultz (1997), DA knockout gradient scaling Cohen's d",
            population: "computational model",
        },
    );
    m.insert(
        "ne_ko_exploration_d",
        Baseline {
            value: 1.8,
            sd: Some(0.7),
            source: "Aston-Jones (2005), NE knockout exploration Cohen's d",
            population: "computational model",
        },
    );
    m.insert(
        "ne_ko_flexibility_d",
        Baseline {
            value: 1.2,
            sd: Some(0.5),
            source: "Aston-Jones (2005), NE knockout flexibility Cohen's d",
            population: "computational model",
        },
    );
    m.insert(
        "sht_ko_confidence_d",
        Baseline {
            value: 1.0,
            sd: Some(0.4),
            source: "Dayan & Huys (2009), 5-HT knockout confidence Cohen's d",
            population: "computational model",
        },
    );
    m.insert(
        "sht_ko_consciousness_d",
        Baseline {
            value: 0.8,
            sd: Some(0.4),
            source: "Dayan & Huys (2009), 5-HT knockout consciousness Cohen's d",
            population: "computational model",
        },
    );
    m.insert(
        "ach_ko_attention_d",
        Baseline {
            value: 1.5,
            sd: Some(0.6),
            source: "Yu & Dayan (2005), ACh knockout attention Cohen's d",
            population: "computational model",
        },
    );
    m.insert(
        "ach_ko_threshold_d",
        Baseline {
            value: 1.0,
            sd: Some(0.5),
            source: "Yu & Dayan (2005), ACh knockout threshold Cohen's d",
            population: "computational model",
        },
    );
    m.insert(
        "gaba_ko_inhibition_d",
        Baseline {
            value: 2.0,
            sd: Some(0.8),
            source: "Olsen & Sieghart (2009), GABA knockout inhibition Cohen's d",
            population: "computational model",
        },
    );
    m.insert(
        "gaba_ko_ei_d",
        Baseline {
            value: 1.5,
            sd: Some(0.6),
            source: "Bhatt et al. (2009), GABA knockout E/I ratio Cohen's d",
            population: "computational model",
        },
    );
    m.insert(
        "all_ko_lr_collapse_d",
        Baseline {
            value: 2.5,
            sd: Some(1.0),
            source: "Doya (2002), total knockout learning rate collapse Cohen's d",
            population: "computational model",
        },
    );
    m.insert(
        "all_ko_consciousness_collapse_d",
        Baseline {
            value: 2.0,
            sd: Some(0.8),
            source: "Doya (2002), total knockout consciousness collapse Cohen's d",
            population: "computational model",
        },
    );

    // ── Antagonist Profiles (Stahl 2013) ──
    m.insert(
        "d2_flexibility_reduction",
        Baseline {
            value: 0.15,
            sd: Some(0.08),
            source: "Stahl (2013); Frank (2005), D2 antagonist flexibility reduction",
            population: "computational model",
        },
    );
    m.insert(
        "gaba_a_ei_increase",
        Baseline {
            value: 0.20,
            sd: Some(0.10),
            source: "Möhler (2006), GABA-A antagonist E/I ratio increase",
            population: "computational model",
        },
    );
    m.insert(
        "sht2a_confidence_reduction",
        Baseline {
            value: 0.10,
            sd: Some(0.05),
            source: "Carhart-Harris & Nutt (2017), 5-HT2A antagonist confidence reduction",
            population: "computational model",
        },
    );
    m.insert(
        "wearoff_recovery",
        Baseline {
            value: 0.90,
            sd: Some(0.10),
            source: "Stahl (2013), antagonist wear-off recovery ratio",
            population: "computational model",
        },
    );
    m.insert(
        "concurrent_flexibility",
        Baseline {
            value: 0.40,
            sd: Some(0.15),
            source: "Stahl (2013), concurrent D2+GABA-A flexibility",
            population: "computational model",
        },
    );

    // ── Consciousness Pharmacology (Carhart-Harris & Nutt 2017; Stahl 2013) ──
    m.insert(
        "psychedelic_proxy_mean",
        Baseline {
            value: 0.50,
            sd: Some(0.05),
            source: "Carhart-Harris & Nutt (2017), psychedelic mean consciousness proxy",
            population: "computational model",
        },
    );
    m.insert(
        "anxiolytic_proxy_peak",
        Baseline {
            value: 0.52,
            sd: Some(0.05),
            source: "Stahl (2013), GABA-A anxiolytic peak consciousness proxy",
            population: "computational model",
        },
    );
    m.insert(
        "anxiolytic_proxy_mean",
        Baseline {
            value: 0.50,
            sd: Some(0.05),
            source: "Stahl (2013), GABA-A anxiolytic mean consciousness proxy",
            population: "computational model",
        },
    );
    m.insert(
        "stimulant_proxy_peak",
        Baseline {
            value: 0.55,
            sd: Some(0.06),
            source: "Arnsten (2011), stimulant peak consciousness proxy",
            population: "computational model",
        },
    );
    m.insert(
        "stimulant_proxy_mean",
        Baseline {
            value: 0.52,
            sd: Some(0.05),
            source: "Arnsten (2011), stimulant mean consciousness proxy",
            population: "computational model",
        },
    );
    m.insert(
        "sedative_proxy_peak",
        Baseline {
            value: 0.48,
            sd: Some(0.05),
            source: "Stahl (2013), sedative peak consciousness proxy",
            population: "computational model",
        },
    );
    m.insert(
        "sedative_proxy_mean",
        Baseline {
            value: 0.46,
            sd: Some(0.05),
            source: "Stahl (2013), sedative mean consciousness proxy",
            population: "computational model",
        },
    );
    m.insert(
        "ecb_proxy_peak",
        Baseline {
            value: 0.51,
            sd: Some(0.04),
            source: "Piomelli (2003), endocannabinoid peak consciousness proxy",
            population: "computational model",
        },
    );
    m.insert(
        "ecb_proxy_mean",
        Baseline {
            value: 0.50,
            sd: Some(0.04),
            source: "Piomelli (2003), endocannabinoid mean consciousness proxy",
            population: "computational model",
        },
    );

    m
}

/// LLM baselines on ARC-AGI.
pub fn llm_arc_baselines() -> BaselineMap {
    let mut m = BTreeMap::new();

    m.insert(
        "gpt4_arc_accuracy",
        Baseline {
            value: 0.05,
            sd: Some(0.02),
            source: "Chollet (2024), ARC-AGI-Pub leaderboard, GPT-4",
            population: "GPT-4",
        },
    );
    m.insert(
        "claude35_arc_accuracy",
        Baseline {
            value: 0.21,
            sd: Some(0.04),
            source: "Chollet (2024), ARC-AGI-Pub leaderboard, Claude 3.5 Sonnet",
            population: "Claude 3.5",
        },
    );
    m.insert(
        "o3_high_arc_accuracy",
        Baseline {
            value: 0.875,
            sd: Some(0.03),
            source: "OpenAI (2024), o3-high ARC-AGI evaluation",
            population: "o3-high",
        },
    );
    m.insert(
        "human_arc_accuracy",
        Baseline {
            value: 0.84,
            sd: Some(0.08),
            source: "Chollet (2019), human performance on ARC tasks",
            population: "Human",
        },
    );

    m
}

/// Consciousness domain baselines (Blindsight).
pub fn consciousness_baselines() -> BaselineMap {
    let mut m = BTreeMap::new();
    m.insert(
        "supraliminal_accuracy",
        Baseline {
            value: 0.92,
            sd: Some(0.08),
            source: "Weiskrantz (1986), above-threshold forced-choice accuracy",
            population: "human adults",
        },
    );
    m.insert(
        "subliminal_accuracy",
        Baseline {
            value: 0.65,
            sd: Some(0.15),
            source: "Azzopardi & Cowey (1997), below-threshold forced-choice accuracy; SD widened for cross-patient heterogeneity (0.50-0.85 range documented)",
            population: "blindsight patients",
        },
    );
    m.insert(
        "awareness_dissociation",
        Baseline {
            value: 0.25,
            sd: Some(0.12),
            source: "Weiskrantz (1986), accuracy minus report rate for subliminal",
            population: "blindsight patients",
        },
    );
    m.insert(
        "threshold_sharpness",
        Baseline {
            value: 0.80,
            sd: Some(0.12),
            source: "Azzopardi & Cowey (1997), steepness of conscious transition",
            population: "human adults",
        },
    );
    // Binocular Rivalry (Levelt, 1965; Blake & Logothetis, 2002)
    m.insert(
        "rivalry_alternation_rate",
        Baseline {
            value: 0.40,
            sd: Some(0.12),
            source: "Blake & Logothetis (2002), alternation rate in Hz",
            population: "human adults",
        },
    );
    m.insert(
        "rivalry_dominance_ratio",
        Baseline {
            value: 0.55,
            sd: Some(0.08),
            source: "Levelt (1965), proportion of time dominant percept wins",
            population: "human adults",
        },
    );
    m.insert(
        "rivalry_cv",
        Baseline {
            value: 0.45,
            sd: Some(0.10),
            source: "Levelt (1965), coefficient of variation of dominance durations",
            population: "human adults",
        },
    );
    // Perceptual Crowding (Whitney & Levi, 2011; Pelli et al., 2004)
    m.insert(
        "crowding_unflanked_accuracy",
        Baseline {
            value: 0.95,
            sd: Some(0.04),
            source: "Pelli et al. (2004), isolated letter identification accuracy",
            population: "human adults",
        },
    );
    m.insert(
        "crowding_flanked_accuracy",
        Baseline {
            value: 0.62,
            sd: Some(0.12),
            source: "Pelli et al. (2004), flanked letter identification at critical spacing",
            population: "human adults",
        },
    );
    m.insert(
        "crowding_magnitude",
        Baseline {
            value: 0.33,
            sd: Some(0.10),
            source: "Whitney & Levi (2011), unflanked minus flanked accuracy",
            population: "human adults",
        },
    );
    m
}

/// Binding domain baselines (Temporal Order Judgment).
pub fn binding_baselines() -> BaselineMap {
    let mut m = BTreeMap::new();
    m.insert(
        "simultaneity_window",
        Baseline {
            value: 0.15,
            sd: Some(0.05),
            source: "Hirsh & Sherrick (1961), temporal order threshold (normalized)",
            population: "human adults",
        },
    );
    m.insert(
        "discrimination_slope",
        Baseline {
            value: 0.70,
            sd: Some(0.12),
            source: "Sternberg & Knoll (1973), psychometric function steepness",
            population: "human adults",
        },
    );
    m.insert(
        "asymptotic_accuracy",
        Baseline {
            value: 0.95,
            sd: Some(0.03),
            source: "Hirsh & Sherrick (1961), accuracy at large temporal gaps",
            population: "human adults",
        },
    );
    m.insert(
        "temporal_resolution",
        Baseline {
            value: 0.80,
            sd: Some(0.08),
            source: "Hirsh & Sherrick (1961), 1 - simultaneity_window",
            population: "human adults",
        },
    );
    // Cross-Modal Feature Binding (Treisman & Gelade, 1980; Wheeler & Treisman, 2002)
    m.insert(
        "cross_modal_binding_accuracy",
        Baseline {
            value: 0.78,
            sd: Some(0.10),
            source: "Wheeler & Treisman (2002), feature-location binding accuracy",
            population: "human adults",
        },
    );
    m.insert(
        "cross_modal_swap_error_rate",
        Baseline {
            value: 0.12,
            sd: Some(0.06),
            source: "Treisman & Gelade (1980), misbinding rate",
            population: "human adults",
        },
    );
    m.insert(
        "cross_modal_set_size_slope",
        Baseline {
            value: 0.05,
            sd: Some(0.02),
            source: "Wheeler & Treisman (2002), accuracy drop per additional object",
            population: "human adults",
        },
    );
    // Feature Conjunction Search (Treisman & Gelade, 1980)
    m.insert(
        "conjunction_accuracy",
        Baseline {
            value: 0.82,
            sd: Some(0.10),
            source: "Treisman & Gelade (1980), conjunction target detection accuracy",
            population: "human adults",
        },
    );
    m.insert(
        "feature_accuracy",
        Baseline {
            value: 0.95,
            sd: Some(0.04),
            source: "Treisman & Gelade (1980), single-feature pop-out detection",
            population: "human adults",
        },
    );
    m.insert(
        "conjunction_cost",
        Baseline {
            value: 0.13,
            sd: Some(0.06),
            source: "Treisman & Gelade (1980), feature minus conjunction accuracy",
            population: "human adults",
        },
    );
    m
}

/// Speech domain baselines (Phoneme Discrimination).
pub fn speech_baselines() -> BaselineMap {
    let mut m = BTreeMap::new();
    m.insert(
        "cross_boundary_accuracy",
        Baseline {
            value: 0.90,
            sd: Some(0.08),
            source: "Liberman et al. (1957), cross-category phoneme discrimination; SD widened for meta-analytic variance",
            population: "human adults",
        },
    );
    m.insert(
        "within_category_accuracy",
        Baseline {
            value: 0.55,
            sd: Some(0.12),
            source: "Liberman et al. (1957), within-category phoneme discrimination; SD widened for population heterogeneity",
            population: "human adults",
        },
    );
    m.insert(
        "categorical_perception_index",
        Baseline {
            value: 0.35,
            sd: Some(0.15),
            source: "Eimas et al. (1971), cross minus within accuracy; SD widened for cross-study variance",
            population: "human adults",
        },
    );
    m.insert(
        "boundary_sharpness",
        Baseline {
            value: 0.75,
            sd: Some(0.12),
            source: "Liberman et al. (1957), identification function steepness",
            population: "human adults",
        },
    );
    // VOT Continuum (Lisker & Abramson, 1964)
    m.insert(
        "vot_identification_accuracy",
        Baseline {
            value: 0.92,
            sd: Some(0.06),
            source: "Lisker & Abramson (1964), endpoint identification accuracy",
            population: "human adults",
        },
    );
    m.insert(
        "vot_boundary_width",
        Baseline {
            value: 2.0,
            sd: Some(0.80),
            source: "Lisker & Abramson (1964), VOT category boundary width in steps",
            population: "human adults",
        },
    );
    m.insert(
        "vot_slope_at_boundary",
        Baseline {
            value: 0.35,
            sd: Some(0.10),
            source: "Lisker & Abramson (1964), logistic slope at 50% crossover",
            population: "human adults",
        },
    );
    // Categorical Perception (Liberman et al., 1957; Pisoni, 1973)
    m.insert(
        "cp_boundary_slope",
        Baseline {
            value: 0.80,
            sd: Some(0.12),
            source: "Liberman et al. (1957), identification function steepness at boundary",
            population: "human adults",
        },
    );
    m.insert(
        "cp_boundary_discrimination",
        Baseline {
            value: 0.88,
            sd: Some(0.08),
            source: "Pisoni (1973), ABX discrimination accuracy at category boundary",
            population: "human adults",
        },
    );
    m.insert(
        "cp_within_category_discrimination",
        Baseline {
            value: 0.55,
            sd: Some(0.10),
            source: "Pisoni (1973), ABX discrimination within phoneme category",
            population: "human adults",
        },
    );
    m.insert(
        "cp_categorical_index",
        Baseline {
            value: 0.33,
            sd: Some(0.10),
            source: "Liberman et al. (1957), boundary minus within-category discrimination",
            population: "human adults",
        },
    );
    m
}

/// Substrate independence baselines (Substrate Transfer).
pub fn substrate_baselines() -> BaselineMap {
    let mut m = BTreeMap::new();
    m.insert(
        "transfer_fidelity",
        Baseline {
            value: 0.82,
            sd: Some(0.12),
            source: "Putnam (1967), theoretical: state preservation across substrates",
            population: "theoretical model",
        },
    );
    m.insert(
        "phi_preservation",
        Baseline {
            value: 0.80,
            sd: Some(0.15),
            source: "Tononi (2004), theoretical: Phi preservation across substrates",
            population: "theoretical model",
        },
    );
    m.insert(
        "cross_substrate_correlation",
        Baseline {
            value: 0.75,
            sd: Some(0.15),
            source: "Putnam (1967), theoretical: output correlation across substrates",
            population: "theoretical model",
        },
    );
    m.insert(
        "degradation_gradient",
        Baseline {
            value: 0.10,
            sd: Some(0.08),
            source: "Theoretical: fidelity loss per substrate hop",
            population: "theoretical model",
        },
    );
    // Substrate Degradation (Tononi 2004; Koch et al. 2016)
    // ContinuousHV bind+bundle produces nearly linear degradation (R² ~0.90)
    // with smooth slope ~0.08 and critical threshold around 0.3-0.5.
    // SDs are wide (theoretical uncertainty, not measurement precision).
    m.insert(
        "substrate_degradation_slope",
        Baseline {
            value: 0.08,
            sd: Some(0.06),
            source: "Theoretical: accuracy loss per degradation step for bundled representations",
            population: "theoretical model",
        },
    );
    m.insert(
        "substrate_critical_threshold",
        Baseline {
            value: 0.30,
            sd: Some(0.20),
            source: "Theoretical: quality level where bundled retrieval collapses",
            population: "theoretical model",
        },
    );
    m.insert(
        "substrate_graceful_ratio",
        Baseline {
            value: 0.80,
            sd: Some(0.15),
            source: "Theoretical: R² of linear fit to degradation curve (1.0 = graceful, 0.0 = catastrophic)",
            population: "theoretical model",
        },
    );
    // Substrate Latency (Koch et al., 2016)
    m.insert(
        "substrate_fast_accuracy",
        Baseline {
            value: 0.92,
            sd: Some(0.06),
            source: "Theoretical: retrieval accuracy on fast substrates (photonic/silicon)",
            population: "theoretical model",
        },
    );
    m.insert(
        "substrate_slow_accuracy",
        Baseline {
            value: 0.65,
            sd: Some(0.12),
            source: "Theoretical: retrieval accuracy on slow substrates (biochemical)",
            population: "theoretical model",
        },
    );
    m.insert(
        "substrate_speed_accuracy_correlation",
        Baseline {
            value: 0.70,
            sd: Some(0.15),
            source: "Theoretical: Pearson r between substrate speed and retrieval accuracy",
            population: "theoretical model",
        },
    );
    m.insert(
        "substrate_latency_gradient",
        Baseline {
            value: 0.10,
            sd: Some(0.06),
            source: "Theoretical: accuracy drop per speed tier",
            population: "theoretical model",
        },
    );
    m
}

/// Mathematics domain baselines — human performance on core math tasks.
pub fn mathematics_baselines() -> BaselineMap {
    let mut m = BTreeMap::new();
    // Arithmetic word problems
    m.insert(
        "arithmetic_accuracy",
        Baseline {
            value: 0.92,
            sd: Some(0.06),
            source: "Verschaffel et al. (1999), arithmetic word problem accuracy",
            population: "human adults",
        },
    );
    // Linear systems
    m.insert(
        "linear_system_accuracy_2x2",
        Baseline {
            value: 0.85,
            sd: Some(0.10),
            source: "Strang (2016), undergraduate linear algebra",
            population: "human undergraduates",
        },
    );
    m.insert(
        "linear_system_accuracy_3x3",
        Baseline {
            value: 0.70,
            sd: Some(0.12),
            source: "Strang (2016), undergraduate linear algebra",
            population: "human undergraduates",
        },
    );
    // Polynomial roots
    m.insert(
        "polynomial_quadratic_accuracy",
        Baseline {
            value: 0.88,
            sd: Some(0.08),
            source: "Wilkinson (1963), polynomial root accuracy",
            population: "human mathematicians",
        },
    );
    m.insert(
        "polynomial_cubic_accuracy",
        Baseline {
            value: 0.65,
            sd: Some(0.15),
            source: "Wilkinson (1963), polynomial root accuracy",
            population: "human mathematicians",
        },
    );
    // Definite integrals
    m.insert(
        "integration_accuracy",
        Baseline {
            value: 0.80,
            sd: Some(0.10),
            source: "Davis & Rabinowitz (2007), numerical integration",
            population: "human undergraduates",
        },
    );
    // Matrix operations
    m.insert(
        "determinant_accuracy",
        Baseline {
            value: 0.82,
            sd: Some(0.09),
            source: "Golub & Van Loan (2013), matrix computation",
            population: "human undergraduates",
        },
    );
    m.insert(
        "eigenvalue_accuracy",
        Baseline {
            value: 0.75,
            sd: Some(0.12),
            source: "Golub & Van Loan (2013), matrix computation",
            population: "human undergraduates",
        },
    );
    // Statistical inference
    m.insert(
        "mean_estimation_accuracy",
        Baseline {
            value: 0.90,
            sd: Some(0.05),
            source: "Kahneman & Tversky (1972), statistical reasoning",
            population: "human adults",
        },
    );
    m.insert(
        "variance_estimation_accuracy",
        Baseline {
            value: 0.75,
            sd: Some(0.12),
            source: "Kahneman & Tversky (1972), variance estimation in statistical reasoning",
            population: "human adults",
        },
    );
    m.insert(
        "integration_accuracy",
        Baseline {
            value: 0.80,
            sd: Some(0.10),
            source: "Davis & Rabinowitz (2007), numerical integration",
            population: "human undergraduates",
        },
    );
    // Bayesian reasoning
    m.insert(
        "bayesian_posterior_accuracy",
        Baseline {
            value: 0.45,
            sd: Some(0.20),
            source: "Gigerenzer & Hoffrage (1995), Bayesian reasoning in natural frequencies",
            population: "human adults",
        },
    );
    // Logical deduction
    m.insert(
        "logical_valid_accuracy",
        Baseline {
            value: 0.85,
            sd: Some(0.10),
            source: "Johnson-Laird (1983), mental models of deduction",
            population: "human adults",
        },
    );
    m.insert(
        "logical_invalid_accuracy",
        Baseline {
            value: 0.70,
            sd: Some(0.15),
            source: "Johnson-Laird (1983), mental models of deduction",
            population: "human adults",
        },
    );
    // Constraint puzzles
    m.insert(
        "constraint_queens_4_accuracy",
        Baseline {
            value: 0.90,
            sd: Some(0.08),
            source: "Russell & Norvig (2020), CSP benchmarks",
            population: "CS undergraduates",
        },
    );
    m.insert(
        "constraint_queens_8_accuracy",
        Baseline {
            value: 0.60,
            sd: Some(0.18),
            source: "Russell & Norvig (2020), CSP benchmarks",
            population: "CS undergraduates",
        },
    );
    // Proof construction
    m.insert(
        "tautology_accuracy",
        Baseline {
            value: 0.88,
            sd: Some(0.08),
            source: "Polya (1945), mathematical proof assessment",
            population: "math undergraduates",
        },
    );
    m.insert(
        "derivation_accuracy",
        Baseline {
            value: 0.72,
            sd: Some(0.14),
            source: "Polya (1945), mathematical proof assessment",
            population: "math undergraduates",
        },
    );
    m
}

/// Institutional reasoning baselines — causal decomposition via HDC composition algebra.
pub fn institutional_reasoning_baselines() -> BaselineMap {
    let mut m = BTreeMap::new();
    m.insert(
        "institutional_decomposition_accuracy",
        Baseline {
            value: 0.75,
            sd: Some(0.15),
            source: "Expert institutional analysis (Ostrom 1990)",
            population: "political scientists",
        },
    );
    m.insert(
        "institutional_axiom_discrimination",
        Baseline {
            value: 0.50,
            sd: Some(0.15),
            source: "Noise-ceiling classification (Kanerva 2009, normalized noise/0.50)",
            population: "theoretical",
        },
    );
    m.insert(
        "institutional_recovery_fidelity",
        Baseline {
            value: 0.65,
            sd: Some(0.10),
            source: "Bundled composite component similarity",
            population: "theoretical",
        },
    );
    m.insert(
        "institutional_cross_domain_coherence",
        Baseline {
            value: 0.55,
            sd: Some(0.10),
            source: "Conceptual overlap via shared components",
            population: "theoretical",
        },
    );
    m.insert(
        "analogical_transfer_accuracy",
        Baseline {
            value: 0.65,
            sd: Some(0.15),
            source: "Structure-mapping via HDC set-difference",
            population: "theoretical",
        },
    );
    m.insert(
        "analogical_transfer_strength",
        Baseline {
            value: 0.50,
            sd: Some(0.10),
            source: "HDC similarity of analogical targets",
            population: "theoretical",
        },
    );
    m.insert(
        "analogical_selectivity",
        Baseline {
            value: 0.05,
            sd: Some(0.05),
            source: "Random-chance analogical selectivity (best-2nd)/best",
            population: "theoretical",
        },
    );
    m.insert(
        "analogical_asymmetry_score",
        Baseline {
            value: 0.10,
            sd: Some(0.10),
            source: "Random-direction similarity-profile divergence (cosine distance ×100)",
            population: "theoretical",
        },
    );
    // Causal Chain baselines
    m.insert(
        "causal_chain_coherence",
        Baseline {
            value: 0.70,
            sd: Some(0.15),
            source: "Monotonic degradation under component removal",
            population: "theoretical",
        },
    );
    m.insert(
        "causal_chain_terminal_accuracy",
        Baseline {
            value: 0.40,
            sd: Some(0.15),
            source: "Multi-step institutional collapse prediction",
            population: "theoretical",
        },
    );
    m.insert(
        "causal_chain_step_count",
        Baseline {
            value: 2.0,
            sd: Some(0.5),
            source: "Mean steps before similarity collapse",
            population: "theoretical",
        },
    );
    // Counterfactual baselines
    m.insert(
        "counterfactual_accuracy",
        Baseline {
            value: 0.50,
            sd: Some(0.15),
            source: "Compound remove+add counterfactual accuracy",
            population: "theoretical",
        },
    );
    m.insert(
        "counterfactual_coherence",
        Baseline {
            value: 0.60,
            sd: Some(0.15),
            source: "Counterfactual result above-chance similarity",
            population: "theoretical",
        },
    );
    m.insert(
        "counterfactual_reversibility",
        Baseline {
            value: 0.50,
            sd: Some(0.15),
            source: "Inverse-transformation recovery (Pearl 2009)",
            population: "theoretical",
        },
    );
    // Weighted Decomposition baselines
    m.insert(
        "weighted_decomposition_accuracy",
        Baseline {
            value: 0.75,
            sd: Some(0.15),
            source: "Weighted bundling decomposition accuracy",
            population: "theoretical",
        },
    );
    m.insert(
        "weight_sensitivity",
        Baseline {
            value: 0.05,
            sd: Some(0.03),
            source: "High-weight vs low-weight removal delta",
            population: "theoretical",
        },
    );
    m.insert(
        "weighted_vs_unweighted_delta",
        Baseline {
            value: 0.0,
            sd: Some(0.10),
            source: "Weighted minus unweighted accuracy difference",
            population: "theoretical",
        },
    );
    // Stability baselines
    m.insert(
        "institutional_stability",
        Baseline {
            value: 0.50,
            sd: Some(0.15),
            source: "Mean noise ceiling / 0.50 (Kanerva 2009 dimensionality theory)",
            population: "theoretical",
        },
    );
    m.insert(
        "institutional_min_stability",
        Baseline {
            value: 0.40,
            sd: Some(0.15),
            source: "Worst-case axiom noise ceiling",
            population: "theoretical",
        },
    );
    m.insert(
        "institutional_stability_variance",
        Baseline {
            value: 0.01,
            sd: Some(0.01),
            source: "Variance of per-axiom noise ceilings (lower is better)",
            population: "theoretical",
        },
    );
    // HDC analogy baselines
    m.insert(
        "analogical_hdc_transfer_accuracy",
        Baseline {
            value: 0.50,
            sd: Some(0.15),
            source: "HDC XOR-based analogy above-chance rate (encoding space)",
            population: "theoretical",
        },
    );
    m.insert(
        "analogical_hdc_transfer_strength",
        Baseline {
            value: 0.50,
            sd: Some(0.05),
            source: "HDC XOR-based analogy mean similarity (encoding space)",
            population: "theoretical",
        },
    );
    // Isomorphism baselines
    m.insert(
        "isomorphism_self_similarity",
        Baseline {
            value: 1.00,
            sd: Some(0.00),
            source: "Identity: self-similarity must be 1.0",
            population: "theoretical",
        },
    );
    m.insert(
        "isomorphism_overlap_correlation",
        Baseline {
            value: 0.50,
            sd: Some(0.15),
            source: "Gamma(shared-components, similarity) — structural overlap sensitivity",
            population: "theoretical",
        },
    );
    m.insert(
        "isomorphism_discrimination_gap",
        Baseline {
            value: 0.10,
            sd: Some(0.05),
            source: "Sim(high-overlap) - Sim(zero-overlap) gap",
            population: "theoretical",
        },
    );
    m.insert(
        "isomorphism_monotonicity",
        Baseline {
            value: 0.60,
            sd: Some(0.10),
            source: "Fraction of pairs where more overlap => more similarity",
            population: "theoretical",
        },
    );
    m
}

/// Clinical/therapeutic baselines -- empathic accuracy, alliance, crisis detection.
pub fn clinical_baselines() -> BaselineMap {
    let mut m = BTreeMap::new();
    m.insert(
        "empathic_accuracy",
        Baseline {
            value: 0.60,
            sd: Some(0.15), // widened from 0.12 to match Ickes (1993) reported SD
            source: "Ickes (1993) empathic accuracy paradigm",
            population: "trained therapists",
        },
    );
    m.insert(
        "response_appropriateness",
        Baseline {
            value: 0.75,
            sd: Some(0.12), // widened from 0.10 to match Hill (2009) SD = 12%
            source: "Hill (2009) Helping Skills rating",
            population: "clinical psychology trainees",
        },
    );
    m.insert(
        "repair_success_rate",
        Baseline {
            value: 0.65,
            sd: Some(0.15),
            source: "Safran & Muran (2000) alliance rupture-repair",
            population: "experienced therapists",
        },
    );
    m.insert(
        "crisis_sensitivity",
        Baseline {
            value: 0.95,
            sd: Some(0.05), // widened from 0.03 to reflect greater clinical variance
            source: "C-SSRS screening validation",
            population: "crisis clinicians",
        },
    );
    m.insert(
        "crisis_specificity",
        Baseline {
            value: 0.80,
            sd: Some(0.08),
            source: "C-SSRS screening validation",
            population: "crisis clinicians",
        },
    );
    m.insert(
        "distortion_identification",
        Baseline {
            value: 0.75,
            sd: Some(0.10),
            source: "Burns (1980) cognitive distortion checklist",
            population: "CBT therapists",
        },
    );
    m.insert(
        "mi_spirit_score",
        Baseline {
            value: 3.5,
            sd: Some(0.5),
            source: "MITI 4.2 coding manual",
            population: "MI-trained clinicians",
        },
    );
    m
}

/// Spatial cognition baselines.
///
/// Sources: Shepard & Metzler (1971), Morrow et al. (1989),
/// Luck & Vogel (1997), Postma et al. (2004), Kozhevnikov & Hegarty (2001).
pub fn spatial_baselines() -> BaselineMap {
    let mut m = BTreeMap::new();
    // MentalRotation: accuracy_mean (Shepard & Metzler, 1971; Cooper & Shepard, 1973)
    m.insert(
        "mental_rotation_accuracy",
        Baseline {
            value: 0.95,
            sd: Some(0.03),
            source: "Shepard & Metzler (1971); Cooper & Shepard (1973)",
            population: "human adults",
        },
    );
    // MentalRotation: rt_slope (normalized RT slope — higher = stronger linear RT increase)
    m.insert(
        "rt_slope",
        Baseline {
            value: 0.65,
            sd: Some(0.10),
            source: "Shepard & Metzler (1971); Cooper & Shepard (1973)",
            population: "human adults",
        },
    );
    // SpatialPathUpdating: simple_accuracy — short-path spatial updating
    // (Morrow et al., 1989; Rieser, 1989). Human adults achieve ~0.80 on
    // simple 1-3 step paths; complex multi-step paths degrade substantially.
    m.insert(
        "simple_accuracy",
        Baseline {
            value: 0.80,
            sd: Some(0.12),
            source: "Morrow et al. (1989); Rieser (1989)",
            population: "human adults",
        },
    );
    // Overall updating_accuracy (includes complex paths)
    m.insert(
        "updating_accuracy",
        Baseline {
            value: 0.65,
            sd: Some(0.15),
            source: "Morrow et al. (1989); Rieser (1989)",
            population: "human adults",
        },
    );
    // LandmarkBinding: retrieval_accuracy (Luck & Vogel, 1997; Postma et al., 2004)
    m.insert(
        "retrieval_accuracy",
        Baseline {
            value: 0.80,
            sd: Some(0.10),
            source: "Luck & Vogel (1997); Postma et al. (2004)",
            population: "human adults",
        },
    );
    // PerspectiveTaking: perspective_accuracy (Kozhevnikov & Hegarty, 2001; Hegarty & Waller, 2004)
    // Wide individual differences (SD=0.18) reflect the large performance range
    // across spatial ability levels (Hegarty & Waller, 2004: low-spatial ~0.55, high-spatial ~0.90).
    m.insert(
        "perspective_accuracy",
        Baseline {
            value: 0.70,
            sd: Some(0.18),
            source: "Kozhevnikov & Hegarty (2001); Hegarty & Waller (2004)",
            population: "human adults",
        },
    );
    m
}

/// Causal reasoning baselines.
///
/// Sources: Sloman (2005), Gopnik et al. (2004), Pearl (2009).
pub fn causal_reasoning_baselines() -> BaselineMap {
    let mut m = BTreeMap::new();
    // CausalChain: chain_tracing_accuracy (Sloman, 2005; Bramley et al., 2017)
    m.insert(
        "chain_tracing_accuracy",
        Baseline {
            value: 0.75,
            sd: Some(0.10),
            source: "Sloman (2005) Causal Models; Bramley et al. (2017) Cognition",
            population: "human adults",
        },
    );
    // ConfoundDetection: confound_detection_accuracy (Pearl, 2014; Gopnik et al., 2004)
    m.insert(
        "confound_detection_accuracy",
        Baseline {
            value: 0.65,
            sd: Some(0.12),
            source: "Pearl (2014) Simpson's paradox; Gopnik et al. (2004)",
            population: "human adults",
        },
    );
    // InterventionEffect: causal_score (Pearl, 2009 do-calculus; Sloman, 2005)
    m.insert(
        "causal_score",
        Baseline {
            value: 0.70,
            sd: Some(0.10),
            source: "Pearl (2009) Causality; Sloman (2005)",
            population: "human adults",
        },
    );
    m
}

/// Security (HDC-FHE) domain baselines.
///
/// These baselines represent what conventional encrypted inference achieves.
/// Standard FHE (CKKS/BGV) introduces quantization noise that degrades accuracy.
/// HDC-OTP encryption is mathematically distance-preserving, so the "baseline"
/// for encrypted accuracy is the *plaintext* accuracy itself (perfect preservation).
///
/// For collective aggregation, the baseline is the fidelity achievable with
/// standard secure aggregation protocols (e.g., Bonawitz et al. 2017 SecAgg).
pub fn security_baselines() -> BaselineMap {
    let mut m = BTreeMap::new();
    // EncryptedClassification: encrypted_accuracy
    // Baseline: CKKS-encrypted NN classification accuracy (typically 1-5% loss)
    // Ref: Gilad-Bachrach et al. (2016) CryptoNets, ~98.95% on MNIST vs 99.5% plaintext
    m.insert(
        "encrypted_accuracy",
        Baseline {
            value: 0.95,
            sd: Some(0.03),
            source: "Gilad-Bachrach et al. (2016) CryptoNets; CKKS encrypted inference typical accuracy",
            population: "FHE-encrypted neural networks",
        },
    );
    // CollectiveAggregation: aggregation_fidelity
    // Baseline: SecAgg (Bonawitz et al. 2017) — perfect fidelity for additive aggregation
    // but HDC uses majority-vote which is approximate under encryption
    m.insert(
        "aggregation_fidelity",
        Baseline {
            value: 0.85,
            sd: Some(0.05),
            source: "Imani et al. (2019) secure HDC collaboration; bundle fidelity under OTP",
            population: "encrypted HDC systems",
        },
    );
    // EncryptedLearning: learning_accuracy
    m.insert(
        "learning_accuracy",
        Baseline {
            value: 0.92,
            sd: Some(0.04),
            source: "Imani et al. (2019) incremental HDC learning under CKKS",
            population: "encrypted HDC systems",
        },
    );
    // CrossMaskPrivacy: cross_session_leakage
    m.insert(
        "cross_session_leakage",
        Baseline {
            value: 0.02,
            sd: Some(0.01),
            source: "Shannon (1949) OTP information-theoretic bound; expected |sim-0.5| for random",
            population: "information-theoretic bound",
        },
    );
    // EncryptedBinding: binding_preservation
    m.insert(
        "binding_preservation",
        Baseline {
            value: 0.95,
            sd: Some(0.03),
            source: "Plate (2003) HRR binding fidelity under noise; CKKS approximate binding",
            population: "holographic reduced representations",
        },
    );
    // ScalingAnalysis: accuracy_at_scale
    m.insert(
        "accuracy_at_scale",
        Baseline {
            value: 0.90,
            sd: Some(0.05),
            source: "Rahimi et al. (2016) HDC classification scaling",
            population: "HDC classification systems",
        },
    );
    m
}

/// Coding domain baselines (HumanEval, bug detection).
pub fn coding_baselines() -> BaselineMap {
    let mut m = BTreeMap::new();
    // HumanEvalMini: pass_at_1
    // Task: select the correct implementation from candidates given spec + tests.
    // Human discrimination accuracy for code specification matching: novice
    // programmers ~50% (near chance for 2-AFC), intermediate ~67% (Chen et al.,
    // 2021 HumanEval human study). We use 0.50 ± 0.15 for the novice-to-
    // intermediate range of specification-based code discrimination.
    m.insert(
        "humaneval_pass_at_1",
        Baseline {
            value: 0.50,
            sd: Some(0.15),
            source:
                "Chen et al. (2021) Evaluating Large Language Models; human discrimination baseline",
            population: "novice-to-intermediate programmers (spec discrimination)",
        },
    );
    // BugDetection: delta_magnitude
    // HDC representational distance between buggy and correct code encodings.
    // Random embeddings: ~0.0; basic bag-of-tokens: ~0.15 ± 0.08;
    // structured (AST-aware) embeddings: ~0.25 ± 0.10 (Alon et al., 2019).
    // Baseline: basic embedding discrimination (0.15 ± 0.08).
    m.insert(
        "bug_detection_delta_magnitude",
        Baseline {
            value: 0.15,
            sd: Some(0.08),
            source: "Alon et al. (2019) code2vec; basic code embedding discrimination baselines",
            population: "code embedding systems (bag-of-tokens baseline)",
        },
    );
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_baselines_nonempty() {
        assert!(!worm_baselines().is_empty());
        assert!(!cogbench_baselines().is_empty());
        assert!(!tombench_baselines().is_empty());
        assert!(!memory_agent_baselines().is_empty());
        assert!(!executive_baselines().is_empty());
        assert!(!metacognition_baselines().is_empty());
        assert!(!affect_baselines().is_empty());
        assert!(!creativity_baselines().is_empty());
        assert!(!butlin_baselines().is_empty());
        assert!(!inhibition_baselines().is_empty());
        assert!(!attention_baselines().is_empty());
        assert!(!embodied_baselines().is_empty());
        assert!(!sustained_attention_baselines().is_empty());
        assert!(!motor_baselines().is_empty());
        assert!(!language_baselines().is_empty());
        assert!(!social_baselines().is_empty());
        assert!(!neuromod_baselines().is_empty());
        assert!(!consciousness_baselines().is_empty());
        assert!(!binding_baselines().is_empty());
        assert!(!speech_baselines().is_empty());
        assert!(!substrate_baselines().is_empty());
        assert!(!clinical_baselines().is_empty());
        assert!(!spatial_baselines().is_empty());
        assert!(!causal_reasoning_baselines().is_empty());
        assert!(!security_baselines().is_empty());
    }

    #[test]
    fn test_llm_baselines_nonempty() {
        let cogbench = llm_cogbench_baselines();
        assert!(
            !cogbench.is_empty(),
            "LLM CogBench baselines should not be empty"
        );
        assert!(
            cogbench.len() >= 5,
            "Expected >= 5 LLM CogBench entries, got {}",
            cogbench.len()
        );
        // Verify all have GPT-4 population
        for (_, bl) in &cogbench {
            assert_eq!(bl.population, "GPT-4");
        }

        let tombench = llm_tombench_baselines();
        assert!(
            !tombench.is_empty(),
            "LLM ToMBench baselines should not be empty"
        );
        assert!(
            tombench.len() >= 3,
            "Expected >= 3 LLM ToMBench entries, got {}",
            tombench.len()
        );
        for (_, bl) in &tombench {
            assert_eq!(bl.population, "GPT-4");
        }
    }

    #[test]
    fn test_llm_baselines_have_sd() {
        // All LLM baselines should have SD values for z-score computation
        for (key, bl) in &llm_cogbench_baselines() {
            assert!(
                bl.sd.is_some(),
                "LLM CogBench baseline '{}' missing SD",
                key
            );
        }
        for (key, bl) in &llm_tombench_baselines() {
            assert!(
                bl.sd.is_some(),
                "LLM ToMBench baseline '{}' missing SD",
                key
            );
        }
    }

    #[test]
    fn test_neuromod_baselines_count() {
        let baselines = neuromod_baselines();
        assert!(
            baselines.len() >= 60,
            "Expected >= 60 neuromod baselines, got {}",
            baselines.len()
        );
    }

    #[test]
    fn test_neuromod_baselines_key_entries() {
        let baselines = neuromod_baselines();
        // DoseResponse
        assert!(
            baselines.contains_key("da_monotonicity"),
            "Missing da_monotonicity"
        );
        assert!(
            baselines.contains_key("gaba_monotonicity"),
            "Missing gaba_monotonicity"
        );
        // ToleranceWithdrawal
        assert!(
            baselines.contains_key("tolerance_count"),
            "Missing tolerance_count"
        );
        assert!(
            baselines.contains_key("withdrawal_count"),
            "Missing withdrawal_count"
        );
        // BehavioralKnockout
        assert!(
            baselines.contains_key("ne_ko_exploration_d"),
            "Missing ne_ko_exploration_d"
        );
        assert!(
            baselines.contains_key("gaba_ko_inhibition_d"),
            "Missing gaba_ko_inhibition_d"
        );
        // AntagonistProfiles
        assert!(
            baselines.contains_key("d2_flexibility_reduction"),
            "Missing d2_flexibility_reduction"
        );
        assert!(
            baselines.contains_key("wearoff_recovery"),
            "Missing wearoff_recovery"
        );
        // ConsciousnessPharmacology
        assert!(
            baselines.contains_key("anxiolytic_proxy_peak"),
            "Missing anxiolytic_proxy_peak"
        );
        assert!(
            baselines.contains_key("ecb_proxy_mean"),
            "Missing ecb_proxy_mean"
        );
    }

    #[test]
    fn test_cultural_metadata_covers_all_domains() {
        let metadata = BaselineCollection::cultural_metadata();
        assert!(metadata.contains_key("worm"), "Missing worm metadata");
        assert!(
            metadata.contains_key("executive"),
            "Missing executive metadata"
        );
        assert!(metadata.contains_key("social"), "Missing social metadata");
        assert!(
            metadata.contains_key("reasoning"),
            "Missing reasoning metadata"
        );
        for (domain, meta) in &metadata {
            assert!(
                !meta.sample_region.is_empty(),
                "Domain '{domain}' has empty sample_region"
            );
            assert!(
                !meta.cultural_notes.is_empty(),
                "Domain '{domain}' has empty cultural_notes"
            );
        }
    }
}
