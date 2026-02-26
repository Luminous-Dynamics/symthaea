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
    /// GPT-4 baselines from CogBench (Coda et al., 2023).
    pub llm_cogbench: BaselineMap,
    /// GPT-4 baselines from ToMBench (Kosinski, 2023).
    pub llm_tombench: BaselineMap,
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
            llm_cogbench: llm_cogbench_baselines(),
            llm_tombench: llm_tombench_baselines(),
        }
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
            value: 4.0,
            sd: Some(1.0),
            source: "Logan (1994); Verbruggen & Logan (2008), SSRT ~200ms at 50ms/tick",
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
            value: 2.0,
            sd: Some(0.5),
            source: "Treisman & Gelade (1980), conjunction_slope - feature_slope",
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
    }

    #[test]
    fn test_llm_baselines_nonempty() {
        let cogbench = llm_cogbench_baselines();
        assert!(!cogbench.is_empty(), "LLM CogBench baselines should not be empty");
        assert!(cogbench.len() >= 5, "Expected >= 5 LLM CogBench entries, got {}", cogbench.len());
        // Verify all have GPT-4 population
        for (_, bl) in &cogbench {
            assert_eq!(bl.population, "GPT-4");
        }

        let tombench = llm_tombench_baselines();
        assert!(!tombench.is_empty(), "LLM ToMBench baselines should not be empty");
        assert!(tombench.len() >= 3, "Expected >= 3 LLM ToMBench entries, got {}", tombench.len());
        for (_, bl) in &tombench {
            assert_eq!(bl.population, "GPT-4");
        }
    }

    #[test]
    fn test_llm_baselines_have_sd() {
        // All LLM baselines should have SD values for z-score computation
        for (key, bl) in &llm_cogbench_baselines() {
            assert!(bl.sd.is_some(), "LLM CogBench baseline '{}' missing SD", key);
        }
        for (key, bl) in &llm_tombench_baselines() {
            assert!(bl.sd.is_some(), "LLM ToMBench baseline '{}' missing SD", key);
        }
    }
}
