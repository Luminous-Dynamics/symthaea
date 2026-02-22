//! Published human and LLM reference data for benchmark comparison.
//!
//! All values are sourced from the respective benchmark papers.
//! References are cited inline.

use std::collections::BTreeMap;

/// A reference baseline value with source citation.
#[derive(Debug, Clone)]
pub struct Baseline {
    /// The reference value.
    pub value: f64,
    /// Source description (e.g., "Cowan (2001), Table 2").
    pub source: &'static str,
    /// Population (e.g., "human adults", "GPT-4").
    pub population: &'static str,
}

/// Get all WorM baselines.
pub fn worm_baselines() -> BTreeMap<&'static str, Baseline> {
    let mut m = BTreeMap::new();

    // Working memory capacity (Cowan's K)
    m.insert(
        "cowan_k",
        Baseline {
            value: 4.0,
            source: "Cowan (2001), The magical number 4",
            population: "human adults",
        },
    );

    // N-back accuracy at n=2
    m.insert(
        "nback_2_accuracy",
        Baseline {
            value: 0.85,
            source: "Jaeggi et al. (2010), meta-analysis",
            population: "human adults",
        },
    );

    // N-back accuracy at n=3
    m.insert(
        "nback_3_accuracy",
        Baseline {
            value: 0.70,
            source: "Jaeggi et al. (2010), meta-analysis",
            population: "human adults",
        },
    );

    // Change detection accuracy at K=4
    m.insert(
        "change_detection_k4",
        Baseline {
            value: 0.75,
            source: "Luck & Vogel (1997)",
            population: "human adults",
        },
    );

    // Spatial updating accuracy (mean across 3-10 updates)
    m.insert(
        "spatial_updating_accuracy",
        Baseline {
            value: 0.85,
            source: "Oberauer et al. (2003); Ecker et al. (2010), spatial updating paradigm",
            population: "human adults",
        },
    );

    // Binding accuracy (mean across set sizes 2-6)
    m.insert(
        "binding_accuracy",
        Baseline {
            value: 0.75,
            source: "Luck & Vogel (1997); Wheeler & Treisman (2002), feature binding in VWM",
            population: "human adults",
        },
    );

    // Serial recall primacy advantage
    m.insert(
        "serial_primacy_advantage",
        Baseline {
            value: 0.15,
            source: "Murdock (1962), serial position curve",
            population: "human adults",
        },
    );

    // Digit span forward (Wechsler, 2008; Woods et al., 2011)
    m.insert(
        "digit_span_forward",
        Baseline {
            value: 6.8,
            source: "Wechsler (2008); Woods et al. (2011), WAIS-IV norms",
            population: "human adults",
        },
    );

    // Digit span backward
    m.insert(
        "digit_span_backward",
        Baseline {
            value: 5.1,
            source: "Wechsler (2008); Woods et al. (2011), WAIS-IV norms",
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
            source: "Wilson et al. (2014), Horizon task",
            population: "human adults",
        },
    );

    // Model-basedness in two-step task
    m.insert(
        "model_basedness",
        Baseline {
            value: 0.60,
            source: "Daw et al. (2011), Two-step task",
            population: "human adults",
        },
    );

    // Temporal discounting score
    m.insert(
        "discounting_score",
        Baseline {
            value: 0.50,
            source: "Kirby et al. (1999), MCQ",
            population: "human adults",
        },
    );

    // BART average pumps
    m.insert(
        "bart_avg_pumps",
        Baseline {
            value: 30.0,
            source: "Lejuez et al. (2002), BART",
            population: "human adults",
        },
    );

    // Restless bandit: reward tracking over changing payoffs
    m.insert(
        "restless_bandit_regret",
        Baseline {
            value: 0.25,
            source: "Speekenbrink & Konstantinidis (2015), Information & choice in a changing world",
            population: "human adults",
        },
    );

    m.insert(
        "restless_bandit_accuracy",
        Baseline {
            value: 0.75,
            source: "Speekenbrink & Konstantinidis (2015), 1 - normalized regret",
            population: "human adults",
        },
    );

    // Instrumental conditioning: contingency sensitivity
    m.insert(
        "instrumental_sensitivity",
        Baseline {
            value: 0.70,
            source: "Dickinson (1985), Actions and habits",
            population: "human adults (estimated from instrumental learning literature)",
        },
    );

    // Probabilistic reasoning: likelihood weight (Bayesian updating)
    m.insert(
        "probabilistic_likelihood_weight",
        Baseline {
            value: 0.50,
            source: "Phillips & Edwards (1966); Grether (1980), conservatism in probability updating",
            population: "human adults (Bayesian normative = 0.50 for symmetric evidence)",
        },
    );

    // Reversal learning (Cools et al. 2002; Clark et al. 2004)
    m.insert(
        "reversal_win_stay",
        Baseline {
            value: 0.85,
            source: "Cools et al. (2002), Defining the neural mechanisms of probabilistic reversal learning",
            population: "human adults",
        },
    );
    m.insert(
        "reversal_lose_shift",
        Baseline {
            value: 0.70,
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
            source: "Cools et al. (2002); Clark et al. (2004), binary reversal paradigm (200 trials)",
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
            source: "Baron-Cohen et al. (1985), Sally-Anne",
            population: "human adults",
        },
    );

    // Faux pas recognition
    m.insert(
        "faux_pas_accuracy",
        Baseline {
            value: 0.85,
            source: "Baron-Cohen et al. (1999), Faux Pas test",
            population: "human adults",
        },
    );

    // Hinting task accuracy
    m.insert(
        "hinting_accuracy",
        Baseline {
            value: 0.80,
            source: "Corcoran et al. (1995), Hinting Task",
            population: "human adults",
        },
    );

    // Persuasion detection
    m.insert(
        "persuasion_detection",
        Baseline {
            value: 0.85,
            source: "Happé (1994), An advanced test of theory of mind",
            population: "human adults",
        },
    );

    // Strange story accuracy
    m.insert(
        "strange_story_accuracy",
        Baseline {
            value: 0.85,
            source: "Happé (1994), An advanced test of theory of mind",
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
            source: "Kohli & Kaur (2006), WCST norms",
            population: "human adults",
        },
    );
    m.insert(
        "wcst_perseverative_errors",
        Baseline {
            value: 8.29,
            source: "Kohli & Kaur (2006), WCST norms",
            population: "human adults",
        },
    );
    m.insert(
        "wcst_trials_to_first",
        Baseline {
            value: 12.17,
            source: "Kohli & Kaur (2006), WCST norms",
            population: "human adults",
        },
    );

    // IGT (Bechara et al., 1994; Steingroever et al., 2015)
    m.insert(
        "igt_overall_net_score",
        Baseline {
            value: 17.5,
            source: "Bechara et al. (1994); Steingroever et al. (2015), midpoint of +10 to +25",
            population: "human adults",
        },
    );
    m.insert(
        "igt_deck_preference_good",
        Baseline {
            value: 0.65,
            source: "Steingroever et al. (2015), last 40 trials",
            population: "human adults",
        },
    );

    // Raven's Progressive Matrices (Raven, 1938; Murphy et al., 2023)
    m.insert(
        "ravens_overall_accuracy",
        Baseline {
            value: 0.78,
            source: "Raven (1938); Murphy et al. (2023), SPM ~47/60",
            population: "human adults",
        },
    );
    m.insert(
        "ravens_easy_accuracy",
        Baseline {
            value: 0.95,
            source: "Raven (1938), Set A-B",
            population: "human adults",
        },
    );

    // Stroop Color-Word Interference (MacLeod, 1991; Stroop, 1935)
    m.insert(
        "stroop_congruent_accuracy",
        Baseline {
            value: 0.98,
            source: "MacLeod (1991), Half a century of research on the Stroop effect",
            population: "human adults",
        },
    );
    m.insert(
        "stroop_incongruent_accuracy",
        Baseline {
            value: 0.88,
            source: "MacLeod (1991), Half a century of research on the Stroop effect",
            population: "human adults",
        },
    );
    m.insert(
        "stroop_effect",
        Baseline {
            value: 0.10,
            source: "MacLeod (1991), accuracy-based Stroop effect",
            population: "human adults",
        },
    );

    // Eriksen Flanker Task (Eriksen & Eriksen, 1974; Ridderinkhof et al., 2021)
    m.insert(
        "flanker_congruent_accuracy",
        Baseline {
            value: 0.97,
            source: "Eriksen & Eriksen (1974); Ridderinkhof et al. (2021)",
            population: "human adults",
        },
    );
    m.insert(
        "flanker_incongruent_accuracy",
        Baseline {
            value: 0.90,
            source: "Eriksen & Eriksen (1974); Ridderinkhof et al. (2021)",
            population: "human adults",
        },
    );
    m.insert(
        "flanker_effect",
        Baseline {
            value: 0.07,
            source: "Eriksen & Eriksen (1974), accuracy-based flanker effect",
            population: "human adults",
        },
    );

    // Tower of London (Shallice, 1982; Kaller et al., 2016)
    m.insert(
        "tol_overall_optimal_rate",
        Baseline {
            value: 0.63,
            source: "Kaller et al. (2016), TOL-F norms",
            population: "human adults",
        },
    );
    m.insert(
        "tol_planning_efficiency",
        Baseline {
            value: 0.82,
            source: "Kaller et al. (2016), optimal/actual moves ratio",
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
            source: "Fleming & Lau (2014), midpoint of 0.10-0.20",
            population: "human adults",
        },
    );
    m.insert(
        "discrimination_gamma",
        Baseline {
            value: 0.50,
            source: "Fleming & Lau (2014), midpoint of 0.40-0.60",
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
            source: "Tulving (1985), Memory and consciousness; Roediger & McDermott (1995), DRM paradigm false recall ~15%",
            population: "human adults",
        },
    );

    m.insert(
        "test_time_learning",
        Baseline {
            value: 0.75,
            source: "Karpicke & Roediger (2008), The critical importance of retrieval for learning",
            population: "human adults",
        },
    );

    // Long-range retention at 50-cycle delay
    m.insert(
        "long_range_delay_50",
        Baseline {
            value: 0.70,
            source: "Baddeley (1997), Human Memory: Theory and Practice",
            population: "human adults",
        },
    );

    // Conflict resolution: recency preference
    m.insert(
        "conflict_recency_preference",
        Baseline {
            value: 0.65,
            source: "Oberauer (2002), Access to information in working memory",
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
            source: "Bradley & Lang (1999), IAPS affective ratings",
            population: "human adults",
        },
    );

    // Mood-congruent recall congruence ratio (Blaney, 1986)
    m.insert(
        "congruence_ratio",
        Baseline {
            value: 0.60,
            source: "Blaney (1986), Affect and memory: a review",
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
            source: "Butlin et al. (2023), Consciousness in Artificial Intelligence: Insights from the Science of Consciousness",
            population: "human adults (all indicators present by definition)",
        },
    );

    m.insert(
        "presence_ratio",
        Baseline {
            value: 1.0,
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
            source: "Bowden & Jung-Beeman (2003), Normative data for 144 compound remote associate problems",
            population: "human adults",
        },
    );

    // Alternate Uses Task fluency (Torrance, 1974)
    m.insert(
        "aut_fluency",
        Baseline {
            value: 8.0,
            source: "Torrance (1974), Torrance Tests of Creative Thinking",
            population: "human adults",
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
    }
}
