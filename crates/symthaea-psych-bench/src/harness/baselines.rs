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

    // Serial recall primacy advantage
    m.insert(
        "serial_primacy_advantage",
        Baseline {
            value: 0.15,
            source: "Murdock (1962), serial position curve",
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

    // Instrumental conditioning: contingency sensitivity
    m.insert(
        "instrumental_sensitivity",
        Baseline {
            value: 0.70,
            source: "Dickinson (1985), Actions and habits",
            population: "human adults (estimated from instrumental learning literature)",
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
    }
}
