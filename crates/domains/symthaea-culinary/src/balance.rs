//! `balance_score` — a dish-cohesion heuristic, and the honest replacement for
//! the pitch's "Φ of a plate".
//!
//! **This is a documented heuristic, not a consciousness/integration measure.**
//! Φ measures irreducible causal integration of an information-processing system;
//! a dish has no such causal state to compute over, so applying Φ here would be a
//! decorative metric. Instead we score two things chefs actually reason about:
//! how *balanced* the five basic tastes are, and how much textural/thermal
//! *contrast* the dish has. No claim beyond "these rules of thumb, made explicit".

/// Intensities of the five basic tastes, each in [0, 1].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TasteProfile {
    pub sweet: f64,
    pub salty: f64,
    pub sour: f64,
    pub bitter: f64,
    pub umami: f64,
}

impl TasteProfile {
    fn tastes(&self) -> [f64; 5] {
        [self.sweet, self.salty, self.sour, self.bitter, self.umami]
    }
}

/// Result of the heuristic, with its two components exposed so callers can see
/// *why*, not just a single opaque number.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BalanceScore {
    /// How evenly the present tastes are balanced, in [0, 1] (1 = perfectly even).
    pub taste_balance: f64,
    /// Textural/thermal contrast, in [0, 1] (1 = strong contrast).
    pub contrast: f64,
    /// Combined heuristic score in [0, 1].
    pub score: f64,
}

/// Score a dish. `contrast` is a caller-supplied [0,1] proxy for textural/thermal
/// variety (e.g. crisp-vs-soft, hot-vs-cold present). The taste-balance term
/// rewards profiles whose non-trivial tastes are close in magnitude (a dish that
/// is *only* sweet scores low; sweet-sour-salty in balance scores high).
pub fn balance_score(taste: &TasteProfile, contrast: f64) -> BalanceScore {
    let contrast = contrast.clamp(0.0, 1.0);
    let t = taste.tastes();
    let present: Vec<f64> = t.iter().copied().filter(|&x| x > 0.05).collect();

    let taste_balance = if present.len() < 2 {
        // one (or no) dominant taste: unbalanced by construction.
        0.0
    } else {
        let mean = present.iter().sum::<f64>() / present.len() as f64;
        let var = present.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / present.len() as f64;
        let cv = if mean > 0.0 { var.sqrt() / mean } else { 1.0 };
        // low coefficient of variation ⇒ well balanced. Also reward having more
        // distinct tastes present (up to the five).
        let evenness = (1.0 - cv).clamp(0.0, 1.0);
        let breadth = (present.len() as f64 - 1.0) / 4.0; // 0 at 1 taste, 1 at 5
        0.7 * evenness + 0.3 * breadth
    };

    let score = 0.6 * taste_balance + 0.4 * contrast;
    BalanceScore {
        taste_balance,
        contrast,
        score,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_sweet_is_unbalanced() {
        let s = balance_score(
            &TasteProfile {
                sweet: 0.9,
                ..Default::default()
            },
            0.0,
        );
        assert_eq!(s.taste_balance, 0.0);
        assert!(s.score < 0.2);
    }

    #[test]
    fn balanced_multi_taste_scores_higher_than_monotone() {
        let balanced = balance_score(
            &TasteProfile {
                sweet: 0.6,
                salty: 0.6,
                sour: 0.5,
                umami: 0.55,
                ..Default::default()
            },
            0.7,
        );
        let monotone = balance_score(
            &TasteProfile {
                salty: 0.9,
                ..Default::default()
            },
            0.7,
        );
        assert!(
            balanced.score > monotone.score,
            "balanced {} should beat monotone {}",
            balanced.score,
            monotone.score
        );
    }

    #[test]
    fn score_stays_in_unit_range() {
        let s = balance_score(
            &TasteProfile {
                sweet: 1.0,
                salty: 1.0,
                sour: 1.0,
                bitter: 1.0,
                umami: 1.0,
            },
            1.0,
        );
        assert!((0.0..=1.0).contains(&s.score));
    }
}
