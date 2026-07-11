// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bayesian updating and binary-classifier / diagnostic-test metrics — the
//! layer that connects most directly to Symthaea's calibration loop (updating a
//! belief given evidence of known reliability).

/// Posterior P(H | E) from prior P(H), the likelihood P(E | H), and the
/// false-positive likelihood P(E | ¬H). Returns `None` if the evidence has zero
/// total probability.
pub fn posterior(prior: f64, likelihood: f64, likelihood_given_not: f64) -> Option<f64> {
    let joint = prior * likelihood;
    let evidence = joint + (1.0 - prior) * likelihood_given_not;
    if evidence == 0.0 {
        return None;
    }
    Some(joint / evidence)
}

/// A 2×2 confusion matrix for a binary classifier / diagnostic test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Confusion {
    pub true_positive: f64,
    pub false_positive: f64,
    pub false_negative: f64,
    pub true_negative: f64,
}

impl Confusion {
    /// Sensitivity / recall / true-positive rate: TP / (TP + FN).
    pub fn sensitivity(&self) -> Option<f64> {
        let d = self.true_positive + self.false_negative;
        (d > 0.0).then_some(self.true_positive / d)
    }

    /// Specificity / true-negative rate: TN / (TN + FP).
    pub fn specificity(&self) -> Option<f64> {
        let d = self.true_negative + self.false_positive;
        (d > 0.0).then_some(self.true_negative / d)
    }

    /// Precision / positive predictive value: TP / (TP + FP).
    pub fn precision(&self) -> Option<f64> {
        let d = self.true_positive + self.false_positive;
        (d > 0.0).then_some(self.true_positive / d)
    }

    /// Negative predictive value: TN / (TN + FN).
    pub fn npv(&self) -> Option<f64> {
        let d = self.true_negative + self.false_negative;
        (d > 0.0).then_some(self.true_negative / d)
    }

    /// Accuracy: (TP + TN) / total.
    pub fn accuracy(&self) -> Option<f64> {
        let total =
            self.true_positive + self.false_positive + self.false_negative + self.true_negative;
        (total > 0.0).then_some((self.true_positive + self.true_negative) / total)
    }

    /// F1 score: harmonic mean of precision and recall.
    pub fn f1(&self) -> Option<f64> {
        let p = self.precision()?;
        let r = self.sensitivity()?;
        (p + r > 0.0).then_some(2.0 * p * r / (p + r))
    }

    /// Positive likelihood ratio: sensitivity / (1 − specificity).
    pub fn likelihood_ratio_positive(&self) -> Option<f64> {
        let (sens, spec) = (self.sensitivity()?, self.specificity()?);
        (spec < 1.0).then_some(sens / (1.0 - spec))
    }
}

/// Diagnostic posterior: P(disease | positive test) from prevalence,
/// sensitivity, and specificity — Bayes' theorem in its most quoted clinical
/// form.
pub fn posterior_positive(prevalence: f64, sensitivity: f64, specificity: f64) -> Option<f64> {
    posterior(prevalence, sensitivity, 1.0 - specificity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classic_rare_disease_ppv() {
        // Prevalence 1%, sensitivity 99%, specificity 95% → PPV ≈ 16.7%.
        let p = posterior_positive(0.01, 0.99, 0.95).unwrap();
        assert!((p - 0.166_666).abs() < 1e-4, "{p}");
    }

    #[test]
    fn confusion_metrics() {
        let c = Confusion {
            true_positive: 90.0,
            false_positive: 10.0,
            false_negative: 20.0,
            true_negative: 80.0,
        };
        assert!((c.sensitivity().unwrap() - 90.0 / 110.0).abs() < 1e-12);
        assert!((c.specificity().unwrap() - 80.0 / 90.0).abs() < 1e-12);
        assert!((c.precision().unwrap() - 0.9).abs() < 1e-12);
        assert!((c.accuracy().unwrap() - 170.0 / 200.0).abs() < 1e-12);
        // LR+ = sens / (1-spec).
        let lr = c.likelihood_ratio_positive().unwrap();
        assert!((lr - (90.0 / 110.0) / (10.0 / 90.0)).abs() < 1e-9);
    }

    #[test]
    fn posterior_is_a_probability() {
        // Uninformative evidence leaves the prior unchanged.
        let p = posterior(0.3, 0.5, 0.5).unwrap();
        assert!((p - 0.3).abs() < 1e-12);
        assert!(posterior(0.0, 0.0, 0.0).is_none());
    }
}
