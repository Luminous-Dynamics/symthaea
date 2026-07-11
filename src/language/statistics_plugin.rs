// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Statistics domain plugin — the Bayesian diagnostic (positive predictive
//! value) and the standard-normal CDF, answered deterministically.
//!
//! The PPV case is the flagship: "prevalence 1%, sensitivity 99%, specificity
//! 95% — what's the chance of disease given a positive test?" is a calculation
//! language models routinely get wrong (base-rate neglect); here it is exact.

use crate::language::domain_plugin::{ComputedResult, DomainPlugin, Entity};
use crate::mind::structured_thought::{ETier, EpistemicCube, MTier, NTier};
use symthaea_statistics::distributions::normal_cdf;
use symthaea_statistics::posterior_positive;

pub struct StatisticsDomainPlugin;

fn result(answer: String) -> ComputedResult {
    ComputedResult {
        answer,
        cube: EpistemicCube {
            e: ETier::E4,
            n: NTier::N3,
            m: MTier::M3,
            h: None,
        },
        psi: 0.0,
        proof_available: false,
    }
}

/// The probability value following any of `keys`. Understands `%` (and treats a
/// bare value > 1 as a percentage), and tolerates "sensitivity **of** 99%".
fn prob_after(text: &str, keys: &[&str]) -> Option<f64> {
    let flat = text.to_lowercase().replace(" of ", " ").replace('=', " ");
    let toks: Vec<&str> = flat
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|t| !t.is_empty())
        .collect();
    for (i, t) in toks.iter().enumerate() {
        if keys.contains(t) {
            if let Some(next) = toks.get(i + 1) {
                let had_pct = next.contains('%');
                let cleaned: String = next
                    .trim_matches(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))
                    .to_string();
                if let Ok(v) = cleaned.parse::<f64>() {
                    return Some(if had_pct || v > 1.0 { v / 100.0 } else { v });
                }
            }
        }
    }
    None
}

impl StatisticsDomainPlugin {
    fn is_diagnostic(text: &str) -> bool {
        let t = text.to_lowercase();
        t.contains("sensitivity") && t.contains("specificity")
    }

    fn is_normal_cdf(text: &str) -> bool {
        let t = text.to_lowercase();
        (t.contains("z-score") || t.contains("z score") || t.contains("standard normal"))
            && (t.contains("probability") || t.contains("percentile") || t.contains("below"))
    }

    fn diagnostic(input: &str) -> Option<ComputedResult> {
        let prevalence = prob_after(input, &["prevalence", "rate", "baserate"])?;
        let sensitivity = prob_after(input, &["sensitivity", "sensitive"])?;
        let specificity = prob_after(input, &["specificity", "specific"])?;
        let ppv = posterior_positive(prevalence, sensitivity, specificity)?;
        Some(result(format!(
            "Bayesian diagnostic: with prevalence {:.1}%, sensitivity {:.1}%, and specificity \
             {:.1}%, a positive test means P(disease | positive) = {:.1}% (the positive \
             predictive value). Base-rate neglect makes this feel much higher than it is.",
            prevalence * 100.0,
            sensitivity * 100.0,
            specificity * 100.0,
            ppv * 100.0
        )))
    }

    fn normal(input: &str) -> Option<ComputedResult> {
        // A z-score is a plain signed number (not a probability, so no %
        // handling): take the first number in the query.
        let z = crate::language::plugin_parse::signed_numbers(input)
            .into_iter()
            .next()?;
        let p = normal_cdf(z);
        Some(result(format!(
            "For a standard normal, P(Z < {z}) = {:.4} (i.e. the {:.1}th percentile).",
            p,
            p * 100.0
        )))
    }
}

impl DomainPlugin for StatisticsDomainPlugin {
    fn domain_name(&self) -> &str {
        "statistics"
    }

    fn extract_entities(&self, _text: &str) -> Vec<Entity> {
        Vec::new()
    }

    fn is_in_domain(&self, topic: &str) -> f64 {
        if Self::is_diagnostic(topic) || Self::is_normal_cdf(topic) {
            0.9
        } else {
            0.1
        }
    }

    fn vocabulary(&self) -> Vec<String> {
        [
            "sensitivity",
            "specificity",
            "prevalence",
            "probability",
            "normal",
            "percentile",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn compute(&self, input: &str, _entities: &[Entity]) -> Option<ComputedResult> {
        if Self::is_diagnostic(input) {
            return Self::diagnostic(input);
        }
        if Self::is_normal_cdf(input) {
            return Self::normal(input);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rare_disease_ppv() {
        let p = StatisticsDomainPlugin;
        let r = p
            .compute(
                "prevalence 1%, sensitivity 99%, specificity 95% — chance of disease if positive?",
                &[],
            )
            .unwrap();
        // Classic answer ≈ 16.7%.
        assert!(r.answer.contains("16.7"), "{}", r.answer);
    }

    #[test]
    fn accepts_fractions_and_of_phrasing() {
        let p = StatisticsDomainPlugin;
        let r = p
            .compute(
                "sensitivity of 0.9 and specificity of 0.9 with prevalence of 0.5",
                &[],
            )
            .unwrap();
        // Balanced prior, symmetric test → PPV 90%.
        assert!(r.answer.contains("90.0%"), "{}", r.answer);
    }

    #[test]
    fn normal_cdf_query() {
        let p = StatisticsDomainPlugin;
        let r = p
            .compute(
                "standard normal: probability Z below a z-score of 1.96",
                &[],
            )
            .unwrap();
        assert!(r.answer.contains("0.975"), "{}", r.answer);
    }

    #[test]
    fn unrelated_input_none() {
        let p = StatisticsDomainPlugin;
        assert!(p.compute("what is the weather today?", &[]).is_none());
    }
}
