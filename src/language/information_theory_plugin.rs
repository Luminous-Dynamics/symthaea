// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Information-theory domain plugin — Shannon entropy of a distribution, in
//! bits, answered deterministically.

use crate::language::domain_plugin::{ComputedResult, DomainPlugin, Entity};
use crate::language::plugin_parse::signed_numbers;
use crate::mind::structured_thought::{ETier, EpistemicCube, MTier, NTier};
use symthaea_information_theory::entropy;

pub struct InformationTheoryDomainPlugin;

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

impl InformationTheoryDomainPlugin {
    fn has_cue(text: &str) -> bool {
        let t = text.to_lowercase();
        t.contains("entropy") || t.contains("information content") || t.contains("surprisal")
    }

    /// Extract a probability distribution from the query: a named uniform source
    /// ("fair coin", "fair die", "fair 8-sided die") or an explicit list of
    /// numbers that form a valid distribution (sum ≈ 1, all in [0, 1]).
    fn distribution(input: &str) -> Option<Vec<f64>> {
        let t = input.to_lowercase();
        if t.contains("fair coin") {
            return Some(vec![0.5, 0.5]);
        }
        if t.contains("fair die") || t.contains("fair dice") {
            return Some(vec![1.0 / 6.0; 6]);
        }
        // "fair N-sided die" / "uniform over N".
        if t.contains("fair") && (t.contains("sided") || t.contains("uniform")) {
            if let Some(&n) = signed_numbers(input).first() {
                let n = n as usize;
                if n >= 2 {
                    return Some(vec![1.0 / n as f64; n]);
                }
            }
        }
        // Explicit probabilities.
        let nums = signed_numbers(input);
        if nums.len() >= 2
            && nums.iter().all(|&p| (0.0..=1.0).contains(&p))
            && (nums.iter().sum::<f64>() - 1.0).abs() < 1e-6
        {
            return Some(nums);
        }
        None
    }
}

impl DomainPlugin for InformationTheoryDomainPlugin {
    fn domain_name(&self) -> &str {
        "information_theory"
    }

    fn extract_entities(&self, _text: &str) -> Vec<Entity> {
        Vec::new()
    }

    fn is_in_domain(&self, topic: &str) -> f64 {
        if Self::has_cue(topic) { 0.9 } else { 0.1 }
    }

    fn vocabulary(&self) -> Vec<String> {
        [
            "entropy",
            "information",
            "bits",
            "surprisal",
            "uncertainty",
            "shannon",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn compute(&self, input: &str, _entities: &[Entity]) -> Option<ComputedResult> {
        if !Self::has_cue(input) {
            return None;
        }
        let dist = Self::distribution(input)?;
        let h = entropy(&dist);
        Some(result(format!(
            "The Shannon entropy of that distribution ({} outcomes) is {h:.4} bits.",
            dist.len()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fair_coin_is_one_bit() {
        let p = InformationTheoryDomainPlugin;
        let r = p
            .compute("what is the entropy of a fair coin?", &[])
            .unwrap();
        assert!(r.answer.contains("1.0000"), "{}", r.answer);
    }

    #[test]
    fn fair_die_is_log2_six() {
        let p = InformationTheoryDomainPlugin;
        let r = p.compute("entropy of a fair die", &[]).unwrap();
        // log2(6) ≈ 2.585 bits.
        assert!(r.answer.contains("2.585"), "{}", r.answer);
    }

    #[test]
    fn explicit_distribution() {
        let p = InformationTheoryDomainPlugin;
        // A degenerate certain outcome has zero entropy.
        let r = p
            .compute("shannon entropy of the distribution 1.0 0.0", &[])
            .unwrap();
        assert!(r.answer.contains("0.0000"), "{}", r.answer);
    }

    #[test]
    fn no_cue_none() {
        let p = InformationTheoryDomainPlugin;
        assert!(p.compute("what should I have for lunch?", &[]).is_none());
    }
}
