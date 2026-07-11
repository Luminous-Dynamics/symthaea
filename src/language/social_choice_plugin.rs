// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Social-choice domain plugin — seat apportionment (D'Hondt / Sainte-Laguë /
//! Hamilton) and weighted voting-power indices (Banzhaf / Shapley-Shubik),
//! answered deterministically. Consumer alignment: Mycelix governance.

use crate::language::domain_plugin::{ComputedResult, DomainPlugin, Entity};
use crate::mind::structured_thought::{ETier, EpistemicCube, MTier, NTier};
use symthaea_social_choice::{banzhaf, dhondt, hamilton, sainte_lague, shapley_shubik};

pub struct SocialChoiceDomainPlugin;

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

const STOP: &[&str] = &[
    "allocate",
    "distribute",
    "apportion",
    "seats",
    "seat",
    "among",
    "to",
    "parties",
    "party",
    "using",
    "with",
    "votes",
    "vote",
    "and",
    "the",
    "by",
    "method",
    "for",
    "between",
    "dhondt",
    "hondt",
    "jefferson",
    "sainte",
    "lague",
    "laguë",
    "webster",
    "hamilton",
    "remainder",
    "largest",
    "quota",
    "weights",
    "weight",
];

fn trim_punct(s: &str) -> String {
    s.trim_matches(|c: char| !c.is_alphanumeric()).to_string()
}

/// Extract `(name, count)` pairs: a non-numeric, non-stopword token immediately
/// followed by an integer (handles both `A 100` and `A=100`).
fn parse_pairs(input: &str) -> Vec<(String, u64)> {
    let normalized = input.replace('=', " ");
    let toks: Vec<&str> = normalized
        .split(|c: char| c.is_whitespace() || c == ',' || c == ';')
        .filter(|t| !t.is_empty())
        .collect();
    let mut pairs = Vec::new();
    let mut i = 0;
    while i + 1 < toks.len() {
        let name = trim_punct(toks[i]);
        let is_word = !name.is_empty() && name.chars().any(|c| c.is_alphabetic());
        let is_stop = STOP.contains(&name.to_lowercase().as_str());
        if is_word && !is_stop {
            if let Ok(v) = trim_punct(toks[i + 1]).parse::<u64>() {
                pairs.push((name, v));
                i += 2;
                continue;
            }
        }
        i += 1;
    }
    pairs
}

/// The integer immediately preceding the word "seats".
fn parse_seats(input: &str) -> Option<u64> {
    let lower = input.to_lowercase();
    let toks: Vec<&str> = lower
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|t| !t.is_empty())
        .collect();
    for (i, t) in toks.iter().enumerate() {
        if (*t == "seats" || *t == "seat") && i > 0 {
            if let Ok(n) = trim_punct(toks[i - 1]).parse::<u64>() {
                return Some(n);
            }
        }
    }
    None
}

/// Parse a weighted voting game: the numbers following "weights" (up to
/// "quota") are the weights; the number after "quota"/"threshold" is the quota.
fn parse_weighted_game(input: &str) -> Option<(Vec<u64>, u64)> {
    let lower = input.to_lowercase().replace('=', " ");
    let toks: Vec<&str> = lower
        .split(|c: char| c.is_whitespace() || c == ',' || c == ';')
        .filter(|t| !t.is_empty())
        .collect();
    let wpos = toks.iter().position(|t| t.starts_with("weight"))?;
    let qpos = toks
        .iter()
        .position(|t| *t == "quota" || *t == "threshold" || *t == "majority")?;
    let mut weights = Vec::new();
    for t in &toks[wpos + 1..qpos.max(wpos + 1)] {
        if let Ok(v) = trim_punct(t).parse::<u64>() {
            weights.push(v);
        }
    }
    let quota = toks
        .get(qpos + 1)
        .and_then(|t| trim_punct(t).parse::<u64>().ok())?;
    if weights.is_empty() {
        return None;
    }
    Some((weights, quota))
}

impl SocialChoiceDomainPlugin {
    fn apportionment_method(text: &str) -> Option<&'static str> {
        let t = text.to_lowercase();
        if t.contains("sainte")
            || t.contains("lague")
            || t.contains("laguë")
            || t.contains("webster")
        {
            Some("sainte-lague")
        } else if t.contains("hamilton") || t.contains("largest remainder") {
            Some("hamilton")
        } else if t.contains("hondt") || t.contains("jefferson") {
            Some("dhondt")
        } else {
            None
        }
    }

    fn is_power(text: &str) -> bool {
        let t = text.to_lowercase();
        t.contains("banzhaf") || t.contains("shapley") || t.contains("voting power")
    }

    fn apportionment(input: &str) -> Option<ComputedResult> {
        let method = Self::apportionment_method(input)?;
        if !input.to_lowercase().contains("seat") {
            return None;
        }
        let parties = parse_pairs(input);
        let seats = parse_seats(input)?;
        if parties.len() < 2 {
            return None;
        }
        let alloc = match method {
            "sainte-lague" => sainte_lague(&parties, seats),
            "hamilton" => hamilton(&parties, seats),
            _ => dhondt(&parties, seats),
        };
        let seats_str = alloc
            .iter()
            .map(|(n, s)| format!("{n}: {s}"))
            .collect::<Vec<_>>()
            .join(", ");
        let pretty = match method {
            "sainte-lague" => "Sainte-Laguë",
            "hamilton" => "Hamilton (largest remainder)",
            _ => "D'Hondt",
        };
        Some(result(format!(
            "{pretty} apportionment of {seats} seats: {seats_str}."
        )))
    }

    fn power(input: &str) -> Option<ComputedResult> {
        let (weights, quota) = parse_weighted_game(input)?;
        let t = input.to_lowercase();
        let fmt = |v: &[f64]| {
            v.iter()
                .enumerate()
                .map(|(i, p)| format!("player {} {:.3}", i + 1, p))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let mut lines = Vec::new();
        if t.contains("banzhaf") || !t.contains("shapley") {
            lines.push(format!("Banzhaf index: {}", fmt(&banzhaf(&weights, quota))));
        }
        if t.contains("shapley") || !t.contains("banzhaf") {
            lines.push(format!(
                "Shapley-Shubik index: {}",
                fmt(&shapley_shubik(&weights, quota))
            ));
        }
        Some(result(format!(
            "Weighted voting game (weights {weights:?}, quota {quota}). {}",
            lines.join("; ")
        )))
    }
}

impl DomainPlugin for SocialChoiceDomainPlugin {
    fn domain_name(&self) -> &str {
        "social_choice"
    }

    fn extract_entities(&self, _text: &str) -> Vec<Entity> {
        Vec::new()
    }

    fn is_in_domain(&self, topic: &str) -> f64 {
        if Self::apportionment_method(topic).is_some() || Self::is_power(topic) {
            0.9
        } else {
            0.1
        }
    }

    fn vocabulary(&self) -> Vec<String> {
        [
            "apportionment",
            "seats",
            "dhondt",
            "banzhaf",
            "shapley",
            "quota",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn compute(&self, input: &str, _entities: &[Entity]) -> Option<ComputedResult> {
        if Self::apportionment_method(input).is_some() {
            if let Some(r) = Self::apportionment(input) {
                return Some(r);
            }
        }
        if Self::is_power(input) {
            return Self::power(input);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dhondt_apportionment() {
        let p = SocialChoiceDomainPlugin;
        let r = p
            .compute(
                "allocate 8 seats using D'Hondt among A 100, B 80, C 30, D 20",
                &[],
            )
            .unwrap();
        // Classic result 4/3/1/0.
        assert!(r.answer.contains("A: 4"), "{}", r.answer);
        assert!(r.answer.contains("B: 3"));
        assert!(r.answer.contains("C: 1"));
        assert!(r.answer.contains("D: 0"));
    }

    #[test]
    fn sainte_lague_differs() {
        let p = SocialChoiceDomainPlugin;
        let r = p
            .compute(
                "distribute 8 seats by Sainte-Laguë with votes A=100 B=80 C=30 D=20",
                &[],
            )
            .unwrap();
        // Sainte-Laguë gives D a seat: 3/3/1/1.
        assert!(r.answer.contains("D: 1"), "{}", r.answer);
    }

    #[test]
    fn banzhaf_power() {
        let p = SocialChoiceDomainPlugin;
        let r = p
            .compute("Banzhaf voting power for weights 3 2 1 quota 4", &[])
            .unwrap();
        // player 1 holds 0.600.
        assert!(r.answer.contains("player 1 0.600"), "{}", r.answer);
    }

    #[test]
    fn unrelated_none() {
        let p = SocialChoiceDomainPlugin;
        assert!(p.compute("who won the election last night?", &[]).is_none());
    }
}
