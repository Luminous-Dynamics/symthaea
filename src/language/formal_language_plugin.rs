// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Formal-language domain plugin — decide whether a regular expression matches
//! a string, using the real Thompson-NFA engine (anchored whole-string match).

use crate::language::domain_plugin::{ComputedResult, DomainPlugin, Entity};
use crate::mind::structured_thought::{ETier, EpistemicCube, MTier, NTier};
use symthaea_formal_language::Regex;

pub struct FormalLanguageDomainPlugin;

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
        proof_available: true,
    }
}

/// Pull out substrings delimited by backticks, single, or double quotes, in
/// order. Used to separate the pattern from the test string.
fn delimited(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '`' || c == '\'' || c == '"' {
            if let Some(end) = chars[i + 1..].iter().position(|&d| d == c) {
                let seg: String = chars[i + 1..i + 1 + end].iter().collect();
                out.push(seg);
                i += end + 2;
                continue;
            }
        }
        i += 1;
    }
    out
}

impl FormalLanguageDomainPlugin {
    fn has_cue(text: &str) -> bool {
        let t = text.to_lowercase();
        (t.contains("regex") || t.contains("regular expression"))
            && (t.contains("match") || t.contains("accept"))
    }
}

impl DomainPlugin for FormalLanguageDomainPlugin {
    fn domain_name(&self) -> &str {
        "formal_language"
    }

    fn extract_entities(&self, _text: &str) -> Vec<Entity> {
        Vec::new()
    }

    fn is_in_domain(&self, topic: &str) -> f64 {
        if Self::has_cue(topic) { 0.9 } else { 0.1 }
    }

    fn vocabulary(&self) -> Vec<String> {
        [
            "regex",
            "regular",
            "expression",
            "match",
            "automaton",
            "pattern",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn compute(&self, input: &str, _entities: &[Entity]) -> Option<ComputedResult> {
        if !Self::has_cue(input) {
            return None;
        }
        // Expect two delimited segments: the pattern, then the test string.
        let parts = delimited(input);
        if parts.len() < 2 {
            return None;
        }
        let (pattern, string) = (&parts[0], &parts[1]);
        match Regex::new(pattern) {
            Ok(re) => {
                let matched = re.matches(string);
                Some(result(format!(
                    "The regular expression `{pattern}` {} `{string}` (anchored / \
                     whole-string match).",
                    if matched { "MATCHES" } else { "does NOT match" }
                )))
            }
            Err(e) => Some(result(format!(
                "The pattern `{pattern}` is not a valid regular expression: {e}."
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pattern() {
        let p = FormalLanguageDomainPlugin;
        let r = p
            .compute("does the regex `a(b|c)*d` match `abcbcd`?", &[])
            .unwrap();
        assert!(r.answer.contains("MATCHES"), "{}", r.answer);
    }

    #[test]
    fn rejects_non_match() {
        let p = FormalLanguageDomainPlugin;
        // anchored: "a+" does not match "b", and "cat" does not match "cats"
        let r = p.compute("does regex 'cat' match 'cats'?", &[]).unwrap();
        assert!(r.answer.contains("does NOT match"), "{}", r.answer);
    }

    #[test]
    fn reports_invalid_pattern() {
        let p = FormalLanguageDomainPlugin;
        let r = p.compute("does the regex `(a` match `a`?", &[]).unwrap();
        assert!(r.answer.contains("not a valid"), "{}", r.answer);
    }

    #[test]
    fn needs_two_delimited_parts() {
        let p = FormalLanguageDomainPlugin;
        assert!(p.compute("does a regex match things?", &[]).is_none());
    }
}
