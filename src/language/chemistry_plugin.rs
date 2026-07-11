// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Chemistry domain plugin.
//!
//! Wires `symthaea-organic-chemistry` into the facade's domain-plugin path so
//! Symthaea answers structural-chemistry questions **deterministically** — it
//! parses a SMILES string and computes molecular formula, weight, functional
//! groups, ring count, and Lipinski drug-likeness in Rust, bypassing LLM
//! generation (via `DomainPlugin::compute`). This turns "we have a chemistry
//! library" into "the facade understands chemistry."

use crate::language::domain_plugin::{ComputedResult, DomainPlugin, Entity};
use crate::mind::structured_thought::{ETier, EpistemicCube, MTier, NTier};
use symthaea_organic_chemistry::{Molecule, detect, lipinski};

/// Domain plugin for structural organic chemistry.
pub struct ChemistryDomainPlugin;

/// Keywords that signal a chemistry query. A SMILES token is only extracted when
/// one is present, so ordinary prose is not mis-read as a molecule.
const CUES: &[&str] = &[
    "smiles",
    "molecul",
    "molar mass",
    "chemical formula",
    "formula",
    "functional group",
    "drug-like",
    "druglike",
    "lipinski",
    "compound",
    "chemistry",
    "ring",
];

impl ChemistryDomainPlugin {
    fn has_cue(text: &str) -> bool {
        let t = text.to_lowercase();
        CUES.iter().any(|c| t.contains(c))
    }

    /// A token "looks like" SMILES if it carries a chemistry-plausible
    /// character — an uppercase organic-subset letter, a ring digit, or a bond/
    /// branch symbol — which filters out lowercase English words.
    fn looks_like_smiles(tok: &str) -> bool {
        tok.chars().any(|c| {
            matches!(c, 'B' | 'C' | 'N' | 'O' | 'P' | 'S' | 'F' | 'I')
                || c.is_ascii_digit()
                || matches!(c, '=' | '#' | '(' | '[')
        })
    }

    /// Find SMILES tokens in text (cue-gated). Returns `(token, start, end)`.
    fn extract_smiles(text: &str) -> Vec<(String, usize, usize)> {
        if !Self::has_cue(text) {
            return Vec::new();
        }
        let mut out = Vec::new();
        for raw in text.split_whitespace() {
            // Strip sentence punctuation but keep SMILES-significant () [].
            let tok = raw.trim_matches(|c: char| ".,!?;:\"'".contains(c));
            if tok.len() < 2 || !Self::looks_like_smiles(tok) {
                continue;
            }
            if let Ok(m) = Molecule::from_smiles(tok) {
                if m.atoms.len() >= 2 {
                    let start = text.find(tok).unwrap_or(0);
                    out.push((tok.to_string(), start, start + tok.len()));
                }
            }
        }
        out
    }

    /// Epistemic cube for a computed chemistry fact: deterministic and
    /// well-grounded (standard atomic weights), mirroring the math plugin.
    fn chemistry_cube() -> EpistemicCube {
        EpistemicCube {
            e: ETier::E4,
            n: NTier::N3,
            m: MTier::M3,
            h: None,
        }
    }
}

impl DomainPlugin for ChemistryDomainPlugin {
    fn domain_name(&self) -> &str {
        "chemistry"
    }

    fn extract_entities(&self, text: &str) -> Vec<Entity> {
        Self::extract_smiles(text)
            .into_iter()
            .map(|(v, s, e)| Entity::new("smiles", v, s, e))
            .collect()
    }

    fn is_in_domain(&self, topic: &str) -> f64 {
        if Self::has_cue(topic) { 0.9 } else { 0.1 }
    }

    fn vocabulary(&self) -> Vec<String> {
        [
            "molecule",
            "molecular",
            "smiles",
            "formula",
            "weight",
            "functional",
            "group",
            "hydroxyl",
            "carboxyl",
            "aromatic",
            "ring",
            "lipinski",
            "compound",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn compute(&self, input: &str, entities: &[Entity]) -> Option<ComputedResult> {
        let smiles = entities.iter().find(|e| e.entity_type == "smiles")?;
        let mol = Molecule::from_smiles(&smiles.value).ok()?;
        let s = &smiles.value;
        let lc = input.to_lowercase();

        let answer = if lc.contains("weight") || lc.contains("mass") {
            format!(
                "{s} has molecular formula {} and molecular weight {:.3} g/mol.",
                mol.molecular_formula(),
                mol.molecular_weight()
            )
        } else if lc.contains("formula") {
            format!("{s} has molecular formula {}.", mol.molecular_formula())
        } else if lc.contains("group") {
            format!("{s} contains functional groups: {:?}.", detect(&mol))
        } else if lc.contains("drug") || lc.contains("lipinski") {
            let l = lipinski(&mol);
            format!(
                "{s} — Lipinski: MW {:.1}, {} H-bond donor(s), {} acceptor(s), \
                 {} rule violation(s) → {}.",
                l.molecular_weight,
                l.hbond_donors,
                l.hbond_acceptors,
                l.violations,
                if l.drug_like {
                    "drug-like"
                } else {
                    "not drug-like"
                }
            )
        } else {
            format!(
                "{s}: formula {}, MW {:.3} g/mol, {} ring(s), groups {:?}.",
                mol.molecular_formula(),
                mol.molecular_weight(),
                mol.ring_count(),
                detect(&mol)
            )
        };

        Some(ComputedResult {
            answer,
            cube: Self::chemistry_cube(),
            psi: 0.0,
            proof_available: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_smiles_only_with_a_cue() {
        let p = ChemistryDomainPlugin;
        // Cue present → extracts CCO.
        let ents = p.extract_entities("what is the molecular weight of CCO?");
        assert_eq!(ents.len(), 1);
        assert_eq!(ents[0].value, "CCO");
        // No cue → no extraction (avoids reading prose as molecules).
        assert!(
            p.extract_entities("we should go to the CCO office")
                .is_empty()
        );
    }

    #[test]
    fn computes_molecular_weight() {
        let p = ChemistryDomainPlugin;
        let ents = p.extract_entities("molecular weight of CCO");
        let r = p.compute("molecular weight of CCO", &ents).unwrap();
        assert!(r.answer.contains("C2H6O"));
        assert!(r.answer.contains("46.0"));
        assert!(!r.proof_available);
    }

    #[test]
    fn computes_lipinski() {
        let p = ChemistryDomainPlugin;
        let ents = p.extract_entities("is the compound CC(=O)Oc1ccccc1C(=O)O drug-like?");
        let r = p
            .compute("is CC(=O)Oc1ccccc1C(=O)O drug-like?", &ents)
            .unwrap();
        assert!(r.answer.contains("drug-like"));
    }

    #[test]
    fn no_smiles_no_computation() {
        let p = ChemistryDomainPlugin;
        assert!(p.compute("tell me about chemistry", &[]).is_none());
    }

    #[test]
    fn in_domain_scoring() {
        let p = ChemistryDomainPlugin;
        assert!(p.is_in_domain("molecular formula of caffeine") > 0.5);
        assert!(p.is_in_domain("what time is it") < 0.5);
    }
}
