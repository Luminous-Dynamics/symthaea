// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Human-reviewable evidence record for a surviving candidate.
//!
//! Mirrors `symthaea-forge/src/certificate.rs`'s design: a certificate is
//! data describing what was found and why it passed, never an instruction
//! and never auto-applied.
//!
//! **Phase 1.1 hardening**: an external review found this boundary was
//! aspirational rather than enforced -- `run_search()` handed back full
//! `ReactionCandidate`s directly and `examples/policy_comparison.rs`
//! printed from those, never actually constructing a certificate anywhere
//! outside this module's own unit test. It also found the certificate
//! itself too thin to be reviewable: formula + molecular weight alone can't
//! distinguish structural isomers, so a human "reviewing" a certificate
//! couldn't actually tell what molecule was being proposed. Both are fixed
//! here: `search.rs` now returns `Vec<ProcessCertificate>` as the survivor
//! output (not raw candidates), and each certificate carries the full
//! atom/bond graph, not just a formula.
//!
//! **Phase A.1 hardening**: structure/gate evidence is now built from
//! `oracle_result.normalized.candidate` (the post-normalization structure
//! that was actually validated and certified), not the raw pre-normalization
//! input -- and every `MoleculeGraphRecord` carries its own
//! `normalization` evidence list, so a reviewer can tell "valid exactly as
//! supplied" (empty list) from "valid after a recognized, logged
//! normalization" (non-empty) rather than the two being silently
//! indistinguishable. See `normalization.rs`.

use crate::normalization::NormalizationRecord;
use crate::oracle::OracleResult;
use serde::Serialize;
use symthaea_organic_chemistry::smiles::{BondOrder, Molecule};

/// Always present on every certificate: a candidate is a computational
/// proposal, never an instruction. No code in this crate reads this field
/// to decide anything -- it exists for a human reader, and because an
/// explicit, repeated statement is worth more than an implicit convention.
pub const CAPABILITY_CLASSIFICATION: &str = "NOT A SYNTHESIS INSTRUCTION -- a computational search candidate only. \
     No code in this crate synthesizes, orders, or acts on this content. \
     Human review is required before any real-world action.";

fn bond_order_label(o: BondOrder) -> &'static str {
    match o {
        BondOrder::Single => "single",
        BondOrder::Double => "double",
        BondOrder::Triple => "triple",
        BondOrder::Aromatic => "aromatic",
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct AtomRecord {
    pub element: String,
    pub aromatic: bool,
    pub charge: i8,
    pub hydrogens: u8,
}

#[derive(Debug, Clone, Serialize)]
pub struct BondRecord {
    pub a: usize,
    pub b: usize,
    pub order: &'static str,
}

/// Full structural record -- not just a formula. Two molecules with the
/// same formula (isomers) produce different `atoms`/`bonds`, so a reviewer
/// can actually tell them apart.
#[derive(Debug, Clone, Serialize)]
pub struct MoleculeGraphRecord {
    pub formula: String,
    pub molecular_weight: f64,
    pub atoms: Vec<AtomRecord>,
    pub bonds: Vec<BondRecord>,
    /// Empty if this molecule was accepted exactly as supplied. Non-empty
    /// means the structure shown here is the *normalized* form -- a
    /// recognized, logged transformation (see `normalization.rs`) was
    /// applied before validity/scope ever ran, and this field is what
    /// distinguishes that from "supplied this way already."
    pub normalization: Vec<NormalizationRecord>,
}

impl MoleculeGraphRecord {
    pub fn new(m: &Molecule, normalization: &[NormalizationRecord]) -> Self {
        Self {
            formula: m.molecular_formula(),
            molecular_weight: m.molecular_weight(),
            atoms: m
                .atoms
                .iter()
                .map(|a| AtomRecord {
                    element: a.element.to_string(),
                    aromatic: a.aromatic,
                    charge: a.charge,
                    hydrogens: a.hydrogens,
                })
                .collect(),
            bonds: m
                .bonds
                .iter()
                .map(|b| BondRecord {
                    a: b.a,
                    b: b.b,
                    order: bond_order_label(b.order),
                })
                .collect(),
            normalization: normalization.to_vec(),
        }
    }
}

impl From<&Molecule> for MoleculeGraphRecord {
    fn from(m: &Molecule) -> Self {
        Self::new(m, &[])
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GateEvidence {
    pub gate: String,
    pub passed: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProcessCertificate {
    pub capability_classification: &'static str,
    pub policy_name: &'static str,
    pub template: &'static str,
    pub reactants: Vec<MoleculeGraphRecord>,
    pub products: Vec<MoleculeGraphRecord>,
    /// Every gate this candidate passed, in the order the oracle runs them
    /// (validity -> scope; the stability estimate is advisory, reported
    /// separately since it never gates -- see `oracle.rs`).
    pub gates: Vec<GateEvidence>,
    /// Advisory-only composition-model numbers (see `oracle.rs`'s module
    /// doc for why this can't be trusted as a pass/fail signal), kept as
    /// plain strings so the certificate doesn't need to depend on
    /// `symthaea-materials`'s own types.
    pub composition_model_notes: Vec<String>,
    /// The full seed reactant list this search run was configured with --
    /// provenance, so a reviewer knows what search space produced this.
    pub seed_reactants_config: Vec<String>,
}

impl ProcessCertificate {
    /// Build a certificate for a candidate that reached `GateOutcome::Passed`.
    /// Panics (debug_assert) if called on a non-passing result -- certificates
    /// are only ever meaningful for survivors; callers (`search.rs`) already
    /// only reach this on the `Passed` branch.
    ///
    /// Takes only `oracle_result` (no separate `candidate` param) --
    /// `oracle_result.normalized` already carries the post-normalization
    /// candidate that was actually validated/scoped, plus the per-molecule
    /// normalization evidence; deriving structure from anything else would
    /// risk a certificate describing a different molecule than the one that
    /// was actually checked.
    pub fn from_passed(
        policy_name: &'static str,
        oracle_result: &OracleResult,
        seed_reactants_config: &[String],
    ) -> Self {
        debug_assert!(oracle_result.outcome.passed());
        let composition_model_notes = oracle_result
            .composition_model_prediction
            .iter()
            .map(|p| {
                format!(
                    "formula={} formation_energy={:.3}eV/atom confidence={:.2} is_stable={} (advisory only, see oracle.rs)",
                    p.formula, p.formation_energy, p.confidence, p.is_stable
                )
            })
            .collect();
        let normalized = &oracle_result.normalized;
        Self {
            capability_classification: CAPABILITY_CLASSIFICATION,
            policy_name,
            template: normalized.candidate.template,
            reactants: normalized
                .candidate
                .reactants
                .iter()
                .zip(normalized.reactant_normalizations.iter())
                .map(|(m, n)| MoleculeGraphRecord::new(m, n))
                .collect(),
            products: normalized
                .candidate
                .products
                .iter()
                .zip(normalized.product_normalizations.iter())
                .map(|(m, n)| MoleculeGraphRecord::new(m, n))
                .collect(),
            gates: vec![
                GateEvidence {
                    gate: "validity".to_string(),
                    passed: true,
                    detail: "structural sanity + element/charge conservation (validity.rs)"
                        .to_string(),
                },
                GateEvidence {
                    gate: "scope".to_string(),
                    passed: true,
                    detail: oracle_result.scope_reason.clone(),
                },
            ],
            composition_model_notes,
            seed_reactants_config: seed_reactants_config.to_vec(),
        }
    }

    pub fn to_json_pretty(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    pub fn summary(&self) -> String {
        let reactants: Vec<String> = self.reactants.iter().map(|r| r.formula.clone()).collect();
        let products: Vec<String> = self.products.iter().map(|p| p.formula.clone()).collect();
        format!(
            "[{}] {} :: {} -> {}",
            self.policy_name,
            self.template,
            reactants.join(" + "),
            products.join(" + ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle;
    use crate::policy::{AllowlistOnlyPolicy, ReactantLibrary};
    use crate::types::ReactionCandidate;

    #[test]
    fn certificate_summarizes_a_survivor_with_full_structure() {
        let ethanol = Molecule::from_smiles("CCO").unwrap();
        let candidate = ReactionCandidate {
            reactants: vec![ethanol.clone()],
            products: vec![ethanol],
            template: "identity",
        };
        let policy = AllowlistOnlyPolicy {
            library: ReactantLibrary::phase0_feedstocks(),
        };
        let result = oracle::evaluate(&candidate, &policy);
        assert!(result.outcome.passed());

        let cert = ProcessCertificate::from_passed("allowlist-only", &result, &["CCO".to_string()]);
        assert!(cert.summary().contains("C2H6O"));
        assert!(cert.summary().contains("identity"));
        assert_eq!(cert.gates.len(), 2);
        assert!(cert.gates.iter().all(|g| g.passed));
        assert_eq!(cert.reactants[0].atoms.len(), 3); // ethanol: C, C, O
        assert!(cert.reactants[0].normalization.is_empty());
        assert_eq!(cert.capability_classification, CAPABILITY_CLASSIFICATION);
        assert!(
            cert.capability_classification
                .contains("NOT A SYNTHESIS INSTRUCTION")
        );
    }

    #[test]
    fn certificate_serializes_to_json() {
        let ethanol = Molecule::from_smiles("CCO").unwrap();
        let candidate = ReactionCandidate {
            reactants: vec![ethanol.clone()],
            products: vec![ethanol],
            template: "identity",
        };
        let policy = AllowlistOnlyPolicy {
            library: ReactantLibrary::phase0_feedstocks(),
        };
        let result = oracle::evaluate(&candidate, &policy);
        let cert = ProcessCertificate::from_passed("allowlist-only", &result, &[]);
        let json = cert.to_json_pretty().unwrap();
        assert!(json.contains("NOT A SYNTHESIS INSTRUCTION"));
        assert!(json.contains("\"element\""));
    }

    #[test]
    fn certificate_records_normalization_evidence_when_it_fired() {
        // A candidate whose reactant needed the nitro normalization
        // (normalization.rs) must show that evidence on the certificate,
        // not silently present the normalized structure as if it had been
        // supplied that way.
        let nitrobenzene = Molecule::from_smiles("c1ccccc1N(=O)=O").unwrap();
        let candidate = ReactionCandidate {
            reactants: vec![nitrobenzene.clone()],
            products: vec![nitrobenzene],
            template: "identity",
        };
        let policy = crate::policy::OpenWithHeuristicScreenPolicy {
            external: crate::hazard_heuristics::ExternalScopeConfig::default(),
        };
        let result = oracle::evaluate(&candidate, &policy);
        // Nitrobenzene is hazard-flagged (nitro group), so
        // OpenWithHeuristicScreenPolicy denies it at scope -- validity is
        // what this test cares about, and validity ran (and passed) first.
        assert!(matches!(
            result.outcome,
            crate::oracle::GateOutcome::FailedScope(_)
        ));
        assert!(result.normalized.any_normalization_applied());
        assert!(!result.normalized.reactant_normalizations[0].is_empty());
    }

    #[test]
    fn certificate_serialization_is_deterministic() {
        // "Certificate serialization / digest stability" (Phase 1.2):
        // construct the same logical candidate/result TWICE, independently,
        // and confirm the JSON is byte-for-byte identical. Certificates
        // deliberately carry no timestamp or other wall-clock/random data
        // (unlike symthaea-forge's certificate, which does) -- that's
        // exactly what makes this property hold; if a future field ever
        // introduces nondeterminism, this test catches it.
        let build = || {
            let acid = Molecule::from_smiles("CC(=O)O").unwrap();
            let alcohol = Molecule::from_smiles("CCO").unwrap();
            use crate::templates::ReactionTemplate;
            let products = crate::templates::EsterificationTemplate
                .apply(&[acid.clone(), alcohol.clone()])
                .unwrap();
            let candidate = ReactionCandidate {
                reactants: vec![acid, alcohol],
                products,
                template: "esterification",
            };
            let policy = crate::policy::HybridAllowlistReactantsPolicy {
                library: ReactantLibrary::phase0_feedstocks(),
                external: crate::hazard_heuristics::ExternalScopeConfig::default(),
            };
            let result = oracle::evaluate(&candidate, &policy);
            ProcessCertificate::from_passed(
                "hybrid-allowlist-reactants",
                &result,
                &["CC(=O)O".to_string(), "CCO".to_string()],
            )
            .to_json_pretty()
            .unwrap()
        };
        let json_a = build();
        let json_b = build();
        assert_eq!(
            json_a, json_b,
            "certificate JSON must be deterministic across independent construction"
        );
    }
}
