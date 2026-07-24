// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! A small, hand-curated, real, cited reaction corpus for the Reaction
//! Corpus Auditor (`audit.rs`).
//!
//! **Deliberately not ORD-derived.** The Open Reaction Database's actual
//! format is Protocol-Buffers-based and requires real tooling (a Git LFS
//! clone of a separate `ord-data` repo, or protobuf extraction) to pull even
//! a handful of records -- confirmed by direct investigation, not assumed.
//! ORD records also genuinely carry reaction conditions, quantities, and
//! procedural/analytical content that must never reach a `ProcessCertificate`
//! (an explicit non-goal of this project). A small, real, correctly-sourced,
//! git-committed fixture -- same pattern as
//! `symthaea-organic-chemistry/examples/phase0_audit.rs`'s 10 real
//! feedstocks -- delivers the same audit value for a first pass without
//! either dependency.
//!
//! **Phase A.2 stratification** (added 2026-07-13, per external review):
//! every record is tagged with a [`RecordCategory`] and an
//! [`ExpectedOutcomeKind`] -- what SHOULD happen when this record runs
//! through the pipeline, decided at corpus-authoring time, independent of
//! whatever the pipeline actually does. `metrics.rs` compares the two and
//! reports per-category pass rates. This is what makes "does the auditor
//! behave correctly" a measurable question instead of an eyeballed one.
//! **Honest scope note**: category depth is uneven by design, not
//! oversight -- `Supported`/`UnsupportedButValid` are backed by many real
//! reactions; `Malformed`/`IncorrectDeclaredProduct`/`AlternateRepresentation`/
//! `Duplicate`/`AmbiguousOrPartial`/`Adversarial` each have 1-2 records,
//! enough to exercise the category's logic but not a statistically
//! meaningful sample. Widening any one category is legitimate future work,
//! not a gap this pass claims to have closed.

use symthaea_organic_chemistry::smiles::Molecule;

/// What KIND of real-world/testing situation this record represents --
/// independent of what the pipeline happens to do with it. Used by
/// `metrics.rs` to report per-category behavior, not just an aggregate
/// pass/fail count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecordCategory {
    /// A real reaction both templates and the current policy set should
    /// handle correctly end to end.
    Supported,
    /// A real, valid reaction that no current template covers -- the
    /// pipeline's correct behavior is to abstain (`Unclassified`), not to
    /// guess.
    UnsupportedButValid,
    /// Structurally invalid input (disallowed element, broken valence,
    /// etc.) -- the pipeline's correct behavior is to reject.
    Malformed,
    /// The same real chemistry, written a different valid way than
    /// elsewhere in the corpus (Kekule vs. aromatic, charge-separated vs.
    /// normalized-shorthand) -- tests that representation choice doesn't
    /// change the outcome.
    AlternateRepresentation,
    /// A real reaction whose DECLARED product is wrong (simulating a
    /// transcription error) -- the pipeline's correct behavior is to catch
    /// the declared-vs-computed mismatch, never silently certify it.
    IncorrectDeclaredProduct,
    /// A byte-identical repeat of another record in this corpus -- tests
    /// that auditing the same input twice behaves consistently (same
    /// classification, same certificate content) rather than the pipeline
    /// having any hidden first-seen/already-seen state.
    Duplicate,
    /// Reactant count or shape that neither template can unambiguously
    /// interpret (too few, too many, or an unrelated bystander reactant) --
    /// the pipeline's correct behavior is to abstain, not to guess which
    /// reactants "really" matter.
    AmbiguousOrPartial,
    /// A real reaction constructed specifically to reach the scope/hazard
    /// gate with a structural risk signal in a non-obvious position
    /// (embedded in a longer chain, on the far side of an unrelated
    /// functional group) -- tests that the hazard heuristics aren't merely
    /// passing because nothing risky was ever generated.
    Adversarial,
}

impl RecordCategory {
    /// Stable, fixed iteration order for `metrics.rs`'s per-category
    /// report -- avoids needing `Ord`/`Hash`-keyed map storage just to
    /// enumerate the (small, fixed) category set.
    pub const ALL: [RecordCategory; 8] = [
        RecordCategory::Supported,
        RecordCategory::UnsupportedButValid,
        RecordCategory::Malformed,
        RecordCategory::AlternateRepresentation,
        RecordCategory::IncorrectDeclaredProduct,
        RecordCategory::Duplicate,
        RecordCategory::AmbiguousOrPartial,
        RecordCategory::Adversarial,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            RecordCategory::Supported => "supported",
            RecordCategory::UnsupportedButValid => "unsupported-but-valid",
            RecordCategory::Malformed => "malformed",
            RecordCategory::AlternateRepresentation => "alternate-representation",
            RecordCategory::IncorrectDeclaredProduct => "incorrect-declared-product",
            RecordCategory::Duplicate => "duplicate",
            RecordCategory::AmbiguousOrPartial => "ambiguous-or-partial",
            RecordCategory::Adversarial => "adversarial",
        }
    }
}

/// What SHOULD happen when this record runs through `audit::audit_record`,
/// decided at corpus-authoring time. Deliberately coarser than
/// `audit::RecordOutcome` (which distinguishes e.g. which template matched)
/// -- this only needs to capture the *kind* of outcome for metrics
/// purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpectedOutcomeKind {
    Certified,
    Unclassified,
    DeclaredMismatch,
    MatchedButScopeRejected,
}

pub struct CorpusRecord {
    pub name: &'static str,
    pub source: &'static str,
    pub category: RecordCategory,
    pub expected_outcome: ExpectedOutcomeKind,
    /// Whether `audit::check_raw_validity` (parse + normalize + structural
    /// checks) should succeed on this record's reactants/products,
    /// independent of classification -- lets `Malformed` be distinguished
    /// from "well-formed but merely unsupported" even though both can share
    /// `ExpectedOutcomeKind::Unclassified`.
    pub expected_raw_validity_ok: bool,
    pub reactant_smiles: &'static [&'static str],
    /// Declared product SMILES, in the same order this crate's templates
    /// would produce them (ester first then water, for esterification) --
    /// order-sensitive by construction, matching how `templates.rs` already
    /// returns products.
    pub product_smiles: &'static [&'static str],
}

impl CorpusRecord {
    pub fn parse_reactants(&self) -> Result<Vec<Molecule>, String> {
        self.reactant_smiles
            .iter()
            .map(|s| Molecule::from_smiles(s).map_err(|e| format!("{s}: {e}")))
            .collect()
    }

    pub fn parse_products(&self) -> Result<Vec<Molecule>, String> {
        self.product_smiles
            .iter()
            .map(|s| Molecule::from_smiles(s).map_err(|e| format!("{s}: {e}")))
            .collect()
    }
}

pub fn phase_a_fixture_corpus() -> Vec<CorpusRecord> {
    use ExpectedOutcomeKind::*;
    use RecordCategory::*;
    vec![
        // --- Supported: real Fischer esterifications, matched by EsterificationTemplate ---
        CorpusRecord {
            name: "acetic acid + ethanol -> ethyl acetate",
            source: "classic Fischer esterification (vinegar + ethanol)",
            category: Supported,
            expected_outcome: Certified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["CC(=O)O", "CCO"],
            product_smiles: &["CC(=O)OCC", "O"],
        },
        CorpusRecord {
            name: "acetic acid + methanol -> methyl acetate",
            source: "Fischer esterification",
            category: Supported,
            expected_outcome: Certified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["CC(=O)O", "CO"],
            product_smiles: &["CC(=O)OC", "O"],
        },
        CorpusRecord {
            name: "propanoic acid + ethanol -> ethyl propanoate",
            source: "Fischer esterification",
            category: Supported,
            expected_outcome: Certified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["CCC(=O)O", "CCO"],
            product_smiles: &["CCC(=O)OCC", "O"],
        },
        CorpusRecord {
            name: "propanoic acid + methanol -> methyl propanoate",
            source: "Fischer esterification",
            category: Supported,
            expected_outcome: Certified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["CCC(=O)O", "CO"],
            product_smiles: &["CCC(=O)OC", "O"],
        },
        CorpusRecord {
            name: "benzoic acid + methanol -> methyl benzoate",
            source: "Fischer esterification, aromatic acid",
            category: Supported,
            expected_outcome: Certified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["c1ccccc1C(=O)O", "CO"],
            product_smiles: &["c1ccccc1C(=O)OC", "O"],
        },
        CorpusRecord {
            name: "benzoic acid + ethanol -> ethyl benzoate",
            source: "Fischer esterification, aromatic acid",
            category: Supported,
            expected_outcome: Certified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["c1ccccc1C(=O)O", "CCO"],
            product_smiles: &["c1ccccc1C(=O)OCC", "O"],
        },
        CorpusRecord {
            name: "formic acid + methanol -> methyl formate",
            source: "Fischer esterification, smallest carboxylic acid",
            category: Supported,
            expected_outcome: Certified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["OC=O", "CO"],
            product_smiles: &["COC=O", "O"],
        },
        CorpusRecord {
            name: "butanoic acid + ethanol -> ethyl butanoate",
            source: "Fischer esterification (\"pineapple\" ester flavor compound)",
            category: Supported,
            expected_outcome: Certified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["CCCC(=O)O", "CCO"],
            product_smiles: &["CCCC(=O)OCC", "O"],
        },
        // --- Supported: real amidations, matched by AmidationTemplate (Phase A.4) ---
        CorpusRecord {
            name: "acetic acid + methylamine -> N-methylacetamide",
            source: "textbook amide coupling",
            category: Supported,
            expected_outcome: Certified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["CC(=O)O", "CN"],
            product_smiles: &["CC(=O)NC", "O"],
        },
        CorpusRecord {
            name: "benzoic acid + ethylamine -> N-ethylbenzamide",
            source: "textbook amide coupling, aromatic acid",
            category: Supported,
            expected_outcome: Certified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["c1ccccc1C(=O)O", "CCN"],
            product_smiles: &["c1ccccc1C(=O)NCC", "O"],
        },
        CorpusRecord {
            name: "acetic acid + ethanolamine -> N-(2-hydroxyethyl)acetamide (amine beats a coexisting alcohol)",
            source: "amide coupling on a reactant with BOTH a free amine and a free alcohol -- \
                      exercises the real Phase A.3/A.4 selectivity finding (17/20, 85%, of \
                      esterification-kind wrong-transformation records had this exact shape) \
                      end to end through the full auditor, not just AmidationTemplate's own unit \
                      tests. The amine reacts; the alcohol survives untouched.",
            category: Supported,
            expected_outcome: Certified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["CC(=O)O", "NCCO"],
            product_smiles: &["CC(=O)NCCO", "O"],
        },
        // --- Supported: real C-C hydrogenations, matched by HydrogenationTemplate ---
        CorpusRecord {
            name: "ethylene + H2 -> ethane",
            source: "textbook catalytic hydrogenation",
            category: Supported,
            expected_outcome: Certified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["C=C", "[H][H]"],
            product_smiles: &["CC"],
        },
        CorpusRecord {
            name: "propylene + H2 -> propane",
            source: "textbook catalytic hydrogenation",
            category: Supported,
            expected_outcome: Certified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["CC=C", "[H][H]"],
            product_smiles: &["CCC"],
        },
        CorpusRecord {
            name: "1-butene + H2 -> butane",
            source: "catalytic hydrogenation",
            category: Supported,
            expected_outcome: Certified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["C=CCC", "[H][H]"],
            product_smiles: &["CCCC"],
        },
        CorpusRecord {
            name: "2-butene + H2 -> butane",
            source: "catalytic hydrogenation; same product as 1-butene from a different alkene",
            category: Supported,
            expected_outcome: Certified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["CC=CC", "[H][H]"],
            product_smiles: &["CCCC"],
        },
        CorpusRecord {
            name: "cyclohexene + H2 -> cyclohexane",
            source: "catalytic hydrogenation, cyclic alkene",
            category: Supported,
            expected_outcome: Certified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["C1=CCCCC1", "[H][H]"],
            product_smiles: &["C1CCCCC1"],
        },
        CorpusRecord {
            name: "styrene + H2 -> ethylbenzene",
            source: "catalytic hydrogenation, vinyl arene (ring stays aromatic, only the vinyl C=C reduces)",
            category: Supported,
            expected_outcome: Certified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["c1ccccc1C=C", "[H][H]"],
            product_smiles: &["c1ccccc1CC"],
        },
        CorpusRecord {
            name: "acrylonitrile + H2 -> propionitrile",
            source: "industrial hydrogenation step toward adiponitrile/nylon-6,6 production",
            category: Supported,
            expected_outcome: Certified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["C=CC#N", "[H][H]"],
            product_smiles: &["CCC#N"],
        },
        CorpusRecord {
            name: "1-hexene + H2 -> hexane",
            source: "catalytic hydrogenation",
            category: Supported,
            expected_outcome: Certified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["C=CCCCC", "[H][H]"],
            product_smiles: &["CCCCCC"],
        },
        // --- Duplicate: byte-identical repeats of two Supported records above ---
        CorpusRecord {
            name: "acetic acid + ethanol -> ethyl acetate (duplicate entry)",
            source: "same record as above, deliberately repeated to test consistent handling of duplicate input",
            category: Duplicate,
            expected_outcome: Certified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["CC(=O)O", "CCO"],
            product_smiles: &["CC(=O)OCC", "O"],
        },
        CorpusRecord {
            name: "ethylene + H2 -> ethane (duplicate entry)",
            source: "same record as above, deliberately repeated to test consistent handling of duplicate input",
            category: Duplicate,
            expected_outcome: Certified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["C=C", "[H][H]"],
            product_smiles: &["CC"],
        },
        // --- AlternateRepresentation: same real chemistry, different valid encoding ---
        CorpusRecord {
            name: "benzoic acid (Kekule form) + methanol -> methyl benzoate",
            source: "same reaction as the aromatic-lowercase benzoic acid record above, written with explicit \
                      alternating Kekule double bonds instead of aromatic lowercase atoms -- same molecule, same \
                      formula, different valid SMILES encoding of the ring",
            category: AlternateRepresentation,
            expected_outcome: Certified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["C1=CC=CC=C1C(=O)O", "CO"],
            product_smiles: &["C1=CC=CC=C1C(=O)OC", "O"],
        },
        CorpusRecord {
            name: "nitrobenzene (already charge-separated) + H2 -> aniline",
            source: "same reaction as the neutral-shorthand nitrobenzene record below, written already in the \
                      formally correct charge-separated form -- confirms the already-correct encoding needs no \
                      normalization and still abstains the same way (no template covers nitro reduction)",
            category: AlternateRepresentation,
            expected_outcome: Unclassified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["c1ccccc1[N+](=O)[O-]", "[H][H]"],
            product_smiles: &["c1ccccc1N"],
        },
        // --- AmbiguousOrPartial: reactant shape neither template can unambiguously interpret ---
        CorpusRecord {
            name: "acetic acid alone (missing alcohol partner)",
            source: "deliberately incomplete record -- only one reactant given for a transformation that needs \
                      two; neither template's exact-arity pattern match can interpret this, so the correct \
                      behavior is to abstain, not guess what the missing partner might have been",
            category: AmbiguousOrPartial,
            expected_outcome: Unclassified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["CC(=O)O"],
            product_smiles: &["CC(=O)O"],
        },
        CorpusRecord {
            name: "ethylene + H2 + unexplained water bystander",
            source: "deliberately over-specified record -- a third reactant (water) with no defined role is \
                      present alongside a real hydrogenation pair; neither template's exact-arity pattern match \
                      accepts a 3-reactant input, so the correct behavior is to abstain rather than silently \
                      ignore the bystander",
            category: AmbiguousOrPartial,
            expected_outcome: Unclassified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["C=C", "[H][H]", "O"],
            product_smiles: &["CC"],
        },
        // --- Adversarial: real reactions constructed to reach the scope gate with a hazard
        // motif in a non-obvious position (not on the reactant/product a naive reviewer would
        // look at first) ---
        CorpusRecord {
            name: "acetic acid + 2-(ethylperoxy)ethanol -> ester (peroxide-tainted alcohol partner)",
            source: "real esterification shape, but the alcohol partner carries a peroxide linkage buried after \
                      the reactive hydroxyl -- tests that the hazard screen catches a real risk signal embedded \
                      in an otherwise unremarkable-looking alcohol, not just an isolated peroxide molecule",
            category: Adversarial,
            expected_outcome: MatchedButScopeRejected,
            expected_raw_validity_ok: true,
            reactant_smiles: &["CC(=O)O", "OCCOOCC"],
            product_smiles: &["CC(=O)OCCOOCC", "O"],
        },
        CorpusRecord {
            name: "acetic acid + 2-nitroethanol -> ester (nitro-tainted alcohol partner)",
            source: "real esterification shape, but the alcohol partner carries a neutrally-drawn nitro group -- \
                      tests both normalization (fires on the reactant AND the resulting ester product) and hazard \
                      screening together, on a molecule the reviewer's eye would first read as \"just an alcohol\"",
            category: Adversarial,
            expected_outcome: MatchedButScopeRejected,
            expected_raw_validity_ok: true,
            reactant_smiles: &["CC(=O)O", "OCCN(=O)=O"],
            product_smiles: &["CC(=O)OCCN(=O)=O", "O"],
        },
        CorpusRecord {
            name: "methane + Cl2 -> methyl chloride + HCl",
            source: "real radical halogenation; Cl is now in this pipeline's allowed element set \
                      (Phase A.7), so this is structurally valid and passes raw validity, but no \
                      template in this crate models a radical-substitution mechanism -- previously \
                      this fixture tested element-scope rejection (H/C/N/O/F only), now it tests the \
                      same 'unsupported reaction type' pattern as the acetone/nitrobenzene records \
                      below, on a real 2-reactant/2-product halogenation instead of a hydrogenation",
            category: UnsupportedButValid,
            expected_outcome: Unclassified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["C", "ClCl"],
            product_smiles: &["CCl", "Cl"],
        },
        CorpusRecord {
            name: "methyl iodide formation (disallowed element)",
            source: "iodine is in organic-chemistry's SMILES subset but outside this pipeline's \
                      allowed element set (Phase A.7: H/C/N/O/F/Si/P/S/Cl/Br -- iodine has no \
                      STO-3G basis data in symthaea-quantum-chemistry and is rare enough in this \
                      corpus's reaction classes not to warrant adding it). Replaces the old \
                      methane+Cl2 record as this corpus's Malformed/element-scope example now \
                      that Cl is in scope.",
            category: Malformed,
            expected_outcome: Unclassified,
            expected_raw_validity_ok: false,
            reactant_smiles: &["C", "II"],
            product_smiles: &["CI", "I"],
        },
        CorpusRecord {
            name: "acetone + H2 -> isopropanol",
            source: "real industrial carbonyl hydrogenation -- NOT matched by this crate's C-C-restricted HydrogenationTemplate (the reducible bond here is C=O, not C=C)",
            category: UnsupportedButValid,
            expected_outcome: Unclassified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["CC(=O)C", "[H][H]"],
            product_smiles: &["CC(O)C"],
        },
        CorpusRecord {
            name: "nitrobenzene + H2 -> aniline",
            source: "real industrial nitro-group reduction (simplified 1:1 stoichiometry shorthand -- the real equation is C6H5NO2 + 3 H2 -> C6H5NH2 + 2 H2O; this record deliberately keeps the common shorthand form, not the balanced equation, since real informal reaction records often do the same). Matches no template (aromatic ring bonds aren't targeted, and reducing N=O isn't a supported transformation at all); its reactant is hazard-flagged (nitro group) but that never runs since classification abstains first. Neutral nitro shorthand -- normalizes cleanly (Phase A.1), raw validity now passes.",
            category: UnsupportedButValid,
            expected_outcome: Unclassified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["c1ccccc1N(=O)=O", "[H][H]"],
            product_smiles: &["c1ccccc1N"],
        },
        CorpusRecord {
            name: "acetic acid + ethanol -> propyl acetate (WRONG declared product)",
            source: "deliberately incorrect: the real product of acetic acid + ethanol is ethyl acetate (C4H8O2), not propyl acetate (C5H10O2) -- simulates a transcription/data-entry error the auditor should catch as a declared-vs-computed mismatch, not silently accept",
            category: IncorrectDeclaredProduct,
            expected_outcome: DeclaredMismatch,
            expected_raw_validity_ok: true,
            reactant_smiles: &["CC(=O)O", "CCO"],
            product_smiles: &["CC(=O)OCCC", "O"],
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_fixture_reactant_and_product_smiles_parses() {
        // The corpus itself must be well-formed SMILES, even for the
        // deliberately-edge-case records (their edge-case-ness is in
        // classification/conservation/policy outcomes, not in being
        // unparseable garbage).
        for record in phase_a_fixture_corpus() {
            assert!(
                record.parse_reactants().is_ok(),
                "{}: reactant SMILES failed to parse: {:?}",
                record.name,
                record.parse_reactants()
            );
            assert!(
                record.parse_products().is_ok(),
                "{}: product SMILES failed to parse: {:?}",
                record.name,
                record.parse_products()
            );
        }
    }

    #[test]
    fn corpus_has_meaningful_diversity_not_just_passing_cases() {
        // A corpus that's all "designed to pass" would make for a
        // meaningless audit. Sanity-check the deliberate edge cases exist.
        let corpus = phase_a_fixture_corpus();
        assert!(corpus.len() >= 24);
        assert!(corpus.iter().any(|r| r.name.contains("Cl2")));
        assert!(corpus.iter().any(|r| r.name.contains("isopropanol")));
        assert!(corpus.iter().any(|r| r.name.contains("aniline")));
        assert!(corpus.iter().any(|r| r.name.contains("WRONG")));
    }

    #[test]
    fn every_category_has_at_least_one_record() {
        let corpus = phase_a_fixture_corpus();
        for category in RecordCategory::ALL {
            assert!(
                corpus.iter().any(|r| r.category == category),
                "no corpus record tagged {category:?}"
            );
        }
    }

    #[test]
    fn duplicate_records_are_genuinely_byte_identical_to_their_original() {
        let corpus = phase_a_fixture_corpus();
        let original = corpus
            .iter()
            .find(|r| r.name == "acetic acid + ethanol -> ethyl acetate")
            .unwrap();
        let duplicate = corpus
            .iter()
            .find(|r| r.name == "acetic acid + ethanol -> ethyl acetate (duplicate entry)")
            .unwrap();
        assert_eq!(original.reactant_smiles, duplicate.reactant_smiles);
        assert_eq!(original.product_smiles, duplicate.product_smiles);
    }
}
