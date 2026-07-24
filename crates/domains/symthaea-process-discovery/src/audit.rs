// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reaction Corpus Auditor: runs `corpus.rs`'s fixture records through this
//! crate's existing validity/template/policy/certificate pipeline and
//! reports what happened, honestly -- parse failures, disallowed elements,
//! unsupported reaction families, declared-vs-computed product mismatches,
//! and genuinely certified matches all get their own distinct, counted
//! outcome. This is auditing, not generation: nothing here invents a
//! molecule that wasn't already in the corpus record.
//!
//! **Policy choice, deliberately different from `search.rs`'s generator**:
//! uses [`OpenWithHeuristicScreenPolicy`], not the library-restricted
//! policies. The generator's job is "don't invent reactants beyond a
//! curated set"; the auditor's job is "verify structure/conservation/hazard
//! on whatever real reaction it's handed" -- restricting real external
//! reactants to this crate's 10-molecule library would reject almost the
//! entire corpus for the wrong reason (not in our list) rather than for
//! anything actually wrong with the record. The hazard screen still applies
//! (structural signals are about the molecule, not about curation).
//!
//! **PubChem cross-reference is advisory-only, per-distinct-SMILES,
//! deduplicated and throttled** -- see `pubchem.rs`'s module doc for the
//! network-access boundary this introduces.
//!
//! **Source-injected, not a bare on/off switch (Phase A.1)**: `run_audit`
//! takes `Option<&dyn PubChemSource>`, not a `bool`. `None` is a fully
//! offline audit (used by most tests). `Some(&LivePubChemSource)` is the
//! real thing. `Some(&AlwaysUnavailableSource)` proves -- not just asserts
//! -- that a total PubChem outage never changes which candidates get
//! certified, rejected, or left unclassified: see
//! `network_fault_never_changes_local_verdict` below, which runs the whole
//! corpus both ways and diffs every `outcome`/`certificate`.
//! `Some(&cache::ReplaySource)` replays a frozen fixture with zero network
//! access, for deterministic reproduction of a prior live run.
//!
//! **RDKit is a second, independent advisory cross-reference (Phase A.2)**,
//! same source-injection shape as PubChem (`run_audit`'s `rdkit_source:
//! Option<&dyn RdkitSource>`) -- see `rdkit.rs`'s module doc for why it's a
//! subprocess bridge, not a build-time dependency.

use crate::certificate::ProcessCertificate;
use crate::corpus::{CorpusRecord, ExpectedOutcomeKind, RecordCategory};
use crate::normalization;
use crate::oracle::{self, GateOutcome};
use crate::policy::ScopePolicy;
use crate::pubchem::{PubChemQueryOutcome, PubChemSource};
use crate::rdkit::{RdkitQueryOutcome, RdkitSource};
use crate::templates::{
    AmidationTemplate, EsterificationTemplate, ExhaustiveHydrogenationTemplate,
    HydrogenationTemplate, ReactionTemplate, count_reducible_cc_bonds, molecular_hydrogen,
};
use crate::types::ReactionCandidate;
use crate::validity;
use std::collections::HashMap;
use symthaea_organic_chemistry::smiles::Molecule;

/// Whether PubChem's reported formula for a compound agrees with what this
/// crate independently computed from the same SMILES. **Purely
/// informational** -- never read by `oracle.rs`/`policy.rs`/`validity.rs`,
/// and never changes a `RecordOutcome`. Distinguishes "PubChem confirms our
/// parse," "PubChem contradicts our parse" (worth a human look -- could be
/// either side's bug), and "we have no PubChem data to compare against"
/// (network failure vs. a genuine not-found are kept separate too, since
/// they mean different things to a reviewer).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PubChemAgreement {
    Agrees,
    /// The formula strings differ, but parse to the same element-count map
    /// (e.g. `"HCl"` vs. `"ClH"`) -- a presentation-convention difference,
    /// NOT a disagreement about what the molecule is. Tracked separately
    /// from `Disagrees` so an aggregate "N disagreements" count never
    /// implies a chemically meaningful conflict when the only difference
    /// is element ordering. See `formula.rs`'s module doc for the real
    /// finding that motivated this split.
    RepresentationOnlyDifference,
    /// A genuine compositional disagreement: the parsed element-count maps
    /// differ, or either formula string failed to parse at all.
    Disagrees,
    NotFoundInPubChem,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PubChemCrossReference {
    pub smiles: String,
    pub our_formula: String,
    pub outcome: PubChemQueryOutcome,
    pub agreement: PubChemAgreement,
}

fn compute_agreement(our_formula: &str, outcome: &PubChemQueryOutcome) -> PubChemAgreement {
    match outcome {
        PubChemQueryOutcome::Found(record) => {
            match crate::formula::compare_formulas(our_formula, &record.molecular_formula) {
                crate::formula::FormulaComparison::ExactMatch => PubChemAgreement::Agrees,
                crate::formula::FormulaComparison::RepresentationOnlyDifference => {
                    PubChemAgreement::RepresentationOnlyDifference
                }
                crate::formula::FormulaComparison::CompositionDisagreement => {
                    PubChemAgreement::Disagrees
                }
            }
        }
        PubChemQueryOutcome::NotFound => PubChemAgreement::NotFoundInPubChem,
        PubChemQueryOutcome::Unavailable(_) => PubChemAgreement::Unavailable,
    }
}

/// Same role as `PubChemAgreement`, for the RDKit cross-reference
/// (`rdkit.rs`). A second, independent source lets a disagreement here be
/// read alongside (not merged with) a PubChem disagreement -- if both
/// external sources agree with each other but not with this crate, that's
/// a much stronger signal than either alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RdkitAgreement {
    Agrees,
    /// See `PubChemAgreement::RepresentationOnlyDifference` -- same
    /// meaning, same reason for keeping it distinct from `Disagrees`.
    RepresentationOnlyDifference,
    Disagrees,
    RejectedByRdkit,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RdkitCrossReference {
    pub smiles: String,
    pub our_formula: String,
    pub outcome: RdkitQueryOutcome,
    pub agreement: RdkitAgreement,
}

fn compute_rdkit_agreement(our_formula: &str, outcome: &RdkitQueryOutcome) -> RdkitAgreement {
    match outcome {
        RdkitQueryOutcome::Found(record) => {
            match crate::formula::compare_formulas(our_formula, &record.molecular_formula) {
                crate::formula::FormulaComparison::ExactMatch => RdkitAgreement::Agrees,
                crate::formula::FormulaComparison::RepresentationOnlyDifference => {
                    RdkitAgreement::RepresentationOnlyDifference
                }
                crate::formula::FormulaComparison::CompositionDisagreement => {
                    RdkitAgreement::Disagrees
                }
            }
        }
        RdkitQueryOutcome::RejectedByRdkit(_) => RdkitAgreement::RejectedByRdkit,
        RdkitQueryOutcome::Unavailable(_) => RdkitAgreement::Unavailable,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RecordOutcome {
    ParseFailed(String),
    /// Matched a template's shape, but the record's DECLARED product
    /// disagrees with what the template actually computes -- e.g. a
    /// transcription error in the source record.
    DeclaredProductMismatch {
        template: &'static str,
        computed_formulas: Vec<String>,
        declared_formulas: Vec<String>,
    },
    /// Matched a template, computed product agrees with the declared
    /// product, but the oracle's validity gate rejected it (e.g. an
    /// element outside this pipeline's allowed set).
    MatchedButFailedValidity {
        template: &'static str,
        reason: String,
    },
    /// Matched a template, passed validity, but the scope/hazard policy
    /// rejected it.
    MatchedButScopeRejected {
        template: &'static str,
        reason: String,
    },
    /// Matched a template, passed every gate -- a real certificate exists.
    Certified {
        template: &'static str,
    },
    /// Neither template's shape matched this record's reactants at all --
    /// a real reaction this crate has no supported transformation for.
    Unclassified,
}

#[derive(Debug, Clone)]
pub struct RecordAudit {
    pub name: &'static str,
    pub source: &'static str,
    /// Carried through from the `CorpusRecord` (Phase A.2) so
    /// `metrics.rs` can report per-category behavior without needing to
    /// re-zip against the original corpus slice by index.
    pub category: RecordCategory,
    pub outcome: RecordOutcome,
    /// Whether `outcome` matches the corpus record's own
    /// `expected_outcome` -- the core Phase A.2 metric. Computed via
    /// `outcome_matches_expected`, not by the caller, so there's one
    /// authoritative mapping from `RecordOutcome` variants to
    /// `ExpectedOutcomeKind`.
    pub matched_expectation: bool,
    pub certificate: Option<ProcessCertificate>,
    /// Structural validity of every reactant/declared-product molecule,
    /// independent of classification -- lets an `Unclassified` record still
    /// distinguish "genuinely well-formed chemistry we just don't support"
    /// from "malformed or uses a disallowed element."
    pub raw_molecule_validity: Result<(), String>,
    /// Whether `raw_molecule_validity.is_ok()` matches the corpus record's
    /// `expected_raw_validity_ok`.
    pub raw_validity_matched_expectation: bool,
    /// Whether normalization (`normalization.rs`) fired on any reactant or
    /// declared product -- computed from `check_raw_validity`, which runs
    /// for every record regardless of classification, so this is accurate
    /// even for records that never reach a certificate.
    pub normalization_applied: bool,
    /// PubChem cross-reference for each distinct reactant/product SMILES in
    /// this record (looked up once per distinct SMILES across the whole
    /// audit run, not once per record -- see `run_audit`). Empty when
    /// `run_audit` was called with `pubchem_source: None`.
    pub pubchem: Vec<PubChemCrossReference>,
    /// RDKit cross-reference, same shape and same rules as `pubchem` --
    /// empty when `run_audit` was called with `rdkit_source: None`.
    pub rdkit: Vec<RdkitCrossReference>,
}

/// The single authoritative mapping from a concrete `RecordOutcome` to the
/// coarser `ExpectedOutcomeKind` a corpus record declares up front. No
/// `ExpectedOutcomeKind` variant currently exists for `ParseFailed` or
/// `MatchedButFailedValidity` (no corpus record targets either yet) --
/// those always report a mismatch rather than silently matching nothing,
/// so a future record that DOES target one of those states will visibly
/// fail this check until `ExpectedOutcomeKind` is extended to cover it.
fn outcome_matches_expected(outcome: &RecordOutcome, expected: ExpectedOutcomeKind) -> bool {
    matches!(
        (outcome, expected),
        (
            RecordOutcome::Certified { .. },
            ExpectedOutcomeKind::Certified
        ) | (
            RecordOutcome::Unclassified,
            ExpectedOutcomeKind::Unclassified
        ) | (
            RecordOutcome::DeclaredProductMismatch { .. },
            ExpectedOutcomeKind::DeclaredMismatch
        ) | (
            RecordOutcome::MatchedButScopeRejected { .. },
            ExpectedOutcomeKind::MatchedButScopeRejected
        )
    )
}

#[derive(Debug, Clone, Default)]
pub struct AuditSummary {
    pub total_records: usize,
    pub parse_failed: usize,
    pub declared_product_mismatch: usize,
    pub matched_failed_validity: usize,
    pub matched_scope_rejected: usize,
    pub certified: usize,
    pub unclassified: usize,
    /// Counts across every `PubChemCrossReference` in every record --
    /// purely informational, never gates. Zero across the board when
    /// `pubchem_source` was `None`. `pubchem_disagreements` counts only
    /// genuine compositional disagreements, NOT `RepresentationOnlyDifference`
    /// -- keeping the two separate is exactly what prevents this count from
    /// implying a chemically meaningful conflict when the only difference
    /// is a formula-string presentation convention.
    pub pubchem_agreements: usize,
    pub pubchem_representation_only: usize,
    pub pubchem_disagreements: usize,
    pub pubchem_not_found: usize,
    pub pubchem_unavailable: usize,
    /// Same as the `pubchem_*` counters above, for the RDKit
    /// cross-reference. Zero across the board when `rdkit_source` was
    /// `None`.
    pub rdkit_agreements: usize,
    pub rdkit_representation_only: usize,
    pub rdkit_disagreements: usize,
    pub rdkit_rejected: usize,
    pub rdkit_unavailable: usize,
}

#[derive(Debug, Clone)]
pub struct AuditReport {
    pub records: Vec<RecordAudit>,
    pub summary: AuditSummary,
}

/// Returns every template whose shape matches these reactants, each with its
/// own computed products -- NOT a single "first match wins" decision.
///
/// **Phase A.4 redesign, replacing a fixed dispatch priority.** The first
/// version of this function tried Amidation, then Esterification, then
/// Hydrogenation, committing to whichever matched first -- chemically
/// motivated (amines are generally better nucleophiles than alcohols toward
/// electrophilic carbonyls absent protection), and it correctly resolved
/// 8/8 genuine amine-vs-alcohol competition cases on a live re-run. But a
/// full 1,282-record re-evaluation also found it wrong on 12 OTHER records:
/// real synthetic selectivity depends on ring electronics (aniline/
/// heteroarylamine-conjugated nitrogens are poor nucleophiles despite being
/// structurally "free"), conjugation (amidine/hydrazone nitrogens), sterics,
/// and synthesis context that no fixed structural priority can reliably
/// predict -- see `PROCESS_DISCOVERY_PHASE_A4_AMIDATION_REEVALUATION_2026-07-15.md`
/// for the full analysis, including a spot-check proving that narrowing the
/// amine detector isn't a clean fix either (the exact nitrogen shape that
/// was wrong in 6 regressions was right in one of the genuine gains).
///
/// The auditor doesn't need to predict selectivity at all: it has a
/// **declared product** to check against. `audit_record`/`classify_candidate`
/// try every candidate this function returns and accept whichever computed
/// product actually matches the declared one, falling back to reporting a
/// mismatch only if none do. This sidesteps the whole prediction problem by
/// using the ground truth that's already available. (The generator in
/// `search.rs` has no declared product and so keeps emitting every matching
/// template's candidate independently -- unaffected by this change.)
///
/// **Returns each candidate's OWN reactants list, not a shared one (Phase
/// A.5).** `ExhaustiveHydrogenationTemplate` needs a different-length
/// reactants list than the other three templates share (one `h2()` per
/// degree of unsaturation, not always exactly one) -- see that template's
/// doc comment. Detected here whenever the input is hydrogenation-shaped
/// (`[unsaturated, single_h2]`) and the unsaturated reactant has at least
/// one reducible C-C bond; the exhaustive candidate is then built with its
/// own correctly-sized H2 list, independent of what the caller originally
/// supplied.
fn classify_all(reactants: &[Molecule]) -> Vec<(&'static str, Vec<Molecule>, Vec<Molecule>)> {
    let mut out = Vec::new();
    if let Some(products) = AmidationTemplate.apply(reactants) {
        out.push(("amidation", reactants.to_vec(), products));
    }
    if let Some(products) = EsterificationTemplate.apply(reactants) {
        out.push(("esterification", reactants.to_vec(), products));
    }
    if let Some(products) = HydrogenationTemplate.apply(reactants) {
        out.push(("hydrogenation", reactants.to_vec(), products));
    }
    if let [unsaturated, h2] = reactants {
        if h2.molecular_formula() == "H2" {
            let needed = count_reducible_cc_bonds(unsaturated);
            if needed > 0 {
                let mut exhaustive_reactants = vec![unsaturated.clone()];
                exhaustive_reactants.extend((0..needed).map(|_| molecular_hydrogen()));
                if let Some(products) = ExhaustiveHydrogenationTemplate.apply(&exhaustive_reactants)
                {
                    out.push(("exhaustive_hydrogenation", exhaustive_reactants, products));
                }
            }
        }
    }
    out
}

/// True if `computed` and `declared` have the same length and each position
/// is structurally isomorphic (after normalization) to its counterpart --
/// real graph comparison, not formula-string equality (see
/// `audit_record`'s call site for why formula equality is provably
/// insufficient once more than one template can match the same reactants).
/// Positional, not set-based: this crate's templates and corpus fixtures
/// both document/rely on a fixed "product, then water" ordering.
fn products_are_isomorphic(computed: &[Molecule], declared: &[Molecule]) -> bool {
    if computed.len() != declared.len() {
        return false;
    }
    computed.iter().zip(declared.iter()).all(|(c, d)| {
        let (c_norm, _) = normalization::normalize_molecule(c);
        let (d_norm, _) = normalization::normalize_molecule(d);
        crate::isomorphism::is_isomorphic_detailed(&c_norm, &d_norm)
            == crate::isomorphism::IsomorphismOutcome::Isomorphic
    })
}

/// Structural validity of every reactant/declared-product molecule,
/// independent of classification. **Normalizes first** (Phase A.1) -- the
/// same recognized-encoding normalization `oracle::evaluate` applies to
/// every classified candidate applies here too, so an `Unclassified`
/// record's raw validity reflects the same rules a classified record would
/// have been checked against, not a stricter pre-normalization view of the
/// same chemistry. This is what fixes the nitrobenzene fixture: a neutrally
/// drawn nitro group now normalizes to its charge-separated form before
/// this check runs, same as it would inside `oracle::evaluate`.
///
/// Also reports whether ANY normalization fired (second tuple element) --
/// this runs for every record regardless of classification outcome, so
/// it's the one place `metrics.rs` can learn "was normalization applied"
/// even for records that never reach a certificate (e.g. a scope-rejected
/// or unclassified record whose reactant still needed normalizing).
fn check_raw_validity(reactants: &[Molecule], products: &[Molecule]) -> (Result<(), String>, bool) {
    let mut any_normalized = false;
    for m in reactants.iter().chain(products.iter()) {
        let (normalized, records) = normalization::normalize_molecule(m);
        if !records.is_empty() {
            any_normalized = true;
        }
        if let Err(e) = validity::check_molecule(&normalized) {
            return (Err(e), any_normalized);
        }
    }
    (Ok(()), any_normalized)
}

#[allow(clippy::too_many_arguments)]
fn audit_record(
    record: &CorpusRecord,
    policy: &dyn ScopePolicy,
    pubchem_cache: &mut HashMap<String, PubChemQueryOutcome>,
    pubchem_source: Option<&dyn PubChemSource>,
    rdkit_cache: &mut HashMap<String, RdkitQueryOutcome>,
    rdkit_source: Option<&dyn RdkitSource>,
) -> RecordAudit {
    let reactants = match record.parse_reactants() {
        Ok(r) => r,
        Err(e) => {
            return RecordAudit {
                name: record.name,
                source: record.source,
                category: record.category,
                outcome: RecordOutcome::ParseFailed(e),
                matched_expectation: false, // no ExpectedOutcomeKind targets ParseFailed yet
                certificate: None,
                raw_molecule_validity: Err("not parsed".to_string()),
                // No corpus record currently expects a parse failure -- an
                // unparseable SMILES is always a genuine bug, not a
                // designed-in case, so this never counts as "matched."
                raw_validity_matched_expectation: false,
                normalization_applied: false,
                pubchem: vec![],
                rdkit: vec![],
            };
        }
    };
    let declared_products = match record.parse_products() {
        Ok(p) => p,
        Err(e) => {
            return RecordAudit {
                name: record.name,
                source: record.source,
                category: record.category,
                outcome: RecordOutcome::ParseFailed(e),
                matched_expectation: false,
                certificate: None,
                raw_molecule_validity: Err("not parsed".to_string()),
                // No corpus record currently expects a parse failure -- an
                // unparseable SMILES is always a genuine bug, not a
                // designed-in case, so this never counts as "matched."
                raw_validity_matched_expectation: false,
                normalization_applied: false,
                pubchem: vec![],
                rdkit: vec![],
            };
        }
    };

    let (raw_molecule_validity, normalization_applied) =
        check_raw_validity(&reactants, &declared_products);

    let pubchem_results: Vec<PubChemCrossReference> = if let Some(source) = pubchem_source {
        record
            .reactant_smiles
            .iter()
            .zip(reactants.iter())
            .chain(record.product_smiles.iter().zip(declared_products.iter()))
            .map(|(s, m)| {
                let our_formula = m.molecular_formula();
                // Dedup by distinct SMILES within this run, not once per
                // occurrence -- the throttle (when `source` actually
                // touches the network) lives in `pubchem.rs::real_http_get`
                // itself, so a cache hit here costs nothing extra.
                let outcome = pubchem_cache
                    .entry(s.to_string())
                    .or_insert_with(|| source.lookup(s))
                    .clone();
                let agreement = compute_agreement(&our_formula, &outcome);
                PubChemCrossReference {
                    smiles: s.to_string(),
                    our_formula,
                    outcome,
                    agreement,
                }
            })
            .collect()
    } else {
        vec![]
    };

    let rdkit_results: Vec<RdkitCrossReference> = if let Some(source) = rdkit_source {
        record
            .reactant_smiles
            .iter()
            .zip(reactants.iter())
            .chain(record.product_smiles.iter().zip(declared_products.iter()))
            .map(|(s, m)| {
                let our_formula = m.molecular_formula();
                let outcome = rdkit_cache
                    .entry(s.to_string())
                    .or_insert_with(|| source.lookup(s))
                    .clone();
                let agreement = compute_rdkit_agreement(&our_formula, &outcome);
                RdkitCrossReference {
                    smiles: s.to_string(),
                    our_formula,
                    outcome,
                    agreement,
                }
            })
            .collect()
    } else {
        vec![]
    };

    let candidates = classify_all(&reactants);
    let declared_formulas: Vec<String> = declared_products
        .iter()
        .map(|m| m.molecular_formula())
        .collect();

    // Try every matching template in order, accept the first whose computed
    // product actually agrees with the declared one -- see classify_all's
    // doc comment for why this replaced a fixed dispatch priority.
    //
    // **Real structural comparison, not formula-string equality.** An
    // amidation and an esterification of the same two reactants both lose
    // the same H2O byproduct from the same total mass -- their products are
    // ALWAYS constitutional isomers with an IDENTICAL molecular formula
    // regardless of whether the new bond forms at the amine's N or the
    // alcohol's O. Formula-string comparison can never tell them apart;
    // found live by `multiple_matching_templates_are_all_tried_esterification_wins_when_declared`
    // failing until this was fixed. Mirrors `examples/uspto_evaluation.rs`'s
    // already-correct approach: normalize, then compare via bounded exact
    // graph isomorphism, positionally (this crate's templates and corpus
    // both document/rely on "product, then water, in that order").
    let agreeing = candidates
        .iter()
        .find(|(_, _, products)| products_are_isomorphic(products, &declared_products));

    let outcome_and_cert = match (agreeing, candidates.first()) {
        (None, None) => (RecordOutcome::Unclassified, None),
        (None, Some((template, _, computed_products))) => {
            // At least one template matched this reactant shape, but NONE
            // of their computed products agree with the declared one --
            // report the mismatch using the first candidate tried
            // (deterministic; no candidate here is "more correct" than any
            // other since none of them matched).
            let computed_formulas: Vec<String> = computed_products
                .iter()
                .map(|m| m.molecular_formula())
                .collect();
            (
                RecordOutcome::DeclaredProductMismatch {
                    template,
                    computed_formulas,
                    declared_formulas: declared_formulas.clone(),
                },
                None,
            )
        }
        (Some((template, candidate_reactants, computed_products)), _) => {
            let candidate = ReactionCandidate {
                reactants: candidate_reactants.clone(),
                products: computed_products.clone(),
                template,
            };
            let result = oracle::evaluate(&candidate, policy);
            match &result.outcome {
                GateOutcome::FailedValidity(reason) => (
                    RecordOutcome::MatchedButFailedValidity {
                        template,
                        reason: reason.clone(),
                    },
                    None,
                ),
                GateOutcome::FailedScope(reason) => (
                    RecordOutcome::MatchedButScopeRejected {
                        template,
                        reason: reason.clone(),
                    },
                    None,
                ),
                GateOutcome::Passed => {
                    let seed_reactants_config: Vec<String> = record
                        .reactant_smiles
                        .iter()
                        .map(|s| s.to_string())
                        .collect();
                    let cert = ProcessCertificate::from_passed(
                        policy.name(),
                        &result,
                        &seed_reactants_config,
                    );
                    (RecordOutcome::Certified { template }, Some(cert))
                }
            }
        }
    };

    let matched_expectation =
        outcome_matches_expected(&outcome_and_cert.0, record.expected_outcome);
    let raw_validity_matched_expectation =
        raw_molecule_validity.is_ok() == record.expected_raw_validity_ok;

    RecordAudit {
        name: record.name,
        source: record.source,
        category: record.category,
        outcome: outcome_and_cert.0,
        matched_expectation,
        certificate: outcome_and_cert.1,
        raw_molecule_validity,
        raw_validity_matched_expectation,
        normalization_applied,
        pubchem: pubchem_results,
        rdkit: rdkit_results,
    }
}

/// Run the full corpus through the pipeline. `pubchem_source` gates whether
/// any network call happens at all -- `None` runs a fully offline audit;
/// see the module doc for what each concrete source means.
pub fn run_audit(
    corpus: &[CorpusRecord],
    policy: &dyn ScopePolicy,
    pubchem_source: Option<&dyn PubChemSource>,
    rdkit_source: Option<&dyn RdkitSource>,
) -> AuditReport {
    let mut pubchem_cache = HashMap::new();
    let mut rdkit_cache = HashMap::new();
    let mut summary = AuditSummary {
        total_records: corpus.len(),
        ..Default::default()
    };
    let records: Vec<RecordAudit> = corpus
        .iter()
        .map(|r| {
            audit_record(
                r,
                policy,
                &mut pubchem_cache,
                pubchem_source,
                &mut rdkit_cache,
                rdkit_source,
            )
        })
        .inspect(|audit| {
            match &audit.outcome {
                RecordOutcome::ParseFailed(_) => summary.parse_failed += 1,
                RecordOutcome::DeclaredProductMismatch { .. } => {
                    summary.declared_product_mismatch += 1
                }
                RecordOutcome::MatchedButFailedValidity { .. } => {
                    summary.matched_failed_validity += 1
                }
                RecordOutcome::MatchedButScopeRejected { .. } => {
                    summary.matched_scope_rejected += 1
                }
                RecordOutcome::Certified { .. } => summary.certified += 1,
                RecordOutcome::Unclassified => summary.unclassified += 1,
            }
            for xref in &audit.pubchem {
                match xref.agreement {
                    PubChemAgreement::Agrees => summary.pubchem_agreements += 1,
                    PubChemAgreement::RepresentationOnlyDifference => {
                        summary.pubchem_representation_only += 1
                    }
                    PubChemAgreement::Disagrees => summary.pubchem_disagreements += 1,
                    PubChemAgreement::NotFoundInPubChem => summary.pubchem_not_found += 1,
                    PubChemAgreement::Unavailable => summary.pubchem_unavailable += 1,
                }
            }
            for xref in &audit.rdkit {
                match xref.agreement {
                    RdkitAgreement::Agrees => summary.rdkit_agreements += 1,
                    RdkitAgreement::RepresentationOnlyDifference => {
                        summary.rdkit_representation_only += 1
                    }
                    RdkitAgreement::Disagrees => summary.rdkit_disagreements += 1,
                    RdkitAgreement::RejectedByRdkit => summary.rdkit_rejected += 1,
                    RdkitAgreement::Unavailable => summary.rdkit_unavailable += 1,
                }
            }
        })
        .collect();

    AuditReport { records, summary }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hazard_heuristics::ExternalScopeConfig;
    use crate::policy::OpenWithHeuristicScreenPolicy;

    fn policy() -> OpenWithHeuristicScreenPolicy {
        OpenWithHeuristicScreenPolicy {
            external: ExternalScopeConfig::default(),
        }
    }

    #[test]
    fn known_good_esterification_is_certified() {
        let record = crate::corpus::CorpusRecord {
            name: "test",
            source: "test",
            category: RecordCategory::Supported,
            expected_outcome: ExpectedOutcomeKind::Certified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["CC(=O)O", "CCO"],
            product_smiles: &["CC(=O)OCC", "O"],
        };
        let p = policy();
        let mut cache = HashMap::new();
        let audit = audit_record(&record, &p, &mut cache, None, &mut HashMap::new(), None);
        assert!(matches!(audit.outcome, RecordOutcome::Certified { .. }));
        assert!(audit.certificate.is_some());
        assert!(audit.raw_molecule_validity.is_ok());
        assert!(audit.matched_expectation);
        assert!(audit.raw_validity_matched_expectation);
    }

    #[test]
    fn disallowed_element_is_caught_via_raw_validity_even_when_unclassified() {
        let record = crate::corpus::CorpusRecord {
            name: "test",
            source: "test",
            category: RecordCategory::Malformed,
            expected_outcome: ExpectedOutcomeKind::Unclassified,
            expected_raw_validity_ok: false,
            reactant_smiles: &["C", "II"],
            product_smiles: &["CI", "I"],
        };
        let p = policy();
        let mut cache = HashMap::new();
        let audit = audit_record(&record, &p, &mut cache, None, &mut HashMap::new(), None);
        assert_eq!(audit.outcome, RecordOutcome::Unclassified);
        assert!(audit.raw_molecule_validity.is_err());
        assert!(audit.matched_expectation);
        assert!(audit.raw_validity_matched_expectation);
    }

    #[test]
    fn wrong_declared_product_is_caught_not_silently_certified() {
        let record = crate::corpus::CorpusRecord {
            name: "test",
            source: "test",
            category: RecordCategory::IncorrectDeclaredProduct,
            expected_outcome: ExpectedOutcomeKind::DeclaredMismatch,
            expected_raw_validity_ok: true,
            reactant_smiles: &["CC(=O)O", "CCO"],
            product_smiles: &["CC(=O)OCCC", "O"], // wrong: propyl not ethyl
        };
        let p = policy();
        let mut cache = HashMap::new();
        let audit = audit_record(&record, &p, &mut cache, None, &mut HashMap::new(), None);
        assert!(matches!(
            audit.outcome,
            RecordOutcome::DeclaredProductMismatch { .. }
        ));
        assert!(audit.certificate.is_none());
        assert!(audit.matched_expectation);
    }

    #[test]
    fn multiple_matching_templates_are_all_tried_esterification_wins_when_declared() {
        // Phase A.4 regression test: 2-(phenylamino)ethanol (OCCNc1ccccc1)
        // has BOTH a free (aniline-type) amine and a free alcohol, so BOTH
        // AmidationTemplate and EsterificationTemplate match this pair --
        // exactly the shape that caused a real, measured regression when
        // `classify` committed to a fixed "amidation first" priority (see
        // PROCESS_DISCOVERY_PHASE_A4_AMIDATION_REEVALUATION_2026-07-15.md).
        // The declared product here is the ESTER (real chemistry: aniline
        // nitrogens are poor nucleophiles). `classify_all` must try both
        // candidates and certify via esterification, not report a mismatch
        // just because amidation -- tried first -- doesn't match.
        let record = crate::corpus::CorpusRecord {
            name: "test",
            source: "test",
            category: RecordCategory::Supported,
            expected_outcome: ExpectedOutcomeKind::Certified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["CC(=O)O", "OCCNc1ccccc1"],
            product_smiles: &["CC(=O)OCCNc1ccccc1", "O"],
        };
        let p = policy();
        let mut cache = HashMap::new();
        let audit = audit_record(&record, &p, &mut cache, None, &mut HashMap::new(), None);
        assert_eq!(
            audit.outcome,
            RecordOutcome::Certified {
                template: "esterification"
            },
            "esterification must be found even though amidation is tried first and also matches"
        );
        assert!(audit.certificate.is_some());
        assert!(audit.matched_expectation);
    }

    #[test]
    fn exhaustive_hydrogenation_wins_when_declared_is_fully_saturated() {
        // Phase A.5: 1,4-pentadiene (C=CCC=C) has 2 reducible C-C bonds, so
        // BOTH HydrogenationTemplate (single-step, -> 1-pentene) and
        // ExhaustiveHydrogenationTemplate (-> pentane) match this pair.
        // The corpus record only needs to name ONE H2 in reactant_smiles
        // (classify_all computes the exact count itself for the exhaustive
        // candidate); the declared product here is the FULLY saturated
        // pentane, so the exhaustive candidate must win.
        let record = crate::corpus::CorpusRecord {
            name: "test",
            source: "test",
            category: RecordCategory::Supported,
            expected_outcome: ExpectedOutcomeKind::Certified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["C=CCC=C", "[H][H]"],
            product_smiles: &["CCCCC"],
        };
        let p = policy();
        let mut cache = HashMap::new();
        let audit = audit_record(&record, &p, &mut cache, None, &mut HashMap::new(), None);
        assert_eq!(
            audit.outcome,
            RecordOutcome::Certified {
                template: "exhaustive_hydrogenation"
            },
            "the fully-saturated declared product must select the exhaustive candidate, \
             not the single-step one tried first"
        );
        assert!(audit.certificate.is_some());
        assert!(audit.matched_expectation);
    }

    #[test]
    fn single_step_hydrogenation_still_wins_when_declared_is_partially_reduced() {
        // The inverse of the test above: same reactant, but the declared
        // product is only PARTIALLY reduced (1-pentene) -- confirms adding
        // the exhaustive candidate didn't silently break the existing
        // single-step behavior for records where that's the real outcome
        // (e.g. a deliberately chemoselective reduction).
        let record = crate::corpus::CorpusRecord {
            name: "test",
            source: "test",
            category: RecordCategory::Supported,
            expected_outcome: ExpectedOutcomeKind::Certified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["C=CCC=C", "[H][H]"],
            product_smiles: &["CCCC=C"],
        };
        let p = policy();
        let mut cache = HashMap::new();
        let audit = audit_record(&record, &p, &mut cache, None, &mut HashMap::new(), None);
        assert_eq!(
            audit.outcome,
            RecordOutcome::Certified {
                template: "hydrogenation"
            },
            "the partially-reduced declared product must select the single-step candidate"
        );
        assert!(audit.certificate.is_some());
        assert!(audit.matched_expectation);
    }

    #[test]
    fn unsupported_carbonyl_hydrogenation_is_unclassified_not_a_false_certificate() {
        let record = crate::corpus::CorpusRecord {
            name: "test",
            source: "test",
            category: RecordCategory::UnsupportedButValid,
            expected_outcome: ExpectedOutcomeKind::Unclassified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["CC(=O)C", "[H][H]"],
            product_smiles: &["CC(O)C"],
        };
        let p = policy();
        let mut cache = HashMap::new();
        let audit = audit_record(&record, &p, &mut cache, None, &mut HashMap::new(), None);
        assert_eq!(audit.outcome, RecordOutcome::Unclassified);
        assert!(audit.raw_molecule_validity.is_ok()); // well-formed, just unsupported
        assert!(audit.matched_expectation);
        assert!(audit.raw_validity_matched_expectation);
    }

    #[test]
    fn neutrally_drawn_nitro_group_now_passes_raw_validity_after_normalization() {
        // Regression test for the exact gap the first live audit run
        // surfaced (Phase A.1): nitrobenzene's neutral N(=O)=O shorthand
        // used to fail raw_molecule_validity outright. It's still
        // Unclassified (no template covers nitro reduction) -- but raw
        // structural validity must now pass, since the shorthand
        // normalizes to a real, valid charge-separated structure first.
        let record = crate::corpus::CorpusRecord {
            name: "test",
            source: "test",
            category: RecordCategory::UnsupportedButValid,
            expected_outcome: ExpectedOutcomeKind::Unclassified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["c1ccccc1N(=O)=O", "[H][H]"],
            product_smiles: &["c1ccccc1N"],
        };
        let p = policy();
        let mut cache = HashMap::new();
        let audit = audit_record(&record, &p, &mut cache, None, &mut HashMap::new(), None);
        assert_eq!(audit.outcome, RecordOutcome::Unclassified);
        assert!(
            audit.raw_molecule_validity.is_ok(),
            "expected normalization to fix raw validity: {:?}",
            audit.raw_molecule_validity
        );
        assert!(audit.raw_validity_matched_expectation);
    }

    #[test]
    fn full_corpus_runs_without_panicking_and_summary_counts_add_up() {
        let corpus = crate::corpus::phase_a_fixture_corpus();
        let p = policy();
        let report = run_audit(&corpus, &p, None, None);
        assert_eq!(report.records.len(), corpus.len());
        let s = &report.summary;
        assert_eq!(
            s.parse_failed
                + s.declared_product_mismatch
                + s.matched_failed_validity
                + s.matched_scope_rejected
                + s.certified
                + s.unclassified,
            s.total_records
        );
        // At least one of each of the deliberate edge cases should land
        // where designed.
        assert!(s.certified > 0);
        assert!(s.unclassified > 0);
        assert!(s.declared_product_mismatch > 0);
        // pubchem_source was None -- no cross-reference counts at all.
        assert_eq!(s.pubchem_agreements, 0);
        assert_eq!(s.pubchem_disagreements, 0);
        assert_eq!(s.pubchem_not_found, 0);
        assert_eq!(s.pubchem_unavailable, 0);

        // Phase A.2's core self-consistency check: every corpus record's
        // hand-authored `expected_outcome`/`expected_raw_validity_ok` must
        // actually match what the pipeline produces. A failure here means
        // either a corpus-authoring mistake (e.g. a hand-computed product
        // formula for an adversarial/alternate-representation record is
        // wrong) or a real pipeline regression -- this test can't tell
        // which, but it catches both rather than letting either drift
        // silently.
        for record in &report.records {
            assert!(
                record.matched_expectation,
                "{}: outcome {:?} did not match its declared expectation",
                record.name, record.outcome
            );
            assert!(
                record.raw_validity_matched_expectation,
                "{}: raw_molecule_validity {:?} did not match its declared expectation",
                record.name, record.raw_molecule_validity
            );
        }
    }

    #[test]
    fn pubchem_agreement_is_computed_when_a_source_is_provided() {
        use crate::pubchem::{PubChemQueryOutcome, PubChemRecord, PubChemSource};

        struct FixedSource;
        impl PubChemSource for FixedSource {
            fn lookup(&self, smiles: &str) -> PubChemQueryOutcome {
                match smiles {
                    "CCO" => PubChemQueryOutcome::Found(PubChemRecord {
                        cid: 702,
                        molecular_formula: "C2H6O".to_string(), // agrees
                        connectivity_smiles: None,
                        iupac_name: None,
                    }),
                    "CC(=O)O" => PubChemQueryOutcome::Found(PubChemRecord {
                        cid: 176,
                        molecular_formula: "WRONG_FORMULA".to_string(), // disagrees
                        connectivity_smiles: None,
                        iupac_name: None,
                    }),
                    _ => PubChemQueryOutcome::NotFound,
                }
            }
        }

        let record = crate::corpus::CorpusRecord {
            name: "test",
            source: "test",
            category: RecordCategory::Supported,
            expected_outcome: ExpectedOutcomeKind::Certified,
            expected_raw_validity_ok: true,
            reactant_smiles: &["CC(=O)O", "CCO"],
            product_smiles: &["CC(=O)OCC", "O"],
        };
        let p = policy();
        let mut cache = HashMap::new();
        let source = FixedSource;
        let audit = audit_record(
            &record,
            &p,
            &mut cache,
            Some(&source),
            &mut HashMap::new(),
            None,
        );
        assert_eq!(audit.pubchem.len(), 4); // 2 reactants + 2 products, none repeated
        let ethanol = audit.pubchem.iter().find(|x| x.smiles == "CCO").unwrap();
        assert_eq!(ethanol.agreement, PubChemAgreement::Agrees);
        let acid = audit
            .pubchem
            .iter()
            .find(|x| x.smiles == "CC(=O)O")
            .unwrap();
        assert_eq!(acid.agreement, PubChemAgreement::Disagrees);
        let water = audit.pubchem.iter().find(|x| x.smiles == "O").unwrap();
        assert_eq!(water.agreement, PubChemAgreement::NotFoundInPubChem);
    }

    // The core reproducibility guarantee (Phase A.1): a total PubChem
    // outage must never change which candidates get certified, rejected,
    // or left unclassified. Runs the WHOLE fixture corpus twice -- once
    // with a fault-injected source that fails every lookup, once fully
    // offline -- and diffs every record's outcome and certificate JSON.
    #[test]
    fn network_fault_never_changes_local_verdict() {
        use crate::pubchem::AlwaysUnavailableSource;

        let corpus = crate::corpus::phase_a_fixture_corpus();
        let p = policy();
        let faulty = AlwaysUnavailableSource;

        let with_fault = run_audit(&corpus, &p, Some(&faulty), None);
        let offline = run_audit(&corpus, &p, None, None);

        assert_eq!(with_fault.records.len(), offline.records.len());
        for (a, b) in with_fault.records.iter().zip(offline.records.iter()) {
            assert_eq!(a.name, b.name);
            assert_eq!(
                a.outcome, b.outcome,
                "a network fault changed the verdict for {}",
                a.name
            );
            assert_eq!(
                a.raw_molecule_validity.is_ok(),
                b.raw_molecule_validity.is_ok(),
                "a network fault changed raw validity for {}",
                a.name
            );
            let cert_json_a = a.certificate.as_ref().map(|c| c.to_json_pretty().unwrap());
            let cert_json_b = b.certificate.as_ref().map(|c| c.to_json_pretty().unwrap());
            assert_eq!(
                cert_json_a, cert_json_b,
                "a network fault changed the certificate for {}",
                a.name
            );
        }
        // The fault case still recorded that PubChem was unavailable --
        // proving the parity above isn't just because pubchem was skipped
        // both times.
        assert!(with_fault.records.iter().any(|r| {
            !r.pubchem.is_empty()
                && r.pubchem
                    .iter()
                    .all(|x| x.agreement == PubChemAgreement::Unavailable)
        }));
    }

    // Same guarantee as `network_fault_never_changes_local_verdict`, for
    // the RDKit cross-reference specifically -- a total RDKit outage
    // (not installed, subprocess unreachable, whatever the cause) must
    // never change a local verdict either.
    #[test]
    fn rdkit_fault_never_changes_local_verdict() {
        use crate::rdkit::AlwaysUnavailableRdkitSource;

        let corpus = crate::corpus::phase_a_fixture_corpus();
        let p = policy();
        let faulty = AlwaysUnavailableRdkitSource;

        let with_fault = run_audit(&corpus, &p, None, Some(&faulty));
        let offline = run_audit(&corpus, &p, None, None);

        assert_eq!(with_fault.records.len(), offline.records.len());
        for (a, b) in with_fault.records.iter().zip(offline.records.iter()) {
            assert_eq!(a.name, b.name);
            assert_eq!(
                a.outcome, b.outcome,
                "an rdkit fault changed the verdict for {}",
                a.name
            );
            let cert_json_a = a.certificate.as_ref().map(|c| c.to_json_pretty().unwrap());
            let cert_json_b = b.certificate.as_ref().map(|c| c.to_json_pretty().unwrap());
            assert_eq!(
                cert_json_a, cert_json_b,
                "an rdkit fault changed the certificate for {}",
                a.name
            );
        }
        assert!(with_fault.records.iter().any(|r| {
            !r.rdkit.is_empty()
                && r.rdkit
                    .iter()
                    .all(|x| x.agreement == RdkitAgreement::Unavailable)
        }));
    }

    // Live-vs-replay parity: `fixtures/pubchem_corpus_fixture.json` is a
    // real frozen recording of a live `--record` run against the actual
    // PubChem API (last re-recorded 2026-07-15, Phase A.4, when the corpus
    // grew from 28 to 31 records with the new AmidationTemplate fixtures --
    // see CHEMICAL_PROCESS_DISCOVERY_PLAN's Phase A.1/A.4 sections for the
    // exact commands). Embedded at compile time (`include_str!`, not a
    // runtime file path) so this test is not sensitive to the working
    // directory `cargo test` happens to run from. Replaying it must
    // reproduce the exact same summary that live run printed -- proving the
    // recording round-trips faithfully, not just that `ReplaySource` can
    // answer *something*.
    #[test]
    fn replaying_the_frozen_fixture_reproduces_the_recorded_live_run() {
        use crate::cache::{PubChemFixtureCache, ReplaySource};

        let raw = include_str!("../fixtures/pubchem_corpus_fixture.json");
        let cache: PubChemFixtureCache = serde_json::from_str(raw).unwrap();
        let source = ReplaySource::from_cache(cache);

        let corpus = crate::corpus::phase_a_fixture_corpus();
        let p = policy();
        let report = run_audit(&corpus, &p, Some(&source), None);
        let s = &report.summary;

        // These exact counts are what the real live run printed against the
        // CURRENT (Phase A.7, 32-record) corpus -- see the plan doc's
        // Phase A.1/A.4/A.7 sections for the exact commands and the
        // fixture's recording date. A change here means either the corpus
        // or the pipeline's classification logic changed since the fixture
        // was recorded, which is exactly the drift this test exists to
        // catch -- it caught exactly this at Phase A.4 (certified 19 -> 22)
        // and again at Phase A.7 (total 31 -> 32, unclassified 6 -> 7, when
        // the methane+Cl2 record was reclassified from Malformed to
        // UnsupportedButValid and a new iodine-based Malformed record was
        // added, now that Cl is in scope -- certified is unaffected since
        // this fixture corpus doesn't otherwise exercise Si/P/S/Cl/Br).
        assert_eq!(s.total_records, corpus.len());
        assert_eq!(s.certified, 22);
        assert_eq!(s.unclassified, 7);
        assert_eq!(s.declared_product_mismatch, 1);
        assert_eq!(s.parse_failed, 0);
        assert_eq!(s.matched_failed_validity, 0);
        assert_eq!(s.matched_scope_rejected, 2);
        assert_eq!(s.pubchem_disagreements, 0);
        assert_eq!(s.pubchem_unavailable, 3);
        assert!(s.pubchem_agreements > 0);
    }
}
