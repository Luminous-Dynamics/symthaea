// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phase A.3: frozen USPTO template-coverage evaluation.
//!
//! Runs the crate's existing, UNMODIFIED validity/normalization/template/
//! isomorphism/policy pipeline against 1,282 real USPTO-50K reaction
//! records that superficially resemble this crate's two supported
//! templates (per `external_corpus/FREEZE_MANIFEST.md`), and answers: of
//! reactions that look supported, how many does the current system
//! independently reconstruct and certify -- and where it doesn't, why not?
//!
//! **Only calls existing public API** (`oracle::evaluate`,
//! `templates::{EsterificationTemplate,HydrogenationTemplate}`,
//! `isomorphism::is_isomorphic_detailed`, `normalization`, `policy`) --
//! `normalization.rs`/`templates.rs`/`oracle.rs`/`validity.rs`/`policy.rs`/
//! `isomorphism.rs` are not touched by this file. See
//! `FREEZE_MANIFEST.md`'s closing section: no chemistry changes until this
//! evaluation's first full result is committed.
//!
//! **Graph-equivalence, not formula-string or byte-string equality.** Per
//! review: comparing serialized SMILES/formula strings for exact equality
//! would conflate genuine mismatches with mere representation differences
//! (atom ordering, aromatic-vs-Kekule, resonance/charge notation). Both the
//! computed and declared product are run through `normalization.rs` (the
//! same normalization `oracle::evaluate` applies internally) and then
//! compared via `isomorphism::is_isomorphic_detailed` -- bounded exact
//! labeled-graph isomorphism, not string comparison.
//!
//! **Known, disclosed data-preparation step**: USPTO-50K's retrosynthesis
//! framing omits H2 as an explicit reactant for hydrogenations (it's an
//! omitted reagent in the source, not a written molecule) -- this harness
//! appends `templates::molecular_hydrogen()` to every hydrogenation
//! candidate's reactant list before classification. This is data
//! preparation (matching real records to the shape the frozen template
//! requires), not a chemistry change.
//!
//! **Ambiguous-reactive-site is a cross-cutting diagnostic flag, not a
//! mutually-exclusive category.** A record can have multiple candidate
//! carboxyl/alcohol/unsaturation sites AND still land in any of the other
//! seven outcome buckets (e.g. correctly certified because the first match
//! happened to be the right one, or wrong-transformation because it wasn't).
//! Forcing it into a single mutually-exclusive taxonomy slot would hide
//! that co-occurrence, so it's reported as `ambiguous_site: bool` alongside
//! the primary category instead. The detector functions below
//! (`count_*_candidates`) are new, read-only, parallel implementations of
//! the same matching logic `templates.rs`'s private `find_carboxyl`/
//! `find_alcohol_hydroxyl`/hydrogenation-bond-search use -- they do not
//! call into or modify `templates.rs`, and never influence which candidate
//! the real (frozen) template actually picks.

use std::env;
use std::fs;
use symthaea_organic_chemistry::smiles::{BondOrder, Molecule};
use symthaea_process_discovery::hazard_heuristics::ExternalScopeConfig;
use symthaea_process_discovery::isomorphism::{self, IsomorphismOutcome};
use symthaea_process_discovery::normalization;
use symthaea_process_discovery::oracle::{self, GateOutcome};
use symthaea_process_discovery::policy::{OpenWithHeuristicScreenPolicy, ScopePolicy};
use symthaea_process_discovery::templates::{
    AmidationTemplate, EsterificationTemplate, ExhaustiveHydrogenationTemplate,
    HydrogenationTemplate, ReactionTemplate, count_reducible_cc_bonds, molecular_hydrogen,
};
use symthaea_process_discovery::types::ReactionCandidate;

const ESTER_TSV: &str =
    include_str!("../external_corpus/uspto50k_scaffold_split/ester_candidates_split.tsv");
const HYDRO_TSV: &str =
    include_str!("../external_corpus/uspto50k_scaffold_split/hydro_candidates_split.tsv");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReactionKind {
    Esterification,
    Hydrogenation,
}

impl ReactionKind {
    fn label(self) -> &'static str {
        match self {
            ReactionKind::Esterification => "esterification",
            ReactionKind::Hydrogenation => "hydrogenation",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Split {
    Dev,
    Validation,
    Holdout,
}

impl Split {
    fn parse(s: &str) -> Self {
        match s {
            "dev" => Split::Dev,
            "validation" => Split::Validation,
            "holdout" => Split::Holdout,
            other => panic!("unknown split label in frozen data: {other:?}"),
        }
    }
    fn label(self) -> &'static str {
        match self {
            Split::Dev => "dev",
            Split::Validation => "validation",
            Split::Holdout => "holdout",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutcomeCategory {
    CertifiedExact,
    StructurallyShapedWrongTransformation,
    UnsupportedMultiTransformation,
    ValidityOrConservationFailure,
    PolicyOrHazardRejection,
    RepresentationOrNormalizationFailure,
    ResourceBoundedUncertainty,
}

impl OutcomeCategory {
    const ALL: [OutcomeCategory; 7] = [
        OutcomeCategory::CertifiedExact,
        OutcomeCategory::StructurallyShapedWrongTransformation,
        OutcomeCategory::UnsupportedMultiTransformation,
        OutcomeCategory::ValidityOrConservationFailure,
        OutcomeCategory::PolicyOrHazardRejection,
        OutcomeCategory::RepresentationOrNormalizationFailure,
        OutcomeCategory::ResourceBoundedUncertainty,
    ];

    fn label(self) -> &'static str {
        match self {
            OutcomeCategory::CertifiedExact => "certified_exact_transformation",
            OutcomeCategory::StructurallyShapedWrongTransformation => {
                "structurally_shaped_wrong_transformation"
            }
            OutcomeCategory::UnsupportedMultiTransformation => "unsupported_reaction_context",
            OutcomeCategory::ValidityOrConservationFailure => "validity_or_conservation_failure",
            OutcomeCategory::PolicyOrHazardRejection => "policy_or_hazard_rejection",
            OutcomeCategory::RepresentationOrNormalizationFailure => {
                "representation_or_normalization_failure"
            }
            OutcomeCategory::ResourceBoundedUncertainty => "resource_bounded_uncertainty",
        }
    }
}

struct CandidateRow {
    kind: ReactionKind,
    split: Split,
    reactants_smiles: String,
    product_smiles: String,
}

struct CandidateResult {
    row_index: usize,
    kind: ReactionKind,
    split: Split,
    reactants_smiles: String,
    product_smiles: String,
    category: OutcomeCategory,
    ambiguous_site: bool,
    detail: String,
}

fn parse_tsv(tsv: &str, kind: ReactionKind) -> Vec<CandidateRow> {
    let mut lines = tsv.lines();
    lines.next(); // header
    lines
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let cols: Vec<&str> = line.split('\t').collect();
            assert_eq!(
                cols.len(),
                4,
                "expected 4 tab-separated columns, got {}: {line:?}",
                cols.len()
            );
            CandidateRow {
                kind,
                split: Split::parse(cols[3]),
                reactants_smiles: cols[0].to_string(),
                product_smiles: cols[1].to_string(),
            }
        })
        .collect()
}

// --- Non-invasive ambiguity diagnostics: parallel counting versions of
// templates.rs's private matcher logic. Never call into templates.rs, never
// influence which site the real (frozen) template picks. ---

fn count_carboxyl_candidates(m: &Molecule) -> usize {
    let mut count = 0;
    for (i, atom) in m.atoms.iter().enumerate() {
        if atom.element != "C" {
            continue;
        }
        let neighbors = m.neighbors(i);
        let has_carbonyl_o = neighbors
            .iter()
            .any(|(j, o)| *o == BondOrder::Double && m.atoms[*j].element == "O");
        if !has_carbonyl_o {
            continue;
        }
        let has_oh = neighbors.iter().any(|(j, o)| {
            *o == BondOrder::Single && m.atoms[*j].element == "O" && m.atoms[*j].hydrogens >= 1
        });
        if has_oh {
            count += 1;
        }
    }
    count
}

fn count_alcohol_candidates(m: &Molecule) -> usize {
    let mut count = 0;
    for (i, atom) in m.atoms.iter().enumerate() {
        if atom.element != "O" || atom.hydrogens == 0 {
            continue;
        }
        let neighbors = m.neighbors(i);
        if neighbors.len() != 1 {
            continue;
        }
        let (c_idx, order) = neighbors[0];
        if order != BondOrder::Single || m.atoms[c_idx].element != "C" {
            continue;
        }
        let c_neighbors = m.neighbors(c_idx);
        let is_carbonyl_carbon = c_neighbors
            .iter()
            .any(|(j, o)| *o == BondOrder::Double && m.atoms[*j].element == "O");
        if !is_carbonyl_carbon {
            count += 1;
        }
    }
    count
}

fn count_cc_unsaturation_candidates(m: &Molecule) -> usize {
    m.bonds
        .iter()
        .filter(|b| {
            matches!(b.order, BondOrder::Double | BondOrder::Triple)
                && m.atoms[b.a].element == "C"
                && m.atoms[b.b].element == "C"
        })
        .count()
}

/// Phase A.4: parallel, read-only mirror of `templates::find_free_amine`'s
/// matching logic -- same convention as `count_carboxyl_candidates`/
/// `count_alcohol_candidates` above, does not call into or modify
/// `templates.rs`.
fn count_amine_candidates(m: &Molecule) -> usize {
    let mut count = 0;
    for (i, atom) in m.atoms.iter().enumerate() {
        if atom.element != "N" || atom.aromatic || atom.charge != 0 || atom.hydrogens == 0 {
            continue;
        }
        let already_amide = m.neighbors(i).iter().any(|&(j, order)| {
            order != BondOrder::Aromatic
                && m.atoms[j].element == "C"
                && m.neighbors(j)
                    .iter()
                    .any(|&(k, o2)| o2 == BondOrder::Double && m.atoms[k].element == "O")
        });
        if !already_amide {
            count += 1;
        }
    }
    count
}

fn classify_candidate(
    row: &CandidateRow,
    policy: &dyn ScopePolicy,
) -> (OutcomeCategory, bool, String) {
    let reactant_smiles_list: Vec<&str> = row.reactants_smiles.split('.').collect();
    let mut reactants = Vec::with_capacity(reactant_smiles_list.len());
    for s in &reactant_smiles_list {
        match Molecule::from_smiles(s) {
            Ok(m) => reactants.push(m),
            Err(e) => {
                return (
                    OutcomeCategory::RepresentationOrNormalizationFailure,
                    false,
                    format!("reactant '{s}' failed to parse: {e}"),
                );
            }
        }
    }
    let declared_product = match Molecule::from_smiles(&row.product_smiles) {
        Ok(m) => m,
        Err(e) => {
            return (
                OutcomeCategory::RepresentationOrNormalizationFailure,
                false,
                format!(
                    "declared product '{}' failed to parse: {e}",
                    row.product_smiles
                ),
            );
        }
    };

    let ambiguous = match row.kind {
        ReactionKind::Esterification => {
            let carboxyl_count: usize = reactants.iter().map(count_carboxyl_candidates).sum();
            let alcohol_count: usize = reactants.iter().map(count_alcohol_candidates).sum();
            let amine_count: usize = reactants.iter().map(count_amine_candidates).sum();
            carboxyl_count > 1
                || alcohol_count > 1
                || amine_count > 1
                // Phase A.4: a competing DIFFERENT group type (amine vs.
                // alcohol) is exactly the shape the amine-competition
                // pattern found -- the original same-type-only check missed
                // this entirely (see PROCESS_DISCOVERY_PHASE_A3_V2_ADJUDICATION_2026-07-15.md's
                // "a real limitation of the diagnostic flag itself" note).
                || (alcohol_count >= 1 && amine_count >= 1)
        }
        ReactionKind::Hydrogenation => {
            reactants
                .iter()
                .map(count_cc_unsaturation_candidates)
                .sum::<usize>()
                > 1
        }
    };

    // Data preparation, not a chemistry change: USPTO-50K omits H2 as an
    // explicit hydrogenation reactant (see module doc).
    let mut classify_reactants = reactants.clone();
    if row.kind == ReactionKind::Hydrogenation {
        classify_reactants.push(molecular_hydrogen());
    }

    // Phase A.4: try every template whose shape matches, not a fixed
    // priority order. A live 1,282-record re-evaluation found the earlier
    // "amidation always wins" priority wrong on 12 records where a
    // technically-free-but-poorly-nucleophilic nitrogen (aniline-conjugated,
    // amidine-conjugated, or sterically hindered) wrongly outranked the
    // alcohol the real reaction actually used -- see
    // PROCESS_DISCOVERY_PHASE_A4_AMIDATION_REEVALUATION_2026-07-15.md.
    // Mirrors audit.rs's classify_all: this harness DOES have a declared
    // product to check against, so it doesn't need to predict selectivity.
    //
    // **Phase A.5**: each candidate now carries its OWN reactants list, not
    // a shared one -- ExhaustiveHydrogenationTemplate needs a different H2
    // count than the other three templates share (one h2() per degree of
    // unsaturation, computed exactly, not always exactly one). See that
    // template's doc comment for why an exact count matters for mass
    // balance.
    let mut candidates: Vec<(&'static str, Vec<Molecule>, Vec<Molecule>)> = [
        AmidationTemplate
            .apply(&classify_reactants)
            .map(|p| ("amidation", classify_reactants.clone(), p)),
        EsterificationTemplate
            .apply(&classify_reactants)
            .map(|p| ("esterification", classify_reactants.clone(), p)),
        HydrogenationTemplate
            .apply(&classify_reactants)
            .map(|p| ("hydrogenation", classify_reactants.clone(), p)),
    ]
    .into_iter()
    .flatten()
    .collect();
    if let [unsaturated, h2] = classify_reactants.as_slice() {
        if h2.molecular_formula() == "H2" {
            let needed = count_reducible_cc_bonds(unsaturated);
            if needed > 0 {
                let mut exhaustive_reactants = vec![unsaturated.clone()];
                exhaustive_reactants.extend((0..needed).map(|_| molecular_hydrogen()));
                if let Some(products) = ExhaustiveHydrogenationTemplate.apply(&exhaustive_reactants)
                {
                    candidates.push(("exhaustive_hydrogenation", exhaustive_reactants, products));
                }
            }
        }
    }

    if candidates.is_empty() {
        return (
            OutcomeCategory::UnsupportedMultiTransformation,
            ambiguous,
            "neither template's exact-arity pattern matched this reactant set".to_string(),
        );
    }

    // Normalized labeled-graph equivalence, not string/formula equality.
    let (declared_norm, _) = normalization::normalize_molecule(&declared_product);

    // Try each candidate; accept the first whose computed product is
    // actually isomorphic to the declared one. A resource-bound hit on any
    // candidate is reported over a NotIsomorphic verdict from another --
    // "couldn't verify" must never be silently overridden by "confidently
    // wrong" from an unrelated template attempt.
    let mut resource_bounded_detail: Option<String> = None;
    for (template_name, candidate_reactants, computed_products) in &candidates {
        // Esterification/amidation compute [product, water]; USPTO's single
        // declared product is the main organic product only (water
        // omitted). Hydrogenation/exhaustive-hydrogenation compute exactly
        // 1 product.
        let primary_computed = &computed_products[0];
        let (computed_norm, _) = normalization::normalize_molecule(primary_computed);
        match isomorphism::is_isomorphic_detailed(&computed_norm, &declared_norm) {
            IsomorphismOutcome::Isomorphic => {
                let candidate = ReactionCandidate {
                    reactants: candidate_reactants.clone(),
                    products: computed_products.clone(),
                    template: template_name,
                };
                let result = oracle::evaluate(&candidate, policy);
                return match result.outcome {
                    GateOutcome::FailedValidity(reason) => (
                        OutcomeCategory::ValidityOrConservationFailure,
                        ambiguous,
                        reason,
                    ),
                    GateOutcome::FailedScope(reason) => {
                        (OutcomeCategory::PolicyOrHazardRejection, ambiguous, reason)
                    }
                    GateOutcome::Passed => (
                        OutcomeCategory::CertifiedExact,
                        ambiguous,
                        "certified".to_string(),
                    ),
                };
            }
            IsomorphismOutcome::AtomLimitExceeded | IsomorphismOutcome::SearchBudgetExceeded => {
                if resource_bounded_detail.is_none() {
                    resource_bounded_detail = Some(format!("{template_name}: unable to verify"));
                }
            }
            IsomorphismOutcome::NotIsomorphic => {}
        }
    }

    if let Some(detail) = resource_bounded_detail {
        return (
            OutcomeCategory::ResourceBoundedUncertainty,
            ambiguous,
            detail,
        );
    }

    // None of the candidates' computed products matched the declared one --
    // report using the first candidate tried (deterministic; no candidate
    // here is "more correct" than any other since none of them matched).
    let (first_template, _, first_products) = &candidates[0];
    let (first_norm, _) = normalization::normalize_molecule(&first_products[0]);
    (
        OutcomeCategory::StructurallyShapedWrongTransformation,
        ambiguous,
        format!(
            "computed={} (via {first_template}) declared={} -- tried: {}",
            first_norm.molecular_formula(),
            declared_norm.molecular_formula(),
            candidates
                .iter()
                .map(|(t, _, _)| *t)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    )
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let out_dir = args
        .get(1)
        .unwrap_or_else(|| panic!("usage: uspto_evaluation <output_dir>"));
    fs::create_dir_all(out_dir).unwrap();

    let policy = OpenWithHeuristicScreenPolicy {
        external: ExternalScopeConfig::default(),
    };

    let mut rows = parse_tsv(ESTER_TSV, ReactionKind::Esterification);
    rows.extend(parse_tsv(HYDRO_TSV, ReactionKind::Hydrogenation));
    println!("Loaded {} frozen candidates.", rows.len());

    let start = std::time::Instant::now();
    let results: Vec<CandidateResult> = rows
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let (category, ambiguous_site, detail) = classify_candidate(row, &policy);
            CandidateResult {
                row_index: i,
                kind: row.kind,
                split: row.split,
                reactants_smiles: row.reactants_smiles.clone(),
                product_smiles: row.product_smiles.clone(),
                category,
                ambiguous_site,
                detail,
            }
        })
        .collect();
    let elapsed = start.elapsed();
    println!(
        "Classified {} candidates in {:.2?}.",
        results.len(),
        elapsed
    );

    // Raw per-record output -- the artifact that enables future stratified
    // sampling for manual adjudication.
    let mut raw = String::from(
        "row_index\tkind\tsplit\tcategory\tambiguous_site\treactants\tproduct\tdetail\n",
    );
    for r in &results {
        raw.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            r.row_index,
            r.kind.label(),
            r.split.label(),
            r.category.label(),
            r.ambiguous_site,
            r.reactants_smiles,
            r.product_smiles,
            r.detail.replace('\t', " ").replace('\n', " ")
        ));
    }
    fs::write(format!("{out_dir}/per_record_results.tsv"), raw).unwrap();

    // Summary: overall, per-kind, per-split.
    let mut summary = String::new();
    summary.push_str("# Phase A.3 USPTO Template-Coverage Evaluation -- First Frozen Run\n\n");
    summary.push_str(&format!(
        "total_candidates={} elapsed={:.2?}\n\n",
        results.len(),
        elapsed
    ));

    let report_block = |label: &str, subset: &[&CandidateResult]| -> String {
        let mut s = format!("## {label} (n={})\n\n", subset.len());
        for cat in OutcomeCategory::ALL {
            let count = subset.iter().filter(|r| r.category == cat).count();
            let pct = if subset.is_empty() {
                0.0
            } else {
                100.0 * count as f64 / subset.len() as f64
            };
            s.push_str(&format!("- {}: {} ({:.1}%)\n", cat.label(), count, pct));
        }
        let ambiguous_count = subset.iter().filter(|r| r.ambiguous_site).count();
        s.push_str(&format!(
            "- ambiguous_reactive_site (cross-cutting flag, any category): {} ({:.1}%)\n\n",
            ambiguous_count,
            if subset.is_empty() {
                0.0
            } else {
                100.0 * ambiguous_count as f64 / subset.len() as f64
            }
        ));
        s
    };

    let all_refs: Vec<&CandidateResult> = results.iter().collect();
    summary.push_str(&report_block("Overall", &all_refs));

    for kind in [ReactionKind::Esterification, ReactionKind::Hydrogenation] {
        let subset: Vec<&CandidateResult> = results.iter().filter(|r| r.kind == kind).collect();
        summary.push_str(&report_block(kind.label(), &subset));
    }

    for split in [Split::Dev, Split::Validation, Split::Holdout] {
        let subset: Vec<&CandidateResult> = results.iter().filter(|r| r.split == split).collect();
        summary.push_str(&report_block(&format!("split={}", split.label()), &subset));
    }

    let iso_diag = isomorphism::diagnostics();
    summary.push_str(&format!(
        "## Isomorphism module diagnostics (cumulative, process-global)\n\ncomparisons_attempted={} atom_limit_rejections={} budget_exhaustions={} worst_steps_used={} worst_depth_reached={}\n",
        iso_diag.comparisons_attempted,
        iso_diag.atom_limit_rejections,
        iso_diag.budget_exhaustions,
        iso_diag.worst_steps_used,
        iso_diag.worst_depth_reached
    ));

    println!("{summary}");
    fs::write(format!("{out_dir}/summary.md"), &summary).unwrap();
    println!("Wrote {out_dir}/summary.md and {out_dir}/per_record_results.tsv");
}
