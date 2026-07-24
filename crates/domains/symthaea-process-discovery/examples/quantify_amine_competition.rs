// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phase A.3 follow-up: how much of `structurally_shaped_wrong_transformation`
//! is explained by the amine-vs-alcohol competition pattern the v2
//! adjudication found (`PROCESS_DISCOVERY_PHASE_A3_V2_ADJUDICATION_2026-07-15.md`)?
//! That pass hand-inspected only 5 of 139 records (all newly-reclassified by
//! the ordering fix); 4/5 shared one shape -- a reactant with BOTH a free
//! amine and a free alcohol, where the real USPTO product formed an amide at
//! the amine but `EsterificationTemplate` (blind to amines) silently
//! esterified the alcohol instead. This scans the FULL frozen
//! wrong-transformation category (not a sample) for that same reactant
//! shape, to inform whether an `AmidationTemplate` is worth building.
//!
//! **Diagnostic only -- no chemistry code touched.** Reads the already-
//! frozen, already-committed v3 report (`uspto_evaluation_report_v3_aromatic_valence_fix/per_record_results.tsv`)
//! and re-parses each wrong-transformation record's reactants with the
//! crate's real SMILES parser (not a regex over the string, unlike the
//! ordering-bug estimate) to test for a genuine free amine. This is more
//! precise than a text heuristic, consistent with the project's general
//! preference for real structural checks over string matching, but it is
//! still a proxy for "the real reaction formed an amide" (it can't know
//! that without re-deriving the actual USPTO reaction mechanism) --
//! reported as "candidate for the pattern," not "confirmed."

use symthaea_organic_chemistry::smiles::{BondOrder, Molecule};

const REPORT_TSV: &str = include_str!(
    "../external_corpus/uspto_evaluation_report_v3_aromatic_valence_fix/per_record_results.tsv"
);

/// True if `m` has at least one free (unreacted) amine nitrogen: aliphatic
/// (not aromatic), neutral, at least one hydrogen (primary or secondary --
/// tertiary amines have no H to lose in a simple amide-forming
/// substitution), and NOT already bonded to a carbonyl carbon (which would
/// make it an existing amide/carbamate/urea nitrogen, not a free amine
/// competing with the alcohol for the *new* bond this reaction forms).
fn has_free_amine(m: &Molecule) -> bool {
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
            return true;
        }
    }
    false
}

struct Row {
    row_index: String,
    kind: String,
    category: String,
    reactants: String,
}

fn parse_tsv(s: &str) -> Vec<Row> {
    let mut lines = s.lines();
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let col = |name: &str| header.iter().position(|h| *h == name).unwrap();
    let (ri, ki, ci, rxi) = (
        col("row_index"),
        col("kind"),
        col("category"),
        col("reactants"),
    );
    lines
        .filter(|l| !l.is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            Row {
                row_index: f[ri].to_string(),
                kind: f[ki].to_string(),
                category: f[ci].to_string(),
                reactants: f[rxi].to_string(),
            }
        })
        .collect()
}

fn main() {
    let rows = parse_tsv(REPORT_TSV);
    let wrong: Vec<&Row> = rows
        .iter()
        .filter(|r| r.category == "structurally_shaped_wrong_transformation")
        .collect();

    let mut amine_candidates = Vec::new();
    let mut other = Vec::new();
    let mut unparseable = 0usize;

    for r in &wrong {
        let mols: Result<Vec<Molecule>, _> =
            r.reactants.split('.').map(Molecule::from_smiles).collect();
        let Ok(mols) = mols else {
            unparseable += 1;
            continue;
        };
        if mols.iter().any(has_free_amine) {
            amine_candidates.push(r.row_index.clone());
        } else {
            other.push(r.row_index.clone());
        }
    }

    let esterification_wrong = wrong.iter().filter(|r| r.kind == "esterification").count();
    let hydrogenation_wrong = wrong.iter().filter(|r| r.kind == "hydrogenation").count();
    let amine_esterification = wrong
        .iter()
        .filter(|r| r.kind == "esterification" && amine_candidates.contains(&r.row_index))
        .count();
    let amine_hydrogenation = wrong
        .iter()
        .filter(|r| r.kind == "hydrogenation" && amine_candidates.contains(&r.row_index))
        .count();

    println!("# Amine-competition scan of structurally_shaped_wrong_transformation\n");
    println!("total_wrong_transformation={}", wrong.len());
    println!(
        "amine_candidate={} ({:.1}%)",
        amine_candidates.len(),
        100.0 * amine_candidates.len() as f64 / wrong.len() as f64
    );
    println!(
        "other={} ({:.1}%)",
        other.len(),
        100.0 * other.len() as f64 / wrong.len() as f64
    );
    println!("unparseable={unparseable}\n");
    println!(
        "esterification: {}/{} amine-candidate",
        amine_esterification, esterification_wrong
    );
    println!(
        "hydrogenation:  {}/{} amine-candidate",
        amine_hydrogenation, hydrogenation_wrong
    );
    println!("\namine_candidate_row_indices={:?}", amine_candidates);
}
