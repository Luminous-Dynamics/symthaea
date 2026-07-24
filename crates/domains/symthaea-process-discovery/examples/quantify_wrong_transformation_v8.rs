// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phase A.9: full quantification of `structurally_shaped_wrong_transformation`
//! on the current frozen corpus (v8, element-scope widening -- 104 records:
//! 12 esterification + 92 hydrogenation). Prior phases (A.4, A.5) already
//! fixed the two dominant real bugs behind this category (esterification's
//! amine-vs-alcohol competition, hydrogenation's single-vs-exhaustive
//! saturation); this asks what's left in what remains, using the real SMILES
//! parser and real graph-theoretic ring counting (cyclomatic number:
//! `bonds - atoms + 1` per connected fragment), not string heuristics.
//!
//! **Diagnostic only -- no chemistry code touched.**
//!
//! Two automated buckets, both computed from real parsed structure:
//! - `small_declared_product`: declared product's real atom count is a small
//!   fraction of the combined reactants' -- a proxy for the "declared
//!   product is actually a reagent/workup byproduct" USPTO extraction
//!   artifact already found twice before (TFA, once each in the A.1 and A.6
//!   closeouts).
//! - `ring_count_increased`: declared product's real ring count (summed
//!   cyclomatic number across its connected fragments) exceeds the combined
//!   reactants' -- a proxy for the "SMARTS pre-filter only sees a C=C, but
//!   the real reaction is an intramolecular cyclization/aromatization, not
//!   hydrogenation" artifact class first found in Phase A.5's own
//!   quantification.
//!
//! Everything not caught by either bucket was individually hand-inspected
//! (not sampled -- all of them, small enough); see
//! `PROCESS_DISCOVERY_PHASE_A9_WRONG_TRANSFORMATION_CLOSEOUT_2026-07-18.md`
//! for the per-record mechanistic classification.

use symthaea_organic_chemistry::smiles::Molecule;

const REPORT_TSV: &str = include_str!(
    "../external_corpus/uspto_evaluation_report_v8_element_scope/per_record_results.tsv"
);

struct Row {
    row_index: String,
    kind: String,
    category: String,
    ambiguous_site: bool,
    reactants: String,
    product: String,
}

fn parse_tsv(s: &str) -> Vec<Row> {
    let mut lines = s.lines();
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let col = |name: &str| header.iter().position(|h| *h == name).unwrap();
    let (ri, ki, ci, ai, rxi, pi) = (
        col("row_index"),
        col("kind"),
        col("category"),
        col("ambiguous_site"),
        col("reactants"),
        col("product"),
    );
    lines
        .filter(|l| !l.is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            Row {
                row_index: f[ri].to_string(),
                kind: f[ki].to_string(),
                category: f[ci].to_string(),
                ambiguous_site: f[ai] == "true",
                reactants: f[rxi].to_string(),
                product: f[pi].to_string(),
            }
        })
        .collect()
}

/// Real atom count and real ring count (sum of per-fragment cyclomatic
/// numbers `bonds - atoms + 1`) for a `.`-joined SMILES string. Returns
/// `None` if any fragment fails to parse.
fn atoms_and_rings(smiles: &str) -> Option<(usize, usize)> {
    let mut total_atoms = 0usize;
    let mut total_rings = 0usize;
    for frag in smiles.split('.') {
        let m = Molecule::from_smiles(frag).ok()?;
        let n_atoms = m.atoms.len();
        let n_bonds: usize = (0..n_atoms).map(|i| m.neighbors(i).len()).sum::<usize>() / 2;
        total_atoms += n_atoms;
        // cyclomatic number for one connected fragment; saturate at 0 in
        // case of any parser edge case rather than underflow-panic.
        total_rings += n_bonds.saturating_sub(n_atoms).saturating_add(1);
        if n_bonds == 0 && n_atoms <= 1 {
            total_rings = total_rings.saturating_sub(1); // single atom, no ring
        }
    }
    Some((total_atoms, total_rings))
}

fn main() {
    let rows = parse_tsv(REPORT_TSV);
    let wrong: Vec<&Row> = rows
        .iter()
        .filter(|r| r.category == "structurally_shaped_wrong_transformation")
        .collect();

    let mut already_ambiguous = Vec::new();
    let mut small_declared_product = Vec::new();
    let mut ring_count_increased = Vec::new();
    let mut other = Vec::new();
    let mut unparseable = Vec::new();

    for r in &wrong {
        if r.ambiguous_site {
            already_ambiguous.push(r.row_index.clone());
            continue;
        }
        let Some((reactant_atoms, reactant_rings)) = atoms_and_rings(&r.reactants) else {
            unparseable.push(r.row_index.clone());
            continue;
        };
        let Some((product_atoms, product_rings)) = atoms_and_rings(&r.product) else {
            unparseable.push(r.row_index.clone());
            continue;
        };
        if product_atoms > 0 && (product_atoms as f64) < 0.4 * (reactant_atoms as f64) {
            small_declared_product.push(r.row_index.clone());
        } else if product_rings > reactant_rings {
            ring_count_increased.push(r.row_index.clone());
        } else {
            other.push(r.row_index.clone());
        }
    }

    println!(
        "# Phase A.9: structurally_shaped_wrong_transformation full quantification (v8, n=104)\n"
    );
    println!("total_wrong_transformation={}", wrong.len());
    println!(
        "already_ambiguous_site={} ({:.1}%) -- known class, Phase A.5",
        already_ambiguous.len(),
        100.0 * already_ambiguous.len() as f64 / wrong.len() as f64
    );
    println!(
        "small_declared_product={} ({:.1}%) -- reagent/byproduct extraction artifact, real atom-count check",
        small_declared_product.len(),
        100.0 * small_declared_product.len() as f64 / wrong.len() as f64
    );
    println!(
        "ring_count_increased={} ({:.1}%) -- cyclization/aromatization mis-tagged by SMARTS pre-filter, real ring-count check",
        ring_count_increased.len(),
        100.0 * ring_count_increased.len() as f64 / wrong.len() as f64
    );
    println!(
        "other={} ({:.1}%) -- individually hand-inspected, see closeout doc",
        other.len(),
        100.0 * other.len() as f64 / wrong.len() as f64
    );
    println!("unparseable={}", unparseable.len());
    println!("\nother_row_indices={other:?}");
    println!(
        "\nby kind: esterification={}/{}, hydrogenation={}/{}",
        wrong.iter().filter(|r| r.kind == "esterification").count(),
        wrong.len(),
        wrong.iter().filter(|r| r.kind == "hydrogenation").count(),
        wrong.len(),
    );
}
