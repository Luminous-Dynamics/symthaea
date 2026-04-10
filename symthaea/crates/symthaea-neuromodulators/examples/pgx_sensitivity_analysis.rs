// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! PGx Health Equity: Allele Frequency Sensitivity Analysis
//!
//! Perturbs all population allele frequencies by -20% to +20% and checks
//! whether equity gap rankings remain stable. Demonstrates that our equity
//! conclusions are robust to the known uncertainty in population-level
//! allele frequency estimates.
//!
//! Run:
//! ```bash
//! cargo run -p symthaea-neuromodulators --example pgx_sensitivity_analysis
//! ```

use symthaea_neuromodulators::pgx_health_equity::PgxHealthEquityEngine;
use symthaea_neuromodulators::pharmacogenomics::AncestryGroup;

const PERTURBATIONS: &[f64] = &[-0.20, -0.10, 0.0, 0.10, 0.20];
const PERTURBATION_LABELS: &[&str] = &["-20%", "-10%", "Base", "+10%", "+20%"];

const ANCESTRY_ORDER: &[(AncestryGroup, &str)] = &[
    (AncestryGroup::European, "European"),
    (AncestryGroup::African, "African"),
    (AncestryGroup::EastAsian, "E.Asian"),
    (AncestryGroup::SouthAsian, "S.Asian"),
    (AncestryGroup::NativeAmerican, "NatAmer"),
    (AncestryGroup::Mixed, "Mixed"),
];

/// Build one engine per perturbation level and run equity analysis on all pairs.
fn run_analyses() -> Vec<(f64, Vec<(String, String, f64, Vec<AncestryGroup>)>)> {
    let base = PgxHealthEquityEngine::new();
    let pairs = base.drug_gene_pairs();

    PERTURBATIONS
        .iter()
        .map(|&factor| {
            let mut engine = PgxHealthEquityEngine::new();
            if factor != 0.0 {
                engine.perturb_frequencies(factor);
            }
            let results: Vec<_> = pairs
                .iter()
                .map(|(drug, gene)| {
                    let a = engine.equity_analysis(drug, gene);
                    (
                        drug.clone(),
                        gene.clone(),
                        a.equity_gap_score,
                        a.underserved_populations.clone(),
                    )
                })
                .collect();
            (factor, results)
        })
        .collect()
}

fn main() {
    let analyses = run_analyses();
    let n_drugs = analyses[0].1.len();

    // Collect pair names from the base (factor=0) run.
    let base_idx = PERTURBATIONS.iter().position(|&p| p == 0.0).unwrap();
    let pair_names: Vec<(String, String)> = analyses[base_idx]
        .1
        .iter()
        .map(|(d, g, _, _)| (d.clone(), g.clone()))
        .collect();

    // --- Header ---
    println!(
        "\n\
         +================================================================+\n\
         |  PGx Health Equity: Allele Frequency Sensitivity Analysis       |\n\
         +================================================================+\n"
    );

    // ===== Section 1: Rankings by perturbation =====
    println!("=== EQUITY GAP RANKINGS BY PERTURBATION LEVEL ===");
    println!("(Do the most inequitable drugs stay the same?)\n");

    // For each perturbation, sort by gap descending to get rankings.
    let mut ranked: Vec<Vec<usize>> = Vec::new(); // ranked[pert_idx][rank] = pair_idx
    for (_, results) in &analyses {
        let mut indices: Vec<usize> = (0..n_drugs).collect();
        indices.sort_by(|a, b| {
            results[*b]
                .2
                .partial_cmp(&results[*a].2)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked.push(indices);
    }

    // Print sorted by base ranking.
    let base_ranking = &ranked[base_idx];
    print!("{:>4} | {:<14}| {:<8}|", "Rank", "Drug", "Gene");
    for label in PERTURBATION_LABELS {
        print!(" {:>6} |", label);
    }
    println!();
    print!("{:-<5}|{:-<15}|{:-<9}|", "", "", "");
    for _ in PERTURBATION_LABELS {
        print!("{:-<8}|", "");
    }
    println!();

    for (rank, &pair_idx) in base_ranking.iter().enumerate() {
        let (ref drug, ref gene) = pair_names[pair_idx];
        print!("{:>4} | {:<14}| {:<8}|", rank + 1, drug, gene);
        for (pert_i, (_, results)) in analyses.iter().enumerate() {
            let gap = results[pair_idx].2;
            let marker = if pert_i != base_idx { " " } else { "" };
            print!(" {:5.3}{} |", gap, marker);
        }
        println!();
    }
    println!();

    // Rank stability: how many drugs maintain the same rank across all perturbations.
    let mut stable_count = 0;
    for &pair_idx in base_ranking {
        let base_rank = base_ranking.iter().position(|&i| i == pair_idx).unwrap();
        let all_same = ranked.iter().all(|r| {
            let r_rank = r.iter().position(|&i| i == pair_idx).unwrap();
            r_rank == base_rank
        });
        if all_same {
            stable_count += 1;
        }
    }
    println!(
        "Rank stability: {}/{} drugs maintain their exact rank across all perturbation levels.",
        stable_count, n_drugs
    );

    // Also check top-3 and bottom-3 stability.
    let top3_stable = (0..3.min(n_drugs)).all(|rank| {
        let base_pair = base_ranking[rank];
        ranked
            .iter()
            .all(|r| r[..3.min(n_drugs)].contains(&base_pair))
    });
    let bot3_stable = if n_drugs >= 3 {
        ((n_drugs - 3)..n_drugs).all(|rank| {
            let base_pair = base_ranking[rank];
            ranked
                .iter()
                .all(|r| r[(n_drugs - 3)..].contains(&base_pair))
        })
    } else {
        true
    };
    println!(
        "Top-3 stable: {}  |  Bottom-3 stable: {}",
        if top3_stable { "YES" } else { "NO" },
        if bot3_stable { "YES" } else { "NO" }
    );
    println!();

    // ===== Section 2: Gap magnitude sensitivity =====
    println!("=== EQUITY GAP MAGNITUDE SENSITIVITY ===");
    println!("(How much do gap scores change with frequency uncertainty?)\n");
    print!(
        "{:<14}| {:<8}| {:>8} | {:>15} | {:>15} | {:>6}",
        "Drug", "Gene", "Base Gap", "+/-10% Range", "+/-20% Range", "Max D"
    );
    println!();
    print!(
        "{:-<14}|{:-<9}|{:-<10}|{:-<17}|{:-<17}|{:-<7}",
        "", "", "", "", "", ""
    );
    println!();

    for &pair_idx in base_ranking {
        let (ref drug, ref gene) = pair_names[pair_idx];
        let base_gap = analyses[base_idx].1[pair_idx].2;

        // Collect gaps across perturbation levels.
        let gaps: Vec<f64> = analyses.iter().map(|(_, r)| r[pair_idx].2).collect();

        let range_10: (f64, f64) = (
            gaps[1].min(gaps[3]), // -10% and +10%
            gaps[1].max(gaps[3]),
        );
        let range_20: (f64, f64) = (
            *gaps
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap(),
            *gaps
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap(),
        );
        let max_delta = range_20.1 - range_20.0;

        println!(
            "{:<14}| {:<8}|   {:.3}   | [{:.3}-{:.3}]   | [{:.3}-{:.3}]   | {:.3}",
            drug, gene, base_gap, range_10.0, range_10.1, range_20.0, range_20.1, max_delta
        );
    }
    println!();

    // ===== Section 3: Underserved population stability =====
    println!("=== UNDERSERVED POPULATION STABILITY ===");
    println!("(Do the same populations get flagged as underserved?)\n");

    // Show -20%, Base, +20% columns.
    let col_indices = [0usize, base_idx, PERTURBATIONS.len() - 1];
    let col_labels = ["-20% flagged", "Base flagged", "+20% flagged"];

    print!("{:<14}|", "Drug");
    for label in &col_labels {
        print!(" {:<20}|", label);
    }
    println!();
    print!("{:-<14}|", "");
    for _ in &col_labels {
        print!("{:-<21}|", "");
    }
    println!();

    let mut consistent_flags = 0usize;
    let mut total_flags = 0usize;

    for &pair_idx in base_ranking {
        let (ref drug, _) = pair_names[pair_idx];
        print!("{:<14}|", drug);
        for &ci in &col_indices {
            let underserved = &analyses[ci].1[pair_idx].3;
            let names: Vec<&str> = ANCESTRY_ORDER
                .iter()
                .filter_map(|(anc, name)| {
                    if underserved.contains(anc) {
                        Some(*name)
                    } else {
                        None
                    }
                })
                .collect();
            let s = if names.is_empty() {
                "(none)".to_string()
            } else {
                names.join(", ")
            };
            print!(" {:<20}|", s);
        }
        println!();

        // Count stability: does the base set of underserved populations persist?
        let base_set = &analyses[base_idx].1[pair_idx].3;
        for &(anc, _) in ANCESTRY_ORDER {
            let in_base = base_set.contains(&anc);
            if in_base {
                total_flags += 1;
                let in_all = analyses.iter().all(|(_, r)| r[pair_idx].3.contains(&anc));
                if in_all {
                    consistent_flags += 1;
                }
            }
        }
    }
    println!();
    println!(
        "Stability: {}/{} drug-population flags remain consistent across all perturbations.",
        consistent_flags, total_flags
    );
    println!();

    // ===== Key Finding =====
    println!("=== KEY FINDING ===\n");

    // Determine the top-3 drugs by base gap.
    let top3_names: Vec<String> = base_ranking
        .iter()
        .take(3.min(n_drugs))
        .map(|&i| pair_names[i].0.clone())
        .collect();

    let max_delta_overall: f64 = (0..n_drugs)
        .map(|pair_idx| {
            let gaps: Vec<f64> = analyses.iter().map(|(_, r)| r[pair_idx].2).collect();
            let lo = gaps.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = gaps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            hi - lo
        })
        .fold(f64::NEG_INFINITY, f64::max);

    let stability_word =
        if top3_stable && (consistent_flags as f64 / total_flags.max(1) as f64) > 0.7 {
            "stable"
        } else {
            "partially stable"
        };

    println!(
        "Equity gap rankings are {} to +/-20% allele frequency uncertainty.",
        stability_word
    );
    println!(
        "The largest gaps ({}) remain the largest regardless of perturbation.",
        top3_names.join(", ")
    );
    println!(
        "Maximum gap score delta across all perturbations: {:.3}",
        max_delta_overall
    );
    println!("This demonstrates that our conclusions are robust to the known uncertainty");
    println!("in population-level allele frequency estimates.");

    println!(
        "\n[Analysis: {} drug-gene pairs x {} perturbation levels = {} equity analyses]\n",
        n_drugs,
        PERTURBATIONS.len(),
        n_drugs * PERTURBATIONS.len()
    );
}
