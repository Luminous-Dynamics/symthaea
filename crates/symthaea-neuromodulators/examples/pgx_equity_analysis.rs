// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Pharmacogenomics Health Equity Analysis
//!
//! Generates comprehensive tables and figure data quantifying ancestry bias
//! in CPIC Level A drug dosing guidelines. Designed to produce data for
//! publication figures and supplementary tables.
//!
//! Run:
//! ```bash
//! cargo run -p symthaea-neuromodulators --example pgx_equity_analysis
//! ```

use symthaea_neuromodulators::pgx_health_equity::{AdmixedPopulation, PgxHealthEquityEngine};
use symthaea_neuromodulators::pharmacogenomics::{AncestryGroup, MetabolizerPhenotype};

/// US population estimates by ancestry (Census 2020, rounded to millions).
const US_POP: &[(AncestryGroup, f64, &str)] = &[
    (AncestryGroup::European, 191.0, "European"),
    (AncestryGroup::African, 41.0, "African"),
    (AncestryGroup::EastAsian, 19.0, "E.Asian"),
    (AncestryGroup::SouthAsian, 5.4, "S.Asian"),
    (AncestryGroup::NativeAmerican, 3.7, "NatAmer"),
    (AncestryGroup::Mixed, 33.8, "Mixed"),
];

const ANCESTRY_ORDER: &[(AncestryGroup, &str)] = &[
    (AncestryGroup::European, "European"),
    (AncestryGroup::African, "African"),
    (AncestryGroup::EastAsian, "E.Asian"),
    (AncestryGroup::SouthAsian, "S.Asian"),
    (AncestryGroup::NativeAmerican, "NatAmer"),
    (AncestryGroup::Mixed, "Mixed"),
];

const DRUG_GENE_PAIRS: &[(&str, &str)] = &[
    ("codeine", "CYP2D6"),
    ("clopidogrel", "CYP2C19"),
    ("warfarin", "CYP2C9"),
    ("sertraline", "CYP2C19"),
    ("efavirenz", "CYP2B6"),
    ("tamoxifen", "CYP2D6"),
    ("atomoxetine", "CYP2D6"),
    ("ondansetron", "CYP2D6"),
    ("voriconazole", "CYP2C19"),
    ("tacrolimus", "CYP3A5"),
    ("phenytoin", "CYP2C9"),
];

const GENES: &[&str] = &["CYP2D6", "CYP2C19", "CYP2C9", "CYP3A4", "CYP2B6", "CYP3A5"];

const PHENOTYPE_ORDER: &[(MetabolizerPhenotype, &str)] = &[
    (MetabolizerPhenotype::UltraRapid, "UltraRapid"),
    (MetabolizerPhenotype::Normal, "Normal"),
    (MetabolizerPhenotype::Intermediate, "Intermed"),
    (MetabolizerPhenotype::Poor, "Poor"),
];

fn ancestry_name(a: &AncestryGroup) -> &'static str {
    match a {
        AncestryGroup::European => "European",
        AncestryGroup::African => "African",
        AncestryGroup::EastAsian => "E.Asian",
        AncestryGroup::SouthAsian => "S.Asian",
        AncestryGroup::NativeAmerican => "NatAmer",
        AncestryGroup::Mixed => "Mixed",
    }
}

fn main() {
    let engine = PgxHealthEquityEngine::new();

    print_header();
    print_table1_metabolizer_distributions(&engine);
    print_table2_equity_gap(&engine);
    print_table3_clinical_impact(&engine);
    print_figure_heatmap(&engine);
    print_admixed_population_analysis(&engine);
    print_key_findings(&engine);
}

fn print_header() {
    println!(
        "\n\
         +==============================================================+\n\
         |  Pharmacogenomics Health Equity Analysis                      |\n\
         |  Quantifying ancestry bias in CPIC Level A drug dosing        |\n\
         +==============================================================+\n"
    );
}

fn print_table1_metabolizer_distributions(engine: &PgxHealthEquityEngine) {
    println!("=== TABLE 1: Metabolizer Phenotype Distributions ===");
    println!("(Hardy-Weinberg derived from validated allele frequencies)\n");

    // Header
    print!("{:<8}| {:<10}|", "Gene", "Phenotype");
    for (_, label) in ANCESTRY_ORDER {
        print!(" {:>8} |", label);
    }
    println!();

    print!("{:-<8}|{:-<11}|", "", "");
    for _ in ANCESTRY_ORDER {
        print!("{:-<11}|", "");
    }
    println!();

    for &gene in GENES {
        for &(pheno, pheno_label) in PHENOTYPE_ORDER {
            print!("{:<8}| {:<10}|", gene, pheno_label);
            for &(anc, _) in ANCESTRY_ORDER {
                let dist = engine.metabolizer_distribution(gene, anc);
                let frac = dist.get(&pheno).copied().unwrap_or(0.0);
                print!("   {:5.1}% |", frac * 100.0);
            }
            println!();
        }
        // Separator between genes
        print!("{:-<8}|{:-<11}|", "", "");
        for _ in ANCESTRY_ORDER {
            print!("{:-<11}|", "");
        }
        println!();
    }
    println!();
}

fn print_table2_equity_gap(engine: &PgxHealthEquityEngine) {
    println!("=== TABLE 2: Equity Gap Analysis by Drug ===\n");
    println!(
        "{:<14}| {:<8}| {:<10}| {:<24}| Underserved Populations",
        "Drug", "Gene", "Gap Score", "Most Affected"
    );
    println!("{:-<14}|{:-<9}|{:-<11}|{:-<25}|{:-<30}", "", "", "", "", "");

    for &(drug, gene) in DRUG_GENE_PAIRS {
        let analysis = engine.equity_analysis(drug, gene);

        // Find most affected ancestry: highest combined adverse + efficacy risk
        let most_affected = analysis
            .population_risk
            .iter()
            .max_by(|a, b| {
                let score_a = a.1.adverse_event_risk + a.1.efficacy_risk;
                let score_b = b.1.adverse_event_risk + b.1.efficacy_risk;
                score_a.partial_cmp(&score_b).unwrap()
            })
            .unwrap();

        // Find highest actionable phenotype frequency for the most affected group
        let dist = &most_affected.1.phenotype_distribution;
        let (worst_pheno, worst_frac) = dist
            .iter()
            .filter(|(p, _)| **p != MetabolizerPhenotype::Normal)
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        let pheno_tag = match worst_pheno {
            MetabolizerPhenotype::UltraRapid => "UM",
            MetabolizerPhenotype::Intermediate => "IM",
            MetabolizerPhenotype::Poor => "PM",
            MetabolizerPhenotype::Normal => "NM",
        };

        let affected_str = format!(
            "{} ({} {:.0}%)",
            ancestry_name(most_affected.0),
            pheno_tag,
            worst_frac * 100.0
        );

        let underserved_str: String = analysis
            .underserved_populations
            .iter()
            .map(|a| ancestry_name(a))
            .collect::<Vec<_>>()
            .join(", ");

        let underserved_display = if underserved_str.is_empty() {
            "(none)".to_string()
        } else {
            underserved_str
        };

        println!(
            "{:<14}| {:<8}|    {:.2}   | {:<24}| {}",
            drug, gene, analysis.equity_gap_score, affected_str, underserved_display
        );
    }
    println!();
}

fn print_table3_clinical_impact(engine: &PgxHealthEquityEngine) {
    println!("=== TABLE 3: Clinical Impact Estimates (US Population) ===\n");
    println!(
        "{:<14}| {:<8}| {:>13} | {:<24}| Risk Description",
        "Drug", "Gene", "Total At Risk", "Most Affected Group"
    );
    println!("{:-<14}|{:-<9}|{:-<15}|{:-<25}|{:-<30}", "", "", "", "", "");

    for &(drug, gene) in DRUG_GENE_PAIRS {
        let analysis = engine.equity_analysis(drug, gene);
        let is_prodrug = drug == "codeine" || drug == "clopidogrel" || drug == "tamoxifen";

        let mut total_at_risk = 0.0_f64;
        let mut worst_group = AncestryGroup::European;
        let mut worst_count = 0.0_f64;
        let mut worst_risk_desc = String::new();

        for &(anc, _, _label) in US_POP {
            let risk = match analysis.population_risk.get(&anc) {
                Some(r) => r,
                None => continue,
            };
            let dist = &risk.phenotype_distribution;
            let pop = US_POP.iter().find(|(a, _, _)| *a == anc).unwrap().1;

            // At-risk = those with actionable phenotypes
            let actionable_frac = if is_prodrug {
                // Prodrugs: UM (toxicity) + PM (no efficacy)
                dist.get(&MetabolizerPhenotype::UltraRapid)
                    .copied()
                    .unwrap_or(0.0)
                    + dist
                        .get(&MetabolizerPhenotype::Poor)
                        .copied()
                        .unwrap_or(0.0)
            } else {
                // Active drugs: PM (toxicity) + UM (sub-therapeutic)
                dist.get(&MetabolizerPhenotype::Poor)
                    .copied()
                    .unwrap_or(0.0)
                    + dist
                        .get(&MetabolizerPhenotype::UltraRapid)
                        .copied()
                        .unwrap_or(0.0)
            };

            let at_risk = pop * actionable_frac;
            total_at_risk += at_risk;

            if at_risk > worst_count {
                worst_count = at_risk;
                worst_group = anc;
                worst_risk_desc = if is_prodrug {
                    let um = dist
                        .get(&MetabolizerPhenotype::UltraRapid)
                        .copied()
                        .unwrap_or(0.0);
                    let pm = dist
                        .get(&MetabolizerPhenotype::Poor)
                        .copied()
                        .unwrap_or(0.0);
                    if um > pm {
                        "Toxicity risk (UM)".to_string()
                    } else {
                        "No efficacy (PM)".to_string()
                    }
                } else {
                    let pm = dist
                        .get(&MetabolizerPhenotype::Poor)
                        .copied()
                        .unwrap_or(0.0);
                    let um = dist
                        .get(&MetabolizerPhenotype::UltraRapid)
                        .copied()
                        .unwrap_or(0.0);
                    if pm > um {
                        "Toxicity risk (PM)".to_string()
                    } else {
                        "Sub-therapeutic (UM)".to_string()
                    }
                };
            }
        }

        println!(
            "{:<14}| {:<8}|      {:5.1}M   | {:<24}| {}",
            drug,
            gene,
            total_at_risk,
            format!("{} ({:.1}M)", ancestry_name(&worst_group), worst_count),
            worst_risk_desc
        );
    }
    println!();
}

fn print_figure_heatmap(engine: &PgxHealthEquityEngine) {
    println!("=== FIGURE DATA: Equity Gap Heatmap ===");
    println!("(Drug x Ancestry, values = adverse_event_risk + efficacy_risk)\n");

    // Header
    print!("{:<14}|", "");
    for (_, label) in ANCESTRY_ORDER {
        print!(" {:>8} |", label);
    }
    println!();
    print!("{:-<14}|", "");
    for _ in ANCESTRY_ORDER {
        print!("{:-<11}|", "");
    }
    println!();

    for &(drug, gene) in DRUG_GENE_PAIRS {
        let analysis = engine.equity_analysis(drug, gene);
        print!("{:<14}|", drug);
        for &(anc, _) in ANCESTRY_ORDER {
            let risk = analysis
                .population_risk
                .get(&anc)
                .map(|r| r.adverse_event_risk + r.efficacy_risk)
                .unwrap_or(0.0);
            print!("     {:.3}  |", risk);
        }
        println!();
    }
    println!();
}

fn print_admixed_population_analysis(engine: &PgxHealthEquityEngine) {
    println!(
        "\n\
         +==============================================================+\n\
         |  ADMIXED POPULATION ANALYSIS                                  |\n\
         |  Modeling real-world populations as ancestry mixtures          |\n\
         |  (Bryc et al. 2015; Salzano & Sans 2014)                      |\n\
         +==============================================================+\n"
    );

    let populations = AdmixedPopulation::all_presets();

    let admixed_genes = &["CYP2D6", "CYP2C19", "CYP3A5"];

    // Header
    print!("{:<24}| {:<32}|", "Population", "Composition");
    for &gene in admixed_genes {
        print!(" {} PM |", gene);
    }
    println!();
    print!("{:-<24}|{:-<33}|", "", "");
    for _ in admixed_genes {
        print!("{:-<12}|", "");
    }
    println!();

    for pop in &populations {
        let comp_str = pop.composition_string();
        print!("{:<24}| {:<32}|", pop.name, comp_str);
        for &gene in admixed_genes {
            let dist = engine.admixed_metabolizer_distribution(gene, pop);
            let pm = dist
                .get(&MetabolizerPhenotype::Poor)
                .copied()
                .unwrap_or(0.0);
            print!("    {:5.1}%  |", pm * 100.0);
        }
        println!();
    }

    // Comparison with pure ancestral groups
    println!();
    println!("--- Comparison: Pure Ancestral vs Admixed (CYP2D6 PM %) ---");
    println!();

    let pure_groups = [
        ("European (pure)", AncestryGroup::European),
        ("African (pure)", AncestryGroup::African),
        ("Native American (pure)", AncestryGroup::NativeAmerican),
    ];
    for (name, anc) in &pure_groups {
        let dist = engine.metabolizer_distribution("CYP2D6", *anc);
        let pm = dist
            .get(&MetabolizerPhenotype::Poor)
            .copied()
            .unwrap_or(0.0);
        println!("  {:<28} PM = {:5.1}%", name, pm * 100.0);
    }
    for pop in &populations {
        let dist = engine.admixed_metabolizer_distribution("CYP2D6", pop);
        let pm = dist
            .get(&MetabolizerPhenotype::Poor)
            .copied()
            .unwrap_or(0.0);
        println!("  {:<28} PM = {:5.1}%", pop.name, pm * 100.0);
    }

    // Equity gap analysis for admixed populations
    println!();
    println!("--- Admixed Population Equity Gaps ---");
    println!();
    println!(
        "{:<24}| {:<14}| {:<8}| Flags",
        "Population", "Drug/Gene", "Gap"
    );
    println!("{:-<24}|{:-<15}|{:-<9}|{:-<30}", "", "", "", "");

    for &(drug, gene) in DRUG_GENE_PAIRS {
        let results = engine.admixed_equity_analysis(drug, gene, &populations);
        for (name, gap, flags) in &results {
            let flag_str = if flags.is_empty() {
                "(none)".to_string()
            } else {
                flags.join(", ")
            };
            println!(
                "{:<24}| {:<6}/{:<6} |  {:.3}  | {}",
                name, drug, gene, gap, flag_str
            );
        }
    }
    println!();
}

fn print_key_findings(engine: &PgxHealthEquityEngine) {
    println!("=== KEY FINDINGS ===\n");

    // 1. Largest equity gap
    let mut largest_gap = 0.0_f64;
    let mut largest_drug = "";
    let mut largest_gene = "";
    for &(drug, gene) in DRUG_GENE_PAIRS {
        let gap = engine.equity_analysis(drug, gene).equity_gap_score;
        if gap > largest_gap {
            largest_gap = gap;
            largest_drug = drug;
            largest_gene = gene;
        }
    }
    println!(
        "1. Largest equity gap: {}/{} (score {:.3})",
        largest_drug, largest_gene, largest_gap
    );

    // 2. Most affected population overall (appears underserved most often)
    let mut underserved_counts: Vec<(AncestryGroup, usize)> = ANCESTRY_ORDER
        .iter()
        .map(|&(anc, _)| {
            let count = DRUG_GENE_PAIRS
                .iter()
                .filter(|&&(drug, gene)| {
                    engine
                        .equity_analysis(drug, gene)
                        .underserved_populations
                        .contains(&anc)
                })
                .count();
            (anc, count)
        })
        .collect();
    underserved_counts.sort_by(|a, b| b.1.cmp(&a.1));
    let (most_affected_anc, most_affected_count) = underserved_counts[0];
    println!(
        "2. Most affected population overall: {} (underserved in {}/{} drug-gene pairs)",
        ancestry_name(&most_affected_anc),
        most_affected_count,
        DRUG_GENE_PAIRS.len()
    );

    // 3. Total patients at risk across all guidelines
    let mut grand_total = 0.0_f64;
    for &(drug, gene) in DRUG_GENE_PAIRS {
        let analysis = engine.equity_analysis(drug, gene);
        let is_prodrug = drug == "codeine" || drug == "clopidogrel" || drug == "tamoxifen";
        for &(anc, pop, _) in US_POP {
            if let Some(risk) = analysis.population_risk.get(&anc) {
                let dist = &risk.phenotype_distribution;
                let frac = if is_prodrug {
                    dist.get(&MetabolizerPhenotype::UltraRapid)
                        .copied()
                        .unwrap_or(0.0)
                        + dist
                            .get(&MetabolizerPhenotype::Poor)
                            .copied()
                            .unwrap_or(0.0)
                } else {
                    dist.get(&MetabolizerPhenotype::Poor)
                        .copied()
                        .unwrap_or(0.0)
                        + dist
                            .get(&MetabolizerPhenotype::UltraRapid)
                            .copied()
                            .unwrap_or(0.0)
                };
                grand_total += pop * frac;
            }
        }
    }
    println!(
        "3. Total patients at risk across all Level A guidelines: {:.1} million (US)",
        grand_total
    );

    // 4. Gene with greatest inter-ancestry variation
    let mut max_variation = 0.0_f64;
    let mut max_var_gene = "";
    for &gene in GENES {
        let mut all_um = Vec::new();
        let mut all_pm = Vec::new();
        for &(anc, _) in ANCESTRY_ORDER {
            let dist = engine.metabolizer_distribution(gene, anc);
            all_um.push(
                dist.get(&MetabolizerPhenotype::UltraRapid)
                    .copied()
                    .unwrap_or(0.0),
            );
            all_pm.push(
                dist.get(&MetabolizerPhenotype::Poor)
                    .copied()
                    .unwrap_or(0.0),
            );
        }
        let um_range = all_um.iter().cloned().fold(0.0_f64, f64::max)
            - all_um.iter().cloned().fold(1.0_f64, f64::min);
        let pm_range = all_pm.iter().cloned().fold(0.0_f64, f64::max)
            - all_pm.iter().cloned().fold(1.0_f64, f64::min);
        let variation = um_range + pm_range;
        if variation > max_variation {
            max_variation = variation;
            max_var_gene = gene;
        }
    }
    println!(
        "4. Gene with greatest inter-ancestry variation: {} (UM+PM range {:.1}%)",
        max_var_gene,
        max_variation * 100.0
    );

    // 5. Guideline applicability summary
    println!("\n--- Guideline Applicability (1.0 = fully applicable) ---");
    for &(drug, gene) in DRUG_GENE_PAIRS {
        let analysis = engine.equity_analysis(drug, gene);
        print!("  {:<14} {:<8}: ", drug, gene);
        for &(anc, label) in ANCESTRY_ORDER {
            let applicability = analysis
                .population_risk
                .get(&anc)
                .map(|r| r.guideline_applicability)
                .unwrap_or(0.0);
            print!("{}={:.2}  ", label, applicability);
        }
        println!();
    }

    println!(
        "\n[Analysis engine: PgxHealthEquityEngine with {} allele frequencies, {} CPIC guidelines]",
        engine.allele_frequencies.len(),
        engine.cpic_guidelines.len()
    );
}
