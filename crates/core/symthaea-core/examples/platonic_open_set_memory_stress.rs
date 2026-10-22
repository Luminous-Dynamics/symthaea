// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Stress selective retrieval as the registered prototype memory grows.
//!
//! This is a controlled synthetic stress benchmark, not an SOTA claim. It holds
//! dimension and total prototype payload constant within each memory-size point,
//! then contrasts independent memories with memories containing correlated
//! near-duplicate entries. One fixed similarity/margin rule is diagnostic only;
//! it is not calibrated and carries no finite-sample risk guarantee.

use std::collections::BTreeMap;
use std::error::Error;
use std::time::Instant;

use symthaea_core::hdc::pattern_interaction::PatternVector;

const DIMENSION: usize = 4_096;
const PROTOTYPE_BUDGETS: [usize; 3] = [16, 64, 256];
const SEEDS: [u64; 3] = [17, 29, 41];
const QUERIES_PER_STRATUM: usize = 128;
const KNOWN_NOISE_LEVELS: [f64; 4] = [0.0, 0.10, 0.20, 0.30];
const NEAR_UNKNOWN_NOISE_LEVELS: [f64; 3] = [0.05, 0.10, 0.20];
const SHADOW_NOISE: f64 = 0.02;
const MIN_SIMILARITY: f64 = 0.25;
const MIN_TOP_TWO_MARGIN: f64 = 0.05;

type BenchResult<T> = Result<T, Box<dyn Error>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BankMode {
    Independent,
    CorrelatedShadows,
}

impl BankMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Independent => "independent",
            Self::CorrelatedShadows => "correlated_shadows",
        }
    }
}

#[derive(Debug, Clone)]
struct StoredPattern {
    label: String,
    vector: PatternVector,
}

#[derive(Debug, Clone)]
struct Query {
    vector: PatternVector,
    target_label: Option<String>,
    stratum: String,
}

#[derive(Debug, Clone, Copy)]
struct Observation {
    target_known: bool,
    top1_correct: bool,
    accepted: bool,
    best_similarity: f64,
    top_two_margin: f64,
    latency_us: f64,
}

fn main() -> BenchResult<()> {
    let started = Instant::now();
    println!("# protocol=platonic-open-set-memory-stress-v1");
    println!("# dimension={DIMENSION}");
    println!("# prototype_budgets=16,64,256; total stored entries held equal across bank modes");
    println!("# seeds=17,29,41; queries_per_stratum={QUERIES_PER_STRATUM}");
    println!("# known_noise=0,0.10,0.20,0.30; near_unknown_noise=0.05,0.10,0.20");
    println!(
        "# correlated_shadows=half the budget are independent bases, half are distinct labels 2% bit-flips from bases"
    );
    println!(
        "# decision_rule=similarity>={MIN_SIMILARITY},margin>={MIN_TOP_TWO_MARGIN}; fixed diagnostic only, not calibrated"
    );
    println!(
        "# vector_payload_bytes=prototype_count*dimension*sizeof(i8); excludes labels, allocations, runtime and indexes"
    );
    println!(
        "record_type,seed,dimension,prototype_budget,bank_mode,stratum,query_n,known_n,accepted_n,top1_correct,top1_accuracy,coverage,accepted_errors,selective_risk,selective_risk_upper95,unknown_false_accept_rate,unknown_false_accept_upper95,mean_best_similarity,mean_top_two_margin,p50_latency_us,p95_latency_us,vector_payload_bytes"
    );

    for seed in SEEDS {
        for prototype_budget in PROTOTYPE_BUDGETS {
            for mode in [BankMode::Independent, BankMode::CorrelatedShadows] {
                let (bank, bases) = build_bank(DIMENSION, prototype_budget, mode, seed)?;
                let queries = build_queries(&bases, seed, DIMENSION)?;
                let mut grouped: BTreeMap<String, Vec<Observation>> = BTreeMap::new();

                for query in &queries {
                    let started_query = Instant::now();
                    let (best_label, best_similarity, top_two_margin) = rank(&bank, &query.vector)?;
                    let latency_us = started_query.elapsed().as_nanos() as f64 / 1_000.0;
                    let accepted = accepts(best_similarity, top_two_margin);
                    let target_known = query.target_label.is_some();
                    let top1_correct = query
                        .target_label
                        .as_ref()
                        .is_some_and(|target| target == &best_label);
                    grouped
                        .entry(query.stratum.clone())
                        .or_default()
                        .push(Observation {
                            target_known,
                            top1_correct,
                            accepted,
                            best_similarity,
                            top_two_margin,
                            latency_us,
                        });
                }

                let vector_payload_bytes = bank.len() * DIMENSION * std::mem::size_of::<i8>();
                for (stratum, observations) in grouped {
                    report_group(
                        seed,
                        prototype_budget,
                        mode,
                        &stratum,
                        &observations,
                        vector_payload_bytes,
                    );
                }
            }
        }
    }

    println!("# total_runtime_ms={}", started.elapsed().as_millis());
    Ok(())
}

fn build_bank(
    dimension: usize,
    prototype_budget: usize,
    mode: BankMode,
    seed: u64,
) -> BenchResult<(Vec<StoredPattern>, Vec<(String, PatternVector)>)> {
    if prototype_budget < 4 || prototype_budget % 2 != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "prototype budget must be even and at least four",
        )
        .into());
    }

    let base_count = match mode {
        BankMode::Independent => prototype_budget,
        BankMode::CorrelatedShadows => prototype_budget / 2,
    };
    let mut bank = Vec::with_capacity(prototype_budget);
    let mut bases = Vec::with_capacity(base_count);

    for index in 0..base_count {
        let label = format!("id-{index:04}");
        let prototype =
            PatternVector::seeded_random(dimension, derive_seed(seed, 0x5052_4f54, index as u64))?;
        bases.push((label.clone(), prototype.clone()));
        bank.push(StoredPattern {
            label: label.clone(),
            vector: prototype.clone(),
        });

        if mode == BankMode::CorrelatedShadows {
            let shadow =
                prototype.with_noise(SHADOW_NOISE, derive_seed(seed, 0x5348_4144, index as u64))?;
            bank.push(StoredPattern {
                label: format!("shadow-{index:04}"),
                vector: shadow,
            });
        }
    }

    debug_assert_eq!(bank.len(), prototype_budget);
    Ok((bank, bases))
}

fn build_queries(
    bases: &[(String, PatternVector)],
    seed: u64,
    dimension: usize,
) -> BenchResult<Vec<Query>> {
    let mut queries = Vec::new();

    for (noise_index, noise_fraction) in KNOWN_NOISE_LEVELS.iter().copied().enumerate() {
        for query_index in 0..QUERIES_PER_STRATUM {
            let base_index = (query_index * 7 + noise_index + seed as usize) % bases.len();
            let (target_label, prototype) = &bases[base_index];
            let vector = prototype.with_noise(
                noise_fraction,
                derive_seed(seed, 0x4b4e_4f57 + noise_index as u64, query_index as u64),
            )?;
            queries.push(Query {
                vector,
                target_label: Some(target_label.clone()),
                stratum: format!(
                    "known_noise_{:02}",
                    (noise_fraction * 100.0).round() as usize
                ),
            });
        }
    }

    for query_index in 0..QUERIES_PER_STRATUM {
        queries.push(Query {
            vector: PatternVector::seeded_random(
                dimension,
                derive_seed(seed, 0x4f4f_4421, query_index as u64),
            )?,
            target_label: None,
            stratum: "random_ood".to_owned(),
        });
    }

    for query_index in 0..QUERIES_PER_STRATUM {
        let count = bases.len();
        let first = (query_index * 3 + seed as usize) % count;
        let second = (first + 1 + query_index % (count - 1)) % count;
        let mut third = (first + 2 + query_index % (count - 2)) % count;
        while third == first || third == second {
            third = (third + 1) % count;
        }
        let constituents = [
            bases[first].1.clone(),
            bases[second].1.clone(),
            bases[third].1.clone(),
        ];
        let mixture = PatternVector::bundle(
            &constituents,
            derive_seed(seed, 0x4d49_5854, query_index as u64),
        )?
        .with_noise(0.05, derive_seed(seed, 0x4d49_584e, query_index as u64))?;
        queries.push(Query {
            vector: mixture,
            target_label: None,
            stratum: "three_way_mixture".to_owned(),
        });
    }

    for (noise_index, noise_fraction) in NEAR_UNKNOWN_NOISE_LEVELS.iter().copied().enumerate() {
        for query_index in 0..QUERIES_PER_STRATUM {
            let base_index = (query_index * 5 + noise_index + seed as usize) % bases.len();
            let vector = bases[base_index].1.with_noise(
                noise_fraction,
                derive_seed(seed, 0x554e_4b4e + noise_index as u64, query_index as u64),
            )?;
            queries.push(Query {
                vector,
                target_label: None,
                stratum: format!(
                    "near_unknown_noise_{:02}",
                    (noise_fraction * 100.0).round() as usize
                ),
            });
        }
    }

    Ok(queries)
}

fn rank(bank: &[StoredPattern], query: &PatternVector) -> BenchResult<(String, f64, f64)> {
    let mut best_label = String::new();
    let mut best_similarity = f64::NEG_INFINITY;
    let mut runner_up_similarity = f64::NEG_INFINITY;

    for entry in bank {
        let similarity = query.similarity(&entry.vector)?;
        if similarity > best_similarity
            || (similarity == best_similarity && entry.label < best_label)
        {
            runner_up_similarity = best_similarity;
            best_similarity = similarity;
            best_label.clone_from(&entry.label);
        } else if similarity > runner_up_similarity {
            runner_up_similarity = similarity;
        }
    }

    let margin = if runner_up_similarity.is_finite() {
        best_similarity - runner_up_similarity
    } else {
        2.0
    };
    Ok((best_label, best_similarity, margin))
}

fn accepts(best_similarity: f64, top_two_margin: f64) -> bool {
    best_similarity >= MIN_SIMILARITY && top_two_margin >= MIN_TOP_TWO_MARGIN
}

fn report_group(
    seed: u64,
    prototype_budget: usize,
    mode: BankMode,
    stratum: &str,
    observations: &[Observation],
    vector_payload_bytes: usize,
) {
    let total = observations.len();
    let known_n = observations.iter().filter(|row| row.target_known).count();
    let accepted_n = observations.iter().filter(|row| row.accepted).count();
    let top1_correct = observations.iter().filter(|row| row.top1_correct).count();
    let accepted_errors = observations
        .iter()
        .filter(|row| row.accepted && (!row.target_known || !row.top1_correct))
        .count();
    let unknown_n = total - known_n;
    let unknown_accepted = observations
        .iter()
        .filter(|row| row.accepted && !row.target_known)
        .count();

    let coverage = ratio(accepted_n, total);
    let selective_risk = if accepted_n > 0 {
        format!("{:.6}", ratio(accepted_errors, accepted_n))
    } else {
        "NA".to_owned()
    };
    let selective_risk_upper95 = if accepted_n > 0 {
        format!(
            "{:.6}",
            wilson_upper_95(accepted_errors, accepted_n).expect("nonzero accepted count")
        )
    } else {
        "NA".to_owned()
    };
    let false_accept_rate = if unknown_n > 0 {
        format!("{:.6}", ratio(unknown_accepted, unknown_n))
    } else {
        "NA".to_owned()
    };
    let false_accept_upper95 = if unknown_n > 0 {
        format!(
            "{:.6}",
            wilson_upper_95(unknown_accepted, unknown_n).expect("nonzero unknown count")
        )
    } else {
        "NA".to_owned()
    };
    let top1_accuracy = if known_n > 0 {
        format!("{:.6}", ratio(top1_correct, known_n))
    } else {
        "NA".to_owned()
    };
    let top1_correct_field = if known_n > 0 {
        top1_correct.to_string()
    } else {
        "NA".to_owned()
    };
    let mean_best_similarity = observations
        .iter()
        .map(|row| row.best_similarity)
        .sum::<f64>()
        / total as f64;
    let mean_margin = observations
        .iter()
        .map(|row| row.top_two_margin)
        .sum::<f64>()
        / total as f64;
    let mut latencies = observations
        .iter()
        .map(|row| row.latency_us)
        .collect::<Vec<_>>();
    let p50_latency_us = percentile(&mut latencies, 0.50);
    let p95_latency_us = percentile(&mut latencies, 0.95);

    println!(
        "evaluation,{seed},{DIMENSION},{prototype_budget},{},{stratum},{total},{known_n},{accepted_n},{top1_correct_field},{top1_accuracy},{coverage:.6},{accepted_errors},{selective_risk},{selective_risk_upper95},{false_accept_rate},{false_accept_upper95},{mean_best_similarity:.6},{mean_margin:.6},{p50_latency_us:.4},{p95_latency_us:.4},{vector_payload_bytes}",
        mode.as_str()
    );
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

/// Wilson upper endpoint for a two-sided 95% interval. This is descriptive
/// only: repeated queries share stored prototypes, so it is not a clustered
/// or finite-sample risk certificate for a deployment distribution.
fn wilson_upper_95(successes: usize, trials: usize) -> Option<f64> {
    if trials == 0 || successes > trials {
        return None;
    }
    let n = trials as f64;
    let p = successes as f64 / n;
    let z = 1.959_963_984_540_054_f64;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = p + z2 / (2.0 * n);
    let radius = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt());
    Some((center + radius) / denominator)
}

fn percentile(values: &mut [f64], probability: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    let index = ((values.len() as f64 * probability).ceil() as usize)
        .saturating_sub(1)
        .min(values.len() - 1);
    values[index]
}

fn derive_seed(seed: u64, domain: u64, index: u64) -> u64 {
    mix64(seed ^ mix64(domain) ^ mix64(index.wrapping_add(0x9e37_79b9_7f4a_7c15)))
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wilson_upper_bound_exposes_uncertainty_when_no_errors_are_observed() {
        let upper = wilson_upper_95(0, 128).expect("nonzero sample size");
        assert!(upper > 0.0);
        assert!(upper < 0.05, "expected upper bound below 5%, got {upper}");
        assert!(wilson_upper_95(0, 0).is_none());
    }

    #[test]
    fn correlated_bank_holds_total_memory_budget_fixed() {
        let (independent, independent_bases) =
            build_bank(1_024, 16, BankMode::Independent, 17).expect("independent bank");
        let (correlated, correlated_bases) =
            build_bank(1_024, 16, BankMode::CorrelatedShadows, 17).expect("correlated bank");

        assert_eq!(independent.len(), 16);
        assert_eq!(correlated.len(), 16);
        assert_eq!(independent_bases.len(), 16);
        assert_eq!(correlated_bases.len(), 8);
    }

    #[test]
    fn correlated_shadow_entries_are_near_duplicates_by_construction() {
        let (bank, bases) =
            build_bank(1_024, 16, BankMode::CorrelatedShadows, 29).expect("correlated bank");

        for (label, base) in &bases {
            let shadow_label = label.replacen("id-", "shadow-", 1);
            let shadow = bank
                .iter()
                .find(|entry| entry.label == shadow_label)
                .expect("matching shadow");
            assert!(
                base.similarity(&shadow.vector).expect("similarity") >= 0.95,
                "shadow for {label} should be a near duplicate"
            );
        }
    }

    #[test]
    fn fixed_decision_rule_rejects_ambiguous_neighbours() {
        assert!(!accepts(0.95, 0.04));
        assert!(!accepts(0.20, 0.50));
        assert!(accepts(0.95, 0.20));
    }

    #[test]
    fn independent_memory_recalls_clean_queries_exactly() {
        let (bank, bases) =
            build_bank(1_024, 16, BankMode::Independent, 41).expect("independent bank");
        let (target, query) = &bases[3];
        let (nearest, score, margin) = rank(&bank, query).expect("rank");
        assert_eq!(&nearest, target);
        assert_eq!(score, 1.0);
        assert!(margin >= MIN_TOP_TWO_MARGIN);
        assert!(accepts(score, margin));
    }
}
