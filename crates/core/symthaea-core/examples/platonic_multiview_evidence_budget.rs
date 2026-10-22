// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Measure the value and hard limits of additional independent observations.
//!
//! Known patterns and near-duplicate unregistered latent patterns are sampled through
//! independent noisy views. Majority fusion is compared at 1, 3 and 9 views. An
//! exact-vector semantic twin reuses the known query's identical retrieval scores
//! while changing only the hidden semantic label. No vector-only rule may distinguish
//! this control. Threshold rows are descriptive sweeps, not risk certificates.

use std::collections::BTreeMap;
use std::error::Error;
use std::time::Instant;

use symthaea_core::hdc::pattern_interaction::{PatternSpaceError, PatternVector};

const DIMENSION: usize = 4_096;
const PROTOTYPE_COUNT: usize = 32;
const SEEDS: [u64; 5] = [13, 29, 41, 53, 67];
const REPLICATES_PER_IDENTITY: usize = 4;
const VIEW_COUNTS: [usize; 3] = [1, 3, 9];
const MAX_VIEWS: usize = 9;
const OBSERVATION_NOISE_LEVELS: [f64; 3] = [0.10, 0.20, 0.30];
const UNKNOWN_SHADOW_NOISE: f64 = 0.02;
const MIN_TOP_TWO_MARGIN: f64 = 0.02;
const SIMILARITY_THRESHOLDS: [f64; 8] = [0.70, 0.80, 0.90, 0.94, 0.96, 0.98, 0.99, 1.00];

type BenchResult<T> = Result<T, Box<dyn Error>>;
type GroupKey = (u64, usize, usize, &'static str);

#[derive(Debug, Clone, Copy)]
struct Score {
    correct_top1: Option<bool>,
    best_similarity: f64,
    top_two_margin: f64,
    fusion_us: f64,
    scan_us: f64,
}

#[derive(Debug, Clone, Copy)]
struct Summary {
    queries: usize,
    known_queries: usize,
    accepted: usize,
    correct_top1: usize,
    accepted_errors: usize,
    unknown_false_accepts: usize,
    mean_best_similarity: f64,
    mean_margin: f64,
    mean_fusion_us: f64,
    mean_scan_us: f64,
}

fn main() -> BenchResult<()> {
    let started = Instant::now();
    println!("# protocol=platonic-multiview-evidence-budget-v1");
    println!("# dimension={DIMENSION},registered_prototypes={PROTOTYPE_COUNT}");
    println!("# seeds=13,29,41,53,67;replicates_per_identity={REPLICATES_PER_IDENTITY}");
    println!("# views=1,3,9;observation_noise=0.10,0.20,0.30;near_unknown_latent_distance=0.02");
    println!("# score thresholds are a descriptive sweep; no threshold was selected on evaluation rows");
    println!("# exact_twin_unknown reuses the known query's ranked vector scores but has no registered identity");
    println!("# Wilson upper endpoints are descriptive IID approximations, not clustered risk certificates");
    println!("record_type,seed,noise_fraction,view_count,query_kind,score_threshold,margin_threshold,queries,known_queries,accepted,top1_correct,top1_accuracy,coverage,accepted_errors,selective_risk,risk_upper95,unknown_false_accepts,unknown_false_accept_rate,false_accept_upper95,mean_best_similarity,mean_top_two_margin,mean_fusion_us,mean_scan_us,stored_vector_payload_bytes,observed_view_payload_bytes");

    for seed in SEEDS {
        let bank = build_bank(seed)?;
        let mut scores: BTreeMap<GroupKey, Vec<Score>> = BTreeMap::new();

        for (noise_index, noise_fraction) in OBSERVATION_NOISE_LEVELS.iter().copied().enumerate() {
            for (identity_index, (identity, latent)) in bank.iter().enumerate() {
                let unknown_latent = latent.with_noise(
                    UNKNOWN_SHADOW_NOISE,
                    derive_seed(seed, 0x5348_4144, identity_index as u64),
                )?;

                for replicate in 0..REPLICATES_PER_IDENTITY {
                    let known_views = make_views(
                        latent, noise_fraction, seed, 0x4b4e_4f57 + noise_index as u64,
                        identity_index, replicate,
                    )?;
                    let unknown_views = make_views(
                        &unknown_latent, noise_fraction, seed, 0x554e_4b4e + noise_index as u64,
                        identity_index, replicate,
                    )?;

                    for view_count in VIEW_COUNTS {
                        let (known_query, known_fusion_us) = fuse_prefix(
                            &known_views,
                            view_count,
                            derive_seed(seed, 0x4655_534b, (identity_index * 100 + replicate) as u64),
                        )?;
                        let (known_rank, known_scan_us) = rank(&bank, &known_query)?;
                        let known = Score {
                            correct_top1: Some(known_rank.0.as_str() == identity),
                            best_similarity: known_rank.1,
                            top_two_margin: known_rank.2,
                            fusion_us: known_fusion_us,
                            scan_us: known_scan_us,
                        };
                        scores.entry((seed, noise_index, view_count, "known"))
                            .or_default().push(known);

                        // The semantic label changes; no observable vector value does.
                        scores.entry((seed, noise_index, view_count, "exact_twin_unknown"))
                            .or_default().push(Score { correct_top1: None, ..known });

                        let (unknown_query, unknown_fusion_us) = fuse_prefix(
                            &unknown_views,
                            view_count,
                            derive_seed(seed, 0x4655_4e4b, (identity_index * 100 + replicate) as u64),
                        )?;
                        let (unknown_rank, unknown_scan_us) = rank(&bank, &unknown_query)?;
                        scores.entry((seed, noise_index, view_count, "near_unknown"))
                            .or_default().push(Score {
                                correct_top1: None,
                                best_similarity: unknown_rank.1,
                                top_two_margin: unknown_rank.2,
                                fusion_us: unknown_fusion_us,
                                scan_us: unknown_scan_us,
                            });
                    }
                }
            }
        }

        for ((row_seed, noise_index, view_count, query_kind), values) in scores {
            for threshold in SIMILARITY_THRESHOLDS {
                let summary = summarize(&values, query_kind, threshold, MIN_TOP_TWO_MARGIN);
                let noise_fraction = OBSERVATION_NOISE_LEVELS[noise_index];
                let risk = ratio(summary.accepted_errors, summary.accepted);
                let risk_upper = if summary.accepted > 0 {
                    format!("{:.6}", wilson_upper_95(summary.accepted_errors, summary.accepted).expect("nonzero accepted count"))
                } else {
                    "NA".to_owned()
                };
                let false_accept_rate = if summary.known_queries == 0 {
                    ratio(summary.unknown_false_accepts, summary.queries)
                } else {
                    0.0
                };
                let false_accept_upper = if summary.known_queries == 0 {
                    format!("{:.6}", wilson_upper_95(summary.unknown_false_accepts, summary.queries).expect("nonzero unknown query count"))
                } else {
                    "NA".to_owned()
                };
                let top1_accuracy = if summary.known_queries > 0 {
                    format!("{:.6}", ratio(summary.correct_top1, summary.known_queries))
                } else {
                    "NA".to_owned()
                };
                let stored_bytes = PROTOTYPE_COUNT * DIMENSION * std::mem::size_of::<i8>();
                let observed_bytes = view_count * DIMENSION * std::mem::size_of::<i8>();

                println!(
                    "evaluation,{row_seed},{noise_fraction:.2},{view_count},{query_kind},{threshold:.2},{MIN_TOP_TWO_MARGIN:.2},{},{},{},{},{top1_accuracy},{:.6},{},{:.6},{risk_upper},{},{false_accept_rate:.6},{false_accept_upper},{:.6},{:.6},{:.4},{:.4},{stored_bytes},{observed_bytes}",
                    summary.queries,
                    summary.known_queries,
                    summary.accepted,
                    if summary.known_queries > 0 { summary.correct_top1.to_string() } else { "NA".to_owned() },
                    ratio(summary.accepted, summary.queries),
                    summary.accepted_errors,
                    risk,
                    summary.unknown_false_accepts,
                    summary.mean_best_similarity,
                    summary.mean_margin,
                    summary.mean_fusion_us,
                    summary.mean_scan_us,
                );
            }
        }
    }

    println!("# total_runtime_ms={}", started.elapsed().as_millis());
    Ok(())
}

fn build_bank(seed: u64) -> BenchResult<Vec<(String, PatternVector)>> {
    let mut bank = Vec::with_capacity(PROTOTYPE_COUNT);
    for index in 0..PROTOTYPE_COUNT {
        bank.push((
            format!("identity-{index:04}"),
            PatternVector::seeded_random(DIMENSION, derive_seed(seed, 0x5052_4f54, index as u64))?,
        ));
    }
    Ok(bank)
}

fn make_views(
    latent: &PatternVector,
    noise_fraction: f64,
    seed: u64,
    domain: u64,
    identity_index: usize,
    replicate: usize,
) -> BenchResult<Vec<PatternVector>> {
    (0..MAX_VIEWS)
        .map(|view_index| {
            let index = ((identity_index * REPLICATES_PER_IDENTITY + replicate) * MAX_VIEWS
                + view_index) as u64;
            Ok(latent.with_noise(
                noise_fraction,
                derive_seed(seed, domain + identity_index as u64, index),
            )?)
        })
        .collect()
}

fn fuse_prefix(
    views: &[PatternVector],
    view_count: usize,
    tie_seed: u64,
) -> Result<(PatternVector, f64), PatternSpaceError> {
    let started = Instant::now();
    let query = if view_count == 1 {
        views.first().ok_or(PatternSpaceError::EmptyVector)?.clone()
    } else {
        PatternVector::bundle(
            views.get(..view_count).ok_or(PatternSpaceError::EmptyVector)?,
            tie_seed,
        )?
    };
    Ok((query, started.elapsed().as_nanos() as f64 / 1_000.0))
}

fn rank(
    bank: &[(String, PatternVector)],
    query: &PatternVector,
) -> Result<((String, f64, f64), f64), PatternSpaceError> {
    let started = Instant::now();
    let mut best_label = String::new();
    let mut best_similarity = f64::NEG_INFINITY;
    let mut runner_up = f64::NEG_INFINITY;
    for (label, prototype) in bank {
        let similarity = query.similarity(prototype)?;
        if similarity > best_similarity || (similarity == best_similarity && label < &best_label) {
            runner_up = best_similarity;
            best_similarity = similarity;
            best_label.clone_from(label);
        } else if similarity > runner_up {
            runner_up = similarity;
        }
    }
    let margin = if runner_up.is_finite() { best_similarity - runner_up } else { 2.0 };
    Ok(((best_label, best_similarity, margin), started.elapsed().as_nanos() as f64 / 1_000.0))
}

fn summarize(values: &[Score], query_kind: &str, threshold: f64, margin_threshold: f64) -> Summary {
    let mut summary = Summary {
        queries: values.len(), known_queries: 0, accepted: 0, correct_top1: 0,
        accepted_errors: 0, unknown_false_accepts: 0, mean_best_similarity: 0.0,
        mean_margin: 0.0, mean_fusion_us: 0.0, mean_scan_us: 0.0,
    };

    for score in values {
        summary.mean_best_similarity += score.best_similarity;
        summary.mean_margin += score.top_two_margin;
        summary.mean_fusion_us += score.fusion_us;
        summary.mean_scan_us += score.scan_us;
        if let Some(correct) = score.correct_top1 {
            summary.known_queries += 1;
            summary.correct_top1 += usize::from(correct);
        }

        if accepts(score.best_similarity, score.top_two_margin, threshold, margin_threshold) {
            summary.accepted += 1;
            if score.correct_top1 != Some(true) {
                summary.accepted_errors += 1;
            }
            if query_kind != "known" {
                summary.unknown_false_accepts += 1;
            }
        }
    }

    let n = summary.queries.max(1) as f64;
    summary.mean_best_similarity /= n;
    summary.mean_margin /= n;
    summary.mean_fusion_us /= n;
    summary.mean_scan_us /= n;
    summary
}

fn accepts(score: f64, margin: f64, threshold: f64, margin_threshold: f64) -> bool {
    score >= threshold && margin >= margin_threshold
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 { 0.0 } else { numerator as f64 / denominator as f64 }
}

/// Wilson upper endpoint of a two-sided 95% binomial confidence interval.
/// Descriptive only: observations share latent identities and are not IID.
fn wilson_upper_95(successes: usize, trials: usize) -> Option<f64> {
    if trials == 0 || successes > trials { return None; }
    let n = trials as f64;
    let p = successes as f64 / n;
    let z = 1.959_963_984_540_054_f64;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = p + z2 / (2.0 * n);
    let radius = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt());
    Some((center + radius) / denominator)
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
    fn majority_fusion_improves_a_fixed_noisy_pattern() {
        let pattern = PatternVector::seeded_random(DIMENSION, 211).expect("pattern");
        let views = make_views(&pattern, 0.20, 223, 0x5445_5354, 0, 0).expect("views");
        let single = views[0].similarity(&pattern).expect("single similarity");
        let fused = PatternVector::bundle(&views, 227).expect("fused observation");
        let fused_similarity = fused.similarity(&pattern).expect("fused similarity");
        assert!(fused_similarity > single + 0.10, "single={single}, fused={fused_similarity}");
    }

    #[test]
    fn exact_vector_twins_have_identical_acceptance_for_every_threshold() {
        let pattern = PatternVector::seeded_random(DIMENSION, 229).expect("pattern");
        let bank = build_bank(233).expect("bank");
        let (ranked, _) = rank(&bank, &pattern).expect("rank");
        for threshold in SIMILARITY_THRESHOLDS {
            let known_decision = accepts(ranked.1, ranked.2, threshold, MIN_TOP_TWO_MARGIN);
            let unknown_twin_decision = accepts(ranked.1, ranked.2, threshold, MIN_TOP_TWO_MARGIN);
            assert_eq!(known_decision, unknown_twin_decision);
        }
    }

    #[test]
    fn similarity_threshold_sweep_has_monotone_acceptance() {
        let score = 0.963;
        let margin = 0.4;
        let counts = SIMILARITY_THRESHOLDS.iter()
            .map(|threshold| accepts(score, margin, *threshold, MIN_TOP_TWO_MARGIN))
            .collect::<Vec<_>>();
        assert!(counts.windows(2).all(|pair| !pair[1] || pair[0]));
    }

    #[test]
    fn wilson_bound_does_not_report_zero_uncertainty() {
        let upper = wilson_upper_95(0, 64).expect("nonzero sample");
        assert!(upper > 0.0);
        assert!(upper < 0.10);
    }
}
