// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Measure open-set retrieval failure under hostile prototype enrollment.
//!
//! This synthetic insertion-attack benchmark adds attacker-labelled copies of
//! genuine prototypes at a fixed vector dimension. It reports attack success,
//! clean accuracy, selective coverage/risk, memory payload and exact-scan cost.
//! Similarity thresholds are frozen diagnostics, not a risk certificate.

use std::error::Error;
use std::time::Instant;

use symthaea_core::hdc::pattern_interaction::PatternVector;

const DIMENSION: usize = 1_024;
const CLASS_COUNT: usize = 16;
const QUERIES_PER_CLASS: usize = 8;
const SEEDS: [u64; 3] = [17, 29, 41];
const POISON_COUNTS: [usize; 5] = [0, 1, 2, 4, 8];
const POISON_NOISE_LEVELS: [f64; 3] = [0.0, 0.02, 0.05];
const QUERY_NOISE_LEVELS: [f64; 3] = [0.0, 0.10, 0.20];
const MIN_SIMILARITY: f64 = 0.25;
const MIN_TOP_TWO_MARGIN: f64 = 0.05;

type BenchResult<T> = Result<T, Box<dyn Error>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttackMode {
    Insertion,
    Replacement,
}

impl AttackMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Insertion => "insertion",
            Self::Replacement => "replacement",
        }
    }
}

#[derive(Debug, Clone)]
struct Entry {
    label: String,
    vector: PatternVector,
    attacker_target: Option<usize>,
}

#[derive(Debug, Clone)]
struct Query {
    target_class: usize,
    noise_index: usize,
    vector: PatternVector,
}

#[derive(Debug, Clone)]
struct Ranked {
    label: String,
    similarity: f64,
    margin: f64,
    attacker_target: Option<usize>,
}

fn main() -> BenchResult<()> {
    let started = Instant::now();

    println!("# protocol=platonic-registry-poisoning-stress-v1");
    println!("# dimension={DIMENSION}; genuine_classes={CLASS_COUNT}; queries_per_class={QUERIES_PER_CLASS}");
    println!("# seeds=17,29,41; poison_counts=0,1,2,4,8; poison_noise=0,0.02,0.05");
    println!("# query_noise=0,0.10,0.20; same genuine query corpus reused across attack profiles");
    println!("# attack_modes=insertion (extra entries) and replacement (fixed entry count, genuine target class removed)");
    println!("# acceptance=similarity>={MIN_SIMILARITY},top_two_margin>={MIN_TOP_TWO_MARGIN}; fixed diagnostic, not calibrated");
    println!("# payload=entry_count*dimension*sizeof(i8); excludes labels, allocator, runtime, and indexes");
    println!("seed,attack_mode,dimension,genuine_classes,entry_count,poison_count,poison_noise,query_noise,query_count,attacked_classes,targeted_query_count,targeted_top1_correct,targeted_top1_accuracy,targeted_accepted_count,targeted_coverage,targeted_false_rejections,targeted_false_rejection_rate,target_attack_success_count,target_attack_success_rate,accepted_target_attack_success_count,accepted_target_attack_success_rate,retained_query_count,retained_top1_correct,retained_top1_accuracy,retained_accepted_count,retained_coverage,top1_correct,top1_accuracy,accepted_count,coverage,accepted_errors,selective_risk,attacker_prediction_count,attacker_prediction_rate,p50_latency_us,p95_latency_us,scan_wall_ms,scan_qps,vector_payload_bytes");

    for seed in SEEDS {
        let prototypes = build_prototypes(seed)?;
        let queries = build_queries(&prototypes, seed)?;
        let target_classes: Vec<usize> = (0..CLASS_COUNT).collect();

        for attack_mode in [AttackMode::Insertion, AttackMode::Replacement] {
            for poison_count in POISON_COUNTS {
                for poison_noise in POISON_NOISE_LEVELS {
                if poison_count == 0 && poison_noise != 0.0 {
                    continue;
                }

                let build_started = Instant::now();
                let (bank, attacked_classes) =
                    build_bank(&prototypes, seed, poison_count, poison_noise, attack_mode)?;
                let build_ms = build_started.elapsed().as_secs_f64() * 1_000.0;
                let attacked_class_count = attacked_classes.len();

                for noise_index in 0..QUERY_NOISE_LEVELS.len() {
                    let query_group = queries
                        .iter()
                        .filter(|query| query.noise_index == noise_index)
                        .collect::<Vec<_>>();
                    let mut top1_correct = 0usize;
                    let mut accepted_count = 0usize;
                    let mut accepted_errors = 0usize;
                    let mut attacker_prediction_count = 0usize;
                    let mut targeted_query_count = 0usize;
                    let mut target_attack_success_count = 0usize;
                    let mut targeted_accepted_count = 0usize;
                    let mut targeted_top1_correct = 0usize;
                    let mut accepted_target_attack_success_count = 0usize;
                    let mut retained_query_count = 0usize;
                    let mut retained_top1_correct = 0usize;
                    let mut retained_accepted_count = 0usize;
                    let mut latencies = Vec::with_capacity(query_group.len());

                    let scan_started = Instant::now();
                    for query in &query_group {
                        let query_started = Instant::now();
                        let ranked = rank(&bank, &query.vector)?;
                        latencies.push(
                            query_started.elapsed().as_nanos() as f64 / 1_000.0,
                        );

                        let expected_label = format!("id-{:02}", query.target_class);
                        let correct = ranked.label == expected_label;
                        if correct {
                            top1_correct += 1;
                        }
                        let accepted = accepts(ranked.similarity, ranked.margin);
                        if accepted {
                            accepted_count += 1;
                            if !correct {
                                accepted_errors += 1;
                            }
                        }

                        if ranked.attacker_target.is_some() {
                            attacker_prediction_count += 1;
                        }

                        let is_targeted = attacked_classes.contains(&query.target_class);
                        if is_targeted {
                            targeted_query_count += 1;
                            if correct {
                                targeted_top1_correct += 1;
                            }
                            if accepted {
                                targeted_accepted_count += 1;
                            }
                            let successful_attack =
                                ranked.attacker_target == Some(query.target_class);
                            if successful_attack {
                                target_attack_success_count += 1;
                                if accepted {
                                    accepted_target_attack_success_count += 1;
                                }
                            }
                        } else {
                            retained_query_count += 1;
                            if correct {
                                retained_top1_correct += 1;
                            }
                            if accepted {
                                retained_accepted_count += 1;
                            }
                        }
                    }
                    let scan_wall_ms = scan_started.elapsed().as_secs_f64() * 1_000.0;
                    let scan_qps = if scan_wall_ms > 0.0 {
                        query_group.len() as f64 / (scan_wall_ms / 1_000.0)
                    } else {
                        0.0
                    };
                    let accepted_target_attack_success_rate = if targeted_accepted_count > 0 {
                        Some(
                            accepted_target_attack_success_count as f64
                                / targeted_accepted_count as f64,
                        )
                    } else {
                        None
                    };
                    let vector_payload_bytes =
                        bank.len() * DIMENSION * std::mem::size_of::<i8>();

                    let targeted_false_rejections =
                        targeted_query_count.saturating_sub(targeted_accepted_count);
                    let retained_top1_accuracy =
                        optional_ratio(retained_top1_correct, retained_query_count);
                    let retained_coverage =
                        optional_ratio(retained_accepted_count, retained_query_count);
                    let targeted_top1_accuracy =
                        optional_ratio(targeted_top1_correct, targeted_query_count);
                    let targeted_coverage =
                        optional_ratio(targeted_accepted_count, targeted_query_count);
                    let targeted_false_rejection_rate =
                        optional_ratio(targeted_false_rejections, targeted_query_count);

                    println!(
                        "{seed},{},{DIMENSION},{CLASS_COUNT},{},{poison_count},{poison_noise:.3},{:.3},{},{attacked_class_count},{targeted_query_count},{targeted_top1_correct},{targeted_top1_accuracy},{targeted_accepted_count},{targeted_coverage},{targeted_false_rejections},{targeted_false_rejection_rate},{target_attack_success_count},{},{accepted_target_attack_success_count},{},{retained_query_count},{retained_top1_correct},{retained_top1_accuracy},{retained_accepted_count},{retained_coverage},{top1_correct},{:.6},{accepted_count},{:.6},{accepted_errors},{},{attacker_prediction_count},{:.6},{:.4},{:.4},{scan_wall_ms:.3},{scan_qps:.2},{vector_payload_bytes}",
                        attack_mode.as_str(),
                        bank.len(),
                        QUERY_NOISE_LEVELS[noise_index],
                        query_group.len(),
                        optional_ratio(target_attack_success_count, targeted_query_count),
                        optional_rate(accepted_target_attack_success_rate),
                        ratio(top1_correct, query_group.len()),
                        ratio(accepted_count, query_group.len()),
                        optional_ratio(accepted_errors, accepted_count),
                        ratio(attacker_prediction_count, query_group.len()),
                        percentile(&mut latencies, 0.50),
                        percentile(&mut latencies, 0.95),
                    );

                    // Include construction as a comment-level per-profile resource metric.
                    println!(
                        "# profile_runtime=seed:{seed},attack_mode:{},poison_count:{poison_count},poison_noise:{poison_noise:.3},query_noise:{:.3},bank_build_ms:{build_ms:.3},entry_count:{}",
                        attack_mode.as_str(),
                        QUERY_NOISE_LEVELS[noise_index],
                        bank.len()
                    );
                }
            }
        }
        }
        // Keep this named value visible to make the fixed genuine-class population explicit.
        debug_assert_eq!(target_classes.len(), CLASS_COUNT);
    }

    println!("# total_runtime_ms={}", started.elapsed().as_millis());
    Ok(())
}

fn build_prototypes(seed: u64) -> BenchResult<Vec<PatternVector>> {
    (0..CLASS_COUNT)
        .map(|class| {
            PatternVector::seeded_random(
                DIMENSION,
                derive_seed(seed, 0x5052_4f54, class as u64),
            )
            .map_err(Into::into)
        })
        .collect()
}

fn build_bank(
    prototypes: &[PatternVector],
    seed: u64,
    poison_count: usize,
    poison_noise: f64,
    attack_mode: AttackMode,
) -> BenchResult<(Vec<Entry>, Vec<usize>)> {
    if poison_count > CLASS_COUNT / 2 {
        return Err("poison count exceeds this protocol's unique-target budget".into());
    }
    let mut attacked_classes = Vec::with_capacity(poison_count);
    for poison_index in 0..poison_count {
        // Seven is coprime with 16, so this produces distinct target classes
        // throughout the declared 0..=8 attack budget.
        let target_class = (poison_index * 7 + seed as usize) % CLASS_COUNT;
        attacked_classes.push(target_class);
    }

    let extra_entries = if attack_mode == AttackMode::Insertion {
        poison_count
    } else {
        0
    };
    let mut bank = Vec::with_capacity(prototypes.len() + extra_entries);
    for (class, prototype) in prototypes.iter().enumerate() {
        if attack_mode == AttackMode::Replacement && attacked_classes.contains(&class) {
            continue;
        }
        bank.push(Entry {
            label: format!("id-{class:02}"),
            vector: prototype.clone(),
            attacker_target: None,
        });
    }

    for (poison_index, target_class) in attacked_classes.iter().copied().enumerate() {
        let vector = prototypes[target_class].with_noise(
            poison_noise,
            derive_seed(seed, 0x504f_4953 + poison_index as u64, target_class as u64),
        )?;
        bank.push(Entry {
            label: format!("attacker-{target_class:02}-{poison_index:02}"),
            vector,
            attacker_target: Some(target_class),
        });
    }

    let expected_entry_count = match attack_mode {
        AttackMode::Insertion => prototypes.len() + poison_count,
        AttackMode::Replacement => prototypes.len(),
    };
    debug_assert_eq!(bank.len(), expected_entry_count);
    Ok((bank, attacked_classes))
}

fn build_queries(prototypes: &[PatternVector], seed: u64) -> BenchResult<Vec<Query>> {
    let mut queries = Vec::with_capacity(
        CLASS_COUNT * QUERIES_PER_CLASS * QUERY_NOISE_LEVELS.len(),
    );
    for (noise_index, noise) in QUERY_NOISE_LEVELS.iter().copied().enumerate() {
        for (class, prototype) in prototypes.iter().enumerate() {
            for query_index in 0..QUERIES_PER_CLASS {
                let sample_id = ((class as u64) << 32) | query_index as u64;
                let vector = prototype.with_noise(
                    noise,
                    derive_seed(
                        seed,
                        0x5155_4552 + noise_index as u64,
                        sample_id,
                    ),
                )?;
                queries.push(Query {
                    target_class: class,
                    noise_index,
                    vector,
                });
            }
        }
    }
    Ok(queries)
}

fn rank(bank: &[Entry], query: &PatternVector) -> BenchResult<Ranked> {
    let mut best_index: Option<usize> = None;
    let mut best_similarity = f64::NEG_INFINITY;
    let mut runner_up_similarity = f64::NEG_INFINITY;

    for (index, entry) in bank.iter().enumerate() {
        let similarity = query.similarity(&entry.vector)?;
        let better = match best_index {
            None => true,
            Some(best) => {
                similarity > best_similarity
                    || (similarity == best_similarity && entry.label < bank[best].label)
            }
        };
        if better {
            runner_up_similarity = best_similarity;
            best_similarity = similarity;
            best_index = Some(index);
        } else if similarity > runner_up_similarity {
            runner_up_similarity = similarity;
        }
    }

    let winner = best_index.ok_or("cannot rank an empty registry")?;
    let margin = if runner_up_similarity.is_finite() {
        best_similarity - runner_up_similarity
    } else {
        2.0
    };
    Ok(Ranked {
        label: bank[winner].label.clone(),
        similarity: best_similarity,
        margin,
        attacker_target: bank[winner].attacker_target,
    })
}

fn accepts(similarity: f64, margin: f64) -> bool {
    similarity >= MIN_SIMILARITY && margin >= MIN_TOP_TWO_MARGIN
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn optional_ratio(numerator: usize, denominator: usize) -> String {
    if denominator == 0 {
        "NA".to_owned()
    } else {
        format!("{:.6}", ratio(numerator, denominator))
    }
}

fn optional_rate(value: Option<f64>) -> String {
    value
        .map(|rate| format!("{rate:.6}"))
        .unwrap_or_else(|| "NA".to_owned())
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
    fn exact_prototype_copy_can_capture_identity_tie() {
        let seed = 17;
        let prototypes = build_prototypes(seed).expect("prototypes");
        let (bank, attacked_classes) =
            build_bank(&prototypes, seed, 1, 0.0, AttackMode::Insertion).expect("poisoned bank");
        let target = attacked_classes[0];
        let ranked = rank(&bank, &prototypes[target]).expect("rank exact prototype");

        assert_eq!(ranked.similarity, 1.0);
        assert_eq!(ranked.attacker_target, Some(target));
        assert_eq!(ranked.margin, 0.0);
        assert!(ranked.label.starts_with("attacker-"));
        assert!(!accepts(ranked.similarity, ranked.margin));
    }

    #[test]
    fn exact_clone_replacement_preserves_capacity_but_can_win_with_high_margin() {
        let seed = 17;
        let prototypes = build_prototypes(seed).expect("prototypes");
        let (clean, _) = build_bank(&prototypes, seed, 0, 0.0, AttackMode::Insertion)
            .expect("clean bank");
        let (replaced, attacked_classes) =
            build_bank(&prototypes, seed, 1, 0.0, AttackMode::Replacement)
                .expect("replacement bank");
        let target = attacked_classes[0];
        let clean_ranked = rank(&clean, &prototypes[target]).expect("clean rank");
        let poisoned_ranked = rank(&replaced, &prototypes[target]).expect("replacement rank");

        assert_eq!(clean.len(), CLASS_COUNT);
        assert_eq!(replaced.len(), clean.len(), "replacement must preserve entry budget");
        assert!(replaced.iter().all(|entry| entry.label != format!("id-{target:02}")));
        assert_eq!(poisoned_ranked.label, format!("attacker-{target:02}-00"));
        assert!(poisoned_ranked.similarity > clean_ranked.similarity - f64::EPSILON);
        assert!(poisoned_ranked.margin >= MIN_TOP_TWO_MARGIN);
        assert!(accepts(poisoned_ranked.similarity, poisoned_ranked.margin));
    }

    #[test]
    fn replacement_budget_remains_constant_and_target_classes_are_unique() {
        let seed = 29;
        let prototypes = build_prototypes(seed).expect("prototypes");
        for poison_count in POISON_COUNTS {
            let (bank, attacked_classes) = build_bank(
                &prototypes,
                seed,
                poison_count,
                0.02,
                AttackMode::Replacement,
            )
            .expect("replacement bank");
            assert_eq!(bank.len(), CLASS_COUNT);
            assert_eq!(attacked_classes.len(), poison_count);
            for left in 0..attacked_classes.len() {
                for right in (left + 1)..attacked_classes.len() {
                    assert_ne!(attacked_classes[left], attacked_classes[right]);
                }
            }
        }
    }

    #[test]
    fn attack_budget_adds_only_declared_unique_near_duplicate_entries() {
        let seed = 29;
        let prototypes = build_prototypes(seed).expect("prototypes");
        let (bank, attacked_classes) =
            build_bank(&prototypes, seed, 8, 0.02, AttackMode::Insertion).expect("poisoned bank");

        assert_eq!(bank.len(), CLASS_COUNT + 8);
        assert_eq!(attacked_classes.len(), 8);
        for left in 0..attacked_classes.len() {
            for right in (left + 1)..attacked_classes.len() {
                assert_ne!(attacked_classes[left], attacked_classes[right]);
            }
        }
        assert_eq!(
            bank.iter().filter(|entry| entry.attacker_target.is_some()).count(),
            8
        );
    }

    #[test]
    fn no_poison_baseline_contains_only_genuine_labels() {
        let seed = 41;
        let prototypes = build_prototypes(seed).expect("prototypes");
        let (bank, attacked_classes) =
            build_bank(&prototypes, seed, 0, 0.0, AttackMode::Insertion).expect("clean bank");

        assert_eq!(bank.len(), CLASS_COUNT);
        assert!(attacked_classes.is_empty());
        assert!(bank.iter().all(|entry| entry.attacker_target.is_none()));
        for (class, prototype) in prototypes.iter().enumerate() {
            let ranked = rank(&bank, prototype).expect("rank clean prototype");
            assert_eq!(ranked.label, format!("id-{class:02}"));
        }
    }

    #[test]
    fn fixed_acceptance_rule_is_fail_closed_at_equality_boundaries() {
        assert!(accepts(MIN_SIMILARITY, MIN_TOP_TWO_MARGIN));
        assert!(!accepts(MIN_SIMILARITY - f64::EPSILON, 1.0));
        assert!(!accepts(1.0, MIN_TOP_TWO_MARGIN - f64::EPSILON));
    }

    #[test]
    fn deterministic_query_generation_is_paired_across_attack_profiles() {
        let first = build_queries(&build_prototypes(17).expect("prototypes"), 17)
            .expect("queries");
        let second = build_queries(&build_prototypes(17).expect("prototypes"), 17)
            .expect("repeat queries");

        assert_eq!(first.len(), CLASS_COUNT * QUERIES_PER_CLASS * QUERY_NOISE_LEVELS.len());
        for (left, right) in first.iter().zip(second.iter()) {
            assert_eq!(left.target_class, right.target_class);
            assert_eq!(left.noise_index, right.noise_index);
            assert_eq!(left.vector.similarity(&right.vector).expect("similarity"), 1.0);
        }
    }
}
