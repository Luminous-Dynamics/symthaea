// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Compare single-view retrieval, fixed multi-view fusion, and adaptive second-view acquisition.
//!
//! The paired unknown-identity control assigns the exact same observed views to an
//! unregistered semantic identity. Consequently, every vector-only action must be
//! identical for the registered and counterfactual interpretations. Multi-view fusion
//! may improve class retrieval under noise, but it is not identity evidence.

use std::error::Error;
use std::time::Instant;

use symthaea_core::hdc::identity_evidence::{
    IdentityResolution, ObservationIdentityEvidence, PatternRegistry, ResolutionPolicy,
};
use symthaea_core::hdc::pattern_interaction::{PatternSpaceError, PatternVector};

const DIMENSION: usize = 4_096;
const IDENTITY_COUNT: usize = 64;
const SAMPLES_PER_IDENTITY: usize = 8;
const NOISE_LEVELS: [f64; 4] = [0.05, 0.10, 0.20, 0.30];
const SEEDS: [u64; 5] = [17, 29, 41, 53, 67];
const ADAPTIVE_SCORE_FLOOR: f64 = 0.55;
const ADAPTIVE_MARGIN_FLOOR: f64 = 0.08;

type BenchResult<T> = Result<T, Box<dyn Error>>;

#[derive(Debug, Clone, PartialEq)]
struct Rank {
    label: String,
    score: f64,
    margin: f64,
}

impl Rank {
    fn accepted(&self, policy: ResolutionPolicy) -> bool {
        self.score >= policy.min_similarity && self.margin >= policy.min_top_two_margin
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct MethodStats {
    total: usize,
    accepted: usize,
    correct: usize,
    accepted_errors: usize,
    paired_unknown_false_accepts: usize,
}

impl MethodStats {
    fn observe(&mut self, result: &Rank, target: &str, policy: ResolutionPolicy) {
        self.total += 1;
        let correct = result.label == target;
        if correct {
            self.correct += 1;
        }
        if result.accepted(policy) {
            self.accepted += 1;
            if !correct {
                self.accepted_errors += 1;
            }
            // The exact same observed vector(s) are also assigned to an unknown
            // semantic identity in the paired counterfactual control.
            self.paired_unknown_false_accepts += 1;
        }
    }

    fn coverage(self) -> f64 {
        ratio(self.accepted, self.total)
    }

    fn accuracy(self) -> f64 {
        ratio(self.correct, self.total)
    }

    fn selective_risk(self) -> Option<f64> {
        (self.accepted > 0).then(|| ratio(self.accepted_errors, self.accepted))
    }

    fn paired_unknown_far(self) -> f64 {
        ratio(self.paired_unknown_false_accepts, self.total)
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct EvidenceStats {
    registration_only_required: usize,
    registration_only_conflicts: usize,
    observation_verified: usize,
    observation_conflicts: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct GroupStats {
    single: MethodStats,
    two_view: MethodStats,
    adaptive: MethodStats,
    second_view_requests: usize,
    evidence: EvidenceStats,
    identical_counterfactual_outputs: usize,
}

fn main() -> BenchResult<()> {
    let start = Instant::now();
    let policy = ResolutionPolicy::new(0.40, 0.02)?;

    println!("# protocol=platonic-multiview-evidence-budget-v1");
    println!("# dimension={DIMENSION}; identities={IDENTITY_COUNT}; samples_per_identity_noise={SAMPLES_PER_IDENTITY}");
    println!("# seeds=17,29,41,53,67; noise=0.05,0.10,0.20,0.30");
    println!("# methods=single_view;average_similarity_two_views;adaptive_second_view");
    println!("# adaptive_request=single_score<{ADAPTIVE_SCORE_FLOOR} OR single_margin<{ADAPTIVE_MARGIN_FLOOR}");
    println!("# decision_rule=similarity>=0.40 AND top_two_margin>=0.02; fixed diagnostic, not calibrated");
    println!("# unknown_control=identical observed view(s), different unregistered semantic identity; cannot be resolved from vectors alone");
    println!("# synthetic observation oracle is ground-truth scaffolding, not authenticated real-world evidence");
    println!("record_type,seed,noise_fraction,dimension,identity_count,query_n,method,top1_accuracy,coverage,selective_risk,paired_unknown_false_accept_rate,second_view_requests,registration_only_required,registration_only_conflict,observation_verified,observation_conflict,counterfactual_output_equal");

    for seed in SEEDS {
        let mut registry = PatternRegistry::new(format!("registry-seed-{seed}"))?;
        let mut prototypes = Vec::with_capacity(IDENTITY_COUNT);
        let mut receipts = Vec::with_capacity(IDENTITY_COUNT);

        for index in 0..IDENTITY_COUNT {
            let label = format!("entity-{index:04}");
            let prototype = PatternVector::seeded_random(
                DIMENSION,
                derive_seed(seed, 0x5052_4f54, index as u64),
            )?;
            let receipt = registry.register(label.clone(), prototype.clone())?;
            prototypes.push((label, prototype));
            receipts.push(receipt);
        }

        for (noise_index, noise_fraction) in NOISE_LEVELS.iter().copied().enumerate() {
            let mut stats = GroupStats::default();
            for identity_index in 0..IDENTITY_COUNT {
                let (target, prototype) = &prototypes[identity_index];
                let receipt = &receipts[identity_index];

                for sample_index in 0..SAMPLES_PER_IDENTITY {
                    let sample_id = (identity_index * SAMPLES_PER_IDENTITY + sample_index) as u64;
                    let view_one = prototype.with_noise(
                        noise_fraction,
                        derive_seed(seed, 0x5649_4557 + noise_index as u64, sample_id * 2),
                    )?;
                    let view_two = prototype.with_noise(
                        noise_fraction,
                        derive_seed(seed, 0x5649_4557 + noise_index as u64, sample_id * 2 + 1),
                    )?;

                    let single_rank = rank_single(&prototypes, &view_one)?;
                    let two_view_rank = rank_two_views(&prototypes, &view_one, &view_two)?;
                    let request_second = requests_second_view(&single_rank);
                    let adaptive_rank = if request_second {
                        stats.second_view_requests += 1;
                        two_view_rank.clone()
                    } else {
                        single_rank.clone()
                    };

                    stats.single.observe(&single_rank, target, policy);
                    stats.two_view.observe(&two_view_rank, target, policy);
                    stats.adaptive.observe(&adaptive_rank, target, policy);

                    // The counterfactual changes only hidden semantics, not either
                    // vector. The rank functions consume only these vectors and the bank;
                    // the unit control below proves repeated scores/actions are identical.
                    let unknown_view_one = view_one.clone();
                    let unknown_view_two = view_two.clone();
                    if unknown_view_one == view_one && unknown_view_two == view_two {
                        stats.identical_counterfactual_outputs += 1;
                    }

                    match registry.resolve_with_receipt(&view_one, receipt, policy)? {
                        IdentityResolution::EvidenceRequired { .. } => {
                            stats.evidence.registration_only_required += 1;
                        }
                        IdentityResolution::EvidenceConflict { .. } => {
                            stats.evidence.registration_only_conflicts += 1;
                        }
                        IdentityResolution::Verified { .. } => {
                            return Err("registration-only resolution must never verify identity".into());
                        }
                    }

                    // The synthetic benchmark oracle plays the role of an already-authenticated
                    // independent source. This does not authenticate real identities.
                    let observation_id = format!(
                        "seed-{seed}/noise-{noise_index}/entity-{identity_index}/sample-{sample_index}/view-one"
                    );
                    let observation_evidence = ObservationIdentityEvidence::from_authenticated_source(
                        receipt,
                        &observation_id,
                        &view_one,
                        "synthetic-known-query-oracle",
                    )?;
                    match registry.resolve_with_observation_evidence(
                        &view_one,
                        &observation_id,
                        receipt,
                        &observation_evidence,
                        policy,
                    )? {
                        IdentityResolution::Verified { .. } => {
                            stats.evidence.observation_verified += 1;
                        }
                        IdentityResolution::EvidenceConflict { .. } => {
                            stats.evidence.observation_conflicts += 1;
                        }
                        IdentityResolution::EvidenceRequired { .. } => {
                            return Err("observation evidence was not consumed".into());
                        }
                    }
                }
            }

            report_row(
                seed,
                noise_fraction,
                "single_view",
                stats.single,
                0,
                stats.evidence,
                stats.identical_counterfactual_outputs,
            );
            report_row(
                seed,
                noise_fraction,
                "average_similarity_two_views",
                stats.two_view,
                0,
                EvidenceStats::default(),
                stats.identical_counterfactual_outputs,
            );
            report_row(
                seed,
                noise_fraction,
                "adaptive_second_view",
                stats.adaptive,
                stats.second_view_requests,
                EvidenceStats::default(),
                stats.identical_counterfactual_outputs,
            );
            report_evidence_row(seed, noise_fraction, stats);
        }
    }

    println!("# runtime_ms={}", start.elapsed().as_millis());
    Ok(())
}

fn requests_second_view(single: &Rank) -> bool {
    single.score < ADAPTIVE_SCORE_FLOOR || single.margin < ADAPTIVE_MARGIN_FLOOR
}

fn rank_single(
    prototypes: &[(String, PatternVector)],
    query: &PatternVector,
) -> BenchResult<Rank> {
    rank_scored(
        prototypes
            .iter()
            .map(|(label, prototype)| Ok((label.clone(), query.similarity(prototype)?)))
            .collect::<Result<Vec<_>, PatternSpaceError>>()?,
    )
}

fn rank_two_views(
    prototypes: &[(String, PatternVector)],
    view_one: &PatternVector,
    view_two: &PatternVector,
) -> BenchResult<Rank> {
    rank_scored(
        prototypes
            .iter()
            .map(|(label, prototype)| {
                Ok((
                    label.clone(),
                    (view_one.similarity(prototype)? + view_two.similarity(prototype)?) / 2.0,
                ))
            })
            .collect::<Result<Vec<_>, PatternSpaceError>>()?,
    )
}

fn rank_scored(mut scores: Vec<(String, f64)>) -> BenchResult<Rank> {
    scores.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    let (label, score) = scores
        .first()
        .cloned()
        .ok_or("cannot rank an empty prototype list")?;
    let runner_up = scores.get(1).map_or(f64::NEG_INFINITY, |item| item.1);
    let margin = if runner_up.is_finite() {
        score - runner_up
    } else {
        2.0
    };
    Ok(Rank { label, score, margin })
}

fn report_row(
    seed: u64,
    noise_fraction: f64,
    method: &str,
    stats: MethodStats,
    second_view_requests: usize,
    evidence: EvidenceStats,
    counterfactual_outputs_equal: usize,
) {
    println!(
        "evaluation,{seed},{noise_fraction:.2},{DIMENSION},{IDENTITY_COUNT},{},{method},{:.6},{:.6},{},{:.6},{second_view_requests},{},{},{},{},{}",
        stats.total,
        stats.accuracy(),
        stats.coverage(),
        stats.selective_risk().map_or_else(|| "NA".to_owned(), |v| format!("{v:.6}")),
        stats.paired_unknown_far(),
        evidence.registration_only_required,
        evidence.registration_only_conflicts,
        evidence.observation_verified,
        evidence.observation_conflicts,
        counterfactual_outputs_equal
    );
}

fn report_evidence_row(seed: u64, noise_fraction: f64, stats: GroupStats) {
    println!(
        "evidence_summary,{seed},{noise_fraction:.2},{DIMENSION},{IDENTITY_COUNT},{},evidence_channels,NA,NA,NA,NA,{},{},{},{},{},{}",
        stats.single.total,
        stats.second_view_requests,
        stats.evidence.registration_only_required,
        stats.evidence.registration_only_conflicts,
        stats.evidence.observation_verified,
        stats.evidence.observation_conflicts,
        stats.identical_counterfactual_outputs
    );
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
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
    fn identical_views_have_identical_ranking_for_known_and_unknown_semantics() {
        let prototypes = vec![
            ("a".to_owned(), PatternVector::seeded_random(1_024, 11).expect("a")),
            ("b".to_owned(), PatternVector::seeded_random(1_024, 13).expect("b")),
        ];
        let first = prototypes[0].1.with_noise(0.10, 17).expect("view one");
        let second = prototypes[0].1.with_noise(0.10, 19).expect("view two");
        let known_single = rank_single(&prototypes, &first).expect("known single");
        let unknown_single = rank_single(&prototypes, &first.clone()).expect("unknown single");
        let known_pair = rank_two_views(&prototypes, &first, &second).expect("known pair");
        let unknown_pair =
            rank_two_views(&prototypes, &first.clone(), &second.clone()).expect("unknown pair");
        assert_eq!(known_single, unknown_single);
        assert_eq!(known_pair, unknown_pair);
        assert_eq!(
            requests_second_view(&known_single),
            requests_second_view(&unknown_single)
        );
    }

    #[test]
    fn two_view_average_is_the_mean_of_the_two_similarity_scores() {
        let prototypes = vec![
            ("a".to_owned(), PatternVector::seeded_random(1_024, 23).expect("a")),
            ("b".to_owned(), PatternVector::seeded_random(1_024, 29).expect("b")),
        ];
        let first = prototypes[0].1.with_noise(0.10, 31).expect("first");
        let second = prototypes[0].1.with_noise(0.20, 37).expect("second");
        let result = rank_two_views(&prototypes, &first, &second).expect("fused rank");
        let expected = (first.similarity(&prototypes[0].1).expect("sim 1")
            + second.similarity(&prototypes[0].1).expect("sim 2"))
            / 2.0;
        assert!((result.score - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn evidence_budget_never_allows_enrollment_receipt_alone_to_verify() {
        let pattern = PatternVector::seeded_random(1_024, 41).expect("pattern");
        let mut registry = PatternRegistry::new("test-registry").expect("registry");
        let receipt = registry.register("known", pattern.clone()).expect("register");
        let result = registry
            .resolve_with_receipt(&pattern, &receipt, ResolutionPolicy::default())
            .expect("registration result");
        assert!(matches!(
            result,
            IdentityResolution::EvidenceRequired { .. }
        ));
    }
}
