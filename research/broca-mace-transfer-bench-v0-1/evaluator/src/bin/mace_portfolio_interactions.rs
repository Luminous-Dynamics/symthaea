// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Synthetic held-out benchmark for non-additive memory portfolio value.
//!
//! The candidate sees singleton predictions, pair-relation features, and a
//! pairwise model fitted on disjoint historical clusters. The evaluator alone
//! owns held-out singleton outcomes and realized pair interactions.

use std::collections::{BTreeMap, BTreeSet};

use sha2::{Digest, Sha256};

const RELATIONS: [Relation; 4] = [
    Relation::Redundant,
    Relation::Complementary,
    Relation::Independent,
    Relation::Conflicting,
];
const TRAIN_CLUSTERS: u64 = 96;
const CALIBRATION_CLUSTERS: u64 = 48;
const STABLE_TEST_CLUSTERS: u64 = 64;
const SHIFTED_TEST_CLUSTERS: u64 = 32;
const TOKEN_BUDGET: u8 = 4;
const BOOTSTRAP_RESAMPLES: usize = 2_000;
const ITEM_VALUES: [i64; 6] = [25, 23, 5, 4, 16, 10];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum Relation {
    Redundant,
    Complementary,
    Independent,
    Conflicting,
    Unknown,
}

impl Relation {
    fn code(self) -> &'static str {
        match self {
            Self::Redundant => "redundant",
            Self::Complementary => "complementary",
            Self::Independent => "independent",
            Self::Conflicting => "conflicting",
            Self::Unknown => "unknown",
        }
    }

    fn all() -> [Self; 4] {
        RELATIONS
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Regime {
    Stable,
    Shifted,
}

#[derive(Clone, Copy, Debug)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        self.state = value;
        value.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn signed(&mut self, magnitude: i64) -> i64 {
        let width = (magnitude * 2 + 1) as u64;
        (self.next() % width) as i64 - magnitude
    }
}

#[derive(Clone, Debug)]
struct TrainingPair {
    cluster_id: u64,
    source_root_id: u64,
    task_family: u8,
    relation: Relation,
    observed_interaction_micros: i64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RelationEstimate {
    mean_micros: i64,
    calibration_radius_micros: u64,
    lower_bound_micros: i64,
    upper_bound_micros: i64,
    train_count: usize,
    calibration_count: usize,
}

#[derive(Clone, Debug)]
struct PairwiseModel {
    estimates: BTreeMap<Relation, RelationEstimate>,
    train_clusters: BTreeSet<u64>,
    calibration_clusters: BTreeSet<u64>,
    train_roots: BTreeSet<u64>,
    calibration_roots: BTreeSet<u64>,
    train_families: BTreeSet<u8>,
    calibration_families: BTreeSet<u8>,
    model_sha256: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FitError {
    EmptySplit,
    ClusterLeakage,
    RootLeakage,
    TaskFamilyLeakage,
    InsufficientRelationSupport,
    UnknownTrainingRelation,
    ValueOverflow,
}

fn fit_pairwise_model(
    training: &[TrainingPair],
    calibration: &[TrainingPair],
) -> Result<PairwiseModel, FitError> {
    if training.is_empty() || calibration.is_empty() {
        return Err(FitError::EmptySplit);
    }

    let train_clusters = training.iter().map(|row| row.cluster_id).collect::<BTreeSet<_>>();
    let calibration_clusters =
        calibration.iter().map(|row| row.cluster_id).collect::<BTreeSet<_>>();
    if !train_clusters.is_disjoint(&calibration_clusters) {
        return Err(FitError::ClusterLeakage);
    }

    let train_roots = training.iter().map(|row| row.source_root_id).collect::<BTreeSet<_>>();
    let calibration_roots =
        calibration.iter().map(|row| row.source_root_id).collect::<BTreeSet<_>>();
    if !train_roots.is_disjoint(&calibration_roots) {
        return Err(FitError::RootLeakage);
    }

    let train_families = training.iter().map(|row| row.task_family).collect::<BTreeSet<_>>();
    let calibration_families = calibration
        .iter()
        .map(|row| row.task_family)
        .collect::<BTreeSet<_>>();
    if !train_families.is_disjoint(&calibration_families) {
        return Err(FitError::TaskFamilyLeakage);
    }

    let mut estimates = BTreeMap::new();
    for relation in Relation::all() {
        let train = training
            .iter()
            .filter(|row| row.relation == relation)
            .collect::<Vec<_>>();
        let calibrate = calibration
            .iter()
            .filter(|row| row.relation == relation)
            .collect::<Vec<_>>();
        if train.len() < 32 || calibrate.len() < 16 {
            return Err(FitError::InsufficientRelationSupport);
        }

        let sum = train.iter().try_fold(0_i64, |total, row| {
            total.checked_add(row.observed_interaction_micros)
        }).ok_or(FitError::ValueOverflow)?;
        let mean = sum / train.len() as i64;
        let mut radius = 0_u64;
        for row in &calibrate {
            let residual = row
                .observed_interaction_micros
                .checked_sub(mean)
                .ok_or(FitError::ValueOverflow)?
                .unsigned_abs();
            radius = radius.max(residual);
        }
        // A one-unit pad avoids claiming a zero-width interval from a finite
        // calibration sample. This remains an empirical envelope, not a CI.
        radius = radius.checked_add(1).ok_or(FitError::ValueOverflow)?;
        let radius_i64 = i64::try_from(radius).map_err(|_| FitError::ValueOverflow)?;
        let lower = mean.checked_sub(radius_i64).ok_or(FitError::ValueOverflow)?;
        let upper = mean.checked_add(radius_i64).ok_or(FitError::ValueOverflow)?;

        estimates.insert(
            relation,
            RelationEstimate {
                mean_micros: mean,
                calibration_radius_micros: radius,
                lower_bound_micros: lower,
                upper_bound_micros: upper,
                train_count: train.len(),
                calibration_count: calibrate.len(),
            },
        );
    }

    let mut model = PairwiseModel {
        estimates,
        train_clusters,
        calibration_clusters,
        train_roots,
        calibration_roots,
        train_families,
        calibration_families,
        model_sha256: String::new(),
    };
    model.model_sha256 = model_digest(&model);
    Ok(model)
}

fn model_digest(model: &PairwiseModel) -> String {
    let mut fields = Vec::new();
    for relation in Relation::all() {
        let estimate = model.estimates[&relation];
        fields.push(relation.code().to_owned());
        fields.push(estimate.mean_micros.to_string());
        fields.push(estimate.calibration_radius_micros.to_string());
        fields.push(estimate.lower_bound_micros.to_string());
        fields.push(estimate.upper_bound_micros.to_string());
        fields.push(estimate.train_count.to_string());
        fields.push(estimate.calibration_count.to_string());
    }
    fields.extend(model.train_clusters.iter().map(u64::to_string));
    fields.extend(model.calibration_clusters.iter().map(u64::to_string));
    fields.extend(model.train_roots.iter().map(u64::to_string));
    fields.extend(model.calibration_roots.iter().map(u64::to_string));
    fields.extend(model.train_families.iter().map(u8::to_string));
    fields.extend(model.calibration_families.iter().map(u8::to_string));
    digest_fields(b"mace:pairwise-interaction-model:v1\0", &fields)
}

fn digest_fields(domain: &[u8], fields: &[String]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    for field in fields {
        hasher.update((field.len() as u64).to_be_bytes());
        hasher.update(field.as_bytes());
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn base_interaction(relation: Relation, regime: Regime) -> i64 {
    match (relation, regime) {
        (Relation::Redundant, Regime::Stable) => -22,
        (Relation::Complementary, Regime::Stable) => 28,
        (Relation::Independent, Regime::Stable) => 0,
        (Relation::Conflicting, Regime::Stable) => -15,
        (Relation::Redundant, Regime::Shifted) => 7,
        (Relation::Complementary, Regime::Shifted) => -18,
        (Relation::Independent, Regime::Shifted) => 0,
        (Relation::Conflicting, Regime::Shifted) => 6,
        (Relation::Unknown, _) => 0,
    }
}

fn training_split(
    seed: u64,
    start_cluster: u64,
    clusters: u64,
    task_families: &[u8],
) -> Vec<TrainingPair> {
    let mut rng = Rng::new(seed);
    let mut rows = Vec::new();
    for offset in 0..clusters {
        let cluster_id = start_cluster + offset;
        let family = task_families[(offset as usize) % task_families.len()];
        for (index, relation) in Relation::all().into_iter().enumerate() {
            let root = cluster_id * 16 + index as u64 + 1;
            rows.push(TrainingPair {
                cluster_id,
                source_root_id: root,
                task_family: family,
                relation,
                observed_interaction_micros: base_interaction(
                    relation,
                    Regime::Stable,
                ) + rng.signed(2),
            });
        }
    }
    rows
}

#[derive(Clone, Debug)]
struct CandidateItem {
    item_id: u8,
    event_group: u8,
    token_cost: u8,
    source_root_id: u64,
    singleton_estimate_micros: i64,
}

#[derive(Clone, Debug)]
struct CandidatePortfolio {
    cluster_id: u64,
    task_family: u8,
    regime: Regime,
    budget_tokens: u8,
    items: Vec<CandidateItem>,
    pair_relations: BTreeMap<(u8, u8), Relation>,
}

#[derive(Clone, Debug)]
struct OracleTruth {
    singleton_values_micros: BTreeMap<u8, i64>,
    pair_values_micros: BTreeMap<(u8, u8), i64>,
}

#[derive(Clone, Debug)]
struct Case {
    candidate: CandidatePortfolio,
    truth: OracleTruth,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Selection {
    item_ids: Vec<u8>,
    predicted_value_micros: i64,
}

fn pair_key(left: u8, right: u8) -> (u8, u8) {
    if left < right {
        (left, right)
    } else {
        (right, left)
    }
}

fn known_relation(left: u8, right: u8) -> Relation {
    match pair_key(left, right) {
        (0, 1) => Relation::Redundant,
        (2, 3) => Relation::Complementary,
        (2, 4) => Relation::Conflicting,
        _ => Relation::Independent,
    }
}

fn generate_case(
    cluster_id: u64,
    task_family: u8,
    regime: Regime,
    seed: u64,
) -> Case {
    let mut rng = Rng::new(seed ^ cluster_id.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut items = Vec::new();
    let mut singleton_values = BTreeMap::new();
    for (index, base) in ITEM_VALUES.iter().enumerate() {
        let item_id = index as u8;
        let singleton_truth = *base + rng.signed(2);
        let singleton_estimate = singleton_truth + rng.signed(2);
        singleton_values.insert(item_id, singleton_truth);
        let event_group = if cluster_id % 7 == 0 && item_id == 5 {
            4
        } else {
            item_id
        };
        items.push(CandidateItem {
            item_id,
            event_group,
            token_cost: 1,
            source_root_id: cluster_id * 100 + u64::from(item_id) + 1,
            singleton_estimate_micros: singleton_estimate,
        });
    }

    let mut pair_relations = BTreeMap::new();
    let mut pair_values = BTreeMap::new();
    for left in 0..items.len() as u8 {
        for right in (left + 1)..items.len() as u8 {
            let actual_relation = known_relation(left, right);
            let mut candidate_relation = actual_relation;
            let perturb = rng.next() % 100;
            if perturb < 3 {
                candidate_relation = Relation::Unknown;
            } else if perturb < 5 {
                candidate_relation = match actual_relation {
                    Relation::Redundant => Relation::Independent,
                    Relation::Complementary => Relation::Conflicting,
                    Relation::Independent => Relation::Redundant,
                    Relation::Conflicting => Relation::Complementary,
                    Relation::Unknown => Relation::Unknown,
                };
            }
            pair_relations.insert((left, right), candidate_relation);
            let actual = base_interaction(actual_relation, regime) + rng.signed(1);
            pair_values.insert((left, right), actual);
        }
    }

    Case {
        candidate: CandidatePortfolio {
            cluster_id,
            task_family,
            regime,
            budget_tokens: TOKEN_BUDGET,
            items,
            pair_relations,
        },
        truth: OracleTruth {
            singleton_values_micros: singleton_values,
            pair_values_micros: pair_values,
        },
    }
}

fn candidate_is_eligible(candidate: &CandidatePortfolio, model: &PairwiseModel) -> bool {
    candidate.regime == Regime::Stable
        && !model.train_families.contains(&candidate.task_family)
        && !model.calibration_families.contains(&candidate.task_family)
        && !model.train_clusters.contains(&candidate.cluster_id)
        && !model.calibration_clusters.contains(&candidate.cluster_id)
        && candidate.items.iter().all(|item| {
            item.source_root_id != 0
                && !model.train_roots.contains(&item.source_root_id)
                && !model.calibration_roots.contains(&item.source_root_id)
        })
        && model.train_clusters.is_disjoint(&model.calibration_clusters)
        && model.train_roots.is_disjoint(&model.calibration_roots)
}

fn enumerate_best(
    candidate: &CandidatePortfolio,
    mut value_for: impl FnMut(&[u8]) -> Option<i64>,
) -> Option<Selection> {
    if candidate.items.len() > 8 {
        return None;
    }
    let combinations = 1_u64.checked_shl(candidate.items.len() as u32)?;
    let mut best = Selection {
        item_ids: Vec::new(),
        predicted_value_micros: 0,
    };

    for mask in 0..combinations {
        let mut ids = Vec::new();
        let mut event_groups = BTreeSet::new();
        let mut tokens = 0_u16;
        for (index, item) in candidate.items.iter().enumerate() {
            if mask & (1_u64 << index) == 0 {
                continue;
            }
            if !event_groups.insert(item.event_group) {
                ids.clear();
                break;
            }
            tokens += u16::from(item.token_cost);
            if tokens > u16::from(candidate.budget_tokens) {
                ids.clear();
                break;
            }
            ids.push(item.item_id);
        }
        if ids.is_empty() && mask != 0 {
            continue;
        }
        ids.sort_unstable();
        let Some(value) = value_for(&ids) else {
            continue;
        };
        if value > best.predicted_value_micros
            || (value == best.predicted_value_micros && ids < best.item_ids)
        {
            best = Selection {
                item_ids: ids,
                predicted_value_micros: value,
            };
        }
    }
    Some(best)
}

fn select_additive(candidate: &CandidatePortfolio) -> Option<Selection> {
    enumerate_best(candidate, |ids| {
        ids.iter().try_fold(0_i64, |total, id| {
            let item = candidate.items.iter().find(|item| item.item_id == *id)?;
            total.checked_add(item.singleton_estimate_micros)
        })
    })
}

fn select_pairwise(
    candidate: &CandidatePortfolio,
    model: &PairwiseModel,
) -> Option<Selection> {
    if !candidate_is_eligible(candidate, model) {
        return None;
    }
    if candidate.pair_relations.values().any(|relation| *relation == Relation::Unknown) {
        return None;
    }

    enumerate_best(candidate, |ids| {
        let mut value = ids.iter().try_fold(0_i64, |total, id| {
            let item = candidate.items.iter().find(|item| item.item_id == *id)?;
            total.checked_add(item.singleton_estimate_micros)
        })?;
        for left_index in 0..ids.len() {
            for right_index in (left_index + 1)..ids.len() {
                let key = pair_key(ids[left_index], ids[right_index]);
                let relation = candidate.pair_relations.get(&key)?;
                let estimate = model.estimates.get(relation)?;
                value = value.checked_add(estimate.lower_bound_micros)?;
            }
        }
        Some(value)
    })
}

fn select_oracle(candidate: &CandidatePortfolio, truth: &OracleTruth) -> Option<Selection> {
    enumerate_best(candidate, |ids| actual_utility(ids, truth))
}

fn actual_utility(ids: &[u8], truth: &OracleTruth) -> Option<i64> {
    let mut total = 0_i64;
    for id in ids {
        total = total.checked_add(*truth.singleton_values_micros.get(id)?)?;
    }
    for left_index in 0..ids.len() {
        for right_index in (left_index + 1)..ids.len() {
            let key = pair_key(ids[left_index], ids[right_index]);
            total = total.checked_add(*truth.pair_values_micros.get(&key)?)?;
        }
    }
    Some(total)
}

#[derive(Clone, Debug, Default)]
struct ArmMetrics {
    utility_sum: i64,
    regret_sum: i64,
    selected_items: u64,
    token_spend: u64,
    redundant_loss: i64,
    synergy_captured: i64,
    false_synergy: u64,
    false_redundancy: u64,
    pairwise_abstentions: u64,
}

fn add_arm_metrics(
    metrics: &mut ArmMetrics,
    candidate: &CandidatePortfolio,
    truth: &OracleTruth,
    selected: &Selection,
    oracle: &Selection,
    predicted_interactions: Option<&PairwiseModel>,
) {
    let actual = actual_utility(&selected.item_ids, truth).unwrap_or(0);
    let optimal = actual_utility(&oracle.item_ids, truth).unwrap_or(0);
    metrics.utility_sum += actual;
    metrics.regret_sum += optimal.saturating_sub(actual);
    metrics.selected_items += selected.item_ids.len() as u64;
    metrics.token_spend += selected
        .item_ids
        .iter()
        .filter_map(|id| candidate.items.iter().find(|item| item.item_id == *id))
        .map(|item| u64::from(item.token_cost))
        .sum::<u64>();

    for left_index in 0..selected.item_ids.len() {
        for right_index in (left_index + 1)..selected.item_ids.len() {
            let left = selected.item_ids[left_index];
            let right = selected.item_ids[right_index];
            let key = pair_key(left, right);
            let actual_delta = *truth.pair_values_micros.get(&key).unwrap_or(&0);
            if actual_delta < 0 {
                metrics.redundant_loss += -actual_delta;
            } else {
                metrics.synergy_captured += actual_delta;
            }
            if let Some(model) = predicted_interactions {
                if let Some(relation) = candidate.pair_relations.get(&key) {
                    if let Some(estimate) = model.estimates.get(relation) {
                        if estimate.lower_bound_micros > 0 && actual_delta <= 0 {
                            metrics.false_synergy += 1;
                        }
                        if estimate.upper_bound_micros < 0 && actual_delta >= 0 {
                            metrics.false_redundancy += 1;
                        }
                    }
                }
            }
        }
    }
}

fn mean(value: i64, count: usize) -> f64 {
    value as f64 / count.max(1) as f64
}

fn bootstrap_ci(differences: &[i64], samples: usize, seed: u64) -> Option<(f64, f64)> {
    if differences.is_empty() || samples == 0 {
        return None;
    }
    let mut rng = Rng::new(seed);
    let mut means = Vec::with_capacity(samples);
    for _ in 0..samples {
        let mut total = 0_i64;
        for _ in 0..differences.len() {
            let index = (rng.next() % differences.len() as u64) as usize;
            total = total.checked_add(differences[index])?;
        }
        means.push(total as f64 / differences.len() as f64);
    }
    means.sort_by(|left, right| left.total_cmp(right));
    let lower = ((samples as f64) * 0.025).floor() as usize;
    let upper = ((samples as f64) * 0.975).floor() as usize;
    Some((
        means[lower.min(samples - 1)],
        means[upper.min(samples - 1)],
    ))
}

fn run_seed(seed: u64) -> Option<(ArmMetrics, ArmMetrics, Vec<i64>, usize)> {
    let training = training_split(seed + 101, 1_000 + seed * 1_000, TRAIN_CLUSTERS, &[1, 2]);
    let calibration = training_split(
        seed + 202,
        2_000 + seed * 1_000,
        CALIBRATION_CLUSTERS,
        &[3],
    );
    let model = fit_pairwise_model(&training, &calibration).ok()?;

    let mut additive = ArmMetrics::default();
    let mut pairwise = ArmMetrics::default();
    let mut differences = Vec::new();
    let mut cases = 0_usize;
    let mut unknown_abstentions = 0_usize;

    for offset in 0..(STABLE_TEST_CLUSTERS + SHIFTED_TEST_CLUSTERS) {
        let shifted = offset >= STABLE_TEST_CLUSTERS;
        let regime = if shifted { Regime::Shifted } else { Regime::Stable };
        let task_family = if shifted { 5 } else { 4 };
        let cluster_id = 3_000 + seed * 1_000 + offset;
        let case = generate_case(
            cluster_id,
            task_family,
            regime,
            seed.wrapping_mul(7919) + offset,
        );
        if model.train_clusters.contains(&case.candidate.cluster_id)
            || model.calibration_clusters.contains(&case.candidate.cluster_id)
        {
            return None;
        }

        let additive_selection = select_additive(&case.candidate)?;
        let pairwise_prediction = select_pairwise(&case.candidate, &model);
        let pairwise_selection = pairwise_prediction
            .clone()
            .unwrap_or_else(|| additive_selection.clone());
        if pairwise_prediction.is_none() {
            unknown_abstentions += 1;
            pairwise.pairwise_abstentions += 1;
        }

        let oracle = select_oracle(&case.candidate, &case.truth)?;
        let additive_utility = actual_utility(&additive_selection.item_ids, &case.truth)?;
        let pairwise_utility = actual_utility(&pairwise_selection.item_ids, &case.truth)?;
        differences.push(pairwise_utility - additive_utility);

        add_arm_metrics(
            &mut additive,
            &case.candidate,
            &case.truth,
            &additive_selection,
            &oracle,
            None,
        );
        add_arm_metrics(
            &mut pairwise,
            &case.candidate,
            &case.truth,
            &pairwise_selection,
            &oracle,
            if pairwise_prediction.is_some() {
                Some(&model)
            } else {
                None
            },
        );
        cases += 1;
    }

    let _ = unknown_abstentions;
    Some((additive, pairwise, differences, cases))
}

fn report() {
    println!("MACE portfolio interaction benchmark v0.1");
    println!(
        concat!(
            "train_clusters={}, calibration_clusters={}, ",
            "stable_test_clusters={}, shifted_test_clusters={}"
        ),
        TRAIN_CLUSTERS,
        CALIBRATION_CLUSTERS,
        STABLE_TEST_CLUSTERS,
        SHIFTED_TEST_CLUSTERS
    );
    println!(
        concat!(
            "arm,mean_utility,mean_regret,selected_items,token_spend,",
            "redundancy_loss,synergy_captured,false_synergy,",
            "false_redundancy,pairwise_abstentions"
        )
    );

    let mut additive_total = ArmMetrics::default();
    let mut pairwise_total = ArmMetrics::default();
    let mut paired = Vec::new();
    let mut cases = 0_usize;
    for seed in 1..=5 {
        let Some((additive, pairwise, differences, count)) = run_seed(seed) else {
            println!("ERROR,model_fit_or_case_generation_failed");
            return;
        };
        additive_total.utility_sum += additive.utility_sum;
        additive_total.regret_sum += additive.regret_sum;
        additive_total.selected_items += additive.selected_items;
        additive_total.token_spend += additive.token_spend;
        additive_total.redundant_loss += additive.redundant_loss;
        additive_total.synergy_captured += additive.synergy_captured;

        pairwise_total.utility_sum += pairwise.utility_sum;
        pairwise_total.regret_sum += pairwise.regret_sum;
        pairwise_total.selected_items += pairwise.selected_items;
        pairwise_total.token_spend += pairwise.token_spend;
        pairwise_total.redundant_loss += pairwise.redundant_loss;
        pairwise_total.synergy_captured += pairwise.synergy_captured;
        pairwise_total.false_synergy += pairwise.false_synergy;
        pairwise_total.false_redundancy += pairwise.false_redundancy;
        pairwise_total.pairwise_abstentions += pairwise.pairwise_abstentions;
        paired.extend(differences);
        cases += count;
    }

    let print_arm = |name: &str, metrics: &ArmMetrics| {
        println!(
            "{},{:.3},{:.3},{},{},{},{},{},{},{}",
            name,
            mean(metrics.utility_sum, cases),
            mean(metrics.regret_sum, cases),
            metrics.selected_items,
            metrics.token_spend,
            metrics.redundant_loss,
            metrics.synergy_captured,
            metrics.false_synergy,
            metrics.false_redundancy,
            metrics.pairwise_abstentions
        );
    };
    print_arm("additive_exact", &additive_total);
    print_arm("pairwise_conservative", &pairwise_total);

    let oracle_regret = mean(pairwise_total.regret_sum, cases);
    let average_difference = paired.iter().sum::<i64>() as f64 / paired.len().max(1) as f64;
    let ci = bootstrap_ci(&paired, BOOTSTRAP_RESAMPLES, 0xC0FF_EE11);
    println!(
        "paired_pairwise_minus_additive_mean_micros={:.3}",
        average_difference
    );
    if let Some((lower, upper)) = ci {
        println!("paired_cluster_bootstrap_95pct_ci_micros={:.3},{:.3}", lower, upper);
    }
    println!("pairwise_mean_regret_vs_evaluator_oracle_micros={:.3}", oracle_regret);
    println!("cases={}", cases);
    println!("claim=synthetic_diagnostic_only_no_external_sota_claim");
}

fn main() {
    report();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_and_calibration_splits_reject_cluster_leakage() {
        let mut train = training_split(10, 1_000, 64, &[1, 2]);
        let calibration = training_split(11, 2_000, 32, &[3]);
        train[0].cluster_id = 2_000;
        assert_eq!(
            fit_pairwise_model(&train, &calibration).unwrap_err(),
            FitError::ClusterLeakage
        );
    }

    #[test]
    fn train_and_calibration_splits_reject_upstream_root_leakage() {
        let train = training_split(10, 1_000, 64, &[1, 2]);
        let mut calibration = training_split(11, 2_000, 32, &[3]);
        calibration[0].source_root_id = train[0].source_root_id;
        assert_eq!(
            fit_pairwise_model(&train, &calibration).unwrap_err(),
            FitError::RootLeakage
        );
    }

    #[test]
    fn train_and_calibration_splits_reject_task_family_leakage() {
        let train = training_split(10, 1_000, 64, &[1, 2]);
        let calibration = training_split(11, 2_000, 32, &[2]);
        assert_eq!(
            fit_pairwise_model(&train, &calibration).unwrap_err(),
            FitError::TaskFamilyLeakage
        );
    }

    #[test]
    fn pairwise_policy_captures_complementarity_and_avoids_redundancy() {
        let train = training_split(10, 1_000, 96, &[1, 2]);
        let calibration = training_split(11, 2_000, 48, &[3]);
        let model = fit_pairwise_model(&train, &calibration).unwrap();
        let mut case = generate_case(3_000, 4, Regime::Stable, 99);
        for relation in case.candidate.pair_relations.values_mut() {
            if *relation == Relation::Unknown {
                *relation = Relation::Independent;
            }
        }

        let additive = select_additive(&case.candidate).unwrap();
        let pairwise = select_pairwise(&case.candidate, &model).unwrap();
        let additive_utility = actual_utility(&additive.item_ids, &case.truth).unwrap();
        let pairwise_utility = actual_utility(&pairwise.item_ids, &case.truth).unwrap();

        assert!(pairwise_utility > additive_utility);
        assert!(!pairwise.item_ids.contains(&0) || !pairwise.item_ids.contains(&1));
        assert!(pairwise.item_ids.contains(&2) && pairwise.item_ids.contains(&3));
    }

    #[test]
    fn pairwise_model_abstains_after_regime_shift() {
        let train = training_split(10, 1_000, 96, &[1, 2]);
        let calibration = training_split(11, 2_000, 48, &[3]);
        let model = fit_pairwise_model(&train, &calibration).unwrap();
        let shifted = generate_case(3_000, 4, Regime::Shifted, 99);
        assert!(select_pairwise(&shifted.candidate, &model).is_none());
    }

    #[test]
    fn unknown_pair_relation_forces_pairwise_abstention() {
        let train = training_split(10, 1_000, 96, &[1, 2]);
        let calibration = training_split(11, 2_000, 48, &[3]);
        let model = fit_pairwise_model(&train, &calibration).unwrap();
        let mut case = generate_case(3_000, 4, Regime::Stable, 99);
        case.candidate
            .pair_relations
            .insert((0, 1), Relation::Unknown);
        assert!(select_pairwise(&case.candidate, &model).is_none());
    }

    #[test]
    fn portfolio_selection_obeys_event_exclusivity_and_token_budget() {
        let train = training_split(10, 1_000, 96, &[1, 2]);
        let calibration = training_split(11, 2_000, 48, &[3]);
        let model = fit_pairwise_model(&train, &calibration).unwrap();
        let mut case = generate_case(3_007, 4, Regime::Stable, 99);
        case.candidate.budget_tokens = 2;
        let selection = select_pairwise(&case.candidate, &model).unwrap();
        assert!(selection.item_ids.len() <= 2);
        assert!(!selection.item_ids.contains(&4) || !selection.item_ids.contains(&5));
    }

    #[test]
    fn fit_is_deterministic_and_model_digest_is_stable() {
        let train = training_split(10, 1_000, 96, &[1, 2]);
        let calibration = training_split(11, 2_000, 48, &[3]);
        let first = fit_pairwise_model(&train, &calibration).unwrap();
        let second = fit_pairwise_model(&train, &calibration).unwrap();
        assert_eq!(first.model_sha256, second.model_sha256);
        assert_eq!(first.estimates, second.estimates);
    }

    #[test]
    fn bootstrap_interval_is_deterministic_and_ordered() {
        let differences = [3, -1, 5, 7, -2, 4];
        let first = bootstrap_ci(&differences, 500, 77).unwrap();
        let second = bootstrap_ci(&differences, 500, 77).unwrap();
        assert_eq!(first, second);
        assert!(first.0 <= first.1);
    }

    #[test]
    fn pairwise_model_rejects_test_source_root_reuse() {
        let train = training_split(10, 1_000, 96, &[1, 2]);
        let calibration = training_split(11, 2_000, 48, &[3]);
        let model = fit_pairwise_model(&train, &calibration).unwrap();
        let mut heldout = generate_case(3_000, 4, Regime::Stable, 99);
        heldout.candidate.items[0].source_root_id = *model.train_roots.iter().next().unwrap();
        assert!(select_pairwise(&heldout.candidate, &model).is_none());
    }

    #[test]
    fn heldout_task_family_is_disjoint_from_train_and_calibration() {
        let train = training_split(10, 1_000, 96, &[1, 2]);
        let calibration = training_split(11, 2_000, 48, &[3]);
        let model = fit_pairwise_model(&train, &calibration).unwrap();
        let heldout = generate_case(3_000, 4, Regime::Stable, 99);
        assert!(candidate_is_eligible(&heldout.candidate, &model));
    }
}
