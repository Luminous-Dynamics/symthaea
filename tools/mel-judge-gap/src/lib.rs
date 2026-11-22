use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const SCHEMA_VERSION: &str = "mel-judge-gap-qualification-v1";
const ASSIGNMENT_ALGORITHM: &str = "sha256-pair-order-v1";
const ASSIGNMENT_DOMAIN: &[u8] = b"MEL-JUDGE-GAP-QUALIFICATION-V1\0";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationV1 {
    pub schema_version: String,
    pub assay_id: String,
    pub frozen_at: String,
    pub assignment_algorithm: String,
    pub assignment_seed_commitment_sha256: String,
    pub assignment_seed_revealed: String,
    pub judge: JudgeV1,
    pub reference: ReferenceV1,
    pub items: Vec<ItemV1>,
    pub difficulty_bins: Vec<DifficultyBinV1>,
    pub pairs: Vec<PairV1>,
    pub claim_ceiling: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JudgeV1 {
    pub judge_id: String,
    pub judge_version: String,
    pub adapter_sha256: String,
    pub protocol_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceV1 {
    pub reference_id: String,
    pub source_kind: String,
    pub cohort_sha256: String,
    pub estimator_sha256: String,
    pub minimum_directional_gap: f64,
    pub confidence_z: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ItemV1 {
    pub item_id: String,
    pub reference_strength_logit: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DifficultyBinV1 {
    pub name: String,
    pub min_gap: f64,
    pub max_gap_exclusive: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairV1 {
    pub pair_id: String,
    pub item_a_id: String,
    pub item_b_id: String,
    pub reference_gap_standard_error_logit: f64,
    pub presentations: Vec<PresentationV1>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PresentationV1 {
    pub presentation_id: String,
    pub ordinal: u8,
    pub left_item_id: String,
    pub choice: String,
    pub prompt_sha256: String,
    pub response_sha256: String,
}

#[derive(Debug, Serialize)]
pub struct QualificationReport {
    pub schema_version: String,
    pub assay_id: String,
    pub judge_id: String,
    pub judge_version: String,
    pub valid: bool,
    pub pair_count: usize,
    pub presentation_count: usize,
    pub complete_pair_count: usize,
    pub swap_stable_pair_count: usize,
    pub swap_inconsistent_pair_count: usize,
    pub incomplete_pair_count: usize,
    pub swap_stability_rate: Option<f64>,
    pub left_choice_rate: Option<f64>,
    pub gold_directional_pair_count: usize,
    pub stable_directional_decision_count: usize,
    pub gold_agreement_count: usize,
    pub gold_coverage: Option<f64>,
    pub conditional_gold_accuracy: Option<f64>,
    pub effective_gold_accuracy: Option<f64>,
    pub observed_item_count: usize,
    pub ranking_graph_connected: bool,
    pub gap_bins: Vec<GapBinReport>,
    pub judge_bradley_terry_ranking: Vec<ItemRankReport>,
    pub spearman_rank_correlation_to_gold: Option<f64>,
    pub claim_ceiling: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct GapBinReport {
    pub name: String,
    pub min_gap: f64,
    pub max_gap_exclusive: Option<f64>,
    pub pair_count: usize,
    pub gold_directional_count: usize,
    pub stable_directional_count: usize,
    pub swap_stable_count: usize,
    pub swap_inconsistent_count: usize,
    pub incomplete_count: usize,
    pub correct_count: usize,
    pub coverage: Option<f64>,
    pub conditional_accuracy: Option<f64>,
    pub effective_accuracy: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct ItemRankReport {
    pub item_id: String,
    pub human_reference_strength_logit: f64,
    pub judge_bradley_terry_strength: f64,
}

#[derive(Debug, Default)]
struct BinCounter {
    pair_count: usize,
    gold_directional_count: usize,
    stable_directional_count: usize,
    swap_stable_count: usize,
    swap_inconsistent_count: usize,
    incomplete_count: usize,
    correct_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParsedChoice {
    Left,
    Right,
    Tie,
    Abstain,
    Invalid,
}

#[derive(Debug, Clone, Copy)]
struct BtObservation {
    item_a: usize,
    item_b: usize,
    winner: usize,
}

pub fn parse_and_qualify(input: &str) -> Result<QualificationReport, String> {
    let qualification: QualificationV1 = serde_json::from_str(input)
        .map_err(|error| format!("invalid judge qualification JSON: {error}"))?;
    qualify(&qualification)
}

pub fn qualify(input: &QualificationV1) -> Result<QualificationReport, String> {
    validate_header(input)?;

    let mut item_indices = BTreeMap::new();
    for (index, item) in input.items.iter().enumerate() {
        if item.item_id.trim().is_empty() {
            return Err("item_id must not be empty".to_owned());
        }
        if !item.reference_strength_logit.is_finite() {
            return Err(format!("invalid human reference strength for {}", item.item_id));
        }
        if item_indices.insert(item.item_id.clone(), index).is_some() {
            return Err(format!("duplicate item_id: {}", item.item_id));
        }
    }

    validate_bins(&input.difficulty_bins)?;

    let mut pair_ids = BTreeSet::new();
    let mut unordered_pairs = BTreeSet::new();
    let mut presentation_ids = BTreeSet::new();
    let mut bin_counters: Vec<BinCounter> =
        (0..input.difficulty_bins.len()).map(|_| BinCounter::default()).collect();
    let mut bt_observations = Vec::new();

    let mut complete_pair_count = 0usize;
    let mut swap_stable_pair_count = 0usize;
    let mut swap_inconsistent_pair_count = 0usize;
    let mut incomplete_pair_count = 0usize;
    let mut left_choices = 0usize;
    let mut decisive_position_choices = 0usize;
    let mut gold_directional_pair_count = 0usize;
    let mut stable_directional_decision_count = 0usize;
    let mut gold_agreement_count = 0usize;

    for pair in &input.pairs {
        if pair.pair_id.trim().is_empty() {
            return Err("pair_id must not be empty".to_owned());
        }
        if !pair_ids.insert(pair.pair_id.clone()) {
            return Err(format!("duplicate pair_id: {}", pair.pair_id));
        }
        if pair.item_a_id == pair.item_b_id {
            return Err(format!("pair {} compares an item with itself", pair.pair_id));
        }
        let item_a_index = *item_indices
            .get(&pair.item_a_id)
            .ok_or_else(|| format!("pair {} references unknown item {}", pair.pair_id, pair.item_a_id))?;
        let item_b_index = *item_indices
            .get(&pair.item_b_id)
            .ok_or_else(|| format!("pair {} references unknown item {}", pair.pair_id, pair.item_b_id))?;
        let canonical_pair = if pair.item_a_id < pair.item_b_id {
            (pair.item_a_id.clone(), pair.item_b_id.clone())
        } else {
            (pair.item_b_id.clone(), pair.item_a_id.clone())
        };
        if !unordered_pairs.insert(canonical_pair) {
            return Err(format!("duplicate unordered item pair: {}", pair.pair_id));
        }
        if pair.presentations.len() != 2 {
            return Err(format!("pair {} must have exactly two swapped-order presentations", pair.pair_id));
        }

        let first_is_a_left = assignment_is_a_left(
            &input.assignment_seed_revealed,
            &input.assay_id,
            &pair.pair_id,
        );
        let expected_first_left = if first_is_a_left {
            pair.item_a_id.as_str()
        } else {
            pair.item_b_id.as_str()
        };
        let expected_second_left = if first_is_a_left {
            pair.item_b_id.as_str()
        } else {
            pair.item_a_id.as_str()
        };

        let first = &pair.presentations[0];
        let second = &pair.presentations[1];
        validate_presentation(first, 1, expected_first_left, pair, &mut presentation_ids)?;
        validate_presentation(second, 2, expected_second_left, pair, &mut presentation_ids)?;

        let first_choice = parse_choice(&first.choice)?;
        let second_choice = parse_choice(&second.choice)?;

        for presentation in [first, second] {
            match parse_choice(&presentation.choice)? {
                ParsedChoice::Left | ParsedChoice::Right => {
                    decisive_position_choices += 1;
                    if presentation.choice == "left" {
                        left_choices += 1;
                    }
                }
                ParsedChoice::Tie | ParsedChoice::Abstain | ParsedChoice::Invalid => {}
            }
        }

        let first_selected = selected_item(first, pair, first_choice);
        let second_selected = selected_item(second, pair, second_choice);
        let first_complete = matches!(first_choice, ParsedChoice::Left | ParsedChoice::Right | ParsedChoice::Tie);
        let second_complete = matches!(second_choice, ParsedChoice::Left | ParsedChoice::Right | ParsedChoice::Tie);
        let incomplete = !first_complete || !second_complete;
        let stable_tie = first_choice == ParsedChoice::Tie && second_choice == ParsedChoice::Tie;
        let stable_winner = match (first_selected, second_selected) {
            (Some(left), Some(right)) if left == right => Some(left),
            _ => None,
        };
        let swap_stable = stable_winner.is_some() || stable_tie;
        let swap_inconsistent = !incomplete && !swap_stable;

        if incomplete {
            incomplete_pair_count += 1;
        } else {
            complete_pair_count += 1;
        }
        if swap_stable {
            swap_stable_pair_count += 1;
        }
        if swap_inconsistent {
            swap_inconsistent_pair_count += 1;
        }

        let item_a = &input.items[item_a_index];
        let item_b = &input.items[item_b_index];
        let gap = (item_a.reference_strength_logit - item_b.reference_strength_logit).abs();
        if !pair.reference_gap_standard_error_logit.is_finite()
            || pair.reference_gap_standard_error_logit < 0.0
        {
            return Err(format!("pair {} has an invalid reference gap standard error", pair.pair_id));
        }
        let uncertainty_margin =
            input.reference.confidence_z * pair.reference_gap_standard_error_logit;
        let gold_directional = gap - uncertainty_margin > input.reference.minimum_directional_gap;
        let gold_winner_index = if gold_directional {
            Some(if item_a.reference_strength_logit > item_b.reference_strength_logit {
                item_a_index
            } else {
                item_b_index
            })
        } else {
            None
        };
        let bin_index = find_bin(&input.difficulty_bins, gap)
            .ok_or_else(|| format!("no difficulty bin covers pair {} with gap {gap}", pair.pair_id))?;

        let counter = &mut bin_counters[bin_index];
        counter.pair_count += 1;
        if gold_directional {
            counter.gold_directional_count += 1;
            gold_directional_pair_count += 1;
        }
        if swap_stable {
            counter.swap_stable_count += 1;
        }
        if swap_inconsistent {
            counter.swap_inconsistent_count += 1;
        }
        if incomplete {
            counter.incomplete_count += 1;
        }

        if gold_directional {
            if let (Some(gold_winner), Some(judge_winner_id)) = (gold_winner_index, stable_winner) {
                let judge_winner = *item_indices
                    .get(&judge_winner_id)
                    .expect("selected candidate must be one of the pair's known items");
                counter.stable_directional_count += 1;
                stable_directional_decision_count += 1;
                bt_observations.push(BtObservation {
                    item_a: item_a_index,
                    item_b: item_b_index,
                    winner: judge_winner,
                });
                if gold_winner == judge_winner {
                    counter.correct_count += 1;
                    gold_agreement_count += 1;
                }
            }
        }
    }

    let observed_items: BTreeSet<usize> = bt_observations
        .iter()
        .flat_map(|observation| [observation.item_a, observation.item_b])
        .collect();
    let ranking_graph_connected =
        comparison_graph_connected(input.items.len(), &bt_observations);
    let fitted = if ranking_graph_connected {
        fit_bradley_terry(input.items.len(), &bt_observations)
    } else {
        None
    };
    let spearman = fitted.as_ref().and_then(|scores| {
        let gold: Vec<f64> = input.items.iter().map(|item| item.reference_strength_logit).collect();
        spearman(&gold, scores)
    });
    let judge_bradley_terry_ranking = match fitted {
        Some(scores) => input
            .items
            .iter()
            .zip(scores)
            .map(|(item, strength)| ItemRankReport {
                item_id: item.item_id.clone(),
                human_reference_strength_logit: item.reference_strength_logit,
                judge_bradley_terry_strength: strength,
            })
            .collect(),
        None => Vec::new(),
    };

    let gap_bins = input
        .difficulty_bins
        .iter()
        .zip(bin_counters)
        .map(|(bin, counter)| GapBinReport {
            name: bin.name.clone(),
            min_gap: bin.min_gap,
            max_gap_exclusive: bin.max_gap_exclusive,
            pair_count: counter.pair_count,
            gold_directional_count: counter.gold_directional_count,
            stable_directional_count: counter.stable_directional_count,
            swap_stable_count: counter.swap_stable_count,
            swap_inconsistent_count: counter.swap_inconsistent_count,
            incomplete_count: counter.incomplete_count,
            correct_count: counter.correct_count,
            coverage: rate(counter.stable_directional_count, counter.gold_directional_count),
            conditional_accuracy: rate(counter.correct_count, counter.stable_directional_count),
            effective_accuracy: rate(counter.correct_count, counter.gold_directional_count),
        })
        .collect();

    Ok(QualificationReport {
        schema_version: SCHEMA_VERSION.to_owned(),
        assay_id: input.assay_id.clone(),
        judge_id: input.judge.judge_id.clone(),
        judge_version: input.judge.judge_version.clone(),
        valid: true,
        pair_count: input.pairs.len(),
        presentation_count: input.pairs.len() * 2,
        complete_pair_count,
        swap_stable_pair_count,
        swap_inconsistent_pair_count,
        incomplete_pair_count,
        swap_stability_rate: rate(swap_stable_pair_count, complete_pair_count),
        left_choice_rate: rate(left_choices, decisive_position_choices),
        gold_directional_pair_count,
        stable_directional_decision_count,
        gold_agreement_count,
        gold_coverage: rate(stable_directional_decision_count, gold_directional_pair_count),
        conditional_gold_accuracy: rate(gold_agreement_count, stable_directional_decision_count),
        effective_gold_accuracy: rate(gold_agreement_count, gold_directional_pair_count),
        observed_item_count: observed_items.len(),
        ranking_graph_connected,
        gap_bins,
        judge_bradley_terry_ranking,
        spearman_rank_correlation_to_gold: spearman,
        claim_ceiling: input.claim_ceiling.clone(),
    })
}

fn validate_header(input: &QualificationV1) -> Result<(), String> {
    if input.schema_version != SCHEMA_VERSION {
        return Err(format!("unsupported schema_version: {}", input.schema_version));
    }
    if input.assay_id.trim().is_empty() || input.frozen_at.trim().is_empty() {
        return Err("assay_id and frozen_at must not be empty".to_owned());
    }
    if input.assignment_algorithm != ASSIGNMENT_ALGORITHM {
        return Err(format!("unsupported assignment_algorithm: {}", input.assignment_algorithm));
    }
    if !is_lower_hex_64(&input.assignment_seed_revealed)
        || !is_lower_hex_64(&input.assignment_seed_commitment_sha256)
    {
        return Err("assignment seed and commitment must be 64 lowercase hexadecimal characters".to_owned());
    }
    if sha256_hex(input.assignment_seed_revealed.as_bytes()) != input.assignment_seed_commitment_sha256 {
        return Err("assignment seed does not match its SHA-256 commitment".to_owned());
    }
    if input.judge.judge_id.trim().is_empty() || input.judge.judge_version.trim().is_empty() {
        return Err("judge identity must not be empty".to_owned());
    }
    if !is_lower_hex_64(&input.judge.adapter_sha256) || !is_lower_hex_64(&input.judge.protocol_sha256) {
        return Err("judge adapter and protocol digests must be SHA-256".to_owned());
    }
    if input.reference.reference_id.trim().is_empty()
        || input.reference.source_kind != "human_bradley_terry_mle"
        || !is_lower_hex_64(&input.reference.cohort_sha256)
        || !is_lower_hex_64(&input.reference.estimator_sha256)
        || !input.reference.minimum_directional_gap.is_finite()
        || input.reference.minimum_directional_gap < 0.0
        || !input.reference.confidence_z.is_finite()
        || input.reference.confidence_z <= 0.0
    {
        return Err("invalid human reference provenance or directional-gap policy".to_owned());
    }
    if input.items.len() < 3 {
        return Err("at least three items are required to assess a ranking".to_owned());
    }
    if input.pairs.is_empty() {
        return Err("pairs must not be empty".to_owned());
    }
    if input.claim_ceiling.is_empty() || input.claim_ceiling.iter().any(|claim| claim.trim().is_empty()) {
        return Err("claim_ceiling must contain non-empty limits".to_owned());
    }
    Ok(())
}

fn validate_bins(bins: &[DifficultyBinV1]) -> Result<(), String> {
    if bins.is_empty() {
        return Err("difficulty_bins must not be empty".to_owned());
    }
    let mut names = BTreeSet::new();
    let mut expected_min = 0.0;
    for (index, bin) in bins.iter().enumerate() {
        if bin.name.trim().is_empty() || !names.insert(bin.name.clone()) {
            return Err("difficulty bin names must be non-empty and unique".to_owned());
        }
        if !bin.min_gap.is_finite() || (bin.min_gap - expected_min).abs() > 1e-12 {
            return Err(format!("difficulty bins must be contiguous from zero; invalid minimum for {}", bin.name));
        }
        if let Some(maximum) = bin.max_gap_exclusive {
            if !maximum.is_finite() || maximum <= bin.min_gap {
                return Err(format!("invalid upper bound for difficulty bin {}", bin.name));
            }
            expected_min = maximum;
            if index + 1 == bins.len() {
                return Err("last difficulty bin must have no upper bound".to_owned());
            }
        } else if index + 1 != bins.len() {
            return Err("only the last difficulty bin may be open-ended".to_owned());
        }
    }
    if bins.last().is_some_and(|bin| bin.max_gap_exclusive.is_some()) {
        return Err("last difficulty bin must be open-ended".to_owned());
    }
    Ok(())
}

fn validate_presentation(
    presentation: &PresentationV1,
    expected_ordinal: u8,
    expected_left_item_id: &str,
    pair: &PairV1,
    used_ids: &mut BTreeSet<String>,
) -> Result<(), String> {
    if presentation.presentation_id.trim().is_empty() || !used_ids.insert(presentation.presentation_id.clone()) {
        return Err(format!("empty or duplicate presentation_id in pair {}", pair.pair_id));
    }
    if presentation.ordinal != expected_ordinal {
        return Err(format!("pair {} presentations must be ordered 1 then 2", pair.pair_id));
    }
    if presentation.left_item_id != expected_left_item_id {
        return Err(format!("pair {} violates precommitted swapped display order", pair.pair_id));
    }
    if presentation.left_item_id != pair.item_a_id && presentation.left_item_id != pair.item_b_id {
        return Err(format!("pair {} presentation references an unknown left item", pair.pair_id));
    }
    if !is_lower_hex_64(&presentation.prompt_sha256) || !is_lower_hex_64(&presentation.response_sha256) {
        return Err(format!("presentation {} has a malformed evidence digest", presentation.presentation_id));
    }
    parse_choice(&presentation.choice)?;
    Ok(())
}

fn parse_choice(value: &str) -> Result<ParsedChoice, String> {
    match value {
        "left" => Ok(ParsedChoice::Left),
        "right" => Ok(ParsedChoice::Right),
        "tie" => Ok(ParsedChoice::Tie),
        "abstain" => Ok(ParsedChoice::Abstain),
        "invalid" => Ok(ParsedChoice::Invalid),
        other => Err(format!("unsupported judge choice: {other}")),
    }
}

fn selected_item(presentation: &PresentationV1, pair: &PairV1, choice: ParsedChoice) -> Option<String> {
    match choice {
        ParsedChoice::Left => Some(presentation.left_item_id.clone()),
        ParsedChoice::Right => {
            if presentation.left_item_id == pair.item_a_id {
                Some(pair.item_b_id.clone())
            } else {
                Some(pair.item_a_id.clone())
            }
        }
        ParsedChoice::Tie | ParsedChoice::Abstain | ParsedChoice::Invalid => None,
    }
}

fn assignment_is_a_left(seed: &str, assay_id: &str, pair_id: &str) -> bool {
    let mut hasher = Sha256::new();
    hasher.update(ASSIGNMENT_DOMAIN);
    hasher.update(seed.as_bytes());
    hasher.update(b"\0");
    hasher.update(assay_id.as_bytes());
    hasher.update(b"\0");
    hasher.update(pair_id.as_bytes());
    let digest = hasher.finalize();
    (digest[0] & 0x80) == 0
}

fn find_bin(bins: &[DifficultyBinV1], gap: f64) -> Option<usize> {
    bins.iter().position(|bin| {
        gap >= bin.min_gap && bin.max_gap_exclusive.is_none_or(|maximum| gap < maximum)
    })
}

fn is_lower_hex_64(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(input: &[u8]) -> String {
    let digest = Sha256::digest(input);
    let mut encoded = String::with_capacity(64);
    for byte in digest.iter() {
        write!(&mut encoded, "{:02x}", *byte).expect("writing to String cannot fail");
    }
    encoded
}

fn rate(numerator: usize, denominator: usize) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
}

fn comparison_graph_connected(item_count: usize, observations: &[BtObservation]) -> bool {
    if item_count < 2 || observations.is_empty() {
        return false;
    }
    let mut adjacency = vec![Vec::<usize>::new(); item_count];
    for observation in observations {
        adjacency[observation.item_a].push(observation.item_b);
        adjacency[observation.item_b].push(observation.item_a);
    }
    let mut visited = vec![false; item_count];
    let mut stack = vec![0usize];
    visited[0] = true;
    while let Some(item) = stack.pop() {
        for neighbor in &adjacency[item] {
            if !visited[*neighbor] {
                visited[*neighbor] = true;
                stack.push(*neighbor);
            }
        }
    }
    visited.into_iter().all(|seen| seen)
}

fn fit_bradley_terry(item_count: usize, observations: &[BtObservation]) -> Option<Vec<f64>> {
    if observations.is_empty() {
        return None;
    }
    let mut strengths = vec![0.0_f64; item_count];
    let learning_rate = 0.12;
    let ridge = 0.01;
    let denominator = observations.len() as f64;

    for _ in 0..5000 {
        let mut gradient = vec![0.0_f64; item_count];
        for observation in observations {
            let difference = strengths[observation.item_a] - strengths[observation.item_b];
            let probability_a_wins = 1.0 / (1.0 + (-difference).exp());
            let residual = if observation.winner == observation.item_a {
                1.0 - probability_a_wins
            } else {
                -probability_a_wins
            };
            gradient[observation.item_a] += residual;
            gradient[observation.item_b] -= residual;
        }
        for index in 0..item_count {
            gradient[index] = gradient[index] / denominator - ridge * strengths[index];
            strengths[index] += learning_rate * gradient[index];
        }
        let mean = strengths.iter().sum::<f64>() / item_count as f64;
        for strength in &mut strengths {
            *strength -= mean;
        }
    }
    Some(strengths)
}

fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|left, right| values[*left].total_cmp(&values[*right]));
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0usize;
    while start < indices.len() {
        let mut end = start + 1;
        while end < indices.len() && values[indices[start]] == values[indices[end]] {
            end += 1;
        }
        let average_rank = ((start + 1 + end) as f64) / 2.0;
        for index in &indices[start..end] {
            ranks[*index] = average_rank;
        }
        start = end;
    }
    ranks
}

fn spearman(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 3 {
        return None;
    }
    let left_ranks = average_ranks(left);
    let right_ranks = average_ranks(right);
    pearson(&left_ranks, &right_ranks)
}

fn pearson(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 2 {
        return None;
    }
    let left_mean = left.iter().sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().sum::<f64>() / right.len() as f64;
    let mut covariance = 0.0;
    let mut left_variance = 0.0;
    let mut right_variance = 0.0;
    for (left_value, right_value) in left.iter().zip(right) {
        let left_delta = left_value - left_mean;
        let right_delta = right_value - right_mean;
        covariance += left_delta * right_delta;
        left_variance += left_delta.powi(2);
        right_variance += right_delta.powi(2);
    }
    let denominator = (left_variance * right_variance).sqrt();
    if denominator <= f64::EPSILON {
        None
    } else {
        Some(covariance / denominator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../fixtures/valid-judge-gap-v1.json");

    fn fixture() -> QualificationV1 {
        serde_json::from_str(FIXTURE).expect("fixture should deserialize")
    }

    fn choice_for_winner(presentation: &PresentationV1, winner: &str) -> String {
        if presentation.left_item_id == winner {
            "left".to_owned()
        } else {
            "right".to_owned()
        }
    }

    #[test]
    fn valid_fixture_recovers_human_reference_ranking() {
        let report = parse_and_qualify(FIXTURE).expect("fixture should qualify");
        assert!(report.valid);
        assert_eq!(report.pair_count, 10);
        assert_eq!(report.presentation_count, 20);
        assert_eq!(report.swap_stable_pair_count, 10);
        assert_eq!(report.swap_inconsistent_pair_count, 0);
        assert_eq!(report.gold_directional_pair_count, 10);
        assert_eq!(report.gold_agreement_count, 10);
        assert_eq!(report.observed_item_count, 5);
        assert!(report.ranking_graph_connected);
        assert_eq!(report.left_choice_rate, Some(0.5));
        assert!(report.spearman_rank_correlation_to_gold.unwrap() > 0.99);
    }

    #[test]
    fn position_only_judge_is_exposed_by_order_swaps() {
        let mut input = fixture();
        for pair in &mut input.pairs {
            for presentation in &mut pair.presentations {
                presentation.choice = "left".to_owned();
            }
        }
        let report = qualify(&input).expect("position-only responses are structurally valid");
        assert_eq!(report.swap_inconsistent_pair_count, 10);
        assert_eq!(report.swap_stability_rate, Some(0.0));
        assert_eq!(report.left_choice_rate, Some(1.0));
        assert!(!report.ranking_graph_connected);
        assert!(report.spearman_rank_correlation_to_gold.is_none());
    }

    #[test]
    fn gap_conditioning_localizes_far_pair_errors() {
        let mut input = fixture();
        let strengths: BTreeMap<String, f64> = input
            .items
            .iter()
            .map(|item| (item.item_id.clone(), item.reference_strength_logit))
            .collect();
        for pair in &mut input.pairs {
            let gap = (strengths[&pair.item_a_id] - strengths[&pair.item_b_id]).abs();
            let preferred = if gap >= 2.0 {
                if strengths[&pair.item_a_id] < strengths[&pair.item_b_id] {
                    pair.item_a_id.clone()
                } else {
                    pair.item_b_id.clone()
                }
            } else if strengths[&pair.item_a_id] > strengths[&pair.item_b_id] {
                pair.item_a_id.clone()
            } else {
                pair.item_b_id.clone()
            };
            for presentation in &mut pair.presentations {
                presentation.choice = choice_for_winner(presentation, &preferred);
            }
        }
        let report = qualify(&input).expect("gap-conditioned fixture should qualify");
        let near = report.gap_bins.iter().find(|bin| bin.name == "near").unwrap();
        let far = report.gap_bins.iter().find(|bin| bin.name == "far").unwrap();
        assert_eq!(near.conditional_accuracy, Some(1.0));
        assert_eq!(far.conditional_accuracy, Some(0.0));
    }

    #[test]
    fn missing_reverse_presentation_is_rejected() {
        let mut input = fixture();
        input.pairs[0].presentations.pop();
        let error = qualify(&input).expect_err("one-sided pair must fail");
        assert!(error.contains("exactly two swapped-order presentations"));
    }

    #[test]
    fn altered_sequence_is_rejected() {
        let mut input = fixture();
        let wrong_left = input.pairs[0].item_a_id.clone();
        input.pairs[0].presentations[0].left_item_id = wrong_left;
        let error = qualify(&input).expect_err("tampered sequence must fail");
        assert!(error.contains("precommitted swapped display order"));
    }

    #[test]
    fn seed_commitment_mismatch_is_rejected() {
        let mut input = fixture();
        input.assignment_seed_revealed = "0".repeat(64);
        let error = qualify(&input).expect_err("changed seed must fail");
        assert!(error.contains("does not match its SHA-256 commitment"));
    }

    #[test]
    fn uncertain_human_reference_pair_is_not_called_directional() {
        let mut input = fixture();
        input.pairs[0].reference_gap_standard_error_logit = 100.0;
        let report = qualify(&input).expect("wide reference uncertainty should remain reportable");
        assert!(report.gold_directional_pair_count < report.pair_count);
        let near = report.gap_bins.iter().find(|bin| bin.name == "near").unwrap();
        assert!(near.gold_directional_count <= near.pair_count);
    }

    #[test]
    fn invalid_digest_is_rejected() {
        let mut input = fixture();
        input.pairs[0].presentations[0].response_sha256 = "not-a-digest".to_owned();
        let error = qualify(&input).expect_err("invalid evidence digest must fail");
        assert!(error.contains("malformed evidence digest"));
    }
}
