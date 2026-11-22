use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const MANIFEST_SCHEMA: &str = "mel-sealed-evaluation-v1";
const PREDICTIONS_SCHEMA: &str = "mel-sealed-predictions-v1";
const REFERENCE_SCHEMA: &str = "mel-sealed-human-reference-v1";
const ORDER_ALGORITHM: &str = "sha256-pair-order-v1";
const ORDER_DOMAIN: &[u8] = b"MEL-SEALED-EVAL-V1\0";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestV1 {
    schema_version: String,
    experiment_id: String,
    frozen_at_utc: String,
    split: SplitV1,
    judge: JudgeV1,
    protocol: ProtocolV1,
    assignment: AssignmentV1,
    minimum_directional_gap_logit: f64,
    confidence_z: f64,
    difficulty_bins: Vec<DifficultyBinV1>,
    candidates: Vec<CandidateV1>,
    pairs: Vec<PairV1>,
    claim_ceiling: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SplitV1 {
    split_id: String,
    split_kind: String,
    dataset_commitment_sha256: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct JudgeV1 {
    judge_id: String,
    judge_version: String,
    adapter_sha256: String,
    prompt_sha256: String,
    settings_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProtocolV1 {
    comparison_protocol_sha256: String,
    audio_delivery_sha256: String,
    normalization_sha256: String,
    presentation_ui_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AssignmentV1 {
    algorithm: String,
    seed_commitment_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DifficultyBinV1 {
    name: String,
    minimum_gap_logit: f64,
    maximum_gap_logit_exclusive: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CandidateV1 {
    item_id: String,
    asset_ref: String,
    audio_sha256: String,
    prompt_sha256: String,
    generator_id: String,
    generator_version: String,
    generation_config_sha256: String,
    rights_receipt_sha256: String,
    modality: String,
    genre: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PairV1 {
    pair_id: String,
    item_a_id: String,
    item_b_id: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PredictionsV1 {
    schema_version: String,
    experiment_id: String,
    manifest_sha256: String,
    frozen_at_utc: String,
    order_seed_revealed: String,
    freeze_receipt_sha256: String,
    judge: JudgeV1,
    pairs: Vec<PredictedPairV1>,
    claim_ceiling: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PredictedPairV1 {
    pair_id: String,
    probability_a: Option<f64>,
    presentations: Vec<PresentationV1>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PresentationV1 {
    presentation_id: String,
    ordinal: u8,
    left_item_id: String,
    choice: String,
    response_sha256: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct HumanReferenceV1 {
    schema_version: String,
    experiment_id: String,
    manifest_sha256: String,
    predictions_sha256: String,
    reference_id: String,
    cohort_sha256: String,
    estimator_sha256: String,
    released_at_utc: String,
    release_receipt_sha256: String,
    item_strengths: Vec<ItemStrengthV1>,
    pair_uncertainty: Vec<PairUncertaintyV1>,
    claim_ceiling: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ItemStrengthV1 {
    item_id: String,
    strength_logit: f64,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PairUncertaintyV1 {
    pair_id: String,
    gap_standard_error_logit: f64,
}

#[derive(Debug, Serialize)]
pub struct EvaluationReport {
    pub schema_version: String,
    pub experiment_id: String,
    pub manifest_sha256: String,
    pub predictions_sha256: String,
    pub human_reference_sha256: String,
    pub temporal_order_assurance: String,
    pub pair_count: usize,
    pub human_directional_pair_count: usize,
    pub judge_stable_directional_pair_count: usize,
    pub judge_stable_tie_pair_count: usize,
    pub judge_order_inconsistent_pair_count: usize,
    pub judge_incomplete_pair_count: usize,
    pub correct_directional_decision_count: usize,
    pub coverage_of_human_directional_pairs: Option<f64>,
    pub conditional_accuracy: Option<f64>,
    pub effective_accuracy: Option<f64>,
    pub brier_score_a_probability: Option<f64>,
    pub brier_pair_count: usize,
    pub gap_bins: Vec<GapBinReport>,
    pub claim_ceiling: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct GapBinReport {
    pub name: String,
    pub pair_count: usize,
    pub human_directional_count: usize,
    pub judge_stable_directional_count: usize,
    pub correct_count: usize,
    pub coverage: Option<f64>,
    pub conditional_accuracy: Option<f64>,
    pub effective_accuracy: Option<f64>,
}

#[derive(Debug, Default)]
struct BinCounter {
    pair_count: usize,
    human_directional_count: usize,
    judge_stable_directional_count: usize,
    correct_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JudgeOutcome {
    StableDirectional,
    StableTie,
    OrderInconsistent,
    Incomplete,
}

pub fn evaluate(
    manifest_json: &str,
    predictions_json: &str,
    human_reference_json: &str,
) -> Result<EvaluationReport, String> {
    let manifest: ManifestV1 = serde_json::from_str(manifest_json)
        .map_err(|error| format!("invalid manifest JSON: {error}"))?;
    let predictions: PredictionsV1 = serde_json::from_str(predictions_json)
        .map_err(|error| format!("invalid predictions JSON: {error}"))?;
    let human_reference: HumanReferenceV1 = serde_json::from_str(human_reference_json)
        .map_err(|error| format!("invalid human-reference JSON: {error}"))?;

    validate_manifest(&manifest)?;
    validate_prediction_header(&predictions, &manifest, manifest_json)?;
    validate_reference_header(
        &human_reference,
        &manifest,
        &predictions,
        manifest_json,
        predictions_json,
        human_reference_json,
    )?;

    let candidate_ids: BTreeSet<String> = manifest
        .candidates
        .iter()
        .map(|candidate| candidate.item_id.clone())
        .collect();
    let mut pair_by_id = BTreeMap::new();
    for pair in &manifest.pairs {
        pair_by_id.insert(pair.pair_id.as_str(), pair);
    }

    let mut predictions_by_pair = BTreeMap::new();
    let mut presentation_ids = BTreeSet::new();
    for pair in &predictions.pairs {
        if predictions_by_pair.insert(pair.pair_id.as_str(), pair).is_some() {
            return Err(format!("duplicate prediction pair_id: {}", pair.pair_id));
        }
        if pair.presentations.len() != 2 {
            return Err(format!(
                "pair {} must have exactly two swapped-order presentations",
                pair.pair_id
            ));
        }
        if let Some(probability) = pair.probability_a {
            if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
                return Err(format!(
                    "pair {} probability_a must be finite and in [0, 1]",
                    pair.pair_id
                ));
            }
        }
        for presentation in &pair.presentations {
            if presentation.presentation_id.trim().is_empty()
                || !presentation_ids.insert(presentation.presentation_id.clone())
            {
                return Err(format!(
                    "empty or duplicate presentation_id in pair {}",
                    pair.pair_id
                ));
            }
            require_digest(
                &presentation.response_sha256,
                &format!("response_sha256 in {}", presentation.presentation_id),
            )?;
            if !matches!(
                presentation.choice.as_str(),
                "left" | "right" | "tie" | "abstain"
            ) {
                return Err(format!(
                    "unsupported choice {} in {}",
                    presentation.choice, presentation.presentation_id
                ));
            }
        }
    }

    if predictions_by_pair.len() != pair_by_id.len()
        || predictions_by_pair
            .keys()
            .any(|pair_id| !pair_by_id.contains_key(pair_id))
    {
        return Err("prediction pair IDs must exactly match manifest pair IDs".to_owned());
    }

    let strengths: BTreeMap<&str, f64> = human_reference
        .item_strengths
        .iter()
        .map(|item| (item.item_id.as_str(), item.strength_logit))
        .collect();
    if strengths.len() != human_reference.item_strengths.len()
        || strengths.len() != candidate_ids.len()
        || strengths.keys().any(|item_id| !candidate_ids.contains(*item_id))
        || human_reference
            .item_strengths
            .iter()
            .any(|item| !item.strength_logit.is_finite())
    {
        return Err("human item strengths must uniquely cover every manifest candidate".to_owned());
    }

    let uncertainty: BTreeMap<&str, f64> = human_reference
        .pair_uncertainty
        .iter()
        .map(|pair| (pair.pair_id.as_str(), pair.gap_standard_error_logit))
        .collect();
    if uncertainty.len() != human_reference.pair_uncertainty.len()
        || uncertainty.len() != pair_by_id.len()
        || uncertainty.keys().any(|pair_id| !pair_by_id.contains_key(pair_id))
        || human_reference
            .pair_uncertainty
            .iter()
            .any(|pair| !pair.gap_standard_error_logit.is_finite()
                || pair.gap_standard_error_logit < 0.0)
    {
        return Err("pair uncertainty must uniquely cover every manifest pair".to_owned());
    }

    let mut counters: BTreeMap<String, BinCounter> = manifest
        .difficulty_bins
        .iter()
        .map(|bin| (bin.name.clone(), BinCounter::default()))
        .collect();

    let mut human_directional_pair_count = 0usize;
    let mut judge_stable_directional_pair_count = 0usize;
    let mut judge_stable_tie_pair_count = 0usize;
    let mut judge_order_inconsistent_pair_count = 0usize;
    let mut judge_incomplete_pair_count = 0usize;
    let mut correct_directional_decision_count = 0usize;
    let mut brier_sum = 0.0_f64;
    let mut brier_pair_count = 0usize;

    for pair in &manifest.pairs {
        let predicted = predictions_by_pair
            .get(pair.pair_id.as_str())
            .ok_or_else(|| format!("missing prediction for pair {}", pair.pair_id))?;
        let first_a_left = assignment_selects_a_left(
            &predictions.order_seed_revealed,
            &manifest.experiment_id,
            &pair.pair_id,
        );
        let expected_first_left = if first_a_left {
            pair.item_a_id.as_str()
        } else {
            pair.item_b_id.as_str()
        };
        let expected_second_left = if first_a_left {
            pair.item_b_id.as_str()
        } else {
            pair.item_a_id.as_str()
        };

        let first = &predicted.presentations[0];
        let second = &predicted.presentations[1];
        validate_presentation(first, 1, expected_first_left, pair)?;
        validate_presentation(second, 2, expected_second_left, pair)?;

        let first_selected = selected_item(first, pair);
        let second_selected = selected_item(second, pair);
        let outcome = classify_outcome(first, second, first_selected, second_selected);
        match outcome {
            JudgeOutcome::StableDirectional => judge_stable_directional_pair_count += 1,
            JudgeOutcome::StableTie => judge_stable_tie_pair_count += 1,
            JudgeOutcome::OrderInconsistent => judge_order_inconsistent_pair_count += 1,
            JudgeOutcome::Incomplete => judge_incomplete_pair_count += 1,
        }

        let gap = strengths[pair.item_a_id.as_str()] - strengths[pair.item_b_id.as_str()];
        let absolute_gap = gap.abs();
        let standard_error = uncertainty[pair.pair_id.as_str()];
        let is_human_directional = absolute_gap
            > manifest.minimum_directional_gap_logit + manifest.confidence_z * standard_error;
        let human_winner = if gap > 0.0 {
            Some(pair.item_a_id.as_str())
        } else if gap < 0.0 {
            Some(pair.item_b_id.as_str())
        } else {
            None
        };

        let bin_index = find_bin(&manifest.difficulty_bins, absolute_gap)
            .ok_or_else(|| format!("no difficulty bin contains pair {}", pair.pair_id))?;
        let bin = &manifest.difficulty_bins[bin_index];
        let counter = counters
            .get_mut(&bin.name)
            .ok_or_else(|| format!("missing difficulty counter for {}", bin.name))?;
        counter.pair_count += 1;

        if is_human_directional {
            human_directional_pair_count += 1;
            counter.human_directional_count += 1;
            if let JudgeOutcome::StableDirectional = outcome {
                counter.judge_stable_directional_count += 1;
                let selected = first_selected.ok_or_else(|| {
                    format!("stable directional pair {} has no selected item", pair.pair_id)
                })?;
                if Some(selected) == human_winner {
                    correct_directional_decision_count += 1;
                    counter.correct_count += 1;
                }
            }
        }

        if let Some(probability_a) = predicted.probability_a {
            let human_probability_a = logistic(gap);
            brier_sum += (probability_a - human_probability_a).powi(2);
            brier_pair_count += 1;
        }
    }

    let judged_gold_directional: usize = counters
        .values()
        .map(|counter| counter.judge_stable_directional_count)
        .sum();
    let gap_bins = manifest
        .difficulty_bins
        .iter()
        .map(|bin| {
            let counter = counters.get(&bin.name).expect("bin counter was created");
            GapBinReport {
                name: bin.name.clone(),
                pair_count: counter.pair_count,
                human_directional_count: counter.human_directional_count,
                judge_stable_directional_count: counter.judge_stable_directional_count,
                correct_count: counter.correct_count,
                coverage: rate(
                    counter.judge_stable_directional_count,
                    counter.human_directional_count,
                ),
                conditional_accuracy: rate(
                    counter.correct_count,
                    counter.judge_stable_directional_count,
                ),
                effective_accuracy: rate(
                    counter.correct_count,
                    counter.human_directional_count,
                ),
            }
        })
        .collect();

    let mut claim_ceiling = manifest.claim_ceiling.clone();
    claim_ceiling.extend(predictions.claim_ceiling.iter().cloned());
    claim_ceiling.extend(human_reference.claim_ceiling.iter().cloned());

    Ok(EvaluationReport {
        schema_version: "mel-sealed-evaluation-report-v1".to_owned(),
        experiment_id: manifest.experiment_id,
        manifest_sha256: sha256_hex(manifest_json.as_bytes()),
        predictions_sha256: sha256_hex(predictions_json.as_bytes()),
        human_reference_sha256: sha256_hex(human_reference_json.as_bytes()),
        temporal_order_assurance: "declared_metadata_only_external_attestation_unverified".to_owned(),
        pair_count: manifest.pairs.len(),
        human_directional_pair_count,
        judge_stable_directional_pair_count: judged_gold_directional,
        judge_stable_tie_pair_count,
        judge_order_inconsistent_pair_count,
        judge_incomplete_pair_count,
        correct_directional_decision_count,
        coverage_of_human_directional_pairs: rate(
            judged_gold_directional,
            human_directional_pair_count,
        ),
        conditional_accuracy: rate(
            correct_directional_decision_count,
            judged_gold_directional,
        ),
        effective_accuracy: rate(
            correct_directional_decision_count,
            human_directional_pair_count,
        ),
        brier_score_a_probability: if brier_pair_count == 0 {
            None
        } else {
            Some(brier_sum / brier_pair_count as f64)
        },
        brier_pair_count,
        gap_bins,
        claim_ceiling,
    })
}

fn validate_manifest(manifest: &ManifestV1) -> Result<(), String> {
    if manifest.schema_version != MANIFEST_SCHEMA
        || manifest.experiment_id.trim().is_empty()
        || !is_canonical_utc_timestamp(&manifest.frozen_at_utc)
    {
        return Err("manifest schema, experiment_id or frozen_at_utc is invalid".to_owned());
    }
    if manifest.split.split_id.trim().is_empty() || manifest.split.split_kind != "held_out" {
        return Err("manifest must identify a held_out split".to_owned());
    }
    require_digest(
        &manifest.split.dataset_commitment_sha256,
        "dataset_commitment_sha256",
    )?;
    validate_judge(&manifest.judge)?;
    for (name, digest) in [
        ("comparison_protocol_sha256", &manifest.protocol.comparison_protocol_sha256),
        ("audio_delivery_sha256", &manifest.protocol.audio_delivery_sha256),
        ("normalization_sha256", &manifest.protocol.normalization_sha256),
        ("presentation_ui_sha256", &manifest.protocol.presentation_ui_sha256),
    ] {
        require_digest(digest, name)?;
    }
    if manifest.assignment.algorithm != ORDER_ALGORITHM {
        return Err(format!(
            "unsupported order algorithm: {}",
            manifest.assignment.algorithm
        ));
    }
    require_digest(
        &manifest.assignment.seed_commitment_sha256,
        "assignment seed commitment",
    )?;
    if !manifest.minimum_directional_gap_logit.is_finite()
        || manifest.minimum_directional_gap_logit < 0.0
        || !manifest.confidence_z.is_finite()
        || manifest.confidence_z <= 0.0
    {
        return Err("directional-gap and confidence thresholds must be finite and valid".to_owned());
    }
    validate_bins(&manifest.difficulty_bins)?;
    if manifest.candidates.len() < 2 || manifest.pairs.is_empty() {
        return Err("manifest needs at least two candidates and one pair".to_owned());
    }
    let mut candidate_ids = BTreeSet::new();
    for candidate in &manifest.candidates {
        if candidate.item_id.trim().is_empty()
            || candidate.asset_ref.trim().is_empty()
            || candidate.generator_id.trim().is_empty()
            || candidate.generator_version.trim().is_empty()
            || candidate.modality.trim().is_empty()
            || candidate.genre.trim().is_empty()
            || !candidate_ids.insert(candidate.item_id.as_str())
        {
            return Err("candidate identity or provenance fields are empty or duplicated".to_owned());
        }
        for (name, digest) in [
            ("audio_sha256", &candidate.audio_sha256),
            ("prompt_sha256", &candidate.prompt_sha256),
            ("generation_config_sha256", &candidate.generation_config_sha256),
            ("rights_receipt_sha256", &candidate.rights_receipt_sha256),
        ] {
            require_digest(digest, &format!("{name} for {}", candidate.item_id))?;
        }
    }
    let mut pair_ids = BTreeSet::new();
    for pair in &manifest.pairs {
        if pair.pair_id.trim().is_empty()
            || pair.item_a_id == pair.item_b_id
            || !candidate_ids.contains(pair.item_a_id.as_str())
            || !candidate_ids.contains(pair.item_b_id.as_str())
            || !pair_ids.insert(pair.pair_id.as_str())
        {
            return Err(format!("invalid or duplicate manifest pair {}", pair.pair_id));
        }
    }
    validate_claim_ceiling(&manifest.claim_ceiling, "manifest")
}

fn validate_prediction_header(
    predictions: &PredictionsV1,
    manifest: &ManifestV1,
    manifest_json: &str,
) -> Result<(), String> {
    if predictions.schema_version != PREDICTIONS_SCHEMA
        || predictions.experiment_id != manifest.experiment_id
        || !is_canonical_utc_timestamp(&predictions.frozen_at_utc)
        || predictions.frozen_at_utc <= manifest.frozen_at_utc
    {
        return Err("prediction header does not follow the frozen manifest".to_owned());
    }
    if predictions.manifest_sha256 != sha256_hex(manifest_json.as_bytes()) {
        return Err("predictions do not bind the exact manifest bytes".to_owned());
    }
    if !is_lower_hex_64(&predictions.order_seed_revealed)
        || sha256_hex(predictions.order_seed_revealed.as_bytes())
            != manifest.assignment.seed_commitment_sha256
    {
        return Err("revealed order seed does not match the precommitted seed digest".to_owned());
    }
    require_digest(&predictions.freeze_receipt_sha256, "freeze_receipt_sha256")?;
    if predictions.judge != manifest.judge {
        return Err("prediction judge identity differs from the frozen manifest".to_owned());
    }
    validate_claim_ceiling(&predictions.claim_ceiling, "predictions")
}

fn validate_reference_header(
    reference: &HumanReferenceV1,
    manifest: &ManifestV1,
    predictions: &PredictionsV1,
    manifest_json: &str,
    predictions_json: &str,
    _human_reference_json: &str,
) -> Result<(), String> {
    if reference.schema_version != REFERENCE_SCHEMA
        || reference.experiment_id != manifest.experiment_id
        || reference.reference_id.trim().is_empty()
        || !is_canonical_utc_timestamp(&reference.released_at_utc)
        || reference.released_at_utc <= predictions.frozen_at_utc
    {
        return Err("human reference must be separately released after prediction freeze".to_owned());
    }
    if reference.manifest_sha256 != sha256_hex(manifest_json.as_bytes())
        || reference.predictions_sha256 != sha256_hex(predictions_json.as_bytes())
    {
        return Err("human reference does not bind the exact manifest and frozen prediction bytes".to_owned());
    }
    require_digest(&reference.cohort_sha256, "human cohort digest")?;
    require_digest(&reference.estimator_sha256, "human estimator digest")?;
    require_digest(&reference.release_receipt_sha256, "release_receipt_sha256")?;
    validate_claim_ceiling(&reference.claim_ceiling, "human reference")
}

fn validate_judge(judge: &JudgeV1) -> Result<(), String> {
    if judge.judge_id.trim().is_empty() || judge.judge_version.trim().is_empty() {
        return Err("judge identity must not be empty".to_owned());
    }
    for (name, digest) in [
        ("judge adapter", &judge.adapter_sha256),
        ("judge prompt", &judge.prompt_sha256),
        ("judge settings", &judge.settings_sha256),
    ] {
        require_digest(digest, name)?;
    }
    Ok(())
}

fn validate_bins(bins: &[DifficultyBinV1]) -> Result<(), String> {
    if bins.is_empty() {
        return Err("difficulty_bins must not be empty".to_owned());
    }
    let mut names = BTreeSet::new();
    let mut expected_minimum = 0.0_f64;
    for (index, bin) in bins.iter().enumerate() {
        if bin.name.trim().is_empty()
            || !names.insert(bin.name.as_str())
            || !bin.minimum_gap_logit.is_finite()
            || (bin.minimum_gap_logit - expected_minimum).abs() > 1e-12
        {
            return Err("difficulty bins must have unique names and contiguous finite bounds".to_owned());
        }
        match bin.maximum_gap_logit_exclusive {
            Some(maximum)
                if maximum.is_finite()
                    && maximum > bin.minimum_gap_logit
                    && index + 1 < bins.len() =>
            {
                expected_minimum = maximum;
            }
            Some(_) => {
                return Err("only non-final bins may have a finite exclusive maximum".to_owned());
            }
            None if index + 1 == bins.len() => {}
            None => {
                return Err("only the final difficulty bin may be open-ended".to_owned());
            }
        }
    }
    Ok(())
}

fn validate_presentation(
    presentation: &PresentationV1,
    expected_ordinal: u8,
    expected_left_item_id: &str,
    pair: &PairV1,
) -> Result<(), String> {
    if presentation.ordinal != expected_ordinal
        || presentation.left_item_id != expected_left_item_id
        || (presentation.left_item_id != pair.item_a_id
            && presentation.left_item_id != pair.item_b_id)
    {
        return Err(format!(
            "presentation {} violates the committed swapped order",
            presentation.presentation_id
        ));
    }
    Ok(())
}

fn selected_item<'a>(presentation: &'a PresentationV1, pair: &'a PairV1) -> Option<&'a str> {
    match presentation.choice.as_str() {
        "left" => Some(presentation.left_item_id.as_str()),
        "right" => {
            if presentation.left_item_id == pair.item_a_id {
                Some(pair.item_b_id.as_str())
            } else {
                Some(pair.item_a_id.as_str())
            }
        }
        "tie" | "abstain" => None,
        _ => None,
    }
}

fn classify_outcome(
    first: &PresentationV1,
    second: &PresentationV1,
    first_selected: Option<&str>,
    second_selected: Option<&str>,
) -> JudgeOutcome {
    if first.choice == "abstain" || second.choice == "abstain" {
        JudgeOutcome::Incomplete
    } else if first.choice == "tie" && second.choice == "tie" {
        JudgeOutcome::StableTie
    } else if let (Some(first_item), Some(second_item)) = (first_selected, second_selected) {
        if first_item == second_item {
            JudgeOutcome::StableDirectional
        } else {
            JudgeOutcome::OrderInconsistent
        }
    } else {
        JudgeOutcome::OrderInconsistent
    }
}

fn assignment_selects_a_left(seed: &str, experiment_id: &str, pair_id: &str) -> bool {
    let mut hasher = Sha256::new();
    hasher.update(ORDER_DOMAIN);
    hasher.update(seed.as_bytes());
    hasher.update(b"\0");
    hasher.update(experiment_id.as_bytes());
    hasher.update(b"\0");
    hasher.update(pair_id.as_bytes());
    let digest = hasher.finalize();
    (digest[0] & 0x80) == 0
}

fn find_bin(bins: &[DifficultyBinV1], gap: f64) -> Option<usize> {
    bins.iter().position(|bin| {
        gap >= bin.minimum_gap_logit
            && bin
                .maximum_gap_logit_exclusive
                .is_none_or(|maximum| gap < maximum)
    })
}

fn is_canonical_utc_timestamp(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 20
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
        || bytes[19] != b'Z'
    {
        return false;
    }
    for (index, byte) in bytes.iter().enumerate() {
        if !matches!(index, 4 | 7 | 10 | 13 | 16 | 19) && !byte.is_ascii_digit() {
            return false;
        }
    }
    let month = value[5..7].parse::<u32>().unwrap_or(0);
    let day = value[8..10].parse::<u32>().unwrap_or(0);
    let hour = value[11..13].parse::<u32>().unwrap_or(99);
    let minute = value[14..16].parse::<u32>().unwrap_or(99);
    let second = value[17..19].parse::<u32>().unwrap_or(99);
    month > 0 && month <= 12 && day > 0 && day <= 31 && hour < 24 && minute < 60 && second < 60
}

fn validate_claim_ceiling(claims: &[String], source: &str) -> Result<(), String> {
    if claims.is_empty() || claims.iter().any(|claim| claim.trim().is_empty()) {
        Err(format!("{source} claim_ceiling must contain non-empty limits"))
    } else {
        Ok(())
    }
}

fn require_digest(value: &str, name: &str) -> Result<(), String> {
    if is_lower_hex_64(value) {
        Ok(())
    } else {
        Err(format!("{name} must be a 64-character lowercase SHA-256 digest"))
    }
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
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

fn logistic(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn rate(numerator: usize, denominator: usize) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MANIFEST: &str = include_str!("../fixtures/manifest-v1.json");
    const PREDICTIONS: &str = include_str!("../fixtures/predictions-v1.json");
    const HUMAN_REFERENCE: &str = include_str!("../fixtures/human-reference-v1.json");

    #[test]
    fn separated_fixture_scores_only_after_prediction_freeze() {
        let report = evaluate(MANIFEST, PREDICTIONS, HUMAN_REFERENCE)
            .expect("synthetic sealed fixture should replay");
        assert_eq!(report.pair_count, 5);
        assert_eq!(report.human_directional_pair_count, 4);
        assert_eq!(report.judge_stable_directional_pair_count, 3);
        assert_eq!(report.judge_stable_tie_pair_count, 1);
        assert_eq!(report.judge_order_inconsistent_pair_count, 1);
        assert_eq!(report.judge_incomplete_pair_count, 0);
        assert_eq!(report.correct_directional_decision_count, 2);
        assert_eq!(report.coverage_of_human_directional_pairs, Some(0.75));
        assert!((report.conditional_accuracy.unwrap() - (2.0 / 3.0)).abs() < 1e-12);
        assert_eq!(report.effective_accuracy, Some(0.5));
        assert_eq!(report.temporal_order_assurance, "declared_metadata_only_external_attestation_unverified");
        assert_eq!(report.brier_pair_count, 5);
    }

    #[test]
    fn manifest_byte_change_breaks_prediction_binding() {
        let changed_manifest = format!("{MANIFEST} ");
        let error = evaluate(&changed_manifest, PREDICTIONS, HUMAN_REFERENCE)
            .expect_err("different manifest bytes must fail");
        assert!(error.contains("exact manifest bytes"));
    }

    #[test]
    fn human_reference_cannot_precede_prediction_freeze() {
        let mut reference: serde_json::Value =
            serde_json::from_str(HUMAN_REFERENCE).expect("fixture JSON");
        reference["released_at_utc"] = serde_json::json!("2026-11-22T13:39:59Z");
        let changed_reference = serde_json::to_string(&reference).expect("serialize");
        let error = evaluate(MANIFEST, PREDICTIONS, &changed_reference)
            .expect_err("early reference release must fail");
        assert!(error.contains("released after prediction freeze"));
    }

    #[test]
    fn manifest_rejects_human_reference_fields() {
        let mut manifest: serde_json::Value =
            serde_json::from_str(MANIFEST).expect("fixture JSON");
        manifest["human_reference"] = serde_json::json!({
            "item_strengths": [{"item_id":"c1","strength_logit":123.0}]
        });
        let changed = serde_json::to_string(&manifest).expect("serialize");
        let error = serde_json::from_str::<ManifestV1>(&changed)
            .expect_err("judge manifest must not accept gold fields");
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn order_only_judge_is_not_given_directional_credit() {
        let mut predictions: serde_json::Value =
            serde_json::from_str(PREDICTIONS).expect("fixture JSON");
        for pair in predictions["pairs"].as_array_mut().expect("pairs array") {
            for presentation in pair["presentations"].as_array_mut().expect("presentations") {
                presentation["choice"] = serde_json::json!("left");
            }
        }
        let changed_predictions = serde_json::to_string(&predictions).expect("serialize");
        let mut reference: serde_json::Value =
            serde_json::from_str(HUMAN_REFERENCE).expect("fixture JSON");
        reference["predictions_sha256"] =
            serde_json::json!(sha256_hex(changed_predictions.as_bytes()));
        let changed_reference = serde_json::to_string(&reference).expect("serialize");
        let report = evaluate(MANIFEST, &changed_predictions, &changed_reference)
            .expect("order-only outputs remain structurally valid");
        assert_eq!(report.judge_order_inconsistent_pair_count, 5);
        assert_eq!(report.judge_stable_directional_pair_count, 0);
        assert_eq!(report.correct_directional_decision_count, 0);
        assert_eq!(report.coverage_of_human_directional_pairs, Some(0.0));
    }

    #[test]
    fn altered_order_assignment_is_rejected() {
        let mut predictions: serde_json::Value =
            serde_json::from_str(PREDICTIONS).expect("fixture JSON");
        predictions["pairs"][0]["presentations"][0]["left_item_id"] =
            serde_json::json!("c1");
        let changed_predictions = serde_json::to_string(&predictions).expect("serialize");
        let mut reference: serde_json::Value =
            serde_json::from_str(HUMAN_REFERENCE).expect("fixture JSON");
        reference["predictions_sha256"] =
            serde_json::json!(sha256_hex(changed_predictions.as_bytes()));
        let changed_reference = serde_json::to_string(&reference).expect("serialize");
        let error = evaluate(MANIFEST, &changed_predictions, &changed_reference)
            .expect_err("altered order must fail");
        assert!(error.contains("committed swapped order"));
    }
}
