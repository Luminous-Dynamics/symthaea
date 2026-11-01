//! Reproducible text-overlap baselines and per-tier detector evaluation.
//!
//! Scores are risk-like similarity scores in [0, 1000], not calibrated
//! probabilities. Per-tier confusion matrices are descriptive for the supplied
//! planted controls; uncertainty and generalization require separate analysis.

use crate::{
    ChallengeManifestV1, ExposureMechanismV1, GENERATOR_VERSION, MANIFEST_SCHEMA,
    PUBLIC_PACKET_SCHEMA, PublicChallengePacketV1, SEALED_KEY_SCHEMA, SealedChallengeKeyV1, TierV1,
    hash_domain, validate_pair_packet,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const DETECTOR_SPEC_SCHEMA: &str = "symthaea.contamination_panel_detector_spec.v1";
pub const DETECTOR_RUN_SCHEMA: &str = "symthaea.contamination_panel_detector_run.v1";
pub const DETECTOR_REPORT_SCHEMA: &str = "symthaea.contamination_panel_detector_report.v1";

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DetectorKindV1 {
    ExactTextEquality,
    NormalizedTextEquality,
    TokenNgramJaccard,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorSpecV1 {
    pub schema: String,
    pub detector_id: String,
    pub detector_version: String,
    pub kind: DetectorKindV1,
    pub threshold_permille: u16,
    /// Required in [2, 8] for TokenNgramJaccard; must be zero otherwise.
    pub token_ngram_size: u8,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PairRiskScoreV1 {
    pub pair_id: String,
    pub risk_permille: u16,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorRunV1 {
    pub schema: String,
    pub generation_id: String,
    pub public_packet_sha256: String,
    pub detector_id: String,
    pub detector_version: String,
    pub threshold_permille: u16,
    pub kind: DetectorKindV1,
    pub token_ngram_size: u8,
    pub scores: Vec<PairRiskScoreV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TierConfusionMetricsV1 {
    pub tier: TierV1,
    pub positive_support: u64,
    pub negative_support: u64,
    pub true_positive: u64,
    pub false_positive: u64,
    pub false_negative: u64,
    pub true_negative: u64,
    /// All proportions use integer permille, rounded down. None means the
    /// denominator is zero and the metric is not estimable for this tier.
    pub precision_permille: Option<u16>,
    pub recall_permille: Option<u16>,
    pub specificity_permille: Option<u16>,
    pub false_positive_rate_permille: Option<u16>,
    pub false_discovery_rate_permille: Option<u16>,
    pub accuracy_permille: Option<u16>,
    pub balanced_accuracy_permille: Option<u16>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorEvaluationReportV1 {
    pub schema: String,
    pub generation_id: String,
    pub public_packet_sha256: String,
    pub sealed_key_sha256: String,
    pub manifest_sha256: String,
    pub detector_id: String,
    pub detector_version: String,
    pub detector_kind: DetectorKindV1,
    pub threshold_permille: u16,
    pub pair_count: u64,
    pub metrics_by_tier: Vec<TierConfusionMetricsV1>,
    pub score_semantics: String,
    pub uncertainty_status: String,
    pub claim_boundary: String,
}

fn validate_spec(spec: &DetectorSpecV1) -> Result<(), String> {
    if spec.schema != DETECTOR_SPEC_SCHEMA {
        return Err(format!(
            "detector spec schema must be {DETECTOR_SPEC_SCHEMA}"
        ));
    }
    if spec.detector_id.trim().is_empty()
        || spec.detector_id.len() > 128
        || spec.detector_version.trim().is_empty()
        || spec.detector_version.len() > 128
    {
        return Err(
            "detector_id and detector_version must be nonempty and <=128 characters".to_owned(),
        );
    }
    if spec.threshold_permille > 1_000 {
        return Err("threshold_permille must be in [0, 1000]".to_owned());
    }
    match spec.kind {
        DetectorKindV1::TokenNgramJaccard if !(2..=8).contains(&spec.token_ngram_size) => {
            Err("token_ngram_size must be in [2, 8] for TokenNgramJaccard".to_owned())
        }
        DetectorKindV1::ExactTextEquality | DetectorKindV1::NormalizedTextEquality
            if spec.token_ngram_size != 0 =>
        {
            Err("token_ngram_size must be zero for non-ngram detectors".to_owned())
        }
        _ => Ok(()),
    }
}

fn normalize_text(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    let mut previous_space = true;
    for character in value.chars() {
        if character.is_ascii_alphanumeric() {
            output.push(character.to_ascii_lowercase());
            previous_space = false;
        } else if character.is_whitespace() || character.is_ascii_punctuation() {
            if !previous_space {
                output.push(' ');
                previous_space = true;
            }
        } else {
            // Preserve non-ASCII letters as a separated unit. This baseline is
            // intentionally not a Unicode normalization or transliteration tool.
            if !previous_space {
                output.push(' ');
            }
            output.push(character);
            output.push(' ');
            previous_space = true;
        }
    }
    output.trim().to_owned()
}

fn token_ngrams(value: &str, n: usize) -> BTreeSet<String> {
    let normalized = normalize_text(value);
    let tokens: Vec<&str> = normalized.split_whitespace().collect();
    if tokens.len() < n {
        return BTreeSet::new();
    }
    tokens.windows(n).map(|window| window.join(" ")).collect()
}

fn jaccard_permille(left: &BTreeSet<String>, right: &BTreeSet<String>) -> u16 {
    let intersection = left.intersection(right).count() as u128;
    let union = left.len() as u128 + right.len() as u128 - intersection;
    if union == 0 {
        return 0;
    }
    ((intersection * 1_000) / union).min(1_000) as u16
}

fn pair_score(pair: &crate::PublicPairV1, spec: &DetectorSpecV1) -> u16 {
    match spec.kind {
        DetectorKindV1::ExactTextEquality => {
            if pair.left_text == pair.right_text {
                1_000
            } else {
                0
            }
        }
        DetectorKindV1::NormalizedTextEquality => {
            if normalize_text(&pair.left_text) == normalize_text(&pair.right_text) {
                1_000
            } else {
                0
            }
        }
        DetectorKindV1::TokenNgramJaccard => {
            let n = usize::from(spec.token_ngram_size);
            jaccard_permille(
                &token_ngrams(&pair.left_text, n),
                &token_ngrams(&pair.right_text, n),
            )
        }
    }
}

/// Run one deterministic built-in detector against the public packet only.
/// No sealed labels, tier metadata, or source lineage is accepted by this API.
pub fn score_public_packet(
    packet: &PublicChallengePacketV1,
    spec: &DetectorSpecV1,
) -> Result<DetectorRunV1, String> {
    validate_pair_packet(packet)?;
    validate_spec(spec)?;

    let packet_bytes = serde_json::to_vec_pretty(packet)
        .map_err(|error| format!("cannot serialize public packet: {error}"))?;
    let public_packet_sha256 = hash_domain(b"public-packet.v1", &packet_bytes);
    let mut scores = Vec::with_capacity(packet.pairs.len());
    for pair in &packet.pairs {
        scores.push(PairRiskScoreV1 {
            pair_id: pair.pair_id.clone(),
            risk_permille: pair_score(pair, spec),
        });
    }

    Ok(DetectorRunV1 {
        schema: DETECTOR_RUN_SCHEMA.to_owned(),
        generation_id: packet.generation_id.clone(),
        public_packet_sha256,
        detector_id: spec.detector_id.clone(),
        detector_version: spec.detector_version.clone(),
        threshold_permille: spec.threshold_permille,
        kind: spec.kind,
        token_ngram_size: spec.token_ngram_size,
        scores,
    })
}

fn ratio_permille(numerator: u64, denominator: u64) -> Option<u16> {
    if denominator == 0 {
        return None;
    }
    let scaled = (u128::from(numerator) * 1_000) / u128::from(denominator);
    Some(scaled.min(1_000) as u16)
}

fn balanced_accuracy_permille(recall: Option<u16>, specificity: Option<u16>) -> Option<u16> {
    match (recall, specificity) {
        (Some(recall), Some(specificity)) => {
            Some(((u32::from(recall) + u32::from(specificity)) / 2) as u16)
        }
        _ => None,
    }
}

/// Evaluate a committed detector run against the separately supplied sealed
/// key. This should execute only inside the evaluator boundary after detector
/// scoring is complete.
pub fn evaluate_detector_run(
    packet: &PublicChallengePacketV1,
    sealed_key: &SealedChallengeKeyV1,
    manifest: &ChallengeManifestV1,
    run: &DetectorRunV1,
) -> Result<DetectorEvaluationReportV1, String> {
    validate_pair_packet(packet)?;
    if sealed_key.schema != SEALED_KEY_SCHEMA {
        return Err(format!("sealed key schema must be {SEALED_KEY_SCHEMA}"));
    }
    if manifest.schema != MANIFEST_SCHEMA || manifest.generator_version != GENERATOR_VERSION {
        return Err("challenge manifest schema or generator version is unsupported".to_owned());
    }
    if manifest.pairs_per_tier == 0 || manifest.pairs_per_tier > crate::MAX_PAIRS_PER_TIER {
        return Err("manifest pairs_per_tier is outside the supported range".to_owned());
    }
    let expected_pair_count = u64::from(manifest.pairs_per_tier)
        .checked_mul(8)
        .ok_or_else(|| "manifest pair count overflows u64".to_owned())?;
    if manifest.pair_count != expected_pair_count
        || u64::try_from(packet.pairs.len()).ok() != Some(expected_pair_count)
    {
        return Err("manifest pair count does not match the declared generator design".to_owned());
    }
    if !crate::valid_digest(&manifest.generator_config_sha256)
        || !crate::valid_digest(&manifest.public_packet_sha256)
        || !crate::valid_digest(&manifest.sealed_key_sha256)
        || !crate::valid_digest(&manifest.seed_commitment_sha256)
    {
        return Err("manifest contains an invalid digest".to_owned());
    }
    if manifest.generation_id != packet.generation_id {
        return Err("manifest generation ID does not match the public packet".to_owned());
    }
    if sealed_key.generation_id != packet.generation_id || run.generation_id != packet.generation_id
    {
        return Err("packet, sealed key and detector run generation IDs must match".to_owned());
    }
    if sealed_key.seed_commitment_sha256 != manifest.seed_commitment_sha256 {
        return Err("sealed key seed commitment does not match the manifest".to_owned());
    }
    if run.schema != DETECTOR_RUN_SCHEMA {
        return Err(format!("detector run schema must be {DETECTOR_RUN_SCHEMA}"));
    }
    let expected_public_packet_sha256 = packet_digest(packet)?;
    if run.public_packet_sha256 != expected_public_packet_sha256
        || manifest.public_packet_sha256 != expected_public_packet_sha256
    {
        return Err("detector run or manifest does not bind the exact public packet".to_owned());
    }
    if sealed_key.public_packet_sha256 != run.public_packet_sha256 {
        return Err("sealed key does not bind the detector run public packet".to_owned());
    }
    let sealed_key_bytes = serde_json::to_vec_pretty(sealed_key)
        .map_err(|error| format!("cannot serialize sealed key: {error}"))?;
    let expected_sealed_key_sha256 = hash_domain(b"sealed-key.v1", &sealed_key_bytes);
    if manifest.sealed_key_sha256 != expected_sealed_key_sha256 {
        return Err("sealed key bytes do not match the manifest commitment".to_owned());
    }
    if run.detector_id.trim().is_empty()
        || run.detector_version.trim().is_empty()
        || run.threshold_permille > 1_000
    {
        return Err("detector run has invalid identity or threshold".to_owned());
    }
    match run.kind {
        DetectorKindV1::TokenNgramJaccard if !(2..=8).contains(&run.token_ngram_size) => {
            return Err("detector run has unsupported token_ngram_size".to_owned());
        }
        DetectorKindV1::ExactTextEquality | DetectorKindV1::NormalizedTextEquality
            if run.token_ngram_size != 0 =>
        {
            return Err("non-ngram detector run must have token_ngram_size=0".to_owned());
        }
        _ => {}
    }
    if !crate::valid_digest(&sealed_key.seed_commitment_sha256)
        || !crate::valid_digest(&sealed_key.public_packet_sha256)
    {
        return Err("sealed key has an invalid artifact or seed commitment".to_owned());
    }
    if sealed_key.pair_truth.len() != packet.pairs.len() {
        return Err("sealed-key truth count does not match public packet pair count".to_owned());
    }

    let mut truth_by_id = BTreeMap::new();
    for truth in &sealed_key.pair_truth {
        if !crate::valid_digest(&truth.pair_id) {
            return Err("sealed-key row contains an invalid pair ID".to_owned());
        }
        if truth.target_mechanism != truth.target_tier.target_mechanism() {
            return Err(format!(
                "sealed-key target mechanism does not match tier for pair {}",
                truth.pair_id
            ));
        }
        let expected_label = if truth
            .planted_mechanisms_present
            .contains(&truth.target_mechanism)
        {
            crate::TargetControlLabelV1::PositiveForTargetTier
        } else {
            crate::TargetControlLabelV1::NegativeForTargetTier
        };
        if truth.target_control_label != expected_label {
            return Err(format!(
                "sealed-key target control label is inconsistent for pair {}",
                truth.pair_id
            ));
        }
        if !crate::valid_digest(&truth.left_source_lineage_sha256)
            || !crate::valid_digest(&truth.right_source_lineage_sha256)
            || !crate::valid_digest(&truth.left_recipe_family_sha256)
            || !crate::valid_digest(&truth.right_recipe_family_sha256)
        {
            return Err(format!(
                "sealed-key lineage digest is invalid for pair {}",
                truth.pair_id
            ));
        }
        let mechanism_has_same_source = truth.planted_mechanisms_present.iter().any(|mechanism| {
            matches!(
                mechanism,
                ExposureMechanismV1::ExactInstance
                    | ExposureMechanismV1::DerivativeInstance
                    | ExposureMechanismV1::SemanticEquivalentInstance
            )
        });
        let source_lineage_is_shared =
            truth.left_source_lineage_sha256 == truth.right_source_lineage_sha256;
        if mechanism_has_same_source != source_lineage_is_shared {
            return Err(format!(
                "sealed-key source lineage contradicts planted mechanisms for pair {}",
                truth.pair_id
            ));
        }
        let mechanism_has_shared_recipe = truth
            .planted_mechanisms_present
            .contains(&ExposureMechanismV1::SharedTaskRecipe);
        let recipe_lineage_is_shared =
            truth.left_recipe_family_sha256 == truth.right_recipe_family_sha256;
        if mechanism_has_shared_recipe != recipe_lineage_is_shared {
            return Err(format!(
                "sealed-key recipe lineage contradicts planted mechanisms for pair {}",
                truth.pair_id
            ));
        }
        let mut sorted_mechanisms = truth.planted_mechanisms_present.clone();
        sorted_mechanisms.sort();
        sorted_mechanisms.dedup();
        if sorted_mechanisms != truth.planted_mechanisms_present {
            return Err(format!(
                "sealed-key mechanisms are duplicated or not canonically ordered for pair {}",
                truth.pair_id
            ));
        }
        if truth_by_id.insert(truth.pair_id.as_str(), truth).is_some() {
            return Err(format!("sealed key repeats pair ID {}", truth.pair_id));
        }
    }

    let pair_ids: BTreeSet<&str> = packet
        .pairs
        .iter()
        .map(|pair| pair.pair_id.as_str())
        .collect();
    let mut scores_by_id = BTreeMap::<&str, u16>::new();
    for score in &run.scores {
        if !pair_ids.contains(score.pair_id.as_str()) {
            return Err(format!(
                "detector run includes unknown pair ID {}",
                score.pair_id
            ));
        }
        if score.risk_permille > 1_000 {
            return Err(format!(
                "risk score for pair {} exceeds 1000",
                score.pair_id
            ));
        }
        if scores_by_id
            .insert(score.pair_id.as_str(), score.risk_permille)
            .is_some()
        {
            return Err(format!("detector run repeats pair ID {}", score.pair_id));
        }
    }
    if scores_by_id.len() != pair_ids.len() {
        return Err(format!(
            "detector run must score every pair exactly once; expected {}, observed {}",
            pair_ids.len(),
            scores_by_id.len()
        ));
    }

    let mut metrics_by_tier = Vec::with_capacity(TierV1::ALL.len());
    for tier in TierV1::ALL {
        let mut positive_support = 0_u64;
        let mut negative_support = 0_u64;
        let mut true_positive = 0_u64;
        let mut false_positive = 0_u64;
        let mut false_negative = 0_u64;
        let mut true_negative = 0_u64;

        for pair in &packet.pairs {
            let truth = truth_by_id
                .get(pair.pair_id.as_str())
                .ok_or_else(|| format!("sealed key is missing pair {}", pair.pair_id))?;
            if truth.target_tier != tier {
                continue;
            }
            let actual_positive =
                truth.target_control_label == crate::TargetControlLabelV1::PositiveForTargetTier;
            let predicted_positive = *scores_by_id
                .get(pair.pair_id.as_str())
                .ok_or_else(|| format!("detector run is missing pair {}", pair.pair_id))?
                >= run.threshold_permille;

            if actual_positive {
                positive_support = positive_support
                    .checked_add(1)
                    .ok_or_else(|| "positive support counter overflow".to_owned())?;
                if predicted_positive {
                    true_positive = true_positive
                        .checked_add(1)
                        .ok_or_else(|| "true-positive counter overflow".to_owned())?;
                } else {
                    false_negative = false_negative
                        .checked_add(1)
                        .ok_or_else(|| "false-negative counter overflow".to_owned())?;
                }
            } else {
                negative_support = negative_support
                    .checked_add(1)
                    .ok_or_else(|| "negative support counter overflow".to_owned())?;
                if predicted_positive {
                    false_positive = false_positive
                        .checked_add(1)
                        .ok_or_else(|| "false-positive counter overflow".to_owned())?;
                } else {
                    true_negative = true_negative
                        .checked_add(1)
                        .ok_or_else(|| "true-negative counter overflow".to_owned())?;
                }
            }
        }

        let total = positive_support
            .checked_add(negative_support)
            .ok_or_else(|| "tier support counter overflow".to_owned())?;
        let correct = true_positive
            .checked_add(true_negative)
            .ok_or_else(|| "correct prediction counter overflow".to_owned())?;

        let precision_denominator = true_positive
            .checked_add(false_positive)
            .ok_or_else(|| "precision denominator overflow".to_owned())?;
        let fpr_denominator = false_positive
            .checked_add(true_negative)
            .ok_or_else(|| "FPR denominator overflow".to_owned())?;
        let fdr_denominator = true_positive
            .checked_add(false_positive)
            .ok_or_else(|| "FDR denominator overflow".to_owned())?;

        let precision = ratio_permille(true_positive, precision_denominator);
        let recall = ratio_permille(true_positive, positive_support);
        let specificity = ratio_permille(true_negative, negative_support);

        metrics_by_tier.push(TierConfusionMetricsV1 {
            tier,
            positive_support,
            negative_support,
            true_positive,
            false_positive,
            false_negative,
            true_negative,
            precision_permille: precision,
            recall_permille: recall,
            specificity_permille: specificity,
            false_positive_rate_permille: ratio_permille(false_positive, fpr_denominator),
            false_discovery_rate_permille: ratio_permille(false_positive, fdr_denominator),
            accuracy_permille: ratio_permille(correct, total),
            balanced_accuracy_permille: balanced_accuracy_permille(recall, specificity),
        });
    }

    let manifest_bytes = serde_json::to_vec_pretty(manifest)
        .map_err(|error| format!("cannot serialize challenge manifest: {error}"))?;
    Ok(DetectorEvaluationReportV1 {
        schema: DETECTOR_REPORT_SCHEMA.to_owned(),
        generation_id: packet.generation_id.clone(),
        public_packet_sha256: run.public_packet_sha256.clone(),
        sealed_key_sha256: expected_sealed_key_sha256,
        manifest_sha256: hash_domain(b"manifest.v1", &manifest_bytes),
        detector_id: run.detector_id.clone(),
        detector_version: run.detector_version.clone(),
        detector_kind: run.kind,
        threshold_permille: run.threshold_permille,
        pair_count: u64::try_from(packet.pairs.len()).map_err(|_| "pair count overflow")?,
        metrics_by_tier,
        score_semantics: "risk_like_similarity_score_not_a_calibrated_probability".to_owned(),
        uncertainty_status: "descriptive_planted_control_counts_only_cluster_aware_inference_not_included".to_owned(),
        claim_boundary: "synthetic_fixture_detector_performance_only_not_real_world_training_contamination_detection_or_absence".to_owned(),
    })
}

fn packet_digest(packet: &PublicChallengePacketV1) -> Result<String, String> {
    if packet.schema != PUBLIC_PACKET_SCHEMA {
        return Err(format!(
            "public packet schema must be {PUBLIC_PACKET_SCHEMA}"
        ));
    }
    let bytes = serde_json::to_vec_pretty(packet)
        .map_err(|error| format!("cannot serialize public packet: {error}"))?;
    Ok(hash_domain(b"public-packet.v1", &bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExposureMechanismV1, GeneratorConfigV1, generate_challenge_bundle};
    use serde_json::json;

    fn fixture() -> (
        PublicChallengePacketV1,
        SealedChallengeKeyV1,
        ChallengeManifestV1,
    ) {
        let bundle = generate_challenge_bundle(
            &GeneratorConfigV1 {
                generation_id: "detector-fixture".to_owned(),
                pairs_per_tier: 4,
            },
            &[0x41; 32],
        )
        .unwrap();
        let packet: PublicChallengePacketV1 =
            serde_json::from_slice(&bundle.public_packet_bytes).unwrap();
        let key: SealedChallengeKeyV1 = serde_json::from_slice(&bundle.sealed_key_bytes).unwrap();
        let manifest: ChallengeManifestV1 = serde_json::from_slice(&bundle.manifest_bytes).unwrap();
        (packet, key, manifest)
    }

    fn spec(kind: DetectorKindV1, threshold: u16, ngram: u8) -> DetectorSpecV1 {
        DetectorSpecV1 {
            schema: DETECTOR_SPEC_SCHEMA.to_owned(),
            detector_id: "fixture-detector".to_owned(),
            detector_version: "1".to_owned(),
            kind,
            threshold_permille: threshold,
            token_ngram_size: ngram,
        }
    }

    #[test]
    fn exact_text_baseline_separates_exact_positive_and_negative_controls() {
        let (packet, key, manifest) = fixture();
        let detector_run =
            score_public_packet(&packet, &spec(DetectorKindV1::ExactTextEquality, 1_000, 0))
                .unwrap();
        let report = evaluate_detector_run(&packet, &key, &manifest, &detector_run).unwrap();
        let t1 = report
            .metrics_by_tier
            .iter()
            .find(|metric| metric.tier == TierV1::T1Exact)
            .unwrap();
        assert_eq!(t1.positive_support, 4);
        assert_eq!(t1.negative_support, 4);
        assert_eq!(t1.true_positive, 4);
        assert_eq!(t1.false_negative, 0);
        assert_eq!(t1.false_positive, 0);
        assert_eq!(t1.true_negative, 4);
    }

    #[test]
    fn normalized_text_baseline_separates_syntactic_positive_controls() {
        let (packet, key, manifest) = fixture();
        let detector_run = score_public_packet(
            &packet,
            &spec(DetectorKindV1::NormalizedTextEquality, 1_000, 0),
        )
        .unwrap();
        let report = evaluate_detector_run(&packet, &key, &manifest, &detector_run).unwrap();
        let t2 = report
            .metrics_by_tier
            .iter()
            .find(|metric| metric.tier == TierV1::T2Syntactic)
            .unwrap();
        assert_eq!(t2.true_positive, 4);
        assert_eq!(t2.false_negative, 0);
        assert_eq!(t2.false_positive, 0);
        assert_eq!(t2.true_negative, 4);
    }

    #[test]
    fn ngram_similarity_is_deterministic_and_bounded() {
        let (packet, _, _) = fixture();
        let detector_spec = spec(DetectorKindV1::TokenNgramJaccard, 500, 3);
        let first = score_public_packet(&packet, &detector_spec).unwrap();
        let second = score_public_packet(&packet, &detector_spec).unwrap();
        assert_eq!(
            serde_json::to_vec(&first).unwrap(),
            serde_json::to_vec(&second).unwrap()
        );
        assert!(
            first
                .scores
                .iter()
                .all(|score| score.risk_permille <= 1_000)
        );
    }

    #[test]
    fn detector_run_must_cover_each_public_pair_exactly_once() {
        let (packet, key, manifest) = fixture();
        let mut run =
            score_public_packet(&packet, &spec(DetectorKindV1::ExactTextEquality, 500, 0)).unwrap();
        run.scores.pop();
        assert!(
            evaluate_detector_run(&packet, &key, &manifest, &run)
                .unwrap_err()
                .contains("score every pair exactly once")
        );
    }

    #[test]
    fn duplicate_or_unknown_score_ids_fail_closed() {
        let (packet, key, manifest) = fixture();
        let mut run =
            score_public_packet(&packet, &spec(DetectorKindV1::ExactTextEquality, 500, 0)).unwrap();
        run.scores[1].pair_id = run.scores[0].pair_id.clone();
        assert!(
            evaluate_detector_run(&packet, &key, &manifest, &run)
                .unwrap_err()
                .contains("repeats pair ID")
        );

        let mut run =
            score_public_packet(&packet, &spec(DetectorKindV1::ExactTextEquality, 500, 0)).unwrap();
        run.scores[0].pair_id = "not-a-pair".to_owned();
        assert!(
            evaluate_detector_run(&packet, &key, &manifest, &run)
                .unwrap_err()
                .contains("unknown pair ID")
        );
    }

    #[test]
    fn sealed_key_mutation_without_manifest_update_fails_closed() {
        let (packet, mut key, manifest) = fixture();
        let run =
            score_public_packet(&packet, &spec(DetectorKindV1::ExactTextEquality, 500, 0)).unwrap();
        key.pair_truth[0].target_control_label = if key.pair_truth[0].target_control_label
            == crate::TargetControlLabelV1::PositiveForTargetTier
        {
            crate::TargetControlLabelV1::NegativeForTargetTier
        } else {
            crate::TargetControlLabelV1::PositiveForTargetTier
        };
        let error = evaluate_detector_run(&packet, &key, &manifest, &run).unwrap_err();
        assert!(error.contains("sealed key bytes do not match the manifest commitment"));
    }

    #[test]
    fn wrong_sealed_key_or_packet_commitment_fails_closed() {
        let (packet, key, manifest) = fixture();
        let run =
            score_public_packet(&packet, &spec(DetectorKindV1::ExactTextEquality, 500, 0)).unwrap();

        let mut wrong_key = key.clone();
        wrong_key.public_packet_sha256 = "0".repeat(64);
        assert!(
            evaluate_detector_run(&packet, &wrong_key, &manifest, &run)
                .unwrap_err()
                .contains("does not bind")
        );

        let mut wrong_run = run;
        wrong_run.public_packet_sha256 = "1".repeat(64);
        assert!(
            evaluate_detector_run(&packet, &key, &manifest, &wrong_run)
                .unwrap_err()
                .contains("does not bind the exact public packet")
        );
    }

    #[test]
    fn unsupported_ngram_sizes_and_scores_outside_range_are_rejected() {
        let (packet, key, manifest) = fixture();
        assert!(
            score_public_packet(&packet, &spec(DetectorKindV1::TokenNgramJaccard, 500, 1)).is_err()
        );

        let mut run =
            score_public_packet(&packet, &spec(DetectorKindV1::ExactTextEquality, 500, 0)).unwrap();
        run.scores[0].risk_permille = 1_001;
        assert!(
            evaluate_detector_run(&packet, &key, &manifest, &run)
                .unwrap_err()
                .contains("exceeds 1000")
        );
    }

    #[test]
    fn empty_class_denominators_are_not_estimable() {
        let (packet, mut key, mut manifest) = fixture();
        for truth in &mut key.pair_truth {
            if truth.target_tier == TierV1::T1Exact {
                truth.target_control_label = crate::TargetControlLabelV1::PositiveForTargetTier;
                truth
                    .planted_mechanisms_present
                    .push(ExposureMechanismV1::ExactInstance);
                truth.planted_mechanisms_present.sort();
                truth.planted_mechanisms_present.dedup();
                truth.right_source_lineage_sha256 = truth.left_source_lineage_sha256.clone();
            }
        }
        let key_bytes = serde_json::to_vec_pretty(&key).unwrap();
        manifest.sealed_key_sha256 = hash_domain(b"sealed-key.v1", &key_bytes);
        // This fixture intentionally violates the generated balanced-control
        // pattern but remains internally consistent for the metrics contract.
        let run =
            score_public_packet(&packet, &spec(DetectorKindV1::ExactTextEquality, 500, 0)).unwrap();
        let report = evaluate_detector_run(&packet, &key, &manifest, &run).unwrap();
        let t1 = report
            .metrics_by_tier
            .iter()
            .find(|metric| metric.tier == TierV1::T1Exact)
            .unwrap();
        assert_eq!(t1.negative_support, 0);
        assert_eq!(t1.specificity_permille, None);
        assert_eq!(t1.false_positive_rate_permille, None);
        assert_eq!(t1.balanced_accuracy_permille, None);
    }

    #[test]
    fn normalized_text_is_deliberately_not_unicode_normalization() {
        let left = normalize_text("MÜNCHEN");
        let right = normalize_text("münchen");
        assert_ne!(left, right);
    }

    #[test]
    fn strict_detector_run_deserialization_rejects_unknown_fields() {
        let (packet, _, _) = fixture();
        let run =
            score_public_packet(&packet, &spec(DetectorKindV1::ExactTextEquality, 500, 0)).unwrap();
        let mut value = serde_json::to_value(run).unwrap();
        value["unexpected"] = json!(true);
        assert!(serde_json::from_value::<DetectorRunV1>(value).is_err());
    }
}
