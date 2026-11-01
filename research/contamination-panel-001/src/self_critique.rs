//! Self-Critique entropy-path invariance score for RL-post-training contamination.
//!
//! This module is a model-agnostic scoring and evidence-binding kernel. A
//! backend adapter supplies per-generated-token entropy values and generation
//! receipts. It does not load a model or generate prompts.
//!
//! Paper score: zero-pad the shorter entropy sequence, calculate cosine
//! similarity, then multiply by min(sequence lengths) / max(sequence lengths).
//! Higher scores indicate greater entropy-path similarity under the declared
//! self-critique probe. The score is a detector feature, not membership proof.

use serde::{Deserialize, Serialize};

pub const SELF_CRITIQUE_REQUEST_SCHEMA: &str =
    "symthaea.self_critique_scoring_request.v1";
pub const SELF_CRITIQUE_REPORT_SCHEMA: &str =
    "symthaea.self_critique_score_report.v1";
pub const SELF_CRITIQUE_PAPER_IMPLEMENTATION_ID: &str =
    "symthaea.self_critique.penalized_entropy_cosine.v1";
pub const MAX_ENTROPY_STEPS: usize = 32_768;
pub const MAX_GENERATED_TOKENS: u64 = 65_536;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EntropySourceV1 {
    /// Released RL-Data-Contamination path: exponentiate the returned vLLM
    /// top-K log-probabilities and sum -p*ln(p) without renormalizing the tail.
    VllmTopKRawPartial,
    /// Exact Shannon entropy from the complete vocabulary softmax/logit vector.
    FullVocabSoftmaxExact,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GreedyDecodeEvidenceV1 {
    pub greedy: bool,
    pub temperature: f64,
    pub max_new_tokens: u64,
    pub generation_config_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SelfCritiqueScoringRequestV1 {
    pub schema: String,
    pub sample_id: String,
    pub model_checkpoint_sha256: String,
    pub tokenizer_sha256: String,
    pub chat_template_sha256: String,
    pub initial_prompt_sha256: String,
    pub critique_instruction_sha256: String,
    pub initial_response_sha256: String,
    pub critique_prompt_sha256: String,
    pub entropy_extractor_sha256: String,
    pub entropy_source: EntropySourceV1,
    /// Required only for the released top-K raw-partial mode.
    pub requested_top_k: Option<u32>,
    pub vocabulary_size: Option<u32>,
    pub initial_decode: GreedyDecodeEvidenceV1,
    pub critique_decode: GreedyDecodeEvidenceV1,
    pub initial_generated_token_count: u64,
    pub critique_generated_token_count: u64,
    /// Values must be finite, non-negative token entropies in nats.
    /// The entropy sequence may be shorter than generated token count if the
    /// backend omitted per-step logprobs; the report exposes that coverage.
    pub initial_entropy_nats: Vec<f64>,
    pub critique_entropy_nats: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SelfCritiqueScoreReportV1 {
    pub schema: String,
    pub implementation_id: String,
    pub sample_id: String,
    pub model_checkpoint_sha256: String,
    pub tokenizer_sha256: String,
    pub chat_template_sha256: String,
    pub initial_prompt_sha256: String,
    pub critique_instruction_sha256: String,
    pub initial_response_sha256: String,
    pub critique_prompt_sha256: String,
    pub entropy_extractor_sha256: String,
    pub request_sha256: String,
    pub entropy_source: EntropySourceV1,
    pub requested_top_k: Option<u32>,
    pub vocabulary_size: Option<u32>,
    pub initial_generation_config_sha256: String,
    pub critique_generation_config_sha256: String,
    pub initial_generated_token_count: u64,
    pub critique_generated_token_count: u64,
    pub initial_entropy_step_count: u64,
    pub critique_entropy_step_count: u64,
    pub initial_entropy_coverage_permille: u16,
    pub critique_entropy_coverage_permille: u16,
    pub initial_missing_entropy_steps: u64,
    pub critique_missing_entropy_steps: u64,
    pub initial_entropy_sequence_sha256: String,
    pub critique_entropy_sequence_sha256: String,
    pub padded_cosine_similarity: f64,
    pub length_penalty: f64,
    /// Higher means more entropy-path similarity under the declared probe.
    pub contamination_score: f64,
    pub paper_faithful_score_formula: bool,
    pub released_topk_entropy_reproduction: bool,
    pub independently_authenticated: bool,
    pub confirmatory_eligible: bool,
    pub claim_boundary: String,
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_digest(value: &str, label: &str) -> Result<(), String> {
    if valid_digest(value) {
        Ok(())
    } else {
        Err(format!("{label} must be a lowercase SHA-256 digest"))
    }
}

fn validate_entropy(values: &[f64], label: &str) -> Result<(), String> {
    if values.is_empty() || values.len() > MAX_ENTROPY_STEPS {
        return Err(format!(
            "{label} entropy step count must be in 1..={MAX_ENTROPY_STEPS}"
        ));
    }
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(format!("{label} entropy values must be finite and non-negative"));
    }
    Ok(())
}

fn sequence_sha256(values: &[f64], label: &str) -> Result<String, String> {
    let bytes = serde_json::to_vec(values)
        .map_err(|error| format!("cannot serialize {label} entropy sequence: {error}"))?;
    Ok(crate::hash_domain(
        b"self-critique-entropy-sequence.v1",
        &bytes,
    ))
}

/// Exact released penalized-cosine formula. Empty sequences are rejected by
/// this API; the official detector maps missing sequences to NaN before scoring.
pub fn penalized_entropy_cosine(
    initial: &[f64],
    critique: &[f64],
) -> Result<(f64, f64, f64), String> {
    validate_entropy(initial, "initial")?;
    validate_entropy(critique, "critique")?;
    let max_len = initial.len().max(critique.len());
    let min_len = initial.len().min(critique.len());
    let mut dot = 0.0;
    let mut norm_initial_sq = 0.0;
    let mut norm_critique_sq = 0.0;

    for index in 0..max_len {
        let left = initial.get(index).copied().unwrap_or(0.0);
        let right = critique.get(index).copied().unwrap_or(0.0);
        dot += left * right;
        norm_initial_sq += left * left;
        norm_critique_sq += right * right;
    }

    let norm_initial = norm_initial_sq.sqrt();
    let norm_critique = norm_critique_sq.sqrt();
    let cosine = if norm_initial == 0.0 || norm_critique == 0.0 {
        0.0
    } else {
        dot / (norm_initial * norm_critique)
    };
    let length_penalty = min_len as f64 / max_len as f64;
    let score = cosine * length_penalty;

    for (label, value) in [
        ("padded cosine", cosine),
        ("length penalty", length_penalty),
        ("penalized score", score),
    ] {
        if !value.is_finite() {
            return Err(format!("Self-Critique {label} is non-finite"));
        }
    }
    if !(-1.0e-12..=1.0 + 1.0e-12).contains(&cosine)
        || !(0.0..=1.0).contains(&length_penalty)
        || !(0.0..=1.0 + 1.0e-12).contains(&score)
    {
        return Err("Self-Critique similarity is outside its expected [0, 1] range".to_owned());
    }
    Ok((cosine.clamp(0.0, 1.0), length_penalty, score.clamp(0.0, 1.0)))
}

/// Validate a per-item receipt and compute its released paper score.
/// Both generations must use greedy decoding (temperature exactly zero).
pub fn score_self_critique_request(
    request: &SelfCritiqueScoringRequestV1,
) -> Result<SelfCritiqueScoreReportV1, String> {
    if request.schema != SELF_CRITIQUE_REQUEST_SCHEMA {
        return Err(format!("request schema must be {SELF_CRITIQUE_REQUEST_SCHEMA}"));
    }
    if request.sample_id.trim().is_empty() || request.sample_id.len() > 256 {
        return Err("sample_id must be nonempty and at most 256 bytes".to_owned());
    }

    for (digest, label) in [
        (&request.model_checkpoint_sha256, "model_checkpoint_sha256"),
        (&request.tokenizer_sha256, "tokenizer_sha256"),
        (&request.chat_template_sha256, "chat_template_sha256"),
        (&request.initial_prompt_sha256, "initial_prompt_sha256"),
        (&request.critique_instruction_sha256, "critique_instruction_sha256"),
        (&request.initial_response_sha256, "initial_response_sha256"),
        (&request.critique_prompt_sha256, "critique_prompt_sha256"),
        (&request.entropy_extractor_sha256, "entropy_extractor_sha256"),
        (
            &request.initial_decode.generation_config_sha256,
            "initial generation_config_sha256",
        ),
        (
            &request.critique_decode.generation_config_sha256,
            "critique generation_config_sha256",
        ),
    ] {
        require_digest(digest, label)?;
    }

    for (decode, label) in [
        (&request.initial_decode, "initial"),
        (&request.critique_decode, "critique"),
    ] {
        if !decode.greedy || decode.temperature != 0.0 {
            return Err(format!("{label} generation must be greedy with temperature=0"));
        }
        if decode.max_new_tokens == 0 || decode.max_new_tokens > MAX_GENERATED_TOKENS {
            return Err(format!("{label} max_new_tokens is outside the supported range"));
        }
    }

    if request.initial_generated_token_count == 0
        || request.critique_generated_token_count == 0
        || request.initial_generated_token_count > request.initial_decode.max_new_tokens
        || request.critique_generated_token_count > request.critique_decode.max_new_tokens
    {
        return Err("generated token counts must be positive and cannot exceed configured limits".to_owned());
    }
    validate_entropy(&request.initial_entropy_nats, "initial")?;
    validate_entropy(&request.critique_entropy_nats, "critique")?;
    if request.initial_entropy_nats.len() as u64 > request.initial_generated_token_count
        || request.critique_entropy_nats.len() as u64 > request.critique_generated_token_count
    {
        return Err("observed entropy steps cannot exceed generated token counts".to_owned());
    }

    match request.entropy_source {
        EntropySourceV1::VllmTopKRawPartial => {
            let top_k = request
                .requested_top_k
                .ok_or_else(|| "vllm_top_k_raw_partial requires requested_top_k".to_owned())?;
            let vocabulary = request
                .vocabulary_size
                .ok_or_else(|| "vllm_top_k_raw_partial requires vocabulary_size".to_owned())?;
            if top_k == 0 || vocabulary <= top_k {
                return Err("top_k must be positive and vocabulary_size must exceed top_k".to_owned());
            }
        }
        EntropySourceV1::FullVocabSoftmaxExact => {
            if request.requested_top_k.is_some() {
                return Err("full_vocab_softmax_exact must not declare requested_top_k".to_owned());
            }
            if request
                .vocabulary_size
                .is_some_and(|vocabulary| vocabulary < 2)
            {
                return Err("full-vocabulary size, when supplied, must be at least 2".to_owned());
            }
        }
    }

    let (padded_cosine_similarity, length_penalty, contamination_score) =
        penalized_entropy_cosine(
            &request.initial_entropy_nats,
            &request.critique_entropy_nats,
        )?;

    let initial_missing =
        request.initial_generated_token_count - request.initial_entropy_nats.len() as u64;
    let critique_missing =
        request.critique_generated_token_count - request.critique_entropy_nats.len() as u64;
    let initial_coverage = (request.initial_entropy_nats.len() as f64 * 1_000.0
        / request.initial_generated_token_count as f64)
        .floor() as u16;
    let critique_coverage = (request.critique_entropy_nats.len() as f64 * 1_000.0
        / request.critique_generated_token_count as f64)
        .floor() as u16;

    let request_bytes = serde_json::to_vec(request)
        .map_err(|error| format!("cannot serialize Self-Critique request: {error}"))?;
    let initial_entropy_sequence_sha256 =
        sequence_sha256(&request.initial_entropy_nats, "initial")?;
    let critique_entropy_sequence_sha256 =
        sequence_sha256(&request.critique_entropy_nats, "critique")?;

    Ok(SelfCritiqueScoreReportV1 {
        schema: SELF_CRITIQUE_REPORT_SCHEMA.to_owned(),
        implementation_id: SELF_CRITIQUE_PAPER_IMPLEMENTATION_ID.to_owned(),
        sample_id: request.sample_id.clone(),
        model_checkpoint_sha256: request.model_checkpoint_sha256.clone(),
        tokenizer_sha256: request.tokenizer_sha256.clone(),
        chat_template_sha256: request.chat_template_sha256.clone(),
        initial_prompt_sha256: request.initial_prompt_sha256.clone(),
        critique_instruction_sha256: request.critique_instruction_sha256.clone(),
        initial_response_sha256: request.initial_response_sha256.clone(),
        critique_prompt_sha256: request.critique_prompt_sha256.clone(),
        entropy_extractor_sha256: request.entropy_extractor_sha256.clone(),
        request_sha256: crate::hash_domain(b"self-critique-request.v1", &request_bytes),
        entropy_source: request.entropy_source,
        requested_top_k: request.requested_top_k,
        vocabulary_size: request.vocabulary_size,
        initial_generation_config_sha256: request.initial_decode.generation_config_sha256.clone(),
        critique_generation_config_sha256: request.critique_decode.generation_config_sha256.clone(),
        initial_generated_token_count: request.initial_generated_token_count,
        critique_generated_token_count: request.critique_generated_token_count,
        initial_entropy_step_count: request.initial_entropy_nats.len() as u64,
        critique_entropy_step_count: request.critique_entropy_nats.len() as u64,
        initial_entropy_coverage_permille: initial_coverage,
        critique_entropy_coverage_permille: critique_coverage,
        initial_missing_entropy_steps: initial_missing,
        critique_missing_entropy_steps: critique_missing,
        initial_entropy_sequence_sha256,
        critique_entropy_sequence_sha256,
        padded_cosine_similarity,
        length_penalty,
        contamination_score,
        paper_faithful_score_formula: true,
        released_topk_entropy_reproduction: matches!(
            request.entropy_source,
            EntropySourceV1::VllmTopKRawPartial
        ),
        independently_authenticated: false,
        confirmatory_eligible: false,
        claim_boundary: "This score measures length-penalized cosine similarity between caller-supplied initial and self-critique token-entropy sequences. The vllm_top_k_raw_partial mode reproduces the released top-K partial-sum extraction, not full-vocabulary Shannon entropy. The full_vocab_softmax_exact mode is a separate extraction variant. The report binds declared model, tokenizer, prompt, generation and extraction identities but cannot authenticate them, prove training membership, prove that exposure caused an answer, or establish universal detection performance. Missing per-token entropy steps are explicitly reported as coverage loss.".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn request(source: EntropySourceV1) -> SelfCritiqueScoringRequestV1 {
        SelfCritiqueScoringRequestV1 {
            schema: SELF_CRITIQUE_REQUEST_SCHEMA.to_owned(),
            sample_id: "item-001".to_owned(),
            model_checkpoint_sha256: digest('a'),
            tokenizer_sha256: digest('b'),
            chat_template_sha256: digest('c'),
            initial_prompt_sha256: digest('d'),
            critique_instruction_sha256: digest('e'),
            initial_response_sha256: digest('f'),
            critique_prompt_sha256: digest('1'),
            entropy_extractor_sha256: digest('2'),
            entropy_source: source,
            requested_top_k: matches!(source, EntropySourceV1::VllmTopKRawPartial).then_some(50),
            vocabulary_size: Some(151_936),
            initial_decode: GreedyDecodeEvidenceV1 {
                greedy: true,
                temperature: 0.0,
                max_new_tokens: 256,
                generation_config_sha256: digest('3'),
            },
            critique_decode: GreedyDecodeEvidenceV1 {
                greedy: true,
                temperature: 0.0,
                max_new_tokens: 512,
                generation_config_sha256: digest('4'),
            },
            initial_generated_token_count: 3,
            critique_generated_token_count: 3,
            initial_entropy_nats: vec![0.5, 1.0, 0.5],
            critique_entropy_nats: vec![0.5, 1.0, 0.5],
        }
    }

    #[test]
    fn identical_sequences_score_one() {
        let (cosine, penalty, score) =
            penalized_entropy_cosine(&[0.5, 1.0, 0.5], &[0.5, 1.0, 0.5]).unwrap();
        assert!((cosine - 1.0).abs() < 1.0e-12);
        assert_eq!(penalty, 1.0);
        assert!((score - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn zero_padding_and_length_penalty_match_official_formula() {
        let (cosine, penalty, score) =
            penalized_entropy_cosine(&[1.0, 0.0], &[1.0]).unwrap();
        assert_eq!(cosine, 1.0);
        assert_eq!(penalty, 0.5);
        assert_eq!(score, 0.5);
    }

    #[test]
    fn disjoint_entropy_paths_have_zero_similarity() {
        let (_, _, score) =
            penalized_entropy_cosine(&[1.0, 0.0], &[0.0, 1.0]).unwrap();
        assert_eq!(score, 0.0);
    }

    #[test]
    fn zero_norm_matches_official_zero_similarity_fallback() {
        let (_, _, score) = penalized_entropy_cosine(&[0.0, 0.0], &[1.0, 2.0]).unwrap();
        assert_eq!(score, 0.0);
    }

    #[test]
    fn empty_sequences_are_rejected_as_missing_measurements() {
        assert!(penalized_entropy_cosine(&[], &[1.0]).is_err());
        assert!(penalized_entropy_cosine(&[1.0], &[]).is_err());
    }

    #[test]
    fn negative_and_non_finite_entropies_are_rejected() {
        assert!(penalized_entropy_cosine(&[-0.1], &[0.2]).is_err());
        assert!(penalized_entropy_cosine(&[f64::NAN], &[0.2]).is_err());
        assert!(penalized_entropy_cosine(&[0.1], &[f64::INFINITY]).is_err());
    }

    #[test]
    fn topk_mode_requires_declared_truncation_metadata() {
        let mut r = request(EntropySourceV1::VllmTopKRawPartial);
        r.requested_top_k = None;
        assert!(score_self_critique_request(&r).is_err());
    }

    #[test]
    fn full_vocabulary_mode_rejects_topk_metadata() {
        let mut r = request(EntropySourceV1::FullVocabSoftmaxExact);
        r.requested_top_k = Some(50);
        assert!(score_self_critique_request(&r).is_err());
    }

    #[test]
    fn full_vocabulary_mode_scores_and_is_not_marked_released_topk() {
        let r = request(EntropySourceV1::FullVocabSoftmaxExact);
        let report = score_self_critique_request(&r).unwrap();
        assert!(!report.released_topk_entropy_reproduction);
        assert!(!report.confirmatory_eligible);
        assert_eq!(report.initial_entropy_coverage_permille, 1_000);
    }

    #[test]
    fn topk_mode_scores_and_exposes_missing_entropy_steps() {
        let mut r = request(EntropySourceV1::VllmTopKRawPartial);
        r.initial_generated_token_count = 4;
        let report = score_self_critique_request(&r).unwrap();
        assert!(report.released_topk_entropy_reproduction);
        assert_eq!(report.initial_missing_entropy_steps, 1);
        assert_eq!(report.initial_entropy_coverage_permille, 750);
    }

    #[test]
    fn non_greedy_generation_is_rejected_for_paper_mode() {
        let mut r = request(EntropySourceV1::FullVocabSoftmaxExact);
        r.critique_decode.temperature = 0.2;
        assert!(score_self_critique_request(&r).is_err());
    }

    #[test]
    fn per_step_entropy_cannot_outnumber_generated_tokens() {
        let mut r = request(EntropySourceV1::FullVocabSoftmaxExact);
        r.initial_generated_token_count = 2;
        assert!(score_self_critique_request(&r).is_err());
    }
}
