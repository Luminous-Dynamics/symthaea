//! Prefix-only continuation-value features for compute-aware reasoning.
//!
//! Every feature emitted by extract_prefix_features is computed from the
//! current checkpoint and earlier records only. Future correctness labels are
//! exposed through a separate evaluator-only API; they must never be passed
//! into the inference-time feature extractor.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const PPM: u64 = 1_000_000;
const MAX_CHECKPOINTS: usize = 100_000;
const MAX_OBSERVATIONS_PER_CHECKPOINT: usize = 10_000;
const MAX_EVENTS_PER_CHECKPOINT: usize = 100_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinuationError {
    pub field: String,
    pub message: String,
}
impl ContinuationError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { field: field.into(), message: message.into() }
    }
}
impl std::fmt::Display for ContinuationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}
impl std::error::Error for ContinuationError {}
fn fail(field: impl Into<String>, message: impl Into<String>) -> ContinuationError {
    ContinuationError::new(field, message)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerifierVerdict {
    SupportsCurrentAnswer,
    ContradictsCurrentAnswer,
    Abstains,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerifierObservation {
    pub verifier_id: String,
    /// A model family / implementation lineage label. Different IDs in one
    /// family are not counted as independent agreement sources.
    pub verifier_family_id: String,
    pub available_at_token: u64,
    pub verdict: VerifierVerdict,
    pub verifier_receipt_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceEventKind {
    VerifiedSupport,
    VerifiedContradiction,
    UnverifiedObservation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceEvent {
    pub event_id: String,
    pub available_at_token: u64,
    pub kind: EvidenceEventKind,
    pub evidence_receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReasoningCheckpoint {
    pub checkpoint_id: String,
    pub token_count: u64,
    pub confidence_ppm: u32,
    pub answer_digest_sha256: String,
    pub verifier_observations: Vec<VerifierObservation>,
    pub new_evidence_events: Vec<EvidenceEvent>,
    pub unresolved_obligation_count: u32,
    /// Evaluator truth for this checkpoint. Prefix feature extraction never
    /// reads this field; only the evaluator-only target function does.
    pub evaluator_answer_correct: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReasoningTrajectory {
    pub task_id: String,
    pub signal_contract_id: String,
    pub hard_token_budget: u64,
    pub checkpoints: Vec<ReasoningCheckpoint>,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

impl ReasoningTrajectory {
    pub fn validate(&self) -> Result<(), ContinuationError> {
        if self.task_id.trim().is_empty() || self.signal_contract_id.trim().is_empty() {
            return Err(fail("trajectory_identity", "task_id and signal_contract_id must be non-empty"));
        }
        if self.hard_token_budget == 0 {
            return Err(fail("hard_token_budget", "must be positive"));
        }
        if self.checkpoints.len() < 2 || self.checkpoints.len() > MAX_CHECKPOINTS {
            return Err(fail("checkpoints", format!("must contain 2..={MAX_CHECKPOINTS} records")));
        }
        let mut previous_token = 0u64;
        let mut checkpoint_ids = BTreeSet::new();
        let mut event_ids = BTreeSet::new();
        for (i, checkpoint) in self.checkpoints.iter().enumerate() {
            if checkpoint.checkpoint_id.trim().is_empty() || !checkpoint_ids.insert(checkpoint.checkpoint_id.as_str()) {
                return Err(fail(format!("checkpoints[{i}].checkpoint_id"), "must be non-empty and unique"));
            }
            if checkpoint.token_count <= previous_token || checkpoint.token_count > self.hard_token_budget {
                return Err(fail(format!("checkpoints[{i}].token_count"), "must increase strictly and not exceed the hard budget"));
            }
            if checkpoint.confidence_ppm > PPM as u32 {
                return Err(fail(format!("checkpoints[{i}].confidence_ppm"), "must lie in [0, 1,000,000]"));
            }
            if !is_sha256(&checkpoint.answer_digest_sha256) {
                return Err(fail(format!("checkpoints[{i}].answer_digest_sha256"), "must be a SHA-256 digest"));
            }
            if checkpoint.verifier_observations.len() > MAX_OBSERVATIONS_PER_CHECKPOINT {
                return Err(fail(format!("checkpoints[{i}].verifier_observations"), "observation count exceeds configured cap"));
            }
            let mut verifier_ids = BTreeSet::new();
            let mut verifier_families = BTreeSet::new();
            for (j, observation) in checkpoint.verifier_observations.iter().enumerate() {
                if observation.verifier_id.trim().is_empty() || observation.verifier_family_id.trim().is_empty() {
                    return Err(fail(format!("checkpoints[{i}].verifier_observations[{j}]"), "verifier ID and family must be non-empty"));
                }
                if !verifier_ids.insert(observation.verifier_id.as_str()) {
                    return Err(fail(format!("checkpoints[{i}].verifier_observations[{j}].verifier_id"), "duplicate verifier ID at checkpoint"));
                }
                if !verifier_families.insert(observation.verifier_family_id.as_str()) {
                    return Err(fail(format!("checkpoints[{i}].verifier_observations[{j}].verifier_family_id"), "multiple observations from the same family would double-count agreement"));
                }
                if observation.available_at_token > checkpoint.token_count {
                    return Err(fail(format!("checkpoints[{i}].verifier_observations[{j}].available_at_token"), "future verifier result leaks into this prefix"));
                }
                if !is_sha256(&observation.verifier_receipt_sha256) {
                    return Err(fail(format!("checkpoints[{i}].verifier_observations[{j}].verifier_receipt_sha256"), "must be a SHA-256 digest"));
                }
            }
            if checkpoint.new_evidence_events.len() > MAX_EVENTS_PER_CHECKPOINT {
                return Err(fail(format!("checkpoints[{i}].new_evidence_events"), "event count exceeds configured cap"));
            }
            for (j, event) in checkpoint.new_evidence_events.iter().enumerate() {
                if event.event_id.trim().is_empty() || !event_ids.insert(event.event_id.as_str()) {
                    return Err(fail(format!("checkpoints[{i}].new_evidence_events[{j}].event_id"), "event ID must be non-empty and unique across the trajectory"));
                }
                if event.available_at_token > checkpoint.token_count {
                    return Err(fail(format!("checkpoints[{i}].new_evidence_events[{j}].available_at_token"), "future evidence leaks into this prefix"));
                }
                if !is_sha256(&event.evidence_receipt_sha256) {
                    return Err(fail(format!("checkpoints[{i}].new_evidence_events[{j}].evidence_receipt_sha256"), "must be a SHA-256 digest"));
                }
            }
            previous_token = checkpoint.token_count;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefixFeatures {
    pub task_id: String,
    pub checkpoint_id: String,
    pub checkpoint_index: usize,
    pub token_count: u64,
    pub remaining_token_budget: u64,
    pub confidence_ppm: u32,
    pub confidence_delta_ppm: Option<i64>,
    pub confidence_slope_ppm_per_1k_tokens: Option<i64>,
    /// Fraction of adjacent answer-digest pairs that remain unchanged over
    /// the trailing window. Descriptive stability, not correctness.
    pub answer_stability_ppm: Option<u32>,
    pub verifier_family_count: u32,
    pub verifier_support_count: u32,
    pub verifier_contradiction_count: u32,
    pub verifier_abstention_count: u32,
    /// Support / (support + contradiction), excluding abstentions. Agreement
    /// is not a probability of truth or an independence certificate.
    pub verifier_support_fraction_ppm: Option<u32>,
    pub new_verified_support_events: u32,
    pub new_verified_contradiction_events: u32,
    pub new_unverified_observations: u32,
    pub unresolved_obligation_count: u32,
    pub obligations_resolved_since_previous: u32,
    pub obligations_opened_since_previous: u32,
}

fn signed_round_div(numerator: i128, denominator: i128) -> Result<i64, ContinuationError> {
    if denominator <= 0 {
        return Err(fail("slope", "denominator must be positive"));
    }
    let sign = if numerator < 0 { -1i128 } else { 1i128 };
    let abs = numerator.checked_abs().ok_or_else(|| fail("slope", "absolute value overflow"))?;
    let rounded = abs.checked_add(denominator / 2)
        .ok_or_else(|| fail("slope", "rounding overflow"))? / denominator;
    let signed = rounded.checked_mul(sign).ok_or_else(|| fail("slope", "signed rounding overflow"))?;
    i64::try_from(signed).map_err(|_| fail("slope", "result does not fit i64"))
}

fn answer_stability_ppm(checkpoints: &[ReasoningCheckpoint], window: usize) -> Option<u32> {
    if checkpoints.len() < 2 || window == 0 {
        return None;
    }
    let start = checkpoints.len().saturating_sub(window + 1);
    let recent = &checkpoints[start..];
    if recent.len() < 2 {
        return None;
    }
    let equal = recent.windows(2).filter(|pair| pair[0].answer_digest_sha256 == pair[1].answer_digest_sha256).count() as u128;
    let transitions = (recent.len() - 1) as u128;
    Some(((equal * u128::from(PPM) + transitions / 2) / transitions) as u32)
}

/// Extract an inference-time feature row from the prefix ending at each
/// checkpoint. It never reads evaluator_answer_correct or a future checkpoint.
pub fn extract_prefix_features(
    trajectory: &ReasoningTrajectory,
    answer_stability_window: usize,
) -> Result<Vec<PrefixFeatures>, ContinuationError> {
    trajectory.validate()?;
    if answer_stability_window == 0 {
        return Err(fail("answer_stability_window", "must be positive"));
    }
    let mut output = Vec::with_capacity(trajectory.checkpoints.len());
    for i in 0..trajectory.checkpoints.len() {
        let checkpoint = &trajectory.checkpoints[i];
        let prefix = &trajectory.checkpoints[..=i];
        let previous = i.checked_sub(1).map(|index| &trajectory.checkpoints[index]);
        let (confidence_delta, confidence_slope, resolved, opened) = if let Some(previous) = previous {
            let delta = i64::from(checkpoint.confidence_ppm) - i64::from(previous.confidence_ppm);
            let token_delta = checkpoint.token_count - previous.token_count;
            let slope = signed_round_div(i128::from(delta) * 1_000, i128::from(token_delta))?;
            let resolved = previous.unresolved_obligation_count.saturating_sub(checkpoint.unresolved_obligation_count);
            let opened = checkpoint.unresolved_obligation_count.saturating_sub(previous.unresolved_obligation_count);
            (Some(delta), Some(slope), resolved, opened)
        } else {
            (None, None, 0, 0)
        };

        let mut support = 0u32;
        let mut contradiction = 0u32;
        let mut abstention = 0u32;
        for verifier in &checkpoint.verifier_observations {
            match verifier.verdict {
                VerifierVerdict::SupportsCurrentAnswer => support += 1,
                VerifierVerdict::ContradictsCurrentAnswer => contradiction += 1,
                VerifierVerdict::Abstains => abstention += 1,
            }
        }
        let non_abstaining = support + contradiction;
        let support_fraction = if non_abstaining == 0 {
            None
        } else {
            Some(((u64::from(support) * PPM + u64::from(non_abstaining) / 2) / u64::from(non_abstaining)) as u32)
        };

        let mut new_support = 0u32;
        let mut new_contradiction = 0u32;
        let mut new_unverified = 0u32;
        for event in &checkpoint.new_evidence_events {
            match event.kind {
                EvidenceEventKind::VerifiedSupport => new_support += 1,
                EvidenceEventKind::VerifiedContradiction => new_contradiction += 1,
                EvidenceEventKind::UnverifiedObservation => new_unverified += 1,
            }
        }
        output.push(PrefixFeatures {
            task_id: trajectory.task_id.clone(),
            checkpoint_id: checkpoint.checkpoint_id.clone(),
            checkpoint_index: i,
            token_count: checkpoint.token_count,
            remaining_token_budget: trajectory.hard_token_budget - checkpoint.token_count,
            confidence_ppm: checkpoint.confidence_ppm,
            confidence_delta_ppm: confidence_delta,
            confidence_slope_ppm_per_1k_tokens: confidence_slope,
            answer_stability_ppm: answer_stability_ppm(prefix, answer_stability_window),
            verifier_family_count: checkpoint.verifier_observations.len() as u32,
            verifier_support_count: support,
            verifier_contradiction_count: contradiction,
            verifier_abstention_count: abstention,
            verifier_support_fraction_ppm: support_fraction,
            new_verified_support_events: new_support,
            new_verified_contradiction_events: new_contradiction,
            new_unverified_observations: new_unverified,
            unresolved_obligation_count: checkpoint.unresolved_obligation_count,
            obligations_resolved_since_previous: resolved,
            obligations_opened_since_previous: opened,
        });
    }
    Ok(output)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContinuationTarget {
    pub task_id: String,
    pub origin_checkpoint_id: String,
    pub horizon_checkpoints: usize,
    pub current_answer_correct: bool,
    pub future_correct_within_horizon: bool,
    pub first_future_correct_checkpoint_offset: Option<usize>,
    pub future_correct_fraction_ppm: u32,
    pub additional_tokens_until_first_correct: Option<u64>,
}

/// Evaluator-only target. It reads future correctness labels and MUST NOT be
/// passed to extract_prefix_features or any deployed stopping decision.
pub fn label_future_continuation_value(
    trajectory: &ReasoningTrajectory,
    origin_index: usize,
    horizon_checkpoints: usize,
) -> Result<ContinuationTarget, ContinuationError> {
    trajectory.validate()?;
    if origin_index >= trajectory.checkpoints.len() {
        return Err(fail("origin_index", "outside trajectory"));
    }
    if horizon_checkpoints == 0 {
        return Err(fail("horizon_checkpoints", "must be positive"));
    }
    let current = &trajectory.checkpoints[origin_index];
    let end = (origin_index + horizon_checkpoints + 1).min(trajectory.checkpoints.len());
    let future = &trajectory.checkpoints[origin_index + 1..end];
    let correct_count = future.iter().filter(|x| x.evaluator_answer_correct).count() as u128;
    let fraction = if future.is_empty() { 0 } else {
        ((correct_count * u128::from(PPM) + future.len() as u128 / 2) / future.len() as u128) as u32
    };
    let first = future.iter().position(|x| x.evaluator_answer_correct).map(|x| x + 1);
    let additional_tokens = first.map(|offset| {
        trajectory.checkpoints[origin_index + offset].token_count.saturating_sub(current.token_count)
    });
    Ok(ContinuationTarget {
        task_id: trajectory.task_id.clone(),
        origin_checkpoint_id: current.checkpoint_id.clone(),
        horizon_checkpoints,
        current_answer_correct: current.evaluator_answer_correct,
        future_correct_within_horizon: first.is_some(),
        first_future_correct_checkpoint_offset: first,
        future_correct_fraction_ppm: fraction,
        additional_tokens_until_first_correct: additional_tokens,
    })
}

pub fn prefix_feature_digest(features: &[PrefixFeatures]) -> Result<String, ContinuationError> {
    let bytes = serde_json::to_vec(features).map_err(|e| fail("feature_serialization", e.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"SYMTHAEA-CONTINUATION-PREFIX-FEATURES-V0.1");
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(out, "{byte:02x}");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(c: char) -> String { std::iter::repeat(c).take(64).collect() }
    fn obs(id:&str,family:&str,token:u64,verdict:VerifierVerdict)->VerifierObservation{
        VerifierObservation{verifier_id:id.into(),verifier_family_id:family.into(),available_at_token:token,verdict,verifier_receipt_sha256:h('a')}
    }
    fn evidence(id:&str,token:u64,kind:EvidenceEventKind)->EvidenceEvent{
        EvidenceEvent{event_id:id.into(),available_at_token:token,kind,evidence_receipt_sha256:h('b')}
    }
    fn trajectory()->ReasoningTrajectory{
        ReasoningTrajectory{
            task_id:"t1".into(),signal_contract_id:"confidence-stability-verifier-v1".into(),hard_token_budget:8000,
            checkpoints:vec![
                ReasoningCheckpoint{checkpoint_id:"c0".into(),token_count:1000,confidence_ppm:200000,answer_digest_sha256:h('1'),verifier_observations:vec![obs("v1","family-a",1000,VerifierVerdict::SupportsCurrentAnswer)],new_evidence_events:vec![evidence("e1",1000,EvidenceEventKind::VerifiedSupport)],unresolved_obligation_count:4,evaluator_answer_correct:false},
                ReasoningCheckpoint{checkpoint_id:"c1".into(),token_count:2000,confidence_ppm:500000,answer_digest_sha256:h('1'),verifier_observations:vec![obs("v1","family-a",2000,VerifierVerdict::SupportsCurrentAnswer),obs("v2","family-b",2000,VerifierVerdict::ContradictsCurrentAnswer)],new_evidence_events:vec![evidence("e2",2000,EvidenceEventKind::VerifiedContradiction)],unresolved_obligation_count:3,evaluator_answer_correct:false},
                ReasoningCheckpoint{checkpoint_id:"c2".into(),token_count:4000,confidence_ppm:850000,answer_digest_sha256:h('2'),verifier_observations:vec![obs("v1","family-a",4000,VerifierVerdict::SupportsCurrentAnswer),obs("v2","family-b",4000,VerifierVerdict::SupportsCurrentAnswer)],new_evidence_events:vec![evidence("e3",4000,EvidenceEventKind::VerifiedSupport)],unresolved_obligation_count:1,evaluator_answer_correct:true},
                ReasoningCheckpoint{checkpoint_id:"c3".into(),token_count:8000,confidence_ppm:920000,answer_digest_sha256:h('2'),verifier_observations:vec![],new_evidence_events:vec![evidence("e4",8000,EvidenceEventKind::UnverifiedObservation)],unresolved_obligation_count:0,evaluator_answer_correct:true},
            ]
        }
    }

    #[test]
    fn prefix_features_are_computed_per_checkpoint() {
        let t=trajectory();
        let f=extract_prefix_features(&t,3).unwrap();
        assert_eq!(f.len(),4);
        assert_eq!(f[0].confidence_delta_ppm,None);
        assert_eq!(f[1].confidence_delta_ppm,Some(300000));
        assert_eq!(f[1].confidence_slope_ppm_per_1k_tokens,Some(300000));
        assert_eq!(f[2].answer_stability_ppm,Some(500000));
        assert_eq!(f[1].obligations_resolved_since_previous,1);
    }

    #[test]
    fn verifier_agreement_is_a_count_not_a_probability_of_truth() {
        let t=trajectory();
        let f=extract_prefix_features(&t,3).unwrap();
        assert_eq!(f[1].verifier_family_count,2);
        assert_eq!(f[1].verifier_support_count,1);
        assert_eq!(f[1].verifier_contradiction_count,1);
        assert_eq!(f[1].verifier_support_fraction_ppm,Some(500000));
    }

    #[test]
    fn independent_family_count_is_not_inferred_from_duplicate_ids() {
        let mut t=trajectory();
        t.checkpoints[1].verifier_observations[1].verifier_family_id="family-a".into();
        assert!(t.validate().is_err());
    }

    #[test]
    fn future_verifier_and_evidence_observations_fail_prefix_validation() {
        let mut t=trajectory();
        t.checkpoints[1].verifier_observations[0].available_at_token=3000;
        assert!(t.validate().is_err());
        let mut t=trajectory();
        t.checkpoints[1].new_evidence_events[0].available_at_token=3000;
        assert!(t.validate().is_err());
    }

    #[test]
    fn duplicate_evidence_ids_fail_closed() {
        let mut t=trajectory();
        t.checkpoints[2].new_evidence_events[0].event_id="e2".into();
        assert!(t.validate().is_err());
    }

    #[test]
    fn continuation_targets_are_separate_future_evaluator_labels() {
        let t=trajectory();
        let labels=label_future_continuation_value(&t,0,2).unwrap();
        assert!(!labels.current_answer_correct);
        assert!(labels.future_correct_within_horizon);
        assert_eq!(labels.first_future_correct_checkpoint_offset,Some(2));
        assert_eq!(labels.additional_tokens_until_first_correct,Some(3000));
        let features=extract_prefix_features(&t,3).unwrap();
        assert!(!serde_json::to_string(&features).unwrap().contains("future_correct"));
    }

    #[test]
    fn evaluator_labels_do_not_change_prefix_features() {
        let a=trajectory();
        let mut b=trajectory();
        b.checkpoints[2].evaluator_answer_correct=false;
        b.checkpoints[3].evaluator_answer_correct=false;
        assert_eq!(extract_prefix_features(&a,3).unwrap(),extract_prefix_features(&b,3).unwrap());
    }

    #[test]
    fn answer_stability_is_not_misrepresented_as_correctness() {
        let mut t=trajectory();
        t.checkpoints[0].answer_digest_sha256=h('9');
        t.checkpoints[1].answer_digest_sha256=h('9');
        t.checkpoints[2].answer_digest_sha256=h('9');
        for checkpoint in &mut t.checkpoints { checkpoint.evaluator_answer_correct=false; }
        let f=extract_prefix_features(&t,3).unwrap();
        assert_eq!(f[2].answer_stability_ppm,Some(1_000_000));
        assert!(!t.checkpoints[2].evaluator_answer_correct);
    }

    #[test]
    fn unverified_observation_is_not_counted_as_verified_evidence_gain() {
        let t=trajectory();
        let f=extract_prefix_features(&t,3).unwrap();
        assert_eq!(f[3].new_unverified_observations,1);
        assert_eq!(f[3].new_verified_support_events,0);
        assert_eq!(f[3].new_verified_contradiction_events,0);
    }

    #[test]
    fn prefix_features_have_replayable_digest() {
        let t=trajectory();
        let a=extract_prefix_features(&t,3).unwrap();
        let b=extract_prefix_features(&t,3).unwrap();
        assert_eq!(prefix_feature_digest(&a).unwrap(),prefix_feature_digest(&b).unwrap());
    }
}
