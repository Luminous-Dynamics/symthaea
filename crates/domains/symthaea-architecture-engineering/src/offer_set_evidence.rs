//! Retained evaluator-state artifacts for independently reconstructable offer sets.
//!
//! OfferSetManifest v0.1 commits to a source-state digest but does not carry the state bytes.
//! This module retains the evaluator-owned state, recomputes its digest, reconstructs each
//! OfferSetSnapshot, and binds the resulting evidence bundle to the policy EventTape.
//!
//! Content integrity does not prove evaluator independence, honesty, or scientific correctness.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::event_tape::EventTape;
use super::offer_set::{verify_matched_offer_sets, OfferSetManifest, OfferSetMatchSummary, OfferSetSnapshot};
use super::paired_inference::{
    analyze_qualified_paired_metric, PairedInferenceError, PairedInferencePlan,
    PairedMetricObservation, QualifiedPairedInference,
};

const SCHEMA: &str = "symthaea.ar5b4.offer-set-evidence-bundle";
const VERSION: &str = "0.1.0";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OfferSetEvidenceError {
    pub path: String,
    pub message: String,
}

impl OfferSetEvidenceError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self { path: path.into(), message: message.into() }
    }
}

impl fmt::Display for OfferSetEvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for OfferSetEvidenceError {}

/// Full evaluator-owned state from which one decision-boundary offer set was derived.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OfferSetSourceState {
    pub schema: String,
    pub version: String,
    pub campaign_id: String,
    pub source_commit: String,
    pub round_id: u64,
    pub batch_id: Option<String>,
    pub observed_through_sequence: u64,
    /// Canonical direction IDs derived from the retained evaluator state.
    pub available_direction_ids: Vec<String>,
    pub new_proposal_allowed: bool,
    /// Complete evaluator-owned state needed to reproduce the offer decision.
    pub evaluator_state: Value,
}

impl OfferSetSourceState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        campaign_id: impl Into<String>,
        source_commit: impl Into<String>,
        round_id: u64,
        batch_id: Option<String>,
        observed_through_sequence: u64,
        mut available_direction_ids: Vec<String>,
        new_proposal_allowed: bool,
        evaluator_state: Value,
    ) -> Result<Self, OfferSetEvidenceError> {
        available_direction_ids.sort_unstable();
        let state = Self {
            schema: SCHEMA.to_owned(),
            version: VERSION.to_owned(),
            campaign_id: campaign_id.into(),
            source_commit: source_commit.into(),
            round_id,
            batch_id,
            observed_through_sequence,
            available_direction_ids,
            new_proposal_allowed,
            evaluator_state,
        };
        state.validate()?;
        Ok(state)
    }

    pub fn validate(&self) -> Result<(), OfferSetEvidenceError> {
        if self.schema != SCHEMA || self.version != VERSION {
            return Err(OfferSetEvidenceError::new("state.schema", "unsupported source-state schema/version"));
        }
        for (path, value) in [
            ("campaign_id", self.campaign_id.as_str()),
            ("source_commit", self.source_commit.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(OfferSetEvidenceError::new(format!("state.{path}"), "value must be non-empty"));
            }
        }
        if self.round_id == 0 {
            return Err(OfferSetEvidenceError::new("state.round_id", "round IDs begin at one"));
        }
        if self.batch_id.as_ref().is_some_and(|batch| batch.trim().is_empty()) {
            return Err(OfferSetEvidenceError::new("state.batch_id", "batch ID cannot be empty"));
        }
        if self.available_direction_ids.iter().any(|id| id.trim().is_empty())
            || self.available_direction_ids.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(OfferSetEvidenceError::new(
                "state.available_direction_ids",
                "direction IDs must be non-empty, sorted, and unique",
            ));
        }
        if !self.evaluator_state.is_object() {
            return Err(OfferSetEvidenceError::new(
                "state.evaluator_state",
                "retained evaluator state must be a JSON object",
            ));
        }
        let source_offers = self.evaluator_state
            .get("available_direction_ids")
            .and_then(Value::as_array)
            .ok_or_else(|| OfferSetEvidenceError::new(
                "state.evaluator_state.available_direction_ids",
                "retained state must include the exact eligible direction list",
            ))?;
        let source_ids = source_offers.iter()
            .map(|value| value.as_str().map(str::to_owned).ok_or_else(|| {
                OfferSetEvidenceError::new(
                    "state.evaluator_state.available_direction_ids",
                    "every eligible direction ID must be a string",
                )
            }))
            .collect::<Result<Vec<_>, _>>()?;
        if source_ids != self.available_direction_ids {
            return Err(OfferSetEvidenceError::new(
                "state.evaluator_state.available_direction_ids",
                "retained state's eligible direction list differs from the extracted offer set",
            ));
        }
        let source_new_permission = self.evaluator_state
            .get("new_proposal_allowed")
            .and_then(Value::as_bool)
            .ok_or_else(|| OfferSetEvidenceError::new(
                "state.evaluator_state.new_proposal_allowed",
                "retained state must include the NEW permission boolean",
            ))?;
        if source_new_permission != self.new_proposal_allowed {
            return Err(OfferSetEvidenceError::new(
                "state.evaluator_state.new_proposal_allowed",
                "retained state's NEW permission differs from the extracted offer set",
            ));
        }
        if !self.evaluator_state.get("state").is_some_and(Value::is_object) {
            return Err(OfferSetEvidenceError::new(
                "state.evaluator_state.state",
                "retained evaluator state payload must be a JSON object",
            ));
        }
        Ok(())
    }

    fn boundary_key(&self) -> (u64, Option<String>) {
        (self.round_id, self.batch_id.clone())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OfferSetStateArtifact {
    pub state: OfferSetSourceState,
    /// SHA-256 of the complete serialized source-state object, including evaluator_state.
    pub state_sha256: String,
}

impl OfferSetStateArtifact {
    pub fn new(state: OfferSetSourceState) -> Result<Self, OfferSetEvidenceError> {
        state.validate()?;
        let state_sha256 = digest_serializable(&state)?;
        Ok(Self { state, state_sha256 })
    }

    pub fn validate(&self) -> Result<(), OfferSetEvidenceError> {
        self.state.validate()?;
        validate_sha256(&self.state_sha256, "state_sha256")?;
        if digest_serializable(&self.state)? != self.state_sha256 {
            return Err(OfferSetEvidenceError::new(
                "state_sha256",
                "retained evaluator-state artifact does not match its content digest",
            ));
        }
        Ok(())
    }

    pub fn to_snapshot(&self) -> Result<OfferSetSnapshot, OfferSetEvidenceError> {
        self.validate()?;
        OfferSetSnapshot::new(
            self.state.round_id,
            self.state.batch_id.clone(),
            self.state.observed_through_sequence,
            self.state.available_direction_ids.clone(),
            self.state.new_proposal_allowed,
            &self.state_sha256,
        ).map_err(|error| OfferSetEvidenceError::new("snapshot", error.to_string()))
    }

    fn matches_snapshot(&self, snapshot: &OfferSetSnapshot) -> Result<(), OfferSetEvidenceError> {
        self.validate()?;
        if self.state.round_id != snapshot.round_id
            || self.state.batch_id != snapshot.batch_id
            || self.state.observed_through_sequence != snapshot.observed_through_sequence
            || self.state.available_direction_ids != snapshot.available_direction_ids
            || self.state.new_proposal_allowed != snapshot.new_proposal_allowed
            || self.state_sha256 != snapshot.source_state_sha256
        {
            return Err(OfferSetEvidenceError::new(
                "state_snapshot",
                "retained evaluator state does not reconstruct the manifest snapshot",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OfferSetEvidenceBundle {
    pub schema: String,
    pub version: String,
    pub manifest: OfferSetManifest,
    /// One retained source-state artifact per manifest boundary, in canonical order.
    pub state_artifacts: Vec<OfferSetStateArtifact>,
    pub bundle_sha256: String,
}

#[derive(Serialize)]
struct UnsignedBundle<'a> {
    schema: &'a str,
    version: &'a str,
    manifest: &'a OfferSetManifest,
    state_artifacts: &'a [OfferSetStateArtifact],
}

impl OfferSetEvidenceBundle {
    pub fn new(tape: &EventTape, states: Vec<OfferSetSourceState>) -> Result<Self, OfferSetEvidenceError> {
        let mut state_artifacts = states.into_iter()
            .map(OfferSetStateArtifact::new)
            .collect::<Result<Vec<_>, _>>()?;
        state_artifacts.sort_by_key(|artifact| artifact.state.boundary_key());

        let snapshots = state_artifacts.iter()
            .map(OfferSetStateArtifact::to_snapshot)
            .collect::<Result<Vec<_>, _>>()?;
        let manifest = OfferSetManifest::new(tape, snapshots)
            .map_err(|error| OfferSetEvidenceError::new("manifest", error.to_string()))?;

        let mut bundle = Self {
            schema: SCHEMA.to_owned(),
            version: VERSION.to_owned(),
            manifest,
            state_artifacts,
            bundle_sha256: String::new(),
        };
        bundle.bundle_sha256 = bundle.compute_digest()?;
        bundle.validate_for_tape(tape)?;
        Ok(bundle)
    }

    /// Verify bundle digest, manifest/tape binding, retained state hashes, and exact snapshot reconstruction.
    pub fn validate_for_tape(&self, tape: &EventTape) -> Result<(), OfferSetEvidenceError> {
        if self.schema != SCHEMA || self.version != VERSION {
            return Err(OfferSetEvidenceError::new("bundle.schema", "unsupported evidence-bundle schema/version"));
        }
        self.manifest.validate_for_tape(tape)
            .map_err(|error| OfferSetEvidenceError::new("manifest", error.to_string()))?;
        if self.state_artifacts.len() != self.manifest.snapshots.len() {
            return Err(OfferSetEvidenceError::new(
                "state_artifacts",
                "exactly one retained source-state artifact is required per manifest snapshot",
            ));
        }

        let mut seen = BTreeSet::new();
        for (index, (artifact, snapshot)) in self.state_artifacts.iter()
            .zip(&self.manifest.snapshots).enumerate()
        {
            artifact.validate().map_err(|error| {
                OfferSetEvidenceError::new(format!("state_artifacts[{index}]"), error.to_string())
            })?;
            let state = &artifact.state;
            if state.campaign_id != tape.header().campaign_id
                || state.source_commit != tape.header().source_commit
            {
                return Err(OfferSetEvidenceError::new(
                    format!("state_artifacts[{index}].identity"),
                    "source-state campaign or source commit differs from the event tape",
                ));
            }
            if !seen.insert(state.boundary_key()) {
                return Err(OfferSetEvidenceError::new(
                    format!("state_artifacts[{index}].boundary"),
                    "duplicate retained source-state decision boundary",
                ));
            }
            if index > 0 && self.state_artifacts[index - 1].state.boundary_key() >= state.boundary_key() {
                return Err(OfferSetEvidenceError::new(
                    format!("state_artifacts[{index}].boundary"),
                    "source-state artifacts must be sorted by round and batch",
                ));
            }
            artifact.matches_snapshot(snapshot).map_err(|error| {
                OfferSetEvidenceError::new(format!("state_artifacts[{index}].reconstruction"), error.to_string())
            })?;
        }

        validate_sha256(&self.bundle_sha256, "bundle_sha256")?;
        if self.compute_digest()? != self.bundle_sha256 {
            return Err(OfferSetEvidenceError::new("bundle_sha256", "evidence-bundle digest does not match its payload"));
        }
        Ok(())
    }

    pub fn manifest(&self) -> &OfferSetManifest {
        &self.manifest
    }

    fn compute_digest(&self) -> Result<String, OfferSetEvidenceError> {
        digest_serializable(&UnsignedBundle {
            schema: &self.schema,
            version: &self.version,
            manifest: &self.manifest,
            state_artifacts: &self.state_artifacts,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OfferSetEvidenceMatchSummary {
    pub offer_sets: OfferSetMatchSummary,
    pub policy_bundle_sha256: String,
    pub retained_state_artifact_count: usize,
}

#[derive(Serialize)]
struct EvidenceSetDigest<'a> {
    policy_bundles: Vec<(&'a str, &'a str)>,
}

/// Verify all retained source artifacts, then compare their reconstructed offer manifests.
pub fn verify_matched_offer_set_evidence(
    tapes: &[EventTape],
    bundles: &[OfferSetEvidenceBundle],
) -> Result<OfferSetEvidenceMatchSummary, OfferSetEvidenceError> {
    let (offer_sets, _, retained_state_artifact_count) = validate_evidence_for_tapes(tapes, bundles)?;
    Ok(OfferSetEvidenceMatchSummary {
        offer_sets,
        policy_bundle_sha256: evidence_set_digest(bundles)?,
        retained_state_artifact_count,
    })
}

/// Strongest current paired-analysis entry point: retained evaluator state is verified before
/// matched execution qualification and preregistered paired inference.
pub fn analyze_qualified_paired_metric_with_state_evidence(
    plan: &PairedInferencePlan,
    observations: &[PairedMetricObservation],
    tapes: &[EventTape],
    bundles: &[OfferSetEvidenceBundle],
) -> Result<QualifiedEvidencePairedInference, OfferSetEvidenceError> {
    let (offer_sets, manifests, retained_state_artifact_count) = validate_evidence_for_tapes(tapes, bundles)?;
    let paired_inference = analyze_qualified_paired_metric(plan, observations, tapes, &manifests)
        .map_err(|error: PairedInferenceError| OfferSetEvidenceError::new("paired_inference", error.to_string()))?;
    Ok(QualifiedEvidencePairedInference {
        offer_sets: OfferSetEvidenceMatchSummary {
            offer_sets,
            policy_bundle_sha256: evidence_set_digest(bundles)?,
            retained_state_artifact_count,
        },
        paired_inference,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct QualifiedEvidencePairedInference {
    pub offer_sets: OfferSetEvidenceMatchSummary,
    pub paired_inference: QualifiedPairedInference,
}

fn validate_evidence_for_tapes(
    tapes: &[EventTape],
    bundles: &[OfferSetEvidenceBundle],
) -> Result<(OfferSetMatchSummary, Vec<OfferSetManifest>, usize), OfferSetEvidenceError> {
    if tapes.len() < 2 {
        return Err(OfferSetEvidenceError::new("tapes", "at least two policy event tapes are required"));
    }
    if tapes.len() != bundles.len() {
        return Err(OfferSetEvidenceError::new("bundles", "one evidence bundle is required for every policy tape"));
    }

    let mut manifests = Vec::with_capacity(tapes.len());
    let mut seen_policy_ids = BTreeSet::new();
    let mut retained_count = 0usize;

    for (index, tape) in tapes.iter().enumerate() {
        let policy_id = tape.header().policy.policy_id.as_str();
        if !seen_policy_ids.insert(policy_id) {
            return Err(OfferSetEvidenceError::new(format!("tapes[{index}].policy_id"), "duplicate policy tape"));
        }
        let matching = bundles.iter()
            .filter(|bundle| bundle.manifest.policy_id == policy_id)
            .collect::<Vec<_>>();
        if matching.len() != 1 {
            return Err(OfferSetEvidenceError::new(
                format!("bundles[{index}].policy_id"),
                "exactly one evidence bundle must match each event-tape policy",
            ));
        }
        let bundle = matching[0];
        bundle.validate_for_tape(tape).map_err(|error| {
            OfferSetEvidenceError::new(format!("policy[{policy_id}].evidence"), error.to_string())
        })?;
        retained_count = retained_count.checked_add(bundle.state_artifacts.len())
            .ok_or_else(|| OfferSetEvidenceError::new("retained_state_artifact_count", "count overflow"))?;
        manifests.push(bundle.manifest.clone());
    }

    let bundled_policy_ids = bundles.iter()
        .map(|bundle| bundle.manifest.policy_id.as_str())
        .collect::<BTreeSet<_>>();
    if bundled_policy_ids != seen_policy_ids {
        return Err(OfferSetEvidenceError::new("bundles", "evidence bundles contain a missing or unexpected policy"));
    }

    let offer_sets = verify_matched_offer_sets(tapes, &manifests)
        .map_err(|error| OfferSetEvidenceError::new("matched_offer_sets", error.to_string()))?;
    Ok((offer_sets, manifests, retained_count))
}

fn evidence_set_digest(bundles: &[OfferSetEvidenceBundle]) -> Result<String, OfferSetEvidenceError> {
    let mut canonical = bundles.iter().collect::<Vec<_>>();
    canonical.sort_by(|left, right| left.manifest.policy_id.cmp(&right.manifest.policy_id));
    let policy_bundles = canonical.into_iter()
        .map(|bundle| (bundle.manifest.policy_id.as_str(), bundle.bundle_sha256.as_str()))
        .collect();
    digest_serializable(&EvidenceSetDigest { policy_bundles })
}

fn validate_sha256(value: &str, path: &str) -> Result<(), OfferSetEvidenceError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(OfferSetEvidenceError::new(path, "expected a 64-character hexadecimal SHA-256 digest"));
    }
    Ok(())
}

fn digest_serializable<T: Serialize>(value: &T) -> Result<String, OfferSetEvidenceError> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        OfferSetEvidenceError::new("serialization", format!("cannot serialize digest input: {error}"))
    })?;
    let digest = Sha256::digest(bytes);
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use crate::event_tape::{
        ActionEvent, ActionIntent, ActionOutcome, BudgetModel, CandidateDisposition,
        ComparisonMode, EventTapeHeader, ExecutionContract, ExecutionMode, MeterValue,
        PolicySnapshot, ResourceCostWeights, ResourceReceipt,
    };
    use sha2::{Digest, Sha256};

    const H1: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const H2: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const H3: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const H4: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const H5: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";

    fn digest<T: Serialize>(value: &T) -> String {
        let bytes = serde_json::to_vec(value).unwrap();
        let hash = Sha256::digest(bytes);
        hash.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn tape(policy_id: &str) -> EventTape {
        let measured = |value| MeterValue::Measured { value };
        let budget_model = BudgetModel::WeightedResources {
            weights: ResourceCostWeights {
                host_cpu_millis: 1,
                wall_millis: 0,
                accelerator_millis: 0,
                host_memory_byte_millis: 0,
                accelerator_memory_byte_millis: 0,
                energy_millijoules: 0,
                input_tokens: 0,
                output_tokens: 0,
                tool_calls: 0,
                evaluator_calls: 0,
                verifier_calls: 0,
            },
        };
        let config = json!({"policy_parameter": 1});
        let header = EventTapeHeader {
            schema: "symthaea.ar5b4.event-tape".to_owned(),
            version: "0.1.0".to_owned(),
            campaign_id: "campaign-1".to_owned(),
            source_commit: "0123456789abcdef0123456789abcdef01234567".to_owned(),
            execution_environment_sha256: H1.to_owned(),
            comparison_mode: ComparisonMode::MechanismMatched,
            proposal_source_id: "shared-source".to_owned(),
            proposal_stream_sha256: Some(H2.to_owned()),
            task_recipe_sha256: H3.to_owned(),
            comparison_profile_sha256: H4.to_owned(),
            evaluator_sha256: H5.to_owned(),
            verifier_sha256: H1.to_owned(),
            budget_model_id: "weighted-v1".to_owned(),
            budget_model_sha256: digest(&budget_model),
            budget_model,
            budget_limit_microunits: 100,
            initial_direction_ids: vec!["A".to_owned()],
            execution: ExecutionContract {
                mode: ExecutionMode::Sequential,
                worker_count: 1,
                seats_per_round: 1,
                first_round_claim_cap: None,
                steady_state_claim_cap: None,
            },
            policy: PolicySnapshot {
                policy_id: policy_id.to_owned(),
                algorithm_id: format!("algorithm-{policy_id}"),
                config_sha256: digest(&config),
                config,
                has_hidden_oracle_access: false,
            },
        };
        let mut tape = EventTape::new(header).unwrap();
        tape.append(ActionEvent {
            event_id: format!("event-{policy_id}"),
            policy_id: policy_id.to_owned(),
            worker_id: 0,
            round_id: 1,
            batch_id: None,
            observed_through_sequence: 0,
            visible_history_sha256: String::new(),
            retry_of_event_id: None,
            intent: ActionIntent::EvaluateCandidate {
                candidate_sha256: H5.to_owned(),
                parent_candidate_sha256: None,
                semantic_delta_sha256: H3.to_owned(),
                direction_id: "A".to_owned(),
            },
            outcome: ActionOutcome::CandidateEvaluation {
                task_recipe_sha256: H3.to_owned(),
                comparison_profile_sha256: H4.to_owned(),
                evaluator_sha256: H5.to_owned(),
                verifier_sha256: H1.to_owned(),
                verifier_receipt_sha256: H2.to_owned(),
                disposition: CandidateDisposition::Verified,
                score: Some(0.9),
                uncertainty: Some(0.0),
                evidence_sha256: Some(H3.to_owned()),
            },
            resources: ResourceReceipt {
                host_cpu_millis: measured(10),
                wall_millis: measured(5),
                accelerator_millis: measured(0),
                peak_memory_bytes: measured(1024),
                accelerator_peak_memory_bytes: measured(0),
                energy_millijoules: measured(1),
                input_tokens: measured(0),
                output_tokens: measured(0),
                tool_calls: measured(0),
                evaluator_calls: measured(1),
                verifier_calls: measured(1),
                charged_budget_microunits: measured(10),
            },
        }).unwrap();
        tape
    }

    fn source_state(ids: Vec<String>, evaluator_state: Value) -> OfferSetSourceState {
        OfferSetSourceState::new(
            "campaign-1",
            "0123456789abcdef0123456789abcdef01234567",
            1,
            None,
            0,
            ids,
            true,
            evaluator_state,
        ).unwrap()
    }

    #[test]
    fn source_state_canonicalizes_offers_and_binds_full_evaluator_payload() {
        let left = OfferSetStateArtifact::new(source_state(
            vec!["B".into(), "A".into()],
            json!({"available_direction_ids": ["A", "B"], "new_proposal_allowed": true, "state": {"verifier_epoch": 7}}),
        )).unwrap();
        let right = OfferSetStateArtifact::new(source_state(
            vec!["A".into(), "B".into()],
            json!({"available_direction_ids": ["A", "B"], "new_proposal_allowed": true, "state": {"verifier_epoch": 7}}),
        )).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.state.available_direction_ids, vec!["A", "B"]);
        assert_eq!(left.to_snapshot().unwrap().source_state_sha256, left.state_sha256);
    }

    #[test]
    fn tampered_evaluator_state_is_rejected_even_when_offer_list_is_unchanged() {
        let mut artifact = OfferSetStateArtifact::new(source_state(
            vec!["A".into()],
            json!({"available_direction_ids": ["A"], "new_proposal_allowed": true, "state": {"eligibility_rule": "verified_only", "verifier_epoch": 7}}),
        )).unwrap();
        artifact.state.evaluator_state["state"]["verifier_epoch"] = json!(8);
        let error = artifact.validate().unwrap_err();
        assert!(error.message.contains("does not match its content digest"));
    }

    #[test]
    fn snapshot_reconstruction_binds_boundary_and_new_permission() {
        let artifact = OfferSetStateArtifact::new(source_state(
            vec!["A".into(), "B".into()],
            json!({"available_direction_ids": ["A", "B"], "new_proposal_allowed": true, "state": {"new_permission": true}}),
        )).unwrap();
        let snapshot = artifact.to_snapshot().unwrap();
        artifact.matches_snapshot(&snapshot).unwrap();
        let changed = OfferSetSnapshot::new(
            1, None, 0, vec!["A".into(), "B".into()], false, &artifact.state_sha256,
        ).unwrap();
        assert!(artifact.matches_snapshot(&changed).is_err());
    }

    #[test]
    fn matched_policies_reuse_the_same_policy_neutral_source_state_artifact() {
        let left_tape = tape("board");
        let right_tape = tape("thompson");
        let shared_state = OfferSetSourceState::new(
            "campaign-1",
            "0123456789abcdef0123456789abcdef01234567",
            1,
            None,
            0,
            vec!["A".into()],
            true,
            json!({"available_direction_ids": ["A"], "new_proposal_allowed": true, "state": {"verifier_epoch": 3}}),
        ).unwrap();
        let left_bundle = OfferSetEvidenceBundle::new(&left_tape, vec![shared_state.clone()]).unwrap();
        let right_bundle = OfferSetEvidenceBundle::new(&right_tape, vec![shared_state]).unwrap();

        assert_eq!(
            left_bundle.state_artifacts[0].state_sha256,
            right_bundle.state_artifacts[0].state_sha256
        );
        let summary = verify_matched_offer_set_evidence(
            &[left_tape, right_tape],
            &[right_bundle, left_bundle],
        ).unwrap();
        assert_eq!(summary.offer_sets.shared_decision_boundaries, 1);
        assert_eq!(summary.retained_state_artifact_count, 2);
    }

    #[test]
    fn bundle_validation_rejects_tampered_retained_state() {
        let event_tape = tape("board");
        let state = OfferSetSourceState::new(
            "campaign-1",
            "0123456789abcdef0123456789abcdef01234567",
            1,
            None,
            0,
            vec!["A".into()],
            true,
            json!({"available_direction_ids": ["A"], "new_proposal_allowed": true, "state": {"verifier_epoch": 3}}),
        ).unwrap();
        let mut bundle = OfferSetEvidenceBundle::new(&event_tape, vec![state]).unwrap();
        bundle.state_artifacts[0].state.evaluator_state["state"]["verifier_epoch"] = json!(4);
        assert!(bundle.validate_for_tape(&event_tape).is_err());
    }

    #[test]
    fn duplicate_or_non_object_source_state_is_rejected() {
        assert!(OfferSetSourceState::new(
            "campaign-1", "0123456789abcdef0123456789abcdef01234567",
            1, None, 0, vec!["A".into(), "A".into()], true, json!({"available_direction_ids": ["A"], "new_proposal_allowed": true, "state": {}}),
        ).is_err());
        assert!(OfferSetSourceState::new(
            "campaign-1", "0123456789abcdef0123456789abcdef01234567",
            1, None, 0, vec!["A".into()], true, json!(["not", "an", "object"]),
        ).is_err());
        assert!(OfferSetSourceState::new(
            "campaign-1", "0123456789abcdef0123456789abcdef01234567",
            1, None, 0, vec!["A".into()], true,
            json!({"available_direction_ids": ["B"], "new_proposal_allowed": true, "state": {}}),
        ).is_err());
    }
}
