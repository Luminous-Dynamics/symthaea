//! Compatibility gate for separately trained camera-conditioned vehicle policies.
//!
//! This module does not load or invoke a VehicleController and does not emit
//! actuator commands. It verifies that a declared policy artifact was trained
//! for the exact fused camera/proprioceptive input contract and scope.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{DrivingObservationReceipt, FusedDrivingInput};

pub const CAMERA_POLICY_MANIFEST_SCHEMA_VERSION: u32 = 1;

const INPUT_SCHEMA_DESCRIPTION: &str = "symthaea-camera-driving-fused-input-v1|ContinuousHV|dimension=16384|fusion=weighted-visual-and-proprioceptive|receipt-bound=true";
const ACTION_SCHEMA_DESCRIPTION: &str = "symthaea-vehicle-command-v1|order=steering,throttle,brake|steering=[-1,1]|throttle=[0,1]|brake=[0,1]";

fn digest_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut result = String::with_capacity(64);
    use std::fmt::Write as _;
    for byte in digest.iter() {
        let _ = write!(result, "{byte:02x}");
    }
    result
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub fn camera_policy_input_schema_sha256() -> String {
    digest_hex(INPUT_SCHEMA_DESCRIPTION.as_bytes())
}

pub fn vehicle_action_schema_sha256() -> String {
    digest_hex(ACTION_SCHEMA_DESCRIPTION.as_bytes())
}

/// How the training pipeline obtained target/expert information.
/// This is an explicit disclosure field, not an endorsement or a benchmark rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CameraPolicyPrivilegeProfile {
    NoPrivilegedInputs,
    PrivilegedExpert,
    GroundTruthPerception,
    MixedDeclared,
    Unknown,
}

/// Exact policy-input and training-scope manifest required for camera-conditioned use.
///
/// A legacy proprioceptive-only controller checkpoint does not contain this
/// manifest and therefore cannot pass this compatibility gate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CameraDrivingPolicyManifest {
    pub schema_version: u32,
    pub policy_revision: String,
    /// SHA-256 of the exact serialized checkpoint bytes supplied to validation.
    pub checkpoint_sha256: String,
    /// Stable identity of the architecture/training implementation.
    pub architecture_sha256: String,
    /// SHA-256 of the canonical fused input schema.
    pub fused_input_schema_sha256: String,
    /// SHA-256 of the steering/throttle/brake output schema.
    pub action_output_schema_sha256: String,
    /// Exact VisionManifold configuration used for training.
    pub vision_config_sha256: String,
    /// Exact proprioceptive encoder seed identity used for training.
    pub vehicle_encoder_seed_sha256: String,
    /// Exact camera calibration identity used for training.
    pub camera_id: String,
    pub camera_calibration_sha256: String,
    /// Simulator and renderer identities; changes require an explicit new manifest.
    pub simulator_revision: String,
    pub renderer_revision: String,
    /// Visual fusion weight in parts per million.
    pub visual_weight_ppm: u32,
    pub hdc_dimension: usize,
    /// Input envelope declared during training.
    pub maximum_frame_width: u32,
    pub maximum_frame_height: u32,
    pub maximum_capture_delivery_age_us: u64,
    /// Exact training data and episode split identities.
    pub training_corpus_sha256: String,
    pub training_split_manifest_sha256: String,
    /// Exact privilege disclosure manifest; digest is an identity pointer only.
    pub privilege_manifest_sha256: String,
    pub privilege_profile: CameraPolicyPrivilegeProfile,
    /// Qualified policy scope / ODD identity.
    pub qualified_scope_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraPolicyCompatibilityError {
    UnsupportedManifestVersion,
    MissingPolicyRevision,
    InvalidDigest,
    CheckpointDigestMismatch,
    InputSchemaMismatch,
    ActionSchemaMismatch,
    VisionConfigMismatch,
    EncoderSeedMismatch,
    CameraIdentityMismatch,
    CameraCalibrationMismatch,
    SimulatorRevisionMismatch,
    RendererMismatch,
    VisualWeightMismatch,
    HdcDimensionMismatch,
    PolicyRevisionMismatch,
    ScopeMismatch,
    FrameDimensionsExceeded,
    InvalidTimestampOrder,
    CaptureDeliveryAgeExceeded,
    VisionLearningNotFrozen,
    ReceiptCommitmentMismatch,
    PolicyInputDigestMismatch,
    UnknownPrivilegeProfile,
}

impl CameraDrivingPolicyManifest {
    /// Validate artifact bytes and one actual bridge output against this manifest.
    ///
    /// The caller must invoke this before passing the fused vector to a separately
    /// trained controller. A successful result establishes contract compatibility
    /// only; it does not establish learned driving competence or physical safety.
    pub fn validate_for(
        &self,
        checkpoint_bytes: &[u8],
        observation: &FusedDrivingInput,
        expected_scope_sha256: &str,
    ) -> Result<CameraPolicyCompatibilityReceipt, CameraPolicyCompatibilityError> {
        if self.schema_version != CAMERA_POLICY_MANIFEST_SCHEMA_VERSION {
            return Err(CameraPolicyCompatibilityError::UnsupportedManifestVersion);
        }
        if self.policy_revision.trim().is_empty()
            || self.camera_id.trim().is_empty()
            || self.simulator_revision.trim().is_empty()
            || self.renderer_revision.trim().is_empty()
        {
            return Err(CameraPolicyCompatibilityError::MissingPolicyRevision);
        }

        for digest in [
            self.checkpoint_sha256.as_str(),
            self.architecture_sha256.as_str(),
            self.fused_input_schema_sha256.as_str(),
            self.action_output_schema_sha256.as_str(),
            self.vision_config_sha256.as_str(),
            self.vehicle_encoder_seed_sha256.as_str(),
            self.camera_calibration_sha256.as_str(),
            self.training_corpus_sha256.as_str(),
            self.training_split_manifest_sha256.as_str(),
            self.privilege_manifest_sha256.as_str(),
            self.qualified_scope_sha256.as_str(),
            expected_scope_sha256,
        ] {
            if !valid_sha256(digest) {
                return Err(CameraPolicyCompatibilityError::InvalidDigest);
            }
        }
        if self.privilege_profile == CameraPolicyPrivilegeProfile::Unknown {
            return Err(CameraPolicyCompatibilityError::UnknownPrivilegeProfile);
        }
        if self.checkpoint_sha256 != digest_hex(checkpoint_bytes) {
            return Err(CameraPolicyCompatibilityError::CheckpointDigestMismatch);
        }
        if self.fused_input_schema_sha256 != camera_policy_input_schema_sha256() {
            return Err(CameraPolicyCompatibilityError::InputSchemaMismatch);
        }
        if self.action_output_schema_sha256 != vehicle_action_schema_sha256() {
            return Err(CameraPolicyCompatibilityError::ActionSchemaMismatch);
        }

        let receipt: &DrivingObservationReceipt = &observation.receipt;
        if receipt.schema_version != crate::RECEIPT_SCHEMA_VERSION {
            return Err(CameraPolicyCompatibilityError::ReceiptCommitmentMismatch);
        }
        if receipt.receipt_sha256 != crate::digest_receipt(receipt) {
            return Err(CameraPolicyCompatibilityError::ReceiptCommitmentMismatch);
        }
        if digest_hex(&{
            let mut bytes = Vec::with_capacity((observation.policy_input.dim() + 1) * 4);
            bytes.extend_from_slice(&(observation.policy_input.dim() as u64).to_le_bytes());
            for value in observation.policy_input.as_slice() {
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
            bytes
        }) != receipt.fused_policy_input_sha256
        {
            return Err(CameraPolicyCompatibilityError::PolicyInputDigestMismatch);
        }

        if self.policy_revision != receipt.identity.policy_revision {
            return Err(CameraPolicyCompatibilityError::PolicyRevisionMismatch);
        }
        if self.qualified_scope_sha256 != expected_scope_sha256 {
            return Err(CameraPolicyCompatibilityError::ScopeMismatch);
        }
        if self.vision_config_sha256 != receipt.vision_config_sha256 {
            return Err(CameraPolicyCompatibilityError::VisionConfigMismatch);
        }
        if self.vehicle_encoder_seed_sha256 != receipt.vehicle_encoder_seed_sha256 {
            return Err(CameraPolicyCompatibilityError::EncoderSeedMismatch);
        }
        if self.camera_id != receipt.identity.camera_id {
            return Err(CameraPolicyCompatibilityError::CameraIdentityMismatch);
        }
        if self.simulator_revision != receipt.identity.simulator_revision {
            return Err(CameraPolicyCompatibilityError::SimulatorRevisionMismatch);
        }
        if receipt.capture_timestamp_us != receipt.upstream_capture_timestamp_us
            || receipt.identity.agent_id != receipt.upstream_agent_id
        {
            return Err(CameraPolicyCompatibilityError::ReceiptCommitmentMismatch);
        }
        if self.camera_calibration_sha256 != receipt.upstream_camera_calibration_sha256 {
            return Err(CameraPolicyCompatibilityError::CameraCalibrationMismatch);
        }
        if self.renderer_revision != receipt.upstream_renderer_revision {
            return Err(CameraPolicyCompatibilityError::RendererMismatch);
        }
        if self.visual_weight_ppm != receipt.visual_weight_ppm {
            return Err(CameraPolicyCompatibilityError::VisualWeightMismatch);
        }
        if self.hdc_dimension != receipt.hdc_dimension
            || observation.policy_input.dim() != receipt.hdc_dimension
        {
            return Err(CameraPolicyCompatibilityError::HdcDimensionMismatch);
        }
        if receipt.width > self.maximum_frame_width || receipt.height > self.maximum_frame_height {
            return Err(CameraPolicyCompatibilityError::FrameDimensionsExceeded);
        }
        if self.maximum_capture_delivery_age_us == 0
            || receipt.upstream_render_completed_timestamp_us
                < receipt.upstream_capture_timestamp_us
            || receipt.upstream_delivered_timestamp_us
                < receipt.upstream_render_completed_timestamp_us
            || receipt.upstream_delivered_timestamp_us < receipt.upstream_capture_timestamp_us
        {
            return Err(CameraPolicyCompatibilityError::InvalidTimestampOrder);
        }
        if receipt.upstream_delivered_timestamp_us - receipt.upstream_capture_timestamp_us
            > self.maximum_capture_delivery_age_us
        {
            return Err(CameraPolicyCompatibilityError::CaptureDeliveryAgeExceeded);
        }
        if !receipt.vision_learning_frozen
            || receipt.vision_training_steps_before != receipt.vision_training_steps_after
            || receipt.vision_telemetry.training_triggered
        {
            return Err(CameraPolicyCompatibilityError::VisionLearningNotFrozen);
        }

        Ok(CameraPolicyCompatibilityReceipt {
            manifest_schema_version: self.schema_version,
            policy_revision: self.policy_revision.clone(),
            checkpoint_sha256: self.checkpoint_sha256.clone(),
            fused_input_schema_sha256: self.fused_input_schema_sha256.clone(),
            action_output_schema_sha256: self.action_output_schema_sha256.clone(),
            qualified_scope_sha256: self.qualified_scope_sha256.clone(),
            privilege_profile: self.privilege_profile,
            bridge_receipt_sha256: receipt.receipt_sha256.clone(),
            fused_policy_input_sha256: receipt.fused_policy_input_sha256.clone(),
            compatibility: "INPUT_AND_ARTIFACT_COMPATIBLE_ONLY".to_owned(),
            controller_invoked: false,
            actuator_command_emitted: false,
            operational_authority_granted: false,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CameraPolicyCompatibilityReceipt {
    pub manifest_schema_version: u32,
    pub policy_revision: String,
    pub checkpoint_sha256: String,
    pub fused_input_schema_sha256: String,
    pub action_output_schema_sha256: String,
    pub qualified_scope_sha256: String,
    pub privilege_profile: CameraPolicyPrivilegeProfile,
    pub bridge_receipt_sha256: String,
    pub fused_policy_input_sha256: String,
    pub compatibility: String,
    pub controller_invoked: bool,
    pub actuator_command_emitted: bool,
    pub operational_authority_granted: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

    fn fixture() -> (
        CameraDrivingPolicyManifest,
        Vec<u8>,
        FusedDrivingInput,
        String,
    ) {
        let checkpoint_bytes = b"synthetic-camera-conditioned-checkpoint-v1".to_vec();
        let scope_sha = "a".repeat(64);
        let policy_input = ContinuousHV::random(HDC_DIMENSION, 1234);
        let mut receipt = DrivingObservationReceipt {
            schema_version: 1,
            identity: crate::DrivingRunIdentity {
                run_id: "run-1".into(),
                episode_id: "episode-1".into(),
                scenario_id: "scenario-1".into(),
                agent_id: "agent-1".into(),
                camera_id: "front-camera".into(),
                simulator_revision: "sim-r1".into(),
                policy_revision: "policy-r1".into(),
            },
            capture_sequence: 1,
            capture_timestamp_us: 1_000,
            vehicle_state_timestamp_us: 1_000,
            derived_dt_us: 5_000,
            width: 32,
            height: 32,
            channels: 3,
            visual_weight_ppm: 600_000,
            vision_config_sha256: "b".repeat(64),
            vehicle_encoder_seed_sha256: "c".repeat(64),
            hdc_dimension: HDC_DIMENSION,
            vehicle_encoder_levels: 32,
            upstream_observation_id: "obs-1".into(),
            upstream_receipt_sha256: "d".repeat(64),
            upstream_frame_sha256: "e".repeat(64),
            upstream_clock_id: "clock-1".into(),
            upstream_tick: 1,
            upstream_agent_id: "agent-1".into(),
            upstream_camera_calibration_sha256: "f".repeat(64),
            upstream_renderer_revision: "renderer-r1".into(),
            upstream_capture_timestamp_us: 1_000,
            upstream_render_completed_timestamp_us: 1_010,
            upstream_delivered_timestamp_us: 1_020,
            upstream_applied_action_receipt_id: None,
            upstream_applied_action_receipt_sha256: None,
            upstream_applied_action_command_id: None,
            upstream_applied_action_tick: None,
            previous_receipt_sha256: None,
            raw_frame_sha256: "1".repeat(64),
            visual_state_sha256: "2".repeat(64),
            vehicle_state_sha256: "3".repeat(64),
            previous_vehicle_state_sha256: None,
            previous_visual_state_sha256: None,
            proprioceptive_hv_sha256: "4".repeat(64),
            fused_policy_input_sha256: String::new(),
            vision_telemetry: crate::VisionTelemetryReceipt {
                prediction_error: 0.1,
                manifold_coherence: 0.9,
                attention_entropy: 0.2,
                motion_surprise: 0.1,
                num_salient_patches: 4,
                frame_sequence: 1,
                training_triggered: false,
            },
            vision_learning_frozen: true,
            vision_training_steps_before: 0,
            vision_training_steps_after: 0,
            receipt_sha256: String::new(),
        };
        let mut input_bytes = Vec::with_capacity((policy_input.dim() + 1) * 4);
        input_bytes.extend_from_slice(&(policy_input.dim() as u64).to_le_bytes());
        for value in policy_input.as_slice() {
            input_bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        receipt.fused_policy_input_sha256 = digest_hex(&input_bytes);
        receipt.receipt_sha256 = crate::digest_receipt(&receipt);

        let manifest = CameraDrivingPolicyManifest {
            schema_version: CAMERA_POLICY_MANIFEST_SCHEMA_VERSION,
            policy_revision: "policy-r1".into(),
            checkpoint_sha256: digest_hex(&checkpoint_bytes),
            architecture_sha256: "9".repeat(64),
            fused_input_schema_sha256: camera_policy_input_schema_sha256(),
            action_output_schema_sha256: vehicle_action_schema_sha256(),
            vision_config_sha256: "b".repeat(64),
            vehicle_encoder_seed_sha256: "c".repeat(64),
            camera_id: "front-camera".into(),
            camera_calibration_sha256: "f".repeat(64),
            simulator_revision: "sim-r1".into(),
            renderer_revision: "renderer-r1".into(),
            visual_weight_ppm: 600_000,
            hdc_dimension: HDC_DIMENSION,
            maximum_frame_width: 64,
            maximum_frame_height: 64,
            maximum_capture_delivery_age_us: 100,
            training_corpus_sha256: "8".repeat(64),
            training_split_manifest_sha256: "7".repeat(64),
            privilege_manifest_sha256: "6".repeat(64),
            privilege_profile: CameraPolicyPrivilegeProfile::NoPrivilegedInputs,
            qualified_scope_sha256: scope_sha.clone(),
        };
        (
            manifest,
            checkpoint_bytes,
            FusedDrivingInput {
                policy_input,
                receipt,
            },
            scope_sha,
        )
    }

    #[test]
    fn exact_camera_manifest_and_checkpoint_are_compatible() {
        let (manifest, bytes, input, scope) = fixture();
        let receipt = manifest.validate_for(&bytes, &input, &scope).unwrap();
        assert_eq!(receipt.compatibility, "INPUT_AND_ARTIFACT_COMPATIBLE_ONLY");
        assert!(!receipt.controller_invoked);
        assert!(!receipt.actuator_command_emitted);
        assert!(!receipt.operational_authority_granted);
    }

    #[test]
    fn legacy_or_wrong_checkpoint_bytes_do_not_pass_camera_gate() {
        let (manifest, _, input, scope) = fixture();
        let error = manifest.validate_for(b"legacy-proprioceptive-checkpoint", &input, &scope);
        assert_eq!(
            error,
            Err(CameraPolicyCompatibilityError::CheckpointDigestMismatch)
        );
    }

    #[test]
    fn policy_scope_must_match_exact_expected_scope() {
        let (manifest, bytes, input, _) = fixture();
        let error = manifest.validate_for(&bytes, &input, &"0".repeat(64));
        assert_eq!(error, Err(CameraPolicyCompatibilityError::ScopeMismatch));
    }

    #[test]
    fn camera_calibration_change_requires_a_new_manifest() {
        let (mut manifest, bytes, input, scope) = fixture();
        manifest.camera_calibration_sha256 = "0".repeat(64);
        let error = manifest.validate_for(&bytes, &input, &scope);
        assert_eq!(
            error,
            Err(CameraPolicyCompatibilityError::CameraCalibrationMismatch)
        );
    }

    #[test]
    fn camera_identity_change_requires_a_new_manifest() {
        let (mut manifest, bytes, input, scope) = fixture();
        manifest.camera_id = "rear-camera".into();
        let error = manifest.validate_for(&bytes, &input, &scope);
        assert_eq!(
            error,
            Err(CameraPolicyCompatibilityError::CameraIdentityMismatch)
        );
    }

    #[test]
    fn simulator_revision_change_requires_a_new_manifest() {
        let (mut manifest, bytes, input, scope) = fixture();
        manifest.simulator_revision = "sim-r2".into();
        let error = manifest.validate_for(&bytes, &input, &scope);
        assert_eq!(
            error,
            Err(CameraPolicyCompatibilityError::SimulatorRevisionMismatch)
        );
    }

    #[test]
    fn renderer_change_requires_a_new_manifest() {
        let (mut manifest, bytes, input, scope) = fixture();
        manifest.renderer_revision = "renderer-r2".into();
        let error = manifest.validate_for(&bytes, &input, &scope);
        assert_eq!(error, Err(CameraPolicyCompatibilityError::RendererMismatch));
    }

    #[test]
    fn visual_weight_change_requires_a_new_manifest() {
        let (mut manifest, bytes, input, scope) = fixture();
        manifest.visual_weight_ppm = 500_000;
        let error = manifest.validate_for(&bytes, &input, &scope);
        assert_eq!(
            error,
            Err(CameraPolicyCompatibilityError::VisualWeightMismatch)
        );
    }

    #[test]
    fn unqualified_or_unknown_privilege_is_not_accepted() {
        let (mut manifest, bytes, input, scope) = fixture();
        manifest.privilege_profile = CameraPolicyPrivilegeProfile::Unknown;
        let error = manifest.validate_for(&bytes, &input, &scope);
        assert_eq!(
            error,
            Err(CameraPolicyCompatibilityError::UnknownPrivilegeProfile)
        );
    }

    #[test]
    fn vision_configuration_change_requires_a_new_manifest() {
        let (mut manifest, bytes, input, scope) = fixture();
        manifest.vision_config_sha256 = "0".repeat(64);
        let error = manifest.validate_for(&bytes, &input, &scope);
        assert_eq!(
            error,
            Err(CameraPolicyCompatibilityError::VisionConfigMismatch)
        );
    }

    #[test]
    fn proprioceptive_encoder_change_requires_a_new_manifest() {
        let (mut manifest, bytes, input, scope) = fixture();
        manifest.vehicle_encoder_seed_sha256 = "0".repeat(64);
        let error = manifest.validate_for(&bytes, &input, &scope);
        assert_eq!(
            error,
            Err(CameraPolicyCompatibilityError::EncoderSeedMismatch)
        );
    }

    #[test]
    fn fused_input_schema_change_requires_a_new_manifest() {
        let (mut manifest, bytes, input, scope) = fixture();
        manifest.fused_input_schema_sha256 = "0".repeat(64);
        let error = manifest.validate_for(&bytes, &input, &scope);
        assert_eq!(
            error,
            Err(CameraPolicyCompatibilityError::InputSchemaMismatch)
        );
    }

    #[test]
    fn action_output_schema_change_requires_a_new_manifest() {
        let (mut manifest, bytes, input, scope) = fixture();
        manifest.action_output_schema_sha256 = "0".repeat(64);
        let error = manifest.validate_for(&bytes, &input, &scope);
        assert_eq!(
            error,
            Err(CameraPolicyCompatibilityError::ActionSchemaMismatch)
        );
    }

    #[test]
    fn changed_hdc_dimension_is_rejected() {
        let (mut manifest, bytes, input, scope) = fixture();
        manifest.hdc_dimension = 8_192;
        let error = manifest.validate_for(&bytes, &input, &scope);
        assert_eq!(
            error,
            Err(CameraPolicyCompatibilityError::HdcDimensionMismatch)
        );
    }

    #[test]
    fn camera_learning_must_remain_frozen() {
        let (manifest, bytes, mut input, scope) = fixture();
        input.receipt.vision_training_steps_after = 1;
        input.receipt.receipt_sha256 = crate::digest_receipt(&input.receipt);
        let error = manifest.validate_for(&bytes, &input, &scope);
        assert_eq!(
            error,
            Err(CameraPolicyCompatibilityError::VisionLearningNotFrozen)
        );
    }

    #[test]
    fn receipt_mutation_is_rejected_even_when_manifest_matches() {
        let (manifest, bytes, mut input, scope) = fixture();
        input.receipt.upstream_renderer_revision = "renderer-r2".into();
        let error = manifest.validate_for(&bytes, &input, &scope);
        assert_eq!(
            error,
            Err(CameraPolicyCompatibilityError::ReceiptCommitmentMismatch)
        );
    }

    #[test]
    fn manifest_rejects_unknown_json_fields() {
        let (manifest, _, _, _) = fixture();
        let mut value = serde_json::to_value(manifest).unwrap();
        value["controller_authorized"] = serde_json::json!(true);
        assert!(serde_json::from_value::<CameraDrivingPolicyManifest>(value).is_err());
    }

    #[test]
    fn stale_bridge_delivery_is_rejected_before_policy_compatibility() {
        let (manifest, bytes, mut input, scope) = fixture();
        input.receipt.upstream_delivered_timestamp_us = 1_200;
        input.receipt.receipt_sha256 = crate::digest_receipt(&input.receipt);
        let error = manifest.validate_for(&bytes, &input, &scope);
        assert_eq!(
            error,
            Err(CameraPolicyCompatibilityError::CaptureDeliveryAgeExceeded)
        );
    }
}
