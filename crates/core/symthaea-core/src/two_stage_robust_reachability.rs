// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Typed two-stage robust-reachability information semantics for MANIFOLD-006D-A.
//!
//! This module is deliberately **pre-theorem**. It defines the information-pattern
//! waist that later 006D tranches may consume: explicit causal chronologies, a sealed
//! stage-1 observation capability, fixed-before-disturbance strategy objects, and
//! pure-data affine/constant second-stage policies.
//!
//! Runtime identities in this module use BLAKE3 and are intentionally distinct from
//! the audited preregistration SHA3-256 semantic identities. The latter are carried
//! only as typed external evidence bindings. No equivalence between those identity
//! domains is claimed here.

use blake3::Hasher;
use thiserror::Error;

use crate::certified_interval::canonical_zero;

/// A BLAKE3 identity for a Rust runtime object in the 006D-A semantic waist.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RuntimeIdentity([u8; 32]);

impl RuntimeIdentity {
    /// Return the raw 32-byte BLAKE3 digest.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// A SHA3-256 semantic identifier imported from the audited preregistration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PreregisteredSha3Identity([u8; 32]);

impl PreregisteredSha3Identity {
    /// Return the raw 32-byte SHA3-256 semantic identifier.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// A SHA-256 digest for an external evidence artifact or source file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExternalSha256Digest([u8; 32]);

impl ExternalSha256Digest {
    /// Return the raw 32-byte SHA-256 digest.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// A 20-byte Git object identifier carried as external lineage evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GitObjectId([u8; 20]);

impl GitObjectId {
    /// Return the raw Git object identifier bytes.
    pub fn as_bytes(&self) -> &[u8; 20] {
        &self.0
    }
}

/// Audited external evidence binding that authorizes the 006D-A semantic waist.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Manifold006dPreregistrationBinding {
    control_commit: GitObjectId,
    control_tree: GitObjectId,
    numerical_parent_commit: GitObjectId,
    numerical_parent_tree: GitObjectId,
    hosted_run_id: u64,
    artifact_id: u64,
    artifact_zip_sha256: ExternalSha256Digest,
    preregistration_json_sha256: ExternalSha256Digest,
    validator_sha256: ExternalSha256Digest,
    workflow_sha256: ExternalSha256Digest,
    qualified_kernel_sha256: ExternalSha256Digest,
    preregistration_id: PreregisteredSha3Identity,
    policy_classes_id: PreregisteredSha3Identity,
    embedding_id: PreregisteredSha3Identity,
}

impl Manifold006dPreregistrationBinding {
    /// Exact audited preregistration control commit.
    pub fn control_commit(&self) -> GitObjectId {
        self.control_commit
    }

    /// Exact audited preregistration control tree.
    pub fn control_tree(&self) -> GitObjectId {
        self.control_tree
    }

    /// Exact executable-qualified MANIFOLD-006N-B parent commit.
    pub fn numerical_parent_commit(&self) -> GitObjectId {
        self.numerical_parent_commit
    }

    /// Exact executable-qualified MANIFOLD-006N-B parent tree.
    pub fn numerical_parent_tree(&self) -> GitObjectId {
        self.numerical_parent_tree
    }

    /// Hosted preregistration run identifier.
    pub fn hosted_run_id(&self) -> u64 {
        self.hosted_run_id
    }

    /// Hosted preregistration artifact identifier.
    pub fn artifact_id(&self) -> u64 {
        self.artifact_id
    }

    /// Audited ZIP digest of the preregistration evidence capsule.
    pub fn artifact_zip_sha256(&self) -> ExternalSha256Digest {
        self.artifact_zip_sha256
    }

    /// Exact SHA-256 digest of the frozen preregistration JSON.
    pub fn preregistration_json_sha256(&self) -> ExternalSha256Digest {
        self.preregistration_json_sha256
    }

    /// Exact SHA-256 digest of the independent preregistration validator.
    pub fn validator_sha256(&self) -> ExternalSha256Digest {
        self.validator_sha256
    }

    /// Exact SHA-256 digest of the audited preregistration workflow.
    pub fn workflow_sha256(&self) -> ExternalSha256Digest {
        self.workflow_sha256
    }

    /// Exact SHA-256 digest of the qualified certified-interval source bound by v3.5.
    pub fn qualified_kernel_sha256(&self) -> ExternalSha256Digest {
        self.qualified_kernel_sha256
    }

    /// Frozen v3 preregistration semantic identifier.
    pub fn preregistration_id(&self) -> PreregisteredSha3Identity {
        self.preregistration_id
    }

    /// Frozen v3 policy-class semantic identifier.
    pub fn policy_classes_id(&self) -> PreregisteredSha3Identity {
        self.policy_classes_id
    }

    /// Frozen v3 open-loop-to-feedback embedding semantic identifier.
    pub fn embedding_id(&self) -> PreregisteredSha3Identity {
        self.embedding_id
    }
}

/// Exact audited v3.5 preregistration evidence consumed by MANIFOLD-006D-A.
pub const MANIFOLD_006D_PREREGISTRATION: Manifold006dPreregistrationBinding =
    Manifold006dPreregistrationBinding {
        control_commit: GitObjectId([
            0xc0, 0xf5, 0x77, 0x6d, 0x28, 0xd3, 0x21, 0xd4, 0x68, 0x01, 0xca, 0x0f, 0x96,
            0x20, 0xa6, 0x22, 0xd3, 0x5c, 0xa2, 0x8c,
        ]),
        control_tree: GitObjectId([
            0xfc, 0xd3, 0x86, 0x83, 0xfd, 0x65, 0x95, 0x2e, 0x72, 0x76, 0x82, 0x73, 0xf5,
            0x17, 0x03, 0x32, 0x3c, 0x55, 0x08, 0xaa,
        ]),
        numerical_parent_commit: GitObjectId([
            0xee, 0x79, 0xf8, 0x6a, 0x6d, 0x43, 0x1c, 0x46, 0x2a, 0xe8, 0x70, 0x10, 0x94,
            0xa3, 0xe2, 0x9b, 0xe8, 0x8c, 0x4a, 0x5e,
        ]),
        numerical_parent_tree: GitObjectId([
            0x64, 0x5b, 0x16, 0x44, 0x11, 0x2a, 0x82, 0x2c, 0x79, 0xb9, 0x7b, 0xcd, 0x9e,
            0xb2, 0x26, 0xcc, 0x60, 0x4b, 0x3a, 0xf8,
        ]),
        hosted_run_id: 34_919_856_614,
        artifact_id: 10_377_995_319,
        artifact_zip_sha256: ExternalSha256Digest([
            0xc9, 0x8e, 0xdc, 0x97, 0x72, 0x28, 0xa0, 0x94, 0x6d, 0x3a, 0x1a, 0x96, 0x1e,
            0xf4, 0x5f, 0xf0, 0x2f, 0x5f, 0xad, 0x2a, 0x87, 0xbe, 0xed, 0x09, 0x37, 0x26,
            0x63, 0x9a, 0x49, 0x24, 0xb2, 0x14,
        ]),
        preregistration_json_sha256: ExternalSha256Digest([
            0x40, 0x92, 0xb9, 0x71, 0xb2, 0x9a, 0x64, 0xa7, 0xeb, 0x62, 0x2e, 0x5f, 0xde,
            0xe1, 0xf8, 0x0d, 0x0b, 0xaf, 0xbe, 0x7b, 0x9e, 0x6d, 0xc6, 0x51, 0x39, 0x95,
            0xf6, 0x92, 0xac, 0x74, 0x37, 0x1b,
        ]),
        validator_sha256: ExternalSha256Digest([
            0xda, 0x77, 0x45, 0xb8, 0x62, 0xf4, 0x32, 0x60, 0xd7, 0x7a, 0x86, 0xb4, 0x52,
            0xce, 0xbc, 0xcb, 0xc1, 0x74, 0xfa, 0x4f, 0xb0, 0x32, 0xd1, 0x30, 0x00, 0xde,
            0x26, 0xb4, 0xe3, 0x57, 0x27, 0x00,
        ]),
        workflow_sha256: ExternalSha256Digest([
            0xa8, 0x86, 0x5e, 0x35, 0x7e, 0xfa, 0x69, 0x83, 0x46, 0xc1, 0xd5, 0x03, 0x58,
            0x07, 0xac, 0x72, 0xfd, 0xc9, 0x1a, 0xfd, 0xbd, 0xdf, 0x37, 0x29, 0xd1, 0x7a,
            0xf9, 0xb7, 0xb4, 0xf6, 0xa8, 0xf5,
        ]),
        qualified_kernel_sha256: ExternalSha256Digest([
            0xcf, 0xfe, 0x55, 0x74, 0xd0, 0x00, 0xa4, 0x4a, 0x87, 0xa7, 0x57, 0xa7, 0x88,
            0xd0, 0x8c, 0xb7, 0xba, 0x47, 0x48, 0x10, 0xe9, 0xc6, 0xa3, 0xd8, 0x60, 0xd5,
            0x7a, 0xc5, 0x73, 0x52, 0xeb, 0x3c,
        ]),
        preregistration_id: PreregisteredSha3Identity([
            0xfa, 0x62, 0x08, 0x4a, 0x56, 0x2a, 0x06, 0x14, 0x45, 0xc7, 0x40, 0x88, 0x91,
            0x46, 0x16, 0x48, 0x0e, 0xfc, 0x60, 0x08, 0xd2, 0xa1, 0x6d, 0xb5, 0x6b, 0x23,
            0xe9, 0x34, 0x06, 0x12, 0x33, 0x7e,
        ]),
        policy_classes_id: PreregisteredSha3Identity([
            0x9f, 0x8b, 0x34, 0xbc, 0x34, 0x6d, 0x56, 0x91, 0xc3, 0x63, 0xde, 0x6f, 0xf4,
            0xa9, 0x41, 0x10, 0xf4, 0x92, 0xe3, 0xfe, 0xcc, 0x48, 0xed, 0x1f, 0x4d, 0x62,
            0x33, 0xde, 0x9d, 0x98, 0xcf, 0xe9,
        ]),
        embedding_id: PreregisteredSha3Identity([
            0x2e, 0x59, 0xb1, 0x77, 0x0d, 0xa3, 0x5a, 0x1d, 0xf8, 0x70, 0x00, 0x91, 0x6c,
            0xf1, 0xbe, 0xdc, 0x73, 0xde, 0x97, 0x56, 0x73, 0x1d, 0xcd, 0x15, 0x4c, 0xe8,
            0xe8, 0x70, 0x2b, 0xff, 0x57, 0xf1,
        ]),
    };

/// Causal events that define when information becomes available in a two-stage strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TwoStageChronologyEvent {
    /// Select both open-loop controls before either disturbance is realized.
    SelectOpenLoopControls,
    /// Select the first control and the complete feedback policy before disturbance.
    SelectFeedbackStrategy,
    /// Realize the first adversarial disturbance.
    RealizeFirstDisturbance,
    /// Establish the exact intermediate state.
    EstablishStage1State,
    /// Publish the declared exact stage-1 observation capability.
    PublishStage1Observation,
    /// Apply the already-selected second open-loop control.
    ApplySecondOpenLoopControl,
    /// Evaluate the already-selected policy using only the stage-1 observation.
    EvaluateFixedStage1Policy,
    /// Realize the second adversarial disturbance.
    RealizeSecondDisturbance,
    /// Establish the terminal state.
    EstablishTerminalState,
}

impl TwoStageChronologyEvent {
    fn tag(self) -> &'static [u8] {
        match self {
            Self::SelectOpenLoopControls => b"select-open-loop-controls",
            Self::SelectFeedbackStrategy => b"select-feedback-strategy",
            Self::RealizeFirstDisturbance => b"realize-d0",
            Self::EstablishStage1State => b"establish-x1",
            Self::PublishStage1Observation => b"publish-obs1",
            Self::ApplySecondOpenLoopControl => b"apply-u1",
            Self::EvaluateFixedStage1Policy => b"evaluate-fixed-pi",
            Self::RealizeSecondDisturbance => b"realize-d1",
            Self::EstablishTerminalState => b"establish-x2",
        }
    }
}

const OPEN_LOOP_CHRONOLOGY: [TwoStageChronologyEvent; 6] = [
    TwoStageChronologyEvent::SelectOpenLoopControls,
    TwoStageChronologyEvent::RealizeFirstDisturbance,
    TwoStageChronologyEvent::EstablishStage1State,
    TwoStageChronologyEvent::ApplySecondOpenLoopControl,
    TwoStageChronologyEvent::RealizeSecondDisturbance,
    TwoStageChronologyEvent::EstablishTerminalState,
];

const FEEDBACK_CHRONOLOGY: [TwoStageChronologyEvent; 7] = [
    TwoStageChronologyEvent::SelectFeedbackStrategy,
    TwoStageChronologyEvent::RealizeFirstDisturbance,
    TwoStageChronologyEvent::EstablishStage1State,
    TwoStageChronologyEvent::PublishStage1Observation,
    TwoStageChronologyEvent::EvaluateFixedStage1Policy,
    TwoStageChronologyEvent::RealizeSecondDisturbance,
    TwoStageChronologyEvent::EstablishTerminalState,
];

/// Supported 006D-A information-pattern classes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TwoStageInformationPatternKind {
    /// Both controls are fixed before either disturbance is realized.
    OpenLoopTwoStage,
    /// A complete policy is fixed before disturbance and evaluated on exact `x1` after stage 1.
    Stage1ExactObservation1D,
}

/// Runtime representation of one typed two-stage information pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TwoStageInformationPattern1D {
    kind: TwoStageInformationPatternKind,
    identity: RuntimeIdentity,
}

impl TwoStageInformationPattern1D {
    /// Construct the two-stage open-loop information pattern.
    pub fn open_loop() -> Self {
        Self::from_parts(
            TwoStageInformationPatternKind::OpenLoopTwoStage,
            &OPEN_LOOP_CHRONOLOGY,
        )
    }

    /// Construct the exact stage-1 observation feedback information pattern.
    pub fn stage1_exact_observation() -> Self {
        Self::from_parts(
            TwoStageInformationPatternKind::Stage1ExactObservation1D,
            &FEEDBACK_CHRONOLOGY,
        )
    }

    fn from_parts(kind: TwoStageInformationPatternKind, chronology: &[TwoStageChronologyEvent]) -> Self {
        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-manifold-006d-information-pattern-runtime-v1\0");
        match kind {
            TwoStageInformationPatternKind::OpenLoopTwoStage => hasher.update(b"open-loop-two-stage\0"),
            TwoStageInformationPatternKind::Stage1ExactObservation1D => {
                hasher.update(b"stage1-exact-observation-1d\0")
            }
        }
        for event in chronology {
            hasher.update(event.tag());
            hasher.update(b"\0");
        }
        Self {
            kind,
            identity: RuntimeIdentity(*hasher.finalize().as_bytes()),
        }
    }

    /// Information-pattern class.
    pub fn kind(&self) -> TwoStageInformationPatternKind {
        self.kind
    }

    /// Exact causal chronology for this information pattern.
    pub fn chronology(&self) -> &'static [TwoStageChronologyEvent] {
        match self.kind {
            TwoStageInformationPatternKind::OpenLoopTwoStage => &OPEN_LOOP_CHRONOLOGY,
            TwoStageInformationPatternKind::Stage1ExactObservation1D => &FEEDBACK_CHRONOLOGY,
        }
    }

    /// BLAKE3 runtime identity for this typed information pattern.
    pub fn runtime_identity(&self) -> RuntimeIdentity {
        self.identity
    }
}

/// Exact stage-1 observation capability for the feedback information pattern.
///
/// Construction is crate-private so external callers cannot mint observations by
/// claiming arbitrary hidden state. Later theorem code must receive this capability
/// from the declared stage chronology rather than a raw disturbance channel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stage1Observation1D {
    state: f64,
    chronology_identity: RuntimeIdentity,
    information_pattern_identity: RuntimeIdentity,
}

impl Stage1Observation1D {
    #[allow(dead_code)]
    pub(crate) fn issue(
        pattern: TwoStageInformationPattern1D,
        state: f64,
    ) -> Result<Self, TwoStageSemanticError> {
        if pattern.kind() != TwoStageInformationPatternKind::Stage1ExactObservation1D {
            return Err(TwoStageSemanticError::ObservationPatternMismatch);
        }
        if !state.is_finite() {
            return Err(TwoStageSemanticError::NonFiniteValue {
                field: "stage1 observation state",
            });
        }
        let state = canonical_zero(state);
        let chronology_identity = chronology_identity(pattern.chronology());
        Ok(Self {
            state,
            chronology_identity,
            information_pattern_identity: pattern.runtime_identity(),
        })
    }

    /// Exact observed stage-1 state.
    pub fn state(&self) -> f64 {
        self.state
    }

    /// Runtime identity of the chronology that issued this observation.
    pub fn chronology_identity(&self) -> RuntimeIdentity {
        self.chronology_identity
    }

    /// Runtime identity of the information pattern that issued this observation.
    pub fn information_pattern_identity(&self) -> RuntimeIdentity {
        self.information_pattern_identity
    }
}

/// Pure-data constant second-stage policy selected before the first disturbance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstantStage1Policy1D {
    value: f64,
    information_pattern_identity: RuntimeIdentity,
    identity: RuntimeIdentity,
}

impl ConstantStage1Policy1D {
    /// Construct a finite constant policy. Admissibility against actuator bounds is a later theorem obligation.
    pub fn new(value: f64) -> Result<Self, TwoStageSemanticError> {
        if !value.is_finite() {
            return Err(TwoStageSemanticError::NonFiniteValue {
                field: "constant policy value",
            });
        }
        let value = canonical_zero(value);
        let pattern = TwoStageInformationPattern1D::stage1_exact_observation();
        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-manifold-006d-constant-policy-runtime-v1\0");
        hasher.update(pattern.runtime_identity().as_bytes());
        hasher.update(&value.to_bits().to_le_bytes());
        Ok(Self {
            value,
            information_pattern_identity: pattern.runtime_identity(),
            identity: RuntimeIdentity(*hasher.finalize().as_bytes()),
        })
    }

    /// Constant action encoded by the policy.
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Feedback information-pattern runtime identity bound into this policy.
    pub fn information_pattern_identity(&self) -> RuntimeIdentity {
        self.information_pattern_identity
    }

    /// BLAKE3 runtime identity of this policy object.
    pub fn runtime_identity(&self) -> RuntimeIdentity {
        self.identity
    }
}

/// Pure-data affine second-stage policy selected before the first disturbance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AffineStage1Policy1D {
    gain: f64,
    bias: f64,
    domain_lower: f64,
    domain_upper: f64,
    information_pattern_identity: RuntimeIdentity,
    identity: RuntimeIdentity,
}

impl AffineStage1Policy1D {
    /// Construct `u1 = gain * x1 + bias` over one finite closed observation domain.
    ///
    /// This constructor establishes only policy syntax and identity. Domain coverage,
    /// actuator-image containment, and terminal-set containment are later obligations.
    pub fn new(
        gain: f64,
        bias: f64,
        domain_lower: f64,
        domain_upper: f64,
    ) -> Result<Self, TwoStageSemanticError> {
        if !gain.is_finite() {
            return Err(TwoStageSemanticError::NonFiniteValue { field: "affine gain" });
        }
        if !bias.is_finite() {
            return Err(TwoStageSemanticError::NonFiniteValue { field: "affine bias" });
        }
        if !domain_lower.is_finite() || !domain_upper.is_finite() {
            return Err(TwoStageSemanticError::NonFiniteValue {
                field: "affine policy domain",
            });
        }
        if domain_lower > domain_upper {
            return Err(TwoStageSemanticError::InvalidPolicyDomain);
        }
        let gain = canonical_zero(gain);
        let bias = canonical_zero(bias);
        let domain_lower = canonical_zero(domain_lower);
        let domain_upper = canonical_zero(domain_upper);
        let pattern = TwoStageInformationPattern1D::stage1_exact_observation();
        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-manifold-006d-affine-policy-runtime-v1\0");
        hasher.update(pattern.runtime_identity().as_bytes());
        for value in [gain, bias, domain_lower, domain_upper] {
            hasher.update(&value.to_bits().to_le_bytes());
        }
        Ok(Self {
            gain,
            bias,
            domain_lower,
            domain_upper,
            information_pattern_identity: pattern.runtime_identity(),
            identity: RuntimeIdentity(*hasher.finalize().as_bytes()),
        })
    }

    /// Affine gain.
    pub fn gain(&self) -> f64 {
        self.gain
    }

    /// Affine bias.
    pub fn bias(&self) -> f64 {
        self.bias
    }

    /// Inclusive lower bound of the declared policy domain.
    pub fn domain_lower(&self) -> f64 {
        self.domain_lower
    }

    /// Inclusive upper bound of the declared policy domain.
    pub fn domain_upper(&self) -> f64 {
        self.domain_upper
    }

    /// Feedback information-pattern runtime identity bound into this policy.
    pub fn information_pattern_identity(&self) -> RuntimeIdentity {
        self.information_pattern_identity
    }

    /// BLAKE3 runtime identity of this policy object.
    pub fn runtime_identity(&self) -> RuntimeIdentity {
        self.identity
    }
}

/// Sealed pure-data second-stage policy classes available to 006D-A.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Stage1Policy1D {
    /// Constant policy that ignores the observation value.
    Constant(ConstantStage1Policy1D),
    /// Affine policy over one declared observation domain.
    Affine(AffineStage1Policy1D),
}

impl Stage1Policy1D {
    /// Runtime identity of the selected policy object.
    pub fn runtime_identity(&self) -> RuntimeIdentity {
        match self {
            Self::Constant(policy) => policy.runtime_identity(),
            Self::Affine(policy) => policy.runtime_identity(),
        }
    }

    /// Runtime identity of the information pattern bound into the policy.
    pub fn information_pattern_identity(&self) -> RuntimeIdentity {
        match self {
            Self::Constant(policy) => policy.information_pattern_identity(),
            Self::Affine(policy) => policy.information_pattern_identity(),
        }
    }
}

impl From<ConstantStage1Policy1D> for Stage1Policy1D {
    fn from(value: ConstantStage1Policy1D) -> Self {
        Self::Constant(value)
    }
}

impl From<AffineStage1Policy1D> for Stage1Policy1D {
    fn from(value: AffineStage1Policy1D) -> Self {
        Self::Affine(value)
    }
}

/// Fixed two-stage open-loop strategy selected before either disturbance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpenLoopTwoStageStrategy1D {
    first_control: f64,
    second_control: f64,
    information_pattern_identity: RuntimeIdentity,
    identity: RuntimeIdentity,
}

impl OpenLoopTwoStageStrategy1D {
    /// Construct a finite fixed pair `(u0, u1)` with no later reselection channel.
    pub fn new(first_control: f64, second_control: f64) -> Result<Self, TwoStageSemanticError> {
        if !first_control.is_finite() {
            return Err(TwoStageSemanticError::NonFiniteValue {
                field: "open-loop first control",
            });
        }
        if !second_control.is_finite() {
            return Err(TwoStageSemanticError::NonFiniteValue {
                field: "open-loop second control",
            });
        }
        let first_control = canonical_zero(first_control);
        let second_control = canonical_zero(second_control);
        let pattern = TwoStageInformationPattern1D::open_loop();
        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-manifold-006d-open-loop-strategy-runtime-v1\0");
        hasher.update(pattern.runtime_identity().as_bytes());
        hasher.update(&first_control.to_bits().to_le_bytes());
        hasher.update(&second_control.to_bits().to_le_bytes());
        Ok(Self {
            first_control,
            second_control,
            information_pattern_identity: pattern.runtime_identity(),
            identity: RuntimeIdentity(*hasher.finalize().as_bytes()),
        })
    }

    /// First-stage control selected before disturbance.
    pub fn first_control(&self) -> f64 {
        self.first_control
    }

    /// Second-stage control selected before disturbance.
    pub fn second_control(&self) -> f64 {
        self.second_control
    }

    /// Open-loop information-pattern runtime identity.
    pub fn information_pattern_identity(&self) -> RuntimeIdentity {
        self.information_pattern_identity
    }

    /// BLAKE3 runtime identity of this fixed strategy.
    pub fn runtime_identity(&self) -> RuntimeIdentity {
        self.identity
    }
}

/// Fixed feedback strategy `(u0, pi)` selected before the first disturbance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stage1FeedbackStrategy1D {
    first_control: f64,
    policy: Stage1Policy1D,
    information_pattern_identity: RuntimeIdentity,
    identity: RuntimeIdentity,
}

impl Stage1FeedbackStrategy1D {
    /// Construct one fixed first control and one already-selected stage-1 policy.
    pub fn new(first_control: f64, policy: Stage1Policy1D) -> Result<Self, TwoStageSemanticError> {
        if !first_control.is_finite() {
            return Err(TwoStageSemanticError::NonFiniteValue {
                field: "feedback first control",
            });
        }
        let first_control = canonical_zero(first_control);
        let pattern = TwoStageInformationPattern1D::stage1_exact_observation();
        if policy.information_pattern_identity() != pattern.runtime_identity() {
            return Err(TwoStageSemanticError::PolicyPatternMismatch);
        }
        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-manifold-006d-feedback-strategy-runtime-v1\0");
        hasher.update(pattern.runtime_identity().as_bytes());
        hasher.update(&first_control.to_bits().to_le_bytes());
        hasher.update(policy.runtime_identity().as_bytes());
        Ok(Self {
            first_control,
            policy,
            information_pattern_identity: pattern.runtime_identity(),
            identity: RuntimeIdentity(*hasher.finalize().as_bytes()),
        })
    }

    /// First-stage control selected before disturbance.
    pub fn first_control(&self) -> f64 {
        self.first_control
    }

    /// Already-selected pure-data second-stage policy.
    pub fn policy(&self) -> Stage1Policy1D {
        self.policy
    }

    /// Feedback information-pattern runtime identity.
    pub fn information_pattern_identity(&self) -> RuntimeIdentity {
        self.information_pattern_identity
    }

    /// BLAKE3 runtime identity of this fixed strategy.
    pub fn runtime_identity(&self) -> RuntimeIdentity {
        self.identity
    }
}

/// Construction errors for the pre-theorem 006D-A semantic waist.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum TwoStageSemanticError {
    /// A required scalar was NaN or infinite.
    #[error("{field} must be finite")]
    NonFiniteValue {
        /// Field whose finite-value contract was violated.
        field: &'static str,
    },
    /// The affine policy domain was not a valid closed interval.
    #[error("affine policy domain must satisfy lower <= upper")]
    InvalidPolicyDomain,
    /// A stage-1 observation was requested under an information pattern that does not publish it.
    #[error("stage-1 observation capability requires Stage1ExactObservation1D")]
    ObservationPatternMismatch,
    /// A policy was bound to a different information pattern than the feedback strategy.
    #[error("stage-1 policy information-pattern identity mismatch")]
    PolicyPatternMismatch,
}

fn chronology_identity(events: &[TwoStageChronologyEvent]) -> RuntimeIdentity {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-manifold-006d-chronology-runtime-v1\0");
    for event in events {
        hasher.update(event.tag());
        hasher.update(b"\0");
    }
    RuntimeIdentity(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn information_patterns_have_distinct_runtime_identities_and_frozen_order() {
        let open_loop = TwoStageInformationPattern1D::open_loop();
        let feedback = TwoStageInformationPattern1D::stage1_exact_observation();
        assert_ne!(open_loop.runtime_identity(), feedback.runtime_identity());
        assert_eq!(
            open_loop.chronology(),
            &[
                TwoStageChronologyEvent::SelectOpenLoopControls,
                TwoStageChronologyEvent::RealizeFirstDisturbance,
                TwoStageChronologyEvent::EstablishStage1State,
                TwoStageChronologyEvent::ApplySecondOpenLoopControl,
                TwoStageChronologyEvent::RealizeSecondDisturbance,
                TwoStageChronologyEvent::EstablishTerminalState,
            ]
        );
        assert_eq!(
            feedback.chronology(),
            &[
                TwoStageChronologyEvent::SelectFeedbackStrategy,
                TwoStageChronologyEvent::RealizeFirstDisturbance,
                TwoStageChronologyEvent::EstablishStage1State,
                TwoStageChronologyEvent::PublishStage1Observation,
                TwoStageChronologyEvent::EvaluateFixedStage1Policy,
                TwoStageChronologyEvent::RealizeSecondDisturbance,
                TwoStageChronologyEvent::EstablishTerminalState,
            ]
        );
    }

    #[test]
    fn observation_capability_is_feedback_only_and_canonicalizes_zero() {
        let feedback = TwoStageInformationPattern1D::stage1_exact_observation();
        let observation = Stage1Observation1D::issue(feedback, -0.0).expect("feedback observation");
        assert_eq!(observation.state().to_bits(), 0.0f64.to_bits());
        assert_eq!(observation.information_pattern_identity(), feedback.runtime_identity());
        assert_eq!(observation.chronology_identity(), chronology_identity(feedback.chronology()));

        let error = Stage1Observation1D::issue(TwoStageInformationPattern1D::open_loop(), 0.0)
            .expect_err("open loop has no stage-1 observation capability");
        assert_eq!(error, TwoStageSemanticError::ObservationPatternMismatch);
    }
}
