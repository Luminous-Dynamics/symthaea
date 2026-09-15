// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Typed two-stage information semantics for MANIFOLD-006D-A.
//!
//! This module is deliberately pre-theorem. It carries the independently audited
//! preregistration boundary into Rust, freezes the two supported causal information
//! patterns, and defines fixed-before-disturbance pure-data strategy objects.
//!
//! External preregistration identities remain typed canonical hexadecimal strings.
//! Runtime object identities use BLAKE3. No equivalence between those identity
//! domains is claimed by this tranche.

use blake3::Hasher;
use thiserror::Error;

use crate::certified_interval::canonical_zero;

/// A BLAKE3 identity for one Rust runtime object in the 006D-A semantic waist.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RuntimeIdentity([u8; 32]);

impl RuntimeIdentity {
    /// Return the raw 32-byte BLAKE3 digest.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// A SHA3-256 semantic identity imported from the audited preregistration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PreregisteredSha3Identity(&'static str);

impl PreregisteredSha3Identity {
    /// Return the canonical lowercase hexadecimal SHA3-256 identity.
    pub fn as_hex(&self) -> &'static str {
        self.0
    }
}

/// A SHA-256 digest for an external evidence artifact or frozen source file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExternalSha256Digest(&'static str);

impl ExternalSha256Digest {
    /// Return the canonical lowercase hexadecimal SHA-256 digest.
    pub fn as_hex(&self) -> &'static str {
        self.0
    }
}

/// A Git object identifier carried as external lineage evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GitObjectId(&'static str);

impl GitObjectId {
    /// Return the canonical lowercase hexadecimal Git object identifier.
    pub fn as_hex(&self) -> &'static str {
        self.0
    }
}

/// A preregistered semantic-schema version tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PreregisteredSemanticVersion(&'static str);

impl PreregisteredSemanticVersion {
    /// Return the exact preregistered version tag.
    pub fn as_str(&self) -> &'static str {
        self.0
    }
}

const CHRONOLOGY_V3: PreregisteredSemanticVersion =
    PreregisteredSemanticVersion("m006d.chron.v3");
const INFORMATION_PATTERN_V3: PreregisteredSemanticVersion =
    PreregisteredSemanticVersion("m006d.ip.v3");
const OBSERVATION_V3: PreregisteredSemanticVersion =
    PreregisteredSemanticVersion("m006d.obs.v3");
const AFFINE_POLICY_V3: PreregisteredSemanticVersion =
    PreregisteredSemanticVersion("m006d.affine_pi.v3");
const STRATEGY_V2: PreregisteredSemanticVersion =
    PreregisteredSemanticVersion("m006d.strategy.v2");
const POLICY_CLASSES_V2: PreregisteredSemanticVersion =
    PreregisteredSemanticVersion("m006d.classes.v2");

/// Audited external evidence that authorizes the 006D-A semantic waist.
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
    open_loop_chronology_id: PreregisteredSha3Identity,
    feedback_chronology_id: PreregisteredSha3Identity,
    open_loop_information_pattern_id: PreregisteredSha3Identity,
    feedback_information_pattern_id: PreregisteredSha3Identity,
    observation_contract_id: PreregisteredSha3Identity,
    reference_affine_policy_id: PreregisteredSha3Identity,
    open_loop_strategy_contract_id: PreregisteredSha3Identity,
    feedback_strategy_contract_id: PreregisteredSha3Identity,
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

    /// Exact SHA-256 digest of the qualified certified-interval source.
    pub fn qualified_kernel_sha256(&self) -> ExternalSha256Digest {
        self.qualified_kernel_sha256
    }

    /// Frozen v3 preregistration semantic identity.
    pub fn preregistration_id(&self) -> PreregisteredSha3Identity {
        self.preregistration_id
    }

    /// Frozen v3 policy-class semantic identity.
    pub fn policy_classes_id(&self) -> PreregisteredSha3Identity {
        self.policy_classes_id
    }

    /// Frozen open-loop-to-feedback embedding semantic identity.
    pub fn embedding_id(&self) -> PreregisteredSha3Identity {
        self.embedding_id
    }

    /// Frozen open-loop chronology semantic identity.
    pub fn open_loop_chronology_id(&self) -> PreregisteredSha3Identity {
        self.open_loop_chronology_id
    }

    /// Frozen feedback chronology semantic identity.
    pub fn feedback_chronology_id(&self) -> PreregisteredSha3Identity {
        self.feedback_chronology_id
    }

    /// Frozen open-loop information-pattern semantic identity.
    pub fn open_loop_information_pattern_id(&self) -> PreregisteredSha3Identity {
        self.open_loop_information_pattern_id
    }

    /// Frozen feedback information-pattern semantic identity.
    pub fn feedback_information_pattern_id(&self) -> PreregisteredSha3Identity {
        self.feedback_information_pattern_id
    }

    /// Frozen stage-1 observation-contract semantic identity.
    pub fn observation_contract_id(&self) -> PreregisteredSha3Identity {
        self.observation_contract_id
    }

    /// Frozen reference affine-policy semantic identity.
    pub fn reference_affine_policy_id(&self) -> PreregisteredSha3Identity {
        self.reference_affine_policy_id
    }

    /// Frozen open-loop strategy-contract semantic identity.
    pub fn open_loop_strategy_contract_id(&self) -> PreregisteredSha3Identity {
        self.open_loop_strategy_contract_id
    }

    /// Frozen feedback strategy-contract semantic identity.
    pub fn feedback_strategy_contract_id(&self) -> PreregisteredSha3Identity {
        self.feedback_strategy_contract_id
    }

    /// Chronology schema version frozen by preregistration.
    pub fn chronology_version(&self) -> PreregisteredSemanticVersion {
        CHRONOLOGY_V3
    }

    /// Information-pattern schema version frozen by preregistration.
    pub fn information_pattern_version(&self) -> PreregisteredSemanticVersion {
        INFORMATION_PATTERN_V3
    }

    /// Observation schema version frozen by preregistration.
    pub fn observation_version(&self) -> PreregisteredSemanticVersion {
        OBSERVATION_V3
    }

    /// Affine-policy schema version frozen by preregistration.
    pub fn affine_policy_version(&self) -> PreregisteredSemanticVersion {
        AFFINE_POLICY_V3
    }

    /// Strategy schema version frozen by preregistration.
    pub fn strategy_version(&self) -> PreregisteredSemanticVersion {
        STRATEGY_V2
    }

    /// Policy-class schema version frozen by preregistration.
    pub fn policy_classes_version(&self) -> PreregisteredSemanticVersion {
        POLICY_CLASSES_V2
    }
}

/// Exact audited v3.5 preregistration evidence consumed by MANIFOLD-006D-A.
pub const MANIFOLD_006D_PREREGISTRATION: Manifold006dPreregistrationBinding =
    Manifold006dPreregistrationBinding {
        control_commit: GitObjectId("c0f5776d28d321d46801ca0f9620a622d35ca28c"),
        control_tree: GitObjectId("fcd38683fd65952e72768273f51703323c5508aa"),
        numerical_parent_commit: GitObjectId("ee79f86a6d431c462ae8701094a3e29be88c4a5e"),
        numerical_parent_tree: GitObjectId("645b1644112a822c79b97bcd9eb226cc604b3af8"),
        hosted_run_id: 34_919_856_614,
        artifact_id: 10_377_995_319,
        artifact_zip_sha256: ExternalSha256Digest(
            "c98edc977228a0946d3a1a961ef45ff02f5fad2a87beed093726639a4924b214",
        ),
        preregistration_json_sha256: ExternalSha256Digest(
            "4092b971b29a64a7eb622e5fdee1f80d0bafbe7b9e6dc6513995f692ac74371b",
        ),
        validator_sha256: ExternalSha256Digest(
            "da7745b862f43260d77a86b452cebccbc174fa4fb032d13000de26b4e3572700",
        ),
        workflow_sha256: ExternalSha256Digest(
            "a8865e357efa698346c1d5035807ac72fdc91afdbddf3729d17af9b7b4f6a8f5",
        ),
        qualified_kernel_sha256: ExternalSha256Digest(
            "cffe5574d000a44a87a757a788d08cb7ba474810e9c6a3d860d57ac57352eb3c",
        ),
        preregistration_id: PreregisteredSha3Identity(
            "fa62084a562a061445c74088914616480efc6008d2a16db56b23e9340612337e",
        ),
        policy_classes_id: PreregisteredSha3Identity(
            "9f8b34bc346d5691c363de6ff4a94110f492e3fecc48ed1f4d6233de9d98cfe9",
        ),
        embedding_id: PreregisteredSha3Identity(
            "2e59b1770da35a1df87000916cf1bedc73de9756731dcd154ce8e8702bff57f1",
        ),
        open_loop_chronology_id: PreregisteredSha3Identity(
            "994798f9d79ffc2b1724acbf5b90c77f23b38290923e177801cd4af88b26379f",
        ),
        feedback_chronology_id: PreregisteredSha3Identity(
            "31e904c6adf8e21937dd978ef710e8e49a0ab1421c96c29ad5c046a72f16a9bd",
        ),
        open_loop_information_pattern_id: PreregisteredSha3Identity(
            "3548205626704f0ce21932783dc7721d1fa8e995925da0fbac7382728c379ad0",
        ),
        feedback_information_pattern_id: PreregisteredSha3Identity(
            "17984644548708695607c3dcb0a5afdaa7dc41a7dd7ec2029b01a9210b2b6d34",
        ),
        observation_contract_id: PreregisteredSha3Identity(
            "d8793739213b6de12202ab46504f12677bb799304f9a637f23f64c2272cb174a",
        ),
        reference_affine_policy_id: PreregisteredSha3Identity(
            "f84ef73cdebf2ed9ef887b3e7be05bcc3200d36dd43dc9027221522283e84fb3",
        ),
        open_loop_strategy_contract_id: PreregisteredSha3Identity(
            "4c47a248a5fdf7ff2b490ce88c792cccd74600d78661cbba039915b8b2b761f3",
        ),
        feedback_strategy_contract_id: PreregisteredSha3Identity(
            "28a53f1bea4734cd54c92a5f5de2617b58a6f961842d24ed6f4ebd87ad47fbe7",
        ),
    };

/// Causal events that define information availability in a two-stage strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TwoStageChronologyEvent {
    /// Select both open-loop controls before either disturbance is realized.
    SelectOpenLoopControls,
    /// Select the first control and complete feedback policy before disturbance.
    SelectFeedbackStrategy,
    /// Realize the first adversarial disturbance.
    RealizeFirstDisturbance,
    /// Establish the exact intermediate state.
    EstablishStage1State,
    /// Publish the declared exact stage-1 observation capability.
    PublishStage1Observation,
    /// Apply the already-selected second open-loop control.
    ApplySecondOpenLoopControl,
    /// Evaluate the already-selected policy from the declared observation.
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

/// Typed runtime representation of one two-stage information pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TwoStageInformationPattern1D {
    kind: TwoStageInformationPatternKind,
    runtime_identity: RuntimeIdentity,
    preregistered_information_pattern_id: PreregisteredSha3Identity,
    preregistered_chronology_id: PreregisteredSha3Identity,
    semantics_version: PreregisteredSemanticVersion,
}

impl TwoStageInformationPattern1D {
    /// Construct the two-stage open-loop information pattern.
    pub fn open_loop() -> Self {
        Self::from_parts(
            TwoStageInformationPatternKind::OpenLoopTwoStage,
            &OPEN_LOOP_CHRONOLOGY,
            MANIFOLD_006D_PREREGISTRATION.open_loop_information_pattern_id(),
            MANIFOLD_006D_PREREGISTRATION.open_loop_chronology_id(),
        )
    }

    /// Construct the exact stage-1 observation feedback information pattern.
    pub fn stage1_exact_observation() -> Self {
        Self::from_parts(
            TwoStageInformationPatternKind::Stage1ExactObservation1D,
            &FEEDBACK_CHRONOLOGY,
            MANIFOLD_006D_PREREGISTRATION.feedback_information_pattern_id(),
            MANIFOLD_006D_PREREGISTRATION.feedback_chronology_id(),
        )
    }

    fn from_parts(
        kind: TwoStageInformationPatternKind,
        chronology: &[TwoStageChronologyEvent],
        preregistered_information_pattern_id: PreregisteredSha3Identity,
        preregistered_chronology_id: PreregisteredSha3Identity,
    ) -> Self {
        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-manifold-006d-information-pattern-runtime-v1\0");
        match kind {
            TwoStageInformationPatternKind::OpenLoopTwoStage => {
                hasher.update(b"open-loop-two-stage\0");
            }
            TwoStageInformationPatternKind::Stage1ExactObservation1D => {
                hasher.update(b"stage1-exact-observation-1d\0");
            }
        }
        for event in chronology {
            hasher.update(event.tag());
            hasher.update(b"\0");
        }
        Self {
            kind,
            runtime_identity: RuntimeIdentity(*hasher.finalize().as_bytes()),
            preregistered_information_pattern_id,
            preregistered_chronology_id,
            semantics_version: INFORMATION_PATTERN_V3,
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
        self.runtime_identity
    }

    /// Independently preregistered SHA3-256 identity of this information-pattern contract.
    pub fn preregistered_information_pattern_id(&self) -> PreregisteredSha3Identity {
        self.preregistered_information_pattern_id
    }

    /// Independently preregistered SHA3-256 identity of this chronology contract.
    pub fn preregistered_chronology_id(&self) -> PreregisteredSha3Identity {
        self.preregistered_chronology_id
    }

    /// Preregistered information-pattern schema version.
    pub fn semantics_version(&self) -> PreregisteredSemanticVersion {
        self.semantics_version
    }
}

/// Exact stage-1 observation capability for the feedback information pattern.
///
/// Construction is module-private. Sibling modules cannot mint this capability by
/// asserting arbitrary hidden state. A later stage executor inside this authority
/// module may issue it only after the declared stage-1 chronology is satisfied.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stage1Observation1D {
    stage: u8,
    state: f64,
    chronology_identity: RuntimeIdentity,
    information_pattern_identity: RuntimeIdentity,
    preregistered_observation_contract_id: PreregisteredSha3Identity,
    preregistered_chronology_id: PreregisteredSha3Identity,
    preregistered_information_pattern_id: PreregisteredSha3Identity,
    semantics_version: PreregisteredSemanticVersion,
}

impl Stage1Observation1D {
    #[allow(dead_code)]
    fn issue(
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
        Ok(Self {
            stage: 1,
            state: canonical_zero(state),
            chronology_identity: chronology_identity(pattern.chronology()),
            information_pattern_identity: pattern.runtime_identity(),
            preregistered_observation_contract_id: MANIFOLD_006D_PREREGISTRATION
                .observation_contract_id(),
            preregistered_chronology_id: pattern.preregistered_chronology_id(),
            preregistered_information_pattern_id: pattern.preregistered_information_pattern_id(),
            semantics_version: OBSERVATION_V3,
        })
    }

    /// Stage index bound into this observation capability.
    pub fn stage(&self) -> u8 {
        self.stage
    }

    /// Exact observed stage-1 state.
    pub fn state(&self) -> f64 {
        self.state
    }

    /// BLAKE3 runtime identity of the chronology that issued this observation.
    pub fn chronology_identity(&self) -> RuntimeIdentity {
        self.chronology_identity
    }

    /// BLAKE3 runtime identity of the information pattern that issued this observation.
    pub fn information_pattern_identity(&self) -> RuntimeIdentity {
        self.information_pattern_identity
    }

    /// Preregistered observation-contract semantic identity.
    pub fn preregistered_observation_contract_id(&self) -> PreregisteredSha3Identity {
        self.preregistered_observation_contract_id
    }

    /// Preregistered chronology semantic identity carried by this capability.
    pub fn preregistered_chronology_id(&self) -> PreregisteredSha3Identity {
        self.preregistered_chronology_id
    }

    /// Preregistered information-pattern semantic identity carried by this capability.
    pub fn preregistered_information_pattern_id(&self) -> PreregisteredSha3Identity {
        self.preregistered_information_pattern_id
    }

    /// Preregistered observation schema version.
    pub fn semantics_version(&self) -> PreregisteredSemanticVersion {
        self.semantics_version
    }
}

/// Pure-data constant second-stage policy selected before the first disturbance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstantStage1Policy1D {
    value: f64,
    information_pattern_identity: RuntimeIdentity,
    policy_classes_id: PreregisteredSha3Identity,
    identity: RuntimeIdentity,
}

impl ConstantStage1Policy1D {
    /// Construct a finite constant policy; actuator admissibility is a later obligation.
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
            policy_classes_id: MANIFOLD_006D_PREREGISTRATION.policy_classes_id(),
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

    /// Preregistered policy-class identity under which this object is interpreted.
    pub fn policy_classes_id(&self) -> PreregisteredSha3Identity {
        self.policy_classes_id
    }

    /// Preregistered policy-class schema version.
    pub fn semantics_version(&self) -> PreregisteredSemanticVersion {
        POLICY_CLASSES_V2
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
    policy_classes_id: PreregisteredSha3Identity,
    identity: RuntimeIdentity,
}

impl AffineStage1Policy1D {
    /// Construct `u1 = gain * x1 + bias` over one finite closed observation domain.
    ///
    /// This establishes policy syntax only. Domain coverage, actuator-image containment,
    /// and terminal-set containment remain later proof obligations.
    pub fn new(
        gain: f64,
        bias: f64,
        domain_lower: f64,
        domain_upper: f64,
    ) -> Result<Self, TwoStageSemanticError> {
        if !gain.is_finite() {
            return Err(TwoStageSemanticError::NonFiniteValue {
                field: "affine gain",
            });
        }
        if !bias.is_finite() {
            return Err(TwoStageSemanticError::NonFiniteValue {
                field: "affine bias",
            });
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
            policy_classes_id: MANIFOLD_006D_PREREGISTRATION.policy_classes_id(),
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

    /// Preregistered policy-class identity under which this object is interpreted.
    pub fn policy_classes_id(&self) -> PreregisteredSha3Identity {
        self.policy_classes_id
    }

    /// Preregistered affine-policy schema version.
    pub fn semantics_version(&self) -> PreregisteredSemanticVersion {
        AFFINE_POLICY_V3
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
    strategy_contract_id: PreregisteredSha3Identity,
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
            strategy_contract_id: MANIFOLD_006D_PREREGISTRATION.open_loop_strategy_contract_id(),
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

    /// Preregistered open-loop strategy-contract semantic identity.
    pub fn strategy_contract_id(&self) -> PreregisteredSha3Identity {
        self.strategy_contract_id
    }

    /// Preregistered strategy schema version.
    pub fn semantics_version(&self) -> PreregisteredSemanticVersion {
        STRATEGY_V2
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
    strategy_contract_id: PreregisteredSha3Identity,
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
            strategy_contract_id: MANIFOLD_006D_PREREGISTRATION.feedback_strategy_contract_id(),
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

    /// Preregistered feedback strategy-contract semantic identity.
    pub fn strategy_contract_id(&self) -> PreregisteredSha3Identity {
        self.strategy_contract_id
    }

    /// Preregistered strategy schema version.
    pub fn semantics_version(&self) -> PreregisteredSemanticVersion {
        STRATEGY_V2
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
    fn information_patterns_bind_runtime_and_external_identity_domains_separately() {
        let open_loop = TwoStageInformationPattern1D::open_loop();
        let feedback = TwoStageInformationPattern1D::stage1_exact_observation();
        assert_ne!(open_loop.runtime_identity(), feedback.runtime_identity());
        assert_eq!(
            open_loop.preregistered_information_pattern_id(),
            MANIFOLD_006D_PREREGISTRATION.open_loop_information_pattern_id()
        );
        assert_eq!(
            feedback.preregistered_information_pattern_id(),
            MANIFOLD_006D_PREREGISTRATION.feedback_information_pattern_id()
        );
        assert_eq!(
            feedback.semantics_version().as_str(),
            "m006d.ip.v3"
        );
    }

    #[test]
    fn observation_capability_is_stage_bound_feedback_only_and_canonicalizes_zero() {
        let feedback = TwoStageInformationPattern1D::stage1_exact_observation();
        let observation = Stage1Observation1D::issue(feedback, -0.0).expect("feedback observation");
        assert_eq!(observation.stage(), 1);
        assert_eq!(observation.state().to_bits(), 0.0f64.to_bits());
        assert_eq!(
            observation.information_pattern_identity(),
            feedback.runtime_identity()
        );
        assert_eq!(
            observation.preregistered_information_pattern_id(),
            MANIFOLD_006D_PREREGISTRATION.feedback_information_pattern_id()
        );
        assert_eq!(
            observation.preregistered_chronology_id(),
            MANIFOLD_006D_PREREGISTRATION.feedback_chronology_id()
        );
        assert_eq!(
            observation.preregistered_observation_contract_id(),
            MANIFOLD_006D_PREREGISTRATION.observation_contract_id()
        );
        assert_eq!(observation.semantics_version().as_str(), "m006d.obs.v3");

        let error = Stage1Observation1D::issue(TwoStageInformationPattern1D::open_loop(), 0.0)
            .expect_err("open loop has no stage-1 observation capability");
        assert_eq!(error, TwoStageSemanticError::ObservationPatternMismatch);
    }

    #[test]
    fn signed_zero_does_not_split_runtime_strategy_identity() {
        let a = OpenLoopTwoStageStrategy1D::new(0.0, -0.0).expect("strategy");
        let b = OpenLoopTwoStageStrategy1D::new(-0.0, 0.0).expect("strategy");
        assert_eq!(a.runtime_identity(), b.runtime_identity());
    }
}
