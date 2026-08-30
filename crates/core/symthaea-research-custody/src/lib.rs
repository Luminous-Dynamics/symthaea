//! Staged evaluation-custody contracts for Symthaea research.
//!
//! This crate records **who is allowed to access which research artifact, for what purpose, and
//! from which experiment phase onward**. It is intentionally a provenance/policy layer, not an
//! operating-system sandbox or cryptographic access-control implementation. A future Xenia
//! adapter can enforce the same policy at the capability boundary.
//!
//! The central distinction is that Evaluation predictor inputs and Evaluation outcomes are not
//! necessarily the same secret. A model may legitimately read final-evaluation imagery while the
//! ground-truth label or future observation remains sealed until after a forecast is committed.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use symthaea_research_split::{PartitionRole, ResearchSplitManifest};
use thiserror::Error;

const CUSTODY_SCHEMA: &str = "symthaea-research-custody/v1";
const ACCESS_RECEIPT_SCHEMA: &str = "symthaea-research-custody-access/v1";

pub type Result<T> = std::result::Result<T, CustodyError>;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CustodyError {
    #[error("{0} must not be empty")]
    EmptyField(&'static str),
    #[error("custody manifest requires at least one asset")]
    MissingAsset,
    #[error("duplicate custody asset id: {0}")]
    DuplicateAsset(String),
    #[error("duplicate access rule for asset {asset_id}: {principal:?}/{action:?}")]
    DuplicateRule {
        asset_id: String,
        principal: CustodyPrincipal,
        action: CustodyAction,
    },
    #[error("unsafe early rule for sealed outcome asset {asset_id}: {principal:?}/{action:?} from {earliest_phase:?}")]
    UnsafeOutcomeRule {
        asset_id: String,
        principal: CustodyPrincipal,
        action: CustodyAction,
        earliest_phase: CustodyPhase,
    },
    #[error("custody asset references sample outside split: {0}")]
    UnknownSample(String),
    #[error("custody asset sample role changed for {sample_id}: recorded={recorded:?}, actual={actual:?}")]
    SampleRoleMismatch {
        sample_id: String,
        recorded: PartitionRole,
        actual: PartitionRole,
    },
    #[error("custody asset sample digest changed for {0}")]
    SampleDigestMismatch(String),
    #[error("evaluation outcome/label asset must belong to Evaluation: {0}")]
    OutcomeNotEvaluation(String),
    #[error("access denied for asset {asset_id}: {principal:?}/{action:?} at {phase:?}")]
    AccessDenied {
        asset_id: String,
        principal: CustodyPrincipal,
        action: CustodyAction,
        phase: CustodyPhase,
    },
    #[error("custody access references unknown asset: {0}")]
    UnknownAsset(String),
    #[error("split manifest digest does not match custody manifest")]
    SplitManifestMismatch,
    #[error("custody manifest digest mismatch")]
    ManifestDigestMismatch,
    #[error("custody access receipt digest mismatch")]
    ReceiptDigestMismatch,
    #[error("custody serialization failed: {0}")]
    Serialization(String),
    #[error("split validation failed: {0}")]
    Split(String),
}

fn non_empty(value: &str, field: &'static str) -> Result<()> {
    if value.trim().is_empty() {
        Err(CustodyError::EmptyField(field))
    } else {
        Ok(())
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum CustodyPhase {
    /// Protocol/split/model development. Final evaluation must not be consumed for selection.
    Development,
    /// Candidate choice and fit artifacts are frozen.
    SelectionFrozen,
    /// Final predictor inputs may be opened where the study design permits it.
    EvaluationInputsOpen,
    /// Predictions/outputs are committed and verification outcomes may be revealed/scored.
    OutcomeRevealed,
    /// Final result/evidence package may be disclosed according to study policy.
    Published,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CustodyPrincipal {
    ModelProcess,
    Verifier,
    ResearchOperator,
    Public,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CustodyAction {
    Read,
    Transform,
    Score,
    Reveal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CustodyAssetKind {
    PredictorInput,
    VerificationOutcome,
    GroundTruthLabel,
    AncillaryMetadata,
    DerivedArtifact,
}

impl CustodyAssetKind {
    fn requires_evaluation_role(self) -> bool {
        matches!(self, Self::VerificationOutcome | Self::GroundTruthLabel)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccessRule {
    pub principal: CustodyPrincipal,
    pub action: CustodyAction,
    pub earliest_phase: CustodyPhase,
}

impl AccessRule {
    pub const fn new(
        principal: CustodyPrincipal,
        action: CustodyAction,
        earliest_phase: CustodyPhase,
    ) -> Self {
        Self {
            principal,
            action,
            earliest_phase,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CustodyAsset {
    pub asset_id: String,
    pub sample_id: String,
    pub sample_role: PartitionRole,
    pub sample_content_digest: String,
    /// Digest of the specific asset under custody. This may differ from the split-unit digest when
    /// a sample contains separate predictor inputs, labels, outcomes, or derived products.
    pub asset_content_digest: String,
    pub kind: CustodyAssetKind,
    pub access_rules: Vec<AccessRule>,
}

impl CustodyAsset {
    pub fn from_split(
        split: &ResearchSplitManifest,
        asset_id: impl Into<String>,
        sample_id: &str,
        asset_content_digest: impl Into<String>,
        kind: CustodyAssetKind,
        access_rules: Vec<AccessRule>,
    ) -> Result<Self> {
        split
            .verify_digest()
            .map_err(|error| CustodyError::Split(error.to_string()))?;
        let asset_id = asset_id.into();
        let asset_content_digest = asset_content_digest.into();
        non_empty(&asset_id, "custody asset id")?;
        non_empty(sample_id, "custody sample id")?;
        non_empty(&asset_content_digest, "custody asset content digest")?;
        let assignment = split
            .assignments
            .iter()
            .find(|assignment| assignment.unit.sample_id == sample_id)
            .ok_or_else(|| CustodyError::UnknownSample(sample_id.to_string()))?;
        if kind.requires_evaluation_role() && assignment.role != PartitionRole::Evaluation {
            return Err(CustodyError::OutcomeNotEvaluation(sample_id.to_string()));
        }
        validate_rules(&asset_id, kind, &access_rules)?;
        Ok(Self {
            asset_id,
            sample_id: assignment.unit.sample_id.clone(),
            sample_role: assignment.role,
            sample_content_digest: assignment.unit.content_digest.clone(),
            asset_content_digest,
            kind,
            access_rules,
        })
    }

    /// Convenience policy for evaluation predictor inputs: the model may read/transform only once
    /// the final input phase opens; the verifier/operator may inspect from SelectionFrozen.
    pub fn evaluation_input(
        split: &ResearchSplitManifest,
        asset_id: impl Into<String>,
        sample_id: &str,
        asset_content_digest: impl Into<String>,
    ) -> Result<Self> {
        let assignment = split
            .assignments
            .iter()
            .find(|assignment| assignment.unit.sample_id == sample_id)
            .ok_or_else(|| CustodyError::UnknownSample(sample_id.to_string()))?;
        if assignment.role != PartitionRole::Evaluation {
            return Err(CustodyError::OutcomeNotEvaluation(sample_id.to_string()));
        }
        Self::from_split(
            split,
            asset_id,
            sample_id,
            asset_content_digest,
            CustodyAssetKind::PredictorInput,
            vec![
                AccessRule::new(
                    CustodyPrincipal::ModelProcess,
                    CustodyAction::Read,
                    CustodyPhase::EvaluationInputsOpen,
                ),
                AccessRule::new(
                    CustodyPrincipal::ModelProcess,
                    CustodyAction::Transform,
                    CustodyPhase::EvaluationInputsOpen,
                ),
                AccessRule::new(
                    CustodyPrincipal::Verifier,
                    CustodyAction::Read,
                    CustodyPhase::SelectionFrozen,
                ),
                AccessRule::new(
                    CustodyPrincipal::ResearchOperator,
                    CustodyAction::Read,
                    CustodyPhase::SelectionFrozen,
                ),
            ],
        )
    }

    /// Convenience policy for held-out outcomes/labels: a verifier may hold/read them while the
    /// model and research operator remain barred until OutcomeRevealed.
    pub fn evaluation_outcome(
        split: &ResearchSplitManifest,
        asset_id: impl Into<String>,
        sample_id: &str,
        asset_content_digest: impl Into<String>,
        kind: CustodyAssetKind,
    ) -> Result<Self> {
        if !kind.requires_evaluation_role() {
            return Err(CustodyError::OutcomeNotEvaluation(sample_id.to_string()));
        }
        Self::from_split(
            split,
            asset_id,
            sample_id,
            asset_content_digest,
            kind,
            vec![
                AccessRule::new(
                    CustodyPrincipal::Verifier,
                    CustodyAction::Read,
                    CustodyPhase::SelectionFrozen,
                ),
                AccessRule::new(
                    CustodyPrincipal::Verifier,
                    CustodyAction::Score,
                    CustodyPhase::OutcomeRevealed,
                ),
                AccessRule::new(
                    CustodyPrincipal::ModelProcess,
                    CustodyAction::Read,
                    CustodyPhase::OutcomeRevealed,
                ),
                AccessRule::new(
                    CustodyPrincipal::ResearchOperator,
                    CustodyAction::Read,
                    CustodyPhase::OutcomeRevealed,
                ),
                AccessRule::new(
                    CustodyPrincipal::Public,
                    CustodyAction::Reveal,
                    CustodyPhase::Published,
                ),
            ],
        )
    }
}

fn validate_rules(asset_id: &str, kind: CustodyAssetKind, rules: &[AccessRule]) -> Result<()> {
    let mut seen = HashSet::new();
    for rule in rules {
        if !seen.insert((rule.principal, rule.action)) {
            return Err(CustodyError::DuplicateRule {
                asset_id: asset_id.to_string(),
                principal: rule.principal,
                action: rule.action,
            });
        }
        if kind.requires_evaluation_role() {
            let early_model_or_operator_access = matches!(
                rule.principal,
                CustodyPrincipal::ModelProcess | CustodyPrincipal::ResearchOperator
            ) && matches!(rule.action, CustodyAction::Read | CustodyAction::Transform)
                && rule.earliest_phase < CustodyPhase::OutcomeRevealed;
            let early_scoring = rule.action == CustodyAction::Score
                && rule.earliest_phase < CustodyPhase::OutcomeRevealed;
            let early_public_reveal = rule.principal == CustodyPrincipal::Public
                && rule.action == CustodyAction::Reveal
                && rule.earliest_phase < CustodyPhase::Published;
            if early_model_or_operator_access || early_scoring || early_public_reveal {
                return Err(CustodyError::UnsafeOutcomeRule {
                    asset_id: asset_id.to_string(),
                    principal: rule.principal,
                    action: rule.action,
                    earliest_phase: rule.earliest_phase,
                });
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResearchCustodyManifest {
    pub custody_id: String,
    pub split_manifest_digest: String,
    pub assets: Vec<CustodyAsset>,
    pub manifest_digest: String,
}

#[derive(Deserialize)]
struct ResearchCustodyManifestRepr {
    custody_id: String,
    split_manifest_digest: String,
    assets: Vec<CustodyAsset>,
    manifest_digest: String,
}

#[derive(Serialize)]
struct CustodyDigestView<'a> {
    schema: &'static str,
    custody_id: &'a str,
    split_manifest_digest: &'a str,
    assets: &'a [CustodyAsset],
}

impl ResearchCustodyManifest {
    pub fn new(
        split: &ResearchSplitManifest,
        custody_id: impl Into<String>,
        mut assets: Vec<CustodyAsset>,
    ) -> Result<Self> {
        split
            .verify_digest()
            .map_err(|error| CustodyError::Split(error.to_string()))?;
        let custody_id = custody_id.into();
        non_empty(&custody_id, "custody manifest id")?;
        if assets.is_empty() {
            return Err(CustodyError::MissingAsset);
        }
        let mut ids = HashSet::new();
        for asset in &assets {
            validate_asset_against_split(asset, split)?;
            if !ids.insert(asset.asset_id.clone()) {
                return Err(CustodyError::DuplicateAsset(asset.asset_id.clone()));
            }
        }
        assets.sort_by(|a, b| a.asset_id.cmp(&b.asset_id));
        let mut result = Self {
            custody_id,
            split_manifest_digest: split.manifest_digest.clone(),
            assets,
            manifest_digest: String::new(),
        };
        result.manifest_digest = result.compute_digest()?;
        Ok(result)
    }

    fn digest_view(&self) -> CustodyDigestView<'_> {
        CustodyDigestView {
            schema: CUSTODY_SCHEMA,
            custody_id: &self.custody_id,
            split_manifest_digest: &self.split_manifest_digest,
            assets: &self.assets,
        }
    }

    pub fn compute_digest(&self) -> Result<String> {
        let bytes = serde_json::to_vec(&self.digest_view())
            .map_err(|error| CustodyError::Serialization(error.to_string()))?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }

    pub fn verify_digest(&self) -> Result<()> {
        validate_internal(self)?;
        if self.compute_digest()? != self.manifest_digest {
            return Err(CustodyError::ManifestDigestMismatch);
        }
        Ok(())
    }

    pub fn verify_against_split(&self, split: &ResearchSplitManifest) -> Result<()> {
        self.verify_digest()?;
        split
            .verify_digest()
            .map_err(|error| CustodyError::Split(error.to_string()))?;
        if self.split_manifest_digest != split.manifest_digest {
            return Err(CustodyError::SplitManifestMismatch);
        }
        for asset in &self.assets {
            validate_asset_against_split(asset, split)?;
        }
        Ok(())
    }

    pub fn asset(&self, asset_id: &str) -> Option<&CustodyAsset> {
        self.assets.iter().find(|asset| asset.asset_id == asset_id)
    }

    pub fn is_allowed(
        &self,
        asset_id: &str,
        principal: CustodyPrincipal,
        action: CustodyAction,
        phase: CustodyPhase,
    ) -> Result<bool> {
        self.verify_digest()?;
        let asset = self
            .asset(asset_id)
            .ok_or_else(|| CustodyError::UnknownAsset(asset_id.to_string()))?;
        Ok(asset.access_rules.iter().any(|rule| {
            rule.principal == principal && rule.action == action && phase >= rule.earliest_phase
        }))
    }
}

impl TryFrom<ResearchCustodyManifestRepr> for ResearchCustodyManifest {
    type Error = CustodyError;

    fn try_from(value: ResearchCustodyManifestRepr) -> Result<Self> {
        let result = Self {
            custody_id: value.custody_id,
            split_manifest_digest: value.split_manifest_digest,
            assets: value.assets,
            manifest_digest: value.manifest_digest,
        };
        result.verify_digest()?;
        Ok(result)
    }
}

impl<'de> Deserialize<'de> for ResearchCustodyManifest {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let repr = ResearchCustodyManifestRepr::deserialize(deserializer)?;
        Self::try_from(repr).map_err(serde::de::Error::custom)
    }
}

fn validate_internal(manifest: &ResearchCustodyManifest) -> Result<()> {
    non_empty(&manifest.custody_id, "custody manifest id")?;
    non_empty(&manifest.split_manifest_digest, "split manifest digest")?;
    if manifest.assets.is_empty() {
        return Err(CustodyError::MissingAsset);
    }
    let mut ids = HashSet::new();
    for asset in &manifest.assets {
        non_empty(&asset.asset_id, "custody asset id")?;
        non_empty(&asset.sample_id, "custody sample id")?;
        non_empty(&asset.sample_content_digest, "custody sample digest")?;
        non_empty(&asset.asset_content_digest, "custody asset digest")?;
        if asset.kind.requires_evaluation_role() && asset.sample_role != PartitionRole::Evaluation {
            return Err(CustodyError::OutcomeNotEvaluation(asset.sample_id.clone()));
        }
        validate_rules(&asset.asset_id, asset.kind, &asset.access_rules)?;
        if !ids.insert(asset.asset_id.clone()) {
            return Err(CustodyError::DuplicateAsset(asset.asset_id.clone()));
        }
    }
    Ok(())
}

fn validate_asset_against_split(asset: &CustodyAsset, split: &ResearchSplitManifest) -> Result<()> {
    let assignment = split
        .assignments
        .iter()
        .find(|assignment| assignment.unit.sample_id == asset.sample_id)
        .ok_or_else(|| CustodyError::UnknownSample(asset.sample_id.clone()))?;
    if assignment.role != asset.sample_role {
        return Err(CustodyError::SampleRoleMismatch {
            sample_id: asset.sample_id.clone(),
            recorded: asset.sample_role,
            actual: assignment.role,
        });
    }
    if assignment.unit.content_digest != asset.sample_content_digest {
        return Err(CustodyError::SampleDigestMismatch(asset.sample_id.clone()));
    }
    if asset.kind.requires_evaluation_role() && assignment.role != PartitionRole::Evaluation {
        return Err(CustodyError::OutcomeNotEvaluation(asset.sample_id.clone()));
    }
    validate_rules(&asset.asset_id, asset.kind, &asset.access_rules)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AccessReceipt {
    pub receipt_id: String,
    pub custody_manifest_digest: String,
    pub asset_id: String,
    pub asset_content_digest: String,
    pub principal: CustodyPrincipal,
    pub action: CustodyAction,
    pub phase: CustodyPhase,
    /// Digest of the evidence that justifies the current experiment phase, such as a frozen
    /// selection receipt or a forecast/output commitment. This crate records the reference but
    /// does not independently validate the external artifact.
    pub phase_evidence_digest: String,
    pub occurred_at_unix_ms: i64,
    pub receipt_digest: String,
}

#[derive(Deserialize)]
struct AccessReceiptRepr {
    receipt_id: String,
    custody_manifest_digest: String,
    asset_id: String,
    asset_content_digest: String,
    principal: CustodyPrincipal,
    action: CustodyAction,
    phase: CustodyPhase,
    phase_evidence_digest: String,
    occurred_at_unix_ms: i64,
    receipt_digest: String,
}

#[derive(Serialize)]
struct AccessReceiptDigestView<'a> {
    schema: &'static str,
    receipt_id: &'a str,
    custody_manifest_digest: &'a str,
    asset_id: &'a str,
    asset_content_digest: &'a str,
    principal: CustodyPrincipal,
    action: CustodyAction,
    phase: CustodyPhase,
    phase_evidence_digest: &'a str,
    occurred_at_unix_ms: i64,
}

impl AccessReceipt {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        manifest: &ResearchCustodyManifest,
        receipt_id: impl Into<String>,
        asset_id: &str,
        principal: CustodyPrincipal,
        action: CustodyAction,
        phase: CustodyPhase,
        phase_evidence_digest: impl Into<String>,
        occurred_at_unix_ms: i64,
    ) -> Result<Self> {
        manifest.verify_digest()?;
        let receipt_id = receipt_id.into();
        let phase_evidence_digest = phase_evidence_digest.into();
        non_empty(&receipt_id, "custody receipt id")?;
        non_empty(&phase_evidence_digest, "phase evidence digest")?;
        let asset = manifest
            .asset(asset_id)
            .ok_or_else(|| CustodyError::UnknownAsset(asset_id.to_string()))?;
        if !manifest.is_allowed(asset_id, principal, action, phase)? {
            return Err(CustodyError::AccessDenied {
                asset_id: asset_id.to_string(),
                principal,
                action,
                phase,
            });
        }
        let mut result = Self {
            receipt_id,
            custody_manifest_digest: manifest.manifest_digest.clone(),
            asset_id: asset.asset_id.clone(),
            asset_content_digest: asset.asset_content_digest.clone(),
            principal,
            action,
            phase,
            phase_evidence_digest,
            occurred_at_unix_ms,
            receipt_digest: String::new(),
        };
        result.receipt_digest = result.compute_digest()?;
        Ok(result)
    }

    fn digest_view(&self) -> AccessReceiptDigestView<'_> {
        AccessReceiptDigestView {
            schema: ACCESS_RECEIPT_SCHEMA,
            receipt_id: &self.receipt_id,
            custody_manifest_digest: &self.custody_manifest_digest,
            asset_id: &self.asset_id,
            asset_content_digest: &self.asset_content_digest,
            principal: self.principal,
            action: self.action,
            phase: self.phase,
            phase_evidence_digest: &self.phase_evidence_digest,
            occurred_at_unix_ms: self.occurred_at_unix_ms,
        }
    }

    pub fn compute_digest(&self) -> Result<String> {
        let bytes = serde_json::to_vec(&self.digest_view())
            .map_err(|error| CustodyError::Serialization(error.to_string()))?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }

    pub fn verify_digest(&self) -> Result<()> {
        non_empty(&self.receipt_id, "custody receipt id")?;
        non_empty(&self.custody_manifest_digest, "custody manifest digest")?;
        non_empty(&self.asset_id, "custody asset id")?;
        non_empty(&self.asset_content_digest, "custody asset digest")?;
        non_empty(&self.phase_evidence_digest, "phase evidence digest")?;
        if self.compute_digest()? != self.receipt_digest {
            return Err(CustodyError::ReceiptDigestMismatch);
        }
        Ok(())
    }

    pub fn verify_against_manifest(&self, manifest: &ResearchCustodyManifest) -> Result<()> {
        self.verify_digest()?;
        manifest.verify_digest()?;
        if self.custody_manifest_digest != manifest.manifest_digest {
            return Err(CustodyError::ManifestDigestMismatch);
        }
        let asset = manifest
            .asset(&self.asset_id)
            .ok_or_else(|| CustodyError::UnknownAsset(self.asset_id.clone()))?;
        if asset.asset_content_digest != self.asset_content_digest {
            return Err(CustodyError::SampleDigestMismatch(asset.sample_id.clone()));
        }
        if !manifest.is_allowed(&self.asset_id, self.principal, self.action, self.phase)? {
            return Err(CustodyError::AccessDenied {
                asset_id: self.asset_id.clone(),
                principal: self.principal,
                action: self.action,
                phase: self.phase,
            });
        }
        Ok(())
    }
}

impl TryFrom<AccessReceiptRepr> for AccessReceipt {
    type Error = CustodyError;

    fn try_from(value: AccessReceiptRepr) -> Result<Self> {
        let result = Self {
            receipt_id: value.receipt_id,
            custody_manifest_digest: value.custody_manifest_digest,
            asset_id: value.asset_id,
            asset_content_digest: value.asset_content_digest,
            principal: value.principal,
            action: value.action,
            phase: value.phase,
            phase_evidence_digest: value.phase_evidence_digest,
            occurred_at_unix_ms: value.occurred_at_unix_ms,
            receipt_digest: value.receipt_digest,
        };
        result.verify_digest()?;
        Ok(result)
    }
}

impl<'de> Deserialize<'de> for AccessReceipt {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let repr = AccessReceiptRepr::deserialize(deserializer)?;
        Self::try_from(repr).map_err(serde::de::Error::custom)
    }
}
