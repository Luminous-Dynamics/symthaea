// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical SHA-256 manifests for Reach qualification perturbations.
//!
//! Qualification evidence previously accepted an upstream 64-bit perturbation
//! configuration fingerprint. This module gives the humanoid domain a typed,
//! canonical manifest whose full contents are committed with SHA-256. A derived
//! 64-bit value remains available only for compatibility with the lower campaign
//! machinery; promotion-grade code must use `manifest_digest`.

use std::collections::BTreeSet;

use crate::evidence_digest::{HumanoidEvidenceDigest, HumanoidEvidenceHasher};
use crate::qualification::HumanoidQualificationSubject;
use crate::reach_qualification_lineage::HumanoidReachPerturbationProfileBinding;
use crate::types::{ActuationMode, HumanoidTask};

pub const HUMANOID_REACH_PERTURBATION_MANIFEST_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq)]
pub enum HumanoidReachPerturbationValue {
    Bool(bool),
    U64(u64),
    F64(f64),
    Text(String),
    F64Range { minimum: f64, maximum: f64 },
    F64Set(Vec<f64>),
    TextSet(Vec<String>),
    Digest(HumanoidEvidenceDigest),
}

impl HumanoidReachPerturbationValue {
    fn validate(&self) -> bool {
        match self {
            Self::Bool(_) | Self::U64(_) => true,
            Self::F64(value) => value.is_finite(),
            Self::Text(value) => valid_id(value),
            Self::F64Range { minimum, maximum } => {
                minimum.is_finite() && maximum.is_finite() && maximum >= minimum
            }
            Self::F64Set(values) => {
                if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
                    return false;
                }
                let mut seen = BTreeSet::new();
                values.iter().all(|value| seen.insert(value.to_bits()))
            }
            Self::TextSet(values) => {
                if values.is_empty() || values.iter().any(|value| !valid_id(value)) {
                    return false;
                }
                let mut seen = BTreeSet::new();
                values.iter().all(|value| seen.insert(value.as_str()))
            }
            Self::Digest(value) => !value.is_zero(),
        }
    }

    fn hash_into(&self, hasher: &mut HumanoidEvidenceHasher) {
        match self {
            Self::Bool(value) => {
                hasher.u64(1).bool(*value);
            }
            Self::U64(value) => {
                hasher.u64(2).u64(*value);
            }
            Self::F64(value) => {
                hasher.u64(3).f64(*value);
            }
            Self::Text(value) => {
                hasher.u64(4).string(value);
            }
            Self::F64Range { minimum, maximum } => {
                hasher.u64(5).f64(*minimum).f64(*maximum);
            }
            Self::F64Set(values) => {
                let mut ordered = values.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
                ordered.sort_unstable();
                hasher.u64(6).usize(ordered.len());
                for bits in ordered {
                    hasher.f64(f64::from_bits(bits));
                }
            }
            Self::TextSet(values) => {
                let mut ordered = values.iter().map(String::as_str).collect::<Vec<_>>();
                ordered.sort_unstable();
                hasher.u64(7).usize(ordered.len());
                for value in ordered {
                    hasher.string(value);
                }
            }
            Self::Digest(value) => {
                hasher.u64(8).digest(*value);
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachPerturbationEntry {
    /// Stable semantic path such as `dynamics.mass_scale` or
    /// `sensor.hand_position.noise_std_m`.
    pub path: String,
    pub value: HumanoidReachPerturbationValue,
}

impl HumanoidReachPerturbationEntry {
    pub fn validate(&self) -> bool {
        valid_path(&self.path) && self.value.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HumanoidReachPerturbationSeedScheme {
    /// The campaign's explicit `trial_seed` is the root deterministic seed.
    TrialSeed64,
    /// The episode campaign's explicit `episode_seed` is the root deterministic seed.
    EpisodeSeed64,
    /// An upstream deterministic generator supplies a 64-bit seed per evidence unit.
    ExternalDeterministic64 { generator_id: String },
}

impl HumanoidReachPerturbationSeedScheme {
    fn validate(&self) -> bool {
        match self {
            Self::TrialSeed64 | Self::EpisodeSeed64 => true,
            Self::ExternalDeterministic64 { generator_id } => valid_id(generator_id),
        }
    }

    pub const fn supports_trial_evidence(&self) -> bool {
        matches!(Self::TrialSeed64 | Self::ExternalDeterministic64 { .. }, self)
    }

    pub const fn supports_episode_evidence(&self) -> bool {
        matches!(Self::EpisodeSeed64 | Self::ExternalDeterministic64 { .. }, self)
    }

    fn hash_into(&self, hasher: &mut HumanoidEvidenceHasher) {
        match self {
            Self::TrialSeed64 => {
                hasher.u64(1);
            }
            Self::EpisodeSeed64 => {
                hasher.u64(2);
            }
            Self::ExternalDeterministic64 { generator_id } => {
                hasher.u64(3).string(generator_id);
            }
        }
    }
}

/// Complete perturbation/randomization identity for one Reach scenario profile.
///
/// `producer_artifact_digest` commits the exact randomizer/fault-injector build.
/// `environment_artifact_digest` commits the simulator/HIL/world/model bundle to
/// which the randomizer was applied. Version strings remain useful metadata but
/// are never treated as artifact identity.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachPerturbationManifest {
    schema_version: u32,
    profile_id: String,
    producer_id: String,
    producer_version: String,
    producer_artifact_digest: HumanoidEvidenceDigest,
    environment_id: String,
    environment_version: String,
    environment_artifact_digest: HumanoidEvidenceDigest,
    subject_digest: HumanoidEvidenceDigest,
    seed_scheme: HumanoidReachPerturbationSeedScheme,
    entries: Vec<HumanoidReachPerturbationEntry>,
    manifest_digest: HumanoidEvidenceDigest,
}

impl HumanoidReachPerturbationManifest {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        subject: &HumanoidQualificationSubject,
        profile_id: impl Into<String>,
        producer_id: impl Into<String>,
        producer_version: impl Into<String>,
        producer_artifact_digest: HumanoidEvidenceDigest,
        environment_id: impl Into<String>,
        environment_version: impl Into<String>,
        environment_artifact_digest: HumanoidEvidenceDigest,
        seed_scheme: HumanoidReachPerturbationSeedScheme,
        entries: Vec<HumanoidReachPerturbationEntry>,
    ) -> Option<Self> {
        let subject_digest = digest_subject(subject)?;
        let mut manifest = Self {
            schema_version: HUMANOID_REACH_PERTURBATION_MANIFEST_SCHEMA_VERSION,
            profile_id: profile_id.into(),
            producer_id: producer_id.into(),
            producer_version: producer_version.into(),
            producer_artifact_digest,
            environment_id: environment_id.into(),
            environment_version: environment_version.into(),
            environment_artifact_digest,
            subject_digest,
            seed_scheme,
            entries,
            manifest_digest: HumanoidEvidenceDigest::ZERO,
        };
        if !manifest.validate_shape(subject) {
            return None;
        }
        manifest.manifest_digest = digest_manifest(&manifest);
        manifest.validate_for(subject).then_some(manifest)
    }

    pub fn profile_id(&self) -> &str {
        &self.profile_id
    }

    pub const fn manifest_digest(&self) -> HumanoidEvidenceDigest {
        self.manifest_digest
    }

    pub const fn producer_artifact_digest(&self) -> HumanoidEvidenceDigest {
        self.producer_artifact_digest
    }

    pub const fn environment_artifact_digest(&self) -> HumanoidEvidenceDigest {
        self.environment_artifact_digest
    }

    pub fn seed_scheme(&self) -> &HumanoidReachPerturbationSeedScheme {
        &self.seed_scheme
    }

    pub fn entries(&self) -> &[HumanoidReachPerturbationEntry] {
        &self.entries
    }

    pub fn validate_for(&self, subject: &HumanoidQualificationSubject) -> bool {
        self.validate_shape(subject)
            && !self.manifest_digest.is_zero()
            && self.manifest_digest == digest_manifest(self)
    }

    /// Compatibility binding for the lower FNV-based campaign machinery.
    /// Promotion-grade callers must retain and compare the full SHA-256 digest.
    pub fn legacy_binding(&self) -> HumanoidReachPerturbationProfileBinding {
        HumanoidReachPerturbationProfileBinding {
            profile_id: self.profile_id.clone(),
            configuration_fingerprint: legacy_u64(self.manifest_digest),
        }
    }

    fn validate_shape(&self, subject: &HumanoidQualificationSubject) -> bool {
        if self.schema_version != HUMANOID_REACH_PERTURBATION_MANIFEST_SCHEMA_VERSION
            || digest_subject(subject) != Some(self.subject_digest)
            || !valid_id(&self.profile_id)
            || !valid_id(&self.producer_id)
            || !valid_id(&self.producer_version)
            || self.producer_artifact_digest.is_zero()
            || !valid_id(&self.environment_id)
            || !valid_id(&self.environment_version)
            || self.environment_artifact_digest.is_zero()
            || !self.seed_scheme.validate()
            || self.entries.is_empty()
            || self.entries.iter().any(|entry| !entry.validate())
        {
            return false;
        }
        let mut paths = BTreeSet::new();
        self.entries.iter().all(|entry| paths.insert(entry.path.as_str()))
    }
}

fn digest_subject(subject: &HumanoidQualificationSubject) -> Option<HumanoidEvidenceDigest> {
    if !subject.validate() || subject.task != HumanoidTask::Reach {
        return None;
    }
    let mut h = HumanoidEvidenceHasher::new("reach.perturbation-subject.v1");
    h.u32(subject.schema_version)
        .string(subject.morphology.schema_id())
        .u64(task_id(subject.task))
        .u64(actuation_mode_id(subject.actuation_mode))
        .string(&subject.backend_profile_id);
    Some(h.finish())
}

fn digest_manifest(manifest: &HumanoidReachPerturbationManifest) -> HumanoidEvidenceDigest {
    let mut entries = manifest.entries.iter().collect::<Vec<_>>();
    entries.sort_by(|left, right| left.path.cmp(&right.path));

    let mut h = HumanoidEvidenceHasher::new("reach.perturbation-manifest.v1");
    h.u32(manifest.schema_version)
        .string(&manifest.profile_id)
        .string(&manifest.producer_id)
        .string(&manifest.producer_version)
        .digest(manifest.producer_artifact_digest)
        .string(&manifest.environment_id)
        .string(&manifest.environment_version)
        .digest(manifest.environment_artifact_digest)
        .digest(manifest.subject_digest);
    manifest.seed_scheme.hash_into(&mut h);
    h.usize(entries.len());
    for entry in entries {
        h.string(&entry.path);
        entry.value.hash_into(&mut h);
    }
    h.finish()
}

fn legacy_u64(digest: HumanoidEvidenceDigest) -> u64 {
    let bytes = digest.as_bytes();
    let mut value = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]);
    if value == 0 {
        value = 1;
    }
    value
}

fn task_id(task: HumanoidTask) -> u64 {
    match task {
        HumanoidTask::Stand => 1,
        HumanoidTask::Walk => 2,
        HumanoidTask::Run => 3,
        HumanoidTask::Reach => 4,
        HumanoidTask::Grasp => 5,
    }
}

fn actuation_mode_id(mode: ActuationMode) -> u64 {
    match mode {
        ActuationMode::NormalizedTorque => 1,
        ActuationMode::TorqueNewtonMetres => 2,
        ActuationMode::NormalizedPosition => 3,
        ActuationMode::PositionTargetRadians => 4,
    }
}

fn valid_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && value.len() <= 256
        && value
            .bytes()
            .all(|byte| byte.is_ascii_graphic() && !byte.is_ascii_whitespace())
}

fn valid_path(value: &str) -> bool {
    valid_id(value)
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b'/' | b'[' | b']')
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphology::HumanoidMorphology;

    fn subject() -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dexterous53,
            HumanoidTask::Reach,
            ActuationMode::NormalizedTorque,
            "perturbation-test-backend-v1",
        )
    }

    fn manifest(entries: Vec<HumanoidReachPerturbationEntry>) -> HumanoidReachPerturbationManifest {
        HumanoidReachPerturbationManifest::new(
            &subject(),
            "nominal-plus-noise-v1",
            "humanoid-qualification-runner",
            "1.0.0",
            HumanoidEvidenceDigest::from_bytes([9; 32]),
            "mujoco-world-bundle",
            "3.8.0",
            HumanoidEvidenceDigest::from_bytes([10; 32]),
            HumanoidReachPerturbationSeedScheme::TrialSeed64,
            entries,
        )
        .unwrap()
    }

    #[test]
    fn entry_order_does_not_change_manifest_digest() {
        let a = HumanoidReachPerturbationEntry {
            path: "dynamics.mass_scale".into(),
            value: HumanoidReachPerturbationValue::F64Range {
                minimum: 0.9,
                maximum: 1.1,
            },
        };
        let b = HumanoidReachPerturbationEntry {
            path: "sensor.hand_position.noise_std_m".into(),
            value: HumanoidReachPerturbationValue::F64(0.002),
        };
        assert_eq!(
            manifest(vec![a.clone(), b.clone()]).manifest_digest(),
            manifest(vec![b, a]).manifest_digest()
        );
    }

    #[test]
    fn set_order_does_not_change_manifest_digest() {
        let a = manifest(vec![HumanoidReachPerturbationEntry {
            path: "fault.mode".into(),
            value: HumanoidReachPerturbationValue::TextSet(vec![
                "encoder-delay".into(),
                "torque-derate".into(),
            ]),
        }]);
        let b = manifest(vec![HumanoidReachPerturbationEntry {
            path: "fault.mode".into(),
            value: HumanoidReachPerturbationValue::TextSet(vec![
                "torque-derate".into(),
                "encoder-delay".into(),
            ]),
        }]);
        assert_eq!(a.manifest_digest(), b.manifest_digest());
    }

    #[test]
    fn parameter_change_changes_manifest_digest() {
        let a = manifest(vec![HumanoidReachPerturbationEntry {
            path: "dynamics.mass_scale".into(),
            value: HumanoidReachPerturbationValue::F64Range {
                minimum: 0.9,
                maximum: 1.1,
            },
        }]);
        let b = manifest(vec![HumanoidReachPerturbationEntry {
            path: "dynamics.mass_scale".into(),
            value: HumanoidReachPerturbationValue::F64Range {
                minimum: 0.8,
                maximum: 1.2,
            },
        }]);
        assert_ne!(a.manifest_digest(), b.manifest_digest());
    }

    #[test]
    fn artifact_change_changes_manifest_digest() {
        let entries = vec![HumanoidReachPerturbationEntry {
            path: "environment.gravity_scale".into(),
            value: HumanoidReachPerturbationValue::F64Range {
                minimum: 0.99,
                maximum: 1.01,
            },
        }];
        let a = manifest(entries.clone());
        let b = HumanoidReachPerturbationManifest::new(
            &subject(),
            "nominal-plus-noise-v1",
            "humanoid-qualification-runner",
            "1.0.0",
            HumanoidEvidenceDigest::from_bytes([11; 32]),
            "mujoco-world-bundle",
            "3.8.0",
            HumanoidEvidenceDigest::from_bytes([10; 32]),
            HumanoidReachPerturbationSeedScheme::TrialSeed64,
            entries,
        )
        .unwrap();
        assert_ne!(a.manifest_digest(), b.manifest_digest());
    }

    #[test]
    fn legacy_binding_is_nonzero_but_not_the_trust_anchor() {
        let manifest = manifest(vec![HumanoidReachPerturbationEntry {
            path: "environment.gravity_scale".into(),
            value: HumanoidReachPerturbationValue::F64Range {
                minimum: 0.99,
                maximum: 1.01,
            },
        }]);
        assert_ne!(manifest.legacy_binding().configuration_fingerprint, 0);
        assert!(!manifest.manifest_digest().is_zero());
    }
}
