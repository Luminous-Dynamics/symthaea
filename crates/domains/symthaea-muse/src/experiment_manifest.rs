// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Frozen manifests for the confirmatory Symthaea–Muse cognition study.
//!
//! The manifest is the authority for what may enter pilot and confirmatory
//! analysis. Related musical material is grouped by `family_id`; validation
//! rejects any family that crosses the pilot/confirmatory boundary.

use crate::cognitive_experiment::{CognitivePolicyArm, FrozenTrialKey};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const STUDY_MANIFEST_VERSION: &str = "symthaea-muse-cognition-study-v1";
pub const MIN_CONFIRMATORY_FIXTURES: usize = 24;
pub const MIN_PILOT_FIXTURES: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum StudySplit {
    Pilot,
    Confirmatory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ConfirmatoryEndpoint {
    ReturnRecognition,
    EarnedRecapitulation,
    Preference,
    KeepRate,
    LowerTimeToCommit,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenStudyFixture {
    pub key: FrozenTrialKey,
    /// Related themes, variants, or orchestrations share one family identifier.
    /// A family may occur in only one split.
    pub family_id: String,
    pub split: StudySplit,
    pub frozen_input_sha256: String,
    pub subject_sha256: String,
    pub renderer_sha256: String,
    pub soundfont_sha256: String,
    pub theory_constraints_sha256: String,
    pub tonic: String,
    pub meter: String,
    pub orchestration: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrozenStudyManifest {
    pub manifest_version: String,
    pub preregistration_sha256: String,
    pub analysis_plan_sha256: String,
    /// Commitment to the private randomization seed. The seed itself belongs in
    /// a separately protected codebook and must not appear in the public file.
    pub randomization_commitment_sha256: String,
    pub policy_versions: BTreeMap<CognitivePolicyArm, String>,
    pub primary_endpoints: Vec<ConfirmatoryEndpoint>,
    pub alpha: f64,
    pub fixtures: Vec<FrozenStudyFixture>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StudyManifestIssue {
    WrongManifestVersion { found: String },
    InvalidDigest { field: String },
    InvalidAlpha,
    MissingPolicyVersion { arm: CognitivePolicyArm },
    EmptyPrimaryEndpoints,
    DuplicatePrimaryEndpoint { endpoint: ConfirmatoryEndpoint },
    TooFewPilotFixtures { found: usize, required: usize },
    TooFewConfirmatoryFixtures { found: usize, required: usize },
    EmptyFixtureId { index: usize },
    EmptyFamilyId { key: FrozenTrialKey },
    EmptyStratumField { key: FrozenTrialKey, field: String },
    DuplicateFixtureKey { key: FrozenTrialKey },
    DuplicateFrozenInput { frozen_input_sha256: String },
    FamilyCrossesSplits { family_id: String },
    ConfirmatorySeedCollision { seed: u64 },
}

impl FrozenStudyManifest {
    pub fn validate(&self) -> Vec<StudyManifestIssue> {
        let mut issues = Vec::new();
        if self.manifest_version != STUDY_MANIFEST_VERSION {
            issues.push(StudyManifestIssue::WrongManifestVersion {
                found: self.manifest_version.clone(),
            });
        }
        for (field, digest) in [
            ("preregistration_sha256", &self.preregistration_sha256),
            ("analysis_plan_sha256", &self.analysis_plan_sha256),
            (
                "randomization_commitment_sha256",
                &self.randomization_commitment_sha256,
            ),
        ] {
            if !is_sha256(digest) {
                issues.push(StudyManifestIssue::InvalidDigest {
                    field: field.into(),
                });
            }
        }
        if !self.alpha.is_finite() || !(0.0..=0.10).contains(&self.alpha) || self.alpha == 0.0 {
            issues.push(StudyManifestIssue::InvalidAlpha);
        }
        for arm in CognitivePolicyArm::ALL {
            if self
                .policy_versions
                .get(&arm)
                .is_none_or(|value| value.trim().is_empty())
            {
                issues.push(StudyManifestIssue::MissingPolicyVersion { arm });
            }
        }
        if self.primary_endpoints.is_empty() {
            issues.push(StudyManifestIssue::EmptyPrimaryEndpoints);
        }
        let mut endpoints = BTreeSet::new();
        for endpoint in &self.primary_endpoints {
            if !endpoints.insert(*endpoint) {
                issues.push(StudyManifestIssue::DuplicatePrimaryEndpoint {
                    endpoint: *endpoint,
                });
            }
        }

        let pilot_count = self
            .fixtures
            .iter()
            .filter(|fixture| fixture.split == StudySplit::Pilot)
            .count();
        let confirmatory_count = self
            .fixtures
            .iter()
            .filter(|fixture| fixture.split == StudySplit::Confirmatory)
            .count();
        if pilot_count < MIN_PILOT_FIXTURES {
            issues.push(StudyManifestIssue::TooFewPilotFixtures {
                found: pilot_count,
                required: MIN_PILOT_FIXTURES,
            });
        }
        if confirmatory_count < MIN_CONFIRMATORY_FIXTURES {
            issues.push(StudyManifestIssue::TooFewConfirmatoryFixtures {
                found: confirmatory_count,
                required: MIN_CONFIRMATORY_FIXTURES,
            });
        }

        let mut keys = BTreeSet::new();
        let mut digests = BTreeSet::new();
        let mut family_splits: BTreeMap<&str, StudySplit> = BTreeMap::new();
        let mut confirmatory_seeds = BTreeSet::new();
        for (index, fixture) in self.fixtures.iter().enumerate() {
            if fixture.key.fixture_id.trim().is_empty() {
                issues.push(StudyManifestIssue::EmptyFixtureId { index });
            }
            if fixture.family_id.trim().is_empty() {
                issues.push(StudyManifestIssue::EmptyFamilyId {
                    key: fixture.key.clone(),
                });
            }
            for (field, value) in [
                ("tonic", fixture.tonic.as_str()),
                ("meter", fixture.meter.as_str()),
                ("orchestration", fixture.orchestration.as_str()),
            ] {
                if value.trim().is_empty() {
                    issues.push(StudyManifestIssue::EmptyStratumField {
                        key: fixture.key.clone(),
                        field: field.into(),
                    });
                }
            }
            if !keys.insert(fixture.key.clone()) {
                issues.push(StudyManifestIssue::DuplicateFixtureKey {
                    key: fixture.key.clone(),
                });
            }
            if !is_sha256(&fixture.frozen_input_sha256) {
                issues.push(StudyManifestIssue::InvalidDigest {
                    field: format!("fixture.{}.frozen_input_sha256", fixture.key.fixture_id),
                });
            } else if !digests.insert(fixture.frozen_input_sha256.clone()) {
                issues.push(StudyManifestIssue::DuplicateFrozenInput {
                    frozen_input_sha256: fixture.frozen_input_sha256.clone(),
                });
            }
            for (field, digest) in [
                ("subject_sha256", &fixture.subject_sha256),
                ("renderer_sha256", &fixture.renderer_sha256),
                ("soundfont_sha256", &fixture.soundfont_sha256),
                (
                    "theory_constraints_sha256",
                    &fixture.theory_constraints_sha256,
                ),
            ] {
                if !is_sha256(digest) {
                    issues.push(StudyManifestIssue::InvalidDigest {
                        field: format!("fixture.{}.{}", fixture.key.fixture_id, field),
                    });
                }
            }
            if let Some(previous) = family_splits.insert(&fixture.family_id, fixture.split) {
                if previous != fixture.split {
                    issues.push(StudyManifestIssue::FamilyCrossesSplits {
                        family_id: fixture.family_id.clone(),
                    });
                }
            }
            if fixture.split == StudySplit::Confirmatory
                && !confirmatory_seeds.insert(fixture.key.seed)
            {
                issues.push(StudyManifestIssue::ConfirmatorySeedCollision {
                    seed: fixture.key.seed,
                });
            }
        }
        issues
    }

    pub fn fixture(&self, key: &FrozenTrialKey) -> Option<&FrozenStudyFixture> {
        self.fixtures.iter().find(|fixture| &fixture.key == key)
    }

    pub fn confirmatory_fixture_count(&self) -> usize {
        self.fixtures
            .iter()
            .filter(|fixture| fixture.split == StudySplit::Confirmatory)
            .count()
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn fixture(index: usize, split: StudySplit) -> FrozenStudyFixture {
        FrozenStudyFixture {
            key: FrozenTrialKey {
                fixture_id: format!("fixture-{index}"),
                seed: index as u64 + 1,
            },
            family_id: format!("family-{index}"),
            split,
            frozen_input_sha256: format!("{index:064x}"),
            subject_sha256: DIGEST.into(),
            renderer_sha256: DIGEST.into(),
            soundfont_sha256: DIGEST.into(),
            theory_constraints_sha256: DIGEST.into(),
            tonic: "C".into(),
            meter: "4/4".into(),
            orchestration: "piano".into(),
        }
    }

    fn manifest() -> FrozenStudyManifest {
        let mut fixtures = Vec::new();
        for index in 1..=MIN_PILOT_FIXTURES {
            fixtures.push(fixture(index, StudySplit::Pilot));
        }
        for index in 100..100 + MIN_CONFIRMATORY_FIXTURES {
            fixtures.push(fixture(index, StudySplit::Confirmatory));
        }
        FrozenStudyManifest {
            manifest_version: STUDY_MANIFEST_VERSION.into(),
            preregistration_sha256: DIGEST.into(),
            analysis_plan_sha256: DIGEST.into(),
            randomization_commitment_sha256: DIGEST.into(),
            policy_versions: CognitivePolicyArm::ALL
                .into_iter()
                .map(|arm| (arm, "frozen-v1".into()))
                .collect(),
            primary_endpoints: vec![
                ConfirmatoryEndpoint::ReturnRecognition,
                ConfirmatoryEndpoint::Preference,
            ],
            alpha: 0.05,
            fixtures,
        }
    }

    #[test]
    fn valid_manifest_passes() {
        assert!(manifest().validate().is_empty());
    }

    #[test]
    fn related_material_cannot_cross_splits() {
        let mut manifest = manifest();
        manifest.fixtures[0].family_id = manifest.fixtures.last().unwrap().family_id.clone();
        assert!(
            manifest
                .validate()
                .iter()
                .any(|issue| matches!(issue, StudyManifestIssue::FamilyCrossesSplits { .. }))
        );
    }

    #[test]
    fn confirmatory_seed_reuse_is_rejected() {
        let mut manifest = manifest();
        let seed = manifest.fixtures[MIN_PILOT_FIXTURES].key.seed;
        manifest.fixtures[MIN_PILOT_FIXTURES + 1].key.seed = seed;
        assert!(
            manifest
                .validate()
                .iter()
                .any(|issue| matches!(issue, StudyManifestIssue::ConfirmatorySeedCollision { .. }))
        );
    }
}
