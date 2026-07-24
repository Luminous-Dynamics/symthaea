// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Balanced blinded presentation schedules for the cognition study.
//!
//! The public schedule contains anonymous presentation codes and artifact
//! digests only. The arm mapping lives in a separate private codebook.

use crate::cognitive_experiment::{CognitivePolicyArm, FrozenTrialKey};
use crate::evidence_digest::{canonical_json_sha256, sha256_hex};
use crate::experiment_manifest::{FrozenStudyManifest, StudySplit};
use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const BLINDED_SCHEDULE_VERSION: &str = "symthaea-muse-blinded-schedule-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArmArtifactBinding {
    pub key: FrozenTrialKey,
    pub arm: CognitivePolicyArm,
    pub audio_sha256: String,
    pub recipe_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindedPresentation {
    pub presentation_id: String,
    pub key: FrozenTrialKey,
    pub split: StudySplit,
    pub position: u8,
    pub anonymous_code: String,
    pub audio_sha256: String,
    pub recipe_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindedSchedule {
    pub schedule_version: String,
    pub manifest_sha256: String,
    pub randomization_commitment_sha256: String,
    pub presentations: Vec<BlindedPresentation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindingCodebookEntry {
    pub presentation_id: String,
    pub anonymous_code: String,
    pub key: FrozenTrialKey,
    pub arm: CognitivePolicyArm,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindingCodebook {
    pub schedule_version: String,
    pub manifest_sha256: String,
    pub randomization_commitment_sha256: String,
    pub entries: Vec<BlindingCodebookEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlindingIssue {
    InvalidManifest,
    InvalidManifestDigest,
    ManifestSerializationFailed,
    CommitmentMismatch,
    SecretCommitmentMismatch,
    MissingArtifact {
        key: FrozenTrialKey,
        arm: CognitivePolicyArm,
    },
    DuplicateArtifact {
        key: FrozenTrialKey,
        arm: CognitivePolicyArm,
    },
    UnknownArtifactFixture {
        key: FrozenTrialKey,
    },
    InvalidArtifactDigest {
        key: FrozenTrialKey,
        arm: CognitivePolicyArm,
        field: String,
    },
    WrongScheduleVersion {
        found: String,
    },
    DuplicatePresentationId {
        presentation_id: String,
    },
    DuplicateAnonymousCode {
        anonymous_code: String,
    },
    UnknownScheduleFixture {
        key: FrozenTrialKey,
    },
    ScheduleSplitMismatch {
        key: FrozenTrialKey,
    },
    InvalidPosition {
        presentation_id: String,
    },
    IncompleteFixtureSchedule {
        key: FrozenTrialKey,
        found: usize,
    },
    DuplicateFixturePosition {
        key: FrozenTrialKey,
        position: u8,
    },
    CodebookEntryMismatch {
        presentation_id: String,
    },
    MissingCodebookEntry {
        presentation_id: String,
    },
    DuplicateCodebookPresentation {
        presentation_id: String,
    },
    UnexpectedCodebookEntry {
        presentation_id: String,
    },
    DuplicateCodebookArm {
        key: FrozenTrialKey,
        arm: CognitivePolicyArm,
    },
}

/// Build a public schedule and private codebook.
///
/// `secret_seed` must be stored separately from the public schedule. The
/// manifest contains only its externally produced SHA-256 commitment. Random
/// generation here is deterministic for replay, not a cryptographic primitive.
pub fn build_blinded_schedule(
    manifest: &FrozenStudyManifest,
    artifacts: &[ArmArtifactBinding],
    secret_key: [u8; 32],
) -> Result<(BlindedSchedule, BlindingCodebook), Vec<BlindingIssue>> {
    let manifest_sha256 = match canonical_json_sha256(manifest) {
        Ok(value) => value,
        Err(_) => return Err(vec![BlindingIssue::ManifestSerializationFailed]),
    };
    let mut issues = Vec::new();
    if !manifest.validate().is_empty() {
        issues.push(BlindingIssue::InvalidManifest);
    }
    if !is_sha256(&manifest_sha256) {
        issues.push(BlindingIssue::InvalidManifestDigest);
    }
    if sha256_hex(&secret_key) != manifest.randomization_commitment_sha256 {
        issues.push(BlindingIssue::SecretCommitmentMismatch);
    }

    let mut artifact_map = BTreeMap::new();
    for artifact in artifacts {
        if manifest.fixture(&artifact.key).is_none() {
            issues.push(BlindingIssue::UnknownArtifactFixture {
                key: artifact.key.clone(),
            });
        }
        let identity = (artifact.key.clone(), artifact.arm);
        if artifact_map.insert(identity.clone(), artifact).is_some() {
            issues.push(BlindingIssue::DuplicateArtifact {
                key: identity.0,
                arm: identity.1,
            });
        }
        for (field, digest) in [
            ("audio_sha256", &artifact.audio_sha256),
            ("recipe_sha256", &artifact.recipe_sha256),
        ] {
            if !is_sha256(digest) {
                issues.push(BlindingIssue::InvalidArtifactDigest {
                    key: artifact.key.clone(),
                    arm: artifact.arm,
                    field: field.into(),
                });
            }
        }
    }
    for fixture in &manifest.fixtures {
        for arm in CognitivePolicyArm::ALL {
            if !artifact_map.contains_key(&(fixture.key.clone(), arm)) {
                issues.push(BlindingIssue::MissingArtifact {
                    key: fixture.key.clone(),
                    arm,
                });
            }
        }
    }
    if !issues.is_empty() {
        return Err(issues);
    }

    let mut rng = StdRng::from_seed(secret_key);
    let mut base = CognitivePolicyArm::ALL;
    for index in (1..base.len()).rev() {
        let swap_with = rng.gen_range(0..=index);
        base.swap(index, swap_with);
    }

    let mut fixtures: Vec<_> = manifest.fixtures.iter().collect();
    fixtures.sort_by(|left, right| left.key.cmp(&right.key));
    let mut presentations = Vec::with_capacity(fixtures.len() * 4);
    let mut entries = Vec::with_capacity(fixtures.len() * 4);
    let mut anonymous_codes = BTreeSet::new();

    for (fixture_index, fixture) in fixtures.into_iter().enumerate() {
        let rotation = fixture_index % base.len();
        for position in 0..4usize {
            let arm = base[(position + rotation) % base.len()];
            let artifact = artifact_map[&(fixture.key.clone(), arm)];
            let presentation_id = format!(
                "{}-{:016x}-p{}",
                fixture.key.fixture_id, fixture.key.seed, position
            );
            let anonymous_code = loop {
                let value = format!("M{:016X}{:016X}", rng.next_u64(), rng.next_u64());
                if anonymous_codes.insert(value.clone()) {
                    break value;
                }
            };
            presentations.push(BlindedPresentation {
                presentation_id: presentation_id.clone(),
                key: fixture.key.clone(),
                split: fixture.split,
                position: position as u8,
                anonymous_code: anonymous_code.clone(),
                audio_sha256: artifact.audio_sha256.clone(),
                recipe_sha256: artifact.recipe_sha256.clone(),
            });
            entries.push(BlindingCodebookEntry {
                presentation_id,
                anonymous_code,
                key: fixture.key.clone(),
                arm,
            });
        }
    }

    let schedule = BlindedSchedule {
        schedule_version: BLINDED_SCHEDULE_VERSION.into(),
        manifest_sha256: manifest_sha256.clone(),
        randomization_commitment_sha256: manifest.randomization_commitment_sha256.clone(),
        presentations,
    };
    let codebook = BlindingCodebook {
        schedule_version: BLINDED_SCHEDULE_VERSION.into(),
        manifest_sha256,
        randomization_commitment_sha256: manifest.randomization_commitment_sha256.clone(),
        entries,
    };
    let validation = validate_blinded_schedule(manifest, &schedule, Some(&codebook));
    if validation.is_empty() {
        Ok((schedule, codebook))
    } else {
        Err(validation)
    }
}

pub fn validate_blinded_schedule(
    manifest: &FrozenStudyManifest,
    schedule: &BlindedSchedule,
    codebook: Option<&BlindingCodebook>,
) -> Vec<BlindingIssue> {
    let mut issues = Vec::new();
    if schedule.schedule_version != BLINDED_SCHEDULE_VERSION {
        issues.push(BlindingIssue::WrongScheduleVersion {
            found: schedule.schedule_version.clone(),
        });
    }
    if !is_sha256(&schedule.manifest_sha256) {
        issues.push(BlindingIssue::InvalidManifestDigest);
    }
    if schedule.randomization_commitment_sha256 != manifest.randomization_commitment_sha256 {
        issues.push(BlindingIssue::CommitmentMismatch);
    }

    let mut ids = BTreeSet::new();
    let mut codes = BTreeSet::new();
    let mut by_fixture: BTreeMap<FrozenTrialKey, Vec<&BlindedPresentation>> = BTreeMap::new();
    for presentation in &schedule.presentations {
        if !ids.insert(presentation.presentation_id.clone()) {
            issues.push(BlindingIssue::DuplicatePresentationId {
                presentation_id: presentation.presentation_id.clone(),
            });
        }
        if !codes.insert(presentation.anonymous_code.clone()) {
            issues.push(BlindingIssue::DuplicateAnonymousCode {
                anonymous_code: presentation.anonymous_code.clone(),
            });
        }
        match manifest.fixture(&presentation.key) {
            None => issues.push(BlindingIssue::UnknownScheduleFixture {
                key: presentation.key.clone(),
            }),
            Some(fixture) if fixture.split != presentation.split => {
                issues.push(BlindingIssue::ScheduleSplitMismatch {
                    key: presentation.key.clone(),
                });
            }
            Some(_) => {}
        }
        if presentation.position > 3 {
            issues.push(BlindingIssue::InvalidPosition {
                presentation_id: presentation.presentation_id.clone(),
            });
        }
        for (field, digest) in [
            ("audio_sha256", &presentation.audio_sha256),
            ("recipe_sha256", &presentation.recipe_sha256),
        ] {
            if !is_sha256(digest) {
                issues.push(BlindingIssue::InvalidArtifactDigest {
                    key: presentation.key.clone(),
                    arm: CognitivePolicyArm::Fixed,
                    field: field.into(),
                });
            }
        }
        by_fixture
            .entry(presentation.key.clone())
            .or_default()
            .push(presentation);
    }
    for fixture in &manifest.fixtures {
        let presentations = by_fixture.get(&fixture.key).cloned().unwrap_or_default();
        if presentations.len() != 4 {
            issues.push(BlindingIssue::IncompleteFixtureSchedule {
                key: fixture.key.clone(),
                found: presentations.len(),
            });
        }
        let mut positions = BTreeSet::new();
        for presentation in presentations {
            if !positions.insert(presentation.position) {
                issues.push(BlindingIssue::DuplicateFixturePosition {
                    key: fixture.key.clone(),
                    position: presentation.position,
                });
            }
        }
    }

    if let Some(codebook) = codebook {
        if codebook.schedule_version != schedule.schedule_version
            || codebook.manifest_sha256 != schedule.manifest_sha256
            || codebook.randomization_commitment_sha256 != schedule.randomization_commitment_sha256
        {
            issues.push(BlindingIssue::CommitmentMismatch);
        }
        let schedule_ids: BTreeSet<_> = schedule
            .presentations
            .iter()
            .map(|presentation| presentation.presentation_id.as_str())
            .collect();
        let mut codebook_ids = BTreeSet::new();
        for entry in &codebook.entries {
            if !codebook_ids.insert(entry.presentation_id.as_str()) {
                issues.push(BlindingIssue::DuplicateCodebookPresentation {
                    presentation_id: entry.presentation_id.clone(),
                });
            }
            if !schedule_ids.contains(entry.presentation_id.as_str()) {
                issues.push(BlindingIssue::UnexpectedCodebookEntry {
                    presentation_id: entry.presentation_id.clone(),
                });
            }
        }
        let entries: BTreeMap<_, _> = codebook
            .entries
            .iter()
            .map(|entry| (entry.presentation_id.as_str(), entry))
            .collect();
        let mut fixture_arms = BTreeSet::new();
        for presentation in &schedule.presentations {
            match entries.get(presentation.presentation_id.as_str()) {
                None => issues.push(BlindingIssue::MissingCodebookEntry {
                    presentation_id: presentation.presentation_id.clone(),
                }),
                Some(entry)
                    if entry.anonymous_code != presentation.anonymous_code
                        || entry.key != presentation.key =>
                {
                    issues.push(BlindingIssue::CodebookEntryMismatch {
                        presentation_id: presentation.presentation_id.clone(),
                    });
                }
                Some(entry) => {
                    if !fixture_arms.insert((entry.key.clone(), entry.arm)) {
                        issues.push(BlindingIssue::DuplicateCodebookArm {
                            key: entry.key.clone(),
                            arm: entry.arm,
                        });
                    }
                }
            }
        }
    }
    issues
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_digest::sha256_hex;
    use crate::experiment_manifest::{
        ConfirmatoryEndpoint, FrozenStudyFixture, MIN_CONFIRMATORY_FIXTURES, MIN_PILOT_FIXTURES,
        STUDY_MANIFEST_VERSION,
    };

    const DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SECRET: [u8; 32] = [0x42; 32];

    fn manifest() -> FrozenStudyManifest {
        let mut fixtures = Vec::new();
        for index in 0..MIN_PILOT_FIXTURES + MIN_CONFIRMATORY_FIXTURES {
            let split = if index < MIN_PILOT_FIXTURES {
                StudySplit::Pilot
            } else {
                StudySplit::Confirmatory
            };
            fixtures.push(FrozenStudyFixture {
                key: FrozenTrialKey {
                    fixture_id: format!("fixture-{index:03}"),
                    seed: index as u64 + 1,
                },
                family_id: format!("family-{index}"),
                split,
                frozen_input_sha256: format!("{:064x}", index + 1),
                subject_sha256: DIGEST.into(),
                renderer_sha256: DIGEST.into(),
                soundfont_sha256: DIGEST.into(),
                theory_constraints_sha256: DIGEST.into(),
                tonic: "C".into(),
                meter: "4/4".into(),
                orchestration: "piano".into(),
            });
        }
        FrozenStudyManifest {
            manifest_version: STUDY_MANIFEST_VERSION.into(),
            preregistration_sha256: DIGEST.into(),
            analysis_plan_sha256: DIGEST.into(),
            randomization_commitment_sha256: sha256_hex(&SECRET),
            policy_versions: CognitivePolicyArm::ALL
                .into_iter()
                .map(|arm| (arm, "policy-v1".into()))
                .collect(),
            primary_endpoints: vec![ConfirmatoryEndpoint::Preference],
            alpha: 0.05,
            fixtures,
        }
    }

    fn artifacts(manifest: &FrozenStudyManifest) -> Vec<ArmArtifactBinding> {
        manifest
            .fixtures
            .iter()
            .flat_map(|fixture| {
                CognitivePolicyArm::ALL
                    .into_iter()
                    .map(move |arm| ArmArtifactBinding {
                        key: fixture.key.clone(),
                        arm,
                        audio_sha256: DIGEST.into(),
                        recipe_sha256: DIGEST.into(),
                    })
            })
            .collect()
    }

    #[test]
    fn public_schedule_and_private_codebook_validate() {
        let manifest = manifest();
        let (schedule, codebook) =
            build_blinded_schedule(&manifest, &artifacts(&manifest), SECRET).unwrap();
        assert!(validate_blinded_schedule(&manifest, &schedule, Some(&codebook)).is_empty());
        assert_eq!(schedule.presentations.len(), manifest.fixtures.len() * 4);
    }

    #[test]
    fn public_schedule_serialization_contains_no_arm_labels() {
        let manifest = manifest();
        let (schedule, _) =
            build_blinded_schedule(&manifest, &artifacts(&manifest), SECRET).unwrap();
        let json = serde_json::to_string(&schedule).unwrap();
        assert!(!json.contains("\"arm\""));
    }

    #[test]
    fn latin_rotation_balances_positions_in_complete_blocks() {
        let manifest = manifest();
        let (schedule, codebook) =
            build_blinded_schedule(&manifest, &artifacts(&manifest), SECRET).unwrap();
        let first_four: BTreeSet<_> = schedule
            .presentations
            .iter()
            .filter(|presentation| {
                manifest
                    .fixture(&presentation.key)
                    .is_some_and(|fixture| fixture.split == StudySplit::Pilot)
            })
            .take(16)
            .map(|presentation| {
                let arm = codebook
                    .entries
                    .iter()
                    .find(|entry| entry.presentation_id == presentation.presentation_id)
                    .unwrap()
                    .arm;
                (arm, presentation.position)
            })
            .collect();
        assert_eq!(first_four.len(), 16);
    }

    #[test]
    fn private_key_must_open_the_public_commitment() {
        let manifest = manifest();
        let issues =
            build_blinded_schedule(&manifest, &artifacts(&manifest), [0x99; 32]).unwrap_err();
        assert!(issues.contains(&BlindingIssue::SecretCommitmentMismatch));
    }

    #[test]
    fn missing_artifact_prevents_randomization() {
        let manifest = manifest();
        let mut artifacts = artifacts(&manifest);
        artifacts.pop();
        assert!(build_blinded_schedule(&manifest, &artifacts, SECRET).is_err());
    }
}
