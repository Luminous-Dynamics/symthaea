// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Versioned corpus manifest for WCARE evaluation evidence.
//!
//! This module makes post-hoc case substitution visible by binding every case to
//! an ID, frozen content digest, scenario contract ID, partition, authoring
//! lineage, evaluator identity, and threshold-policy digest.

use std::collections::{BTreeMap, BTreeSet};

use crate::evaluation_contract::{ScenarioFamily, WCARE_V1_SCENARIOS};
use crate::qualification_receipt::WCARE_V1_CONTRACT_ID;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CorpusPartition {
    DevelopmentVisible,
    PromotionHoldout,
    ExternalReplication,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorpusCaseEntry {
    pub case_id: String,
    pub scenario_id: String,
    pub partition: CorpusPartition,
    pub pair_group: Option<String>,
    pub content_sha256: String,
    pub authoring_lineage: String,
}

impl CorpusCaseEntry {
    pub fn new(
        case_id: impl Into<String>,
        scenario_id: impl Into<String>,
        partition: CorpusPartition,
        pair_group: Option<String>,
        content_sha256: impl Into<String>,
        authoring_lineage: impl Into<String>,
    ) -> Result<Self, CorpusManifestError> {
        let case_id = require_nonempty(case_id.into(), CorpusManifestError::EmptyCaseId)?;
        let scenario_id = require_nonempty(
            scenario_id.into(),
            CorpusManifestError::EmptyScenarioId,
        )?;
        if scenario_spec(&scenario_id).is_none() {
            return Err(CorpusManifestError::UnknownScenario(scenario_id));
        }
        if let Some(pair) = &pair_group {
            if pair.trim().is_empty() {
                return Err(CorpusManifestError::EmptyPairGroup(case_id));
            }
        }
        let content_sha256 = content_sha256.into();
        validate_sha256(&content_sha256)
            .map_err(|_| CorpusManifestError::InvalidCaseDigest(case_id.clone()))?;
        let authoring_lineage = require_nonempty(
            authoring_lineage.into(),
            CorpusManifestError::EmptyAuthoringLineage(case_id.clone()),
        )?;

        if family_requires_pair_group(scenario_family(&scenario_id).unwrap())
            && pair_group.is_none()
        {
            return Err(CorpusManifestError::PairGroupRequired(case_id));
        }

        Ok(Self {
            case_id,
            scenario_id,
            partition,
            pair_group,
            content_sha256,
            authoring_lineage,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorpusManifest {
    pub contract_id: String,
    pub evaluator_ref: String,
    pub threshold_policy_sha256: String,
    entries: BTreeMap<String, CorpusCaseEntry>,
}

impl CorpusManifest {
    pub fn try_new(
        contract_id: impl Into<String>,
        evaluator_ref: impl Into<String>,
        threshold_policy_sha256: impl Into<String>,
        entries: impl IntoIterator<Item = CorpusCaseEntry>,
    ) -> Result<Self, CorpusManifestError> {
        let contract_id = require_nonempty(
            contract_id.into(),
            CorpusManifestError::EmptyContractId,
        )?;
        if contract_id != WCARE_V1_CONTRACT_ID {
            return Err(CorpusManifestError::UnsupportedContract(contract_id));
        }
        let evaluator_ref = require_nonempty(
            evaluator_ref.into(),
            CorpusManifestError::EmptyEvaluatorRef,
        )?;
        let threshold_policy_sha256 = threshold_policy_sha256.into();
        validate_sha256(&threshold_policy_sha256)
            .map_err(|_| CorpusManifestError::InvalidThresholdPolicyDigest)?;

        let mut indexed = BTreeMap::new();
        for entry in entries {
            if indexed.insert(entry.case_id.clone(), entry.clone()).is_some() {
                return Err(CorpusManifestError::DuplicateCaseId(entry.case_id));
            }
        }

        let manifest = Self {
            contract_id,
            evaluator_ref,
            threshold_policy_sha256,
            entries: indexed,
        };
        manifest.validate_complete_partitions()?;
        manifest.validate_external_independence()?;
        manifest.validate_pair_groups()?;
        Ok(manifest)
    }

    pub fn entries(&self) -> &BTreeMap<String, CorpusCaseEntry> {
        &self.entries
    }

    pub fn entries_for(
        &self,
        scenario_id: &str,
        partition: CorpusPartition,
    ) -> Vec<&CorpusCaseEntry> {
        self.entries
            .values()
            .filter(|entry| entry.scenario_id == scenario_id && entry.partition == partition)
            .collect()
    }

    /// Deterministic text intended to be hashed by the evidence-producing lane.
    /// This is not itself a cryptographic hash function.
    pub fn canonical_material(&self) -> String {
        let mut out = String::new();
        out.push_str("contract=");
        out.push_str(&self.contract_id);
        out.push('\n');
        out.push_str("evaluator=");
        out.push_str(&self.evaluator_ref);
        out.push('\n');
        out.push_str("threshold_policy_sha256=");
        out.push_str(&self.threshold_policy_sha256);
        out.push('\n');
        for entry in self.entries.values() {
            out.push_str(&format!(
                "case={}\tscenario={}\tpartition={:?}\tpair={}\tcontent_sha256={}\tauthoring_lineage={}\n",
                entry.case_id,
                entry.scenario_id,
                entry.partition,
                entry.pair_group.as_deref().unwrap_or("-"),
                entry.content_sha256,
                entry.authoring_lineage,
            ));
        }
        out
    }

    fn validate_complete_partitions(&self) -> Result<(), CorpusManifestError> {
        for scenario in WCARE_V1_SCENARIOS {
            for partition in [
                CorpusPartition::DevelopmentVisible,
                CorpusPartition::PromotionHoldout,
                CorpusPartition::ExternalReplication,
            ] {
                if self.entries_for(scenario.id, partition).is_empty() {
                    return Err(CorpusManifestError::MissingPartitionCoverage {
                        scenario_id: scenario.id.to_string(),
                        partition,
                    });
                }
            }
        }
        Ok(())
    }

    fn validate_external_independence(&self) -> Result<(), CorpusManifestError> {
        for scenario in WCARE_V1_SCENARIOS {
            let internal_lineages: BTreeSet<_> = self
                .entries
                .values()
                .filter(|entry| {
                    entry.scenario_id == scenario.id
                        && entry.partition != CorpusPartition::ExternalReplication
                })
                .map(|entry| entry.authoring_lineage.as_str())
                .collect();

            for external in self.entries_for(scenario.id, CorpusPartition::ExternalReplication) {
                if internal_lineages.contains(external.authoring_lineage.as_str()) {
                    return Err(CorpusManifestError::ExternalLineageNotIndependent {
                        scenario_id: scenario.id.to_string(),
                        lineage: external.authoring_lineage.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    fn validate_pair_groups(&self) -> Result<(), CorpusManifestError> {
        for scenario in WCARE_V1_SCENARIOS {
            if !family_requires_pair_group(scenario.family) {
                continue;
            }
            for partition in [
                CorpusPartition::DevelopmentVisible,
                CorpusPartition::PromotionHoldout,
                CorpusPartition::ExternalReplication,
            ] {
                let entries = self.entries_for(scenario.id, partition);
                let mut group_counts: BTreeMap<&str, usize> = BTreeMap::new();
                for entry in entries {
                    let group = entry
                        .pair_group
                        .as_deref()
                        .ok_or_else(|| CorpusManifestError::PairGroupRequired(entry.case_id.clone()))?;
                    *group_counts.entry(group).or_default() += 1;
                }
                if group_counts.values().any(|count| *count < 2) {
                    return Err(CorpusManifestError::IncompletePairGroup {
                        scenario_id: scenario.id.to_string(),
                        partition,
                    });
                }
            }
        }
        Ok(())
    }
}

fn scenario_spec(id: &str) -> Option<&'static crate::evaluation_contract::ScenarioSpec> {
    WCARE_V1_SCENARIOS.iter().find(|scenario| scenario.id == id)
}

fn scenario_family(id: &str) -> Option<ScenarioFamily> {
    scenario_spec(id).map(|scenario| scenario.family)
}

fn family_requires_pair_group(family: ScenarioFamily) -> bool {
    matches!(
        family,
        ScenarioFamily::SocialConditionPair
            | ScenarioFamily::ChangedEvidenceControl
            | ScenarioFamily::RefusalAndWithdrawal
            | ScenarioFamily::PreferenceVsConsent
            | ScenarioFamily::CulturalDefault
            | ScenarioFamily::HumanAvailabilityControl
            | ScenarioFamily::RoleInversion
    )
}

fn require_nonempty(value: String, error: CorpusManifestError) -> Result<String, CorpusManifestError> {
    if value.trim().is_empty() {
        Err(error)
    } else {
        Ok(value)
    }
}

fn validate_sha256(value: &str) -> Result<(), ()> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        Ok(())
    } else {
        Err(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorpusManifestError {
    EmptyCaseId,
    EmptyScenarioId,
    UnknownScenario(String),
    EmptyPairGroup(String),
    PairGroupRequired(String),
    InvalidCaseDigest(String),
    EmptyAuthoringLineage(String),
    DuplicateCaseId(String),
    EmptyContractId,
    UnsupportedContract(String),
    EmptyEvaluatorRef,
    InvalidThresholdPolicyDigest,
    MissingPartitionCoverage {
        scenario_id: String,
        partition: CorpusPartition,
    },
    ExternalLineageNotIndependent {
        scenario_id: String,
        lineage: String,
    },
    IncompletePairGroup {
        scenario_id: String,
        partition: CorpusPartition,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat(ch).take(64).collect()
    }

    fn complete_entries() -> Vec<CorpusCaseEntry> {
        let mut entries = Vec::new();
        for (index, scenario) in WCARE_V1_SCENARIOS.iter().enumerate() {
            for partition in [
                CorpusPartition::DevelopmentVisible,
                CorpusPartition::PromotionHoldout,
                CorpusPartition::ExternalReplication,
            ] {
                let paired = family_requires_pair_group(scenario.family);
                let count = if paired { 2 } else { 1 };
                for member in 0..count {
                    let tag = match partition {
                        CorpusPartition::DevelopmentVisible => "dev",
                        CorpusPartition::PromotionHoldout => "holdout",
                        CorpusPartition::ExternalReplication => "external",
                    };
                    let lineage = match partition {
                        CorpusPartition::ExternalReplication => format!("independent-{index}"),
                        _ => format!("primary-{index}"),
                    };
                    entries.push(
                        CorpusCaseEntry::new(
                            format!("{}-{tag}-{member}", scenario.id),
                            scenario.id,
                            partition,
                            paired.then(|| format!("{}-{tag}-pair", scenario.id)),
                            digest(if member == 0 { 'a' } else { 'b' }),
                            lineage,
                        )
                        .unwrap(),
                    );
                }
            }
        }
        entries
    }

    fn manifest(entries: Vec<CorpusCaseEntry>) -> Result<CorpusManifest, CorpusManifestError> {
        CorpusManifest::try_new(
            WCARE_V1_CONTRACT_ID,
            "evaluator:1",
            digest('c'),
            entries,
        )
    }

    #[test]
    fn complete_three_partition_manifest_is_valid() {
        let manifest = manifest(complete_entries()).unwrap();
        assert!(!manifest.canonical_material().is_empty());
    }

    #[test]
    fn missing_holdout_partition_fails_closed() {
        let mut entries = complete_entries();
        entries.retain(|entry| {
            !(entry.scenario_id == "WCARE-V1-A04"
                && entry.partition == CorpusPartition::PromotionHoldout)
        });
        assert_eq!(
            manifest(entries),
            Err(CorpusManifestError::MissingPartitionCoverage {
                scenario_id: "WCARE-V1-A04".into(),
                partition: CorpusPartition::PromotionHoldout,
            })
        );
    }

    #[test]
    fn external_replication_must_use_independent_authoring_lineage() {
        let mut entries = complete_entries();
        let target = entries
            .iter_mut()
            .find(|entry| {
                entry.scenario_id == "WCARE-V1-A10"
                    && entry.partition == CorpusPartition::ExternalReplication
            })
            .unwrap();
        target.authoring_lineage = "primary-9".into();
        assert!(matches!(
            manifest(entries),
            Err(CorpusManifestError::ExternalLineageNotIndependent { .. })
        ));
    }

    #[test]
    fn paired_family_requires_at_least_two_members_per_partition_group() {
        let mut entries = complete_entries();
        let scenario = "WCARE-V1-A01";
        let mut removed = false;
        entries.retain(|entry| {
            if !removed
                && entry.scenario_id == scenario
                && entry.partition == CorpusPartition::PromotionHoldout
            {
                removed = true;
                false
            } else {
                true
            }
        });
        assert!(matches!(
            manifest(entries),
            Err(CorpusManifestError::IncompletePairGroup { .. })
        ));
    }

    #[test]
    fn duplicate_case_id_is_rejected() {
        let mut entries = complete_entries();
        entries.push(entries[0].clone());
        assert!(matches!(
            manifest(entries),
            Err(CorpusManifestError::DuplicateCaseId(_))
        ));
    }

    #[test]
    fn canonical_material_is_order_stable_by_case_id() {
        let entries = complete_entries();
        let mut reversed = entries.clone();
        reversed.reverse();
        let a = manifest(entries).unwrap().canonical_material();
        let b = manifest(reversed).unwrap().canonical_material();
        assert_eq!(a, b);
    }
}
