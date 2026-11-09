//! Static validation for paired A0/A2 benchmark campaign manifests.
//!
//! This module validates design metadata only. It does not generate secrets,
//! assign evaluator truth, run agents, or own the sealed benchmark service.

use crate::protocol::WorldFamily;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;

pub const CAMPAIGN_MANIFEST_VERSION: &str = "symthaea-open-world-campaign-v0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonDesign {
    /// Both access arms receive the same frozen intervention/evidence trace.
    MatchedTrace,
    /// Both arms share starting evidence, legal actions and maximum budgets;
    /// their subsequent adaptive actions may differ.
    AdaptiveBudget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccessArm {
    A0,
    A2,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CampaignManifest {
    pub schema_version: String,
    pub campaign_id: String,
    pub benchmark_revision: String,
    pub source_commit_sha: String,
    pub agent_artifact_sha256: String,
    pub policy_config_sha256: String,
    pub task_contract_sha256: String,
    /// Digest of evaluator-held seed ledger. The secret ledger itself must not
    /// be committed with this public campaign plan.
    pub seed_ledger_sha256: String,
    /// Digest of evaluator-held confirmation split ledger.
    pub confirmation_ledger_sha256: String,
    pub pairs: Vec<CampaignPair>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CampaignPair {
    pub pair_id: String,
    /// Opaque world alias; do not encode the true mechanism or seed here.
    pub world_alias: String,
    pub world_family: WorldFamily,
    /// v0 compares exactly these two arms, sharing the agent artifact,
    /// policy configuration, initial evidence and action/compute budgets.
    pub access_arms: [AccessArm; 2],
    pub design: ComparisonDesign,
    /// Digest of evidence shared by A0 and A2 before the structure-access
    /// treatment is applied.
    pub shared_initial_evidence_sha256: String,
    /// Required for MatchedTrace, absent for AdaptiveBudget.
    pub intervention_trace_sha256: Option<String>,
    pub action_budget: usize,
    pub compute_budget_units: u64,
}

impl CampaignManifest {
    pub fn validate(&self) -> Result<(), CampaignError> {
        if self.schema_version != CAMPAIGN_MANIFEST_VERSION
            || self.campaign_id.trim().is_empty()
            || self.benchmark_revision.trim().is_empty()
            || self.pairs.is_empty()
        {
            return Err(CampaignError::InvalidHeader);
        }

        validate_git_sha(&self.source_commit_sha)?;
        validate_digest(&self.agent_artifact_sha256)?;
        validate_digest(&self.policy_config_sha256)?;
        validate_digest(&self.task_contract_sha256)?;
        validate_digest(&self.seed_ledger_sha256)?;
        validate_digest(&self.confirmation_ledger_sha256)?;

        let mut pair_ids = BTreeSet::new();
        for pair in &self.pairs {
            if pair.pair_id.trim().is_empty()
                || pair.world_alias.trim().is_empty()
                || !pair_ids.insert(pair.pair_id.as_str())
                || pair.access_arms != [AccessArm::A0, AccessArm::A2]
                || pair.action_budget == 0
                || pair.compute_budget_units == 0
            {
                return Err(CampaignError::InvalidPair);
            }
            validate_digest(&pair.shared_initial_evidence_sha256)?;

            match (pair.design, &pair.intervention_trace_sha256) {
                (ComparisonDesign::MatchedTrace, Some(digest)) => validate_digest(digest)?,
                (ComparisonDesign::MatchedTrace, None) => {
                    return Err(CampaignError::MissingMatchedTrace);
                }
                (ComparisonDesign::AdaptiveBudget, None) => {}
                (ComparisonDesign::AdaptiveBudget, Some(_)) => {
                    return Err(CampaignError::UnexpectedFixedTrace);
                }
            }
        }
        Ok(())
    }
}

fn validate_git_sha(value: &str) -> Result<(), CampaignError> {
    if value.len() != 40 || !value.bytes().all(is_lower_hex) {
        return Err(CampaignError::InvalidDigest);
    }
    Ok(())
}

fn validate_digest(value: &str) -> Result<(), CampaignError> {
    let hex = value
        .strip_prefix("sha256:")
        .ok_or(CampaignError::InvalidDigest)?;
    if hex.len() != 64 || !hex.bytes().all(is_lower_hex) {
        return Err(CampaignError::InvalidDigest);
    }
    Ok(())
}

fn is_lower_hex(byte: u8) -> bool {
    byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CampaignError {
    InvalidHeader,
    InvalidPair,
    InvalidDigest,
    MissingMatchedTrace,
    UnexpectedFixedTrace,
}

impl fmt::Display for CampaignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for CampaignError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        format!("sha256:{}", byte.to_string().repeat(64))
    }

    fn valid_manifest() -> CampaignManifest {
        CampaignManifest {
            schema_version: CAMPAIGN_MANIFEST_VERSION.to_owned(),
            campaign_id: "pilot-v0".to_owned(),
            benchmark_revision: "open-world-ansatz-v1".to_owned(),
            source_commit_sha: "a".repeat(40),
            agent_artifact_sha256: digest('b'),
            policy_config_sha256: digest('c'),
            task_contract_sha256: digest('d'),
            seed_ledger_sha256: digest('e'),
            confirmation_ledger_sha256: digest('f'),
            pairs: vec![CampaignPair {
                pair_id: "pair-0001".to_owned(),
                world_alias: "world-0001".to_owned(),
                world_family: WorldFamily::NumericMechanism,
                access_arms: [AccessArm::A0, AccessArm::A2],
                design: ComparisonDesign::MatchedTrace,
                shared_initial_evidence_sha256: digest('1'),
                intervention_trace_sha256: Some(digest('2')),
                action_budget: 8,
                compute_budget_units: 100,
            }],
        }
    }

    #[test]
    fn accepts_matched_trace_pair_with_shared_manifest_fields() {
        assert!(valid_manifest().validate().is_ok());
    }

    #[test]
    fn rejects_matched_trace_pair_without_trace_digest() {
        let mut manifest = valid_manifest();
        manifest.pairs[0].intervention_trace_sha256 = None;
        assert_eq!(manifest.validate(), Err(CampaignError::MissingMatchedTrace));
    }

    #[test]
    fn rejects_fixed_trace_on_adaptive_budget_pair() {
        let mut manifest = valid_manifest();
        manifest.pairs[0].design = ComparisonDesign::AdaptiveBudget;
        assert_eq!(
            manifest.validate(),
            Err(CampaignError::UnexpectedFixedTrace)
        );
    }

    #[test]
    fn rejects_non_a0_a2_arm_combinations() {
        let mut manifest = valid_manifest();
        manifest.pairs[0].access_arms = [AccessArm::A2, AccessArm::A2];
        assert_eq!(manifest.validate(), Err(CampaignError::InvalidPair));
    }

    #[test]
    fn rejects_duplicate_pair_id() {
        let mut manifest = valid_manifest();
        let duplicate = manifest.pairs[0].clone();
        manifest.pairs.push(duplicate);
        assert_eq!(manifest.validate(), Err(CampaignError::InvalidPair));
    }

    #[test]
    fn allows_repeated_world_alias_across_distinct_pairs() {
        let mut manifest = valid_manifest();
        let mut second_pair = manifest.pairs[0].clone();
        second_pair.pair_id = "pair-0002".to_owned();
        second_pair.shared_initial_evidence_sha256 = digest('3');
        second_pair.intervention_trace_sha256 = Some(digest('4'));
        manifest.pairs.push(second_pair);
        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn rejects_zero_budget_and_malformed_digests() {
        let mut manifest = valid_manifest();
        manifest.pairs[0].action_budget = 0;
        assert_eq!(manifest.validate(), Err(CampaignError::InvalidPair));

        let mut manifest = valid_manifest();
        manifest.seed_ledger_sha256 = "not-a-digest".to_owned();
        assert_eq!(manifest.validate(), Err(CampaignError::InvalidDigest));
    }

    #[test]
    fn public_manifest_does_not_contain_truth_or_seed_fields() {
        let serialized = serde_json::to_string(&valid_manifest()).unwrap();
        assert!(!serialized.contains(r#""mechanism""#));
        assert!(!serialized.contains("ground_truth"));
        assert!(!serialized.contains("world_seed"));
        assert!(!serialized.contains("true_model"));
        assert!(serialized.contains("numeric_mechanism"));
        assert!(serialized.contains(r#""seed_ledger_sha256""#));
    }
}
