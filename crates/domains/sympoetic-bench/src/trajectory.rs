//! Versioned receipts for persistent human-AI team trajectories.
//!
//! This module verifies continuity and explicit configuration changes across
//! successive task cycles. It does not infer collective intelligence from a
//! trajectory; that requires qualified multi-task measurement and longitudinal
//! evidence beyond this structural contract.

use std::collections::HashSet;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::validate::sha256_hex;

const TRAJECTORY_SCHEMA: &str = "sympoetic-team-trajectory/0.1";
const CYCLE_SCHEMA: &str = "sympoetic-team-cycle/0.1";
const PARTICIPANT_TYPES: &[&str] = &["human", "agent"];
const MEMORY_POLICIES: &[&str] = &["none", "task_only", "team_opt_in_persistent"];
const MEMORY_CONSENT: &[&str] = &["not_applicable", "task_only", "explicit_opt_in"];
const CHANGE_TYPES: &[&str] = &[
    "member_added",
    "member_removed",
    "role_changed",
    "authority_changed",
    "memory_policy_changed",
    "memory_consent_changed",
    "model_snapshot_changed",
];

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TeamTrajectory {
    pub schema_version: String,
    pub team_id: String,
    pub cycles: Vec<TeamCycle>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TeamCycle {
    pub schema_version: String,
    pub cycle_index: u64,
    pub run_id: String,
    pub scenario_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prior_cycle_digest: Option<String>,
    pub team_state_before_digest: String,
    pub team_state_after_digest: String,
    pub artifact_digest: String,
    pub outcome_digest: String,
    pub memory_policy: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub consent_digest: Option<String>,
    pub members: Vec<TeamMember>,
    pub changes: Vec<TeamChange>,
    pub cycle_digest: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TeamMember {
    /// Pseudonymous principal identifier; never a real-world identity.
    pub principal_id: String,
    pub participant_type: String,
    pub role: String,
    pub authority_digest: String,
    pub memory_consent: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_snapshot: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TeamChange {
    pub change_id: String,
    pub change_type: String,
    pub actor_principal_id: String,
    pub subject_principal_id: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrajectoryValidation {
    pub status: &'static str,
    pub trajectory_sha256: String,
    pub team_id: String,
    pub cycle_count: usize,
    pub unique_run_count: usize,
    pub state_chain_verified: bool,
}

pub fn compute_cycle_digest(cycle: &TeamCycle) -> Result<String> {
    let mut value: Value = serde_json::to_value(cycle)?;
    let object = value
        .as_object_mut()
        .context("team cycle serialization did not produce an object")?;
    object.remove("cycle_digest");
    Ok(sha256_hex(&serde_json::to_vec(&value)?))
}

pub fn validate_trajectory(
    trajectory: &TeamTrajectory,
    original_bytes: &[u8],
) -> Result<TrajectoryValidation> {
    if trajectory.schema_version != TRAJECTORY_SCHEMA {
        bail!("unsupported team trajectory schema version");
    }
    require_nonempty("team_id", &trajectory.team_id)?;
    if trajectory.cycles.is_empty() {
        bail!("team trajectory must contain at least one cycle");
    }

    let mut run_ids = HashSet::new();
    let mut previous_cycle: Option<&TeamCycle> = None;
    let mut previous_members = Vec::new();

    for (position, cycle) in trajectory.cycles.iter().enumerate() {
        if cycle.schema_version != CYCLE_SCHEMA {
            bail!("cycle {position}: unsupported team cycle schema version");
        }
        if cycle.cycle_index != position as u64 {
            bail!(
                "cycle index must be contiguous from zero; expected {}, got {}",
                position,
                cycle.cycle_index
            );
        }
        require_nonempty("run_id", &cycle.run_id)?;
        require_nonempty("scenario_id", &cycle.scenario_id)?;
        if !valid_scenario_id(&cycle.scenario_id) {
            bail!("cycle {} has invalid scenario_id {:?}", cycle.cycle_index, cycle.scenario_id);
        }
        if !run_ids.insert(cycle.run_id.as_str()) {
            bail!("duplicate run_id {:?} in team trajectory", cycle.run_id);
        }

        for (name, digest) in [
            ("team_state_before_digest", cycle.team_state_before_digest.as_str()),
            ("team_state_after_digest", cycle.team_state_after_digest.as_str()),
            ("artifact_digest", cycle.artifact_digest.as_str()),
            ("outcome_digest", cycle.outcome_digest.as_str()),
        ] {
            require_sha256(name, digest)?;
        }
        if let Some(digest) = cycle.consent_digest.as_deref() {
            require_sha256("consent_digest", digest)?;
        }
        require_member("memory_policy", &cycle.memory_policy, MEMORY_POLICIES)?;

        if cycle.members.is_empty() {
            bail!("cycle {} must contain at least one team member", cycle.cycle_index);
        }
        let mut member_ids = HashSet::new();
        for member in &cycle.members {
            require_nonempty("member principal_id", &member.principal_id)?;
            require_nonempty("member role", &member.role)?;
            require_member("participant_type", &member.participant_type, PARTICIPANT_TYPES)?;
            require_member("memory_consent", &member.memory_consent, MEMORY_CONSENT)?;
            require_sha256("member authority_digest", &member.authority_digest)?;
            if let Some(snapshot) = member.model_snapshot.as_deref() {
                require_nonempty("model_snapshot", snapshot)?;
            }
            if !member_ids.insert(member.principal_id.as_str()) {
                bail!(
                    "cycle {} contains duplicate principal {:?}",
                    cycle.cycle_index,
                    member.principal_id
                );
            }
        }

        if cycle.memory_policy == "team_opt_in_persistent" {
            if cycle.consent_digest.is_none() {
                bail!("persistent team memory requires a consent_digest");
            }
            if cycle
                .members
                .iter()
                .any(|member| member.memory_consent != "explicit_opt_in")
            {
                bail!("persistent team memory requires explicit opt-in from every current member");
            }
        }

        let mut change_ids = HashSet::new();
        for change in &cycle.changes {
            require_nonempty("change_id", &change.change_id)?;
            require_nonempty("change actor", &change.actor_principal_id)?;
            require_nonempty("change subject", &change.subject_principal_id)?;
            require_member("change_type", &change.change_type, CHANGE_TYPES)?;
            if !change_ids.insert(change.change_id.as_str()) {
                bail!("cycle {} contains duplicate change_id {:?}", cycle.cycle_index, change.change_id);
            }
            let actor_exists = cycle
                .members
                .iter()
                .any(|member| member.principal_id == change.actor_principal_id)
                || previous_members
                    .iter()
                    .any(|member: &TeamMember| member.principal_id == change.actor_principal_id);
            if !actor_exists {
                bail!(
                    "cycle {} change actor {:?} is not in the before/after roster",
                    cycle.cycle_index,
                    change.actor_principal_id
                );
            }
        }

        match previous_cycle {
            None => {
                if cycle.prior_cycle_digest.is_some() {
                    bail!("first team cycle must not reference a prior cycle digest");
                }
                if !cycle.changes.is_empty() {
                    bail!("first team cycle establishes the initial roster; it must not claim prior-cycle changes");
                }
            }
            Some(previous) => {
                let prior = cycle
                    .prior_cycle_digest
                    .as_deref()
                    .context("subsequent team cycle requires prior_cycle_digest")?;
                if prior != previous.cycle_digest {
                    bail!("cycle {} does not reference the preceding cycle receipt", cycle.cycle_index);
                }
                if cycle.team_state_before_digest != previous.team_state_after_digest {
                    bail!("cycle {} breaks the team-state continuity chain", cycle.cycle_index);
                }
                validate_membership_changes(previous, cycle)?;
            }
        }

        let expected_digest = compute_cycle_digest(cycle)?;
        require_sha256("cycle_digest", &cycle.cycle_digest)?;
        if expected_digest != cycle.cycle_digest {
            bail!("cycle {} cycle_digest does not match canonical cycle content", cycle.cycle_index);
        }

        previous_members = cycle.members.clone();
        previous_cycle = Some(cycle);
    }

    Ok(TrajectoryValidation {
        status: "pass",
        trajectory_sha256: sha256_hex(original_bytes),
        team_id: trajectory.team_id.clone(),
        cycle_count: trajectory.cycles.len(),
        unique_run_count: run_ids.len(),
        state_chain_verified: true,
    })
}

fn validate_membership_changes(previous: &TeamCycle, current: &TeamCycle) -> Result<()> {
    let required_change = |change_type: &str, principal_id: &str| {
        current.changes.iter().any(|change| {
            change.change_type == change_type && change.subject_principal_id == principal_id
        })
    };

    for before in &previous.members {
        match current
            .members
            .iter()
            .find(|after| after.principal_id == before.principal_id)
        {
            None => {
                if !required_change("member_removed", &before.principal_id) {
                    bail!(
                        "cycle {} removed member {:?} without member_removed event",
                        current.cycle_index,
                        before.principal_id
                    );
                }
            }
            Some(after) => {
                if before.participant_type != after.participant_type {
                    bail!(
                        "cycle {} changes principal {:?} participant_type; use a new principal_id",
                        current.cycle_index,
                        before.principal_id
                    );
                }
                if before.role != after.role && !required_change("role_changed", &before.principal_id) {
                    bail!(
                        "cycle {} changed role for {:?} without role_changed event",
                        current.cycle_index,
                        before.principal_id
                    );
                }
                if before.authority_digest != after.authority_digest
                    && !required_change("authority_changed", &before.principal_id)
                {
                    bail!(
                        "cycle {} changed authority for {:?} without authority_changed event",
                        current.cycle_index,
                        before.principal_id
                    );
                }
                if before.memory_consent != after.memory_consent
                    && !required_change("memory_consent_changed", &before.principal_id)
                {
                    bail!(
                        "cycle {} changed memory consent for {:?} without memory_consent_changed event",
                        current.cycle_index,
                        before.principal_id
                    );
                }
                if before.model_snapshot != after.model_snapshot
                    && !required_change("model_snapshot_changed", &before.principal_id)
                {
                    bail!(
                        "cycle {} changed model snapshot for {:?} without model_snapshot_changed event",
                        current.cycle_index,
                        before.principal_id
                    );
                }
            }
        }
    }

    for after in &current.members {
        if !previous
            .members
            .iter()
            .any(|before| before.principal_id == after.principal_id)
            && !required_change("member_added", &after.principal_id)
        {
            bail!(
                "cycle {} added member {:?} without member_added event",
                current.cycle_index,
                after.principal_id
            );
        }
    }

    if previous.memory_policy != current.memory_policy
        && !required_change("memory_policy_changed", &current.members[0].principal_id)
    {
        bail!("cycle {} changed memory policy without memory_policy_changed event", current.cycle_index);
    }

    for change in &current.changes {
        let before = previous
            .members
            .iter()
            .find(|member| member.principal_id == change.subject_principal_id);
        let after = current
            .members
            .iter()
            .find(|member| member.principal_id == change.subject_principal_id);
        let matches_transition = match change.change_type.as_str() {
            "member_added" => before.is_none() && after.is_some(),
            "member_removed" => before.is_some() && after.is_none(),
            "role_changed" => before.zip(after).is_some_and(|(a, b)| a.role != b.role),
            "authority_changed" => before
                .zip(after)
                .is_some_and(|(a, b)| a.authority_digest != b.authority_digest),
            "memory_policy_changed" => previous.memory_policy != current.memory_policy,
            "memory_consent_changed" => before
                .zip(after)
                .is_some_and(|(a, b)| a.memory_consent != b.memory_consent),
            "model_snapshot_changed" => before
                .zip(after)
                .is_some_and(|(a, b)| a.model_snapshot != b.model_snapshot),
            _ => false,
        };
        if !matches_transition {
            bail!(
                "cycle {} change {:?} does not correspond to an observed team configuration transition",
                current.cycle_index,
                change.change_id
            );
        }
    }

    Ok(())
}

fn valid_scenario_id(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 10
        && &bytes[0..3] == b"SP-"
        && bytes[3..6].iter().all(u8::is_ascii_uppercase)
        && bytes[6] == b'-'
        && bytes[7..10].iter().all(u8::is_ascii_digit)
}

fn require_nonempty(label: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        bail!("{label} must not be empty");
    }
    Ok(())
}

fn require_member(label: &str, value: &str, allowed: &[&str]) -> Result<()> {
    if !allowed.contains(&value) {
        bail!("{label} has unsupported value {value:?}");
    }
    Ok(())
}

fn require_sha256(label: &str, digest: &str) -> Result<()> {
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("{label} must be a lowercase 64-character SHA-256 digest");
    }
    Ok(())
}
