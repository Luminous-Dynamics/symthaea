// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Agent Constraints
//!
//! Operational limits and constraint enforcement for AI agents.

use super::InstrumentalActor;
use serde::{Deserialize, Serialize};

#[cfg(feature = "ts-export")]
use ts_rs::TS;

/// Agent classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum AgentClass {
    /// Fully automated, no human-in-loop
    Autonomous,
    /// Human approval required for high-value actions
    Supervised,
    /// Human initiates, agent executes
    Assistive,
    /// Read-only participation
    Observer,
}

impl AgentClass {
    /// Get class-specific limits
    pub fn limits(&self) -> ClassLimits {
        match self {
            AgentClass::Autonomous => ClassLimits {
                max_kredit_per_epoch: 5_000,
                max_tx_per_hour: 100,
                max_tx_size_sap: 1_000,
                requires_approval_above: None,
                api_rate_per_minute: 60,
            },
            AgentClass::Supervised => ClassLimits {
                max_kredit_per_epoch: 10_000,
                max_tx_per_hour: 500,
                max_tx_size_sap: 5_000,
                requires_approval_above: Some(1_000),
                api_rate_per_minute: 300,
            },
            AgentClass::Assistive => ClassLimits {
                max_kredit_per_epoch: 15_000,
                max_tx_per_hour: 1_000,
                max_tx_size_sap: 10_000,
                requires_approval_above: None,
                api_rate_per_minute: 600,
            },
            AgentClass::Observer => ClassLimits {
                max_kredit_per_epoch: 500,
                max_tx_per_hour: 10,
                max_tx_size_sap: 0,
                requires_approval_above: Some(0),
                api_rate_per_minute: 60,
            },
        }
    }
}

/// Class-specific operational limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassLimits {
    /// Maximum KREDIT per 30-day epoch
    pub max_kredit_per_epoch: u64,
    /// Maximum transactions per hour
    pub max_tx_per_hour: u32,
    /// Maximum transaction size in SAP
    pub max_tx_size_sap: u64,
    /// SAP threshold requiring sponsor approval (None = never required)
    pub requires_approval_above: Option<u64>,
    /// API calls per minute
    pub api_rate_per_minute: u32,
}

/// Agent operational constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConstraints {
    /// Can vote on governance
    pub can_vote_governance: bool,
    /// Can become validator
    pub can_become_validator: bool,
    /// Can govern HEARTH pools
    pub can_govern_hearth: bool,
    /// Can sponsor other agents
    pub can_sponsor_agents: bool,
    /// Can accumulate CIV
    pub can_hold_civ: bool,
    /// Can hold SAP directly
    pub can_hold_sap: bool,
    /// Can receive CGC
    pub can_receive_cgc: bool,
    /// Can send CGC
    pub can_send_cgc: bool,
    /// Custom action whitelist (None = all allowed)
    pub action_whitelist: Option<Vec<String>>,
    /// Action blacklist
    pub action_blacklist: Vec<String>,
}

impl Default for AgentConstraints {
    fn default() -> Self {
        Self {
            // Constitutional constraints
            can_vote_governance: false,
            can_become_validator: false,
            can_govern_hearth: false,
            can_sponsor_agents: false,
            can_hold_civ: false,
            can_hold_sap: false,
            can_receive_cgc: true,
            can_send_cgc: false,
            // No custom filters by default
            action_whitelist: None,
            action_blacklist: vec![],
        }
    }
}

impl AgentConstraints {
    /// Merge with constitutional constraints
    pub fn merge_constitutional(&mut self) {
        // Constitutional constraints are immutable - enforce them
        self.can_vote_governance = false;
        self.can_become_validator = false;
        self.can_govern_hearth = false;
        self.can_sponsor_agents = false;
        self.can_hold_civ = false;
        self.can_hold_sap = false;
    }

    /// Check if action is allowed
    pub fn is_action_allowed(&self, action_type: &str) -> bool {
        // Check blacklist first
        if self.action_blacklist.contains(&action_type.to_string()) {
            return false;
        }

        // Check whitelist if present
        if let Some(ref whitelist) = self.action_whitelist {
            return whitelist.contains(&action_type.to_string());
        }

        true
    }
}

/// Constraint violation types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstraintViolation {
    /// Constitutional constraint violated
    Constitutional(String),
    /// Class-specific limit exceeded
    ClassLimit(String),
    /// Rate limit exceeded
    RateLimit(String),
    /// Sponsor approval required
    ApprovalRequired,
    /// Action blacklisted
    Blacklisted(String),
    /// Action not whitelisted
    NotWhitelisted(String),
    /// Agent not operational
    NotOperational,
}

/// Action details for constraint checking
#[derive(Debug, Clone)]
pub struct AgentAction {
    /// Action type identifier
    pub action_type: String,
    /// SAP value involved
    pub sap_value: u64,
    /// KREDIT cost
    pub kredit_cost: u64,
    /// Has sponsor approval
    pub has_sponsor_approval: bool,
}

impl AgentAction {
    /// Check if this is a governance action
    pub fn is_governance_vote(&self) -> bool {
        self.action_type == "governance_vote" || self.action_type == "mip_vote"
    }
}

/// Enforce constraints on agent action
pub fn enforce_constraints(
    agent: &InstrumentalActor,
    action: &AgentAction,
) -> Result<(), ConstraintViolation> {
    // Check operational status
    if !agent.is_operational() {
        return Err(ConstraintViolation::NotOperational);
    }

    let constraints = &agent.constraints;
    let class_limits = agent.agent_class.limits();

    // Constitutional constraints
    if action.is_governance_vote() && !constraints.can_vote_governance {
        return Err(ConstraintViolation::Constitutional(
            "Agents cannot vote on governance".to_string(),
        ));
    }

    // Blacklist check
    if constraints.action_blacklist.contains(&action.action_type) {
        return Err(ConstraintViolation::Blacklisted(action.action_type.clone()));
    }

    // Whitelist check
    if let Some(ref whitelist) = constraints.action_whitelist {
        if !whitelist.contains(&action.action_type) {
            return Err(ConstraintViolation::NotWhitelisted(
                action.action_type.clone(),
            ));
        }
    }

    // Class limits
    if action.kredit_cost > class_limits.max_kredit_per_epoch {
        return Err(ConstraintViolation::ClassLimit(
            "KREDIT per action exceeds class limit".to_string(),
        ));
    }

    if action.sap_value > class_limits.max_tx_size_sap {
        return Err(ConstraintViolation::ClassLimit(
            "Transaction size exceeds class limit".to_string(),
        ));
    }

    // Rate limit
    if agent.actions_this_hour >= class_limits.max_tx_per_hour {
        return Err(ConstraintViolation::RateLimit(
            "Hourly transaction limit exceeded".to_string(),
        ));
    }

    // Approval requirement
    if let Some(threshold) = class_limits.requires_approval_above {
        if action.sap_value > threshold && !action.has_sponsor_approval {
            return Err(ConstraintViolation::ApprovalRequired);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::{AgentStatus, InstrumentalActor};
    use super::*;
    use crate::agentic::UncertaintyCalibration;
    use crate::matl::KVector;

    fn test_agent(class: AgentClass) -> InstrumentalActor {
        InstrumentalActor {
            agent_id: super::super::AgentId::generate(),
            sponsor_did: "did:test:sponsor".to_string(),
            agent_class: class,
            kredit_balance: 5000,
            kredit_cap: 10000,
            constraints: AgentConstraints::default(),
            behavior_log: vec![],
            status: AgentStatus::Active,
            created_at: 0,
            last_activity: 0,
            actions_this_hour: 0,
            k_vector: KVector::new_participant(),
            epistemic_stats: super::super::EpistemicStats::default(),
            output_history: vec![],
            uncertainty_calibration: UncertaintyCalibration::default(),
            pending_escalations: vec![],
        }
    }

    #[test]
    fn test_constitutional_constraints() {
        let agent = test_agent(AgentClass::Supervised);
        let action = AgentAction {
            action_type: "governance_vote".to_string(),
            sap_value: 0,
            kredit_cost: 10,
            has_sponsor_approval: true,
        };

        let result = enforce_constraints(&agent, &action);
        assert!(matches!(
            result,
            Err(ConstraintViolation::Constitutional(_))
        ));
    }

    #[test]
    fn test_approval_required() {
        let agent = test_agent(AgentClass::Supervised);
        let action = AgentAction {
            action_type: "transfer".to_string(),
            sap_value: 5000, // Above 1000 threshold
            kredit_cost: 500,
            has_sponsor_approval: false,
        };

        let result = enforce_constraints(&agent, &action);
        assert_eq!(result, Err(ConstraintViolation::ApprovalRequired));
    }

    #[test]
    fn test_action_allowed() {
        let agent = test_agent(AgentClass::Supervised);
        let action = AgentAction {
            action_type: "transfer".to_string(),
            sap_value: 500, // Below threshold
            kredit_cost: 50,
            has_sponsor_approval: false,
        };

        let result = enforce_constraints(&agent, &action);
        assert!(result.is_ok());
    }

    #[test]
    fn test_blacklist() {
        let mut agent = test_agent(AgentClass::Supervised);
        agent.constraints.action_blacklist = vec!["risky_action".to_string()];

        let action = AgentAction {
            action_type: "risky_action".to_string(),
            sap_value: 100,
            kredit_cost: 10,
            has_sponsor_approval: true,
        };

        let result = enforce_constraints(&agent, &action);
        assert!(matches!(result, Err(ConstraintViolation::Blacklisted(_))));
    }
}
