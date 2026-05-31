// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! # HEARTH - Liquid Commons Pools
//!
//! Implementation of MIP-E-002 Article IV: Liquid Commons Hearths
//!
//! HEARTH pools are community-governed liquid commons funds with
//! warming (deposit), tending (labor), and harvesting (withdrawal) mechanics.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// HEARTH pool state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hearth {
    /// Pool identifier (matches Local DAO ID)
    pub pool_id: String,
    /// Total SAP in the pool
    pub total_sap: u64,
    /// Total TEND credits issued
    pub total_tend: u64,
    /// Member contributions (DID -> contribution)
    pub contributions: HashMap<String, HearthContribution>,
    /// Pending harvest requests
    pub pending_harvests: Vec<HarvestRequest>,
    /// Pool governance parameters
    pub governance: HearthGovernance,
    /// Creation timestamp
    pub created_at: u64,
    /// Last activity timestamp
    pub last_activity: u64,
}

/// Individual member's contribution to HEARTH
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HearthContribution {
    /// SAP contributed (warming)
    pub sap_warmed: u64,
    /// TEND credits earned (tending)
    pub tend_earned: u64,
    /// TEND credits spent (harvesting)
    pub tend_spent: u64,
    /// First contribution timestamp
    pub first_contribution: u64,
    /// Last contribution timestamp
    pub last_contribution: u64,
}

impl HearthContribution {
    /// Calculate available TEND for harvesting
    pub fn available_tend(&self) -> u64 {
        self.tend_earned.saturating_sub(self.tend_spent)
    }

    /// Calculate share of pool
    pub fn pool_share(&self, total_sap: u64) -> f64 {
        if total_sap == 0 {
            0.0
        } else {
            self.sap_warmed as f64 / total_sap as f64
        }
    }
}

/// HEARTH governance parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HearthGovernance {
    /// Maximum contribution per member (% of pool)
    pub max_contribution_pct: f64,
    /// Harvest approval threshold (% of pool)
    pub harvest_approval_threshold: f64,
    /// Minimum warming amount
    pub min_warming_amount: u64,
    /// TEND earning rate (TEND per SAP per day)
    pub tend_rate: f64,
    /// Emergency unlock enabled
    pub emergency_unlock_enabled: bool,
}

impl Default for HearthGovernance {
    fn default() -> Self {
        Self {
            max_contribution_pct: 0.10,       // 10% cap per MIP-E-002
            harvest_approval_threshold: 0.05, // 5% requires approval
            min_warming_amount: 100,
            tend_rate: 0.001, // 0.1% TEND per SAP per day
            emergency_unlock_enabled: false,
        }
    }
}

/// Pool allocation for a member
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HearthPool {
    /// Pool ID
    pub pool_id: String,
    /// Member's total SAP in pool
    pub member_sap: u64,
    /// Member's earned TEND
    pub member_tend: u64,
    /// Pool total for share calculation
    pub pool_total: u64,
}

/// Warming (deposit) action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmingAction {
    /// Member DID
    pub member_did: String,
    /// Amount of SAP to warm
    pub amount: u64,
    /// Target HEARTH pool
    pub pool_id: String,
    /// Timestamp
    pub timestamp: u64,
}

/// Tending (labor contribution) action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TendingAction {
    /// Member DID
    pub member_did: String,
    /// Description of labor contributed
    pub description: String,
    /// TEND credits to award
    pub tend_amount: u64,
    /// Verified by (list of attestors)
    pub verified_by: Vec<String>,
    /// Pool ID
    pub pool_id: String,
    /// Timestamp
    pub timestamp: u64,
}

/// Harvest (withdrawal) request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarvestRequest {
    /// Request ID
    pub request_id: String,
    /// Member DID
    pub member_did: String,
    /// SAP amount requested
    pub sap_amount: u64,
    /// TEND to burn
    pub tend_burn: u64,
    /// Pool ID
    pub pool_id: String,
    /// Reason for harvest
    pub reason: String,
    /// Request timestamp
    pub requested_at: u64,
    /// Approval status
    pub status: HarvestStatus,
    /// Votes (DID -> approve/reject)
    pub votes: HashMap<String, bool>,
}

/// Status of a harvest request
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HarvestStatus {
    /// Awaiting quorum
    Pending,
    /// Approved by quorum
    Approved,
    /// Rejected by quorum
    Rejected,
    /// Completed (SAP transferred)
    Completed,
    /// Cancelled by requester
    Cancelled,
}

/// Result of HEARTH operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HearthResult {
    /// Warming successful
    Warmed {
        /// SAP added to the pool
        sap: u64,
        /// TEND reputation earned
        tend_earned: u64,
    },
    /// Tending recorded
    Tended {
        /// TEND reputation earned
        tend_earned: u64,
    },
    /// Harvest pending approval
    HarvestPending {
        /// ID of the harvest request
        request_id: String,
    },
    /// Harvest approved and completed
    Harvested {
        /// SAP withdrawn from pool
        sap: u64,
        /// TEND reputation consumed
        tend_burned: u64,
    },
    /// Error occurred
    Error {
        /// Error message
        message: String,
    },
}

impl Hearth {
    /// Create new HEARTH pool
    pub fn new(pool_id: String, timestamp: u64) -> Self {
        Self {
            pool_id,
            total_sap: 0,
            total_tend: 0,
            contributions: HashMap::new(),
            pending_harvests: Vec::new(),
            governance: HearthGovernance::default(),
            created_at: timestamp,
            last_activity: timestamp,
        }
    }

    /// Warm the HEARTH (deposit SAP)
    pub fn warm(&mut self, action: WarmingAction) -> HearthResult {
        // Check minimum
        if action.amount < self.governance.min_warming_amount {
            return HearthResult::Error {
                message: format!(
                    "Minimum warming amount is {}",
                    self.governance.min_warming_amount
                ),
            };
        }

        // Check contribution cap
        let contribution = self
            .contributions
            .entry(action.member_did.clone())
            .or_insert_with(|| HearthContribution {
                sap_warmed: 0,
                tend_earned: 0,
                tend_spent: 0,
                first_contribution: action.timestamp,
                last_contribution: action.timestamp,
            });

        let new_total = contribution.sap_warmed + action.amount;
        let pool_after = self.total_sap + action.amount;

        // Check contribution cap (skip for bootstrap when pool is empty or very small)
        if self.total_sap > 0 {
            let share_after = new_total as f64 / pool_after as f64;
            if share_after > self.governance.max_contribution_pct {
                return HearthResult::Error {
                    message: format!(
                        "Contribution would exceed {}% cap",
                        self.governance.max_contribution_pct * 100.0
                    ),
                };
            }
        }

        // Apply warming
        contribution.sap_warmed += action.amount;
        contribution.last_contribution = action.timestamp;

        // Calculate TEND earned from warming
        let tend_earned = (action.amount as f64 * self.governance.tend_rate * 30.0) as u64;
        contribution.tend_earned += tend_earned;

        self.total_sap += action.amount;
        self.total_tend += tend_earned;
        self.last_activity = action.timestamp;

        HearthResult::Warmed {
            sap: action.amount,
            tend_earned,
        }
    }

    /// Record tending (labor contribution)
    pub fn tend(&mut self, action: TendingAction) -> HearthResult {
        // Verify attestors exist
        if action.verified_by.is_empty() {
            return HearthResult::Error {
                message: "Tending must be verified by at least one member".to_string(),
            };
        }

        let contribution = self
            .contributions
            .entry(action.member_did.clone())
            .or_insert_with(|| HearthContribution {
                sap_warmed: 0,
                tend_earned: 0,
                tend_spent: 0,
                first_contribution: action.timestamp,
                last_contribution: action.timestamp,
            });

        contribution.tend_earned += action.tend_amount;
        contribution.last_contribution = action.timestamp;

        self.total_tend += action.tend_amount;
        self.last_activity = action.timestamp;

        HearthResult::Tended {
            tend_earned: action.tend_amount,
        }
    }

    /// Request harvest (withdrawal)
    pub fn request_harvest(&mut self, request: HarvestRequest) -> HearthResult {
        // Check member has contribution
        let contribution = match self.contributions.get(&request.member_did) {
            Some(c) => c,
            None => {
                return HearthResult::Error {
                    message: "No contribution found for member".to_string(),
                };
            }
        };

        // Check TEND availability
        if contribution.available_tend() < request.tend_burn {
            return HearthResult::Error {
                message: format!(
                    "Insufficient TEND: have {}, need {}",
                    contribution.available_tend(),
                    request.tend_burn
                ),
            };
        }

        // Check SAP availability in pool
        if self.total_sap < request.sap_amount {
            return HearthResult::Error {
                message: "Insufficient SAP in pool".to_string(),
            };
        }

        // Check if approval required (>5% of pool)
        let harvest_ratio = request.sap_amount as f64 / self.total_sap as f64;
        if harvest_ratio <= self.governance.harvest_approval_threshold {
            // Auto-approve small harvests
            return self.execute_harvest(&request);
        }

        // Queue for approval
        let request_id = request.request_id.clone();
        self.pending_harvests.push(request);

        HearthResult::HarvestPending { request_id }
    }

    /// Execute approved harvest
    fn execute_harvest(&mut self, request: &HarvestRequest) -> HearthResult {
        // Deduct from member contribution
        if let Some(contribution) = self.contributions.get_mut(&request.member_did) {
            contribution.tend_spent += request.tend_burn;
        }

        // Deduct from pool
        self.total_sap = self.total_sap.saturating_sub(request.sap_amount);
        self.total_tend = self.total_tend.saturating_sub(request.tend_burn);

        HearthResult::Harvested {
            sap: request.sap_amount,
            tend_burned: request.tend_burn,
        }
    }

    /// Vote on pending harvest
    pub fn vote_harvest(
        &mut self,
        request_id: &str,
        voter_did: &str,
        approve: bool,
    ) -> Result<HarvestStatus, String> {
        let request = self
            .pending_harvests
            .iter_mut()
            .find(|r| r.request_id == request_id)
            .ok_or("Harvest request not found")?;

        // Check voter is a contributor
        if !self.contributions.contains_key(voter_did) {
            return Err("Only contributors can vote".to_string());
        }

        request.votes.insert(voter_did.to_string(), approve);

        // Check if quorum reached (simple majority of contributors)
        let total_contributors = self.contributions.len();
        let total_votes = request.votes.len();
        let approvals = request.votes.values().filter(|&&v| v).count();

        if total_votes > total_contributors / 2 {
            if approvals > total_votes / 2 {
                request.status = HarvestStatus::Approved;
            } else {
                request.status = HarvestStatus::Rejected;
            }
        }

        Ok(request.status)
    }

    /// Get member's pool share
    pub fn member_share(&self, member_did: &str) -> Option<f64> {
        self.contributions
            .get(member_did)
            .map(|c| c.pool_share(self.total_sap))
    }

    /// Calculate available SAP for borrowing (reputation-collateralized lending)
    pub fn available_for_lending(&self) -> u64 {
        // 30% of pool available for lending
        (self.total_sap as f64 * 0.30) as u64
    }
}

/// Cross-HEARTH mutual aid bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HearthBridge {
    /// Source HEARTH ID
    pub source_pool: String,
    /// Target HEARTH ID
    pub target_pool: String,
    /// Maximum transfer amount
    pub max_transfer: u64,
    /// Activation threshold (Vitality Index)
    pub activation_threshold: f64,
    /// Currently active
    pub active: bool,
}

impl HearthBridge {
    /// Check if bridge should activate
    pub fn should_activate(&self, target_vitality: f64) -> bool {
        target_vitality < self.activation_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hearth_warming() {
        let mut hearth = Hearth::new("local-dao-1".to_string(), 1000);

        let action = WarmingAction {
            member_did: "did:example:alice".to_string(),
            amount: 1000,
            pool_id: "local-dao-1".to_string(),
            timestamp: 1001,
        };

        match hearth.warm(action) {
            HearthResult::Warmed { sap, tend_earned } => {
                assert_eq!(sap, 1000);
                assert!(tend_earned > 0);
            }
            _ => panic!("Expected Warmed result"),
        }

        assert_eq!(hearth.total_sap, 1000);
    }

    #[test]
    fn test_contribution_cap() {
        let mut hearth = Hearth::new("local-dao-1".to_string(), 1000);

        // First member warms 9000
        let action1 = WarmingAction {
            member_did: "did:example:alice".to_string(),
            amount: 9000,
            pool_id: "local-dao-1".to_string(),
            timestamp: 1001,
        };
        hearth.warm(action1);

        // Second member tries to warm 2000 (would be >10% after)
        let action2 = WarmingAction {
            member_did: "did:example:bob".to_string(),
            amount: 2000,
            pool_id: "local-dao-1".to_string(),
            timestamp: 1002,
        };

        // This should succeed as bob's contribution would be 2000/(9000+2000) = 18%
        // which is over 10%, so it should fail
        match hearth.warm(action2) {
            HearthResult::Error { .. } => {} // Expected
            _ => panic!("Expected error due to cap"),
        }
    }

    #[test]
    fn test_harvest_auto_approve() {
        let mut hearth = Hearth::new("local-dao-1".to_string(), 1000);

        // Warm first
        let warm = WarmingAction {
            member_did: "did:example:alice".to_string(),
            amount: 10000,
            pool_id: "local-dao-1".to_string(),
            timestamp: 1001,
        };
        hearth.warm(warm);

        // Small harvest (< 5%) should auto-approve
        let harvest = HarvestRequest {
            request_id: "harvest-1".to_string(),
            member_did: "did:example:alice".to_string(),
            sap_amount: 100, // 1% of pool
            tend_burn: 10,
            pool_id: "local-dao-1".to_string(),
            reason: "Emergency".to_string(),
            requested_at: 1002,
            status: HarvestStatus::Pending,
            votes: HashMap::new(),
        };

        match hearth.request_harvest(harvest) {
            HearthResult::Harvested { sap, .. } => {
                assert_eq!(sap, 100);
            }
            _ => panic!("Expected auto-approved harvest"),
        }
    }
}
