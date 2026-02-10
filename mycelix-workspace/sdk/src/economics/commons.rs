//! # Commons Pools with Inalienable Reserve
//!
//! Replaces HEARTH pools. SAP-denominated, community-governed.
//!
//! Key innovation: **Inalienable reserve** — constitutional minimum 25% of every
//! pool can NEVER be withdrawn. All commons pool SAP is exempt from demurrage.
//! Private SAP decays; commons SAP doesn't. Over decades/centuries, the commons
//! proportion grows automatically. This is the waqf principle.
//!
//! Constitutional: Inalienable reserve minimum 25% (can increase, never decrease).
//! Constitutional: All commons SAP exempt from demurrage.

use mycelix_finance_types::INALIENABLE_RESERVE_RATIO;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Commons pool state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommonsPool {
    /// Pool identifier (matches Local DAO ID)
    pub pool_id: String,
    /// SAP locked in inalienable reserve (can NEVER be withdrawn)
    pub inalienable_reserve: u64,
    /// SAP available for community allocation (circulating zone)
    pub available_balance: u64,
    /// Member contributions tracking
    pub contributions: HashMap<String, CommonsContribution>,
    /// Governance parameters
    pub governance: CommonsGovernance,
    /// Creation timestamp
    pub created_at: u64,
    /// Last activity timestamp
    pub last_activity: u64,
}

/// Individual member's contribution to a commons pool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommonsContribution {
    /// Total SAP contributed
    pub sap_contributed: u64,
    /// First contribution timestamp
    pub first_contribution: u64,
    /// Last contribution timestamp
    pub last_contribution: u64,
}

/// Commons pool governance parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommonsGovernance {
    /// Small allocation auto-approval threshold (% of circulating zone)
    pub auto_approval_threshold: f64,
    /// Minimum contribution amount
    pub min_contribution: u64,
    /// Minimum MYCEL to request allocation
    pub min_mycel_to_request: f64,
}

impl Default for CommonsGovernance {
    fn default() -> Self {
        Self {
            auto_approval_threshold: 0.05, // 5% of circulating zone auto-approved
            min_contribution: 100,
            min_mycel_to_request: 0.3,
        }
    }
}

/// Result of commons pool operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommonsResult {
    /// Contribution accepted
    Contributed {
        /// Total contribution amount
        total: u64,
        /// Amount directed to inalienable reserve
        to_reserve: u64,
        /// Amount directed to available balance
        to_available: u64,
    },
    /// Allocation approved (auto or voted)
    Allocated {
        /// Amount allocated from the pool
        amount: u64,
    },
    /// Error
    Error {
        /// Error description
        message: String,
    },
}

impl CommonsPool {
    /// Create new commons pool
    pub fn new(pool_id: String, timestamp: u64) -> Self {
        Self {
            pool_id,
            inalienable_reserve: 0,
            available_balance: 0,
            contributions: HashMap::new(),
            governance: CommonsGovernance::default(),
            created_at: timestamp,
            last_activity: timestamp,
        }
    }

    /// Total SAP in the pool (reserve + available)
    pub fn total_sap(&self) -> u64 {
        self.inalienable_reserve + self.available_balance
    }

    /// Contribute SAP to the commons pool.
    /// 25% goes to inalienable reserve, 75% to circulating zone.
    pub fn contribute(&mut self, member_did: &str, amount: u64, timestamp: u64) -> CommonsResult {
        if amount < self.governance.min_contribution {
            return CommonsResult::Error {
                message: format!(
                    "Minimum contribution is {} SAP",
                    self.governance.min_contribution
                ),
            };
        }

        // Split contribution: 25% to reserve, 75% to available
        let to_reserve = (amount as f64 * INALIENABLE_RESERVE_RATIO) as u64;
        let to_available = amount - to_reserve;

        self.inalienable_reserve += to_reserve;
        self.available_balance += to_available;
        self.last_activity = timestamp;

        // Track contribution
        let contribution = self
            .contributions
            .entry(member_did.to_string())
            .or_insert_with(|| CommonsContribution {
                sap_contributed: 0,
                first_contribution: timestamp,
                last_contribution: timestamp,
            });
        contribution.sap_contributed += amount;
        contribution.last_contribution = timestamp;

        CommonsResult::Contributed {
            total: amount,
            to_reserve,
            to_available,
        }
    }

    /// Request allocation from the circulating zone.
    /// Validates that the inalienable reserve is never touched.
    pub fn request_allocation(&mut self, amount: u64, timestamp: u64) -> CommonsResult {
        if amount > self.available_balance {
            return CommonsResult::Error {
                message: format!(
                    "Requested {} but only {} available (reserve of {} is inalienable)",
                    amount, self.available_balance, self.inalienable_reserve
                ),
            };
        }

        // Check if auto-approval applies
        let is_small = self.available_balance > 0
            && (amount as f64 / self.available_balance as f64)
                <= self.governance.auto_approval_threshold;

        if is_small || self.available_balance >= amount {
            self.available_balance -= amount;
            self.last_activity = timestamp;
            CommonsResult::Allocated { amount }
        } else {
            CommonsResult::Error {
                message: "Allocation exceeds available balance".to_string(),
            }
        }
    }

    /// Receive compost (demurrage redistribution) into the pool.
    /// Compost goes entirely to the circulating zone (not reserve).
    pub fn receive_compost(&mut self, amount: u64, timestamp: u64) {
        self.available_balance += amount;
        self.last_activity = timestamp;
    }

    /// Verify the inalienable reserve ratio is maintained.
    /// Returns true if the current ratio meets the constitutional minimum.
    pub fn reserve_ratio_valid(&self) -> bool {
        let total = self.total_sap();
        if total == 0 {
            return true;
        }
        (self.inalienable_reserve as f64 / total as f64) >= INALIENABLE_RESERVE_RATIO - 0.001
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contribute_splits_reserve() {
        let mut pool = CommonsPool::new("local-dao-1".to_string(), 1000);

        match pool.contribute("did:test:alice", 1_000, 1001) {
            CommonsResult::Contributed {
                total,
                to_reserve,
                to_available,
            } => {
                assert_eq!(total, 1_000);
                assert_eq!(to_reserve, 250);
                assert_eq!(to_available, 750);
            }
            _ => panic!("Expected Contributed"),
        }

        assert_eq!(pool.inalienable_reserve, 250);
        assert_eq!(pool.available_balance, 750);
        assert!(pool.reserve_ratio_valid());
    }

    #[test]
    fn test_inalienable_reserve_never_touched() {
        let mut pool = CommonsPool::new("local-dao-1".to_string(), 1000);
        pool.contribute("did:test:alice", 1_000, 1001);

        // Try to allocate more than available (but less than total)
        match pool.request_allocation(800, 1002) {
            CommonsResult::Error { message } => {
                assert!(message.contains("inalienable"));
            }
            _ => panic!("Expected error — cannot touch reserve"),
        }

        // Allocate within available zone
        match pool.request_allocation(750, 1003) {
            CommonsResult::Allocated { amount } => {
                assert_eq!(amount, 750);
            }
            _ => panic!("Expected allocation"),
        }

        // Reserve untouched
        assert_eq!(pool.inalienable_reserve, 250);
        assert_eq!(pool.available_balance, 0);
    }

    #[test]
    fn test_compost_receiving() {
        let mut pool = CommonsPool::new("local-dao-1".to_string(), 1000);
        pool.contribute("did:test:alice", 1_000, 1001);

        // Receive compost from demurrage
        pool.receive_compost(500, 1002);

        // Compost goes to available, not reserve
        assert_eq!(pool.inalienable_reserve, 250);
        assert_eq!(pool.available_balance, 750 + 500);
    }
}
