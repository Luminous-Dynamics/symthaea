// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Minimal, mechanism-native Mutual Credit semantics.
//!
//! This module models a bounded credit-clearing network in which a trade
//! simultaneously creates a positive claim for the seller and a negative
//! obligation for the buyer. The aggregate network balance is conserved at
//! zero. It deliberately does not model commercial banking, interest,
//! collateral, equity, or Creditism settlement semantics.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq)]
pub enum MutualCreditError {
    UnknownMember(String),
    DuplicateMember(String),
    NonPositiveAmount,
    NonFiniteAmount,
    CreditLimitExceeded {
        member: String,
        balance: f64,
        requested_balance: f64,
        lower_limit: f64,
        upper_limit: f64,
    },
    SelfTrade,
    DefaultedMember(String),
    NonZeroBalance(String),
    InsufficientPositiveBalance {
        member: String,
        balance: f64,
        requested: f64,
    },
    ExternalSettlementWouldExceedLimit {
        member: String,
        balance: f64,
        requested_balance: f64,
        lower_limit: f64,
        upper_limit: f64,
    },
    NonFiniteResult,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemberLimit {
    /// Maximum negative balance the member may carry.
    pub debit_limit: f64,
    /// Maximum positive balance the member may carry.
    pub credit_limit: f64,
}

impl MemberLimit {
    pub fn new(debit_limit: f64, credit_limit: f64) -> Result<Self, MutualCreditError> {
        if !debit_limit.is_finite() || !credit_limit.is_finite() {
            return Err(MutualCreditError::NonFiniteAmount);
        }
        if debit_limit < 0.0 || credit_limit < 0.0 {
            return Err(MutualCreditError::NonPositiveAmount);
        }
        Ok(Self {
            debit_limit,
            credit_limit,
        })
    }

    fn lower(&self) -> f64 {
        -self.debit_limit
    }

    fn upper(&self) -> f64 {
        self.credit_limit
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MutualTrade {
    pub amount: f64,
    pub buyer_new_balance: f64,
    pub seller_new_balance: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MutualCreditNetwork {
    balances: BTreeMap<String, f64>,
    limits: BTreeMap<String, MemberLimit>,
    defaults: BTreeSet<String>,
}

impl MutualCreditNetwork {
    pub fn new() -> Self {
        Self {
            balances: BTreeMap::new(),
            limits: BTreeMap::new(),
            defaults: BTreeSet::new(),
        }
    }

    pub fn register_member(
        &mut self,
        member: impl Into<String>,
        limit: MemberLimit,
    ) -> Result<(), MutualCreditError> {
        let member = member.into();
        if member.is_empty() {
            return Err(MutualCreditError::UnknownMember(member));
        }
        if self.balances.contains_key(&member) {
            return Err(MutualCreditError::DuplicateMember(member));
        }
        self.balances.insert(member.clone(), 0.0);
        self.limits.insert(member, limit);
        Ok(())
    }

    pub fn balance(&self, member: &str) -> Result<f64, MutualCreditError> {
        self.balances
            .get(member)
            .copied()
            .ok_or_else(|| MutualCreditError::UnknownMember(member.to_owned()))
    }

    pub fn member_count(&self) -> usize {
        self.balances.len()
    }

    pub fn total_balance(&self) -> f64 {
        self.balances.values().sum()
    }

    pub fn mark_defaulted(&mut self, member: &str) -> Result<(), MutualCreditError> {
        self.require_member(member)?;
        self.defaults.insert(member.to_owned());
        Ok(())
    }

    pub fn is_defaulted(&self, member: &str) -> bool {
        self.defaults.contains(member)
    }

    /// Execute a trade inside the network.
    ///
    /// No external money is introduced. The buyer's negative position and
    /// seller's positive position are created atomically and preserve a
    /// zero-sum network balance.
    pub fn trade(
        &mut self,
        buyer: &str,
        seller: &str,
        amount: f64,
    ) -> Result<MutualTrade, MutualCreditError> {
        validate_amount(amount)?;
        if buyer == seller {
            return Err(MutualCreditError::SelfTrade);
        }
        self.require_active_member(buyer)?;
        self.require_active_member(seller)?;

        let buyer_limit = self.limits[buyer];
        let seller_limit = self.limits[seller];
        let buyer_balance = self.balances[buyer];
        let seller_balance = self.balances[seller];

        let buyer_new_balance = buyer_balance - amount;
        let seller_new_balance = seller_balance + amount;

        validate_balance(
            buyer,
            buyer_new_balance,
            buyer_limit,
        )?;
        validate_balance(
            seller,
            seller_new_balance,
            seller_limit,
        )?;

        if !buyer_new_balance.is_finite() || !seller_new_balance.is_finite() {
            return Err(MutualCreditError::NonFiniteResult);
        }

        self.balances.insert(buyer.to_owned(), buyer_new_balance);
        self.balances
            .insert(seller.to_owned(), seller_new_balance);

        Ok(MutualTrade {
            amount,
            buyer_new_balance,
            seller_new_balance,
        })
    }

    /// Settle a positive network balance into external money.
    ///
    /// External settlement reduces the member's positive mutual-credit claim;
    /// it never creates network credit. A negative balance therefore cannot be
    /// "paid out" through this path.
    pub fn settle_external(
        &mut self,
        member: &str,
        amount: f64,
    ) -> Result<(), MutualCreditError> {
        validate_amount(amount)?;
        self.require_active_member(member)?;

        let limit = self.limits[member];
        let balance = self.balances[member];
        if balance <= 0.0 || amount > balance {
            return Err(MutualCreditError::InsufficientPositiveBalance {
                member: member.to_owned(),
                balance,
                requested: amount,
            });
        }

        let requested_balance = balance - amount;
        if requested_balance < limit.lower() || requested_balance > limit.upper() {
            return Err(
                MutualCreditError::ExternalSettlementWouldExceedLimit {
                    member: member.to_owned(),
                    balance,
                    requested_balance,
                    lower_limit: limit.lower(),
                    upper_limit: limit.upper(),
                },
            );
        }
        if !requested_balance.is_finite() {
            return Err(MutualCreditError::NonFiniteResult);
        }
        self.balances.insert(member.to_owned(), requested_balance);
        Ok(())
    }

    /// A zero aggregate is an invariant for internal mutual-credit creation
    /// and clearing.
    pub fn reconciles_zero_sum(&self, tolerance: f64) -> bool {
        tolerance.is_finite() && tolerance >= 0.0 && self.total_balance().abs() <= tolerance
    }

    /// A member can leave only after its network position has been cleared.
    pub fn can_exit(&self, member: &str) -> Result<bool, MutualCreditError> {
        self.require_member(member)?;
        Ok(self.balance(member)?.abs() <= f64::EPSILON)
    }

    fn require_member(&self, member: &str) -> Result<(), MutualCreditError> {
        if self.balances.contains_key(member) {
            Ok(())
        } else {
            Err(MutualCreditError::UnknownMember(member.to_owned()))
        }
    }

    fn require_active_member(&self, member: &str) -> Result<(), MutualCreditError> {
        self.require_member(member)?;
        if self.defaults.contains(member) {
            return Err(MutualCreditError::DefaultedMember(member.to_owned()));
        }
        Ok(())
    }
}

impl Default for MutualCreditNetwork {
    fn default() -> Self {
        Self::new()
    }
}

fn validate_amount(amount: f64) -> Result<(), MutualCreditError> {
    if !amount.is_finite() {
        return Err(MutualCreditError::NonFiniteAmount);
    }
    if amount <= 0.0 {
        return Err(MutualCreditError::NonPositiveAmount);
    }
    Ok(())
}

fn validate_balance(
    member: &str,
    requested_balance: f64,
    limit: MemberLimit,
) -> Result<(), MutualCreditError> {
    if requested_balance < limit.lower() || requested_balance > limit.upper() {
        return Err(MutualCreditError::CreditLimitExceeded {
            member: member.to_owned(),
            balance: requested_balance,
            requested_balance,
            lower_limit: limit.lower(),
            upper_limit: limit.upper(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limit() -> MemberLimit {
        MemberLimit::new(100.0, 100.0).unwrap()
    }

    fn network() -> MutualCreditNetwork {
        let mut network = MutualCreditNetwork::new();
        network.register_member("alice", limit()).unwrap();
        network.register_member("bob", limit()).unwrap();
        network.register_member("carol", limit()).unwrap();
        network
    }

    #[test]
    fn trade_creates_reciprocal_positions() {
        let mut network = network();
        let trade = network.trade("alice", "bob", 40.0).unwrap();

        assert_eq!(trade.buyer_new_balance, -40.0);
        assert_eq!(trade.seller_new_balance, 40.0);
        assert!(network.reconciles_zero_sum(1e-12));
    }

    #[test]
    fn multilateral_cycle_clears_without_external_money() {
        let mut network = network();
        network.trade("alice", "bob", 40.0).unwrap();
        network.trade("bob", "carol", 40.0).unwrap();
        network.trade("carol", "alice", 40.0).unwrap();

        assert_eq!(network.balance("alice").unwrap(), 0.0);
        assert_eq!(network.balance("bob").unwrap(), 0.0);
        assert_eq!(network.balance("carol").unwrap(), 0.0);
        assert!(network.reconciles_zero_sum(1e-12));
    }

    #[test]
    fn credit_limit_failure_is_atomic() {
        let mut network = network();
        let before = network.clone();

        let result = network.trade("alice", "bob", 101.0);

        assert!(matches!(
            result,
            Err(MutualCreditError::CreditLimitExceeded { .. })
        ));
        assert_eq!(network, before);
    }

    #[test]
    fn defaulted_member_cannot_trade() {
        let mut network = network();
        network.mark_defaulted("alice").unwrap();

        assert_eq!(
            network.trade("alice", "bob", 1.0),
            Err(MutualCreditError::DefaultedMember("alice".into()))
        );
        assert_eq!(
            network.trade("bob", "alice", 1.0),
            Err(MutualCreditError::DefaultedMember("alice".into()))
        );
        assert!(network.reconciles_zero_sum(1e-12));
    }

    #[test]
    fn external_settlement_requires_positive_claim() {
        let mut network = network();
        network.trade("alice", "bob", 40.0).unwrap();

        assert_eq!(
            network.settle_external("alice", 15.0),
            Err(MutualCreditError::InsufficientPositiveBalance {
                member: "alice".into(),
                balance: -40.0,
                requested: 15.0,
            })
        );

        network.settle_external("bob", 15.0).unwrap();

        assert_eq!(network.balance("bob").unwrap(), 25.0);
        assert_eq!(network.balance("alice").unwrap(), -40.0);
        assert_eq!(network.total_balance(), -15.0);
    }

    #[test]
    fn nonmember_and_self_trade_fail_closed() {
        let mut network = network();
        assert_eq!(
            network.trade("unknown", "bob", 1.0),
            Err(MutualCreditError::UnknownMember("unknown".into()))
        );
        assert_eq!(
            network.trade("alice", "alice", 1.0),
            Err(MutualCreditError::SelfTrade)
        );
    }

    #[test]
    fn exit_requires_zero_position() {
        let mut network = network();
        network.trade("alice", "bob", 20.0).unwrap();

        assert!(!network.can_exit("alice").unwrap());
        assert!(!network.can_exit("bob").unwrap());
    }

    #[test]
    fn same_operations_are_deterministic() {
        let mut first = network();
        let mut second = network();

        for target in [&mut first, &mut second] {
            target.trade("alice", "bob", 20.0).unwrap();
            target.trade("bob", "carol", 7.0).unwrap();
            target.trade("carol", "alice", 13.0).unwrap();
        }

        assert_eq!(first, second);
    }
}
