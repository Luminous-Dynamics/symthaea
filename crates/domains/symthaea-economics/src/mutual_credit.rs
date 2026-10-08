// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Minimal, mechanism-native Mutual Credit semantics.
//!
//! This module models a bounded credit-clearing network in which a trade
//! simultaneously creates a positive claim for the seller and a negative
//! obligation for the buyer. Internal trade is exactly zero-sum. External
//! settlement is tracked as an explicit conservation-boundary event.
//!
//! The v0 kernel deliberately does not inherit commercial banking, interest,
//! collateral, equity, or Creditism settlement semantics.

use std::collections::{BTreeMap, BTreeSet};

pub type CreditUnit = i128;

#[derive(Debug, Clone, PartialEq)]
pub enum MutualCreditError {
    UnknownMember(String),
    DuplicateMember(String),
    InvalidEventId(String),
    DuplicateEventId(String),
    NonPositiveAmount,
    InvalidLimit,
    ArithmeticOverflow,
    CreditLimitExceeded {
        member: String,
        balance: CreditUnit,
        requested_balance: CreditUnit,
        lower_limit: CreditUnit,
        upper_limit: CreditUnit,
    },
    SelfTrade,
    DefaultedMember(String),
    InsufficientPositiveBalance {
        member: String,
        balance: CreditUnit,
        requested: CreditUnit,
    },
    ExternalSettlementWouldExceedLimit {
        member: String,
        balance: CreditUnit,
        requested_balance: CreditUnit,
        lower_limit: CreditUnit,
        upper_limit: CreditUnit,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemberLimit {
    /// Maximum negative balance the member may carry.
    pub debit_limit: CreditUnit,
    /// Maximum positive balance the member may carry.
    pub credit_limit: CreditUnit,
}

impl MemberLimit {
    pub fn new(debit_limit: CreditUnit, credit_limit: CreditUnit) -> Result<Self, MutualCreditError> {
        if debit_limit < 0 || credit_limit < 0 {
            return Err(MutualCreditError::InvalidLimit);
        }
        Ok(Self {
            debit_limit,
            credit_limit,
        })
    }

    fn lower(&self) -> CreditUnit {
        -self.debit_limit
    }

    fn upper(&self) -> CreditUnit {
        self.credit_limit
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MutualTrade {
    pub id: String,
    pub amount: CreditUnit,
    pub buyer_new_balance: CreditUnit,
    pub seller_new_balance: CreditUnit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MutualCreditNetwork {
    balances: BTreeMap<String, CreditUnit>,
    limits: BTreeMap<String, MemberLimit>,
    defaults: BTreeSet<String>,
    seen_event_ids: BTreeSet<String>,
    external_settled_total: CreditUnit,
}

impl MutualCreditNetwork {
    pub fn new() -> Self {
        Self {
            balances: BTreeMap::new(),
            limits: BTreeMap::new(),
            defaults: BTreeSet::new(),
            seen_event_ids: BTreeSet::new(),
            external_settled_total: 0,
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
        self.balances.insert(member.clone(), 0);
        self.limits.insert(member, limit);
        Ok(())
    }

    pub fn balance(&self, member: &str) -> Result<CreditUnit, MutualCreditError> {
        self.balances
            .get(member)
            .copied()
            .ok_or_else(|| MutualCreditError::UnknownMember(member.to_owned()))
    }

    pub fn member_count(&self) -> usize {
        self.balances.len()
    }

    /// Returns the aggregate internal balance or an overflow error.
    pub fn total_balance(&self) -> Result<CreditUnit, MutualCreditError> {
        self.balances
            .values()
            .try_fold(0, |total, balance| {
                total
                    .checked_add(*balance)
                    .ok_or(MutualCreditError::ArithmeticOverflow)
            })
    }

    pub fn external_settled_total(&self) -> CreditUnit {
        self.external_settled_total
    }

    /// Exact internal zero-sum invariant.
    pub fn reconciles_zero_sum(&self) -> bool {
        self.total_balance() == Ok(0)
    }

    /// Exact conservation relation across the external settlement boundary:
    /// internal balance + externally settled claim = zero.
    pub fn reconciles_external_boundary(&self) -> bool {
        match self.total_balance() {
            Ok(total) => total
                .checked_add(self.external_settled_total)
                == Some(0),
            Err(_) => false,
        }
    }

    pub fn mark_defaulted(&mut self, member: &str) -> Result<(), MutualCreditError> {
        self.require_member(member)?;
        self.defaults.insert(member.to_owned());
        Ok(())
    }

    pub fn is_defaulted(&self, member: &str) -> bool {
        self.defaults.contains(member)
    }

    /// Execute an identified trade inside the network.
    ///
    /// No external money is introduced. The buyer's negative position and
    /// seller's positive position are created atomically and preserve a
    /// zero-sum network balance.
    pub fn trade(
        &mut self,
        event_id: impl Into<String>,
        buyer: &str,
        seller: &str,
        amount: CreditUnit,
    ) -> Result<MutualTrade, MutualCreditError> {
        let event_id = self.validate_event_id(event_id)?;
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

        let buyer_new_balance = buyer_balance
            .checked_sub(amount)
            .ok_or(MutualCreditError::ArithmeticOverflow)?;
        let seller_new_balance = seller_balance
            .checked_add(amount)
            .ok_or(MutualCreditError::ArithmeticOverflow)?;

        validate_balance(
            buyer,
            buyer_balance,
            buyer_new_balance,
            buyer_limit,
        )?;
        validate_balance(
            seller,
            seller_balance,
            seller_new_balance,
            seller_limit,
        )?;

        self.seen_event_ids.insert(event_id.clone());
        self.balances.insert(buyer.to_owned(), buyer_new_balance);
        self.balances
            .insert(seller.to_owned(), seller_new_balance);

        Ok(MutualTrade {
            id: event_id,
            amount,
            buyer_new_balance,
            seller_new_balance,
        })
    }

    /// Settle a positive network balance into external money.
    ///
    /// External settlement reduces the member's positive mutual-credit claim;
    /// it never creates network credit. A negative balance cannot be paid out
    /// through this path.
    pub fn settle_external(
        &mut self,
        event_id: impl Into<String>,
        member: &str,
        amount: CreditUnit,
    ) -> Result<(), MutualCreditError> {
        let event_id = self.validate_event_id(event_id)?;
        validate_amount(amount)?;
        self.require_active_member(member)?;

        let limit = self.limits[member];
        let balance = self.balances[member];
        if balance <= 0 || amount > balance {
            return Err(MutualCreditError::InsufficientPositiveBalance {
                member: member.to_owned(),
                balance,
                requested: amount,
            });
        }

        let requested_balance = balance
            .checked_sub(amount)
            .ok_or(MutualCreditError::ArithmeticOverflow)?;
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

        let next_external_total = self
            .external_settled_total
            .checked_add(amount)
            .ok_or(MutualCreditError::ArithmeticOverflow)?;

        self.seen_event_ids.insert(event_id);
        self.balances.insert(member.to_owned(), requested_balance);
        self.external_settled_total = next_external_total;
        Ok(())
    }

    /// A member can leave only after its network position has been cleared.
    pub fn can_exit(&self, member: &str) -> Result<bool, MutualCreditError> {
        self.require_member(member)?;
        Ok(self.balance(member)? == 0)
    }

    fn validate_event_id(
        &self,
        event_id: impl Into<String>,
    ) -> Result<String, MutualCreditError> {
        let event_id = event_id.into();
        if event_id.is_empty() {
            return Err(MutualCreditError::InvalidEventId(event_id));
        }
        if self.seen_event_ids.contains(&event_id) {
            return Err(MutualCreditError::DuplicateEventId(event_id));
        }
        Ok(event_id)
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

fn validate_amount(amount: CreditUnit) -> Result<(), MutualCreditError> {
    if amount <= 0 {
        return Err(MutualCreditError::NonPositiveAmount);
    }
    Ok(())
}

fn validate_balance(
    member: &str,
    balance: CreditUnit,
    requested_balance: CreditUnit,
    limit: MemberLimit,
) -> Result<(), MutualCreditError> {
    if requested_balance < limit.lower() || requested_balance > limit.upper() {
        return Err(MutualCreditError::CreditLimitExceeded {
            member: member.to_owned(),
            balance,
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
        MemberLimit::new(100, 100).unwrap()
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
        let trade = network.trade("mc-1", "alice", "bob", 40).unwrap();

        assert_eq!(trade.id, "mc-1");
        assert_eq!(trade.amount, 40);
        assert_eq!(trade.buyer_new_balance, -40);
        assert_eq!(trade.seller_new_balance, 40);
        assert!(network.reconciles_zero_sum());
        assert!(network.reconciles_external_boundary());
    }

    #[test]
    fn multilateral_reciprocal_trades_clear_without_external_money() {
        let mut network = network();
        network.trade("mc-1", "alice", "bob", 40).unwrap();
        network.trade("mc-2", "bob", "carol", 40).unwrap();
        network.trade("mc-3", "carol", "alice", 40).unwrap();

        assert_eq!(network.balance("alice").unwrap(), 0);
        assert_eq!(network.balance("bob").unwrap(), 0);
        assert_eq!(network.balance("carol").unwrap(), 0);
        assert!(network.reconciles_zero_sum());
    }

    #[test]
    fn credit_limit_failure_is_atomic() {
        let mut network = network();
        let before = network.clone();

        let result = network.trade("mc-1", "alice", "bob", 101);

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
            network.trade("mc-1", "alice", "bob", 1),
            Err(MutualCreditError::DefaultedMember("alice".into()))
        );
        assert_eq!(
            network.trade("mc-2", "bob", "alice", 1),
            Err(MutualCreditError::DefaultedMember("alice".into()))
        );
        assert!(network.reconciles_zero_sum());
    }

    #[test]
    fn external_settlement_requires_positive_claim_and_tracks_boundary() {
        let mut network = network();
        network.trade("mc-1", "alice", "bob", 40).unwrap();

        assert_eq!(
            network.settle_external("mc-2", "alice", 15),
            Err(MutualCreditError::InsufficientPositiveBalance {
                member: "alice".into(),
                balance: -40,
                requested: 15,
            })
        );
        assert_eq!(
            network.settle_external("mc-3", "bob", 50),
            Err(MutualCreditError::InsufficientPositiveBalance {
                member: "bob".into(),
                balance: 40,
                requested: 50,
            })
        );

        network.settle_external("mc-4", "bob", 15).unwrap();

        assert_eq!(network.balance("bob").unwrap(), 25);
        assert_eq!(network.balance("alice").unwrap(), -40);
        assert_eq!(network.total_balance(), Ok(-15));
        assert_eq!(network.external_settled_total(), 15);
        assert!(network.reconciles_external_boundary());
    }

    #[test]
    fn replayed_event_id_fails_closed_without_mutation() {
        let mut network = network();
        network.trade("mc-1", "alice", "bob", 20).unwrap();
        let before = network.clone();

        let result = network.trade("mc-1", "alice", "bob", 20);

        assert_eq!(
            result,
            Err(MutualCreditError::DuplicateEventId("mc-1".into()))
        );
        assert_eq!(network, before);
    }

    #[test]
    fn distinct_event_ids_permit_legitimate_repetition() {
        let mut network = network();
        network.trade("mc-1", "alice", "bob", 20).unwrap();
        network.trade("mc-2", "alice", "bob", 20).unwrap();

        assert_eq!(network.balance("alice").unwrap(), -40);
        assert_eq!(network.balance("bob").unwrap(), 40);
        assert!(network.reconciles_zero_sum());
    }

    #[test]
    fn nonmember_and_self_trade_fail_closed() {
        let mut network = network();
        assert_eq!(
            network.trade("mc-1", "unknown", "bob", 1),
            Err(MutualCreditError::UnknownMember("unknown".into()))
        );
        assert_eq!(
            network.trade("mc-2", "alice", "alice", 1),
            Err(MutualCreditError::SelfTrade)
        );
    }

    #[test]
    fn exit_requires_zero_position() {
        let mut network = network();
        network.trade("mc-1", "alice", "bob", 20).unwrap();

        assert!(!network.can_exit("alice").unwrap());
        assert!(!network.can_exit("bob").unwrap());
    }

    #[test]
    fn same_operations_are_deterministic() {
        let mut first = network();
        let mut second = network();

        for (target, prefix) in [(&mut first, "a"), (&mut second, "a")] {
            target.trade(format!("{prefix}-1"), "alice", "bob", 20).unwrap();
            target.trade(format!("{prefix}-2"), "bob", "carol", 7).unwrap();
            target.trade(format!("{prefix}-3"), "carol", "alice", 13).unwrap();
        }

        assert_eq!(first, second);
        assert!(first.reconciles_zero_sum());
    }

    #[test]
    fn invalid_event_ids_fail_closed() {
        let mut network = network();

        assert_eq!(
            network.trade("", "alice", "bob", 1),
            Err(MutualCreditError::InvalidEventId("".into()))
        );
        assert_eq!(
            network.trade("mc-1", "alice", "bob", 0),
            Err(MutualCreditError::NonPositiveAmount)
        );
        network.trade("mc-1", "alice", "bob", 1).unwrap();
    }
}
