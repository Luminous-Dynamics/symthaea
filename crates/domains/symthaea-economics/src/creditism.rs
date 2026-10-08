// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Minimal, mechanism-native Creditism settlement semantics.
//!
//! This module deliberately does not model Personal Credit as a conventional
//! transferable financial asset. It only captures the bounded mechanics needed
//! for synthetic qualification:
//!
//! - issuance;
//! - deletion at defined use;
//! - separate contribution recognition;
//! - exchange seller recognition bounded by prior acquisition cost;
//! - rejection of transfer/loan/collateral/investment/inheritance attempts;
//! - exact Personal Credit stock reconciliation.
//!
//! It is not a macroeconomic simulator, governance engine, or policy oracle.

use std::collections::{BTreeMap, btree_map::Entry};

#[derive(Debug, Clone, PartialEq)]
pub enum CreditismError {
    UnknownAccount(String),
    NonPositiveAmount,
    InsufficientPersonalCredit {
        account: String,
        available: f64,
        requested: f64,
    },
    NonTransferable,
    NotCollateralizable,
    NotInterestBearing,
    NotInvestable,
    NotInheritable,
    NonFiniteAmount,
    NonFiniteResult,
    DuplicateAccount(String),
    SelfExchange,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExchangeSettlement {
    pub buyer_deleted: f64,
    pub seller_recognized: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PersonalCreditLedger {
    opening_stock: f64,
    balances: BTreeMap<String, f64>,
    issued: f64,
    deleted: f64,
}

impl PersonalCreditLedger {
    /// Construct an empty ledger.
    pub fn new() -> Self {
        Self {
            opening_stock: 0.0,
            balances: BTreeMap::new(),
            issued: 0.0,
            deleted: 0.0,
        }
    }

    /// Construct from an explicit opening state.
    ///
    /// The opening state is treated as already reconciled stock, not as a
    /// current-period issuance event.
    pub fn from_opening_balances(
        balances: impl IntoIterator<Item = (String, f64)>,
    ) -> Result<Self, CreditismError> {
        let mut ledger = Self::new();
        for (account, amount) in balances {
            validate_account(&account)?;
            validate_amount(amount)?;
            match ledger.balances.entry(account) {
                Entry::Vacant(entry) => {
                    entry.insert(amount);
                }
                Entry::Occupied(entry) => {
                    return Err(CreditismError::DuplicateAccount(entry.key().clone()));
                }
            }
        }
        ledger.opening_stock = ledger.total_balance();
        Ok(ledger)
    }

    pub fn balance(&self, account: &str) -> f64 {
        self.balances.get(account).copied().unwrap_or(0.0)
    }

    pub fn total_balance(&self) -> f64 {
        self.balances.values().sum()
    }

    pub fn opening_stock(&self) -> f64 {
        self.opening_stock
    }

    pub fn issued(&self) -> f64 {
        self.issued
    }

    pub fn deleted(&self) -> f64 {
        self.deleted
    }

    /// Issue Personal Credit for existence/basic entitlement.
    pub fn issue_existence(
        &mut self,
        account: &str,
        amount: f64,
    ) -> Result<(), CreditismError> {
        self.issue(account, amount)
    }

    /// Issue Personal Credit for a separately verified contribution.
    ///
    /// This function intentionally does not accept a buyer/purchase identifier.
    /// Contribution recognition remains independent of consumer settlement.
    pub fn issue_contribution(
        &mut self,
        account: &str,
        amount: f64,
    ) -> Result<(), CreditismError> {
        self.issue(account, amount)
    }

    /// Issue a bonus as a separate recognition event.
    pub fn issue_bonus(&mut self, account: &str, amount: f64) -> Result<(), CreditismError> {
        self.issue(account, amount)
    }

    /// Delete Personal Credit at a Marketplace purchase.
    pub fn marketplace_purchase(
        &mut self,
        buyer: &str,
        amount: f64,
    ) -> Result<f64, CreditismError> {
        self.delete(buyer, amount)?;
        Ok(amount)
    }

    /// Execute the documented Exchange-style settlement:
    ///
    /// - the buyer loses the full purchase amount;
    /// - the seller receives separate recognition bounded by what they
    ///   previously paid to acquire the exchanged good;
    /// - the purchase price is not automatically treated as seller revenue.
    pub fn exchange(
        &mut self,
        buyer: &str,
        seller: &str,
        purchase_price: f64,
        seller_acquisition_cost: f64,
    ) -> Result<ExchangeSettlement, CreditismError> {
        validate_amount(purchase_price)?;
        validate_nonnegative_amount(seller_acquisition_cost)?;
        if buyer == seller {
            return Err(CreditismError::SelfExchange);
        }
        if seller.is_empty() {
            return Err(CreditismError::UnknownAccount(seller.to_owned()));
        }

        let buyer_balance = self
            .balances
            .get(buyer)
            .copied()
            .ok_or_else(|| CreditismError::UnknownAccount(buyer.to_owned()))?;
        if buyer_balance < purchase_price {
            return Err(CreditismError::InsufficientPersonalCredit {
                account: buyer.to_owned(),
                available: buyer_balance,
                requested: purchase_price,
            });
        }

        let seller_recognized = purchase_price.min(seller_acquisition_cost);
        let new_seller_balance = self.balance(seller) + seller_recognized;
        let new_deleted = self.deleted + purchase_price;
        let new_issued = self.issued + seller_recognized;
        if !new_seller_balance.is_finite()
            || !new_deleted.is_finite()
            || !new_issued.is_finite()
        {
            return Err(CreditismError::NonFiniteResult);
        }

        if let Some(balance) = self.balances.get_mut(buyer) {
            *balance -= purchase_price;
        } else {
            unreachable!("buyer balance was checked above");
        }
        *self.balances.entry(seller.to_owned()).or_default() = new_seller_balance;
        self.deleted = new_deleted;
        self.issued = new_issued;

        Ok(ExchangeSettlement {
            buyer_deleted: purchase_price,
            seller_recognized,
        })
    }

    pub fn attempt_transfer(
        &self,
        _from: &str,
        _to: &str,
        _amount: f64,
    ) -> Result<(), CreditismError> {
        Err(CreditismError::NonTransferable)
    }

    pub fn attempt_loan(
        &self,
        _from: &str,
        _to: &str,
        _amount: f64,
        _interest_rate: f64,
    ) -> Result<(), CreditismError> {
        Err(CreditismError::NotInterestBearing)
    }

    pub fn attempt_collateralization(
        &self,
        _account: &str,
        _amount: f64,
    ) -> Result<(), CreditismError> {
        Err(CreditismError::NotCollateralizable)
    }

    pub fn attempt_investment(
        &self,
        _account: &str,
        _amount: f64,
    ) -> Result<(), CreditismError> {
        Err(CreditismError::NotInvestable)
    }

    pub fn attempt_inheritance(
        &self,
        _from: &str,
        _to: &str,
        _amount: f64,
    ) -> Result<(), CreditismError> {
        Err(CreditismError::NotInheritable)
    }

    /// Check the exact Personal Credit stock identity.
    pub fn reconciles(&self, tolerance: f64) -> bool {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return false;
        }
        let expected = self.opening_stock + self.issued - self.deleted;
        (self.total_balance() - expected).abs() <= tolerance
    }

    fn issue(&mut self, account: &str, amount: f64) -> Result<(), CreditismError> {
        validate_amount(amount)?;
        validate_account(account)?;
        let current_balance = self.balance(account);
        let new_balance = current_balance + amount;
        let new_issued = self.issued + amount;
        if !new_balance.is_finite() || !new_issued.is_finite() {
            return Err(CreditismError::NonFiniteResult);
        }
        self.balances.insert(account.to_owned(), new_balance);
        self.issued = new_issued;
        Ok(())
    }

    fn delete(&mut self, account: &str, amount: f64) -> Result<(), CreditismError> {
        validate_amount(amount)?;
        let balance = self
            .balances
            .get_mut(account)
            .ok_or_else(|| CreditismError::UnknownAccount(account.to_owned()))?;

        if *balance < amount {
            return Err(CreditismError::InsufficientPersonalCredit {
                account: account.to_owned(),
                available: *balance,
                requested: amount,
            });
        }

        let new_deleted = self.deleted + amount;
        if !new_deleted.is_finite() {
            return Err(CreditismError::NonFiniteResult);
        }

        *balance -= amount;
        self.deleted = new_deleted;
        Ok(())
    }
}

impl Default for PersonalCreditLedger {
    fn default() -> Self {
        Self::new()
    }
}

fn validate_account(account: &str) -> Result<(), CreditismError> {
    if account.is_empty() {
        return Err(CreditismError::UnknownAccount(account.to_owned()));
    }
    Ok(())
}

fn validate_amount(amount: f64) -> Result<(), CreditismError> {
    if !amount.is_finite() {
        return Err(CreditismError::NonFiniteAmount);
    }
    if amount <= 0.0 {
        return Err(CreditismError::NonPositiveAmount);
    }
    Ok(())
}

fn validate_nonnegative_amount(amount: f64) -> Result<(), CreditismError> {
    if !amount.is_finite() {
        return Err(CreditismError::NonFiniteAmount);
    }
    if amount < 0.0 {
        return Err(CreditismError::NonPositiveAmount);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marketplace_deletion_is_not_seller_payment() {
        let mut ledger = PersonalCreditLedger::from_opening_balances([
            ("buyer".to_owned(), 100.0),
            ("seller".to_owned(), 0.0),
        ])
        .unwrap();

        ledger.marketplace_purchase("buyer", 30.0).unwrap();

        assert_eq!(ledger.balance("buyer"), 70.0);
        assert_eq!(ledger.balance("seller"), 0.0);
        assert_eq!(ledger.deleted(), 30.0);
        assert!(ledger.reconciles(0.0));
    }

    #[test]
    fn contribution_recognition_is_independent_of_purchase() {
        let mut ledger = PersonalCreditLedger::new();
        ledger.issue_existence("buyer", 100.0).unwrap();

        ledger.marketplace_purchase("buyer", 30.0).unwrap();
        ledger.issue_contribution("producer", 25.0).unwrap();

        assert_eq!(ledger.balance("buyer"), 70.0);
        assert_eq!(ledger.balance("producer"), 25.0);
        assert_eq!(ledger.deleted(), 30.0);
        assert_eq!(ledger.issued(), 125.0);
        assert!(ledger.reconciles(0.0));
    }

    #[test]
    fn exchange_deletes_premium_without_making_it_seller_income() {
        let mut ledger = PersonalCreditLedger::from_opening_balances([
            ("buyer".to_owned(), 100.0),
            ("seller".to_owned(), 10.0),
        ])
        .unwrap();

        let settlement = ledger.exchange("buyer", "seller", 100.0, 10.0).unwrap();

        assert_eq!(settlement.buyer_deleted, 100.0);
        assert_eq!(settlement.seller_recognized, 10.0);
        assert_eq!(ledger.balance("buyer"), 0.0);
        assert_eq!(ledger.balance("seller"), 20.0);
        assert_eq!(ledger.deleted(), 100.0);
        assert_eq!(ledger.issued(), 10.0);
        assert!(ledger.reconciles(0.0));
    }

    #[test]
    fn forbidden_capabilities_fail_closed() {
        let ledger = PersonalCreditLedger::new();

        assert_eq!(
            ledger.attempt_transfer("a", "b", 1.0),
            Err(CreditismError::NonTransferable)
        );
        assert_eq!(
            ledger.attempt_collateralization("a", 1.0),
            Err(CreditismError::NotCollateralizable)
        );
        assert_eq!(
            ledger.attempt_loan("a", "b", 1.0, 0.1),
            Err(CreditismError::NotInterestBearing)
        );
        assert_eq!(
            ledger.attempt_investment("a", 1.0),
            Err(CreditismError::NotInvestable)
        );
        assert_eq!(
            ledger.attempt_inheritance("a", "b", 1.0),
            Err(CreditismError::NotInheritable)
        );
    }

    #[test]
    fn failed_exchange_is_atomic() {
        let mut ledger = PersonalCreditLedger::from_opening_balances([
            ("buyer".to_owned(), 100.0),
            ("seller".to_owned(), 10.0),
        ])
        .unwrap();
        let before = ledger.clone();

        assert_eq!(
            ledger.exchange("buyer", "", 50.0, 10.0),
            Err(CreditismError::UnknownAccount(String::new()))
        );
        assert_eq!(ledger, before);
    }

    #[test]
    fn self_exchange_fails_closed_without_mutation() {
        let mut ledger = PersonalCreditLedger::from_opening_balances([
            ("same".to_owned(), 100.0),
        ])
        .unwrap();
        let before = ledger.clone();

        assert_eq!(
            ledger.exchange("same", "same", 50.0, 10.0),
            Err(CreditismError::SelfExchange)
        );
        assert_eq!(ledger, before);
    }

    #[test]
    fn zero_acquisition_cost_is_valid_exchange_input() {
        let mut ledger = PersonalCreditLedger::from_opening_balances([
            ("buyer".to_owned(), 20.0),
            ("seller".to_owned(), 0.0),
        ])
        .unwrap();

        let settlement = ledger.exchange("buyer", "seller", 20.0, 0.0).unwrap();

        assert_eq!(settlement.buyer_deleted, 20.0);
        assert_eq!(settlement.seller_recognized, 0.0);
        assert!(ledger.reconciles(0.0));
    }

    #[test]
    fn opening_state_rejects_duplicate_accounts() {
        assert_eq!(
            PersonalCreditLedger::from_opening_balances([
                ("alice".to_owned(), 10.0),
                ("alice".to_owned(), 20.0),
            ]),
            Err(CreditismError::DuplicateAccount("alice".to_owned()))
        );
    }

    #[test]
    fn opening_state_rejects_empty_accounts() {
        assert_eq!(
            PersonalCreditLedger::from_opening_balances([("".to_owned(), 10.0)]),
            Err(CreditismError::UnknownAccount(String::new()))
        );
    }

    #[test]
    fn malformed_amounts_fail_closed() {
        let mut ledger = PersonalCreditLedger::new();
        assert_eq!(
            ledger.issue_existence("a", f64::NAN),
            Err(CreditismError::NonFiniteAmount)
        );
        assert_eq!(
            ledger.issue_existence("a", 0.0),
            Err(CreditismError::NonPositiveAmount)
        );
        assert_eq!(
            ledger.issue_existence("a", -1.0),
            Err(CreditismError::NonPositiveAmount)
        );
    }

    #[test]
    fn insufficient_balance_does_not_mutate_ledger() {
        let mut ledger = PersonalCreditLedger::new();
        ledger.issue_existence("buyer", 5.0).unwrap();

        let before = ledger.clone();
        assert!(ledger.marketplace_purchase("buyer", 6.0).is_err());
        assert_eq!(ledger, before);
    }
}
