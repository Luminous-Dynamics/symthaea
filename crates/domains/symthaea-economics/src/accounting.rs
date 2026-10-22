// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exact double-entry accounting primitives.
//!
//! Amounts are integer atoms in an explicitly named unit. The kernel therefore
//! enforces balance exactly rather than hiding accounting drift behind a
//! floating-point tolerance. Issuance, destruction, equity, and external-sector
//! effects must be represented through explicit counterpart accounts.

use std::collections::{BTreeMap, BTreeSet, btree_map::Entry};

use crate::error::{EconomicsError, Result};
use crate::ontology::{AccountId, UnitId};

/// Compare positive and negative magnitudes by cancellation instead of summing
/// them. This remains exact even when each side's aggregate magnitude would
/// exceed `u128`.
fn signed_values_balance(values: impl IntoIterator<Item = i128>) -> bool {
    let mut positive = Vec::new();
    let mut negative = Vec::new();
    for value in values {
        if value > 0 {
            positive.push(value as u128);
        } else if value < 0 {
            negative.push(value.unsigned_abs());
        }
    }

    let (mut p_index, mut n_index) = (0_usize, 0_usize);
    let (mut p_remaining, mut n_remaining) = (0_u128, 0_u128);
    loop {
        if p_remaining == 0 && p_index < positive.len() {
            p_remaining = positive[p_index];
            p_index += 1;
        }
        if n_remaining == 0 && n_index < negative.len() {
            n_remaining = negative[n_index];
            n_index += 1;
        }

        if p_remaining == 0 || n_remaining == 0 {
            return p_remaining == 0
                && n_remaining == 0
                && p_index == positive.len()
                && n_index == negative.len();
        }

        let cancelled = p_remaining.min(n_remaining);
        p_remaining -= cancelled;
        n_remaining -= cancelled;
    }
}

/// One signed posting in integer atoms of a journal entry's declared unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Posting {
    account: AccountId,
    amount_atoms: i128,
}

impl Posting {
    pub fn new(account: AccountId, amount_atoms: i128) -> Result<Self> {
        if amount_atoms == 0 {
            return Err(EconomicsError::InvalidParameter {
                context: "zero accounting posting",
            });
        }
        Ok(Self {
            account,
            amount_atoms,
        })
    }

    pub fn account(&self) -> &AccountId {
        &self.account
    }

    pub fn amount_atoms(&self) -> i128 {
        self.amount_atoms
    }
}

/// An exactly balanced journal entry in one declared unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JournalEntry {
    unit: UnitId,
    postings: Vec<Posting>,
    memo: Option<String>,
}

impl JournalEntry {
    pub fn new(unit: UnitId, postings: Vec<Posting>, memo: Option<String>) -> Result<Self> {
        if postings.len() < 2 {
            return Err(EconomicsError::InvalidParameter {
                context: "journal entry requires at least two postings",
            });
        }
        if memo.as_ref().is_some_and(|memo| memo.trim().is_empty()) {
            return Err(EconomicsError::InvalidParameter {
                context: "journal entry memo",
            });
        }

        let mut seen = BTreeSet::new();
        for posting in &postings {
            if !seen.insert(posting.account.clone()) {
                return Err(EconomicsError::InvalidParameter {
                    context: "duplicate account in journal entry",
                });
            }
        }
        if !signed_values_balance(postings.iter().map(Posting::amount_atoms)) {
            return Err(EconomicsError::InvalidParameter {
                context: "unbalanced journal entry",
            });
        }

        Ok(Self {
            unit,
            postings,
            memo,
        })
    }

    pub fn unit(&self) -> &UnitId {
        &self.unit
    }

    pub fn postings(&self) -> &[Posting] {
        &self.postings
    }

    pub fn memo(&self) -> Option<&str> {
        self.memo.as_deref()
    }
}

/// A closed double-entry ledger for one explicit unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DoubleEntryLedger {
    unit: UnitId,
    balances: BTreeMap<AccountId, i128>,
}

impl DoubleEntryLedger {
    pub fn new(unit: UnitId) -> Self {
        Self {
            unit,
            balances: BTreeMap::new(),
        }
    }

    pub fn unit(&self) -> &UnitId {
        &self.unit
    }

    pub fn register_account(&mut self, account: AccountId) -> Result<()> {
        match self.balances.entry(account) {
            Entry::Vacant(slot) => {
                slot.insert(0);
                Ok(())
            }
            Entry::Occupied(_) => Err(EconomicsError::InvalidParameter {
                context: "duplicate ledger account",
            }),
        }
    }

    pub fn balance(&self, account: &AccountId) -> Result<i128> {
        self.balances
            .get(account)
            .copied()
            .ok_or(EconomicsError::InvalidParameter {
                context: "unknown ledger account",
            })
    }

    pub fn account_count(&self) -> usize {
        self.balances.len()
    }

    /// Apply a balanced entry atomically. Any validation or arithmetic failure
    /// leaves the ledger unchanged.
    pub fn apply(&mut self, entry: &JournalEntry) -> Result<()> {
        if entry.unit != self.unit {
            return Err(EconomicsError::InvalidParameter {
                context: "journal unit does not match ledger unit",
            });
        }

        let mut updates = Vec::with_capacity(entry.postings.len());
        for posting in &entry.postings {
            let current = self.balances.get(&posting.account).copied().ok_or(
                EconomicsError::InvalidParameter {
                    context: "journal references unknown ledger account",
                },
            )?;
            let updated = current.checked_add(posting.amount_atoms).ok_or(
                EconomicsError::NumericalFailure {
                    context: "ledger balance overflow",
                },
            )?;
            updates.push((posting.account.clone(), updated));
        }

        for (account, updated) in updates {
            self.balances.insert(account, updated);
        }
        debug_assert!(self.is_balanced());
        Ok(())
    }

    /// Exact global double-entry invariant without aggregate overflow.
    pub fn is_balanced(&self) -> bool {
        signed_values_balance(self.balances.values().copied())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn account(id: &str) -> AccountId {
        AccountId::new(id).unwrap()
    }

    fn unit() -> UnitId {
        UnitId::new("usd:cent").unwrap()
    }

    #[test]
    fn unbalanced_entries_fail_closed() {
        let entry = JournalEntry::new(
            unit(),
            vec![
                Posting::new(account("cash"), 100).unwrap(),
                Posting::new(account("equity"), -99).unwrap(),
            ],
            None,
        );
        assert!(entry.is_err());
    }

    #[test]
    fn balanced_entry_preserves_exact_ledger_identity() {
        let cash = account("cash");
        let equity = account("equity");
        let mut ledger = DoubleEntryLedger::new(unit());
        ledger.register_account(cash.clone()).unwrap();
        ledger.register_account(equity.clone()).unwrap();

        let entry = JournalEntry::new(
            unit(),
            vec![
                Posting::new(cash.clone(), 10_000).unwrap(),
                Posting::new(equity.clone(), -10_000).unwrap(),
            ],
            Some("opening balance".into()),
        )
        .unwrap();
        ledger.apply(&entry).unwrap();

        assert_eq!(ledger.balance(&cash).unwrap(), 10_000);
        assert_eq!(ledger.balance(&equity).unwrap(), -10_000);
        assert!(ledger.is_balanced());
    }

    #[test]
    fn duplicate_registration_does_not_reset_existing_balance() {
        let cash = account("cash");
        let equity = account("equity");
        let mut ledger = DoubleEntryLedger::new(unit());
        ledger.register_account(cash.clone()).unwrap();
        ledger.register_account(equity.clone()).unwrap();
        ledger
            .apply(
                &JournalEntry::new(
                    unit(),
                    vec![
                        Posting::new(cash.clone(), 500).unwrap(),
                        Posting::new(equity.clone(), -500).unwrap(),
                    ],
                    None,
                )
                .unwrap(),
            )
            .unwrap();

        assert!(ledger.register_account(cash.clone()).is_err());
        assert_eq!(ledger.balance(&cash).unwrap(), 500);
        assert_eq!(ledger.balance(&equity).unwrap(), -500);
        assert!(ledger.is_balanced());
    }

    #[test]
    fn failed_application_is_atomic() {
        let asset = account("asset");
        let equity = account("equity");
        let mut ledger = DoubleEntryLedger::new(unit());
        ledger.register_account(asset.clone()).unwrap();
        ledger.register_account(equity.clone()).unwrap();

        ledger
            .apply(
                &JournalEntry::new(
                    unit(),
                    vec![
                        Posting::new(asset.clone(), i128::MAX).unwrap(),
                        Posting::new(equity.clone(), -i128::MAX).unwrap(),
                    ],
                    None,
                )
                .unwrap(),
            )
            .unwrap();

        let before_asset = ledger.balance(&asset).unwrap();
        let before_equity = ledger.balance(&equity).unwrap();
        let overflow = JournalEntry::new(
            unit(),
            vec![
                Posting::new(asset.clone(), 1).unwrap(),
                Posting::new(equity.clone(), -1).unwrap(),
            ],
            None,
        )
        .unwrap();

        assert!(ledger.apply(&overflow).is_err());
        assert_eq!(ledger.balance(&asset).unwrap(), before_asset);
        assert_eq!(ledger.balance(&equity).unwrap(), before_equity);
        assert!(ledger.is_balanced());
    }

    #[test]
    fn units_cannot_be_silently_mixed() {
        let a = account("a");
        let b = account("b");
        let mut ledger = DoubleEntryLedger::new(unit());
        ledger.register_account(a.clone()).unwrap();
        ledger.register_account(b.clone()).unwrap();
        let entry = JournalEntry::new(
            UnitId::new("eur:cent").unwrap(),
            vec![Posting::new(a, 1).unwrap(), Posting::new(b, -1).unwrap()],
            None,
        )
        .unwrap();
        assert!(ledger.apply(&entry).is_err());
    }

    #[test]
    fn extreme_balanced_magnitudes_do_not_overflow_the_invariant_check() {
        let postings = vec![
            Posting::new(account("a"), i128::MAX).unwrap(),
            Posting::new(account("b"), i128::MAX).unwrap(),
            Posting::new(account("c"), -i128::MAX).unwrap(),
            Posting::new(account("d"), -i128::MAX).unwrap(),
        ];
        assert!(JournalEntry::new(unit(), postings, None).is_ok());
    }
}
