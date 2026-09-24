// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Stock-flow-consistent accounting primitives.
//!
//! This module owns accounting closure for synthetic financial-system models.
//! It deliberately does not own behavioral macroeconomics, causal claims,
//! forecasting, policy recommendations, financing-regime classifications, or
//! live evidence ingestion.
//!
//! The core distinction is:
//!
//! ```text
//! financial claim != productive capital != market valuation
//! ```
//!
//! Every financial position binds both a holder and an issuer/counterparty.
//! Therefore a position cannot exist as an unowned asset without a matching
//! financial obligation/equity claim somewhere else in the modeled system.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::error::{EconomicsError, Result, ensure_finite};

const MAX_ID_BYTES: usize = 128;

macro_rules! id_type {
    ($name:ident, $context:literal) => {
        #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self> {
                let value = value.into();
                if value.is_empty()
                    || value.chars().all(char::is_whitespace)
                    || value.len() > MAX_ID_BYTES
                {
                    return Err(EconomicsError::InvalidParameter { context: $context });
                }
                Ok(Self(value))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }
    };
}

id_type!(SectorId, "stock-flow sector id must be non-empty and bounded");
id_type!(InstrumentId, "stock-flow instrument id must be non-empty and bounded");
id_type!(PositionId, "stock-flow position id must be non-empty and bounded");

/// Numerical profile used only for accounting-identity closure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClosureProfile {
    pub absolute_tolerance: f64,
}

impl ClosureProfile {
    pub fn new(absolute_tolerance: f64) -> Result<Self> {
        ensure_finite(absolute_tolerance, "stock-flow closure tolerance")?;
        if absolute_tolerance < 0.0 {
            return Err(EconomicsError::InvalidParameter {
                context: "stock-flow closure tolerance must be non-negative",
            });
        }
        Ok(Self { absolute_tolerance })
    }
}

impl Default for ClosureProfile {
    fn default() -> Self {
        Self {
            absolute_tolerance: 1e-9,
        }
    }
}

/// One financial position followed across an accounting period.
///
/// The closing identity is:
///
/// ```text
/// closing = opening + transaction_change + valuation_change - writeoff
/// ```
///
/// `transaction_change` may be positive or negative. `valuation_change` may
/// also be positive or negative. `writeoff` is a non-negative loss-recognition
/// amount and is kept separate from both cash-flow transactions and valuation.
#[derive(Clone, Debug, PartialEq)]
pub struct FinancialPositionTransition {
    pub position_id: PositionId,
    pub holder: SectorId,
    pub issuer: SectorId,
    pub instrument: InstrumentId,
    pub opening: f64,
    pub transaction_change: f64,
    pub valuation_change: f64,
    pub writeoff: f64,
    pub closing: f64,
}

impl FinancialPositionTransition {
    pub fn expected_closing(&self) -> f64 {
        self.opening + self.transaction_change + self.valuation_change - self.writeoff
    }

    pub fn closure_residual(&self) -> f64 {
        self.closing - self.expected_closing()
    }

    pub fn validate(&self, profile: ClosureProfile) -> Result<()> {
        for (value, context) in [
            (self.opening, "stock-flow opening position"),
            (self.transaction_change, "stock-flow transaction change"),
            (self.valuation_change, "stock-flow valuation change"),
            (self.writeoff, "stock-flow writeoff"),
            (self.closing, "stock-flow closing position"),
        ] {
            ensure_finite(value, context)?;
        }

        if self.opening < 0.0 || self.closing < 0.0 || self.writeoff < 0.0 {
            return Err(EconomicsError::InvalidParameter {
                context: "stock-flow opening, closing, and writeoff amounts must be non-negative",
            });
        }

        let residual = self.closure_residual();
        ensure_finite(residual, "stock-flow closure residual")?;
        if residual.abs() > profile.absolute_tolerance {
            return Err(EconomicsError::InvalidParameter {
                context: "stock-flow position does not satisfy the closing identity",
            });
        }
        Ok(())
    }
}

/// A payer/receiver income flow represented once, preventing one-sided counting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IncomeFlowKind {
    Wage,
    Interest,
    Dividend,
    Tax,
    Transfer,
    Other,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IncomeTransfer {
    pub payer: SectorId,
    pub receiver: SectorId,
    pub kind: IncomeFlowKind,
    pub amount: f64,
}

impl IncomeTransfer {
    pub fn new(
        payer: SectorId,
        receiver: SectorId,
        kind: IncomeFlowKind,
        amount: f64,
    ) -> Result<Self> {
        ensure_finite(amount, "stock-flow income transfer")?;
        if amount < 0.0 {
            return Err(EconomicsError::InvalidParameter {
                context: "stock-flow income transfer amount must be non-negative",
            });
        }
        Ok(Self {
            payer,
            receiver,
            kind,
            amount,
        })
    }
}

/// Real productive-capital formation, deliberately separate from financial
/// asset ownership and valuation changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RealCapitalCategory {
    Structures,
    Machinery,
    ResearchAndDevelopment,
    SoftwareAndIntangibles,
    PublicInfrastructure,
    Other,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RealCapitalFormation {
    pub sector: SectorId,
    pub category: RealCapitalCategory,
    pub amount: f64,
}

impl RealCapitalFormation {
    pub fn new(sector: SectorId, category: RealCapitalCategory, amount: f64) -> Result<Self> {
        ensure_finite(amount, "real capital formation")?;
        if amount < 0.0 {
            return Err(EconomicsError::InvalidParameter {
                context: "real capital formation amount must be non-negative",
            });
        }
        Ok(Self {
            sector,
            category,
            amount,
        })
    }
}

/// A synthetic accounting period. This is not a behavioral economy model.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StockFlowSystem {
    pub positions: Vec<FinancialPositionTransition>,
    pub income_transfers: Vec<IncomeTransfer>,
    pub real_capital_formation: Vec<RealCapitalFormation>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClosureReport {
    pub position_count: usize,
    pub income_transfer_count: usize,
    pub real_capital_formation_count: usize,
    pub sector_net_financial_positions: BTreeMap<SectorId, f64>,
    pub aggregate_net_financial_position: f64,
    pub total_real_capital_formation: f64,
    pub total_valuation_change: f64,
    pub total_writeoffs: f64,
    pub max_abs_position_residual: f64,
}

impl StockFlowSystem {
    /// Validate every position and compute a deterministic closure report.
    ///
    /// Because each financial position names both holder and issuer, its
    /// closing carrying amount is added to the holder's financial assets and
    /// subtracted from the issuer's financial liabilities/equity claims.
    pub fn validate(&self, profile: ClosureProfile) -> Result<ClosureReport> {
        let mut seen_positions = BTreeSet::new();
        let mut sector_net_financial_positions = BTreeMap::<SectorId, f64>::new();
        let mut max_abs_position_residual: f64 = 0.0;
        let mut total_valuation_change = 0.0;
        let mut total_writeoffs = 0.0;

        for position in &self.positions {
            if !seen_positions.insert(position.position_id.clone()) {
                return Err(EconomicsError::InvalidParameter {
                    context: "duplicate stock-flow position id",
                });
            }
            position.validate(profile)?;
            max_abs_position_residual =
                max_abs_position_residual.max(position.closure_residual().abs());
            total_valuation_change += position.valuation_change;
            total_writeoffs += position.writeoff;

            *sector_net_financial_positions
                .entry(position.holder.clone())
                .or_insert(0.0) += position.closing;
            *sector_net_financial_positions
                .entry(position.issuer.clone())
                .or_insert(0.0) -= position.closing;
        }

        for transfer in &self.income_transfers {
            ensure_finite(transfer.amount, "stock-flow income transfer")?;
            if transfer.amount < 0.0 {
                return Err(EconomicsError::InvalidParameter {
                    context: "stock-flow income transfer amount must be non-negative",
                });
            }
        }

        let mut total_real_capital_formation = 0.0;
        for formation in &self.real_capital_formation {
            ensure_finite(formation.amount, "real capital formation")?;
            if formation.amount < 0.0 {
                return Err(EconomicsError::InvalidParameter {
                    context: "real capital formation amount must be non-negative",
                });
            }
            total_real_capital_formation += formation.amount;
        }

        let aggregate_net_financial_position: f64 =
            sector_net_financial_positions.values().copied().sum();
        ensure_finite(
            aggregate_net_financial_position,
            "aggregate net financial position",
        )?;
        if aggregate_net_financial_position.abs() > profile.absolute_tolerance {
            return Err(EconomicsError::InvalidParameter {
                context: "closed stock-flow system net financial position does not reconcile",
            });
        }

        Ok(ClosureReport {
            position_count: self.positions.len(),
            income_transfer_count: self.income_transfers.len(),
            real_capital_formation_count: self.real_capital_formation.len(),
            sector_net_financial_positions,
            aggregate_net_financial_position,
            total_real_capital_formation,
            total_valuation_change,
            total_writeoffs,
            max_abs_position_residual,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sector(id: &str) -> SectorId {
        SectorId::new(id).unwrap()
    }

    fn instrument(id: &str) -> InstrumentId {
        InstrumentId::new(id).unwrap()
    }

    fn position(
        id: &str,
        holder: &str,
        issuer: &str,
        opening: f64,
        transaction_change: f64,
        valuation_change: f64,
        writeoff: f64,
        closing: f64,
    ) -> FinancialPositionTransition {
        FinancialPositionTransition {
            position_id: PositionId::new(id).unwrap(),
            holder: sector(holder),
            issuer: sector(issuer),
            instrument: instrument(id),
            opening,
            transaction_change,
            valuation_change,
            writeoff,
            closing,
        }
    }

    #[test]
    fn bank_loan_and_matching_deposit_close_without_free_wealth() {
        let system = StockFlowSystem {
            positions: vec![
                position("loan", "bank", "firm", 0.0, 100.0, 0.0, 0.0, 100.0),
                position("deposit", "firm", "bank", 0.0, 100.0, 0.0, 0.0, 100.0),
            ],
            ..Default::default()
        };
        let report = system.validate(ClosureProfile::default()).unwrap();
        assert_eq!(report.aggregate_net_financial_position, 0.0);
        assert_eq!(report.sector_net_financial_positions[&sector("bank")], 0.0);
        assert_eq!(report.sector_net_financial_positions[&sector("firm")], 0.0);
    }

    #[test]
    fn principal_repayment_reduces_positions_exactly_once() {
        let system = StockFlowSystem {
            positions: vec![
                position("loan", "bank", "firm", 100.0, -20.0, 0.0, 0.0, 80.0),
                position("deposit", "firm", "bank", 100.0, -20.0, 0.0, 0.0, 80.0),
            ],
            ..Default::default()
        };
        let report = system.validate(ClosureProfile::default()).unwrap();
        assert_eq!(report.aggregate_net_financial_position, 0.0);
    }

    #[test]
    fn interest_transfer_does_not_masquerade_as_principal_repayment() {
        let unchanged_loan = position("loan", "bank", "firm", 100.0, 0.0, 0.0, 0.0, 100.0);
        let system = StockFlowSystem {
            positions: vec![unchanged_loan.clone()],
            income_transfers: vec![IncomeTransfer::new(
                sector("firm"),
                sector("bank"),
                IncomeFlowKind::Interest,
                5.0,
            )
            .unwrap()],
            ..Default::default()
        };
        system.validate(ClosureProfile::default()).unwrap();
        assert_eq!(unchanged_loan.closing, 100.0);
        assert_eq!(system.income_transfers[0].amount, 5.0);
    }

    #[test]
    fn equity_financing_and_real_capital_are_distinct_records() {
        let system = StockFlowSystem {
            positions: vec![position(
                "equity",
                "household",
                "firm",
                0.0,
                100.0,
                0.0,
                0.0,
                100.0,
            )],
            real_capital_formation: vec![RealCapitalFormation::new(
                sector("firm"),
                RealCapitalCategory::Machinery,
                100.0,
            )
            .unwrap()],
            ..Default::default()
        };
        let report = system.validate(ClosureProfile::default()).unwrap();
        assert_eq!(report.total_real_capital_formation, 100.0);
        assert_eq!(report.total_valuation_change, 0.0);
    }

    #[test]
    fn secondary_market_transfer_does_not_create_productive_capital() {
        let system = StockFlowSystem {
            positions: vec![
                position(
                    "old-owner",
                    "household",
                    "firm",
                    100.0,
                    -100.0,
                    0.0,
                    0.0,
                    0.0,
                ),
                position("new-owner", "fund", "firm", 0.0, 100.0, 0.0, 0.0, 100.0),
            ],
            ..Default::default()
        };
        let report = system.validate(ClosureProfile::default()).unwrap();
        assert_eq!(report.total_real_capital_formation, 0.0);
        assert_eq!(report.aggregate_net_financial_position, 0.0);
    }

    #[test]
    fn asset_price_appreciation_is_not_productive_capital_formation() {
        let system = StockFlowSystem {
            positions: vec![position(
                "equity",
                "household",
                "firm",
                100.0,
                0.0,
                20.0,
                0.0,
                120.0,
            )],
            ..Default::default()
        };
        let report = system.validate(ClosureProfile::default()).unwrap();
        assert_eq!(report.total_valuation_change, 20.0);
        assert_eq!(report.total_real_capital_formation, 0.0);
    }

    #[test]
    fn writeoff_is_explicit_and_does_not_disappear() {
        let system = StockFlowSystem {
            positions: vec![position(
                "loan",
                "bank",
                "firm",
                100.0,
                0.0,
                0.0,
                25.0,
                75.0,
            )],
            ..Default::default()
        };
        let report = system.validate(ClosureProfile::default()).unwrap();
        assert_eq!(report.total_writeoffs, 25.0);
    }

    #[test]
    fn inconsistent_closing_amount_fails() {
        let system = StockFlowSystem {
            positions: vec![position(
                "broken", "bank", "firm", 100.0, -20.0, 0.0, 0.0, 90.0,
            )],
            ..Default::default()
        };
        assert!(matches!(
            system.validate(ClosureProfile::default()),
            Err(EconomicsError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn non_finite_values_fail_closed() {
        let system = StockFlowSystem {
            positions: vec![position(
                "nan", "bank", "firm", f64::NAN, 0.0, 0.0, 0.0, 0.0,
            )],
            ..Default::default()
        };
        assert!(matches!(
            system.validate(ClosureProfile::default()),
            Err(EconomicsError::NonFiniteInput { .. })
        ));
    }

    #[test]
    fn duplicate_position_ids_are_rejected() {
        let one = position("same", "bank", "firm", 0.0, 10.0, 0.0, 0.0, 10.0);
        let two = position("same", "firm", "bank", 0.0, 10.0, 0.0, 0.0, 10.0);
        let system = StockFlowSystem {
            positions: vec![one, two],
            ..Default::default()
        };
        assert!(matches!(
            system.validate(ClosureProfile::default()),
            Err(EconomicsError::InvalidParameter { .. })
        ));
    }
}
