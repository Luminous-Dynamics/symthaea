// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Thermodynamic Economy — "Tend" mutual credit linked to primary production.

use serde::{Deserialize, Serialize};

/// The economic engine of a sovereign settlement.
///
/// "Tend" is minted only through primary production (e.g. water purification,
/// material recycling, structural manufacturing). Entropy production
/// acts as a natural burn/debit mechanism.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicLedger {
    pub total_tend_supply: f64,
    pub primary_production_index: f32,
    pub cumulative_entropy: f64,
}

impl Default for ThermodynamicLedger {
    fn default() -> Self {
        Self {
            total_tend_supply: 1000.0, // Initial endowment
            primary_production_index: 1.0,
            cumulative_entropy: 0.0,
        }
    }
}

impl ThermodynamicLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mint "Tend" based on metabolic primary production.
    ///
    /// # Arguments
    /// * `production_mw` - Power successfully routed to fabrication/filtration.
    /// * `efficiency` - Metabolic efficiency (e.g. water clarity).
    pub fn mint_production_credit(&mut self, production_mw: f32, efficiency: f32) -> f64 {
        if efficiency < 0.1 {
            return 0.0;
        }

        // Value formula: Credit = Production * Efficiency^2
        let credit = (production_mw as f64 * (efficiency as f64).powi(2)) * 0.1;
        self.total_tend_supply += credit;
        self.primary_production_index =
            (self.primary_production_index * 0.95) + (production_mw * 0.05);

        tracing::info!(
            "🪙  Economy: Minted {:.4} Tend from primary production.",
            credit
        );
        credit
    }

    /// Burn "Tend" based on systemic entropy (prediction error/surprise).
    pub fn apply_entropy_tax(&mut self, surprise: f32) -> f64 {
        let tax = (surprise as f64 * 0.5).min(self.total_tend_supply * 0.1);
        self.total_tend_supply -= tax;
        self.cumulative_entropy += surprise as f64;

        if tax > 0.1 {
            tracing::warn!(
                "🔥 Economy: Burned {:.4} Tend due to systemic surprise (Entropy).",
                tax
            );
        }
        tax
    }

    pub fn current_balance(&self) -> f64 {
        self.total_tend_supply
    }
}
