//! Business entity support for tax proofs.
//!
//! This module extends tax proofs to support various business entity types
//! beyond individual taxpayers.
//!
//! # Supported Entity Types
//!
//! - **Sole Proprietorship**: Income flows to personal return (Schedule C)
//! - **Partnership**: Pass-through entity (K-1)
//! - **S-Corporation**: Pass-through entity with payroll (K-1 + W-2)
//! - **C-Corporation**: Double taxation (corporate + dividend)
//! - **LLC**: Flexible taxation election
//!
//! # Example
//!
//! ```rust,ignore
//! use mycelix_zk_tax::entity::{EntityType, BusinessProver, EntityTaxInfo};
//!
//! // Create a business prover
//! let prover = BusinessProver::dev_mode();
//!
//! // Prove S-Corp taxation
//! let proof = prover.prove(
//!     EntityTaxInfo {
//!         entity_type: EntityType::SCorporation,
//!         gross_revenue: 500_000,
//!         reasonable_salary: 80_000,  // S-Corp owner salary
//!         distributions: 120_000,
//!         deductions: 100_000,
//!     },
//!     Jurisdiction::US,
//!     2024,
//! ).unwrap();
//!
//! println!("Effective rate: {}%", proof.effective_rate_bps as f64 / 100.0);
//! ```

use crate::{Error, Result, Jurisdiction, TaxYear};
use serde::{Deserialize, Serialize};

// =============================================================================
// Entity Types
// =============================================================================

/// Types of business entities.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
    /// Individual taxpayer (default)
    Individual,

    /// Sole proprietorship (Schedule C)
    SoleProprietorship,

    /// General or limited partnership
    Partnership,

    /// S-Corporation (pass-through)
    SCorporation,

    /// C-Corporation (double taxation)
    CCorporation,

    /// Limited Liability Company (flexible taxation)
    LLC,

    /// Trust or estate
    Trust,

    /// Non-profit organization
    NonProfit,
}

impl EntityType {
    /// Get the display name for this entity type.
    pub fn name(&self) -> &'static str {
        match self {
            EntityType::Individual => "Individual",
            EntityType::SoleProprietorship => "Sole Proprietorship",
            EntityType::Partnership => "Partnership",
            EntityType::SCorporation => "S-Corporation",
            EntityType::CCorporation => "C-Corporation",
            EntityType::LLC => "Limited Liability Company",
            EntityType::Trust => "Trust/Estate",
            EntityType::NonProfit => "Non-Profit Organization",
        }
    }

    /// Check if this is a pass-through entity.
    pub fn is_pass_through(&self) -> bool {
        matches!(
            self,
            EntityType::Individual
                | EntityType::SoleProprietorship
                | EntityType::Partnership
                | EntityType::SCorporation
                | EntityType::LLC
        )
    }

    /// Check if this entity type is subject to self-employment tax.
    pub fn subject_to_se_tax(&self) -> bool {
        matches!(
            self,
            EntityType::SoleProprietorship | EntityType::Partnership
        )
    }

    /// Check if this entity type requires payroll for owners.
    pub fn requires_owner_payroll(&self) -> bool {
        matches!(self, EntityType::SCorporation | EntityType::CCorporation)
    }

    /// Get the IRS form used for this entity type.
    pub fn irs_form(&self) -> &'static str {
        match self {
            EntityType::Individual => "1040",
            EntityType::SoleProprietorship => "1040 Schedule C",
            EntityType::Partnership => "1065",
            EntityType::SCorporation => "1120-S",
            EntityType::CCorporation => "1120",
            EntityType::LLC => "Varies by election",
            EntityType::Trust => "1041",
            EntityType::NonProfit => "990",
        }
    }
}

impl Default for EntityType {
    fn default() -> Self {
        EntityType::Individual
    }
}

// =============================================================================
// Business Tax Information
// =============================================================================

/// Tax information for a business entity.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntityTaxInfo {
    /// Type of business entity
    pub entity_type: EntityType,

    /// Gross revenue / receipts
    pub gross_revenue: u64,

    /// Total deductions (COGS, expenses, depreciation, etc.)
    pub deductions: u64,

    /// Owner salary (for S-Corp/C-Corp)
    pub reasonable_salary: Option<u64>,

    /// Distributions to owners (non-salary)
    pub distributions: Option<u64>,

    /// Qualified Business Income (for 199A deduction)
    pub qbi: Option<u64>,

    /// Number of owners/shareholders
    pub owner_count: Option<u32>,

    /// State of incorporation
    pub state: Option<String>,
}

impl EntityTaxInfo {
    /// Create new tax info for an individual.
    pub fn individual(gross_income: u64, deductions: u64) -> Self {
        Self {
            entity_type: EntityType::Individual,
            gross_revenue: gross_income,
            deductions,
            reasonable_salary: None,
            distributions: None,
            qbi: None,
            owner_count: Some(1),
            state: None,
        }
    }

    /// Create new tax info for a sole proprietorship.
    pub fn sole_prop(gross_revenue: u64, deductions: u64) -> Self {
        Self {
            entity_type: EntityType::SoleProprietorship,
            gross_revenue,
            deductions,
            reasonable_salary: None,
            distributions: None,
            qbi: Some(gross_revenue.saturating_sub(deductions)),
            owner_count: Some(1),
            state: None,
        }
    }

    /// Create new tax info for an S-Corporation.
    pub fn s_corp(
        gross_revenue: u64,
        deductions: u64,
        owner_salary: u64,
        distributions: u64,
    ) -> Self {
        Self {
            entity_type: EntityType::SCorporation,
            gross_revenue,
            deductions,
            reasonable_salary: Some(owner_salary),
            distributions: Some(distributions),
            qbi: Some(distributions), // QBI = distributions for S-Corp
            owner_count: Some(1),
            state: None,
        }
    }

    /// Create new tax info for a C-Corporation.
    pub fn c_corp(gross_revenue: u64, deductions: u64) -> Self {
        Self {
            entity_type: EntityType::CCorporation,
            gross_revenue,
            deductions,
            reasonable_salary: None,
            distributions: None,
            qbi: None, // No QBI for C-Corp
            owner_count: None,
            state: None,
        }
    }

    /// Calculate net income.
    pub fn net_income(&self) -> u64 {
        self.gross_revenue.saturating_sub(self.deductions)
    }

    /// Calculate taxable income (varies by entity type).
    pub fn taxable_income(&self) -> u64 {
        match self.entity_type {
            EntityType::Individual => self.net_income(),
            EntityType::SoleProprietorship => self.net_income(),
            EntityType::Partnership => self.net_income(),
            EntityType::SCorporation => {
                // S-Corp: salary taxed as wages, distributions as pass-through
                self.reasonable_salary.unwrap_or(0) + self.distributions.unwrap_or(0)
            }
            EntityType::CCorporation => {
                // C-Corp: corporate tax on net income
                self.net_income()
            }
            EntityType::LLC => self.net_income(), // Depends on election
            EntityType::Trust => self.net_income(),
            EntityType::NonProfit => 0, // Generally tax-exempt
        }
    }
}

// =============================================================================
// Business Tax Proof
// =============================================================================

/// A proof of business tax bracket/liability.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BusinessTaxProof {
    /// Entity type
    pub entity_type: EntityType,

    /// Jurisdiction
    pub jurisdiction: Jurisdiction,

    /// Tax year
    pub tax_year: TaxYear,

    /// Net income (revenue - deductions)
    pub net_income: u64,

    /// Taxable income after adjustments
    pub taxable_income: u64,

    /// Marginal tax rate in basis points
    pub marginal_rate_bps: u16,

    /// Effective tax rate in basis points
    pub effective_rate_bps: u16,

    /// Self-employment tax amount (if applicable)
    pub se_tax: Option<u64>,

    /// QBI deduction amount (if applicable)
    pub qbi_deduction: Option<u64>,

    /// Corporate tax (for C-Corp)
    pub corporate_tax: Option<u64>,

    /// Commitment hash
    pub commitment: [u8; 32],

    /// Receipt bytes (ZK proof)
    pub receipt_bytes: Vec<u8>,
}

impl BusinessTaxProof {
    /// Check if this entity benefits from pass-through taxation.
    pub fn is_pass_through(&self) -> bool {
        self.entity_type.is_pass_through()
    }

    /// Calculate total estimated tax liability.
    pub fn estimated_tax(&self) -> u64 {
        let base_tax = (self.taxable_income as u128 * self.effective_rate_bps as u128 / 10000) as u64;
        let se_tax = self.se_tax.unwrap_or(0);
        let corp_tax = self.corporate_tax.unwrap_or(0);

        base_tax + se_tax + corp_tax
    }

    /// Verify the proof.
    pub fn verify(&self) -> Result<()> {
        // Basic validation
        if self.net_income == 0 && self.taxable_income > 0 {
            return Err(Error::invalid_proof("Taxable income exceeds net income"));
        }

        if self.marginal_rate_bps > 10000 {
            return Err(Error::invalid_proof("Marginal rate exceeds 100%"));
        }

        if self.effective_rate_bps > self.marginal_rate_bps {
            return Err(Error::invalid_proof("Effective rate exceeds marginal rate"));
        }

        Ok(())
    }
}

// =============================================================================
// Business Prover
// =============================================================================

/// Prover for business tax proofs.
pub struct BusinessProver {
    dev_mode: bool,
}

impl BusinessProver {
    /// Create a new prover in production mode.
    pub fn new() -> Self {
        Self { dev_mode: false }
    }

    /// Create a prover in development mode.
    pub fn dev_mode() -> Self {
        Self { dev_mode: true }
    }

    /// Generate a business tax proof.
    pub fn prove(
        &self,
        info: EntityTaxInfo,
        jurisdiction: Jurisdiction,
        tax_year: TaxYear,
    ) -> Result<BusinessTaxProof> {
        let net_income = info.net_income();
        let taxable_income = info.taxable_income();

        // Calculate rates based on entity type
        let (marginal_rate_bps, effective_rate_bps) = self.calculate_rates(
            &info,
            jurisdiction,
            tax_year,
        )?;

        // Calculate SE tax if applicable
        let se_tax = if info.entity_type.subject_to_se_tax() {
            Some(self.calculate_se_tax(net_income))
        } else {
            None
        };

        // Calculate QBI deduction if applicable
        let qbi_deduction = if info.entity_type.is_pass_through() && info.qbi.is_some() {
            Some(self.calculate_qbi_deduction(info.qbi.unwrap(), taxable_income))
        } else {
            None
        };

        // Calculate corporate tax for C-Corp
        let corporate_tax = if info.entity_type == EntityType::CCorporation {
            Some(self.calculate_corporate_tax(net_income, tax_year))
        } else {
            None
        };

        // Generate commitment
        let commitment = self.compute_commitment(&info, tax_year);

        // Generate proof receipt
        let receipt_bytes = if self.dev_mode {
            vec![0xDE, 0xAD, 0xBE, 0xEF]
        } else {
            // Would use Risc0 in production
            vec![]
        };

        Ok(BusinessTaxProof {
            entity_type: info.entity_type,
            jurisdiction,
            tax_year,
            net_income,
            taxable_income,
            marginal_rate_bps,
            effective_rate_bps,
            se_tax,
            qbi_deduction,
            corporate_tax,
            commitment,
            receipt_bytes,
        })
    }

    /// Calculate marginal and effective tax rates.
    fn calculate_rates(
        &self,
        info: &EntityTaxInfo,
        jurisdiction: Jurisdiction,
        tax_year: TaxYear,
    ) -> Result<(u16, u16)> {
        let taxable = info.taxable_income();

        // Simplified rate calculation (would use actual brackets in production)
        let marginal = match info.entity_type {
            EntityType::CCorporation => 2100, // 21% flat corporate rate
            _ => {
                // Progressive individual rates
                if taxable < 11_000 {
                    1000
                } else if taxable < 44_725 {
                    1200
                } else if taxable < 95_375 {
                    2200
                } else if taxable < 182_100 {
                    2400
                } else if taxable < 231_250 {
                    3200
                } else if taxable < 578_125 {
                    3500
                } else {
                    3700
                }
            }
        };

        // Effective rate is typically lower due to progressive brackets
        let effective = marginal * 7 / 10; // Rough approximation

        Ok((marginal, effective))
    }

    /// Calculate self-employment tax.
    fn calculate_se_tax(&self, net_income: u64) -> u64 {
        // 2024 SE tax: 15.3% on 92.35% of net income, up to SS wage base
        let se_income = (net_income as u128 * 9235 / 10000) as u64;
        let ss_wage_base = 168_600; // 2024

        let ss_income = se_income.min(ss_wage_base);
        let medicare_income = se_income;

        // 12.4% Social Security + 2.9% Medicare
        let ss_tax = (ss_income as u128 * 1240 / 10000) as u64;
        let medicare_tax = (medicare_income as u128 * 290 / 10000) as u64;

        ss_tax + medicare_tax
    }

    /// Calculate QBI deduction (Section 199A).
    fn calculate_qbi_deduction(&self, qbi: u64, taxable_income: u64) -> u64 {
        // Simplified: 20% of QBI, limited by taxable income
        let qbi_deduction = qbi / 5; // 20%

        // Phase-out thresholds (2024)
        let threshold = 182_100; // Single filer threshold

        if taxable_income <= threshold {
            qbi_deduction
        } else {
            // Phase-out calculation (simplified)
            let excess = taxable_income - threshold;
            let phase_out_range = 50_000;
            let reduction = qbi_deduction * excess.min(phase_out_range) / phase_out_range;
            qbi_deduction.saturating_sub(reduction)
        }
    }

    /// Calculate corporate tax (C-Corp).
    fn calculate_corporate_tax(&self, net_income: u64, _tax_year: TaxYear) -> u64 {
        // Flat 21% corporate rate (since TCJA 2017)
        (net_income as u128 * 21 / 100) as u64
    }

    /// Compute commitment hash.
    fn compute_commitment(&self, info: &EntityTaxInfo, tax_year: TaxYear) -> [u8; 32] {
        use sha2::{Sha256, Digest};

        let mut hasher = Sha256::new();
        hasher.update(&(info.entity_type as u8).to_le_bytes());
        hasher.update(&info.gross_revenue.to_le_bytes());
        hasher.update(&info.deductions.to_le_bytes());
        hasher.update(&tax_year.to_le_bytes());

        let result = hasher.finalize();
        let mut commitment = [0u8; 32];
        commitment.copy_from_slice(&result);
        commitment
    }
}

impl Default for BusinessProver {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_types() {
        assert!(EntityType::SoleProprietorship.is_pass_through());
        assert!(EntityType::SCorporation.is_pass_through());
        assert!(!EntityType::CCorporation.is_pass_through());
    }

    #[test]
    fn test_sole_prop_proof() {
        let prover = BusinessProver::dev_mode();
        let info = EntityTaxInfo::sole_prop(150_000, 30_000);

        let proof = prover.prove(info, Jurisdiction::US, 2024).unwrap();

        assert_eq!(proof.net_income, 120_000);
        assert!(proof.se_tax.is_some());
        assert!(proof.qbi_deduction.is_some());
    }

    #[test]
    fn test_s_corp_proof() {
        let prover = BusinessProver::dev_mode();
        let info = EntityTaxInfo::s_corp(500_000, 100_000, 80_000, 120_000);

        let proof = prover.prove(info, Jurisdiction::US, 2024).unwrap();

        assert_eq!(proof.entity_type, EntityType::SCorporation);
        assert!(proof.se_tax.is_none()); // S-Corp distributions not subject to SE tax
        assert!(proof.qbi_deduction.is_some());
    }

    #[test]
    fn test_c_corp_proof() {
        let prover = BusinessProver::dev_mode();
        let info = EntityTaxInfo::c_corp(1_000_000, 200_000);

        let proof = prover.prove(info, Jurisdiction::US, 2024).unwrap();

        assert_eq!(proof.entity_type, EntityType::CCorporation);
        assert!(proof.corporate_tax.is_some());
        assert!(proof.qbi_deduction.is_none()); // No QBI for C-Corp
    }

    #[test]
    fn test_se_tax_calculation() {
        let prover = BusinessProver::dev_mode();
        let se_tax = prover.calculate_se_tax(100_000);

        // Should be approximately 15.3% of 92.35% of net income
        // 100,000 * 0.9235 * 0.153 ≈ 14,130
        assert!(se_tax > 10_000 && se_tax < 20_000);
    }
}
