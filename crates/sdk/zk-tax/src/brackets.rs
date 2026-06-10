// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tax bracket definitions for all supported jurisdictions.

use crate::jurisdiction::{FilingStatus, Jurisdiction};
use crate::types::{BracketCommitment, TaxYear};
use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// A single tax bracket definition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaxBracket {
    /// Bracket index (0-based)
    pub index: u8,
    /// Lower bound of income (inclusive)
    pub lower: u64,
    /// Upper bound of income (exclusive)
    pub upper: u64,
    /// Marginal rate in basis points (e.g., 2200 = 22%)
    pub rate_bps: u16,
}

impl TaxBracket {
    /// Check if an income falls within this bracket.
    pub fn contains(&self, income: u64) -> bool {
        income >= self.lower && income < self.upper
    }

    /// Get the marginal rate as a percentage.
    pub fn rate_percent(&self) -> f64 {
        self.rate_bps as f64 / 100.0
    }

    /// Compute the cryptographic commitment for this bracket.
    pub fn commitment(&self, tax_year: TaxYear) -> BracketCommitment {
        compute_commitment(self.lower, self.upper, tax_year)
    }
}

impl std::fmt::Display for TaxBracket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.upper == u64::MAX {
            write!(f, "Bracket {}: ${}+ at {}%",
                   self.index,
                   format_currency(self.lower),
                   self.rate_percent())
        } else {
            write!(f, "Bracket {}: ${}-${} at {}%",
                   self.index,
                   format_currency(self.lower),
                   format_currency(self.upper),
                   self.rate_percent())
        }
    }
}

/// Get tax brackets for a jurisdiction, year, and filing status.
pub fn get_brackets(
    jurisdiction: Jurisdiction,
    year: TaxYear,
    status: FilingStatus,
) -> Result<&'static [TaxBracket]> {
    match (jurisdiction, year, status) {
        // US 2020
        (Jurisdiction::US, 2020, FilingStatus::Single) => Ok(&US_2020_SINGLE),
        (Jurisdiction::US, 2020, FilingStatus::MarriedFilingJointly) => Ok(&US_2020_MFJ),
        (Jurisdiction::US, 2020, FilingStatus::MarriedFilingSeparately) => Ok(&US_2020_SINGLE), // Same as single
        (Jurisdiction::US, 2020, FilingStatus::HeadOfHousehold) => Ok(&US_2020_SINGLE), // Fallback

        // US 2021
        (Jurisdiction::US, 2021, FilingStatus::Single) => Ok(&US_2021_SINGLE),
        (Jurisdiction::US, 2021, FilingStatus::MarriedFilingJointly) => Ok(&US_2021_MFJ),
        (Jurisdiction::US, 2021, FilingStatus::MarriedFilingSeparately) => Ok(&US_2021_SINGLE),
        (Jurisdiction::US, 2021, FilingStatus::HeadOfHousehold) => Ok(&US_2021_SINGLE),

        // US 2022
        (Jurisdiction::US, 2022, FilingStatus::Single) => Ok(&US_2022_SINGLE),
        (Jurisdiction::US, 2022, FilingStatus::MarriedFilingJointly) => Ok(&US_2022_MFJ),
        (Jurisdiction::US, 2022, FilingStatus::MarriedFilingSeparately) => Ok(&US_2022_SINGLE),
        (Jurisdiction::US, 2022, FilingStatus::HeadOfHousehold) => Ok(&US_2022_SINGLE),

        // US 2023
        (Jurisdiction::US, 2023, FilingStatus::Single) => Ok(&US_2023_SINGLE),
        (Jurisdiction::US, 2023, FilingStatus::MarriedFilingJointly) => Ok(&US_2023_MFJ),
        (Jurisdiction::US, 2023, FilingStatus::MarriedFilingSeparately) => Ok(&US_2023_SINGLE),
        (Jurisdiction::US, 2023, FilingStatus::HeadOfHousehold) => Ok(&US_2023_SINGLE),

        // US 2024
        (Jurisdiction::US, 2024, FilingStatus::Single) => Ok(&US_2024_SINGLE),
        (Jurisdiction::US, 2024, FilingStatus::MarriedFilingJointly) => Ok(&US_2024_MFJ),
        (Jurisdiction::US, 2024, FilingStatus::MarriedFilingSeparately) => Ok(&US_2024_MFS),
        (Jurisdiction::US, 2024, FilingStatus::HeadOfHousehold) => Ok(&US_2024_HOH),

        // US 2025
        (Jurisdiction::US, 2025, FilingStatus::Single) => Ok(&US_2025_SINGLE),
        (Jurisdiction::US, 2025, FilingStatus::MarriedFilingJointly) => Ok(&US_2025_MFJ),
        (Jurisdiction::US, 2025, FilingStatus::MarriedFilingSeparately) => Ok(&US_2025_MFS),
        (Jurisdiction::US, 2025, FilingStatus::HeadOfHousehold) => Ok(&US_2025_HOH),

        // UK 2020-2023
        (Jurisdiction::UK, 2020, FilingStatus::Single) => Ok(&UK_2020),
        (Jurisdiction::UK, 2021, FilingStatus::Single) => Ok(&UK_2021),
        (Jurisdiction::UK, 2022, FilingStatus::Single) => Ok(&UK_2022),
        (Jurisdiction::UK, 2023, FilingStatus::Single) => Ok(&UK_2023),

        // UK 2024/2025
        (Jurisdiction::UK, 2024, FilingStatus::Single) => Ok(&UK_2024),
        (Jurisdiction::UK, 2025, FilingStatus::Single) => Ok(&UK_2025),

        // Canada 2020-2025 (Federal rates - using closest available)
        (Jurisdiction::CA, 2020..=2023, FilingStatus::Single) => Ok(&CA_2024), // Approximate
        (Jurisdiction::CA, 2024, FilingStatus::Single) => Ok(&CA_2024),
        (Jurisdiction::CA, 2025, FilingStatus::Single) => Ok(&CA_2025),

        // Germany 2020-2025
        (Jurisdiction::DE, 2020..=2023, FilingStatus::Single) => Ok(&DE_2024_SINGLE),
        (Jurisdiction::DE, 2020..=2023, FilingStatus::MarriedFilingJointly) => Ok(&DE_2024_MFJ),
        (Jurisdiction::DE, 2024, FilingStatus::Single) => Ok(&DE_2024_SINGLE),
        (Jurisdiction::DE, 2024, FilingStatus::MarriedFilingJointly) => Ok(&DE_2024_MFJ),
        (Jurisdiction::DE, 2025, FilingStatus::Single) => Ok(&DE_2025_SINGLE),
        (Jurisdiction::DE, 2025, FilingStatus::MarriedFilingJointly) => Ok(&DE_2025_MFJ),

        // Australia 2020-2025
        (Jurisdiction::AU, 2020..=2023, FilingStatus::Single) => Ok(&AU_2024),
        (Jurisdiction::AU, 2024, FilingStatus::Single) => Ok(&AU_2024),
        (Jurisdiction::AU, 2025, FilingStatus::Single) => Ok(&AU_2025),

        // France 2020-2025
        (Jurisdiction::FR, 2020..=2023, FilingStatus::Single) => Ok(&FR_2024),
        (Jurisdiction::FR, 2024, FilingStatus::Single) => Ok(&FR_2024),
        (Jurisdiction::FR, 2025, FilingStatus::Single) => Ok(&FR_2025),

        // Japan 2020-2025
        (Jurisdiction::JP, 2020..=2023, FilingStatus::Single) => Ok(&JP_2024),
        (Jurisdiction::JP, 2024, FilingStatus::Single) => Ok(&JP_2024),
        (Jurisdiction::JP, 2025, FilingStatus::Single) => Ok(&JP_2025),

        // India 2020-2025 (New Tax Regime)
        (Jurisdiction::IN, 2020..=2023, FilingStatus::Single) => Ok(&IN_2024),
        (Jurisdiction::IN, 2024, FilingStatus::Single) => Ok(&IN_2024),
        (Jurisdiction::IN, 2025, FilingStatus::Single) => Ok(&IN_2025),

        // Brazil 2020-2025
        (Jurisdiction::BR, 2020..=2023, FilingStatus::Single) => Ok(&BR_2024),
        (Jurisdiction::BR, 2024, FilingStatus::Single) => Ok(&BR_2024),
        (Jurisdiction::BR, 2025, FilingStatus::Single) => Ok(&BR_2025),

        // Argentina 2020-2025
        (Jurisdiction::AR, 2020..=2023, FilingStatus::Single) => Ok(&AR_2024),
        (Jurisdiction::AR, 2024, FilingStatus::Single) => Ok(&AR_2024),
        (Jurisdiction::AR, 2025, FilingStatus::Single) => Ok(&AR_2025),

        // Mexico 2020-2025
        (Jurisdiction::MX, 2020..=2023, FilingStatus::Single) => Ok(&MX_2024),
        (Jurisdiction::MX, 2024, FilingStatus::Single) => Ok(&MX_2024),
        (Jurisdiction::MX, 2025, FilingStatus::Single) => Ok(&MX_2025),

        // Italy 2020-2025
        (Jurisdiction::IT, 2020..=2023, FilingStatus::Single) => Ok(&IT_2024),
        (Jurisdiction::IT, 2024, FilingStatus::Single) => Ok(&IT_2024),
        (Jurisdiction::IT, 2025, FilingStatus::Single) => Ok(&IT_2025),

        // Russia 2020-2025
        (Jurisdiction::RU, 2020..=2023, FilingStatus::Single) => Ok(&RU_2024),
        (Jurisdiction::RU, 2024, FilingStatus::Single) => Ok(&RU_2024),
        (Jurisdiction::RU, 2025, FilingStatus::Single) => Ok(&RU_2025),

        // Turkey 2020-2025
        (Jurisdiction::TR, 2020..=2023, FilingStatus::Single) => Ok(&TR_2024),
        (Jurisdiction::TR, 2024, FilingStatus::Single) => Ok(&TR_2024),
        (Jurisdiction::TR, 2025, FilingStatus::Single) => Ok(&TR_2025),

        // China 2020-2025
        (Jurisdiction::CN, 2020..=2023, FilingStatus::Single) => Ok(&CN_2024),
        (Jurisdiction::CN, 2024, FilingStatus::Single) => Ok(&CN_2024),
        (Jurisdiction::CN, 2025, FilingStatus::Single) => Ok(&CN_2025),

        // South Korea 2020-2025
        (Jurisdiction::KR, 2020..=2023, FilingStatus::Single) => Ok(&KR_2024),
        (Jurisdiction::KR, 2024, FilingStatus::Single) => Ok(&KR_2024),
        (Jurisdiction::KR, 2025, FilingStatus::Single) => Ok(&KR_2025),

        // Indonesia 2020-2025
        (Jurisdiction::ID, 2020..=2023, FilingStatus::Single) => Ok(&ID_2024),
        (Jurisdiction::ID, 2024, FilingStatus::Single) => Ok(&ID_2024),
        (Jurisdiction::ID, 2025, FilingStatus::Single) => Ok(&ID_2025),

        // Saudi Arabia 2020-2025 (no personal income tax)
        (Jurisdiction::SA, 2020..=2023, FilingStatus::Single) => Ok(&SA_2024),
        (Jurisdiction::SA, 2024, FilingStatus::Single) => Ok(&SA_2024),
        (Jurisdiction::SA, 2025, FilingStatus::Single) => Ok(&SA_2025),

        // South Africa 2020-2025
        (Jurisdiction::ZA, 2020..=2023, FilingStatus::Single) => Ok(&ZA_2024),
        (Jurisdiction::ZA, 2024, FilingStatus::Single) => Ok(&ZA_2024),
        (Jurisdiction::ZA, 2025, FilingStatus::Single) => Ok(&ZA_2025),

        // =========================================================================
        // Americas (Extended)
        // =========================================================================

        // Chile 2020-2025
        (Jurisdiction::CL, 2020..=2023, FilingStatus::Single) => Ok(&CL_2024),
        (Jurisdiction::CL, 2024, FilingStatus::Single) => Ok(&CL_2024),
        (Jurisdiction::CL, 2025, FilingStatus::Single) => Ok(&CL_2025),

        // Colombia 2020-2025
        (Jurisdiction::CO, 2020..=2023, FilingStatus::Single) => Ok(&CO_2024),
        (Jurisdiction::CO, 2024, FilingStatus::Single) => Ok(&CO_2024),
        (Jurisdiction::CO, 2025, FilingStatus::Single) => Ok(&CO_2025),

        // Peru 2020-2025
        (Jurisdiction::PE, 2020..=2023, FilingStatus::Single) => Ok(&PE_2024),
        (Jurisdiction::PE, 2024, FilingStatus::Single) => Ok(&PE_2024),
        (Jurisdiction::PE, 2025, FilingStatus::Single) => Ok(&PE_2025),

        // Ecuador 2020-2025
        (Jurisdiction::EC, 2020..=2023, FilingStatus::Single) => Ok(&EC_2024),
        (Jurisdiction::EC, 2024, FilingStatus::Single) => Ok(&EC_2024),
        (Jurisdiction::EC, 2025, FilingStatus::Single) => Ok(&EC_2025),

        // Uruguay 2020-2025
        (Jurisdiction::UY, 2020..=2023, FilingStatus::Single) => Ok(&UY_2024),
        (Jurisdiction::UY, 2024, FilingStatus::Single) => Ok(&UY_2024),
        (Jurisdiction::UY, 2025, FilingStatus::Single) => Ok(&UY_2025),

        // =========================================================================
        // Europe (EU Member States)
        // =========================================================================

        // Spain 2020-2025
        (Jurisdiction::ES, 2020..=2023, FilingStatus::Single) => Ok(&ES_2024),
        (Jurisdiction::ES, 2024, FilingStatus::Single) => Ok(&ES_2024),
        (Jurisdiction::ES, 2025, FilingStatus::Single) => Ok(&ES_2025),

        // Netherlands 2020-2025
        (Jurisdiction::NL, 2020..=2023, FilingStatus::Single) => Ok(&NL_2024),
        (Jurisdiction::NL, 2024, FilingStatus::Single) => Ok(&NL_2024),
        (Jurisdiction::NL, 2025, FilingStatus::Single) => Ok(&NL_2025),

        // Belgium 2020-2025
        (Jurisdiction::BE, 2020..=2023, FilingStatus::Single) => Ok(&BE_2024),
        (Jurisdiction::BE, 2024, FilingStatus::Single) => Ok(&BE_2024),
        (Jurisdiction::BE, 2025, FilingStatus::Single) => Ok(&BE_2025),

        // Austria 2020-2025
        (Jurisdiction::AT, 2020..=2023, FilingStatus::Single) => Ok(&AT_2024),
        (Jurisdiction::AT, 2024, FilingStatus::Single) => Ok(&AT_2024),
        (Jurisdiction::AT, 2025, FilingStatus::Single) => Ok(&AT_2025),

        // Portugal 2020-2025
        (Jurisdiction::PT, 2020..=2023, FilingStatus::Single) => Ok(&PT_2024),
        (Jurisdiction::PT, 2024, FilingStatus::Single) => Ok(&PT_2024),
        (Jurisdiction::PT, 2025, FilingStatus::Single) => Ok(&PT_2025),

        // Ireland 2020-2025
        (Jurisdiction::IE, 2020..=2023, FilingStatus::Single) => Ok(&IE_2024),
        (Jurisdiction::IE, 2024, FilingStatus::Single) => Ok(&IE_2024),
        (Jurisdiction::IE, 2025, FilingStatus::Single) => Ok(&IE_2025),

        // Poland 2020-2025
        (Jurisdiction::PL, 2020..=2023, FilingStatus::Single) => Ok(&PL_2024),
        (Jurisdiction::PL, 2024, FilingStatus::Single) => Ok(&PL_2024),
        (Jurisdiction::PL, 2025, FilingStatus::Single) => Ok(&PL_2025),

        // Sweden 2020-2025
        (Jurisdiction::SE, 2020..=2023, FilingStatus::Single) => Ok(&SE_2024),
        (Jurisdiction::SE, 2024, FilingStatus::Single) => Ok(&SE_2024),
        (Jurisdiction::SE, 2025, FilingStatus::Single) => Ok(&SE_2025),

        // Denmark 2020-2025
        (Jurisdiction::DK, 2020..=2023, FilingStatus::Single) => Ok(&DK_2024),
        (Jurisdiction::DK, 2024, FilingStatus::Single) => Ok(&DK_2024),
        (Jurisdiction::DK, 2025, FilingStatus::Single) => Ok(&DK_2025),

        // Finland 2020-2025
        (Jurisdiction::FI, 2020..=2023, FilingStatus::Single) => Ok(&FI_2024),
        (Jurisdiction::FI, 2024, FilingStatus::Single) => Ok(&FI_2024),
        (Jurisdiction::FI, 2025, FilingStatus::Single) => Ok(&FI_2025),

        // Norway 2020-2025
        (Jurisdiction::NO, 2020..=2023, FilingStatus::Single) => Ok(&NO_2024),
        (Jurisdiction::NO, 2024, FilingStatus::Single) => Ok(&NO_2024),
        (Jurisdiction::NO, 2025, FilingStatus::Single) => Ok(&NO_2025),

        // Switzerland 2020-2025
        (Jurisdiction::CH, 2020..=2023, FilingStatus::Single) => Ok(&CH_2024),
        (Jurisdiction::CH, 2024, FilingStatus::Single) => Ok(&CH_2024),
        (Jurisdiction::CH, 2025, FilingStatus::Single) => Ok(&CH_2025),

        // Czech Republic 2020-2025
        (Jurisdiction::CZ, 2020..=2023, FilingStatus::Single) => Ok(&CZ_2024),
        (Jurisdiction::CZ, 2024, FilingStatus::Single) => Ok(&CZ_2024),
        (Jurisdiction::CZ, 2025, FilingStatus::Single) => Ok(&CZ_2025),

        // Greece 2020-2025
        (Jurisdiction::GR, 2020..=2023, FilingStatus::Single) => Ok(&GR_2024),
        (Jurisdiction::GR, 2024, FilingStatus::Single) => Ok(&GR_2024),
        (Jurisdiction::GR, 2025, FilingStatus::Single) => Ok(&GR_2025),

        // Hungary 2020-2025 (flat 15%)
        (Jurisdiction::HU, 2020..=2023, FilingStatus::Single) => Ok(&HU_2024),
        (Jurisdiction::HU, 2024, FilingStatus::Single) => Ok(&HU_2024),
        (Jurisdiction::HU, 2025, FilingStatus::Single) => Ok(&HU_2025),

        // Romania 2020-2025 (flat 10%)
        (Jurisdiction::RO, 2020..=2023, FilingStatus::Single) => Ok(&RO_2024),
        (Jurisdiction::RO, 2024, FilingStatus::Single) => Ok(&RO_2024),
        (Jurisdiction::RO, 2025, FilingStatus::Single) => Ok(&RO_2025),

        // =========================================================================
        // Europe (Non-EU)
        // =========================================================================

        // Ukraine 2020-2025 (flat 18%)
        (Jurisdiction::UA, 2020..=2023, FilingStatus::Single) => Ok(&UA_2024),
        (Jurisdiction::UA, 2024, FilingStatus::Single) => Ok(&UA_2024),
        (Jurisdiction::UA, 2025, FilingStatus::Single) => Ok(&UA_2025),

        // =========================================================================
        // Asia-Pacific
        // =========================================================================

        // New Zealand 2020-2025
        (Jurisdiction::NZ, 2020..=2023, FilingStatus::Single) => Ok(&NZ_2024),
        (Jurisdiction::NZ, 2024, FilingStatus::Single) => Ok(&NZ_2024),
        (Jurisdiction::NZ, 2025, FilingStatus::Single) => Ok(&NZ_2025),

        // Singapore 2020-2025
        (Jurisdiction::SG, 2020..=2023, FilingStatus::Single) => Ok(&SG_2024),
        (Jurisdiction::SG, 2024, FilingStatus::Single) => Ok(&SG_2024),
        (Jurisdiction::SG, 2025, FilingStatus::Single) => Ok(&SG_2025),

        // Hong Kong 2020-2025
        (Jurisdiction::HK, 2020..=2023, FilingStatus::Single) => Ok(&HK_2024),
        (Jurisdiction::HK, 2024, FilingStatus::Single) => Ok(&HK_2024),
        (Jurisdiction::HK, 2025, FilingStatus::Single) => Ok(&HK_2025),

        // Taiwan 2020-2025
        (Jurisdiction::TW, 2020..=2023, FilingStatus::Single) => Ok(&TW_2024),
        (Jurisdiction::TW, 2024, FilingStatus::Single) => Ok(&TW_2024),
        (Jurisdiction::TW, 2025, FilingStatus::Single) => Ok(&TW_2025),

        // Malaysia 2020-2025
        (Jurisdiction::MY, 2020..=2023, FilingStatus::Single) => Ok(&MY_2024),
        (Jurisdiction::MY, 2024, FilingStatus::Single) => Ok(&MY_2024),
        (Jurisdiction::MY, 2025, FilingStatus::Single) => Ok(&MY_2025),

        // Thailand 2020-2025
        (Jurisdiction::TH, 2020..=2023, FilingStatus::Single) => Ok(&TH_2024),
        (Jurisdiction::TH, 2024, FilingStatus::Single) => Ok(&TH_2024),
        (Jurisdiction::TH, 2025, FilingStatus::Single) => Ok(&TH_2025),

        // Vietnam 2020-2025
        (Jurisdiction::VN, 2020..=2023, FilingStatus::Single) => Ok(&VN_2024),
        (Jurisdiction::VN, 2024, FilingStatus::Single) => Ok(&VN_2024),
        (Jurisdiction::VN, 2025, FilingStatus::Single) => Ok(&VN_2025),

        // Philippines 2020-2025
        (Jurisdiction::PH, 2020..=2023, FilingStatus::Single) => Ok(&PH_2024),
        (Jurisdiction::PH, 2024, FilingStatus::Single) => Ok(&PH_2024),
        (Jurisdiction::PH, 2025, FilingStatus::Single) => Ok(&PH_2025),

        // Pakistan 2020-2025
        (Jurisdiction::PK, 2020..=2023, FilingStatus::Single) => Ok(&PK_2024),
        (Jurisdiction::PK, 2024, FilingStatus::Single) => Ok(&PK_2024),
        (Jurisdiction::PK, 2025, FilingStatus::Single) => Ok(&PK_2025),

        // =========================================================================
        // Middle East
        // =========================================================================

        // UAE 2020-2025 (0% personal income tax)
        (Jurisdiction::AE, 2020..=2023, FilingStatus::Single) => Ok(&AE_2024),
        (Jurisdiction::AE, 2024, FilingStatus::Single) => Ok(&AE_2024),
        (Jurisdiction::AE, 2025, FilingStatus::Single) => Ok(&AE_2025),

        // Israel 2020-2025
        (Jurisdiction::IL, 2020..=2023, FilingStatus::Single) => Ok(&IL_2024),
        (Jurisdiction::IL, 2024, FilingStatus::Single) => Ok(&IL_2024),
        (Jurisdiction::IL, 2025, FilingStatus::Single) => Ok(&IL_2025),

        // Egypt 2020-2025
        (Jurisdiction::EG, 2020..=2023, FilingStatus::Single) => Ok(&EG_2024),
        (Jurisdiction::EG, 2024, FilingStatus::Single) => Ok(&EG_2024),
        (Jurisdiction::EG, 2025, FilingStatus::Single) => Ok(&EG_2025),

        // Qatar 2020-2025 (0% personal income tax)
        (Jurisdiction::QA, 2020..=2023, FilingStatus::Single) => Ok(&QA_2024),
        (Jurisdiction::QA, 2024, FilingStatus::Single) => Ok(&QA_2024),
        (Jurisdiction::QA, 2025, FilingStatus::Single) => Ok(&QA_2025),

        // =========================================================================
        // Africa
        // =========================================================================

        // Nigeria 2020-2025
        (Jurisdiction::NG, 2020..=2023, FilingStatus::Single) => Ok(&NG_2024),
        (Jurisdiction::NG, 2024, FilingStatus::Single) => Ok(&NG_2024),
        (Jurisdiction::NG, 2025, FilingStatus::Single) => Ok(&NG_2025),

        // Kenya 2020-2025
        (Jurisdiction::KE, 2020..=2023, FilingStatus::Single) => Ok(&KE_2024),
        (Jurisdiction::KE, 2024, FilingStatus::Single) => Ok(&KE_2024),
        (Jurisdiction::KE, 2025, FilingStatus::Single) => Ok(&KE_2025),

        // Morocco 2020-2025
        (Jurisdiction::MA, 2020..=2023, FilingStatus::Single) => Ok(&MA_2024),
        (Jurisdiction::MA, 2024, FilingStatus::Single) => Ok(&MA_2024),
        (Jurisdiction::MA, 2025, FilingStatus::Single) => Ok(&MA_2025),

        // Ghana 2020-2025
        (Jurisdiction::GH, 2020..=2023, FilingStatus::Single) => Ok(&GH_2024),
        (Jurisdiction::GH, 2024, FilingStatus::Single) => Ok(&GH_2024),
        (Jurisdiction::GH, 2025, FilingStatus::Single) => Ok(&GH_2025),

        // Unsupported tax year (2020-2025 supported)
        (_, year, _) if !(2020..=2025).contains(&year) => {
            Err(Error::unsupported_year(year))
        }

        // Invalid filing status for jurisdiction
        (j, _, status) if !j.is_valid_filing_status(status) => {
            let valid: Vec<&str> = j.valid_filing_statuses()
                .iter()
                .map(|s| s.code())
                .collect();
            Err(Error::invalid_filing_status(
                status.name(),
                j.name(),
                &valid,
            ))
        }

        _ => Err(Error::unsupported_jurisdiction(jurisdiction.name())),
    }
}

/// Find the bracket for a given income.
pub fn find_bracket(
    income: u64,
    jurisdiction: Jurisdiction,
    year: TaxYear,
    status: FilingStatus,
) -> Result<&'static TaxBracket> {
    let brackets = get_brackets(jurisdiction, year, status)?;
    brackets
        .iter()
        .find(|b| b.contains(income))
        .ok_or(Error::no_bracket(income))
}

/// Compute deterministic commitment to bracket parameters.
///
/// Uses FNV-1a hash expanded to 32 bytes - matches guest code exactly.
pub fn compute_commitment(lower: u64, upper: u64, tax_year: TaxYear) -> BracketCommitment {
    // FNV-1a hash (must match guest code exactly!)
    let mut hash: u64 = 0xcbf29ce484222325;
    let prime: u64 = 0x100000001b3;

    // Hash lower bound
    for byte in lower.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(prime);
    }

    // Hash upper bound
    for byte in upper.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(prime);
    }

    // Hash tax year
    for byte in tax_year.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(prime);
    }

    // Expand to 32 bytes by running additional rounds
    let mut commitment = [0u8; 32];
    let mut h = hash;
    for chunk in commitment.chunks_mut(8) {
        chunk.copy_from_slice(&h.to_le_bytes());
        h = h.wrapping_mul(prime);
    }

    BracketCommitment::from_bytes(commitment)
}

fn format_currency(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

// =============================================================================
// US TAX BRACKETS
// =============================================================================

/// US 2020 Single Filer (IRS Revenue Procedure 2019-44)
pub static US_2020_SINGLE: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,       upper: 9_875,    rate_bps: 1000 },
    TaxBracket { index: 1, lower: 9_875,   upper: 40_125,   rate_bps: 1200 },
    TaxBracket { index: 2, lower: 40_125,  upper: 85_525,   rate_bps: 2200 },
    TaxBracket { index: 3, lower: 85_525,  upper: 163_300,  rate_bps: 2400 },
    TaxBracket { index: 4, lower: 163_300, upper: 207_350,  rate_bps: 3200 },
    TaxBracket { index: 5, lower: 207_350, upper: 518_400,  rate_bps: 3500 },
    TaxBracket { index: 6, lower: 518_400, upper: u64::MAX, rate_bps: 3700 },
];

/// US 2020 Married Filing Jointly
pub static US_2020_MFJ: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,        upper: 19_750,   rate_bps: 1000 },
    TaxBracket { index: 1, lower: 19_750,   upper: 80_250,   rate_bps: 1200 },
    TaxBracket { index: 2, lower: 80_250,   upper: 171_050,  rate_bps: 2200 },
    TaxBracket { index: 3, lower: 171_050,  upper: 326_600,  rate_bps: 2400 },
    TaxBracket { index: 4, lower: 326_600,  upper: 414_700,  rate_bps: 3200 },
    TaxBracket { index: 5, lower: 414_700,  upper: 622_050,  rate_bps: 3500 },
    TaxBracket { index: 6, lower: 622_050,  upper: u64::MAX, rate_bps: 3700 },
];

/// US 2021 Single Filer (IRS Revenue Procedure 2020-45)
pub static US_2021_SINGLE: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,       upper: 9_950,    rate_bps: 1000 },
    TaxBracket { index: 1, lower: 9_950,   upper: 40_525,   rate_bps: 1200 },
    TaxBracket { index: 2, lower: 40_525,  upper: 86_375,   rate_bps: 2200 },
    TaxBracket { index: 3, lower: 86_375,  upper: 164_925,  rate_bps: 2400 },
    TaxBracket { index: 4, lower: 164_925, upper: 209_425,  rate_bps: 3200 },
    TaxBracket { index: 5, lower: 209_425, upper: 523_600,  rate_bps: 3500 },
    TaxBracket { index: 6, lower: 523_600, upper: u64::MAX, rate_bps: 3700 },
];

/// US 2021 Married Filing Jointly
pub static US_2021_MFJ: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,        upper: 19_900,   rate_bps: 1000 },
    TaxBracket { index: 1, lower: 19_900,   upper: 81_050,   rate_bps: 1200 },
    TaxBracket { index: 2, lower: 81_050,   upper: 172_750,  rate_bps: 2200 },
    TaxBracket { index: 3, lower: 172_750,  upper: 329_850,  rate_bps: 2400 },
    TaxBracket { index: 4, lower: 329_850,  upper: 418_850,  rate_bps: 3200 },
    TaxBracket { index: 5, lower: 418_850,  upper: 628_300,  rate_bps: 3500 },
    TaxBracket { index: 6, lower: 628_300,  upper: u64::MAX, rate_bps: 3700 },
];

/// US 2022 Single Filer (IRS Revenue Procedure 2021-45)
pub static US_2022_SINGLE: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,       upper: 10_275,   rate_bps: 1000 },
    TaxBracket { index: 1, lower: 10_275,  upper: 41_775,   rate_bps: 1200 },
    TaxBracket { index: 2, lower: 41_775,  upper: 89_075,   rate_bps: 2200 },
    TaxBracket { index: 3, lower: 89_075,  upper: 170_050,  rate_bps: 2400 },
    TaxBracket { index: 4, lower: 170_050, upper: 215_950,  rate_bps: 3200 },
    TaxBracket { index: 5, lower: 215_950, upper: 539_900,  rate_bps: 3500 },
    TaxBracket { index: 6, lower: 539_900, upper: u64::MAX, rate_bps: 3700 },
];

/// US 2022 Married Filing Jointly
pub static US_2022_MFJ: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,        upper: 20_550,   rate_bps: 1000 },
    TaxBracket { index: 1, lower: 20_550,   upper: 83_550,   rate_bps: 1200 },
    TaxBracket { index: 2, lower: 83_550,   upper: 178_150,  rate_bps: 2200 },
    TaxBracket { index: 3, lower: 178_150,  upper: 340_100,  rate_bps: 2400 },
    TaxBracket { index: 4, lower: 340_100,  upper: 431_900,  rate_bps: 3200 },
    TaxBracket { index: 5, lower: 431_900,  upper: 647_850,  rate_bps: 3500 },
    TaxBracket { index: 6, lower: 647_850,  upper: u64::MAX, rate_bps: 3700 },
];

/// US 2023 Single Filer (IRS Revenue Procedure 2022-38)
pub static US_2023_SINGLE: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,       upper: 11_000,   rate_bps: 1000 },
    TaxBracket { index: 1, lower: 11_000,  upper: 44_725,   rate_bps: 1200 },
    TaxBracket { index: 2, lower: 44_725,  upper: 95_375,   rate_bps: 2200 },
    TaxBracket { index: 3, lower: 95_375,  upper: 182_100,  rate_bps: 2400 },
    TaxBracket { index: 4, lower: 182_100, upper: 231_250,  rate_bps: 3200 },
    TaxBracket { index: 5, lower: 231_250, upper: 578_125,  rate_bps: 3500 },
    TaxBracket { index: 6, lower: 578_125, upper: u64::MAX, rate_bps: 3700 },
];

/// US 2023 Married Filing Jointly
pub static US_2023_MFJ: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,        upper: 22_000,   rate_bps: 1000 },
    TaxBracket { index: 1, lower: 22_000,   upper: 89_450,   rate_bps: 1200 },
    TaxBracket { index: 2, lower: 89_450,   upper: 190_750,  rate_bps: 2200 },
    TaxBracket { index: 3, lower: 190_750,  upper: 364_200,  rate_bps: 2400 },
    TaxBracket { index: 4, lower: 364_200,  upper: 462_500,  rate_bps: 3200 },
    TaxBracket { index: 5, lower: 462_500,  upper: 693_750,  rate_bps: 3500 },
    TaxBracket { index: 6, lower: 693_750,  upper: u64::MAX, rate_bps: 3700 },
];

/// US 2024 Single Filer (IRS Revenue Procedure 2023-34)
pub static US_2024_SINGLE: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,       upper: 11_600,   rate_bps: 1000 },
    TaxBracket { index: 1, lower: 11_600,  upper: 47_150,   rate_bps: 1200 },
    TaxBracket { index: 2, lower: 47_150,  upper: 100_525,  rate_bps: 2200 },
    TaxBracket { index: 3, lower: 100_525, upper: 191_950,  rate_bps: 2400 },
    TaxBracket { index: 4, lower: 191_950, upper: 243_725,  rate_bps: 3200 },
    TaxBracket { index: 5, lower: 243_725, upper: 609_350,  rate_bps: 3500 },
    TaxBracket { index: 6, lower: 609_350, upper: u64::MAX, rate_bps: 3700 },
];

/// US 2024 Married Filing Jointly
pub static US_2024_MFJ: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,        upper: 23_200,   rate_bps: 1000 },
    TaxBracket { index: 1, lower: 23_200,   upper: 94_300,   rate_bps: 1200 },
    TaxBracket { index: 2, lower: 94_300,   upper: 201_050,  rate_bps: 2200 },
    TaxBracket { index: 3, lower: 201_050,  upper: 383_900,  rate_bps: 2400 },
    TaxBracket { index: 4, lower: 383_900,  upper: 487_450,  rate_bps: 3200 },
    TaxBracket { index: 5, lower: 487_450,  upper: 731_200,  rate_bps: 3500 },
    TaxBracket { index: 6, lower: 731_200,  upper: u64::MAX, rate_bps: 3700 },
];

/// US 2024 Married Filing Separately
pub static US_2024_MFS: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,        upper: 11_600,   rate_bps: 1000 },
    TaxBracket { index: 1, lower: 11_600,   upper: 47_150,   rate_bps: 1200 },
    TaxBracket { index: 2, lower: 47_150,   upper: 100_525,  rate_bps: 2200 },
    TaxBracket { index: 3, lower: 100_525,  upper: 191_950,  rate_bps: 2400 },
    TaxBracket { index: 4, lower: 191_950,  upper: 243_725,  rate_bps: 3200 },
    TaxBracket { index: 5, lower: 243_725,  upper: 365_600,  rate_bps: 3500 },
    TaxBracket { index: 6, lower: 365_600,  upper: u64::MAX, rate_bps: 3700 },
];

/// US 2024 Head of Household
pub static US_2024_HOH: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,        upper: 16_550,   rate_bps: 1000 },
    TaxBracket { index: 1, lower: 16_550,   upper: 63_100,   rate_bps: 1200 },
    TaxBracket { index: 2, lower: 63_100,   upper: 100_500,  rate_bps: 2200 },
    TaxBracket { index: 3, lower: 100_500,  upper: 191_950,  rate_bps: 2400 },
    TaxBracket { index: 4, lower: 191_950,  upper: 243_700,  rate_bps: 3200 },
    TaxBracket { index: 5, lower: 243_700,  upper: 609_350,  rate_bps: 3500 },
    TaxBracket { index: 6, lower: 609_350,  upper: u64::MAX, rate_bps: 3700 },
];

/// US 2025 Single Filer (IRS Revenue Procedure 2024-40)
pub static US_2025_SINGLE: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,        upper: 11_925,   rate_bps: 1000 },
    TaxBracket { index: 1, lower: 11_925,   upper: 48_475,   rate_bps: 1200 },
    TaxBracket { index: 2, lower: 48_475,   upper: 103_350,  rate_bps: 2200 },
    TaxBracket { index: 3, lower: 103_350,  upper: 197_300,  rate_bps: 2400 },
    TaxBracket { index: 4, lower: 197_300,  upper: 250_525,  rate_bps: 3200 },
    TaxBracket { index: 5, lower: 250_525,  upper: 626_350,  rate_bps: 3500 },
    TaxBracket { index: 6, lower: 626_350,  upper: u64::MAX, rate_bps: 3700 },
];

/// US 2025 Married Filing Jointly
pub static US_2025_MFJ: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,        upper: 23_850,   rate_bps: 1000 },
    TaxBracket { index: 1, lower: 23_850,   upper: 96_950,   rate_bps: 1200 },
    TaxBracket { index: 2, lower: 96_950,   upper: 206_700,  rate_bps: 2200 },
    TaxBracket { index: 3, lower: 206_700,  upper: 394_600,  rate_bps: 2400 },
    TaxBracket { index: 4, lower: 394_600,  upper: 501_050,  rate_bps: 3200 },
    TaxBracket { index: 5, lower: 501_050,  upper: 751_600,  rate_bps: 3500 },
    TaxBracket { index: 6, lower: 751_600,  upper: u64::MAX, rate_bps: 3700 },
];

/// US 2025 Married Filing Separately
pub static US_2025_MFS: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,        upper: 11_925,   rate_bps: 1000 },
    TaxBracket { index: 1, lower: 11_925,   upper: 48_475,   rate_bps: 1200 },
    TaxBracket { index: 2, lower: 48_475,   upper: 103_350,  rate_bps: 2200 },
    TaxBracket { index: 3, lower: 103_350,  upper: 197_300,  rate_bps: 2400 },
    TaxBracket { index: 4, lower: 197_300,  upper: 250_525,  rate_bps: 3200 },
    TaxBracket { index: 5, lower: 250_525,  upper: 375_800,  rate_bps: 3500 },
    TaxBracket { index: 6, lower: 375_800,  upper: u64::MAX, rate_bps: 3700 },
];

/// US 2025 Head of Household
pub static US_2025_HOH: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,        upper: 17_000,   rate_bps: 1000 },
    TaxBracket { index: 1, lower: 17_000,   upper: 64_850,   rate_bps: 1200 },
    TaxBracket { index: 2, lower: 64_850,   upper: 103_350,  rate_bps: 2200 },
    TaxBracket { index: 3, lower: 103_350,  upper: 197_300,  rate_bps: 2400 },
    TaxBracket { index: 4, lower: 197_300,  upper: 250_500,  rate_bps: 3200 },
    TaxBracket { index: 5, lower: 250_500,  upper: 626_350,  rate_bps: 3500 },
    TaxBracket { index: 6, lower: 626_350,  upper: u64::MAX, rate_bps: 3700 },
];

// =============================================================================
// UK TAX BRACKETS
// =============================================================================

/// UK 2020/21 Tax Bands (HMRC)
pub static UK_2020: [TaxBracket; 4] = [
    TaxBracket { index: 0, lower: 0,        upper: 12_500,   rate_bps: 0 },     // Personal Allowance
    TaxBracket { index: 1, lower: 12_500,   upper: 50_000,   rate_bps: 2000 },  // Basic rate 20%
    TaxBracket { index: 2, lower: 50_000,   upper: 150_000,  rate_bps: 4000 },  // Higher rate 40%
    TaxBracket { index: 3, lower: 150_000,  upper: u64::MAX, rate_bps: 4500 },  // Additional rate 45%
];

/// UK 2021/22 Tax Bands (HMRC)
pub static UK_2021: [TaxBracket; 4] = [
    TaxBracket { index: 0, lower: 0,        upper: 12_570,   rate_bps: 0 },     // Personal Allowance
    TaxBracket { index: 1, lower: 12_570,   upper: 50_270,   rate_bps: 2000 },  // Basic rate 20%
    TaxBracket { index: 2, lower: 50_270,   upper: 150_000,  rate_bps: 4000 },  // Higher rate 40%
    TaxBracket { index: 3, lower: 150_000,  upper: u64::MAX, rate_bps: 4500 },  // Additional rate 45%
];

/// UK 2022/23 Tax Bands (HMRC)
pub static UK_2022: [TaxBracket; 4] = [
    TaxBracket { index: 0, lower: 0,        upper: 12_570,   rate_bps: 0 },     // Personal Allowance
    TaxBracket { index: 1, lower: 12_570,   upper: 50_270,   rate_bps: 2000 },  // Basic rate 20%
    TaxBracket { index: 2, lower: 50_270,   upper: 150_000,  rate_bps: 4000 },  // Higher rate 40%
    TaxBracket { index: 3, lower: 150_000,  upper: u64::MAX, rate_bps: 4500 },  // Additional rate 45%
];

/// UK 2023/24 Tax Bands (HMRC)
pub static UK_2023: [TaxBracket; 4] = [
    TaxBracket { index: 0, lower: 0,        upper: 12_570,   rate_bps: 0 },     // Personal Allowance
    TaxBracket { index: 1, lower: 12_570,   upper: 50_270,   rate_bps: 2000 },  // Basic rate 20%
    TaxBracket { index: 2, lower: 50_270,   upper: 125_140,  rate_bps: 4000 },  // Higher rate 40%
    TaxBracket { index: 3, lower: 125_140,  upper: u64::MAX, rate_bps: 4500 },  // Additional rate 45%
];

/// UK 2024/25 Tax Bands (HMRC)
/// Note: UK uses tax years April-April, "2024" means 2024/25
pub static UK_2024: [TaxBracket; 4] = [
    TaxBracket { index: 0, lower: 0,        upper: 12_570,   rate_bps: 0 },     // Personal Allowance
    TaxBracket { index: 1, lower: 12_570,   upper: 50_270,   rate_bps: 2000 },  // Basic rate 20%
    TaxBracket { index: 2, lower: 50_270,   upper: 125_140,  rate_bps: 4000 },  // Higher rate 40%
    TaxBracket { index: 3, lower: 125_140,  upper: u64::MAX, rate_bps: 4500 },  // Additional rate 45%
];

/// UK 2025/26 Tax Bands (HMRC - frozen)
pub static UK_2025: [TaxBracket; 4] = [
    TaxBracket { index: 0, lower: 0,        upper: 12_570,   rate_bps: 0 },     // Personal Allowance
    TaxBracket { index: 1, lower: 12_570,   upper: 50_270,   rate_bps: 2000 },  // Basic rate 20%
    TaxBracket { index: 2, lower: 50_270,   upper: 125_140,  rate_bps: 4000 },  // Higher rate 40%
    TaxBracket { index: 3, lower: 125_140,  upper: u64::MAX, rate_bps: 4500 },  // Additional rate 45%
];

// =============================================================================
// CANADA TAX BRACKETS (Federal only, in CAD)
// =============================================================================

/// Canada 2024 Federal Tax Brackets (CRA)
pub static CA_2024: [TaxBracket; 5] = [
    TaxBracket { index: 0, lower: 0,        upper: 55_867,   rate_bps: 1500 },  // 15%
    TaxBracket { index: 1, lower: 55_867,   upper: 111_733,  rate_bps: 2050 },  // 20.5%
    TaxBracket { index: 2, lower: 111_733,  upper: 173_205,  rate_bps: 2600 },  // 26%
    TaxBracket { index: 3, lower: 173_205,  upper: 246_752,  rate_bps: 2900 },  // 29%
    TaxBracket { index: 4, lower: 246_752,  upper: u64::MAX, rate_bps: 3300 },  // 33%
];

/// Canada 2025 Federal Tax Brackets (CRA - inflation adjusted)
pub static CA_2025: [TaxBracket; 5] = [
    TaxBracket { index: 0, lower: 0,        upper: 57_375,   rate_bps: 1500 },  // 15%
    TaxBracket { index: 1, lower: 57_375,   upper: 114_750,  rate_bps: 2050 },  // 20.5%
    TaxBracket { index: 2, lower: 114_750,  upper: 177_882,  rate_bps: 2600 },  // 26%
    TaxBracket { index: 3, lower: 177_882,  upper: 253_414,  rate_bps: 2900 },  // 29%
    TaxBracket { index: 4, lower: 253_414,  upper: u64::MAX, rate_bps: 3300 },  // 33%
];

// =============================================================================
// GERMANY TAX BRACKETS (in EUR)
// =============================================================================

/// Germany 2024 Tax Zones (Einkommensteuer - Single)
/// Note: Germany uses progressive formulas, these are simplified brackets
pub static DE_2024_SINGLE: [TaxBracket; 5] = [
    TaxBracket { index: 0, lower: 0,        upper: 11_604,   rate_bps: 0 },     // Grundfreibetrag (tax-free)
    TaxBracket { index: 1, lower: 11_604,   upper: 17_005,   rate_bps: 1400 },  // Zone 2: 14-24%
    TaxBracket { index: 2, lower: 17_005,   upper: 66_760,   rate_bps: 2400 },  // Zone 3: 24-42%
    TaxBracket { index: 3, lower: 66_760,   upper: 277_825,  rate_bps: 4200 },  // Zone 4: 42%
    TaxBracket { index: 4, lower: 277_825,  upper: u64::MAX, rate_bps: 4500 },  // Zone 5: 45% (Reichensteuer)
];

/// Germany 2024 Tax Zones (Married - Splitting Tariff)
pub static DE_2024_MFJ: [TaxBracket; 5] = [
    TaxBracket { index: 0, lower: 0,        upper: 23_208,   rate_bps: 0 },     // 2x Grundfreibetrag
    TaxBracket { index: 1, lower: 23_208,   upper: 34_010,   rate_bps: 1400 },  // Zone 2
    TaxBracket { index: 2, lower: 34_010,   upper: 133_520,  rate_bps: 2400 },  // Zone 3
    TaxBracket { index: 3, lower: 133_520,  upper: 555_650,  rate_bps: 4200 },  // Zone 4
    TaxBracket { index: 4, lower: 555_650,  upper: u64::MAX, rate_bps: 4500 },  // Zone 5
];

/// Germany 2025 Tax Zones (Single - inflation adjusted)
pub static DE_2025_SINGLE: [TaxBracket; 5] = [
    TaxBracket { index: 0, lower: 0,        upper: 12_096,   rate_bps: 0 },     // Grundfreibetrag
    TaxBracket { index: 1, lower: 12_096,   upper: 17_443,   rate_bps: 1400 },  // Zone 2
    TaxBracket { index: 2, lower: 17_443,   upper: 68_480,   rate_bps: 2400 },  // Zone 3
    TaxBracket { index: 3, lower: 68_480,   upper: 277_825,  rate_bps: 4200 },  // Zone 4
    TaxBracket { index: 4, lower: 277_825,  upper: u64::MAX, rate_bps: 4500 },  // Zone 5
];

/// Germany 2025 Tax Zones (Married - Splitting)
pub static DE_2025_MFJ: [TaxBracket; 5] = [
    TaxBracket { index: 0, lower: 0,        upper: 24_192,   rate_bps: 0 },     // 2x Grundfreibetrag
    TaxBracket { index: 1, lower: 24_192,   upper: 34_886,   rate_bps: 1400 },  // Zone 2
    TaxBracket { index: 2, lower: 34_886,   upper: 136_960,  rate_bps: 2400 },  // Zone 3
    TaxBracket { index: 3, lower: 136_960,  upper: 555_650,  rate_bps: 4200 },  // Zone 4
    TaxBracket { index: 4, lower: 555_650,  upper: u64::MAX, rate_bps: 4500 },  // Zone 5
];

// =============================================================================
// AUSTRALIA TAX BRACKETS (in AUD)
// =============================================================================

/// Australia 2024 Tax Brackets (ATO - Resident rates)
pub static AU_2024: [TaxBracket; 5] = [
    TaxBracket { index: 0, lower: 0,        upper: 18_200,   rate_bps: 0 },     // Tax-free threshold
    TaxBracket { index: 1, lower: 18_200,   upper: 45_000,   rate_bps: 1900 },  // 19%
    TaxBracket { index: 2, lower: 45_000,   upper: 120_000,  rate_bps: 3250 },  // 32.5%
    TaxBracket { index: 3, lower: 120_000,  upper: 180_000,  rate_bps: 3700 },  // 37%
    TaxBracket { index: 4, lower: 180_000,  upper: u64::MAX, rate_bps: 4500 },  // 45%
];

/// Australia 2025 Tax Brackets (ATO - Stage 3 tax cuts)
pub static AU_2025: [TaxBracket; 5] = [
    TaxBracket { index: 0, lower: 0,        upper: 18_200,   rate_bps: 0 },     // Tax-free threshold
    TaxBracket { index: 1, lower: 18_200,   upper: 45_000,   rate_bps: 1600 },  // 16% (reduced)
    TaxBracket { index: 2, lower: 45_000,   upper: 135_000,  rate_bps: 3000 },  // 30% (reduced)
    TaxBracket { index: 3, lower: 135_000,  upper: 190_000,  rate_bps: 3700 },  // 37%
    TaxBracket { index: 4, lower: 190_000,  upper: u64::MAX, rate_bps: 4500 },  // 45%
];

// =============================================================================
// FRANCE TAX BRACKETS (in EUR)
// =============================================================================

/// France 2024 Income Tax Brackets (DGFiP - Barème progressif)
/// Note: Using single "part" (célibataire sans enfant)
pub static FR_2024: [TaxBracket; 5] = [
    TaxBracket { index: 0, lower: 0,        upper: 11_294,   rate_bps: 0 },     // 0%
    TaxBracket { index: 1, lower: 11_294,   upper: 28_797,   rate_bps: 1100 },  // 11%
    TaxBracket { index: 2, lower: 28_797,   upper: 82_341,   rate_bps: 3000 },  // 30%
    TaxBracket { index: 3, lower: 82_341,   upper: 177_106,  rate_bps: 4100 },  // 41%
    TaxBracket { index: 4, lower: 177_106,  upper: u64::MAX, rate_bps: 4500 },  // 45%
];

/// France 2025 Income Tax Brackets (inflation adjusted)
pub static FR_2025: [TaxBracket; 5] = [
    TaxBracket { index: 0, lower: 0,        upper: 11_520,   rate_bps: 0 },     // 0%
    TaxBracket { index: 1, lower: 11_520,   upper: 29_373,   rate_bps: 1100 },  // 11%
    TaxBracket { index: 2, lower: 29_373,   upper: 83_988,   rate_bps: 3000 },  // 30%
    TaxBracket { index: 3, lower: 83_988,   upper: 180_648,  rate_bps: 4100 },  // 41%
    TaxBracket { index: 4, lower: 180_648,  upper: u64::MAX, rate_bps: 4500 },  // 45%
];

// =============================================================================
// JAPAN TAX BRACKETS (in JPY)
// =============================================================================

/// Japan 2024 Income Tax Brackets (NTA - Shotokuzei)
/// Note: Excludes local inhabitant tax (10%)
pub static JP_2024: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,           upper: 1_950_000,   rate_bps: 500 },   // 5%
    TaxBracket { index: 1, lower: 1_950_000,   upper: 3_300_000,   rate_bps: 1000 },  // 10%
    TaxBracket { index: 2, lower: 3_300_000,   upper: 6_950_000,   rate_bps: 2000 },  // 20%
    TaxBracket { index: 3, lower: 6_950_000,   upper: 9_000_000,   rate_bps: 2300 },  // 23%
    TaxBracket { index: 4, lower: 9_000_000,   upper: 18_000_000,  rate_bps: 3300 },  // 33%
    TaxBracket { index: 5, lower: 18_000_000,  upper: 40_000_000,  rate_bps: 4000 },  // 40%
    TaxBracket { index: 6, lower: 40_000_000,  upper: u64::MAX,    rate_bps: 4500 },  // 45%
];

/// Japan 2025 Income Tax Brackets (same as 2024)
pub static JP_2025: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,           upper: 1_950_000,   rate_bps: 500 },   // 5%
    TaxBracket { index: 1, lower: 1_950_000,   upper: 3_300_000,   rate_bps: 1000 },  // 10%
    TaxBracket { index: 2, lower: 3_300_000,   upper: 6_950_000,   rate_bps: 2000 },  // 20%
    TaxBracket { index: 3, lower: 6_950_000,   upper: 9_000_000,   rate_bps: 2300 },  // 23%
    TaxBracket { index: 4, lower: 9_000_000,   upper: 18_000_000,  rate_bps: 3300 },  // 33%
    TaxBracket { index: 5, lower: 18_000_000,  upper: 40_000_000,  rate_bps: 4000 },  // 40%
    TaxBracket { index: 6, lower: 40_000_000,  upper: u64::MAX,    rate_bps: 4500 },  // 45%
];

// =============================================================================
// INDIA TAX BRACKETS (in INR - New Tax Regime)
// =============================================================================

/// India 2024-25 Income Tax - New Tax Regime (IT Department)
/// FY 2024-25 (AY 2025-26)
pub static IN_2024: [TaxBracket; 6] = [
    TaxBracket { index: 0, lower: 0,          upper: 300_000,    rate_bps: 0 },     // Nil
    TaxBracket { index: 1, lower: 300_000,    upper: 700_000,    rate_bps: 500 },   // 5%
    TaxBracket { index: 2, lower: 700_000,    upper: 1_000_000,  rate_bps: 1000 },  // 10%
    TaxBracket { index: 3, lower: 1_000_000,  upper: 1_200_000,  rate_bps: 1500 },  // 15%
    TaxBracket { index: 4, lower: 1_200_000,  upper: 1_500_000,  rate_bps: 2000 },  // 20%
    TaxBracket { index: 5, lower: 1_500_000,  upper: u64::MAX,   rate_bps: 3000 },  // 30%
];

/// India 2025-26 Income Tax - New Tax Regime (projected same)
pub static IN_2025: [TaxBracket; 6] = [
    TaxBracket { index: 0, lower: 0,          upper: 300_000,    rate_bps: 0 },     // Nil
    TaxBracket { index: 1, lower: 300_000,    upper: 700_000,    rate_bps: 500 },   // 5%
    TaxBracket { index: 2, lower: 700_000,    upper: 1_000_000,  rate_bps: 1000 },  // 10%
    TaxBracket { index: 3, lower: 1_000_000,  upper: 1_200_000,  rate_bps: 1500 },  // 15%
    TaxBracket { index: 4, lower: 1_200_000,  upper: 1_500_000,  rate_bps: 2000 },  // 20%
    TaxBracket { index: 5, lower: 1_500_000,  upper: u64::MAX,   rate_bps: 3000 },  // 30%
];

// =============================================================================
// BRAZIL TAX BRACKETS (in BRL)
// =============================================================================

/// Brazil 2024 IRPF Tax Brackets (Receita Federal)
pub static BR_2024: [TaxBracket; 5] = [
    TaxBracket { index: 0, lower: 0,          upper: 24_511,     rate_bps: 0 },     // Isento
    TaxBracket { index: 1, lower: 24_511,     upper: 33_919,     rate_bps: 750 },   // 7.5%
    TaxBracket { index: 2, lower: 33_919,     upper: 45_012,     rate_bps: 1500 },  // 15%
    TaxBracket { index: 3, lower: 45_012,     upper: 55_976,     rate_bps: 2250 },  // 22.5%
    TaxBracket { index: 4, lower: 55_976,     upper: u64::MAX,   rate_bps: 2750 },  // 27.5%
];

/// Brazil 2025 IRPF Tax Brackets (inflation adjusted)
pub static BR_2025: [TaxBracket; 5] = [
    TaxBracket { index: 0, lower: 0,          upper: 26_963,     rate_bps: 0 },     // Isento (adjusted)
    TaxBracket { index: 1, lower: 26_963,     upper: 37_311,     rate_bps: 750 },   // 7.5%
    TaxBracket { index: 2, lower: 37_311,     upper: 49_513,     rate_bps: 1500 },  // 15%
    TaxBracket { index: 3, lower: 49_513,     upper: 61_574,     rate_bps: 2250 },  // 22.5%
    TaxBracket { index: 4, lower: 61_574,     upper: u64::MAX,   rate_bps: 2750 },  // 27.5%
];

// =============================================================================
// ARGENTINA TAX BRACKETS (in ARS - Note: High inflation, brackets update frequently)
// =============================================================================

/// Argentina 2024 Income Tax Brackets (AFIP - Impuesto a las Ganancias)
pub static AR_2024: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,             upper: 2_475_479,     rate_bps: 500 },   // 5%
    TaxBracket { index: 1, lower: 2_475_479,     upper: 4_950_958,     rate_bps: 900 },   // 9%
    TaxBracket { index: 2, lower: 4_950_958,     upper: 7_426_437,     rate_bps: 1200 },  // 12%
    TaxBracket { index: 3, lower: 7_426_437,     upper: 9_901_916,     rate_bps: 1500 },  // 15%
    TaxBracket { index: 4, lower: 9_901_916,     upper: 14_852_874,    rate_bps: 1900 },  // 19%
    TaxBracket { index: 5, lower: 14_852_874,    upper: 19_803_832,    rate_bps: 2300 },  // 23%
    TaxBracket { index: 6, lower: 19_803_832,    upper: u64::MAX,      rate_bps: 3500 },  // 35%
];

/// Argentina 2025 Income Tax Brackets (inflation adjusted estimate)
pub static AR_2025: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,             upper: 3_713_219,     rate_bps: 500 },   // 5%
    TaxBracket { index: 1, lower: 3_713_219,     upper: 7_426_437,     rate_bps: 900 },   // 9%
    TaxBracket { index: 2, lower: 7_426_437,     upper: 11_139_656,    rate_bps: 1200 },  // 12%
    TaxBracket { index: 3, lower: 11_139_656,    upper: 14_852_874,    rate_bps: 1500 },  // 15%
    TaxBracket { index: 4, lower: 14_852_874,    upper: 22_279_311,    rate_bps: 1900 },  // 19%
    TaxBracket { index: 5, lower: 22_279_311,    upper: 29_705_748,    rate_bps: 2300 },  // 23%
    TaxBracket { index: 6, lower: 29_705_748,    upper: u64::MAX,      rate_bps: 3500 },  // 35%
];

// =============================================================================
// MEXICO TAX BRACKETS (in MXN)
// =============================================================================

/// Mexico 2024 ISR Tax Brackets (SAT)
pub static MX_2024: [TaxBracket; 8] = [
    TaxBracket { index: 0, lower: 0,          upper: 8_952,       rate_bps: 192 },   // 1.92%
    TaxBracket { index: 1, lower: 8_952,      upper: 75_984,      rate_bps: 640 },   // 6.4%
    TaxBracket { index: 2, lower: 75_984,     upper: 133_536,     rate_bps: 1088 },  // 10.88%
    TaxBracket { index: 3, lower: 133_536,    upper: 155_229,     rate_bps: 1600 },  // 16%
    TaxBracket { index: 4, lower: 155_229,    upper: 185_852,     rate_bps: 1792 },  // 17.92%
    TaxBracket { index: 5, lower: 185_852,    upper: 374_837,     rate_bps: 2136 },  // 21.36%
    TaxBracket { index: 6, lower: 374_837,    upper: 590_795,     rate_bps: 2394 },  // 23.94%
    TaxBracket { index: 7, lower: 590_795,    upper: u64::MAX,    rate_bps: 3500 },  // 35%
];

/// Mexico 2025 ISR Tax Brackets (inflation adjusted)
pub static MX_2025: [TaxBracket; 8] = [
    TaxBracket { index: 0, lower: 0,          upper: 9_400,       rate_bps: 192 },   // 1.92%
    TaxBracket { index: 1, lower: 9_400,      upper: 79_783,      rate_bps: 640 },   // 6.4%
    TaxBracket { index: 2, lower: 79_783,     upper: 140_213,     rate_bps: 1088 },  // 10.88%
    TaxBracket { index: 3, lower: 140_213,    upper: 162_990,     rate_bps: 1600 },  // 16%
    TaxBracket { index: 4, lower: 162_990,    upper: 195_145,     rate_bps: 1792 },  // 17.92%
    TaxBracket { index: 5, lower: 195_145,    upper: 393_579,     rate_bps: 2136 },  // 21.36%
    TaxBracket { index: 6, lower: 393_579,    upper: 620_335,     rate_bps: 2394 },  // 23.94%
    TaxBracket { index: 7, lower: 620_335,    upper: u64::MAX,    rate_bps: 3500 },  // 35%
];

// =============================================================================
// ITALY TAX BRACKETS (in EUR - IRPEF)
// =============================================================================

/// Italy 2024 IRPEF Tax Brackets (Agenzia delle Entrate)
pub static IT_2024: [TaxBracket; 4] = [
    TaxBracket { index: 0, lower: 0,        upper: 28_000,    rate_bps: 2300 },  // 23%
    TaxBracket { index: 1, lower: 28_000,   upper: 50_000,    rate_bps: 3500 },  // 35%
    TaxBracket { index: 2, lower: 50_000,   upper: u64::MAX,  rate_bps: 4300 },  // 43%
    TaxBracket { index: 3, lower: 0, upper: 0, rate_bps: 0 }, // Placeholder for array size
];

/// Italy 2025 IRPEF Tax Brackets (same structure)
pub static IT_2025: [TaxBracket; 4] = [
    TaxBracket { index: 0, lower: 0,        upper: 28_000,    rate_bps: 2300 },  // 23%
    TaxBracket { index: 1, lower: 28_000,   upper: 50_000,    rate_bps: 3500 },  // 35%
    TaxBracket { index: 2, lower: 50_000,   upper: u64::MAX,  rate_bps: 4300 },  // 43%
    TaxBracket { index: 3, lower: 0, upper: 0, rate_bps: 0 }, // Placeholder
];

// =============================================================================
// RUSSIA TAX BRACKETS (in RUB)
// =============================================================================

/// Russia 2024 NDFL Tax (FNS - progressive from 2021)
pub static RU_2024: [TaxBracket; 2] = [
    TaxBracket { index: 0, lower: 0,           upper: 5_000_000,  rate_bps: 1300 },  // 13%
    TaxBracket { index: 1, lower: 5_000_000,   upper: u64::MAX,   rate_bps: 1500 },  // 15%
];

/// Russia 2025 NDFL Tax (expanded brackets)
pub static RU_2025: [TaxBracket; 5] = [
    TaxBracket { index: 0, lower: 0,            upper: 2_400_000,   rate_bps: 1300 },  // 13%
    TaxBracket { index: 1, lower: 2_400_000,    upper: 5_000_000,   rate_bps: 1500 },  // 15%
    TaxBracket { index: 2, lower: 5_000_000,    upper: 20_000_000,  rate_bps: 1800 },  // 18%
    TaxBracket { index: 3, lower: 20_000_000,   upper: 50_000_000,  rate_bps: 2000 },  // 20%
    TaxBracket { index: 4, lower: 50_000_000,   upper: u64::MAX,    rate_bps: 2200 },  // 22%
];

// =============================================================================
// TURKEY TAX BRACKETS (in TRY)
// =============================================================================

/// Turkey 2024 Gelir Vergisi (GIB)
pub static TR_2024: [TaxBracket; 5] = [
    TaxBracket { index: 0, lower: 0,          upper: 110_000,     rate_bps: 1500 },  // 15%
    TaxBracket { index: 1, lower: 110_000,    upper: 230_000,     rate_bps: 2000 },  // 20%
    TaxBracket { index: 2, lower: 230_000,    upper: 580_000,     rate_bps: 2700 },  // 27%
    TaxBracket { index: 3, lower: 580_000,    upper: 3_000_000,   rate_bps: 3500 },  // 35%
    TaxBracket { index: 4, lower: 3_000_000,  upper: u64::MAX,    rate_bps: 4000 },  // 40%
];

/// Turkey 2025 Gelir Vergisi (inflation adjusted)
pub static TR_2025: [TaxBracket; 5] = [
    TaxBracket { index: 0, lower: 0,          upper: 158_000,     rate_bps: 1500 },  // 15%
    TaxBracket { index: 1, lower: 158_000,    upper: 330_000,     rate_bps: 2000 },  // 20%
    TaxBracket { index: 2, lower: 330_000,    upper: 800_000,     rate_bps: 2700 },  // 27%
    TaxBracket { index: 3, lower: 800_000,    upper: 4_300_000,   rate_bps: 3500 },  // 35%
    TaxBracket { index: 4, lower: 4_300_000,  upper: u64::MAX,    rate_bps: 4000 },  // 40%
];

// =============================================================================
// CHINA TAX BRACKETS (in CNY)
// =============================================================================

/// China 2024 IIT Tax Brackets (SAT - comprehensive income)
pub static CN_2024: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,          upper: 36_000,      rate_bps: 300 },   // 3%
    TaxBracket { index: 1, lower: 36_000,     upper: 144_000,     rate_bps: 1000 },  // 10%
    TaxBracket { index: 2, lower: 144_000,    upper: 300_000,     rate_bps: 2000 },  // 20%
    TaxBracket { index: 3, lower: 300_000,    upper: 420_000,     rate_bps: 2500 },  // 25%
    TaxBracket { index: 4, lower: 420_000,    upper: 660_000,     rate_bps: 3000 },  // 30%
    TaxBracket { index: 5, lower: 660_000,    upper: 960_000,     rate_bps: 3500 },  // 35%
    TaxBracket { index: 6, lower: 960_000,    upper: u64::MAX,    rate_bps: 4500 },  // 45%
];

/// China 2025 IIT Tax Brackets (same structure)
pub static CN_2025: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,          upper: 36_000,      rate_bps: 300 },   // 3%
    TaxBracket { index: 1, lower: 36_000,     upper: 144_000,     rate_bps: 1000 },  // 10%
    TaxBracket { index: 2, lower: 144_000,    upper: 300_000,     rate_bps: 2000 },  // 20%
    TaxBracket { index: 3, lower: 300_000,    upper: 420_000,     rate_bps: 2500 },  // 25%
    TaxBracket { index: 4, lower: 420_000,    upper: 660_000,     rate_bps: 3000 },  // 30%
    TaxBracket { index: 5, lower: 660_000,    upper: 960_000,     rate_bps: 3500 },  // 35%
    TaxBracket { index: 6, lower: 960_000,    upper: u64::MAX,    rate_bps: 4500 },  // 45%
];

// =============================================================================
// SOUTH KOREA TAX BRACKETS (in KRW)
// =============================================================================

/// South Korea 2024 Income Tax (NTS)
pub static KR_2024: [TaxBracket; 8] = [
    TaxBracket { index: 0, lower: 0,             upper: 14_000_000,    rate_bps: 600 },   // 6%
    TaxBracket { index: 1, lower: 14_000_000,    upper: 50_000_000,    rate_bps: 1500 },  // 15%
    TaxBracket { index: 2, lower: 50_000_000,    upper: 88_000_000,    rate_bps: 2400 },  // 24%
    TaxBracket { index: 3, lower: 88_000_000,    upper: 150_000_000,   rate_bps: 3500 },  // 35%
    TaxBracket { index: 4, lower: 150_000_000,   upper: 300_000_000,   rate_bps: 3800 },  // 38%
    TaxBracket { index: 5, lower: 300_000_000,   upper: 500_000_000,   rate_bps: 4000 },  // 40%
    TaxBracket { index: 6, lower: 500_000_000,   upper: 1_000_000_000, rate_bps: 4200 },  // 42%
    TaxBracket { index: 7, lower: 1_000_000_000, upper: u64::MAX,      rate_bps: 4500 },  // 45%
];

/// South Korea 2025 Income Tax (same structure)
pub static KR_2025: [TaxBracket; 8] = [
    TaxBracket { index: 0, lower: 0,             upper: 14_000_000,    rate_bps: 600 },   // 6%
    TaxBracket { index: 1, lower: 14_000_000,    upper: 50_000_000,    rate_bps: 1500 },  // 15%
    TaxBracket { index: 2, lower: 50_000_000,    upper: 88_000_000,    rate_bps: 2400 },  // 24%
    TaxBracket { index: 3, lower: 88_000_000,    upper: 150_000_000,   rate_bps: 3500 },  // 35%
    TaxBracket { index: 4, lower: 150_000_000,   upper: 300_000_000,   rate_bps: 3800 },  // 38%
    TaxBracket { index: 5, lower: 300_000_000,   upper: 500_000_000,   rate_bps: 4000 },  // 40%
    TaxBracket { index: 6, lower: 500_000_000,   upper: 1_000_000_000, rate_bps: 4200 },  // 42%
    TaxBracket { index: 7, lower: 1_000_000_000, upper: u64::MAX,      rate_bps: 4500 },  // 45%
];

// =============================================================================
// INDONESIA TAX BRACKETS (in IDR)
// =============================================================================

/// Indonesia 2024 PPh Tax Brackets (DJP)
pub static ID_2024: [TaxBracket; 5] = [
    TaxBracket { index: 0, lower: 0,             upper: 60_000_000,    rate_bps: 500 },   // 5%
    TaxBracket { index: 1, lower: 60_000_000,    upper: 250_000_000,   rate_bps: 1500 },  // 15%
    TaxBracket { index: 2, lower: 250_000_000,   upper: 500_000_000,   rate_bps: 2500 },  // 25%
    TaxBracket { index: 3, lower: 500_000_000,   upper: 5_000_000_000, rate_bps: 3000 },  // 30%
    TaxBracket { index: 4, lower: 5_000_000_000, upper: u64::MAX,      rate_bps: 3500 },  // 35%
];

/// Indonesia 2025 PPh Tax Brackets (same structure)
pub static ID_2025: [TaxBracket; 5] = [
    TaxBracket { index: 0, lower: 0,             upper: 60_000_000,    rate_bps: 500 },   // 5%
    TaxBracket { index: 1, lower: 60_000_000,    upper: 250_000_000,   rate_bps: 1500 },  // 15%
    TaxBracket { index: 2, lower: 250_000_000,   upper: 500_000_000,   rate_bps: 2500 },  // 25%
    TaxBracket { index: 3, lower: 500_000_000,   upper: 5_000_000_000, rate_bps: 3000 },  // 30%
    TaxBracket { index: 4, lower: 5_000_000_000, upper: u64::MAX,      rate_bps: 3500 },  // 35%
];

// =============================================================================
// SAUDI ARABIA (in SAR - No personal income tax, Zakat for nationals)
// =============================================================================

/// Saudi Arabia 2024 - No personal income tax (ZATCA)
/// Saudi nationals pay 2.5% Zakat on net worth, not income
/// Expatriates pay no income tax
pub static SA_2024: [TaxBracket; 1] = [
    TaxBracket { index: 0, lower: 0, upper: u64::MAX, rate_bps: 0 },  // 0% income tax
];

/// Saudi Arabia 2025 - No personal income tax
pub static SA_2025: [TaxBracket; 1] = [
    TaxBracket { index: 0, lower: 0, upper: u64::MAX, rate_bps: 0 },  // 0% income tax
];

// =============================================================================
// SOUTH AFRICA TAX BRACKETS (in ZAR)
// =============================================================================

/// South Africa 2024 Income Tax (SARS - Tax Year March-Feb)
pub static ZA_2024: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,          upper: 237_100,     rate_bps: 1800 },  // 18%
    TaxBracket { index: 1, lower: 237_100,    upper: 370_500,     rate_bps: 2600 },  // 26%
    TaxBracket { index: 2, lower: 370_500,    upper: 512_800,     rate_bps: 3100 },  // 31%
    TaxBracket { index: 3, lower: 512_800,    upper: 673_000,     rate_bps: 3600 },  // 36%
    TaxBracket { index: 4, lower: 673_000,    upper: 857_900,     rate_bps: 3900 },  // 39%
    TaxBracket { index: 5, lower: 857_900,    upper: 1_817_000,   rate_bps: 4100 },  // 41%
    TaxBracket { index: 6, lower: 1_817_000,  upper: u64::MAX,    rate_bps: 4500 },  // 45%
];

/// South Africa 2025 Income Tax (inflation adjusted)
pub static ZA_2025: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,          upper: 248_976,     rate_bps: 1800 },  // 18%
    TaxBracket { index: 1, lower: 248_976,    upper: 389_025,     rate_bps: 2600 },  // 26%
    TaxBracket { index: 2, lower: 389_025,    upper: 538_440,     rate_bps: 3100 },  // 31%
    TaxBracket { index: 3, lower: 538_440,    upper: 706_650,     rate_bps: 3600 },  // 36%
    TaxBracket { index: 4, lower: 706_650,    upper: 900_795,     rate_bps: 3900 },  // 39%
    TaxBracket { index: 5, lower: 900_795,    upper: 1_907_850,   rate_bps: 4100 },  // 41%
    TaxBracket { index: 6, lower: 1_907_850,  upper: u64::MAX,    rate_bps: 4500 },  // 45%
];

// =============================================================================
// ADDITIONAL AMERICAS
// =============================================================================

/// Chile 2024 Tax Brackets (SII - in UTA, converted to CLP)
pub static CL_2024: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,             upper: 8_775_000,    rate_bps: 0 },     // Exempt
    TaxBracket { index: 1, lower: 8_775_000,     upper: 19_500_000,   rate_bps: 400 },   // 4%
    TaxBracket { index: 2, lower: 19_500_000,    upper: 32_500_000,   rate_bps: 800 },   // 8%
    TaxBracket { index: 3, lower: 32_500_000,    upper: 45_500_000,   rate_bps: 1350 },  // 13.5%
    TaxBracket { index: 4, lower: 45_500_000,    upper: 58_500_000,   rate_bps: 2300 },  // 23%
    TaxBracket { index: 5, lower: 58_500_000,    upper: 78_000_000,   rate_bps: 3040 },  // 30.4%
    TaxBracket { index: 6, lower: 78_000_000,    upper: u64::MAX,     rate_bps: 4000 },  // 40%
];

pub static CL_2025: [TaxBracket; 7] = CL_2024;

/// Colombia 2024 Tax Brackets (DIAN - in UVT, converted to COP)
pub static CO_2024: [TaxBracket; 5] = [
    TaxBracket { index: 0, lower: 0,             upper: 49_850_000,   rate_bps: 0 },     // 0%
    TaxBracket { index: 1, lower: 49_850_000,    upper: 78_000_000,   rate_bps: 1900 },  // 19%
    TaxBracket { index: 2, lower: 78_000_000,    upper: 169_000_000,  rate_bps: 2800 },  // 28%
    TaxBracket { index: 3, lower: 169_000_000,   upper: 674_000_000,  rate_bps: 3300 },  // 33%
    TaxBracket { index: 4, lower: 674_000_000,   upper: u64::MAX,     rate_bps: 3900 },  // 39%
];

pub static CO_2025: [TaxBracket; 5] = CO_2024;

/// Peru 2024 Tax Brackets (SUNAT - in UIT, converted to PEN)
pub static PE_2024: [TaxBracket; 5] = [
    TaxBracket { index: 0, lower: 0,          upper: 24_750,    rate_bps: 800 },   // 8%
    TaxBracket { index: 1, lower: 24_750,     upper: 99_000,    rate_bps: 1400 },  // 14%
    TaxBracket { index: 2, lower: 99_000,     upper: 184_800,   rate_bps: 1700 },  // 17%
    TaxBracket { index: 3, lower: 184_800,    upper: 247_500,   rate_bps: 2000 },  // 20%
    TaxBracket { index: 4, lower: 247_500,    upper: u64::MAX,  rate_bps: 3000 },  // 30%
];

pub static PE_2025: [TaxBracket; 5] = PE_2024;

/// Ecuador 2024 Tax Brackets (SRI - in USD)
pub static EC_2024: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,          upper: 11_722,    rate_bps: 0 },     // 0%
    TaxBracket { index: 1, lower: 11_722,     upper: 14_930,    rate_bps: 500 },   // 5%
    TaxBracket { index: 2, lower: 14_930,     upper: 19_385,    rate_bps: 1000 },  // 10%
    TaxBracket { index: 3, lower: 19_385,     upper: 25_638,    rate_bps: 1200 },  // 12%
    TaxBracket { index: 4, lower: 25_638,     upper: 33_738,    rate_bps: 1500 },  // 15%
    TaxBracket { index: 5, lower: 33_738,     upper: 67_476,    rate_bps: 2000 },  // 20%
    TaxBracket { index: 6, lower: 67_476,     upper: u64::MAX,  rate_bps: 2500 },  // 25%
];

pub static EC_2025: [TaxBracket; 7] = EC_2024;

/// Uruguay 2024 Tax Brackets (DGI - IRPF in UYU)
pub static UY_2024: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,             upper: 445_200,      rate_bps: 0 },     // 0%
    TaxBracket { index: 1, lower: 445_200,       upper: 636_000,      rate_bps: 1000 },  // 10%
    TaxBracket { index: 2, lower: 636_000,       upper: 954_000,      rate_bps: 1500 },  // 15%
    TaxBracket { index: 3, lower: 954_000,       upper: 1_908_000,    rate_bps: 2400 },  // 24%
    TaxBracket { index: 4, lower: 1_908_000,     upper: 3_180_000,    rate_bps: 2500 },  // 25%
    TaxBracket { index: 5, lower: 3_180_000,     upper: 4_770_000,    rate_bps: 2700 },  // 27%
    TaxBracket { index: 6, lower: 4_770_000,     upper: u64::MAX,     rate_bps: 3600 },  // 36%
];

pub static UY_2025: [TaxBracket; 7] = UY_2024;

// =============================================================================
// ADDITIONAL EUROPE - EU
// =============================================================================

/// Spain 2024 IRPF Tax Brackets (AEAT - in EUR, general state rates)
pub static ES_2024: [TaxBracket; 5] = [
    TaxBracket { index: 0, lower: 0,          upper: 12_450,    rate_bps: 1900 },  // 19%
    TaxBracket { index: 1, lower: 12_450,     upper: 20_200,    rate_bps: 2400 },  // 24%
    TaxBracket { index: 2, lower: 20_200,     upper: 35_200,    rate_bps: 3000 },  // 30%
    TaxBracket { index: 3, lower: 35_200,     upper: 60_000,    rate_bps: 3700 },  // 37%
    TaxBracket { index: 4, lower: 60_000,     upper: u64::MAX,  rate_bps: 4500 },  // 45%
];

pub static ES_2025: [TaxBracket; 5] = ES_2024;

/// Netherlands 2024 Tax Brackets (Belastingdienst - in EUR)
pub static NL_2024: [TaxBracket; 3] = [
    TaxBracket { index: 0, lower: 0,          upper: 75_518,    rate_bps: 3693 },  // 36.93%
    TaxBracket { index: 1, lower: 75_518,     upper: u64::MAX,  rate_bps: 4950 },  // 49.50%
    TaxBracket { index: 2, lower: 0, upper: 0, rate_bps: 0 },  // placeholder
];

pub static NL_2025: [TaxBracket; 3] = NL_2024;

/// Belgium 2024 Tax Brackets (SPF Finances - in EUR)
pub static BE_2024: [TaxBracket; 4] = [
    TaxBracket { index: 0, lower: 0,          upper: 15_200,    rate_bps: 2500 },  // 25%
    TaxBracket { index: 1, lower: 15_200,     upper: 26_830,    rate_bps: 4000 },  // 40%
    TaxBracket { index: 2, lower: 26_830,     upper: 46_440,    rate_bps: 4500 },  // 45%
    TaxBracket { index: 3, lower: 46_440,     upper: u64::MAX,  rate_bps: 5000 },  // 50%
];

pub static BE_2025: [TaxBracket; 4] = BE_2024;

/// Austria 2024 Tax Brackets (BMF - in EUR)
pub static AT_2024: [TaxBracket; 5] = [
    TaxBracket { index: 0, lower: 0,          upper: 12_816,    rate_bps: 0 },     // 0%
    TaxBracket { index: 1, lower: 12_816,     upper: 20_818,    rate_bps: 2000 },  // 20%
    TaxBracket { index: 2, lower: 20_818,     upper: 34_513,    rate_bps: 3000 },  // 30%
    TaxBracket { index: 3, lower: 34_513,     upper: 66_612,    rate_bps: 4000 },  // 40%
    TaxBracket { index: 4, lower: 66_612,     upper: u64::MAX,  rate_bps: 5000 },  // 50%
];

pub static AT_2025: [TaxBracket; 5] = AT_2024;

/// Portugal 2024 Tax Brackets (AT - in EUR)
pub static PT_2024: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,          upper: 7_703,     rate_bps: 1350 },  // 13.5%
    TaxBracket { index: 1, lower: 7_703,      upper: 11_623,    rate_bps: 1800 },  // 18%
    TaxBracket { index: 2, lower: 11_623,     upper: 16_472,    rate_bps: 2300 },  // 23%
    TaxBracket { index: 3, lower: 16_472,     upper: 21_321,    rate_bps: 2600 },  // 26%
    TaxBracket { index: 4, lower: 21_321,     upper: 27_146,    rate_bps: 3275 },  // 32.75%
    TaxBracket { index: 5, lower: 27_146,     upper: 80_882,    rate_bps: 3700 },  // 37%
    TaxBracket { index: 6, lower: 80_882,     upper: u64::MAX,  rate_bps: 4800 },  // 48%
];

pub static PT_2025: [TaxBracket; 7] = PT_2024;

/// Ireland 2024 Tax Brackets (Revenue - in EUR)
pub static IE_2024: [TaxBracket; 2] = [
    TaxBracket { index: 0, lower: 0,          upper: 42_000,    rate_bps: 2000 },  // 20%
    TaxBracket { index: 1, lower: 42_000,     upper: u64::MAX,  rate_bps: 4000 },  // 40%
];

pub static IE_2025: [TaxBracket; 2] = IE_2024;

/// Poland 2024 Tax Brackets (KAS - in PLN)
pub static PL_2024: [TaxBracket; 2] = [
    TaxBracket { index: 0, lower: 0,          upper: 120_000,   rate_bps: 1200 },  // 12%
    TaxBracket { index: 1, lower: 120_000,    upper: u64::MAX,  rate_bps: 3200 },  // 32%
];

pub static PL_2025: [TaxBracket; 2] = PL_2024;

/// Sweden 2024 Tax Brackets (Skatteverket - in SEK, state tax only)
pub static SE_2024: [TaxBracket; 2] = [
    TaxBracket { index: 0, lower: 0,          upper: 598_500,   rate_bps: 0 },     // 0% state (local ~32%)
    TaxBracket { index: 1, lower: 598_500,    upper: u64::MAX,  rate_bps: 2000 },  // 20% state
];

pub static SE_2025: [TaxBracket; 2] = SE_2024;

/// Denmark 2024 Tax Brackets (Skattestyrelsen - in DKK, state tax)
pub static DK_2024: [TaxBracket; 3] = [
    TaxBracket { index: 0, lower: 0,          upper: 46_600,    rate_bps: 0 },     // Personal allowance
    TaxBracket { index: 1, lower: 46_600,     upper: 588_900,   rate_bps: 1230 },  // 12.3% bottom
    TaxBracket { index: 2, lower: 588_900,    upper: u64::MAX,  rate_bps: 2730 },  // 27.3% top
];

pub static DK_2025: [TaxBracket; 3] = DK_2024;

/// Finland 2024 Tax Brackets (Vero - in EUR, state tax)
pub static FI_2024: [TaxBracket; 5] = [
    TaxBracket { index: 0, lower: 0,          upper: 19_900,    rate_bps: 0 },     // 0%
    TaxBracket { index: 1, lower: 19_900,     upper: 29_700,    rate_bps: 1264 },  // 12.64%
    TaxBracket { index: 2, lower: 29_700,     upper: 49_000,    rate_bps: 1700 },  // 17%
    TaxBracket { index: 3, lower: 49_000,     upper: 85_800,    rate_bps: 2164 },  // 21.64%
    TaxBracket { index: 4, lower: 85_800,     upper: u64::MAX,  rate_bps: 3144 },  // 31.44%
];

pub static FI_2025: [TaxBracket; 5] = FI_2024;

/// Norway 2024 Tax Brackets (Skatteetaten - in NOK, bracket tax)
pub static NO_2024: [TaxBracket; 5] = [
    TaxBracket { index: 0, lower: 0,          upper: 208_050,   rate_bps: 0 },     // 0%
    TaxBracket { index: 1, lower: 208_050,    upper: 292_850,   rate_bps: 170 },   // 1.7%
    TaxBracket { index: 2, lower: 292_850,    upper: 670_000,   rate_bps: 400 },   // 4.0%
    TaxBracket { index: 3, lower: 670_000,    upper: 937_900,   rate_bps: 1370 },  // 13.7%
    TaxBracket { index: 4, lower: 937_900,    upper: u64::MAX,  rate_bps: 1640 },  // 16.4%
];

pub static NO_2025: [TaxBracket; 5] = NO_2024;

/// Switzerland 2024 Tax Brackets (FTA - in CHF, federal direct tax)
pub static CH_2024: [TaxBracket; 6] = [
    TaxBracket { index: 0, lower: 0,          upper: 17_800,    rate_bps: 0 },     // 0%
    TaxBracket { index: 1, lower: 17_800,     upper: 31_600,    rate_bps: 77 },    // 0.77%
    TaxBracket { index: 2, lower: 31_600,     upper: 41_400,    rate_bps: 88 },    // 0.88%
    TaxBracket { index: 3, lower: 41_400,     upper: 55_200,    rate_bps: 264 },   // 2.64%
    TaxBracket { index: 4, lower: 55_200,     upper: 755_200,   rate_bps: 654 },   // 6.54%
    TaxBracket { index: 5, lower: 755_200,    upper: u64::MAX,  rate_bps: 1150 },  // 11.5%
];

pub static CH_2025: [TaxBracket; 6] = CH_2024;

/// Czech Republic 2024 Tax Brackets (FS - in CZK)
pub static CZ_2024: [TaxBracket; 2] = [
    TaxBracket { index: 0, lower: 0,             upper: 1_935_552,  rate_bps: 1500 },  // 15%
    TaxBracket { index: 1, lower: 1_935_552,     upper: u64::MAX,   rate_bps: 2300 },  // 23%
];

pub static CZ_2025: [TaxBracket; 2] = CZ_2024;

/// Greece 2024 Tax Brackets (AADE - in EUR)
pub static GR_2024: [TaxBracket; 4] = [
    TaxBracket { index: 0, lower: 0,          upper: 10_000,    rate_bps: 900 },   // 9%
    TaxBracket { index: 1, lower: 10_000,     upper: 20_000,    rate_bps: 2200 },  // 22%
    TaxBracket { index: 2, lower: 20_000,     upper: 30_000,    rate_bps: 2800 },  // 28%
    TaxBracket { index: 3, lower: 30_000,     upper: u64::MAX,  rate_bps: 3600 },  // 36%
];

pub static GR_2025: [TaxBracket; 4] = GR_2024;

/// Hungary 2024 Flat Tax (NAV - in HUF)
pub static HU_2024: [TaxBracket; 1] = [
    TaxBracket { index: 0, lower: 0, upper: u64::MAX, rate_bps: 1500 },  // 15% flat
];

pub static HU_2025: [TaxBracket; 1] = HU_2024;

/// Romania 2024 Flat Tax (ANAF - in RON)
pub static RO_2024: [TaxBracket; 1] = [
    TaxBracket { index: 0, lower: 0, upper: u64::MAX, rate_bps: 1000 },  // 10% flat
];

pub static RO_2025: [TaxBracket; 1] = RO_2024;

// =============================================================================
// EUROPE - NON-EU
// =============================================================================

/// Ukraine 2024 Tax Brackets (DPS - in UAH)
pub static UA_2024: [TaxBracket; 1] = [
    TaxBracket { index: 0, lower: 0, upper: u64::MAX, rate_bps: 1800 },  // 18% flat (PDFO)
];

pub static UA_2025: [TaxBracket; 1] = UA_2024;

// =============================================================================
// ADDITIONAL ASIA-PACIFIC
// =============================================================================

/// New Zealand 2024 Tax Brackets (IRD - in NZD)
pub static NZ_2024: [TaxBracket; 5] = [
    TaxBracket { index: 0, lower: 0,          upper: 14_000,    rate_bps: 1050 },  // 10.5%
    TaxBracket { index: 1, lower: 14_000,     upper: 48_000,    rate_bps: 1750 },  // 17.5%
    TaxBracket { index: 2, lower: 48_000,     upper: 70_000,    rate_bps: 3000 },  // 30%
    TaxBracket { index: 3, lower: 70_000,     upper: 180_000,   rate_bps: 3300 },  // 33%
    TaxBracket { index: 4, lower: 180_000,    upper: u64::MAX,  rate_bps: 3900 },  // 39%
];

pub static NZ_2025: [TaxBracket; 5] = NZ_2024;

/// Singapore 2024 Tax Brackets (IRAS - in SGD)
pub static SG_2024: [TaxBracket; 9] = [
    TaxBracket { index: 0, lower: 0,          upper: 20_000,    rate_bps: 0 },     // 0%
    TaxBracket { index: 1, lower: 20_000,     upper: 30_000,    rate_bps: 200 },   // 2%
    TaxBracket { index: 2, lower: 30_000,     upper: 40_000,    rate_bps: 350 },   // 3.5%
    TaxBracket { index: 3, lower: 40_000,     upper: 80_000,    rate_bps: 700 },   // 7%
    TaxBracket { index: 4, lower: 80_000,     upper: 120_000,   rate_bps: 1150 },  // 11.5%
    TaxBracket { index: 5, lower: 120_000,    upper: 160_000,   rate_bps: 1500 },  // 15%
    TaxBracket { index: 6, lower: 160_000,    upper: 200_000,   rate_bps: 1800 },  // 18%
    TaxBracket { index: 7, lower: 200_000,    upper: 320_000,   rate_bps: 1900 },  // 19%
    TaxBracket { index: 8, lower: 320_000,    upper: u64::MAX,  rate_bps: 2200 },  // 22%
];

pub static SG_2025: [TaxBracket; 9] = SG_2024;

/// Hong Kong 2024 Tax Brackets (IRD - in HKD)
pub static HK_2024: [TaxBracket; 5] = [
    TaxBracket { index: 0, lower: 0,          upper: 50_000,    rate_bps: 200 },   // 2%
    TaxBracket { index: 1, lower: 50_000,     upper: 100_000,   rate_bps: 600 },   // 6%
    TaxBracket { index: 2, lower: 100_000,    upper: 150_000,   rate_bps: 1000 },  // 10%
    TaxBracket { index: 3, lower: 150_000,    upper: 200_000,   rate_bps: 1400 },  // 14%
    TaxBracket { index: 4, lower: 200_000,    upper: u64::MAX,  rate_bps: 1700 },  // 17%
];

pub static HK_2025: [TaxBracket; 5] = HK_2024;

/// Taiwan 2024 Tax Brackets (NTA - in TWD)
pub static TW_2024: [TaxBracket; 5] = [
    TaxBracket { index: 0, lower: 0,          upper: 560_000,   rate_bps: 500 },   // 5%
    TaxBracket { index: 1, lower: 560_000,    upper: 1_260_000, rate_bps: 1200 },  // 12%
    TaxBracket { index: 2, lower: 1_260_000,  upper: 2_520_000, rate_bps: 2000 },  // 20%
    TaxBracket { index: 3, lower: 2_520_000,  upper: 4_720_000, rate_bps: 3000 },  // 30%
    TaxBracket { index: 4, lower: 4_720_000,  upper: u64::MAX,  rate_bps: 4000 },  // 40%
];

pub static TW_2025: [TaxBracket; 5] = TW_2024;

/// Malaysia 2024 Tax Brackets (LHDN - in MYR)
pub static MY_2024: [TaxBracket; 8] = [
    TaxBracket { index: 0, lower: 0,          upper: 5_000,     rate_bps: 0 },     // 0%
    TaxBracket { index: 1, lower: 5_000,      upper: 20_000,    rate_bps: 100 },   // 1%
    TaxBracket { index: 2, lower: 20_000,     upper: 35_000,    rate_bps: 300 },   // 3%
    TaxBracket { index: 3, lower: 35_000,     upper: 50_000,    rate_bps: 600 },   // 6%
    TaxBracket { index: 4, lower: 50_000,     upper: 70_000,    rate_bps: 1100 },  // 11%
    TaxBracket { index: 5, lower: 70_000,     upper: 100_000,   rate_bps: 1900 },  // 19%
    TaxBracket { index: 6, lower: 100_000,    upper: 400_000,   rate_bps: 2500 },  // 25%
    TaxBracket { index: 7, lower: 400_000,    upper: u64::MAX,  rate_bps: 2800 },  // 28%
];

pub static MY_2025: [TaxBracket; 8] = MY_2024;

/// Thailand 2024 Tax Brackets (RD - in THB)
pub static TH_2024: [TaxBracket; 8] = [
    TaxBracket { index: 0, lower: 0,          upper: 150_000,   rate_bps: 0 },     // 0%
    TaxBracket { index: 1, lower: 150_000,    upper: 300_000,   rate_bps: 500 },   // 5%
    TaxBracket { index: 2, lower: 300_000,    upper: 500_000,   rate_bps: 1000 },  // 10%
    TaxBracket { index: 3, lower: 500_000,    upper: 750_000,   rate_bps: 1500 },  // 15%
    TaxBracket { index: 4, lower: 750_000,    upper: 1_000_000, rate_bps: 2000 },  // 20%
    TaxBracket { index: 5, lower: 1_000_000,  upper: 2_000_000, rate_bps: 2500 },  // 25%
    TaxBracket { index: 6, lower: 2_000_000,  upper: 5_000_000, rate_bps: 3000 },  // 30%
    TaxBracket { index: 7, lower: 5_000_000,  upper: u64::MAX,  rate_bps: 3500 },  // 35%
];

pub static TH_2025: [TaxBracket; 8] = TH_2024;

/// Vietnam 2024 Tax Brackets (GDT - in VND)
pub static VN_2024: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,             upper: 60_000_000,    rate_bps: 500 },   // 5%
    TaxBracket { index: 1, lower: 60_000_000,    upper: 120_000_000,   rate_bps: 1000 },  // 10%
    TaxBracket { index: 2, lower: 120_000_000,   upper: 216_000_000,   rate_bps: 1500 },  // 15%
    TaxBracket { index: 3, lower: 216_000_000,   upper: 384_000_000,   rate_bps: 2000 },  // 20%
    TaxBracket { index: 4, lower: 384_000_000,   upper: 624_000_000,   rate_bps: 2500 },  // 25%
    TaxBracket { index: 5, lower: 624_000_000,   upper: 960_000_000,   rate_bps: 3000 },  // 30%
    TaxBracket { index: 6, lower: 960_000_000,   upper: u64::MAX,      rate_bps: 3500 },  // 35%
];

pub static VN_2025: [TaxBracket; 7] = VN_2024;

/// Philippines 2024 Tax Brackets (BIR - in PHP)
pub static PH_2024: [TaxBracket; 6] = [
    TaxBracket { index: 0, lower: 0,          upper: 250_000,    rate_bps: 0 },     // 0%
    TaxBracket { index: 1, lower: 250_000,    upper: 400_000,    rate_bps: 1500 },  // 15%
    TaxBracket { index: 2, lower: 400_000,    upper: 800_000,    rate_bps: 2000 },  // 20%
    TaxBracket { index: 3, lower: 800_000,    upper: 2_000_000,  rate_bps: 2500 },  // 25%
    TaxBracket { index: 4, lower: 2_000_000,  upper: 8_000_000,  rate_bps: 3000 },  // 30%
    TaxBracket { index: 5, lower: 8_000_000,  upper: u64::MAX,   rate_bps: 3500 },  // 35%
];

pub static PH_2025: [TaxBracket; 6] = PH_2024;

/// Pakistan 2024 Tax Brackets (FBR - in PKR)
pub static PK_2024: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,             upper: 600_000,      rate_bps: 0 },     // 0%
    TaxBracket { index: 1, lower: 600_000,       upper: 1_200_000,    rate_bps: 500 },   // 5%
    TaxBracket { index: 2, lower: 1_200_000,     upper: 2_400_000,    rate_bps: 1000 },  // 10%
    TaxBracket { index: 3, lower: 2_400_000,     upper: 3_600_000,    rate_bps: 1500 },  // 15%
    TaxBracket { index: 4, lower: 3_600_000,     upper: 6_000_000,    rate_bps: 2000 },  // 20%
    TaxBracket { index: 5, lower: 6_000_000,     upper: 12_000_000,   rate_bps: 2500 },  // 25%
    TaxBracket { index: 6, lower: 12_000_000,    upper: u64::MAX,     rate_bps: 3500 },  // 35%
];

pub static PK_2025: [TaxBracket; 7] = PK_2024;

// =============================================================================
// ADDITIONAL MIDDLE EAST
// =============================================================================

/// UAE 2024 - No personal income tax (FTA)
pub static AE_2024: [TaxBracket; 1] = [
    TaxBracket { index: 0, lower: 0, upper: u64::MAX, rate_bps: 0 },  // 0%
];

pub static AE_2025: [TaxBracket; 1] = AE_2024;

/// Israel 2024 Tax Brackets (ITA - in ILS)
pub static IL_2024: [TaxBracket; 6] = [
    TaxBracket { index: 0, lower: 0,          upper: 84_120,    rate_bps: 1000 },  // 10%
    TaxBracket { index: 1, lower: 84_120,     upper: 120_720,   rate_bps: 1400 },  // 14%
    TaxBracket { index: 2, lower: 120_720,    upper: 193_800,   rate_bps: 2000 },  // 20%
    TaxBracket { index: 3, lower: 193_800,    upper: 269_280,   rate_bps: 3100 },  // 31%
    TaxBracket { index: 4, lower: 269_280,    upper: 560_280,   rate_bps: 3500 },  // 35%
    TaxBracket { index: 5, lower: 560_280,    upper: u64::MAX,  rate_bps: 4700 },  // 47%
];

pub static IL_2025: [TaxBracket; 6] = IL_2024;

/// Egypt 2024 Tax Brackets (ETA - in EGP)
pub static EG_2024: [TaxBracket; 6] = [
    TaxBracket { index: 0, lower: 0,          upper: 40_000,    rate_bps: 0 },     // 0%
    TaxBracket { index: 1, lower: 40_000,     upper: 55_000,    rate_bps: 1000 },  // 10%
    TaxBracket { index: 2, lower: 55_000,     upper: 70_000,    rate_bps: 1500 },  // 15%
    TaxBracket { index: 3, lower: 70_000,     upper: 200_000,   rate_bps: 2000 },  // 20%
    TaxBracket { index: 4, lower: 200_000,    upper: 400_000,   rate_bps: 2250 },  // 22.5%
    TaxBracket { index: 5, lower: 400_000,    upper: u64::MAX,  rate_bps: 2500 },  // 25%
];

pub static EG_2025: [TaxBracket; 6] = EG_2024;

/// Qatar 2024 - No personal income tax (GTA)
pub static QA_2024: [TaxBracket; 1] = [
    TaxBracket { index: 0, lower: 0, upper: u64::MAX, rate_bps: 0 },  // 0%
];

pub static QA_2025: [TaxBracket; 1] = QA_2024;

// =============================================================================
// ADDITIONAL AFRICA
// =============================================================================

/// Nigeria 2024 Tax Brackets (FIRS - in NGN)
pub static NG_2024: [TaxBracket; 6] = [
    TaxBracket { index: 0, lower: 0,          upper: 300_000,      rate_bps: 700 },   // 7%
    TaxBracket { index: 1, lower: 300_000,    upper: 600_000,      rate_bps: 1100 },  // 11%
    TaxBracket { index: 2, lower: 600_000,    upper: 1_100_000,    rate_bps: 1500 },  // 15%
    TaxBracket { index: 3, lower: 1_100_000,  upper: 1_600_000,    rate_bps: 1900 },  // 19%
    TaxBracket { index: 4, lower: 1_600_000,  upper: 3_200_000,    rate_bps: 2100 },  // 21%
    TaxBracket { index: 5, lower: 3_200_000,  upper: u64::MAX,     rate_bps: 2400 },  // 24%
];

pub static NG_2025: [TaxBracket; 6] = NG_2024;

/// Kenya 2024 Tax Brackets (KRA - in KES)
pub static KE_2024: [TaxBracket; 5] = [
    TaxBracket { index: 0, lower: 0,          upper: 288_000,      rate_bps: 1000 },  // 10%
    TaxBracket { index: 1, lower: 288_000,    upper: 388_000,      rate_bps: 2500 },  // 25%
    TaxBracket { index: 2, lower: 388_000,    upper: 6_000_000,    rate_bps: 3000 },  // 30%
    TaxBracket { index: 3, lower: 6_000_000,  upper: 9_600_000,    rate_bps: 3250 },  // 32.5%
    TaxBracket { index: 4, lower: 9_600_000,  upper: u64::MAX,     rate_bps: 3500 },  // 35%
];

pub static KE_2025: [TaxBracket; 5] = KE_2024;

/// Morocco 2024 Tax Brackets (DGI - in MAD)
pub static MA_2024: [TaxBracket; 6] = [
    TaxBracket { index: 0, lower: 0,          upper: 30_000,    rate_bps: 0 },     // 0%
    TaxBracket { index: 1, lower: 30_000,     upper: 50_000,    rate_bps: 1000 },  // 10%
    TaxBracket { index: 2, lower: 50_000,     upper: 60_000,    rate_bps: 2000 },  // 20%
    TaxBracket { index: 3, lower: 60_000,     upper: 80_000,    rate_bps: 3000 },  // 30%
    TaxBracket { index: 4, lower: 80_000,     upper: 180_000,   rate_bps: 3400 },  // 34%
    TaxBracket { index: 5, lower: 180_000,    upper: u64::MAX,  rate_bps: 3700 },  // 37%
];

pub static MA_2025: [TaxBracket; 6] = MA_2024;

/// Ghana 2024 Tax Brackets (GRA - in GHS)
pub static GH_2024: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,          upper: 5_880,     rate_bps: 0 },     // 0%
    TaxBracket { index: 1, lower: 5_880,      upper: 7_080,     rate_bps: 500 },   // 5%
    TaxBracket { index: 2, lower: 7_080,      upper: 10_560,    rate_bps: 1000 },  // 10%
    TaxBracket { index: 3, lower: 10_560,     upper: 43_560,    rate_bps: 1750 },  // 17.5%
    TaxBracket { index: 4, lower: 43_560,     upper: 235_560,   rate_bps: 2500 },  // 25%
    TaxBracket { index: 5, lower: 235_560,    upper: 900_000,   rate_bps: 3000 },  // 30%
    TaxBracket { index: 6, lower: 900_000,    upper: u64::MAX,  rate_bps: 3500 },  // 35%
];

pub static GH_2025: [TaxBracket; 7] = GH_2024;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_us_bracket_lookup() {
        let bracket = find_bracket(85_000, Jurisdiction::US, 2024, FilingStatus::Single).unwrap();
        assert_eq!(bracket.index, 2);
        assert_eq!(bracket.rate_bps, 2200);
    }

    #[test]
    fn test_uk_bracket_lookup() {
        let bracket = find_bracket(60_000, Jurisdiction::UK, 2024, FilingStatus::Single).unwrap();
        assert_eq!(bracket.index, 2);
        assert_eq!(bracket.rate_bps, 4000);
    }

    #[test]
    fn test_commitment_deterministic() {
        let c1 = compute_commitment(47_150, 100_525, 2024);
        let c2 = compute_commitment(47_150, 100_525, 2024);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_commitment_differs_by_year() {
        let c1 = compute_commitment(47_150, 100_525, 2024);
        let c2 = compute_commitment(47_150, 100_525, 2025);
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_bracket_contains() {
        let bracket = &US_2024_SINGLE[2];
        assert!(!bracket.contains(47_149));
        assert!(bracket.contains(47_150));
        assert!(bracket.contains(85_000));
        assert!(!bracket.contains(100_525));
    }

    #[test]
    fn test_canada_bracket_lookup() {
        // 80,000 CAD should be in 20.5% bracket
        let bracket = find_bracket(80_000, Jurisdiction::CA, 2024, FilingStatus::Single).unwrap();
        assert_eq!(bracket.index, 1);
        assert_eq!(bracket.rate_bps, 2050);
    }

    #[test]
    fn test_germany_bracket_lookup() {
        // 50,000 EUR should be in 24% zone (Zone 3)
        let bracket = find_bracket(50_000, Jurisdiction::DE, 2024, FilingStatus::Single).unwrap();
        assert_eq!(bracket.index, 2);
        assert_eq!(bracket.rate_bps, 2400);
    }

    #[test]
    fn test_australia_bracket_lookup() {
        // 100,000 AUD should be in 32.5% bracket
        let bracket = find_bracket(100_000, Jurisdiction::AU, 2024, FilingStatus::Single).unwrap();
        assert_eq!(bracket.index, 2);
        assert_eq!(bracket.rate_bps, 3250);
    }

    #[test]
    fn test_germany_married() {
        // 100,000 EUR married should be in Zone 3 (splitting tariff)
        let bracket = find_bracket(100_000, Jurisdiction::DE, 2024, FilingStatus::MarriedFilingJointly).unwrap();
        assert_eq!(bracket.index, 2);
        assert_eq!(bracket.rate_bps, 2400);
    }

    #[test]
    fn test_france_bracket_lookup() {
        // 50,000 EUR should be in 30% bracket
        let bracket = find_bracket(50_000, Jurisdiction::FR, 2024, FilingStatus::Single).unwrap();
        assert_eq!(bracket.index, 2);
        assert_eq!(bracket.rate_bps, 3000);
    }

    #[test]
    fn test_japan_bracket_lookup() {
        // 5,000,000 JPY should be in 20% bracket
        let bracket = find_bracket(5_000_000, Jurisdiction::JP, 2024, FilingStatus::Single).unwrap();
        assert_eq!(bracket.index, 2);
        assert_eq!(bracket.rate_bps, 2000);
    }

    #[test]
    fn test_india_bracket_lookup() {
        // 800,000 INR should be in 10% bracket (New Tax Regime)
        let bracket = find_bracket(800_000, Jurisdiction::IN, 2024, FilingStatus::Single).unwrap();
        assert_eq!(bracket.index, 2);
        assert_eq!(bracket.rate_bps, 1000);
    }

    #[test]
    fn test_brazil_bracket_lookup() {
        // 40,000 BRL should be in 15% bracket
        let bracket = find_bracket(40_000, Jurisdiction::BR, 2024, FilingStatus::Single).unwrap();
        assert_eq!(bracket.index, 2);
        assert_eq!(bracket.rate_bps, 1500);
    }

    #[test]
    fn test_all_g20_jurisdictions_2024() {
        // Test that all G20 jurisdictions work for 2024
        let jurisdictions = [
            // Americas
            (Jurisdiction::US, 85_000_u64, 2_u8),
            (Jurisdiction::CA, 80_000, 1),
            (Jurisdiction::MX, 100_000, 2),
            (Jurisdiction::BR, 40_000, 2),
            (Jurisdiction::AR, 5_000_000, 2),  // 12% bracket
            // Europe
            (Jurisdiction::UK, 60_000, 2),
            (Jurisdiction::DE, 50_000, 2),
            (Jurisdiction::FR, 50_000, 2),
            (Jurisdiction::IT, 35_000, 1),
            (Jurisdiction::RU, 3_000_000, 0),
            (Jurisdiction::TR, 200_000, 1),
            // Asia-Pacific
            (Jurisdiction::JP, 5_000_000, 2),
            (Jurisdiction::CN, 200_000, 2),
            (Jurisdiction::IN, 800_000, 2),
            (Jurisdiction::KR, 30_000_000, 1),
            (Jurisdiction::ID, 100_000_000, 1),
            (Jurisdiction::AU, 100_000, 2),
            // Middle East & Africa
            (Jurisdiction::SA, 1_000_000, 0),  // 0% tax
            (Jurisdiction::ZA, 300_000, 1),
        ];

        for (j, income, expected_index) in jurisdictions {
            let bracket = find_bracket(income, j, 2024, FilingStatus::Single).unwrap();
            assert_eq!(bracket.index, expected_index, "Failed for {:?} with income {}", j, income);
        }
    }

    #[test]
    fn test_mexico_bracket_lookup() {
        // 100,000 MXN should be in 10.88% bracket
        let bracket = find_bracket(100_000, Jurisdiction::MX, 2024, FilingStatus::Single).unwrap();
        assert_eq!(bracket.index, 2);
        assert_eq!(bracket.rate_bps, 1088);
    }

    #[test]
    fn test_argentina_bracket_lookup() {
        // 5,000,000 ARS should be in 12% bracket (bracket 1 ends at 4,950,958)
        let bracket = find_bracket(5_000_000, Jurisdiction::AR, 2024, FilingStatus::Single).unwrap();
        assert_eq!(bracket.index, 2);
        assert_eq!(bracket.rate_bps, 1200);
    }

    #[test]
    fn test_china_bracket_lookup() {
        // 200,000 CNY should be in 20% bracket
        let bracket = find_bracket(200_000, Jurisdiction::CN, 2024, FilingStatus::Single).unwrap();
        assert_eq!(bracket.index, 2);
        assert_eq!(bracket.rate_bps, 2000);
    }

    #[test]
    fn test_korea_bracket_lookup() {
        // 30,000,000 KRW should be in 15% bracket
        let bracket = find_bracket(30_000_000, Jurisdiction::KR, 2024, FilingStatus::Single).unwrap();
        assert_eq!(bracket.index, 1);
        assert_eq!(bracket.rate_bps, 1500);
    }

    #[test]
    fn test_saudi_no_income_tax() {
        // Saudi Arabia has 0% income tax
        let bracket = find_bracket(1_000_000, Jurisdiction::SA, 2024, FilingStatus::Single).unwrap();
        assert_eq!(bracket.index, 0);
        assert_eq!(bracket.rate_bps, 0);
    }

    #[test]
    fn test_south_africa_bracket_lookup() {
        // 300,000 ZAR should be in 26% bracket
        let bracket = find_bracket(300_000, Jurisdiction::ZA, 2024, FilingStatus::Single).unwrap();
        assert_eq!(bracket.index, 1);
        assert_eq!(bracket.rate_bps, 2600);
    }

    // =======================================================================
    // HISTORICAL YEAR TESTS (2020-2023)
    // =======================================================================

    #[test]
    fn test_us_historical_years() {
        // Test US brackets for all supported years
        for year in [2020, 2021, 2022, 2023, 2024, 2025] {
            let bracket = find_bracket(85_000, Jurisdiction::US, year, FilingStatus::Single).unwrap();
            assert_eq!(bracket.index, 2, "Year {} should have bracket index 2", year);
            assert_eq!(bracket.rate_bps, 2200, "Year {} should have 22% rate", year);
        }
    }

    #[test]
    fn test_uk_historical_years() {
        // Test UK brackets for all supported years
        for year in [2020, 2021, 2022, 2023, 2024, 2025] {
            let bracket = find_bracket(60_000, Jurisdiction::UK, year, FilingStatus::Single).unwrap();
            assert_eq!(bracket.index, 2, "Year {} should have bracket index 2", year);
            assert_eq!(bracket.rate_bps, 4000, "Year {} should have 40% rate", year);
        }
    }

    #[test]
    fn test_us_2020_specific_brackets() {
        // 2020 had different bracket thresholds
        let bracket = find_bracket(40_000, Jurisdiction::US, 2020, FilingStatus::Single).unwrap();
        assert_eq!(bracket.index, 1); // 12% bracket
        assert_eq!(bracket.rate_bps, 1200);

        let bracket = find_bracket(85_000, Jurisdiction::US, 2020, FilingStatus::Single).unwrap();
        assert_eq!(bracket.index, 2); // 22% bracket
        assert_eq!(bracket.rate_bps, 2200);
    }

    #[test]
    fn test_us_2023_specific_brackets() {
        // 2023 brackets (inflation adjusted)
        let bracket = find_bracket(11_000, Jurisdiction::US, 2023, FilingStatus::Single).unwrap();
        assert_eq!(bracket.index, 1); // Just above 10% threshold
        assert_eq!(bracket.rate_bps, 1200);

        let bracket = find_bracket(95_000, Jurisdiction::US, 2023, FilingStatus::Single).unwrap();
        assert_eq!(bracket.index, 2); // 22% bracket
        assert_eq!(bracket.rate_bps, 2200);
    }

    #[test]
    fn test_us_mfj_historical() {
        // Test MFJ for historical years
        for year in [2020, 2021, 2022, 2023] {
            let bracket = find_bracket(150_000, Jurisdiction::US, year, FilingStatus::MarriedFilingJointly).unwrap();
            assert_eq!(bracket.index, 2, "Year {} MFJ should have bracket index 2", year);
            assert_eq!(bracket.rate_bps, 2200, "Year {} MFJ should have 22% rate", year);
        }
    }

    #[test]
    fn test_all_g20_historical_support() {
        // All G20 jurisdictions should work for all years 2020-2025
        let jurisdictions = [
            Jurisdiction::US, Jurisdiction::CA, Jurisdiction::MX, Jurisdiction::BR, Jurisdiction::AR,
            Jurisdiction::UK, Jurisdiction::DE, Jurisdiction::FR, Jurisdiction::IT, Jurisdiction::RU, Jurisdiction::TR,
            Jurisdiction::JP, Jurisdiction::CN, Jurisdiction::IN, Jurisdiction::KR, Jurisdiction::ID, Jurisdiction::AU,
            Jurisdiction::SA, Jurisdiction::ZA,
        ];

        for year in [2020, 2021, 2022, 2023, 2024, 2025] {
            for j in &jurisdictions {
                let result = get_brackets(*j, year, FilingStatus::Single);
                assert!(result.is_ok(), "Year {} should work for {:?}", year, j);
            }
        }
    }

    #[test]
    fn test_unsupported_year_rejected() {
        // Years outside 2020-2025 should be rejected
        for year in [2019, 2026, 2030] {
            let result = get_brackets(Jurisdiction::US, year, FilingStatus::Single);
            assert!(result.is_err(), "Year {} should be rejected", year);
        }
    }
}
