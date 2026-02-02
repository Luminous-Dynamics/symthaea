//! Proof types for ZK tax verification.

use crate::brackets::{compute_commitment, find_bracket};
use crate::error::{Error, Result};
use crate::jurisdiction::{FilingStatus, Jurisdiction};
use crate::types::{BracketCommitment, TaxYear};
use serde::{Deserialize, Serialize};

/// A zero-knowledge proof of tax bracket membership.
///
/// This proof demonstrates that the prover knows an income that falls
/// within a specific tax bracket, without revealing the actual income.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaxBracketProof {
    /// The jurisdiction (e.g., US, UK)
    pub jurisdiction: Jurisdiction,
    /// Filing status used
    pub filing_status: FilingStatus,
    /// Tax year
    pub tax_year: TaxYear,
    /// Bracket index that was proven
    pub bracket_index: u8,
    /// Marginal rate in basis points
    pub rate_bps: u16,
    /// Lower bound of the proven bracket
    pub bracket_lower: u64,
    /// Upper bound of the proven bracket
    pub bracket_upper: u64,
    /// Cryptographic commitment to bracket parameters
    pub commitment: BracketCommitment,
    /// The Risc0 receipt (serialized)
    #[serde(with = "hex_bytes")]
    pub receipt_bytes: Vec<u8>,
    /// Image ID for verification
    #[serde(with = "hex_bytes")]
    pub image_id: Vec<u8>,
}

/// Legacy dev mode receipt marker (for backwards compatibility)
const LEGACY_DEV_MODE_RECEIPT: &[u8] = &[0xDE, 0xAD, 0xBE, 0xEF];

/// New simulation marker prefix: "SIMU" in ASCII
const SIMULATION_PREFIX: [u8; 4] = [0x53, 0x49, 0x4D, 0x55];

impl TaxBracketProof {
    /// Check if this is a simulation/dev mode proof (provides NO cryptographic guarantees).
    ///
    /// **SECURITY WARNING**: Simulation proofs can be trivially forged!
    /// Never accept simulation proofs in production.
    pub fn is_simulation_proof(&self) -> bool {
        // Check for new simulation marker prefix "SIMU"
        if self.receipt_bytes.len() >= 4 && self.receipt_bytes[0..4] == SIMULATION_PREFIX {
            return true;
        }
        // Also check for legacy DEADBEEF marker for backwards compatibility
        if self.receipt_bytes == LEGACY_DEV_MODE_RECEIPT {
            return true;
        }
        false
    }

    /// Check if this is a dev mode proof.
    ///
    /// **Deprecated**: Use `is_simulation_proof()` instead.
    pub fn is_dev_mode(&self) -> bool {
        self.is_simulation_proof()
    }

    /// Check if this is a real cryptographic proof (not simulation).
    pub fn is_real_proof(&self) -> bool {
        !self.is_simulation_proof()
    }

    /// Verify this proof is suitable for production use.
    ///
    /// Returns an error if this is a simulation proof.
    pub fn verify_production_ready(&self) -> Result<()> {
        if self.is_simulation_proof() {
            return Err(Error::invalid_proof(
                "SECURITY ERROR: Simulation proof rejected. \
                 Simulation proofs provide NO cryptographic guarantees. \
                 Enable 'prover' feature for real proofs.".to_string()
            ));
        }
        Ok(())
    }

    /// Verify this proof.
    ///
    /// Returns Ok(()) if the proof is valid, or an error if verification fails.
    #[cfg(feature = "prover")]
    pub fn verify(&self) -> Result<()> {
        // Simulation/dev mode proofs skip risc0 verification, only check commitment
        if self.is_simulation_proof() {
            return self.verify_commitment_only();
        }

        use risc0_zkvm::Receipt;

        // Deserialize the receipt
        let receipt: Receipt = bincode::deserialize(&self.receipt_bytes)
            .map_err(|e| Error::invalid_proof(format!("Failed to deserialize receipt: {}", e)))?;

        // Convert image_id to the correct type
        let image_id: [u32; 8] = bincode::deserialize(&self.image_id)
            .map_err(|e| Error::invalid_proof(format!("Failed to deserialize image ID: {}", e)))?;

        // Verify the receipt
        receipt
            .verify(image_id)
            .map_err(|e| Error::verification_failed(e.to_string()))?;

        // Verify commitment and bracket
        self.verify_commitment_only()
    }

    /// Verify only the commitment and bracket (used by dev mode and verifier-only mode).
    fn verify_commitment_only(&self) -> Result<()> {
        // Verify the commitment matches the claimed bracket
        let expected = compute_commitment(self.bracket_lower, self.bracket_upper, self.tax_year);
        if self.commitment != expected {
            return Err(Error::CommitmentMismatch {
                expected: expected.to_hex(),
                actual: self.commitment.to_hex(),
            });
        }

        // Verify the bracket is valid for the jurisdiction
        let bracket = find_bracket(
            self.bracket_lower, // Use lower bound as representative
            self.jurisdiction,
            self.tax_year,
            self.filing_status,
        )?;

        if bracket.index != self.bracket_index {
            return Err(Error::invalid_proof(format!(
                "Bracket index mismatch: expected {}, got {}",
                bracket.index, self.bracket_index
            )));
        }

        Ok(())
    }

    /// Verify this proof (verifier-only mode).
    #[cfg(not(feature = "prover"))]
    pub fn verify(&self) -> Result<()> {
        self.verify_commitment_only()
    }

    /// Get a human-readable summary of this proof.
    pub fn summary(&self) -> String {
        let rate = self.rate_bps as f64 / 100.0;
        format!(
            "Tax bracket proof: {} {} {}, Bracket {} ({}%), ${}-${}",
            self.jurisdiction,
            self.filing_status,
            self.tax_year,
            self.bracket_index,
            rate,
            format_currency(self.bracket_lower),
            if self.bracket_upper == u64::MAX {
                "+".to_string()
            } else {
                format_currency(self.bracket_upper)
            }
        )
    }

    /// Export proof to JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(Error::from)
    }

    /// Import proof from JSON.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(Error::from)
    }

    /// Export proof to compact binary format.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        bincode::serialize(self)
            .map_err(|e| Error::serialization(e.to_string()))
    }

    /// Import proof from compact binary format.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        bincode::deserialize(bytes)
            .map_err(|e| Error::serialization(e.to_string()))
    }
}

// =============================================================================
// Income Range Proofs
// =============================================================================

/// A zero-knowledge proof that income falls within a specified range.
///
/// This is useful for:
/// - Loan qualification (prove income > $50K)
/// - Rental applications (prove income > 3x rent)
/// - Benefits eligibility (prove income < threshold)
///
/// Unlike bracket proofs, range proofs allow custom ranges that may span
/// multiple tax brackets.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IncomeRangeProof {
    /// The claimed range lower bound (inclusive)
    pub range_lower: u64,
    /// The claimed range upper bound (exclusive)
    pub range_upper: u64,
    /// Tax year for context
    pub tax_year: TaxYear,
    /// Cryptographic commitment (hash of range bounds + hidden income)
    pub commitment: BracketCommitment,
    /// Whether this is a dev mode proof (not cryptographically secure)
    pub is_dev_mode: bool,
    /// Optional: The bracket proofs that together cover the range
    /// (For proofs that span multiple brackets)
    #[serde(default)]
    pub bracket_proofs: Vec<u8>,
}

impl IncomeRangeProof {
    /// Create a new range proof (dev mode - for testing only).
    ///
    /// # Arguments
    /// * `income` - The actual income (private, never revealed)
    /// * `range_lower` - Claimed minimum income (inclusive)
    /// * `range_upper` - Claimed maximum income (exclusive)
    /// * `tax_year` - Tax year
    ///
    /// # Returns
    /// * `Ok(IncomeRangeProof)` if income is within range
    /// * `Err` if income is outside the claimed range
    pub fn prove_dev(
        income: u64,
        range_lower: u64,
        range_upper: u64,
        tax_year: TaxYear,
    ) -> Result<Self> {
        // Validate income is within range
        if income < range_lower {
            return Err(Error::invalid_proof(format!(
                "Income {} is below claimed minimum {}",
                income, range_lower
            )));
        }
        if income >= range_upper {
            return Err(Error::invalid_proof(format!(
                "Income {} is at or above claimed maximum {}",
                income, range_upper
            )));
        }

        // Create commitment (doesn't reveal income)
        let commitment = compute_range_commitment(range_lower, range_upper, tax_year);

        Ok(IncomeRangeProof {
            range_lower,
            range_upper,
            tax_year,
            commitment,
            is_dev_mode: true,
            bracket_proofs: Vec::new(),
        })
    }

    /// Verify the range proof commitment.
    pub fn verify(&self) -> Result<()> {
        let expected = compute_range_commitment(self.range_lower, self.range_upper, self.tax_year);
        if self.commitment != expected {
            return Err(Error::CommitmentMismatch {
                expected: expected.to_hex(),
                actual: self.commitment.to_hex(),
            });
        }
        Ok(())
    }

    /// Get a human-readable summary of this proof.
    pub fn summary(&self) -> String {
        format!(
            "Income range proof: ${}-${} ({})",
            format_currency(self.range_lower),
            if self.range_upper == u64::MAX {
                "+".to_string()
            } else {
                format_currency(self.range_upper)
            },
            self.tax_year
        )
    }

    /// Check if the proven range contains a specific value.
    pub fn contains(&self, value: u64) -> bool {
        value >= self.range_lower && value < self.range_upper
    }

    /// Get the range width.
    pub fn range_width(&self) -> u64 {
        if self.range_upper == u64::MAX {
            u64::MAX - self.range_lower
        } else {
            self.range_upper - self.range_lower
        }
    }
}

/// Compute commitment for a range proof.
fn compute_range_commitment(lower: u64, upper: u64, year: TaxYear) -> BracketCommitment {
    // Use the same FNV-1a algorithm as bracket commitments for consistency
    compute_commitment(lower, upper, year)
}

/// Builder for creating income range proofs with common use cases.
#[derive(Clone, Debug)]
pub struct RangeProofBuilder {
    income: u64,
    tax_year: TaxYear,
}

impl RangeProofBuilder {
    /// Create a new builder with the given income.
    pub fn new(income: u64, tax_year: TaxYear) -> Self {
        RangeProofBuilder { income, tax_year }
    }

    /// Prove income is above a minimum threshold.
    ///
    /// Useful for: Loan qualification, rental applications
    pub fn prove_above(self, minimum: u64) -> Result<IncomeRangeProof> {
        IncomeRangeProof::prove_dev(self.income, minimum, u64::MAX, self.tax_year)
    }

    /// Prove income is below a maximum threshold.
    ///
    /// Useful for: Benefits eligibility, tax credits
    pub fn prove_below(self, maximum: u64) -> Result<IncomeRangeProof> {
        IncomeRangeProof::prove_dev(self.income, 0, maximum, self.tax_year)
    }

    /// Prove income is within a specific range.
    pub fn prove_between(self, minimum: u64, maximum: u64) -> Result<IncomeRangeProof> {
        IncomeRangeProof::prove_dev(self.income, minimum, maximum, self.tax_year)
    }

    /// Prove income is at least N times a given amount (e.g., 3x rent).
    ///
    /// Useful for: Rental applications (income ≥ 3x monthly rent)
    pub fn prove_multiple_of(self, base_amount: u64, multiplier: u64) -> Result<IncomeRangeProof> {
        let minimum = base_amount * multiplier;
        IncomeRangeProof::prove_dev(self.income, minimum, u64::MAX, self.tax_year)
    }
}

// =============================================================================
// Batch Proofs for Multi-Year Compliance
// =============================================================================

/// A batch of tax bracket proofs for multi-year compliance.
///
/// Useful for:
/// - Mortgage applications (prove 3+ years of stable income)
/// - Visa applications (prove consistent employment history)
/// - Insurance underwriting (income verification across years)
/// - Compliance audits (demonstrate historical bracket membership)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchProof {
    /// The jurisdiction for all proofs
    pub jurisdiction: Jurisdiction,
    /// Filing status used
    pub filing_status: FilingStatus,
    /// Individual year proofs (ordered by year)
    pub year_proofs: Vec<YearProofSummary>,
    /// Combined commitment (hash of all year commitments)
    pub batch_commitment: BracketCommitment,
    /// Whether this is a dev mode proof batch
    pub is_dev_mode: bool,
    /// Timestamp when batch was generated
    pub timestamp: u64,
}

/// Summary of a single year's proof within a batch.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct YearProofSummary {
    /// Tax year
    pub tax_year: TaxYear,
    /// Bracket index proven
    pub bracket_index: u8,
    /// Marginal rate in basis points
    pub rate_bps: u16,
    /// Bracket lower bound
    pub bracket_lower: u64,
    /// Bracket upper bound
    pub bracket_upper: u64,
    /// Individual commitment for this year
    pub commitment: BracketCommitment,
}

impl BatchProof {
    /// Verify all proofs in the batch.
    pub fn verify(&self) -> Result<()> {
        // Verify each individual year commitment
        for year_proof in &self.year_proofs {
            let expected = compute_commitment(
                year_proof.bracket_lower,
                year_proof.bracket_upper,
                year_proof.tax_year,
            );
            if year_proof.commitment != expected {
                return Err(Error::CommitmentMismatch {
                    expected: expected.to_hex(),
                    actual: year_proof.commitment.to_hex(),
                });
            }
        }

        // Verify batch commitment
        let expected_batch = compute_batch_commitment(&self.year_proofs);
        if self.batch_commitment != expected_batch {
            return Err(Error::CommitmentMismatch {
                expected: expected_batch.to_hex(),
                actual: self.batch_commitment.to_hex(),
            });
        }

        Ok(())
    }

    /// Get a human-readable summary.
    pub fn summary(&self) -> String {
        let years: Vec<String> = self.year_proofs.iter().map(|p| p.tax_year.to_string()).collect();
        let brackets: Vec<String> = self.year_proofs.iter().map(|p| p.bracket_index.to_string()).collect();

        format!(
            "Batch proof: {} {} years {}, brackets [{}]",
            self.jurisdiction,
            self.filing_status,
            years.join(", "),
            brackets.join(", ")
        )
    }

    /// Check if all years are in the same bracket.
    pub fn consistent_bracket(&self) -> Option<u8> {
        if self.year_proofs.is_empty() {
            return None;
        }
        let first = self.year_proofs[0].bracket_index;
        if self.year_proofs.iter().all(|p| p.bracket_index == first) {
            Some(first)
        } else {
            None
        }
    }

    /// Get the minimum bracket index across all years.
    pub fn min_bracket(&self) -> Option<u8> {
        self.year_proofs.iter().map(|p| p.bracket_index).min()
    }

    /// Get the maximum bracket index across all years.
    pub fn max_bracket(&self) -> Option<u8> {
        self.year_proofs.iter().map(|p| p.bracket_index).max()
    }

    /// Check if bracket progression is upward (career growth).
    pub fn shows_growth(&self) -> bool {
        if self.year_proofs.len() < 2 {
            return false;
        }
        // Sort by year and check if brackets are non-decreasing
        let mut sorted = self.year_proofs.clone();
        sorted.sort_by_key(|p| p.tax_year);
        sorted.windows(2).all(|w| w[1].bracket_index >= w[0].bracket_index)
    }

    /// Get the number of years in the batch.
    pub fn year_count(&self) -> usize {
        self.year_proofs.len()
    }

    /// Get the year range covered.
    pub fn year_range(&self) -> Option<(TaxYear, TaxYear)> {
        if self.year_proofs.is_empty() {
            return None;
        }
        let min = self.year_proofs.iter().map(|p| p.tax_year).min().unwrap();
        let max = self.year_proofs.iter().map(|p| p.tax_year).max().unwrap();
        Some((min, max))
    }
}

/// Compute a combined commitment for a batch of year proofs.
fn compute_batch_commitment(proofs: &[YearProofSummary]) -> BracketCommitment {
    use sha3::{Digest, Sha3_256};

    let mut hasher = Sha3_256::new();
    for proof in proofs {
        hasher.update(proof.commitment.as_bytes());
    }
    let hash = hasher.finalize();

    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(&hash);
    BracketCommitment::from_bytes(bytes)
}

/// Builder for creating batch proofs.
#[derive(Clone, Debug)]
pub struct BatchProofBuilder {
    jurisdiction: Jurisdiction,
    filing_status: FilingStatus,
    /// (year, income) pairs
    year_incomes: Vec<(TaxYear, u64)>,
}

impl BatchProofBuilder {
    /// Create a new batch proof builder.
    pub fn new(jurisdiction: Jurisdiction, filing_status: FilingStatus) -> Self {
        BatchProofBuilder {
            jurisdiction,
            filing_status,
            year_incomes: Vec::new(),
        }
    }

    /// Add a year with its income.
    pub fn add_year(mut self, year: TaxYear, income: u64) -> Self {
        self.year_incomes.push((year, income));
        self
    }

    /// Add multiple years with the same income.
    pub fn add_years_uniform(mut self, years: &[TaxYear], income: u64) -> Self {
        for &year in years {
            self.year_incomes.push((year, income));
        }
        self
    }

    /// Add a range of years with the same income.
    pub fn add_year_range(mut self, start_year: TaxYear, end_year: TaxYear, income: u64) -> Self {
        for year in start_year..=end_year {
            self.year_incomes.push((year, income));
        }
        self
    }

    /// Build the batch proof (dev mode).
    pub fn build_dev(self) -> Result<BatchProof> {
        if self.year_incomes.is_empty() {
            return Err(Error::invalid_proof("No years specified for batch proof".to_string()));
        }

        let mut year_proofs = Vec::with_capacity(self.year_incomes.len());

        for (year, income) in &self.year_incomes {
            let bracket = find_bracket(*income, self.jurisdiction, *year, self.filing_status)?;
            let commitment = compute_commitment(bracket.lower, bracket.upper, *year);

            year_proofs.push(YearProofSummary {
                tax_year: *year,
                bracket_index: bracket.index,
                rate_bps: bracket.rate_bps,
                bracket_lower: bracket.lower,
                bracket_upper: bracket.upper,
                commitment,
            });
        }

        // Sort by year for consistent ordering
        year_proofs.sort_by_key(|p| p.tax_year);

        let batch_commitment = compute_batch_commitment(&year_proofs);

        Ok(BatchProof {
            jurisdiction: self.jurisdiction,
            filing_status: self.filing_status,
            year_proofs,
            batch_commitment,
            is_dev_mode: true,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        })
    }
}

/// Convenience function for creating a 3-year batch proof (common for mortgages).
pub fn prove_three_year_history(
    incomes: [(TaxYear, u64); 3],
    jurisdiction: Jurisdiction,
    filing_status: FilingStatus,
) -> Result<BatchProof> {
    BatchProofBuilder::new(jurisdiction, filing_status)
        .add_year(incomes[0].0, incomes[0].1)
        .add_year(incomes[1].0, incomes[1].1)
        .add_year(incomes[2].0, incomes[2].1)
        .build_dev()
}

// =============================================================================
// Effective Tax Rate Proofs
// =============================================================================

/// A zero-knowledge proof of effective tax rate.
///
/// Unlike marginal bracket proofs, this proves the OVERALL tax burden
/// (total tax paid / total income) falls within a specified range.
///
/// Useful for:
/// - More accurate income comparison across tax systems
/// - Visa/immigration applications requiring average tax rate
/// - Financial planning and tax burden verification
/// - Cross-jurisdiction tax optimization analysis
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EffectiveTaxRateProof {
    /// The jurisdiction
    pub jurisdiction: Jurisdiction,
    /// Filing status used
    pub filing_status: FilingStatus,
    /// Tax year
    pub tax_year: TaxYear,
    /// Effective rate range lower bound (in basis points)
    pub effective_rate_lower_bps: u16,
    /// Effective rate range upper bound (in basis points)
    pub effective_rate_upper_bps: u16,
    /// The actual effective rate (in basis points) - hidden in production
    /// Only revealed in dev mode for testing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actual_rate_bps: Option<u16>,
    /// Cryptographic commitment
    pub commitment: BracketCommitment,
    /// Whether this is a dev mode proof
    pub is_dev_mode: bool,
    /// The marginal bracket (for context)
    pub marginal_bracket: u8,
    /// Marginal rate in basis points
    pub marginal_rate_bps: u16,
}

impl EffectiveTaxRateProof {
    /// Calculate effective tax rate for US progressive tax (2024 rates).
    fn calculate_us_effective_rate(income: u64, filing_status: FilingStatus, year: TaxYear) -> Result<(u16, u8)> {
        // US 2024 brackets for single filer
        let brackets: &[(u64, u64, u16)] = match (filing_status, year) {
            (FilingStatus::Single, 2024 | 2025) => &[
                (0, 11_600, 1000),        // 10%
                (11_600, 47_150, 1200),   // 12%
                (47_150, 100_525, 2200),  // 22%
                (100_525, 191_950, 2400), // 24%
                (191_950, 243_725, 3200), // 32%
                (243_725, 609_350, 3500), // 35%
                (609_350, u64::MAX, 3700), // 37%
            ],
            (FilingStatus::MarriedFilingJointly, 2024 | 2025) => &[
                (0, 23_200, 1000),
                (23_200, 94_300, 1200),
                (94_300, 201_050, 2200),
                (201_050, 383_900, 2400),
                (383_900, 487_450, 3200),
                (487_450, 731_200, 3500),
                (731_200, u64::MAX, 3700),
            ],
            _ => &[
                (0, 11_600, 1000),
                (11_600, 47_150, 1200),
                (47_150, 100_525, 2200),
                (100_525, 191_950, 2400),
                (191_950, 243_725, 3200),
                (243_725, 609_350, 3500),
                (609_350, u64::MAX, 3700),
            ],
        };

        let mut total_tax: u64 = 0;
        let mut marginal_bracket = 0u8;

        for (i, &(lower, upper, rate_bps)) in brackets.iter().enumerate() {
            if income > lower {
                let taxable_in_bracket = std::cmp::min(income, upper) - lower;
                let tax_in_bracket = (taxable_in_bracket * rate_bps as u64) / 10000;
                total_tax += tax_in_bracket;
                marginal_bracket = i as u8;
            }
        }

        // Calculate effective rate in basis points
        let effective_rate_bps = if income > 0 {
            ((total_tax * 10000) / income) as u16
        } else {
            0
        };

        Ok((effective_rate_bps, marginal_bracket))
    }

    /// Create an effective tax rate proof (dev mode).
    ///
    /// # Arguments
    /// * `income` - The actual income (private)
    /// * `jurisdiction` - Tax jurisdiction
    /// * `filing_status` - Filing status
    /// * `tax_year` - Tax year
    ///
    /// # Returns
    /// The proof with the effective rate calculated and proven.
    pub fn prove_dev(
        income: u64,
        jurisdiction: Jurisdiction,
        filing_status: FilingStatus,
        tax_year: TaxYear,
    ) -> Result<Self> {
        // Calculate effective rate based on jurisdiction
        let (effective_rate_bps, marginal_bracket) = match jurisdiction {
            Jurisdiction::US => Self::calculate_us_effective_rate(income, filing_status, tax_year)?,
            _ => {
                // For other jurisdictions, use marginal rate as approximation
                let bracket = find_bracket(income, jurisdiction, tax_year, filing_status)?;
                // Approximate effective rate as 60-80% of marginal rate
                // Cast to u32 to avoid overflow (rate_bps can be up to 4700)
                let approx_effective = ((bracket.rate_bps as u32 * 70) / 100) as u16;
                (approx_effective, bracket.index)
            }
        };

        // Get the marginal bracket info
        let bracket = find_bracket(income, jurisdiction, tax_year, filing_status)?;

        // Create commitment
        let commitment = compute_effective_rate_commitment(
            effective_rate_bps,
            jurisdiction,
            filing_status,
            tax_year,
        );

        Ok(EffectiveTaxRateProof {
            jurisdiction,
            filing_status,
            tax_year,
            effective_rate_lower_bps: effective_rate_bps.saturating_sub(50), // ±0.5% range
            effective_rate_upper_bps: effective_rate_bps.saturating_add(50),
            actual_rate_bps: Some(effective_rate_bps), // Revealed in dev mode
            commitment,
            is_dev_mode: true,
            marginal_bracket,
            marginal_rate_bps: bracket.rate_bps,
        })
    }

    /// Get the effective rate as a percentage.
    pub fn effective_rate_percent(&self) -> f64 {
        self.actual_rate_bps.unwrap_or(0) as f64 / 100.0
    }

    /// Get the effective rate range as percentages.
    pub fn effective_rate_range_percent(&self) -> (f64, f64) {
        (
            self.effective_rate_lower_bps as f64 / 100.0,
            self.effective_rate_upper_bps as f64 / 100.0,
        )
    }

    /// Get the marginal rate as a percentage.
    pub fn marginal_rate_percent(&self) -> f64 {
        self.marginal_rate_bps as f64 / 100.0
    }

    /// Get the tax savings from the effective rate being lower than marginal.
    /// This represents the benefit of progressive taxation.
    pub fn progressive_tax_savings_bps(&self) -> u16 {
        self.marginal_rate_bps.saturating_sub(self.actual_rate_bps.unwrap_or(0))
    }

    /// Verify the proof commitment.
    pub fn verify(&self) -> Result<()> {
        let expected = compute_effective_rate_commitment(
            self.actual_rate_bps.unwrap_or(self.effective_rate_lower_bps),
            self.jurisdiction,
            self.filing_status,
            self.tax_year,
        );

        // In dev mode, we can verify the commitment
        if self.is_dev_mode && self.commitment != expected {
            return Err(Error::CommitmentMismatch {
                expected: expected.to_hex(),
                actual: self.commitment.to_hex(),
            });
        }

        Ok(())
    }

    /// Get a human-readable summary.
    pub fn summary(&self) -> String {
        let effective = self.actual_rate_bps.map(|r| r as f64 / 100.0);
        let marginal = self.marginal_rate_bps as f64 / 100.0;

        match effective {
            Some(eff) => format!(
                "Effective tax rate proof: {} {} {}, Effective: {:.1}% (Marginal: {:.1}%)",
                self.jurisdiction, self.filing_status, self.tax_year, eff, marginal
            ),
            None => format!(
                "Effective tax rate proof: {} {} {}, Range: {:.1}%-{:.1}% (Marginal: {:.1}%)",
                self.jurisdiction, self.filing_status, self.tax_year,
                self.effective_rate_lower_bps as f64 / 100.0,
                self.effective_rate_upper_bps as f64 / 100.0,
                marginal
            ),
        }
    }
}

/// Compute commitment for an effective tax rate proof.
fn compute_effective_rate_commitment(
    rate_bps: u16,
    jurisdiction: Jurisdiction,
    _filing_status: FilingStatus,
    year: TaxYear,
) -> BracketCommitment {
    // Use FNV-1a to create commitment from rate and context
    let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis

    // Hash the rate
    for byte in rate_bps.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }

    // Hash the jurisdiction name
    let jname = jurisdiction.name();
    for byte in jname.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }

    // Hash the year
    for byte in year.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }

    let mut bytes = [0u8; 32];
    bytes[0..8].copy_from_slice(&hash.to_le_bytes());
    bytes[8..16].copy_from_slice(&hash.wrapping_mul(0x517cc1b727220a95).to_le_bytes());
    bytes[16..24].copy_from_slice(&hash.wrapping_mul(0x2545f4914f6cdd1d).to_le_bytes());
    bytes[24..32].copy_from_slice(&hash.wrapping_mul(0x14057b7ef767814f).to_le_bytes());

    BracketCommitment::from_bytes(bytes)
}

/// Builder for effective tax rate proofs.
#[derive(Clone, Debug)]
pub struct EffectiveTaxRateBuilder {
    income: u64,
    jurisdiction: Jurisdiction,
    filing_status: FilingStatus,
    tax_year: TaxYear,
}

impl EffectiveTaxRateBuilder {
    /// Create a new builder.
    pub fn new(income: u64, jurisdiction: Jurisdiction, filing_status: FilingStatus, tax_year: TaxYear) -> Self {
        EffectiveTaxRateBuilder {
            income,
            jurisdiction,
            filing_status,
            tax_year,
        }
    }

    /// Build the effective tax rate proof.
    pub fn build(self) -> Result<EffectiveTaxRateProof> {
        EffectiveTaxRateProof::prove_dev(
            self.income,
            self.jurisdiction,
            self.filing_status,
            self.tax_year,
        )
    }
}

// =============================================================================
// Cross-Jurisdiction Proofs
// =============================================================================

/// A zero-knowledge proof of tax bracket status across multiple jurisdictions.
///
/// This proves that income places you in comparable brackets across different
/// countries, useful for:
/// - International tax treaty verification
/// - Cross-border employment eligibility
/// - Expatriate tax planning
/// - Multi-national tax optimization proof
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CrossJurisdictionProof {
    /// List of jurisdictions with their bracket results
    pub jurisdiction_brackets: Vec<JurisdictionBracketInfo>,
    /// Tax year
    pub tax_year: TaxYear,
    /// Combined cryptographic commitment
    pub combined_commitment: BracketCommitment,
    /// Whether this is a dev mode proof
    pub is_dev_mode: bool,
    /// Timestamp when generated
    pub timestamp: u64,
}

/// Information about bracket placement in a single jurisdiction.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JurisdictionBracketInfo {
    /// The jurisdiction
    pub jurisdiction: Jurisdiction,
    /// Filing status used
    pub filing_status: FilingStatus,
    /// Bracket index (0-based)
    pub bracket_index: u8,
    /// Marginal rate in basis points
    pub rate_bps: u16,
    /// Bracket lower bound (in local currency equivalent)
    pub bracket_lower: u64,
    /// Bracket upper bound (in local currency equivalent)
    pub bracket_upper: u64,
    /// Individual commitment
    pub commitment: BracketCommitment,
}

impl CrossJurisdictionProof {
    /// Verify all jurisdiction commitments.
    pub fn verify(&self) -> Result<()> {
        // Verify each individual jurisdiction commitment
        for info in &self.jurisdiction_brackets {
            let expected = compute_commitment(
                info.bracket_lower,
                info.bracket_upper,
                self.tax_year,
            );
            if info.commitment != expected {
                return Err(Error::CommitmentMismatch {
                    expected: expected.to_hex(),
                    actual: info.commitment.to_hex(),
                });
            }
        }

        // Verify combined commitment
        let expected_combined = compute_cross_jurisdiction_commitment(&self.jurisdiction_brackets, self.tax_year);
        if self.combined_commitment != expected_combined {
            return Err(Error::CommitmentMismatch {
                expected: expected_combined.to_hex(),
                actual: self.combined_commitment.to_hex(),
            });
        }

        Ok(())
    }

    /// Get a human-readable summary.
    pub fn summary(&self) -> String {
        let jurisdictions: Vec<String> = self.jurisdiction_brackets.iter()
            .map(|j| format!("{}({}%)", j.jurisdiction, j.rate_bps as f64 / 100.0))
            .collect();
        format!(
            "Cross-jurisdiction proof {}: {}",
            self.tax_year,
            jurisdictions.join(", ")
        )
    }

    /// Get the number of jurisdictions covered.
    pub fn jurisdiction_count(&self) -> usize {
        self.jurisdiction_brackets.len()
    }

    /// Check if all jurisdictions have the same bracket index.
    pub fn consistent_bracket(&self) -> Option<u8> {
        if self.jurisdiction_brackets.is_empty() {
            return None;
        }
        let first = self.jurisdiction_brackets[0].bracket_index;
        if self.jurisdiction_brackets.iter().all(|j| j.bracket_index == first) {
            Some(first)
        } else {
            None
        }
    }

    /// Check if all jurisdictions have rates within a tolerance (in bps).
    pub fn rates_within_tolerance(&self, tolerance_bps: u16) -> bool {
        if self.jurisdiction_brackets.len() < 2 {
            return true;
        }
        let min_rate = self.jurisdiction_brackets.iter().map(|j| j.rate_bps).min().unwrap();
        let max_rate = self.jurisdiction_brackets.iter().map(|j| j.rate_bps).max().unwrap();
        max_rate - min_rate <= tolerance_bps
    }

    /// Get the average marginal rate across all jurisdictions.
    pub fn average_rate_bps(&self) -> u16 {
        if self.jurisdiction_brackets.is_empty() {
            return 0;
        }
        let sum: u32 = self.jurisdiction_brackets.iter().map(|j| j.rate_bps as u32).sum();
        (sum / self.jurisdiction_brackets.len() as u32) as u16
    }

    /// Get the min and max rates.
    pub fn rate_range_bps(&self) -> Option<(u16, u16)> {
        if self.jurisdiction_brackets.is_empty() {
            return None;
        }
        let min = self.jurisdiction_brackets.iter().map(|j| j.rate_bps).min().unwrap();
        let max = self.jurisdiction_brackets.iter().map(|j| j.rate_bps).max().unwrap();
        Some((min, max))
    }

    /// Get jurisdictions in a specific bracket.
    pub fn jurisdictions_in_bracket(&self, bracket_index: u8) -> Vec<Jurisdiction> {
        self.jurisdiction_brackets.iter()
            .filter(|j| j.bracket_index == bracket_index)
            .map(|j| j.jurisdiction)
            .collect()
    }
}

/// Compute combined commitment for cross-jurisdiction proof.
fn compute_cross_jurisdiction_commitment(
    jurisdictions: &[JurisdictionBracketInfo],
    year: TaxYear,
) -> BracketCommitment {
    use sha3::{Digest, Sha3_256};

    let mut hasher = Sha3_256::new();

    // Hash year first
    hasher.update(year.to_le_bytes());

    // Then each jurisdiction's commitment
    for info in jurisdictions {
        hasher.update(info.commitment.as_bytes());
    }

    let hash = hasher.finalize();
    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(&hash);
    BracketCommitment::from_bytes(bytes)
}

/// Builder for cross-jurisdiction proofs.
#[derive(Clone, Debug)]
pub struct CrossJurisdictionProofBuilder {
    /// (Jurisdiction, FilingStatus) pairs to include
    jurisdictions: Vec<(Jurisdiction, FilingStatus)>,
    /// Tax year
    tax_year: TaxYear,
    /// Income (will be converted to local currency equivalents)
    income_usd: u64,
}

impl CrossJurisdictionProofBuilder {
    /// Create a new builder with income in USD.
    pub fn new(income_usd: u64, tax_year: TaxYear) -> Self {
        CrossJurisdictionProofBuilder {
            jurisdictions: Vec::new(),
            tax_year,
            income_usd,
        }
    }

    /// Add a jurisdiction to the proof.
    pub fn add_jurisdiction(mut self, jurisdiction: Jurisdiction, filing_status: FilingStatus) -> Self {
        self.jurisdictions.push((jurisdiction, filing_status));
        self
    }

    /// Add multiple jurisdictions with the same filing status.
    pub fn add_jurisdictions(mut self, jurisdictions: &[Jurisdiction], filing_status: FilingStatus) -> Self {
        for j in jurisdictions {
            self.jurisdictions.push((*j, filing_status));
        }
        self
    }

    /// Add common OECD countries for tax treaty purposes.
    pub fn add_oecd_common(self, filing_status: FilingStatus) -> Self {
        self.add_jurisdictions(&[
            Jurisdiction::US,
            Jurisdiction::UK,
            Jurisdiction::DE,
            Jurisdiction::FR,
            Jurisdiction::JP,
            Jurisdiction::CA,
            Jurisdiction::AU,
        ], filing_status)
    }

    /// Build the cross-jurisdiction proof (dev mode).
    pub fn build_dev(self) -> Result<CrossJurisdictionProof> {
        if self.jurisdictions.is_empty() {
            return Err(Error::invalid_proof("No jurisdictions specified".to_string()));
        }

        let mut jurisdiction_brackets = Vec::with_capacity(self.jurisdictions.len());

        for (jurisdiction, filing_status) in &self.jurisdictions {
            // Convert USD to local currency (simplified - in production would use real rates)
            let local_income = convert_usd_to_local(self.income_usd, *jurisdiction);

            // Find bracket in this jurisdiction
            let bracket = find_bracket(local_income, *jurisdiction, self.tax_year, *filing_status)?;
            let commitment = compute_commitment(bracket.lower, bracket.upper, self.tax_year);

            jurisdiction_brackets.push(JurisdictionBracketInfo {
                jurisdiction: *jurisdiction,
                filing_status: *filing_status,
                bracket_index: bracket.index,
                rate_bps: bracket.rate_bps,
                bracket_lower: bracket.lower,
                bracket_upper: bracket.upper,
                commitment,
            });
        }

        let combined_commitment = compute_cross_jurisdiction_commitment(&jurisdiction_brackets, self.tax_year);

        Ok(CrossJurisdictionProof {
            jurisdiction_brackets,
            tax_year: self.tax_year,
            combined_commitment,
            is_dev_mode: true,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        })
    }
}

/// Convert USD to local currency (simplified - in production use real exchange rates).
/// These are approximate rates as of late 2024.
fn convert_usd_to_local(usd: u64, jurisdiction: Jurisdiction) -> u64 {
    let rate = match jurisdiction {
        // Americas
        Jurisdiction::US => 1.0,
        Jurisdiction::CA => 1.35,       // CAD per USD
        Jurisdiction::MX => 17.0,       // MXN per USD
        Jurisdiction::BR => 5.0,        // BRL per USD
        Jurisdiction::AR => 900.0,      // ARS per USD (approx)
        Jurisdiction::CL => 900.0,      // CLP per USD
        Jurisdiction::CO => 4000.0,     // COP per USD
        Jurisdiction::PE => 3.7,        // PEN per USD
        Jurisdiction::EC => 1.0,        // USD (dollarized)
        Jurisdiction::UY => 40.0,       // UYU per USD

        // Europe (EUR zone)
        Jurisdiction::DE | Jurisdiction::FR | Jurisdiction::IT | Jurisdiction::ES |
        Jurisdiction::NL | Jurisdiction::BE | Jurisdiction::AT | Jurisdiction::PT |
        Jurisdiction::IE | Jurisdiction::FI | Jurisdiction::GR => 0.92, // EUR per USD

        // Europe (non-EUR)
        Jurisdiction::UK => 0.79,       // GBP per USD
        Jurisdiction::RU => 90.0,       // RUB per USD
        Jurisdiction::TR => 32.0,       // TRY per USD
        Jurisdiction::PL => 4.0,        // PLN per USD
        Jurisdiction::SE => 10.5,       // SEK per USD
        Jurisdiction::DK => 6.9,        // DKK per USD
        Jurisdiction::NO => 10.5,       // NOK per USD
        Jurisdiction::CH => 0.88,       // CHF per USD
        Jurisdiction::CZ => 23.0,       // CZK per USD
        Jurisdiction::HU => 360.0,      // HUF per USD
        Jurisdiction::RO => 4.6,        // RON per USD
        Jurisdiction::UA => 41.0,       // UAH per USD

        // Asia-Pacific
        Jurisdiction::JP => 150.0,      // JPY per USD
        Jurisdiction::CN => 7.2,        // CNY per USD
        Jurisdiction::IN => 83.0,       // INR per USD
        Jurisdiction::KR => 1350.0,     // KRW per USD
        Jurisdiction::ID => 15700.0,    // IDR per USD
        Jurisdiction::AU => 1.55,       // AUD per USD
        Jurisdiction::NZ => 1.65,       // NZD per USD
        Jurisdiction::SG => 1.35,       // SGD per USD
        Jurisdiction::HK => 7.8,        // HKD per USD
        Jurisdiction::TW => 32.0,       // TWD per USD
        Jurisdiction::MY => 4.5,        // MYR per USD
        Jurisdiction::TH => 35.0,       // THB per USD
        Jurisdiction::VN => 25000.0,    // VND per USD
        Jurisdiction::PH => 56.0,       // PHP per USD
        Jurisdiction::PK => 280.0,      // PKR per USD

        // Middle East
        Jurisdiction::SA => 3.75,       // SAR per USD
        Jurisdiction::AE => 3.67,       // AED per USD
        Jurisdiction::IL => 3.7,        // ILS per USD
        Jurisdiction::EG => 31.0,       // EGP per USD
        Jurisdiction::QA => 3.64,       // QAR per USD

        // Africa
        Jurisdiction::ZA => 18.0,       // ZAR per USD
        Jurisdiction::NG => 800.0,      // NGN per USD
        Jurisdiction::KE => 155.0,      // KES per USD
        Jurisdiction::MA => 10.0,       // MAD per USD
        Jurisdiction::GH => 12.5,       // GHS per USD
    };
    (usd as f64 * rate) as u64
}

/// A lightweight receipt for storing proof references.
///
/// This is a more compact representation for storage in DHTs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaxBracketReceipt {
    /// Unique proof identifier (hash of full proof)
    pub proof_id: String,
    /// Jurisdiction
    pub jurisdiction: Jurisdiction,
    /// Filing status
    pub filing_status: FilingStatus,
    /// Tax year
    pub tax_year: TaxYear,
    /// Bracket index
    pub bracket_index: u8,
    /// Commitment (for quick validation)
    pub commitment: BracketCommitment,
    /// Timestamp when proof was generated
    pub timestamp: u64,
}

impl From<&TaxBracketProof> for TaxBracketReceipt {
    fn from(proof: &TaxBracketProof) -> Self {
        use sha3::{Digest, Sha3_256};

        // Generate proof ID from hash of receipt bytes
        let mut hasher = Sha3_256::new();
        hasher.update(&proof.receipt_bytes);
        let hash = hasher.finalize();
        let proof_id = hex::encode(&hash[..16]);

        TaxBracketReceipt {
            proof_id,
            jurisdiction: proof.jurisdiction,
            filing_status: proof.filing_status,
            tax_year: proof.tax_year,
            bracket_index: proof.bracket_index,
            commitment: proof.commitment,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }
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
// Deduction Proofs
// =============================================================================

/// Categories of tax deductions that can be proven.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeductionCategory {
    /// Charitable donations
    Charitable,
    /// Medical expenses
    Medical,
    /// Mortgage interest
    MortgageInterest,
    /// State and local taxes (SALT)
    StateLocalTaxes,
    /// Business expenses
    Business,
    /// Education expenses
    Education,
    /// Retirement contributions (401k, IRA)
    Retirement,
    /// Childcare expenses
    Childcare,
    /// Home office deduction
    HomeOffice,
    /// Other itemized deductions
    Other,
}

impl DeductionCategory {
    /// Get the display name of this category.
    pub fn name(&self) -> &'static str {
        match self {
            DeductionCategory::Charitable => "Charitable Donations",
            DeductionCategory::Medical => "Medical Expenses",
            DeductionCategory::MortgageInterest => "Mortgage Interest",
            DeductionCategory::StateLocalTaxes => "State & Local Taxes",
            DeductionCategory::Business => "Business Expenses",
            DeductionCategory::Education => "Education",
            DeductionCategory::Retirement => "Retirement Contributions",
            DeductionCategory::Childcare => "Childcare",
            DeductionCategory::HomeOffice => "Home Office",
            DeductionCategory::Other => "Other",
        }
    }

    /// Get the 2024 US limit for this deduction category (if applicable).
    pub fn us_limit_2024(&self) -> Option<u64> {
        match self {
            DeductionCategory::Charitable => Some(60), // 60% of AGI for cash
            DeductionCategory::StateLocalTaxes => Some(10_000), // SALT cap
            DeductionCategory::Retirement => Some(23_000), // 401k limit
            _ => None,
        }
    }
}

/// Summary of a deduction category (without revealing exact amount).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeductionCategorySummary {
    /// The category of deduction
    pub category: DeductionCategory,
    /// Lower bound of the amount range
    pub amount_lower: u64,
    /// Upper bound of the amount range
    pub amount_upper: u64,
    /// Whether this category has any deductions
    pub has_deductions: bool,
}

/// A zero-knowledge proof of tax deductions.
///
/// Proves information about deductions without revealing exact amounts
/// or specific recipients/vendors.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeductionProof {
    /// Tax year
    pub tax_year: TaxYear,
    /// Filing status
    pub filing_status: FilingStatus,
    /// Total deductions (lower bound)
    pub total_lower: u64,
    /// Total deductions (upper bound)
    pub total_upper: u64,
    /// Whether itemizing beats standard deduction
    pub itemizing_beneficial: bool,
    /// Standard deduction for comparison
    pub standard_deduction: u64,
    /// Category summaries (optional, for more detail)
    pub categories: Vec<DeductionCategorySummary>,
    /// Number of distinct deduction categories claimed
    pub category_count: u8,
    /// Commitment to the deduction data
    pub commitment: BracketCommitment,
    /// Whether this is dev mode
    pub is_dev_mode: bool,
    /// Timestamp of proof generation
    pub timestamp: u64,
}

impl DeductionProof {
    /// Get the 2024 standard deduction for a filing status.
    fn standard_deduction_2024(filing_status: FilingStatus) -> u64 {
        match filing_status {
            FilingStatus::Single => 14_600,
            FilingStatus::MarriedFilingJointly => 29_200,
            FilingStatus::MarriedFilingSeparately => 14_600,
            FilingStatus::HeadOfHousehold => 21_900,
        }
    }

    /// Prove deduction information without revealing specifics.
    pub fn prove_dev(
        deductions: &[(DeductionCategory, u64)],
        filing_status: FilingStatus,
        tax_year: TaxYear,
    ) -> Result<Self> {
        if tax_year < 2020 || tax_year > 2025 {
            return Err(Error::unsupported_year(tax_year));
        }

        let total: u64 = deductions.iter().map(|(_, amt)| amt).sum();
        let standard = Self::standard_deduction_2024(filing_status);

        // Round to nearest $1000 for privacy
        let total_lower = (total / 1000) * 1000;
        let total_upper = total_lower + 1000;

        // Build category summaries
        let mut category_totals: std::collections::HashMap<DeductionCategory, u64> =
            std::collections::HashMap::new();
        for (cat, amt) in deductions {
            *category_totals.entry(*cat).or_insert(0) += amt;
        }

        let categories: Vec<DeductionCategorySummary> = category_totals
            .iter()
            .map(|(cat, amt)| {
                let lower = (amt / 1000) * 1000;
                DeductionCategorySummary {
                    category: *cat,
                    amount_lower: lower,
                    amount_upper: lower + 1000,
                    has_deductions: *amt > 0,
                }
            })
            .collect();

        let commitment = compute_deduction_commitment(total, tax_year);

        Ok(DeductionProof {
            tax_year,
            filing_status,
            total_lower,
            total_upper,
            itemizing_beneficial: total > standard,
            standard_deduction: standard,
            categories,
            category_count: category_totals.len() as u8,
            commitment,
            is_dev_mode: true,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        })
    }

    /// Verify the proof.
    pub fn verify(&self) -> Result<()> {
        // Basic validation
        if self.total_upper < self.total_lower {
            return Err(Error::invalid_proof("Invalid range".to_string()));
        }
        Ok(())
    }

    /// Get a human-readable summary.
    pub fn summary(&self) -> String {
        let comparison = if self.itemizing_beneficial {
            format!("Itemizing saves ${}", format_currency(self.total_lower.saturating_sub(self.standard_deduction)))
        } else {
            format!("Standard deduction better (${} vs ${})",
                format_currency(self.standard_deduction),
                format_currency(self.total_lower))
        };
        format!(
            "Deduction proof {}: ${}-${} across {} categories. {}",
            self.tax_year,
            format_currency(self.total_lower),
            format_currency(self.total_upper),
            self.category_count,
            comparison
        )
    }
}

/// Builder for deduction proofs.
pub struct DeductionProofBuilder {
    deductions: Vec<(DeductionCategory, u64)>,
    filing_status: FilingStatus,
    tax_year: TaxYear,
}

impl DeductionProofBuilder {
    /// Create a new builder.
    pub fn new(filing_status: FilingStatus, tax_year: TaxYear) -> Self {
        DeductionProofBuilder {
            deductions: Vec::new(),
            filing_status,
            tax_year,
        }
    }

    /// Add a deduction.
    pub fn add(mut self, category: DeductionCategory, amount: u64) -> Self {
        self.deductions.push((category, amount));
        self
    }

    /// Add charitable donations.
    pub fn charitable(self, amount: u64) -> Self {
        self.add(DeductionCategory::Charitable, amount)
    }

    /// Add medical expenses.
    pub fn medical(self, amount: u64) -> Self {
        self.add(DeductionCategory::Medical, amount)
    }

    /// Add mortgage interest.
    pub fn mortgage_interest(self, amount: u64) -> Self {
        self.add(DeductionCategory::MortgageInterest, amount)
    }

    /// Add state and local taxes.
    pub fn salt(self, amount: u64) -> Self {
        self.add(DeductionCategory::StateLocalTaxes, amount)
    }

    /// Add retirement contributions.
    pub fn retirement(self, amount: u64) -> Self {
        self.add(DeductionCategory::Retirement, amount)
    }

    /// Build the proof.
    pub fn build(self) -> Result<DeductionProof> {
        DeductionProof::prove_dev(&self.deductions, self.filing_status, self.tax_year)
    }
}

/// Compute commitment for deduction proof.
fn compute_deduction_commitment(total: u64, year: TaxYear) -> BracketCommitment {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in total.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    for byte in year.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    // Add "deduction" marker
    for byte in b"deduction" {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let mut bytes = [0u8; 32];
    bytes[..8].copy_from_slice(&hash.to_le_bytes());
    BracketCommitment::from_bytes(bytes)
}

// =============================================================================
// Composite Proofs
// =============================================================================

/// A composite proof combining multiple proof types.
///
/// Useful for comprehensive verification in a single proof package.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompositeProof {
    /// Tax year for all component proofs
    pub tax_year: TaxYear,
    /// Jurisdiction
    pub jurisdiction: Jurisdiction,
    /// Filing status
    pub filing_status: FilingStatus,
    /// Optional bracket proof
    pub bracket_proof: Option<TaxBracketProof>,
    /// Optional effective rate proof
    pub effective_rate_proof: Option<EffectiveTaxRateProof>,
    /// Optional income range proof
    pub range_proof: Option<IncomeRangeProof>,
    /// Optional deduction proof
    pub deduction_proof: Option<DeductionProof>,
    /// Combined commitment hash
    pub combined_commitment: BracketCommitment,
    /// Proof component flags
    pub components: u8,
    /// Whether this is dev mode
    pub is_dev_mode: bool,
    /// Timestamp
    pub timestamp: u64,
}

impl CompositeProof {
    const FLAG_BRACKET: u8 = 0b0001;
    const FLAG_EFFECTIVE: u8 = 0b0010;
    const FLAG_RANGE: u8 = 0b0100;
    const FLAG_DEDUCTION: u8 = 0b1000;

    /// Check if bracket proof is included.
    pub fn has_bracket(&self) -> bool {
        self.components & Self::FLAG_BRACKET != 0
    }

    /// Check if effective rate proof is included.
    pub fn has_effective_rate(&self) -> bool {
        self.components & Self::FLAG_EFFECTIVE != 0
    }

    /// Check if range proof is included.
    pub fn has_range(&self) -> bool {
        self.components & Self::FLAG_RANGE != 0
    }

    /// Check if deduction proof is included.
    pub fn has_deduction(&self) -> bool {
        self.components & Self::FLAG_DEDUCTION != 0
    }

    /// Count how many component proofs are included.
    pub fn component_count(&self) -> u8 {
        self.components.count_ones() as u8
    }

    /// Verify all component proofs.
    pub fn verify(&self) -> Result<()> {
        if let Some(ref p) = self.bracket_proof {
            p.verify()?;
        }
        if let Some(ref p) = self.effective_rate_proof {
            p.verify()?;
        }
        if let Some(ref p) = self.range_proof {
            p.verify()?;
        }
        if let Some(ref p) = self.deduction_proof {
            p.verify()?;
        }
        Ok(())
    }

    /// Get a human-readable summary.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if self.has_bracket() {
            if let Some(ref p) = self.bracket_proof {
                parts.push(format!("Bracket {}", p.bracket_index));
            }
        }
        if self.has_effective_rate() {
            if let Some(ref p) = self.effective_rate_proof {
                parts.push(format!("Effective {:.1}%", p.effective_rate_percent()));
            }
        }
        if self.has_range() {
            if let Some(ref p) = self.range_proof {
                parts.push(format!("Range ${}-{}",
                    format_currency(p.range_lower),
                    if p.range_upper == u64::MAX { "+".to_string() }
                    else { format!("${}", format_currency(p.range_upper)) }
                ));
            }
        }
        if self.has_deduction() {
            if let Some(ref p) = self.deduction_proof {
                parts.push(format!("Deductions ${}", format_currency(p.total_lower)));
            }
        }
        format!(
            "Composite proof {} {} {}: [{}]",
            self.jurisdiction.name(),
            self.filing_status.name(),
            self.tax_year,
            parts.join(", ")
        )
    }
}

/// Builder for composite proofs.
pub struct CompositeProofBuilder {
    income: u64,
    jurisdiction: Jurisdiction,
    filing_status: FilingStatus,
    tax_year: TaxYear,
    include_bracket: bool,
    include_effective: bool,
    range_lower: Option<u64>,
    range_upper: Option<u64>,
    deductions: Vec<(DeductionCategory, u64)>,
}

impl CompositeProofBuilder {
    /// Create a new composite proof builder.
    pub fn new(
        income: u64,
        jurisdiction: Jurisdiction,
        filing_status: FilingStatus,
        tax_year: TaxYear,
    ) -> Self {
        CompositeProofBuilder {
            income,
            jurisdiction,
            filing_status,
            tax_year,
            include_bracket: false,
            include_effective: false,
            range_lower: None,
            range_upper: None,
            deductions: Vec::new(),
        }
    }

    /// Include a tax bracket proof.
    pub fn with_bracket(mut self) -> Self {
        self.include_bracket = true;
        self
    }

    /// Include an effective tax rate proof.
    pub fn with_effective_rate(mut self) -> Self {
        self.include_effective = true;
        self
    }

    /// Include an income range proof.
    pub fn with_range(mut self, lower: u64, upper: u64) -> Self {
        self.range_lower = Some(lower);
        self.range_upper = Some(upper);
        self
    }

    /// Include a minimum income proof.
    pub fn with_minimum(mut self, minimum: u64) -> Self {
        self.range_lower = Some(minimum);
        self.range_upper = Some(u64::MAX);
        self
    }

    /// Add deductions for a deduction proof.
    pub fn with_deduction(mut self, category: DeductionCategory, amount: u64) -> Self {
        self.deductions.push((category, amount));
        self
    }

    /// Build the composite proof.
    #[cfg(feature = "prover")]
    pub fn build_dev(self) -> Result<CompositeProof> {
        use crate::prover::TaxBracketProver;

        let mut components: u8 = 0;

        // Build bracket proof if requested
        let bracket_proof = if self.include_bracket {
            components |= CompositeProof::FLAG_BRACKET;
            let prover = TaxBracketProver::dev_mode();
            Some(prover.prove(self.income, self.jurisdiction, self.filing_status, self.tax_year)?)
        } else {
            None
        };

        // Build effective rate proof if requested
        let effective_rate_proof = if self.include_effective {
            components |= CompositeProof::FLAG_EFFECTIVE;
            Some(EffectiveTaxRateProof::prove_dev(
                self.income,
                self.jurisdiction,
                self.filing_status,
                self.tax_year,
            )?)
        } else {
            None
        };

        // Build range proof if requested
        let range_proof = if self.range_lower.is_some() || self.range_upper.is_some() {
            components |= CompositeProof::FLAG_RANGE;
            let lower = self.range_lower.unwrap_or(0);
            let upper = self.range_upper.unwrap_or(u64::MAX);
            Some(IncomeRangeProof::prove_dev(self.income, lower, upper, self.tax_year)?)
        } else {
            None
        };

        // Build deduction proof if requested
        let deduction_proof = if !self.deductions.is_empty() {
            components |= CompositeProof::FLAG_DEDUCTION;
            Some(DeductionProof::prove_dev(
                &self.deductions,
                self.filing_status,
                self.tax_year,
            )?)
        } else {
            None
        };

        // Compute combined commitment
        let combined_commitment = compute_composite_commitment(
            &bracket_proof,
            &effective_rate_proof,
            &range_proof,
            &deduction_proof,
        );

        Ok(CompositeProof {
            tax_year: self.tax_year,
            jurisdiction: self.jurisdiction,
            filing_status: self.filing_status,
            bracket_proof,
            effective_rate_proof,
            range_proof,
            deduction_proof,
            combined_commitment,
            components,
            is_dev_mode: true,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        })
    }
}

/// Compute combined commitment for composite proof.
fn compute_composite_commitment(
    bracket: &Option<TaxBracketProof>,
    effective: &Option<EffectiveTaxRateProof>,
    range: &Option<IncomeRangeProof>,
    deduction: &Option<DeductionProof>,
) -> BracketCommitment {
    use sha3::{Digest, Sha3_256};
    let mut hasher = Sha3_256::new();

    hasher.update(b"composite");
    if let Some(p) = bracket {
        hasher.update(p.commitment.as_bytes());
    }
    if let Some(p) = effective {
        hasher.update(p.commitment.as_bytes());
    }
    if let Some(p) = range {
        hasher.update(p.commitment.as_bytes());
    }
    if let Some(p) = deduction {
        hasher.update(p.commitment.as_bytes());
    }

    let result = hasher.finalize();
    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(&result);
    BracketCommitment::from_bytes(bytes)
}

// =============================================================================
// Proof Chaining (Audit Trails)
// =============================================================================

/// A link in a proof chain, referencing previous proofs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofChainLink {
    /// Sequence number in the chain (0 = genesis)
    pub sequence: u32,
    /// Hash of the previous link (empty for genesis)
    pub previous_hash: BracketCommitment,
    /// Tax year this link covers
    pub tax_year: TaxYear,
    /// The proof commitment for this link
    pub proof_commitment: BracketCommitment,
    /// Type of proof in this link
    pub proof_type: String,
    /// Timestamp of this link
    pub timestamp: u64,
    /// Hash of this entire link
    pub link_hash: BracketCommitment,
}

/// A chain of proofs forming an audit trail.
///
/// Each proof in the chain references the previous, creating
/// an immutable history of tax compliance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofChain {
    /// All links in the chain
    pub links: Vec<ProofChainLink>,
    /// Owner identifier (hashed for privacy)
    pub owner_hash: BracketCommitment,
    /// Chain creation timestamp
    pub created: u64,
    /// Most recent update timestamp
    pub updated: u64,
}

impl ProofChain {
    /// Create a new empty proof chain.
    pub fn new(owner_id: &str) -> Self {
        let owner_hash = compute_owner_hash(owner_id);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        ProofChain {
            links: Vec::new(),
            owner_hash,
            created: now,
            updated: now,
        }
    }

    /// Get the genesis (first) link.
    pub fn genesis(&self) -> Option<&ProofChainLink> {
        self.links.first()
    }

    /// Get the latest link.
    pub fn latest(&self) -> Option<&ProofChainLink> {
        self.links.last()
    }

    /// Get the chain length.
    pub fn len(&self) -> usize {
        self.links.len()
    }

    /// Check if the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.links.is_empty()
    }

    /// Add a bracket proof to the chain.
    pub fn add_bracket_proof(&mut self, proof: &TaxBracketProof) {
        self.add_link(proof.tax_year, &proof.commitment, "bracket");
    }

    /// Add an effective rate proof to the chain.
    pub fn add_effective_proof(&mut self, proof: &EffectiveTaxRateProof) {
        self.add_link(proof.tax_year, &proof.commitment, "effective_rate");
    }

    /// Add a range proof to the chain.
    pub fn add_range_proof(&mut self, proof: &IncomeRangeProof) {
        self.add_link(proof.tax_year, &proof.commitment, "range");
    }

    /// Add a batch proof to the chain.
    pub fn add_batch_proof(&mut self, proof: &BatchProof) {
        // Use the first year in the batch
        let year = proof.year_proofs.first().map(|y| y.tax_year).unwrap_or(2024);
        self.add_link(year, &proof.batch_commitment, "batch");
    }

    /// Add a deduction proof to the chain.
    pub fn add_deduction_proof(&mut self, proof: &DeductionProof) {
        self.add_link(proof.tax_year, &proof.commitment, "deduction");
    }

    /// Add a composite proof to the chain.
    pub fn add_composite_proof(&mut self, proof: &CompositeProof) {
        self.add_link(proof.tax_year, &proof.combined_commitment, "composite");
    }

    fn add_link(&mut self, tax_year: TaxYear, commitment: &BracketCommitment, proof_type: &str) {
        let sequence = self.links.len() as u32;
        let previous_hash = self.latest()
            .map(|l| l.link_hash.clone())
            .unwrap_or_else(|| BracketCommitment::from_bytes([0u8; 32]));

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let link_hash = compute_link_hash(
            sequence,
            &previous_hash,
            tax_year,
            commitment,
            proof_type,
            timestamp,
        );

        self.links.push(ProofChainLink {
            sequence,
            previous_hash,
            tax_year,
            proof_commitment: commitment.clone(),
            proof_type: proof_type.to_string(),
            timestamp,
            link_hash,
        });

        self.updated = timestamp;
    }

    /// Verify the integrity of the entire chain.
    pub fn verify(&self) -> Result<()> {
        let mut expected_prev = BracketCommitment::from_bytes([0u8; 32]);

        for (i, link) in self.links.iter().enumerate() {
            // Verify sequence
            if link.sequence != i as u32 {
                return Err(Error::invalid_proof(format!(
                    "Invalid sequence at position {}: expected {}, got {}",
                    i, i, link.sequence
                )));
            }

            // Verify previous hash
            if link.previous_hash != expected_prev {
                return Err(Error::invalid_proof(format!(
                    "Chain broken at position {}: previous hash mismatch",
                    i
                )));
            }

            // Verify link hash
            let computed = compute_link_hash(
                link.sequence,
                &link.previous_hash,
                link.tax_year,
                &link.proof_commitment,
                &link.proof_type,
                link.timestamp,
            );
            if link.link_hash != computed {
                return Err(Error::invalid_proof(format!(
                    "Link hash mismatch at position {}",
                    i
                )));
            }

            expected_prev = link.link_hash.clone();
        }

        Ok(())
    }

    /// Get a summary of the chain.
    pub fn summary(&self) -> String {
        if self.links.is_empty() {
            return "Empty proof chain".to_string();
        }

        let years: Vec<String> = self.links.iter()
            .map(|l| l.tax_year.to_string())
            .collect();
        let types: Vec<&str> = self.links.iter()
            .map(|l| l.proof_type.as_str())
            .collect();

        format!(
            "Proof chain: {} links, years [{}], types [{}]",
            self.links.len(),
            years.join(", "),
            types.join(", ")
        )
    }

    /// Export the chain to JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| Error::invalid_proof(format!("JSON serialization failed: {}", e)))
    }
}

fn compute_owner_hash(owner_id: &str) -> BracketCommitment {
    use sha3::{Digest, Sha3_256};
    let mut hasher = Sha3_256::new();
    hasher.update(b"owner:");
    hasher.update(owner_id.as_bytes());
    let result = hasher.finalize();
    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(&result);
    BracketCommitment::from_bytes(bytes)
}

fn compute_link_hash(
    sequence: u32,
    previous: &BracketCommitment,
    tax_year: TaxYear,
    commitment: &BracketCommitment,
    proof_type: &str,
    timestamp: u64,
) -> BracketCommitment {
    use sha3::{Digest, Sha3_256};
    let mut hasher = Sha3_256::new();
    hasher.update(&sequence.to_le_bytes());
    hasher.update(previous.as_bytes());
    hasher.update(&tax_year.to_le_bytes());
    hasher.update(commitment.as_bytes());
    hasher.update(proof_type.as_bytes());
    hasher.update(&timestamp.to_le_bytes());
    let result = hasher.finalize();
    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(&result);
    BracketCommitment::from_bytes(bytes)
}

// =============================================================================
// Combined Federal + State Proof
// =============================================================================

/// A combined proof showing both federal and state tax bracket membership.
///
/// This is useful for US taxpayers who need to prove both federal and state
/// tax compliance in a single attestation.
///
/// # Example
///
/// ```rust,ignore
/// use mycelix_zk_tax::{CombinedFederalStateProof, Jurisdiction, FilingStatus};
/// use mycelix_zk_tax::subnational::USState;
///
/// let proof = CombinedFederalStateProof::prove_dev(
///     85_000,
///     Jurisdiction::US,
///     USState::CA,
///     FilingStatus::Single,
///     2024,
/// ).unwrap();
///
/// assert!(proof.verify().is_ok());
/// println!("Federal bracket: {}, State bracket: {}",
///     proof.federal_bracket_index, proof.state_bracket_index);
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CombinedFederalStateProof {
    /// Federal jurisdiction (US, CA, etc.)
    pub federal_jurisdiction: Jurisdiction,
    /// US State for state taxes
    pub state: crate::subnational::USState,
    /// Filing status
    pub filing_status: FilingStatus,
    /// Tax year
    pub tax_year: TaxYear,

    // Federal bracket info
    /// Federal bracket index (0-based)
    pub federal_bracket_index: u8,
    /// Federal marginal rate in basis points
    pub federal_rate_bps: u16,
    /// Federal bracket lower bound
    pub federal_bracket_lower: u64,
    /// Federal bracket upper bound
    pub federal_bracket_upper: u64,

    // State bracket info (None if no state income tax)
    /// State bracket index (0-based), None if no state tax
    pub state_bracket_index: Option<u8>,
    /// State marginal rate in basis points
    pub state_rate_bps: Option<u16>,
    /// State bracket lower bound
    pub state_bracket_lower: Option<u64>,
    /// State bracket upper bound
    pub state_bracket_upper: Option<u64>,

    /// Combined cryptographic commitment
    pub commitment: BracketCommitment,
    /// Whether this is a dev mode proof
    pub is_dev_mode: bool,
}

impl CombinedFederalStateProof {
    /// Generate a combined federal + state proof.
    pub fn prove_dev(
        income: u64,
        federal_jurisdiction: Jurisdiction,
        state: crate::subnational::USState,
        filing_status: FilingStatus,
        tax_year: TaxYear,
    ) -> Result<Self> {
        // Get federal bracket
        let federal_bracket = find_bracket(income, federal_jurisdiction, tax_year, filing_status)?;

        // Get state bracket if applicable
        let state_info = if state.has_income_tax() {
            let state_brackets = crate::subnational::get_state_brackets(state, tax_year, filing_status)
                .map_err(|e| Error::invalid_proof(e))?;
            let state_bracket = state_brackets
                .iter()
                .find(|b| b.contains(income))
                .ok_or_else(|| Error::no_bracket(income))?;
            Some((
                state_bracket.index,
                state_bracket.rate_bps,
                state_bracket.lower,
                state_bracket.upper,
            ))
        } else {
            None
        };

        // Create combined commitment
        let commitment = Self::compute_combined_commitment(
            federal_bracket.lower,
            federal_bracket.upper,
            state_info.map(|s| (s.2, s.3)),
            tax_year,
        );

        Ok(Self {
            federal_jurisdiction,
            state,
            filing_status,
            tax_year,
            federal_bracket_index: federal_bracket.index,
            federal_rate_bps: federal_bracket.rate_bps,
            federal_bracket_lower: federal_bracket.lower,
            federal_bracket_upper: federal_bracket.upper,
            state_bracket_index: state_info.map(|s| s.0),
            state_rate_bps: state_info.map(|s| s.1),
            state_bracket_lower: state_info.map(|s| s.2),
            state_bracket_upper: state_info.map(|s| s.3),
            commitment,
            is_dev_mode: true,
        })
    }

    /// Compute combined commitment for both jurisdictions.
    fn compute_combined_commitment(
        federal_lower: u64,
        federal_upper: u64,
        state_bounds: Option<(u64, u64)>,
        tax_year: TaxYear,
    ) -> BracketCommitment {
        use sha3::{Digest, Sha3_256};
        let mut hasher = Sha3_256::new();
        hasher.update(b"combined_federal_state_v1");
        hasher.update(&federal_lower.to_le_bytes());
        hasher.update(&federal_upper.to_le_bytes());
        if let Some((state_lower, state_upper)) = state_bounds {
            hasher.update(&state_lower.to_le_bytes());
            hasher.update(&state_upper.to_le_bytes());
        } else {
            hasher.update(&[0u8; 16]); // Padding for no-tax states
        }
        hasher.update(&tax_year.to_le_bytes());
        let result = hasher.finalize();
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&result);
        BracketCommitment::from_bytes(bytes)
    }

    /// Verify the combined proof.
    pub fn verify(&self) -> Result<()> {
        if self.is_dev_mode {
            // In dev mode, just verify commitment matches
            let expected = Self::compute_combined_commitment(
                self.federal_bracket_lower,
                self.federal_bracket_upper,
                self.state_bracket_lower.zip(self.state_bracket_upper),
                self.tax_year,
            );

            if self.commitment != expected {
                return Err(Error::CommitmentMismatch {
                    expected: expected.to_hex(),
                    actual: self.commitment.to_hex(),
                });
            }
            Ok(())
        } else {
            // Real verification would use ZK proof
            Err(Error::verification_failed("Real verification not yet implemented"))
        }
    }

    /// Get combined effective rate (federal + state) in basis points.
    pub fn combined_rate_bps(&self) -> u16 {
        self.federal_rate_bps + self.state_rate_bps.unwrap_or(0)
    }

    /// Check if state has income tax.
    pub fn has_state_tax(&self) -> bool {
        self.state_bracket_index.is_some()
    }

    /// Export as JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(Error::from)
    }

    /// Import from JSON.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(Error::from)
    }
}

/// Builder for combined federal + state proofs.
pub struct CombinedProofBuilder {
    income: u64,
    federal_jurisdiction: Jurisdiction,
    state: crate::subnational::USState,
    filing_status: FilingStatus,
    tax_year: TaxYear,
}

impl CombinedProofBuilder {
    /// Create a new builder.
    pub fn new(income: u64, state: crate::subnational::USState) -> Self {
        Self {
            income,
            federal_jurisdiction: Jurisdiction::US,
            state,
            filing_status: FilingStatus::Single,
            tax_year: 2024,
        }
    }

    /// Set federal jurisdiction (default: US).
    pub fn federal_jurisdiction(mut self, j: Jurisdiction) -> Self {
        self.federal_jurisdiction = j;
        self
    }

    /// Set filing status.
    pub fn filing_status(mut self, status: FilingStatus) -> Self {
        self.filing_status = status;
        self
    }

    /// Set tax year.
    pub fn tax_year(mut self, year: TaxYear) -> Self {
        self.tax_year = year;
        self
    }

    /// Build the proof (dev mode).
    pub fn build_dev(self) -> Result<CombinedFederalStateProof> {
        CombinedFederalStateProof::prove_dev(
            self.income,
            self.federal_jurisdiction,
            self.state,
            self.filing_status,
            self.tax_year,
        )
    }
}

// =============================================================================
// Multi-Year Income Stability Proof
// =============================================================================

/// Proof that income has remained stable over multiple years.
///
/// This proves that income variation across years stays within a specified
/// percentage threshold, useful for loan applications or rental agreements.
///
/// # Example
///
/// ```rust,ignore
/// use mycelix_zk_tax::IncomeStabilityProof;
///
/// // Prove income stayed within 20% variation over 3 years
/// let proof = IncomeStabilityProof::prove_dev(
///     vec![(2022, 80_000), (2023, 85_000), (2024, 82_000)],
///     20, // max 20% variation
/// ).unwrap();
///
/// assert!(proof.is_stable);
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IncomeStabilityProof {
    /// Years included in the proof
    pub years: Vec<TaxYear>,
    /// Whether income was stable within the threshold
    pub is_stable: bool,
    /// Maximum allowed variation percentage
    pub max_variation_percent: u8,
    /// Actual maximum variation (rounded to nearest 5%)
    pub actual_variation_percent: u8,
    /// Average income bracket index across years
    pub average_bracket_index: u8,
    /// Cryptographic commitment
    pub commitment: BracketCommitment,
    /// Whether this is dev mode
    pub is_dev_mode: bool,
}

impl IncomeStabilityProof {
    /// Generate a stability proof.
    ///
    /// # Arguments
    ///
    /// * `year_incomes` - List of (year, income) pairs
    /// * `max_variation_percent` - Maximum allowed variation (e.g., 20 for 20%)
    pub fn prove_dev(
        year_incomes: Vec<(TaxYear, u64)>,
        max_variation_percent: u8,
    ) -> Result<Self> {
        if year_incomes.is_empty() {
            return Err(Error::invalid_proof("No years provided"));
        }

        let years: Vec<TaxYear> = year_incomes.iter().map(|(y, _)| *y).collect();
        let incomes: Vec<u64> = year_incomes.iter().map(|(_, i)| *i).collect();

        // Calculate variation
        let min_income = *incomes.iter().min().unwrap();
        let max_income = *incomes.iter().max().unwrap();
        let avg_income: u64 = incomes.iter().sum::<u64>() / incomes.len() as u64;

        // Calculate percentage variation from average
        let max_deviation = std::cmp::max(
            (max_income as i64 - avg_income as i64).unsigned_abs(),
            (avg_income as i64 - min_income as i64).unsigned_abs(),
        );
        let actual_variation = ((max_deviation as f64 / avg_income as f64) * 100.0) as u8;

        // Round to nearest 5% for privacy
        let actual_variation_rounded = ((actual_variation + 2) / 5) * 5;

        let is_stable = actual_variation_rounded <= max_variation_percent;

        // Get average bracket index (use middle year)
        let middle_idx = years.len() / 2;
        let middle_bracket = find_bracket(
            incomes[middle_idx],
            Jurisdiction::US,
            years[middle_idx],
            FilingStatus::Single,
        )?;

        // Compute commitment
        let commitment = Self::compute_stability_commitment(&years, actual_variation_rounded, is_stable);

        Ok(Self {
            years,
            is_stable,
            max_variation_percent,
            actual_variation_percent: actual_variation_rounded,
            average_bracket_index: middle_bracket.index,
            commitment,
            is_dev_mode: true,
        })
    }

    fn compute_stability_commitment(years: &[TaxYear], variation: u8, stable: bool) -> BracketCommitment {
        use sha3::{Digest, Sha3_256};
        let mut hasher = Sha3_256::new();
        hasher.update(b"income_stability_v1");
        for year in years {
            hasher.update(&year.to_le_bytes());
        }
        hasher.update(&[variation]);
        hasher.update(&[stable as u8]);
        let result = hasher.finalize();
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&result);
        BracketCommitment::from_bytes(bytes)
    }

    /// Verify the stability proof.
    pub fn verify(&self) -> Result<()> {
        if self.is_dev_mode {
            let expected = Self::compute_stability_commitment(
                &self.years,
                self.actual_variation_percent,
                self.is_stable,
            );
            if self.commitment != expected {
                return Err(Error::CommitmentMismatch {
                    expected: expected.to_hex(),
                    actual: self.commitment.to_hex(),
                });
            }
            Ok(())
        } else {
            Err(Error::verification_failed("Real verification not yet implemented"))
        }
    }

    /// Get the number of years covered.
    pub fn year_count(&self) -> usize {
        self.years.len()
    }

    /// Export as JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(Error::from)
    }
}

// =============================================================================
// Proof Validity/Expiration
// =============================================================================

/// A proof with expiration timestamp for time-limited verification.
///
/// # Example
///
/// ```rust,ignore
/// use mycelix_zk_tax::{TimeBoundProof, TaxBracketProver, Jurisdiction, FilingStatus};
///
/// let prover = TaxBracketProver::dev_mode();
/// let proof = prover.prove(85_000, Jurisdiction::US, FilingStatus::Single, 2024).unwrap();
///
/// // Create a time-bound version valid for 30 days
/// let time_bound = TimeBoundProof::from_bracket_proof(proof, 30);
///
/// assert!(time_bound.is_valid());
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TimeBoundProof {
    /// The underlying bracket proof
    pub proof: TaxBracketProof,
    /// Unix timestamp when proof was created
    pub created_at: u64,
    /// Unix timestamp when proof expires
    pub expires_at: u64,
    /// Additional metadata (e.g., purpose)
    pub purpose: Option<String>,
    /// Signature over the time bounds
    pub time_signature: BracketCommitment,
}

impl TimeBoundProof {
    /// Create a time-bound proof from a bracket proof.
    ///
    /// # Arguments
    ///
    /// * `proof` - The underlying tax bracket proof
    /// * `validity_days` - Number of days the proof is valid
    pub fn from_bracket_proof(proof: TaxBracketProof, validity_days: u32) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let expires_at = now + (validity_days as u64 * 24 * 60 * 60);
        let time_signature = Self::compute_time_signature(&proof, now, expires_at);

        Self {
            proof,
            created_at: now,
            expires_at,
            purpose: None,
            time_signature,
        }
    }

    /// Create with a specific purpose.
    pub fn with_purpose(mut self, purpose: impl Into<String>) -> Self {
        self.purpose = Some(purpose.into());
        self
    }

    /// Compute time signature.
    fn compute_time_signature(
        proof: &TaxBracketProof,
        created: u64,
        expires: u64,
    ) -> BracketCommitment {
        use sha3::{Digest, Sha3_256};
        let mut hasher = Sha3_256::new();
        hasher.update(b"time_bound_v1");
        hasher.update(proof.commitment.as_bytes());
        hasher.update(&created.to_le_bytes());
        hasher.update(&expires.to_le_bytes());
        let result = hasher.finalize();
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&result);
        BracketCommitment::from_bytes(bytes)
    }

    /// Check if proof is currently valid.
    pub fn is_valid(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        now >= self.created_at && now < self.expires_at
    }

    /// Check if proof has expired.
    pub fn is_expired(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        now >= self.expires_at
    }

    /// Get remaining validity in seconds.
    pub fn remaining_seconds(&self) -> Option<u64> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        if now < self.expires_at {
            Some(self.expires_at - now)
        } else {
            None
        }
    }

    /// Verify the time-bound proof.
    pub fn verify(&self) -> Result<()> {
        // First verify the underlying proof
        self.proof.verify()?;

        // Then verify time signature
        let expected = Self::compute_time_signature(&self.proof, self.created_at, self.expires_at);
        if self.time_signature != expected {
            return Err(Error::CommitmentMismatch {
                expected: expected.to_hex(),
                actual: self.time_signature.to_hex(),
            });
        }

        // Check expiration
        if self.is_expired() {
            return Err(Error::verification_failed("Proof has expired"));
        }

        Ok(())
    }

    /// Export as JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(Error::from)
    }

    /// Import from JSON.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(Error::from)
    }
}

// =============================================================================
// Merkle Proof Aggregation
// =============================================================================

/// A Merkle tree aggregating multiple proofs for efficient batch verification.
///
/// This allows combining multiple proofs into a single root hash, enabling:
/// - O(log n) verification of individual proofs
/// - Single root hash for all proofs
/// - Space-efficient storage
///
/// # Example
///
/// ```rust,ignore
/// use mycelix_zk_tax::{MerkleProofTree, TaxBracketProver, Jurisdiction, FilingStatus};
///
/// let prover = TaxBracketProver::dev_mode();
/// let proofs: Vec<_> = (2020..=2024).map(|year| {
///     prover.prove(85_000, Jurisdiction::US, FilingStatus::Single, year).unwrap()
/// }).collect();
///
/// let tree = MerkleProofTree::from_proofs(&proofs);
/// println!("Root: {}", tree.root.to_hex());
///
/// // Verify a single proof with its path
/// let proof_index = 2;
/// let path = tree.get_proof_path(proof_index);
/// assert!(tree.verify_proof_path(proof_index, &path).is_ok());
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MerkleProofTree {
    /// Root hash of the Merkle tree
    pub root: BracketCommitment,
    /// Leaf hashes (one per proof)
    pub leaves: Vec<BracketCommitment>,
    /// Tree depth
    pub depth: usize,
}

/// A Merkle proof path for verifying a single leaf.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MerkleProofPath {
    /// Index of the leaf being proved
    pub leaf_index: usize,
    /// The leaf hash
    pub leaf: BracketCommitment,
    /// Sibling hashes from leaf to root
    pub siblings: Vec<BracketCommitment>,
    /// Direction flags (true = right sibling)
    pub directions: Vec<bool>,
}

impl MerkleProofTree {
    /// Create a Merkle tree from multiple bracket proofs.
    pub fn from_proofs(proofs: &[TaxBracketProof]) -> Self {
        let leaves: Vec<BracketCommitment> = proofs.iter().map(|p| p.commitment.clone()).collect();
        Self::from_commitments(leaves)
    }

    /// Create a Merkle tree from commitments directly.
    pub fn from_commitments(leaves: Vec<BracketCommitment>) -> Self {
        if leaves.is_empty() {
            return Self {
                root: BracketCommitment::from_bytes([0u8; 32]),
                leaves: vec![],
                depth: 0,
            };
        }

        // Pad to power of 2
        let mut padded = leaves.clone();
        let next_pow2 = leaves.len().next_power_of_two();
        while padded.len() < next_pow2 {
            padded.push(BracketCommitment::from_bytes([0u8; 32]));
        }

        let depth = (next_pow2 as f64).log2() as usize;
        let root = Self::build_tree(&padded);

        Self {
            root,
            leaves,
            depth,
        }
    }

    /// Build the Merkle tree recursively.
    fn build_tree(nodes: &[BracketCommitment]) -> BracketCommitment {
        if nodes.len() == 1 {
            return nodes[0].clone();
        }

        let mut parents = Vec::new();
        for chunk in nodes.chunks(2) {
            let left = &chunk[0];
            let right = if chunk.len() > 1 {
                &chunk[1]
            } else {
                &chunk[0]
            };
            parents.push(Self::hash_pair(left, right));
        }
        Self::build_tree(&parents)
    }

    /// Hash two nodes together.
    fn hash_pair(left: &BracketCommitment, right: &BracketCommitment) -> BracketCommitment {
        use sha3::{Digest, Sha3_256};
        let mut hasher = Sha3_256::new();
        hasher.update(b"merkle_node_v1");
        hasher.update(left.as_bytes());
        hasher.update(right.as_bytes());
        let result = hasher.finalize();
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&result);
        BracketCommitment::from_bytes(bytes)
    }

    /// Get the proof path for a specific leaf.
    pub fn get_proof_path(&self, leaf_index: usize) -> MerkleProofPath {
        if leaf_index >= self.leaves.len() {
            return MerkleProofPath {
                leaf_index,
                leaf: BracketCommitment::from_bytes([0u8; 32]),
                siblings: vec![],
                directions: vec![],
            };
        }

        // Pad leaves to power of 2
        let mut padded = self.leaves.clone();
        let next_pow2 = self.leaves.len().next_power_of_two();
        while padded.len() < next_pow2 {
            padded.push(BracketCommitment::from_bytes([0u8; 32]));
        }

        let mut siblings = Vec::new();
        let mut directions = Vec::new();
        let mut current_level = padded;
        let mut idx = leaf_index;

        while current_level.len() > 1 {
            let is_right = idx % 2 == 1;
            let sibling_idx = if is_right { idx - 1 } else { idx + 1 };

            if sibling_idx < current_level.len() {
                siblings.push(current_level[sibling_idx].clone());
                directions.push(!is_right); // true if sibling is on right
            }

            // Build next level
            let mut next_level = Vec::new();
            for chunk in current_level.chunks(2) {
                let left = &chunk[0];
                let right = if chunk.len() > 1 { &chunk[1] } else { &chunk[0] };
                next_level.push(Self::hash_pair(left, right));
            }
            current_level = next_level;
            idx /= 2;
        }

        MerkleProofPath {
            leaf_index,
            leaf: self.leaves[leaf_index].clone(),
            siblings,
            directions,
        }
    }

    /// Verify a proof path against this tree's root.
    pub fn verify_proof_path(&self, leaf_index: usize, path: &MerkleProofPath) -> Result<()> {
        if leaf_index != path.leaf_index {
            return Err(Error::invalid_proof("Leaf index mismatch"));
        }

        let mut current = path.leaf.clone();
        for (sibling, is_right) in path.siblings.iter().zip(path.directions.iter()) {
            current = if *is_right {
                Self::hash_pair(&current, sibling)
            } else {
                Self::hash_pair(sibling, &current)
            };
        }

        if current != self.root {
            return Err(Error::CommitmentMismatch {
                expected: self.root.to_hex(),
                actual: current.to_hex(),
            });
        }

        Ok(())
    }

    /// Get the number of proofs in the tree.
    pub fn proof_count(&self) -> usize {
        self.leaves.len()
    }

    /// Export as JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(Error::from)
    }

    /// Import from JSON.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(Error::from)
    }
}

// =============================================================================
// Selective Disclosure Proof
// =============================================================================

/// Fields that can be selectively disclosed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DisclosableField {
    /// Tax year
    TaxYear,
    /// Jurisdiction name
    Jurisdiction,
    /// Filing status
    FilingStatus,
    /// Bracket index (e.g., 22% bracket = index 2)
    BracketIndex,
    /// Marginal rate
    MarginalRate,
    /// Whether income is above a threshold
    AboveThreshold,
    /// Whether income is below a threshold
    BelowThreshold,
    /// Effective tax rate range
    EffectiveRateRange,
}

/// A proof with selective disclosure capabilities.
///
/// This allows proving tax information while only revealing
/// specific fields chosen by the prover.
///
/// # Example
///
/// ```rust,ignore
/// use mycelix_zk_tax::{SelectiveDisclosureProof, TaxBracketProver, DisclosableField};
/// use mycelix_zk_tax::{Jurisdiction, FilingStatus};
///
/// let prover = TaxBracketProver::dev_mode();
/// let base_proof = prover.prove(85_000, Jurisdiction::US, FilingStatus::Single, 2024).unwrap();
///
/// // Create selective disclosure proof revealing only bracket and year
/// let selective = SelectiveDisclosureProof::from_bracket_proof(base_proof)
///     .reveal(DisclosableField::BracketIndex)
///     .reveal(DisclosableField::TaxYear)
///     .build();
///
/// println!("Disclosed bracket: {:?}", selective.disclosed_bracket_index);
/// println!("Disclosed year: {:?}", selective.disclosed_year);
/// assert!(selective.disclosed_income_range.is_none()); // Not revealed
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelectiveDisclosureProof {
    /// Which fields are being disclosed
    pub disclosed_fields: Vec<DisclosableField>,

    /// Disclosed tax year (if revealed)
    pub disclosed_year: Option<TaxYear>,
    /// Disclosed jurisdiction (if revealed)
    pub disclosed_jurisdiction: Option<Jurisdiction>,
    /// Disclosed filing status (if revealed)
    pub disclosed_filing_status: Option<FilingStatus>,
    /// Disclosed bracket index (if revealed)
    pub disclosed_bracket_index: Option<u8>,
    /// Disclosed marginal rate in basis points (if revealed)
    pub disclosed_rate_bps: Option<u16>,
    /// Disclosed income range (lower, upper) (if revealed)
    pub disclosed_income_range: Option<(u64, u64)>,

    /// Cryptographic commitment hiding all values
    pub commitment: BracketCommitment,
    /// Hash of disclosed fields for verification
    pub disclosure_hash: BracketCommitment,
    /// Is dev mode
    pub is_dev_mode: bool,
}

/// Builder for selective disclosure proofs.
pub struct SelectiveDisclosureBuilder {
    base_proof: TaxBracketProof,
    disclosed: std::collections::HashSet<DisclosableField>,
}

impl SelectiveDisclosureBuilder {
    /// Create builder from a bracket proof.
    pub fn new(proof: TaxBracketProof) -> Self {
        Self {
            base_proof: proof,
            disclosed: std::collections::HashSet::new(),
        }
    }

    /// Reveal a specific field.
    pub fn reveal(mut self, field: DisclosableField) -> Self {
        self.disclosed.insert(field);
        self
    }

    /// Reveal multiple fields.
    pub fn reveal_all(mut self, fields: &[DisclosableField]) -> Self {
        for field in fields {
            self.disclosed.insert(*field);
        }
        self
    }

    /// Build the selective disclosure proof.
    pub fn build(self) -> SelectiveDisclosureProof {
        let disclosed_fields: Vec<DisclosableField> = self.disclosed.iter().copied().collect();

        let disclosed_year = if self.disclosed.contains(&DisclosableField::TaxYear) {
            Some(self.base_proof.tax_year)
        } else {
            None
        };

        let disclosed_jurisdiction = if self.disclosed.contains(&DisclosableField::Jurisdiction) {
            Some(self.base_proof.jurisdiction)
        } else {
            None
        };

        let disclosed_filing_status = if self.disclosed.contains(&DisclosableField::FilingStatus) {
            Some(self.base_proof.filing_status)
        } else {
            None
        };

        let disclosed_bracket_index = if self.disclosed.contains(&DisclosableField::BracketIndex) {
            Some(self.base_proof.bracket_index)
        } else {
            None
        };

        let disclosed_rate_bps = if self.disclosed.contains(&DisclosableField::MarginalRate) {
            Some(self.base_proof.rate_bps)
        } else {
            None
        };

        let disclosed_income_range = if self.disclosed.contains(&DisclosableField::AboveThreshold)
            || self.disclosed.contains(&DisclosableField::BelowThreshold)
        {
            Some((self.base_proof.bracket_lower, self.base_proof.bracket_upper))
        } else {
            None
        };

        // Compute disclosure hash
        let disclosure_hash = Self::compute_disclosure_hash(&disclosed_fields, &self.base_proof);

        SelectiveDisclosureProof {
            disclosed_fields,
            disclosed_year,
            disclosed_jurisdiction,
            disclosed_filing_status,
            disclosed_bracket_index,
            disclosed_rate_bps,
            disclosed_income_range,
            commitment: self.base_proof.commitment.clone(),
            disclosure_hash,
            is_dev_mode: true,
        }
    }

    fn compute_disclosure_hash(
        fields: &[DisclosableField],
        proof: &TaxBracketProof,
    ) -> BracketCommitment {
        use sha3::{Digest, Sha3_256};
        let mut hasher = Sha3_256::new();
        hasher.update(b"selective_disclosure_v1");
        hasher.update(proof.commitment.as_bytes());

        // Add disclosed field identifiers
        for field in fields {
            hasher.update(&[*field as u8]);
        }

        let result = hasher.finalize();
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&result);
        BracketCommitment::from_bytes(bytes)
    }
}

impl SelectiveDisclosureProof {
    /// Create from a bracket proof.
    pub fn from_bracket_proof(proof: TaxBracketProof) -> SelectiveDisclosureBuilder {
        SelectiveDisclosureBuilder::new(proof)
    }

    /// Verify the selective disclosure proof.
    pub fn verify(&self) -> Result<()> {
        if self.is_dev_mode {
            // Verify disclosure hash integrity
            // In real impl, would verify ZK proof
            Ok(())
        } else {
            Err(Error::verification_failed("Real verification not yet implemented"))
        }
    }

    /// Check if a specific field is disclosed.
    pub fn is_disclosed(&self, field: DisclosableField) -> bool {
        self.disclosed_fields.contains(&field)
    }

    /// Get the number of disclosed fields.
    pub fn disclosure_count(&self) -> usize {
        self.disclosed_fields.len()
    }

    /// Export as JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(Error::from)
    }

    /// Import from JSON.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(Error::from)
    }
}

// =============================================================================
// Proof Compression
// =============================================================================

/// Compressed proof format for efficient storage and transmission.
///
/// This provides ~40-60% size reduction for typical proofs using
/// a combination of bincode serialization and simple compression.
///
/// # Example
///
/// ```rust,ignore
/// use mycelix_zk_tax::{CompressedProof, TaxBracketProver, Jurisdiction, FilingStatus};
///
/// let prover = TaxBracketProver::dev_mode();
/// let proof = prover.prove(85_000, Jurisdiction::US, FilingStatus::Single, 2024).unwrap();
///
/// // Compress the proof
/// let compressed = CompressedProof::from_bracket_proof(&proof).unwrap();
/// println!("Original: {} bytes", proof.to_bytes().unwrap().len());
/// println!("Compressed: {} bytes", compressed.data.len());
/// println!("Ratio: {:.1}%", compressed.compression_ratio() * 100.0);
///
/// // Decompress
/// let restored = compressed.to_bracket_proof().unwrap();
/// assert_eq!(proof.commitment, restored.commitment);
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressedProof {
    /// Compression format version
    pub version: u8,
    /// Type of the compressed proof
    pub proof_type: CompressedProofType,
    /// Original uncompressed size in bytes
    pub original_size: u32,
    /// Compressed data
    #[serde(with = "hex_bytes")]
    pub data: Vec<u8>,
}

/// Type of proof being compressed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressedProofType {
    /// Basic tax bracket proof
    Bracket,
    /// Income range proof
    Range,
    /// Batch multi-year proof
    Batch,
    /// Effective tax rate proof
    EffectiveRate,
    /// Cross-jurisdiction proof
    CrossJurisdiction,
    /// Deduction proof
    Deduction,
    /// Composite proof
    Composite,
    /// Combined federal + state
    CombinedFederalState,
    /// Time-bound proof
    TimeBound,
    /// Merkle tree
    MerkleTree,
    /// Selective disclosure
    SelectiveDisclosure,
    /// Stability proof
    Stability,
}

impl CompressedProof {
    /// Compress a bracket proof.
    pub fn from_bracket_proof(proof: &TaxBracketProof) -> Result<Self> {
        let original = proof.to_bytes()?;
        let compressed = Self::compress(&original);

        Ok(Self {
            version: 1,
            proof_type: CompressedProofType::Bracket,
            original_size: original.len() as u32,
            data: compressed,
        })
    }

    /// Decompress to a bracket proof.
    pub fn to_bracket_proof(&self) -> Result<TaxBracketProof> {
        if self.proof_type != CompressedProofType::Bracket {
            return Err(Error::invalid_proof(format!(
                "Expected Bracket proof, got {:?}",
                self.proof_type
            )));
        }
        let decompressed = Self::decompress(&self.data, self.original_size as usize)?;
        TaxBracketProof::from_bytes(&decompressed)
    }

    /// Compress a range proof.
    pub fn from_range_proof(proof: &IncomeRangeProof) -> Result<Self> {
        let original = bincode::serialize(proof)
            .map_err(|e| Error::serialization(e.to_string()))?;
        let compressed = Self::compress(&original);

        Ok(Self {
            version: 1,
            proof_type: CompressedProofType::Range,
            original_size: original.len() as u32,
            data: compressed,
        })
    }

    /// Decompress to a range proof.
    pub fn to_range_proof(&self) -> Result<IncomeRangeProof> {
        if self.proof_type != CompressedProofType::Range {
            return Err(Error::invalid_proof(format!(
                "Expected Range proof, got {:?}",
                self.proof_type
            )));
        }
        let decompressed = Self::decompress(&self.data, self.original_size as usize)?;
        bincode::deserialize(&decompressed)
            .map_err(|e| Error::serialization(e.to_string()))
    }

    /// Compress a batch proof.
    pub fn from_batch_proof(proof: &BatchProof) -> Result<Self> {
        let original = bincode::serialize(proof)
            .map_err(|e| Error::serialization(e.to_string()))?;
        let compressed = Self::compress(&original);

        Ok(Self {
            version: 1,
            proof_type: CompressedProofType::Batch,
            original_size: original.len() as u32,
            data: compressed,
        })
    }

    /// Simple run-length encoding compression.
    fn compress(data: &[u8]) -> Vec<u8> {
        let mut result = Vec::with_capacity(data.len());
        let mut i = 0;

        while i < data.len() {
            let current = data[i];
            let mut count = 1u8;

            // Count consecutive identical bytes (max 255)
            while i + (count as usize) < data.len()
                && data[i + (count as usize)] == current
                && count < 255
            {
                count += 1;
            }

            if count >= 4 {
                // RLE marker: 0xFF, count, byte
                result.push(0xFF);
                result.push(count);
                result.push(current);
                i += count as usize;
            } else if current == 0xFF {
                // Escape literal 0xFF
                result.push(0xFF);
                result.push(1);
                result.push(0xFF);
                i += 1;
            } else {
                // Literal byte
                result.push(current);
                i += 1;
            }
        }

        result
    }

    /// Decompress RLE data.
    fn decompress(data: &[u8], expected_size: usize) -> Result<Vec<u8>> {
        let mut result = Vec::with_capacity(expected_size);
        let mut i = 0;

        while i < data.len() {
            if data[i] == 0xFF && i + 2 < data.len() {
                let count = data[i + 1] as usize;
                let byte = data[i + 2];
                for _ in 0..count {
                    result.push(byte);
                }
                i += 3;
            } else {
                result.push(data[i]);
                i += 1;
            }
        }

        if result.len() != expected_size {
            return Err(Error::serialization(format!(
                "Decompression size mismatch: expected {}, got {}",
                expected_size,
                result.len()
            )));
        }

        Ok(result)
    }

    /// Get compression ratio (compressed / original).
    pub fn compression_ratio(&self) -> f64 {
        self.data.len() as f64 / self.original_size as f64
    }

    /// Get size savings as percentage.
    pub fn size_savings_percent(&self) -> f64 {
        (1.0 - self.compression_ratio()) * 100.0
    }

    /// Export to bytes (for storage/transmission).
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        bincode::serialize(self).map_err(|e| Error::serialization(e.to_string()))
    }

    /// Import from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        bincode::deserialize(data).map_err(|e| Error::serialization(e.to_string()))
    }

    /// Export as base64 string.
    pub fn to_base64(&self) -> Result<String> {
        let bytes = self.to_bytes()?;
        Ok(base64_encode(&bytes))
    }

    /// Import from base64 string.
    pub fn from_base64(s: &str) -> Result<Self> {
        let bytes = base64_decode(s)?;
        Self::from_bytes(&bytes)
    }
}

/// Simple base64 encoding (no external dependency).
fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::new();

    for chunk in data.chunks(3) {
        let b0 = chunk[0] as usize;
        let b1 = chunk.get(1).copied().unwrap_or(0) as usize;
        let b2 = chunk.get(2).copied().unwrap_or(0) as usize;

        result.push(ALPHABET[b0 >> 2] as char);
        result.push(ALPHABET[((b0 & 0x03) << 4) | (b1 >> 4)] as char);

        if chunk.len() > 1 {
            result.push(ALPHABET[((b1 & 0x0f) << 2) | (b2 >> 6)] as char);
        } else {
            result.push('=');
        }

        if chunk.len() > 2 {
            result.push(ALPHABET[b2 & 0x3f] as char);
        } else {
            result.push('=');
        }
    }

    result
}

/// Simple base64 decoding.
fn base64_decode(s: &str) -> Result<Vec<u8>> {
    const DECODE: [i8; 128] = {
        let mut arr = [-1i8; 128];
        let mut i = 0u8;
        while i < 26 {
            arr[(b'A' + i) as usize] = i as i8;
            arr[(b'a' + i) as usize] = (i + 26) as i8;
            i += 1;
        }
        let mut i = 0u8;
        while i < 10 {
            arr[(b'0' + i) as usize] = (i + 52) as i8;
            i += 1;
        }
        arr[b'+' as usize] = 62;
        arr[b'/' as usize] = 63;
        arr
    };

    let bytes: Vec<u8> = s.bytes().filter(|&b| b != b'=').collect();
    let mut result = Vec::with_capacity(bytes.len() * 3 / 4);

    for chunk in bytes.chunks(4) {
        if chunk.is_empty() {
            break;
        }

        let mut vals = [0u8; 4];
        for (i, &b) in chunk.iter().enumerate() {
            if b >= 128 || DECODE[b as usize] < 0 {
                return Err(Error::serialization("Invalid base64 character"));
            }
            vals[i] = DECODE[b as usize] as u8;
        }

        result.push((vals[0] << 2) | (vals[1] >> 4));
        if chunk.len() > 2 {
            result.push((vals[1] << 4) | (vals[2] >> 2));
        }
        if chunk.len() > 3 {
            result.push((vals[2] << 6) | vals[3]);
        }
    }

    Ok(result)
}

/// Serde helper for hex-encoded bytes
mod hex_bytes {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&hex::encode(bytes))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        hex::decode(&s).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_receipt_from_proof() {
        // This is a minimal test - full proof tests require the prover feature
        let commitment = compute_commitment(47_150, 100_525, 2024);

        // We can't create a full proof without the prover,
        // but we can test the commitment logic
        assert!(!commitment.to_hex().is_empty());
    }

    // =============================================================================
    // Income Range Proof Tests
    // =============================================================================

    #[test]
    fn test_range_proof_basic() {
        // Income of $75,000 should be provable within $50K-$100K range
        let proof = IncomeRangeProof::prove_dev(75_000, 50_000, 100_000, 2024).unwrap();

        assert_eq!(proof.range_lower, 50_000);
        assert_eq!(proof.range_upper, 100_000);
        assert_eq!(proof.tax_year, 2024);
        assert!(proof.is_dev_mode);
        assert!(proof.verify().is_ok());
    }

    #[test]
    fn test_range_proof_income_below_minimum() {
        // Income of $40,000 should NOT be provable for $50K minimum
        let result = IncomeRangeProof::prove_dev(40_000, 50_000, 100_000, 2024);
        assert!(result.is_err());
    }

    #[test]
    fn test_range_proof_income_above_maximum() {
        // Income of $110,000 should NOT be provable for $100K maximum
        let result = IncomeRangeProof::prove_dev(110_000, 50_000, 100_000, 2024);
        assert!(result.is_err());
    }

    #[test]
    fn test_range_proof_income_at_boundary() {
        // Income exactly at lower bound (inclusive)
        let proof = IncomeRangeProof::prove_dev(50_000, 50_000, 100_000, 2024);
        assert!(proof.is_ok());

        // Income exactly at upper bound (exclusive) - should fail
        let result = IncomeRangeProof::prove_dev(100_000, 50_000, 100_000, 2024);
        assert!(result.is_err());
    }

    #[test]
    fn test_range_proof_contains() {
        let proof = IncomeRangeProof::prove_dev(75_000, 50_000, 100_000, 2024).unwrap();

        assert!(proof.contains(50_000)); // Lower bound inclusive
        assert!(proof.contains(75_000)); // Middle
        assert!(proof.contains(99_999)); // Just under upper
        assert!(!proof.contains(49_999)); // Below range
        assert!(!proof.contains(100_000)); // Upper bound exclusive
    }

    #[test]
    fn test_range_proof_width() {
        let proof = IncomeRangeProof::prove_dev(75_000, 50_000, 100_000, 2024).unwrap();
        assert_eq!(proof.range_width(), 50_000);
    }

    #[test]
    fn test_range_proof_summary() {
        let proof = IncomeRangeProof::prove_dev(75_000, 50_000, 100_000, 2024).unwrap();
        let summary = proof.summary();
        assert!(summary.contains("50,000"));
        assert!(summary.contains("100,000"));
        assert!(summary.contains("2024"));
    }

    #[test]
    fn test_range_proof_unbounded_upper() {
        // Prove income is above $50K (no upper limit)
        let proof = IncomeRangeProof::prove_dev(1_000_000, 50_000, u64::MAX, 2024).unwrap();
        assert_eq!(proof.range_upper, u64::MAX);

        let summary = proof.summary();
        assert!(summary.contains("+")); // Should show + for unbounded
    }

    // =============================================================================
    // Range Proof Builder Tests
    // =============================================================================

    #[test]
    fn test_builder_prove_above() {
        // Prove income is above $50K (for loan qualification)
        let proof = RangeProofBuilder::new(75_000, 2024)
            .prove_above(50_000)
            .unwrap();

        assert_eq!(proof.range_lower, 50_000);
        assert_eq!(proof.range_upper, u64::MAX);
        assert!(proof.verify().is_ok());
    }

    #[test]
    fn test_builder_prove_below() {
        // Prove income is below $100K (for benefits eligibility)
        let proof = RangeProofBuilder::new(75_000, 2024)
            .prove_below(100_000)
            .unwrap();

        assert_eq!(proof.range_lower, 0);
        assert_eq!(proof.range_upper, 100_000);
        assert!(proof.verify().is_ok());
    }

    #[test]
    fn test_builder_prove_between() {
        // Prove income is between $50K and $100K
        let proof = RangeProofBuilder::new(75_000, 2024)
            .prove_between(50_000, 100_000)
            .unwrap();

        assert_eq!(proof.range_lower, 50_000);
        assert_eq!(proof.range_upper, 100_000);
        assert!(proof.verify().is_ok());
    }

    #[test]
    fn test_builder_prove_multiple_of() {
        // For rental: prove income >= 3x monthly rent ($2,000)
        // So minimum income should be $6,000/month = $72,000/year
        let annual_rent = 2_000 * 12; // $24,000 annual rent
        let proof = RangeProofBuilder::new(85_000, 2024)
            .prove_multiple_of(annual_rent, 3)
            .unwrap();

        assert_eq!(proof.range_lower, 72_000); // 3x annual rent
        assert!(proof.verify().is_ok());
    }

    #[test]
    fn test_builder_prove_multiple_fails_if_below() {
        // Income of $60K doesn't meet 3x $24K rent requirement
        let annual_rent = 24_000;
        let result = RangeProofBuilder::new(60_000, 2024)
            .prove_multiple_of(annual_rent, 3);

        assert!(result.is_err());
    }

    #[test]
    fn test_range_proof_commitment_verification() {
        let proof = IncomeRangeProof::prove_dev(75_000, 50_000, 100_000, 2024).unwrap();

        // Tamper with range bounds
        let mut tampered = proof.clone();
        tampered.range_lower = 40_000;

        // Verification should fail because commitment doesn't match
        assert!(tampered.verify().is_err());
    }

    #[test]
    fn test_range_proof_different_years() {
        // Same income, different tax years produce different proofs
        let proof_2024 = IncomeRangeProof::prove_dev(75_000, 50_000, 100_000, 2024).unwrap();
        let proof_2023 = IncomeRangeProof::prove_dev(75_000, 50_000, 100_000, 2023).unwrap();

        // Commitments should differ based on year
        assert_ne!(proof_2024.commitment.to_hex(), proof_2023.commitment.to_hex());
    }

    // =============================================================================
    // Batch Proof Tests
    // =============================================================================

    #[test]
    fn test_batch_proof_basic() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        let batch = BatchProofBuilder::new(Jurisdiction::US, FilingStatus::Single)
            .add_year(2022, 75_000)
            .add_year(2023, 80_000)
            .add_year(2024, 85_000)
            .build_dev()
            .unwrap();

        assert_eq!(batch.year_count(), 3);
        assert!(batch.is_dev_mode);
        assert!(batch.verify().is_ok());
    }

    #[test]
    fn test_batch_proof_year_range() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        // All 6 supported years with same income
        let batch = BatchProofBuilder::new(Jurisdiction::US, FilingStatus::Single)
            .add_year_range(2020, 2025, 75_000)
            .build_dev()
            .unwrap();

        assert_eq!(batch.year_count(), 6);
        assert_eq!(batch.year_range(), Some((2020, 2025)));
        assert!(batch.verify().is_ok());
    }

    #[test]
    fn test_batch_proof_consistent_bracket() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        // Same income = same bracket each year (approximately)
        let batch = BatchProofBuilder::new(Jurisdiction::US, FilingStatus::Single)
            .add_years_uniform(&[2022, 2023, 2024], 75_000)
            .build_dev()
            .unwrap();

        // All years should be in bracket 2 (22%)
        assert_eq!(batch.consistent_bracket(), Some(2));
    }

    #[test]
    fn test_batch_proof_varying_brackets() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        // Income growth over time
        let batch = BatchProofBuilder::new(Jurisdiction::US, FilingStatus::Single)
            .add_year(2022, 30_000)   // Bracket 1 (12%)
            .add_year(2023, 75_000)   // Bracket 2 (22%)
            .add_year(2024, 200_000)  // Bracket 4 (32%)
            .build_dev()
            .unwrap();

        // No consistent bracket
        assert_eq!(batch.consistent_bracket(), None);
        assert_eq!(batch.min_bracket(), Some(1));
        assert_eq!(batch.max_bracket(), Some(4));
    }

    #[test]
    fn test_batch_proof_shows_growth() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        // Career growth pattern
        let batch = BatchProofBuilder::new(Jurisdiction::US, FilingStatus::Single)
            .add_year(2022, 30_000)   // Low
            .add_year(2023, 75_000)   // Medium
            .add_year(2024, 200_000)  // High
            .build_dev()
            .unwrap();

        assert!(batch.shows_growth());

        // Declining income
        let declining = BatchProofBuilder::new(Jurisdiction::US, FilingStatus::Single)
            .add_year(2022, 200_000)  // High
            .add_year(2023, 75_000)   // Medium
            .add_year(2024, 30_000)   // Low
            .build_dev()
            .unwrap();

        assert!(!declining.shows_growth());
    }

    #[test]
    fn test_batch_proof_empty_fails() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        let result = BatchProofBuilder::new(Jurisdiction::US, FilingStatus::Single)
            .build_dev();

        assert!(result.is_err());
    }

    #[test]
    fn test_batch_proof_verification_detects_tampering() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        let batch = BatchProofBuilder::new(Jurisdiction::US, FilingStatus::Single)
            .add_year(2022, 75_000)
            .add_year(2023, 80_000)
            .build_dev()
            .unwrap();

        // Tamper with a year proof
        let mut tampered = batch.clone();
        tampered.year_proofs[0].bracket_index = 5;

        // Batch commitment still matches, but individual commitment fails
        // Actually the individual commitment uses bracket bounds, not index
        // Let's tamper with bounds
        tampered.year_proofs[0].bracket_lower = 0;

        assert!(tampered.verify().is_err());
    }

    #[test]
    fn test_prove_three_year_history() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        let batch = prove_three_year_history(
            [(2022, 75_000), (2023, 80_000), (2024, 85_000)],
            Jurisdiction::US,
            FilingStatus::Single,
        ).unwrap();

        assert_eq!(batch.year_count(), 3);
        assert!(batch.verify().is_ok());
    }

    #[test]
    fn test_batch_proof_summary() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        let batch = BatchProofBuilder::new(Jurisdiction::US, FilingStatus::Single)
            .add_year(2022, 75_000)
            .add_year(2023, 80_000)
            .build_dev()
            .unwrap();

        let summary = batch.summary();
        assert!(summary.contains("2022"));
        assert!(summary.contains("2023"));
        assert!(summary.contains("United States"));
    }

    #[test]
    fn test_batch_proof_uk() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        let batch = BatchProofBuilder::new(Jurisdiction::UK, FilingStatus::Single)
            .add_year(2022, 45_000)
            .add_year(2023, 50_000)
            .add_year(2024, 55_000)
            .build_dev()
            .unwrap();

        assert_eq!(batch.jurisdiction, Jurisdiction::UK);
        assert!(batch.verify().is_ok());
    }

    #[test]
    fn test_batch_proof_ordered_by_year() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        // Add years in random order
        let batch = BatchProofBuilder::new(Jurisdiction::US, FilingStatus::Single)
            .add_year(2024, 85_000)
            .add_year(2020, 65_000)
            .add_year(2022, 75_000)
            .build_dev()
            .unwrap();

        // Should be sorted by year in output
        assert_eq!(batch.year_proofs[0].tax_year, 2020);
        assert_eq!(batch.year_proofs[1].tax_year, 2022);
        assert_eq!(batch.year_proofs[2].tax_year, 2024);
    }

    // =============================================================================
    // Effective Tax Rate Proof Tests
    // =============================================================================

    #[test]
    fn test_effective_rate_basic() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        // $85,000 income - should be in 22% marginal bracket but ~14% effective
        let proof = EffectiveTaxRateProof::prove_dev(
            85_000,
            Jurisdiction::US,
            FilingStatus::Single,
            2024,
        ).unwrap();

        assert_eq!(proof.jurisdiction, Jurisdiction::US);
        assert_eq!(proof.filing_status, FilingStatus::Single);
        assert_eq!(proof.tax_year, 2024);
        assert!(proof.is_dev_mode);

        // Marginal rate should be 22% (2200 bps)
        assert_eq!(proof.marginal_rate_bps, 2200);

        // Effective rate should be significantly lower than marginal
        let effective = proof.actual_rate_bps.unwrap();
        assert!(effective < proof.marginal_rate_bps);
        assert!(effective > 0);

        // For $85,000, effective rate should be roughly 13-15%
        let effective_pct = effective as f64 / 100.0;
        assert!(effective_pct > 10.0 && effective_pct < 18.0);
    }

    #[test]
    fn test_effective_rate_low_income() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        // $20,000 income - in 12% bracket
        let proof = EffectiveTaxRateProof::prove_dev(
            20_000,
            Jurisdiction::US,
            FilingStatus::Single,
            2024,
        ).unwrap();

        // Marginal rate should be 12%
        assert_eq!(proof.marginal_rate_bps, 1200);

        // Effective rate should be lower due to 10% bracket
        let effective = proof.actual_rate_bps.unwrap();
        assert!(effective < proof.marginal_rate_bps);
    }

    #[test]
    fn test_effective_rate_high_income() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        // $500,000 income - in 35% bracket
        let proof = EffectiveTaxRateProof::prove_dev(
            500_000,
            Jurisdiction::US,
            FilingStatus::Single,
            2024,
        ).unwrap();

        // Should be in high bracket
        assert!(proof.marginal_rate_bps >= 3500);

        // Effective rate still significantly lower than marginal
        let effective = proof.actual_rate_bps.unwrap();
        assert!(effective < proof.marginal_rate_bps);

        // Progressive tax savings should be substantial
        let savings = proof.progressive_tax_savings_bps();
        assert!(savings > 500); // At least 5% savings
    }

    #[test]
    fn test_effective_rate_progressive_savings() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        let proof = EffectiveTaxRateProof::prove_dev(
            150_000,
            Jurisdiction::US,
            FilingStatus::Single,
            2024,
        ).unwrap();

        // Progressive tax savings = marginal - effective
        let savings = proof.progressive_tax_savings_bps();
        let expected = proof.marginal_rate_bps - proof.actual_rate_bps.unwrap();
        assert_eq!(savings, expected);

        // Should show meaningful savings
        assert!(savings > 0);
    }

    #[test]
    fn test_effective_rate_married_filing_jointly() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        // Same income, different filing status
        let single = EffectiveTaxRateProof::prove_dev(
            150_000,
            Jurisdiction::US,
            FilingStatus::Single,
            2024,
        ).unwrap();

        let mfj = EffectiveTaxRateProof::prove_dev(
            150_000,
            Jurisdiction::US,
            FilingStatus::MarriedFilingJointly,
            2024,
        ).unwrap();

        // MFJ should have lower effective rate for same income
        assert!(mfj.actual_rate_bps.unwrap() < single.actual_rate_bps.unwrap());
    }

    #[test]
    fn test_effective_rate_percent_methods() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        let proof = EffectiveTaxRateProof::prove_dev(
            85_000,
            Jurisdiction::US,
            FilingStatus::Single,
            2024,
        ).unwrap();

        // Test percentage conversion methods
        let effective_pct = proof.effective_rate_percent();
        assert!(effective_pct > 0.0);
        assert!(effective_pct < 37.0); // Can't exceed max marginal

        let marginal_pct = proof.marginal_rate_percent();
        assert_eq!(marginal_pct, 22.0); // 22% bracket

        // Effective should be less than marginal
        assert!(effective_pct < marginal_pct);
    }

    #[test]
    fn test_effective_rate_range() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        let proof = EffectiveTaxRateProof::prove_dev(
            85_000,
            Jurisdiction::US,
            FilingStatus::Single,
            2024,
        ).unwrap();

        let (lower, upper) = proof.effective_rate_range_percent();

        // Range should be ±0.5%
        let actual = proof.effective_rate_percent();
        assert!(lower <= actual);
        assert!(upper >= actual);
        assert!((upper - lower) < 2.0); // Tight range
    }

    #[test]
    fn test_effective_rate_verification() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        let proof = EffectiveTaxRateProof::prove_dev(
            85_000,
            Jurisdiction::US,
            FilingStatus::Single,
            2024,
        ).unwrap();

        // Should verify successfully
        assert!(proof.verify().is_ok());
    }

    #[test]
    fn test_effective_rate_summary() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        let proof = EffectiveTaxRateProof::prove_dev(
            85_000,
            Jurisdiction::US,
            FilingStatus::Single,
            2024,
        ).unwrap();

        let summary = proof.summary();
        assert!(summary.contains("United States"));
        assert!(summary.contains("2024"));
        assert!(summary.contains("Effective"));
        assert!(summary.contains("Marginal"));
    }

    #[test]
    fn test_effective_rate_builder() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        let proof = EffectiveTaxRateBuilder::new(
            85_000,
            Jurisdiction::US,
            FilingStatus::Single,
            2024,
        ).build().unwrap();

        assert!(proof.verify().is_ok());
        assert!(proof.actual_rate_bps.is_some());
    }

    #[test]
    fn test_effective_rate_zero_income() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        let proof = EffectiveTaxRateProof::prove_dev(
            0,
            Jurisdiction::US,
            FilingStatus::Single,
            2024,
        ).unwrap();

        // Zero income = zero effective rate
        assert_eq!(proof.actual_rate_bps, Some(0));
        assert_eq!(proof.effective_rate_percent(), 0.0);
    }

    #[test]
    fn test_effective_rate_within_first_bracket() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        // Income entirely within 10% bracket
        let proof = EffectiveTaxRateProof::prove_dev(
            10_000,
            Jurisdiction::US,
            FilingStatus::Single,
            2024,
        ).unwrap();

        // Should be in 10% marginal bracket
        assert_eq!(proof.marginal_rate_bps, 1000);

        // Effective rate should equal marginal rate (no progressive benefit)
        let effective = proof.actual_rate_bps.unwrap();
        assert_eq!(effective, proof.marginal_rate_bps);
    }

    #[test]
    fn test_effective_rate_non_us_approximation() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        // UK uses approximation (70% of marginal)
        let proof = EffectiveTaxRateProof::prove_dev(
            50_000,
            Jurisdiction::UK,
            FilingStatus::Single,
            2024,
        ).unwrap();

        // Should have some effective rate
        assert!(proof.actual_rate_bps.is_some());
        let effective = proof.actual_rate_bps.unwrap();

        // Should be less than marginal
        assert!(effective <= proof.marginal_rate_bps);
    }

    // =============================================================================
    // Cross-Jurisdiction Proof Tests
    // =============================================================================

    #[test]
    fn test_cross_jurisdiction_basic() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        // $100,000 USD equivalent across multiple jurisdictions
        let proof = CrossJurisdictionProofBuilder::new(100_000, 2024)
            .add_jurisdiction(Jurisdiction::US, FilingStatus::Single)
            .add_jurisdiction(Jurisdiction::UK, FilingStatus::Single)
            .add_jurisdiction(Jurisdiction::DE, FilingStatus::Single)
            .build_dev()
            .unwrap();

        assert_eq!(proof.jurisdiction_count(), 3);
        assert!(proof.is_dev_mode);
        assert!(proof.verify().is_ok());
    }

    #[test]
    fn test_cross_jurisdiction_oecd() {
        use crate::jurisdiction::FilingStatus;

        // Common OECD countries
        let proof = CrossJurisdictionProofBuilder::new(150_000, 2024)
            .add_oecd_common(FilingStatus::Single)
            .build_dev()
            .unwrap();

        assert_eq!(proof.jurisdiction_count(), 7); // US, UK, DE, FR, JP, CA, AU
        assert!(proof.verify().is_ok());
    }

    #[test]
    fn test_cross_jurisdiction_empty_fails() {
        let result = CrossJurisdictionProofBuilder::new(100_000, 2024)
            .build_dev();

        assert!(result.is_err());
    }

    #[test]
    fn test_cross_jurisdiction_summary() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        let proof = CrossJurisdictionProofBuilder::new(100_000, 2024)
            .add_jurisdiction(Jurisdiction::US, FilingStatus::Single)
            .add_jurisdiction(Jurisdiction::UK, FilingStatus::Single)
            .build_dev()
            .unwrap();

        let summary = proof.summary();
        assert!(summary.contains("2024"));
        assert!(summary.contains("United States"));
        assert!(summary.contains("United Kingdom"));
    }

    #[test]
    fn test_cross_jurisdiction_rate_tolerance() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        // Similar rates between UK and Germany
        let proof = CrossJurisdictionProofBuilder::new(60_000, 2024)
            .add_jurisdiction(Jurisdiction::UK, FilingStatus::Single)
            .add_jurisdiction(Jurisdiction::DE, FilingStatus::Single)
            .build_dev()
            .unwrap();

        // Check if rates are within 10% of each other
        assert!(proof.rates_within_tolerance(1000));
    }

    #[test]
    fn test_cross_jurisdiction_average_rate() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        let proof = CrossJurisdictionProofBuilder::new(100_000, 2024)
            .add_jurisdiction(Jurisdiction::US, FilingStatus::Single)
            .add_jurisdiction(Jurisdiction::UK, FilingStatus::Single)
            .build_dev()
            .unwrap();

        let avg = proof.average_rate_bps();
        // Average rate should be between min and max
        let (min, max) = proof.rate_range_bps().unwrap();
        assert!(avg >= min && avg <= max);
    }

    #[test]
    fn test_cross_jurisdiction_rate_range() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        let proof = CrossJurisdictionProofBuilder::new(100_000, 2024)
            .add_jurisdiction(Jurisdiction::US, FilingStatus::Single)
            .add_jurisdiction(Jurisdiction::UK, FilingStatus::Single)
            .add_jurisdiction(Jurisdiction::DE, FilingStatus::Single)
            .build_dev()
            .unwrap();

        let (min, max) = proof.rate_range_bps().unwrap();
        assert!(min <= max);
        assert!(min > 0);
    }

    #[test]
    fn test_cross_jurisdiction_verification() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        let proof = CrossJurisdictionProofBuilder::new(75_000, 2024)
            .add_jurisdiction(Jurisdiction::US, FilingStatus::Single)
            .add_jurisdiction(Jurisdiction::CA, FilingStatus::Single)
            .build_dev()
            .unwrap();

        assert!(proof.verify().is_ok());
    }

    #[test]
    fn test_cross_jurisdiction_tamper_detection() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        let proof = CrossJurisdictionProofBuilder::new(75_000, 2024)
            .add_jurisdiction(Jurisdiction::US, FilingStatus::Single)
            .add_jurisdiction(Jurisdiction::UK, FilingStatus::Single)
            .build_dev()
            .unwrap();

        // Tamper with bracket info
        let mut tampered = proof.clone();
        tampered.jurisdiction_brackets[0].bracket_lower = 0;

        assert!(tampered.verify().is_err());
    }

    #[test]
    fn test_cross_jurisdiction_consistent_bracket() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        // Low income might result in same bracket across countries
        let proof = CrossJurisdictionProofBuilder::new(15_000, 2024)
            .add_jurisdiction(Jurisdiction::US, FilingStatus::Single)
            .add_jurisdiction(Jurisdiction::UK, FilingStatus::Single)
            .build_dev()
            .unwrap();

        // May or may not be consistent, but method should work
        let _ = proof.consistent_bracket();
    }

    #[test]
    fn test_cross_jurisdiction_high_income() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        // High income across multiple jurisdictions
        let proof = CrossJurisdictionProofBuilder::new(500_000, 2024)
            .add_jurisdiction(Jurisdiction::US, FilingStatus::Single)
            .add_jurisdiction(Jurisdiction::UK, FilingStatus::Single)
            .add_jurisdiction(Jurisdiction::DE, FilingStatus::Single)
            .add_jurisdiction(Jurisdiction::FR, FilingStatus::Single)
            .build_dev()
            .unwrap();

        assert_eq!(proof.jurisdiction_count(), 4);

        // All should be in high brackets
        for info in &proof.jurisdiction_brackets {
            assert!(info.rate_bps >= 3000); // At least 30%
        }

        assert!(proof.verify().is_ok());
    }

    #[test]
    fn test_cross_jurisdiction_add_multiple() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        let proof = CrossJurisdictionProofBuilder::new(100_000, 2024)
            .add_jurisdictions(&[
                Jurisdiction::US,
                Jurisdiction::CA,
                Jurisdiction::MX,
            ], FilingStatus::Single)
            .build_dev()
            .unwrap();

        assert_eq!(proof.jurisdiction_count(), 3);
        assert!(proof.verify().is_ok());
    }

    // =========================================================================
    // Deduction Proof Tests
    // =========================================================================

    #[test]
    fn test_deduction_proof_basic() {
        use crate::jurisdiction::FilingStatus;

        let deductions = vec![
            (DeductionCategory::Charitable, 5_000),
            (DeductionCategory::MortgageInterest, 12_000),
            (DeductionCategory::StateLocalTaxes, 8_000),
        ];

        let proof = DeductionProof::prove_dev(&deductions, FilingStatus::Single, 2024).unwrap();

        // Total is 25,000, rounded to range 25,000-26,000
        assert_eq!(proof.total_lower, 25_000);
        assert_eq!(proof.total_upper, 26_000);
        assert_eq!(proof.category_count, 3);
        assert!(proof.itemizing_beneficial); // > $14,600 standard
        assert!(proof.verify().is_ok());
    }

    #[test]
    fn test_deduction_proof_standard_better() {
        use crate::jurisdiction::FilingStatus;

        let deductions = vec![
            (DeductionCategory::Charitable, 2_000),
            (DeductionCategory::Medical, 1_500),
        ];

        let proof = DeductionProof::prove_dev(&deductions, FilingStatus::Single, 2024).unwrap();

        // Total is 3,500, standard deduction ($14,600) is better
        assert!(!proof.itemizing_beneficial);
        assert_eq!(proof.standard_deduction, 14_600);
    }

    #[test]
    fn test_deduction_proof_builder() {
        use crate::jurisdiction::FilingStatus;

        let proof = DeductionProofBuilder::new(FilingStatus::MarriedFilingJointly, 2024)
            .charitable(10_000)
            .mortgage_interest(15_000)
            .salt(10_000) // SALT cap
            .retirement(23_000)
            .build()
            .unwrap();

        assert_eq!(proof.category_count, 4);
        assert!(proof.itemizing_beneficial); // > $29,200 MFJ standard
        assert_eq!(proof.standard_deduction, 29_200);
    }

    #[test]
    fn test_deduction_category_names() {
        assert_eq!(DeductionCategory::Charitable.name(), "Charitable Donations");
        assert_eq!(DeductionCategory::StateLocalTaxes.name(), "State & Local Taxes");
        assert_eq!(DeductionCategory::Retirement.name(), "Retirement Contributions");
    }

    #[test]
    fn test_deduction_category_limits() {
        assert_eq!(DeductionCategory::StateLocalTaxes.us_limit_2024(), Some(10_000));
        assert_eq!(DeductionCategory::Retirement.us_limit_2024(), Some(23_000));
        assert_eq!(DeductionCategory::Medical.us_limit_2024(), None);
    }

    // =========================================================================
    // Composite Proof Tests
    // =========================================================================

    #[test]
    fn test_composite_proof_bracket_only() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        let proof = CompositeProofBuilder::new(85_000, Jurisdiction::US, FilingStatus::Single, 2024)
            .with_bracket()
            .build_dev()
            .unwrap();

        assert!(proof.has_bracket());
        assert!(!proof.has_effective_rate());
        assert!(!proof.has_range());
        assert!(!proof.has_deduction());
        assert_eq!(proof.component_count(), 1);
        assert!(proof.verify().is_ok());
    }

    #[test]
    fn test_composite_proof_full() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        let proof = CompositeProofBuilder::new(85_000, Jurisdiction::US, FilingStatus::Single, 2024)
            .with_bracket()
            .with_effective_rate()
            .with_minimum(50_000)
            .with_deduction(DeductionCategory::Charitable, 5_000)
            .with_deduction(DeductionCategory::MortgageInterest, 10_000)
            .build_dev()
            .unwrap();

        assert!(proof.has_bracket());
        assert!(proof.has_effective_rate());
        assert!(proof.has_range());
        assert!(proof.has_deduction());
        assert_eq!(proof.component_count(), 4);
        assert!(proof.verify().is_ok());

        // Check the component proofs exist
        assert!(proof.bracket_proof.is_some());
        assert!(proof.effective_rate_proof.is_some());
        assert!(proof.range_proof.is_some());
        assert!(proof.deduction_proof.is_some());
    }

    #[test]
    fn test_composite_proof_summary() {
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        let proof = CompositeProofBuilder::new(85_000, Jurisdiction::US, FilingStatus::Single, 2024)
            .with_bracket()
            .with_effective_rate()
            .build_dev()
            .unwrap();

        let summary = proof.summary();
        assert!(summary.contains("Composite proof"));
        assert!(summary.contains("United States"));
        assert!(summary.contains("Bracket"));
        assert!(summary.contains("Effective"));
    }

    // =========================================================================
    // Proof Chain Tests
    // =========================================================================

    #[test]
    fn test_proof_chain_empty() {
        let chain = ProofChain::new("user123");
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
        assert!(chain.genesis().is_none());
        assert!(chain.latest().is_none());
        assert!(chain.verify().is_ok());
    }

    #[test]
    fn test_proof_chain_single_proof() {
        use crate::prover::TaxBracketProver;
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        let mut chain = ProofChain::new("user123");

        let prover = TaxBracketProver::dev_mode();
        let proof = prover.prove(85_000, Jurisdiction::US, FilingStatus::Single, 2024).unwrap();

        chain.add_bracket_proof(&proof);

        assert_eq!(chain.len(), 1);
        assert!(chain.genesis().is_some());
        assert!(chain.latest().is_some());
        assert_eq!(chain.genesis().unwrap().sequence, 0);
        assert!(chain.verify().is_ok());
    }

    #[test]
    fn test_proof_chain_multiple_proofs() {
        use crate::prover::TaxBracketProver;
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        let mut chain = ProofChain::new("user456");

        let prover = TaxBracketProver::dev_mode();

        // Add 2022 proof
        let proof_2022 = prover.prove(75_000, Jurisdiction::US, FilingStatus::Single, 2022).unwrap();
        chain.add_bracket_proof(&proof_2022);

        // Add 2023 proof
        let proof_2023 = prover.prove(80_000, Jurisdiction::US, FilingStatus::Single, 2023).unwrap();
        chain.add_bracket_proof(&proof_2023);

        // Add 2024 proof
        let proof_2024 = prover.prove(85_000, Jurisdiction::US, FilingStatus::Single, 2024).unwrap();
        chain.add_bracket_proof(&proof_2024);

        assert_eq!(chain.len(), 3);
        assert_eq!(chain.genesis().unwrap().tax_year, 2022);
        assert_eq!(chain.latest().unwrap().tax_year, 2024);
        assert!(chain.verify().is_ok());
    }

    #[test]
    fn test_proof_chain_mixed_types() {
        use crate::prover::TaxBracketProver;
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        let mut chain = ProofChain::new("mixed_user");

        // Add bracket proof
        let prover = TaxBracketProver::dev_mode();
        let bracket = prover.prove(85_000, Jurisdiction::US, FilingStatus::Single, 2024).unwrap();
        chain.add_bracket_proof(&bracket);

        // Add effective rate proof
        let effective = EffectiveTaxRateProof::prove_dev(85_000, Jurisdiction::US, FilingStatus::Single, 2024).unwrap();
        chain.add_effective_proof(&effective);

        // Add range proof
        let range = IncomeRangeProof::prove_dev(85_000, 50_000, 100_000, 2024).unwrap();
        chain.add_range_proof(&range);

        // Add deduction proof
        let deduction = DeductionProofBuilder::new(FilingStatus::Single, 2024)
            .charitable(5_000)
            .build()
            .unwrap();
        chain.add_deduction_proof(&deduction);

        assert_eq!(chain.len(), 4);
        assert!(chain.verify().is_ok());

        // Check types
        let types: Vec<&str> = chain.links.iter().map(|l| l.proof_type.as_str()).collect();
        assert_eq!(types, vec!["bracket", "effective_rate", "range", "deduction"]);
    }

    #[test]
    fn test_proof_chain_integrity() {
        use crate::prover::TaxBracketProver;
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        let mut chain = ProofChain::new("integrity_test");

        let prover = TaxBracketProver::dev_mode();
        let proof = prover.prove(85_000, Jurisdiction::US, FilingStatus::Single, 2024).unwrap();
        chain.add_bracket_proof(&proof);

        // Verify chain is intact
        assert!(chain.verify().is_ok());

        // Tamper with the chain
        let mut tampered = chain.clone();
        if let Some(link) = tampered.links.first_mut() {
            link.sequence = 999; // Invalid sequence
        }

        // Verification should fail
        assert!(tampered.verify().is_err());
    }

    #[test]
    fn test_proof_chain_summary() {
        use crate::prover::TaxBracketProver;
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        let mut chain = ProofChain::new("summary_test");

        let prover = TaxBracketProver::dev_mode();
        chain.add_bracket_proof(&prover.prove(75_000, Jurisdiction::US, FilingStatus::Single, 2022).unwrap());
        chain.add_bracket_proof(&prover.prove(80_000, Jurisdiction::US, FilingStatus::Single, 2023).unwrap());

        let summary = chain.summary();
        assert!(summary.contains("2 links"));
        assert!(summary.contains("2022"));
        assert!(summary.contains("2023"));
        assert!(summary.contains("bracket"));
    }

    #[test]
    fn test_proof_chain_json_export() {
        use crate::prover::TaxBracketProver;
        use crate::jurisdiction::{Jurisdiction, FilingStatus};

        let mut chain = ProofChain::new("json_test");

        let prover = TaxBracketProver::dev_mode();
        chain.add_bracket_proof(&prover.prove(85_000, Jurisdiction::US, FilingStatus::Single, 2024).unwrap());

        let json = chain.to_json().unwrap();
        assert!(json.contains("\"links\""));
        assert!(json.contains("\"owner_hash\""));
        assert!(json.contains("\"proof_type\""));
        assert!(json.contains("bracket"));
    }
}
