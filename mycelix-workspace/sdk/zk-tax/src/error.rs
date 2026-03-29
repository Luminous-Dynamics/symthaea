// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Error types for ZK tax SDK.
//!
//! This module provides detailed, actionable error messages for all
//! SDK operations. Each error includes context and suggestions for resolution.

use thiserror::Error;

/// Result type alias for SDK operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during ZK tax proof operations.
///
/// All errors include detailed context and suggestions for resolution.
#[derive(Error, Debug)]
pub enum Error {
    /// Income doesn't fall within the specified bracket bounds.
    #[error(
        "Income validation failed: ${income} is outside bracket bounds [${lower}, ${upper})\n\
         \n\
         This usually means:\n\
         1. The bracket data is incorrect for this jurisdiction/year\n\
         2. The income value may need adjustment (ensure it's annual gross income)\n\
         \n\
         Tip: Use `find_bracket()` to get the correct bracket for your income."
    )]
    IncomeOutOfBounds {
        income: u64,
        lower: u64,
        upper: u64,
    },

    /// The specified tax year is not supported.
    #[error(
        "Unsupported tax year: {year}\n\
         \n\
         Supported years: {}\n\
         \n\
         Note: Tax bracket data is typically available for the current year\n\
         and 5 years back. Future years may use projected brackets.\n\
         \n\
         Did you mean: {}?",
        supported_years_str(),
        suggest_year(*year)
    )]
    UnsupportedTaxYear { year: u32 },

    /// The specified jurisdiction is not supported.
    #[error(
        "Unsupported jurisdiction: '{input}'\n\
         \n\
         Supported jurisdictions (G20 countries):\n\
         \n\
         Americas: US, CA, MX, BR, AR\n\
         Europe:   UK, DE, FR, IT, RU, TR\n\
         Asia:     JP, CN, IN, KR, ID, AU\n\
         Other:    SA, ZA\n\
         \n\
         {}",
        suggest_jurisdiction(input)
    )]
    UnsupportedJurisdiction { input: String },

    /// The filing status is not valid for the jurisdiction.
    #[error(
        "Invalid filing status: '{status}' is not valid for {jurisdiction}\n\
         \n\
         Valid filing statuses for {jurisdiction}:\n\
         {valid_statuses}\n\
         \n\
         Common mapping:\n\
         - single: Single/unmarried filer\n\
         - mfj:    Married Filing Jointly (US/CA)\n\
         - mfs:    Married Filing Separately (US/CA)\n\
         - hoh:    Head of Household (US only)"
    )]
    InvalidFilingStatus {
        status: String,
        jurisdiction: String,
        valid_statuses: String,
    },

    /// No bracket found for the given income.
    #[error(
        "No tax bracket found for income: ${income}\n\
         \n\
         This is unexpected and may indicate:\n\
         1. Missing bracket data for this jurisdiction/year combination\n\
         2. An extremely high income that exceeds defined brackets\n\
         \n\
         Please check:\n\
         - The jurisdiction supports the requested tax year\n\
         - The income value is reasonable (USD equivalent)"
    )]
    NoBracketFound { income: u64 },

    /// Proof generation failed.
    #[error(
        "Proof generation failed: {reason}\n\
         \n\
         Context: {context}\n\
         \n\
         Possible causes:\n\
         1. Input values don't satisfy proof constraints\n\
         2. Insufficient memory for ZK circuit execution\n\
         3. Internal prover error\n\
         \n\
         Suggestion: {suggestion}"
    )]
    ProofGenerationFailed {
        reason: String,
        context: String,
        suggestion: String,
    },

    /// Proof verification failed.
    #[error(
        "Proof verification failed: {reason}\n\
         \n\
         This means the proof is invalid and should NOT be trusted.\n\
         \n\
         Possible causes:\n\
         1. Proof was tampered with\n\
         2. Proof was generated with incorrect inputs\n\
         3. Proof format is corrupted\n\
         \n\
         Security: Never accept a proof that fails verification."
    )]
    VerificationFailed { reason: String },

    /// Invalid proof format or data.
    #[error(
        "Invalid proof format: {reason}\n\
         \n\
         The proof data could not be parsed or validated.\n\
         \n\
         Common issues:\n\
         1. Proof was truncated during transmission\n\
         2. Wrong proof type (e.g., using bracket proof as range proof)\n\
         3. Proof version mismatch\n\
         \n\
         Ensure you're using matching SDK versions for proving and verifying."
    )]
    InvalidProof { reason: String },

    /// Commitment mismatch during verification.
    #[error(
        "Cryptographic commitment mismatch\n\
         \n\
         Expected: {expected}\n\
         Actual:   {actual}\n\
         \n\
         This indicates the proof inputs don't match the claimed outputs.\n\
         The proof should be considered INVALID.\n\
         \n\
         This could mean:\n\
         1. The prover used different inputs than claimed\n\
         2. The proof data was modified after generation\n\
         3. Different SDK versions were used"
    )]
    CommitmentMismatch { expected: String, actual: String },

    /// Serialization or deserialization error.
    #[error(
        "Serialization error: {reason}\n\
         \n\
         Failed to serialize or deserialize proof data.\n\
         \n\
         Common causes:\n\
         1. Corrupted proof bytes\n\
         2. Incompatible format version\n\
         3. Truncated data\n\
         \n\
         Suggestion: Ensure proof data is transmitted without modification."
    )]
    SerializationError { reason: String },

    /// Subnational region not supported.
    #[error(
        "Unsupported subnational region: '{region}' ({region_type})\n\
         \n\
         {available_regions}\n\
         \n\
         Note: Subnational tax data is available for:\n\
         - US: All 50 states + DC\n\
         - Canada: All provinces and territories\n\
         - Switzerland: All 26 cantons\n\
         - UK: England, Wales, Scotland, Northern Ireland\n\
         - Brazil: All 27 states"
    )]
    UnsupportedSubnationalRegion {
        region: String,
        region_type: String,
        available_regions: String,
    },

    /// Batch proof years don't match incomes.
    #[error(
        "Mismatched batch proof inputs\n\
         \n\
         Years provided: {year_count}\n\
         Incomes provided: {income_count}\n\
         \n\
         Each year must have a corresponding income value.\n\
         \n\
         Example:\n\
         ```\n\
         BatchProofBuilder::new(Jurisdiction::US, FilingStatus::Single)\n\
             .add_year(2022, 75_000)\n\
             .add_year(2023, 80_000)\n\
             .add_year(2024, 85_000)\n\
             .build_dev()\n\
         ```"
    )]
    BatchInputMismatch {
        year_count: usize,
        income_count: usize,
    },

    /// WASM-specific error (only available with wasm feature).
    #[cfg(feature = "wasm")]
    #[error(
        "WASM error: {reason}\n\
         \n\
         This error occurred in the WebAssembly runtime.\n\
         \n\
         Common causes:\n\
         1. Module not initialized (call init() first)\n\
         2. Invalid parameter types from JavaScript\n\
         3. Memory allocation failure\n\
         \n\
         Ensure `await init()` is called before using any SDK functions."
    )]
    WasmError { reason: String },
}

// =============================================================================
// Error Construction Helpers
// =============================================================================

impl Error {
    /// Create a simple proof generation error.
    pub fn proof_generation(reason: impl Into<String>) -> Self {
        Error::ProofGenerationFailed {
            reason: reason.into(),
            context: "Unknown".to_string(),
            suggestion: "Check inputs and try again".to_string(),
        }
    }

    /// Create a proof generation error with context.
    pub fn proof_generation_with_context(
        reason: impl Into<String>,
        context: impl Into<String>,
        suggestion: impl Into<String>,
    ) -> Self {
        Error::ProofGenerationFailed {
            reason: reason.into(),
            context: context.into(),
            suggestion: suggestion.into(),
        }
    }

    /// Create a simple invalid proof error.
    pub fn invalid_proof(reason: impl Into<String>) -> Self {
        Error::InvalidProof { reason: reason.into() }
    }

    /// Create a verification failed error.
    pub fn verification_failed(reason: impl Into<String>) -> Self {
        Error::VerificationFailed { reason: reason.into() }
    }

    /// Create an unsupported tax year error.
    pub fn unsupported_year(year: u32) -> Self {
        Error::UnsupportedTaxYear { year }
    }

    /// Create an unsupported jurisdiction error.
    pub fn unsupported_jurisdiction(input: impl Into<String>) -> Self {
        Error::UnsupportedJurisdiction { input: input.into() }
    }

    /// Create an invalid filing status error with valid options.
    pub fn invalid_filing_status(
        status: impl Into<String>,
        jurisdiction: impl Into<String>,
        valid: &[&str],
    ) -> Self {
        Error::InvalidFilingStatus {
            status: status.into(),
            jurisdiction: jurisdiction.into(),
            valid_statuses: valid.join(", "),
        }
    }

    /// Create a no bracket found error.
    pub fn no_bracket(income: u64) -> Self {
        Error::NoBracketFound { income }
    }

    /// Create a serialization error.
    pub fn serialization(reason: impl Into<String>) -> Self {
        Error::SerializationError { reason: reason.into() }
    }
}

// =============================================================================
// Helper Functions for Error Messages
// =============================================================================

/// Get comma-separated list of supported years.
fn supported_years_str() -> String {
    "2020, 2021, 2022, 2023, 2024, 2025".to_string()
}

/// Suggest the closest valid year.
fn suggest_year(year: u32) -> String {
    if year < 2020 {
        format!("2020 (earliest supported)")
    } else if year > 2025 {
        format!("2025 (latest supported)")
    } else {
        format!("{}", year)
    }
}

/// Suggest the closest jurisdiction based on input.
fn suggest_jurisdiction(input: &str) -> String {
    let input_upper = input.to_uppercase();

    // Common aliases
    let suggestion = match input_upper.as_str() {
        "USA" | "UNITED STATES" | "AMERICA" => Some("US (United States)"),
        "UNITED KINGDOM" | "BRITAIN" | "ENGLAND" | "SCOTLAND" | "WALES" => Some("UK (United Kingdom)"),
        "GERMANY" | "DEUTSCHLAND" => Some("DE (Germany)"),
        "FRANCE" => Some("FR (France)"),
        "JAPAN" | "NIHON" => Some("JP (Japan)"),
        "CHINA" | "ZHONGGUO" | "PRC" => Some("CN (China)"),
        "INDIA" | "BHARAT" => Some("IN (India)"),
        "CANADA" => Some("CA (Canada)"),
        "MEXICO" | "MÉXICO" => Some("MX (Mexico)"),
        "BRAZIL" | "BRASIL" => Some("BR (Brazil)"),
        "ARGENTINA" => Some("AR (Argentina)"),
        "ITALY" | "ITALIA" => Some("IT (Italy)"),
        "RUSSIA" | "РОССИЯ" => Some("RU (Russia)"),
        "TURKEY" | "TÜRKIYE" => Some("TR (Turkey)"),
        "SOUTH KOREA" | "KOREA" | "한국" => Some("KR (South Korea)"),
        "INDONESIA" => Some("ID (Indonesia)"),
        "AUSTRALIA" => Some("AU (Australia)"),
        "SAUDI" | "SAUDI ARABIA" | "KSA" => Some("SA (Saudi Arabia)"),
        "SOUTH AFRICA" | "ZA" => Some("ZA (South Africa)"),
        _ => None,
    };

    match suggestion {
        Some(s) => format!("Did you mean: {}?", s),
        None => "Use the 2-letter country code (ISO 3166-1 alpha-2).".to_string(),
    }
}

// =============================================================================
// Trait Implementations
// =============================================================================

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::SerializationError { reason: e.to_string() }
    }
}

// Note: Risc0 uses anyhow::Error internally, so we convert via string

// =============================================================================
// Error Code for Programmatic Handling
// =============================================================================

impl Error {
    /// Get a machine-readable error code for programmatic handling.
    ///
    /// These codes are stable across SDK versions.
    pub fn code(&self) -> &'static str {
        match self {
            Error::IncomeOutOfBounds { .. } => "E001_INCOME_OUT_OF_BOUNDS",
            Error::UnsupportedTaxYear { .. } => "E002_UNSUPPORTED_TAX_YEAR",
            Error::UnsupportedJurisdiction { .. } => "E003_UNSUPPORTED_JURISDICTION",
            Error::InvalidFilingStatus { .. } => "E004_INVALID_FILING_STATUS",
            Error::NoBracketFound { .. } => "E005_NO_BRACKET_FOUND",
            Error::ProofGenerationFailed { .. } => "E010_PROOF_GENERATION_FAILED",
            Error::VerificationFailed { .. } => "E011_VERIFICATION_FAILED",
            Error::InvalidProof { .. } => "E012_INVALID_PROOF",
            Error::CommitmentMismatch { .. } => "E013_COMMITMENT_MISMATCH",
            Error::SerializationError { .. } => "E020_SERIALIZATION_ERROR",
            Error::UnsupportedSubnationalRegion { .. } => "E030_UNSUPPORTED_SUBNATIONAL",
            Error::BatchInputMismatch { .. } => "E040_BATCH_INPUT_MISMATCH",
            #[cfg(feature = "wasm")]
            Error::WasmError { .. } => "E100_WASM_ERROR",
        }
    }

    /// Check if this error is recoverable (can be fixed by user action).
    pub fn is_recoverable(&self) -> bool {
        match self {
            Error::IncomeOutOfBounds { .. } => true,
            Error::UnsupportedTaxYear { .. } => true,
            Error::UnsupportedJurisdiction { .. } => true,
            Error::InvalidFilingStatus { .. } => true,
            Error::NoBracketFound { .. } => true,
            Error::BatchInputMismatch { .. } => true,
            Error::ProofGenerationFailed { .. } => false,
            Error::VerificationFailed { .. } => false,
            Error::InvalidProof { .. } => false,
            Error::CommitmentMismatch { .. } => false,
            Error::SerializationError { .. } => false,
            Error::UnsupportedSubnationalRegion { .. } => true,
            #[cfg(feature = "wasm")]
            Error::WasmError { .. } => true,
        }
    }
}
