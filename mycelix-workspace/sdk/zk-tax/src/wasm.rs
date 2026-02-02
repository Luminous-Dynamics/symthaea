//! WebAssembly bindings for browser-based tax proof generation.
//!
//! This module provides wasm-bindgen exports for use in web applications.
//!
//! # Usage in JavaScript
//!
//! ```javascript
//! import init, {
//!     prove_bracket,
//!     verify_proof,
//!     get_brackets,
//!     compress_proof,
//!     decompress_proof
//! } from 'mycelix-zk-tax';
//!
//! // Initialize the WASM module
//! await init();
//!
//! // Generate a proof
//! const proof = prove_bracket(85000, "US", "single", 2024);
//! console.log("Bracket:", proof.bracket_index);
//! console.log("Rate:", proof.rate_bps / 100, "%");
//!
//! // Verify a proof
//! const isValid = verify_proof(proof);
//! console.log("Valid:", isValid);
//!
//! // Compress for storage
//! const compressed = compress_proof(proof);
//! console.log("Compressed size:", compressed.length);
//! ```

#![cfg(feature = "wasm")]

use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use crate::{
    Jurisdiction, FilingStatus, TaxBracketProof,
    proof::{CompressedProof, CompressedProofType, RangeProofBuilder, BatchProofBuilder},
    brackets::get_brackets as sdk_get_brackets,
};

// =============================================================================
// Initialization
// =============================================================================

/// Initialize the WASM module. Call this before using any other functions.
#[wasm_bindgen(start)]
pub fn init() {
    // Set up panic hook for better error messages
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

// =============================================================================
// Proof Generation
// =============================================================================

/// Generate a tax bracket proof.
///
/// # Arguments
/// * `income` - Annual gross income in dollars
/// * `jurisdiction` - Country code (US, UK, DE, etc.)
/// * `filing_status` - Filing status (single, mfj, mfs, hoh)
/// * `tax_year` - Tax year (2020-2025)
///
/// # Returns
/// A JSON string containing the proof
#[wasm_bindgen]
pub fn prove_bracket(
    income: u64,
    jurisdiction: &str,
    filing_status: &str,
    tax_year: u32,
) -> Result<JsValue, JsError> {
    let jurisdiction = parse_jurisdiction(jurisdiction)?;
    let filing_status = parse_filing_status(filing_status)?;

    // Always use dev mode in WASM (real proofs require too much memory)
    let prover = crate::TaxBracketProver::dev_mode();

    let proof = prover
        .prove(income, jurisdiction, filing_status, tax_year)
        .map_err(|e| JsError::new(&e.to_string()))?;

    Ok(serde_wasm_bindgen::to_value(&proof)?)
}

/// Generate a range proof.
#[wasm_bindgen]
pub fn prove_range(
    income: u64,
    tax_year: u32,
    min_income: Option<u64>,
    max_income: Option<u64>,
) -> Result<JsValue, JsError> {
    let mut builder = RangeProofBuilder::new(income, tax_year);

    if let Some(min) = min_income {
        builder = builder.min_income(min);
    }
    if let Some(max) = max_income {
        builder = builder.max_income(max);
    }

    let proof = builder
        .build_dev()
        .map_err(|e| JsError::new(&e.to_string()))?;

    Ok(serde_wasm_bindgen::to_value(&proof)?)
}

/// Generate a batch (multi-year) proof.
///
/// # Arguments
/// * `years_json` - JSON array of {year, income} objects
/// * `jurisdiction` - Country code
/// * `filing_status` - Filing status
#[wasm_bindgen]
pub fn prove_batch(
    years_json: &str,
    jurisdiction: &str,
    filing_status: &str,
) -> Result<JsValue, JsError> {
    #[derive(Deserialize)]
    struct YearIncome {
        year: u32,
        income: u64,
    }

    let years: Vec<YearIncome> = serde_json::from_str(years_json)
        .map_err(|e| JsError::new(&format!("Invalid years JSON: {}", e)))?;

    let jurisdiction = parse_jurisdiction(jurisdiction)?;
    let filing_status = parse_filing_status(filing_status)?;

    let mut builder = BatchProofBuilder::new(jurisdiction, filing_status);
    for yi in years {
        builder = builder.add_year(yi.year, yi.income);
    }

    let proof = builder
        .build_dev()
        .map_err(|e| JsError::new(&e.to_string()))?;

    Ok(serde_wasm_bindgen::to_value(&proof)?)
}

// =============================================================================
// Verification
// =============================================================================

/// Verify a tax bracket proof.
///
/// # Arguments
/// * `proof` - The proof object (from prove_bracket)
///
/// # Returns
/// True if valid, throws error if invalid
#[wasm_bindgen]
pub fn verify_proof(proof: JsValue) -> Result<bool, JsError> {
    let proof: TaxBracketProof = serde_wasm_bindgen::from_value(proof)?;

    proof
        .verify()
        .map_err(|e| JsError::new(&e.to_string()))?;

    Ok(true)
}

/// Get verification details for a proof.
#[wasm_bindgen]
pub fn get_proof_info(proof: JsValue) -> Result<JsValue, JsError> {
    let proof: TaxBracketProof = serde_wasm_bindgen::from_value(proof)?;

    #[derive(Serialize)]
    struct ProofInfo {
        jurisdiction: String,
        jurisdiction_name: String,
        filing_status: String,
        tax_year: u32,
        bracket_index: u8,
        rate_percent: f64,
        bracket_lower: u64,
        bracket_upper: u64,
        commitment: String,
        is_dev_mode: bool,
    }

    let info = ProofInfo {
        jurisdiction: proof.jurisdiction.code().to_string(),
        jurisdiction_name: proof.jurisdiction.name().to_string(),
        filing_status: proof.filing_status.name().to_string(),
        tax_year: proof.tax_year,
        bracket_index: proof.bracket_index,
        rate_percent: proof.rate_bps as f64 / 100.0,
        bracket_lower: proof.bracket_lower,
        bracket_upper: proof.bracket_upper,
        commitment: proof.commitment.to_hex(),
        is_dev_mode: proof.receipt_bytes == vec![0xDE, 0xAD, 0xBE, 0xEF],
    };

    Ok(serde_wasm_bindgen::to_value(&info)?)
}

// =============================================================================
// Compression
// =============================================================================

/// Compress a proof for efficient storage.
#[wasm_bindgen]
pub fn compress_proof(proof: JsValue) -> Result<JsValue, JsError> {
    let proof: TaxBracketProof = serde_wasm_bindgen::from_value(proof)?;

    let compressed = CompressedProof::compress(&proof, CompressedProofType::RleBase64)
        .map_err(|e| JsError::new(&e.to_string()))?;

    Ok(serde_wasm_bindgen::to_value(&compressed)?)
}

/// Decompress a proof.
#[wasm_bindgen]
pub fn decompress_proof(compressed: JsValue) -> Result<JsValue, JsError> {
    let compressed: CompressedProof = serde_wasm_bindgen::from_value(compressed)?;

    let proof = compressed
        .decompress()
        .map_err(|e| JsError::new(&e.to_string()))?;

    Ok(serde_wasm_bindgen::to_value(&proof)?)
}

// =============================================================================
// Bracket Information
// =============================================================================

/// Get tax brackets for a jurisdiction.
#[wasm_bindgen]
pub fn get_brackets(
    jurisdiction: &str,
    tax_year: u32,
    filing_status: &str,
) -> Result<JsValue, JsError> {
    let jurisdiction = parse_jurisdiction(jurisdiction)?;
    let filing_status = parse_filing_status(filing_status)?;

    let brackets = sdk_get_brackets(jurisdiction, tax_year, filing_status)
        .map_err(|e| JsError::new(&e.to_string()))?;

    #[derive(Serialize)]
    struct BracketInfo {
        index: usize,
        lower: u64,
        upper: u64,
        rate_percent: f64,
    }

    let infos: Vec<BracketInfo> = brackets
        .iter()
        .enumerate()
        .map(|(i, b)| BracketInfo {
            index: i,
            lower: b.lower,
            upper: b.upper,
            rate_percent: b.rate_bps as f64 / 100.0,
        })
        .collect();

    Ok(serde_wasm_bindgen::to_value(&infos)?)
}

/// Get list of supported jurisdictions (all 58).
#[wasm_bindgen]
pub fn get_jurisdictions() -> JsValue {
    #[derive(Serialize)]
    struct JurisdictionInfo {
        code: &'static str,
        name: &'static str,
        currency: &'static str,
        region: &'static str,
    }

    let jurisdictions = vec![
        // Americas
        JurisdictionInfo { code: "US", name: "United States", currency: "USD", region: "Americas" },
        JurisdictionInfo { code: "CA", name: "Canada", currency: "CAD", region: "Americas" },
        JurisdictionInfo { code: "MX", name: "Mexico", currency: "MXN", region: "Americas" },
        JurisdictionInfo { code: "BR", name: "Brazil", currency: "BRL", region: "Americas" },
        JurisdictionInfo { code: "AR", name: "Argentina", currency: "ARS", region: "Americas" },
        JurisdictionInfo { code: "CL", name: "Chile", currency: "CLP", region: "Americas" },
        JurisdictionInfo { code: "CO", name: "Colombia", currency: "COP", region: "Americas" },
        JurisdictionInfo { code: "PE", name: "Peru", currency: "PEN", region: "Americas" },
        JurisdictionInfo { code: "EC", name: "Ecuador", currency: "USD", region: "Americas" },
        JurisdictionInfo { code: "UY", name: "Uruguay", currency: "UYU", region: "Americas" },
        // Europe
        JurisdictionInfo { code: "UK", name: "United Kingdom", currency: "GBP", region: "Europe" },
        JurisdictionInfo { code: "DE", name: "Germany", currency: "EUR", region: "Europe" },
        JurisdictionInfo { code: "FR", name: "France", currency: "EUR", region: "Europe" },
        JurisdictionInfo { code: "IT", name: "Italy", currency: "EUR", region: "Europe" },
        JurisdictionInfo { code: "ES", name: "Spain", currency: "EUR", region: "Europe" },
        JurisdictionInfo { code: "NL", name: "Netherlands", currency: "EUR", region: "Europe" },
        JurisdictionInfo { code: "BE", name: "Belgium", currency: "EUR", region: "Europe" },
        JurisdictionInfo { code: "AT", name: "Austria", currency: "EUR", region: "Europe" },
        JurisdictionInfo { code: "PT", name: "Portugal", currency: "EUR", region: "Europe" },
        JurisdictionInfo { code: "IE", name: "Ireland", currency: "EUR", region: "Europe" },
        JurisdictionInfo { code: "PL", name: "Poland", currency: "PLN", region: "Europe" },
        JurisdictionInfo { code: "SE", name: "Sweden", currency: "SEK", region: "Europe" },
        JurisdictionInfo { code: "DK", name: "Denmark", currency: "DKK", region: "Europe" },
        JurisdictionInfo { code: "FI", name: "Finland", currency: "EUR", region: "Europe" },
        JurisdictionInfo { code: "NO", name: "Norway", currency: "NOK", region: "Europe" },
        JurisdictionInfo { code: "CH", name: "Switzerland", currency: "CHF", region: "Europe" },
        JurisdictionInfo { code: "CZ", name: "Czech Republic", currency: "CZK", region: "Europe" },
        JurisdictionInfo { code: "GR", name: "Greece", currency: "EUR", region: "Europe" },
        JurisdictionInfo { code: "HU", name: "Hungary", currency: "HUF", region: "Europe" },
        JurisdictionInfo { code: "RO", name: "Romania", currency: "RON", region: "Europe" },
        JurisdictionInfo { code: "RU", name: "Russia", currency: "RUB", region: "Europe" },
        JurisdictionInfo { code: "TR", name: "Turkey", currency: "TRY", region: "Europe" },
        JurisdictionInfo { code: "UA", name: "Ukraine", currency: "UAH", region: "Europe" },
        // Asia-Pacific
        JurisdictionInfo { code: "JP", name: "Japan", currency: "JPY", region: "Asia-Pacific" },
        JurisdictionInfo { code: "CN", name: "China", currency: "CNY", region: "Asia-Pacific" },
        JurisdictionInfo { code: "IN", name: "India", currency: "INR", region: "Asia-Pacific" },
        JurisdictionInfo { code: "KR", name: "South Korea", currency: "KRW", region: "Asia-Pacific" },
        JurisdictionInfo { code: "ID", name: "Indonesia", currency: "IDR", region: "Asia-Pacific" },
        JurisdictionInfo { code: "AU", name: "Australia", currency: "AUD", region: "Asia-Pacific" },
        JurisdictionInfo { code: "NZ", name: "New Zealand", currency: "NZD", region: "Asia-Pacific" },
        JurisdictionInfo { code: "SG", name: "Singapore", currency: "SGD", region: "Asia-Pacific" },
        JurisdictionInfo { code: "HK", name: "Hong Kong", currency: "HKD", region: "Asia-Pacific" },
        JurisdictionInfo { code: "TW", name: "Taiwan", currency: "TWD", region: "Asia-Pacific" },
        JurisdictionInfo { code: "MY", name: "Malaysia", currency: "MYR", region: "Asia-Pacific" },
        JurisdictionInfo { code: "TH", name: "Thailand", currency: "THB", region: "Asia-Pacific" },
        JurisdictionInfo { code: "VN", name: "Vietnam", currency: "VND", region: "Asia-Pacific" },
        JurisdictionInfo { code: "PH", name: "Philippines", currency: "PHP", region: "Asia-Pacific" },
        JurisdictionInfo { code: "PK", name: "Pakistan", currency: "PKR", region: "Asia-Pacific" },
        // Middle East
        JurisdictionInfo { code: "SA", name: "Saudi Arabia", currency: "SAR", region: "Middle East" },
        JurisdictionInfo { code: "AE", name: "UAE", currency: "AED", region: "Middle East" },
        JurisdictionInfo { code: "IL", name: "Israel", currency: "ILS", region: "Middle East" },
        JurisdictionInfo { code: "EG", name: "Egypt", currency: "EGP", region: "Middle East" },
        JurisdictionInfo { code: "QA", name: "Qatar", currency: "QAR", region: "Middle East" },
        // Africa
        JurisdictionInfo { code: "ZA", name: "South Africa", currency: "ZAR", region: "Africa" },
        JurisdictionInfo { code: "NG", name: "Nigeria", currency: "NGN", region: "Africa" },
        JurisdictionInfo { code: "KE", name: "Kenya", currency: "KES", region: "Africa" },
        JurisdictionInfo { code: "MA", name: "Morocco", currency: "MAD", region: "Africa" },
        JurisdictionInfo { code: "GH", name: "Ghana", currency: "GHS", region: "Africa" },
    ];

    serde_wasm_bindgen::to_value(&jurisdictions).unwrap()
}

/// Get supported tax years.
#[wasm_bindgen]
pub fn get_supported_years() -> Vec<u32> {
    vec![2020, 2021, 2022, 2023, 2024, 2025]
}

/// Get SDK version.
#[wasm_bindgen]
pub fn get_version() -> String {
    crate::VERSION.to_string()
}

// =============================================================================
// Helpers
// =============================================================================

fn parse_jurisdiction(s: &str) -> Result<Jurisdiction, JsError> {
    Jurisdiction::from_code(s)
        .ok_or_else(|| JsError::new(&format!("Unknown jurisdiction: {}", s)))
}

fn parse_filing_status(s: &str) -> Result<FilingStatus, JsError> {
    match s.to_lowercase().as_str() {
        "single" | "s" => Ok(FilingStatus::Single),
        "mfj" | "married_filing_jointly" => Ok(FilingStatus::MarriedFilingJointly),
        "mfs" | "married_filing_separately" => Ok(FilingStatus::MarriedFilingSeparately),
        "hoh" | "head_of_household" => Ok(FilingStatus::HeadOfHousehold),
        _ => Err(JsError::new(&format!("Invalid filing status: {}", s))),
    }
}

// =============================================================================
// TypeScript Type Definitions
// =============================================================================

#[wasm_bindgen(typescript_custom_section)]
const TS_TYPES: &'static str = r#"
export interface TaxBracketProof {
    jurisdiction: string;
    filing_status: string;
    tax_year: number;
    bracket_index: number;
    rate_bps: number;
    bracket_lower: number;
    bracket_upper: number;
    commitment: number[];
    receipt_bytes: number[];
    image_id: number[];
}

export interface ProofInfo {
    jurisdiction: string;
    jurisdiction_name: string;
    filing_status: string;
    tax_year: number;
    bracket_index: number;
    rate_percent: number;
    bracket_lower: number;
    bracket_upper: number;
    commitment: string;
    is_dev_mode: boolean;
}

export interface BracketInfo {
    index: number;
    lower: number;
    upper: number;
    rate_percent: number;
}

export interface JurisdictionInfo {
    code: string;
    name: string;
    currency: string;
    region: string;
}

export interface RangeProof {
    income_commitment: number[];
    min_bound: number | null;
    max_bound: number | null;
    tax_year: number;
    receipt_bytes: number[];
    image_id: number[];
}

export interface BatchProof {
    jurisdiction: string;
    filing_status: string;
    proofs: TaxBracketProof[];
    aggregation_commitment: number[];
}

export type Region = 'Americas' | 'Europe' | 'Asia-Pacific' | 'Middle East' | 'Africa';
export type FilingStatusCode = 'single' | 'mfj' | 'mfs' | 'hoh';
export type JurisdictionCode =
    | 'US' | 'CA' | 'MX' | 'BR' | 'AR' | 'CL' | 'CO' | 'PE' | 'EC' | 'UY'  // Americas
    | 'UK' | 'DE' | 'FR' | 'IT' | 'ES' | 'NL' | 'BE' | 'AT' | 'PT' | 'IE'  // Europe
    | 'PL' | 'SE' | 'DK' | 'FI' | 'NO' | 'CH' | 'CZ' | 'GR' | 'HU' | 'RO'
    | 'RU' | 'TR' | 'UA'
    | 'JP' | 'CN' | 'IN' | 'KR' | 'ID' | 'AU' | 'NZ' | 'SG' | 'HK' | 'TW'  // Asia-Pacific
    | 'MY' | 'TH' | 'VN' | 'PH' | 'PK'
    | 'SA' | 'AE' | 'IL' | 'EG' | 'QA'  // Middle East
    | 'ZA' | 'NG' | 'KE' | 'MA' | 'GH'; // Africa

export interface CompressedProof {
    compression_type: string;
    original_size: number;
    compressed_data: string;
}
"#;
