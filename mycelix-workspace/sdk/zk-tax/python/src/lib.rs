//! Python bindings for ZK Tax SDK.
//!
//! This module provides Python bindings using PyO3, enabling use of
//! zero-knowledge tax proofs from Python applications.
//!
//! # Example (Python)
//!
//! ```python
//! from zk_tax import Jurisdiction, FilingStatus, find_bracket, get_brackets
//!
//! # Find bracket for an income
//! bracket = find_bracket(85000, Jurisdiction.US, 2024, FilingStatus.SINGLE)
//! print(f"Bracket: {bracket.index}, Rate: {bracket.rate_percent}%")
//!
//! # Get all brackets for a jurisdiction
//! brackets = get_brackets(Jurisdiction.US, 2024, FilingStatus.SINGLE)
//! for b in brackets:
//!     print(f"  {b.lower} - {b.upper}: {b.rate_percent}%")
//! ```

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use mycelix_zk_tax::{
    Jurisdiction as RustJurisdiction,
    FilingStatus as RustFilingStatus,
    TaxBracket as RustTaxBracket,
    brackets::{find_bracket as rust_find_bracket, get_brackets as rust_get_brackets},
    TaxBracketProof as RustTaxBracketProof,
};

// =============================================================================
// Jurisdiction Enum
// =============================================================================

/// Tax jurisdiction (country).
#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Jurisdiction {
    US,
    UK,
    DE,
    FR,
    CA,
    AU,
    JP,
    CN,
    IN,
    BR,
    MX,
    KR,
    IT,
    RU,
    TR,
    AR,
    ID,
    SA,
    ZA,
    // Extended jurisdictions
    CL,
    CO,
    PE,
    EC,
    UY,
    ES,
    NL,
    BE,
    AT,
    PT,
    IE,
    PL,
    SE,
    DK,
    FI,
    NO,
    CH,
    CZ,
    GR,
    HU,
    RO,
    UA,
    NZ,
    SG,
    HK,
    TW,
    MY,
    TH,
    VN,
    PH,
    PK,
    AE,
    IL,
    EG,
    QA,
    NG,
    KE,
    MA,
    GH,
}

impl From<Jurisdiction> for RustJurisdiction {
    fn from(j: Jurisdiction) -> Self {
        match j {
            Jurisdiction::US => RustJurisdiction::US,
            Jurisdiction::UK => RustJurisdiction::UK,
            Jurisdiction::DE => RustJurisdiction::DE,
            Jurisdiction::FR => RustJurisdiction::FR,
            Jurisdiction::CA => RustJurisdiction::CA,
            Jurisdiction::AU => RustJurisdiction::AU,
            Jurisdiction::JP => RustJurisdiction::JP,
            Jurisdiction::CN => RustJurisdiction::CN,
            Jurisdiction::IN => RustJurisdiction::IN,
            Jurisdiction::BR => RustJurisdiction::BR,
            Jurisdiction::MX => RustJurisdiction::MX,
            Jurisdiction::KR => RustJurisdiction::KR,
            Jurisdiction::IT => RustJurisdiction::IT,
            Jurisdiction::RU => RustJurisdiction::RU,
            Jurisdiction::TR => RustJurisdiction::TR,
            Jurisdiction::AR => RustJurisdiction::AR,
            Jurisdiction::ID => RustJurisdiction::ID,
            Jurisdiction::SA => RustJurisdiction::SA,
            Jurisdiction::ZA => RustJurisdiction::ZA,
            Jurisdiction::CL => RustJurisdiction::CL,
            Jurisdiction::CO => RustJurisdiction::CO,
            Jurisdiction::PE => RustJurisdiction::PE,
            Jurisdiction::EC => RustJurisdiction::EC,
            Jurisdiction::UY => RustJurisdiction::UY,
            Jurisdiction::ES => RustJurisdiction::ES,
            Jurisdiction::NL => RustJurisdiction::NL,
            Jurisdiction::BE => RustJurisdiction::BE,
            Jurisdiction::AT => RustJurisdiction::AT,
            Jurisdiction::PT => RustJurisdiction::PT,
            Jurisdiction::IE => RustJurisdiction::IE,
            Jurisdiction::PL => RustJurisdiction::PL,
            Jurisdiction::SE => RustJurisdiction::SE,
            Jurisdiction::DK => RustJurisdiction::DK,
            Jurisdiction::FI => RustJurisdiction::FI,
            Jurisdiction::NO => RustJurisdiction::NO,
            Jurisdiction::CH => RustJurisdiction::CH,
            Jurisdiction::CZ => RustJurisdiction::CZ,
            Jurisdiction::GR => RustJurisdiction::GR,
            Jurisdiction::HU => RustJurisdiction::HU,
            Jurisdiction::RO => RustJurisdiction::RO,
            Jurisdiction::UA => RustJurisdiction::UA,
            Jurisdiction::NZ => RustJurisdiction::NZ,
            Jurisdiction::SG => RustJurisdiction::SG,
            Jurisdiction::HK => RustJurisdiction::HK,
            Jurisdiction::TW => RustJurisdiction::TW,
            Jurisdiction::MY => RustJurisdiction::MY,
            Jurisdiction::TH => RustJurisdiction::TH,
            Jurisdiction::VN => RustJurisdiction::VN,
            Jurisdiction::PH => RustJurisdiction::PH,
            Jurisdiction::PK => RustJurisdiction::PK,
            Jurisdiction::AE => RustJurisdiction::AE,
            Jurisdiction::IL => RustJurisdiction::IL,
            Jurisdiction::EG => RustJurisdiction::EG,
            Jurisdiction::QA => RustJurisdiction::QA,
            Jurisdiction::NG => RustJurisdiction::NG,
            Jurisdiction::KE => RustJurisdiction::KE,
            Jurisdiction::MA => RustJurisdiction::MA,
            Jurisdiction::GH => RustJurisdiction::GH,
        }
    }
}

impl From<RustJurisdiction> for Jurisdiction {
    fn from(j: RustJurisdiction) -> Self {
        match j {
            RustJurisdiction::US => Jurisdiction::US,
            RustJurisdiction::UK => Jurisdiction::UK,
            RustJurisdiction::DE => Jurisdiction::DE,
            RustJurisdiction::FR => Jurisdiction::FR,
            RustJurisdiction::CA => Jurisdiction::CA,
            RustJurisdiction::AU => Jurisdiction::AU,
            RustJurisdiction::JP => Jurisdiction::JP,
            RustJurisdiction::CN => Jurisdiction::CN,
            RustJurisdiction::IN => Jurisdiction::IN,
            RustJurisdiction::BR => Jurisdiction::BR,
            RustJurisdiction::MX => Jurisdiction::MX,
            RustJurisdiction::KR => Jurisdiction::KR,
            RustJurisdiction::IT => Jurisdiction::IT,
            RustJurisdiction::RU => Jurisdiction::RU,
            RustJurisdiction::TR => Jurisdiction::TR,
            RustJurisdiction::AR => Jurisdiction::AR,
            RustJurisdiction::ID => Jurisdiction::ID,
            RustJurisdiction::SA => Jurisdiction::SA,
            RustJurisdiction::ZA => Jurisdiction::ZA,
            RustJurisdiction::CL => Jurisdiction::CL,
            RustJurisdiction::CO => Jurisdiction::CO,
            RustJurisdiction::PE => Jurisdiction::PE,
            RustJurisdiction::EC => Jurisdiction::EC,
            RustJurisdiction::UY => Jurisdiction::UY,
            RustJurisdiction::ES => Jurisdiction::ES,
            RustJurisdiction::NL => Jurisdiction::NL,
            RustJurisdiction::BE => Jurisdiction::BE,
            RustJurisdiction::AT => Jurisdiction::AT,
            RustJurisdiction::PT => Jurisdiction::PT,
            RustJurisdiction::IE => Jurisdiction::IE,
            RustJurisdiction::PL => Jurisdiction::PL,
            RustJurisdiction::SE => Jurisdiction::SE,
            RustJurisdiction::DK => Jurisdiction::DK,
            RustJurisdiction::FI => Jurisdiction::FI,
            RustJurisdiction::NO => Jurisdiction::NO,
            RustJurisdiction::CH => Jurisdiction::CH,
            RustJurisdiction::CZ => Jurisdiction::CZ,
            RustJurisdiction::GR => Jurisdiction::GR,
            RustJurisdiction::HU => Jurisdiction::HU,
            RustJurisdiction::RO => Jurisdiction::RO,
            RustJurisdiction::UA => Jurisdiction::UA,
            RustJurisdiction::NZ => Jurisdiction::NZ,
            RustJurisdiction::SG => Jurisdiction::SG,
            RustJurisdiction::HK => Jurisdiction::HK,
            RustJurisdiction::TW => Jurisdiction::TW,
            RustJurisdiction::MY => Jurisdiction::MY,
            RustJurisdiction::TH => Jurisdiction::TH,
            RustJurisdiction::VN => Jurisdiction::VN,
            RustJurisdiction::PH => Jurisdiction::PH,
            RustJurisdiction::PK => Jurisdiction::PK,
            RustJurisdiction::AE => Jurisdiction::AE,
            RustJurisdiction::IL => Jurisdiction::IL,
            RustJurisdiction::EG => Jurisdiction::EG,
            RustJurisdiction::QA => Jurisdiction::QA,
            RustJurisdiction::NG => Jurisdiction::NG,
            RustJurisdiction::KE => Jurisdiction::KE,
            RustJurisdiction::MA => Jurisdiction::MA,
            RustJurisdiction::GH => Jurisdiction::GH,
        }
    }
}

// =============================================================================
// Filing Status Enum
// =============================================================================

/// Tax filing status.
#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(non_camel_case_types)] // Python convention uses SCREAMING_CASE for enum values
pub enum FilingStatus {
    SINGLE,
    MARRIED_FILING_JOINTLY,
    MARRIED_FILING_SEPARATELY,
    HEAD_OF_HOUSEHOLD,
}

impl From<FilingStatus> for RustFilingStatus {
    fn from(s: FilingStatus) -> Self {
        match s {
            FilingStatus::SINGLE => RustFilingStatus::Single,
            FilingStatus::MARRIED_FILING_JOINTLY => RustFilingStatus::MarriedFilingJointly,
            FilingStatus::MARRIED_FILING_SEPARATELY => RustFilingStatus::MarriedFilingSeparately,
            FilingStatus::HEAD_OF_HOUSEHOLD => RustFilingStatus::HeadOfHousehold,
        }
    }
}

impl From<RustFilingStatus> for FilingStatus {
    fn from(s: RustFilingStatus) -> Self {
        match s {
            RustFilingStatus::Single => FilingStatus::SINGLE,
            RustFilingStatus::MarriedFilingJointly => FilingStatus::MARRIED_FILING_JOINTLY,
            RustFilingStatus::MarriedFilingSeparately => FilingStatus::MARRIED_FILING_SEPARATELY,
            RustFilingStatus::HeadOfHousehold => FilingStatus::HEAD_OF_HOUSEHOLD,
        }
    }
}

// =============================================================================
// Tax Bracket
// =============================================================================

/// A tax bracket with rate and bounds.
#[pyclass]
#[derive(Clone, Debug)]
pub struct TaxBracket {
    /// Bracket index (0-based)
    #[pyo3(get)]
    pub index: u8,
    /// Marginal tax rate in basis points (e.g., 2200 = 22%)
    #[pyo3(get)]
    pub rate_bps: u16,
    /// Lower bound of the bracket (inclusive)
    #[pyo3(get)]
    pub lower: u64,
    /// Upper bound of the bracket (exclusive)
    #[pyo3(get)]
    pub upper: u64,
}

#[pymethods]
impl TaxBracket {
    /// Get the tax rate as a percentage (e.g., 22.0 for 22%).
    #[getter]
    fn rate_percent(&self) -> f64 {
        self.rate_bps as f64 / 100.0
    }

    /// Check if an income falls within this bracket.
    fn contains(&self, income: u64) -> bool {
        income >= self.lower && (self.upper == u64::MAX || income < self.upper)
    }

    fn __repr__(&self) -> String {
        format!(
            "TaxBracket(index={}, rate={}%, lower={}, upper={})",
            self.index,
            self.rate_bps as f64 / 100.0,
            self.lower,
            if self.upper == u64::MAX { "inf".to_string() } else { self.upper.to_string() }
        )
    }
}

impl From<&RustTaxBracket> for TaxBracket {
    fn from(b: &RustTaxBracket) -> Self {
        TaxBracket {
            index: b.index,
            rate_bps: b.rate_bps,
            lower: b.lower,
            upper: b.upper,
        }
    }
}

// =============================================================================
// Tax Bracket Proof
// =============================================================================

/// A zero-knowledge proof of tax bracket membership.
#[pyclass]
#[derive(Clone)]
pub struct TaxBracketProof {
    inner: RustTaxBracketProof,
}

#[pymethods]
impl TaxBracketProof {
    /// The jurisdiction this proof is for.
    #[getter]
    fn jurisdiction(&self) -> Jurisdiction {
        self.inner.jurisdiction.into()
    }

    /// The filing status used.
    #[getter]
    fn filing_status(&self) -> FilingStatus {
        self.inner.filing_status.into()
    }

    /// The tax year.
    #[getter]
    fn tax_year(&self) -> u32 {
        self.inner.tax_year
    }

    /// The bracket index (0-based).
    #[getter]
    fn bracket_index(&self) -> u8 {
        self.inner.bracket_index
    }

    /// The marginal tax rate in basis points.
    #[getter]
    fn rate_bps(&self) -> u16 {
        self.inner.rate_bps
    }

    /// The tax rate as a percentage.
    #[getter]
    fn rate_percent(&self) -> f64 {
        self.inner.rate_bps as f64 / 100.0
    }

    /// Lower bound of the bracket.
    #[getter]
    fn bracket_lower(&self) -> u64 {
        self.inner.bracket_lower
    }

    /// Upper bound of the bracket.
    #[getter]
    fn bracket_upper(&self) -> u64 {
        self.inner.bracket_upper
    }

    /// The cryptographic commitment (hex string).
    #[getter]
    fn commitment(&self) -> String {
        self.inner.commitment.to_hex()
    }

    /// Verify this proof.
    ///
    /// Returns True if valid, raises ValueError if invalid.
    fn verify(&self) -> PyResult<bool> {
        self.inner.verify()
            .map(|_| true)
            .map_err(|e| PyValueError::new_err(format!("Proof verification failed: {}", e)))
    }

    /// Serialize to JSON string.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialization failed: {}", e)))
    }

    /// Deserialize from JSON string.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let inner: RustTaxBracketProof = serde_json::from_str(json)
            .map_err(|e| PyValueError::new_err(format!("Invalid JSON: {}", e)))?;
        Ok(TaxBracketProof { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "TaxBracketProof(jurisdiction={:?}, year={}, bracket={}, rate={}%)",
            self.inner.jurisdiction,
            self.inner.tax_year,
            self.inner.bracket_index,
            self.inner.rate_bps as f64 / 100.0
        )
    }
}

// =============================================================================
// Module Functions
// =============================================================================

/// Find the tax bracket for a given income.
///
/// Args:
///     income: Annual income in the jurisdiction's currency
///     jurisdiction: Tax jurisdiction (country)
///     year: Tax year (2020-2025)
///     filing_status: Filing status
///
/// Returns:
///     TaxBracket for the given income
///
/// Raises:
///     ValueError: If jurisdiction/year/status combination is unsupported
#[pyfunction]
fn find_bracket(
    income: u64,
    jurisdiction: Jurisdiction,
    year: u32,
    filing_status: FilingStatus,
) -> PyResult<TaxBracket> {
    rust_find_bracket(income, jurisdiction.into(), year, filing_status.into())
        .map(|b| TaxBracket::from(b))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Get all tax brackets for a jurisdiction.
///
/// Args:
///     jurisdiction: Tax jurisdiction (country)
///     year: Tax year (2020-2025)
///     filing_status: Filing status
///
/// Returns:
///     List of TaxBracket objects, sorted by lower bound
///
/// Raises:
///     ValueError: If jurisdiction/year/status combination is unsupported
#[pyfunction]
fn get_brackets(
    jurisdiction: Jurisdiction,
    year: u32,
    filing_status: FilingStatus,
) -> PyResult<Vec<TaxBracket>> {
    rust_get_brackets(jurisdiction.into(), year, filing_status.into())
        .map(|brackets| brackets.iter().map(TaxBracket::from).collect())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// List all supported jurisdictions.
///
/// Returns:
///     List of (code, name) tuples for all supported jurisdictions
#[pyfunction]
fn list_jurisdictions() -> Vec<(String, String)> {
    vec![
        ("US".to_string(), "United States".to_string()),
        ("UK".to_string(), "United Kingdom".to_string()),
        ("DE".to_string(), "Germany".to_string()),
        ("FR".to_string(), "France".to_string()),
        ("CA".to_string(), "Canada".to_string()),
        ("AU".to_string(), "Australia".to_string()),
        ("JP".to_string(), "Japan".to_string()),
        ("CN".to_string(), "China".to_string()),
        ("IN".to_string(), "India".to_string()),
        ("BR".to_string(), "Brazil".to_string()),
        ("MX".to_string(), "Mexico".to_string()),
        ("KR".to_string(), "South Korea".to_string()),
        ("IT".to_string(), "Italy".to_string()),
        ("RU".to_string(), "Russia".to_string()),
        ("TR".to_string(), "Turkey".to_string()),
        ("AR".to_string(), "Argentina".to_string()),
        ("ID".to_string(), "Indonesia".to_string()),
        ("SA".to_string(), "Saudi Arabia".to_string()),
        ("ZA".to_string(), "South Africa".to_string()),
        ("CL".to_string(), "Chile".to_string()),
        ("CO".to_string(), "Colombia".to_string()),
        ("PE".to_string(), "Peru".to_string()),
        ("EC".to_string(), "Ecuador".to_string()),
        ("UY".to_string(), "Uruguay".to_string()),
        ("ES".to_string(), "Spain".to_string()),
        ("NL".to_string(), "Netherlands".to_string()),
        ("BE".to_string(), "Belgium".to_string()),
        ("AT".to_string(), "Austria".to_string()),
        ("PT".to_string(), "Portugal".to_string()),
        ("IE".to_string(), "Ireland".to_string()),
        ("PL".to_string(), "Poland".to_string()),
        ("SE".to_string(), "Sweden".to_string()),
        ("DK".to_string(), "Denmark".to_string()),
        ("FI".to_string(), "Finland".to_string()),
        ("NO".to_string(), "Norway".to_string()),
        ("CH".to_string(), "Switzerland".to_string()),
        ("CZ".to_string(), "Czech Republic".to_string()),
        ("GR".to_string(), "Greece".to_string()),
        ("HU".to_string(), "Hungary".to_string()),
        ("RO".to_string(), "Romania".to_string()),
        ("UA".to_string(), "Ukraine".to_string()),
        ("NZ".to_string(), "New Zealand".to_string()),
        ("SG".to_string(), "Singapore".to_string()),
        ("HK".to_string(), "Hong Kong".to_string()),
        ("TW".to_string(), "Taiwan".to_string()),
        ("MY".to_string(), "Malaysia".to_string()),
        ("TH".to_string(), "Thailand".to_string()),
        ("VN".to_string(), "Vietnam".to_string()),
        ("PH".to_string(), "Philippines".to_string()),
        ("PK".to_string(), "Pakistan".to_string()),
        ("AE".to_string(), "UAE".to_string()),
        ("IL".to_string(), "Israel".to_string()),
        ("EG".to_string(), "Egypt".to_string()),
        ("QA".to_string(), "Qatar".to_string()),
        ("NG".to_string(), "Nigeria".to_string()),
        ("KE".to_string(), "Kenya".to_string()),
        ("MA".to_string(), "Morocco".to_string()),
        ("GH".to_string(), "Ghana".to_string()),
    ]
}

/// Get the SDK version.
#[pyfunction]
fn version() -> &'static str {
    mycelix_zk_tax::VERSION
}

// =============================================================================
// Module Definition
// =============================================================================

/// ZK Tax Python SDK - Zero-knowledge tax bracket proofs.
#[pymodule]
fn zk_tax(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Jurisdiction>()?;
    m.add_class::<FilingStatus>()?;
    m.add_class::<TaxBracket>()?;
    m.add_class::<TaxBracketProof>()?;

    m.add_function(wrap_pyfunction!(find_bracket, m)?)?;
    m.add_function(wrap_pyfunction!(get_brackets, m)?)?;
    m.add_function(wrap_pyfunction!(list_jurisdictions, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;

    Ok(())
}
