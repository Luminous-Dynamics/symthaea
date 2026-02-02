//! Jurisdiction and filing status definitions.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Supported tax jurisdictions (50+ countries).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Jurisdiction {
    // === Americas (10) ===
    /// United States (IRS)
    US,
    /// Canada (CRA)
    CA,
    /// Mexico (SAT)
    MX,
    /// Brazil (Receita Federal)
    BR,
    /// Argentina (AFIP)
    AR,
    /// Chile (SII)
    CL,
    /// Colombia (DIAN)
    CO,
    /// Peru (SUNAT)
    PE,
    /// Ecuador (SRI)
    EC,
    /// Uruguay (DGI)
    UY,

    // === Europe - EU (20) ===
    /// United Kingdom (HMRC)
    UK,
    /// Germany (Finanzamt)
    DE,
    /// France (DGFiP)
    FR,
    /// Italy (Agenzia delle Entrate)
    IT,
    /// Spain (AEAT)
    ES,
    /// Netherlands (Belastingdienst)
    NL,
    /// Belgium (SPF Finances)
    BE,
    /// Austria (BMF)
    AT,
    /// Portugal (AT)
    PT,
    /// Ireland (Revenue)
    IE,
    /// Poland (KAS)
    PL,
    /// Sweden (Skatteverket)
    SE,
    /// Denmark (Skattestyrelsen)
    DK,
    /// Finland (Vero)
    FI,
    /// Norway (Skatteetaten)
    NO,
    /// Switzerland (FTA)
    CH,
    /// Czech Republic (FS)
    CZ,
    /// Greece (AADE)
    GR,
    /// Hungary (NAV)
    HU,
    /// Romania (ANAF)
    RO,

    // === Europe - Non-EU (3) ===
    /// Russia (FNS)
    RU,
    /// Turkey (GIB)
    TR,
    /// Ukraine (DPS)
    UA,

    // === Asia-Pacific (15) ===
    /// Japan (NTA)
    JP,
    /// China (SAT)
    CN,
    /// India (IT Department)
    IN,
    /// South Korea (NTS)
    KR,
    /// Indonesia (DJP)
    ID,
    /// Australia (ATO)
    AU,
    /// New Zealand (IRD)
    NZ,
    /// Singapore (IRAS)
    SG,
    /// Hong Kong (IRD)
    HK,
    /// Taiwan (NTA)
    TW,
    /// Malaysia (LHDN)
    MY,
    /// Thailand (RD)
    TH,
    /// Vietnam (GDT)
    VN,
    /// Philippines (BIR)
    PH,
    /// Pakistan (FBR)
    PK,

    // === Middle East (5) ===
    /// Saudi Arabia (ZATCA)
    SA,
    /// United Arab Emirates (FTA)
    AE,
    /// Israel (ITA)
    IL,
    /// Egypt (ETA)
    EG,
    /// Qatar (GTA)
    QA,

    // === Africa (5) ===
    /// South Africa (SARS)
    ZA,
    /// Nigeria (FIRS)
    NG,
    /// Kenya (KRA)
    KE,
    /// Morocco (DGI)
    MA,
    /// Ghana (GRA)
    GH,
}

impl Jurisdiction {
    /// Parse jurisdiction from a country code string.
    /// Returns None if the code is not recognized.
    pub fn from_code(code: &str) -> Option<Self> {
        code.parse().ok()
    }

    /// Get the ISO 3166-1 alpha-2 country code.
    pub fn code(&self) -> &'static str {
        match self {
            // Americas
            Jurisdiction::US => "US",
            Jurisdiction::CA => "CA",
            Jurisdiction::MX => "MX",
            Jurisdiction::BR => "BR",
            Jurisdiction::AR => "AR",
            Jurisdiction::CL => "CL",
            Jurisdiction::CO => "CO",
            Jurisdiction::PE => "PE",
            Jurisdiction::EC => "EC",
            Jurisdiction::UY => "UY",
            // Europe - EU
            Jurisdiction::UK => "UK",
            Jurisdiction::DE => "DE",
            Jurisdiction::FR => "FR",
            Jurisdiction::IT => "IT",
            Jurisdiction::ES => "ES",
            Jurisdiction::NL => "NL",
            Jurisdiction::BE => "BE",
            Jurisdiction::AT => "AT",
            Jurisdiction::PT => "PT",
            Jurisdiction::IE => "IE",
            Jurisdiction::PL => "PL",
            Jurisdiction::SE => "SE",
            Jurisdiction::DK => "DK",
            Jurisdiction::FI => "FI",
            Jurisdiction::NO => "NO",
            Jurisdiction::CH => "CH",
            Jurisdiction::CZ => "CZ",
            Jurisdiction::GR => "GR",
            Jurisdiction::HU => "HU",
            Jurisdiction::RO => "RO",
            // Europe - Non-EU
            Jurisdiction::RU => "RU",
            Jurisdiction::TR => "TR",
            Jurisdiction::UA => "UA",
            // Asia-Pacific
            Jurisdiction::JP => "JP",
            Jurisdiction::CN => "CN",
            Jurisdiction::IN => "IN",
            Jurisdiction::KR => "KR",
            Jurisdiction::ID => "ID",
            Jurisdiction::AU => "AU",
            Jurisdiction::NZ => "NZ",
            Jurisdiction::SG => "SG",
            Jurisdiction::HK => "HK",
            Jurisdiction::TW => "TW",
            Jurisdiction::MY => "MY",
            Jurisdiction::TH => "TH",
            Jurisdiction::VN => "VN",
            Jurisdiction::PH => "PH",
            Jurisdiction::PK => "PK",
            // Middle East
            Jurisdiction::SA => "SA",
            Jurisdiction::AE => "AE",
            Jurisdiction::IL => "IL",
            Jurisdiction::EG => "EG",
            Jurisdiction::QA => "QA",
            // Africa
            Jurisdiction::ZA => "ZA",
            Jurisdiction::NG => "NG",
            Jurisdiction::KE => "KE",
            Jurisdiction::MA => "MA",
            Jurisdiction::GH => "GH",
        }
    }

    /// Get the display name for this jurisdiction.
    pub fn name(&self) -> &'static str {
        match self {
            // Americas
            Jurisdiction::US => "United States",
            Jurisdiction::CA => "Canada",
            Jurisdiction::MX => "Mexico",
            Jurisdiction::BR => "Brazil",
            Jurisdiction::AR => "Argentina",
            Jurisdiction::CL => "Chile",
            Jurisdiction::CO => "Colombia",
            Jurisdiction::PE => "Peru",
            Jurisdiction::EC => "Ecuador",
            Jurisdiction::UY => "Uruguay",
            // Europe - EU
            Jurisdiction::UK => "United Kingdom",
            Jurisdiction::DE => "Germany",
            Jurisdiction::FR => "France",
            Jurisdiction::IT => "Italy",
            Jurisdiction::ES => "Spain",
            Jurisdiction::NL => "Netherlands",
            Jurisdiction::BE => "Belgium",
            Jurisdiction::AT => "Austria",
            Jurisdiction::PT => "Portugal",
            Jurisdiction::IE => "Ireland",
            Jurisdiction::PL => "Poland",
            Jurisdiction::SE => "Sweden",
            Jurisdiction::DK => "Denmark",
            Jurisdiction::FI => "Finland",
            Jurisdiction::NO => "Norway",
            Jurisdiction::CH => "Switzerland",
            Jurisdiction::CZ => "Czech Republic",
            Jurisdiction::GR => "Greece",
            Jurisdiction::HU => "Hungary",
            Jurisdiction::RO => "Romania",
            // Europe - Non-EU
            Jurisdiction::RU => "Russia",
            Jurisdiction::TR => "Turkey",
            Jurisdiction::UA => "Ukraine",
            // Asia-Pacific
            Jurisdiction::JP => "Japan",
            Jurisdiction::CN => "China",
            Jurisdiction::IN => "India",
            Jurisdiction::KR => "South Korea",
            Jurisdiction::ID => "Indonesia",
            Jurisdiction::AU => "Australia",
            Jurisdiction::NZ => "New Zealand",
            Jurisdiction::SG => "Singapore",
            Jurisdiction::HK => "Hong Kong",
            Jurisdiction::TW => "Taiwan",
            Jurisdiction::MY => "Malaysia",
            Jurisdiction::TH => "Thailand",
            Jurisdiction::VN => "Vietnam",
            Jurisdiction::PH => "Philippines",
            Jurisdiction::PK => "Pakistan",
            // Middle East
            Jurisdiction::SA => "Saudi Arabia",
            Jurisdiction::AE => "United Arab Emirates",
            Jurisdiction::IL => "Israel",
            Jurisdiction::EG => "Egypt",
            Jurisdiction::QA => "Qatar",
            // Africa
            Jurisdiction::ZA => "South Africa",
            Jurisdiction::NG => "Nigeria",
            Jurisdiction::KE => "Kenya",
            Jurisdiction::MA => "Morocco",
            Jurisdiction::GH => "Ghana",
        }
    }

    /// Get the tax authority name.
    pub fn authority(&self) -> &'static str {
        match self {
            // Americas
            Jurisdiction::US => "IRS",
            Jurisdiction::CA => "CRA",
            Jurisdiction::MX => "SAT",
            Jurisdiction::BR => "Receita Federal",
            Jurisdiction::AR => "AFIP",
            Jurisdiction::CL => "SII",
            Jurisdiction::CO => "DIAN",
            Jurisdiction::PE => "SUNAT",
            Jurisdiction::EC => "SRI",
            Jurisdiction::UY => "DGI",
            // Europe - EU
            Jurisdiction::UK => "HMRC",
            Jurisdiction::DE => "Finanzamt",
            Jurisdiction::FR => "DGFiP",
            Jurisdiction::IT => "Agenzia Entrate",
            Jurisdiction::ES => "AEAT",
            Jurisdiction::NL => "Belastingdienst",
            Jurisdiction::BE => "SPF Finances",
            Jurisdiction::AT => "BMF",
            Jurisdiction::PT => "AT",
            Jurisdiction::IE => "Revenue",
            Jurisdiction::PL => "KAS",
            Jurisdiction::SE => "Skatteverket",
            Jurisdiction::DK => "Skattestyrelsen",
            Jurisdiction::FI => "Vero",
            Jurisdiction::NO => "Skatteetaten",
            Jurisdiction::CH => "FTA",
            Jurisdiction::CZ => "FS",
            Jurisdiction::GR => "AADE",
            Jurisdiction::HU => "NAV",
            Jurisdiction::RO => "ANAF",
            // Europe - Non-EU
            Jurisdiction::RU => "FNS",
            Jurisdiction::TR => "GIB",
            Jurisdiction::UA => "DPS",
            // Asia-Pacific
            Jurisdiction::JP => "NTA",
            Jurisdiction::CN => "SAT",
            Jurisdiction::IN => "IT Department",
            Jurisdiction::KR => "NTS",
            Jurisdiction::ID => "DJP",
            Jurisdiction::AU => "ATO",
            Jurisdiction::NZ => "IRD",
            Jurisdiction::SG => "IRAS",
            Jurisdiction::HK => "IRD",
            Jurisdiction::TW => "NTA",
            Jurisdiction::MY => "LHDN",
            Jurisdiction::TH => "RD",
            Jurisdiction::VN => "GDT",
            Jurisdiction::PH => "BIR",
            Jurisdiction::PK => "FBR",
            // Middle East
            Jurisdiction::SA => "ZATCA",
            Jurisdiction::AE => "FTA",
            Jurisdiction::IL => "ITA",
            Jurisdiction::EG => "ETA",
            Jurisdiction::QA => "GTA",
            // Africa
            Jurisdiction::ZA => "SARS",
            Jurisdiction::NG => "FIRS",
            Jurisdiction::KE => "KRA",
            Jurisdiction::MA => "DGI",
            Jurisdiction::GH => "GRA",
        }
    }

    /// Get the currency code for this jurisdiction.
    pub fn currency(&self) -> &'static str {
        match self {
            // Americas
            Jurisdiction::US => "USD",
            Jurisdiction::CA => "CAD",
            Jurisdiction::MX => "MXN",
            Jurisdiction::BR => "BRL",
            Jurisdiction::AR => "ARS",
            Jurisdiction::CL => "CLP",
            Jurisdiction::CO => "COP",
            Jurisdiction::PE => "PEN",
            Jurisdiction::EC => "USD",  // Ecuador uses USD
            Jurisdiction::UY => "UYU",
            // Europe - EUR zone
            Jurisdiction::DE => "EUR",
            Jurisdiction::FR => "EUR",
            Jurisdiction::IT => "EUR",
            Jurisdiction::ES => "EUR",
            Jurisdiction::NL => "EUR",
            Jurisdiction::BE => "EUR",
            Jurisdiction::AT => "EUR",
            Jurisdiction::PT => "EUR",
            Jurisdiction::IE => "EUR",
            Jurisdiction::FI => "EUR",
            Jurisdiction::GR => "EUR",
            // Europe - Non-EUR
            Jurisdiction::UK => "GBP",
            Jurisdiction::PL => "PLN",
            Jurisdiction::SE => "SEK",
            Jurisdiction::DK => "DKK",
            Jurisdiction::NO => "NOK",
            Jurisdiction::CH => "CHF",
            Jurisdiction::CZ => "CZK",
            Jurisdiction::HU => "HUF",
            Jurisdiction::RO => "RON",
            Jurisdiction::RU => "RUB",
            Jurisdiction::TR => "TRY",
            Jurisdiction::UA => "UAH",
            // Asia-Pacific
            Jurisdiction::JP => "JPY",
            Jurisdiction::CN => "CNY",
            Jurisdiction::IN => "INR",
            Jurisdiction::KR => "KRW",
            Jurisdiction::ID => "IDR",
            Jurisdiction::AU => "AUD",
            Jurisdiction::NZ => "NZD",
            Jurisdiction::SG => "SGD",
            Jurisdiction::HK => "HKD",
            Jurisdiction::TW => "TWD",
            Jurisdiction::MY => "MYR",
            Jurisdiction::TH => "THB",
            Jurisdiction::VN => "VND",
            Jurisdiction::PH => "PHP",
            Jurisdiction::PK => "PKR",
            // Middle East
            Jurisdiction::SA => "SAR",
            Jurisdiction::AE => "AED",
            Jurisdiction::IL => "ILS",
            Jurisdiction::EG => "EGP",
            Jurisdiction::QA => "QAR",
            // Africa
            Jurisdiction::ZA => "ZAR",
            Jurisdiction::NG => "NGN",
            Jurisdiction::KE => "KES",
            Jurisdiction::MA => "MAD",
            Jurisdiction::GH => "GHS",
        }
    }

    /// Get valid filing statuses for this jurisdiction.
    pub fn valid_filing_statuses(&self) -> &'static [FilingStatus] {
        match self {
            // US has multiple filing statuses
            Jurisdiction::US => &[
                FilingStatus::Single,
                FilingStatus::MarriedFilingJointly,
                FilingStatus::MarriedFilingSeparately,
                FilingStatus::HeadOfHousehold,
            ],
            // Germany has income splitting for married couples
            Jurisdiction::DE => &[
                FilingStatus::Single,
                FilingStatus::MarriedFilingJointly,
            ],
            // Most countries use individual taxation (all others)
            _ => &[FilingStatus::Single],
        }
    }

    /// Check if a filing status is valid for this jurisdiction.
    pub fn is_valid_filing_status(&self, status: FilingStatus) -> bool {
        self.valid_filing_statuses().contains(&status)
    }
}

impl fmt::Display for Jurisdiction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl std::str::FromStr for Jurisdiction {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            // Americas
            "US" | "USA" | "UNITED STATES" => Ok(Jurisdiction::US),
            "CA" | "CAN" | "CANADA" => Ok(Jurisdiction::CA),
            "MX" | "MEX" | "MEXICO" => Ok(Jurisdiction::MX),
            "BR" | "BRA" | "BRAZIL" | "BRASIL" => Ok(Jurisdiction::BR),
            "AR" | "ARG" | "ARGENTINA" => Ok(Jurisdiction::AR),
            "CL" | "CHL" | "CHILE" => Ok(Jurisdiction::CL),
            "CO" | "COL" | "COLOMBIA" => Ok(Jurisdiction::CO),
            "PE" | "PER" | "PERU" => Ok(Jurisdiction::PE),
            "EC" | "ECU" | "ECUADOR" => Ok(Jurisdiction::EC),
            "UY" | "URY" | "URUGUAY" => Ok(Jurisdiction::UY),
            // Europe - EU
            "UK" | "GB" | "GBR" | "UNITED KINGDOM" => Ok(Jurisdiction::UK),
            "DE" | "DEU" | "GERMANY" | "DEUTSCHLAND" => Ok(Jurisdiction::DE),
            "FR" | "FRA" | "FRANCE" => Ok(Jurisdiction::FR),
            "IT" | "ITA" | "ITALY" | "ITALIA" => Ok(Jurisdiction::IT),
            "ES" | "ESP" | "SPAIN" | "ESPANA" => Ok(Jurisdiction::ES),
            "NL" | "NLD" | "NETHERLANDS" | "HOLLAND" => Ok(Jurisdiction::NL),
            "BE" | "BEL" | "BELGIUM" => Ok(Jurisdiction::BE),
            "AT" | "AUT" | "AUSTRIA" => Ok(Jurisdiction::AT),
            "PT" | "PRT" | "PORTUGAL" => Ok(Jurisdiction::PT),
            "IE" | "IRL" | "IRELAND" => Ok(Jurisdiction::IE),
            "PL" | "POL" | "POLAND" => Ok(Jurisdiction::PL),
            "SE" | "SWE" | "SWEDEN" => Ok(Jurisdiction::SE),
            "DK" | "DNK" | "DENMARK" => Ok(Jurisdiction::DK),
            "FI" | "FIN" | "FINLAND" => Ok(Jurisdiction::FI),
            "NO" | "NOR" | "NORWAY" => Ok(Jurisdiction::NO),
            "CH" | "CHE" | "SWITZERLAND" => Ok(Jurisdiction::CH),
            "CZ" | "CZE" | "CZECH REPUBLIC" | "CZECHIA" => Ok(Jurisdiction::CZ),
            "GR" | "GRC" | "GREECE" => Ok(Jurisdiction::GR),
            "HU" | "HUN" | "HUNGARY" => Ok(Jurisdiction::HU),
            "RO" | "ROU" | "ROMANIA" => Ok(Jurisdiction::RO),
            // Europe - Non-EU
            "RU" | "RUS" | "RUSSIA" => Ok(Jurisdiction::RU),
            "TR" | "TUR" | "TURKEY" | "TURKIYE" => Ok(Jurisdiction::TR),
            "UA" | "UKR" | "UKRAINE" => Ok(Jurisdiction::UA),
            // Asia-Pacific
            "JP" | "JPN" | "JAPAN" => Ok(Jurisdiction::JP),
            "CN" | "CHN" | "CHINA" => Ok(Jurisdiction::CN),
            "IN" | "IND" | "INDIA" => Ok(Jurisdiction::IN),
            "KR" | "KOR" | "SOUTH KOREA" | "KOREA" => Ok(Jurisdiction::KR),
            "ID" | "IDN" | "INDONESIA" => Ok(Jurisdiction::ID),
            "AU" | "AUS" | "AUSTRALIA" => Ok(Jurisdiction::AU),
            "NZ" | "NZL" | "NEW ZEALAND" => Ok(Jurisdiction::NZ),
            "SG" | "SGP" | "SINGAPORE" => Ok(Jurisdiction::SG),
            "HK" | "HKG" | "HONG KONG" => Ok(Jurisdiction::HK),
            "TW" | "TWN" | "TAIWAN" => Ok(Jurisdiction::TW),
            "MY" | "MYS" | "MALAYSIA" => Ok(Jurisdiction::MY),
            "TH" | "THA" | "THAILAND" => Ok(Jurisdiction::TH),
            "VN" | "VNM" | "VIETNAM" => Ok(Jurisdiction::VN),
            "PH" | "PHL" | "PHILIPPINES" => Ok(Jurisdiction::PH),
            "PK" | "PAK" | "PAKISTAN" => Ok(Jurisdiction::PK),
            // Middle East
            "SA" | "SAU" | "SAUDI ARABIA" | "SAUDI" => Ok(Jurisdiction::SA),
            "AE" | "ARE" | "UAE" | "UNITED ARAB EMIRATES" => Ok(Jurisdiction::AE),
            "IL" | "ISR" | "ISRAEL" => Ok(Jurisdiction::IL),
            "EG" | "EGY" | "EGYPT" => Ok(Jurisdiction::EG),
            "QA" | "QAT" | "QATAR" => Ok(Jurisdiction::QA),
            // Africa
            "ZA" | "ZAF" | "SOUTH AFRICA" => Ok(Jurisdiction::ZA),
            "NG" | "NGA" | "NIGERIA" => Ok(Jurisdiction::NG),
            "KE" | "KEN" | "KENYA" => Ok(Jurisdiction::KE),
            "MA" | "MAR" | "MOROCCO" => Ok(Jurisdiction::MA),
            "GH" | "GHA" | "GHANA" => Ok(Jurisdiction::GH),
            _ => Err(format!(
                "Unknown jurisdiction: {}. Use ISO 2/3 letter country code (e.g., US, UK, DE, JP, CN)",
                s
            )),
        }
    }
}

/// Tax filing status.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FilingStatus {
    /// Single filer (US) or standard individual (UK)
    Single,
    /// Married filing jointly (US only)
    MarriedFilingJointly,
    /// Married filing separately (US only)
    MarriedFilingSeparately,
    /// Head of household (US only)
    HeadOfHousehold,
}

impl FilingStatus {
    /// Get the display name for this filing status.
    pub fn name(&self) -> &'static str {
        match self {
            FilingStatus::Single => "Single",
            FilingStatus::MarriedFilingJointly => "Married Filing Jointly",
            FilingStatus::MarriedFilingSeparately => "Married Filing Separately",
            FilingStatus::HeadOfHousehold => "Head of Household",
        }
    }

    /// Get the short code for this filing status.
    pub fn code(&self) -> &'static str {
        match self {
            FilingStatus::Single => "S",
            FilingStatus::MarriedFilingJointly => "MFJ",
            FilingStatus::MarriedFilingSeparately => "MFS",
            FilingStatus::HeadOfHousehold => "HoH",
        }
    }
}

impl fmt::Display for FilingStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl std::str::FromStr for FilingStatus {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().replace([' ', '_', '-'], "").as_str() {
            "single" | "s" => Ok(FilingStatus::Single),
            "marriedfilingjointly" | "mfj" | "joint" => Ok(FilingStatus::MarriedFilingJointly),
            "marriedfilingseparately" | "mfs" | "separate" => Ok(FilingStatus::MarriedFilingSeparately),
            "headofhousehold" | "hoh" | "head" => Ok(FilingStatus::HeadOfHousehold),
            _ => Err(format!("Unknown filing status: {}", s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jurisdiction_parsing() {
        assert_eq!("US".parse::<Jurisdiction>().unwrap(), Jurisdiction::US);
        assert_eq!("uk".parse::<Jurisdiction>().unwrap(), Jurisdiction::UK);
        assert_eq!("United States".parse::<Jurisdiction>().unwrap(), Jurisdiction::US);
    }

    #[test]
    fn test_filing_status_parsing() {
        assert_eq!("single".parse::<FilingStatus>().unwrap(), FilingStatus::Single);
        assert_eq!("MFJ".parse::<FilingStatus>().unwrap(), FilingStatus::MarriedFilingJointly);
        assert_eq!("head of household".parse::<FilingStatus>().unwrap(), FilingStatus::HeadOfHousehold);
    }

    #[test]
    fn test_valid_filing_statuses() {
        assert!(Jurisdiction::US.is_valid_filing_status(FilingStatus::HeadOfHousehold));
        assert!(!Jurisdiction::UK.is_valid_filing_status(FilingStatus::HeadOfHousehold));
    }
}
