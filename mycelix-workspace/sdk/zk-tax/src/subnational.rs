// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sub-national tax jurisdiction support.
//!
//! This module provides tax bracket data for US states and Canadian provinces
//! that have their own income tax systems.
//!
//! # Usage
//!
//! ```rust,ignore
//! use mycelix_zk_tax::subnational::{USState, CanadianProvince, get_state_brackets};
//!
//! // Get California state tax brackets
//! let brackets = get_state_brackets(USState::CA, 2024, FilingStatus::Single)?;
//! ```

use crate::{FilingStatus, TaxBracket};
use serde::{Deserialize, Serialize};
use std::fmt;

// =============================================================================
// US States
// =============================================================================

/// US States with income tax.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum USState {
    // === States with progressive income tax ===
    /// California - highest state income tax (1-13.3%)
    CA,
    /// New York - progressive (4-10.9%)
    NY,
    /// New Jersey - progressive (1.4-10.75%)
    NJ,
    /// Oregon - progressive (4.75-9.9%)
    OR,
    /// Minnesota - progressive (5.35-9.85%)
    MN,
    /// Hawaii - progressive (1.4-11%)
    HI,
    /// Vermont - progressive (3.35-8.75%)
    VT,
    /// Iowa - progressive (4.4-6%)
    IA,
    /// Wisconsin - progressive (3.54-7.65%)
    WI,
    /// Maine - progressive (5.8-7.15%)
    ME,

    // === States with flat income tax ===
    /// Illinois - 4.95% flat
    IL,
    /// Massachusetts - 5% flat (9% on short-term capital gains >$1M)
    MA,
    /// Michigan - 4.25% flat
    MI,
    /// Indiana - 3.05% flat
    IN,
    /// Pennsylvania - 3.07% flat
    PA,
    /// Utah - 4.65% flat
    UT,
    /// Colorado - 4.4% flat
    CO,
    /// North Carolina - 4.75% flat (2024)
    NC,
    /// Arizona - 2.5% flat (2024)
    AZ,
    /// Kentucky - 4% flat (2024)
    KY,

    // === States with NO income tax ===
    /// Texas - no income tax
    TX,
    /// Florida - no income tax
    FL,
    /// Washington - no income tax (but capital gains tax)
    WA,
    /// Nevada - no income tax
    NV,
    /// Wyoming - no income tax
    WY,
    /// South Dakota - no income tax
    SD,
    /// Alaska - no income tax
    AK,
    /// Tennessee - no income tax (as of 2021)
    TN,
    /// New Hampshire - no income tax (interest/dividends only until 2025)
    NH,
}

impl USState {
    /// Get the display name.
    pub fn name(&self) -> &'static str {
        match self {
            USState::CA => "California",
            USState::NY => "New York",
            USState::NJ => "New Jersey",
            USState::OR => "Oregon",
            USState::MN => "Minnesota",
            USState::HI => "Hawaii",
            USState::VT => "Vermont",
            USState::IA => "Iowa",
            USState::WI => "Wisconsin",
            USState::ME => "Maine",
            USState::IL => "Illinois",
            USState::MA => "Massachusetts",
            USState::MI => "Michigan",
            USState::IN => "Indiana",
            USState::PA => "Pennsylvania",
            USState::UT => "Utah",
            USState::CO => "Colorado",
            USState::NC => "North Carolina",
            USState::AZ => "Arizona",
            USState::KY => "Kentucky",
            USState::TX => "Texas",
            USState::FL => "Florida",
            USState::WA => "Washington",
            USState::NV => "Nevada",
            USState::WY => "Wyoming",
            USState::SD => "South Dakota",
            USState::AK => "Alaska",
            USState::TN => "Tennessee",
            USState::NH => "New Hampshire",
        }
    }

    /// Check if this state has income tax.
    pub fn has_income_tax(&self) -> bool {
        !matches!(
            self,
            USState::TX
                | USState::FL
                | USState::WA
                | USState::NV
                | USState::WY
                | USState::SD
                | USState::AK
                | USState::TN
                | USState::NH
        )
    }

    /// Check if this state has a flat tax rate.
    pub fn is_flat_tax(&self) -> bool {
        matches!(
            self,
            USState::IL
                | USState::MA
                | USState::MI
                | USState::IN
                | USState::PA
                | USState::UT
                | USState::CO
                | USState::NC
                | USState::AZ
                | USState::KY
        )
    }
}

impl fmt::Display for USState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl std::str::FromStr for USState {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "CA" | "CALIFORNIA" => Ok(USState::CA),
            "NY" | "NEW YORK" => Ok(USState::NY),
            "NJ" | "NEW JERSEY" => Ok(USState::NJ),
            "OR" | "OREGON" => Ok(USState::OR),
            "MN" | "MINNESOTA" => Ok(USState::MN),
            "HI" | "HAWAII" => Ok(USState::HI),
            "VT" | "VERMONT" => Ok(USState::VT),
            "IA" | "IOWA" => Ok(USState::IA),
            "WI" | "WISCONSIN" => Ok(USState::WI),
            "ME" | "MAINE" => Ok(USState::ME),
            "IL" | "ILLINOIS" => Ok(USState::IL),
            "MA" | "MASSACHUSETTS" => Ok(USState::MA),
            "MI" | "MICHIGAN" => Ok(USState::MI),
            "IN" | "INDIANA" => Ok(USState::IN),
            "PA" | "PENNSYLVANIA" => Ok(USState::PA),
            "UT" | "UTAH" => Ok(USState::UT),
            "CO" | "COLORADO" => Ok(USState::CO),
            "NC" | "NORTH CAROLINA" => Ok(USState::NC),
            "AZ" | "ARIZONA" => Ok(USState::AZ),
            "KY" | "KENTUCKY" => Ok(USState::KY),
            "TX" | "TEXAS" => Ok(USState::TX),
            "FL" | "FLORIDA" => Ok(USState::FL),
            "WA" | "WASHINGTON" => Ok(USState::WA),
            "NV" | "NEVADA" => Ok(USState::NV),
            "WY" | "WYOMING" => Ok(USState::WY),
            "SD" | "SOUTH DAKOTA" => Ok(USState::SD),
            "AK" | "ALASKA" => Ok(USState::AK),
            "TN" | "TENNESSEE" => Ok(USState::TN),
            "NH" | "NEW HAMPSHIRE" => Ok(USState::NH),
            _ => Err(format!("Unknown US state: {}", s)),
        }
    }
}

// =============================================================================
// Canadian Provinces
// =============================================================================

/// Canadian provinces and territories.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CanadianProvince {
    /// Ontario - largest province
    ON,
    /// Quebec - highest provincial tax
    QC,
    /// British Columbia
    BC,
    /// Alberta - flat 10%
    AB,
    /// Manitoba
    MB,
    /// Saskatchewan
    SK,
    /// Nova Scotia
    NS,
    /// New Brunswick
    NB,
    /// Newfoundland and Labrador
    NL,
    /// Prince Edward Island
    PE,
    /// Northwest Territories
    NT,
    /// Yukon
    YT,
    /// Nunavut
    NU,
}

impl CanadianProvince {
    /// Get the display name.
    pub fn name(&self) -> &'static str {
        match self {
            CanadianProvince::ON => "Ontario",
            CanadianProvince::QC => "Quebec",
            CanadianProvince::BC => "British Columbia",
            CanadianProvince::AB => "Alberta",
            CanadianProvince::MB => "Manitoba",
            CanadianProvince::SK => "Saskatchewan",
            CanadianProvince::NS => "Nova Scotia",
            CanadianProvince::NB => "New Brunswick",
            CanadianProvince::NL => "Newfoundland and Labrador",
            CanadianProvince::PE => "Prince Edward Island",
            CanadianProvince::NT => "Northwest Territories",
            CanadianProvince::YT => "Yukon",
            CanadianProvince::NU => "Nunavut",
        }
    }

    /// Check if this province has a flat tax rate.
    pub fn is_flat_tax(&self) -> bool {
        matches!(self, CanadianProvince::AB)
    }
}

impl fmt::Display for CanadianProvince {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl std::str::FromStr for CanadianProvince {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "ON" | "ONTARIO" => Ok(CanadianProvince::ON),
            "QC" | "QUEBEC" => Ok(CanadianProvince::QC),
            "BC" | "BRITISH COLUMBIA" => Ok(CanadianProvince::BC),
            "AB" | "ALBERTA" => Ok(CanadianProvince::AB),
            "MB" | "MANITOBA" => Ok(CanadianProvince::MB),
            "SK" | "SASKATCHEWAN" => Ok(CanadianProvince::SK),
            "NS" | "NOVA SCOTIA" => Ok(CanadianProvince::NS),
            "NB" | "NEW BRUNSWICK" => Ok(CanadianProvince::NB),
            "NL" | "NEWFOUNDLAND" | "NEWFOUNDLAND AND LABRADOR" => Ok(CanadianProvince::NL),
            "PE" | "PEI" | "PRINCE EDWARD ISLAND" => Ok(CanadianProvince::PE),
            "NT" | "NORTHWEST TERRITORIES" => Ok(CanadianProvince::NT),
            "YT" | "YUKON" => Ok(CanadianProvince::YT),
            "NU" | "NUNAVUT" => Ok(CanadianProvince::NU),
            _ => Err(format!("Unknown Canadian province: {}", s)),
        }
    }
}

// =============================================================================
// US State Tax Brackets (2024)
// =============================================================================

/// California 2024 Single brackets (10 brackets, 1-13.3%)
pub static CA_SINGLE_2024: [TaxBracket; 9] = [
    TaxBracket { index: 0, lower: 0,       upper: 10_412,   rate_bps: 100 },   // 1%
    TaxBracket { index: 1, lower: 10_412,  upper: 24_684,   rate_bps: 200 },   // 2%
    TaxBracket { index: 2, lower: 24_684,  upper: 38_959,   rate_bps: 400 },   // 4%
    TaxBracket { index: 3, lower: 38_959,  upper: 54_081,   rate_bps: 600 },   // 6%
    TaxBracket { index: 4, lower: 54_081,  upper: 68_350,   rate_bps: 800 },   // 8%
    TaxBracket { index: 5, lower: 68_350,  upper: 349_137,  rate_bps: 930 },   // 9.3%
    TaxBracket { index: 6, lower: 349_137, upper: 418_961,  rate_bps: 1030 },  // 10.3%
    TaxBracket { index: 7, lower: 418_961, upper: 698_271,  rate_bps: 1130 },  // 11.3%
    TaxBracket { index: 8, lower: 698_271, upper: u64::MAX, rate_bps: 1330 },  // 13.3%
];

/// New York 2024 Single brackets (8 brackets, 4-10.9%)
pub static NY_SINGLE_2024: [TaxBracket; 8] = [
    TaxBracket { index: 0, lower: 0,        upper: 8_500,     rate_bps: 400 },   // 4%
    TaxBracket { index: 1, lower: 8_500,    upper: 11_700,    rate_bps: 450 },   // 4.5%
    TaxBracket { index: 2, lower: 11_700,   upper: 13_900,    rate_bps: 525 },   // 5.25%
    TaxBracket { index: 3, lower: 13_900,   upper: 80_650,    rate_bps: 550 },   // 5.5%
    TaxBracket { index: 4, lower: 80_650,   upper: 215_400,   rate_bps: 600 },   // 6%
    TaxBracket { index: 5, lower: 215_400,  upper: 1_077_550, rate_bps: 685 },   // 6.85%
    TaxBracket { index: 6, lower: 1_077_550,upper: 5_000_000, rate_bps: 965 },   // 9.65%
    TaxBracket { index: 7, lower: 5_000_000,upper: u64::MAX,  rate_bps: 1090 },  // 10.9%
];

/// Texas 2024 - No income tax (single bracket at 0%)
pub static TX_2024: [TaxBracket; 1] = [
    TaxBracket { index: 0, lower: 0, upper: u64::MAX, rate_bps: 0 },
];

/// Florida 2024 - No income tax
pub static FL_2024: [TaxBracket; 1] = [
    TaxBracket { index: 0, lower: 0, upper: u64::MAX, rate_bps: 0 },
];

/// Illinois 2024 - Flat 4.95%
pub static IL_2024: [TaxBracket; 1] = [
    TaxBracket { index: 0, lower: 0, upper: u64::MAX, rate_bps: 495 },
];

/// Massachusetts 2024 - Flat 5%
pub static MA_2024: [TaxBracket; 1] = [
    TaxBracket { index: 0, lower: 0, upper: u64::MAX, rate_bps: 500 },
];

/// Colorado 2024 - Flat 4.4%
pub static CO_2024: [TaxBracket; 1] = [
    TaxBracket { index: 0, lower: 0, upper: u64::MAX, rate_bps: 440 },
];

// =============================================================================
// Canadian Provincial Tax Brackets (2024)
// =============================================================================

/// Ontario 2024 brackets (5 brackets, 5.05-13.16%)
pub static ON_2024: [TaxBracket; 5] = [
    TaxBracket { index: 0, lower: 0,       upper: 51_446,  rate_bps: 505 },   // 5.05%
    TaxBracket { index: 1, lower: 51_446,  upper: 102_894, rate_bps: 915 },   // 9.15%
    TaxBracket { index: 2, lower: 102_894, upper: 150_000, rate_bps: 1116 },  // 11.16%
    TaxBracket { index: 3, lower: 150_000, upper: 220_000, rate_bps: 1216 },  // 12.16%
    TaxBracket { index: 4, lower: 220_000, upper: u64::MAX, rate_bps: 1316 }, // 13.16%
];

/// Quebec 2024 brackets (4 brackets, 14-25.75%)
pub static QC_2024: [TaxBracket; 4] = [
    TaxBracket { index: 0, lower: 0,       upper: 51_780,  rate_bps: 1400 },  // 14%
    TaxBracket { index: 1, lower: 51_780,  upper: 103_545, rate_bps: 1900 },  // 19%
    TaxBracket { index: 2, lower: 103_545, upper: 126_000, rate_bps: 2400 },  // 24%
    TaxBracket { index: 3, lower: 126_000, upper: u64::MAX, rate_bps: 2575 }, // 25.75%
];

/// British Columbia 2024 brackets (7 brackets, 5.06-20.5%)
pub static BC_2024: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,       upper: 47_937,  rate_bps: 506 },   // 5.06%
    TaxBracket { index: 1, lower: 47_937,  upper: 95_875,  rate_bps: 770 },   // 7.7%
    TaxBracket { index: 2, lower: 95_875,  upper: 110_076, rate_bps: 1050 },  // 10.5%
    TaxBracket { index: 3, lower: 110_076, upper: 133_664, rate_bps: 1229 },  // 12.29%
    TaxBracket { index: 4, lower: 133_664, upper: 181_232, rate_bps: 1470 },  // 14.7%
    TaxBracket { index: 5, lower: 181_232, upper: 252_752, rate_bps: 1680 },  // 16.8%
    TaxBracket { index: 6, lower: 252_752, upper: u64::MAX, rate_bps: 2050 }, // 20.5%
];

/// Alberta 2024 - Flat 10%
pub static AB_2024: [TaxBracket; 1] = [
    TaxBracket { index: 0, lower: 0, upper: u64::MAX, rate_bps: 1000 },
];

// =============================================================================
// German Länder (Federal States)
// =============================================================================

/// German federal states (Länder).
/// Note: Germany doesn't have state income tax, but church tax varies by state.
/// The church tax (Kirchensteuer) is 8% in Bavaria/Baden-Württemberg, 9% elsewhere.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GermanLand {
    /// Bavaria (Bayern) - 8% church tax
    BY,
    /// Baden-Württemberg - 8% church tax
    BW,
    /// Berlin - 9% church tax
    BE,
    /// Brandenburg - 9% church tax
    BB,
    /// Bremen - 9% church tax
    HB,
    /// Hamburg - 9% church tax
    HH,
    /// Hesse (Hessen) - 9% church tax
    HE,
    /// Lower Saxony (Niedersachsen) - 9% church tax
    NI,
    /// Mecklenburg-Vorpommern - 9% church tax
    MV,
    /// North Rhine-Westphalia (Nordrhein-Westfalen) - 9% church tax
    NW,
    /// Rhineland-Palatinate (Rheinland-Pfalz) - 9% church tax
    RP,
    /// Saarland - 9% church tax
    SL,
    /// Saxony (Sachsen) - 9% church tax
    SN,
    /// Saxony-Anhalt (Sachsen-Anhalt) - 9% church tax
    ST,
    /// Schleswig-Holstein - 9% church tax
    SH,
    /// Thuringia (Thüringen) - 9% church tax
    TH,
}

impl GermanLand {
    /// Get display name.
    pub fn name(&self) -> &'static str {
        match self {
            GermanLand::BY => "Bavaria",
            GermanLand::BW => "Baden-Württemberg",
            GermanLand::BE => "Berlin",
            GermanLand::BB => "Brandenburg",
            GermanLand::HB => "Bremen",
            GermanLand::HH => "Hamburg",
            GermanLand::HE => "Hesse",
            GermanLand::NI => "Lower Saxony",
            GermanLand::MV => "Mecklenburg-Vorpommern",
            GermanLand::NW => "North Rhine-Westphalia",
            GermanLand::RP => "Rhineland-Palatinate",
            GermanLand::SL => "Saarland",
            GermanLand::SN => "Saxony",
            GermanLand::ST => "Saxony-Anhalt",
            GermanLand::SH => "Schleswig-Holstein",
            GermanLand::TH => "Thuringia",
        }
    }

    /// Get church tax rate in basis points (applied to income tax, not income).
    /// 800 = 8%, 900 = 9%
    pub fn church_tax_rate_bps(&self) -> u16 {
        match self {
            GermanLand::BY | GermanLand::BW => 800, // 8%
            _ => 900, // 9% in all other states
        }
    }
}

impl fmt::Display for GermanLand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl std::str::FromStr for GermanLand {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "BY" | "BAVARIA" | "BAYERN" => Ok(GermanLand::BY),
            "BW" | "BADEN-WÜRTTEMBERG" | "BADEN-WUERTTEMBERG" => Ok(GermanLand::BW),
            "BE" | "BERLIN" => Ok(GermanLand::BE),
            "BB" | "BRANDENBURG" => Ok(GermanLand::BB),
            "HB" | "BREMEN" => Ok(GermanLand::HB),
            "HH" | "HAMBURG" => Ok(GermanLand::HH),
            "HE" | "HESSE" | "HESSEN" => Ok(GermanLand::HE),
            "NI" | "LOWER SAXONY" | "NIEDERSACHSEN" => Ok(GermanLand::NI),
            "MV" | "MECKLENBURG-VORPOMMERN" => Ok(GermanLand::MV),
            "NW" | "NRW" | "NORTH RHINE-WESTPHALIA" | "NORDRHEIN-WESTFALEN" => Ok(GermanLand::NW),
            "RP" | "RHINELAND-PALATINATE" | "RHEINLAND-PFALZ" => Ok(GermanLand::RP),
            "SL" | "SAARLAND" => Ok(GermanLand::SL),
            "SN" | "SAXONY" | "SACHSEN" => Ok(GermanLand::SN),
            "ST" | "SAXONY-ANHALT" | "SACHSEN-ANHALT" => Ok(GermanLand::ST),
            "SH" | "SCHLESWIG-HOLSTEIN" => Ok(GermanLand::SH),
            "TH" | "THURINGIA" | "THÜRINGEN" | "THUERINGEN" => Ok(GermanLand::TH),
            _ => Err(format!("Unknown German state: {}", s)),
        }
    }
}

// =============================================================================
// Australian States and Territories
// =============================================================================

/// Australian states and territories.
/// Note: Australia uses federal income tax only, but Medicare levy varies
/// and some states have payroll tax that affects employers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AustralianState {
    /// New South Wales
    NSW,
    /// Victoria
    VIC,
    /// Queensland
    QLD,
    /// Western Australia
    WA,
    /// South Australia
    SA,
    /// Tasmania
    TAS,
    /// Australian Capital Territory
    ACT,
    /// Northern Territory
    NT,
}

impl AustralianState {
    /// Get display name.
    pub fn name(&self) -> &'static str {
        match self {
            AustralianState::NSW => "New South Wales",
            AustralianState::VIC => "Victoria",
            AustralianState::QLD => "Queensland",
            AustralianState::WA => "Western Australia",
            AustralianState::SA => "South Australia",
            AustralianState::TAS => "Tasmania",
            AustralianState::ACT => "Australian Capital Territory",
            AustralianState::NT => "Northern Territory",
        }
    }
}

impl fmt::Display for AustralianState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl std::str::FromStr for AustralianState {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "NSW" | "NEW SOUTH WALES" => Ok(AustralianState::NSW),
            "VIC" | "VICTORIA" => Ok(AustralianState::VIC),
            "QLD" | "QUEENSLAND" => Ok(AustralianState::QLD),
            "WA" | "WESTERN AUSTRALIA" => Ok(AustralianState::WA),
            "SA" | "SOUTH AUSTRALIA" => Ok(AustralianState::SA),
            "TAS" | "TASMANIA" => Ok(AustralianState::TAS),
            "ACT" | "AUSTRALIAN CAPITAL TERRITORY" => Ok(AustralianState::ACT),
            "NT" | "NORTHERN TERRITORY" => Ok(AustralianState::NT),
            _ => Err(format!("Unknown Australian state: {}", s)),
        }
    }
}

// =============================================================================
// Indian States and Union Territories
// =============================================================================

/// Indian states with professional tax.
/// Note: India has federal income tax plus state-level Professional Tax (PT)
/// which varies by state (max ₹2,500/year).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IndianState {
    /// Maharashtra (Mumbai) - has Professional Tax
    MH,
    /// Karnataka (Bangalore) - has Professional Tax
    KA,
    /// Tamil Nadu (Chennai) - has Professional Tax
    TN,
    /// Gujarat - has Professional Tax
    GJ,
    /// Andhra Pradesh - has Professional Tax
    AP,
    /// Telangana (Hyderabad) - has Professional Tax
    TS,
    /// West Bengal (Kolkata) - has Professional Tax
    WB,
    /// Kerala - has Professional Tax
    KL,
    /// Madhya Pradesh - has Professional Tax
    MP,
    /// Delhi NCT - no Professional Tax
    DL,
    /// Uttar Pradesh - no Professional Tax
    UP,
    /// Rajasthan - no Professional Tax
    RJ,
    /// Punjab - no Professional Tax
    PB,
    /// Haryana - no Professional Tax
    HR,
}

impl IndianState {
    /// Get display name.
    pub fn name(&self) -> &'static str {
        match self {
            IndianState::MH => "Maharashtra",
            IndianState::KA => "Karnataka",
            IndianState::TN => "Tamil Nadu",
            IndianState::GJ => "Gujarat",
            IndianState::AP => "Andhra Pradesh",
            IndianState::TS => "Telangana",
            IndianState::WB => "West Bengal",
            IndianState::KL => "Kerala",
            IndianState::MP => "Madhya Pradesh",
            IndianState::DL => "Delhi",
            IndianState::UP => "Uttar Pradesh",
            IndianState::RJ => "Rajasthan",
            IndianState::PB => "Punjab",
            IndianState::HR => "Haryana",
        }
    }

    /// Check if this state levies Professional Tax.
    pub fn has_professional_tax(&self) -> bool {
        matches!(
            self,
            IndianState::MH
                | IndianState::KA
                | IndianState::TN
                | IndianState::GJ
                | IndianState::AP
                | IndianState::TS
                | IndianState::WB
                | IndianState::KL
                | IndianState::MP
        )
    }

    /// Get max annual Professional Tax in INR (if applicable).
    /// Most states cap at ₹2,500/year.
    pub fn max_professional_tax_inr(&self) -> u64 {
        if self.has_professional_tax() {
            2_500
        } else {
            0
        }
    }
}

impl fmt::Display for IndianState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl std::str::FromStr for IndianState {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "MH" | "MAHARASHTRA" | "MUMBAI" => Ok(IndianState::MH),
            "KA" | "KARNATAKA" | "BANGALORE" | "BENGALURU" => Ok(IndianState::KA),
            "TN" | "TAMIL NADU" | "CHENNAI" => Ok(IndianState::TN),
            "GJ" | "GUJARAT" | "AHMEDABAD" => Ok(IndianState::GJ),
            "AP" | "ANDHRA PRADESH" => Ok(IndianState::AP),
            "TS" | "TELANGANA" | "HYDERABAD" => Ok(IndianState::TS),
            "WB" | "WEST BENGAL" | "KOLKATA" => Ok(IndianState::WB),
            "KL" | "KERALA" => Ok(IndianState::KL),
            "MP" | "MADHYA PRADESH" => Ok(IndianState::MP),
            "DL" | "DELHI" => Ok(IndianState::DL),
            "UP" | "UTTAR PRADESH" => Ok(IndianState::UP),
            "RJ" | "RAJASTHAN" => Ok(IndianState::RJ),
            "PB" | "PUNJAB" => Ok(IndianState::PB),
            "HR" | "HARYANA" => Ok(IndianState::HR),
            _ => Err(format!("Unknown Indian state: {}", s)),
        }
    }
}

// =============================================================================
// Brazilian States
// =============================================================================

/// Brazilian states (Estados) and Federal District.
/// Brazil uses federal income tax (IRPF), but states have ICMS (sales tax)
/// and some municipal taxes (ISS) that vary by location.
/// This enum enables state-specific proof generation for combined tax analysis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BrazilianState {
    /// Acre
    AC,
    /// Alagoas
    AL,
    /// Amapá
    AP,
    /// Amazonas
    AM,
    /// Bahia
    BA,
    /// Ceará
    CE,
    /// Distrito Federal (Federal District/Brasília)
    DF,
    /// Espírito Santo
    ES,
    /// Goiás
    GO,
    /// Maranhão
    MA,
    /// Mato Grosso
    MT,
    /// Mato Grosso do Sul
    MS,
    /// Minas Gerais
    MG,
    /// Pará
    PA,
    /// Paraíba
    PB,
    /// Paraná
    PR,
    /// Pernambuco
    PE,
    /// Piauí
    PI,
    /// Rio de Janeiro
    RJ,
    /// Rio Grande do Norte
    RN,
    /// Rio Grande do Sul
    RS,
    /// Rondônia
    RO,
    /// Roraima
    RR,
    /// Santa Catarina
    SC,
    /// São Paulo
    SP,
    /// Sergipe
    SE,
    /// Tocantins
    TO,
}

impl BrazilianState {
    /// Get display name.
    pub fn name(&self) -> &'static str {
        match self {
            BrazilianState::AC => "Acre",
            BrazilianState::AL => "Alagoas",
            BrazilianState::AP => "Amapá",
            BrazilianState::AM => "Amazonas",
            BrazilianState::BA => "Bahia",
            BrazilianState::CE => "Ceará",
            BrazilianState::DF => "Distrito Federal",
            BrazilianState::ES => "Espírito Santo",
            BrazilianState::GO => "Goiás",
            BrazilianState::MA => "Maranhão",
            BrazilianState::MT => "Mato Grosso",
            BrazilianState::MS => "Mato Grosso do Sul",
            BrazilianState::MG => "Minas Gerais",
            BrazilianState::PA => "Pará",
            BrazilianState::PB => "Paraíba",
            BrazilianState::PR => "Paraná",
            BrazilianState::PE => "Pernambuco",
            BrazilianState::PI => "Piauí",
            BrazilianState::RJ => "Rio de Janeiro",
            BrazilianState::RN => "Rio Grande do Norte",
            BrazilianState::RS => "Rio Grande do Sul",
            BrazilianState::RO => "Rondônia",
            BrazilianState::RR => "Roraima",
            BrazilianState::SC => "Santa Catarina",
            BrazilianState::SP => "São Paulo",
            BrazilianState::SE => "Sergipe",
            BrazilianState::TO => "Tocantins",
        }
    }

    /// Get the region of Brazil this state belongs to.
    pub fn region(&self) -> &'static str {
        match self {
            BrazilianState::AC | BrazilianState::AP | BrazilianState::AM |
            BrazilianState::PA | BrazilianState::RO | BrazilianState::RR |
            BrazilianState::TO => "Norte",
            BrazilianState::AL | BrazilianState::BA | BrazilianState::CE |
            BrazilianState::MA | BrazilianState::PB | BrazilianState::PE |
            BrazilianState::PI | BrazilianState::RN | BrazilianState::SE => "Nordeste",
            BrazilianState::DF | BrazilianState::GO | BrazilianState::MT |
            BrazilianState::MS => "Centro-Oeste",
            BrazilianState::ES | BrazilianState::MG | BrazilianState::RJ |
            BrazilianState::SP => "Sudeste",
            BrazilianState::PR | BrazilianState::RS | BrazilianState::SC => "Sul",
        }
    }

    /// Check if this is a major economic center (higher cost of living).
    pub fn is_major_economic_center(&self) -> bool {
        matches!(
            self,
            BrazilianState::SP | BrazilianState::RJ | BrazilianState::DF |
            BrazilianState::MG | BrazilianState::RS | BrazilianState::PR
        )
    }
}

impl fmt::Display for BrazilianState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl std::str::FromStr for BrazilianState {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "AC" | "ACRE" => Ok(BrazilianState::AC),
            "AL" | "ALAGOAS" => Ok(BrazilianState::AL),
            "AP" | "AMAPÁ" | "AMAPA" => Ok(BrazilianState::AP),
            "AM" | "AMAZONAS" => Ok(BrazilianState::AM),
            "BA" | "BAHIA" => Ok(BrazilianState::BA),
            "CE" | "CEARÁ" | "CEARA" => Ok(BrazilianState::CE),
            "DF" | "DISTRITO FEDERAL" | "BRASÍLIA" | "BRASILIA" => Ok(BrazilianState::DF),
            "ES" | "ESPÍRITO SANTO" | "ESPIRITO SANTO" => Ok(BrazilianState::ES),
            "GO" | "GOIÁS" | "GOIAS" => Ok(BrazilianState::GO),
            "MA" | "MARANHÃO" | "MARANHAO" => Ok(BrazilianState::MA),
            "MT" | "MATO GROSSO" => Ok(BrazilianState::MT),
            "MS" | "MATO GROSSO DO SUL" => Ok(BrazilianState::MS),
            "MG" | "MINAS GERAIS" => Ok(BrazilianState::MG),
            "PA" | "PARÁ" | "PARA" => Ok(BrazilianState::PA),
            "PB" | "PARAÍBA" | "PARAIBA" => Ok(BrazilianState::PB),
            "PR" | "PARANÁ" | "PARANA" => Ok(BrazilianState::PR),
            "PE" | "PERNAMBUCO" => Ok(BrazilianState::PE),
            "PI" | "PIAUÍ" | "PIAUI" => Ok(BrazilianState::PI),
            "RJ" | "RIO DE JANEIRO" => Ok(BrazilianState::RJ),
            "RN" | "RIO GRANDE DO NORTE" => Ok(BrazilianState::RN),
            "RS" | "RIO GRANDE DO SUL" => Ok(BrazilianState::RS),
            "RO" | "RONDÔNIA" | "RONDONIA" => Ok(BrazilianState::RO),
            "RR" | "RORAIMA" => Ok(BrazilianState::RR),
            "SC" | "SANTA CATARINA" => Ok(BrazilianState::SC),
            "SP" | "SÃO PAULO" | "SAO PAULO" => Ok(BrazilianState::SP),
            "SE" | "SERGIPE" => Ok(BrazilianState::SE),
            "TO" | "TOCANTINS" => Ok(BrazilianState::TO),
            _ => Err(format!("Unknown Brazilian state: {}", s)),
        }
    }
}

// =============================================================================
// Brazilian Federal Income Tax (IRPF) Brackets
// Note: Brazil uses federal income tax with state/municipal surcharges
// Values in BRL (Brazilian Reais)
// =============================================================================

/// Brazil IRPF 2024 federal income tax brackets (monthly values)
/// Based on MP 1.172/2023 updated rates
pub static BR_IRPF_2024_MONTHLY: [TaxBracket; 5] = [
    TaxBracket { index: 0, lower: 0,        upper: 2_259,   rate_bps: 0 },     // Exempt
    TaxBracket { index: 1, lower: 2_259,    upper: 2_826,   rate_bps: 750 },   // 7.5%
    TaxBracket { index: 2, lower: 2_826,    upper: 3_751,   rate_bps: 1500 },  // 15%
    TaxBracket { index: 3, lower: 3_751,    upper: 4_664,   rate_bps: 2250 },  // 22.5%
    TaxBracket { index: 4, lower: 4_664,    upper: u64::MAX, rate_bps: 2750 }, // 27.5%
];

/// Brazil IRPF 2024 annual brackets (monthly * 12)
pub static BR_IRPF_2024_ANNUAL: [TaxBracket; 5] = [
    TaxBracket { index: 0, lower: 0,         upper: 27_108,   rate_bps: 0 },     // Exempt
    TaxBracket { index: 1, lower: 27_108,    upper: 33_912,   rate_bps: 750 },   // 7.5%
    TaxBracket { index: 2, lower: 33_912,    upper: 45_012,   rate_bps: 1500 },  // 15%
    TaxBracket { index: 3, lower: 45_012,    upper: 55_968,   rate_bps: 2250 },  // 22.5%
    TaxBracket { index: 4, lower: 55_968,    upper: u64::MAX, rate_bps: 2750 },  // 27.5%
];

/// São Paulo additional considerations (ISS rates for services)
/// Standard ISS ranges from 2-5% depending on service type
pub static SP_ISS_2024: [TaxBracket; 3] = [
    TaxBracket { index: 0, lower: 0, upper: u64::MAX, rate_bps: 200 },  // 2% (min ISS)
    TaxBracket { index: 1, lower: 0, upper: u64::MAX, rate_bps: 350 },  // 3.5% (common)
    TaxBracket { index: 2, lower: 0, upper: u64::MAX, rate_bps: 500 },  // 5% (max ISS)
];

// =============================================================================
// UK Nations (Scotland has different tax bands)
// =============================================================================

/// UK constituent nations/regions for tax purposes.
/// Scotland sets its own income tax rates since 2017.
/// England, Wales, and Northern Ireland use the UK-wide rates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UKNation {
    /// England - uses UK-wide rates
    England,
    /// Wales - uses UK-wide rates
    Wales,
    /// Scotland - has its own income tax bands since 2017
    Scotland,
    /// Northern Ireland - uses UK-wide rates
    NorthernIreland,
}

impl UKNation {
    /// Get display name.
    pub fn name(&self) -> &'static str {
        match self {
            UKNation::England => "England",
            UKNation::Wales => "Wales",
            UKNation::Scotland => "Scotland",
            UKNation::NorthernIreland => "Northern Ireland",
        }
    }

    /// Check if this nation has devolved income tax powers.
    pub fn has_devolved_tax(&self) -> bool {
        matches!(self, UKNation::Scotland)
    }
}

impl fmt::Display for UKNation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl std::str::FromStr for UKNation {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "ENGLAND" | "ENG" => Ok(UKNation::England),
            "WALES" | "WLS" | "CYMRU" => Ok(UKNation::Wales),
            "SCOTLAND" | "SCO" | "ALBA" => Ok(UKNation::Scotland),
            "NORTHERN IRELAND" | "NI" | "NIR" => Ok(UKNation::NorthernIreland),
            _ => Err(format!("Unknown UK nation: {}", s)),
        }
    }
}

// =============================================================================
// UK Scotland Tax Brackets (2024-25)
// Scotland has 6 bands compared to UK's 3 main bands
// =============================================================================

/// Scotland 2024-25 tax bands (6 bands vs UK's 3)
/// More progressive than rest of UK with additional starter and intermediate rates
pub static SCOTLAND_2024: [TaxBracket; 6] = [
    TaxBracket { index: 0, lower: 0,        upper: 12_570,  rate_bps: 0 },     // Personal Allowance (0%)
    TaxBracket { index: 1, lower: 12_570,   upper: 14_876,  rate_bps: 1900 },  // Starter rate (19%)
    TaxBracket { index: 2, lower: 14_876,   upper: 26_561,  rate_bps: 2000 },  // Basic rate (20%)
    TaxBracket { index: 3, lower: 26_561,   upper: 43_662,  rate_bps: 2100 },  // Intermediate rate (21%)
    TaxBracket { index: 4, lower: 43_662,   upper: 75_000,  rate_bps: 4200 },  // Higher rate (42%)
    TaxBracket { index: 5, lower: 75_000,   upper: u64::MAX, rate_bps: 4700 }, // Top rate (47%)
];

/// Scotland 2025-26 projected rates (based on announced changes)
/// Advanced rate added at 45% for £75,000-£125,140
pub static SCOTLAND_2025: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,         upper: 12_570,   rate_bps: 0 },     // Personal Allowance (0%)
    TaxBracket { index: 1, lower: 12_570,    upper: 15_397,   rate_bps: 1900 },  // Starter rate (19%)
    TaxBracket { index: 2, lower: 15_397,    upper: 27_491,   rate_bps: 2000 },  // Basic rate (20%)
    TaxBracket { index: 3, lower: 27_491,    upper: 43_662,   rate_bps: 2100 },  // Intermediate rate (21%)
    TaxBracket { index: 4, lower: 43_662,    upper: 75_000,   rate_bps: 4200 },  // Higher rate (42%)
    TaxBracket { index: 5, lower: 75_000,    upper: 125_140,  rate_bps: 4500 },  // Advanced rate (45%)
    TaxBracket { index: 6, lower: 125_140,   upper: u64::MAX, rate_bps: 4800 },  // Top rate (48%)
];

/// UK (rest of UK excluding Scotland) 2024-25 tax bands for comparison
pub static UK_REST_2024: [TaxBracket; 4] = [
    TaxBracket { index: 0, lower: 0,        upper: 12_570,  rate_bps: 0 },     // Personal Allowance (0%)
    TaxBracket { index: 1, lower: 12_570,   upper: 50_270,  rate_bps: 2000 },  // Basic rate (20%)
    TaxBracket { index: 2, lower: 50_270,   upper: 125_140, rate_bps: 4000 },  // Higher rate (40%)
    TaxBracket { index: 3, lower: 125_140,  upper: u64::MAX, rate_bps: 4500 }, // Additional rate (45%)
];

// =============================================================================
// Swiss Cantons
// =============================================================================

/// Swiss cantons (Kantone).
/// Switzerland has 26 cantons, each with their own cantonal tax rates.
/// Combined with federal tax (11.5% max) and municipal tax.
/// Rates shown are cantonal base rates before municipal multipliers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SwissCanton {
    /// Zürich - financial center
    ZH,
    /// Bern - capital canton
    BE,
    /// Luzern - central Switzerland
    LU,
    /// Uri - original canton
    UR,
    /// Schwyz - low tax canton
    SZ,
    /// Obwalden - lowest tax canton
    OW,
    /// Nidwalden - very low tax
    NW,
    /// Glarus - eastern Switzerland
    GL,
    /// Zug - very low tax canton (crypto hub)
    ZG,
    /// Fribourg/Freiburg - bilingual canton
    FR,
    /// Solothurn - northwestern Switzerland
    SO,
    /// Basel-Stadt - city canton
    BS,
    /// Basel-Landschaft - Basel area
    BL,
    /// Schaffhausen - northern Switzerland
    SH,
    /// Appenzell Ausserrhoden
    AR,
    /// Appenzell Innerrhoden - smallest canton
    AI,
    /// St. Gallen - eastern Switzerland
    SG,
    /// Graubünden/Grisons - largest canton
    GR,
    /// Aargau - northern Switzerland
    AG,
    /// Thurgau - northeastern Switzerland
    TG,
    /// Ticino/Tessin - Italian-speaking
    TI,
    /// Vaud/Waadt - French-speaking, Lake Geneva
    VD,
    /// Valais/Wallis - Alpine canton
    VS,
    /// Neuchâtel - French-speaking
    NE,
    /// Genève/Geneva - international hub
    GE,
    /// Jura - newest canton (1979)
    JU,
}

impl SwissCanton {
    /// Get display name (German/French/Italian).
    pub fn name(&self) -> &'static str {
        match self {
            SwissCanton::ZH => "Zürich",
            SwissCanton::BE => "Bern",
            SwissCanton::LU => "Luzern",
            SwissCanton::UR => "Uri",
            SwissCanton::SZ => "Schwyz",
            SwissCanton::OW => "Obwalden",
            SwissCanton::NW => "Nidwalden",
            SwissCanton::GL => "Glarus",
            SwissCanton::ZG => "Zug",
            SwissCanton::FR => "Fribourg",
            SwissCanton::SO => "Solothurn",
            SwissCanton::BS => "Basel-Stadt",
            SwissCanton::BL => "Basel-Landschaft",
            SwissCanton::SH => "Schaffhausen",
            SwissCanton::AR => "Appenzell Ausserrhoden",
            SwissCanton::AI => "Appenzell Innerrhoden",
            SwissCanton::SG => "St. Gallen",
            SwissCanton::GR => "Graubünden",
            SwissCanton::AG => "Aargau",
            SwissCanton::TG => "Thurgau",
            SwissCanton::TI => "Ticino",
            SwissCanton::VD => "Vaud",
            SwissCanton::VS => "Valais",
            SwissCanton::NE => "Neuchâtel",
            SwissCanton::GE => "Geneva",
            SwissCanton::JU => "Jura",
        }
    }

    /// Check if this canton is considered "low tax" (< 25% combined max rate).
    pub fn is_low_tax(&self) -> bool {
        matches!(
            self,
            SwissCanton::ZG
                | SwissCanton::SZ
                | SwissCanton::OW
                | SwissCanton::NW
                | SwissCanton::UR
                | SwissCanton::AI
                | SwissCanton::AR
        )
    }

    /// Get the main language of the canton.
    pub fn language(&self) -> &'static str {
        match self {
            SwissCanton::TI => "Italian",
            SwissCanton::GE | SwissCanton::VD | SwissCanton::NE | SwissCanton::JU => "French",
            SwissCanton::FR | SwissCanton::VS | SwissCanton::BE => "German/French",
            SwissCanton::GR => "German/Romansh/Italian",
            _ => "German",
        }
    }
}

impl fmt::Display for SwissCanton {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl std::str::FromStr for SwissCanton {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "ZH" | "ZÜRICH" | "ZURICH" | "ZUERICH" => Ok(SwissCanton::ZH),
            "BE" | "BERN" | "BERNE" => Ok(SwissCanton::BE),
            "LU" | "LUZERN" | "LUCERNE" => Ok(SwissCanton::LU),
            "UR" | "URI" => Ok(SwissCanton::UR),
            "SZ" | "SCHWYZ" => Ok(SwissCanton::SZ),
            "OW" | "OBWALDEN" => Ok(SwissCanton::OW),
            "NW" | "NIDWALDEN" => Ok(SwissCanton::NW),
            "GL" | "GLARUS" => Ok(SwissCanton::GL),
            "ZG" | "ZUG" => Ok(SwissCanton::ZG),
            "FR" | "FRIBOURG" | "FREIBURG" => Ok(SwissCanton::FR),
            "SO" | "SOLOTHURN" => Ok(SwissCanton::SO),
            "BS" | "BASEL-STADT" | "BASEL STADT" | "BASEL" => Ok(SwissCanton::BS),
            "BL" | "BASEL-LANDSCHAFT" | "BASEL LANDSCHAFT" | "BASELLAND" => Ok(SwissCanton::BL),
            "SH" | "SCHAFFHAUSEN" => Ok(SwissCanton::SH),
            "AR" | "APPENZELL AUSSERRHODEN" | "AUSSERRHODEN" => Ok(SwissCanton::AR),
            "AI" | "APPENZELL INNERRHODEN" | "INNERRHODEN" => Ok(SwissCanton::AI),
            "SG" | "ST. GALLEN" | "ST GALLEN" | "SANKT GALLEN" => Ok(SwissCanton::SG),
            "GR" | "GRAUBÜNDEN" | "GRAUBUENDEN" | "GRISONS" => Ok(SwissCanton::GR),
            "AG" | "AARGAU" => Ok(SwissCanton::AG),
            "TG" | "THURGAU" => Ok(SwissCanton::TG),
            "TI" | "TICINO" | "TESSIN" => Ok(SwissCanton::TI),
            "VD" | "VAUD" | "WAADT" => Ok(SwissCanton::VD),
            "VS" | "VALAIS" | "WALLIS" => Ok(SwissCanton::VS),
            "NE" | "NEUCHÂTEL" | "NEUCHATEL" | "NEUENBURG" => Ok(SwissCanton::NE),
            "GE" | "GENEVA" | "GENÈVE" | "GENEVE" | "GENF" => Ok(SwissCanton::GE),
            "JU" | "JURA" => Ok(SwissCanton::JU),
            _ => Err(format!("Unknown Swiss canton: {}", s)),
        }
    }
}

// =============================================================================
// Swiss Cantonal Tax Brackets (2024)
// Note: These are simplified cantonal base rates. Actual rates depend on:
// - Municipal multiplier (Steuerfuss)
// - Church membership
// - Married/Single status
// Values in CHF.
// =============================================================================

/// Zürich 2024 cantonal brackets (simplified, Single, main city)
/// Actual rates vary by municipality (Steuerfuss typically 100-130%)
pub static ZH_SINGLE_2024: [TaxBracket; 8] = [
    TaxBracket { index: 0, lower: 0,        upper: 6_700,    rate_bps: 0 },     // 0% (deduction)
    TaxBracket { index: 1, lower: 6_700,    upper: 11_400,   rate_bps: 200 },   // 2%
    TaxBracket { index: 2, lower: 11_400,   upper: 16_100,   rate_bps: 300 },   // 3%
    TaxBracket { index: 3, lower: 16_100,   upper: 23_700,   rate_bps: 400 },   // 4%
    TaxBracket { index: 4, lower: 23_700,   upper: 33_000,   rate_bps: 500 },   // 5%
    TaxBracket { index: 5, lower: 33_000,   upper: 43_700,   rate_bps: 600 },   // 6%
    TaxBracket { index: 6, lower: 43_700,   upper: 79_100,   rate_bps: 700 },   // 7%
    TaxBracket { index: 7, lower: 79_100,   upper: u64::MAX, rate_bps: 800 },   // 8% (cantonal max)
];

/// Zug 2024 - famous low-tax canton (crypto-friendly)
pub static ZG_SINGLE_2024: [TaxBracket; 5] = [
    TaxBracket { index: 0, lower: 0,        upper: 16_000,   rate_bps: 0 },     // 0% (deduction)
    TaxBracket { index: 1, lower: 16_000,   upper: 28_000,   rate_bps: 100 },   // 1%
    TaxBracket { index: 2, lower: 28_000,   upper: 56_000,   rate_bps: 200 },   // 2%
    TaxBracket { index: 3, lower: 56_000,   upper: 112_000,  rate_bps: 300 },   // 3%
    TaxBracket { index: 4, lower: 112_000,  upper: u64::MAX, rate_bps: 400 },   // 4% (cantonal max!)
];

/// Geneva 2024 - highest tax canton but international hub
pub static GE_SINGLE_2024: [TaxBracket; 9] = [
    TaxBracket { index: 0, lower: 0,        upper: 17_493,   rate_bps: 0 },     // 0% (deduction)
    TaxBracket { index: 1, lower: 17_493,   upper: 21_076,   rate_bps: 800 },   // 8%
    TaxBracket { index: 2, lower: 21_076,   upper: 23_184,   rate_bps: 900 },   // 9%
    TaxBracket { index: 3, lower: 23_184,   upper: 25_291,   rate_bps: 1000 },  // 10%
    TaxBracket { index: 4, lower: 25_291,   upper: 27_399,   rate_bps: 1100 },  // 11%
    TaxBracket { index: 5, lower: 27_399,   upper: 38_019,   rate_bps: 1200 },  // 12%
    TaxBracket { index: 6, lower: 38_019,   upper: 59_259,   rate_bps: 1400 },  // 14%
    TaxBracket { index: 7, lower: 59_259,   upper: 119_080,  rate_bps: 1550 },  // 15.5%
    TaxBracket { index: 8, lower: 119_080,  upper: u64::MAX, rate_bps: 1900 },  // 19% (cantonal max)
];

/// Vaud 2024 - Lake Geneva region
pub static VD_SINGLE_2024: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,        upper: 14_500,   rate_bps: 0 },     // 0% (deduction)
    TaxBracket { index: 1, lower: 14_500,   upper: 31_600,   rate_bps: 230 },   // 2.3%
    TaxBracket { index: 2, lower: 31_600,   upper: 41_300,   rate_bps: 330 },   // 3.3%
    TaxBracket { index: 3, lower: 41_300,   upper: 56_000,   rate_bps: 430 },   // 4.3%
    TaxBracket { index: 4, lower: 56_000,   upper: 103_300,  rate_bps: 630 },   // 6.3%
    TaxBracket { index: 5, lower: 103_300,  upper: 166_800,  rate_bps: 1030 },  // 10.3%
    TaxBracket { index: 6, lower: 166_800,  upper: u64::MAX, rate_bps: 1150 },  // 11.5% (cantonal max)
];

/// Bern 2024 - capital region
pub static BE_SINGLE_2024: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,        upper: 16_600,   rate_bps: 0 },     // 0% (deduction)
    TaxBracket { index: 1, lower: 16_600,   upper: 30_000,   rate_bps: 200 },   // 2%
    TaxBracket { index: 2, lower: 30_000,   upper: 40_000,   rate_bps: 350 },   // 3.5%
    TaxBracket { index: 3, lower: 40_000,   upper: 60_000,   rate_bps: 450 },   // 4.5%
    TaxBracket { index: 4, lower: 60_000,   upper: 100_000,  rate_bps: 550 },   // 5.5%
    TaxBracket { index: 5, lower: 100_000,  upper: 200_000,  rate_bps: 650 },   // 6.5%
    TaxBracket { index: 6, lower: 200_000,  upper: u64::MAX, rate_bps: 680 },   // 6.8% (cantonal max)
];

/// Basel-Stadt 2024 - city canton
pub static BS_SINGLE_2024: [TaxBracket; 6] = [
    TaxBracket { index: 0, lower: 0,        upper: 11_000,   rate_bps: 0 },     // 0% (deduction)
    TaxBracket { index: 1, lower: 11_000,   upper: 50_000,   rate_bps: 225 },   // 2.25%
    TaxBracket { index: 2, lower: 50_000,   upper: 100_000,  rate_bps: 260 },   // 2.6%
    TaxBracket { index: 3, lower: 100_000,  upper: 150_000,  rate_bps: 270 },   // 2.7%
    TaxBracket { index: 4, lower: 150_000,  upper: 200_000,  rate_bps: 280 },   // 2.8%
    TaxBracket { index: 5, lower: 200_000,  upper: u64::MAX, rate_bps: 295 },   // 2.95% (unusually low!)
];

/// Schwyz 2024 - low tax canton
pub static SZ_SINGLE_2024: [TaxBracket; 5] = [
    TaxBracket { index: 0, lower: 0,        upper: 15_000,   rate_bps: 0 },     // 0% (deduction)
    TaxBracket { index: 1, lower: 15_000,   upper: 30_000,   rate_bps: 100 },   // 1%
    TaxBracket { index: 2, lower: 30_000,   upper: 70_000,   rate_bps: 200 },   // 2%
    TaxBracket { index: 3, lower: 70_000,   upper: 150_000,  rate_bps: 300 },   // 3%
    TaxBracket { index: 4, lower: 150_000,  upper: u64::MAX, rate_bps: 400 },   // 4% (cantonal max)
];

/// Obwalden 2024 - lowest tax canton in Switzerland
pub static OW_SINGLE_2024: [TaxBracket; 4] = [
    TaxBracket { index: 0, lower: 0,        upper: 10_000,   rate_bps: 0 },     // 0% (deduction)
    TaxBracket { index: 1, lower: 10_000,   upper: 30_000,   rate_bps: 50 },    // 0.5%
    TaxBracket { index: 2, lower: 30_000,   upper: 100_000,  rate_bps: 120 },   // 1.2%
    TaxBracket { index: 3, lower: 100_000,  upper: u64::MAX, rate_bps: 180 },   // 1.8% (lowest in CH!)
];

/// Ticino 2024 - Italian-speaking canton
pub static TI_SINGLE_2024: [TaxBracket; 7] = [
    TaxBracket { index: 0, lower: 0,        upper: 13_000,   rate_bps: 0 },     // 0% (deduction)
    TaxBracket { index: 1, lower: 13_000,   upper: 25_000,   rate_bps: 200 },   // 2%
    TaxBracket { index: 2, lower: 25_000,   upper: 40_000,   rate_bps: 400 },   // 4%
    TaxBracket { index: 3, lower: 40_000,   upper: 60_000,   rate_bps: 600 },   // 6%
    TaxBracket { index: 4, lower: 60_000,   upper: 100_000,  rate_bps: 800 },   // 8%
    TaxBracket { index: 5, lower: 100_000,  upper: 200_000,  rate_bps: 1000 },  // 10%
    TaxBracket { index: 6, lower: 200_000,  upper: u64::MAX, rate_bps: 1150 },  // 11.5%
];

// =============================================================================
// Indian State Professional Tax Brackets
// =============================================================================

/// Maharashtra Professional Tax brackets (monthly salary slabs)
pub static MH_PT_2024: [TaxBracket; 4] = [
    TaxBracket { index: 0, lower: 0,        upper: 7_500,   rate_bps: 0 },     // Exempt
    TaxBracket { index: 1, lower: 7_500,    upper: 10_000,  rate_bps: 0 },     // ₹175/month (approx 2%)
    TaxBracket { index: 2, lower: 10_000,   upper: u64::MAX, rate_bps: 0 },    // ₹200-300/month
    TaxBracket { index: 3, lower: 0, upper: 0, rate_bps: 0 },                  // Placeholder
];

/// Karnataka Professional Tax brackets
pub static KA_PT_2024: [TaxBracket; 3] = [
    TaxBracket { index: 0, lower: 0,        upper: 15_000,  rate_bps: 0 },     // Exempt
    TaxBracket { index: 1, lower: 15_000,   upper: u64::MAX, rate_bps: 0 },    // ₹200/month (max ₹2,500/year)
    TaxBracket { index: 2, lower: 0, upper: 0, rate_bps: 0 },                  // Placeholder
];

// =============================================================================
// Bracket Lookup Functions
// =============================================================================

/// Get tax brackets for a US state.
pub fn get_state_brackets(
    state: USState,
    year: u32,
    status: FilingStatus,
) -> Result<&'static [TaxBracket], String> {
    // Only single supported for now (MFJ would double the brackets)
    if status != FilingStatus::Single {
        return Err(format!(
            "Only Single filing status supported for US states currently. Got: {:?}",
            status
        ));
    }

    match (state, year) {
        (USState::CA, 2024) => Ok(&CA_SINGLE_2024),
        (USState::CA, 2025) => Ok(&CA_SINGLE_2024), // Use 2024 as estimate
        (USState::NY, 2024) => Ok(&NY_SINGLE_2024),
        (USState::NY, 2025) => Ok(&NY_SINGLE_2024),
        (USState::TX, 2024 | 2025) => Ok(&TX_2024),
        (USState::FL, 2024 | 2025) => Ok(&FL_2024),
        (USState::IL, 2024 | 2025) => Ok(&IL_2024),
        (USState::MA, 2024 | 2025) => Ok(&MA_2024),
        (USState::CO, 2024 | 2025) => Ok(&CO_2024),
        // States without income tax
        (USState::WA | USState::NV | USState::WY | USState::SD | USState::AK | USState::TN | USState::NH, _) => {
            Ok(&TX_2024) // Use 0% brackets
        }
        _ => Err(format!(
            "Tax brackets not yet implemented for {} in {}",
            state, year
        )),
    }
}

/// Get tax brackets for a Canadian province.
pub fn get_provincial_brackets(
    province: CanadianProvince,
    year: u32,
    _status: FilingStatus,
) -> Result<&'static [TaxBracket], String> {
    match (province, year) {
        (CanadianProvince::ON, 2024 | 2025) => Ok(&ON_2024),
        (CanadianProvince::QC, 2024 | 2025) => Ok(&QC_2024),
        (CanadianProvince::BC, 2024 | 2025) => Ok(&BC_2024),
        (CanadianProvince::AB, 2024 | 2025) => Ok(&AB_2024),
        _ => Err(format!(
            "Tax brackets not yet implemented for {} in {}",
            province, year
        )),
    }
}

/// Get Brazilian federal IRPF tax brackets.
/// All Brazilian states use the same federal income tax rates.
pub fn get_brazilian_brackets(
    _state: BrazilianState,
    year: u32,
    _status: FilingStatus,
    monthly: bool,
) -> Result<&'static [TaxBracket], String> {
    match (year, monthly) {
        (2024 | 2025, true) => Ok(&BR_IRPF_2024_MONTHLY),
        (2024 | 2025, false) => Ok(&BR_IRPF_2024_ANNUAL),
        _ => Err(format!("Brazilian tax brackets not yet implemented for {}", year)),
    }
}

/// Find the federal IRPF bracket for a given income in Brazil.
pub fn find_brazilian_bracket(
    income: u64,
    state: BrazilianState,
    year: u32,
    status: FilingStatus,
    monthly: bool,
) -> Result<TaxBracket, String> {
    let brackets = get_brazilian_brackets(state, year, status, monthly)?;

    for bracket in brackets {
        if income >= bracket.lower && income < bracket.upper {
            return Ok(*bracket);
        }
    }

    Err(format!("No bracket found for income {} in {}", income, state))
}

/// Get tax brackets for a UK nation (Scotland has different rates).
pub fn get_uk_nation_brackets(
    nation: UKNation,
    year: u32,
    _status: FilingStatus,
) -> Result<&'static [TaxBracket], String> {
    match (nation, year) {
        (UKNation::Scotland, 2024) => Ok(&SCOTLAND_2024),
        (UKNation::Scotland, 2025) => Ok(&SCOTLAND_2025),
        (UKNation::England | UKNation::Wales | UKNation::NorthernIreland, 2024 | 2025) => Ok(&UK_REST_2024),
        _ => Err(format!(
            "Tax brackets not yet implemented for {} in {}",
            nation, year
        )),
    }
}

/// Find the bracket for a given income in a UK nation.
pub fn find_uk_nation_bracket(
    income: u64,
    nation: UKNation,
    year: u32,
    status: FilingStatus,
) -> Result<TaxBracket, String> {
    let brackets = get_uk_nation_brackets(nation, year, status)?;

    for bracket in brackets {
        if income >= bracket.lower && income < bracket.upper {
            return Ok(*bracket);
        }
    }

    Err(format!("No bracket found for income {} in {}", income, nation))
}

/// Get cantonal tax brackets for a Swiss canton.
pub fn get_cantonal_brackets(
    canton: SwissCanton,
    year: u32,
    _status: FilingStatus,
) -> Result<&'static [TaxBracket], String> {
    match (canton, year) {
        (SwissCanton::ZH, 2024 | 2025) => Ok(&ZH_SINGLE_2024),
        (SwissCanton::ZG, 2024 | 2025) => Ok(&ZG_SINGLE_2024),
        (SwissCanton::GE, 2024 | 2025) => Ok(&GE_SINGLE_2024),
        (SwissCanton::VD, 2024 | 2025) => Ok(&VD_SINGLE_2024),
        (SwissCanton::BE, 2024 | 2025) => Ok(&BE_SINGLE_2024),
        (SwissCanton::BS, 2024 | 2025) => Ok(&BS_SINGLE_2024),
        (SwissCanton::SZ, 2024 | 2025) => Ok(&SZ_SINGLE_2024),
        (SwissCanton::OW, 2024 | 2025) => Ok(&OW_SINGLE_2024),
        (SwissCanton::TI, 2024 | 2025) => Ok(&TI_SINGLE_2024),
        _ => Err(format!(
            "Tax brackets not yet implemented for {} in {}. Available: ZH, ZG, GE, VD, BE, BS, SZ, OW, TI",
            canton, year
        )),
    }
}

/// Find the cantonal bracket for a given income in a Swiss canton.
pub fn find_cantonal_bracket(
    income: u64,
    canton: SwissCanton,
    year: u32,
    status: FilingStatus,
) -> Result<TaxBracket, String> {
    let brackets = get_cantonal_brackets(canton, year, status)?;

    for bracket in brackets {
        if income >= bracket.lower && income < bracket.upper {
            return Ok(*bracket);
        }
    }

    Err(format!("No bracket found for income {} in {}", income, canton))
}

/// Find the bracket for a given income in a US state.
pub fn find_state_bracket(
    income: u64,
    state: USState,
    year: u32,
    status: FilingStatus,
) -> Result<TaxBracket, String> {
    let brackets = get_state_brackets(state, year, status)?;

    for bracket in brackets {
        if income >= bracket.lower && income < bracket.upper {
            return Ok(*bracket);
        }
    }

    // Should never happen with proper brackets
    Err(format!("No bracket found for income {} in {}", income, state))
}

/// Find the bracket for a given income in a Canadian province.
pub fn find_provincial_bracket(
    income: u64,
    province: CanadianProvince,
    year: u32,
    status: FilingStatus,
) -> Result<TaxBracket, String> {
    let brackets = get_provincial_brackets(province, year, status)?;

    for bracket in brackets {
        if income >= bracket.lower && income < bracket.upper {
            return Ok(*bracket);
        }
    }

    Err(format!("No bracket found for income {} in {}", income, province))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_california_brackets() {
        // $100,000 income should be in the 9.3% bracket
        let bracket = find_state_bracket(100_000, USState::CA, 2024, FilingStatus::Single).unwrap();
        assert_eq!(bracket.index, 5);
        assert_eq!(bracket.rate_bps, 930);
    }

    #[test]
    fn test_new_york_brackets() {
        // $150,000 income should be in the 6% bracket
        let bracket = find_state_bracket(150_000, USState::NY, 2024, FilingStatus::Single).unwrap();
        assert_eq!(bracket.index, 4);
        assert_eq!(bracket.rate_bps, 600);
    }

    #[test]
    fn test_texas_no_tax() {
        // Texas has no income tax
        let bracket = find_state_bracket(1_000_000, USState::TX, 2024, FilingStatus::Single).unwrap();
        assert_eq!(bracket.rate_bps, 0);
    }

    #[test]
    fn test_ontario_brackets() {
        // $100,000 CAD should be in the 9.15% bracket
        let bracket = find_provincial_bracket(100_000, CanadianProvince::ON, 2024, FilingStatus::Single).unwrap();
        assert_eq!(bracket.index, 1);
        assert_eq!(bracket.rate_bps, 915);
    }

    #[test]
    fn test_quebec_brackets() {
        // $80,000 CAD should be in the 19% bracket
        let bracket = find_provincial_bracket(80_000, CanadianProvince::QC, 2024, FilingStatus::Single).unwrap();
        assert_eq!(bracket.index, 1);
        assert_eq!(bracket.rate_bps, 1900);
    }

    #[test]
    fn test_alberta_flat_tax() {
        // Alberta has flat 10%
        let bracket = find_provincial_bracket(500_000, CanadianProvince::AB, 2024, FilingStatus::Single).unwrap();
        assert_eq!(bracket.rate_bps, 1000);
    }

    #[test]
    fn test_us_state_parsing() {
        assert_eq!("CA".parse::<USState>().unwrap(), USState::CA);
        assert_eq!("california".parse::<USState>().unwrap(), USState::CA);
        assert_eq!("NY".parse::<USState>().unwrap(), USState::NY);
        assert_eq!("TEXAS".parse::<USState>().unwrap(), USState::TX);
    }

    #[test]
    fn test_canadian_province_parsing() {
        assert_eq!("ON".parse::<CanadianProvince>().unwrap(), CanadianProvince::ON);
        assert_eq!("ontario".parse::<CanadianProvince>().unwrap(), CanadianProvince::ON);
        assert_eq!("QC".parse::<CanadianProvince>().unwrap(), CanadianProvince::QC);
        assert_eq!("BC".parse::<CanadianProvince>().unwrap(), CanadianProvince::BC);
    }

    #[test]
    fn test_has_income_tax() {
        assert!(USState::CA.has_income_tax());
        assert!(USState::NY.has_income_tax());
        assert!(!USState::TX.has_income_tax());
        assert!(!USState::FL.has_income_tax());
        assert!(!USState::WA.has_income_tax());
    }

    #[test]
    fn test_is_flat_tax() {
        assert!(USState::IL.is_flat_tax());
        assert!(USState::MA.is_flat_tax());
        assert!(USState::CO.is_flat_tax());
        assert!(!USState::CA.is_flat_tax());
        assert!(!USState::NY.is_flat_tax());
    }

    // =======================================================================
    // German Länder Tests
    // =======================================================================

    #[test]
    fn test_german_land_parsing() {
        assert_eq!("BY".parse::<GermanLand>().unwrap(), GermanLand::BY);
        assert_eq!("bavaria".parse::<GermanLand>().unwrap(), GermanLand::BY);
        assert_eq!("BAYERN".parse::<GermanLand>().unwrap(), GermanLand::BY);
        assert_eq!("NW".parse::<GermanLand>().unwrap(), GermanLand::NW);
        assert_eq!("NRW".parse::<GermanLand>().unwrap(), GermanLand::NW);
    }

    #[test]
    fn test_german_church_tax_rates() {
        // Bavaria and Baden-Württemberg have 8% church tax
        assert_eq!(GermanLand::BY.church_tax_rate_bps(), 800);
        assert_eq!(GermanLand::BW.church_tax_rate_bps(), 800);

        // All other states have 9%
        assert_eq!(GermanLand::BE.church_tax_rate_bps(), 900);
        assert_eq!(GermanLand::NW.church_tax_rate_bps(), 900);
        assert_eq!(GermanLand::HE.church_tax_rate_bps(), 900);
    }

    // =======================================================================
    // Australian State Tests
    // =======================================================================

    #[test]
    fn test_australian_state_parsing() {
        assert_eq!("NSW".parse::<AustralianState>().unwrap(), AustralianState::NSW);
        assert_eq!("new south wales".parse::<AustralianState>().unwrap(), AustralianState::NSW);
        assert_eq!("VIC".parse::<AustralianState>().unwrap(), AustralianState::VIC);
        assert_eq!("QLD".parse::<AustralianState>().unwrap(), AustralianState::QLD);
    }

    #[test]
    fn test_australian_state_names() {
        assert_eq!(AustralianState::NSW.name(), "New South Wales");
        assert_eq!(AustralianState::VIC.name(), "Victoria");
        assert_eq!(AustralianState::ACT.name(), "Australian Capital Territory");
    }

    // =======================================================================
    // Indian State Tests
    // =======================================================================

    #[test]
    fn test_indian_state_parsing() {
        assert_eq!("MH".parse::<IndianState>().unwrap(), IndianState::MH);
        assert_eq!("maharashtra".parse::<IndianState>().unwrap(), IndianState::MH);
        assert_eq!("MUMBAI".parse::<IndianState>().unwrap(), IndianState::MH);
        assert_eq!("KA".parse::<IndianState>().unwrap(), IndianState::KA);
        assert_eq!("Bangalore".parse::<IndianState>().unwrap(), IndianState::KA);
    }

    #[test]
    fn test_indian_professional_tax() {
        // States with Professional Tax
        assert!(IndianState::MH.has_professional_tax());
        assert!(IndianState::KA.has_professional_tax());
        assert!(IndianState::TN.has_professional_tax());
        assert_eq!(IndianState::MH.max_professional_tax_inr(), 2_500);

        // States without Professional Tax
        assert!(!IndianState::DL.has_professional_tax());
        assert!(!IndianState::UP.has_professional_tax());
        assert_eq!(IndianState::DL.max_professional_tax_inr(), 0);
    }

    // =======================================================================
    // Swiss Canton Tests
    // =======================================================================

    #[test]
    fn test_swiss_canton_parsing() {
        assert_eq!("ZH".parse::<SwissCanton>().unwrap(), SwissCanton::ZH);
        assert_eq!("zurich".parse::<SwissCanton>().unwrap(), SwissCanton::ZH);
        assert_eq!("ZÜRICH".parse::<SwissCanton>().unwrap(), SwissCanton::ZH);
        assert_eq!("ZG".parse::<SwissCanton>().unwrap(), SwissCanton::ZG);
        assert_eq!("zug".parse::<SwissCanton>().unwrap(), SwissCanton::ZG);
        assert_eq!("GE".parse::<SwissCanton>().unwrap(), SwissCanton::GE);
        assert_eq!("geneva".parse::<SwissCanton>().unwrap(), SwissCanton::GE);
        assert_eq!("GENÈVE".parse::<SwissCanton>().unwrap(), SwissCanton::GE);
    }

    #[test]
    fn test_swiss_canton_names() {
        assert_eq!(SwissCanton::ZH.name(), "Zürich");
        assert_eq!(SwissCanton::GE.name(), "Geneva");
        assert_eq!(SwissCanton::TI.name(), "Ticino");
        assert_eq!(SwissCanton::GR.name(), "Graubünden");
    }

    #[test]
    fn test_swiss_low_tax_cantons() {
        // These are the famous low-tax cantons
        assert!(SwissCanton::ZG.is_low_tax());
        assert!(SwissCanton::SZ.is_low_tax());
        assert!(SwissCanton::OW.is_low_tax());
        assert!(SwissCanton::NW.is_low_tax());

        // These are higher-tax cantons
        assert!(!SwissCanton::GE.is_low_tax());
        assert!(!SwissCanton::VD.is_low_tax());
        assert!(!SwissCanton::ZH.is_low_tax());
    }

    #[test]
    fn test_swiss_canton_languages() {
        assert_eq!(SwissCanton::ZH.language(), "German");
        assert_eq!(SwissCanton::GE.language(), "French");
        assert_eq!(SwissCanton::TI.language(), "Italian");
        assert_eq!(SwissCanton::GR.language(), "German/Romansh/Italian");
        assert_eq!(SwissCanton::FR.language(), "German/French");
    }

    #[test]
    fn test_zurich_brackets() {
        // CHF 50,000 income should be in the 7% bracket (43,700 - 79,100)
        let bracket = find_cantonal_bracket(50_000, SwissCanton::ZH, 2024, FilingStatus::Single).unwrap();
        assert_eq!(bracket.index, 6);
        assert_eq!(bracket.rate_bps, 700);
    }

    #[test]
    fn test_zug_low_tax() {
        // CHF 100,000 in Zug - should be in the 3% bracket (very low!)
        let bracket = find_cantonal_bracket(100_000, SwissCanton::ZG, 2024, FilingStatus::Single).unwrap();
        assert_eq!(bracket.index, 3);
        assert_eq!(bracket.rate_bps, 300);
    }

    #[test]
    fn test_obwalden_lowest_tax() {
        // CHF 200,000 in Obwalden - only 1.8% (lowest in Switzerland!)
        let bracket = find_cantonal_bracket(200_000, SwissCanton::OW, 2024, FilingStatus::Single).unwrap();
        assert_eq!(bracket.index, 3);
        assert_eq!(bracket.rate_bps, 180);
    }

    #[test]
    fn test_geneva_highest_tax() {
        // CHF 200,000 in Geneva - 19% (highest cantonal rate)
        let bracket = find_cantonal_bracket(200_000, SwissCanton::GE, 2024, FilingStatus::Single).unwrap();
        assert_eq!(bracket.index, 8);
        assert_eq!(bracket.rate_bps, 1900);
    }

    #[test]
    fn test_swiss_tax_comparison() {
        // Compare CHF 100,000 income across cantons
        let zug = find_cantonal_bracket(100_000, SwissCanton::ZG, 2024, FilingStatus::Single).unwrap();
        let geneva = find_cantonal_bracket(100_000, SwissCanton::GE, 2024, FilingStatus::Single).unwrap();
        let zurich = find_cantonal_bracket(100_000, SwissCanton::ZH, 2024, FilingStatus::Single).unwrap();

        // Zug should have much lower rate than Geneva
        assert!(zug.rate_bps < geneva.rate_bps);
        assert!(zug.rate_bps < zurich.rate_bps);
        // Zurich should be between Zug and Geneva
        assert!(zurich.rate_bps < geneva.rate_bps);
    }

    // =======================================================================
    // UK Nation Tests (Scotland vs Rest of UK)
    // =======================================================================

    #[test]
    fn test_uk_nation_parsing() {
        assert_eq!("SCOTLAND".parse::<UKNation>().unwrap(), UKNation::Scotland);
        assert_eq!("sco".parse::<UKNation>().unwrap(), UKNation::Scotland);
        assert_eq!("alba".parse::<UKNation>().unwrap(), UKNation::Scotland);
        assert_eq!("ENGLAND".parse::<UKNation>().unwrap(), UKNation::England);
        assert_eq!("wales".parse::<UKNation>().unwrap(), UKNation::Wales);
        assert_eq!("CYMRU".parse::<UKNation>().unwrap(), UKNation::Wales);
        assert_eq!("NI".parse::<UKNation>().unwrap(), UKNation::NorthernIreland);
    }

    #[test]
    fn test_uk_nation_names() {
        assert_eq!(UKNation::Scotland.name(), "Scotland");
        assert_eq!(UKNation::England.name(), "England");
        assert_eq!(UKNation::Wales.name(), "Wales");
        assert_eq!(UKNation::NorthernIreland.name(), "Northern Ireland");
    }

    #[test]
    fn test_uk_devolved_tax() {
        assert!(UKNation::Scotland.has_devolved_tax());
        assert!(!UKNation::England.has_devolved_tax());
        assert!(!UKNation::Wales.has_devolved_tax());
        assert!(!UKNation::NorthernIreland.has_devolved_tax());
    }

    #[test]
    fn test_scotland_starter_rate() {
        // £14,000 should be in Scotland's starter rate (19%)
        let bracket = find_uk_nation_bracket(14_000, UKNation::Scotland, 2024, FilingStatus::Single).unwrap();
        assert_eq!(bracket.index, 1);
        assert_eq!(bracket.rate_bps, 1900); // 19%
    }

    #[test]
    fn test_scotland_intermediate_rate() {
        // £35,000 should be in Scotland's intermediate rate (21%)
        let bracket = find_uk_nation_bracket(35_000, UKNation::Scotland, 2024, FilingStatus::Single).unwrap();
        assert_eq!(bracket.index, 3);
        assert_eq!(bracket.rate_bps, 2100); // 21%
    }

    #[test]
    fn test_scotland_top_rate() {
        // £100,000 should be in Scotland's top rate (47%)
        let bracket = find_uk_nation_bracket(100_000, UKNation::Scotland, 2024, FilingStatus::Single).unwrap();
        assert_eq!(bracket.index, 5);
        assert_eq!(bracket.rate_bps, 4700); // 47%
    }

    #[test]
    fn test_scotland_vs_rest_uk() {
        // Compare £50,000 income in Scotland vs England
        let scotland = find_uk_nation_bracket(50_000, UKNation::Scotland, 2024, FilingStatus::Single).unwrap();
        let england = find_uk_nation_bracket(50_000, UKNation::England, 2024, FilingStatus::Single).unwrap();

        // Scotland has higher rate (42%) vs England (20%) at this income
        assert_eq!(scotland.rate_bps, 4200); // 42% higher rate
        assert_eq!(england.rate_bps, 2000);  // 20% basic rate
        assert!(scotland.rate_bps > england.rate_bps);
    }

    #[test]
    fn test_scotland_more_progressive() {
        // Scotland has more brackets making it more progressive
        let scotland_low = find_uk_nation_bracket(15_000, UKNation::Scotland, 2024, FilingStatus::Single).unwrap();
        let england_low = find_uk_nation_bracket(15_000, UKNation::England, 2024, FilingStatus::Single).unwrap();

        // At low income, Scotland has starter rate (19%) vs England basic (20%)
        assert_eq!(scotland_low.rate_bps, 2000);  // 20% basic
        assert_eq!(england_low.rate_bps, 2000);   // 20% basic
        // But Scotland has more granularity (6 bands vs 4)
    }

    #[test]
    fn test_scotland_2025_advanced_rate() {
        // £80,000 should be in Scotland 2025's advanced rate (45%)
        let bracket = find_uk_nation_bracket(80_000, UKNation::Scotland, 2025, FilingStatus::Single).unwrap();
        assert_eq!(bracket.index, 5);
        assert_eq!(bracket.rate_bps, 4500); // 45% advanced rate
    }

    // =======================================================================
    // Brazilian State Tests
    // =======================================================================

    #[test]
    fn test_brazilian_state_parsing() {
        assert_eq!("SP".parse::<BrazilianState>().unwrap(), BrazilianState::SP);
        assert_eq!("SÃO PAULO".parse::<BrazilianState>().unwrap(), BrazilianState::SP);
        assert_eq!("sao paulo".parse::<BrazilianState>().unwrap(), BrazilianState::SP);
        assert_eq!("RJ".parse::<BrazilianState>().unwrap(), BrazilianState::RJ);
        assert_eq!("DF".parse::<BrazilianState>().unwrap(), BrazilianState::DF);
        assert_eq!("BRASÍLIA".parse::<BrazilianState>().unwrap(), BrazilianState::DF);
    }

    #[test]
    fn test_brazilian_state_names() {
        assert_eq!(BrazilianState::SP.name(), "São Paulo");
        assert_eq!(BrazilianState::RJ.name(), "Rio de Janeiro");
        assert_eq!(BrazilianState::DF.name(), "Distrito Federal");
        assert_eq!(BrazilianState::AM.name(), "Amazonas");
    }

    #[test]
    fn test_brazilian_regions() {
        assert_eq!(BrazilianState::SP.region(), "Sudeste");
        assert_eq!(BrazilianState::RJ.region(), "Sudeste");
        assert_eq!(BrazilianState::BA.region(), "Nordeste");
        assert_eq!(BrazilianState::AM.region(), "Norte");
        assert_eq!(BrazilianState::RS.region(), "Sul");
        assert_eq!(BrazilianState::DF.region(), "Centro-Oeste");
    }

    #[test]
    fn test_brazilian_economic_centers() {
        assert!(BrazilianState::SP.is_major_economic_center());
        assert!(BrazilianState::RJ.is_major_economic_center());
        assert!(BrazilianState::DF.is_major_economic_center());
        assert!(!BrazilianState::AC.is_major_economic_center());
        assert!(!BrazilianState::RR.is_major_economic_center());
    }

    #[test]
    fn test_brazilian_irpf_exempt() {
        // R$2,000/month should be exempt (below R$2,259)
        let bracket = find_brazilian_bracket(2_000, BrazilianState::SP, 2024, FilingStatus::Single, true).unwrap();
        assert_eq!(bracket.index, 0);
        assert_eq!(bracket.rate_bps, 0);
    }

    #[test]
    fn test_brazilian_irpf_starter() {
        // R$2,500/month should be in 7.5% bracket
        let bracket = find_brazilian_bracket(2_500, BrazilianState::SP, 2024, FilingStatus::Single, true).unwrap();
        assert_eq!(bracket.index, 1);
        assert_eq!(bracket.rate_bps, 750);
    }

    #[test]
    fn test_brazilian_irpf_max_rate() {
        // R$10,000/month should be in 27.5% bracket
        let bracket = find_brazilian_bracket(10_000, BrazilianState::SP, 2024, FilingStatus::Single, true).unwrap();
        assert_eq!(bracket.index, 4);
        assert_eq!(bracket.rate_bps, 2750);
    }

    #[test]
    fn test_brazilian_irpf_annual() {
        // R$50,000/year should be in 22.5% bracket
        let bracket = find_brazilian_bracket(50_000, BrazilianState::RJ, 2024, FilingStatus::Single, false).unwrap();
        assert_eq!(bracket.index, 3);
        assert_eq!(bracket.rate_bps, 2250);
    }

    #[test]
    fn test_brazilian_same_rates_all_states() {
        // All Brazilian states use the same federal IRPF rates
        let sp = find_brazilian_bracket(5_000, BrazilianState::SP, 2024, FilingStatus::Single, true).unwrap();
        let rj = find_brazilian_bracket(5_000, BrazilianState::RJ, 2024, FilingStatus::Single, true).unwrap();
        let am = find_brazilian_bracket(5_000, BrazilianState::AM, 2024, FilingStatus::Single, true).unwrap();

        assert_eq!(sp.rate_bps, rj.rate_bps);
        assert_eq!(rj.rate_bps, am.rate_bps);
        assert_eq!(sp.index, 4); // All in 27.5% bracket
    }
}
