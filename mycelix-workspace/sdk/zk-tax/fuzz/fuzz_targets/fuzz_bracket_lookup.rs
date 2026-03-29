#![no_main]

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;
use mycelix_zk_tax::{Jurisdiction, FilingStatus, brackets::{find_bracket, get_brackets}};

#[derive(Debug, Arbitrary)]
struct BracketLookupInput {
    income: u64,
    jurisdiction_idx: u8,
    year: u32,
    filing_status_idx: u8,
}

impl BracketLookupInput {
    fn jurisdiction(&self) -> Jurisdiction {
        match self.jurisdiction_idx % 58 {
            0 => Jurisdiction::US,
            1 => Jurisdiction::UK,
            2 => Jurisdiction::DE,
            3 => Jurisdiction::FR,
            4 => Jurisdiction::CA,
            5 => Jurisdiction::AU,
            6 => Jurisdiction::JP,
            7 => Jurisdiction::CN,
            8 => Jurisdiction::IN,
            9 => Jurisdiction::BR,
            10 => Jurisdiction::MX,
            11 => Jurisdiction::KR,
            12 => Jurisdiction::IT,
            13 => Jurisdiction::RU,
            14 => Jurisdiction::TR,
            15 => Jurisdiction::AR,
            16 => Jurisdiction::ID,
            17 => Jurisdiction::SA,
            18 => Jurisdiction::ZA,
            19 => Jurisdiction::CL,
            20 => Jurisdiction::CO,
            21 => Jurisdiction::PE,
            22 => Jurisdiction::EC,
            23 => Jurisdiction::UY,
            24 => Jurisdiction::ES,
            25 => Jurisdiction::NL,
            26 => Jurisdiction::BE,
            27 => Jurisdiction::AT,
            28 => Jurisdiction::PT,
            29 => Jurisdiction::IE,
            30 => Jurisdiction::PL,
            31 => Jurisdiction::SE,
            32 => Jurisdiction::DK,
            33 => Jurisdiction::FI,
            34 => Jurisdiction::NO,
            35 => Jurisdiction::CH,
            36 => Jurisdiction::CZ,
            37 => Jurisdiction::GR,
            38 => Jurisdiction::HU,
            39 => Jurisdiction::RO,
            40 => Jurisdiction::UA,
            41 => Jurisdiction::NZ,
            42 => Jurisdiction::SG,
            43 => Jurisdiction::HK,
            44 => Jurisdiction::TW,
            45 => Jurisdiction::MY,
            46 => Jurisdiction::TH,
            47 => Jurisdiction::VN,
            48 => Jurisdiction::PH,
            49 => Jurisdiction::PK,
            50 => Jurisdiction::AE,
            51 => Jurisdiction::IL,
            52 => Jurisdiction::EG,
            53 => Jurisdiction::QA,
            54 => Jurisdiction::NG,
            55 => Jurisdiction::KE,
            56 => Jurisdiction::MA,
            _ => Jurisdiction::GH,
        }
    }

    fn filing_status(&self) -> FilingStatus {
        match self.filing_status_idx % 4 {
            0 => FilingStatus::Single,
            1 => FilingStatus::MarriedFilingJointly,
            2 => FilingStatus::MarriedFilingSeparately,
            _ => FilingStatus::HeadOfHousehold,
        }
    }

    fn year(&self) -> u32 {
        // Constrain to valid years
        2020 + (self.year % 6)
    }
}

fuzz_target!(|input: BracketLookupInput| {
    let jurisdiction = input.jurisdiction();
    let filing_status = input.filing_status();
    let year = input.year();

    // Test get_brackets - should never panic
    let brackets_result = get_brackets(jurisdiction, year, filing_status);

    // If we got brackets, they should be valid
    if let Ok(brackets) = brackets_result {
        // Brackets should be non-empty
        assert!(!brackets.is_empty(), "Empty brackets for {:?}", jurisdiction);

        // Brackets should be sorted by lower bound
        for window in brackets.windows(2) {
            assert!(window[0].lower < window[1].lower, "Unsorted brackets");
        }

        // Each bracket should have valid bounds
        for bracket in brackets {
            assert!(bracket.lower <= bracket.upper, "Invalid bracket bounds");
        }
    }

    // Test find_bracket - should never panic for valid inputs
    let find_result = find_bracket(input.income, jurisdiction, year, filing_status);

    if let Ok(bracket) = find_result {
        // The found bracket should contain the income
        assert!(input.income >= bracket.lower, "Income below bracket lower bound");
        if bracket.upper != u64::MAX {
            assert!(input.income < bracket.upper, "Income at or above bracket upper bound");
        }
    }
});
