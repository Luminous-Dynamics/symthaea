// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic, implementation-independent property generator for
//! MATH-REP-001D0.
//!
//! Labels come from the construction law used to build each pair. This module
//! intentionally does not import any Symthaea normalizer/retriever.

use symthaea_core::hdc::fol_formula_ext::{NumericType, Term};

pub const GENERATOR_ID: &str = "math-equivalence-property-q0-v2";
pub const AUTHORITY: &str = "MeasurementOnly";
pub const PAIRS_PER_SEED: usize = 128;
pub const REFUSALS_PER_SEED: usize = 24;

pub const DEV_SEEDS: [u64; 4] = [
    0x91A2_3B4C_5D6E_7F01,
    0xC0DE_CAFE_1020_3040,
    0x1357_9BDF_2468_ACE1,
    0x7A11_C0DE_55AA_F00D,
];

pub const EVAL_SEEDS: [u64; 4] = [
    0xA17E_5EED_3141_5926,
    0xD15C_0FFE_E0DD_F00D,
    0x6C8E_9CF5_7093_2BD1,
    0xF1A5_C0DE_BADC_0FFE,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairExpectation {
    SameNormalForm,
    DifferentNormalForm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OracleFamily {
    AddCommutative,
    AddAssociative,
    MulCommutative,
    MulAssociative,
    Distributive,
    AddZero,
    MulOne,
    DoubleNegation,
    SquareAsProduct,
    AdditiveCancellation,
    SubtractAsAddNegative,
    NegateSum,
    IntegerCoefficientSplit,
    RationalCoefficientSplit,
    PowerComposition,
    MultiplyZero,
    PowerOne,
    DifferenceOfSquares,
    FreshTermInjection,
    CoefficientPerturbation,
    ExponentPerturbation,
    OperatorPerturbation,
    SurvivingVariableChange,
    DomainIdentityChange,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefusalFamily {
    InexactRealLiteral,
    ConstantDivision,
    VariableDivision,
    FractionalIntLiteral,
    FractionalNatLiteral,
    ZeroDenominator,
    CoefficientOverflow,
    ExponentOverflow,
}

#[derive(Debug, Clone)]
pub struct GeneratedPair {
    pub id: String,
    pub seed: u64,
    pub index: usize,
    pub family: OracleFamily,
    pub expectation: PairExpectation,
    pub lhs_domain: NumericType,
    pub lhs: Term,
    pub rhs_domain: NumericType,
    pub rhs: Term,
}

#[derive(Debug, Clone)]
pub struct GeneratedRefusal {
    pub id: String,
    pub seed: u64,
    pub index: usize,
    pub family: RefusalFamily,
    pub domain: NumericType,
    pub term: Term,
    pub expected_disposition: &'static str,
    pub expected_receipt_reason: &'static str,
}

#[derive(Debug, Clone, Copy)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64*: fixed algorithm is part of GENERATOR_ID v2 lineage.
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn range(&mut self, upper_exclusive: u64) -> u64 {
        debug_assert!(upper_exclusive > 0);
        self.next_u64() % upper_exclusive
    }
}

pub fn generate_pairs(seed: u64) -> Vec<GeneratedPair> {
    let mut rng = DeterministicRng::new(seed);
    (0..PAIRS_PER_SEED)
        .map(|index| generate_pair(seed, index, &mut rng))
        .collect()
}

pub fn generate_refusals(seed: u64) -> Vec<GeneratedRefusal> {
    let mut rng = DeterministicRng::new(seed ^ 0xA5A5_5A5A_DEAD_BEEF);
    (0..REFUSALS_PER_SEED)
        .map(|index| generate_refusal(seed, index, &mut rng))
        .collect()
}

fn generate_pair(seed: u64, index: usize, rng: &mut DeterministicRng) -> GeneratedPair {
    let family = family_for(index);
    let mut domain = choose_domain(rng);

    // Families using exact rationals force the Real domain. Families that use
    // additive inverses avoid Nat so the oracle does not depend on any
    // truncated-natural subtraction convention.
    if family == OracleFamily::RationalCoefficientSplit {
        domain = NumericType::Real;
    } else if uses_additive_inverse(family) && domain == NumericType::Nat {
        domain = if rng.range(2) == 0 {
            NumericType::Int
        } else {
            NumericType::Real
        };
    }

    let a = nonzero_atom(rng, domain, 0);
    let b = nonzero_atom(rng, domain, 1);
    let c = nonzero_atom(rng, domain, 2);

    let (expectation, lhs_domain, lhs, rhs_domain, rhs) = match family {
        OracleFamily::AddCommutative => (
            PairExpectation::SameNormalForm,
            domain,
            a.clone().add(b.clone()),
            domain,
            b.add(a),
        ),
        OracleFamily::AddAssociative => (
            PairExpectation::SameNormalForm,
            domain,
            a.clone().add(b.clone()).add(c.clone()),
            domain,
            a.add(b.add(c)),
        ),
        OracleFamily::MulCommutative => (
            PairExpectation::SameNormalForm,
            domain,
            a.clone().mul(b.clone()),
            domain,
            b.mul(a),
        ),
        OracleFamily::MulAssociative => (
            PairExpectation::SameNormalForm,
            domain,
            a.clone().mul(b.clone()).mul(c.clone()),
            domain,
            a.mul(b.mul(c)),
        ),
        OracleFamily::Distributive => (
            PairExpectation::SameNormalForm,
            domain,
            a.clone().mul(b.clone().add(c.clone())),
            domain,
            a.clone().mul(b).add(a.mul(c)),
        ),
        OracleFamily::AddZero => (
            PairExpectation::SameNormalForm,
            domain,
            a.clone().add(Term::int(0)),
            domain,
            a,
        ),
        OracleFamily::MulOne => (
            PairExpectation::SameNormalForm,
            domain,
            a.clone().mul(Term::int(1)),
            domain,
            a,
        ),
        OracleFamily::DoubleNegation => (
            PairExpectation::SameNormalForm,
            domain,
            a.clone().neg().neg(),
            domain,
            a,
        ),
        OracleFamily::SquareAsProduct => (
            PairExpectation::SameNormalForm,
            domain,
            a.clone().pow(2),
            domain,
            a.clone().mul(a),
        ),
        OracleFamily::AdditiveCancellation => (
            PairExpectation::SameNormalForm,
            domain,
            a.clone().add(b.clone()).sub(b),
            domain,
            a,
        ),
        OracleFamily::SubtractAsAddNegative => (
            PairExpectation::SameNormalForm,
            domain,
            a.clone().sub(b.clone()),
            domain,
            a.add(b.neg()),
        ),
        OracleFamily::NegateSum => (
            PairExpectation::SameNormalForm,
            domain,
            a.clone().add(b.clone()).neg(),
            domain,
            a.neg().add(b.neg()),
        ),
        OracleFamily::IntegerCoefficientSplit => {
            let k1 = 1 + rng.range(4) as i64;
            let k2 = 1 + rng.range(4) as i64;
            let variable = Term::var("v3");
            (
                PairExpectation::SameNormalForm,
                domain,
                Term::int(k1 + k2).mul(variable.clone()),
                domain,
                Term::int(k1)
                    .mul(variable.clone())
                    .add(Term::int(k2).mul(variable)),
            )
        }
        OracleFamily::RationalCoefficientSplit => {
            let denominator = 2 + rng.range(6) as i64;
            let p1 = 1 + rng.range(denominator as u64) as i64;
            let p2 = 1 + rng.range(denominator as u64) as i64;
            let variable = Term::var("v3");
            (
                PairExpectation::SameNormalForm,
                NumericType::Real,
                Term::rat(p1 + p2, denominator).mul(variable.clone()),
                NumericType::Real,
                Term::rat(p1, denominator)
                    .mul(variable.clone())
                    .add(Term::rat(p2, denominator).mul(variable)),
            )
        }
        OracleFamily::PowerComposition => {
            let variable = Term::var("v3");
            let left_power = 1 + rng.range(3) as u32;
            let right_power = 2 + rng.range(3) as u32;
            (
                PairExpectation::SameNormalForm,
                domain,
                variable.clone().pow(left_power).pow(right_power),
                domain,
                variable.pow(left_power * right_power),
            )
        }
        OracleFamily::MultiplyZero => (
            PairExpectation::SameNormalForm,
            domain,
            a.mul(Term::int(0)),
            domain,
            Term::int(0),
        ),
        OracleFamily::PowerOne => (
            PairExpectation::SameNormalForm,
            domain,
            a.clone().pow(1),
            domain,
            a,
        ),
        OracleFamily::DifferenceOfSquares => (
            PairExpectation::SameNormalForm,
            domain,
            a.clone().sub(b.clone()).mul(a.clone().add(b.clone())),
            domain,
            a.pow(2).sub(b.pow(2)),
        ),
        OracleFamily::FreshTermInjection => {
            let fresh = Term::var(&format!("__delta_{seed:016x}_{index:04}"));
            (
                PairExpectation::DifferentNormalForm,
                domain,
                a.clone(),
                domain,
                a.add(fresh),
            )
        }
        OracleFamily::CoefficientPerturbation => {
            let k = 1 + rng.range(5) as i64;
            let variable = Term::var("v0");
            (
                PairExpectation::DifferentNormalForm,
                domain,
                Term::int(k).mul(variable.clone()),
                domain,
                Term::int(k + 1).mul(variable),
            )
        }
        OracleFamily::ExponentPerturbation => {
            let exponent = 1 + rng.range(3) as u32;
            let variable = Term::var("v0");
            (
                PairExpectation::DifferentNormalForm,
                domain,
                variable.clone().pow(exponent),
                domain,
                variable.pow(exponent + 1),
            )
        }
        OracleFamily::OperatorPerturbation => (
            PairExpectation::DifferentNormalForm,
            domain,
            Term::var("v0").add(Term::var("v1")),
            domain,
            Term::var("v0").mul(Term::var("v1")),
        ),
        OracleFamily::SurvivingVariableChange => (
            PairExpectation::DifferentNormalForm,
            domain,
            Term::var("v0").add(Term::int(1)),
            domain,
            Term::var("v4").add(Term::int(1)),
        ),
        OracleFamily::DomainIdentityChange => {
            let (lhs_domain, rhs_domain) = match domain {
                NumericType::Int => (NumericType::Int, NumericType::Real),
                NumericType::Nat => (NumericType::Nat, NumericType::Real),
                NumericType::Real => (NumericType::Real, NumericType::Int),
            };
            let value = Term::var("v0").add(Term::int(1));
            (
                PairExpectation::DifferentNormalForm,
                lhs_domain,
                value.clone(),
                rhs_domain,
                value,
            )
        }
    };

    GeneratedPair {
        id: format!("PROP_{seed:016x}_{index:04}_{family:?}"),
        seed,
        index,
        family,
        expectation,
        lhs_domain,
        lhs,
        rhs_domain,
        rhs,
    }
}

fn generate_refusal(seed: u64, index: usize, rng: &mut DeterministicRng) -> GeneratedRefusal {
    let family = refusal_family_for(index);
    let (domain, term, disposition, reason) = match family {
        RefusalFamily::InexactRealLiteral => {
            let numerator = 1 + rng.range(99) as f64;
            (NumericType::Real, Term::real(numerator / 10.0), "Unsupported", "OtherUnsupported")
        }
        RefusalFamily::ConstantDivision => (
            NumericType::Real,
            Term::var("v0").add(Term::int(1)).div(Term::int(2)),
            "Unsupported",
            "NonPolynomialDivision",
        ),
        RefusalFamily::VariableDivision => (
            NumericType::Real,
            Term::int(1).div(Term::var("v0").add(Term::int(1))),
            "Unsupported",
            "NonPolynomialDivision",
        ),
        RefusalFamily::FractionalIntLiteral => (
            NumericType::Int,
            Term::rat(1 + rng.range(4) as i64, 5),
            "Unsupported",
            "DomainAmbiguity",
        ),
        RefusalFamily::FractionalNatLiteral => (
            NumericType::Nat,
            Term::rat(1 + rng.range(4) as i64, 5),
            "Unsupported",
            "DomainAmbiguity",
        ),
        RefusalFamily::ZeroDenominator => (
            NumericType::Real,
            Term::RatLit(1, 0),
            "Rejected",
            "OtherUnsupported",
        ),
        RefusalFamily::CoefficientOverflow => (
            NumericType::Int,
            Term::int(i64::MAX).add(Term::int(1 + rng.range(8) as i64)),
            "Unsupported",
            "ResourceLimit",
        ),
        RefusalFamily::ExponentOverflow => (
            NumericType::Real,
            Term::var("v0")
                .pow(u32::MAX)
                .mul(Term::var("v0").pow(1 + rng.range(2) as u32)),
            "Unsupported",
            "ResourceLimit",
        ),
    };

    GeneratedRefusal {
        id: format!("REFUSE_{seed:016x}_{index:04}_{family:?}"),
        seed,
        index,
        family,
        domain,
        term,
        expected_disposition: disposition,
        expected_receipt_reason: reason,
    }
}

fn family_for(index: usize) -> OracleFamily {
    use OracleFamily::*;
    const FAMILIES: [OracleFamily; 24] = [
        AddCommutative,
        AddAssociative,
        MulCommutative,
        MulAssociative,
        Distributive,
        AddZero,
        MulOne,
        DoubleNegation,
        SquareAsProduct,
        AdditiveCancellation,
        SubtractAsAddNegative,
        NegateSum,
        IntegerCoefficientSplit,
        RationalCoefficientSplit,
        PowerComposition,
        MultiplyZero,
        PowerOne,
        DifferenceOfSquares,
        FreshTermInjection,
        CoefficientPerturbation,
        ExponentPerturbation,
        OperatorPerturbation,
        SurvivingVariableChange,
        DomainIdentityChange,
    ];
    FAMILIES[index % FAMILIES.len()]
}

fn refusal_family_for(index: usize) -> RefusalFamily {
    use RefusalFamily::*;
    const FAMILIES: [RefusalFamily; 8] = [
        InexactRealLiteral,
        ConstantDivision,
        VariableDivision,
        FractionalIntLiteral,
        FractionalNatLiteral,
        ZeroDenominator,
        CoefficientOverflow,
        ExponentOverflow,
    ];
    FAMILIES[index % FAMILIES.len()]
}

fn choose_domain(rng: &mut DeterministicRng) -> NumericType {
    match rng.range(3) {
        0 => NumericType::Int,
        1 => NumericType::Real,
        _ => NumericType::Nat,
    }
}

fn uses_additive_inverse(family: OracleFamily) -> bool {
    matches!(
        family,
        OracleFamily::DoubleNegation
            | OracleFamily::AdditiveCancellation
            | OracleFamily::SubtractAsAddNegative
            | OracleFamily::NegateSum
            | OracleFamily::DifferenceOfSquares
    )
}

fn nonzero_atom(rng: &mut DeterministicRng, domain: NumericType, slot: usize) -> Term {
    let coefficient = 1 + rng.range(4) as i64;
    let exponent = 1 + rng.range(3) as u32;
    let offset = rng.range(4) as i64;
    let variable = Term::var(&format!("v{slot}")).pow(exponent);
    let scaled = Term::int(coefficient).mul(variable);

    if offset == 0 {
        scaled
    } else {
        // Positive coefficients/offsets keep Nat fixtures free of an accidental
        // reliance on negative literals while still exercising polynomial ASTs.
        let _ = domain;
        scaled.add(Term::int(offset))
    }
}
