// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Frozen fixture types for MATH-REP-001C.

use symthaea_core::hdc::fol_formula_ext::{NumericType, Term};

pub const FIXTURE_SET_ID: &str = "math-equivalence-retrieval-q0-v1";
pub const AUTHORITY: &str = "MeasurementOnly";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpectedRelation {
    SameNormalForm,
    DifferentNormalForm,
}

#[derive(Debug, Clone)]
pub struct PairCase {
    pub id: &'static str,
    pub lhs_domain: NumericType,
    pub lhs: Term,
    pub rhs_domain: NumericType,
    pub rhs: Term,
    pub expected: ExpectedRelation,
}

#[derive(Debug, Clone)]
pub struct RefusalCase {
    pub id: &'static str,
    pub domain: NumericType,
    pub term: Term,
    pub expected_disposition: &'static str,
    pub expected_receipt_reason: &'static str,
}
