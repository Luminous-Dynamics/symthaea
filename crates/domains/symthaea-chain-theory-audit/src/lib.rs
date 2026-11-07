pub mod physical_consistency;
pub use physical_consistency::*;

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent combinatorial audit of the published Chain Theory alpha count.
//!
//! This module reimplements the four-slot product-record construction,
//! read-exchange symmetrization, qubit commutator records and two exchange
//! signatures. It does not import the author's Lean implementation.
//!
//! It reproduces the *calculation under the stated definitions*. It does not
//! establish that those definitions describe nature, that the read operation
//! captures all physically permitted LOCC distillation, or that the
//! qutrit-qubit link is a photon. The published physical identification is a
//! conjecture, not a theorem of the arithmetic.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

/// Four slots: (read 1, center), (read 1, far), (read 2, center),
/// (read 2, far). Qutrit slots have eight traceless letters; qubit slots
/// have three.
const SLOT_DIMENSIONS: [u8; 4] = [8, 3, 8, 3];

type Assignment = [Option<u8>; 4];

#[derive(Clone, Debug, PartialEq, Eq)]
struct Form {
    /// Keep the first-term order: the Y-move rule acts on the first record
    /// term, as in the paper's definition.
    terms: Vec<(Assignment, i32)>,
}

impl Form {
    fn new(terms: Vec<(Assignment, i32)>) -> Self {
        let mut combined: Vec<(Assignment, i32)> = Vec::new();
        for (assignment, coefficient) in terms {
            if let Some((_, existing)) = combined
                .iter_mut()
                .find(|(candidate, _)| *candidate == assignment)
            {
                *existing += coefficient;
            } else {
                combined.push((assignment, coefficient));
            }
        }
        combined.retain(|(_, coefficient)| *coefficient != 0);
        Self { terms: combined }
    }

    fn single(assignment: Assignment) -> Self {
        Self::new(vec![(assignment, 1)])
    }

    fn exchange_reads(&self) -> Self {
        Self::new(
            self.terms
                .iter()
                .map(|(assignment, coefficient)| (exchange_assignment(*assignment), *coefficient))
                .collect(),
        )
    }

    fn scaled(&self, factor: i32) -> Self {
        Self::new(
            self.terms
                .iter()
                .map(|(assignment, coefficient)| (*assignment, coefficient * factor))
                .collect(),
        )
    }

    fn canonical(&self) -> BTreeMap<Assignment, i32> {
        let mut map = BTreeMap::new();
        for (assignment, coefficient) in &self.terms {
            *map.entry(*assignment).or_insert(0) += coefficient;
        }
        map.retain(|_, coefficient| *coefficient != 0);
        map
    }

    fn same_as_scaled(&self, other: &Self, factor: i32) -> bool {
        self.canonical() == other.scaled(factor).canonical()
    }

    fn first_assignment(&self) -> Option<Assignment> {
        self.terms.first().map(|(assignment, _)| *assignment)
    }

    fn has_two_letters_at_same_end(&self) -> bool {
        let Some(assignment) = self.first_assignment() else {
            return false;
        };
        let slots: Vec<usize> = assignment
            .iter()
            .enumerate()
            .filter_map(|(slot, letter)| letter.map(|_| slot))
            .collect();
        slots.len() == 2 && slots[0] % 2 == slots[1] % 2
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RecordConstruction {
    Odd,
    Even,
    ExchangeFixed,
    Commutator,
}

impl RecordConstruction {
    fn sort_rank(self) -> u8 {
        match self {
            Self::Odd => 0,
            Self::Even => 1,
            Self::ExchangeFixed => 2,
            Self::Commutator => 3,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Odd => "odd",
            Self::Even => "even",
            Self::ExchangeFixed => "exchange-fixed",
            Self::Commutator => "commutator",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SlotKinds {
    Mixed,
    Same,
    Bracket,
}

impl SlotKinds {
    fn sort_rank(self) -> u8 {
        match self {
            Self::Mixed => 0,
            Self::Same => 1,
            Self::Bracket => 2,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Mixed => "mixed",
            Self::Same => "same",
            Self::Bracket => "bracket",
        }
    }
}

#[derive(Clone, Debug)]
struct Record {
    form: Form,
    /// +1 for product records; the qubit commutator records carry -1.
    sign: i32,
    /// The exchanged-read image exists only for records with two letters
    /// at one end, or for the commutator records.
    reads: Option<Form>,
    construction: RecordConstruction,
    slot_kinds: SlotKinds,
}

impl Record {
    fn signature_x(&self) -> i32 {
        if self
            .reads
            .as_ref()
            .is_some_and(|image| image.same_as_scaled(&self.form, self.sign))
        {
            -1
        } else {
            1
        }
    }

    fn first_term_is_moved_by_y(&self) -> bool {
        let Some(first) = self.form.first_assignment() else {
            return false;
        };
        let original: BTreeSet<u8> = occupied_qudits(first).into_iter().collect();
        let moved: BTreeSet<u8> = occupied_qudits(first)
            .into_iter()
            .map(swap_far_qudit)
            .collect();
        original != moved
    }

    fn signature_y(&self) -> i32 {
        let image = self
            .first_term_is_moved_by_y()
            .then(|| self.form.exchange_reads());
        if image.is_some_and(|form| form.same_as_scaled(&self.form, self.sign)) {
            -1
        } else {
            1
        }
    }
}

fn exchange_assignment(a: Assignment) -> Assignment {
    [a[2], a[3], a[0], a[1]]
}

/// Author-compatible lexicographic key: the ordered list of occupied
/// (slot, basis-letter) pairs. This matches the reference enumerator's
/// tuple ordering, unlike a digit-packed key that prioritizes letter values.
fn assignment_key(a: Assignment) -> Vec<(usize, u8)> {
    a.iter()
        .enumerate()
        .filter_map(|(slot, letter)| letter.map(|index| (slot, index)))
        .collect()
}

fn product_slot_kinds(a: Assignment) -> SlotKinds {
    let slots: Vec<usize> = a
        .iter()
        .enumerate()
        .filter_map(|(slot, letter)| letter.map(|_| slot))
        .collect();
    if slots.len() == 2 && slots[0] % 2 == slots[1] % 2 {
        SlotKinds::Same
    } else {
        SlotKinds::Mixed
    }
}

fn occupied_qudits(a: Assignment) -> Vec<u8> {
    a.iter()
        .enumerate()
        .filter_map(|(slot, letter)| {
            letter.map(|_| match slot {
                0 | 2 => 0, // the shared qutrit center
                1 => 1,     // first far qubit
                _ => 2,     // second far qubit
            })
        })
        .collect()
}

fn swap_far_qudit(qudit: u8) -> u8 {
    match qudit {
        1 => 2,
        2 => 1,
        other => other,
    }
}

fn product_assignments() -> Vec<Assignment> {
    let mut products = Vec::new();
    for first_slot in 0..4 {
        for second_slot in (first_slot + 1)..4 {
            for first_letter in 0..SLOT_DIMENSIONS[first_slot] {
                for second_letter in 0..SLOT_DIMENSIONS[second_slot] {
                    let mut assignment = [None; 4];
                    assignment[first_slot] = Some(first_letter);
                    assignment[second_slot] = Some(second_letter);
                    products.push(assignment);
                }
            }
        }
    }
    products
}

fn product_records(products: &[Assignment]) -> Vec<Record> {
    let mut records = Vec::with_capacity(products.len());
    for &product in products {
        let exchanged = exchange_assignment(product);
        let slot_kinds = product_slot_kinds(product);
        if product == exchanged {
            let form = Form::single(product);
            let reads = form
                .has_two_letters_at_same_end()
                .then(|| form.exchange_reads());
            records.push(Record {
                form,
                sign: 1,
                reads,
                construction: RecordConstruction::ExchangeFixed,
                slot_kinds,
            });
        } else if assignment_key(product) < assignment_key(exchanged) {
            let odd = Form::new(vec![(product, 1), (exchanged, -1)]);
            let even = Form::new(vec![(product, 1), (exchanged, 1)]);
            let has_reads = odd.has_two_letters_at_same_end();
            records.push(Record {
                reads: has_reads.then(|| odd.exchange_reads()),
                form: odd,
                sign: 1,
                construction: RecordConstruction::Odd,
                slot_kinds,
            });
            records.push(Record {
                reads: has_reads.then(|| even.exchange_reads()),
                form: even,
                sign: 1,
                construction: RecordConstruction::Even,
                slot_kinds,
            });
        }
    }
    records
}

type Matrix2 = [[i32; 2]; 2];

/// The integer, traceless basis used at a qubit: symmetric off-diagonal,
/// antisymmetric off-diagonal, and diagonal.
fn qubit_traceless_basis() -> [Matrix2; 3] {
    [[[0, 1], [1, 0]], [[0, 1], [-1, 0]], [[1, 0], [0, -1]]]
}

fn matrix_product(a: Matrix2, b: Matrix2) -> Matrix2 {
    let mut out = [[0; 2]; 2];
    for row in 0..2 {
        for col in 0..2 {
            for inner in 0..2 {
                out[row][col] += a[row][inner] * b[inner][col];
            }
        }
    }
    out
}

fn matrix_subtract(a: Matrix2, b: Matrix2) -> Matrix2 {
    let mut out = [[0; 2]; 2];
    for row in 0..2 {
        for col in 0..2 {
            out[row][col] = a[row][col] - b[row][col];
        }
    }
    out
}

fn commutator_coefficients(a: Matrix2, b: Matrix2) -> Vec<(usize, i32)> {
    let commutator = matrix_subtract(matrix_product(a, b), matrix_product(b, a));
    let basis = qubit_traceless_basis();
    let mut coefficients = Vec::new();
    for (index, letter) in basis.iter().enumerate() {
        let mut inner_product = 0;
        let mut norm_squared = 0;
        for row in 0..2 {
            for col in 0..2 {
                inner_product += letter[row][col] * commutator[row][col];
                norm_squared += letter[row][col] * letter[row][col];
            }
        }
        if inner_product != 0 {
            // The three basis matrices are mutually orthogonal and have
            // nonzero integer norm, so the coefficients divide exactly.
            assert_eq!(inner_product % norm_squared, 0);
            coefficients.push((index, inner_product / norm_squared));
        }
    }
    coefficients
}

fn commutator_at(first: usize, second: usize, slot: usize) -> Form {
    let basis = qubit_traceless_basis();
    let coefficients = commutator_coefficients(basis[first], basis[second]);
    Form::new(
        coefficients
            .into_iter()
            .map(|(letter, coefficient)| {
                let mut assignment = [None; 4];
                assignment[slot] = Some(letter as u8);
                (assignment, coefficient)
            })
            .collect(),
    )
}

fn add_forms(a: &Form, b: &Form, sign_b: i32) -> Form {
    let mut terms = a.terms.clone();
    terms.extend(
        b.terms
            .iter()
            .map(|(assignment, coefficient)| (*assignment, coefficient * sign_b)),
    );
    Form::new(terms)
}

fn exchange_parity(form: &Form) -> i32 {
    let exchanged = form.exchange_reads();
    if exchanged.same_as_scaled(form, 1) {
        1
    } else if exchanged.same_as_scaled(form, -1) {
        -1
    } else {
        0
    }
}

fn weyl_phase_is_a_sign(dimension: usize) -> bool {
    (0..dimension).all(|j| {
        let phase = (2 * j + 2 * dimension - 2 * ((j + 1) % dimension)) % (2 * dimension);
        phase == dimension
    })
}

fn commutator_records() -> Vec<Record> {
    let sign = if weyl_phase_is_a_sign(2) { -1 } else { 1 };
    let mut records = Vec::new();
    let basis = qubit_traceless_basis();

    for first in 0..basis.len() {
        for second in (first + 1)..basis.len() {
            let at_first_read = commutator_at(first, second, 1);
            let at_second_read = commutator_at(first, second, 3);
            let reverse_first = commutator_at(second, first, 1);
            let reverse_second = commutator_at(second, first, 3);

            let candidates = [
                add_forms(&at_first_read, &at_second_read, 1),
                add_forms(&at_first_read, &at_second_read, -1),
            ];
            let reverse_candidates = [
                add_forms(&reverse_first, &reverse_second, 1),
                add_forms(&reverse_first, &reverse_second, -1),
            ];

            for (candidate, reverse) in candidates.into_iter().zip(reverse_candidates) {
                if exchange_parity(&candidate) == sign {
                    records.push(Record {
                        form: candidate,
                        sign,
                        reads: Some(reverse),
                        construction: RecordConstruction::Commutator,
                        slot_kinds: SlotKinds::Bracket,
                    });
                }
            }
        }
    }
    records
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExchangeSectors {
    /// Neither exchange is survived.
    pub plus_plus: usize,
    /// Only Y (the far-link exchange) is survived.
    pub plus_minus: usize,
    /// Only X (the two-read exchange) is survived.
    pub minus_plus: usize,
    /// Both exchanges are survived.
    pub minus_minus: usize,
}

impl ExchangeSectors {
    fn add(&mut self, signature_x: i32, signature_y: i32) {
        match (signature_x, signature_y) {
            (1, 1) => self.plus_plus += 1,
            (1, -1) => self.plus_minus += 1,
            (-1, 1) => self.minus_plus += 1,
            (-1, -1) => self.minus_minus += 1,
            _ => unreachable!("exchange signatures are signs"),
        }
    }

    pub fn total(self) -> usize {
        self.plus_plus + self.plus_minus + self.minus_plus + self.minus_minus
    }

    /// Exact algebraic numerator, represented as coefficients of
    /// (sqrt(1), sqrt(2), sqrt(3), sqrt(6)).
    pub fn radical_coefficients(self) -> [usize; 4] {
        [
            self.plus_plus,
            self.plus_minus,
            self.minus_plus,
            self.minus_minus,
        ]
    }

    pub fn weighted_numerator_f64(self) -> f64 {
        self.plus_plus as f64
            + self.plus_minus as f64 * 2.0_f64.sqrt()
            + self.minus_plus as f64 * 3.0_f64.sqrt()
            + self.minus_minus as f64 * 6.0_f64.sqrt()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecordRow {
    /// One-based row position, matching the R001...R172 CSV ordinal.
    pub ordinal: usize,
    pub construction: &'static str,
    pub slot_kinds: &'static str,
    pub signature_x: i32,
    pub signature_y: i32,
    pub ends_passed: &'static str,
    pub weight_squared: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RecordCategoryCounts {
    pub odd_mixed_plus_plus: usize,
    pub odd_same_plus_plus: usize,
    pub even_mixed_plus_minus: usize,
    pub even_same_minus_plus: usize,
    pub exchange_fixed_same_minus_plus: usize,
    pub commutator_minus_minus: usize,
}

impl RecordCategoryCounts {
    pub fn total(self) -> usize {
        self.odd_mixed_plus_plus
            + self.odd_same_plus_plus
            + self.even_mixed_plus_minus
            + self.even_same_minus_plus
            + self.exchange_fixed_same_minus_plus
            + self.commutator_minus_minus
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChainAlphaAudit {
    pub product_assignments: usize,
    pub product_records: usize,
    pub commutator_records: usize,
    pub total_records: usize,
    pub sectors: ExchangeSectors,
    pub categories: RecordCategoryCounts,
    pub readout_dimension: usize,
    pub alpha_candidate: f64,
    pub inverse_alpha_candidate: f64,
    pub codata_2022_inverse: f64,
    pub codata_2022_standard_uncertainty: f64,
    pub difference_in_standard_uncertainties: f64,
}

fn build_records() -> (usize, usize, usize, Vec<Record>) {
    let products = product_assignments();
    let product_record_list = product_records(&products);
    let commutator_record_list = commutator_records();
    let product_record_count = product_record_list.len();
    let commutator_record_count = commutator_record_list.len();
    let mut all_records = product_record_list;
    all_records.extend(commutator_record_list);
    // This is the stable sort used by the author's enumerate_records.py:
    // construction order, then "mixed"/"same"/"bracket" within that class.
    all_records.sort_by_key(|record| {
        (
            record.construction.sort_rank(),
            record.slot_kinds.sort_rank(),
        )
    });
    (
        products.len(),
        product_record_count,
        commutator_record_count,
        all_records,
    )
}

fn record_weight_squared(signature_x: i32, signature_y: i32) -> usize {
    (if signature_x == -1 { 3 } else { 1 }) * (if signature_y == -1 { 2 } else { 1 })
}

fn record_ends_passed(signature_x: i32, signature_y: i32) -> &'static str {
    match (signature_x, signature_y) {
        (1, 1) => "none",
        (-1, 1) => "qutrit",
        (1, -1) => "qubit",
        (-1, -1) => "both",
        _ => unreachable!("exchange signatures are signs"),
    }
}

/// Emit all 172 row classifications in the author's CSV ordering. This
/// supports direct comparison of construction, slot class, signs and weights.
pub fn record_rows() -> Vec<RecordRow> {
    let (_, _, _, records) = build_records();
    records
        .iter()
        .enumerate()
        .map(|(index, record)| {
            let signature_x = record.signature_x();
            let signature_y = record.signature_y();
            RecordRow {
                ordinal: index + 1,
                construction: record.construction.as_str(),
                slot_kinds: record.slot_kinds.as_str(),
                signature_x,
                signature_y,
                ends_passed: record_ends_passed(signature_x, signature_y),
                weight_squared: record_weight_squared(signature_x, signature_y),
            }
        })
        .collect()
}

fn count_record_categories(rows: &[RecordRow]) -> RecordCategoryCounts {
    let mut counts = RecordCategoryCounts::default();
    for row in rows {
        match (
            row.construction,
            row.slot_kinds,
            row.signature_x,
            row.signature_y,
            row.ends_passed,
            row.weight_squared,
        ) {
            ("odd", "mixed", 1, 1, "none", 1) => counts.odd_mixed_plus_plus += 1,
            ("odd", "same", 1, 1, "none", 1) => counts.odd_same_plus_plus += 1,
            ("even", "mixed", 1, -1, "qubit", 2) => counts.even_mixed_plus_minus += 1,
            ("even", "same", -1, 1, "qutrit", 3) => counts.even_same_minus_plus += 1,
            ("exchange-fixed", "same", -1, 1, "qutrit", 3) => {
                counts.exchange_fixed_same_minus_plus += 1
            }
            ("commutator", "bracket", -1, -1, "both", 6) => counts.commutator_minus_minus += 1,
            other => panic!("unexpected Chain Theory CSV row classification: {other:?}"),
        }
    }
    counts
}

/// Reproduce the record count, row classifications, exchange sectors and
/// weighted scalar under the published four-slot qutrit-qubit definitions.
pub fn reproduce() -> ChainAlphaAudit {
    let (product_assignments, product_records, commutator_records, all_records) = build_records();

    let mut sectors = ExchangeSectors::default();
    for record in &all_records {
        sectors.add(record.signature_x(), record.signature_y());
    }
    let rows: Vec<RecordRow> = all_records
        .iter()
        .enumerate()
        .map(|(index, record)| {
            let signature_x = record.signature_x();
            let signature_y = record.signature_y();
            RecordRow {
                ordinal: index + 1,
                construction: record.construction.as_str(),
                slot_kinds: record.slot_kinds.as_str(),
                signature_x,
                signature_y,
                ends_passed: record_ends_passed(signature_x, signature_y),
                weight_squared: record_weight_squared(signature_x, signature_y),
            }
        })
        .collect();
    let categories = count_record_categories(&rows);

    // Two read links each contribute (3*2)^2 operator directions.
    // The unread link contributes (3^2-1)(2^2-1) traceless directions.
    let unread_link_dimension = (3_usize.pow(2) - 1) * (2_usize.pow(2) - 1);
    let read_link_dimension = (3 * 2_usize).pow(2);
    let readout_dimension = unread_link_dimension * read_link_dimension.pow(2);

    let weighted_numerator = sectors.weighted_numerator_f64();
    let inverse_alpha_candidate = readout_dimension as f64 / weighted_numerator;
    let alpha_candidate = 1.0 / inverse_alpha_candidate;
    let codata_2022_inverse = 137.035_999_177;
    let codata_2022_standard_uncertainty = 2.1e-8;
    let difference_in_standard_uncertainties =
        (inverse_alpha_candidate - codata_2022_inverse) / codata_2022_standard_uncertainty;

    ChainAlphaAudit {
        product_assignments,
        product_records,
        commutator_records,
        total_records: all_records.len(),
        sectors,
        categories,
        readout_dimension,
        alpha_candidate,
        inverse_alpha_candidate,
        codata_2022_inverse,
        codata_2022_standard_uncertainty,
        difference_in_standard_uncertainties,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_construction_has_169_two_letter_product_assignments() {
        let products = product_assignments();
        assert_eq!(products.len(), 169);
    }

    #[test]
    fn qubit_commutator_algebra_generates_three_order_records() {
        assert!(weyl_phase_is_a_sign(2));
        assert!(!weyl_phase_is_a_sign(3));
        assert_eq!(commutator_records().len(), 3);
        for record in commutator_records() {
            assert_eq!(record.signature_x(), -1);
            assert_eq!(record.signature_y(), -1);
        }
    }

    #[test]
    fn independent_enumeration_reproduces_the_four_exchange_sectors() {
        let audit = reproduce();
        assert_eq!(audit.product_records, 169);
        assert_eq!(audit.commutator_records, 3);
        assert_eq!(audit.total_records, 172);
        assert_eq!(
            audit.sectors,
            ExchangeSectors {
                plus_plus: 79,
                plus_minus: 48,
                minus_plus: 42,
                minus_minus: 3,
            }
        );
        assert_eq!(audit.sectors.total(), audit.total_records);
        assert_eq!(audit.categories.total(), audit.total_records);
        assert_eq!(
            audit.categories,
            RecordCategoryCounts {
                odd_mixed_plus_plus: 48,
                odd_same_plus_plus: 31,
                even_mixed_plus_minus: 48,
                even_same_minus_plus: 31,
                exchange_fixed_same_minus_plus: 11,
                commutator_minus_minus: 3,
            }
        );
    }

    #[test]
    fn row_classification_matches_all_author_csv_bands() {
        let rows = record_rows();
        assert_eq!(rows.len(), 172);
        let bands = [
            (1, 48, "odd", "mixed", 1, 1, "none", 1),
            (49, 79, "odd", "same", 1, 1, "none", 1),
            (80, 127, "even", "mixed", 1, -1, "qubit", 2),
            (128, 158, "even", "same", -1, 1, "qutrit", 3),
            (159, 169, "exchange-fixed", "same", -1, 1, "qutrit", 3),
            (170, 172, "commutator", "bracket", -1, -1, "both", 6),
        ];
        for (first, last, construction, slot_kinds, x, y, ends, k) in bands {
            for row in &rows[(first - 1)..last] {
                assert_eq!(row.construction, construction, "ordinal={}", row.ordinal);
                assert_eq!(row.slot_kinds, slot_kinds, "ordinal={}", row.ordinal);
                assert_eq!(row.signature_x, x, "ordinal={}", row.ordinal);
                assert_eq!(row.signature_y, y, "ordinal={}", row.ordinal);
                assert_eq!(row.ends_passed, ends, "ordinal={}", row.ordinal);
                assert_eq!(row.weight_squared, k, "ordinal={}", row.ordinal);
            }
        }
    }

    #[test]
    fn readout_space_dimension_is_31104() {
        assert_eq!(reproduce().readout_dimension, 31_104);
    }

    #[test]
    fn radical_sum_matches_published_inverse_alpha_candidate() {
        let audit = reproduce();
        assert!((audit.inverse_alpha_candidate - 137.035_999_189_566_16).abs() < 1e-12);
        assert!((audit.alpha_candidate - 1.0 / 137.035_999_189_566_16).abs() < 1e-15);
        assert!((audit.difference_in_standard_uncertainties - 0.5984).abs() < 0.001);
    }
}
