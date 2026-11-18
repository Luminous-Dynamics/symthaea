// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Exact ground-truth oracle for the finite D4 and polynomial fixtures.
// This is an evaluation oracle, not a learned symmetry-discovery method.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

const FIXTURE: &str =
    include_str!("../../symmetry-discovery-benchmark-001/fixtures/v0.1/cases.json");

type Matrix2 = [[i64; 2]; 2];
type Point2 = [f64; 2];
type Exponents2 = [u32; 2];

#[derive(Debug, Clone, Deserialize)]
struct Manifest {
    schema_version: String,
    benchmark_id: String,
    freeze_status: String,
    ambient_group: AmbientGroup,
    cases: Vec<Case>,
    split_contract: SplitContract,
    benchmark_status: BenchmarkStatus,
}

#[derive(Debug, Clone, Deserialize)]
struct AmbientGroup {
    group_id: String,
    elements: Vec<GroupElement>,
    multiplication_contract: String,
}

#[derive(Debug, Clone, Deserialize)]
struct GroupElement {
    id: String,
    matrix: Matrix2,
}

#[derive(Debug, Clone, Deserialize)]
struct Case {
    case_id: String,
    track: String,
    target_subgroup: Vec<String>,
    expected_subgroup_label: String,
    geometry: Option<Geometry>,
    ambient_group_id: Option<String>,
    polynomial: Option<Polynomial>,
}

#[derive(Debug, Clone, Deserialize)]
struct Geometry {
    dimension: usize,
    points: Vec<Point2>,
    point_order_semantics: String,
}

#[derive(Debug, Clone, Deserialize)]
struct Polynomial {
    variables: Vec<String>,
    terms: Vec<PolynomialTerm>,
    exact_rational: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct PolynomialTerm {
    exponents: Exponents2,
    coefficient: i64,
}

#[derive(Debug, Clone, Deserialize)]
struct SplitContract {
    discovery_split_id: String,
    calibration_split_id: String,
    final_holdout_split_id: String,
}

#[derive(Debug, Clone, Deserialize)]
struct BenchmarkStatus {
    fixture_validation: String,
    external_baseline_runs: String,
    results: String,
    claim_boundary: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CaseResult {
    pub case_id: String,
    pub track: String,
    pub expected_subgroup_label: String,
    pub actual_subgroup_label: String,
    pub expected_elements: Vec<String>,
    pub actual_elements: Vec<String>,
    pub exact_match: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OracleReport {
    pub benchmark_id: String,
    pub ambient_group_id: String,
    pub ambient_group_order: usize,
    pub group_laws_valid: bool,
    pub cases: Vec<CaseResult>,
    pub all_exact_labels_match: bool,
    pub split_ids_are_distinct: bool,
    pub benchmark_claim_boundary_preserved: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OracleError {
    Json(String),
    InvalidManifest(String),
    DuplicateGroupElement(String),
    NonInvertibleGroupElement(String),
    MissingIdentity,
    GroupNotClosed { left: String, right: String },
    MissingInverse(String),
    InvalidSubgroup(String),
    InvalidCase(String),
    DuplicatePoint(String),
    InvalidPolynomial(String),
    ArithmeticOverflow(String),
    TargetMismatch {
        case_id: String,
        expected: Vec<String>,
        actual: Vec<String>,
    },
}

impl fmt::Display for OracleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for OracleError {}

fn multiply_matrix(left: Matrix2, right: Matrix2) -> Matrix2 {
    let mut product = [[0_i64; 2]; 2];
    for row in 0..2 {
        for column in 0..2 {
            for inner in 0..2 {
                product[row][column] += left[row][inner] * right[inner][column];
            }
        }
    }
    product
}

fn determinant(matrix: Matrix2) -> i64 {
    matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
}

fn identity() -> Matrix2 {
    [[1, 0], [0, 1]]
}

fn validate_group(group: &AmbientGroup) -> Result<BTreeMap<String, Matrix2>, OracleError> {
    if group.elements.is_empty() {
        return Err(OracleError::InvalidManifest(
            "ambient group has no elements".to_owned(),
        ));
    }
    let mut by_id = BTreeMap::new();
    let mut by_matrix = BTreeMap::new();
    for element in &group.elements {
        if element.id.trim().is_empty() {
            return Err(OracleError::InvalidManifest(
                "empty group element ID".to_owned(),
            ));
        }
        if by_id.insert(element.id.clone(), element.matrix).is_some() {
            return Err(OracleError::DuplicateGroupElement(element.id.clone()));
        }
        if by_matrix
            .insert(matrix_key(element.matrix), element.id.clone())
            .is_some()
        {
            return Err(OracleError::InvalidManifest(format!(
                "duplicate matrix for element {}",
                element.id
            )));
        }
        if determinant(element.matrix).abs() != 1 {
            return Err(OracleError::NonInvertibleGroupElement(
                element.id.clone(),
            ));
        }
    }
    if !by_matrix.contains_key(&matrix_key(identity())) {
        return Err(OracleError::MissingIdentity);
    }

    for left in &group.elements {
        for right in &group.elements {
            let product = multiply_matrix(left.matrix, right.matrix);
            let Some(_product_id) = by_matrix.get(&matrix_key(product)) else {
                return Err(OracleError::GroupNotClosed {
                    left: left.id.clone(),
                    right: right.id.clone(),
                });
            };
        }
    }
    for element in &group.elements {
        if !group.elements.iter().any(|candidate| {
            multiply_matrix(element.matrix, candidate.matrix) == identity()
                && multiply_matrix(candidate.matrix, element.matrix) == identity()
        }) {
            return Err(OracleError::MissingInverse(element.id.clone()));
        }
    }
    Ok(by_id)
}

fn matrix_key(matrix: Matrix2) -> [i64; 4] {
    [
        matrix[0][0],
        matrix[0][1],
        matrix[1][0],
        matrix[1][1],
    ]
}

fn validate_subgroup(
    group: &AmbientGroup,
    by_id: &BTreeMap<String, Matrix2>,
    case: &Case,
) -> Result<Vec<String>, OracleError> {
    let elements: BTreeSet<String> = case.target_subgroup.iter().cloned().collect();
    if elements.len() != case.target_subgroup.len() || elements.is_empty() {
        return Err(OracleError::InvalidSubgroup(case.case_id.clone()));
    }
    if !elements.contains("e") && !group.elements.iter().any(|element| {
        element.matrix == identity()
            && elements.contains(&element.id)
    }) {
        return Err(OracleError::InvalidSubgroup(case.case_id.clone()));
    }
    for id in &elements {
        if !by_id.contains_key(id) {
            return Err(OracleError::InvalidSubgroup(format!(
                "{} references unknown element {}",
                case.case_id, id
            )));
        }
    }
    for left in &elements {
        for right in &elements {
            let product = multiply_matrix(by_id[left], by_id[right]);
            let product_id = group
                .elements
                .iter()
                .find(|candidate| candidate.matrix == product)
                .map(|candidate| candidate.id.as_str())
                .ok_or_else(|| OracleError::InvalidSubgroup(case.case_id.clone()))?;
            if !elements.contains(product_id) {
                return Err(OracleError::InvalidSubgroup(format!(
                    "{} is not closed under {} * {}",
                    case.case_id, left, right
                )));
            }
        }
    }
    let mut sorted: Vec<String> = elements.into_iter().collect();
    sorted.sort();
    Ok(sorted)
}

fn transform_point(matrix: Matrix2, point: Point2) -> Point2 {
    [
        matrix[0][0] as f64 * point[0] + matrix[0][1] as f64 * point[1],
        matrix[1][0] as f64 * point[0] + matrix[1][1] as f64 * point[1],
    ]
}

fn point_set_stabilizer(
    group: &AmbientGroup,
    geometry: &Geometry,
    case_id: &str,
) -> Result<Vec<String>, OracleError> {
    if geometry.dimension != 2
        || geometry.point_order_semantics != "unordered set"
        || geometry.points.is_empty()
    {
        return Err(OracleError::InvalidCase(case_id.to_owned()));
    }
    let mut seen = BTreeSet::new();
    for point in &geometry.points {
        if !point[0].is_finite() || !point[1].is_finite() {
            return Err(OracleError::InvalidCase(case_id.to_owned()));
        }
        // For this signed-permutation ambient group, multiplication changes
        // only signs and coordinate order. IEEE equality is exact here;
        // no tolerance may promote the near-symmetry control.
        let key = (point[0].to_bits(), point[1].to_bits());
        if !seen.insert(key) {
            return Err(OracleError::DuplicatePoint(case_id.to_owned()));
        }
    }

    let point_set: BTreeSet<(u64, u64)> = geometry
        .points
        .iter()
        .map(|point| canonical_point(*point))
        .collect();

    let mut stabilizer = Vec::new();
    for element in &group.elements {
        let transformed: BTreeSet<(u64, u64)> = geometry
            .points
            .iter()
            .map(|point| canonical_point(transform_point(element.matrix, *point)))
            .collect();
        if transformed == point_set {
            stabilizer.push(element.id.clone());
        }
    }
    stabilizer.sort();
    Ok(stabilizer)
}

fn canonical_point(point: Point2) -> (u64, u64) {
    // Normalize signed zero; it is the same geometric point.
    let x = if point[0] == 0.0 { 0.0 } else { point[0] };
    let y = if point[1] == 0.0 { 0.0 } else { point[1] };
    (x.to_bits(), y.to_bits())
}

fn polynomial_map(
    polynomial: &Polynomial,
    case_id: &str,
) -> Result<BTreeMap<Exponents2, i64>, OracleError> {
    if !polynomial.exact_rational
        || polynomial.variables != ["x".to_owned(), "y".to_owned()]
        || polynomial.terms.is_empty()
    {
        return Err(OracleError::InvalidPolynomial(case_id.to_owned()));
    }
    let mut map = BTreeMap::new();
    for term in &polynomial.terms {
        if term.exponents[0] > 32 || term.exponents[1] > 32 {
            return Err(OracleError::InvalidPolynomial(case_id.to_owned()));
        }
        let previous = map.entry(term.exponents).or_insert(0_i64);
        *previous = previous
            .checked_add(term.coefficient)
            .ok_or_else(|| OracleError::ArithmeticOverflow(case_id.to_owned()))?;
    }
    map.retain(|_, coefficient| *coefficient != 0);
    Ok(map)
}

fn signed_permutation_image(
    matrix: Matrix2,
    exponents: Exponents2,
) -> Result<(Exponents2, i64), OracleError> {
    let row0 = matrix[0];
    let row1 = matrix[1];
    let column0 = row0.iter().position(|value| *value != 0);
    let column1 = row1.iter().position(|value| *value != 0);
    let (Some(column0), Some(column1)) = (column0, column1) else {
        return Err(OracleError::InvalidManifest(
            "expected signed-permutation matrices".to_owned(),
        ));
    };
    let sign0 = row0[column0];
    let sign1 = row1[column1];
    if sign0.abs() != 1 || sign1.abs() != 1 || column0 == column1 {
        return Err(OracleError::InvalidManifest(
            "expected signed-permutation matrices".to_owned(),
        ));
    }
    let mut transformed = [0_u32; 2];
    transformed[column0] = exponents[0];
    transformed[column1] = exponents[1];
    let mut sign = 1_i64;
    if sign0 < 0 && exponents[0] % 2 == 1 {
        sign = -sign;
    }
    if sign1 < 0 && exponents[1] % 2 == 1 {
        sign = -sign;
    }
    Ok((transformed, sign))
}

fn transform_polynomial(
    polynomial: &Polynomial,
    matrix: Matrix2,
    case_id: &str,
) -> Result<BTreeMap<Exponents2, i64>, OracleError> {
    let original = polynomial_map(polynomial, case_id)?;
    let mut transformed = BTreeMap::new();
    for (exponents, coefficient) in original {
        let (image, sign) = signed_permutation_image(matrix, exponents)?;
        let signed_coefficient = coefficient
            .checked_mul(sign)
            .ok_or_else(|| OracleError::ArithmeticOverflow(case_id.to_owned()))?;
        let value = transformed.entry(image).or_insert(0_i64);
        *value = value
            .checked_add(signed_coefficient)
            .ok_or_else(|| OracleError::ArithmeticOverflow(case_id.to_owned()))?;
    }
    transformed.retain(|_, coefficient| *coefficient != 0);
    Ok(transformed)
}

fn polynomial_stabilizer(
    group: &AmbientGroup,
    polynomial: &Polynomial,
    case_id: &str,
) -> Result<Vec<String>, OracleError> {
    let original = polynomial_map(polynomial, case_id)?;
    let mut stabilizer = Vec::new();
    for element in &group.elements {
        if transform_polynomial(polynomial, element.matrix, case_id)? == original {
            stabilizer.push(element.id.clone());
        }
    }
    stabilizer.sort();
    Ok(stabilizer)
}

fn subgroup_label(order: usize) -> &'static str {
    match order {
        1 => "C1",
        2 => "C2",
        4 => "C4",
        8 => "D4",
        _ => "unclassified",
    }
}

fn validate_manifest(manifest: &Manifest) -> Result<OracleReport, OracleError> {
    if manifest.schema_version != "0.1.0"
        || manifest.freeze_status != "protocol_fixtures_not_benchmark_results"
        || manifest.ambient_group.group_id != "D4"
        || manifest.ambient_group.multiplication_contract.trim().is_empty()
        || manifest.cases.is_empty()
    {
        return Err(OracleError::InvalidManifest(
            "unexpected schema, freeze status, ambient group, or empty case set".to_owned(),
        ));
    }
    if manifest.split_contract.discovery_split_id
        == manifest.split_contract.calibration_split_id
        || manifest.split_contract.discovery_split_id
            == manifest.split_contract.final_holdout_split_id
        || manifest.split_contract.calibration_split_id
            == manifest.split_contract.final_holdout_split_id
    {
        return Err(OracleError::InvalidManifest(
            "discovery, calibration and final holdout split IDs must be distinct".to_owned(),
        ));
    }
    let ids = validate_group(&manifest.ambient_group)?;
    let mut results = Vec::new();
    for case in &manifest.cases {
        let expected = validate_subgroup(&manifest.ambient_group, &ids, case)?;
        let actual = match (&case.geometry, &case.polynomial) {
            (Some(geometry), None) if case.track == "exact_finite_structural_stabilizer" => {
                point_set_stabilizer(&manifest.ambient_group, geometry, &case.case_id)?
            }
            (None, Some(polynomial))
                if case.track == "finite_reference_group_regression_discovery" =>
            {
                if case.ambient_group_id.as_deref() != Some(manifest.ambient_group.group_id.as_str()) {
                    return Err(OracleError::InvalidCase(case.case_id.clone()));
                }
                polynomial_stabilizer(&manifest.ambient_group, polynomial, &case.case_id)?
            }
            _ => return Err(OracleError::InvalidCase(case.case_id.clone())),
        };
        if actual != expected {
            return Err(OracleError::TargetMismatch {
                case_id: case.case_id.clone(),
                expected,
                actual,
            });
        }
        let label = subgroup_label(actual.len());
        if case.expected_subgroup_label != label {
            return Err(OracleError::InvalidCase(format!(
                "{} label {} disagrees with subgroup order {}",
                case.case_id,
                case.expected_subgroup_label,
                actual.len()
            )));
        }
        results.push(CaseResult {
            case_id: case.case_id.clone(),
            track: case.track.clone(),
            expected_subgroup_label: case.expected_subgroup_label.clone(),
            actual_subgroup_label: label.to_owned(),
            expected_elements: expected.clone(),
            actual_elements: actual,
            exact_match: true,
        });
    }
    results.sort_by(|left, right| left.case_id.cmp(&right.case_id));
    Ok(OracleReport {
        benchmark_id: manifest.benchmark_id.clone(),
        ambient_group_id: manifest.ambient_group.group_id.clone(),
        ambient_group_order: manifest.ambient_group.elements.len(),
        group_laws_valid: true,
        all_exact_labels_match: true,
        split_ids_are_distinct: true,
        benchmark_claim_boundary_preserved: manifest
            .benchmark_status
            .results
            .eq("none")
            && manifest
                .benchmark_status
                .fixture_validation
                .contains("authored"),
        cases: results,
    })
}

pub fn run_frozen_oracle() -> Result<OracleReport, OracleError> {
    let manifest: Manifest = serde_json::from_str(FIXTURE)
        .map_err(|error| OracleError::Json(error.to_string()))?;
    validate_manifest(&manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_manifest_has_valid_group_laws_and_all_five_exact_labels() {
        let report = run_frozen_oracle().unwrap();
        assert_eq!(report.ambient_group_order, 8);
        assert!(report.group_laws_valid);
        assert!(report.split_ids_are_distinct);
        assert!(report.all_exact_labels_match);
        assert_eq!(report.cases.len(), 5);
        assert_eq!(
            report
                .cases
                .iter()
                .map(|case| case.actual_subgroup_label.as_str())
                .collect::<Vec<_>>(),
            vec!["C1", "C4", "D4", "C4", "D4"]
        );
    }

    #[test]
    fn near_symmetry_is_exactly_trivial_not_tolerance_promoted() {
        let report = run_frozen_oracle().unwrap();
        let near = report
            .cases
            .iter()
            .find(|case| case.case_id == "D4-NEAR-SYMMETRY-BREAK-001")
            .unwrap();
        assert_eq!(near.actual_elements, vec!["e".to_owned()]);
        assert_eq!(near.actual_subgroup_label, "C1");
    }

    #[test]
    fn chiral_point_set_rejects_every_reflection() {
        let report = run_frozen_oracle().unwrap();
        let chiral = report
            .cases
            .iter()
            .find(|case| case.case_id == "C4-CHIRAL-ORBIT-001")
            .unwrap();
        assert_eq!(
            chiral.actual_elements,
            vec![
                "e".to_owned(),
                "r180".to_owned(),
                "r270".to_owned(),
                "r90".to_owned()
            ]
        );
    }

    #[test]
    fn polynomial_targets_separate_d4_from_c4_exactly() {
        let report = run_frozen_oracle().unwrap();
        let d4 = report
            .cases
            .iter()
            .find(|case| case.case_id == "D4-REGRESSION-D4-001")
            .unwrap();
        let c4 = report
            .cases
            .iter()
            .find(|case| case.case_id == "D4-REGRESSION-C4-001")
            .unwrap();
        assert_eq!(d4.actual_subgroup_label, "D4");
        assert_eq!(c4.actual_subgroup_label, "C4");
        assert!(d4.actual_elements.contains(&"fdiag".to_owned()));
        assert!(!c4.actual_elements.contains(&"fx".to_owned()));
    }

    #[test]
    fn subgroup_order_label_is_not_inferred_from_case_name() {
        assert_eq!(subgroup_label(1), "C1");
        assert_eq!(subgroup_label(2), "C2");
        assert_eq!(subgroup_label(4), "C4");
        assert_eq!(subgroup_label(8), "D4");
        assert_eq!(subgroup_label(6), "unclassified");
    }
}
