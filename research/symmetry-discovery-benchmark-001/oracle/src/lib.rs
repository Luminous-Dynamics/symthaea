// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exact, dependency-light semantics around the frozen D4 benchmark manifest.
//! JSON parsing is delegated to serde_json; all group, point-set and
//! polynomial calculations use exact integer arithmetic.

use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

type Matrix2 = [[i128; 2]; 2];
type Point = (i128, i128);
type Polynomial = BTreeMap<(u32, u32), i128>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StabilizerWitnessV1 {
    PointPermutation {
        source_to_target: Vec<usize>,
    },
    MovedPoint {
        source: (i128, i128),
        transformed: (i128, i128),
    },
    PolynomialIdentity {
        transformed_terms: Vec<((u32, u32), i128)>,
    },
    PolynomialCoefficientMismatch {
        exponents: (u32, u32),
        original_coefficient: i128,
        transformed_coefficient: i128,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ElementDecisionV1 {
    pub element_id: String,
    pub accepted: bool,
    pub witness: StabilizerWitnessV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StabilizerCertificateV1 {
    pub case_id: String,
    pub accepted_subgroup: Vec<String>,
    pub decisions: Vec<ElementDecisionV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum SubgroupAxiomViolationV1 {
    MissingIdentity { identity_element_id: String },
    MissingInverse {
        element_id: String,
        inverse_element_id: String,
    },
    MissingProduct {
        left_element_id: String,
        right_element_id: String,
        product_element_id: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProposalElementAuditV1 {
    pub element_id: String,
    pub proposed: bool,
    pub exact: bool,
    /// For false promotions this is a counterexample to invariance.
    /// For missed symmetries this is a constructive acceptance witness.
    pub witness: StabilizerWitnessV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SubgroupProposalAuditV1 {
    pub case_id: String,
    pub proposed_elements: Vec<String>,
    pub exact_stabilizer: Vec<String>,
    pub group_axioms_satisfied: bool,
    pub axiom_violations: Vec<SubgroupAxiomViolationV1>,
    pub true_positive_elements: Vec<String>,
    pub false_promotion_elements: Vec<String>,
    pub missed_symmetry_elements: Vec<String>,
    pub true_negative_elements: Vec<String>,
    pub decisions: Vec<ProposalElementAuditV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Element {
    id: String,
    matrix: Matrix2,
}

#[derive(Debug)]
pub struct Manifest {
    root: Value,
    elements: Vec<Element>,
    by_matrix: BTreeMap<String, String>,
    multiplication: BTreeMap<(String, String), String>,
    inverses: BTreeMap<String, String>,
}

fn field<'a>(value: &'a Value, name: &str) -> Result<&'a Value, String> {
    value
        .get(name)
        .ok_or_else(|| format!("missing required JSON field: {name}"))
}

fn string_field(value: &Value, name: &str) -> Result<String, String> {
    field(value, name)?
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("field {name} must be a string"))
}

fn integer(value: &Value, context: &str) -> Result<i128, String> {
    value
        .as_i64()
        .map(i128::from)
        .ok_or_else(|| format!("{context} must be an exact signed integer"))
}

fn matrix_key(matrix: Matrix2) -> String {
    format!(
        "{},{},{},{}",
        matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    )
}

fn matrix_from_json(value: &Value, context: &str) -> Result<Matrix2, String> {
    let rows = value
        .as_array()
        .ok_or_else(|| format!("{context} must be a 2x2 matrix"))?;
    if rows.len() != 2 {
        return Err(format!("{context} must have two rows"));
    }
    let mut matrix = [[0_i128; 2]; 2];
    for (row_index, row) in rows.iter().enumerate() {
        let columns = row
            .as_array()
            .ok_or_else(|| format!("{context}[{row_index}] must be an array"))?;
        if columns.len() != 2 {
            return Err(format!("{context}[{row_index}] must have two entries"));
        }
        for (column_index, item) in columns.iter().enumerate() {
            matrix[row_index][column_index] =
                integer(item, &format!("{context}[{row_index}][{column_index}]"))?;
        }
    }
    Ok(matrix)
}

fn power_of_ten(exponent: u32) -> Result<i128, String> {
    let mut value = 1_i128;
    for _ in 0..exponent {
        value = value
            .checked_mul(10)
            .ok_or_else(|| "decimal scale overflow".to_owned())?;
    }
    Ok(value)
}

/// Convert a JSON decimal number to an exact integer at the requested scale.
/// No binary floating-point conversion or tolerance is used.
fn decimal_scaled(value: &Value, scale: i128, context: &str) -> Result<i128, String> {
    let number = value
        .as_number()
        .ok_or_else(|| format!("{context} must be a JSON number"))?;
    let text = number.to_string();
    let (mantissa, exponent) = match text.find(|character| character == 'e' || character == 'E') {
        Some(index) => {
            let exponent = text[index + 1..]
                .parse::<i32>()
                .map_err(|_| format!("{context} has an invalid decimal exponent"))?;
            (&text[..index], exponent)
        }
        None => (text.as_str(), 0),
    };

    let negative = mantissa.starts_with('-');
    let unsigned = mantissa.trim_start_matches(|character| character == '-' || character == '+');
    let (whole, fractional) = match unsigned.split_once('.') {
        Some((whole, fractional)) => (whole, fractional),
        None => (unsigned, ""),
    };
    let digits = format!("{whole}{fractional}");
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!("{context} is not a finite decimal number"));
    }
    let mut numerator = digits
        .parse::<i128>()
        .map_err(|_| format!("{context} exceeds the exact integer parser range"))?;
    if negative {
        numerator = -numerator;
    }

    let decimal_power = exponent
        .checked_sub(i32::try_from(fractional.len()).map_err(|_| {
            format!("{context} has too many decimal places")
        })?)
        .ok_or_else(|| format!("{context} decimal exponent overflow"))?;

    let scaled_numerator = numerator
        .checked_mul(scale)
        .ok_or_else(|| format!("{context} scaled numerator overflow"))?;
    if decimal_power >= 0 {
        scaled_numerator
            .checked_mul(power_of_ten(decimal_power as u32)?)
            .ok_or_else(|| format!("{context} scaled value overflow"))
    } else {
        let denominator = power_of_ten(decimal_power.unsigned_abs())?;
        if scaled_numerator % denominator != 0 {
            return Err(format!(
                "{context} is not exactly representable at scale {scale}"
            ));
        }
        Ok(scaled_numerator / denominator)
    }
}

fn point_from_json(value: &Value, scale: i128, context: &str) -> Result<Point, String> {
    let coordinates = value
        .as_array()
        .ok_or_else(|| format!("{context} must be a 2-vector"))?;
    if coordinates.len() != 2 {
        return Err(format!("{context} must contain exactly two coordinates"));
    }
    Ok((
        decimal_scaled(&coordinates[0], scale, &format!("{context}.x"))?,
        decimal_scaled(&coordinates[1], scale, &format!("{context}.y"))?,
    ))
}

fn point_apply(matrix: Matrix2, point: Point) -> Result<Point, String> {
    let x = matrix[0][0]
        .checked_mul(point.0)
        .and_then(|left| {
            matrix[0][1]
                .checked_mul(point.1)
                .and_then(|right| left.checked_add(right))
        })
        .ok_or_else(|| "point action overflow in x coordinate".to_owned())?;
    let y = matrix[1][0]
        .checked_mul(point.0)
        .and_then(|left| {
            matrix[1][1]
                .checked_mul(point.1)
                .and_then(|right| left.checked_add(right))
        })
        .ok_or_else(|| "point action overflow in y coordinate".to_owned())?;
    Ok((x, y))
}

fn multiply(left: Matrix2, right: Matrix2) -> Result<Matrix2, String> {
    let mut product = [[0_i128; 2]; 2];
    for row in 0..2 {
        for column in 0..2 {
            let first = left[row][0]
                .checked_mul(right[0][column])
                .ok_or_else(|| format!("matrix product overflow at ({row}, {column})"))?;
            let second = left[row][1]
                .checked_mul(right[1][column])
                .ok_or_else(|| format!("matrix product overflow at ({row}, {column})"))?;
            product[row][column] = first
                .checked_add(second)
                .ok_or_else(|| format!("matrix product sum overflow at ({row}, {column})"))?;
        }
    }
    Ok(product)
}

fn transpose(matrix: Matrix2) -> Matrix2 {
    [
        [matrix[0][0], matrix[1][0]],
        [matrix[0][1], matrix[1][1]],
    ]
}

/// Verify A^T G A = G using exact checked integer arithmetic.
/// Unlike the Euclidean-only helper, this supports a non-orthogonal
/// coordinate representation together with its congruently transformed metric.
fn validate_metric_isometry(matrix: Matrix2, metric: Matrix2) -> Result<(), String> {
    let pulled_back = multiply(multiply(transpose(matrix), metric)?, matrix)?;
    if pulled_back != metric {
        return Err(format!(
            "metric-isometry defect: A^T G A = {pulled_back:?}, expected {metric:?}"
        ));
    }
    Ok(())
}

fn validate_orthogonal_element(element: &Element) -> Result<(), String> {
    for first_column in 0..2 {
        for second_column in 0..2 {
            let mut inner_product = 0_i128;
            for row in 0..2 {
                let term = element.matrix[row][first_column]
                    .checked_mul(element.matrix[row][second_column])
                    .ok_or_else(|| {
                        format!("orthogonality product overflow for {}", element.id)
                    })?;
                inner_product = inner_product.checked_add(term).ok_or_else(|| {
                    format!("orthogonality sum overflow for {}", element.id)
                })?;
            }
            let expected = if first_column == second_column { 1 } else { 0 };
            if inner_product != expected {
                return Err(format!(
                    "element {} is not orthogonal: column inner product ({first_column}, {second_column}) is {inner_product}, expected {expected}",
                    element.id
                ));
            }
        }
    }
    Ok(())
}

impl Manifest {
    pub fn parse() -> Result<Self, String> {
        let root: Value = serde_json::from_str(include_str!("../../fixtures/v0.1/cases.json"))
            .map_err(|error| format!("invalid frozen case manifest JSON: {error}"))?;
        if string_field(&root, "benchmark_id")? != "LUMINOUS-SYMM-QUALIFIED-DISCOVERY-001" {
            return Err("unexpected benchmark_id".to_owned());
        }

        let ambient = field(&root, "ambient_group")?;
        let element_values = field(ambient, "elements")?
            .as_array()
            .ok_or_else(|| "ambient_group.elements must be an array".to_owned())?;
        let mut elements = Vec::with_capacity(element_values.len());
        let mut by_matrix = BTreeMap::new();
        let mut ids = BTreeSet::new();
        for (index, value) in element_values.iter().enumerate() {
            let id = string_field(value, "id")?;
            if !ids.insert(id.clone()) {
                return Err(format!("duplicate group element ID: {id}"));
            }
            let matrix = matrix_from_json(
                field(value, "matrix")?,
                &format!("ambient_group.elements[{index}].matrix"),
            )?;
            if by_matrix.insert(matrix_key(matrix), id.clone()).is_some() {
                return Err(format!("duplicate group matrix for element: {id}"));
            }
            elements.push(Element { id, matrix });
        }

        let mut multiplication = BTreeMap::new();
        let table = field(ambient, "multiplication_table")?
            .as_array()
            .ok_or_else(|| "multiplication_table must be an array".to_owned())?;
        for item in table {
            let left = string_field(item, "left_element_id")?;
            let right = string_field(item, "right_element_id")?;
            let product = string_field(item, "product_element_id")?;
            if multiplication
                .insert((left.clone(), right.clone()), product)
                .is_some()
            {
                return Err(format!("duplicate multiplication entry ({left}, {right})"));
            }
        }

        let mut inverses = BTreeMap::new();
        let inverse_table = field(ambient, "inverse_table")?
            .as_array()
            .ok_or_else(|| "inverse_table must be an array".to_owned())?;
        for item in inverse_table {
            let element = string_field(item, "element_id")?;
            let inverse = string_field(item, "inverse_element_id")?;
            if inverses.insert(element.clone(), inverse).is_some() {
                return Err(format!("duplicate inverse entry for {element}"));
            }
        }

        Ok(Self {
            root,
            elements,
            by_matrix,
            multiplication,
            inverses,
        })
    }

    pub fn verify_group_axioms(&self) -> Result<(), String> {
        let identity_id = string_field(field(&self.root, "ambient_group")?, "identity_element_id")?;
        let declared_order = field(field(&self.root, "ambient_group")?, "group_order")?
            .as_u64()
            .ok_or_else(|| "ambient_group.group_order must be an unsigned integer".to_owned())?;
        if declared_order != self.elements.len() as u64 {
            return Err(format!(
                "ambient_group.group_order={declared_order}, but {} matrices are declared",
                self.elements.len()
            ));
        }
        if self.elements.len() != 8 || self.multiplication.len() != 64 || self.inverses.len() != 8 {
            return Err(
                "D4 manifest must contain 8 elements, 64 products and 8 inverses".to_owned(),
            );
        }
        let identity = self
            .elements
            .iter()
            .find(|element| element.id == identity_id)
            .ok_or_else(|| "identity ID is not a declared element".to_owned())?;

        for element in &self.elements {
            validate_orthogonal_element(element)?;
        }

        for left in &self.elements {
            for right in &self.elements {
                let product_matrix = multiply(left.matrix, right.matrix)?;
                let product_id = self
                    .by_matrix
                    .get(&matrix_key(product_matrix))
                    .ok_or_else(|| format!("D4 not closed at {} * {}", left.id, right.id))?;
                let table_id = self
                    .multiplication
                    .get(&(left.id.clone(), right.id.clone()))
                    .ok_or_else(|| {
                        format!("missing multiplication entry {} * {}", left.id, right.id)
                    })?;
                if product_id != table_id {
                    return Err(format!(
                        "multiplication table mismatch {} * {}: matrix gives {product_id}, table gives {table_id}",
                        left.id, right.id
                    ));
                }
                if left.id == identity_id && product_id != &right.id {
                    return Err(format!("left identity fails for {}", right.id));
                }
                if right.id == identity_id && product_id != &left.id {
                    return Err(format!("right identity fails for {}", left.id));
                }
            }
            let inverse_id = self
                .inverses
                .get(&left.id)
                .ok_or_else(|| format!("missing inverse for {}", left.id))?;
            let inverse = self
                .elements
                .iter()
                .find(|element| &element.id == inverse_id)
                .ok_or_else(|| format!("unknown inverse ID {inverse_id}"))?;
            if multiply(left.matrix, inverse.matrix)? != identity.matrix
                || multiply(inverse.matrix, left.matrix)? != identity.matrix
            {
                return Err(format!("two-sided inverse fails for {}", left.id));
            }
        }

        for a in &self.elements {
            for b in &self.elements {
                for c in &self.elements {
                    let ab = self.multiplication[&(a.id.clone(), b.id.clone())].clone();
                    let bc = self.multiplication[&(b.id.clone(), c.id.clone())].clone();
                    let lhs = self.multiplication[&(ab, c.id.clone())].clone();
                    let rhs = self.multiplication[&(a.id.clone(), bc)].clone();
                    if lhs != rhs {
                        return Err(format!(
                            "associativity fails for {}, {}, {}",
                            a.id, b.id, c.id
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    fn declared_subgroup(&self, case: &Value) -> Result<BTreeSet<String>, String> {
        let members = field(case, "target_subgroup")?
            .as_array()
            .ok_or_else(|| "target_subgroup must be an array".to_owned())?;
        let mut set = BTreeSet::new();
        for value in members {
            let id = value
                .as_str()
                .ok_or_else(|| "subgroup member ID must be a string".to_owned())?
                .to_owned();
            if !self.elements.iter().any(|element| element.id == id) {
                return Err(format!("subgroup references unknown element {id}"));
            }
            if !set.insert(id.clone()) {
                return Err(format!("duplicate subgroup member {id}"));
            }
        }
        let identity = string_field(field(&self.root, "ambient_group")?, "identity_element_id")?;
        if !set.contains(&identity) {
            return Err("target subgroup omits the identity".to_owned());
        }
        for left in &set {
            for right in &set {
                let product = self.multiplication[&(left.clone(), right.clone())].clone();
                if !set.contains(&product) {
                    return Err(format!(
                        "target subgroup is not closed: {left} * {right} = {product}"
                    ));
                }
            }
        }
        for element in &set {
            let inverse = self.inverses[element].clone();
            if !set.contains(&inverse) {
                return Err(format!("target subgroup omits inverse {inverse}"));
            }
        }
        let label = string_field(case, "expected_subgroup_label")?;
        let expected_ids: &[&str] = match label.as_str() {
            "D4" => &["e", "r90", "r180", "r270", "fx", "fy", "fdiag", "fanti"],
            "C4" => &["e", "r90", "r180", "r270"],
            "C2_R180" => &["e", "r180"],
            "C2_FX" => &["e", "fx"],
            "C2_FY" => &["e", "fy"],
            "C2_FDIAG" => &["e", "fdiag"],
            "C2_FANTI" => &["e", "fanti"],
            "V4_AXIS" => &["e", "r180", "fx", "fy"],
            "V4_DIAG" => &["e", "r180", "fdiag", "fanti"],
            "C1" => &["e"],
            _ => return Err(format!("unknown fixture subgroup label: {label}")),
        };
        let expected_members: BTreeSet<String> =
            expected_ids.iter().map(|id| (*id).to_owned()).collect();
        if set != expected_members {
            return Err(format!(
                "subgroup label {label} does not match its canonical D4 subgroup: {set:?}"
            ));
        }
        Ok(set)
    }

    fn structural_stabilizer(&self, case: &Value) -> Result<BTreeSet<String>, String> {
        let geometry = field(case, "geometry")?;
        let point_values = field(geometry, "points")?
            .as_array()
            .ok_or_else(|| "geometry.points must be an array".to_owned())?;
        let mut points = BTreeSet::new();
        for (index, value) in point_values.iter().enumerate() {
            let point = point_from_json(value, 1000, &format!("geometry.points[{index}]"))?;
            if !points.insert(point) {
                return Err(format!("duplicate geometric point at index {index}"));
            }
        }
        let mut stabilizer = BTreeSet::new();
        for element in &self.elements {
            let transformed: BTreeSet<Point> = points
                .iter()
                .map(|point| point_apply(element.matrix, *point))
                .collect::<Result<_, _>>()?;
            if transformed == points {
                stabilizer.insert(element.id.clone());
            }
        }
        Ok(stabilizer)
    }

    fn structural_certificate(
        &self,
        case: &Value,
    ) -> Result<StabilizerCertificateV1, String> {
        let geometry = field(case, "geometry")?;
        let point_values = field(geometry, "points")?
            .as_array()
            .ok_or_else(|| "geometry.points must be an array".to_owned())?;
        let points: Vec<Point> = point_values
            .iter()
            .enumerate()
            .map(|(index, value)| point_from_json(value, 1000, &format!("geometry.points[{index}]")))
            .collect::<Result<_, _>>()?;
        let point_set: BTreeSet<Point> = points.iter().copied().collect();
        if point_set.len() != points.len() {
            return Err("structural fixture contains duplicate points".to_owned());
        }

        let mut accepted_subgroup = Vec::new();
        let mut decisions = Vec::with_capacity(self.elements.len());
        for element in &self.elements {
            let transformed: Vec<Point> = points
                .iter()
                .map(|point| point_apply(element.matrix, *point))
                .collect::<Result<_, _>>()?;
            let transformed_set: BTreeSet<Point> = transformed.iter().copied().collect();
            if transformed_set == point_set {
                let source_to_target = transformed
                    .iter()
                    .map(|point| {
                        points
                            .iter()
                            .position(|candidate| candidate == point)
                            .ok_or_else(|| "accepted point image has no target index".to_owned())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                accepted_subgroup.push(element.id.clone());
                decisions.push(ElementDecisionV1 {
                    element_id: element.id.clone(),
                    accepted: true,
                    witness: StabilizerWitnessV1::PointPermutation { source_to_target },
                });
            } else {
                let (source, transformed_point) = points
                    .iter()
                    .copied()
                    .zip(transformed.iter().copied())
                    .find(|(_, image)| !point_set.contains(image))
                    .ok_or_else(|| "rejected point action lacks a moved-point witness".to_owned())?;
                decisions.push(ElementDecisionV1 {
                    element_id: element.id.clone(),
                    accepted: false,
                    witness: StabilizerWitnessV1::MovedPoint {
                        source,
                        transformed: transformed_point,
                    },
                });
            }
        }
        accepted_subgroup.sort();
        decisions.sort_by(|left, right| left.element_id.cmp(&right.element_id));
        Ok(StabilizerCertificateV1 {
            case_id: string_field(case, "case_id")?,
            accepted_subgroup,
            decisions,
        })
    }

    fn polynomial_certificate(
        &self,
        case: &Value,
    ) -> Result<StabilizerCertificateV1, String> {
        let polynomial = self.polynomial_from_json(case)?;
        let mut accepted_subgroup = Vec::new();
        let mut decisions = Vec::with_capacity(self.elements.len());
        for element in &self.elements {
            let transformed = transform_polynomial(&polynomial, element.matrix)?;
            if transformed == polynomial {
                accepted_subgroup.push(element.id.clone());
                decisions.push(ElementDecisionV1 {
                    element_id: element.id.clone(),
                    accepted: true,
                    witness: StabilizerWitnessV1::PolynomialIdentity {
                        transformed_terms: transformed.into_iter().collect(),
                    },
                });
            } else {
                let mut exponents = BTreeSet::new();
                exponents.extend(polynomial.keys().copied());
                exponents.extend(transformed.keys().copied());
                let mismatch = exponents
                    .into_iter()
                    .find_map(|key| {
                        let original_coefficient = *polynomial.get(&key).unwrap_or(&0);
                        let transformed_coefficient = *transformed.get(&key).unwrap_or(&0);
                        (original_coefficient != transformed_coefficient).then_some((
                            key,
                            original_coefficient,
                            transformed_coefficient,
                        ))
                    })
                    .ok_or_else(|| "rejected polynomial action lacks a coefficient witness".to_owned())?;
                decisions.push(ElementDecisionV1 {
                    element_id: element.id.clone(),
                    accepted: false,
                    witness: StabilizerWitnessV1::PolynomialCoefficientMismatch {
                        exponents: mismatch.0,
                        original_coefficient: mismatch.1,
                        transformed_coefficient: mismatch.2,
                    },
                });
            }
        }
        accepted_subgroup.sort();
        decisions.sort_by(|left, right| left.element_id.cmp(&right.element_id));
        Ok(StabilizerCertificateV1 {
            case_id: string_field(case, "case_id")?,
            accepted_subgroup,
            decisions,
        })
    }

    fn verify_certificate(
        &self,
        case: &Value,
        certificate: &StabilizerCertificateV1,
    ) -> Result<(), String> {
        let track = string_field(case, "track")?;
        let replayed = match track.as_str() {
            "exact_finite_structural_stabilizer" => self.structural_certificate(case)?,
            "finite_reference_group_regression_discovery" => self.polynomial_certificate(case)?,
            _ => return Err(format!("unsupported certificate track: {track}")),
        };
        if replayed != *certificate {
            return Err(format!("certificate replay mismatch for {}", certificate.case_id));
        }
        let declared = self.declared_subgroup(case)?;
        let accepted: BTreeSet<String> = certificate.accepted_subgroup.iter().cloned().collect();
        if accepted != declared {
            return Err(format!("certificate subgroup mismatch for {}", certificate.case_id));
        }
        if certificate.decisions.len() != self.elements.len() {
            return Err(format!("certificate omits a group element for {}", certificate.case_id));
        }
        Ok(())
    }

    /// Compare an arbitrary method-proposed element set with the exact
    /// stabilizer. Proposal validity as a subgroup is independent of
    /// element-wise invariance: a proper subgroup may be valid but incomplete,
    /// while a set of plausible elements may fail identity/inverse/closure.
    pub fn audit_proposed_subgroup(
        &self,
        case: &Value,
        proposal_ids: &[String],
    ) -> Result<SubgroupProposalAuditV1, String> {
        let mut proposed = BTreeSet::new();
        for element_id in proposal_ids {
            if !self.elements.iter().any(|element| element.id == *element_id) {
                return Err(format!("proposal references unknown D4 element {element_id}"));
            }
            if !proposed.insert(element_id.clone()) {
                return Err(format!("proposal contains duplicate element {element_id}"));
            }
        }

        let track = string_field(case, "track")?;
        let exact_certificate = match track.as_str() {
            "exact_finite_structural_stabilizer" => self.structural_certificate(case)?,
            "finite_reference_group_regression_discovery" => self.polynomial_certificate(case)?,
            _ => return Err(format!("unsupported proposal audit track: {track}")),
        };
        let exact: BTreeSet<String> = exact_certificate.accepted_subgroup.iter().cloned().collect();

        let ambient = field(&self.root, "ambient_group")?;
        let identity_id = string_field(ambient, "identity_element_id")?;
        let mut axiom_violations = Vec::new();
        if !proposed.contains(&identity_id) {
            axiom_violations.push(SubgroupAxiomViolationV1::MissingIdentity {
                identity_element_id: identity_id,
            });
        }
        for left in &proposed {
            let inverse_id = self
                .inverses
                .get(left)
                .ok_or_else(|| format!("oracle has no inverse entry for {left}"))?;
            if !proposed.contains(inverse_id) {
                axiom_violations.push(SubgroupAxiomViolationV1::MissingInverse {
                    element_id: left.clone(),
                    inverse_element_id: inverse_id.clone(),
                });
            }
            for right in &proposed {
                let product_id = self
                    .multiplication
                    .get(&(left.clone(), right.clone()))
                    .ok_or_else(|| format!("oracle has no product entry for {left} * {right}"))?;
                if !proposed.contains(product_id) {
                    axiom_violations.push(SubgroupAxiomViolationV1::MissingProduct {
                        left_element_id: left.clone(),
                        right_element_id: right.clone(),
                        product_element_id: product_id.clone(),
                    });
                }
            }
        }

        let mut true_positive_elements = Vec::new();
        let mut false_promotion_elements = Vec::new();
        let mut missed_symmetry_elements = Vec::new();
        let mut true_negative_elements = Vec::new();
        let mut decisions = Vec::with_capacity(self.elements.len());

        for exact_decision in &exact_certificate.decisions {
            let is_proposed = proposed.contains(&exact_decision.element_id);
            if is_proposed && exact_decision.accepted {
                true_positive_elements.push(exact_decision.element_id.clone());
            } else if is_proposed && !exact_decision.accepted {
                false_promotion_elements.push(exact_decision.element_id.clone());
            } else if !is_proposed && exact_decision.accepted {
                missed_symmetry_elements.push(exact_decision.element_id.clone());
            } else {
                true_negative_elements.push(exact_decision.element_id.clone());
            }
            decisions.push(ProposalElementAuditV1 {
                element_id: exact_decision.element_id.clone(),
                proposed: is_proposed,
                exact: exact_decision.accepted,
                witness: exact_decision.witness.clone(),
            });
        }

        Ok(SubgroupProposalAuditV1 {
            case_id: exact_certificate.case_id,
            proposed_elements: proposed.into_iter().collect(),
            exact_stabilizer: exact_certificate.accepted_subgroup,
            group_axioms_satisfied: axiom_violations.is_empty(),
            axiom_violations,
            true_positive_elements,
            false_promotion_elements,
            missed_symmetry_elements,
            true_negative_elements,
            decisions,
        })
    }

    /// Recompute every proposal classification, subgroup axiom and
    /// counterexample from the frozen case and reject any edited audit.
    pub fn verify_proposal_audit(
        &self,
        case: &Value,
        audit: &SubgroupProposalAuditV1,
    ) -> Result<(), String> {
        let replayed = self.audit_proposed_subgroup(case, &audit.proposed_elements)?;
        if replayed != *audit {
            return Err(format!("proposal audit replay mismatch for {}", audit.case_id));
        }
        Ok(())
    }

    fn find_case(&self, case_id: &str) -> Result<&Value, String> {
        field(&self.root, "cases")?
            .as_array()
            .ok_or_else(|| "cases must be an array".to_owned())?
            .iter()
            .find(|case| {
                string_field(case, "case_id")
                    .is_ok_and(|candidate| candidate == case_id)
            })
            .ok_or_else(|| format!("unknown benchmark case ID: {case_id}"))
    }

    pub fn case_track_and_split(&self, case_id: &str) -> Result<(String, String), String> {
        let case = self.find_case(case_id)?;
        Ok((
            string_field(case, "track")?,
            string_field(case, "split")?,
        ))
    }

    /// Public manifest-bound proposal audit entry point. It computes exact
    /// ground truth and counterexamples; caller-supplied labels are never
    /// accepted as verifier authority.
    pub fn audit_case_proposal(
        &self,
        case_id: &str,
        proposal_ids: &[String],
    ) -> Result<SubgroupProposalAuditV1, String> {
        let case = self.find_case(case_id)?;
        self.audit_proposed_subgroup(case, proposal_ids)
    }

    /// Replay and verify a proposal audit against the exact manifest case.
    pub fn verify_case_proposal_audit(
        &self,
        case_id: &str,
        audit: &SubgroupProposalAuditV1,
    ) -> Result<(), String> {
        let case = self.find_case(case_id)?;
        self.verify_proposal_audit(case, audit)
    }

    pub fn case_ids_for_split_and_track(
        &self,
        split: &str,
        track: &str,
    ) -> Result<Vec<String>, String> {
        let cases = field(&self.root, "cases")?
            .as_array()
            .ok_or_else(|| "cases must be an array".to_owned())?;
        let mut result = Vec::new();
        for case in cases {
            if string_field(case, "split")? == split
                && string_field(case, "track")? == track
            {
                result.push(string_field(case, "case_id")?);
            }
        }
        result.sort();
        Ok(result)
    }

    fn polynomial_from_json(&self, case: &Value) -> Result<Polynomial, String> {
        let polynomial = field(case, "polynomial")?;
        if field(polynomial, "exact_rational")?.as_bool() != Some(true) {
            return Err("finite exact regression oracle requires exact_rational=true".to_owned());
        }
        let variables = field(polynomial, "variables")?
            .as_array()
            .ok_or_else(|| "polynomial.variables must be an array".to_owned())?;
        if variables.len() != 2
            || variables[0].as_str() != Some("x")
            || variables[1].as_str() != Some("y")
        {
            return Err("finite regression oracle requires variables exactly [x, y]".to_owned());
        }
        let terms = field(polynomial, "terms")?
            .as_array()
            .ok_or_else(|| "polynomial.terms must be an array".to_owned())?;
        if terms.is_empty() {
            return Err("polynomial.terms must not be empty".to_owned());
        }
        if terms.len() > 256 {
            return Err("polynomial term count exceeds the exact-oracle bound 256".to_owned());
        }
        let mut result = Polynomial::new();
        let mut seen_exponents = BTreeSet::new();
        for (index, term) in terms.iter().enumerate() {
            let exponents = field(term, "exponents")?
                .as_array()
                .ok_or_else(|| format!("term {index} exponents must be an array"))?;
            if exponents.len() != 2 {
                return Err(format!("term {index} must have two exponents"));
            }
            let x = u32::try_from(
                exponents[0]
                    .as_u64()
                    .ok_or_else(|| format!("term {index} x exponent must be nonnegative"))?,
            )
            .map_err(|_| format!("term {index} x exponent exceeds u32"))?;
            let y = u32::try_from(
                exponents[1]
                    .as_u64()
                    .ok_or_else(|| format!("term {index} y exponent must be nonnegative"))?,
            )
            .map_err(|_| format!("term {index} y exponent exceeds u32"))?;
            const MAX_TOTAL_DEGREE: u32 = 32;
            if x.checked_add(y).is_none_or(|degree| degree > MAX_TOTAL_DEGREE) {
                return Err(format!(
                    "term {index} total degree exceeds the exact-oracle bound {MAX_TOTAL_DEGREE}"
                ));
            }
            if !seen_exponents.insert((x, y)) {
                return Err(format!("duplicate polynomial monomial ({x}, {y})"));
            }
            let coefficient = integer(field(term, "coefficient")?, "polynomial coefficient")?;
            if coefficient == 0 {
                return Err(format!("polynomial term {index} has a zero coefficient"));
            }
            result.insert((x, y), coefficient);
        }
        if result.is_empty() {
            return Err("polynomial must not be the zero polynomial".to_owned());
        }
        Ok(result)
    }

    fn polynomial_stabilizer(&self, case: &Value) -> Result<BTreeSet<String>, String> {
        let polynomial = self.polynomial_from_json(case)?;
        let mut stabilizer = BTreeSet::new();
        for element in &self.elements {
            if transform_polynomial(&polynomial, element.matrix)? == polynomial {
                stabilizer.insert(element.id.clone());
            }
        }
        Ok(stabilizer)
    }

    pub fn validate_all_cases(&self) -> Result<(usize, usize), String> {
        let cases = field(&self.root, "cases")?
            .as_array()
            .ok_or_else(|| "cases must be an array".to_owned())?;
        let split_contract = field(&self.root, "split_contract")?;
        let split_ids = [
            string_field(split_contract, "discovery_split_id")?,
            string_field(split_contract, "calibration_split_id")?,
            string_field(split_contract, "final_holdout_split_id")?,
        ];
        if split_ids.iter().collect::<BTreeSet<_>>().len() != 3 {
            return Err("discovery, calibration and final-holdout split IDs must be distinct".to_owned());
        }
        let mut ids = BTreeSet::new();
        let mut structural_count = 0;
        let mut regression_count = 0;
        let mut structural_families = BTreeMap::<String, usize>::new();
        let mut split_counts = BTreeMap::<String, usize>::new();
        let mut generated_family_splits = BTreeMap::<(String, String), usize>::new();
        for case in cases {
            let case_id = string_field(case, "case_id")?;
            if !ids.insert(case_id.clone()) {
                return Err(format!("duplicate case ID: {case_id}"));
            }
            let split = string_field(case, "split")?;
            if !["fixture_sanity", "discovery", "calibration", "final_holdout"]
                .contains(&split.as_str())
            {
                return Err(format!("invalid benchmark split {split} in {case_id}"));
            }
            *split_counts.entry(split.clone()).or_insert(0) += 1;
            let declared = self.declared_subgroup(case)?;
            let label = string_field(case, "expected_subgroup_label")?;
            *structural_families.entry(label.clone()).or_insert(0) += 1;
            if case_id.starts_with("GENERATED-") {
                if split == "fixture_sanity" {
                    return Err(format!("generated case {case_id} cannot use fixture_sanity"));
                }
                *generated_family_splits
                    .entry((label, split))
                    .or_insert(0) += 1;
            }
            let track = string_field(case, "track")?;
            let computed = match track.as_str() {
                "exact_finite_structural_stabilizer" => {
                    structural_count += 1;
                    self.structural_stabilizer(case)?
                }
                "finite_reference_group_regression_discovery" => {
                    regression_count += 1;
                    self.polynomial_stabilizer(case)?
                }
                _ => return Err(format!("unsupported benchmark track {track} in {case_id}")),
            };
            if computed != declared {
                return Err(format!(
                    "exact oracle mismatch in {case_id}: computed {computed:?}, declared {declared:?}"
                ));
            }
        }
        if structural_count != 903 || regression_count != 2 {
            return Err(format!(
                "unexpected fixture lane counts: structural={structural_count}, regression={regression_count}"
            ));
        }
        let expected_families = BTreeMap::from([
            ("C1".to_owned(), 91_usize),
            ("C2_FX".to_owned(), 90_usize),
            ("C2_FY".to_owned(), 90_usize),
            ("C2_FDIAG".to_owned(), 90_usize),
            ("C2_FANTI".to_owned(), 90_usize),
            ("C2_R180".to_owned(), 90_usize),
            ("C4".to_owned(), 92_usize),
            ("D4".to_owned(), 92_usize),
            ("V4_AXIS".to_owned(), 90_usize),
            ("V4_DIAG".to_owned(), 90_usize),
        ]);
        if structural_families != expected_families {
            return Err(format!(
                "unexpected subgroup-family counts: {structural_families:?}"
            ));
        }
        let expected_split_counts = BTreeMap::from([
            ("fixture_sanity".to_owned(), 5_usize),
            ("discovery".to_owned(), 300_usize),
            ("calibration".to_owned(), 300_usize),
            ("final_holdout".to_owned(), 300_usize),
        ]);
        if split_counts != expected_split_counts {
            return Err(format!("unexpected split counts: {split_counts:?}"));
        }
        let expected_generated_family_splits: BTreeMap<(String, String), usize> = [
            "D4", "C4", "C2_R180", "C2_FX", "C2_FY", "C2_FDIAG", "C2_FANTI",
            "V4_AXIS", "V4_DIAG", "C1",
        ]
        .into_iter()
        .flat_map(|family| {
            ["discovery", "calibration", "final_holdout"]
                .into_iter()
                .map(move |split| ((family.to_owned(), split.to_owned()), 30_usize))
        })
        .collect();
        if generated_family_splits != expected_generated_family_splits {
            return Err(format!(
                "unexpected per-family split counts: {generated_family_splits:?}"
            ));
        }
        Ok((structural_count, regression_count))
    }
}

fn binomial(n: u32, k: u32) -> Result<i128, String> {
    if k > n {
        return Ok(0);
    }
    let k = k.min(n - k);
    let mut value = 1_i128;
    for index in 0..k {
        value = value
            .checked_mul(i128::from(n - index))
            .ok_or_else(|| "binomial coefficient overflow".to_owned())?;
        value /= i128::from(index + 1);
    }
    Ok(value)
}

fn pow(base: i128, exponent: u32) -> Result<i128, String> {
    base.checked_pow(exponent)
        .ok_or_else(|| "polynomial monomial power overflow".to_owned())
}

fn transform_polynomial(polynomial: &Polynomial, matrix: Matrix2) -> Result<Polynomial, String> {
    let mut transformed = Polynomial::new();
    for ((p, q), coefficient) in polynomial {
        for i in 0..=*p {
            for j in 0..=*q {
                let factors = [
                    *coefficient,
                    binomial(*p, i)?,
                    pow(matrix[0][0], i)?,
                    pow(matrix[0][1], *p - i)?,
                    binomial(*q, j)?,
                    pow(matrix[1][0], j)?,
                    pow(matrix[1][1], *q - j)?,
                ];
                let term = factors.iter().try_fold(1_i128, |acc, factor| {
                    acc.checked_mul(*factor)
                        .ok_or_else(|| "polynomial transform overflow".to_owned())
                })?;
                let key = (i + j, *p - i + *q - j);
                let entry = transformed.entry(key).or_insert(0_i128);
                *entry = entry
                    .checked_add(term)
                    .ok_or_else(|| "polynomial coefficient accumulation overflow".to_owned())?;
            }
        }
    }
    transformed.retain(|_, coefficient| *coefficient != 0);
    Ok(transformed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> Manifest {
        let manifest = Manifest::parse().expect("frozen manifest parses");
        manifest.verify_group_axioms().expect("D4 group axioms");
        manifest
    }

    #[test]
    fn frozen_d4_group_table_is_exact_and_associative() {
        let manifest = manifest();
        assert_eq!(manifest.elements.len(), 8);
        assert_eq!(manifest.multiplication.len(), 64);
        assert_eq!(manifest.inverses.len(), 8);
    }

    #[test]
    fn all_903_structural_stabilizers_match_exact_oracle() {
        let counts = manifest()
            .validate_all_cases()
            .expect("all frozen labels must agree with exact integer oracle");
        assert_eq!(counts.0, 903);
    }

    #[test]
    fn all_ten_distinct_d4_subgroups_are_present_in_the_corpus() {
        let manifest = manifest();
        let cases = field(&manifest.root, "cases").unwrap().as_array().unwrap();
        let mut labels = BTreeSet::new();
        let mut subgroups = BTreeSet::new();
        for case in cases {
            let declared = manifest.declared_subgroup(case).unwrap();
            let label = string_field(case, "expected_subgroup_label").unwrap();
            labels.insert(label);
            subgroups.insert(declared);
        }
        assert_eq!(labels.len(), 10);
        assert_eq!(subgroups.len(), 10);
    }

    #[test]
    fn both_regression_polynomial_stabilizers_match_exact_oracle() {
        let counts = manifest().validate_all_cases().unwrap();
        assert_eq!(counts.1, 2);
    }

    #[test]
    fn square_chiral_and_near_symmetry_controls_are_not_collapsed() {
        let manifest = manifest();
        let cases = field(&manifest.root, "cases").unwrap().as_array().unwrap();
        for (case_id, expected_order) in [
            ("D4-SQUARE-POINTSET-001", 8),
            ("C4-CHIRAL-ORBIT-001", 4),
            ("D4-NEAR-SYMMETRY-BREAK-001", 1),
        ] {
            let case = cases
                .iter()
                .find(|case| string_field(case, "case_id").unwrap() == case_id)
                .unwrap();
            assert_eq!(
                manifest.structural_stabilizer(case).unwrap().len(),
                expected_order
            );
        }
    }

    #[test]
    fn accepted_point_actions_carry_bijections_and_rejected_actions_carry_moved_points() {
        let manifest = manifest();
        let cases = field(&manifest.root, "cases").unwrap().as_array().unwrap();
        for case in cases.iter().filter(|case| {
            string_field(case, "track").unwrap() == "exact_finite_structural_stabilizer"
        }) {
            let certificate = manifest.structural_certificate(case).unwrap();
            manifest.verify_certificate(case, &certificate).unwrap();
            assert_eq!(certificate.decisions.len(), 8);
            for decision in &certificate.decisions {
                match &decision.witness {
                    StabilizerWitnessV1::PointPermutation { source_to_target } => {
                        assert!(decision.accepted);
                        let mut sorted = source_to_target.clone();
                        sorted.sort_unstable();
                        assert_eq!(sorted, (0..source_to_target.len()).collect::<Vec<_>>());
                    }
                    StabilizerWitnessV1::MovedPoint { source, transformed } => {
                        assert!(!decision.accepted);
                        assert_ne!(source, transformed);
                    }
                    _ => panic!("structural certificate has a polynomial witness"),
                }
            }
        }
    }

    #[test]
    fn rejected_polynomial_actions_carry_exact_coefficient_counterexamples() {
        let manifest = manifest();
        let cases = field(&manifest.root, "cases").unwrap().as_array().unwrap();
        for case in cases.iter().filter(|case| {
            string_field(case, "track").unwrap() == "finite_reference_group_regression_discovery"
        }) {
            let certificate = manifest.polynomial_certificate(case).unwrap();
            manifest.verify_certificate(case, &certificate).unwrap();
            for decision in &certificate.decisions {
                match &decision.witness {
                    StabilizerWitnessV1::PolynomialIdentity { .. } => assert!(decision.accepted),
                    StabilizerWitnessV1::PolynomialCoefficientMismatch {
                        original_coefficient,
                        transformed_coefficient,
                        ..
                    } => {
                        assert!(!decision.accepted);
                        assert_ne!(original_coefficient, transformed_coefficient);
                    }
                    _ => panic!("polynomial certificate has a geometric witness"),
                }
            }
        }
    }

    #[test]
    fn forged_point_and_polynomial_witnesses_are_rejected() {
        let manifest = manifest();
        let cases = field(&manifest.root, "cases").unwrap().as_array().unwrap();

        let point_case = cases
            .iter()
            .find(|case| string_field(case, "case_id").unwrap() == "C4-CHIRAL-ORBIT-001")
            .unwrap();
        let mut point_certificate = manifest.structural_certificate(point_case).unwrap();
        manifest.verify_certificate(point_case, &point_certificate).unwrap();
        let rejected_point = point_certificate
            .decisions
            .iter_mut()
            .find(|decision| !decision.accepted)
            .expect("C4 chiral fixture has at least one rejected reflection");
        if let StabilizerWitnessV1::MovedPoint { transformed, .. } =
            &mut rejected_point.witness
        {
            transformed.0 += 1;
        } else {
            panic!("rejected structural action must carry a moved-point witness");
        }
        assert!(manifest.verify_certificate(point_case, &point_certificate).is_err());

        let polynomial_case = cases
            .iter()
            .find(|case| string_field(case, "case_id").unwrap() == "D4-REGRESSION-C4-001")
            .unwrap();
        let mut polynomial_certificate = manifest.polynomial_certificate(polynomial_case).unwrap();
        manifest
            .verify_certificate(polynomial_case, &polynomial_certificate)
            .unwrap();
        let rejected_polynomial = polynomial_certificate
            .decisions
            .iter_mut()
            .find(|decision| !decision.accepted)
            .expect("C4-only polynomial has rejected reflections");
        if let StabilizerWitnessV1::PolynomialCoefficientMismatch {
            transformed_coefficient,
            ..
        } = &mut rejected_polynomial.witness
        {
            *transformed_coefficient += 1;
        } else {
            panic!("rejected polynomial action must carry a coefficient witness");
        }
        assert!(
            manifest
                .verify_certificate(polynomial_case, &polynomial_certificate)
                .is_err()
        );
    }

    #[test]
    fn proposal_audit_distinguishes_overpromotion_from_valid_but_incomplete_subgroups() {
        let oracle = manifest();
        let cases = field(&oracle.root, "cases").unwrap().as_array().unwrap();
        let chiral = cases
            .iter()
            .find(|case| string_field(case, "case_id").unwrap() == "C4-CHIRAL-ORBIT-001")
            .unwrap();
        let d4_proposal = vec![
            "e", "r90", "r180", "r270", "fx", "fy", "fdiag", "fanti",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
        let overpromotion = oracle.audit_proposed_subgroup(chiral, &d4_proposal).unwrap();
        assert!(overpromotion.group_axioms_satisfied);
        assert_eq!(overpromotion.true_positive_elements.len(), 4);
        assert_eq!(overpromotion.false_promotion_elements.len(), 4);
        assert!(overpromotion.missed_symmetry_elements.is_empty());
        for element_id in ["fx", "fy", "fdiag", "fanti"] {
            let decision = overpromotion
                .decisions
                .iter()
                .find(|decision| decision.element_id == element_id)
                .unwrap();
            assert!(decision.proposed);
            assert!(!decision.exact);
            assert!(matches!(
                &decision.witness,
                StabilizerWitnessV1::MovedPoint { .. }
            ));
        }
        oracle.verify_proposal_audit(chiral, &overpromotion).unwrap();

        let square = cases
            .iter()
            .find(|case| string_field(case, "case_id").unwrap() == "D4-SQUARE-POINTSET-001")
            .unwrap();
        let c4_proposal = vec!["e", "r90", "r180", "r270"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let underpromotion = oracle.audit_proposed_subgroup(square, &c4_proposal).unwrap();
        assert!(underpromotion.group_axioms_satisfied);
        assert_eq!(underpromotion.true_positive_elements.len(), 4);
        assert_eq!(underpromotion.missed_symmetry_elements.len(), 4);
        assert!(underpromotion.false_promotion_elements.is_empty());
        for element_id in ["fx", "fy", "fdiag", "fanti"] {
            let decision = underpromotion
                .decisions
                .iter()
                .find(|decision| decision.element_id == element_id)
                .unwrap();
            assert!(!decision.proposed);
            assert!(decision.exact);
            assert!(matches!(
                &decision.witness,
                StabilizerWitnessV1::PointPermutation { .. }
            ));
        }
        oracle.verify_proposal_audit(square, &underpromotion).unwrap();
    }

    #[test]
    fn proposal_audit_rejects_non_subgroup_support_sets_and_forged_witnesses() {
        let oracle = manifest();
        let cases = field(&oracle.root, "cases").unwrap().as_array().unwrap();
        let square = cases
            .iter()
            .find(|case| string_field(case, "case_id").unwrap() == "D4-SQUARE-POINTSET-001")
            .unwrap();

        let malformed = vec!["e", "r90", "fx"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let audit = oracle.audit_proposed_subgroup(square, &malformed).unwrap();
        assert!(!audit.group_axioms_satisfied);
        assert!(
            audit
                .axiom_violations
                .iter()
                .any(|violation| matches!(
                    violation,
                    SubgroupAxiomViolationV1::MissingProduct { .. }
                ))
        );

        let chiral = cases
            .iter()
            .find(|case| string_field(case, "case_id").unwrap() == "C4-CHIRAL-ORBIT-001")
            .unwrap();
        let d4 = vec!["e", "r90", "r180", "r270", "fx", "fy", "fdiag", "fanti"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let mut forged = oracle.audit_proposed_subgroup(chiral, &d4).unwrap();
        let reflection = forged
            .decisions
            .iter_mut()
            .find(|decision| decision.proposed && !decision.exact)
            .unwrap();
        if let StabilizerWitnessV1::MovedPoint { transformed, .. } = &mut reflection.witness {
            transformed.1 += 1;
        } else {
            panic!("false promotion must carry a moved-point counterexample");
        }
        assert!(oracle.verify_proposal_audit(chiral, &forged).is_err());
        let incomplete_d4 = oracle.audit_proposed_subgroup(chiral, &d4[..7]).unwrap();
        assert!(!incomplete_d4.group_axioms_satisfied);
        assert_eq!(
            oracle.audit_proposed_subgroup(chiral, &["not-an-element".to_owned()]),
            Err("proposal references unknown D4 element not-an-element".to_owned())
        );
    }

    #[test]
    fn exact_point_action_rejects_signed_integer_overflow() {
        let reflection_x = [[-1_i128, 0], [0, 1]];
        assert!(point_apply(reflection_x, (i128::MIN, 0)).is_err());
    }

    #[test]
    fn exact_group_multiplication_rejects_integer_overflow() {
        let left = [[i128::MAX, 0], [0, 1]];
        let right = [[2_i128, 0], [0, 1]];
        assert!(multiply(left, right).is_err());
    }

    #[test]
    fn non_orthogonal_conjugate_preserves_abstract_d4_but_fails_euclidean_action() {
        let manifest = manifest();
        let shear = [[1_i128, 1], [0, 1]];
        let inverse_shear = [[1_i128, -1], [0, 1]];

        let mut conjugated = Vec::with_capacity(manifest.elements.len());
        let mut conjugated_by_id = BTreeMap::new();
        for element in &manifest.elements {
            let left = multiply(shear, element.matrix).unwrap();
            let matrix = multiply(left, inverse_shear).unwrap();
            assert!(conjugated_by_id.insert(element.id.clone(), matrix).is_none());
            conjugated.push(Element {
                id: element.id.clone(),
                matrix,
            });
        }

        // Conjugation preserves the complete abstract multiplication table.
        for left in &conjugated {
            for right in &conjugated {
                let product = multiply(left.matrix, right.matrix).unwrap();
                let expected_id =
                    &manifest.multiplication[&(left.id.clone(), right.id.clone())];
                assert_eq!(matrix_key(product), matrix_key(conjugated_by_id[expected_id]));
            }
        }

        // But the conjugated representation no longer acts by Euclidean
        // isometries; the exact group laws are not a substitute for the metric.
        assert!(
            conjugated
                .iter()
                .any(|element| validate_orthogonal_element(element).is_err())
        );
    }

    #[test]
    fn dense_conjugation_preserves_the_pulled_back_metric_exactly() {
        let manifest = manifest();
        let fixture: Value = serde_json::from_str(include_str!(
            "../../fixtures/v0.1/dense_coordinate_representation.json"
        ))
        .expect("dense coordinate fixture is valid JSON");

        let basis = matrix_from_json(field(&fixture, "basis_matrix").unwrap(), "basis").unwrap();
        let inverse_basis =
            matrix_from_json(field(&fixture, "inverse_basis_matrix").unwrap(), "inverse").unwrap();
        let euclidean = matrix_from_json(
            field(&fixture, "canonical_metric").unwrap(),
            "canonical_metric",
        )
        .unwrap();
        let transformed_metric = matrix_from_json(
            field(&fixture, "transformed_metric").unwrap(),
            "transformed_metric",
        )
        .unwrap();
        assert_eq!(
            basis[0][0] * basis[1][1] - basis[0][1] * basis[1][0],
            1
        );
        assert_eq!(
            multiply(basis, inverse_basis).unwrap(),
            [[1, 0], [0, 1]]
        );
        assert_eq!(
            multiply(inverse_basis, basis).unwrap(),
            [[1, 0], [0, 1]]
        );
        assert_eq!(
            multiply(transpose(basis), multiply(euclidean, basis).unwrap()).unwrap(),
            transformed_metric
        );

        let declared_values = field(&fixture, "transformed_group_elements")
            .unwrap()
            .as_array()
            .unwrap();
        let declared: BTreeMap<String, Matrix2> = declared_values
            .iter()
            .map(|value| {
                let id = string_field(value, "element_id").unwrap();
                let matrix = matrix_from_json(field(value, "matrix").unwrap(), &id).unwrap();
                (id, matrix)
            })
            .collect();
        assert_eq!(declared.len(), manifest.elements.len());

        let mut saw_non_euclidean_matrix = false;
        let mut transformed_by_id = BTreeMap::new();
        for element in &manifest.elements {
            let conjugated = multiply(
                multiply(inverse_basis, element.matrix).unwrap(),
                basis,
            )
            .unwrap();
            assert_eq!(conjugated, declared[&element.id]);
            validate_metric_isometry(conjugated, transformed_metric)
                .expect("conjugated action must preserve the pulled-back metric");
            if validate_orthogonal_element(&Element {
                id: element.id.clone(),
                matrix: conjugated,
            })
            .is_err()
            {
                saw_non_euclidean_matrix = true;
            }
            transformed_by_id.insert(element.id.clone(), conjugated);
        }

        assert!(saw_non_euclidean_matrix);
        for left in &manifest.elements {
            for right in &manifest.elements {
                let product = multiply(
                    transformed_by_id[&left.id],
                    transformed_by_id[&right.id],
                )
                .unwrap();
                let expected_id =
                    &manifest.multiplication[&(left.id.clone(), right.id.clone())];
                assert_eq!(product, transformed_by_id[expected_id]);
            }
        }

        // A deliberately stale identity metric must reject some, but not
        // all, transformed actions: identity and the central half-turn remain
        // Euclidean-orthogonal while the dense conjugated reflections/rotations do not.
        let stale_metric_failures = transformed_by_id
            .values()
            .filter(|matrix| validate_metric_isometry(**matrix, euclidean).is_err())
            .count();
        assert!(stale_metric_failures > 0);
        assert!(stale_metric_failures < transformed_by_id.len());
    }

    #[test]
    fn every_declared_d4_element_preserves_the_frozen_euclidean_metric() {
        let manifest = manifest();
        for element in &manifest.elements {
            validate_orthogonal_element(element).unwrap();
        }
    }

    #[test]
    fn decimal_coordinates_are_parsed_exactly_without_binary_float_rounding() {
        let exact: Value = serde_json::from_str("1.01").unwrap();
        assert_eq!(decimal_scaled(&exact, 1000, "test"), Ok(1010));

        let scientific: Value = serde_json::from_str("1e-3").unwrap();
        assert_eq!(decimal_scaled(&scientific, 1000, "test"), Ok(1));

        let not_representable: Value = serde_json::from_str("1.0001").unwrap();
        assert!(decimal_scaled(&not_representable, 1000, "test").is_err());

        // These tokens collapse to the same f64 under ordinary serde_json
        // parsing, but the exact-token feature must keep them distinct.
        let near_one: Value = serde_json::from_str("1.00000000000000001").unwrap();
        assert!(decimal_scaled(&near_one, 1000, "test").is_err());
        let exact_small: Value = serde_json::from_str("0.000000000000000001").unwrap();
        assert_eq!(
            decimal_scaled(&exact_small, 1_000_000_000_000_000_000, "test"),
            Ok(1)
        );
    }

    #[test]
    fn polynomial_oracle_rejects_out_of_bound_degree_fractional_and_duplicate_terms() {
        let manifest = manifest();

        let too_large = serde_json::json!({
            "polynomial": {
                "variables": ["x", "y"],
                "exact_rational": true,
                "terms": [{"exponents": [33, 0], "coefficient": 1}]
            }
        });
        assert!(manifest.polynomial_from_json(&too_large).is_err());

        let fractional = serde_json::json!({
            "polynomial": {
                "variables": ["x", "y"],
                "exact_rational": true,
                "terms": [{"exponents": [4, 0], "coefficient": 0.5}]
            }
        });
        assert!(manifest.polynomial_from_json(&fractional).is_err());

        let duplicate = serde_json::json!({
            "polynomial": {
                "variables": ["x", "y"],
                "exact_rational": true,
                "terms": [
                    {"exponents": [4, 0], "coefficient": 1},
                    {"exponents": [4, 0], "coefficient": -1}
                ]
            }
        });
        assert!(manifest.polynomial_from_json(&duplicate).is_err());
    }

    #[test]
    fn false_subgroup_promotion_is_detected_by_exact_point_set_oracle() {
        let manifest = manifest();
        let cases = field(&manifest.root, "cases").unwrap().as_array().unwrap();
        let chiral = cases
            .iter()
            .find(|case| string_field(case, "case_id").unwrap() == "C4-CHIRAL-ORBIT-001")
            .unwrap();
        let exact = manifest.structural_stabilizer(chiral).unwrap();
        assert_eq!(exact.len(), 4);
        assert!(!exact.contains("fx"));
    }
}
