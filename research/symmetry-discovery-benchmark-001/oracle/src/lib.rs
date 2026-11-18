// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exact, dependency-light semantics around the frozen D4 benchmark manifest.
//! JSON parsing is delegated to serde_json; all group, point-set and
//! polynomial calculations use exact integer arithmetic.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

type Matrix2 = [[i128; 2]; 2];
type Point = (i128, i128);
type Polynomial = BTreeMap<(u32, u32), i128>;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Element {
    id: String,
    matrix: Matrix2,
}

#[derive(Debug)]
struct Manifest {
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

fn multiply(left: Matrix2, right: Matrix2) -> Matrix2 {
    let mut product = [[0_i128; 2]; 2];
    for row in 0..2 {
        for column in 0..2 {
            product[row][column] =
                left[row][0] * right[0][column] + left[row][1] * right[1][column];
        }
    }
    product
}

impl Manifest {
    fn parse() -> Result<Self, String> {
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
            if by_matrix
                .insert(matrix_key(matrix), id.clone())
                .is_some()
            {
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
            if multiplication.insert((left.clone(), right.clone()), product).is_some() {
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

    fn verify_group_axioms(&self) -> Result<(), String> {
        let identity_id = string_field(field(&self.root, "ambient_group")?, "identity_element_id")?;
        if self.elements.len() != 8 || self.multiplication.len() != 64 || self.inverses.len() != 8 {
            return Err("D4 manifest must contain 8 elements, 64 products and 8 inverses".to_owned());
        }
        let identity = self
            .elements
            .iter()
            .find(|element| element.id == identity_id)
            .ok_or_else(|| "identity ID is not a declared element".to_owned())?;

        for left in &self.elements {
            for right in &self.elements {
                let product_matrix = multiply(left.matrix, right.matrix);
                let product_id = self
                    .by_matrix
                    .get(&matrix_key(product_matrix))
                    .ok_or_else(|| format!("D4 not closed at {} * {}", left.id, right.id))?;
                let table_id = self
                    .multiplication
                    .get(&(left.id.clone(), right.id.clone()))
                    .ok_or_else(|| format!("missing multiplication entry {} * {}", left.id, right.id))?;
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
            if multiply(left.matrix, inverse.matrix) != identity.matrix
                || multiply(inverse.matrix, left.matrix) != identity.matrix
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
                        return Err(format!("associativity fails for {}, {}, {}", a.id, b.id, c.id));
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
                    return Err(format!("target subgroup is not closed: {left} * {right} = {product}"));
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
        let expected_order = match label.as_str() {
            "D4" => 8,
            "C4" => 4,
            "C1" => 1,
            _ => return Err(format!("unknown fixture subgroup label: {label}")),
        };
        if set.len() != expected_order {
            return Err(format!(
                "subgroup label {label} declares order {expected_order}, but target has order {}",
                set.len()
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
            let coordinates = value
                .as_array()
                .ok_or_else(|| format!("point {index} must be a 2-vector"))?;
            if coordinates.len() != 2 {
                return Err(format!("point {index} must have two coordinates"));
            }
            let mut scaled = [0_i128; 2];
            for axis in 0..2 {
                let coordinate = coordinates[axis]
                    .as_f64()
                    .ok_or_else(|| format!("point {index} coordinate {axis} must be numeric"))?;
                let value = coordinate * 1000.0;
                let rounded = value.round();
                if (value - rounded).abs() > 1.0e-8 {
                    return Err(format!(
                        "point {index} coordinate {axis} is not exactly representable at 1/1000 scale"
                    ));
                }
                scaled[axis] = rounded as i128;
            }
            if !points.insert((scaled[0], scaled[1])) {
                return Err(format!("duplicate geometric point at index {index}"));
            }
        }
        let mut stabilizer = BTreeSet::new();
        for element in &self.elements {
            let transformed: BTreeSet<Point> = points
                .iter()
                .map(|(x, y)| {
                    (
                        element.matrix[0][0] * x + element.matrix[0][1] * y,
                        element.matrix[1][0] * x + element.matrix[1][1] * y,
                    )
                })
                .collect();
            if transformed == points {
                stabilizer.insert(element.id.clone());
            }
        }
        Ok(stabilizer)
    }

    fn polynomial_from_json(&self, case: &Value) -> Result<Polynomial, String> {
        let polynomial = field(case, "polynomial")?;
        if field(polynomial, "exact_rational")?.as_bool() != Some(true) {
            return Err("finite exact regression oracle requires exact_rational=true".to_owned());
        }
        let terms = field(polynomial, "terms")?
            .as_array()
            .ok_or_else(|| "polynomial.terms must be an array".to_owned())?;
        let mut result = Polynomial::new();
        for (index, term) in terms.iter().enumerate() {
            let exponents = field(term, "exponents")?
                .as_array()
                .ok_or_else(|| format!("term {index} exponents must be an array"))?;
            if exponents.len() != 2 {
                return Err(format!("term {index} must have two exponents"));
            }
            let x = exponents[0]
                .as_u64()
                .ok_or_else(|| format!("term {index} x exponent must be nonnegative"))?
                as u32;
            let y = exponents[1]
                .as_u64()
                .ok_or_else(|| format!("term {index} y exponent must be nonnegative"))?
                as u32;
            let coefficient = integer(field(term, "coefficient")?, "polynomial coefficient")?;
            let entry = result.entry((x, y)).or_insert(0);
            *entry = entry
                .checked_add(coefficient)
                .ok_or_else(|| "polynomial coefficient overflow".to_owned())?;
        }
        result.retain(|_, coefficient| *coefficient != 0);
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

    fn validate_all_cases(&self) -> Result<(usize, usize), String> {
        let cases = field(&self.root, "cases")?
            .as_array()
            .ok_or_else(|| "cases must be an array".to_owned())?;
        let mut ids = BTreeSet::new();
        let mut structural_count = 0;
        let mut regression_count = 0;
        for case in cases {
            let case_id = string_field(case, "case_id")?;
            if !ids.insert(case_id.clone()) {
                return Err(format!("duplicate case ID: {case_id}"));
            }
            let declared = self.declared_subgroup(case)?;
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
        if structural_count != 153 || regression_count != 2 {
            return Err(format!(
                "unexpected fixture lane counts: structural={structural_count}, regression={regression_count}"
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

fn transform_polynomial(
    polynomial: &Polynomial,
    matrix: Matrix2,
) -> Result<Polynomial, String> {
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
    fn all_153_structural_stabilizers_match_exact_oracle() {
        let counts = manifest()
            .validate_all_cases()
            .expect("all frozen labels must agree with exact integer oracle");
        assert_eq!(counts.0, 153);
    }

    #[test]
    fn both_regression_polynomial_stabilizers_match_exact_oracle() {
        let counts = manifest().validate_all_cases().unwrap();
        assert_eq!(counts.1, 2);
    }

    #[test]
    fn square_chiral_and_near_symmetry_controls_are_not_collapsed() {
        let manifest = manifest();
        let cases = field(&manifest.root, "cases")
            .unwrap()
            .as_array()
            .unwrap();
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
    fn false_subgroup_promotion_is_detected_by_exact_point_set_oracle() {
        let manifest = manifest();
        let cases = field(&manifest.root, "cases")
            .unwrap()
            .as_array()
            .unwrap();
        let chiral = cases
            .iter()
            .find(|case| string_field(case, "case_id").unwrap() == "C4-CHIRAL-ORBIT-001")
            .unwrap();
        let exact = manifest.structural_stabilizer(chiral).unwrap();
        assert_eq!(exact.len(), 4);
        assert!(!exact.contains("fx"));
    }
}
