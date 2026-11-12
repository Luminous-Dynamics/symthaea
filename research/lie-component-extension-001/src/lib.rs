// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Exact bounded structural qualification for split Lie / finite-component
//! extensions. All arithmetic is checked i128; overflow fails closed.

use std::fmt;

pub type Matrix<const N: usize> = [[i64; N]; N];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FiniteGroupV1 {
    pub group_id: String,
    pub identity: usize,
    pub multiplication: Vec<Vec<usize>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LieAlgebraV1<const N: usize> {
    pub algebra_id: String,
    /// Matrix representation of the declared Lie-algebra basis A_i.
    pub generators: Vec<Matrix<N>>,
    /// structure_constants[i][j][k] means [A_i, A_j] = sum_k c[i][j][k] A_k.
    pub structure_constants: Vec<Vec<Vec<i64>>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentRepresentationV1<const N: usize> {
    /// State-space representation matrices indexed by finite-group element.
    pub state_matrices: Vec<Matrix<N>>,
    /// Adjoint matrices indexed by finite-group element. Rows are output
    /// basis coordinates and columns are input basis coordinates:
    /// Ad(h) A_i = sum_j adjoint[h][j][i] A_j.
    pub adjoint_matrices: Vec<Vec<Vec<i64>>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitExtensionV1<const N: usize> {
    pub extension_id: String,
    pub identity_component_id: String,
    pub component_group: FiniteGroupV1,
    pub lie_algebra: LieAlgebraV1<N>,
    pub component_representation: ComponentRepresentationV1<N>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QualificationProfileV1 {
    pub maximum_state_dimension: usize,
    pub maximum_lie_generators: usize,
    pub maximum_component_order: usize,
    pub maximum_absolute_entry: i64,
    pub maximum_work_units: usize,
}

impl Default for QualificationProfileV1 {
    fn default() -> Self {
        Self {
            maximum_state_dimension: 8,
            maximum_lie_generators: 8,
            maximum_component_order: 16,
            maximum_absolute_entry: 1_000_000,
            maximum_work_units: 2_000_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtensionReceiptV1 {
    pub extension_id: String,
    pub identity_component_id: String,
    pub component_group_id: String,
    pub state_dimension: usize,
    pub lie_dimension: usize,
    pub component_order: usize,
    pub group_axiom_checks: usize,
    pub lie_closure_checks: usize,
    pub jacobi_checks: usize,
    pub state_representation_checks: usize,
    pub adjoint_representation_checks: usize,
    pub bracket_automorphism_checks: usize,
    pub representation_compatibility_checks: usize,
    pub split_extension_qualified: bool,
    pub exact_integer_arithmetic: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualificationErrorV1 {
    InvalidIdentifier,
    InvalidProfile,
    ZeroDimension,
    StateDimensionExceeded,
    LieDimensionExceeded,
    ComponentOrderExceeded,
    WorkBudgetExceeded,
    ArithmeticOverflow,
    EntryOutOfProfile,
    MultiplicationTableNotSquare,
    ProductOutOfRange,
    MultiplicationTableNotLatin,
    InvalidIdentity,
    MissingInverse,
    NonAssociativeGroup,
    LieStructureShapeMismatch,
    LieStructureNotAntisymmetric,
    LieJacobiViolation,
    LieMatrixClosureViolation,
    RepresentationCountMismatch,
    AdjointShapeMismatch,
    StateRepresentationIdentityViolation,
    StateRepresentationLawViolation,
    AdjointIdentityViolation,
    AdjointRepresentationLawViolation,
    AdjointNotInvertible,
    BracketAutomorphismViolation,
    RepresentationCompatibilityViolation,
}

impl fmt::Display for QualificationErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for QualificationErrorV1 {}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn checked_add(left: i128, right: i128) -> Result<i128, QualificationErrorV1> {
    left.checked_add(right)
        .ok_or(QualificationErrorV1::ArithmeticOverflow)
}

fn checked_mul(left: i128, right: i128) -> Result<i128, QualificationErrorV1> {
    left.checked_mul(right)
        .ok_or(QualificationErrorV1::ArithmeticOverflow)
}

fn checked_i64(value: i128) -> Result<i64, QualificationErrorV1> {
    i64::try_from(value).map_err(|_| QualificationErrorV1::ArithmeticOverflow)
}

fn state_identity<const N: usize>() -> Matrix<N> {
    std::array::from_fn(|row| std::array::from_fn(|column| if row == column { 1 } else { 0 }))
}

fn matrix_product<const N: usize>(
    left: &Matrix<N>,
    right: &Matrix<N>,
) -> Result<Matrix<N>, QualificationErrorV1> {
    let mut output = [[0_i64; N]; N];
    for row in 0..N {
        for column in 0..N {
            let mut value = 0_i128;
            for inner in 0..N {
                value = checked_add(
                    value,
                    checked_mul(
                        i128::from(left[row][inner]),
                        i128::from(right[inner][column]),
                    )?,
                )?;
            }
            output[row][column] = checked_i64(value)?;
        }
    }
    Ok(output)
}

fn matrix_commutator<const N: usize>(
    left: &Matrix<N>,
    right: &Matrix<N>,
) -> Result<Matrix<N>, QualificationErrorV1> {
    let ab = matrix_product(left, right)?;
    let ba = matrix_product(right, left)?;
    let mut output = [[0_i64; N]; N];
    for row in 0..N {
        for column in 0..N {
            output[row][column] =
                checked_i64(i128::from(ab[row][column]) - i128::from(ba[row][column]))?;
        }
    }
    Ok(output)
}

fn linear_combination<const N: usize>(
    generators: &[Matrix<N>],
    coefficients: &[i64],
) -> Result<Matrix<N>, QualificationErrorV1> {
    if generators.len() != coefficients.len() {
        return Err(QualificationErrorV1::LieStructureShapeMismatch);
    }
    let mut output = [[0_i64; N]; N];
    for (generator, coefficient) in generators.iter().zip(coefficients) {
        for row in 0..N {
            for column in 0..N {
                let contribution =
                    checked_mul(i128::from(*coefficient), i128::from(generator[row][column]))?;
                output[row][column] =
                    checked_i64(checked_add(i128::from(output[row][column]), contribution)?)?;
            }
        }
    }
    Ok(output)
}

fn dynamic_product(
    left: &[Vec<i64>],
    right: &[Vec<i64>],
) -> Result<Vec<Vec<i64>>, QualificationErrorV1> {
    let size = left.len();
    if size == 0
        || right.len() != size
        || left.iter().any(|row| row.len() != size)
        || right.iter().any(|row| row.len() != size)
    {
        return Err(QualificationErrorV1::AdjointShapeMismatch);
    }
    let mut output = vec![vec![0_i64; size]; size];
    for row in 0..size {
        for column in 0..size {
            let mut value = 0_i128;
            for inner in 0..size {
                value = checked_add(
                    value,
                    checked_mul(
                        i128::from(left[row][inner]),
                        i128::from(right[inner][column]),
                    )?,
                )?;
            }
            output[row][column] = checked_i64(value)?;
        }
    }
    Ok(output)
}

fn validate_group(group: &FiniteGroupV1) -> Result<(Vec<usize>, usize), QualificationErrorV1> {
    let order = group.multiplication.len();
    if order == 0 {
        return Err(QualificationErrorV1::MultiplicationTableNotSquare);
    }
    for row in &group.multiplication {
        if row.len() != order {
            return Err(QualificationErrorV1::MultiplicationTableNotSquare);
        }
        if row.iter().any(|product| *product >= order) {
            return Err(QualificationErrorV1::ProductOutOfRange);
        }
    }

    let mut checks = 0usize;
    for row in 0..order {
        for left in 0..order {
            for right in (left + 1)..order {
                checks = checks
                    .checked_add(1)
                    .ok_or(QualificationErrorV1::ArithmeticOverflow)?;
                if group.multiplication[row][left] == group.multiplication[row][right] {
                    return Err(QualificationErrorV1::MultiplicationTableNotLatin);
                }
                if group.multiplication[left][row] == group.multiplication[right][row] {
                    return Err(QualificationErrorV1::MultiplicationTableNotLatin);
                }
            }
        }
    }
    if group.identity >= order {
        return Err(QualificationErrorV1::InvalidIdentity);
    }

    for element in 0..order {
        checks += 2;
        if group.multiplication[group.identity][element] != element
            || group.multiplication[element][group.identity] != element
        {
            return Err(QualificationErrorV1::InvalidIdentity);
        }
    }

    for a in 0..order {
        for b in 0..order {
            for c in 0..order {
                checks = checks
                    .checked_add(1)
                    .ok_or(QualificationErrorV1::ArithmeticOverflow)?;
                if group.multiplication[group.multiplication[a][b]][c]
                    != group.multiplication[a][group.multiplication[b][c]]
                {
                    return Err(QualificationErrorV1::NonAssociativeGroup);
                }
            }
        }
    }

    let mut inverses = vec![usize::MAX; order];
    for element in 0..order {
        let inverse = (0..order).find(|candidate| {
            group.multiplication[element][*candidate] == group.identity
                && group.multiplication[*candidate][element] == group.identity
        });
        let Some(inverse) = inverse else {
            return Err(QualificationErrorV1::MissingInverse);
        };
        inverses[element] = inverse;
        checks = checks
            .checked_add(order)
            .ok_or(QualificationErrorV1::ArithmeticOverflow)?;
    }
    Ok((inverses, checks))
}

fn validate_lie_algebra<const N: usize>(
    algebra: &LieAlgebraV1<N>,
) -> Result<(usize, usize), QualificationErrorV1> {
    let dimension = algebra.generators.len();
    if dimension == 0 {
        return Err(QualificationErrorV1::LieStructureShapeMismatch);
    }
    if algebra.structure_constants.len() != dimension
        || algebra
            .structure_constants
            .iter()
            .any(|row| row.len() != dimension || row.iter().any(|col| col.len() != dimension))
    {
        return Err(QualificationErrorV1::LieStructureShapeMismatch);
    }

    let mut closure_checks = 0usize;
    for i in 0..dimension {
        for j in 0..dimension {
            for k in 0..dimension {
                closure_checks += 1;
                if i128::from(algebra.structure_constants[i][j][k])
                    != -i128::from(algebra.structure_constants[j][i][k])
                {
                    return Err(QualificationErrorV1::LieStructureNotAntisymmetric);
                }
            }
        }
        if algebra.generators[i]
            .iter()
            .flatten()
            .all(|value| *value == 0)
        {
            return Err(QualificationErrorV1::LieMatrixClosureViolation);
        }
    }

    for i in 0..dimension {
        for j in 0..dimension {
            let commutator = matrix_commutator(&algebra.generators[i], &algebra.generators[j])?;
            let expected =
                linear_combination(&algebra.generators, &algebra.structure_constants[i][j])?;
            closure_checks += 1;
            if commutator != expected {
                return Err(QualificationErrorV1::LieMatrixClosureViolation);
            }
        }
    }

    let mut jacobi_checks = 0usize;
    for i in 0..dimension {
        for j in 0..dimension {
            for k in 0..dimension {
                for output in 0..dimension {
                    let mut sum = 0_i128;
                    for inner in 0..dimension {
                        let term1 = checked_mul(
                            i128::from(algebra.structure_constants[i][j][inner]),
                            i128::from(algebra.structure_constants[inner][k][output]),
                        )?;
                        let term2 = checked_mul(
                            i128::from(algebra.structure_constants[j][k][inner]),
                            i128::from(algebra.structure_constants[inner][i][output]),
                        )?;
                        let term3 = checked_mul(
                            i128::from(algebra.structure_constants[k][i][inner]),
                            i128::from(algebra.structure_constants[inner][j][output]),
                        )?;
                        sum = checked_add(sum, checked_add(term1, checked_add(term2, term3)?)?)?;
                    }
                    jacobi_checks += 1;
                    if sum != 0 {
                        return Err(QualificationErrorV1::LieJacobiViolation);
                    }
                }
            }
        }
    }
    Ok((closure_checks, jacobi_checks))
}

fn validate_profile<const N: usize>(
    extension: &SplitExtensionV1<N>,
    profile: QualificationProfileV1,
) -> Result<(), QualificationErrorV1> {
    if !nonempty(&extension.extension_id)
        || !nonempty(&extension.identity_component_id)
        || !nonempty(&extension.component_group.group_id)
        || !nonempty(&extension.lie_algebra.algebra_id)
    {
        return Err(QualificationErrorV1::InvalidIdentifier);
    }
    if profile.maximum_state_dimension == 0
        || profile.maximum_lie_generators == 0
        || profile.maximum_component_order == 0
        || profile.maximum_absolute_entry <= 0
        || profile.maximum_work_units == 0
    {
        return Err(QualificationErrorV1::InvalidProfile);
    }
    if N == 0 {
        return Err(QualificationErrorV1::ZeroDimension);
    }
    if N > profile.maximum_state_dimension {
        return Err(QualificationErrorV1::StateDimensionExceeded);
    }
    if extension.lie_algebra.generators.len() > profile.maximum_lie_generators {
        return Err(QualificationErrorV1::LieDimensionExceeded);
    }
    if extension.component_group.multiplication.len() > profile.maximum_component_order {
        return Err(QualificationErrorV1::ComponentOrderExceeded);
    }

    let entry_limit = profile.maximum_absolute_entry as u64;
    let outside_profile = |value: &i64| value.unsigned_abs() > entry_limit;
    for generator in &extension.lie_algebra.generators {
        if generator.iter().flatten().any(outside_profile) {
            return Err(QualificationErrorV1::EntryOutOfProfile);
        }
    }
    for row in &extension.lie_algebra.structure_constants {
        if row.iter().flatten().any(outside_profile) {
            return Err(QualificationErrorV1::EntryOutOfProfile);
        }
    }
    for state in &extension.component_representation.state_matrices {
        if state.iter().flatten().any(outside_profile) {
            return Err(QualificationErrorV1::EntryOutOfProfile);
        }
    }
    for adjoint in &extension.component_representation.adjoint_matrices {
        if adjoint.iter().flatten().any(outside_profile) {
            return Err(QualificationErrorV1::EntryOutOfProfile);
        }
    }

    // Upper-bound the actual verifier loops before any quadratic/cubic work.
    // Group associativity: O(g^3); Jacobi and bracket automorphism:
    // O(d^5); matrix products: O(N^3) per composition/conjugation.
    let d = extension.lie_algebra.generators.len();
    let group_order = extension.component_group.multiplication.len();
    let g2 = group_order
        .checked_mul(group_order)
        .ok_or(QualificationErrorV1::ArithmeticOverflow)?;
    let g3 = g2
        .checked_mul(group_order)
        .ok_or(QualificationErrorV1::ArithmeticOverflow)?;
    let d2 = d
        .checked_mul(d)
        .ok_or(QualificationErrorV1::ArithmeticOverflow)?;
    let d3 = d2
        .checked_mul(d)
        .ok_or(QualificationErrorV1::ArithmeticOverflow)?;
    let d4 = d3
        .checked_mul(d)
        .ok_or(QualificationErrorV1::ArithmeticOverflow)?;
    let d5 = d4
        .checked_mul(d)
        .ok_or(QualificationErrorV1::ArithmeticOverflow)?;
    let n2 = N
        .checked_mul(N)
        .ok_or(QualificationErrorV1::ArithmeticOverflow)?;
    let n3 = n2
        .checked_mul(N)
        .ok_or(QualificationErrorV1::ArithmeticOverflow)?;
    let g_d5 = group_order
        .checked_mul(d5)
        .ok_or(QualificationErrorV1::ArithmeticOverflow)?;
    let g_d_n3 = group_order
        .checked_mul(d)
        .and_then(|value| value.checked_mul(n3))
        .ok_or(QualificationErrorV1::ArithmeticOverflow)?;
    let g2_d3 = g2
        .checked_mul(d3)
        .ok_or(QualificationErrorV1::ArithmeticOverflow)?;
    let g2_n3 = g2
        .checked_mul(n3)
        .ok_or(QualificationErrorV1::ArithmeticOverflow)?;
    let d2_n3 = d2
        .checked_mul(n3)
        .ok_or(QualificationErrorV1::ArithmeticOverflow)?;
    let required = g3
        .checked_add(d5)
        .and_then(|value| value.checked_add(g_d5))
        .and_then(|value| value.checked_add(g_d_n3.checked_mul(2)?))
        .and_then(|value| value.checked_add(g2_d3))
        .and_then(|value| value.checked_add(g2_n3))
        .and_then(|value| value.checked_add(d2_n3))
        .ok_or(QualificationErrorV1::ArithmeticOverflow)?;
    if required > profile.maximum_work_units {
        return Err(QualificationErrorV1::WorkBudgetExceeded);
    }
    Ok(())
}

pub fn verify_split_extension<const N: usize>(
    extension: &SplitExtensionV1<N>,
    profile: QualificationProfileV1,
) -> Result<ExtensionReceiptV1, QualificationErrorV1> {
    validate_profile(extension, profile)?;
    let (inverses, group_axiom_checks) = validate_group(&extension.component_group)?;
    let (lie_closure_checks, jacobi_checks) = validate_lie_algebra(&extension.lie_algebra)?;

    let group_order = extension.component_group.multiplication.len();
    let lie_dimension = extension.lie_algebra.generators.len();
    let representations = &extension.component_representation;
    if representations.state_matrices.len() != group_order
        || representations.adjoint_matrices.len() != group_order
    {
        return Err(QualificationErrorV1::RepresentationCountMismatch);
    }
    for matrix in &representations.adjoint_matrices {
        if matrix.len() != lie_dimension || matrix.iter().any(|row| row.len() != lie_dimension) {
            return Err(QualificationErrorV1::AdjointShapeMismatch);
        }
    }

    let state_identity = state_identity::<N>();
    let adjoint_identity: Vec<Vec<i64>> = (0..lie_dimension)
        .map(|row| {
            (0..lie_dimension)
                .map(|column| if row == column { 1_i64 } else { 0_i64 })
                .collect()
        })
        .collect();

    if representations.state_matrices[extension.component_group.identity] != state_identity {
        return Err(QualificationErrorV1::StateRepresentationIdentityViolation);
    }
    if representations.adjoint_matrices[extension.component_group.identity] != adjoint_identity {
        return Err(QualificationErrorV1::AdjointIdentityViolation);
    }

    let mut state_representation_checks = 0usize;
    let mut adjoint_representation_checks = 0usize;
    for left in 0..group_order {
        for right in 0..group_order {
            let product = extension.component_group.multiplication[left][right];
            let state_product = matrix_product(
                &representations.state_matrices[left],
                &representations.state_matrices[right],
            )?;
            state_representation_checks += 1;
            if state_product != representations.state_matrices[product] {
                return Err(QualificationErrorV1::StateRepresentationLawViolation);
            }

            let adjoint_product = dynamic_product(
                &representations.adjoint_matrices[left],
                &representations.adjoint_matrices[right],
            )?;
            adjoint_representation_checks += 1;
            if adjoint_product != representations.adjoint_matrices[product] {
                return Err(QualificationErrorV1::AdjointRepresentationLawViolation);
            }
        }
    }

    let mut bracket_automorphism_checks = 0usize;
    for element in 0..group_order {
        let adjoint = &representations.adjoint_matrices[element];
        for i in 0..lie_dimension {
            for j in 0..lie_dimension {
                for output in 0..lie_dimension {
                    let mut transformed_original = 0_i128;
                    for input in 0..lie_dimension {
                        transformed_original = checked_add(
                            transformed_original,
                            checked_mul(
                                i128::from(adjoint[output][input]),
                                i128::from(extension.lie_algebra.structure_constants[i][j][input]),
                            )?,
                        )?;
                    }

                    let mut bracket_of_transformed = 0_i128;
                    for left in 0..lie_dimension {
                        for right in 0..lie_dimension {
                            bracket_of_transformed = checked_add(
                                bracket_of_transformed,
                                checked_mul(
                                    checked_mul(
                                        i128::from(adjoint[left][i]),
                                        i128::from(adjoint[right][j]),
                                    )?,
                                    i128::from(
                                        extension.lie_algebra.structure_constants[left][right]
                                            [output],
                                    ),
                                )?,
                            )?;
                        }
                    }
                    bracket_automorphism_checks += 1;
                    if transformed_original != bracket_of_transformed {
                        return Err(QualificationErrorV1::BracketAutomorphismViolation);
                    }
                }
            }
        }
    }

    let mut representation_compatibility_checks = 0usize;
    for element in 0..group_order {
        let state = &representations.state_matrices[element];
        let inverse_state = &representations.state_matrices[inverses[element]];
        let adjoint = &representations.adjoint_matrices[element];
        for generator_index in 0..lie_dimension {
            let conjugated = matrix_product(
                &matrix_product(state, &extension.lie_algebra.generators[generator_index])?,
                inverse_state,
            )?;
            let coefficients: Vec<i64> = (0..lie_dimension)
                .map(|output| adjoint[output][generator_index])
                .collect();
            let expected = linear_combination(&extension.lie_algebra.generators, &coefficients)?;
            representation_compatibility_checks += 1;
            if conjugated != expected {
                return Err(QualificationErrorV1::RepresentationCompatibilityViolation);
            }
        }
    }

    Ok(ExtensionReceiptV1 {
        extension_id: extension.extension_id.clone(),
        identity_component_id: extension.identity_component_id.clone(),
        component_group_id: extension.component_group.group_id.clone(),
        state_dimension: N,
        lie_dimension,
        component_order: group_order,
        group_axiom_checks,
        lie_closure_checks,
        jacobi_checks,
        state_representation_checks,
        adjoint_representation_checks,
        bracket_automorphism_checks,
        representation_compatibility_checks,
        split_extension_qualified: true,
        exact_integer_arithmetic: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c2_group() -> FiniteGroupV1 {
        FiniteGroupV1 {
            group_id: "C2".to_owned(),
            identity: 0,
            multiplication: vec![vec![0, 1], vec![1, 0]],
        }
    }

    fn so2_lie_algebra() -> LieAlgebraV1<2> {
        LieAlgebraV1 {
            algebra_id: "so2".to_owned(),
            generators: vec![[[0, -1], [1, 0]]],
            structure_constants: vec![vec![vec![0]]],
        }
    }

    fn o2_extension() -> SplitExtensionV1<2> {
        SplitExtensionV1 {
            extension_id: "O2-as-SO2-semidir-C2".to_owned(),
            identity_component_id: "SO2".to_owned(),
            component_group: c2_group(),
            lie_algebra: so2_lie_algebra(),
            component_representation: ComponentRepresentationV1 {
                state_matrices: vec![[[1, 0], [0, 1]], [[1, 0], [0, -1]]],
                adjoint_matrices: vec![vec![vec![1]], vec![vec![-1]]],
            },
        }
    }

    fn direct_product_control() -> SplitExtensionV1<2> {
        SplitExtensionV1 {
            extension_id: "SO2-times-C2-central-control".to_owned(),
            identity_component_id: "SO2".to_owned(),
            component_group: c2_group(),
            lie_algebra: so2_lie_algebra(),
            component_representation: ComponentRepresentationV1 {
                state_matrices: vec![[[1, 0], [0, 1]], [[-1, 0], [0, -1]]],
                adjoint_matrices: vec![vec![vec![1]], vec![vec![1]]],
            },
        }
    }

    fn affine_lie_algebra() -> LieAlgebraV1<2> {
        // A1 = e11, A2 = e12, [A1,A2] = A2.
        LieAlgebraV1 {
            algebra_id: "affine-line".to_owned(),
            generators: vec![[[1, 0], [0, 0]], [[0, 1], [0, 0]]],
            structure_constants: vec![vec![vec![0, 0], vec![0, 1]], vec![vec![0, -1], vec![0, 0]]],
        }
    }

    fn affine_component_extension() -> SplitExtensionV1<2> {
        // Reflection in the state representation fixes A1 and negates A2.
        SplitExtensionV1 {
            extension_id: "affine-line-with-discrete-sign".to_owned(),
            identity_component_id: "affine-connected-component".to_owned(),
            component_group: c2_group(),
            lie_algebra: affine_lie_algebra(),
            component_representation: ComponentRepresentationV1 {
                state_matrices: vec![[[1, 0], [0, 1]], [[1, 0], [0, -1]]],
                adjoint_matrices: vec![vec![vec![1, 0], vec![0, 1]], vec![vec![1, 0], vec![0, -1]]],
            },
        }
    }

    #[test]
    fn exact_o2_split_extension_is_qualified() {
        let receipt =
            verify_split_extension(&o2_extension(), QualificationProfileV1::default()).unwrap();
        assert!(receipt.split_extension_qualified);
        assert!(receipt.exact_integer_arithmetic);
        assert_eq!(receipt.component_order, 2);
        assert_eq!(receipt.lie_dimension, 1);
        assert_eq!(receipt.representation_compatibility_checks, 2);
    }

    #[test]
    fn direct_product_is_a_distinct_valid_component_action() {
        let receipt =
            verify_split_extension(&direct_product_control(), QualificationProfileV1::default())
                .unwrap();
        assert!(receipt.split_extension_qualified);
    }

    #[test]
    fn nonabelian_affine_lie_algebra_accepts_discrete_automorphism() {
        let receipt = verify_split_extension(
            &affine_component_extension(),
            QualificationProfileV1::default(),
        )
        .unwrap();
        assert_eq!(receipt.jacobi_checks, 16);
        assert_eq!(receipt.bracket_automorphism_checks, 16);
    }

    #[test]
    fn incorrect_adjoint_sign_is_rejected_by_representation_compatibility() {
        let mut extension = o2_extension();
        extension.component_representation.adjoint_matrices[1] = vec![vec![1]];
        assert_eq!(
            verify_split_extension(&extension, QualificationProfileV1::default()),
            Err(QualificationErrorV1::RepresentationCompatibilityViolation)
        );
    }

    #[test]
    fn non_involutive_state_component_is_rejected() {
        let mut extension = o2_extension();
        extension.component_representation.state_matrices[1] = [[1, 1], [0, 1]];
        assert_eq!(
            verify_split_extension(&extension, QualificationProfileV1::default()),
            Err(QualificationErrorV1::StateRepresentationLawViolation)
        );
    }

    #[test]
    fn adjoint_map_must_obey_finite_component_group_law() {
        let mut extension = direct_product_control();
        // A shear is a valid automorphism of the two-dimensional abelian
        // algebra, but its square is not the identity required by C2.
        extension.lie_algebra.generators = vec![[[1, 0], [0, 0]], [[0, 0], [0, 1]]];
        extension.lie_algebra.structure_constants =
            vec![vec![vec![0, 0], vec![0, 0]], vec![vec![0, 0], vec![0, 0]]];
        extension.component_representation.adjoint_matrices[0] =
            vec![vec![1, 0], vec![0, 1]];
        extension.component_representation.adjoint_matrices[1] = vec![vec![1, 1], vec![0, 1]];
        assert_eq!(
            verify_split_extension(&extension, QualificationProfileV1::default()),
            Err(QualificationErrorV1::AdjointRepresentationLawViolation)
        );
    }

    #[test]
    fn bracket_automorphism_is_checked_separately() {
        let mut extension = affine_component_extension();
        // This order-two basis map is invertible but does not preserve
        // [A1,A2] = A2.
        extension.component_representation.adjoint_matrices[1] = vec![vec![-1, 0], vec![0, 1]];
        assert_eq!(
            verify_split_extension(&extension, QualificationProfileV1::default()),
            Err(QualificationErrorV1::BracketAutomorphismViolation)
        );
    }

    #[test]
    fn nonassociative_latin_loop_is_not_admitted_as_component_group() {
        let mut extension = o2_extension();
        extension.component_group = FiniteGroupV1 {
            group_id: "invalid-loop".to_owned(),
            identity: 0,
            multiplication: vec![
                vec![0, 1, 2, 3, 4, 5],
                vec![1, 3, 5, 2, 0, 4],
                vec![2, 5, 4, 1, 3, 0],
                vec![3, 2, 0, 4, 5, 1],
                vec![4, 0, 3, 5, 1, 2],
                vec![5, 4, 1, 0, 2, 3],
            ],
        };
        extension.component_representation.state_matrices = vec![state_identity::<2>(); 6];
        extension.component_representation.adjoint_matrices = vec![vec![vec![1]]; 6];
        assert_eq!(
            verify_split_extension(&extension, QualificationProfileV1::default()),
            Err(QualificationErrorV1::NonAssociativeGroup)
        );
    }

    #[test]
    fn work_budget_and_entry_profile_fail_closed() {
        let too_small = QualificationProfileV1 {
            maximum_work_units: 1,
            ..QualificationProfileV1::default()
        };
        assert_eq!(
            verify_split_extension(&o2_extension(), too_small),
            Err(QualificationErrorV1::WorkBudgetExceeded)
        );

        let mut extension = o2_extension();
        extension.component_representation.state_matrices[1][0][0] = 5_000_000;
        assert_eq!(
            verify_split_extension(&extension, QualificationProfileV1::default()),
            Err(QualificationErrorV1::EntryOutOfProfile)
        );
    }
}
