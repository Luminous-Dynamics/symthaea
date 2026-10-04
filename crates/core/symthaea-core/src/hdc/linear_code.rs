    let algebra = factorization_algebra(factors)?;
    if target.dimension() != factors[0].dimension() {
        return None;
    }

    let combined_basis = concatenate_factor_bases(factors, algebra.factor_dimension_sum);
    let representative_coefficients = solve_linear_combination(target, &combined_basis)?;
    let kernel_basis = factorization_kernel_basis(factors)?;
    if kernel_basis.len() != algebra.kernel_dimension {
        return None;
    }
    let fiber = LinearCodeFactorizationFiber {
        representative_coefficients,
        kernel_basis,
        cardinality: algebra.factorization_count_per_target,
    };

    if fiber.representative_coefficients.len() != algebra.factor_dimension_sum
        || !fiber.basis_is_well_formed()
    {
        return None;
    }
    let mut reconstructed = BinaryCodeword::zero(target.dimension());
    for (&coefficient, generator) in fiber
        .representative_coefficients
        .iter()
        .zip(&combined_basis)
    {
        if coefficient {
            reconstructed.xor_assign(generator);
        }
    }
    if reconstructed != *target {
        return None;
    }

    let zero_mask = vec![false; fiber.kernel_basis.len()];
    if fiber.coefficients_for_mask(&zero_mask)? != fiber.representative_coefficients {
        return None;
    }
    Some(fiber)
}
/// Return the exact fiber cardinality for a target in the factor-span.
/// None means that the target is not representable by the supplied factors.
pub fn factorization_count_for_target(
    target: &BinaryCodeword,
    factors: &[&RandomLinearCode],
) -> Option<ExactPowerOfTwo> {
    let algebra = factorization_algebra(factors)?;
    if target.dimension() != factors[0].dimension() {
        return None;
    }

    let combined_basis = concatenate_factor_bases(factors, algebra.factor_dimension_sum);
    solve_linear_combination(target, &combined_basis)
        .map(|_| algebra.factorization_count_per_target)
}

fn concatenate_factor_bases(factors: &[&RandomLinearCode], capacity: usize) -> Vec<BinaryCodeword> {
    let mut basis = Vec::with_capacity(capacity);