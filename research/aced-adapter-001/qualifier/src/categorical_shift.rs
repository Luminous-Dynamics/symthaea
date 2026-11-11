use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CategoryCount {
    pub state: String,
    pub count: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CategoryShiftInterval {
    pub state: String,
    pub reference_proportion: f64,
    pub intervention_proportion: f64,
    /// intervention - reference
    pub difference: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub exceeds_minimum_effect: bool,
    pub interval_excludes_zero: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CategoricalShiftReport {
    pub reference_n: u64,
    pub intervention_n: u64,
    pub total_variation_distance: f64,
    pub maximum_absolute_category_shift: f64,
    pub familywise_z_critical: f64,
    pub minimum_absolute_effect: f64,
    pub categories: Vec<CategoryShiftInterval>,
}

fn wilson_interval(successes: u64, total: u64, z: f64) -> Result<(f64, f64), String> {
    if total == 0 || successes > total {
        return Err("Wilson interval requires 0 <= successes <= total and total > 0".into());
    }
    let n = total as f64;
    let p = successes as f64 / n;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half = (z / denominator)
        * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt();
    Ok(((center - half).max(0.0), (center + half).min(1.0)))
}

fn normalize_counts(
    rows: &[CategoryCount],
    label: &str,
) -> Result<(BTreeMap<String, u64>, u64), String> {
    if rows.len() < 2 {
        return Err(format!("{label} distribution must contain at least two categories"));
    }
    let mut counts = BTreeMap::new();
    let mut total = 0_u64;
    for row in rows {
        if row.state.trim().is_empty() {
            return Err(format!("{label} contains an empty category identity"));
        }
        if counts.insert(row.state.clone(), row.count).is_some() {
            return Err(format!("{label} contains duplicate category {}", row.state));
        }
        total = total
            .checked_add(row.count)
            .ok_or_else(|| format!("{label} sample count overflows u64"))?;
    }
    if total == 0 {
        return Err(format!("{label} total sample count must be positive"));
    }
    Ok((counts, total))
}

/// Compare two categorical distributions with Newcombe differences built
/// from Wilson score intervals. The supplied critical z value must already
/// account for the preregistered multiple-comparison family.
pub fn compare_categorical_distributions(
    reference: &[CategoryCount],
    intervention: &[CategoryCount],
    familywise_z_critical: f64,
    minimum_absolute_effect: f64,
) -> Result<CategoricalShiftReport, String> {
    if !familywise_z_critical.is_finite() || familywise_z_critical <= 0.0 {
        return Err("familywise z critical value must be finite and positive".into());
    }
    if !minimum_absolute_effect.is_finite()
        || !(0.0..=1.0).contains(&minimum_absolute_effect)
    {
        return Err("minimum absolute effect must be finite and within [0,1]".into());
    }

    let (reference_counts, reference_n) = normalize_counts(reference, "reference")?;
    let (intervention_counts, intervention_n) = normalize_counts(intervention, "intervention")?;
    let reference_states = reference_counts.keys().collect::<BTreeSet<_>>();
    let intervention_states = intervention_counts.keys().collect::<BTreeSet<_>>();
    if reference_states != intervention_states {
        return Err("reference and intervention must declare exactly the same category set".into());
    }

    let mut categories = Vec::with_capacity(reference_counts.len());
    let mut total_variation_distance = 0.0;
    let mut maximum_absolute_category_shift: f64 = 0.0;

    for (state, reference_count) in &reference_counts {
        let intervention_count = intervention_counts[state];
        let p_reference = *reference_count as f64 / reference_n as f64;
        let p_intervention = intervention_count as f64 / intervention_n as f64;
        let difference = p_intervention - p_reference;

        let (reference_lower, reference_upper) =
            wilson_interval(*reference_count, reference_n, familywise_z_critical)?;
        let (intervention_lower, intervention_upper) =
            wilson_interval(intervention_count, intervention_n, familywise_z_critical)?;

        let lower = (difference
            - ((p_intervention - intervention_lower).powi(2)
                + (reference_upper - p_reference).powi(2))
            .sqrt())
        .max(-1.0);
        let upper = (difference
            + ((intervention_upper - p_intervention).powi(2)
                + (p_reference - reference_lower).powi(2))
            .sqrt())
        .min(1.0);
        let absolute_difference = difference.abs();

        total_variation_distance += absolute_difference / 2.0;
        maximum_absolute_category_shift =
            maximum_absolute_category_shift.max(absolute_difference);

        categories.push(CategoryShiftInterval {
            state: state.clone(),
            reference_proportion: p_reference,
            intervention_proportion: p_intervention,
            difference,
            lower_bound: lower,
            upper_bound: upper,
            exceeds_minimum_effect: absolute_difference >= minimum_absolute_effect,
            interval_excludes_zero: lower > 0.0 || upper < 0.0,
        });
    }

    Ok(CategoricalShiftReport {
        reference_n,
        intervention_n,
        total_variation_distance,
        maximum_absolute_category_shift,
        familywise_z_critical,
        minimum_absolute_effect,
        categories,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binary_counts(a: u64, b: u64) -> Vec<CategoryCount> {
        vec![
            CategoryCount { state: "no".into(), count: a },
            CategoryCount { state: "yes".into(), count: b },
        ]
    }

    #[test]
    fn identical_distributions_have_zero_total_variation() {
        let report = compare_categorical_distributions(
            &binary_counts(50, 50),
            &binary_counts(50, 50),
            1.96,
            0.02,
        )
        .unwrap();
        assert_eq!(report.total_variation_distance, 0.0);
        assert_eq!(report.maximum_absolute_category_shift, 0.0);
        assert!(report.categories.iter().all(|item| !item.interval_excludes_zero));
    }

    #[test]
    fn same_percentage_shift_has_different_evidence_at_different_sample_sizes() {
        let small = compare_categorical_distributions(
            &binary_counts(9, 1),
            &binary_counts(6, 4),
            1.96,
            0.02,
        )
        .unwrap();
        let large = compare_categorical_distributions(
            &binary_counts(900, 100),
            &binary_counts(600, 400),
            1.96,
            0.02,
        )
        .unwrap();
        assert!((small.maximum_absolute_category_shift - 0.30).abs() < 1e-12);
        assert!((large.maximum_absolute_category_shift - 0.30).abs() < 1e-12);
        assert!(small.categories.iter().all(|item| !item.interval_excludes_zero));
        assert!(large.categories.iter().any(|item| item.interval_excludes_zero));
    }

    #[test]
    fn effect_size_and_uncertainty_are_reported_separately() {
        let report = compare_categorical_distributions(
            &binary_counts(80, 20),
            &binary_counts(70, 30),
            1.96,
            0.08,
        )
        .unwrap();
        assert!((report.total_variation_distance - 0.10).abs() < 1e-12);
        assert!(report.categories.iter().any(|item| item.exceeds_minimum_effect));
        assert!(report.categories.iter().all(|item| item.lower_bound <= item.upper_bound));
    }

    #[test]
    fn zero_cells_are_valid_and_intervals_remain_bounded() {
        let report = compare_categorical_distributions(
            &binary_counts(100, 0),
            &binary_counts(0, 100),
            1.96,
            0.01,
        )
        .unwrap();
        assert!(report.categories.iter().all(|item| {
            item.lower_bound >= -1.0
                && item.upper_bound <= 1.0
                && item.lower_bound <= item.upper_bound
        }));
        assert!((report.total_variation_distance - 1.0).abs() < 1e-12);
    }

    #[test]
    fn mismatched_category_sets_fail_closed() {
        let reference = vec![
            CategoryCount { state: "a".into(), count: 10 },
            CategoryCount { state: "b".into(), count: 10 },
        ];
        let intervention = vec![
            CategoryCount { state: "a".into(), count: 10 },
            CategoryCount { state: "c".into(), count: 10 },
        ];
        assert!(compare_categorical_distributions(&reference, &intervention, 1.96, 0.0).is_err());
    }

    #[test]
    fn duplicate_or_empty_category_identities_fail_closed() {
        let duplicate = vec![
            CategoryCount { state: "a".into(), count: 10 },
            CategoryCount { state: "a".into(), count: 10 },
        ];
        let empty = vec![
            CategoryCount { state: "".into(), count: 10 },
            CategoryCount { state: "b".into(), count: 10 },
        ];
        assert!(compare_categorical_distributions(&duplicate, &duplicate, 1.96, 0.0).is_err());
        assert!(compare_categorical_distributions(&empty, &empty, 1.96, 0.0).is_err());
    }

    #[test]
    fn invalid_sample_size_and_thresholds_fail_closed() {
        let empty = vec![
            CategoryCount { state: "a".into(), count: 0 },
            CategoryCount { state: "b".into(), count: 0 },
        ];
        assert!(compare_categorical_distributions(&empty, &empty, 1.96, 0.0).is_err());
        assert!(compare_categorical_distributions(&binary_counts(5, 5), &binary_counts(5, 5), f64::NAN, 0.0).is_err());
        assert!(compare_categorical_distributions(&binary_counts(5, 5), &binary_counts(5, 5), 1.96, 1.1).is_err());
    }
}
