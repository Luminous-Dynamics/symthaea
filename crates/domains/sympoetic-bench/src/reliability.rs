//! Inter-rater reliability estimators for nominal human annotations.
//!
//! These coefficients are diagnostics of agreement beyond chance, not proof
//! that the rubric is valid or that raters share a correct ground truth.

use std::collections::BTreeMap;

use anyhow::{Result, bail};

/// Cohen's kappa for two nominal raters with pairwise missing annotations.
///
/// Returns None when there are no jointly rated units or expected chance
/// agreement is one (the coefficient is then undefined).
pub fn cohens_kappa_nominal(
    rater_a: &[Option<&str>],
    rater_b: &[Option<&str>],
) -> Result<Option<f64>> {
    if rater_a.len() != rater_b.len() {
        bail!("Cohen's kappa requires equal-length rater arrays");
    }

    let mut jointly_rated = 0usize;
    let mut observed_agreements = 0usize;
    let mut a_counts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut b_counts: BTreeMap<&str, usize> = BTreeMap::new();

    for (index, (a, b)) in rater_a
        .iter()
        .copied()
        .zip(rater_b.iter().copied())
        .enumerate()
    {
        if let Some(label) = a {
            if label.trim().is_empty() {
                bail!("rater A has an empty label at unit {index}");
            }
        }
        if let Some(label) = b {
            if label.trim().is_empty() {
                bail!("rater B has an empty label at unit {index}");
            }
        }
        let (Some(a), Some(b)) = (a, b) else {
            continue;
        };
        jointly_rated += 1;
        observed_agreements += (a == b) as usize;
        *a_counts.entry(a).or_default() += 1;
        *b_counts.entry(b).or_default() += 1;
    }

    if jointly_rated == 0 {
        return Ok(None);
    }
    let n = jointly_rated as f64;
    let observed = observed_agreements as f64 / n;
    let expected = a_counts
        .iter()
        .map(|(label, count_a)| {
            *count_a as f64 * *b_counts.get(*label).unwrap_or(&0) as f64
        })
        .sum::<f64>()
        / (n * n);

    if (1.0 - expected).abs() <= f64::EPSILON {
        return Ok(None);
    }
    Ok(Some((observed - expected) / (1.0 - expected)))
}

/// Krippendorff's alpha for nominal data arranged as items × raters.
///
/// Missing annotations are represented by None. Items with fewer than two
/// ratings contribute to the marginal distribution but not observed pairwise
/// disagreement. Returns None if there are no pairable items or expected
/// disagreement is zero.
pub fn krippendorff_alpha_nominal(
    ratings: &[Vec<Option<&str>>],
) -> Result<Option<f64>> {
    if ratings.is_empty() {
        return Ok(None);
    }
    let rater_count = ratings[0].len();
    if ratings.iter().any(|item| item.len() != rater_count) {
        bail!("Krippendorff alpha requires a rectangular item-by-rater matrix");
    }
    if rater_count < 2 {
        return Ok(None);
    }

    let mut total_ratings = 0usize;
    let mut marginal_counts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut observed_disagreement = 0.0;
    let mut pairable_items = 0usize;

    for (item_index, item) in ratings.iter().enumerate() {
        let mut item_counts: BTreeMap<&str, usize> = BTreeMap::new();
        for label in item.iter().flatten().copied() {
            if label.trim().is_empty() {
                bail!("Krippendorff alpha has an empty label on item {item_index}");
            }
            *item_counts.entry(label).or_default() += 1;
            *marginal_counts.entry(label).or_default() += 1;
            total_ratings += 1;
        }

        let item_n = item_counts.values().sum::<usize>();
        if item_n < 2 {
            continue;
        }
        let same_ordered_pairs = item_counts
            .values()
            .map(|count| *count * *count)
            .sum::<usize>();
        let all_ordered_pairs = item_n * item_n;
        observed_disagreement +=
            (all_ordered_pairs - same_ordered_pairs) as f64 / (item_n - 1) as f64;
        pairable_items += 1;
    }

    if pairable_items == 0 || total_ratings < 2 {
        return Ok(None);
    }

    let total = total_ratings as f64;
    let same_global_ordered_pairs = marginal_counts
        .values()
        .map(|count| (*count as f64).powi(2))
        .sum::<f64>();
    let expected_disagreement = total * total - same_global_ordered_pairs;
    if expected_disagreement <= f64::EPSILON {
        return Ok(None);
    }

    let alpha = 1.0
        - observed_disagreement * (total - 1.0) / expected_disagreement;
    if !alpha.is_finite() {
        bail!("Krippendorff alpha calculation produced a non-finite value");
    }
    Ok(Some(alpha))
}
