//! Reproduces the headline result of Ahn et al. (2011): do a cuisine's recipes
//! pair ingredients that **share** flavor compounds more (ΔNc > 0) or less
//! (ΔNc < 0) than chance?
//!
//! Metric (matching the paper):
//! - **Nc_real(cuisine)** = mean over recipes of the *per-recipe average* number
//!   of shared compounds over all ingredient pairs. Per-recipe averaging (not
//!   pooling every pair) is what stops the few enormous recipes from dominating —
//!   it is the difference between reproducing the paper and not.
//! - **Nc_null(cuisine)** = the same statistic on a *frequency-conserving* null:
//!   each real recipe of size `n` is replaced by `n` distinct ingredients drawn
//!   from that cuisine's own ingredient-frequency distribution. This preserves
//!   recipe sizes and ingredient popularity while destroying which ingredients
//!   are chosen *together*.
//! - **ΔNc = Nc_real − Nc_null.**
//!
//! Everything is deterministic from `seed` (see [`crate::rng`]).

use crate::data::dataset;
use crate::rng::SplitMix64;
use std::collections::HashMap;

/// Default recipe cap per cuisine used by the convenience helpers. NorthAmerican
/// has ~41k recipes; capping keeps the ground-truth test fast while leaving the
/// sign of ΔNc robust (verified against the full set).
pub const DEFAULT_MAX_RECIPES: usize = 6000;

/// Result of a flavor-pairing analysis for one cuisine.
#[derive(Clone, Debug, PartialEq)]
pub struct DeltaNc {
    pub cuisine: String,
    /// Mean per-recipe shared-compound count in the real recipes.
    pub real: f64,
    /// Same statistic under the frequency-conserving null.
    pub null: f64,
    /// `real - null`. Positive ⇒ pairs ingredients that share compounds.
    pub delta: f64,
    /// Number of recipes actually used (after the cap).
    pub recipes_used: usize,
}

/// Convenience: ΔNc with the default seed and recipe cap.
pub fn delta_nc_default(cuisine: &str) -> Option<DeltaNc> {
    delta_nc(cuisine, 0xC0FFEE, DEFAULT_MAX_RECIPES)
}

/// Compute ΔNc for `cuisine`. Returns `None` for an unknown cuisine.
pub fn delta_nc(cuisine: &str, seed: u64, max_recipes: usize) -> Option<DeltaNc> {
    let d = dataset();
    let all = d.recipes.get(cuisine)?;

    let mut rng = SplitMix64::new(seed);

    // Deterministically down-sample large cuisines (partial Fisher–Yates prefix).
    let sampled: Vec<&Vec<String>> = if all.len() > max_recipes {
        let mut idx: Vec<usize> = (0..all.len()).collect();
        for i in 0..max_recipes {
            let j = i + rng.below(idx.len() - i);
            idx.swap(i, j);
        }
        idx[..max_recipes].iter().map(|&i| &all[i]).collect()
    } else {
        all.iter().collect()
    };

    // Resolve each recipe's ingredients to their compound slices once.
    let empty: &[u16] = &[];
    let compounds = |name: &str| -> &'static [u16] {
        d.ingredient_compounds
            .get(name)
            .map(|v| v.as_slice())
            .unwrap_or(empty)
    };

    let recipes: Vec<Vec<&'static [u16]>> = sampled
        .iter()
        .map(|r| r.iter().map(|n| compounds(n)).collect())
        .collect();

    let real = mean_per_recipe_shared(&recipes);

    // Frequency-conserving null. Build the cuisine's ingredient-frequency table
    // (over the sampled recipes) as parallel arrays of compound-slices and
    // cumulative weights for weighted draws.
    let mut freq: HashMap<&str, u32> = HashMap::new();
    for r in &sampled {
        for n in r.iter() {
            *freq.entry(n.as_str()).or_insert(0) += 1;
        }
    }
    // Sort by name for a deterministic build order — HashMap iteration order is
    // randomized per call, which would make the weighted draws (and thus `null`)
    // non-reproducible from the seed.
    let mut freq: Vec<(&str, u32)> = freq.into_iter().collect();
    freq.sort_unstable_by(|a, b| a.0.cmp(b.0));
    let mut slices: Vec<&'static [u16]> = Vec::with_capacity(freq.len());
    let mut cum: Vec<f64> = Vec::with_capacity(freq.len());
    let mut running = 0.0f64;
    for (name, w) in &freq {
        running += *w as f64;
        slices.push(compounds(name));
        cum.push(running);
    }
    let total = running;

    let null_recipes: Vec<Vec<&'static [u16]>> = recipes
        .iter()
        .map(|r| sample_distinct(r.len(), &slices, &cum, total, &mut rng))
        .collect();
    let null = mean_per_recipe_shared(&null_recipes);

    Some(DeltaNc {
        cuisine: cuisine.to_string(),
        real,
        null,
        delta: real - null,
        recipes_used: sampled.len(),
    })
}

/// Mean over recipes of the per-recipe average shared-compound count.
fn mean_per_recipe_shared(recipes: &[Vec<&[u16]>]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for r in recipes {
        if r.len() < 2 {
            continue;
        }
        let mut pair_sum = 0u64;
        let mut pairs = 0u64;
        for i in 0..r.len() {
            for j in (i + 1)..r.len() {
                pair_sum += intersection_len(r[i], r[j]) as u64;
                pairs += 1;
            }
        }
        sum += pair_sum as f64 / pairs as f64;
        count += 1;
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

/// Draw `n` *distinct* ingredients (compound slices) weighted by `cum` cumulative
/// weights. Falls back gracefully if the pool has fewer than `n` distinct items.
fn sample_distinct(
    n: usize,
    slices: &[&'static [u16]],
    cum: &[f64],
    total: f64,
    rng: &mut SplitMix64,
) -> Vec<&'static [u16]> {
    let want = n.min(slices.len());
    let mut chosen: Vec<usize> = Vec::with_capacity(want);
    let mut attempts = 0usize;
    let cap = want * 32 + 64;
    while chosen.len() < want && attempts < cap {
        attempts += 1;
        let x = rng.unit() * total;
        // first index whose cumulative weight is > x
        let idx = cum.partition_point(|&c| c <= x).min(cum.len() - 1);
        if !chosen.contains(&idx) {
            chosen.push(idx);
        }
    }
    chosen.into_iter().map(|i| slices[i]).collect()
}

/// Linear-merge intersection size of two sorted, deduped slices.
fn intersection_len(a: &[u16], b: &[u16]) -> usize {
    let (mut i, mut j, mut n) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                n += 1;
                i += 1;
                j += 1;
            }
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_across_runs() {
        let a = delta_nc("EastAsian", 42, 2000).unwrap();
        let b = delta_nc("EastAsian", 42, 2000).unwrap();
        assert_eq!(a, b, "same seed must give identical ΔNc");
    }

    #[test]
    fn unknown_cuisine_is_none() {
        assert!(delta_nc("Martian", 1, 100).is_none());
    }
}
