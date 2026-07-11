//! An ingredient as a sparse flavor vector: the set of volatile flavor compounds
//! it contains (Ahn et al. 2011). Pairing similarity is set overlap over those
//! compounds — the operationalization of the food-pairing hypothesis.

use crate::data::dataset;

/// A borrowed view of one ingredient's flavor compounds (sorted compound ids).
#[derive(Clone, Copy, Debug)]
pub struct Ingredient<'a> {
    pub name: &'a str,
    /// Sorted, deduped compound ids — the sparse presence vector.
    pub compounds: &'a [u16],
}

impl<'a> Ingredient<'a> {
    /// Look an ingredient up by name in the embedded dataset.
    pub fn get(name: &str) -> Option<Ingredient<'static>> {
        let d = dataset();
        d.ingredient_compounds
            .get_key_value(name)
            .map(|(k, v)| Ingredient {
                name: k.as_str(),
                compounds: v.as_slice(),
            })
    }

    /// Number of flavor compounds shared with another ingredient
    /// (|A ∩ B| — the quantity Ahn et al. use to test food pairing).
    pub fn shared_compounds(&self, other: &Ingredient<'_>) -> usize {
        intersection_len(self.compounds, other.compounds)
    }

    /// Jaccard overlap |A ∩ B| / |A ∪ B| in [0, 1].
    pub fn jaccard(&self, other: &Ingredient<'_>) -> f64 {
        let inter = self.shared_compounds(other);
        let union = self.compounds.len() + other.compounds.len() - inter;
        if union == 0 {
            0.0
        } else {
            inter as f64 / union as f64
        }
    }

    /// Cosine similarity of the two binary presence vectors in [0, 1].
    pub fn cosine(&self, other: &Ingredient<'_>) -> f64 {
        let inter = self.shared_compounds(other) as f64;
        let denom = ((self.compounds.len() * other.compounds.len()) as f64).sqrt();
        if denom == 0.0 { 0.0 } else { inter / denom }
    }
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
    fn intersection_basic() {
        assert_eq!(intersection_len(&[1, 3, 5, 7], &[3, 4, 5, 9]), 2);
        assert_eq!(intersection_len(&[], &[1, 2]), 0);
        assert_eq!(intersection_len(&[1, 2, 3], &[1, 2, 3]), 3);
    }

    #[test]
    fn self_similarity_is_one() {
        let garlic = Ingredient::get("garlic").expect("garlic present");
        assert_eq!(garlic.shared_compounds(&garlic), garlic.compounds.len());
        assert!((garlic.cosine(&garlic) - 1.0).abs() < 1e-9);
        assert!((garlic.jaccard(&garlic) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn metrics_are_symmetric() {
        let a = Ingredient::get("garlic").unwrap();
        let b = Ingredient::get("onion").unwrap();
        assert_eq!(a.shared_compounds(&b), b.shared_compounds(&a));
        assert!((a.cosine(&b) - b.cosine(&a)).abs() < 1e-12);
    }
}
