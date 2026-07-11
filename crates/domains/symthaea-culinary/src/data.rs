//! Loads the embedded Ahn et al. (2011) flavor-network data (see
//! `data/PROVENANCE.md`) into fast lookup tables, once, lazily.
//!
//! - `ingredient_compounds`: ingredient name -> sorted compound-id vector
//!   (the sparse "flavor vector" — an ingredient's set of volatile compounds).
//! - `recipes`: cuisine -> list of recipes, each a deduped list of ingredient names.

use std::collections::HashMap;
use std::sync::OnceLock;

const INGR_INFO: &str = include_str!("../data/ingr_info.tsv");
const INGR_COMP: &str = include_str!("../data/ingr_comp.tsv");
const RECIPES: &str = include_str!("../data/recipes.csv");

/// The parsed dataset. Cheap to clone-free borrow via [`dataset`].
pub struct Dataset {
    /// Ingredient name -> its flavor compounds (compound ids, sorted, deduped).
    pub ingredient_compounds: HashMap<String, Vec<u16>>,
    /// Cuisine -> recipes; each recipe is a deduped list of ingredient names.
    pub recipes: HashMap<String, Vec<Vec<String>>>,
}

static DATASET: OnceLock<Dataset> = OnceLock::new();

/// Access the parsed flavor-network dataset (parsed once on first call).
pub fn dataset() -> &'static Dataset {
    DATASET.get_or_init(load)
}

fn load() -> Dataset {
    // id -> name
    let mut id2name: HashMap<u32, String> = HashMap::new();
    for line in INGR_INFO.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let mut it = line.split('\t');
        let (Some(id), Some(name)) = (it.next(), it.next()) else {
            continue;
        };
        if let Ok(id) = id.trim().parse::<u32>() {
            id2name.insert(id, name.trim().to_string());
        }
    }

    // ingredient name -> compound ids
    let mut ingredient_compounds: HashMap<String, Vec<u16>> = HashMap::new();
    for line in INGR_COMP.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let (Some(iid), Some(cid)) = (it.next(), it.next()) else {
            continue;
        };
        let (Ok(iid), Ok(cid)) = (iid.parse::<u32>(), cid.parse::<u16>()) else {
            continue;
        };
        if let Some(name) = id2name.get(&iid) {
            ingredient_compounds
                .entry(name.clone())
                .or_default()
                .push(cid);
        }
    }
    // sort + dedup each compound vector so intersections are a linear merge.
    for v in ingredient_compounds.values_mut() {
        v.sort_unstable();
        v.dedup();
    }

    // cuisine -> recipes
    let mut recipes: HashMap<String, Vec<Vec<String>>> = HashMap::new();
    for line in RECIPES.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let mut fields = line.split(',');
        let Some(cuisine) = fields.next() else {
            continue;
        };
        let mut ings: Vec<String> = fields
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        ings.sort_unstable();
        ings.dedup();
        if ings.len() >= 2 {
            recipes.entry(cuisine.to_string()).or_default().push(ings);
        }
    }

    Dataset {
        ingredient_compounds,
        recipes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_expected_scale() {
        let d = dataset();
        // 1530 ingredients in ingr_info; not all appear in ingr_comp, but most do.
        assert!(
            d.ingredient_compounds.len() > 1000,
            "expected >1000 ingredients with compounds, got {}",
            d.ingredient_compounds.len()
        );
        // 11 cuisines in the recipe file.
        assert_eq!(d.recipes.len(), 11, "expected 11 cuisines");
        // NorthAmerican is by far the largest.
        assert!(d.recipes["NorthAmerican"].len() > 30_000);
    }

    #[test]
    fn known_ingredient_has_compounds() {
        let d = dataset();
        // garlic is a canonical, compound-rich ingredient in the dataset.
        let garlic = d.ingredient_compounds.get("garlic");
        assert!(garlic.is_some_and(|c| !c.is_empty()));
    }
}
