// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Kin-term string classification from the relationship-triangle coordinates
//! `(m, n)` — generations from ego up to the nearest common ancestor, and from
//! alter up to it — plus alter's sex. English (Eskimo) terminology.

use crate::genealogy::Sex;

fn ordinal(n: u32) -> String {
    match n {
        1 => "first".into(),
        2 => "second".into(),
        3 => "third".into(),
        4 => "fourth".into(),
        5 => "fifth".into(),
        _ => format!("{n}th"),
    }
}

fn removal(k: u32) -> String {
    match k {
        0 => String::new(),
        1 => "once removed".into(),
        2 => "twice removed".into(),
        3 => "thrice removed".into(),
        _ => format!("{k} times removed"),
    }
}

fn greats(k: u32) -> String {
    "great-".repeat(k as usize)
}

/// Term for an ancestor `m` generations up (m ≥ 1).
pub(crate) fn ancestor_term(m: u32, sex: Sex) -> String {
    let base = match sex {
        Sex::Male => "father",
        Sex::Female => "mother",
        Sex::Unknown => "parent",
    };
    match m {
        1 => base.into(),
        2 => format!("grand{base}"),
        _ => format!("{}grand{base}", greats(m - 2)),
    }
}

/// Term for a descendant `n` generations down (n ≥ 1).
pub(crate) fn descendant_term(n: u32, sex: Sex) -> String {
    let base = match sex {
        Sex::Male => "son",
        Sex::Female => "daughter",
        Sex::Unknown => "child",
    };
    match n {
        1 => base.into(),
        2 => format!("grand{base}"),
        _ => format!("{}grand{base}", greats(n - 2)),
    }
}

pub(crate) fn sibling_term(sex: Sex) -> String {
    match sex {
        Sex::Male => "brother",
        Sex::Female => "sister",
        Sex::Unknown => "sibling",
    }
    .into()
}

/// Parent's-sibling line (m ≥ 2, n = 1): uncle/aunt, granduncle, …
pub(crate) fn pibling_term(m: u32, sex: Sex) -> String {
    let base = match sex {
        Sex::Male => "uncle",
        Sex::Female => "aunt",
        Sex::Unknown => "parent's sibling",
    };
    match m {
        2 => base.into(),
        3 => format!("grand{base}"),
        _ => format!("{}grand{base}", greats(m - 3)),
    }
}

/// Sibling's-descendant line (m = 1, n ≥ 2): niece/nephew, grandniece, …
pub(crate) fn nibling_term(n: u32, sex: Sex) -> String {
    let base = match sex {
        Sex::Male => "nephew",
        Sex::Female => "niece",
        Sex::Unknown => "nibling",
    };
    match n {
        2 => base.into(),
        3 => format!("grand{base}"),
        _ => format!("{}grand{base}", greats(n - 3)),
    }
}

/// Cousins (m ≥ 2, n ≥ 2): "first cousin", "second cousin once removed", …
pub(crate) fn cousin_term(m: u32, n: u32) -> String {
    let degree = m.min(n) - 1;
    let rem = m.abs_diff(n);
    let base = format!("{} cousin", ordinal(degree));
    if rem == 0 {
        base
    } else {
        format!("{base} {}", removal(rem))
    }
}

/// Classify a consanguineal relation from triangle coordinates + alter's sex.
pub(crate) fn term_from_mn(m: u32, n: u32, sex: Sex) -> String {
    match (m, n) {
        (0, 0) => "self".into(),
        (m, 0) => ancestor_term(m, sex),
        (0, n) => descendant_term(n, sex),
        (1, 1) => sibling_term(sex),
        (1, n) => nibling_term(n, sex),
        (m, 1) => pibling_term(m, sex),
        (m, n) => cousin_term(m, n),
    }
}

/// Convert a consanguineal base term into its affinal (in-law) form.
pub(crate) fn in_law(base: &str) -> String {
    format!("{base}-in-law")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ancestor_and_descendant() {
        assert_eq!(ancestor_term(1, Sex::Male), "father");
        assert_eq!(ancestor_term(2, Sex::Female), "grandmother");
        assert_eq!(ancestor_term(3, Sex::Male), "great-grandfather");
        assert_eq!(descendant_term(1, Sex::Female), "daughter");
        assert_eq!(descendant_term(3, Sex::Male), "great-grandson");
    }

    #[test]
    fn collateral_terms() {
        assert_eq!(pibling_term(2, Sex::Male), "uncle");
        assert_eq!(pibling_term(3, Sex::Female), "grandaunt");
        assert_eq!(nibling_term(2, Sex::Female), "niece");
        assert_eq!(nibling_term(3, Sex::Male), "grandnephew");
    }

    #[test]
    fn cousins() {
        assert_eq!(cousin_term(2, 2), "first cousin");
        assert_eq!(cousin_term(3, 3), "second cousin");
        assert_eq!(cousin_term(2, 3), "first cousin once removed");
        assert_eq!(cousin_term(4, 2), "first cousin twice removed");
    }
}
