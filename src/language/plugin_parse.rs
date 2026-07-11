// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Shared numeric parsing for domain plugins — robust to punctuation and to
//! "70 kg" / "70kg" / "beta=0.3" forms. Keeps each plugin's parsing consistent
//! and tested in one place.

/// Trim surrounding non-numeric characters (keeps digits, `.`, leading `-`).
fn numeric(s: &str) -> Option<f64> {
    s.trim_matches(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))
        .parse::<f64>()
        .ok()
}

fn clean_unit(s: &str) -> String {
    s.trim_matches(|c: char| !c.is_ascii_alphabetic())
        .to_lowercase()
}

fn tokens(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| c.is_whitespace() || c == ',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

/// All `(value, unit)` measurements: a number with a unit on the same token
/// ("70kg") or the following one ("70 kg").
pub(crate) fn measurements(text: &str) -> Vec<(f64, String)> {
    let toks = tokens(text);
    let mut out = Vec::new();
    for (i, tok) in toks.iter().enumerate() {
        let split = tok.find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'));
        let (num, inline): (&str, &str) = match split {
            Some(0) | None => (tok, ""),
            Some(j) => (&tok[..j], &tok[j..]),
        };
        let Some(value) = numeric(num) else { continue };
        let unit = if !inline.is_empty() {
            clean_unit(inline)
        } else if let Some(next) = toks.get(i + 1) {
            clean_unit(next)
        } else {
            continue;
        };
        if !unit.is_empty() {
            out.push((value, unit));
        }
    }
    out
}

/// The first measurement whose unit is in `units`.
pub(crate) fn value_for_unit(text: &str, units: &[&str]) -> Option<f64> {
    measurements(text)
        .into_iter()
        .find(|(_, u)| units.contains(&u.as_str()))
        .map(|(v, _)| v)
}

/// All measurement values whose unit is in `units`.
pub(crate) fn values_for_unit(text: &str, units: &[&str]) -> Vec<f64> {
    measurements(text)
        .into_iter()
        .filter(|(_, u)| units.contains(&u.as_str()))
        .map(|(v, _)| v)
        .collect()
}

/// The number following a label token (tolerates `f=10`, `f 10`).
pub(crate) fn labeled(text: &str, labels: &[&str]) -> Option<f64> {
    let flat = text.to_lowercase().replace('=', " ");
    let toks: Vec<&str> = flat
        .split(|c: char| c.is_whitespace() || c == ',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .collect();
    for (i, t) in toks.iter().enumerate() {
        if labels.contains(t) {
            if let Some(v) = toks.get(i + 1).and_then(|n| numeric(n)) {
                return Some(v);
            }
        }
    }
    None
}

/// All signed decimal numbers, in order.
pub(crate) fn signed_numbers(text: &str) -> Vec<f64> {
    text.split(|c: char| c.is_whitespace() || c == ',' || c == ';')
        .filter_map(numeric)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measurements_various_forms() {
        let m = measurements("70 kg and 1.75 m, plus 180cm");
        assert!(m.contains(&(70.0, "kg".into())));
        assert!(m.contains(&(1.75, "m".into())));
        assert!(m.contains(&(180.0, "cm".into())));
    }

    #[test]
    fn value_for_unit_and_multiples() {
        assert_eq!(value_for_unit("temperature 300 k", &["k"]), Some(300.0));
        assert_eq!(
            values_for_unit("between 300 K and 600 K", &["k"]),
            vec![300.0, 600.0]
        );
    }

    #[test]
    fn labeled_tolerates_punctuation() {
        assert_eq!(labeled("beta=0.3, gamma 0.1?", &["beta"]), Some(0.3));
        assert_eq!(labeled("f=10 object 30", &["object"]), Some(30.0));
    }

    #[test]
    fn signed_numbers_preserve_sign_and_order() {
        assert_eq!(
            signed_numbers("-1000 500, 500"),
            vec![-1000.0, 500.0, 500.0]
        );
    }
}
