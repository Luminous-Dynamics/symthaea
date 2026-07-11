//! Package Healer — automatically fix broken package names.
//!
//! When nixpkgs renames or removes a package, the healer finds the replacement
//! using three strategies:
//! 1. Alias lookup (instant, 485 known mappings)
//! 2. HDC semantic search (instant, finds by meaning)
//! 3. Remote nixpkgs search (via relay, searches live nixpkgs)
//!
//! Usage:
//! ```no_run
//! use symthaea_app_db::package_healer::heal_package;
//! let fix = heal_package("firefox-esr");
//! // Returns: Some(HealResult { original: "firefox-esr", replacement: "firefox-esr-128", ... })
//! ```

use crate::aliases;
use crate::semantic_search;

/// Result of attempting to heal a broken package name.
#[derive(Debug, Clone)]
pub struct HealResult {
    /// The original (broken) package name
    pub original: String,
    /// The suggested replacement
    pub replacement: String,
    /// How the replacement was found
    pub method: HealMethod,
    /// Confidence (0.0 to 1.0)
    pub confidence: f32,
    /// Human-readable explanation
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HealMethod {
    /// Found via alias database (high confidence)
    Alias,
    /// Found via HDC semantic similarity (medium confidence)
    Semantic,
    /// Found via prefix/suffix matching (medium confidence)
    NameSimilarity,
    /// Found via remote `nix search` on target (high confidence)
    RemoteSearch,
    /// No replacement found
    NotFound,
}

/// Try to heal a broken package name using client-side strategies.
/// Returns None if no suitable replacement is found.
/// This runs in WASM — no network access needed.
pub fn heal_package(broken_name: &str) -> Option<HealResult> {
    let name = broken_name.trim().to_lowercase();
    if name.is_empty() {
        return None;
    }

    // Strategy 1: Alias lookup (exact match on a known mapping)
    if let Some(nixpkg) = aliases::lookup_alias(&name) {
        if nixpkg != name {
            return Some(HealResult {
                original: broken_name.to_string(),
                replacement: nixpkg.to_string(),
                method: HealMethod::Alias,
                confidence: 0.95,
                reason: format!("'{}' is a known alias for '{}'", broken_name, nixpkg),
            });
        }
    }

    // Strategy 2: Fuzzy alias match (prefix/substring)
    let suggestions = aliases::suggest_similar(&name);
    if let Some((alias, nixpkg)) = suggestions.first() {
        // Only suggest if the match is reasonably close
        let similarity = name_similarity(&name, alias);
        if similarity > 0.6 {
            return Some(HealResult {
                original: broken_name.to_string(),
                replacement: nixpkg.to_string(),
                method: HealMethod::NameSimilarity,
                confidence: similarity * 0.85,
                reason: format!(
                    "'{}' looks similar to '{}' (nixpkg: {})",
                    broken_name, alias, nixpkg
                ),
            });
        }
    }

    // Strategy 3: HDC semantic search (find by meaning)
    let semantic_results = semantic_search::semantic_search(&name, 3);
    if let Some(best) = semantic_results.first() {
        if best.similarity > 0.60 {
            return Some(HealResult {
                original: broken_name.to_string(),
                replacement: best.nixpkg.to_string(),
                method: HealMethod::Semantic,
                confidence: best.similarity * 0.80,
                reason: format!(
                    "'{}' semantically matches '{}' ({})",
                    broken_name, best.name, best.nixpkg
                ),
            });
        }
    }

    // Strategy 4: Common rename patterns
    if let Some(result) = try_common_renames(&name, broken_name) {
        return Some(result);
    }

    None
}

/// Try to heal a list of broken packages. Returns fixes for each.
pub fn heal_packages(broken_names: &[String]) -> Vec<HealResult> {
    broken_names
        .iter()
        .filter_map(|name| heal_package(name))
        .collect()
}

/// Apply healed replacements to a package list.
/// Returns the fixed list and a log of changes.
pub fn apply_healing(packages: &[String], fixes: &[HealResult]) -> (Vec<String>, Vec<String>) {
    let mut fixed = packages.to_vec();
    let mut log = Vec::new();

    for fix in fixes {
        for pkg in fixed.iter_mut() {
            if pkg.to_lowercase() == fix.original.to_lowercase() {
                log.push(format!(
                    "{} → {} ({})",
                    fix.original, fix.replacement, fix.reason
                ));
                *pkg = fix.replacement.clone();
            }
        }
    }

    (fixed, log)
}

/// Simple name similarity based on common prefix length + edit distance heuristic
fn name_similarity(a: &str, b: &str) -> f32 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let max_len = a_chars.len().max(b_chars.len()) as f32;
    if max_len == 0.0 {
        return 1.0;
    }

    // Common prefix length
    let prefix_len = a_chars
        .iter()
        .zip(&b_chars)
        .take_while(|(a, b)| a == b)
        .count() as f32;

    // Common character count (order-independent)
    let mut b_remaining: Vec<char> = b_chars.clone();
    let mut common = 0f32;
    for ch in &a_chars {
        if let Some(pos) = b_remaining.iter().position(|b| b == ch) {
            common += 1.0;
            b_remaining.remove(pos);
        }
    }

    // Weighted: prefix matters more than character overlap
    let prefix_score = prefix_len / max_len;
    let common_score = common / max_len;

    prefix_score * 0.6 + common_score * 0.4
}

/// Try common package rename patterns in nixpkgs
fn try_common_renames(name: &str, original: &str) -> Option<HealResult> {
    // Pattern: package-X → packageX (e.g., python-3 → python3)
    let no_hyphens = name.replace("-", "");
    if let Some(nixpkg) = aliases::lookup_alias(&no_hyphens) {
        return Some(HealResult {
            original: original.to_string(),
            replacement: nixpkg.to_string(),
            method: HealMethod::NameSimilarity,
            confidence: 0.7,
            reason: format!(
                "'{}' may have been renamed to '{}' (hyphen removed)",
                original, nixpkg
            ),
        });
    }

    // Pattern: package → package-bin (binary package)
    let with_bin = format!("{}-bin", name);
    if let Some(nixpkg) = aliases::lookup_alias(&with_bin) {
        return Some(HealResult {
            original: original.to_string(),
            replacement: nixpkg.to_string(),
            method: HealMethod::NameSimilarity,
            confidence: 0.7,
            reason: format!(
                "'{}' may have moved to binary package '{}'",
                original, nixpkg
            ),
        });
    }

    // Pattern: package → package-unwrapped
    let unwrapped = format!("{}-unwrapped", name);
    if let Some(nixpkg) = aliases::lookup_alias(&unwrapped) {
        return Some(HealResult {
            original: original.to_string(),
            replacement: nixpkg.to_string(),
            method: HealMethod::NameSimilarity,
            confidence: 0.6,
            reason: format!("'{}' may be available as '{}'", original, nixpkg),
        });
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heal_known_alias() {
        let result = heal_package("chrome");
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.replacement, "google-chrome");
        assert_eq!(r.method, HealMethod::Alias);
        assert!(r.confidence > 0.9);
    }

    #[test]
    fn heal_similar_name() {
        // "firefo" has enough prefix overlap with "firefox" to match
        let result = heal_package("firefo");
        assert!(result.is_some(), "Should heal 'firefo'");
        let r = result.unwrap();
        assert!(
            r.replacement.contains("firefox") || r.replacement.contains("fire"),
            "Expected firefox-related, got: {}",
            r.replacement
        );
    }

    #[test]
    fn heal_semantic_match() {
        let result = heal_package("web browser");
        assert!(result.is_some(), "Should find a browser semantically");
        let r = result.unwrap();
        assert!(r.method == HealMethod::Semantic);
    }

    #[test]
    fn heal_not_found_for_gibberish() {
        let result = heal_package("xyzzy12345nonexistent");
        // May or may not find something — but if it does, confidence should be low
        if let Some(r) = result {
            assert!(
                r.confidence < 0.7,
                "Gibberish should have low confidence: {}",
                r.confidence
            );
        }
    }

    #[test]
    fn apply_healing_replaces_packages() {
        let packages = vec!["chrome".to_string(), "vim".to_string()];
        let fixes = vec![HealResult {
            original: "chrome".to_string(),
            replacement: "google-chrome".to_string(),
            method: HealMethod::Alias,
            confidence: 0.95,
            reason: "alias".to_string(),
        }];
        let (fixed, log) = apply_healing(&packages, &fixes);
        assert_eq!(fixed, vec!["google-chrome", "vim"]);
        assert_eq!(log.len(), 1);
        assert!(log[0].contains("chrome → google-chrome"));
    }

    #[test]
    fn heal_empty_returns_none() {
        assert!(heal_package("").is_none());
        assert!(heal_package("  ").is_none());
    }

    #[test]
    fn name_similarity_identical() {
        assert!((name_similarity("firefox", "firefox") - 1.0).abs() < 0.01);
    }

    #[test]
    fn name_similarity_prefix() {
        let sim = name_similarity("fire", "firefox");
        assert!(
            sim > 0.5,
            "Common prefix should give high similarity: {}",
            sim
        );
    }
}
