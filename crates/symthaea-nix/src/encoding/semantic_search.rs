// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Semantic Option Search API
//!
//! Encodes natural language queries as HDC vectors and finds the most
//! semantically similar NixOS options using the shared codebook.

use super::codebook::NixCodebook;
use super::option_encoder::OptionEncoder;
use super::user_input_encoder::UserInputEncoder;
use symthaea_core::hdc::ContinuousHV;

/// A search result matching a NixOS option.
#[derive(Debug, Clone)]
pub struct OptionMatch {
    /// The matched option path (e.g., "services.nginx.enable").
    pub path: String,
    /// Similarity score (0.0 to 1.0).
    pub similarity: f32,
    /// Which part of the query matched (if determinable).
    pub match_reason: String,
}

/// Search NixOS options by natural language query.
///
/// Encodes the query using `UserInputEncoder`, then finds the most
/// similar cached tokens in the codebook. Also encodes common option
/// paths to find structural matches.
pub fn search_options(
    query: &str,
    codebook: &mut NixCodebook,
    known_paths: &[&str],
    top_k: usize,
) -> Vec<OptionMatch> {
    // Encode the query as an HDC vector
    let mut input_encoder = UserInputEncoder::new(codebook);
    let query_hv = input_encoder.encode_input(query);

    // Encode all known paths and compute similarity to query
    let mut matches: Vec<OptionMatch> = known_paths
        .iter()
        .map(|path| {
            let mut encoder = OptionEncoder::new(codebook);
            let path_hv = encoder.encode_path(path);
            let sim = query_hv.similarity(&path_hv).max(0.0);
            OptionMatch {
                path: path.to_string(),
                similarity: sim,
                match_reason: format!("HDC similarity: {sim:.3}"),
            }
        })
        .collect();

    // Also check codebook token-level matches for keyword similarity
    let token_results = codebook.search_similar(&query_hv, top_k * 2);
    for (token, sim) in &token_results {
        // Find paths containing this token
        for path in known_paths {
            if path.contains(token.as_str()) {
                if let Some(existing) = matches.iter_mut().find(|m| m.path == *path) {
                    // Boost the similarity if a token match reinforces it
                    if *sim > 0.1 {
                        existing.similarity = (existing.similarity + sim * 0.3).min(1.0);
                        existing.match_reason = format!("HDC + token '{token}' match");
                    }
                }
            }
        }
    }

    matches.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    matches.truncate(top_k);
    matches
}

/// Search options using a pre-encoded query vector.
pub fn search_options_by_vector(
    query_hv: &ContinuousHV,
    codebook: &mut NixCodebook,
    known_paths: &[&str],
    top_k: usize,
) -> Vec<OptionMatch> {
    let mut matches: Vec<OptionMatch> = known_paths
        .iter()
        .map(|path| {
            let mut encoder = OptionEncoder::new(codebook);
            let path_hv = encoder.encode_path(path);
            let sim = query_hv.similarity(&path_hv).max(0.0);
            OptionMatch {
                path: path.to_string(),
                similarity: sim,
                match_reason: format!("Vector similarity: {sim:.3}"),
            }
        })
        .collect();

    matches.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    matches.truncate(top_k);
    matches
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_options_returns_results() {
        let mut codebook = NixCodebook::new();
        let paths = [
            "services.nginx.enable",
            "services.postgresql.enable",
            "services.openssh.enable",
            "networking.firewall.enable",
            "boot.loader.grub.enable",
        ];

        let results = search_options("enable nginx", &mut codebook, &paths, 3);
        assert!(!results.is_empty());
        assert!(results.len() <= 3);
        // Results should be sorted by similarity (descending)
        for window in results.windows(2) {
            assert!(window[0].similarity >= window[1].similarity);
        }
    }

    #[test]
    fn test_search_options_returns_all_paths() {
        let mut codebook = NixCodebook::new();
        let paths = [
            "services.nginx.enable",
            "services.nginx.package",
            "services.postgresql.enable",
            "boot.loader.grub.enable",
        ];

        let results = search_options("nginx web server", &mut codebook, &paths, 4);
        assert_eq!(results.len(), 4);
        // All paths should be scored
        let result_paths: Vec<&str> = results.iter().map(|r| r.path.as_str()).collect();
        assert!(result_paths.contains(&"services.nginx.enable"));
        assert!(result_paths.contains(&"boot.loader.grub.enable"));
        // All similarities should be finite
        for r in &results {
            assert!(
                r.similarity.is_finite(),
                "Similarity should be finite for {}",
                r.path
            );
        }
    }

    #[test]
    fn test_search_by_vector() {
        let mut codebook = NixCodebook::new();
        let paths = ["services.nginx.enable", "networking.firewall.enable"];

        // Encode "nginx" and search by its vector
        let hv = codebook.get_or_create("nginx").clone();
        let results = search_options_by_vector(&hv, &mut codebook, &paths, 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_empty_paths() {
        let mut codebook = NixCodebook::new();
        let results = search_options("nginx", &mut codebook, &[], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_top_k_zero() {
        let mut codebook = NixCodebook::new();
        let paths = ["services.nginx.enable"];
        let results = search_options("nginx", &mut codebook, &paths, 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_top_k_exceeds_paths() {
        let mut codebook = NixCodebook::new();
        let paths = ["services.nginx.enable", "boot.loader.grub.enable"];
        let results = search_options("nginx", &mut codebook, &paths, 100);
        // Should return all paths, not more
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_single_char_query() {
        let mut codebook = NixCodebook::new();
        let paths = ["services.nginx.enable"];
        let results = search_options("x", &mut codebook, &paths, 5);
        assert_eq!(results.len(), 1);
        assert!(results[0].similarity.is_finite());
    }

    #[test]
    fn test_search_by_vector_empty_paths() {
        let mut codebook = NixCodebook::new();
        let hv = ContinuousHV::random(16384, 42);
        let results = search_options_by_vector(&hv, &mut codebook, &[], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_by_vector_top_k_zero() {
        let mut codebook = NixCodebook::new();
        let paths = ["services.nginx.enable"];
        let hv = codebook.get_or_create("nginx").clone();
        let results = search_options_by_vector(&hv, &mut codebook, &paths, 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_results_sorted_descending() {
        let mut codebook = NixCodebook::new();
        let paths = [
            "services.nginx.enable",
            "services.postgresql.enable",
            "networking.firewall.enable",
            "boot.loader.grub.enable",
        ];
        let results = search_options("firewall networking", &mut codebook, &paths, 4);
        for w in results.windows(2) {
            assert!(
                w[0].similarity >= w[1].similarity,
                "Results should be sorted descending: {} >= {}",
                w[0].similarity,
                w[1].similarity
            );
        }
    }

    #[test]
    fn test_match_reason_populated() {
        let mut codebook = NixCodebook::new();
        let paths = ["services.nginx.enable"];
        let results = search_options("nginx", &mut codebook, &paths, 1);
        assert!(!results[0].match_reason.is_empty());

        let hv = codebook.get_or_create("nginx").clone();
        let results2 = search_options_by_vector(&hv, &mut codebook, &paths, 1);
        assert!(results2[0].match_reason.contains("Vector similarity"));
    }
}
