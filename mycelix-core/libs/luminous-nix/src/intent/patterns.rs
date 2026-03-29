// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Regex pattern matching as fallback for intent classification
//!
//! When HDC confidence is low, these patterns provide reliable exact matching.

use regex::Regex;

use super::Intent;

/// Regex-based pattern matcher for NixOS commands
pub struct PatternMatcher {
    search_patterns: Vec<Regex>,
    install_patterns: Vec<Regex>,
    remove_patterns: Vec<Regex>,
    generations_patterns: Vec<Regex>,
    rebuild_patterns: Vec<Regex>,
    gc_patterns: Vec<Regex>,
    status_patterns: Vec<Regex>,
    help_patterns: Vec<Regex>,
    argument_pattern: Regex,
}

impl PatternMatcher {
    /// Create a new pattern matcher with compiled regexes
    pub fn new() -> Self {
        Self {
            search_patterns: vec![
                Regex::new(r"^search\s+(.+)$").expect("valid regex literal"),
                Regex::new(r"^find\s+(.+)$").expect("valid regex literal"),
                Regex::new(r"^look\s+for\s+(.+)$").expect("valid regex literal"),
                Regex::new(r"^looking\s+for\s+(.+)$").expect("valid regex literal"),
                Regex::new(r"^query\s+(.+)$").expect("valid regex literal"),
                Regex::new(r"^locate\s+(.+)$").expect("valid regex literal"),
                Regex::new(r"^where\s+is\s+(.+)$").expect("valid regex literal"),
            ],
            install_patterns: vec![
                Regex::new(r"^install\s+(.+)$").expect("valid regex literal"),
                Regex::new(r"^add\s+(.+)$").expect("valid regex literal"),
                Regex::new(r"^get\s+(.+)$").expect("valid regex literal"),
                Regex::new(r"^setup\s+(.+)$").expect("valid regex literal"),
                Regex::new(r"^enable\s+(.+)$").expect("valid regex literal"),
            ],
            remove_patterns: vec![
                Regex::new(r"^remove\s+(.+)$").expect("valid regex literal"),
                Regex::new(r"^uninstall\s+(.+)$").expect("valid regex literal"),
                Regex::new(r"^delete\s+(.+)$").expect("valid regex literal"),
                Regex::new(r"^rm\s+(.+)$").expect("valid regex literal"),
                Regex::new(r"^disable\s+(.+)$").expect("valid regex literal"),
                Regex::new(r"^purge\s+(.+)$").expect("valid regex literal"),
            ],
            generations_patterns: vec![
                Regex::new(r"^(list\s+)?generations?$").expect("valid regex literal"),
                Regex::new(r"^show\s+generations?$").expect("valid regex literal"),
                Regex::new(r"^generation\s+history$").expect("valid regex literal"),
                Regex::new(r"^system\s+history$").expect("valid regex literal"),
                Regex::new(r"^list\s+system\s+versions?$").expect("valid regex literal"),
            ],
            rebuild_patterns: vec![
                Regex::new(r"^rebuild(\s+system)?$").expect("valid regex literal"),
                Regex::new(r"^nixos-rebuild$").expect("valid regex literal"),
                Regex::new(r"^update\s+system$").expect("valid regex literal"),
                Regex::new(r"^switch\s+configuration$").expect("valid regex literal"),
                Regex::new(r"^apply\s+config(uration)?$").expect("valid regex literal"),
            ],
            gc_patterns: vec![
                Regex::new(r"^gc$").expect("valid regex literal"),
                Regex::new(r"^garbage\s+collect(ion)?$").expect("valid regex literal"),
                Regex::new(r"^clean(\s+up)?$").expect("valid regex literal"),
                Regex::new(r"^cleanup$").expect("valid regex literal"),
                Regex::new(r"^nix-collect-garbage$").expect("valid regex literal"),
                Regex::new(r"^delete\s+old\s+generations?$").expect("valid regex literal"),
                Regex::new(r"^free\s+up\s+space$").expect("valid regex literal"),
                Regex::new(r"^clear\s+cache$").expect("valid regex literal"),
            ],
            status_patterns: vec![
                Regex::new(r"^status$").expect("valid regex literal"),
                Regex::new(r"^system\s+(info|status)$").expect("valid regex literal"),
                Regex::new(r"^show\s+status$").expect("valid regex literal"),
                Regex::new(r"^(what|which)\s+version$").expect("valid regex literal"),
                Regex::new(r"^nixos\s+version$").expect("valid regex literal"),
                Regex::new(r"^current\s+generation$").expect("valid regex literal"),
            ],
            help_patterns: vec![
                Regex::new(r"^help$").expect("valid regex literal"),
                Regex::new(r"^how\s+to\s+use$").expect("valid regex literal"),
                Regex::new(r"^what\s+can\s+you\s+do$").expect("valid regex literal"),
                Regex::new(r"^commands?$").expect("valid regex literal"),
                Regex::new(r"^usage$").expect("valid regex literal"),
            ],
            argument_pattern: Regex::new(r"^\w+\s+(.+)$").expect("valid regex literal"),
        }
    }

    /// Try to match a query against known patterns
    ///
    /// Returns the intent and confidence if a match is found
    pub fn match_query(&self, query: &str) -> Option<(Intent, f32)> {
        let query = query.trim().to_lowercase();

        // Check search patterns
        for pattern in &self.search_patterns {
            if let Some(caps) = pattern.captures(&query) {
                let search_query = caps.get(1).map(|m| m.as_str().to_string()).unwrap_or_default();
                return Some((Intent::Search { query: search_query }, 0.95));
            }
        }

        // Check install patterns
        for pattern in &self.install_patterns {
            if let Some(caps) = pattern.captures(&query) {
                let package = caps.get(1).map(|m| m.as_str().to_string()).unwrap_or_default();
                return Some((Intent::Install { package }, 0.95));
            }
        }

        // Check remove patterns
        for pattern in &self.remove_patterns {
            if let Some(caps) = pattern.captures(&query) {
                let package = caps.get(1).map(|m| m.as_str().to_string()).unwrap_or_default();
                return Some((Intent::Remove { package }, 0.95));
            }
        }

        // Check generations patterns
        for pattern in &self.generations_patterns {
            if pattern.is_match(&query) {
                return Some((Intent::ListGenerations, 0.95));
            }
        }

        // Check rebuild patterns
        for pattern in &self.rebuild_patterns {
            if pattern.is_match(&query) {
                return Some((Intent::Rebuild, 0.95));
            }
        }

        // Check gc patterns
        for pattern in &self.gc_patterns {
            if pattern.is_match(&query) {
                return Some((Intent::GarbageCollect, 0.95));
            }
        }

        // Check status patterns
        for pattern in &self.status_patterns {
            if pattern.is_match(&query) {
                return Some((Intent::Status, 0.95));
            }
        }

        // Check help patterns
        for pattern in &self.help_patterns {
            if pattern.is_match(&query) {
                return Some((Intent::Help, 0.95));
            }
        }

        None
    }

    /// Extract the argument portion from a query
    pub fn extract_argument(&self, query: &str) -> Option<String> {
        self.argument_pattern
            .captures(query)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().to_string())
    }
}

impl Default for PatternMatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_patterns() {
        let matcher = PatternMatcher::new();

        let queries = vec![
            "search firefox",
            "find vim",
            "look for git",
            "looking for neovim",
            "locate htop",
        ];

        for query in queries {
            let result = matcher.match_query(query);
            assert!(result.is_some(), "Failed to match: {}", query);
            assert!(matches!(result.unwrap().0, Intent::Search { .. }));
        }
    }

    #[test]
    fn test_install_patterns() {
        let matcher = PatternMatcher::new();

        let queries = vec!["install firefox", "add vim", "get git", "setup docker"];

        for query in queries {
            let result = matcher.match_query(query);
            assert!(result.is_some(), "Failed to match: {}", query);
            assert!(matches!(result.unwrap().0, Intent::Install { .. }));
        }
    }

    #[test]
    fn test_remove_patterns() {
        let matcher = PatternMatcher::new();

        let queries = vec!["remove firefox", "uninstall vim", "delete git", "rm htop"];

        for query in queries {
            let result = matcher.match_query(query);
            assert!(result.is_some(), "Failed to match: {}", query);
            assert!(matches!(result.unwrap().0, Intent::Remove { .. }));
        }
    }

    #[test]
    fn test_generations_patterns() {
        let matcher = PatternMatcher::new();

        let queries = vec![
            "generations",
            "list generations",
            "show generations",
            "generation history",
        ];

        for query in queries {
            let result = matcher.match_query(query);
            assert!(result.is_some(), "Failed to match: {}", query);
            assert!(matches!(result.unwrap().0, Intent::ListGenerations));
        }
    }

    #[test]
    fn test_extract_argument() {
        let matcher = PatternMatcher::new();

        assert_eq!(
            matcher.extract_argument("search firefox"),
            Some("firefox".to_string())
        );
        assert_eq!(
            matcher.extract_argument("install vim neovim"),
            Some("vim neovim".to_string())
        );
    }
}
