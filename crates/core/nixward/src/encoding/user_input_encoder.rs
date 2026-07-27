// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! User Input Encoder
//!
//! Encodes user natural language input into the **same HDC space** as
//! system state, enabling goal inference via vector similarity.
//!
//! "install firefox" encodes to a vector near "system with firefox",
//! so the gap between current state and encoded input IS the free energy
//! that drives action selection.

use super::codebook::NixCodebook;
use symthaea_core::hdc::ContinuousHV;

/// Known NixOS action verbs and their semantic clusters.
const ACTION_VERBS: &[(&str, &[&str])] = &[
    ("install", &["install", "add", "get", "setup", "enable"]),
    (
        "remove",
        &["remove", "uninstall", "delete", "disable", "drop"],
    ),
    ("search", &["search", "find", "look", "query", "locate"]),
    ("update", &["update", "upgrade", "refresh", "sync"]),
    (
        "rebuild",
        &["rebuild", "switch", "build", "deploy", "apply"],
    ),
    ("rollback", &["rollback", "revert", "undo", "restore"]),
    (
        "diagnose",
        &["why", "diagnose", "debug", "error", "fail", "broken"],
    ),
    (
        "optimize",
        &["optimize", "faster", "speed", "performance", "slow"],
    ),
    (
        "configure",
        &["configure", "config", "set", "change", "modify"],
    ),
    (
        "clean",
        &["clean", "gc", "garbage", "free", "space", "disk"],
    ),
];

/// Encodes user text input into the system-state HDC space.
pub struct UserInputEncoder<'a> {
    codebook: &'a mut NixCodebook,
}

impl<'a> UserInputEncoder<'a> {
    /// Create a new user input encoder.
    pub fn new(codebook: &'a mut NixCodebook) -> Self {
        Self { codebook }
    }

    /// Encode user input text into a ContinuousHV.
    ///
    /// The encoding lives in the same space as system state, so:
    /// - "install firefox" → vector near "system with firefox package"
    /// - "enable nginx" → vector near "system with nginx running"
    /// - "why did my build fail" → vector near "build failure state"
    pub fn encode_input(&mut self, input: &str) -> ContinuousHV {
        let lower = input.to_lowercase();
        let tokens: Vec<&str> = lower.split_whitespace().collect();
        let dim = self.codebook.dim();

        if tokens.is_empty() {
            return ContinuousHV::zero(dim);
        }

        let mut components = Vec::new();
        let mut weights = Vec::new();

        // Detect and encode action intent
        if let Some(action_hv) = self.encode_action_intent(&tokens) {
            components.push(action_hv);
            weights.push(0.6);
        }

        // Encode target entities (packages, services, options)
        let entity_hv = self.encode_entities(&tokens);
        if entity_hv.norm() > 0.0 {
            components.push(entity_hv);
            weights.push(1.0); // Entities have highest weight
        }

        // Encode remaining content tokens
        let content_hv = self.encode_content_tokens(&tokens);
        if content_hv.norm() > 0.0 {
            components.push(content_hv);
            weights.push(0.3);
        }

        if components.is_empty() {
            return ContinuousHV::zero(dim);
        }

        let input_role = self.codebook.input_role.clone();
        let refs: Vec<&ContinuousHV> = components.iter().collect();
        let base = ContinuousHV::weighted_bundle(&refs, &weights);

        // Bind with input role to mark this as user input
        base.bind(&input_role)
    }

    /// Detect action verbs and encode them.
    ///
    /// Iterates tokens in input order so that earlier words (which tend to
    /// express the user's primary intent) take priority over later words
    /// that might incidentally match a different action category.
    fn encode_action_intent(&mut self, tokens: &[&str]) -> Option<ContinuousHV> {
        for token in tokens {
            for (canonical, synonyms) in ACTION_VERBS {
                if synonyms.contains(token) {
                    return Some(self.codebook.get_or_create(canonical).clone());
                }
            }
        }
        None
    }

    /// Resolve package aliases in entity tokens.
    ///
    /// Maps natural language names like "vscode" → "nixpkgs.vscode",
    /// "chrome" → "nixpkgs.google-chrome", etc. using the 600+ alias table.
    fn resolve_aliases(&self, tokens: &[&str]) -> Vec<String> {
        tokens
            .iter()
            .map(|&t| {
                super::package_aliases::resolve_alias(t)
                    .map(|nix_name| nix_name.to_string())
                    .unwrap_or_else(|| t.to_string())
            })
            .collect()
    }

    /// Extract and encode entity tokens (non-stopwords, non-verbs).
    fn encode_entities(&mut self, tokens: &[&str]) -> ContinuousHV {
        let stopwords = [
            "i", "my", "me", "the", "a", "an", "is", "are", "was", "it", "do", "does", "did",
            "can", "could", "would", "should", "will", "to", "for", "in", "on", "at", "of", "with",
            "by", "from", "how", "what", "why", "when", "where", "please", "want", "need", "like",
            "make", "help", "up", "out", "not", "don't", "this", "that", "some", "all", "very",
            "too", "also", "just", "about",
        ];

        // Also exclude known action verbs
        let action_words: Vec<&str> = ACTION_VERBS
            .iter()
            .flat_map(|(_, synonyms)| synonyms.iter().copied())
            .collect();

        let entities: Vec<&str> = tokens
            .iter()
            .copied()
            .filter(|t| t.len() > 1 && !stopwords.contains(t) && !action_words.contains(t))
            .collect();

        if entities.is_empty() {
            return ContinuousHV::zero(self.codebook.dim());
        }

        // Resolve package aliases (e.g. "vscode" → "nixpkgs.vscode")
        let resolved = self.resolve_aliases(&entities);

        // Encode entities with package/service role binding
        let pkg_role = self.codebook.package_role.clone();
        let encoded: Vec<ContinuousHV> = resolved
            .iter()
            .map(|entity| {
                let basis = self.codebook.get_or_create(entity).clone();
                basis.bind(&pkg_role)
            })
            .collect();

        let refs: Vec<&ContinuousHV> = encoded.iter().collect();
        ContinuousHV::bundle(&refs)
    }

    /// Encode remaining content tokens (lower weight, contextual).
    ///
    /// Filters out stopwords and known action verbs to avoid duplicating
    /// signal already captured by `encode_action_intent` and `encode_entities`.
    fn encode_content_tokens(&mut self, tokens: &[&str]) -> ContinuousHV {
        let stopwords = [
            "i", "my", "me", "the", "a", "an", "is", "are", "was", "it", "do", "does", "did",
            "can", "could", "would", "should", "will", "to", "for", "in", "on", "at", "of", "with",
            "by", "from", "how", "what", "why", "when", "where", "please", "want", "need", "like",
            "make", "help", "up", "out", "not", "don't", "this", "that", "some", "all", "very",
            "too", "also", "just", "about",
        ];

        let action_words: Vec<&str> = ACTION_VERBS
            .iter()
            .flat_map(|(_, synonyms)| synonyms.iter().copied())
            .collect();

        let encoded: Vec<ContinuousHV> = tokens
            .iter()
            .filter(|t| t.len() > 2 && !stopwords.contains(t) && !action_words.contains(t))
            .take(10)
            .map(|tok| self.codebook.get_or_create(tok).clone())
            .collect();

        if encoded.is_empty() {
            return ContinuousHV::zero(self.codebook.dim());
        }

        let refs: Vec<&ContinuousHV> = encoded.iter().collect();
        ContinuousHV::bundle(&refs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_similar_intents_similar_vectors() {
        let mut cb = NixCodebook::new();

        let install_ff = {
            let mut enc = UserInputEncoder::new(&mut cb);
            enc.encode_input("install firefox")
        };
        let add_ff = {
            let mut enc = UserInputEncoder::new(&mut cb);
            enc.encode_input("add firefox")
        };
        let remove_python = {
            let mut enc = UserInputEncoder::new(&mut cb);
            enc.encode_input("remove python")
        };

        let sim_install = install_ff.similarity(&add_ff);
        let sim_cross = install_ff.similarity(&remove_python);

        // "install firefox" and "add firefox" should be more similar
        // than "install firefox" and "remove python"
        assert!(
            sim_install > sim_cross,
            "Similar intents ({:.3}) should have higher similarity than different ({:.3})",
            sim_install,
            sim_cross,
        );
    }

    #[test]
    fn test_same_target_detected() {
        let mut cb = NixCodebook::new();

        let install_nginx = {
            let mut enc = UserInputEncoder::new(&mut cb);
            enc.encode_input("install nginx")
        };
        let enable_nginx = {
            let mut enc = UserInputEncoder::new(&mut cb);
            enc.encode_input("enable nginx")
        };
        let install_docker = {
            let mut enc = UserInputEncoder::new(&mut cb);
            enc.encode_input("install docker")
        };

        // Both mention nginx → should share some similarity
        let sim_nginx = install_nginx.similarity(&enable_nginx);
        let sim_different = install_nginx.similarity(&install_docker);

        assert!(
            sim_nginx > sim_different,
            "Same target ({:.3}) should have higher similarity than different target ({:.3})",
            sim_nginx,
            sim_different,
        );
    }

    #[test]
    fn test_diagnostic_intent() {
        let mut cb = NixCodebook::new();

        let why_fail = {
            let mut enc = UserInputEncoder::new(&mut cb);
            enc.encode_input("why did my build fail")
        };
        let debug_error = {
            let mut enc = UserInputEncoder::new(&mut cb);
            enc.encode_input("debug this error")
        };
        let install_pkg = {
            let mut enc = UserInputEncoder::new(&mut cb);
            enc.encode_input("install firefox")
        };

        let sim_diag = why_fail.similarity(&debug_error);
        let sim_cross = why_fail.similarity(&install_pkg);

        assert!(
            sim_diag > sim_cross,
            "Diagnostic intents ({:.3}) should be more similar than install ({:.3})",
            sim_diag,
            sim_cross,
        );
    }

    #[test]
    fn test_empty_input() {
        let mut cb = NixCodebook::new();
        let hv = {
            let mut enc = UserInputEncoder::new(&mut cb);
            enc.encode_input("")
        };
        assert!(hv.norm() < 1e-6, "Empty input should produce zero vector");
    }

    #[test]
    fn test_alias_resolution_improves_similarity() {
        // "install vscode" should produce the same entity encoding as
        // "install code" since both resolve to the same nix package via aliases.
        let mut cb = NixCodebook::new();

        let vscode = {
            let mut enc = UserInputEncoder::new(&mut cb);
            enc.encode_input("install vscode")
        };
        let code = {
            let mut enc = UserInputEncoder::new(&mut cb);
            enc.encode_input("install code")
        };
        let unrelated = {
            let mut enc = UserInputEncoder::new(&mut cb);
            enc.encode_input("install htop")
        };

        // Both "vscode" and "code" should resolve to the same nix package,
        // making them more similar than an unrelated package
        let sim_alias = vscode.similarity(&code);
        let sim_unrelated = vscode.similarity(&unrelated);

        assert!(
            sim_alias > sim_unrelated,
            "Aliased packages ({:.3}) should be more similar than unrelated ({:.3})",
            sim_alias,
            sim_unrelated,
        );
    }

    #[test]
    fn test_resolve_aliases() {
        let mut cb = NixCodebook::new();
        let enc = UserInputEncoder::new(&mut cb);

        let resolved = enc.resolve_aliases(&["firefox", "vscode", "unknown_pkg"]);
        // "unknown_pkg" should pass through unchanged
        assert_eq!(resolved[2], "unknown_pkg");
        assert_eq!(resolved.len(), 3);
    }
}
