// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HDC Translator - Semantic Translation via Hyperdimensional Computing
//!
//! Uses BinaryHV vectors for semantic operations:
//! - Encoding NixOS paths and concepts to hypervectors
//! - Semantic similarity search for widget discovery
//! - Role-based binding for structured mappings
//! - Unbinding for reverse translation

use super::widget_mapper::{NixPath, WidgetBinding, WidgetId};
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use symthaea_core::hdc::binary_hv::BinaryHV;

/// Semantic role for HDC binding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SemanticRole {
    /// Service configuration
    Service,
    /// Package/software
    Package,
    /// Network setting
    Network,
    /// User/permission
    User,
    /// File/path
    FilePath,
    /// Port number
    Port,
    /// Boolean toggle
    Toggle,
    /// Text configuration
    Text,
    /// List/array
    List,
    /// Generic option
    Option,
}

impl SemanticRole {
    /// Get role from NixOS path
    pub fn from_path(path: &str) -> Self {
        if path.contains(".enable") {
            Self::Toggle
        } else if path.contains(".port") {
            Self::Port
        } else if path.contains(".package") || path.contains("Packages") {
            Self::Package
        } else if path.starts_with("services.") {
            Self::Service
        } else if path.starts_with("networking.") {
            Self::Network
        } else if path.starts_with("users.") {
            Self::User
        } else if path.contains("path") || path.contains("Path") {
            Self::FilePath
        } else if path.contains("extra") || path.contains("config") {
            Self::Text
        } else {
            Self::Option
        }
    }

    /// Get base HV for this role
    pub fn to_hv(&self) -> BinaryHV {
        let seed = match self {
            Self::Service => 1000,
            Self::Package => 2000,
            Self::Network => 3000,
            Self::User => 4000,
            Self::FilePath => 5000,
            Self::Port => 6000,
            Self::Toggle => 7000,
            Self::Text => 8000,
            Self::List => 9000,
            Self::Option => 10000,
        };
        BinaryHV::random(seed)
    }
}

/// Semantic binding between widget concept and NixOS concept
#[derive(Debug, Clone)]
pub struct SemanticBinding {
    /// Widget semantic HV
    pub widget_hv: BinaryHV,

    /// Role HV
    pub role_hv: BinaryHV,

    /// Combined bound HV (widget ⊗ role)
    pub bound_hv: BinaryHV,

    /// NixOS path semantic HV
    pub nix_hv: BinaryHV,

    /// Confidence of the binding
    pub confidence: f64,
}

impl SemanticBinding {
    /// Create new semantic binding
    pub fn new(widget_hv: BinaryHV, role: SemanticRole, nix_hv: BinaryHV) -> Self {
        let role_hv = role.to_hv();
        let bound_hv = widget_hv.bind(&role_hv);

        // Calculate confidence based on similarity between bound and nix
        let confidence = bound_hv.similarity(&nix_hv) as f64;

        Self {
            widget_hv,
            role_hv,
            bound_hv,
            nix_hv,
            confidence,
        }
    }

    /// Unbind to recover widget HV from nix HV
    pub fn unbind_widget(&self) -> BinaryHV {
        // nix ⊗ role = widget (approximately, due to XOR self-inverse)
        self.nix_hv.bind(&self.role_hv)
    }

    /// Unbind to recover nix HV from widget HV
    pub fn unbind_nix(&self) -> BinaryHV {
        self.widget_hv.bind(&self.role_hv)
    }
}

/// Translation result
#[derive(Debug, Clone)]
pub struct TranslationResult {
    /// Best matching path
    pub nix_path: NixPath,

    /// Semantic similarity
    pub similarity: f64,

    /// Role detected
    pub role: SemanticRole,

    /// Alternative matches
    pub alternatives: Vec<(NixPath, f64)>,
}

/// Semantic confidence level
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SemanticConfidence {
    /// High confidence (>0.8)
    High,
    /// Medium confidence (0.5-0.8)
    Medium,
    /// Low confidence (<0.5)
    Low,
    /// No match found
    NoMatch,
}

impl SemanticConfidence {
    pub fn from_similarity(sim: f64) -> Self {
        if sim > 0.8 {
            Self::High
        } else if sim > 0.5 {
            Self::Medium
        } else if sim > 0.3 {
            Self::Low
        } else {
            Self::NoMatch
        }
    }
}

/// HDC Translator for semantic operations
pub struct HdcTranslator {
    /// Path to HV cache
    path_cache: HashMap<String, BinaryHV>,

    /// Role HV cache
    role_cache: HashMap<SemanticRole, BinaryHV>,

    /// NixOS vocabulary (common terms -> HV)
    vocabulary: HashMap<String, BinaryHV>,
}

impl HdcTranslator {
    /// Create new HDC translator
    pub fn new() -> Self {
        let mut translator = Self {
            path_cache: HashMap::new(),
            role_cache: HashMap::new(),
            vocabulary: HashMap::new(),
        };

        // Initialize role cache
        for role in [
            SemanticRole::Service,
            SemanticRole::Package,
            SemanticRole::Network,
            SemanticRole::User,
            SemanticRole::FilePath,
            SemanticRole::Port,
            SemanticRole::Toggle,
            SemanticRole::Text,
            SemanticRole::List,
            SemanticRole::Option,
        ] {
            translator.role_cache.insert(role, role.to_hv());
        }

        // Initialize NixOS vocabulary
        translator.init_vocabulary();

        translator
    }

    /// Initialize NixOS vocabulary with semantic HVs
    fn init_vocabulary(&mut self) {
        let terms = [
            // Services
            ("nginx", "web server http proxy"),
            ("postgresql", "database sql postgres"),
            ("docker", "container virtualization"),
            ("openssh", "ssh remote secure shell"),
            ("firewall", "network security iptables"),
            // System
            ("systemPackages", "packages software install"),
            ("enable", "toggle activate turn on"),
            ("disable", "toggle deactivate turn off"),
            ("port", "network port number listen"),
            ("user", "account permission identity"),
            ("group", "permission role access"),
            // Configuration
            ("extraConfig", "custom configuration options"),
            ("settings", "configuration options parameters"),
            ("options", "settings parameters configure"),
        ];

        for (term, description) in terms {
            let hv = self.encode_text(&format!("{term} {description}"));
            self.vocabulary.insert(term.to_string(), hv);
        }
    }

    /// Convert string to u64 seed
    fn string_to_seed(s: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }

    /// Encode text to BinaryHV using n-gram bundling
    fn encode_text(&self, text: &str) -> BinaryHV {
        let mut vectors: Vec<BinaryHV> = Vec::new();
        let chars: Vec<char> = text.to_lowercase().chars().collect();

        // 3-gram encoding
        for window in chars.windows(3) {
            let trigram: String = window.iter().collect();
            let seed = Self::string_to_seed(&trigram);
            vectors.push(BinaryHV::random(seed));
        }

        // Word-level encoding
        for word in text.split_whitespace() {
            let seed = Self::string_to_seed(word);
            vectors.push(BinaryHV::random(seed));
        }

        if vectors.is_empty() {
            BinaryHV::zero()
        } else {
            BinaryHV::bundle(&vectors)
        }
    }

    /// Encode NixOS path to semantic HV
    pub fn encode_path(&mut self, path: &NixPath) -> BinaryHV {
        // Check cache
        if let Some(hv) = self.path_cache.get(&path.0) {
            return *hv;
        }

        // Encode path components
        let mut vectors: Vec<BinaryHV> = Vec::new();

        // Split path and encode each component
        for (i, component) in path.0.split('.').enumerate() {
            // Check vocabulary first
            let component_hv = if let Some(vocab_hv) = self.vocabulary.get(component) {
                *vocab_hv
            } else {
                self.encode_text(component)
            };

            // Apply positional permutation
            let permuted = component_hv.permute(i);
            vectors.push(permuted);
        }

        // Determine role and bind with role HV
        let role = SemanticRole::from_path(&path.0);
        let role_hv = self
            .role_cache
            .get(&role)
            .copied()
            .unwrap_or_else(BinaryHV::zero);
        vectors.push(role_hv);

        let result = if vectors.is_empty() {
            BinaryHV::zero()
        } else {
            BinaryHV::bundle(&vectors)
        };

        // Cache result
        self.path_cache.insert(path.0.clone(), result);

        result
    }

    /// Encode widget concept to semantic HV
    pub fn encode_widget(&self, widget_id: &WidgetId, label: &str) -> BinaryHV {
        // Combine widget ID and label
        let combined = format!("{} {}", widget_id.0, label);
        self.encode_text(&combined)
    }

    /// Create semantic binding between widget and nix path
    pub fn create_binding(
        &mut self,
        widget_id: &WidgetId,
        label: &str,
        path: &NixPath,
    ) -> SemanticBinding {
        let widget_hv = self.encode_widget(widget_id, label);
        let nix_hv = self.encode_path(path);
        let role = SemanticRole::from_path(&path.0);

        SemanticBinding::new(widget_hv, role, nix_hv)
    }

    /// Search for widgets matching a query
    pub fn search(
        &self,
        query: &str,
        bindings: &[WidgetBinding],
        limit: usize,
    ) -> Vec<super::SearchResult> {
        let query_hv = self.encode_text(query);

        let mut results: Vec<(usize, f64)> = bindings
            .iter()
            .enumerate()
            .map(|(i, binding)| {
                let similarity = query_hv.similarity(&binding.semantic_hv) as f64;
                (i, similarity)
            })
            .collect();

        // Sort by similarity descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        results
            .into_iter()
            .take(limit)
            .filter(|(_, sim)| *sim > 0.3)
            .map(|(i, sim)| {
                let binding = bindings[i].clone();
                let category = super::ConfigCategory::from_path(&binding.nix_path.0);

                super::SearchResult {
                    binding,
                    similarity: sim,
                    category,
                }
            })
            .collect()
    }

    /// Find best matching path for a query
    pub fn find_path(&self, query: &str, paths: &[NixPath]) -> Option<TranslationResult> {
        let query_hv = self.encode_text(query);

        let mut matches: Vec<(NixPath, f64)> = paths
            .iter()
            .map(|path| {
                let path_hv = if let Some(hv) = self.path_cache.get(&path.0) {
                    *hv
                } else {
                    self.encode_text(&path.0)
                };
                let similarity = query_hv.similarity(&path_hv) as f64;
                (path.clone(), similarity)
            })
            .collect();

        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((best_path, best_sim)) = matches.first() {
            if *best_sim > 0.3 {
                let role = SemanticRole::from_path(&best_path.0);
                let alternatives: Vec<(NixPath, f64)> =
                    matches.iter().skip(1).take(3).cloned().collect();

                return Some(TranslationResult {
                    nix_path: best_path.clone(),
                    similarity: *best_sim,
                    role,
                    alternatives,
                });
            }
        }

        None
    }

    /// Get semantic role for a path
    pub fn get_role(&self, path: &NixPath) -> SemanticRole {
        SemanticRole::from_path(&path.0)
    }

    /// Calculate semantic similarity between two paths
    pub fn path_similarity(&mut self, path1: &NixPath, path2: &NixPath) -> f64 {
        let hv1 = self.encode_path(path1);
        let hv2 = self.encode_path(path2);
        hv1.similarity(&hv2) as f64
    }
}

impl Default for HdcTranslator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_role_from_path() {
        assert_eq!(
            SemanticRole::from_path("services.nginx.enable"),
            SemanticRole::Toggle
        );
        assert_eq!(
            SemanticRole::from_path("networking.firewall.enable"),
            SemanticRole::Toggle
        );
        assert_eq!(
            SemanticRole::from_path("services.nginx.port"),
            SemanticRole::Port
        );
        assert_eq!(
            SemanticRole::from_path("environment.systemPackages"),
            SemanticRole::Package
        );
    }

    #[test]
    fn test_hdc_translator_creation() {
        let translator = HdcTranslator::new();
        assert!(!translator.vocabulary.is_empty());
        assert!(!translator.role_cache.is_empty());
    }

    #[test]
    fn test_path_encoding() {
        let mut translator = HdcTranslator::new();

        let path1 = NixPath::new("services.nginx.enable");
        let path2 = NixPath::new("services.nginx.port");
        let path3 = NixPath::new("users.users.alice.isNormalUser");

        let hv1 = translator.encode_path(&path1);
        let hv2 = translator.encode_path(&path2);
        let hv3 = translator.encode_path(&path3);

        // Similar paths should have higher similarity
        let sim12 = hv1.similarity(&hv2);
        let sim13 = hv1.similarity(&hv3);

        // nginx.enable and nginx.port should be more similar than nginx and users
        assert!(sim12 > sim13);
    }

    #[test]
    fn test_semantic_binding() {
        let widget_hv = BinaryHV::random(42);
        let nix_hv = BinaryHV::random(43);
        let role = SemanticRole::Service;

        let binding = SemanticBinding::new(widget_hv, role, nix_hv);

        // Binding should create a combined HV
        assert_ne!(binding.bound_hv, widget_hv);
        assert_ne!(binding.bound_hv, nix_hv);
    }

    #[test]
    fn test_semantic_confidence() {
        assert_eq!(
            SemanticConfidence::from_similarity(0.9),
            SemanticConfidence::High
        );
        assert_eq!(
            SemanticConfidence::from_similarity(0.6),
            SemanticConfidence::Medium
        );
        assert_eq!(
            SemanticConfidence::from_similarity(0.4),
            SemanticConfidence::Low
        );
        assert_eq!(
            SemanticConfidence::from_similarity(0.2),
            SemanticConfidence::NoMatch
        );
    }
}
