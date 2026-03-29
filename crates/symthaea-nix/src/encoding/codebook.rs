// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! NixOS HDC Codebook
//!
//! Pre-computed basis vectors for NixOS option path segments, value types,
//! and structural roles. These form the "alphabet" from which all NixOS
//! HDC encodings are composed via bind and bundle operations.
//!
//! The codebook is deterministic — the same segment always maps to the same
//! basis vector, enabling reproducible encoding across sessions.

use std::collections::HashMap;
use symthaea_core::hdc::ContinuousHV;

/// Default HDC dimension for NixOS encoding.
///
/// Matches Symthaea core's `BinaryHV` dimension (16,384 bits = 2,048 bytes).
/// Higher dimensions improve representational capacity but increase memory and
/// compute cost. 16,384 is the sweet spot validated by the Symthaea psych-bench
/// suite for semantic similarity tasks.
pub const NIX_HDC_DIM: usize = 16_384;

/// Seed namespace to avoid collisions with other subsystems.
/// Derived from ASCII "NixOS\0\0\0" to give NixOS encodings a unique basis
/// that is orthogonal (in expectation) to other subsystem codebooks.
const CODEBOOK_SEED_BASE: u64 = 0x4E69_784F_5300_0000;

/// Number of random hyperplanes per LSH hash table.
/// 8 planes → 2^8 = 256 buckets per table. Balances hash collision rate
/// against bucket sparsity for the expected codebook size (~100-1000 tokens).
const LSH_NUM_PLANES: usize = 8;

/// Number of independent LSH hash tables for multi-probe.
/// 4 tables with independent random projections gives high recall (~95%)
/// while keeping memory overhead at 4× the bucket index.
const LSH_NUM_TABLES: usize = 4;

/// Minimum cache size before LSH kicks in (below this, brute-force is faster).
/// At 32 entries, linear scan over f32 vectors is faster than the LSH overhead
/// of hashing + bucket lookup + similarity verification.
const LSH_MIN_ENTRIES: usize = 32;

// Compile-time sanity checks
const _: () = assert!(NIX_HDC_DIM > 0 && NIX_HDC_DIM.is_power_of_two());
const _: () = assert!(LSH_NUM_PLANES > 0 && LSH_NUM_PLANES <= 16);
const _: () = assert!(LSH_NUM_TABLES > 0);
const _: () = assert!(LSH_MIN_ENTRIES > 0);

/// A single LSH hash table using random hyperplane projections.
struct LshTable {
    /// Random hyperplane normals (one per bit of the hash).
    planes: Vec<Vec<f32>>,
    /// Buckets: hash → list of token indices.
    buckets: HashMap<u32, Vec<usize>>,
}

impl LshTable {
    fn new(dim: usize, seed: u64) -> Self {
        let planes: Vec<Vec<f32>> = (0..LSH_NUM_PLANES)
            .map(|i| {
                let hv = ContinuousHV::random(dim, seed.wrapping_add(i as u64));
                hv.as_slice().to_vec()
            })
            .collect();
        Self {
            planes,
            buckets: HashMap::new(),
        }
    }

    fn hash(&self, vector: &ContinuousHV) -> u32 {
        let slice = vector.as_slice();
        let mut h: u32 = 0;
        for (i, plane) in self.planes.iter().enumerate() {
            let dot: f32 = slice.iter().zip(plane.iter()).map(|(a, b)| a * b).sum();
            if dot >= 0.0 {
                h |= 1 << i;
            }
        }
        h
    }

    fn insert(&mut self, vector: &ContinuousHV, index: usize) {
        let h = self.hash(vector);
        self.buckets.entry(h).or_default().push(index);
    }

    fn query(&self, vector: &ContinuousHV) -> Vec<usize> {
        let h = self.hash(vector);
        self.buckets.get(&h).cloned().unwrap_or_default()
    }

    fn clear(&mut self) {
        self.buckets.clear();
    }
}

/// Pre-computed basis vectors for NixOS HDC encoding.
///
/// Provides deterministic, cached mapping from string tokens to hypervectors.
/// All NixOS encoding modules share a single codebook to ensure vectors
/// live in the same semantic space.
///
/// Includes an LSH (Locality-Sensitive Hashing) index for sub-linear
/// approximate nearest-neighbor search when the cache grows large.
pub struct NixCodebook {
    /// Dimension of all vectors in this codebook.
    dim: usize,
    /// Cached basis vectors: token string → ContinuousHV.
    cache: HashMap<String, ContinuousHV>,
    /// Ordered token list (index matches LSH table indices).
    token_order: Vec<String>,
    /// LSH hash tables for fast approximate search.
    lsh_tables: Vec<LshTable>,
    /// Whether the LSH index needs rebuilding.
    lsh_dirty: bool,

    // ── Structural role markers ──────────────────────────────────────
    /// Role vector for "path segment at level N" (hierarchical position).
    level_markers: Vec<ContinuousHV>,
    /// Role vector for option values (bound with value encoding).
    pub value_role: ContinuousHV,
    /// Role vector for option types (bool, string, int, list, attrset).
    pub type_role: ContinuousHV,
    /// Role vector for package names.
    pub package_role: ContinuousHV,
    /// Role vector for service states (enabled/disabled/running/stopped).
    pub service_role: ContinuousHV,
    /// Role vector for user input tokens.
    pub input_role: ContinuousHV,
}

impl NixCodebook {
    /// Create a new codebook with standard dimension.
    pub fn new() -> Self {
        Self::with_dim(NIX_HDC_DIM)
    }

    /// Create a codebook with a specific dimension.
    pub fn with_dim(dim: usize) -> Self {
        // Pre-compute level markers for up to 8 hierarchy levels
        // (e.g., services.xserver.displayManager.gdm.enable = 5 levels)
        let level_markers: Vec<ContinuousHV> = (0..8)
            .map(|level| ContinuousHV::random(dim, CODEBOOK_SEED_BASE.wrapping_add(0x100 + level)))
            .collect();

        let lsh_tables: Vec<LshTable> = (0..LSH_NUM_TABLES)
            .map(|t| {
                LshTable::new(
                    dim,
                    CODEBOOK_SEED_BASE.wrapping_add(0x1000 + t as u64 * 100),
                )
            })
            .collect();

        let mut cb = Self {
            dim,
            cache: HashMap::with_capacity(64),
            token_order: Vec::with_capacity(64),
            lsh_tables,
            lsh_dirty: false,
            level_markers,
            value_role: ContinuousHV::random(dim, CODEBOOK_SEED_BASE.wrapping_add(0x200)),
            type_role: ContinuousHV::random(dim, CODEBOOK_SEED_BASE.wrapping_add(0x201)),
            package_role: ContinuousHV::random(dim, CODEBOOK_SEED_BASE.wrapping_add(0x202)),
            service_role: ContinuousHV::random(dim, CODEBOOK_SEED_BASE.wrapping_add(0x203)),
            input_role: ContinuousHV::random(dim, CODEBOOK_SEED_BASE.wrapping_add(0x204)),
        };
        // Pre-populate common segments to avoid cold-cache penalties
        cb.preload_common_segments();
        cb
    }

    /// Get the dimension of vectors in this codebook.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get or create a basis vector for a token string.
    ///
    /// The same token always produces the same vector (deterministic seeding).
    pub fn get_or_create(&mut self, token: &str) -> &ContinuousHV {
        if !self.cache.contains_key(token) {
            let seed = Self::token_seed(token);
            let hv = ContinuousHV::random(self.dim, seed);
            self.cache.insert(token.to_string(), hv.clone());
            self.token_order.push(token.to_string());
            self.lsh_dirty = true;
        }
        &self.cache[token]
    }

    /// Get the level marker for a given hierarchy depth.
    ///
    /// Level 0 = top-level (services, boot, networking, etc.)
    /// Level 1 = second segment (nginx, xserver, firewall, etc.)
    /// ...
    pub fn level_marker(&self, level: usize) -> &ContinuousHV {
        &self.level_markers[level.min(self.level_markers.len() - 1)]
    }

    /// Encode a single path segment at a given hierarchy level.
    ///
    /// Result = `segment_basis ⊗ level_marker[level]`
    ///
    /// This preserves which level the segment appears at, so
    /// `services` at level 0 differs from `services` at level 2.
    pub fn encode_segment(&mut self, segment: &str, level: usize) -> ContinuousHV {
        let basis = self.get_or_create(segment).clone();
        let marker = self.level_marker(level).clone();
        basis.bind(&marker)
    }

    /// Pre-populate the codebook with common NixOS top-level segments.
    ///
    /// This ensures consistent encoding even before any options are observed.
    pub fn preload_common_segments(&mut self) {
        let common = [
            // Top-level NixOS option categories
            "services",
            "networking",
            "boot",
            "hardware",
            "environment",
            "programs",
            "users",
            "security",
            "system",
            "fileSystems",
            "swapDevices",
            "nix",
            "nixpkgs",
            "systemd",
            "virtualisation",
            "fonts",
            "sound",
            "i18n",
            "console",
            "time",
            "powerManagement",
            // Common second-level segments
            "enable",
            "package",
            "settings",
            "config",
            "extraConfig",
            "openFirewall",
            "user",
            "group",
            "dataDir",
            "port",
            "nginx",
            "postgresql",
            "mysql",
            "openssh",
            "xserver",
            "firewall",
            "wireguard",
            "docker",
            "podman",
            "pipewire",
            "grub",
            "systemd-boot",
            "loader",
            "kernelPackages",
            "allowedTCPPorts",
            "allowedUDPPorts",
            "systemPackages",
            // Value type tokens
            "true",
            "false",
            "null",
            // Action-related tokens
            "install",
            "remove",
            "enable",
            "disable",
            "search",
            "rebuild",
            "switch",
            "rollback",
            "update",
            "upgrade",
        ];
        for seg in &common {
            self.get_or_create(seg);
        }
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Find the top-k most similar cached tokens to a query vector.
    ///
    /// Uses LSH for approximate nearest-neighbor when the cache is large
    /// enough to benefit (>= LSH_MIN_ENTRIES). Falls back to brute-force
    /// for small caches where the overhead isn't worthwhile.
    pub fn search_similar(&mut self, query: &ContinuousHV, top_k: usize) -> Vec<(String, f32)> {
        if self.cache.len() < LSH_MIN_ENTRIES {
            return self.search_brute_force(query, top_k);
        }

        if self.lsh_dirty {
            self.rebuild_lsh();
        }

        // Collect candidate indices from all LSH tables
        let mut candidate_set = std::collections::HashSet::new();
        for table in &self.lsh_tables {
            for idx in table.query(query) {
                candidate_set.insert(idx);
            }
        }

        // If LSH returned too few candidates, fall back to brute force
        if candidate_set.len() < top_k * 2 {
            return self.search_brute_force(query, top_k);
        }

        // Score only the candidates
        let mut results: Vec<(String, f32)> = candidate_set
            .into_iter()
            .filter_map(|idx| {
                self.token_order.get(idx).and_then(|token| {
                    self.cache
                        .get(token)
                        .map(|hv| (token.clone(), query.similarity(hv)))
                })
            })
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Brute-force search over all cached vectors.
    fn search_brute_force(&self, query: &ContinuousHV, top_k: usize) -> Vec<(String, f32)> {
        let mut results: Vec<(String, f32)> = self
            .cache
            .iter()
            .map(|(token, hv)| (token.clone(), query.similarity(hv)))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Rebuild the LSH index from all cached vectors.
    fn rebuild_lsh(&mut self) {
        for table in &mut self.lsh_tables {
            table.clear();
        }
        for (idx, token) in self.token_order.iter().enumerate() {
            if let Some(hv) = self.cache.get(token) {
                for table in &mut self.lsh_tables {
                    table.insert(hv, idx);
                }
            }
        }
        self.lsh_dirty = false;
    }

    /// Deterministic seed from a token string.
    ///
    /// Uses a simple hash to map strings to u64 seeds, ensuring the same
    /// string always produces the same basis vector.
    fn token_seed(token: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        // Include namespace to avoid collisions
        CODEBOOK_SEED_BASE.hash(&mut hasher);
        token.hash(&mut hasher);
        hasher.finish()
    }
}

impl Default for NixCodebook {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_encoding() {
        let mut cb1 = NixCodebook::new();
        let mut cb2 = NixCodebook::new();

        let hv1 = cb1.get_or_create("services").clone();
        let hv2 = cb2.get_or_create("services").clone();

        // Same token → same vector
        assert!((hv1.similarity(&hv2) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_different_tokens_quasi_orthogonal() {
        let mut cb = NixCodebook::new();
        let hv_services = cb.get_or_create("services").clone();
        let hv_boot = cb.get_or_create("boot").clone();

        // Different tokens → nearly orthogonal
        let sim = hv_services.similarity(&hv_boot).abs();
        assert!(
            sim < 0.1,
            "Expected near-orthogonal, got similarity {}",
            sim
        );
    }

    #[test]
    fn test_level_markers_differ() {
        let cb = NixCodebook::new();
        let l0 = cb.level_marker(0);
        let l1 = cb.level_marker(1);

        let sim = l0.similarity(l1).abs();
        assert!(
            sim < 0.1,
            "Level markers should be quasi-orthogonal, got {}",
            sim
        );
    }

    #[test]
    fn test_encode_segment_level_sensitive() {
        let mut cb = NixCodebook::new();

        // "services" at level 0 vs level 1 should differ
        let s0 = cb.encode_segment("services", 0);
        let s1 = cb.encode_segment("services", 1);

        let sim = s0.similarity(&s1).abs();
        assert!(
            sim < 0.15,
            "Same segment at different levels should differ, got {}",
            sim
        );
    }

    #[test]
    fn test_preload() {
        let cb = NixCodebook::new();
        // Constructor now auto-preloads common segments for performance
        assert!(!cb.is_empty());
        assert!(cb.len() > 30);
    }

    #[test]
    fn test_role_markers_quasi_orthogonal() {
        let cb = NixCodebook::new();
        let roles = [
            &cb.value_role,
            &cb.type_role,
            &cb.package_role,
            &cb.service_role,
            &cb.input_role,
        ];
        for i in 0..roles.len() {
            for j in (i + 1)..roles.len() {
                let sim = roles[i].similarity(roles[j]).abs();
                assert!(
                    sim < 0.1,
                    "Role markers {} and {} should be quasi-orthogonal, got {}",
                    i,
                    j,
                    sim
                );
            }
        }
    }

    #[test]
    fn test_search_similar_finds_self() {
        let mut cb = NixCodebook::new();
        let hv = cb.get_or_create("services").clone();
        let results = cb.search_similar(&hv, 3);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "services");
        assert!((results[0].1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_lsh_index_rebuilt_on_search() {
        let mut cb = NixCodebook::new();
        // Preload fills ~40 entries, above LSH_MIN_ENTRIES
        assert!(cb.len() >= 32);
        let hv = cb.get_or_create("services").clone();
        let results = cb.search_similar(&hv, 5);
        assert!(!results.is_empty());
        // After search, LSH should be clean
        assert!(!cb.lsh_dirty);
    }

    #[test]
    fn test_token_order_tracks_insertions() {
        let mut cb = NixCodebook::with_dim(256);
        let preloaded = cb.token_order.len();
        cb.get_or_create("alpha");
        cb.get_or_create("beta");
        cb.get_or_create("gamma");
        assert_eq!(cb.token_order.len(), preloaded + 3);
        assert_eq!(cb.token_order[preloaded], "alpha");
        assert_eq!(cb.token_order[preloaded + 1], "beta");
        assert_eq!(cb.token_order[preloaded + 2], "gamma");
    }

    #[test]
    fn test_constants_valid() {
        assert_eq!(NIX_HDC_DIM, 16_384);
        assert!(NIX_HDC_DIM.is_power_of_two());
        assert_eq!(1 << LSH_NUM_PLANES, 256);
        assert_eq!(LSH_NUM_TABLES, 4);
        assert!(LSH_MIN_ENTRIES > 0);
    }

    #[test]
    fn test_custom_dim_codebook() {
        let cb = NixCodebook::with_dim(512);
        assert_eq!(cb.dim(), 512);
        assert!(!cb.is_empty());
    }
}
