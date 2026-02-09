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
pub const NIX_HDC_DIM: usize = 16_384;

/// Seed namespace to avoid collisions with other subsystems.
const CODEBOOK_SEED_BASE: u64 = 0x4E69_784F_5300_0000; // "NixOS\0\0\0"

/// Pre-computed basis vectors for NixOS HDC encoding.
///
/// Provides deterministic, cached mapping from string tokens to hypervectors.
/// All NixOS encoding modules share a single codebook to ensure vectors
/// live in the same semantic space.
pub struct NixCodebook {
    /// Dimension of all vectors in this codebook.
    dim: usize,
    /// Cached basis vectors: token string → ContinuousHV.
    cache: HashMap<String, ContinuousHV>,

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
            .map(|level| {
                ContinuousHV::random(dim, CODEBOOK_SEED_BASE.wrapping_add(0x100 + level))
            })
            .collect();

        Self {
            dim,
            cache: HashMap::new(),
            level_markers,
            value_role: ContinuousHV::random(dim, CODEBOOK_SEED_BASE.wrapping_add(0x200)),
            type_role: ContinuousHV::random(dim, CODEBOOK_SEED_BASE.wrapping_add(0x201)),
            package_role: ContinuousHV::random(dim, CODEBOOK_SEED_BASE.wrapping_add(0x202)),
            service_role: ContinuousHV::random(dim, CODEBOOK_SEED_BASE.wrapping_add(0x203)),
            input_role: ContinuousHV::random(dim, CODEBOOK_SEED_BASE.wrapping_add(0x204)),
        }
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
            self.cache.insert(token.to_string(), hv);
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
    /// Result = segment_basis ⊗ level_marker[level]
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
            "services", "networking", "boot", "hardware", "environment",
            "programs", "users", "security", "system", "fileSystems",
            "swapDevices", "nix", "nixpkgs", "systemd", "virtualisation",
            "fonts", "sound", "i18n", "console", "time", "powerManagement",
            // Common second-level segments
            "enable", "package", "settings", "config", "extraConfig",
            "openFirewall", "user", "group", "dataDir", "port",
            "nginx", "postgresql", "mysql", "openssh", "xserver",
            "firewall", "wireguard", "docker", "podman", "pipewire",
            "grub", "systemd-boot", "loader", "kernelPackages",
            "allowedTCPPorts", "allowedUDPPorts", "systemPackages",
            // Value type tokens
            "true", "false", "null",
            // Action-related tokens
            "install", "remove", "enable", "disable", "search",
            "rebuild", "switch", "rollback", "update", "upgrade",
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

    /// Deterministic seed from a token string.
    ///
    /// Uses a simple hash to map strings to u64 seeds, ensuring the same
    /// string always produces the same basis vector.
    fn token_seed(token: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
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
        assert!(sim < 0.1, "Expected near-orthogonal, got similarity {}", sim);
    }

    #[test]
    fn test_level_markers_differ() {
        let cb = NixCodebook::new();
        let l0 = cb.level_marker(0);
        let l1 = cb.level_marker(1);

        let sim = l0.similarity(l1).abs();
        assert!(sim < 0.1, "Level markers should be quasi-orthogonal, got {}", sim);
    }

    #[test]
    fn test_encode_segment_level_sensitive() {
        let mut cb = NixCodebook::new();

        // "services" at level 0 vs level 1 should differ
        let s0 = cb.encode_segment("services", 0);
        let s1 = cb.encode_segment("services", 1);

        let sim = s0.similarity(&s1).abs();
        assert!(sim < 0.15, "Same segment at different levels should differ, got {}", sim);
    }

    #[test]
    fn test_preload() {
        let mut cb = NixCodebook::new();
        assert!(cb.is_empty());
        cb.preload_common_segments();
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
                assert!(sim < 0.1, "Role markers {} and {} should be quasi-orthogonal, got {}", i, j, sim);
            }
        }
    }
}
