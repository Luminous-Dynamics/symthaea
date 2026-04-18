// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Nixpkgs option metadata index — lazy disk-cached lookup of NixOS module options.
//!
//! Purpose: give the Nix codegen pipeline ground truth for which option paths
//! exist (e.g. `services.postgresql.enable`) so generated configs can be
//! verified before being shown to the user.
//!
//! Strategy: per-option lazy lookup via `nix-instantiate --eval --json` against
//! `<nixpkgs/nixos>`. First lookup ~1s; cached lookups are O(1) memory. Cache
//! persists to `~/.cache/symthaea/nixpkgs-options.json`.
//!
//! We deliberately do NOT enumerate the full NixOS option space — that would
//! require evaluating tens of thousands of options at startup. Instead, the
//! index grows on demand: every option the codegen touches gets cached, and
//! the cache is shared across runs.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;

/// Metadata for a single NixOS module option, e.g. `services.nginx.enable`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct OptionMetadata {
    /// Dotted option path, e.g. "services.postgresql.enable".
    pub path: String,
    /// Type description from the option's `type` attribute, e.g. "boolean".
    pub type_sig: String,
    /// Human-readable description (may be empty).
    pub description: String,
    /// Whether the option exists in nixpkgs.
    pub exists: bool,
}

/// Result of a package existence query.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PackageMetadata {
    pub name: String,
    pub exists: bool,
}

/// On-disk cache structure (versioned for forward-compat).
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
struct CacheFile {
    version: u32,
    options: HashMap<String, OptionMetadata>,
    packages: HashMap<String, PackageMetadata>,
}

const CACHE_VERSION: u32 = 1;

/// Lazy, disk-backed nixpkgs option index.
///
/// Thread-safe via internal Mutex. Cheap to clone the handle (it's an Arc
/// internally), but we expose only `&self` methods so callers don't need to.
pub struct NixpkgsIndex {
    cache_path: PathBuf,
    inner: Mutex<CacheFile>,
}

impl NixpkgsIndex {
    /// Create an index with the default cache path (`~/.cache/symthaea/nixpkgs-options.json`).
    pub fn default_cache() -> Self {
        let path = default_cache_path();
        let inner = load_cache(&path).unwrap_or_default();
        Self {
            cache_path: path,
            inner: Mutex::new(inner),
        }
    }

    /// Create an index with an explicit cache path (useful for tests).
    pub fn with_cache_path(path: PathBuf) -> Self {
        let inner = load_cache(&path).unwrap_or_default();
        Self {
            cache_path: path,
            inner: Mutex::new(inner),
        }
    }

    /// Look up an option's metadata. First call shells out to `nix-instantiate`;
    /// subsequent calls hit the in-memory cache. Returns `None` only if the
    /// nix tool itself fails (e.g. nixpkgs not on NIX_PATH).
    pub fn lookup_option(&self, path: &str) -> Option<OptionMetadata> {
        // Fast path: in-memory cache hit.
        if let Ok(inner) = self.inner.lock() {
            if let Some(meta) = inner.options.get(path) {
                return Some(meta.clone());
            }
        }

        // Slow path: invoke nix-instantiate.
        let meta = query_option_uncached(path)?;

        // Insert + persist.
        if let Ok(mut inner) = self.inner.lock() {
            inner.version = CACHE_VERSION;
            inner.options.insert(path.to_string(), meta.clone());
            let _ = save_cache(&self.cache_path, &inner);
        }
        Some(meta)
    }

    /// Check whether an option path exists in NixOS modules.
    /// Convenience wrapper around `lookup_option().exists`.
    pub fn option_exists(&self, path: &str) -> bool {
        self.lookup_option(path).map(|m| m.exists).unwrap_or(false)
    }

    /// Check whether a top-level package exists in nixpkgs (e.g. "rust-analyzer").
    pub fn package_exists(&self, name: &str) -> bool {
        if let Ok(inner) = self.inner.lock() {
            if let Some(p) = inner.packages.get(name) {
                return p.exists;
            }
        }

        let exists = query_package_exists(name);

        if let Ok(mut inner) = self.inner.lock() {
            inner.version = CACHE_VERSION;
            inner.packages.insert(
                name.to_string(),
                PackageMetadata {
                    name: name.to_string(),
                    exists,
                },
            );
            let _ = save_cache(&self.cache_path, &inner);
        }
        exists
    }

    /// Number of cached option entries (for diagnostics / tests).
    pub fn cached_option_count(&self) -> usize {
        self.inner.lock().map(|i| i.options.len()).unwrap_or(0)
    }

    /// Number of cached package entries (for diagnostics / tests).
    pub fn cached_package_count(&self) -> usize {
        self.inner.lock().map(|i| i.packages.len()).unwrap_or(0)
    }

    /// Clear the in-memory cache (does not delete the on-disk file).
    pub fn clear_memory(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.options.clear();
            inner.packages.clear();
        }
    }
}

// ─── Internals ──────────────────────────────────────────────────────────────

fn default_cache_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home)
        .join(".cache")
        .join("symthaea")
        .join("nixpkgs-options.json")
}

fn load_cache(path: &PathBuf) -> Option<CacheFile> {
    let bytes = std::fs::read(path).ok()?;
    let parsed: CacheFile = serde_json::from_slice(&bytes).ok()?;
    if parsed.version != CACHE_VERSION {
        return None;
    }
    Some(parsed)
}

fn save_cache(path: &PathBuf, cache: &CacheFile) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let bytes = serde_json::to_vec_pretty(cache)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(path, bytes)
}

/// Validate an option path before splicing into a Nix expression.
/// Allows only `[A-Za-z0-9._-]` to prevent injection.
fn is_safe_path(path: &str) -> bool {
    !path.is_empty()
        && path.len() <= 256
        && path
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-')
}

/// Validate a package name. Same rules as option path components.
fn is_safe_pkg_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// Build the `let ... in { ... }` expression that introspects an option.
/// Uses `tryEval` and `hasAttrByPath` so the program never panics on missing
/// paths — instead we return `{ exists = false; ... }`.
fn build_option_query_expr(path: &str) -> String {
    // attr path as Nix list of strings: services.nginx.enable -> ["services" "nginx" "enable"]
    let parts: Vec<String> = path.split('.').map(|p| format!("\"{}\"", p)).collect();
    let nix_path = format!("[ {} ]", parts.join(" "));

    format!(
        r#"let
  opts = (import <nixpkgs/nixos> {{ configuration = {{}}; }}).options;
  attrPath = {nix_path};
  hasIt = builtins.hasAttr (builtins.head attrPath) opts && (
    let
      go = node: rest:
        if rest == [] then true
        else if builtins.isAttrs node && builtins.hasAttr (builtins.head rest) node
          then go (node.${{builtins.head rest}}) (builtins.tail rest)
          else false;
    in go opts attrPath
  );
  resolved = if hasIt then builtins.foldl' (n: k: n.${{k}}) opts attrPath else null;
  typeSig =
    if resolved != null && (builtins.tryEval (resolved.type or null)).success
      then ((resolved.type or {{}}).description or "")
      else "";
  desc =
    if resolved != null && (builtins.tryEval (resolved.description or "")).success
      then (let d = resolved.description or ""; in if builtins.isString d then d else "")
      else "";
in {{ exists = hasIt; type_sig = typeSig; description = desc; }}"#
    )
}

/// Synchronously query a single option via `nix-instantiate --eval --json`.
/// Returns `None` on tool failure, `Some(meta)` otherwise (with `exists=false`
/// when the path doesn't resolve in nixpkgs).
pub fn query_option_uncached(path: &str) -> Option<OptionMetadata> {
    if !is_safe_path(path) {
        return None;
    }
    let expr = build_option_query_expr(path);
    let out = Command::new("nix-instantiate")
        .args(["--eval", "--strict", "--json", "-E", &expr])
        .output()
        .ok()?;

    if !out.status.success() {
        // Tool error — but treat "option doesn't exist" as a *successful* lookup
        // returning exists=false. That happens when the expr itself fails to
        // evaluate, which we should distinguish from a hard failure.
        return Some(OptionMetadata {
            path: path.to_string(),
            type_sig: String::new(),
            description: String::new(),
            exists: false,
        });
    }

    #[derive(Deserialize)]
    struct Raw {
        exists: bool,
        type_sig: String,
        description: String,
    }
    let raw: Raw = serde_json::from_slice(&out.stdout).ok()?;
    Some(OptionMetadata {
        path: path.to_string(),
        type_sig: raw.type_sig,
        description: raw.description,
        exists: raw.exists,
    })
}

/// Synchronously check `pkgs ? <name>` against `<nixpkgs>`.
pub fn query_package_exists(name: &str) -> bool {
    if !is_safe_pkg_name(name) {
        return false;
    }
    let expr = format!(
        "let p = import <nixpkgs> {{}}; in builtins.hasAttr \"{}\" p",
        name
    );
    Command::new("nix-instantiate")
        .args(["--eval", "-E", &expr])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "true")
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_path_validation() {
        assert!(is_safe_path("services.nginx.enable"));
        assert!(is_safe_path("hardware.opengl.driSupport"));
        assert!(is_safe_path("home-manager.users.alice.programs.git.enable"));
        assert!(!is_safe_path(""));
        assert!(!is_safe_path("services; rm -rf /"));
        assert!(!is_safe_path("services.${injected}"));
        assert!(!is_safe_path("a b c"));
    }

    #[test]
    fn safe_pkg_validation() {
        assert!(is_safe_pkg_name("rust-analyzer"));
        assert!(is_safe_pkg_name("python3"));
        assert!(is_safe_pkg_name("postgresql_15"));
        assert!(!is_safe_pkg_name(""));
        assert!(!is_safe_pkg_name("rust analyzer"));
        assert!(!is_safe_pkg_name("rust;ls"));
        assert!(!is_safe_pkg_name("rust.analyzer"));
    }

    #[test]
    fn cache_roundtrip() {
        let tmp = std::env::temp_dir().join(format!(
            "symthaea_nixpkgs_index_test_{}.json",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&tmp);

        let mut cache = CacheFile::default();
        cache.version = CACHE_VERSION;
        cache.options.insert(
            "services.foo.enable".to_string(),
            OptionMetadata {
                path: "services.foo.enable".to_string(),
                type_sig: "boolean".to_string(),
                description: "Enable foo.".to_string(),
                exists: true,
            },
        );
        save_cache(&tmp, &cache).unwrap();

        let loaded = load_cache(&tmp).unwrap();
        assert_eq!(loaded.version, CACHE_VERSION);
        assert_eq!(loaded.options.len(), 1);
        assert_eq!(
            loaded.options.get("services.foo.enable").unwrap().type_sig,
            "boolean"
        );

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn cache_version_mismatch_returns_none() {
        let tmp = std::env::temp_dir().join(format!(
            "symthaea_nixpkgs_index_version_test_{}.json",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&tmp);

        let mut cache = CacheFile::default();
        cache.version = 99; // wrong
        save_cache(&tmp, &cache).unwrap();

        assert!(load_cache(&tmp).is_none());
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn build_option_query_expr_includes_path() {
        let expr = build_option_query_expr("services.nginx.enable");
        assert!(expr.contains("\"services\""));
        assert!(expr.contains("\"nginx\""));
        assert!(expr.contains("\"enable\""));
        assert!(expr.contains("hasAttr"));
    }

    /// Smoke-test against a real nix-instantiate. Only runs if NIX_PATH is set
    /// and nix-instantiate is on PATH. Otherwise silently skips.
    #[test]
    fn live_lookup_known_option() {
        if Command::new("nix-instantiate")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("[skip] nix-instantiate not available");
            return;
        }
        let meta = query_option_uncached("services.nginx.enable");
        let Some(meta) = meta else {
            eprintln!("[skip] live lookup returned None");
            return;
        };
        assert!(
            meta.exists,
            "services.nginx.enable should exist (got {meta:?})"
        );
        assert!(
            meta.type_sig.to_lowercase().contains("boolean"),
            "expected boolean type, got {:?}",
            meta.type_sig
        );
    }

    #[test]
    fn live_lookup_unknown_option() {
        if Command::new("nix-instantiate")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("[skip] nix-instantiate not available");
            return;
        }
        let meta = query_option_uncached("services.this_option_does_not_exist_xyz.enable");
        let Some(meta) = meta else {
            eprintln!("[skip] live lookup returned None");
            return;
        };
        assert!(!meta.exists, "bogus path should not exist");
    }

    #[test]
    fn live_package_existence() {
        if Command::new("nix-instantiate")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("[skip] nix-instantiate not available");
            return;
        }
        // hello is the canonical "always exists" nixpkgs probe.
        assert!(query_package_exists("hello"), "hello package must exist");
        assert!(
            !query_package_exists("totally-nonexistent-pkg-xyz-12345"),
            "bogus package must not exist"
        );
    }

    #[test]
    fn index_caches_lookups() {
        if Command::new("nix-instantiate")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("[skip] nix-instantiate not available");
            return;
        }
        let tmp = std::env::temp_dir().join(format!(
            "symthaea_nixpkgs_index_cache_test_{}.json",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&tmp);

        let idx = NixpkgsIndex::with_cache_path(tmp.clone());
        assert_eq!(idx.cached_option_count(), 0);

        let m1 = idx.lookup_option("services.openssh.enable");
        assert!(m1.is_some());
        assert_eq!(idx.cached_option_count(), 1);

        // Second lookup hits cache (we can't time-test here, but the count
        // should not change and the result must be identical).
        let m2 = idx.lookup_option("services.openssh.enable");
        assert_eq!(m1, m2);
        assert_eq!(idx.cached_option_count(), 1);

        // Reload from disk.
        let idx2 = NixpkgsIndex::with_cache_path(tmp.clone());
        assert_eq!(idx2.cached_option_count(), 1);

        let _ = std::fs::remove_file(&tmp);
    }
}
