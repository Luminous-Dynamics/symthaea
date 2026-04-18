// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Content-hash cache for `try_nix_module_eval` verdicts (P3).
//!
//! `try_nix_module_eval` wraps a snippet as a NixOS module and runs
//! `nix-instantiate '<nixpkgs/nixos>' -A system` — the strongest
//! pre-build verification we have, at 5–30s per call on a warm cache
//! and much longer cold. The per-problem cost makes it infeasible to
//! enable in the 94-problem NixEval benchmark without caching.
//!
//! This module caches deterministic verdicts (success, eval errors)
//! keyed by `sha256(normalized_snippet) + nixpkgs_rev`. Transient
//! errors (nix-instantiate not found, IO errors) are **not** cached —
//! those should be retried next run.
//!
//! Normalization collapses cosmetic differences (whitespace, line
//! comments) so two snippets that would eval identically share a
//! cache entry.
//!
//! Disk location: `~/.cache/symthaea/module-eval.json`. Format is JSON
//! with a version field; a mismatched version drops the whole file on
//! load (same regenerative policy as `learned_idioms`).

use crate::language::nix_codegen::{try_nix_module_eval, NixVerdict};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// Bumped when the verdict schema or normalization changes. v1 = initial.
const CACHE_VERSION: u32 = 1;

/// Cacheable shape of a verdict. Avoids adding serde derives to the
/// public `NixVerdict` enum (which lives in `nix_codegen.rs` and is
/// exposed as a structural API).
#[derive(Clone, Debug, Serialize, Deserialize)]
struct CachedVerdict {
    /// True if `NixVerdict::ParseOk`, false otherwise.
    ok: bool,
    /// Error message if any. Empty on ParseOk.
    msg: String,
}

impl From<&NixVerdict> for CachedVerdict {
    fn from(v: &NixVerdict) -> Self {
        match v {
            NixVerdict::ParseOk => CachedVerdict {
                ok: true,
                msg: String::new(),
            },
            NixVerdict::ParseError(m) => CachedVerdict {
                ok: false,
                msg: m.clone(),
            },
        }
    }
}

impl From<CachedVerdict> for NixVerdict {
    fn from(c: CachedVerdict) -> NixVerdict {
        if c.ok {
            NixVerdict::ParseOk
        } else {
            NixVerdict::ParseError(c.msg)
        }
    }
}

/// Disk-file shape. `HashMap<cache_key, verdict>`.
#[derive(Debug, Serialize, Deserialize, Default)]
struct CacheFile {
    version: u32,
    entries: HashMap<String, CachedVerdict>,
}

/// Process-wide cache singleton. Reused across benchmark problems so
/// one run populates and the next run reads through it. Initialised
/// lazily to avoid cost on codepaths that never touch module eval.
static SHARED_CACHE: OnceLock<ModuleEvalCache> = OnceLock::new();

pub fn shared_module_eval_cache() -> &'static ModuleEvalCache {
    SHARED_CACHE.get_or_init(ModuleEvalCache::default_cache)
}

pub struct ModuleEvalCache {
    cache_path: PathBuf,
    inner: Mutex<CacheFile>,
}

impl ModuleEvalCache {
    pub fn default_cache() -> Self {
        let path = default_cache_path();
        let inner = load_cache(&path).unwrap_or_default();
        Self {
            cache_path: path,
            inner: Mutex::new(inner),
        }
    }

    pub fn with_cache_path(path: PathBuf) -> Self {
        let inner = load_cache(&path).unwrap_or_default();
        Self {
            cache_path: path,
            inner: Mutex::new(inner),
        }
    }

    /// Number of cached verdicts — primarily for benchmark telemetry.
    pub fn len(&self) -> usize {
        self.inner.lock().map(|i| i.entries.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Look up a cached verdict by snippet. Returns the verdict on hit,
    /// None on miss. Pure read — does not populate.
    pub fn get(&self, snippet: &str) -> Option<NixVerdict> {
        let key = cache_key_for(snippet)?;
        let inner = self.inner.lock().ok()?;
        inner.entries.get(&key).cloned().map(Into::into)
    }

    /// Store a verdict. Idempotent — overwriting is fine (same key =
    /// same input, so the verdict should be stable unless nixpkgs
    /// moved under us; that's what the `nixpkgs_rev` half of the key
    /// guards against).
    pub fn put(&self, snippet: &str, verdict: &NixVerdict) {
        let Some(key) = cache_key_for(snippet) else {
            return;
        };
        if !is_cacheable(verdict) {
            return;
        }
        if let Ok(mut inner) = self.inner.lock() {
            inner.version = CACHE_VERSION;
            inner.entries.insert(key, CachedVerdict::from(verdict));
            let _ = save_cache(&self.cache_path, &inner);
        }
    }
}

/// Top-level cached wrapper. Reads cache, falls through to the
/// uncached `try_nix_module_eval` on miss, stores deterministic
/// verdicts, returns whatever `try_nix_module_eval` returned.
///
/// Returns None only when the underlying function does — i.e. when
/// the snippet doesn't look like a NixOS module.
pub fn cached_module_eval(snippet: &str) -> Option<NixVerdict> {
    cached_module_eval_using(snippet, shared_module_eval_cache())
}

/// Injectable-cache variant for tests + benchmarks that want a
/// per-run tempfile cache. Mirrors the pattern in `learned_idioms.rs`.
pub fn cached_module_eval_using(snippet: &str, cache: &ModuleEvalCache) -> Option<NixVerdict> {
    if let Some(hit) = cache.get(snippet) {
        return Some(hit);
    }
    let verdict = try_nix_module_eval(snippet)?;
    cache.put(snippet, &verdict);
    Some(verdict)
}

// ─── Internals ──────────────────────────────────────────────────────────────

fn default_cache_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home)
        .join(".cache")
        .join("symthaea")
        .join("module-eval.json")
}

/// Compute the cache key: blake3(normalized_snippet + "\0" + nixpkgs_rev).
/// Returns None if normalization fails (empty snippet etc.) — callers
/// treat None as "don't cache this one."
fn cache_key_for(snippet: &str) -> Option<String> {
    let norm = normalize_snippet(snippet);
    if norm.trim().is_empty() {
        return None;
    }
    let rev = detect_nixpkgs_rev().unwrap_or_else(|| "unknown".to_string());
    let mut h = blake3::Hasher::new();
    h.update(norm.as_bytes());
    h.update(b"\0");
    h.update(rev.as_bytes());
    Some(h.finalize().to_hex().to_string())
}

/// Pre-hash normalization. Drops line comments, pads structural
/// punctuation so `a=b;` and `a = b ;` collapse to the same token
/// sequence, then collapses runs of whitespace to a single space.
/// NOT a nixfmt pass — we care about "would this eval the same?",
/// not "does this look pretty?".
fn normalize_snippet(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    for line in src.lines() {
        // Drop `# …` to end-of-line. As in nix_scorer, this is a
        // heuristic — `#` inside a string is rare in NixOS configs;
        // false positives here would only cause spurious cache
        // misses, not wrong answers.
        let code = match line.find('#') {
            Some(idx) => &line[..idx],
            None => line,
        };
        out.push_str(code);
        out.push(' ');
    }
    // Pad structural punctuation so spaces around them are optional.
    let padded = out
        .replace('{', " { ")
        .replace('}', " } ")
        .replace('[', " [ ")
        .replace(']', " ] ")
        .replace(';', " ; ")
        .replace('=', " = ")
        .replace(',', " , ");
    // Collapse whitespace runs.
    padded.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Is this verdict worth caching? Deterministic outcomes (success,
/// module-eval errors with a real nix message) yes. Transient errors
/// that look like nix-instantiate IO failures no — those would stick
/// in the cache forever if the user's PATH is temporarily wrong.
fn is_cacheable(verdict: &NixVerdict) -> bool {
    match verdict {
        NixVerdict::ParseOk => true,
        NixVerdict::ParseError(msg) => {
            // Heuristics for "our setup is broken, not the snippet":
            let m = msg.to_lowercase();
            if m.contains("nix-instantiate:") // spawn / IO error
                || m.contains("write snippet:") // tempfile IO failure
                || m.contains("write wrapper:") // same
                || m.contains("no such file") // toolchain path missing
                || m.contains("permission denied")
            {
                return false;
            }
            true
        }
    }
}

/// Same best-effort detection as `learned_idioms::detect_nixpkgs_rev`.
/// Deliberately duplicated (6 lines) to avoid cross-module visibility
/// gymnastics; the alternative was a `pub fn` leak that nothing else
/// outside these two callers would need.
fn detect_nixpkgs_rev() -> Option<String> {
    if let Ok(v) = std::env::var("SYMTHAEA_NIXPKGS_REV") {
        if !v.trim().is_empty() {
            return Some(v);
        }
    }
    if let Ok(v) = std::fs::read_to_string("/run/current-system/nixos-version") {
        let s = v.trim();
        if !s.is_empty() {
            return Some(s.to_string());
        }
    }
    None
}

fn load_cache(path: &Path) -> Option<CacheFile> {
    let bytes = std::fs::read(path).ok()?;
    let parsed: CacheFile = serde_json::from_slice(&bytes).ok()?;
    if parsed.version != CACHE_VERSION {
        return None;
    }
    Some(parsed)
}

fn save_cache(path: &Path, cache: &CacheFile) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let bytes = serde_json::to_vec_pretty(cache)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(path, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_cache(suffix: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "symthaea_module_eval_cache_{}_{}_{}.json",
            suffix,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ))
    }

    #[test]
    fn normalize_collapses_whitespace_and_comments() {
        let a = "{ services.nginx.enable   = true; # comment here\n}";
        let b = "{services.nginx.enable=true;}";
        // These are cosmetically different but semantically identical.
        // Key should match.
        assert_eq!(
            cache_key_for(a),
            cache_key_for(b),
            "whitespace/comment normalization must produce the same key"
        );
    }

    #[test]
    fn different_content_different_key() {
        let a = "{ services.nginx.enable = true; }";
        let b = "{ services.postgresql.enable = true; }";
        assert_ne!(cache_key_for(a), cache_key_for(b));
    }

    #[test]
    fn empty_snippet_gets_no_key() {
        assert!(cache_key_for("").is_none());
        assert!(cache_key_for("   \n\n # only comment \n").is_none());
    }

    #[test]
    fn cacheable_classifies_transient_errors_as_skip() {
        assert!(is_cacheable(&NixVerdict::ParseOk));
        assert!(is_cacheable(&NixVerdict::ParseError(
            "error: attribute 'enabled' missing, did you mean 'enable'?".into()
        )));
        assert!(!is_cacheable(&NixVerdict::ParseError(
            "nix-instantiate: not found".into()
        )));
        assert!(!is_cacheable(&NixVerdict::ParseError(
            "write snippet: permission denied".into()
        )));
    }

    #[test]
    fn put_and_get_round_trip() {
        let path = tmp_cache("rt");
        let _ = std::fs::remove_file(&path);
        let cache = ModuleEvalCache::with_cache_path(path.clone());
        assert!(cache.is_empty());

        let snippet = "{ services.nginx.enable = true; }";
        cache.put(snippet, &NixVerdict::ParseOk);
        assert_eq!(cache.len(), 1);

        let hit = cache.get(snippet).expect("should hit");
        assert!(hit.is_ok());

        // Reload from disk — verdict must survive.
        let cache2 = ModuleEvalCache::with_cache_path(path.clone());
        let hit2 = cache2.get(snippet).expect("reload hit");
        assert!(hit2.is_ok());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn put_skips_transient_errors() {
        let path = tmp_cache("transient");
        let _ = std::fs::remove_file(&path);
        let cache = ModuleEvalCache::with_cache_path(path.clone());

        cache.put(
            "{ services.foo = true; }",
            &NixVerdict::ParseError("nix-instantiate: not found".into()),
        );
        assert_eq!(
            cache.len(),
            0,
            "transient errors must not populate the cache"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn put_stores_deterministic_errors() {
        let path = tmp_cache("det");
        let _ = std::fs::remove_file(&path);
        let cache = ModuleEvalCache::with_cache_path(path.clone());

        cache.put(
            "{ services.nginx.enabled = true; }",
            &NixVerdict::ParseError(
                "error: The option `services.nginx.enabled' is used but not defined.".into(),
            ),
        );
        assert_eq!(cache.len(), 1);

        let hit = cache.get("{ services.nginx.enabled = true; }").unwrap();
        assert!(!hit.is_ok());
        assert!(hit.message().contains("not defined"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn corrupt_cache_file_is_discarded() {
        let path = tmp_cache("corrupt");
        let _ = std::fs::remove_file(&path);
        std::fs::write(&path, b"not json at all, just garbage {{{}").unwrap();

        let cache = ModuleEvalCache::with_cache_path(path.clone());
        // Corrupt file → empty cache, not a panic.
        assert!(cache.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn version_mismatch_drops_entries() {
        let path = tmp_cache("version");
        let _ = std::fs::remove_file(&path);
        // Hand-write a v0 cache file.
        std::fs::write(
            &path,
            r#"{"version": 0, "entries": {"foo": {"ok": true, "msg": ""}}}"#,
        )
        .unwrap();
        let cache = ModuleEvalCache::with_cache_path(path.clone());
        assert!(cache.is_empty(), "v0 entries must not load into v1 cache");
        let _ = std::fs::remove_file(&path);
    }
}
