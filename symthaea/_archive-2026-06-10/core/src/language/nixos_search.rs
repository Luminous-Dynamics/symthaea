// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tier 2: scoped, authoritative web search against `search.nixos.org`.
//!
//! Lets the codegen pipeline fuzzy-search NixOS module options when the
//! `nixpkgs_index` doesn't have a hit and we don't yet know the exact
//! option path to verify.
//!
//! Strategy: use the public Elasticsearch endpoint behind search.nixos.org
//! via `curl` (sync, no async dep). Cache results to
//! `~/.cache/symthaea/nixos-search.json` so repeat queries are free.
//!
//! We deliberately do NOT scrape Stack Overflow / GitHub / general Google
//! results — too noisy, and the search.nixos.org index is the
//! authoritative source for option metadata.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;

/// Single search hit — a NixOS module option with metadata.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SearchHit {
    /// Dotted option path (e.g. `services.postgresql.enable`).
    pub option_name: String,
    /// Type signature ("boolean", "list of strings", etc.).
    pub option_type: String,
    /// Description (may include rendered HTML — caller should strip if needed).
    pub option_description: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
struct CacheFile {
    version: u32,
    /// Keyed by query string (lowercase).
    queries: HashMap<String, Vec<SearchHit>>,
}

const CACHE_VERSION: u32 = 1;

/// The NixOS Elasticsearch alias. Kept in source so we can bump it as the
/// search index version increments. Verified working: 2026-04-18.
const SEARCH_INDEX: &str = "latest-46-nixos-unstable";
const SEARCH_BASE: &str = "https://search.nixos.org/backend";
/// Public credentials — these are bundled in the search.nixos.org frontend
/// JS, so they're not secret. We pin them here so the lookup keeps working
/// when the user is logged out / hasn't seen the page.
const SEARCH_USER: &str = "aWVSALXpZv";
const SEARCH_PASS: &str = "X8gPHnzL52wFEekuxsfQ9cSh";

/// Disk-cached, sync handle to search.nixos.org option queries.
pub struct NixosSearch {
    cache_path: PathBuf,
    inner: Mutex<CacheFile>,
}

impl NixosSearch {
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

    /// Search for options whose name has the given prefix (e.g. `"services.postgresql"`).
    /// Returns up to `limit` hits ordered by the search index's relevance.
    /// Cached after the first call.
    pub fn search_options_by_prefix(&self, prefix: &str, limit: usize) -> Vec<SearchHit> {
        let key = format!("prefix:{}:{}", prefix.to_lowercase(), limit);
        if let Some(hits) = self.cache_get(&key) {
            return hits;
        }

        let body = build_prefix_query(prefix, limit);
        let hits = run_search(&body).unwrap_or_default();

        self.cache_put(key, hits.clone());
        hits
    }

    /// Look up an exact option-path match (e.g. `services.postgresql.enable`).
    /// Returns the hit if it exists in the search index, else None.
    /// Cached after the first call.
    pub fn lookup_option(&self, name: &str) -> Option<SearchHit> {
        let key = format!("exact:{}", name.to_lowercase());
        if let Some(hits) = self.cache_get(&key) {
            return hits.into_iter().next();
        }

        let body = build_exact_query(name);
        let hits = run_search(&body).unwrap_or_default();
        self.cache_put(key, hits.clone());
        hits.into_iter().next()
    }

    /// Return the number of cached query results (for diagnostics / tests).
    pub fn cached_query_count(&self) -> usize {
        self.inner.lock().map(|i| i.queries.len()).unwrap_or(0)
    }

    fn cache_get(&self, key: &str) -> Option<Vec<SearchHit>> {
        self.inner.lock().ok()?.queries.get(key).cloned()
    }

    fn cache_put(&self, key: String, hits: Vec<SearchHit>) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.version = CACHE_VERSION;
            inner.queries.insert(key, hits);
            let _ = save_cache(&self.cache_path, &inner);
        }
    }
}

// ─── Internals ──────────────────────────────────────────────────────────────

fn default_cache_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home)
        .join(".cache")
        .join("symthaea")
        .join("nixos-search.json")
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

/// Same path-safety rule as `nixpkgs_index::is_safe_path` — prevent JSON
/// injection by rejecting any character outside the option-path alphabet.
fn is_safe_query(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 256
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-')
}

fn build_prefix_query(prefix: &str, limit: usize) -> String {
    let safe = if is_safe_query(prefix) { prefix } else { "" };
    let limit = limit.clamp(1, 100);
    format!(
        r#"{{"query":{{"bool":{{"filter":[{{"term":{{"type":"option"}}}}],"must":[{{"prefix":{{"option_name":"{safe}"}}}}]}}}},"size":{limit}}}"#
    )
}

fn build_exact_query(name: &str) -> String {
    let safe = if is_safe_query(name) { name } else { "" };
    format!(
        r#"{{"query":{{"bool":{{"filter":[{{"term":{{"type":"option"}}}}],"must":[{{"term":{{"option_name":"{safe}"}}}}]}}}},"size":1}}"#
    )
}

fn run_search(body: &str) -> Option<Vec<SearchHit>> {
    let url = format!("{SEARCH_BASE}/{SEARCH_INDEX}/_search");
    let auth = format!("{SEARCH_USER}:{SEARCH_PASS}");

    let out = Command::new("curl")
        .args([
            "-s",
            "-m",
            "10",
            "--fail",
            "-u",
            &auth,
            "-H",
            "Content-Type: application/json",
            "-d",
            body,
            &url,
        ])
        .output()
        .ok()?;

    if !out.status.success() {
        return None;
    }
    parse_response(&out.stdout)
}

fn parse_response(bytes: &[u8]) -> Option<Vec<SearchHit>> {
    #[derive(Deserialize)]
    struct EsResp {
        hits: HitsBlock,
    }
    #[derive(Deserialize)]
    struct HitsBlock {
        hits: Vec<HitWrap>,
    }
    #[derive(Deserialize)]
    struct HitWrap {
        #[serde(rename = "_source")]
        source: serde_json::Value,
    }
    let resp: EsResp = serde_json::from_slice(bytes).ok()?;
    let mut out = Vec::with_capacity(resp.hits.hits.len());
    for h in resp.hits.hits {
        let s = h.source;
        let name = s
            .get("option_name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if name.is_empty() {
            continue;
        }
        let typ = s
            .get("option_type")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let desc = s
            .get("option_description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        out.push(SearchHit {
            option_name: name,
            option_type: typ,
            option_description: desc,
        });
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_query_validation() {
        assert!(is_safe_query("services.postgresql"));
        assert!(is_safe_query("hardware.opengl"));
        assert!(!is_safe_query(""));
        assert!(!is_safe_query("services\";evil"));
        assert!(!is_safe_query("services.${injected}"));
    }

    #[test]
    fn build_prefix_query_escapes_unsafe() {
        // Unsafe input collapses to empty prefix — caller will get empty
        // results rather than an injection.
        let q = build_prefix_query("a;b", 5);
        assert!(
            q.contains(r#""option_name":"""#),
            "unsafe input should be blanked, got: {q}"
        );
    }

    #[test]
    fn build_prefix_query_clamps_limit() {
        let q = build_prefix_query("services", 99999);
        assert!(q.contains("\"size\":100"));
        let q2 = build_prefix_query("services", 0);
        assert!(q2.contains("\"size\":1"));
    }

    #[test]
    fn parse_response_extracts_hits() {
        let raw = br#"{
  "hits": {
    "hits": [
      {"_source": {"option_name": "services.nginx.enable", "option_type": "boolean", "option_description": "Enable nginx."}},
      {"_source": {"option_name": "services.nginx.user", "option_type": "string", "option_description": ""}}
    ]
  }
}"#;
        let hits = parse_response(raw).expect("should parse");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].option_name, "services.nginx.enable");
        assert_eq!(hits[0].option_type, "boolean");
        assert_eq!(hits[1].option_name, "services.nginx.user");
    }

    #[test]
    fn parse_response_skips_empty_names() {
        let raw = br#"{
  "hits": {
    "hits": [
      {"_source": {"option_name": "", "option_type": "x", "option_description": ""}},
      {"_source": {"option_name": "services.openssh.enable", "option_type": "boolean", "option_description": ""}}
    ]
  }
}"#;
        let hits = parse_response(raw).expect("should parse");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].option_name, "services.openssh.enable");
    }

    #[test]
    fn cache_roundtrip() {
        let tmp = std::env::temp_dir().join(format!(
            "symthaea_nixos_search_test_{}.json",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&tmp);

        let mut cache = CacheFile::default();
        cache.version = CACHE_VERSION;
        cache.queries.insert(
            "prefix:services.foo:5".to_string(),
            vec![SearchHit {
                option_name: "services.foo.enable".to_string(),
                option_type: "boolean".to_string(),
                option_description: "Enable foo.".to_string(),
            }],
        );
        save_cache(&tmp, &cache).unwrap();

        let loaded = load_cache(&tmp).unwrap();
        assert_eq!(loaded.queries.len(), 1);
        assert_eq!(
            loaded.queries.get("prefix:services.foo:5").unwrap()[0].option_name,
            "services.foo.enable"
        );

        let _ = std::fs::remove_file(&tmp);
    }

    /// Live test: hits the real search.nixos.org endpoint. Skipped if curl
    /// isn't available or the network is unreachable.
    #[test]
    fn live_prefix_search_postgresql() {
        if Command::new("curl").arg("--version").output().is_err() {
            eprintln!("[skip] curl not available");
            return;
        }
        let tmp = std::env::temp_dir().join(format!(
            "symthaea_nixos_search_live_{}.json",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&tmp);

        let s = NixosSearch::with_cache_path(tmp.clone());
        let hits = s.search_options_by_prefix("services.postgresql", 5);
        if hits.is_empty() {
            eprintln!("[skip] live search returned 0 hits — likely offline");
            return;
        }
        assert!(
            hits.iter()
                .any(|h| h.option_name == "services.postgresql.enable"),
            "expected services.postgresql.enable in hits, got: {hits:?}"
        );

        // Second call should hit cache.
        let before = s.cached_query_count();
        let _ = s.search_options_by_prefix("services.postgresql", 5);
        assert_eq!(s.cached_query_count(), before);

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn live_exact_lookup() {
        if Command::new("curl").arg("--version").output().is_err() {
            eprintln!("[skip] curl not available");
            return;
        }
        let tmp = std::env::temp_dir().join(format!(
            "symthaea_nixos_search_exact_{}.json",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&tmp);

        let s = NixosSearch::with_cache_path(tmp.clone());
        let hit = s.lookup_option("services.openssh.enable");
        let Some(hit) = hit else {
            eprintln!("[skip] live exact lookup returned None — likely offline");
            return;
        };
        assert_eq!(hit.option_name, "services.openssh.enable");
        assert!(hit.option_type.to_lowercase().contains("boolean"));

        let _ = std::fs::remove_file(&tmp);
    }
}
