// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tier 3: self-improvement loop for Nix codegen.
//!
//! Persists every generation that passes the full verification ladder
//! (parse + eval + option-index clean) so future prompts with similar
//! keywords retrieve a known-good answer instead of regenerating from
//! scratch. Cache lives at `~/.cache/symthaea/learned-idioms.json`.
//!
//! Match strategy: prompt → keyword set → Jaccard overlap against each
//! learned idiom → take best match above MIN_OVERLAP. Returns the most
//! recently verified idiom in the matched set (recency wins ties).
//!
//! This closes the loop between Tier 1 verification and codegen: each
//! verified output becomes a richer prior for the next prompt.

use crate::language::nix_codegen::NixIntent;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Mutex;

/// Single learned idiom — a prompt/code pair that previously passed verification.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct LearnedIdiom {
    /// Lowercased keyword set extracted from the original prompt.
    pub keywords: Vec<String>,
    /// Original prompt verbatim (for human inspection / debugging).
    pub prompt: String,
    /// The verified code (parses + evals + 0 unknown options).
    pub code: String,
    /// Classified intent at record time.
    pub intent: String,
    /// Unix epoch seconds at last verification.
    pub verified_at: u64,
    /// How many times this idiom has been retrieved successfully.
    pub hit_count: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
struct CacheFile {
    version: u32,
    idioms: Vec<LearnedIdiom>,
}

const CACHE_VERSION: u32 = 1;
/// Minimum Jaccard overlap on keyword sets to accept a match.
/// 0.4 means at least 40% of the union of {prompt, idiom} keywords
/// must overlap. Tuned to be selective: empirically catches paraphrases
/// without bleeding across intents.
pub const MIN_OVERLAP: f32 = 0.4;

/// Stop-words to drop before keyword extraction. Keep short.
const STOP: &[&str] = &[
    "a",
    "an",
    "the",
    "and",
    "or",
    "to",
    "for",
    "with",
    "of",
    "in",
    "on",
    "at",
    "by",
    "as",
    "is",
    "be",
    "set",
    "up",
    "configure",
    "enable",
    "make",
    "please",
    "i",
    "want",
    "need",
    "this",
    "that",
    "my",
    "me",
    "do",
    "can",
    "you",
    "should",
    "will",
    "would",
    "could",
];

/// Extract a normalized keyword set from a prompt.
pub fn extract_keywords(prompt: &str) -> Vec<String> {
    let lower = prompt.to_lowercase();
    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for tok in lower.split(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '_') {
        let t = tok.trim_matches(|c: char| !c.is_ascii_alphanumeric());
        if t.len() < 2 {
            continue;
        }
        if STOP.contains(&t) {
            continue;
        }
        if seen.insert(t.to_string()) {
            out.push(t.to_string());
        }
    }
    out
}

fn jaccard(a: &[String], b: &[String]) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let aset: HashSet<&String> = a.iter().collect();
    let bset: HashSet<&String> = b.iter().collect();
    let inter = aset.intersection(&bset).count() as f32;
    let union = aset.union(&bset).count() as f32;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Disk-backed cache of learned idioms.
pub struct LearnedIdiomCache {
    cache_path: PathBuf,
    inner: Mutex<CacheFile>,
}

impl LearnedIdiomCache {
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

    /// Number of stored idioms.
    pub fn len(&self) -> usize {
        self.inner.lock().map(|i| i.idioms.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Find the best-matching learned idiom for `prompt`, or None if no
    /// stored idiom clears `MIN_OVERLAP`. Increments `hit_count` on success.
    pub fn find_match(&self, prompt: &str) -> Option<LearnedIdiom> {
        let keywords = extract_keywords(prompt);
        if keywords.is_empty() {
            return None;
        }
        let prompt_set: HashSet<&String> = keywords.iter().collect();

        let mut inner = self.inner.lock().ok()?;

        // Find the highest-scoring idiom that clears MIN_OVERLAP AND
        // satisfies subset containment: every keyword in the new prompt
        // must appear in the cached idiom's keyword set. Without this,
        // "rust dev with sccache" would Jaccard-match a cached "rust dev
        // with rust-analyzer" entry (overlap 2/3 ≥ MIN_OVERLAP) and
        // serve the rust-analyzer answer that doesn't mention sccache.
        //
        // The subset rule treats cached idioms as supersets — if every
        // ask in the new prompt is covered by what the cached prompt
        // asked for, the cached answer is plausibly fit. Adds false
        // negatives when the cached answer is actually a superset of the
        // new ask but the cache prompt lacks a keyword the new prompt
        // mentions; we accept that to avoid the silent wrong-answer
        // failure mode.
        let mut best_idx: Option<usize> = None;
        let mut best_score: f32 = -1.0;
        for (idx, idiom) in inner.idioms.iter().enumerate() {
            let score = jaccard(&keywords, &idiom.keywords);
            if score < MIN_OVERLAP {
                continue;
            }
            let cached_set: HashSet<&String> = idiom.keywords.iter().collect();
            if !prompt_set.is_subset(&cached_set) {
                continue;
            }
            let take = match best_idx {
                None => true,
                Some(b) => {
                    if score > best_score {
                        true
                    } else if score == best_score {
                        let cur = &inner.idioms[b];
                        idiom.verified_at > cur.verified_at
                            || (idiom.verified_at == cur.verified_at
                                && idiom.hit_count > cur.hit_count)
                    } else {
                        false
                    }
                }
            };
            if take {
                best_idx = Some(idx);
                best_score = score;
            }
        }

        let idx = best_idx?;
        inner.idioms[idx].hit_count = inner.idioms[idx].hit_count.saturating_add(1);
        let result = inner.idioms[idx].clone();
        let _ = save_cache(&self.cache_path, &inner);
        Some(result)
    }

    /// Record a verified idiom. If a near-identical one already exists
    /// (same keyword set + same code), bump its hit_count instead of
    /// duplicating.
    pub fn record(&self, prompt: &str, code: &str, intent: NixIntent) {
        let keywords = extract_keywords(prompt);
        if keywords.is_empty() {
            return;
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        if let Ok(mut inner) = self.inner.lock() {
            // De-dup: same keyword set + same code → bump.
            for existing in inner.idioms.iter_mut() {
                if existing.keywords == keywords && existing.code == code {
                    existing.verified_at = now;
                    existing.hit_count = existing.hit_count.saturating_add(1);
                    let _ = save_cache(&self.cache_path, &inner);
                    return;
                }
            }
            inner.version = CACHE_VERSION;
            inner.idioms.push(LearnedIdiom {
                keywords,
                prompt: prompt.to_string(),
                code: code.to_string(),
                intent: format!("{intent:?}"),
                verified_at: now,
                hit_count: 1,
            });
            let _ = save_cache(&self.cache_path, &inner);
        }
    }

    /// Drop all idioms older than `max_age_secs`.
    pub fn prune_older_than(&self, max_age_secs: u64) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if let Ok(mut inner) = self.inner.lock() {
            let before = inner.idioms.len();
            inner
                .idioms
                .retain(|i| now.saturating_sub(i.verified_at) <= max_age_secs);
            if inner.idioms.len() != before {
                let _ = save_cache(&self.cache_path, &inner);
            }
        }
    }
}

// ─── Internals ──────────────────────────────────────────────────────────────

fn default_cache_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home)
        .join(".cache")
        .join("symthaea")
        .join("learned-idioms.json")
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

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_cache(suffix: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "symthaea_learned_idioms_{}_{}_{}.json",
            suffix,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ))
    }

    #[test]
    fn extract_keywords_drops_stopwords() {
        let kw = extract_keywords("Please set up a rust dev environment with rust-analyzer");
        assert!(kw.contains(&"rust".to_string()));
        assert!(kw.contains(&"dev".to_string()));
        assert!(kw.contains(&"environment".to_string()));
        assert!(kw.contains(&"rust-analyzer".to_string()));
        assert!(!kw.contains(&"please".to_string()));
        assert!(!kw.contains(&"set".to_string()));
        assert!(!kw.contains(&"up".to_string()));
        assert!(!kw.contains(&"a".to_string()));
        assert!(!kw.contains(&"with".to_string()));
    }

    #[test]
    fn extract_keywords_dedups() {
        let kw = extract_keywords("rust rust rust");
        assert_eq!(kw, vec!["rust".to_string()]);
    }

    #[test]
    fn jaccard_basic() {
        let a = vec!["rust".to_string(), "dev".to_string()];
        let b = vec!["rust".to_string(), "shell".to_string()];
        assert!((jaccard(&a, &b) - 1.0 / 3.0).abs() < 1e-6);

        let c = vec!["rust".to_string(), "dev".to_string()];
        assert_eq!(jaccard(&a, &c), 1.0);

        let d = vec!["python".to_string(), "data".to_string()];
        assert_eq!(jaccard(&a, &d), 0.0);
    }

    #[test]
    fn cache_round_trip() {
        let path = tmp_cache("rt");
        let _ = std::fs::remove_file(&path);
        let cache = LearnedIdiomCache::with_cache_path(path.clone());
        assert_eq!(cache.len(), 0);

        cache.record(
            "set up postgresql with pgvector",
            "{ services.postgresql.enable = true; }",
            NixIntent::Service,
        );
        assert_eq!(cache.len(), 1);

        // Reload from disk.
        let cache2 = LearnedIdiomCache::with_cache_path(path.clone());
        assert_eq!(cache2.len(), 1);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn find_match_returns_best_overlap() {
        let path = tmp_cache("match");
        let _ = std::fs::remove_file(&path);
        let cache = LearnedIdiomCache::with_cache_path(path.clone());

        cache.record(
            "set up postgresql with pgvector",
            "{ services.postgresql.enable = true; }",
            NixIntent::Service,
        );
        cache.record(
            "set up nginx with reverse proxy",
            "{ services.nginx.enable = true; }",
            NixIntent::Service,
        );

        // Strict subset of #1's keywords — should match (cached idiom is a
        // superset of what the new prompt asks for, so the answer fits).
        let m = cache.find_match("postgresql pgvector");
        assert!(m.is_some(), "subset paraphrase should match: {:?}", m);
        assert!(
            m.unwrap().code.contains("postgresql"),
            "matched the wrong idiom"
        );

        // Has unique keywords not in any cached entry — should NOT match.
        // Without this rejection, "install postgresql for vector database"
        // would Jaccard-match the postgresql idiom and return code that
        // doesn't address the "vector database" specifics.
        let m_extra = cache.find_match("install postgresql for vector database");
        assert!(
            m_extra.is_none(),
            "prompt with extra keywords should NOT match: {:?}",
            m_extra
        );

        // Unrelated → None.
        let m2 = cache.find_match("enable wayland with sway");
        assert!(m2.is_none(), "should not falsely match: {:?}", m2);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn record_dedups_identical() {
        let path = tmp_cache("dedup");
        let _ = std::fs::remove_file(&path);
        let cache = LearnedIdiomCache::with_cache_path(path.clone());

        let prompt = "set up rust dev environment";
        let code = "{ pkgs ? import <nixpkgs> {} }: pkgs.mkShell { buildInputs = [ pkgs.rust-analyzer ]; }";
        cache.record(prompt, code, NixIntent::DevShell);
        cache.record(prompt, code, NixIntent::DevShell);
        cache.record(prompt, code, NixIntent::DevShell);
        assert_eq!(
            cache.len(),
            1,
            "identical record() calls should de-duplicate"
        );
        let m = cache.find_match(prompt).unwrap();
        // 3 records + 1 hit from find_match = 4
        assert_eq!(m.hit_count, 4, "hit count should increment on each touch");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn empty_prompt_does_not_record() {
        let path = tmp_cache("empty");
        let _ = std::fs::remove_file(&path);
        let cache = LearnedIdiomCache::with_cache_path(path.clone());
        cache.record("", "{}", NixIntent::Generic);
        cache.record("    ", "{}", NixIntent::Generic);
        assert_eq!(cache.len(), 0, "empty / whitespace prompts must be dropped");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn find_match_increments_hit_count() {
        let path = tmp_cache("hits");
        let _ = std::fs::remove_file(&path);
        let cache = LearnedIdiomCache::with_cache_path(path.clone());

        cache.record(
            "configure docker daemon",
            "{ virtualisation.docker.enable = true; }",
            NixIntent::Service,
        );
        let _ = cache.find_match("configure docker daemon");
        let m = cache.find_match("docker daemon").unwrap();
        // 1 record + 2 find_match = 3
        assert_eq!(m.hit_count, 3);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn prune_drops_old_entries() {
        let path = tmp_cache("prune");
        let _ = std::fs::remove_file(&path);
        let cache = LearnedIdiomCache::with_cache_path(path.clone());
        cache.record("foo bar baz", "{}", NixIntent::Generic);
        assert_eq!(cache.len(), 1);
        // max_age = 0 → all entries are "older than 0 seconds" once even
        // a moment passes. Sleep a moment to ensure the comparison fires.
        std::thread::sleep(std::time::Duration::from_millis(1100));
        cache.prune_older_than(0);
        assert_eq!(cache.len(), 0, "max_age=0 should drop everything");
        let _ = std::fs::remove_file(&path);
    }
}
