// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HTTP gallery server (feature = `server`).
//!
//! A small, self-contained axum server that makes the flat-file gallery
//! store *visible*: it lists `index.json` entries, renders stored SVGs,
//! and accepts human ratings that write through to the same persisted
//! `AestheticMemory` file that `Symthaea::rate_art` and the cognitive
//! loop's `CreativeManager` use (`.claude/aesthetic_memory.json` by
//! default). VISUAL_ART_IMPROVEMENT_PLAN_2026-07-10 Phase 4.1.
//!
//! Design notes:
//! - symthaea-web is pure static CSR (Trunk/WASM, no backend), so the
//!   gallery surface lives here as a standalone server-rendered binary —
//!   simple and working over fancy.
//! - All handler logic is exposed as plain functions
//!   ([`load_gallery`], [`resolve_visual_svg`], [`apply_rating`],
//!   [`render_gallery_html`]) so it can be tested without a network.
//! - The gallery index is re-read from disk per request: it is a small
//!   flat file, and the generating process may rewrite it at any time.
//! - Ratings use `AestheticTracker::human_feedback_unattributed` — the
//!   server does not know the generation-time harmony state, and (per the
//!   Phase 8.5 honesty note mirrored from `Symthaea::rate_art`) absent
//!   information must mean *no* harmony-bias update, not fabricated data.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::Router;
use axum::extract::{Json, Path as UrlPath, State};
use axum::http::{StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::storage::GalleryStorage;
use crate::{ArtModality, GalleryEntry, GalleryIndex};

/// Default gallery root, sibling convention to `.claude/aesthetic_memory.json`.
pub const DEFAULT_GALLERY_ROOT: &str = ".claude/gallery";
/// Default persisted aesthetic-memory path — must match
/// `creative_bridge::AESTHETIC_MEMORY_PATH` in the main crate so ratings
/// land in the same file the cognitive loop reads.
pub const DEFAULT_AESTHETIC_MEMORY_PATH: &str = ".claude/aesthetic_memory.json";

/// Server configuration.
#[derive(Debug, Clone)]
pub struct GalleryServerConfig {
    /// Root directory of the flat-file gallery store (contains `index.json`).
    pub gallery_root: PathBuf,
    /// Path of the persisted `AestheticMemory` JSON that ratings write to.
    pub aesthetic_memory_path: PathBuf,
}

impl Default for GalleryServerConfig {
    fn default() -> Self {
        Self {
            gallery_root: PathBuf::from(DEFAULT_GALLERY_ROOT),
            aesthetic_memory_path: PathBuf::from(DEFAULT_AESTHETIC_MEMORY_PATH),
        }
    }
}

// ─── Pure handler logic (network-free, unit-testable) ───────────────────────

/// Load the gallery index from a store root. An absent index file is treated
/// as an empty gallery (the generating loop may simply not have saved yet).
pub fn load_gallery(root: &Path) -> std::io::Result<GalleryIndex> {
    let storage = GalleryStorage::new(root);
    match storage.load_index() {
        Ok(index) => Ok(index),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(GalleryIndex::default()),
        Err(e) => Err(e),
    }
}

/// Why an SVG lookup failed.
#[derive(Debug)]
pub enum SvgError {
    /// No entry with that id in the index.
    NotFound,
    /// Entry exists but has no visual SVG component.
    NotVisual,
    /// Stored filename contains a path separator (refuse traversal).
    BadFilename,
    /// Underlying file read failed.
    Io(std::io::Error),
}

/// Resolve the SVG content for an entry id. Visual and Synesthetic
/// modalities have SVG files; Score entries may carry an inline SVG.
pub fn resolve_visual_svg(root: &Path, index: &GalleryIndex, id: Uuid) -> Result<String, SvgError> {
    let entry = index
        .entries
        .iter()
        .find(|e| e.id == id)
        .ok_or(SvgError::NotFound)?;
    let filename = match &entry.modality {
        ArtModality::Visual { filename } => filename,
        ArtModality::Synesthetic {
            visual_filename, ..
        } => visual_filename,
        ArtModality::Score {
            score_svg: Some(svg),
            ..
        } => return Ok(svg.clone()),
        _ => return Err(SvgError::NotVisual),
    };
    if filename.contains('/') || filename.contains('\\') || filename.contains("..") {
        return Err(SvgError::BadFilename);
    }
    let path = GalleryStorage::new(root).visual_dir().join(filename);
    std::fs::read_to_string(path).map_err(SvgError::Io)
}

/// Outcome of applying a human rating.
#[derive(Debug, Clone, Serialize)]
pub struct RatingOutcome {
    /// The rating actually applied (after clamping to [-1, 1]).
    pub rating_applied: f32,
    /// Recalibrated aesthetic expectation (EMA) after the rating.
    pub ema: f32,
    /// Total evaluations ever recorded in the persisted memory.
    pub total_evaluations: u64,
    /// Dopamine delta the rating produced.
    pub dopamine_delta: f32,
}

/// Apply a human rating to the persisted aesthetic memory at `memory_path`
/// and save it back. Mirrors `Symthaea::rate_art` exactly: load →
/// `AestheticTracker::from_memory` → `human_feedback_unattributed` →
/// `to_memory(...).save`. Rating is clamped to [-1, 1].
pub fn apply_rating(memory_path: &Path, rating: f32) -> RatingOutcome {
    let rating = rating.clamp(-1.0, 1.0);
    let memory = symthaea_aesthetic::AestheticMemory::load(memory_path);
    let mut tracker = symthaea_aesthetic::AestheticTracker::from_memory(
        symthaea_aesthetic::AestheticConfig::default(),
        &memory,
    );
    let feedback = tracker.human_feedback_unattributed(rating);
    tracker.to_memory(&memory).save(memory_path);
    RatingOutcome {
        rating_applied: rating,
        ema: tracker.expectation(),
        total_evaluations: tracker.evaluation_count(),
        dopamine_delta: feedback.dopamine_delta,
    }
}

/// Minimal HTML escaping for text interpolated into the page.
fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// Inline artwork markup for one entry, or `None` if the modality has
/// nothing renderable here (music/dance).
fn entry_artwork_html(root: &Path, index: &GalleryIndex, entry: &GalleryEntry) -> Option<String> {
    match &entry.modality {
        ArtModality::Visual { .. }
        | ArtModality::Synesthetic { .. }
        | ArtModality::Score { .. } => match resolve_visual_svg(root, index, entry.id) {
            Ok(svg) => Some(format!("<div class=\"art\">{svg}</div>")),
            Err(_) => {
                Some("<div class=\"art missing\">(SVG file missing from store)</div>".to_string())
            }
        },
        ArtModality::Poetry { text } => Some(format!(
            "<pre class=\"art poetry\">{}</pre>",
            escape_html(text)
        )),
        _ => None,
    }
}

/// Render the whole gallery as a self-contained HTML page (server-rendered,
/// no external assets; rating buttons POST to `/api/rate` via inline JS).
pub fn render_gallery_html(index: &GalleryIndex, root: &Path) -> String {
    let mut cards = String::new();
    // Newest first.
    let mut entries: Vec<&GalleryEntry> = index.entries.iter().collect();
    entries.sort_by(|a, b| b.created_at.cmp(&a.created_at));

    for entry in entries {
        let Some(art) = entry_artwork_html(root, index, entry) else {
            continue;
        };
        let tags = entry
            .tags
            .iter()
            .map(|t| format!("<span class=\"tag\">{}</span>", escape_html(t)))
            .collect::<Vec<_>>()
            .join(" ");
        cards.push_str(&format!(
            r#"<section class="card" id="{id}">
{art}
<div class="meta">
  <span class="modality">{modality}</span>
  <span class="score">composite {score:.3}</span>
  <span class="cycle">cycle {cycle}</span>
  <span class="date">{date}</span>
  {protected}
  <div class="tags">{tags}</div>
</div>
<div class="rate">
  <button onclick="rate('{id}', -1.0)">&#128078; -1</button>
  <button onclick="rate('{id}', -0.5)">-0.5</button>
  <button onclick="rate('{id}', 0.5)">+0.5</button>
  <button onclick="rate('{id}', 1.0)">&#128077; +1</button>
  <span class="rate-result" id="result-{id}"></span>
</div>
</section>
"#,
            id = entry.id,
            art = art,
            modality = escape_html(entry.modality.name()),
            score = entry.aesthetic_score.composite,
            cycle = entry.created_at_cycle,
            date = entry.created_at.format("%Y-%m-%d %H:%M UTC"),
            protected = if entry.protected {
                "<span class=\"protected\">protected</span>"
            } else {
                ""
            },
            tags = tags,
        ));
    }

    if cards.is_empty() {
        cards = "<p class=\"empty\">The gallery is empty — no artworks have been stored yet.</p>"
            .to_string();
    }

    format!(
        r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Symthaea Gallery</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; background: #101014; color: #e8e8ee; }}
  header {{ padding: 1rem 2rem; border-bottom: 1px solid #2a2a33; }}
  header h1 {{ margin: 0; font-size: 1.3rem; font-weight: 600; }}
  header .stats {{ color: #9a9aa8; font-size: 0.85rem; }}
  main {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 1.2rem; padding: 1.5rem 2rem; }}
  .card {{ background: #17171d; border: 1px solid #2a2a33; border-radius: 10px; padding: 1rem; }}
  .art {{ background: #fff; border-radius: 6px; overflow: hidden; }}
  .art svg {{ display: block; width: 100%; height: auto; }}
  .art.missing {{ background: none; color: #77778a; padding: 2rem; text-align: center; }}
  .art.poetry {{ background: none; color: #e8e8ee; white-space: pre-wrap; font-family: Georgia, serif; padding: 0.5rem; }}
  .meta {{ font-size: 0.8rem; color: #9a9aa8; margin-top: 0.6rem; display: flex; flex-wrap: wrap; gap: 0.5rem; }}
  .protected {{ color: #d9b44a; }}
  .tag {{ background: #22222c; border-radius: 4px; padding: 0.1rem 0.4rem; }}
  .rate {{ margin-top: 0.6rem; display: flex; gap: 0.4rem; align-items: center; }}
  .rate button {{ background: #22222c; color: #e8e8ee; border: 1px solid #33333f; border-radius: 6px; padding: 0.3rem 0.7rem; cursor: pointer; }}
  .rate button:hover {{ background: #2d2d3a; }}
  .rate-result {{ font-size: 0.8rem; color: #8fbf8f; }}
  .empty {{ padding: 2rem; color: #9a9aa8; }}
</style>
</head>
<body>
<header>
  <h1>Symthaea Gallery</h1>
  <div class="stats">{count} artworks &middot; average composite {avg:.3}</div>
</header>
<main>
{cards}
</main>
<script>
async function rate(id, rating) {{
  const el = document.getElementById('result-' + id);
  try {{
    const res = await fetch('/api/rate', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ rating: rating, entry_id: id }})
    }});
    const out = await res.json();
    el.textContent = 'recorded ' + out.rating_applied.toFixed(1) + ' → ema ' + out.ema.toFixed(3);
  }} catch (e) {{
    el.textContent = 'rating failed';
  }}
}}
</script>
</body>
</html>
"#,
        count = index.len(),
        avg = index.average_score(),
        cards = cards,
    )
}

// ─── Axum layer ──────────────────────────────────────────────────────────────

/// Request body for `POST /api/rate`.
#[derive(Debug, Deserialize)]
pub struct RateRequest {
    /// Human rating in [-1, 1] (clamped server-side).
    pub rating: f32,
    /// Which artwork was on screen. Currently informational only — the
    /// persisted feedback is unattributed (see module docs).
    #[serde(default)]
    pub entry_id: Option<Uuid>,
}

/// Response body for `POST /api/rate`.
#[derive(Debug, Serialize)]
pub struct RateResponse {
    pub rating_applied: f32,
    pub ema: f32,
    pub total_evaluations: u64,
    pub dopamine_delta: f32,
    pub entry_id: Option<Uuid>,
}

fn io_error_response(e: std::io::Error) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        format!("gallery store error: {e}"),
    )
        .into_response()
}

async fn page_handler(State(config): State<Arc<GalleryServerConfig>>) -> Response {
    match load_gallery(&config.gallery_root) {
        Ok(index) => Html(render_gallery_html(&index, &config.gallery_root)).into_response(),
        Err(e) => io_error_response(e),
    }
}

async fn entries_handler(State(config): State<Arc<GalleryServerConfig>>) -> Response {
    match load_gallery(&config.gallery_root) {
        Ok(index) => Json(index).into_response(),
        Err(e) => io_error_response(e),
    }
}

async fn svg_handler(
    State(config): State<Arc<GalleryServerConfig>>,
    UrlPath(id): UrlPath<Uuid>,
) -> Response {
    let index = match load_gallery(&config.gallery_root) {
        Ok(index) => index,
        Err(e) => return io_error_response(e),
    };
    match resolve_visual_svg(&config.gallery_root, &index, id) {
        Ok(svg) => ([(header::CONTENT_TYPE, "image/svg+xml")], svg).into_response(),
        Err(SvgError::NotFound) => (StatusCode::NOT_FOUND, "no such entry").into_response(),
        Err(SvgError::NotVisual) => {
            (StatusCode::NOT_FOUND, "entry has no SVG component").into_response()
        }
        Err(SvgError::BadFilename) => {
            (StatusCode::BAD_REQUEST, "entry filename rejected").into_response()
        }
        Err(SvgError::Io(e)) => io_error_response(e),
    }
}

async fn rate_handler(
    State(config): State<Arc<GalleryServerConfig>>,
    Json(req): Json<RateRequest>,
) -> Json<RateResponse> {
    let outcome = apply_rating(&config.aesthetic_memory_path, req.rating);
    Json(RateResponse {
        rating_applied: outcome.rating_applied,
        ema: outcome.ema,
        total_evaluations: outcome.total_evaluations,
        dopamine_delta: outcome.dopamine_delta,
        entry_id: req.entry_id,
    })
}

/// Build the gallery router. Routes:
/// - `GET /` — server-rendered HTML gallery
/// - `GET /api/entries` — JSON gallery index
/// - `GET /api/entry/{id}/svg` — raw SVG for one entry
/// - `POST /api/rate` — apply a human rating to persisted aesthetic memory
pub fn router(config: GalleryServerConfig) -> Router {
    Router::new()
        .route("/", get(page_handler))
        .route("/api/entries", get(entries_handler))
        .route("/api/entry/{id}/svg", get(svg_handler))
        .route("/api/rate", post(rate_handler))
        .with_state(Arc::new(config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_entry;
    use symthaea_aesthetic::AestheticScore;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "symthaea-gallery-server-{tag}-{}-{}",
            std::process::id(),
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn test_score() -> AestheticScore {
        let mut s = AestheticScore::uniform(0.6);
        s.compute_composite();
        s
    }

    fn seeded_store(tag: &str) -> (PathBuf, GalleryIndex, Uuid) {
        let root = temp_dir(tag);
        let storage = GalleryStorage::new(&root);
        storage.ensure_dirs().unwrap();
        storage
            .save_visual("art_001.svg", "<svg><circle r=\"5\"/></svg>")
            .unwrap();
        let mut index = GalleryIndex::new(100);
        index.add(create_entry(
            ArtModality::Visual {
                filename: "art_001.svg".into(),
            },
            test_score(),
            [0.9, 0.1, 0.2, 0.8, 0.1, 0.5, 0.7, 0.0],
            42,
        ));
        let id = index.entries[0].id;
        storage.save_index(&index).unwrap();
        (root, index, id)
    }

    #[test]
    fn load_gallery_missing_index_is_empty() {
        let root = temp_dir("missing-index");
        let index = load_gallery(&root).unwrap();
        assert!(index.is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn load_gallery_round_trip() {
        let (root, _, id) = seeded_store("round-trip");
        let loaded = load_gallery(&root).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.entries[0].id, id);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_svg_by_id() {
        let (root, index, id) = seeded_store("resolve");
        let svg = resolve_visual_svg(&root, &index, id).unwrap();
        assert!(svg.contains("<circle"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_svg_unknown_id() {
        let (root, index, _) = seeded_store("unknown-id");
        assert!(matches!(
            resolve_visual_svg(&root, &index, Uuid::new_v4()),
            Err(SvgError::NotFound)
        ));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_svg_rejects_traversal_filename() {
        let (root, mut index, _) = seeded_store("traversal");
        index.add(create_entry(
            ArtModality::Visual {
                filename: "../escape.svg".into(),
            },
            test_score(),
            [0.5; 8],
            1,
        ));
        let evil_id = index.entries[1].id;
        assert!(matches!(
            resolve_visual_svg(&root, &index, evil_id),
            Err(SvgError::BadFilename)
        ));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_svg_non_visual_entry() {
        let (root, mut index, _) = seeded_store("non-visual");
        index.add(create_entry(
            ArtModality::Poetry {
                text: "a poem".into(),
            },
            test_score(),
            [0.5; 8],
            1,
        ));
        let poem_id = index.entries[1].id;
        assert!(matches!(
            resolve_visual_svg(&root, &index, poem_id),
            Err(SvgError::NotVisual)
        ));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn apply_rating_clamps_and_persists() {
        let dir = temp_dir("rating");
        let memory_path = dir.join("aesthetic_memory.json");

        // Way out of range — must clamp to 1.0.
        let outcome = apply_rating(&memory_path, 5.0);
        assert_eq!(outcome.rating_applied, 1.0);
        assert!(outcome.ema > 0.5, "positive rating should raise the EMA");
        assert_eq!(outcome.total_evaluations, 1);

        // Persisted: a second rating starts from the saved state.
        let saved = symthaea_aesthetic::AestheticMemory::load(&memory_path);
        assert_eq!(saved.total_evaluations, 1);
        assert!(saved.ema > 0.5);

        let outcome2 = apply_rating(&memory_path, -3.0);
        assert_eq!(outcome2.rating_applied, -1.0);
        assert_eq!(outcome2.total_evaluations, 2);
        assert!(
            outcome2.ema < outcome.ema,
            "negative rating should lower the EMA"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn apply_rating_leaves_harmony_bias_untouched() {
        let dir = temp_dir("rating-bias");
        let memory_path = dir.join("aesthetic_memory.json");

        let mut memory = symthaea_aesthetic::AestheticMemory::new();
        memory.harmony_bias = [0.3; 8];
        memory.save(&memory_path);

        apply_rating(&memory_path, 0.9);
        let saved = symthaea_aesthetic::AestheticMemory::load(&memory_path);
        assert_eq!(
            saved.harmony_bias, [0.3; 8],
            "unattributed feedback must not touch harmony bias"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn render_html_contains_svg_and_rating_controls() {
        let (root, index, id) = seeded_store("render");
        let html = render_gallery_html(&index, &root);
        assert!(html.contains("<circle"), "SVG should be inlined");
        assert!(html.contains(&id.to_string()), "entry id should appear");
        assert!(html.contains("/api/rate"), "rating JS should be present");
        assert!(html.contains("Symthaea Gallery"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn render_html_empty_gallery() {
        let root = temp_dir("render-empty");
        let index = GalleryIndex::new(10);
        let html = render_gallery_html(&index, &root);
        assert!(html.contains("gallery is empty"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn render_html_escapes_poetry() {
        let root = temp_dir("render-escape");
        let mut index = GalleryIndex::new(10);
        index.add(create_entry(
            ArtModality::Poetry {
                text: "<script>alert(1)</script>".into(),
            },
            test_score(),
            [0.5; 8],
            1,
        ));
        let html = render_gallery_html(&index, &root);
        assert!(!html.contains("<script>alert"));
        assert!(html.contains("&lt;script&gt;"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn router_builds() {
        let _router = router(GalleryServerConfig {
            gallery_root: PathBuf::from("/tmp/nonexistent-gallery"),
            aesthetic_memory_path: PathBuf::from("/tmp/nonexistent-memory.json"),
        });
    }
}
