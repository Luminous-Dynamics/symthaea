// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Client for the `muse_studio` axum API (`symthaea-muse/src/bin/muse_studio.rs`,
//! default `http://127.0.0.1:8400`).
//!
//! Wire types live in `symthaea-muse-protocol` (a small serde-only crate,
//! no audio/ONNX/native deps) and are shared with the server — this
//! module is just the HTTP calls, re-exporting the protocol types
//! callers need.

use gloo_net::http::Request;

use symthaea_muse_protocol::ComposeResponse;
// IdentityCard isn't named directly anywhere yet (only reached via
// Candidate.card) — re-export it once a view actually needs to name it.
pub use symthaea_muse_protocol::{
    AtlasCompareResponse, AtlasPoint, AtlasSummary, BundleEnvelope, Candidate, ComposeRequest,
    EvidenceBasis, EvidenceStatus, KeeperEntry, ListenCompositionBundle, MotifsSummary,
    MusicalTime, PerformedVoice, ResonanceCurve, SectionInfo, StyleFamily,
};

/// Default origin for the `muse_studio` backend this app talks to.
pub const DEFAULT_BACKEND: &str = "http://127.0.0.1:8400";

/// `POST /api/compose`, returning every requested candidate. Used
/// directly by Create Mode's form; `compose_listen_piece` is a thin
/// wrapper for the Listen radio's one-candidate case.
pub async fn compose(backend: &str, req: &ComposeRequest) -> Result<Vec<Candidate>, String> {
    let url = format!("{}/api/compose", backend.trim_end_matches('/'));
    let resp = Request::post(&url)
        .header("content-type", "application/json")
        .json(req)
        .map_err(|e| format!("failed to encode request: {e}"))?
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    if !resp.ok() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(if body.is_empty() {
            format!("backend returned HTTP {status}")
        } else {
            body
        });
    }
    let parsed = resp
        .json::<ComposeResponse>()
        .await
        .map_err(|e| format!("failed to parse response: {e}"))?;
    Ok(parsed.candidates)
}

/// `POST /api/compose` for exactly one Listen-tab candidate, in the given
/// style. Mirrors `composeListenPiece()` in `studio/index.html`: the
/// mood/energy/bars/valence/arousal constants there are deliberately fixed
/// (Listen is meant to feel like tuning a radio into an already-playing
/// station, not a compose form) — only style, seed, and renderer vary.
/// `renderer` is the user's persistent renderer preference
/// (`MuseState::renderer_preference`): `Some("native")`/`Some("fluidsynth")`
/// to force one, or `None` for the server's own default.
pub async fn compose_listen_piece(
    backend: &str,
    style: &str,
    renderer: Option<&str>,
) -> Result<Candidate, String> {
    let seed = (js_sys::Math::random() * 900_000.0) as u64 + 1;
    let tonic = (js_sys::Math::random() * 12.0) as i32;
    // Randomized within Create Mode's own slider bounds (valence -1..1,
    // arousal/energy 0..1) rather than fixed at one neutral triple
    // (0.15/0.45/0.5) for every single piece — real, separate from
    // `vary_premise` (which varies tempo/texture/phrase/ensemble/mode):
    // this is the emotional-intent axis, still the same style. Fixing
    // one intent forever was a real, avoidable share of "the songs sound
    // the same" independent of everything the premise layer covers.
    let valence = (js_sys::Math::random() * 2.0 - 1.0) as f32;
    let arousal = js_sys::Math::random() as f32;
    let energy = js_sys::Math::random() as f32;
    let req = ComposeRequest {
        valence,
        arousal,
        energy,
        tonic,
        style: style.to_string(),
        bars: 4,
        base_seed: seed,
        n_candidates: 1,
        prompt: String::new(),
        spec: None,
        // The Listen radio wants successive pieces to stop sharing one
        // fixed premise — Create-mode's authored composes leave this off
        // so an exact spec/style stays exactly what the user asked for.
        vary_premise: true,
        renderer: renderer.map(str::to_string),
    };
    compose(backend, &req)
        .await?
        .pop()
        .ok_or_else(|| "compose returned no candidates".to_string())
}

/// `POST /api/keeper/{id}` — heart/keep a piece.
pub async fn keep_piece(backend: &str, id: u64) -> Result<(), String> {
    let url = format!("{}/api/keeper/{id}", backend.trim_end_matches('/'));
    let resp = Request::post(&url)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    if !resp.ok() {
        return Err(format!("backend returned HTTP {}", resp.status()));
    }
    Ok(())
}

pub fn audio_url(backend: &str, id: u64) -> String {
    format!("{}/api/audio/{id}", backend.trim_end_matches('/'))
}

pub fn midi_url(backend: &str, id: u64) -> String {
    format!("{}/api/midi/{id}", backend.trim_end_matches('/'))
}

/// `GET /api/notes/{id}` — the performed voices/notes for Research Mode's
/// Score (piano-roll) view.
pub async fn fetch_notes(backend: &str, id: u64) -> Result<Vec<PerformedVoice>, String> {
    let url = format!("{}/api/notes/{id}", backend.trim_end_matches('/'));
    let resp = Request::get(&url)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    if !resp.ok() {
        return Err(format!("backend returned HTTP {}", resp.status()));
    }
    resp.json::<Vec<PerformedVoice>>()
        .await
        .map_err(|e| format!("failed to parse response: {e}"))
}

/// `GET /api/keepers` — every kept piece, most-recent-first. The Liked
/// page's data source.
pub async fn fetch_keepers(backend: &str) -> Result<Vec<KeeperEntry>, String> {
    let url = format!("{}/api/keepers", backend.trim_end_matches('/'));
    let resp = Request::get(&url)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    if !resp.ok() {
        return Err(format!("backend returned HTTP {}", resp.status()));
    }
    resp.json::<Vec<KeeperEntry>>()
        .await
        .map_err(|e| format!("failed to parse response: {e}"))
}

pub fn keeper_audio_url(backend: &str, audio_key: &str) -> String {
    format!(
        "{}/api/keeper-audio/{audio_key}",
        backend.trim_end_matches('/')
    )
}

pub fn keeper_midi_url(backend: &str, audio_key: &str) -> String {
    format!(
        "{}/api/keeper-midi/{audio_key}",
        backend.trim_end_matches('/')
    )
}

pub fn keeper_recipe_url(backend: &str, audio_key: &str) -> String {
    format!(
        "{}/api/keeper-recipe/{audio_key}",
        backend.trim_end_matches('/')
    )
}

/// `GET /api/atlas?lens=...` — every in-session candidate plus persisted
/// keeper, fingerprinted and projected to 2D via the given lens (a
/// reweighting of the same structural fingerprint — `None`/unrecognized
/// resolves to `"combined"`, the original Phase 1 behavior). Powers the
/// Atlas view.
pub async fn fetch_atlas(backend: &str, lens: Option<&str>) -> Result<AtlasSummary, String> {
    let url = match lens {
        Some(l) => format!("{}/api/atlas?lens={l}", backend.trim_end_matches('/')),
        None => format!("{}/api/atlas", backend.trim_end_matches('/')),
    };
    let resp = Request::get(&url)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    if !resp.ok() {
        return Err(format!("backend returned HTTP {}", resp.status()));
    }
    resp.json::<AtlasSummary>()
        .await
        .map_err(|e| format!("failed to parse response: {e}"))
}

/// `GET /api/atlas/compare?a=...&b=...` — a real per-layer structural
/// distance breakdown between two specific Atlas points, powering the "why
/// nearby" evidence panel.
pub async fn fetch_atlas_compare(
    backend: &str,
    a: &str,
    b: &str,
) -> Result<AtlasCompareResponse, String> {
    let url = format!(
        "{}/api/atlas/compare?a={}&b={}",
        backend.trim_end_matches('/'),
        urlencoding_encode(a),
        urlencoding_encode(b),
    );
    let resp = Request::get(&url)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    if !resp.ok() {
        return Err(format!("backend returned HTTP {}", resp.status()));
    }
    resp.json::<AtlasCompareResponse>()
        .await
        .map_err(|e| format!("failed to parse response: {e}"))
}

/// Minimal percent-encoding for Atlas point ids (`"candidate:3"`,
/// `"keeper:abc-123"`) as URL query values — avoids pulling in a whole
/// `urlencoding` crate dependency for encoding just `:`.
fn urlencoding_encode(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
            other => format!("%{:02X}", other as u32),
        })
        .collect()
}

/// `GET /api/piece/{id}/listen-bundle` — the composition-side evidence
/// bundle (sections, phrases, motifs, cadences, sonorities, orchestration,
/// resonance) behind Research's evidence views. Every non-observed field
/// carries its own `EvidenceBasis`, and layers the backend doesn't yet
/// produce arrive as empty vectors, not fabricated ones.
pub async fn fetch_listen_bundle(
    backend: &str,
    id: u64,
) -> Result<BundleEnvelope<ListenCompositionBundle>, String> {
    let url = format!(
        "{}/api/piece/{id}/listen-bundle",
        backend.trim_end_matches('/')
    );
    let resp = Request::get(&url)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    if !resp.ok() {
        return Err(format!("backend returned HTTP {}", resp.status()));
    }
    resp.json::<BundleEnvelope<ListenCompositionBundle>>()
        .await
        .map_err(|e| format!("failed to parse response: {e}"))
}

/// `GET /api/motifs/{id}` — the candidate's discrete section/motif-return
/// structure, when the engine's `Form` pipeline actually applies to it
/// (`has_structure: false` with empty `sections` otherwise — an honest
/// "this piece's form doesn't have discrete motif-return structure", not
/// an error). Powers the Listen page's live current-section indicator.
pub async fn fetch_motifs(backend: &str, id: u64) -> Result<MotifsSummary, String> {
    let url = format!("{}/api/motifs/{id}", backend.trim_end_matches('/'));
    let resp = Request::get(&url)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    if !resp.ok() {
        return Err(format!("backend returned HTTP {}", resp.status()));
    }
    resp.json::<MotifsSummary>()
        .await
        .map_err(|e| format!("failed to parse response: {e}"))
}

/// `GET /api/styles` — every style Muse can compose in, each with its
/// real `Style::grammar_family()`. Fetched once and cached by
/// `MuseState::load_style_families`, not refetched per-compose.
pub async fn fetch_style_families(backend: &str) -> Result<Vec<StyleFamily>, String> {
    let url = format!("{}/api/styles", backend.trim_end_matches('/'));
    let resp = Request::get(&url)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    if !resp.ok() {
        return Err(format!("backend returned HTTP {}", resp.status()));
    }
    resp.json::<Vec<StyleFamily>>()
        .await
        .map_err(|e| format!("failed to parse response: {e}"))
}

/// `GET /api/spec/{style}` — the style's preset composition spec as raw
/// JSON text, for the Create Mode spec editor's "load preset" action.
pub async fn spec_preset(backend: &str, style: &str) -> Result<String, String> {
    let url = format!("{}/api/spec/{style}", backend.trim_end_matches('/'));
    let resp = Request::get(&url)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    if !resp.ok() {
        return Err(format!("backend returned HTTP {}", resp.status()));
    }
    resp.text()
        .await
        .map_err(|e| format!("failed to read response: {e}"))
}

/// `GET /api/specs` — every name the user has saved a spec under.
pub async fn list_specs(backend: &str) -> Result<Vec<String>, String> {
    let url = format!("{}/api/specs", backend.trim_end_matches('/'));
    let resp = Request::get(&url)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    if !resp.ok() {
        return Err(format!("backend returned HTTP {}", resp.status()));
    }
    resp.json::<Vec<String>>()
        .await
        .map_err(|e| format!("failed to parse response: {e}"))
}

/// `GET /api/specs/{name}` — a previously-saved spec's raw JSON text.
pub async fn load_named_spec(backend: &str, name: &str) -> Result<String, String> {
    let url = format!("{}/api/specs/{name}", backend.trim_end_matches('/'));
    let resp = Request::get(&url)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    if !resp.ok() {
        return Err(format!("backend returned HTTP {}", resp.status()));
    }
    resp.text()
        .await
        .map_err(|e| format!("failed to read response: {e}"))
}

/// `POST /api/specs` — save the given raw spec JSON text under `name`.
pub async fn save_spec(backend: &str, name: &str, spec_json: &str) -> Result<(), String> {
    let spec: serde_json::Value =
        serde_json::from_str(spec_json).map_err(|e| format!("spec is not valid JSON: {e}"))?;
    let url = format!("{}/api/specs", backend.trim_end_matches('/'));
    let body = serde_json::json!({ "name": name, "spec": spec });
    let resp = Request::post(&url)
        .header("content-type", "application/json")
        .json(&body)
        .map_err(|e| format!("failed to encode request: {e}"))?
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    if !resp.ok() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(if text.is_empty() {
            format!("backend returned HTTP {status}")
        } else {
            text
        });
    }
    Ok(())
}
