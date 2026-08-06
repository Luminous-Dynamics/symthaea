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
    EvidenceBasis, EvidenceStatus, HarmonySummary, ImportedWorkSummary, JourneyNextRequest,
    JourneyNextResponse, KeeperEntry, ListenCompositionBundle, MotifsSummary, MusicalTime,
    PerformedVoice, ResonanceCurve, SectionInfo, StyleFamily, TeachingCorpusSummary,
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
    choice: &JourneyNextResponse,
    renderer: Option<&str>,
) -> Result<Candidate, String> {
    let req = ComposeRequest {
        valence: choice.valence,
        arousal: choice.arousal,
        energy: choice.energy,
        tonic: choice.tonic,
        style: choice.style.clone(),
        bars: choice.bars,
        base_seed: choice.composition_seed,
        n_candidates: 1,
        prompt: String::new(),
        spec: None,
        // The Listen radio wants successive pieces to stop sharing one
        // fixed premise — Create-mode's authored composes leave this off
        // so an exact spec/style stays exactly what the user asked for.
        vary_premise: true,
        renderer: renderer.map(str::to_string),
        use_motif_foundry: false,
        composition_lesson_id: None,
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

/// Which claim the contributor is making about an imported score. Mirrors
/// `symthaea_muse_protocol::AuthorizationBasis` — kept as a plain string here
/// (not the protocol enum) since this is form-encoded multipart, not JSON.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportAuthorization {
    /// "I created this work" -- becomes `declared_authorship: true` server-side.
    OwnWork,
    /// "I'm authorized to import and privately analyze someone else's work" --
    /// the conservative default; never implies an authorship claim.
    AuthorizedImport,
}

impl ImportAuthorization {
    fn as_wire(self) -> &'static str {
        match self {
            ImportAuthorization::OwnWork => "own_work",
            ImportAuthorization::AuthorizedImport => "authorized_import",
        }
    }
}

/// Private-first symbolic import. The server persists an immutable permission
/// receipt and never projects this operation into Foundry/global learning.
pub async fn import_music(
    backend: &str,
    file: web_sys::File,
    title: &str,
    contributor: &str,
    authorization: ImportAuthorization,
) -> Result<ImportedWorkSummary, String> {
    let form = web_sys::FormData::new().map_err(|_| "could not create upload form")?;
    form.append_with_blob_and_filename("file", &file, &file.name())
        .map_err(|_| "could not attach score")?;
    form.append_with_str("title", title)
        .map_err(|_| "could not attach title")?;
    form.append_with_str("contributor", contributor)
        .map_err(|_| "could not attach contributor")?;
    form.append_with_str("authorization_basis", authorization.as_wire())
        .map_err(|_| "could not attach authorization")?;
    let url = format!("{}/api/music/import", backend.trim_end_matches('/'));
    let resp = Request::post(&url)
        .body(form)
        .map_err(|error| format!("could not prepare import: {error}"))?
        .send()
        .await
        .map_err(|error| format!("import failed: {error}"))?;
    if !resp.ok() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(if body.is_empty() {
            format!("backend returned HTTP {status}")
        } else {
            body
        });
    }
    resp.json()
        .await
        .map_err(|error| format!("could not read import receipt: {error}"))
}

pub fn imported_audio_url(backend: &str, work_id: &str) -> String {
    format!(
        "{}/api/music/import/{work_id}/audio",
        backend.trim_end_matches('/')
    )
}

pub async fn fetch_imported_works(backend: &str) -> Result<Vec<ImportedWorkSummary>, String> {
    let url = format!("{}/api/music/imports", backend.trim_end_matches('/'));
    let response = Request::get(&url)
        .send()
        .await
        .map_err(|error| format!("request failed: {error}"))?;
    if !response.ok() {
        return Err(format!("backend returned HTTP {}", response.status()));
    }
    response
        .json()
        .await
        .map_err(|error| format!("could not read imported works: {error}"))
}

pub async fn fetch_teaching_corpus(backend: &str) -> Result<TeachingCorpusSummary, String> {
    let url = format!("{}/api/teaching", backend.trim_end_matches('/'));
    let resp = Request::get(&url)
        .send()
        .await
        .map_err(|error| format!("request failed: {error}"))?;
    if !resp.ok() {
        let body = resp.text().await.unwrap_or_default();
        return Err(if body.is_empty() {
            "the etude corpus is not installed".to_owned()
        } else {
            body
        });
    }
    resp.json()
        .await
        .map_err(|error| format!("could not read teaching corpus: {error}"))
}

pub fn teaching_audio_url(backend: &str, lesson_id: &str) -> String {
    format!(
        "{}/api/teaching/{lesson_id}/audio",
        backend.trim_end_matches('/')
    )
}

pub async fn choose_journey_style(
    backend: &str,
    request: &JourneyNextRequest,
) -> Result<JourneyNextResponse, String> {
    let url = format!("{}/api/journey/next", backend.trim_end_matches('/'));
    let response = Request::post(&url)
        .header("content-type", "application/json")
        .json(request)
        .map_err(|error| format!("could not encode journey request: {error}"))?
        .send()
        .await
        .map_err(|error| format!("journey request failed: {error}"))?;
    if !response.ok() {
        return Err(format!(
            "journey service returned HTTP {}",
            response.status()
        ));
    }
    response
        .json()
        .await
        .map_err(|error| format!("could not read journey choice: {error}"))
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

/// `GET /api/harmony/{id}` — the harmonic *vocabulary* of the piece's key:
/// tonic, tonality, and the 7 diatonic triads with Roman numerals, chord
/// symbols, and Tonic/Predominant/Dominant function.
///
/// Deliberately the key's static vocabulary, not a chord-by-time analysis:
/// the engine's real per-piece `Progression` is computed mid-compose and
/// discarded before reaching `Candidate`, so "which chords did this piece
/// actually use, when" is not answerable from the server's stored state
/// today. See `harmony_summary()` in `muse_studio.rs`.
pub async fn fetch_harmony(backend: &str, id: u64) -> Result<HarmonySummary, String> {
    let url = format!("{}/api/harmony/{id}", backend.trim_end_matches('/'));
    let resp = Request::get(&url)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    if !resp.ok() {
        return Err(format!("backend returned HTTP {}", resp.status()));
    }
    resp.json::<HarmonySummary>()
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

// There is deliberately no `fetch_catalog` here. The Muse 152 taxonomy is a
// compile-time `const CATALOG: [CanonicalStyle; 152]` in the protocol crate,
// which this app already links — so a view that needs it imports
// `symthaea_muse_protocol::catalog` directly rather than fetching 152 entries
// the binary already contains. Same data by construction, no round-trip, and
// no chance of the two sides disagreeing. (Not re-exported here ahead of that
// first consumer, per this crate's own rule about not adding surface
// speculatively.)
//
// A fetch is also not merely unnecessary but impossible as written:
// `CanonicalStyle`'s fields are `&'static str`, so its derived `Deserialize`
// only satisfies `Deserialize<'de>` for a borrowed-from-'static input and
// cannot deserialize an owned response body ("implementation of `Deserialize`
// is not general enough"). Deserializing it would require an owned mirror
// type with `String` fields — worth adding only if a non-Rust client or a
// server-authoritative catalog ever needs one. `GET /api/catalog` still
// exists server-side for exactly those consumers.

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
