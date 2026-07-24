// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Research Mode's Orchestration view: "who's playing what" — one row per
//! voice with its resolved instrument, note count, and register (pitch
//! range), plus a small register bar comparing every voice's range across
//! the keyboard at a glance.
//!
//! Deliberately not a new backend endpoint: `GET /api/notes/{id}` already
//! returns every voice's resolved instrument and full note list (used by
//! `score_view::ScoreView`'s piano roll) — this view just aggregates and
//! renders the same fetched data a different way, grouped by voice instead
//! of by pitch-over-time.

use leptos::prelude::*;
use leptos::task::spawn_local;

use crate::api::{self, PerformedVoice};
use crate::score_view::voice_color;
use crate::state::MuseState;

/// Standard 88-key piano range (A0..C8), used only to place each voice's
/// register bar consistently — not a claim about instrument compass.
const KEYBOARD_LOW_MIDI: f64 = 21.0;
const KEYBOARD_HIGH_MIDI: f64 = 108.0;

fn midi_of(frequency: f64) -> f64 {
    69.0 + 12.0 * (frequency / 440.0).log2()
}

const NOTE_NAMES: [&str; 12] = [
    "C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B",
];

fn note_name(midi: f64) -> String {
    let rounded = midi.round() as i32;
    let octave = rounded / 12 - 1;
    let pitch_class = rounded.rem_euclid(12) as usize;
    format!("{}{octave}", NOTE_NAMES[pitch_class])
}

struct VoiceSummary {
    name: String,
    instrument: String,
    note_count: usize,
    low_midi: f64,
    high_midi: f64,
}

fn summarize(voice: &PerformedVoice) -> Option<VoiceSummary> {
    if voice.notes.is_empty() {
        return None;
    }
    let mut low = f64::MAX;
    let mut high = f64::MIN;
    for n in &voice.notes {
        let m = midi_of(n.frequency);
        low = low.min(m);
        high = high.max(m);
    }
    Some(VoiceSummary {
        name: voice.name.clone(),
        instrument: voice.instrument.clone(),
        note_count: voice.notes.len(),
        low_midi: low,
        high_midi: high,
    })
}

#[component]
pub fn OrchestrationView(muse: MuseState) -> impl IntoView {
    let voices = RwSignal::new(Vec::<PerformedVoice>::new());
    let load_status = RwSignal::new(String::new());

    // Refetch whenever the shared current piece changes — same source and
    // trigger as ScoreView's fetch, just a second independent request
    // rather than sharing ScoreView's local `voices` signal (the two views
    // don't otherwise share state, and duplicating one cheap GET is
    // simpler than threading data between sibling components).
    Effect::new(move |_| {
        let Some(c) = muse.current.get() else {
            voices.set(Vec::new());
            return;
        };
        let id = c.id;
        load_status.set("loading orchestration…".to_string());
        spawn_local(async move {
            match api::fetch_notes(api::DEFAULT_BACKEND, id).await {
                Ok(v) => {
                    voices.set(v);
                    load_status.set(String::new());
                }
                Err(e) => load_status.set(format!("orchestration unavailable: {e}")),
            }
        });
    });

    view! {
        <div class="orchestration-view">
            {move || {
                let summaries: Vec<VoiceSummary> =
                    voices.get().iter().filter_map(summarize).collect();
                if summaries.is_empty() {
                    view! { <p class="muted small">{move || load_status.get()}</p> }.into_any()
                } else {
                    view! {
                        <div class="orch-rows">
                            {summaries.into_iter().map(|v| {
                                let color = voice_color(&v.name);
                                let span = (KEYBOARD_HIGH_MIDI - KEYBOARD_LOW_MIDI).max(1.0);
                                let left_pct = ((v.low_midi - KEYBOARD_LOW_MIDI) / span * 100.0).clamp(0.0, 100.0);
                                let width_pct = ((v.high_midi - v.low_midi) / span * 100.0).clamp(1.0, 100.0 - left_pct);
                                let register = format!("{}\u{2013}{}", note_name(v.low_midi), note_name(v.high_midi));
                                view! {
                                    <div class="orch-voice-row">
                                        <div class="orch-voice-label">
                                            <b style=format!("color:{color}")>"■"</b>
                                            <span class="orch-voice-name">{v.name}</span>
                                            <span class="muted small">{v.instrument.clone()}</span>
                                        </div>
                                        <div class="orch-register-track">
                                            <div
                                                class="orch-register-fill"
                                                style=format!(
                                                    "left:{left_pct:.1}%;width:{width_pct:.1}%;background:{color};"
                                                )
                                            ></div>
                                        </div>
                                        <span class="muted small orch-voice-stats">
                                            {format!("{} notes \u{00b7} {register}", v.note_count)}
                                        </span>
                                    </div>
                                }
                            }).collect_view()}
                        </div>
                    }.into_any()
                }
            }}
        </div>
    }
}
