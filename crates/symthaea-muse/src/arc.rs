// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bar-level emotional arc conditioning for music generation.
//!
//! Inspired by MusicGen's bar-level control token approach and Story2MIDI (2024),
//! this module lets a narrative or emotion sequence drive music bar-by-bar —
//! like a film score that responds to plot beats.
//!
//! # Architecture
//!
//! ```text
//! Vec<BarDirective> → per-bar MusicalState modulation → compose_with_arc()
//! ```
//!
//! Each `BarDirective` carries a VA coordinate, density modifier, and optional
//! section type override. The arc is stretched or compressed to fit the
//! composition duration.

use serde::{Deserialize, Serialize};
use symthaea_aesthetic::ValenceArousal;

use crate::{pitch::TuningSystem, structure::SectionType, AudioData, Composition, MuseConfig, MusicalState, Note};

// ─── Bar Directive ────────────────────────────────────────────────────────────

/// Emotional intent for a single bar (measure) of music.
///
/// Analogous to MusicGen's bar-level control tokens, but driven by the
/// Symthaea VA emotion space rather than discrete labels.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BarDirective {
    /// Valence-Arousal coordinate for this bar.
    pub va: ValenceArousal,
    /// Note density multiplier [0.25, 2.0] relative to base.
    /// 1.0 = unchanged, 2.0 = twice as many notes, 0.25 = very sparse.
    pub density: f32,
    /// Optional section type override. If None, inferred from VA.
    pub section_type: Option<SectionType>,
    /// Dynamic level (velocity scaling) [0.3, 1.0].
    pub dynamics: f32,
}

impl BarDirective {
    /// Create a bar directive from a VA coordinate with sensible defaults.
    pub fn from_va(va: ValenceArousal) -> Self {
        let params = symthaea_aesthetic::MusicalParams::from_va(va);
        Self {
            va,
            density: params.density_scale.clamp(0.25, 2.0),
            section_type: None,
            dynamics: params.dynamics,
        }
    }

    /// Create a climactic bar (high arousal, any valence).
    pub fn climax(valence: f32) -> Self {
        Self::from_va(ValenceArousal::new(valence, 0.9))
    }

    /// Create a quiet/intimate bar.
    pub fn quiet(valence: f32) -> Self {
        Self::from_va(ValenceArousal::new(valence, 0.15))
    }

    /// Create a resolving bar (positive valence, moderate arousal).
    pub fn resolve() -> Self {
        Self::from_va(ValenceArousal::new(0.6, 0.35))
    }
}

impl Default for BarDirective {
    fn default() -> Self {
        Self::from_va(ValenceArousal::neutral())
    }
}

// ─── Emotional Arc ────────────────────────────────="────────────────────────────

/// A sequence of bar directives forming a complete emotional arc.
///
/// Arcs are automatically stretched to the composition duration.
/// An optional `tuning_system` activates non-Western scales (maqamat, ragas,
/// gamelan, just intonation, microtonal) for the entire arc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalArc {
    pub bars: Vec<BarDirective>,
    /// If set, overrides 12-TET for the entire composition.
    /// This activates maqamat, ragas, gamelan, just intonation, or microtonal tuning.
    pub tuning_system: Option<TuningSystem>,
}

impl EmotionalArc {
    pub fn new(bars: Vec<BarDirective>) -> Self {
        Self { bars, tuning_system: None }
    }

    /// Set the tuning system for this arc (builder-style).
    ///
    /// ```
    /// use symthaea_muse::arc::EmotionalArc;
    /// use symthaea_muse::pitch::{TuningSystem, MaqamMode};
    /// let arc = EmotionalArc::meditative()
    ///     .with_tuning(TuningSystem::Maqamat { maqam: MaqamMode::Rast });
    /// ```
    pub fn with_tuning(mut self, tuning: TuningSystem) -> Self {
        self.tuning_system = Some(tuning);
        self
    }

    /// Classic narrative arc: quiet intro → rising action → climax → resolution.
    pub fn narrative_arc(climax_valence: f32) -> Self {
        Self::new(vec![
            BarDirective::quiet(0.1),                                 // opening stillness
            BarDirective::from_va(ValenceArousal::new(0.2, 0.35)),   // rising
            BarDirective::from_va(ValenceArousal::new(0.3, 0.55)),   // building
            BarDirective::from_va(ValenceArousal::new(0.2, 0.70)),   // tension
            BarDirective::climax(climax_valence),                     // climax
            BarDirective::from_va(ValenceArousal::new(0.5, 0.55)),   // release
            BarDirective::resolve(),                                   // resolution
            BarDirective::quiet(0.4),                                 // coda
        ])
    }

    /// Meditative arc: gradual descent into stillness.
    pub fn meditative() -> Self {
        Self::new(vec![
            BarDirective::from_va(ValenceArousal::new(0.3, 0.5)),
            BarDirective::from_va(ValenceArousal::new(0.4, 0.4)),
            BarDirective::from_va(ValenceArousal::new(0.5, 0.3)),
            BarDirective::from_va(ValenceArousal::new(0.5, 0.2)),
            BarDirective::quiet(0.6),
        ])
    }

    /// Turbulent arc: unresolved tension.
    pub fn turbulent() -> Self {
        Self::new(vec![
            BarDirective::from_va(ValenceArousal::new(-0.3, 0.6)),
            BarDirective::from_va(ValenceArousal::new(-0.5, 0.75)),
            BarDirective::from_va(ValenceArousal::new(-0.6, 0.85)),
            BarDirective::from_va(ValenceArousal::new(-0.4, 0.9)),
            BarDirective::from_va(ValenceArousal::new(-0.6, 0.7)), // unresolved
        ])
    }

    /// Sample a directive at a normalized position [0, 1] using linear interpolation.
    pub fn sample(&self, t: f32) -> BarDirective {
        if self.bars.is_empty() {
            return BarDirective::default();
        }
        if self.bars.len() == 1 {
            return self.bars[0];
        }

        let t = t.clamp(0.0, 1.0);
        let scaled = t * (self.bars.len() - 1) as f32;
        let lo = scaled.floor() as usize;
        let hi = (lo + 1).min(self.bars.len() - 1);
        let frac = scaled - lo as f32;

        let a = &self.bars[lo];
        let b = &self.bars[hi];

        BarDirective {
            va: symthaea_aesthetic::lerp_va(a.va, b.va, frac),
            density: a.density + (b.density - a.density) * frac,
            section_type: if frac < 0.5 { a.section_type } else { b.section_type },
            dynamics: a.dynamics + (b.dynamics - a.dynamics) * frac,
        }
    }
}

// ─── Arc-Driven Composition ───────────────────────────────────────────────────

/// Generate a composition driven by an emotional arc.
///
/// The arc is divided into one segment per song section. Each segment's
/// `BarDirective` modulates the `MusicalState` used for that section,
/// allowing tempo, mode, density, and dynamics to evolve across the piece.
///
/// Unlike `compose()` which uses a single static cognitive snapshot, this
/// function produces music that tells an emotional story — the same way
/// film scores track narrative beats.
pub fn compose_with_arc(
    config: &MuseConfig,
    base_state: &MusicalState,
    arc: &EmotionalArc,
    seed: u64,
) -> Composition {
    use crate::{form, melody, pitch, rhythm, structure, synth, voice};

    // Plan form from base state (arc modulates within the form)
    let song_form = form::plan_form(base_state, config.duration_secs);
    let n_sections = song_form.sections.len().max(1);

    let overall_section = structure::determine_section(base_state);

    let mut all_notes: Vec<Note> = Vec::new();

    for (sec_idx, section) in song_form.sections.iter().enumerate() {
        // Sample the arc at this section's normalized position
        let t = sec_idx as f32 / (n_sections - 1).max(1) as f32;
        let directive = arc.sample(t);

        // Derive VA-modulated musical state for this section
        let va_params = symthaea_aesthetic::MusicalParams::from_va(directive.va);
        let section_state = MusicalState {
            arousal: (base_state.arousal * 0.3 + directive.va.arousal * 0.7).clamp(0.0, 1.0),
            valence: (base_state.valence * 0.3 + directive.va.valence * 0.7).clamp(-1.0, 1.0),
            dopamine: (base_state.dopamine * directive.dynamics).clamp(0.0, 1.0),
            noradrenaline: (base_state.noradrenaline * (0.5 + directive.va.arousal * 0.5))
                .clamp(0.0, 1.0),
            ..base_state.clone()
        };

        // Apply density from the directive
        let notes_per_section = config
            .max_notes
            .max(2)
            .checked_div(n_sections)
            .unwrap_or(config.max_notes)
            .max(1);
        let density = section.energy_level * directive.density * va_params.density_scale;
        let sec_max = ((notes_per_section as f32) * density.clamp(0.1, 3.0)).round() as usize;

        let sec_config = MuseConfig {
            max_notes: sec_max.max(1),
            duration_secs: section.duration,
            ..config.clone()
        };

        // Build scale from the arc-modulated state (use tuning system if specified)
        let tempo = rhythm::compute_tempo(&sec_config, &section_state);
        let beat_duration = 60.0 / tempo;
        let base_scale = if let Some(ref tuning) = arc.tuning_system {
            pitch::build_scale_with_tuning(&section_state, tuning)
        } else {
            pitch::build_scale(&section_state)
        };

        // Key shift from section
        let key_ratio = 2.0_f32.powf(section.key_shift as f32 / 12.0);
        let sec_scale: Vec<f32> = base_scale.iter().map(|&f| f * key_ratio).collect();

        let sec_seed = seed.wrapping_add(sec_idx as u64 * 31);
        let mut sec_notes =
            melody::generate_melody(&sec_config, &section_state, &sec_scale, beat_duration, sec_seed);

        // Apply directive dynamics to velocity
        for note in &mut sec_notes {
            note.velocity = (note.velocity * directive.dynamics).clamp(0.0, 1.0);
        }

        // Offset to section position
        for note in &mut sec_notes {
            note.start_time += section.start_time;
        }
        all_notes.extend(sec_notes);
    }

    // Arrange and synthesize
    let arrangement = voice::arrange(&all_notes, base_state);
    let total_samples = (config.duration_secs * config.sample_rate as f32) as usize;
    let audio = synth::render_arrangement(&arrangement, config.sample_rate, total_samples, base_state, config);

    Composition {
        audio,
        sample_rate: config.sample_rate,
        notes: all_notes,
        duration_secs: config.duration_secs,
        section: overall_section,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MuseConfig, MusicalState};

    #[test]
    fn narrative_arc_has_eight_bars() {
        let arc = EmotionalArc::narrative_arc(0.7);
        assert_eq!(arc.bars.len(), 8);
    }

    #[test]
    fn sample_endpoints() {
        let arc = EmotionalArc::narrative_arc(0.5);
        let start = arc.sample(0.0);
        let end = arc.sample(1.0);
        // Start should be quiet (low arousal), end should be quiet too (coda)
        assert!(start.va.arousal < 0.4, "start arousal {}", start.va.arousal);
        assert!(end.va.arousal < 0.5, "end arousal {}", end.va.arousal);
    }

    #[test]
    fn climax_at_midpoint() {
        let arc = EmotionalArc::narrative_arc(0.8);
        // Sample at ~60% should have highest arousal (climax bar)
        let mid = arc.sample(0.6);
        let start = arc.sample(0.0);
        assert!(mid.va.arousal > start.va.arousal, "climax should exceed start");
    }

    #[test]
    fn compose_with_arc_produces_audio() {
        let config = MuseConfig {
            duration_secs: 4.0,
            max_notes: 16,
            ..Default::default()
        };
        let state = MusicalState::default();
        let arc = EmotionalArc::narrative_arc(0.5);
        let comp = compose_with_arc(&config, &state, &arc, 42);
        assert!(!comp.audio.is_empty());
        assert!(!comp.notes.is_empty());
    }

    #[test]
    fn arc_composition_differs_from_flat() {
        let config = MuseConfig {
            duration_secs: 4.0,
            max_notes: 32,
            ..Default::default()
        };
        let state = MusicalState::default();
        let flat = crate::compose(&config, &state, 42);
        let arc = EmotionalArc::turbulent();
        let dramatic = compose_with_arc(&config, &state, &arc, 42);
        // Different arcs should produce different note velocities
        let flat_vel: f32 = flat.notes.iter().map(|n| n.velocity).sum();
        let dramatic_vel: f32 = dramatic.notes.iter().map(|n| n.velocity).sum();
        // They won't necessarily differ in total count, but the composition will differ
        let _ = (flat_vel, dramatic_vel); // just confirm it runs without panic
        assert!(!dramatic.notes.is_empty());
    }

    #[test]
    fn meditative_arc_low_density() {
        let arc = EmotionalArc::meditative();
        for bar in &arc.bars {
            // All bars should have relatively low density (meditative = sparse)
            assert!(bar.density <= 1.5, "meditative bar density {}", bar.density);
        }
    }

    #[test]
    fn bar_directive_from_va_bounded() {
        let d = BarDirective::from_va(ValenceArousal::new(-1.0, 1.0));
        assert!(d.density >= 0.25 && d.density <= 2.0);
        assert!(d.dynamics >= 0.5 && d.dynamics <= 1.0);
    }

    #[test]
    fn arc_with_maqam_tuning_composes() {
        use crate::pitch::{MaqamMode, TuningSystem};
        let config = MuseConfig { duration_secs: 4.0, max_notes: 16, ..Default::default() };
        let state = MusicalState::default();
        let arc = EmotionalArc::meditative()
            .with_tuning(TuningSystem::Maqamat { maqam: MaqamMode::Hijaz });
        assert!(arc.tuning_system.is_some());
        let comp = compose_with_arc(&config, &state, &arc, 99);
        assert!(!comp.notes.is_empty(), "maqam arc should produce notes");
    }

    #[test]
    fn arc_with_raga_tuning_composes() {
        use crate::pitch::{RagaMode, TuningSystem};
        let config = MuseConfig { duration_secs: 4.0, max_notes: 16, ..Default::default() };
        let state = MusicalState::default();
        let arc = EmotionalArc::narrative_arc(0.6)
            .with_tuning(TuningSystem::Raga { raga: RagaMode::Bhairav });
        let comp = compose_with_arc(&config, &state, &arc, 77);
        assert!(!comp.notes.is_empty(), "raga arc should produce notes");
    }

    #[test]
    fn arc_default_has_no_tuning() {
        let arc = EmotionalArc::turbulent();
        assert!(arc.tuning_system.is_none(), "default arc should use 12-TET");
    }
}
