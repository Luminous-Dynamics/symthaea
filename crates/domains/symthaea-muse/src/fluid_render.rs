// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Optional FluidSynth render backend: send the PERFORMED MIDI through a
//! real soundfont engine instead of the in-crate synthesizer.
//!
//! An A/B listening test settled this architecture question decisively:
//! the same performed MIDI (swing, damage, hook, dynamics all baked in)
//! rendered through FluidSynth + FluidR3 "no longer fights the
//! composition... if I had heard these renders first I would not have
//! said the instruments sound harsh." The composition engine's real
//! product is the performed MIDI; rendering belongs to a dedicated
//! instrument engine when one is available. The in-crate renderer remains
//! the zero-dependency fallback.
//!
//! Availability is environmental, never assumed: `fluidsynth` must be on
//! `PATH` (or named by `SYMTHAEA_FLUIDSYNTH`) and a SoundFont must be
//! named by `SYMTHAEA_SOUNDFONT`. The Muse Studio launcher provides both
//! via `nix-shell`.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The consciousness-state color mapped into FluidSynth settings — so the
/// Studio's state sliders keep DOING something under the soundfont
/// backend. (User bug report the day the backend landed: "the music
/// slider doesn't work" — the sliders drove the native renderer's
/// timbre/reverb, which the soundfont path bypassed entirely.)
#[derive(Debug, Clone, Copy)]
pub struct RenderColor {
    /// synth.reverb.room-size [0,1]
    pub reverb_room: f32,
    /// synth.reverb.level [0,1]
    pub reverb_level: f32,
    /// synth.reverb.damp [0,1] — higher = darker tail
    pub reverb_damp: f32,
    /// synth.chorus.level [0,10] — 0 disables
    pub chorus_level: f32,
}

impl RenderColor {
    /// Map the state the way the Studio's slider labels promise:
    /// consciousness = "precision/space" (room + wet, tightened by
    /// noradrenaline's "urgency"), serotonin = "warmth" (darker reverb
    /// tail), dopamine = "brightness/shimmer" (a little chorus).
    pub fn from_state(state: &crate::MusicalState) -> Self {
        let space = state.consciousness_level.clamp(0.0, 1.0);
        let tight = state.noradrenaline.clamp(0.0, 1.0);
        RenderColor {
            reverb_room: 0.25 + 0.55 * space,
            reverb_level: (0.45 + 0.45 * space) * (1.0 - 0.35 * tight),
            reverb_damp: 0.2 + 0.6 * state.serotonin.clamp(0.0, 1.0),
            chorus_level: 2.5 * state.dopamine.clamp(0.0, 1.0).powi(2),
        }
    }
}

/// Style-conditioned room: modulate the slider-driven base [`RenderColor`]
/// by the piece's [`crate::theory_realize::PerformanceDialect`]. The
/// sliders stay authoritative (the base is computed from state exactly as
/// before); the dialect only SHIFTS the room the way the tradition's
/// recording practice would:
/// - ProcessExact (minimalism/ambient/drone, sacred-like stillness) — a
///   larger, wetter space: the sustained material IS the reverb's food.
/// - DanceLocked / JazzLaidBack — drier and closer: a rhythm section is
///   close-mic'd; a wash would smear the groove that is the clock.
/// - ClassicalRubato / FolkLift — unchanged (returned bit-identical).
#[cfg(feature = "theory")]
pub fn room_for_dialect(
    base: RenderColor,
    dialect: crate::theory_realize::PerformanceDialect,
) -> RenderColor {
    use crate::theory_realize::PerformanceDialect as D;
    match dialect {
        D::ClassicalRubato | D::FolkLift => base,
        D::ProcessExact => RenderColor {
            reverb_room: (base.reverb_room + 0.18).clamp(0.0, 1.0),
            reverb_level: (base.reverb_level + 0.12).clamp(0.0, 1.0),
            ..base
        },
        D::DanceLocked | D::JazzLaidBack => RenderColor {
            reverb_room: (base.reverb_room - 0.12).max(0.0),
            reverb_level: (base.reverb_level * 0.75).clamp(0.0, 1.0),
            ..base
        },
    }
}

/// The fluidsynth binary + soundfont, if the environment provides them.
pub fn available() -> Option<(PathBuf, PathBuf)> {
    let soundfont = std::env::var_os("SYMTHAEA_SOUNDFONT")
        .map(PathBuf::from)
        .filter(|p| p.is_file())?;
    let binary = std::env::var_os("SYMTHAEA_FLUIDSYNTH")
        .map(PathBuf::from)
        .filter(|p| p.is_file())
        .or_else(|| {
            std::env::split_paths(&std::env::var_os("PATH")?)
                .map(|d| d.join("fluidsynth"))
                .find(|p| p.is_file())
        })?;
    Some((binary, soundfont))
}

/// Render a Standard MIDI File to MASTERED 16-bit stereo WAV bytes via
/// FluidSynth. Returns `None` (caller falls back to the native renderer) when
/// the environment lacks fluidsynth/soundfont, the render fails, or the result
/// is silent.
///
/// FluidSynth's default gain (0.2) leaves our deliberately-soft velocities
/// whisper-quiet, so the render runs at gain 0.8 and the PCM then goes through
/// [`master_wav`] — the same BS.1770-4 chain and the same
/// [`crate::auto_master::MasteringConfig`] the native path uses, so the two
/// renderers are post-processed identically.
pub fn render_midi_to_wav(
    midi_path: &Path,
    sample_rate: u32,
    color: Option<RenderColor>,
) -> Option<Vec<u8>> {
    let (binary, soundfont) = available()?;
    let out = std::env::temp_dir().join(format!(
        "muse_fluid_{}_{}.wav",
        std::process::id(),
        midi_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("render")
    ));
    let mut cmd = Command::new(&binary);
    if let Some(c) = color {
        cmd.arg("-o")
            .arg(format!(
                "synth.reverb.room-size={:.3}",
                c.reverb_room.clamp(0.0, 1.0)
            ))
            .arg("-o")
            .arg(format!(
                "synth.reverb.level={:.3}",
                c.reverb_level.clamp(0.0, 1.0)
            ))
            .arg("-o")
            .arg(format!(
                "synth.reverb.damp={:.3}",
                c.reverb_damp.clamp(0.0, 1.0)
            ));
        if c.chorus_level > 0.05 {
            cmd.arg("-o").arg(format!(
                "synth.chorus.level={:.3}",
                c.chorus_level.clamp(0.0, 10.0)
            ));
        } else {
            cmd.arg("-o").arg("synth.chorus.active=0");
        }
    }
    let status = cmd
        .arg("-g")
        .arg("0.8")
        .arg("-T")
        .arg("wav")
        .arg("-F")
        .arg(&out)
        .arg("-r")
        .arg(sample_rate.to_string())
        .arg("-ni")
        .arg(&soundfont)
        .arg(midi_path)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .ok()?;
    if !status.success() {
        let _ = std::fs::remove_file(&out);
        return None;
    }
    let result = master_wav(&out);
    let _ = std::fs::remove_file(&out);
    result
}

/// Read a WAV, run the full mastering chain, re-encode as 16-bit in memory.
///
/// This used to be `normalize_wav`: a single peak scan and one gain multiply to
/// −1.5 dBFS. That made the PREFERRED renderer the one with *strictly weaker*
/// post-processing. The native path runs [`crate::auto_master::auto_master`] —
/// real ITU-R BS.1770-4 K-weighting with both gates, corrective 3-band EQ,
/// section leveling and a brick-wall limiter — at `theory_realize.rs:1643`,
/// while FluidSynth output got none of it. Every one of `auto_master`'s
/// listening-tuned defaults was inert on the renderer an A/B test had chosen.
///
/// Both paths now run the same chain with the same `MasteringConfig::default()`,
/// so a FluidSynth-vs-native comparison measures the renderers rather than the
/// mastering.
///
/// CORRECTION (2026-07-30). An earlier version of this comment claimed the
/// original A/B listening test was confounded by the native arm being
/// unmastered. **That was wrong, and the truth runs the other way.** Checked
/// against history: `auto_master` was wired into `realize_core` on 2026-07-08
/// (`4c613f9d49`), and this file — with its A/B claim — first appeared on
/// 2026-07-10 (`c0fda87d0b`), with the `auto_master` call verified present in
/// `theory_realize.rs` at that commit. So the native arm WAS mastered and the
/// FluidSynth arm was peak-normalized only: FluidSynth won *despite* the weaker
/// post-processing, which strengthens the preference rather than undermining it.
///
/// The error came from misreading `theory_realize.rs`'s own comment ("never
/// called from ANY generation path before this") as describing the state at the
/// time of the listening test, when it describes the state before 2026-07-08 —
/// two days earlier.
///
/// What remains genuinely untested is narrower: now that both arms share one
/// mastering chain, the comparison is FAIR for the first time (it was previously
/// unfair AGAINST FluidSynth), so re-running it measures renderer timbre alone.
fn master_wav(path: &Path) -> Option<Vec<u8>> {
    let mut reader = hound::WavReader::open(path).ok()?;
    let spec_in = reader.spec();
    let samples: Vec<f32> = match spec_in.sample_format {
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec_in.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .filter_map(Result::ok)
                .map(|s| s as f32 * scale)
                .collect()
        }
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(Result::ok).collect(),
    };
    let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    if peak < 1e-5 {
        return None; // silence — treat as a failed render
    }

    // `auto_master` works on stereo frames. FluidSynth renders stereo, but
    // handle mono defensively by duplicating the channel rather than failing.
    let channels = spec_in.channels.max(1) as usize;
    let mut frames: Vec<[f32; 2]> = match channels {
        1 => samples.iter().map(|&s| [s, s]).collect(),
        _ => samples
            .chunks_exact(channels)
            .map(|c| [c[0], c[1 % channels]])
            .collect(),
    };
    crate::auto_master::auto_master(
        &mut frames,
        spec_in.sample_rate,
        &crate::auto_master::MasteringConfig::default(),
    );

    // Re-encode as stereo regardless of input channel count: `frames` is always
    // stereo after the conversion above, so writing `spec_in.channels` would
    // mislabel a mono input's now-stereo data and halve its playback rate.
    let spec_out = hound::WavSpec {
        channels: 2,
        sample_rate: spec_in.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec_out).ok()?;
        for f in &frames {
            for s in f {
                let v = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
                writer.write_sample(v).ok()?;
            }
        }
        writer.finalize().ok()?;
    }
    Some(cursor.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "theory")]
    #[test]
    fn room_follows_the_dialect() {
        use crate::theory_realize::PerformanceDialect as D;
        let base = RenderColor::from_state(&crate::MusicalState::default());
        // Identity dialects: the base passes through bit-identically —
        // the sliders' room is untouched for classical/folk playing.
        for d in [D::ClassicalRubato, D::FolkLift] {
            let r = room_for_dialect(base, d);
            assert_eq!(r.reverb_room, base.reverb_room);
            assert_eq!(r.reverb_level, base.reverb_level);
            assert_eq!(r.reverb_damp, base.reverb_damp);
            assert_eq!(r.chorus_level, base.chorus_level);
        }
        // Process music: a larger, wetter space.
        let process = room_for_dialect(base, D::ProcessExact);
        assert!(process.reverb_room > base.reverb_room);
        assert!(process.reverb_level > base.reverb_level);
        // Groove dialects: drier and closer.
        for d in [D::DanceLocked, D::JazzLaidBack] {
            let dry = room_for_dialect(base, d);
            assert!(dry.reverb_room < base.reverb_room);
            assert!(dry.reverb_level < base.reverb_level);
        }
        // The dialect never touches the warmth/shimmer dimensions.
        assert_eq!(process.reverb_damp, base.reverb_damp);
        assert_eq!(process.chorus_level, base.chorus_level);
        // And everything stays in FluidSynth's legal ranges even at the
        // slider extremes.
        for space in [0.0f32, 1.0] {
            let state = crate::MusicalState {
                consciousness_level: space,
                ..Default::default()
            };
            let b = RenderColor::from_state(&state);
            for d in [D::ProcessExact, D::DanceLocked, D::JazzLaidBack] {
                let r = room_for_dialect(b, d);
                assert!((0.0..=1.0).contains(&r.reverb_room));
                assert!((0.0..=1.0).contains(&r.reverb_level));
            }
        }
    }

    #[test]
    fn unavailable_without_environment() {
        // In a bare test environment (no SYMTHAEA_SOUNDFONT) this must be
        // a clean None, never a panic — the studio falls back to native.
        if std::env::var_os("SYMTHAEA_SOUNDFONT").is_none() {
            assert!(available().is_none());
        }
    }

    #[test]
    fn masters_real_audio_to_the_loudness_target_and_rejects_silence() {
        let dir = std::env::temp_dir();
        let quiet = dir.join(format!("muse_norm_test_{}.wav", std::process::id()));
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        // A very quiet sine must come back LOUDNESS-normalized to the
        // mastering target, not merely peak-scaled. This assertion is the
        // regression: the previous implementation peak-normalized to exactly
        // 0.84 (−1.5 dBFS) and this test asserted that constant, which is
        // precisely the weaker post-processing the FluidSynth path used to get
        // while the native path ran the full BS.1770-4 chain. Asserting LUFS
        // instead of peak is what makes the two paths' behaviour comparable.
        let mut w = hound::WavWriter::create(&quiet, spec).unwrap();
        for i in 0..44100 {
            let s = (i as f32 * 0.06).sin() * 0.01;
            w.write_sample((s * 32767.0) as i16).unwrap();
        }
        w.finalize().unwrap();
        let bytes = master_wav(&quiet).expect("real audio masters");
        let mut r = hound::WavReader::new(std::io::Cursor::new(bytes)).unwrap();
        let out_spec = r.spec();
        // `master_wav` always emits stereo: it upmixes mono before mastering,
        // so writing the input channel count would mislabel the data.
        assert_eq!(out_spec.channels, 2, "master_wav must emit stereo");
        let frames: Vec<[f32; 2]> = r
            .samples::<i16>()
            .filter_map(Result::ok)
            .map(|s| s as f32 / 32767.0)
            .collect::<Vec<_>>()
            .chunks_exact(2)
            .map(|c| [c[0], c[1]])
            .collect();

        let target = crate::auto_master::MasteringConfig::default().target_lufs;
        let out = crate::auto_master::measure_lufs(&frames, out_spec.sample_rate);
        assert!(
            (out.integrated - target).abs() < 2.0,
            "mastered output should land near {target} LUFS, got {} — a peak-only \
             normalizer would not hit a loudness target at all",
            out.integrated
        );

        let peak = frames.iter().flatten().fold(0.0f32, |m, s| m.max(s.abs()));
        let ceiling =
            10f32.powf(crate::auto_master::MasteringConfig::default().limiter_ceiling_db / 20.0);
        assert!(
            peak <= ceiling + 0.02,
            "peak {peak} must stay under the limiter ceiling {ceiling}"
        );
        std::fs::remove_file(&quiet).ok();

        // Silence is a failed render, not a normalizable one.
        let silent = dir.join(format!("muse_norm_silent_{}.wav", std::process::id()));
        let mut w = hound::WavWriter::create(&silent, spec).unwrap();
        for _ in 0..4410 {
            w.write_sample(0i16).unwrap();
        }
        w.finalize().unwrap();
        assert!(master_wav(&silent).is_none());
        std::fs::remove_file(&silent).ok();
    }
}
