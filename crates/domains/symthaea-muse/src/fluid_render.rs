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

/// Render a Standard MIDI File to peak-normalized 16-bit stereo WAV bytes
/// via FluidSynth. Returns `None` (caller falls back to the native
/// renderer) when the environment lacks fluidsynth/soundfont, the render
/// fails, or the result is silent.
///
/// FluidSynth's default gain (0.2) leaves our deliberately-soft velocities
/// whisper-quiet, so the render runs at gain 0.8 and the PCM is then
/// normalized to −1.5 dBFS peak — the same ceiling the native master uses.
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
    let result = normalize_wav(&out);
    let _ = std::fs::remove_file(&out);
    result
}

/// Read a WAV, peak-normalize to −1.5 dBFS, re-encode as 16-bit in memory.
fn normalize_wav(path: &Path) -> Option<Vec<u8>> {
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
    let gain = 0.84 / peak; // −1.5 dBFS, matching the native master ceiling
    let spec_out = hound::WavSpec {
        channels: spec_in.channels,
        sample_rate: spec_in.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec_out).ok()?;
        for s in &samples {
            let v = (s * gain * 32767.0).clamp(-32768.0, 32767.0) as i16;
            writer.write_sample(v).ok()?;
        }
        writer.finalize().ok()?;
    }
    Some(cursor.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_without_environment() {
        // In a bare test environment (no SYMTHAEA_SOUNDFONT) this must be
        // a clean None, never a panic — the studio falls back to native.
        if std::env::var_os("SYMTHAEA_SOUNDFONT").is_none() {
            assert!(available().is_none());
        }
    }

    #[test]
    fn normalize_rejects_silence_and_scales_real_audio() {
        let dir = std::env::temp_dir();
        let quiet = dir.join(format!("muse_norm_test_{}.wav", std::process::id()));
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        // A very quiet sine must normalize up to the −1.5 dBFS ceiling.
        let mut w = hound::WavWriter::create(&quiet, spec).unwrap();
        for i in 0..4410 {
            let s = (i as f32 * 0.06).sin() * 0.01;
            w.write_sample((s * 32767.0) as i16).unwrap();
        }
        w.finalize().unwrap();
        let bytes = normalize_wav(&quiet).expect("real audio normalizes");
        let mut r = hound::WavReader::new(std::io::Cursor::new(bytes)).unwrap();
        let peak = r
            .samples::<i16>()
            .filter_map(Result::ok)
            .fold(0i32, |m, s| m.max((s as i32).abs()));
        assert!(
            (peak as f32 / 32767.0 - 0.84).abs() < 0.02,
            "peak {} should sit at the -1.5dBFS ceiling",
            peak
        );
        std::fs::remove_file(&quiet).ok();

        // Silence is a failed render, not a normalizable one.
        let silent = dir.join(format!("muse_norm_silent_{}.wav", std::process::id()));
        let mut w = hound::WavWriter::create(&silent, spec).unwrap();
        for _ in 0..4410 {
            w.write_sample(0i16).unwrap();
        }
        w.finalize().unwrap();
        assert!(normalize_wav(&silent).is_none());
        std::fs::remove_file(&silent).ok();
    }
}
