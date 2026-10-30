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

use sha2::{Digest, Sha256};
use std::ffi::{OsStr, OsString};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

const PROCESS_DIAGNOSTIC_PREVIEW_BYTES: usize = 4096;
static FLUID_OUTPUT_NONCE: AtomicU64 = AtomicU64::new(0);

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

/// Captured local observation of one FluidSynth child-process invocation.
///
/// The complete stdout/stderr streams are committed by digest; only bounded
/// previews are retained in receipts. Executable and SoundFont hashes are
/// measured on both sides of the invocation. Matching pre/post hashes detect
/// ordinary replacement races, but do not attest which inode or pages the
/// kernel actually loaded and do not defend against a privileged/malicious host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FluidSynthRenderObservation {
    pub executable_path: PathBuf,
    pub soundfont_path: PathBuf,
    pub executable_sha256_before: String,
    pub executable_sha256_after: String,
    pub soundfont_sha256_before: String,
    pub soundfont_sha256_after: String,
    /// Ordered argv excluding argv[0], which is executable_path.
    pub arguments: Vec<String>,
    /// Digest of the exact environment snapshot explicitly passed to the child.
    /// Values are not persisted because process environments can contain secrets.
    pub environment_sha256: String,
    pub stdout_sha256: String,
    pub stderr_sha256: String,
    pub stdout_preview: String,
    pub stderr_preview: String,
    pub stdout_truncated: bool,
    pub stderr_truncated: bool,
    /// None means the process could not be started or did not report an exit code.
    pub exit_code: Option<i32>,
    pub spawn_error: Option<String>,
    /// Present only after exit 0, stable executable/assets and successful mastering.
    pub mastered_wav_bytes: Option<Vec<u8>>,
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> Option<String> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).ok()?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Some(format!("{:x}", hasher.finalize()))
}

#[cfg(unix)]
fn os_str_bytes(value: &OsStr) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    value.as_bytes().to_vec()
}

#[cfg(windows)]
fn os_str_bytes(value: &OsStr) -> Vec<u8> {
    use std::os::windows::ffi::OsStrExt;
    value.encode_wide().flat_map(|unit| unit.to_le_bytes()).collect()
}

#[cfg(not(any(unix, windows)))]
fn os_str_bytes(value: &OsStr) -> Vec<u8> {
    value.to_string_lossy().as_bytes().to_vec()
}

fn snapshot_environment() -> Vec<(OsString, OsString)> {
    let mut environment: Vec<(OsString, OsString)> = std::env::vars_os().collect();
    environment.sort_by(|left, right| left.0.cmp(&right.0));
    environment
}

fn environment_sha256(environment: &[(OsString, OsString)]) -> String {
    let mut ordered = environment.to_vec();
    ordered.sort_by(|left, right| left.0.cmp(&right.0));
    let mut hasher = Sha256::new();
    hasher.update(b"muse-fluidsynth-child-environment-v1\0");
    for (key, value) in ordered {
        let key = os_str_bytes(&key);
        let value = os_str_bytes(&value);
        hasher.update((key.len() as u64).to_le_bytes());
        hasher.update(key);
        hasher.update((value.len() as u64).to_le_bytes());
        hasher.update(value);
    }
    format!("{:x}", hasher.finalize())
}

fn diagnostic_preview(bytes: &[u8]) -> (String, bool) {
    let visible = bytes.len().min(PROCESS_DIAGNOSTIC_PREVIEW_BYTES);
    (
        String::from_utf8_lossy(&bytes[..visible]).into_owned(),
        bytes.len() > PROCESS_DIAGNOSTIC_PREVIEW_BYTES,
    )
}

/// Execute FluidSynth and return the process observation even when the child
/// exits unsuccessfully. No shell is involved: the executable and ordered
/// argument vector are passed directly to Command. The child receives an
/// explicit snapshot of the parent environment, and that snapshot is hashed.
/// A missing executable/SoundFont or an unrecordable non-UTF-8 argv returns
/// None; a spawn/exit/render failure is represented in the returned observation.
pub fn render_midi_to_wav_observed(
    midi_path: &Path,
    sample_rate: u32,
    color: Option<RenderColor>,
) -> Option<FluidSynthRenderObservation> {
    let (binary, soundfont) = available()?;
    let binary = binary.canonicalize().ok()?;
    let soundfont = soundfont.canonicalize().ok()?;
    let midi_path = midi_path.canonicalize().ok()?;

    let executable_sha256_before = sha256_file(&binary)?;
    let soundfont_sha256_before = sha256_file(&soundfont)?;
    let environment = snapshot_environment();
    let environment_sha256 = environment_sha256(&environment);

    let nonce = FLUID_OUTPUT_NONCE.fetch_add(1, Ordering::Relaxed);
    let out = std::env::temp_dir().join(format!(
        "muse_fluid_{}_{}.wav",
        std::process::id(),
        nonce
    ));

    let mut arguments = Vec::<OsString>::new();
    if let Some(c) = color {
        arguments.push(OsString::from("-o"));
        arguments.push(OsString::from(format!(
            "synth.reverb.room-size={:.3}",
            c.reverb_room.clamp(0.0, 1.0)
        )));
        arguments.push(OsString::from("-o"));
        arguments.push(OsString::from(format!(
            "synth.reverb.level={:.3}",
            c.reverb_level.clamp(0.0, 1.0)
        )));
        arguments.push(OsString::from("-o"));
        arguments.push(OsString::from(format!(
            "synth.reverb.damp={:.3}",
            c.reverb_damp.clamp(0.0, 1.0)
        )));
        if c.chorus_level > 0.05 {
            arguments.push(OsString::from("-o"));
            arguments.push(OsString::from(format!(
                "synth.chorus.level={:.3}",
                c.chorus_level.clamp(0.0, 10.0)
            )));
        } else {
            arguments.push(OsString::from("-o"));
            arguments.push(OsString::from("synth.chorus.active=0"));
        }
    }
    arguments.extend([
        OsString::from("-g"),
        OsString::from("0.8"),
        OsString::from("-T"),
        OsString::from("wav"),
        OsString::from("-F"),
        out.as_os_str().to_owned(),
        OsString::from("-r"),
        OsString::from(sample_rate.to_string()),
        OsString::from("-ni"),
        soundfont.as_os_str().to_owned(),
        midi_path.as_os_str().to_owned(),
    ]);
    // A receipt must be able to state the exact argument strings. Refuse
    // platform paths that cannot be represented losslessly as UTF-8.
    let argument_strings: Vec<String> = arguments
        .iter()
        .map(|argument| argument.to_str().map(str::to_owned))
        .collect::<Option<Vec<_>>>()?;

    let mut command = Command::new(&binary);
    command.args(&arguments);
    command.env_clear();
    for (key, value) in &environment {
        command.env(key, value);
    }
    let process_result = command.output();
    let (stdout, stderr, exit_code, spawn_error) = match process_result {
        Ok(output) => (output.stdout, output.stderr, output.status.code(), None),
        Err(error) => (Vec::new(), Vec::new(), None, Some(error.to_string())),
    };

    let executable_sha256_after = sha256_file(&binary).unwrap_or_default();
    let soundfont_sha256_after = sha256_file(&soundfont).unwrap_or_default();
    let (stdout_preview, stdout_truncated) = diagnostic_preview(&stdout);
    let (stderr_preview, stderr_truncated) = diagnostic_preview(&stderr);

    let artifacts_stable = !executable_sha256_after.is_empty()
        && !soundfont_sha256_after.is_empty()
        && executable_sha256_before == executable_sha256_after
        && soundfont_sha256_before == soundfont_sha256_after;
    let mastered_wav_bytes = if exit_code == Some(0) && artifacts_stable {
        master_wav(&out)
    } else {
        None
    };
    let _ = std::fs::remove_file(&out);

    Some(FluidSynthRenderObservation {
        executable_path: binary,
        soundfont_path: soundfont,
        executable_sha256_before,
        executable_sha256_after,
        soundfont_sha256_before,
        soundfont_sha256_after,
        arguments: argument_strings,
        environment_sha256,
        stdout_sha256: sha256_hex(&stdout),
        stderr_sha256: sha256_hex(&stderr),
        stdout_preview,
        stderr_preview,
        stdout_truncated,
        stderr_truncated,
        exit_code,
        spawn_error,
        mastered_wav_bytes,
    })
}

/// Backward-compatible convenience wrapper. Evidence-bearing callers should
/// use render_midi_to_wav_observed and retain its process observation.
pub fn render_midi_to_wav(
    midi_path: &Path,
    sample_rate: u32,
    color: Option<RenderColor>,
) -> Option<Vec<u8>> {
    render_midi_to_wav_observed(midi_path, sample_rate, color)
        .and_then(|observation| observation.mastered_wav_bytes)
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

    #[test]
    fn child_environment_commitment_is_order_independent_and_content_sensitive() {
        let first = vec![
            (OsString::from("A"), OsString::from("one")),
            (OsString::from("B"), OsString::from("two")),
        ];
        let reordered = vec![
            (OsString::from("B"), OsString::from("two")),
            (OsString::from("A"), OsString::from("one")),
        ];
        let changed = vec![
            (OsString::from("A"), OsString::from("changed")),
            (OsString::from("B"), OsString::from("two")),
        ];
        assert_eq!(environment_sha256(&first), environment_sha256(&reordered));
        assert_ne!(environment_sha256(&first), environment_sha256(&changed));
    }

    #[test]
    fn child_diagnostic_preview_is_bounded_and_marks_truncation() {
        let bytes = vec![b'x'; PROCESS_DIAGNOSTIC_PREVIEW_BYTES + 7];
        let (preview, truncated) = diagnostic_preview(&bytes);
        assert_eq!(preview.len(), PROCESS_DIAGNOSTIC_PREVIEW_BYTES);
        assert!(truncated);
        let (full, truncated) = diagnostic_preview(b"renderer ok");
        assert_eq!(full, "renderer ok");
        assert!(!truncated);
    }

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
