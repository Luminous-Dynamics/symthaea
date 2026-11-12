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

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::ffi::{OsStr, OsString};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

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

/// Version identifier for the local FluidSynth child-process observation.
pub const FLUIDSYNTH_INVOCATION_OBSERVATION_SCHEMA_V1: &str =
    "urn:luminous-dynamics:muse:fluidsynth-invocation-observation:v1";

/// Bound retained stdout/stderr preview bytes. Streams continue to be drained
/// after this limit so a verbose child cannot block on a full pipe.
const DIAGNOSTIC_PREVIEW_LIMIT: usize = 8 * 1024;

static FLUID_OUTPUT_NONCE: AtomicU64 = AtomicU64::new(0);

/// One OS argument as passed to the child process. raw_os_hex preserves the
/// platform-native OsStr encoding; display is only for human inspection.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InvocationArgumentV1 {
    pub display: String,
    pub raw_os_hex: String,
}

/// One allowlisted environment variable passed to the renderer. Values are
/// represented only by a digest so secrets cannot leak into the receipt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InvocationEnvironmentV1 {
    pub name: String,
    pub value_sha256: String,
}

/// Result state for a local renderer invocation. These values describe the
/// observable wrapper/child-process path, not a cryptographic attestation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FluidSynthInvocationDispositionV1 {
    NotAvailable,
    InputArtifactUnavailable,
    SpawnFailed,
    ProcessFailed,
    MidiChangedDuringRender,
    RendererAssetChangedDuringRender,
    OutputMissing,
    OutputUnmasterableOrSilent,
    Rendered,
}

/// Structured local observation of the FluidSynth invocation.
///
/// The before/after path digests narrow file-replacement races but do not prove
/// which bytes the operating system loaded into the child process. A malicious
/// host, kernel, dynamic loader or filesystem can still falsify this local
/// observation. Stronger execution attestation needs a separate trusted
/// execution boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluidSynthInvocationObservationV1 {
    pub schema: String,
    pub disposition: FluidSynthInvocationDispositionV1,
    pub program: Option<InvocationArgumentV1>,
    pub arguments: Vec<InvocationArgumentV1>,
    pub environment: Vec<InvocationEnvironmentV1>,
    pub executable_sha256_before: Option<String>,
    pub executable_sha256_after: Option<String>,
    pub soundfont_sha256_before: Option<String>,
    pub soundfont_sha256_after: Option<String>,
    pub midi_sha256_before: Option<String>,
    pub midi_sha256_after: Option<String>,
    pub output_wav_sha256_before_mastering: Option<String>,
    pub mastered_wav_sha256: Option<String>,
    pub process_spawned: bool,
    pub process_success: bool,
    pub process_exit_code: Option<i32>,
    pub spawn_error: Option<String>,
    pub stdout_sha256: Option<String>,
    pub stderr_sha256: Option<String>,
    pub stdout_total_bytes: u64,
    pub stderr_total_bytes: u64,
    pub stdout_preview: String,
    pub stderr_preview: String,
    pub stdout_preview_truncated: bool,
    pub stderr_preview_truncated: bool,
    pub sample_rate_hz: u32,
    pub started_unix_ns: u128,
    pub finished_unix_ns: u128,
    pub elapsed_micros: u128,
    pub trust_boundary: String,
}

/// Output of a render attempt. wav_bytes == None is an explicit failed or
/// unavailable attempt; the observation remains available to the caller.
#[derive(Debug, Clone)]
pub struct FluidSynthRenderAttemptV1 {
    pub wav_bytes: Option<Vec<u8>>,
    pub observation: FluidSynthInvocationObservationV1,
}

fn unix_time_ns() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

fn sha256_hex_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    hex_digest(&digest)
}

fn sha256_file(path: &Path) -> Option<String> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).ok()?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let digest = hasher.finalize();
    Some(hex_digest(&digest))
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

#[cfg(unix)]
fn raw_os_bytes(value: &OsStr) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    value.as_bytes().to_vec()
}

#[cfg(windows)]
fn raw_os_bytes(value: &OsStr) -> Vec<u8> {
    use std::os::windows::ffi::OsStrExt;
    value.encode_wide().flat_map(u16::to_le_bytes).collect()
}

#[cfg(not(any(unix, windows)))]
fn raw_os_bytes(value: &OsStr) -> Vec<u8> {
    value.to_string_lossy().as_bytes().to_vec()
}

fn invocation_argument(value: &OsStr) -> InvocationArgumentV1 {
    InvocationArgumentV1 {
        display: value.to_string_lossy().into_owned(),
        raw_os_hex: hex_digest(&raw_os_bytes(value)),
    }
}

#[derive(Debug)]
struct DrainedStream {
    sha256: String,
    preview: String,
    total_bytes: u64,
    truncated: bool,
}

fn drain_stream<R: Read>(mut reader: R) -> std::io::Result<DrainedStream> {
    let mut hasher = Sha256::new();
    let mut preview = Vec::with_capacity(DIAGNOSTIC_PREVIEW_LIMIT);
    let mut total_bytes = 0u64;
    let mut buffer = [0u8; 8192];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
        total_bytes = total_bytes.saturating_add(count as u64);
        let remaining = DIAGNOSTIC_PREVIEW_LIMIT.saturating_sub(preview.len());
        preview.extend_from_slice(&buffer[..count.min(remaining)]);
    }
    Ok(DrainedStream {
        sha256: hex_digest(hasher.finalize().as_slice()),
        preview: String::from_utf8_lossy(&preview).into_owned(),
        total_bytes,
        truncated: total_bytes > preview.len() as u64,
    })
}

fn empty_observation(
    disposition: FluidSynthInvocationDispositionV1,
    sample_rate: u32,
) -> FluidSynthInvocationObservationV1 {
    let now = unix_time_ns();
    FluidSynthInvocationObservationV1 {
        schema: FLUIDSYNTH_INVOCATION_OBSERVATION_SCHEMA_V1.to_string(),
        disposition,
        program: None,
        arguments: Vec::new(),
        environment: Vec::new(),
        executable_sha256_before: None,
        executable_sha256_after: None,
        soundfont_sha256_before: None,
        soundfont_sha256_after: None,
        midi_sha256_before: None,
        midi_sha256_after: None,
        output_wav_sha256_before_mastering: None,
        mastered_wav_sha256: None,
        process_spawned: false,
        process_success: false,
        process_exit_code: None,
        spawn_error: None,
        stdout_sha256: None,
        stderr_sha256: None,
        stdout_total_bytes: 0,
        stderr_total_bytes: 0,
        stdout_preview: String::new(),
        stderr_preview: String::new(),
        stdout_preview_truncated: false,
        stderr_preview_truncated: false,
        sample_rate_hz: sample_rate,
        started_unix_ns: now,
        finished_unix_ns: now,
        elapsed_micros: 0,
        trust_boundary:
            "local wrapper observation; not proof of child-loaded bytes or trusted-host execution"
                .to_string(),
    }
}

/// Render a Standard MIDI File through FluidSynth and retain both the audio
/// result and a structured invocation observation.
///
/// Unlike the compatibility wrapper, this API preserves failure evidence
/// (missing input, spawn failure, non-zero exit, changed input/assets, missing
/// output and silent/unmasterable output) instead of collapsing every failure
/// to None. Stdout/stderr are drained concurrently; only a bounded preview
/// is retained while full-stream SHA-256 and byte counts are recorded.
pub fn render_midi_to_wav_with_observation(
    midi_path: &Path,
    sample_rate: u32,
    color: Option<RenderColor>,
) -> FluidSynthRenderAttemptV1 {
    let Some((binary, soundfont)) = available() else {
        return FluidSynthRenderAttemptV1 {
            wav_bytes: None,
            observation: empty_observation(
                FluidSynthInvocationDispositionV1::NotAvailable,
                sample_rate,
            ),
        };
    };
    render_midi_to_wav_with_observation_using_paths(
        midi_path,
        sample_rate,
        color,
        &binary,
        &soundfont,
    )
}

/// Render with the exact executable and SoundFont selected by the caller.
/// This prevents a second environment lookup from silently changing the
/// actual command after the caller has frozen its render configuration.
pub fn render_midi_to_wav_with_observation_using_paths(
    midi_path: &Path,
    sample_rate: u32,
    color: Option<RenderColor>,
    binary: &Path,
    soundfont: &Path,
) -> FluidSynthRenderAttemptV1 {
    render_midi_to_wav_with_paths(midi_path, sample_rate, color, binary, soundfont)
}

/// Compatibility API retained for existing callers. New evidence-producing
/// paths should call render_midi_to_wav_with_observation and persist the
/// observation even when no audio bytes are returned.
pub fn render_midi_to_wav(
    midi_path: &Path,
    sample_rate: u32,
    color: Option<RenderColor>,
) -> Option<Vec<u8>> {
    render_midi_to_wav_with_observation(midi_path, sample_rate, color).wav_bytes
}

fn render_midi_to_wav_with_paths(
    midi_path: &Path,
    sample_rate: u32,
    color: Option<RenderColor>,
    binary: &Path,
    soundfont: &Path,
) -> FluidSynthRenderAttemptV1 {
    let started = Instant::now();
    let started_unix_ns = unix_time_ns();
    let nonce = FLUID_OUTPUT_NONCE.fetch_add(1, Ordering::Relaxed);
    let out = std::env::temp_dir().join(format!(
        "muse_fluid_{}_{}.wav",
        std::process::id(),
        nonce
    ));

    let executable_before = sha256_file(binary);
    let soundfont_before = sha256_file(soundfont);
    let midi_before = sha256_file(midi_path);
    if executable_before.is_none() || soundfont_before.is_none() || midi_before.is_none() {
        let mut observation =
            empty_observation(FluidSynthInvocationDispositionV1::InputArtifactUnavailable, sample_rate);
        observation.started_unix_ns = started_unix_ns;
        observation.finished_unix_ns = unix_time_ns();
        observation.elapsed_micros = started.elapsed().as_micros();
        observation.executable_sha256_before = executable_before;
        observation.soundfont_sha256_before = soundfont_before;
        observation.midi_sha256_before = midi_before;
        return FluidSynthRenderAttemptV1 { wav_bytes: None, observation };
    }

    let mut args = Vec::<OsString>::new();
    if let Some(c) = color {
        args.push("-o".into());
        args.push(format!("synth.reverb.room-size={:.3}", c.reverb_room.clamp(0.0, 1.0)).into());
        args.push("-o".into());
        args.push(format!("synth.reverb.level={:.3}", c.reverb_level.clamp(0.0, 1.0)).into());
        args.push("-o".into());
        args.push(format!("synth.reverb.damp={:.3}", c.reverb_damp.clamp(0.0, 1.0)).into());
        if c.chorus_level > 0.05 {
            args.push("-o".into());
            args.push(format!("synth.chorus.level={:.3}", c.chorus_level.clamp(0.0, 10.0)).into());
        } else {
            args.push("-o".into());
            args.push("synth.chorus.active=0".into());
        }
    }
    args.extend([
        "-g".into(),
        "0.8".into(),
        "-T".into(),
        "wav".into(),
        "-F".into(),
        out.as_os_str().to_os_string(),
        "-r".into(),
        sample_rate.to_string().into(),
        "-ni".into(),
        soundfont.as_os_str().to_os_string(),
        midi_path.as_os_str().to_os_string(),
    ]);

    // Avoid inheriting an unbounded process environment (which may contain
    // credentials or unrelated state). Pass only the renderer's explicit
    // runtime allowlist and bind each value by digest in the observation.
    const ENV_ALLOWLIST: [&str; 6] = [
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "XDG_CONFIG_HOME",
    ];
    let child_environment: Vec<(OsString, OsString)> = ENV_ALLOWLIST
        .iter()
        .filter_map(|name| {
            std::env::var_os(name).map(|value| (OsString::from(name), value))
        })
        .collect();
    let environment_observation: Vec<InvocationEnvironmentV1> = child_environment
        .iter()
        .map(|(name, value)| InvocationEnvironmentV1 {
            name: name.to_string_lossy().into_owned(),
            value_sha256: sha256_hex_bytes(&raw_os_bytes(value)),
        })
        .collect();

    let mut command = Command::new(binary);
    command.env_clear();
    command.envs(child_environment.iter().cloned());
    command.args(&args);
    command.stdout(Stdio::piped()).stderr(Stdio::piped());

    let program = invocation_argument(binary.as_os_str());
    let arguments = args.iter().map(|arg| invocation_argument(arg.as_os_str())).collect();
    let mut observation = empty_observation(
        FluidSynthInvocationDispositionV1::SpawnFailed,
        sample_rate,
    );
    observation.program = Some(program);
    observation.arguments = arguments;
    observation.environment = environment_observation;
    observation.executable_sha256_before = executable_before;
    observation.soundfont_sha256_before = soundfont_before;
    observation.midi_sha256_before = midi_before;
    observation.started_unix_ns = started_unix_ns;

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            observation.spawn_error = Some(error.to_string());
            observation.finished_unix_ns = unix_time_ns();
            observation.elapsed_micros = started.elapsed().as_micros();
            observation.executable_sha256_after = sha256_file(binary);
            observation.soundfont_sha256_after = sha256_file(soundfont);
            observation.midi_sha256_after = sha256_file(midi_path);
            return FluidSynthRenderAttemptV1 { wav_bytes: None, observation };
        }
    };
    observation.process_spawned = true;

    let stdout_thread = child.stdout.take().map(|stream| {
        std::thread::spawn(move || drain_stream(stream))
    });
    let stderr_thread = child.stderr.take().map(|stream| {
        std::thread::spawn(move || drain_stream(stream))
    });

    let status = child.wait();
    if let Some(joined) = stdout_thread {
        if let Ok(Ok(stream)) = joined.join() {
            observation.stdout_sha256 = Some(stream.sha256);
            observation.stdout_preview = stream.preview;
            observation.stdout_total_bytes = stream.total_bytes;
            observation.stdout_preview_truncated = stream.truncated;
        }
    }
    if let Some(joined) = stderr_thread {
        if let Ok(Ok(stream)) = joined.join() {
            observation.stderr_sha256 = Some(stream.sha256);
            observation.stderr_preview = stream.preview;
            observation.stderr_total_bytes = stream.total_bytes;
            observation.stderr_preview_truncated = stream.truncated;
        }
    }

    let status = match status {
        Ok(status) => status,
        Err(error) => {
            observation.spawn_error = Some(format!("wait failed: {error}"));
            observation.finished_unix_ns = unix_time_ns();
            observation.elapsed_micros = started.elapsed().as_micros();
            observation.executable_sha256_after = sha256_file(binary);
            observation.soundfont_sha256_after = sha256_file(soundfont);
            observation.midi_sha256_after = sha256_file(midi_path);
            let _ = std::fs::remove_file(&out);
            return FluidSynthRenderAttemptV1 { wav_bytes: None, observation };
        }
    };
    observation.process_exit_code = status.code();
    observation.process_success = status.success();

    observation.executable_sha256_after = sha256_file(binary);
    observation.soundfont_sha256_after = sha256_file(soundfont);
    observation.midi_sha256_after = sha256_file(midi_path);

    let binary_unchanged =
        observation.executable_sha256_before == observation.executable_sha256_after;
    let soundfont_unchanged =
        observation.soundfont_sha256_before == observation.soundfont_sha256_after;
    let midi_unchanged = observation.midi_sha256_before == observation.midi_sha256_after;

    if !status.success() {
        observation.disposition = FluidSynthInvocationDispositionV1::ProcessFailed;
        let _ = std::fs::remove_file(&out);
    } else if !midi_unchanged {
        observation.disposition = FluidSynthInvocationDispositionV1::MidiChangedDuringRender;
        let _ = std::fs::remove_file(&out);
    } else if !binary_unchanged || !soundfont_unchanged {
        observation.disposition =
            FluidSynthInvocationDispositionV1::RendererAssetChangedDuringRender;
        let _ = std::fs::remove_file(&out);
    } else if !out.is_file() {
        observation.disposition = FluidSynthInvocationDispositionV1::OutputMissing;
    } else {
        observation.output_wav_sha256_before_mastering = sha256_file(&out);
        match master_wav(&out) {
            Some(bytes) => {
                observation.mastered_wav_sha256 = Some(sha256_hex_bytes(&bytes));
                observation.disposition = FluidSynthInvocationDispositionV1::Rendered;
                observation.finished_unix_ns = unix_time_ns();
                observation.elapsed_micros = started.elapsed().as_micros();
                let _ = std::fs::remove_file(&out);
                return FluidSynthRenderAttemptV1 {
                    wav_bytes: Some(bytes),
                    observation,
                };
            }
            None => {
                observation.disposition =
                    FluidSynthInvocationDispositionV1::OutputUnmasterableOrSilent;
            }
        }
        let _ = std::fs::remove_file(&out);
    }

    observation.finished_unix_ns = unix_time_ns();
    observation.elapsed_micros = started.elapsed().as_micros();
    FluidSynthRenderAttemptV1 { wav_bytes: None, observation }
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

    #[cfg(unix)]
    mod invocation_observation_tests {
        use super::*;
        use std::os::unix::fs::PermissionsExt;

        fn unique_path(extension: &str) -> PathBuf {
            let nonce = FLUID_OUTPUT_NONCE.fetch_add(1, Ordering::Relaxed);
            std::env::temp_dir().join(format!(
                "muse-fluid-observation-{}-{nonce}.{extension}",
                std::process::id()
            ))
        }

        fn write_test_wav(path: &Path) {
            let spec = hound::WavSpec {
                channels: 2,
                sample_rate: 48_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::create(path, spec).unwrap();
            for frame in 0..4_800 {
                let sample = ((frame as f32 * 0.071).sin() * 0.12 * i16::MAX as f32) as i16;
                writer.write_sample(sample).unwrap();
                writer.write_sample(sample).unwrap();
            }
            writer.finalize().unwrap();
        }

        fn shell_quote(path: &Path) -> String {
            let value = path.to_string_lossy();
            format!("'{}'", value.replace('\'', "'\"'\"'"))
        }

        fn write_fake_executable(body: &str) -> PathBuf {
            let path = unique_path("sh");
            std::fs::write(&path, format!("#!/bin/sh\nset -eu\n{body}\n")).unwrap();
            let mut permissions = std::fs::metadata(&path).unwrap().permissions();
            permissions.set_mode(0o755);
            std::fs::set_permissions(&path, permissions).unwrap();
            path
        }

        fn fake_inputs() -> (PathBuf, PathBuf, PathBuf) {
            let midi = unique_path("mid");
            let soundfont = unique_path("sf2");
            let output_fixture = unique_path("wav");
            std::fs::write(&midi, b"MThd-fixture-midi").unwrap();
            std::fs::write(&soundfont, b"fake-soundfont-bytes").unwrap();
            write_test_wav(&output_fixture);
            (midi, soundfont, output_fixture)
        }

        fn cleanup(paths: &[&Path]) {
            for path in paths {
                let _ = std::fs::remove_file(path);
            }
        }

        #[test]
        fn successful_process_records_arguments_environment_and_both_streams() {
            let (midi, soundfont, fixture_wav) = fake_inputs();
            let body = format!(
                "output=''\nprevious=''\nfor arg in \"$@\"; do\n  if [ \"$previous\" = '-F' ]; then output=\"$arg\"; fi\n  previous=\"$arg\"\ndone\nprintf 'fake renderer stdout\\\\n'\nprintf 'fake renderer stderr\\\\n' >&2\n/bin/cp {} \"$output\"",
                shell_quote(&fixture_wav)
            );
            let script = write_fake_executable(&body);
            let attempt = render_midi_to_wav_with_paths(
                &midi,
                48_000,
                None,
                &script,
                &soundfont,
            );
            let observation = &attempt.observation;
            assert!(attempt.wav_bytes.is_some());
            assert_eq!(observation.disposition, FluidSynthInvocationDispositionV1::Rendered);
            assert!(observation.process_spawned);
            assert!(observation.process_success);
            assert_eq!(observation.process_exit_code, Some(0));
            assert!(observation.stdout_preview.contains("fake renderer stdout"));
            assert!(observation.stderr_preview.contains("fake renderer stderr"));
            assert!(!observation.arguments.is_empty());
            assert_eq!(
                observation.executable_sha256_before,
                observation.executable_sha256_after
            );
            assert_eq!(
                observation.soundfont_sha256_before,
                observation.soundfont_sha256_after
            );
            assert_eq!(
                observation.midi_sha256_before,
                observation.midi_sha256_after
            );
            assert!(!observation.environment.is_empty());
            assert!(
                observation
                    .environment
                    .iter()
                    .all(|item| item.name != "OPENAI_API_KEY" && item.name != "DEEPSEEK_API_KEY")
            );
            cleanup(&[&midi, &soundfont, &fixture_wav, &script]);
        }

        #[test]
        fn nonzero_exit_preserves_failure_diagnostics() {
            let (midi, soundfont, fixture_wav, _) = fake_inputs();
            let script = write_fake_executable(
                "printf 'expected failure\\\\n' >&2\nexit 23",
            );
            let attempt = render_midi_to_wav_with_paths(
                &midi,
                48_000,
                None,
                &script,
                &soundfont,
            );
            assert!(attempt.wav_bytes.is_none());
            assert_eq!(
                attempt.observation.disposition,
                FluidSynthInvocationDispositionV1::ProcessFailed
            );
            assert_eq!(attempt.observation.process_exit_code, Some(23));
            assert!(attempt.observation.stderr_preview.contains("expected failure"));
            assert!(attempt.observation.process_spawned);
            assert!(!attempt.observation.process_success);
            cleanup(&[&midi, &soundfont, &fixture_wav, &script]);
        }

        #[test]
        fn midi_substitution_during_child_execution_fails_closed() {
            let (midi, soundfont, fixture_wav, _) = fake_inputs();
            let script = write_fake_executable(
                "last=''\nfor arg in \"$@\"; do last=\"$arg\"; done\nprintf 'substituted-midi' > \"$last\"\nexit 0",
            );
            let attempt = render_midi_to_wav_with_paths(
                &midi,
                48_000,
                None,
                &script,
                &soundfont,
            );
            assert!(attempt.wav_bytes.is_none());
            assert_eq!(
                attempt.observation.disposition,
                FluidSynthInvocationDispositionV1::MidiChangedDuringRender
            );
            assert_ne!(
                attempt.observation.midi_sha256_before,
                attempt.observation.midi_sha256_after
            );
            cleanup(&[&midi, &soundfont, &fixture_wav, &script]);
        }

        #[test]
        fn stderr_is_drained_beyond_preview_limit_without_deadlock() {
            let (midi, soundfont, fixture_wav, _) = fake_inputs();
            let body = format!(
                "output=''\nprevious=''\nfor arg in \"$@\"; do\n  if [ \"$previous\" = '-F' ]; then output=\"$arg\"; fi\n  previous=\"$arg\"\ndone\ni=0\nwhile [ $i -lt 2000 ]; do printf 'x1234567890\\\\n' >&2; i=$((i+1)); done\n/bin/cp {} \"$output\"",
                shell_quote(&fixture_wav)
            );
            let script = write_fake_executable(&body);
            let attempt = render_midi_to_wav_with_paths(
                &midi,
                48_000,
                None,
                &script,
                &soundfont,
            );
            assert_eq!(
                attempt.observation.disposition,
                FluidSynthInvocationDispositionV1::Rendered
            );
            assert!(attempt.observation.stderr_preview_truncated);
            assert!(attempt.observation.stderr_total_bytes > DIAGNOSTIC_PREVIEW_LIMIT as u64);
            assert!(attempt.observation.stderr_sha256.is_some());
            cleanup(&[&midi, &soundfont, &fixture_wav, &script]);
        }
    }

}
