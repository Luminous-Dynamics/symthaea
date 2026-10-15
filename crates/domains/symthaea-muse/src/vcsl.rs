// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! VCSL sample-library mapper: render notes from REAL recorded instruments.
//!
//! The Versilian Community Sample Library (CC0, ~9.6 GB, 4,231 WAVs,
//! multi-velocity, organized by Hornbostel-Sachs classification) sat on disk
//! at `data/samples/vcsl` completely unreferenced by any code while every
//! note rendered through additive/KS/FM synthesis. Real recordings are the
//! single biggest timbre-quality lever available to this crate — no ML, no
//! new dependencies, just a mapper.
//!
//! Scope and honesty:
//! - Two CC0 libraries, each covering what it actually contains: VCSL
//!   (piano → Steinway B, harp, recorder-as-ney, tenor sax, pipe organ,
//!   glockenspiel, balafon, mbira, vibraphone) and VSCO2 Community Edition
//!   (`data/samples/vsco2-ce`) for the orchestral LEADS — solo violin,
//!   viola section, cello section, flute, oboe, clarinet, bassoon, trumpet,
//!   french horn. Guitar/koto/oud stay on Karplus-Strong (their physical
//!   model is genuinely good) — no fake mappings.
//! - The new orchestral instruments (Viola/Oboe/Bassoon/French Horn/
//!   Vibraphone) were verified against the actual on-disk library layout
//!   before mapping, not guessed: `Strings/Viola Section/susvib`,
//!   `Woodwinds/Oboe/Sus`, `Woodwinds/Bassoon/sus`, `Brass/F Horn/sus` (all
//!   VSCO2-CE), and `Idiophones/Struck Idiophones/Vibraphone/Hard Mallets`
//!   (VCSL) all genuinely exist and use the same `NOTE_v#` filename
//!   convention this module already parses. Timpani was investigated too
//!   and deliberately NOT mapped: both libraries ship real timpani hits
//!   (`Membranophones/Struck Membranophones/Timpani 1/Hit` in VCSL,
//!   `Percussion/Timpani` in VSCO2-CE), but every filename encodes a DRUM
//!   NUMBER (`Timpani1_Hit_v2_rr1_Sum.wav`), not a scientific pitch — this
//!   module's [`parse_note_token`] has nothing to parse there, so a real
//!   mapping isn't honestly possible without a second, drum-number-aware
//!   indexing scheme this module doesn't have. `Instrument::Timpani`
//!   therefore stays synthesis-only.
//! - The wave-2 additions (Harpsichord, Tuba, Trombone, Duduk, ChoirOoh,
//!   Xylophone) are ALSO deliberately left unmapped, for a different reason
//!   than Timpani's: at the time these were added, neither
//!   `data/samples/vcsl` nor `data/samples/vsco2-ce` existed on disk in the
//!   development environment at all, so nothing could be verified — and
//!   unlike the Viola/Oboe/Bassoon/French Horn/Vibraphone additions above
//!   (each checked against a real on-disk path before mapping), this
//!   module's own doc comments don't already establish that any of these
//!   six exist in either library. VCSL is broadly reported to include a
//!   real harpsichord and a range of world/ethnic instruments, and
//!   VSCO2-CE's brass section is reported to include tuba and trombone, but
//!   guessing an exact subdirectory path with no way to confirm it would
//!   risk a confidently-asserted mapping that silently never indexes (wrong
//!   path -> `InstrumentBank::index` returns `None`) or, worse, quietly
//!   indexes the wrong content. Per the Timpani precedent, an honest
//!   "not mapped" beats a guessed mapping — these six stay synthesis-only
//!   until someone can verify the real directory layout on a machine that
//!   has these libraries checked out.
//! - Opt-in: nothing loads unless [`init`] is called or `SYMTHAEA_VCSL_DIR`
//!   is set. When inactive, rendering is bit-identical to before this module
//!   existed.
//! - Metadata (paths, parsed pitch/dynamic/round-robin) is indexed up front;
//!   audio decodes lazily per file on first use and is cached — a whole
//!   piece touches a few dozen files (~tens of MB), never the full 9.6 GB.
//!
//! Filename convention (verified against the library on disk): tokens
//! separated by `_`, one of which is an UPPERCASE note name (`A2`, `C#5`,
//! `A#-1`); dynamics are lowercase (`f1`, `mf1`, `vl3`, `loud`); round-robins
//! are `rr2` / `var3` / trailing `01`. Lowercase-vs-uppercase is the
//! disambiguator that keeps the dynamic token `f1` from reading as the note
//! F1.

#![cfg(not(target_arch = "wasm32"))]

use crate::instruments::Instrument;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Which sample library provides an instrument. VCSL (keyboards, harps,
/// mallets, sax, organ) is complemented by VSCO2 Community Edition (also
/// CC0, `data/samples/vsco2-ce`, github.com/sgossner/VSCO-2-CE), which
/// fills VCSL's biggest gap: the orchestral LEAD instruments — solo violin,
/// cello section, flute, clarinet, trumpet — the most exposed voices in
/// most ensembles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SampleSource {
    Vcsl,
    Vsco2,
}

/// Relative subdirectory (within its library's root) providing samples for
/// a muse instrument, or `None` when neither library has an honest
/// equivalent. Sustained/vibrato articulations are chosen to match how the
/// renderer uses each instrument (long expressive lines, not effects).
fn sample_subdir(instrument: Instrument) -> Option<(SampleSource, &'static str)> {
    use SampleSource::*;
    match instrument {
        Instrument::Piano | Instrument::PianoPP => {
            Some((Vcsl, "Chordophones/Zithers/Grand Piano, Steinway B/Sus"))
        }
        Instrument::Harp => Some((Vcsl, "Chordophones/Composite Chordophones/Concert Harp")),
        // Real flute from VSCO2; the breathy baroque recorder stays the
        // honest stand-in for the ney (an end-blown flute).
        Instrument::Flute => Some((Vsco2, "Woodwinds/Flute/susvib")),
        Instrument::Ney => Some((
            Vcsl,
            "Aerophones/Edge-blown Aerophones/Baroque Alto Recorder/Sustain",
        )),
        Instrument::Violin => Some((Vsco2, "Strings/Solo Violin/Arco Vib")),
        Instrument::Viola => Some((Vsco2, "Strings/Viola Section/susvib")),
        Instrument::Cello => Some((Vsco2, "Strings/Cello Section/susvib")),
        Instrument::Clarinet => Some((Vsco2, "Woodwinds/Clarinet/susLong")),
        Instrument::Oboe => Some((Vsco2, "Woodwinds/Oboe/Sus")),
        Instrument::Bassoon => Some((Vsco2, "Woodwinds/Bassoon/sus")),
        Instrument::Trumpet => Some((Vsco2, "Brass/Trumpet/susvib")),
        Instrument::FrenchHorn => Some((Vsco2, "Brass/F Horn/sus")),
        Instrument::Saxophone => Some((Vcsl, "Aerophones/Reed Aerophones/Tenor Saxophone/Vibrato")),
        Instrument::Organ => Some((Vcsl, "Aerophones/Edge-blown Aerophones/Pipe Organ/Loud")),
        Instrument::Bell => Some((Vcsl, "Idiophones/Struck Idiophones/Glockenspiel")),
        Instrument::Marimba => Some((
            Vcsl,
            "Idiophones/Struck Idiophones/Balafon/Traditional Mallet",
        )),
        Instrument::Kalimba => Some((Vcsl, "Idiophones/Plucked Idiophones/Kalimba, Kenya")),
        Instrument::Vibraphone => {
            Some((Vcsl, "Idiophones/Struck Idiophones/Vibraphone/Hard Mallets"))
        }
        _ => None,
    }
}

/// Short-articulation (staccato/spiccato) sample directories, for notes the
/// renderer decides are played short. Only instruments whose library ships a
/// real short articulation are mapped — no fake mappings.
fn staccato_subdir(instrument: Instrument) -> Option<(SampleSource, &'static str)> {
    use SampleSource::*;
    match instrument {
        Instrument::Violin => Some((Vsco2, "Strings/Solo Violin/spic")),
        Instrument::Viola => Some((Vsco2, "Strings/Viola Section/spic")),
        Instrument::Cello => Some((Vsco2, "Strings/Cello Section/spic")),
        Instrument::Flute => Some((Vsco2, "Woodwinds/Flute/stac")),
        Instrument::Clarinet => Some((Vsco2, "Woodwinds/Clarinet/stac")),
        Instrument::Oboe => Some((Vsco2, "Woodwinds/Oboe/Stacc")),
        Instrument::Bassoon => Some((Vsco2, "Woodwinds/Bassoon/stac")),
        Instrument::Trumpet => Some((Vsco2, "Brass/Trumpet/stac")),
        Instrument::FrenchHorn => Some((Vsco2, "Brass/F Horn/stac")),
        _ => None,
    }
}

/// Parse an UPPERCASE scientific-pitch token (`A4`, `C#5`, `Bb3`, `A#-1`)
/// into a MIDI note number. Lowercase first letters are rejected on purpose:
/// VCSL writes note names uppercase and dynamics lowercase, so this is what
/// keeps `f1`/`mf2` from parsing as pitches.
fn parse_note_token(tok: &str) -> Option<u8> {
    let bytes = tok.as_bytes();
    let letter = *bytes.first()?;
    if !letter.is_ascii_uppercase() || !(b'A'..=b'G').contains(&letter) {
        return None;
    }
    let mut semis: i32 = match letter {
        b'C' => 0,
        b'D' => 2,
        b'E' => 4,
        b'F' => 5,
        b'G' => 7,
        b'A' => 9,
        b'B' => 11,
        _ => return None,
    };
    let mut i = 1;
    match bytes.get(i) {
        Some(b'#') => {
            semis += 1;
            i += 1;
        }
        Some(b'b') => {
            // Flat only counts when digits follow ("Bb3"); a bare "Bb" tail
            // like "Bark" must not sneak through (handled by the parse below).
            semis -= 1;
            i += 1;
        }
        _ => {}
    }
    let octave: i32 = tok.get(i..)?.parse().ok()?;
    let midi = (octave + 1) * 12 + semis;
    u8::try_from(midi).ok().filter(|&m| m <= 127)
}

/// Parse a lowercase dynamic token into a rough strength in [0, 1].
/// `vlN`/`vN` velocity-layer indices are returned raw as `Layer(N)` and
/// normalized per bank once the maximum layer index is known.
enum Dynamic {
    Strength(f32),
    Layer(u32),
}

fn parse_dynamic_token(tok: &str) -> Option<Dynamic> {
    // Strip trailing digits from word-style dynamics: "f1", "mf2".
    let word_end = tok.find(|c: char| c.is_ascii_digit()).unwrap_or(tok.len());
    let (word, digits) = tok.split_at(word_end);
    match word {
        "ppp" | "pp" => return Some(Dynamic::Strength(0.15)),
        "p" | "quiet" | "soft" => return Some(Dynamic::Strength(0.35)),
        "mp" => return Some(Dynamic::Strength(0.5)),
        "mf" | "medium" => return Some(Dynamic::Strength(0.65)),
        "f" => return Some(Dynamic::Strength(0.85)),
        "ff" | "fff" | "loud" => return Some(Dynamic::Strength(1.0)),
        "vl" | "v" => {
            if let Ok(n) = digits.parse::<u32>() {
                return Some(Dynamic::Layer(n));
            }
        }
        _ => {}
    }
    None
}

/// One indexed sample file. Audio decodes lazily (see [`InstrumentBank`]).
/// Round-robin take numbers (`rr2`/`var3` filename tokens) are deliberately
/// NOT parsed: takes surface as multiple same-pitch/same-dynamic candidates,
/// and [`InstrumentBank::pick`] rotates among them by seed — the path-sorted
/// index (not filesystem order, which is nondeterministic) makes that
/// rotation reproducible.
struct SampleMeta {
    path: PathBuf,
    midi: u8,
    /// Dynamic strength [0, 1] (normalized after indexing).
    dynamic: f32,
}

/// Decoded audio: mono frames + source sample rate.
pub(crate) struct DecodedSample {
    pub(crate) frames: Vec<f32>,
    pub(crate) sample_rate: u32,
    pub(crate) midi: u8,
}

/// All indexed samples for one instrument, with a decode cache.
pub(crate) struct InstrumentBank {
    samples: Vec<SampleMeta>,
    cache: Mutex<HashMap<usize, Arc<DecodedSample>>>,
}

impl InstrumentBank {
    fn index(dir: &Path) -> Option<Self> {
        let mut samples = Vec::new();
        let entries = std::fs::read_dir(dir).ok()?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("wav") {
                continue;
            }
            let stem = path.file_stem()?.to_str()?.to_string();
            // Split on '_' ONLY: '-' must stay inside tokens because negative
            // octaves ("A#-1") use it.
            let tokens: Vec<&str> = stem.split('_').collect();
            let mut midi = None;
            let mut dynamic = None;
            for tok in &tokens {
                if midi.is_none()
                    && let Some(m) = parse_note_token(tok)
                {
                    midi = Some(m);
                    continue;
                }
                if dynamic.is_none()
                    && let Some(d) = parse_dynamic_token(tok)
                {
                    dynamic = Some(d);
                }
            }
            let Some(midi) = midi else { continue };
            samples.push((
                SampleMeta {
                    path,
                    midi,
                    dynamic: 0.7, // placeholder, resolved below
                },
                dynamic,
            ));
        }
        if samples.is_empty() {
            return None;
        }
        // Path-sort so pick()'s seed-modulo round-robin is deterministic —
        // read_dir order varies by filesystem.
        samples.sort_by(|a, b| a.0.path.cmp(&b.0.path));
        // Normalize layer-style dynamics against the bank's own max layer.
        let max_layer = samples
            .iter()
            .filter_map(|(_, d)| match d {
                Some(Dynamic::Layer(n)) => Some(*n),
                _ => None,
            })
            .max()
            .unwrap_or(1)
            .max(1);
        let samples = samples
            .into_iter()
            .map(|(mut meta, dynamic)| {
                meta.dynamic = match dynamic {
                    Some(Dynamic::Strength(s)) => s,
                    Some(Dynamic::Layer(n)) => n as f32 / max_layer as f32,
                    None => 0.7,
                };
                meta
            })
            .collect();
        Some(InstrumentBank {
            samples,
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// Pick the best sample for a target pitch/velocity, decode (cached), and
    /// return it. `seed` selects among round-robins so repeated notes vary.
    /// Returns `None` when no sample is within ±7 semitones — repitching
    /// farther than a fifth sounds obviously wrong.
    pub(crate) fn pick(&self, midi: f32, velocity: f32, seed: u32) -> Option<Arc<DecodedSample>> {
        self.pick_with_window(midi, velocity, seed, 5.0, 7.0)
    }

    /// Like [`Self::pick`] but with caller-chosen repitch limits. The
    /// renderer uses a strict window first (+5/−7: past that, up-shifts
    /// chipmunk), then a RELAXED window as fallback — a far-shifted real
    /// recording still beats switching one voice's timbre to synthesis
    /// mid-line, which is how three out-of-range counter-melody notes
    /// became measured 10-18× HF "harsh spots".
    pub(crate) fn pick_with_window(
        &self,
        midi: f32,
        velocity: f32,
        seed: u32,
        max_up: f32,
        max_down: f32,
    ) -> Option<Arc<DecodedSample>> {
        let nearest_signed = self
            .samples
            .iter()
            .map(|s| s.midi as f32 - midi)
            .min_by(|a, b| a.abs().total_cmp(&b.abs()))
            .unwrap_or(f32::MAX);
        // sample below target (negative) = up-shift at play time.
        if nearest_signed < -max_up || nearest_signed > max_down {
            return None;
        }
        let nearest = nearest_signed.abs();
        let at_pitch: Vec<usize> = (0..self.samples.len())
            .filter(|&i| ((self.samples[i].midi as f32 - midi).abs() - nearest).abs() < 0.5)
            .collect();
        let best_dyn = at_pitch
            .iter()
            .map(|&i| (self.samples[i].dynamic - velocity).abs())
            .fold(f32::MAX, f32::min);
        let candidates: Vec<usize> = at_pitch
            .into_iter()
            .filter(|&i| ((self.samples[i].dynamic - velocity).abs() - best_dyn).abs() < 0.05)
            .collect();
        let chosen = candidates[seed as usize % candidates.len()];
        self.decode(chosen)
    }

    fn decode(&self, idx: usize) -> Option<Arc<DecodedSample>> {
        if let Some(hit) = self.cache.lock().ok()?.get(&idx) {
            return Some(hit.clone());
        }
        let meta = &self.samples[idx];
        let mut reader = hound::WavReader::open(&meta.path).ok()?;
        let spec = reader.spec();
        let channels = spec.channels.max(1) as usize;
        let mono: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => {
                let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
                let all: Vec<f32> = reader
                    .samples::<i32>()
                    .filter_map(Result::ok)
                    .map(|s| s as f32 / max)
                    .collect();
                all.chunks(channels)
                    .map(|c| c.iter().sum::<f32>() / channels as f32)
                    .collect()
            }
            hound::SampleFormat::Float => {
                let all: Vec<f32> = reader.samples::<f32>().filter_map(Result::ok).collect();
                all.chunks(channels)
                    .map(|c| c.iter().sum::<f32>() / channels as f32)
                    .collect()
            }
        };
        if mono.is_empty() {
            return None;
        }
        let decoded = Arc::new(DecodedSample {
            frames: mono,
            sample_rate: spec.sample_rate,
            midi: meta.midi,
        });
        self.cache.lock().ok()?.insert(idx, decoded.clone());
        Some(decoded)
    }
}

/// The loaded library: per-instrument banks, indexed once.
pub struct VcslLibrary {
    banks: HashMap<Instrument, InstrumentBank>,
    staccato_banks: HashMap<Instrument, InstrumentBank>,
}

impl VcslLibrary {
    /// Index both libraries. `root` is the VCSL directory; the VSCO2-CE
    /// root (lead instruments) is `SYMTHAEA_VSCO2_DIR` if set, else the
    /// sibling `<root>/../vsco2-ce`. Either library may be absent — each
    /// instrument indexes from whichever source it maps to, and missing
    /// sources just leave those instruments on synthesis. Returns `None`
    /// only when NO mapped directory yields parseable samples.
    pub fn load(root: &Path) -> Option<Self> {
        let vsco2_root: PathBuf = std::env::var_os("SYMTHAEA_VSCO2_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| root.join("../vsco2-ce"));
        let mut banks = HashMap::new();
        let mut staccato_banks = HashMap::new();
        for instrument in [
            Instrument::Piano,
            Instrument::PianoPP,
            Instrument::Harp,
            Instrument::Flute,
            Instrument::Ney,
            Instrument::Violin,
            Instrument::Viola,
            Instrument::Cello,
            Instrument::Clarinet,
            Instrument::Oboe,
            Instrument::Bassoon,
            Instrument::Trumpet,
            Instrument::FrenchHorn,
            Instrument::Saxophone,
            Instrument::Organ,
            Instrument::Bell,
            Instrument::Marimba,
            Instrument::Kalimba,
            Instrument::Vibraphone,
        ] {
            if let Some((source, sub)) = sample_subdir(instrument) {
                let base = match source {
                    SampleSource::Vcsl => root,
                    SampleSource::Vsco2 => &vsco2_root,
                };
                if let Some(bank) = InstrumentBank::index(&base.join(sub)) {
                    banks.insert(instrument, bank);
                }
            }
            if let Some((source, sub)) = staccato_subdir(instrument) {
                let base = match source {
                    SampleSource::Vcsl => root,
                    SampleSource::Vsco2 => &vsco2_root,
                };
                if let Some(bank) = InstrumentBank::index(&base.join(sub)) {
                    staccato_banks.insert(instrument, bank);
                }
            }
        }
        if banks.is_empty() {
            None
        } else {
            Some(Self {
                banks,
                staccato_banks,
            })
        }
    }

    pub(crate) fn bank(&self, instrument: Instrument) -> Option<&InstrumentBank> {
        self.banks.get(&instrument)
    }

    /// The short-articulation bank, when the library ships one — the
    /// renderer uses it for notes short enough to be *played* staccato
    /// (a truncated sustain sample never sounds like a real short bow).
    pub(crate) fn staccato_bank(&self, instrument: Instrument) -> Option<&InstrumentBank> {
        self.staccato_banks.get(&instrument)
    }

    /// Would a note at this pitch/velocity play from a REAL sample, or
    /// fall through to synthesis? Diagnostic surface (range-coverage
    /// probing, future Studio visualization).
    pub fn can_play(&self, instrument: Instrument, midi: f32, velocity: f32) -> bool {
        self.banks
            .get(&instrument)
            .is_some_and(|b| b.pick(midi, velocity, 0).is_some())
    }

    /// Number of instruments with a usable bank (for logging/tests).
    pub fn instrument_count(&self) -> usize {
        self.banks.len()
    }
}

static LIBRARY: OnceLock<Option<VcslLibrary>> = OnceLock::new();

/// Explicitly initialize the global library from `root`. Returns whether a
/// usable library was loaded. Idempotent: the first successful init (or the
/// first env-var lookup) wins for the process lifetime.
pub fn init(root: &Path) -> bool {
    LIBRARY.get_or_init(|| VcslLibrary::load(root)).is_some()
}

/// The process-global library: explicit [`init`] first, else the
/// `SYMTHAEA_VCSL_DIR` env var, else inactive (synthesis-only — the
/// pre-existing behavior, bit-identical).
pub fn library() -> Option<&'static VcslLibrary> {
    LIBRARY
        .get_or_init(|| {
            let dir = std::env::var_os("SYMTHAEA_VCSL_DIR")?;
            let lib = VcslLibrary::load(Path::new(&dir));
            if lib.is_none() {
                eprintln!(
                    "[muse::vcsl] SYMTHAEA_VCSL_DIR set but no usable library at {:?} — \
                     falling back to synthesis",
                    dir
                );
            }
            lib
        })
        .as_ref()
}

/// Source-derived sample and renderer capability snapshot. This reports
/// declared mappings separately from observed on-disk assets: a mapping
/// declaration alone is not evidence that a sample library is installed.
#[derive(Debug, Clone, Serialize)]
pub struct InstrumentCapabilityInventoryV1 {
    pub schema_version: &'static str,
    pub source_revision: Option<String>,
    pub instrument_count: usize,
    pub vcsl_root: AssetRootEvidenceV1,
    pub vsco2_root: AssetRootEvidenceV1,
    pub soundfont: Option<SoundFontEvidenceV1>,
    /// Present only when the caller explicitly opts into FluidSynth runtime
    /// rendering with --fluidsynth. Header matches and rendered output remain
    /// separate evidence dimensions.
    pub fluidsynth_probe: Option<FluidSynthRuntimeProbeV1>,
    pub instruments: Vec<InstrumentCapabilityV1>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FluidSynthRuntimeProbeV1 {
    pub status: &'static str,
    pub binary_name: String,
    /// First non-empty line from fluidsynth --version; null if unsupported or unsuccessful.
    pub binary_version: Option<String>,
    /// SHA-256 of the resolved executable bytes, not of its shared libraries.
    pub binary_sha256: String,
    pub soundfont_sha256: String,
    pub requested_program_count: usize,
    pub probe_count: usize,
    pub rendered_non_silent_probe_count: usize,
    /// Fixed coverage grid. A non-silent result at one cell does not certify the other cells.
    pub probe_notes_midi: Vec<u8>,
    pub probe_velocities: Vec<u8>,
    pub sample_rate_hz: u32,
    pub reverb_disabled: bool,
    pub chorus_disabled: bool,
    /// Stable argument template; temp paths and the executable's full path are excluded.
    pub command_template: &'static str,
    pub programs: Vec<FluidSynthProgramProbeV1>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FluidSynthProgramProbeV1 {
    pub bank: u16,
    pub program_0_based: u8,
    pub note_midi: u8,
    pub velocity: u8,
    pub soundfont_header_preset_names: Vec<String>,
    /// Statuses include skipped_no_bank0_header_match, fluidsynth_exit_nonzero,
    /// fluidsynth_success_without_output_wav, output_wav_decode_failed,
    /// rendered_silent and rendered_non_silent.
    pub status: &'static str,
    pub command_exit_code: Option<i32>,
    pub sample_rate: Option<u32>,
    pub channels: Option<u16>,
    pub frames: Option<u64>,
    /// Hash of the exact output WAV bytes, when an output file was produced and readable.
    pub output_wav_sha256: Option<String>,
    pub peak_absolute_sample: Option<f32>,
    pub rms: Option<f32>,
}

/// Evidence for one library root. The digest binds sorted relative WAV paths
/// and file sizes, not file contents; it is a lightweight asset-layout seal.
#[derive(Debug, Clone, Serialize)]
pub struct AssetRootEvidenceV1 {
    pub root_present: bool,
    pub wav_file_count: usize,
    pub path_size_manifest_sha256: Option<String>,
}

/// Evidence for an optional SoundFont file. Unlike the path/size manifest
/// above, this sha256 digest is computed over the file bytes.
#[derive(Debug, Clone, Serialize)]
pub struct SoundFontEvidenceV1 {
    pub file_name: String,
    pub size_bytes: u64,
    pub sha256: String,
    /// Parsed from the RIFF/SoundFont preset-header table. This is header
    /// inventory only; it does not certify samples, FluidSynth loadability,
    /// or audible fidelity.
    pub preset_inventory: SoundFontPresetInventoryV1,
}

#[derive(Debug, Clone, Serialize)]
pub struct SoundFontPresetInventoryV1 {
    pub status: &'static str,
    pub declared_preset_count: usize,
    pub declared_bank_count: usize,
    pub duplicate_bank_program_slots: usize,
    pub presets: Vec<SoundFontPresetV1>,
}

/// SoundFont preset-header identity. Program and bank are the values stored
/// in the SoundFont header, not a claim about source instruments or quality.
#[derive(Debug, Clone, Serialize)]
pub struct SoundFontPresetV1 {
    pub name: String,
    pub bank: u16,
    pub program: u16,
    pub bag_index: u16,
}

/// A single sustained or short-articulation mapping and its local availability.
#[derive(Debug, Clone, Serialize)]
pub struct SampleDirectoryEvidenceV1 {
    pub source_library: &'static str,
    pub relative_directory: String,
    pub directory_present: bool,
    /// Number of WAV files whose filenames are accepted by the current
    /// sample indexer. This does not prove that file decoding will succeed.
    pub indexed_sample_count: usize,
    /// MIDI-note coverage observed from accepted filenames, not rendered range.
    pub min_midi: Option<u8>,
    pub max_midi: Option<u8>,
    pub distinct_pitch_count: usize,
    /// Distinct normalized dynamic values parsed from indexed filenames; not calibrated layers.
    pub distinct_dynamic_strength_count: usize,
    /// Result of fully reading every WAV accepted by the filename indexer.
    /// This does not establish rendered playback or perceptual quality.
    pub audio_decode_status: &'static str,
    pub decoded_sample_file_count: usize,
    pub failed_sample_decode_count: usize,
    /// Fully decoded files whose peak absolute sample value is <= 1e-5.
    pub silent_sample_file_count: usize,
    /// Distinct sample rates/channels across files that fully decoded.
    pub distinct_sample_rate_count: usize,
    pub distinct_channel_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct InstrumentCapabilityV1 {
    pub instrument: String,
    /// General MIDI program number, zero based. Some instruments use a
    /// documented nearest-family approximation; this is not source identity.
    pub gm_program_0_based: u8,
    /// Source-derived dispatch family. This does not prove audible fidelity.
    pub procedural_renderer_path: &'static str,
    pub sustained_samples: Option<SampleDirectoryEvidenceV1>,
    pub short_articulation_samples: Option<SampleDirectoryEvidenceV1>,
    /// Names declared by an optional SoundFont at bank 0 and this instrument's
    /// selected GM program. None means no SoundFont was supplied; Some(empty)
    /// means the inspected header table had no exact bank/program match.
    /// A match is still not evidence that the preset will render successfully.
    pub soundfont_bank0_program_matches: Option<Vec<String>>,
}

/// Build a deterministic capability snapshot from the current source mappings
/// and the asset roots the caller actually supplied. Missing optional libraries
/// are represented as absent evidence rather than silently counted as coverage.
pub fn capability_inventory(
    vcsl_root: &Path,
    vsco2_root: &Path,
    soundfont_path: Option<&Path>,
) -> std::io::Result<InstrumentCapabilityInventoryV1> {
    capability_inventory_with_fluidsynth(vcsl_root, vsco2_root, soundfont_path, None)
}

/// Build the same inventory with optional, explicit FluidSynth rendering.
/// Passing a binary requests real subprocess probes; no SoundFont or binary
/// is discovered implicitly, so default inventory remains side-effect free.
pub fn capability_inventory_with_fluidsynth(
    vcsl_root: &Path,
    vsco2_root: &Path,
    soundfont_path: Option<&Path>,
    fluidsynth_path: Option<&Path>,
) -> std::io::Result<InstrumentCapabilityInventoryV1> {
    if fluidsynth_path.is_some() && soundfont_path.is_none() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "FluidSynth runtime probes require --soundfont",
        ));
    }

    const INSTRUMENTS: [Instrument; 42] = [
        Instrument::Piano,
        Instrument::PianoPP,
        Instrument::Violin,
        Instrument::Cello,
        Instrument::Flute,
        Instrument::Organ,
        Instrument::ElectricPiano,
        Instrument::Bell,
        Instrument::AcousticGuitar,
        Instrument::Harp,
        Instrument::Pad,
        Instrument::Clarinet,
        Instrument::Trumpet,
        Instrument::Saxophone,
        Instrument::Marimba,
        Instrument::Sitar,
        Instrument::Kalimba,
        Instrument::SawLead,
        Instrument::Koto,
        Instrument::Oud,
        Instrument::Ney,
        Instrument::UprightBass,
        Instrument::Viola,
        Instrument::Oboe,
        Instrument::FrenchHorn,
        Instrument::Bassoon,
        Instrument::Accordion,
        Instrument::Mandolin,
        Instrument::Banjo,
        Instrument::Vibraphone,
        Instrument::Celesta,
        Instrument::ChoirAah,
        Instrument::StringEnsemble,
        Instrument::ElectricBass,
        Instrument::Timpani,
        Instrument::MusicBox,
        Instrument::Harpsichord,
        Instrument::Tuba,
        Instrument::Trombone,
        Instrument::Duduk,
        Instrument::ChoirOoh,
        Instrument::Xylophone,
    ];

    let vcsl_evidence = asset_root_evidence(vcsl_root)?;
    let vsco2_evidence = asset_root_evidence(vsco2_root)?;
    let soundfont = soundfont_path.map(soundfont_evidence).transpose()?;
    let mut sample_directory_cache =
        HashMap::<(SampleSource, String), Option<SampleDirectoryEvidenceV1>>::new();
    let instruments = INSTRUMENTS
        .iter()
        .copied()
        .map(|instrument| {
            let sustained_samples = sample_directory_evidence_cached(
                instrument,
                false,
                vcsl_root,
                vsco2_root,
                &mut sample_directory_cache,
            );
            let short_articulation_samples = sample_directory_evidence_cached(
                instrument,
                true,
                vcsl_root,
                vsco2_root,
                &mut sample_directory_cache,
            );
            let soundfont_bank0_program_matches = soundfont.as_ref().map(|evidence| {
                evidence
                    .preset_inventory
                    .presets
                    .iter()
                    .filter(|preset| {
                        preset.bank == 0 && preset.program == instrument.gm_program() as u16
                    })
                    .map(|preset| preset.name.clone())
                    .collect()
            });
            InstrumentCapabilityV1 {
                instrument: format!("{instrument:?}"),
                gm_program_0_based: instrument.gm_program(),
                procedural_renderer_path: procedural_renderer_path(instrument),
                sustained_samples,
                short_articulation_samples,
                soundfont_bank0_program_matches,
            }
        })
        .collect::<Vec<_>>();
    let source_revision = std::env::var("SYMTHAEA_SOURCE_REV")
        .or_else(|_| std::env::var("GITHUB_SHA"))
        .ok()
        .filter(|value| !value.trim().is_empty());

    let fluidsynth_probe = match (fluidsynth_path, soundfont_path, soundfont.as_ref()) {
        (Some(binary), Some(soundfont_file), Some(evidence)) => Some(
            probe_fluidsynth_programs(
                binary,
                soundfont_file,
                &evidence.sha256,
                &evidence.preset_inventory,
                INSTRUMENTS.iter().copied().map(Instrument::gm_program),
            )?,
        ),
        _ => None,
    };

    Ok(InstrumentCapabilityInventoryV1 {
        schema_version: "muse-instrument-capability-inventory-v1",
        source_revision,
        instrument_count: instruments.len(),
        vcsl_root: vcsl_evidence,
        vsco2_root: vsco2_evidence,
        soundfont,
        fluidsynth_probe,
        instruments,
    })
}

fn procedural_renderer_path(instrument: Instrument) -> &'static str {
    if instrument.uses_karplus_strong() {
        "karplus_strong"
    } else if instrument.uses_fm() {
        "frequency_modulation"
    } else {
        "additive"
    }
}

fn sample_directory_evidence_cached(
    instrument: Instrument,
    short_articulation: bool,
    vcsl_root: &Path,
    vsco2_root: &Path,
    cache: &mut HashMap<(SampleSource, String), Option<SampleDirectoryEvidenceV1>>,
) -> Option<SampleDirectoryEvidenceV1> {
    let mapping = if short_articulation {
        staccato_subdir(instrument)
    } else {
        sample_subdir(instrument)
    }?;
    let key = (mapping.0, mapping.1.to_owned());
    if let Some(cached) = cache.get(&key) {
        return cached.clone();
    }
    let evidence = sample_directory_evidence(instrument, short_articulation, vcsl_root, vsco2_root);
    cache.insert(key, evidence.clone());
    evidence
}

fn sample_directory_evidence(
    instrument: Instrument,
    short_articulation: bool,
    vcsl_root: &Path,
    vsco2_root: &Path,
) -> Option<SampleDirectoryEvidenceV1> {
    let mapping = if short_articulation {
        staccato_subdir(instrument)
    } else {
        sample_subdir(instrument)
    }?;
    let (source, relative_directory) = mapping;
    let source_name = match source {
        SampleSource::Vcsl => "vcsl",
        SampleSource::Vsco2 => "vsco2_ce",
    };
    let root = match source {
        SampleSource::Vcsl => vcsl_root,
        SampleSource::Vsco2 => vsco2_root,
    };
    let directory = root.join(relative_directory);
    let directory_present = directory.is_dir();
    let bank = if directory_present {
        InstrumentBank::index(&directory)
    } else {
        None
    };
    let indexed_sample_count = bank.as_ref().map_or(0, |bank| bank.samples.len());
    let audio_audit = audit_indexed_sample_audio(bank.as_ref());
    let audio_decode_status = if !directory_present {
        "mapped_directory_missing"
    } else if indexed_sample_count == 0 {
        "no_indexed_samples"
    } else if audio_audit.failed_sample_decode_count == 0 {
        "all_indexed_files_fully_decoded"
    } else if audio_audit.decoded_sample_file_count == 0 {
        "no_indexed_files_fully_decoded"
    } else {
        "some_indexed_files_failed_decode"
    };
    let min_midi = bank
        .as_ref()
        .and_then(|bank| bank.samples.iter().map(|sample| sample.midi).min());
    let max_midi = bank
        .as_ref()
        .and_then(|bank| bank.samples.iter().map(|sample| sample.midi).max());
    let distinct_pitch_count = bank.as_ref().map_or(0, |bank| {
        bank.samples
            .iter()
            .map(|sample| sample.midi)
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    });
    let distinct_dynamic_strength_count = bank.as_ref().map_or(0, |bank| {
        bank.samples
            .iter()
            .map(|sample| sample.dynamic.to_bits())
            .collect::<std::collections::HashSet<_>>()
            .len()
    });
    Some(SampleDirectoryEvidenceV1 {
        source_library: source_name,
        relative_directory: relative_directory.to_owned(),
        directory_present,
        indexed_sample_count,
        min_midi,
        max_midi,
        distinct_pitch_count,
        distinct_dynamic_strength_count,
        audio_decode_status,
        decoded_sample_file_count: audio_audit.decoded_sample_file_count,
        failed_sample_decode_count: audio_audit.failed_sample_decode_count,
        silent_sample_file_count: audio_audit.silent_sample_file_count,
        distinct_sample_rate_count: audio_audit.sample_rates.len(),
        distinct_channel_count: audio_audit.channel_counts.len(),
    })
}

#[derive(Default)]
struct IndexedSampleAudioAudit {
    decoded_sample_file_count: usize,
    failed_sample_decode_count: usize,
    silent_sample_file_count: usize,
    sample_rates: std::collections::BTreeSet<u32>,
    channel_counts: std::collections::BTreeSet<u16>,
}

fn audit_indexed_sample_audio(bank: Option<&InstrumentBank>) -> IndexedSampleAudioAudit {
    let mut audit = IndexedSampleAudioAudit::default();
    let Some(bank) = bank else {
        return audit;
    };

    for sample in &bank.samples {
        match audit_one_wav(&sample.path) {
            Ok((sample_rate, channels, peak, _, _)) => {
                audit.decoded_sample_file_count += 1;
                audit.sample_rates.insert(sample_rate);
                audit.channel_counts.insert(channels);
                if peak <= 1e-5 {
                    audit.silent_sample_file_count += 1;
                }
            }
            Err(_) => audit.failed_sample_decode_count += 1,
        }
    }
    audit
}

/// Fully consume the WAV sample iterator rather than treating a valid header
/// as a successful decode. Invalid/truncated sample data, empty audio, invalid
/// channel/rate metadata and non-finite float samples fail this file-level audit.
fn audit_one_wav(path: &Path) -> std::io::Result<(u32, u16, f32, f32, u64)> {
    let mut reader = hound::WavReader::open(path).map_err(|error| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string())
    })?;
    let spec = reader.spec();
    if spec.channels == 0
        || spec.sample_rate == 0
        || spec.bits_per_sample == 0
        || spec.bits_per_sample > 32
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid WAV sample rate, channel count, or bit depth",
        ));
    }

    let mut sample_count = 0u64;
    let mut peak = 0.0f32;
    let mut sum_squares = 0.0f64;
    match spec.sample_format {
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            for sample in reader.samples::<i32>() {
                let sample = sample.map_err(|error| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string())
                })?;
                let normalized = sample as f32 / scale;
                peak = peak.max(normalized.abs());
                sum_squares += f64::from(normalized) * f64::from(normalized);
                sample_count += 1;
            }
        }
        hound::SampleFormat::Float => {
            for sample in reader.samples::<f32>() {
                let sample = sample.map_err(|error| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string())
                })?;
                if !sample.is_finite() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "WAV contains a non-finite floating-point sample",
                    ));
                }
                peak = peak.max(sample.abs());
                sum_squares += f64::from(sample) * f64::from(sample);
                sample_count += 1;
            }
        }
    }
    if sample_count == 0 || sample_count % u64::from(spec.channels) != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "WAV has no complete audio frames",
        ));
    }
    let frames = sample_count / u64::from(spec.channels);
    let rms = (sum_squares / sample_count as f64).sqrt() as f32;
    Ok((spec.sample_rate, spec.channels, peak, rms, frames))
}

fn resolve_executable(binary: &Path) -> std::io::Result<PathBuf> {
    let has_path_component = binary.is_absolute()
        || binary.components().count() > 1
        || binary
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)));
    let resolved = if binary.is_absolute() || has_path_component {
        binary.to_path_buf()
    } else {
        std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
            .map(|directory| directory.join(binary))
            .find(|candidate| candidate.is_file())
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("FluidSynth executable not found on PATH: {}", binary.display()),
                )
            })?
    };
    if !resolved.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("FluidSynth executable is not a file: {}", resolved.display()),
        ));
    }
    Ok(resolved)
}

fn sha256_file(path: &Path) -> std::io::Result<String> {
    use std::io::Read as _;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let digest = hasher.finalize();
    Ok(hex_digest(&digest[..]))
}

static FLUIDSYNTH_PROBE_COUNTER: AtomicU64 = AtomicU64::new(0);

fn probe_midi_bytes(program: u8, note: u8, velocity: u8) -> Vec<u8> {
    // SMF format 0, one track, 480 PPQN at the MIDI default tempo (120 BPM).
    // Program change + one half-second note + half-second release tail.
    let track = [
        0x00, 0xB0, 0x00, 0x00,       // bank-select MSB = 0, channel 0
        0x00, 0xB0, 0x20, 0x00,       // bank-select LSB = 0, channel 0
        0x00, 0xC0, program,          // bank-0 program change, channel 0
        0x00, 0x90, note, velocity,    // note on, channel 0
        0x83, 0x60, 0x80, note, 0x00,  // delta 480, note off
        0x83, 0x60, 0xFF, 0x2F, 0x00,  // delta 480, end of track
    ];
    let mut midi = Vec::with_capacity(14 + 8 + track.len());
    midi.extend_from_slice(b"MThd");
    midi.extend_from_slice(&6u32.to_be_bytes());
    midi.extend_from_slice(&0u16.to_be_bytes());
    midi.extend_from_slice(&1u16.to_be_bytes());
    midi.extend_from_slice(&480u16.to_be_bytes());
    midi.extend_from_slice(b"MTrk");
    midi.extend_from_slice(&(track.len() as u32).to_be_bytes());
    midi.extend_from_slice(&track);
    midi
}

fn probe_fluidsynth_programs(
    binary: &Path,
    soundfont_path: &Path,
    soundfont_sha256: &str,
    preset_inventory: &SoundFontPresetInventoryV1,
    programs: impl Iterator<Item = u8>,
) -> std::io::Result<FluidSynthRuntimeProbeV1> {
    let resolved_binary = resolve_executable(binary)?;
    let binary_sha256 = sha256_file(&resolved_binary)?;
    let version_output = std::process::Command::new(&resolved_binary)
        .arg("--version")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()?;
    let version_text = if version_output.status.success() {
        let stdout = String::from_utf8_lossy(&version_output.stdout);
        let stderr = String::from_utf8_lossy(&version_output.stderr);
        stdout
            .lines()
            .chain(stderr.lines())
            .map(str::trim)
            .find(|line| !line.is_empty())
            .map(|line| line.chars().take(200).collect::<String>())
    } else {
        None
    };

    let unique_programs = programs.collect::<std::collections::BTreeSet<_>>();
    let requested_program_count = unique_programs.len();
    const PROBE_NOTES_MIDI: [u8; 3] = [48, 60, 72];
    const PROBE_VELOCITIES: [u8; 3] = [48, 96, 120];
    let mut results = Vec::with_capacity(unique_programs.len() * PROBE_NOTES_MIDI.len() * PROBE_VELOCITIES.len());
    let mut rendered_non_silent_probe_count = 0usize;

    for program in unique_programs {
        let header_names = preset_inventory
            .presets
            .iter()
            .filter(|preset| preset.bank == 0 && preset.program == u16::from(program))
            .map(|preset| preset.name.clone())
            .collect::<Vec<_>>();

        for note in PROBE_NOTES_MIDI {
            for velocity in PROBE_VELOCITIES {
                if header_names.is_empty() {
                    results.push(FluidSynthProgramProbeV1 {
                        bank: 0,
                        program_0_based: program,
                        note_midi: note,
                        velocity,
                        soundfont_header_preset_names: header_names.clone(),
                        status: "skipped_no_bank0_header_match",
                        command_exit_code: None,
                        sample_rate: None,
                        channels: None,
                        frames: None,
                        output_wav_sha256: None,
                        peak_absolute_sample: None,
                        rms: None,
                    });
                    continue;
                }

                let unique = FLUIDSYNTH_PROBE_COUNTER.fetch_add(1, Ordering::Relaxed);
                let stem = format!(
                    "muse_fluid_probe_{}_{}_{}_{}_{}",
                    std::process::id(), unique, program, note, velocity
                );
                let temp = std::env::temp_dir();
                let midi_path = temp.join(format!("{stem}.mid"));
                let wav_path = temp.join(format!("{stem}.wav"));
                // A previous process may have crashed before cleanup while reusing a
                // PID; a stale WAV must never qualify as this invocation's evidence.
                let _ = std::fs::remove_file(&wav_path);
                let _ = std::fs::remove_file(&midi_path);
                std::fs::write(&midi_path, probe_midi_bytes(program, note, velocity))?;

                // The -n/-i pair disables live MIDI and the shell. -F renders the
                // supplied MIDI to a WAV file without requiring an audio device.
                // Disable chorus/reverb to keep this small mechanical probe controlled.
                let command_result = std::process::Command::new(&resolved_binary)
                    .arg("-n")
                    .arg("-i")
                    .arg("-g")
                    .arg("0.8")
                    .arg("-C0")
                    .arg("-R0")
                    .arg("-T")
                    .arg("wav")
                    .arg("-F")
                    .arg(&wav_path)
                    .arg("-r")
                    .arg("22050")
                    .arg(soundfont_path)
                    .arg(&midi_path)
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status();

                let _ = std::fs::remove_file(&midi_path);
                let mut result = FluidSynthProgramProbeV1 {
                    bank: 0,
                    program_0_based: program,
                    note_midi: note,
                    velocity,
                    soundfont_header_preset_names: header_names.clone(),
                    status: "fluidsynth_exit_nonzero",
                    command_exit_code: None,
                    sample_rate: None,
                    channels: None,
                    frames: None,
                    output_wav_sha256: None,
                    peak_absolute_sample: None,
                    rms: None,
                };
                match command_result {
                    Ok(status) => {
                        result.command_exit_code = status.code();
                        if status.success() {
                            if wav_path.is_file() {
                                result.output_wav_sha256 = sha256_file(&wav_path).ok();
                            }
                            match audit_one_wav(&wav_path) {
                                Ok((sample_rate, channels, peak, rms, frames)) => {
                                    result.sample_rate = Some(sample_rate);
                                    result.channels = Some(channels);
                                    result.frames = Some(frames);
                                    result.peak_absolute_sample = Some(peak);
                                    result.rms = Some(rms);
                                    if peak <= 1e-5 {
                                        result.status = "rendered_silent";
                                    } else {
                                        result.status = "rendered_non_silent";
                                        rendered_non_silent_probe_count += 1;
                                    }
                                }
                                Err(_) => {
                                    result.status = if wav_path.is_file() {
                                        "output_wav_decode_failed"
                                    } else {
                                        "fluidsynth_success_without_output_wav"
                                    };
                                }
                            }
                        }
                    }
                    Err(error) => {
                        let _ = std::fs::remove_file(&wav_path);
                        return Err(std::io::Error::new(
                            error.kind(),
                            format!(
                                "failed to execute FluidSynth probe binary {}: {error}",
                                binary.display()
                            ),
                        ));
                    }
                }
                let _ = std::fs::remove_file(&wav_path);
                results.push(result);
            }
        }
    }

    Ok(FluidSynthRuntimeProbeV1 {
        status: "fluidsynth_render_probes_completed",
        binary_name: resolved_binary
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned(),
        binary_version: version_text,
        binary_sha256,
        soundfont_sha256: soundfont_sha256.to_owned(),
        requested_program_count,
        probe_count: results.len(),
        rendered_non_silent_probe_count,
        probe_notes_midi: PROBE_NOTES_MIDI.to_vec(),
        probe_velocities: PROBE_VELOCITIES.to_vec(),
        sample_rate_hz: 22050,
        reverb_disabled: true,
        chorus_disabled: true,
        command_template: "-n -i -g 0.8 -C0 -R0 -T wav -F <output.wav> -r 22050 <soundfont.sf2> <probe.mid>",
        programs: results,
    })
}

fn asset_root_evidence(root: &Path) -> std::io::Result<AssetRootEvidenceV1> {
    let metadata = match std::fs::metadata(root) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(AssetRootEvidenceV1 {
                root_present: false,
                wav_file_count: 0,
                path_size_manifest_sha256: None,
            });
        }
        Err(error) => return Err(error),
    };
    if !metadata.is_dir() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("sample root is not a directory: {}", root.display()),
        ));
    }

    let mut entries = Vec::<(String, u64)>::new();
    collect_wav_manifest(root, root, &mut entries)?;
    entries.sort_unstable_by(|left, right| left.0.cmp(&right.0));

    let mut hasher = Sha256::new();
    for (relative_path, size_bytes) in &entries {
        // Length-prefix the path and use a fixed-endian size encoding so
        // filenames containing separators cannot make the manifest ambiguous.
        hasher.update((relative_path.len() as u64).to_le_bytes());
        hasher.update(relative_path.as_bytes());
        hasher.update(size_bytes.to_le_bytes());
    }
    let digest = hasher.finalize();
    Ok(AssetRootEvidenceV1 {
        root_present: true,
        wav_file_count: entries.len(),
        path_size_manifest_sha256: Some(hex_digest(&digest[..])),
    })
}

fn collect_wav_manifest(
    root: &Path,
    current: &Path,
    output: &mut Vec<(String, u64)>,
) -> std::io::Result<()> {
    let mut entries = std::fs::read_dir(current)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_unstable_by_key(|entry| entry.file_name());
    for entry in entries {
        let file_type = entry.file_type()?;
        let name = entry.file_name();
        if file_type.is_dir() {
            if name == ".git" || name == ".svn" {
                continue;
            }
            collect_wav_manifest(root, &entry.path(), output)?;
        } else if file_type.is_file()
            && entry
                .path()
                .extension()
                .and_then(|extension| extension.to_str())
                == Some("wav")
        {
            let path = entry.path();
            let relative = path.strip_prefix(root).unwrap_or(&path);
            let relative_path = relative
                .components()
                .map(|component| component.as_os_str().to_string_lossy())
                .collect::<Vec<_>>()
                .join("/");
            output.push((relative_path, entry.metadata()?.len()));
        }
    }
    Ok(())
}

fn soundfont_evidence(path: &Path) -> std::io::Result<SoundFontEvidenceV1> {
    use std::io::{Read as _, Seek as _, SeekFrom};

    let mut file = std::fs::File::open(path)?;
    let size_bytes = file.metadata()?.len();
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let digest = hasher.finalize();
    file.seek(SeekFrom::Start(0))?;
    let preset_inventory = parse_soundfont_preset_headers(&mut file, size_bytes)?;
    Ok(SoundFontEvidenceV1 {
        file_name: path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned(),
        size_bytes,
        sha256: hex_digest(&digest[..]),
        preset_inventory,
    })
}

fn parse_soundfont_preset_headers(
    file: &mut std::fs::File,
    file_size: u64,
) -> std::io::Result<SoundFontPresetInventoryV1> {
    use std::io::{Read as _, Seek as _, SeekFrom};

    let mut header = [0u8; 12];
    file.seek(SeekFrom::Start(0))?;
    file.read_exact(&mut header).map_err(|error| {
        invalid_soundfont(format!("truncated RIFF header: {error}"))
    })?;
    if &header[0..4] != b"RIFF" || &header[8..12] != b"sfbk" {
        return Err(invalid_soundfont(
            "expected a RIFF SoundFont (sfbk) container",
        ));
    }
    let riff_end = 8u64 + u32::from_le_bytes(header[4..8].try_into().unwrap()) as u64;
    if riff_end < 12 || riff_end > file_size {
        return Err(invalid_soundfont(
            "RIFF container length extends beyond the file",
        ));
    }

    let mut cursor = 12u64;
    let mut preset_header_data: Option<(u64, u32)> = None;
    while cursor + 8 <= riff_end {
        let (chunk_id, chunk_size, data_start, data_end) =
            read_riff_chunk_header(file, cursor, riff_end)?;
        if &chunk_id == b"LIST" {
            if chunk_size < 4 {
                return Err(invalid_soundfont("LIST chunk is shorter than its type"));
            }
            let mut list_type = [0u8; 4];
            file.seek(SeekFrom::Start(data_start))?;
            file.read_exact(&mut list_type)?;
            if &list_type == b"pdta" {
                let mut sub_cursor = data_start + 4;
                while sub_cursor + 8 <= data_end {
                    let (sub_id, sub_size, sub_start, sub_end) =
                        read_riff_chunk_header(file, sub_cursor, data_end)?;
                    if &sub_id == b"phdr" {
                        if preset_header_data.is_some() {
                            return Err(invalid_soundfont(
                                "SoundFont contains multiple pdta/phdr chunks",
                            ));
                        }
                        preset_header_data = Some((sub_start, sub_size));
                    }
                    sub_cursor = sub_end + (sub_size as u64 % 2);
                }
                if sub_cursor != data_end {
                    return Err(invalid_soundfont(
                        "malformed chunk boundary inside pdta LIST",
                    ));
                }
            }
        }
        cursor = data_end + (chunk_size as u64 % 2);
    }
    if cursor != riff_end {
        return Err(invalid_soundfont("malformed RIFF chunk boundary"));
    }

    let (phdr_start, phdr_size) = preset_header_data
        .ok_or_else(|| invalid_soundfont("SoundFont has no pdta/phdr preset-header table"))?;
    const PRESET_HEADER_BYTES: u32 = 38;
    if phdr_size < PRESET_HEADER_BYTES || phdr_size % PRESET_HEADER_BYTES != 0 {
        return Err(invalid_soundfont(
            "pdta/phdr size is invalid or omits the required terminal EOP record",
        ));
    }

    let records = phdr_size / PRESET_HEADER_BYTES;
    // Keep parsing bounded even when pointed at an adversarial multi-gigabyte
    // file whose phdr chunk is syntactically large but unreasonable.
    const MAX_PRESET_HEADERS: u32 = 1_000_000;
    if records > MAX_PRESET_HEADERS {
        return Err(invalid_soundfont(format!(
            "pdta/phdr contains {records} records, exceeding the parser budget"
        )));
    }
    let mut presets = Vec::with_capacity(records.saturating_sub(1) as usize);
    for index in 0..records {
        let mut record = [0u8; PRESET_HEADER_BYTES as usize];
        file.seek(SeekFrom::Start(
            phdr_start + index as u64 * PRESET_HEADER_BYTES as u64,
        ))?;
        file.read_exact(&mut record)?;
        let name_end = record[0..20]
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(20);
        let name = String::from_utf8_lossy(&record[0..name_end])
            .trim()
            .to_owned();
        let program = u16::from_le_bytes([record[20], record[21]]);
        let bank = u16::from_le_bytes([record[22], record[23]]);
        let bag_index = u16::from_le_bytes([record[24], record[25]]);

        if index + 1 == records {
            if name != "EOP" {
                return Err(invalid_soundfont(
                    "pdta/phdr terminal record is not named EOP",
                ));
            }
        } else {
            if name.is_empty() {
                return Err(invalid_soundfont(
                    "SoundFont preset header has an empty name",
                ));
            }
            if program > 127 {
                return Err(invalid_soundfont(format!(
                    "SoundFont preset {name:?} has out-of-range program {program}"
                )));
            }
            presets.push(SoundFontPresetV1 {
                name,
                bank,
                program,
                bag_index,
            });
        }
    }

    // Normalize output order so semantically equivalent header tables produce
    // stable output even when the file stored records in a different order.
    presets.sort_unstable_by(|left, right| {
        (left.bank, left.program, &left.name).cmp(&(right.bank, right.program, &right.name))
    });
    let mut banks = std::collections::BTreeSet::new();
    let mut slots = std::collections::BTreeMap::<(u16, u16), usize>::new();
    for preset in &presets {
        banks.insert(preset.bank);
        *slots.entry((preset.bank, preset.program)).or_default() += 1;
    }
    let duplicate_bank_program_slots = slots.values().filter(|count| **count > 1).count();

    Ok(SoundFontPresetInventoryV1 {
        status: "preset_headers_parsed_not_renderability",
        declared_preset_count: presets.len(),
        declared_bank_count: banks.len(),
        duplicate_bank_program_slots,
        presets,
    })
}

fn read_riff_chunk_header(
    file: &mut std::fs::File,
    position: u64,
    parent_end: u64,
) -> std::io::Result<([u8; 4], u32, u64, u64)> {
    use std::io::{Read as _, Seek as _, SeekFrom};

    let mut header = [0u8; 8];
    file.seek(SeekFrom::Start(position))?;
    file.read_exact(&mut header)?;
    let chunk_id = header[0..4].try_into().unwrap();
    let chunk_size = u32::from_le_bytes(header[4..8].try_into().unwrap());
    let data_start = position + 8;
    let data_end = data_start + chunk_size as u64;
    if data_end > parent_end {
        return Err(invalid_soundfont("RIFF chunk extends beyond its parent"));
    }
    Ok((chunk_id, chunk_size, data_start, data_end))
}

fn invalid_soundfont(message: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into())
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    bytes
        .as_ref()
        .iter()
        .flat_map(|byte| {
            let value = *byte;
            [
                HEX[(value >> 4) as usize] as char,
                HEX[(value & 0x0f) as usize] as char,
            ]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn note_token_parses_scientific_pitch() {
        assert_eq!(parse_note_token("A4"), Some(69));
        assert_eq!(parse_note_token("C4"), Some(60));
        assert_eq!(parse_note_token("C#5"), Some(73));
        assert_eq!(parse_note_token("Bb3"), Some(58));
        assert_eq!(parse_note_token("A#-1"), Some(10));
    }

    #[test]
    fn note_token_rejects_dynamics_and_names() {
        // The classic trap: lowercase dynamic tokens must NOT parse as notes.
        assert_eq!(parse_note_token("f1"), None);
        assert_eq!(parse_note_token("mf1"), None);
        assert_eq!(parse_note_token("vl2"), None);
        assert_eq!(parse_note_token("rr3"), None);
        // Instrument-name fragments must not parse either.
        assert_eq!(parse_note_token("KSHarp"), None);
        assert_eq!(parse_note_token("Bark1"), None);
        assert_eq!(parse_note_token("GPiano"), None);
    }

    #[test]
    fn dynamic_tokens_order_sensibly() {
        let s = |t: &str| match parse_dynamic_token(t) {
            Some(Dynamic::Strength(s)) => s,
            _ => panic!("expected strength for {t}"),
        };
        assert!(s("pp") < s("p"));
        assert!(s("p") < s("mf1"));
        assert!(s("mf1") < s("f1"));
        assert!(s("f2") < s("loud"));
        assert!(matches!(
            parse_dynamic_token("vl3"),
            Some(Dynamic::Layer(3))
        ));
    }

    /// The on-disk library (9.6 GB, machine-local, not in git). Integration
    /// tests skip gracefully when absent so CI stays green without it.
    fn local_library() -> Option<VcslLibrary> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../data/samples/vcsl");
        VcslLibrary::load(&root)
    }

    #[test]
    fn indexes_the_real_library_when_present() {
        let Some(lib) = local_library() else {
            eprintln!("VCSL not on disk — skipping");
            return;
        };
        // At minimum the flagship banks must index.
        assert!(lib.bank(Instrument::Piano).is_some(), "Steinway must index");
        assert!(lib.bank(Instrument::Harp).is_some(), "harp must index");
        assert!(lib.instrument_count() >= 5, "most mapped banks should load");
    }

    #[test]
    fn vsco2_lead_instruments_index_when_present() {
        let vsco2 = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../data/samples/vsco2-ce");
        if !vsco2.is_dir() {
            eprintln!("VSCO2-CE not on disk — skipping");
            return;
        }
        let Some(lib) = local_library() else {
            eprintln!("VCSL not on disk — skipping");
            return;
        };
        for instrument in [
            Instrument::Violin,
            Instrument::Viola,
            Instrument::Cello,
            Instrument::Flute,
            Instrument::Clarinet,
            Instrument::Oboe,
            Instrument::Bassoon,
            Instrument::Trumpet,
            Instrument::FrenchHorn,
        ] {
            assert!(
                lib.bank(instrument).is_some(),
                "{instrument:?} must index from VSCO2"
            );
        }
        // A violin A4 must pick and decode real audio.
        let s = lib
            .bank(Instrument::Violin)
            .unwrap()
            .pick(69.0, 0.8, 3)
            .expect("violin near A4");
        assert!((s.midi as f32 - 69.0).abs() <= 7.0);
        assert!(!s.frames.is_empty());
        assert!(s.frames.iter().all(|x| x.is_finite()));
        // A viola A3 must also pick and decode real audio (susvib bank).
        let v = lib
            .bank(Instrument::Viola)
            .unwrap()
            .pick(57.0, 0.8, 1)
            .expect("viola near A3");
        assert!((v.midi as f32 - 57.0).abs() <= 7.0);
        assert!(!v.frames.is_empty());
        assert!(v.frames.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn vibraphone_indexes_from_vcsl_when_present() {
        let Some(lib) = local_library() else {
            eprintln!("VCSL not on disk — skipping");
            return;
        };
        let Some(bank) = lib.bank(Instrument::Vibraphone) else {
            eprintln!("Vibraphone bank not indexed — skipping");
            return;
        };
        let s = bank.pick(69.0, 0.8, 1).expect("vibraphone near A4");
        assert!(!s.frames.is_empty());
        assert!(s.frames.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn timpani_deliberately_has_no_bank_despite_real_samples_on_disk() {
        // Both libraries genuinely ship timpani hits, but their filenames
        // encode a drum number (Timpani1/2/3...), not a scientific pitch —
        // this module's note parser has nothing to extract, so mapping
        // Timpani here would be dishonest. See the module doc comment.
        let Some(lib) = local_library() else {
            eprintln!("VCSL not on disk — skipping");
            return;
        };
        assert!(lib.bank(Instrument::Timpani).is_none());
    }

    #[test]
    fn wave2_instruments_deliberately_have_no_bank_pending_disk_verification() {
        // Harpsichord/Tuba/Trombone/Duduk/ChoirOoh/Xylophone were added
        // while neither `data/samples/vcsl` nor `data/samples/vsco2-ce`
        // existed on disk in this environment, and this module's own doc
        // comments didn't already establish real subdirectories for any of
        // them (unlike the Viola/Oboe/Bassoon/French Horn/Vibraphone
        // additions, which WERE verified before mapping) — so, per the
        // Timpani precedent, they stay unmapped rather than guessed. This
        // test still runs (and would still pass) if the library later
        // appears on disk, since these instruments simply aren't in
        // `sample_subdir`/`staccato_subdir` at all.
        let Some(lib) = local_library() else {
            eprintln!("VCSL not on disk — skipping");
            return;
        };
        for instrument in [
            Instrument::Harpsichord,
            Instrument::Tuba,
            Instrument::Trombone,
            Instrument::Duduk,
            Instrument::ChoirOoh,
            Instrument::Xylophone,
        ] {
            assert!(
                lib.bank(instrument).is_none(),
                "{instrument:?} deliberately unmapped pending real on-disk verification"
            );
        }
    }

    #[test]
    fn staccato_banks_index_for_the_lead_instruments() {
        let vsco2 = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../data/samples/vsco2-ce");
        if !vsco2.is_dir() {
            eprintln!("VSCO2-CE not on disk — skipping");
            return;
        }
        let Some(lib) = local_library() else {
            eprintln!("VCSL not on disk — skipping");
            return;
        };
        for instrument in [
            Instrument::Violin,
            Instrument::Viola,
            Instrument::Cello,
            Instrument::Flute,
            Instrument::Clarinet,
            Instrument::Oboe,
            Instrument::Bassoon,
            Instrument::Trumpet,
            Instrument::FrenchHorn,
        ] {
            let bank = lib
                .staccato_bank(instrument)
                .unwrap_or_else(|| panic!("{instrument:?} staccato bank must index"));
            let s = bank
                .pick(60.0, 0.8, 1)
                .unwrap_or_else(|| panic!("{instrument:?} staccato C4-ish pick"));
            assert!(!s.frames.is_empty());
        }
        // Instruments without a real short articulation honestly have none.
        assert!(lib.staccato_bank(Instrument::Piano).is_none());
    }

    #[test]
    fn relaxed_window_covers_what_the_strict_window_refuses() {
        let Some(lib) = local_library() else {
            eprintln!("VCSL not on disk — skipping");
            return;
        };
        // The cello-section susvib bank tops out at D4 (MIDI 62). B4 (71)
        // is refused by the strict window but MUST be reachable via the
        // relaxed fallback the renderer uses — this exact gap sent the
        // counter-melody's top notes to additive synthesis mid-line.
        let vsco2 = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../data/samples/vsco2-ce");
        if !vsco2.is_dir() {
            eprintln!("VSCO2-CE not on disk — skipping");
            return;
        }
        let bank = lib.bank(Instrument::Cello).unwrap();
        assert!(bank.pick(71.0, 0.6, 1).is_none(), "strict should refuse +9");
        let relaxed = bank
            .pick_with_window(71.0, 0.6, 1, 12.0, 24.0)
            .expect("relaxed window must cover a +9 up-shift");
        assert!(!relaxed.frames.is_empty());
        // And the relaxed window still refuses truly unreachable pitches.
        assert!(bank.pick_with_window(120.0, 0.6, 1, 12.0, 24.0).is_none());
    }

    #[test]
    fn picks_decode_real_audio_near_the_requested_pitch() {
        let Some(lib) = local_library() else {
            eprintln!("VCSL not on disk — skipping");
            return;
        };
        let bank = lib.bank(Instrument::Harp).unwrap();
        let sample = bank.pick(60.0, 0.8, 7).expect("C4 harp sample");
        assert!((sample.midi as f32 - 60.0).abs() <= 7.0);
        assert!(!sample.frames.is_empty());
        assert!(sample.frames.iter().all(|s| s.is_finite()));
        assert!(sample.sample_rate >= 22050);
        // Round-robin/dynamic selection must be deterministic per seed.
        let again = bank.pick(60.0, 0.8, 7).unwrap();
        assert!(Arc::ptr_eq(&sample, &again));
    }

    #[test]
    fn inventory_separates_declared_mappings_from_missing_assets() {
        let root = std::env::temp_dir().join(format!(
            "muse-instrument-inventory-missing-{}",
            std::process::id()
        ));
        let vcsl_root = root.join("vcsl-missing");
        let vsco2_root = root.join("vsco2-missing");
        let first = capability_inventory(&vcsl_root, &vsco2_root, None).unwrap();
        let second = capability_inventory(&vcsl_root, &vsco2_root, None).unwrap();

        assert_eq!(first.instrument_count, 42);
        let guitar = first
            .instruments
            .iter()
            .find(|item| item.instrument == "AcousticGuitar")
            .unwrap();
        assert_eq!(guitar.procedural_renderer_path, "karplus_strong");
        let xylophone = first
            .instruments
            .iter()
            .find(|item| item.instrument == "Xylophone")
            .unwrap();
        assert_eq!(xylophone.procedural_renderer_path, "frequency_modulation");
        let violin = first
            .instruments
            .iter()
            .find(|item| item.instrument == "Violin")
            .unwrap();
        assert_eq!(violin.procedural_renderer_path, "additive");
        assert!(!first.vcsl_root.root_present);
        assert!(!first.vsco2_root.root_present);
        assert!(
            first.fluidsynth_probe.is_none(),
            "the default inventory must not start external FluidSynth processes"
        );
        assert_eq!(
            serde_json::to_vec(&first).unwrap(),
            serde_json::to_vec(&second).unwrap(),
            "the same source and asset roots must produce identical JSON"
        );

        let violin = first
            .instruments
            .iter()
            .find(|item| item.instrument == "Violin")
            .unwrap();
        let mapping = violin.sustained_samples.as_ref().unwrap();
        assert_eq!(mapping.source_library, "vsco2_ce");
        assert!(!mapping.directory_present);
        assert_eq!(mapping.indexed_sample_count, 0);
        assert_eq!(mapping.audio_decode_status, "mapped_directory_missing");
        assert_eq!(mapping.decoded_sample_file_count, 0);
        assert_eq!(mapping.failed_sample_decode_count, 0);

        let timpani = first
            .instruments
            .iter()
            .find(|item| item.instrument == "Timpani")
            .unwrap();
        assert!(
            timpani.sustained_samples.is_none(),
            "unmapped assets must not be inferred from the instrument name"
        );
    }

    fn soundfont_fixture(presets: &[(&str, u16, u16)]) -> Vec<u8> {
        fn append_record(
            output: &mut Vec<u8>,
            name: &str,
            program: u16,
            bank: u16,
            bag_index: u16,
        ) {
            let mut record = [0u8; 38];
            let name_bytes = name.as_bytes();
            let copy_len = name_bytes.len().min(20);
            record[..copy_len].copy_from_slice(&name_bytes[..copy_len]);
            record[20..22].copy_from_slice(&program.to_le_bytes());
            record[22..24].copy_from_slice(&bank.to_le_bytes());
            record[24..26].copy_from_slice(&bag_index.to_le_bytes());
            output.extend_from_slice(&record);
        }

        let mut phdr = Vec::new();
        for (index, (name, bank, program)) in presets.iter().enumerate() {
            append_record(&mut phdr, name, *program, *bank, index as u16);
        }
        append_record(&mut phdr, "EOP", 0, 0, presets.len() as u16);

        let mut pdta_list = b"pdta".to_vec();
        pdta_list.extend_from_slice(b"phdr");
        pdta_list.extend_from_slice(&(phdr.len() as u32).to_le_bytes());
        pdta_list.extend_from_slice(&phdr);

        let mut riff_payload = b"sfbk".to_vec();
        riff_payload.extend_from_slice(b"LIST");
        riff_payload.extend_from_slice(&(pdta_list.len() as u32).to_le_bytes());
        riff_payload.extend_from_slice(&pdta_list);

        let mut file = b"RIFF".to_vec();
        file.extend_from_slice(&(riff_payload.len() as u32).to_le_bytes());
        file.extend_from_slice(&riff_payload);
        file
    }

    #[test]
    fn probe_midi_encodes_requested_program_and_one_note() {
        let midi = probe_midi_bytes(40, 60, 96);
        assert_eq!(&midi[0..4], b"MThd");
        assert_eq!(u32::from_be_bytes(midi[4..8].try_into().unwrap()), 6);
        assert_eq!(u16::from_be_bytes(midi[8..10].try_into().unwrap()), 0);
        assert_eq!(u16::from_be_bytes(midi[10..12].try_into().unwrap()), 1);
        assert_eq!(u16::from_be_bytes(midi[12..14].try_into().unwrap()), 480);
        assert_eq!(&midi[14..18], b"MTrk");
        assert_eq!(u32::from_be_bytes(midi[18..22].try_into().unwrap()), 25);
        assert_eq!(&midi[23..26], &[0xB0, 0x00, 0x00], "bank MSB is zero");
        assert_eq!(&midi[27..30], &[0xB0, 0x20, 0x00], "bank LSB is zero");
        assert_eq!(midi[32], 40, "program change event uses requested GM program");
        assert_eq!(midi[35], 60, "probe note is middle C");
        assert_eq!(midi[36], 96, "probe velocity is fixed");
        assert_eq!(&midi[midi.len() - 3..], &[0xFF, 0x2F, 0x00]);
    }

    #[test]
    fn fluidsynth_probe_requires_explicit_soundfont() {
        let error = capability_inventory_with_fluidsynth(
            Path::new("/missing-vcsl"),
            Path::new("/missing-vsco2"),
            None,
            Some(Path::new("fluidsynth")),
        )
        .unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("--soundfont"));
    }

    #[cfg(unix)]
    #[test]
    fn runtime_probe_records_non_silent_output_and_skips_missing_header_matches() {
        use std::os::unix::fs::PermissionsExt;

        let root = std::env::temp_dir().join(format!(
            "muse-instrument-runtime-probe-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let soundfont = root.join("fixture.sf2");
        std::fs::write(&soundfont, soundfont_fixture(&[("Grand Piano", 0, 0)])).unwrap();
        let evidence = soundfont_evidence(&soundfont).unwrap();

        let output_fixture = root.join("non-silent.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 22050,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&output_fixture, spec).unwrap();
        for index in 0..256 {
            writer.write_sample(if index % 2 == 0 { 8000i16 } else { -8000i16 }).unwrap();
        }
        writer.finalize().unwrap();

        // Isolated fake CLI: copy a valid WAV to the path supplied after -F.
        // This exercises subprocess argument construction and output auditing,
        // without pretending CI has a real SoundFont or installed FluidSynth.
        let fake_binary = root.join("fake-fluidsynth");
        let script = format!(
            "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo 'fake-fluidsynth fixture 1'; exit 0; fi\nout=''\nwhile [ \"$#\" -gt 0 ]; do\n  if [ \"$1\" = \"-F\" ]; then shift; out=\"$1\"; fi\n  shift\ndone\ncp \"{}\" \"$out\"\n",
            output_fixture.display()
        );
        std::fs::write(&fake_binary, script).unwrap();
        let mut permissions = std::fs::metadata(&fake_binary).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&fake_binary, permissions).unwrap();

        let probe = probe_fluidsynth_programs(
            &fake_binary,
            &soundfont,
            &evidence.sha256,
            &evidence.preset_inventory,
            [0u8, 24u8].into_iter(),
        )
        .unwrap();

        assert_eq!(probe.status, "fluidsynth_render_probes_completed");
        assert_eq!(probe.binary_name, "fake-fluidsynth");
        assert_eq!(probe.binary_version.as_deref(), Some("fake-fluidsynth fixture 1"));
        assert_eq!(probe.binary_sha256, sha256_file(&fake_binary).unwrap());
        assert_eq!(probe.soundfont_sha256, evidence.sha256);
        assert_eq!(probe.probe_notes_midi, vec![48, 60, 72]);
        assert_eq!(probe.probe_velocities, vec![48, 96, 120]);
        assert_eq!(probe.sample_rate_hz, 22050);
        assert!(probe.reverb_disabled && probe.chorus_disabled);
        assert!(probe.command_template.contains("<probe.mid>"));
        assert_eq!(probe.requested_program_count, 2);
        assert_eq!(probe.probe_count, 18);
        assert_eq!(probe.programs.len(), 18);
        assert_eq!(probe.rendered_non_silent_probe_count, 9);
        assert_eq!(probe.programs[0].program_0_based, 0);
        assert_eq!(probe.programs[0].note_midi, 48);
        assert_eq!(probe.programs[0].velocity, 48);
        assert_eq!(probe.programs[0].status, "rendered_non_silent");
        assert_eq!(probe.programs[0].sample_rate, Some(22050));
        assert_eq!(probe.programs[0].channels, Some(1));
        assert!(probe.programs[0].frames.unwrap() > 0);
        assert!(probe.programs[0].peak_absolute_sample.unwrap() > 1e-5);
        assert!(probe.programs[0].rms.unwrap() > 0.0);
        let expected_output_hash = sha256_file(&output_fixture).unwrap();
        assert_eq!(
            probe.programs[0].output_wav_sha256.as_deref(),
            Some(expected_output_hash.as_str()),
            "receipt preserves the exact rendered WAV bytes"
        );
        let first_missing_slot = probe.programs.iter()
            .find(|item| item.program_0_based == 24)
            .unwrap();
        assert_eq!(first_missing_slot.note_midi, 48);
        assert_eq!(first_missing_slot.velocity, 48);
        assert_eq!(first_missing_slot.status, "skipped_no_bank0_header_match");
        assert_eq!(first_missing_slot.command_exit_code, None);
        assert!(
            probe.programs.iter().filter(|item| item.program_0_based == 24)
                .all(|item| item.status == "skipped_no_bank0_header_match")
        );

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn invalid_soundfont_is_rejected_instead_of_reporting_no_presets() {
        let root = std::env::temp_dir().join(format!(
            "muse-instrument-inventory-invalid-sf2-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let soundfont = root.join("not-a-soundfont.sf2");
        std::fs::write(&soundfont, b"not a RIFF soundfont").unwrap();

        let error = soundfont_evidence(&soundfont).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn inventory_counts_only_files_accepted_by_the_declared_indexer() {
        let root = std::env::temp_dir().join(format!(
            "muse-instrument-inventory-present-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let vcsl_root = root.join("vcsl");
        let vsco2_root = root.join("vsco2-ce");
        let harp_dir =
            vcsl_root.join("Chordophones/Composite Chordophones/Concert Harp");
        std::fs::create_dir_all(&harp_dir).unwrap();
        fn write_fixture_wav(path: &Path, amplitude: i16) {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 22050,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::create(path, spec).unwrap();
            for index in 0..256 {
                let sample = if amplitude == 0 {
                    0
                } else if index % 2 == 0 {
                    amplitude
                } else {
                    -amplitude
                };
                writer.write_sample(sample).unwrap();
            }
            writer.finalize().unwrap();
        }
        write_fixture_wav(&harp_dir.join("Harp_A4_vl1.wav"), 8000);
        write_fixture_wav(&harp_dir.join("Harp_C3_vl2.wav"), 0);
        std::fs::write(harp_dir.join("Harp_D4_vl3.wav"), b"truncated-not-wav").unwrap();
        std::fs::write(harp_dir.join("not-a-pitch-file.wav"), []).unwrap();
        std::fs::create_dir_all(&vsco2_root).unwrap();
        let soundfont = root.join("fixture.sf2");
        let soundfont_bytes = soundfont_fixture(&[
            ("Grand Piano", 0, 0),
            ("Violin", 0, 40),
            ("Alternate Violin", 0, 40),
            ("Percussion", 128, 0),
        ]);
        std::fs::write(&soundfont, &soundfont_bytes).unwrap();

        let inventory = capability_inventory(&vcsl_root, &vsco2_root, Some(&soundfont)).unwrap();
        assert!(inventory.vcsl_root.root_present);
        assert_eq!(inventory.vcsl_root.wav_file_count, 4);
        assert!(inventory.vcsl_root.path_size_manifest_sha256.is_some());
        assert!(inventory.vsco2_root.root_present);
        assert_eq!(inventory.vsco2_root.wav_file_count, 0);

        let harp = inventory
            .instruments
            .iter()
            .find(|item| item.instrument == "Harp")
            .unwrap();
        let mapping = harp.sustained_samples.as_ref().unwrap();
        assert!(mapping.directory_present);
        assert_eq!(mapping.indexed_sample_count, 3);
        assert_eq!(mapping.min_midi, Some(48));
        assert_eq!(mapping.max_midi, Some(69));
        assert_eq!(mapping.distinct_pitch_count, 3);
        assert_eq!(mapping.distinct_dynamic_strength_count, 3);
        assert_eq!(mapping.audio_decode_status, "some_indexed_files_failed_decode");
        assert_eq!(mapping.decoded_sample_file_count, 2);
        assert_eq!(mapping.failed_sample_decode_count, 1);
        assert_eq!(mapping.silent_sample_file_count, 1);
        assert_eq!(mapping.distinct_sample_rate_count, 1);
        assert_eq!(mapping.distinct_channel_count, 1);

        let soundfont_evidence = inventory.soundfont.as_ref().unwrap();
        assert_eq!(soundfont_evidence.file_name, "fixture.sf2");
        assert_eq!(soundfont_evidence.size_bytes, soundfont_bytes.len() as u64);
        let expected_digest = Sha256::digest(&soundfont_bytes);
        assert_eq!(
            soundfont_evidence.sha256,
            hex_digest(&expected_digest[..])
        );
        let presets = &soundfont_evidence.preset_inventory;
        assert_eq!(presets.status, "preset_headers_parsed_not_renderability");
        assert_eq!(presets.declared_preset_count, 4);
        assert_eq!(presets.declared_bank_count, 2);
        assert_eq!(presets.duplicate_bank_program_slots, 1);
        assert_eq!(presets.presets[0].name, "Grand Piano");
        assert_eq!(presets.presets[0].bank, 0);
        assert_eq!(presets.presets[0].program, 0);
        assert!(presets.presets.iter().any(|preset| {
            preset.name == "Violin" && preset.bank == 0 && preset.program == 40
        }));
        assert!(presets.presets.iter().any(|preset| {
            preset.name == "Percussion" && preset.bank == 128 && preset.program == 0
        }));
        let piano = inventory
            .instruments
            .iter()
            .find(|item| item.instrument == "Piano")
            .unwrap();
        assert_eq!(
            piano.soundfont_bank0_program_matches.as_deref(),
            Some(&["Grand Piano".to_owned()][..])
        );
        let violin = inventory
            .instruments
            .iter()
            .find(|item| item.instrument == "Violin")
            .unwrap();
        assert_eq!(
            violin.soundfont_bank0_program_matches.as_deref(),
            Some(&["Alternate Violin".to_owned(), "Violin".to_owned()][..])
        );
        let xylophone = inventory
            .instruments
            .iter()
            .find(|item| item.instrument == "Xylophone")
            .unwrap();
        assert_eq!(
            xylophone.soundfont_bank0_program_matches.as_deref(),
            Some(&[][..]),
            "a missing program is different from not inspecting a SoundFont"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

}
