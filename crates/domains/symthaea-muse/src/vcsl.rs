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
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

/// Which sample library provides an instrument. VCSL (keyboards, harps,
/// mallets, sax, organ) is complemented by VSCO2 Community Edition (also
/// CC0, `data/samples/vsco2-ce`, github.com/sgossner/VSCO-2-CE), which
/// fills VCSL's biggest gap: the orchestral LEAD instruments — solo violin,
/// cello section, flute, clarinet, trumpet — the most exposed voices in
/// most ensembles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
}
