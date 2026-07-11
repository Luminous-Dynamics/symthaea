// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sample-based synthesis: load WAV samples and pitch-shift for playback.
//!
//! Replaces thin additive synthesis with rich recorded instrument samples.
//! Loads WAV files via hound, pitch-shifts via linear interpolation, and
//! blends with the existing synthesis for a richer sound.
//!
//! Supports loading from a samples directory or generating synthetic
//! reference samples from the existing high-quality additive engine.

use crate::instruments::Instrument;
use std::collections::HashMap;
use std::path::Path;

/// A loaded audio sample.
#[derive(Debug, Clone)]
pub struct Sample {
    /// Mono f32 samples.
    pub data: Vec<f32>,
    /// Sample rate of the recording.
    pub sample_rate: u32,
    /// MIDI note this sample was recorded at.
    pub root_note: u8,
}

/// Sample library: maps (instrument, midi_note_region) → sample.
pub struct SampleLibrary {
    samples: HashMap<(InstrumentKey, u8), Sample>,
}

/// Simplified instrument key for sample mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InstrumentKey {
    Piano,
    Strings, // violin, cello
    Wind,    // flute
    Plucked, // guitar, harp
    Keys,    // e-piano, bell
    Pad,
}

impl InstrumentKey {
    pub fn from_instrument(inst: Instrument) -> Self {
        match inst {
            Instrument::Piano | Instrument::PianoPP => Self::Piano,
            Instrument::Violin | Instrument::Cello => Self::Strings,
            Instrument::Flute
            | Instrument::Ney
            | Instrument::Clarinet
            | Instrument::Trumpet
            | Instrument::Saxophone => Self::Wind,
            Instrument::AcousticGuitar
            | Instrument::Harp
            | Instrument::Koto
            | Instrument::Oud
            | Instrument::Sitar
            | Instrument::UprightBass => Self::Plucked,
            Instrument::ElectricPiano
            | Instrument::Bell
            | Instrument::Marimba
            | Instrument::Kalimba => Self::Keys,
            Instrument::Pad | Instrument::Organ | Instrument::SawLead => Self::Pad,
        }
    }
}

impl SampleLibrary {
    pub fn new() -> Self {
        Self {
            samples: HashMap::new(),
        }
    }

    /// Load samples from a directory.
    /// Expected structure: samples/piano_c4.wav, samples/violin_a3.wav, etc.
    pub fn load_directory(&mut self, dir: &Path) -> usize {
        let mut count = 0;
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return 0,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "wav").unwrap_or(false) {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if let Some((key, note)) = parse_sample_name(stem) {
                        if let Ok(sample) = load_wav(&path, note) {
                            self.samples.insert((key, note), sample);
                            count += 1;
                        }
                    }
                }
            }
        }
        count
    }

    /// Resolve the nearest sample's map key + playback rate for a note.
    ///
    /// This performs the O(N) nearest-note scan over the library. Call it
    /// ONCE per note (at spawn) and cache the result; then fetch per audio
    /// sample with `get_by_key`, which is a single O(1) hash lookup. A
    /// previous version ran this scan once per output sample per active note
    /// (tens of millions of map iterations per second at 44.1 kHz), directly
    /// threatening real-time underruns.
    pub fn resolve_sample(
        &self,
        instrument: Instrument,
        freq: f32,
    ) -> Option<((InstrumentKey, u8), f64)> {
        let key = InstrumentKey::from_instrument(instrument);
        let target_note = freq_to_midi(freq);

        // Find nearest available sample note for this instrument family
        let mut best: Option<u8> = None;
        for &(k, note) in self.samples.keys() {
            if k == key {
                best = match best {
                    Some(b)
                        if (b as i16 - target_note as i16).abs()
                            <= (note as i16 - target_note as i16).abs() =>
                    {
                        Some(b)
                    }
                    _ => Some(note),
                };
            }
        }

        best.map(|root| {
            let playback_rate = 2.0f64.powf((target_note as f64 - root as f64) / 12.0);
            ((key, root), playback_rate)
        })
    }

    /// O(1) fetch by a key previously resolved with `resolve_sample`.
    pub fn get_by_key(&self, key: (InstrumentKey, u8)) -> Option<&Sample> {
        self.samples.get(&key)
    }

    /// Get the nearest sample for an instrument and frequency.
    ///
    /// Convenience combining `resolve_sample` + `get_by_key`. Do NOT call in
    /// per-sample loops — resolve once and use `get_by_key` there.
    pub fn get_sample(&self, instrument: Instrument, freq: f32) -> Option<(&Sample, f64)> {
        self.resolve_sample(instrument, freq)
            .and_then(|(key, rate)| self.samples.get(&key).map(|s| (s, rate)))
    }

    /// Generate synthetic reference samples from the additive engine.
    /// Creates one sample per octave per instrument family.
    pub fn generate_synthetic(&mut self, sample_rate: u32) {
        let families = [
            (InstrumentKey::Piano, Instrument::Piano),
            (InstrumentKey::Strings, Instrument::Violin),
            (InstrumentKey::Wind, Instrument::Flute),
            (InstrumentKey::Plucked, Instrument::AcousticGuitar),
            (InstrumentKey::Keys, Instrument::ElectricPiano),
            (InstrumentKey::Pad, Instrument::Pad),
        ];

        // Generate at C2, C3, C4, C5, C6
        let notes: [(u8, f32); 5] = [
            (36, 65.41),   // C2
            (48, 130.81),  // C3
            (60, 261.63),  // C4
            (72, 523.25),  // C5
            (84, 1046.50), // C6
        ];

        let duration_secs = 2.0;
        let num_samples = (duration_secs * sample_rate as f32) as usize;

        for &(key, instrument) in &families {
            let partials = instrument.partials();

            for &(midi_note, freq) in &notes {
                let mut data = Vec::with_capacity(num_samples);
                let sr = sample_rate as f32;

                for i in 0..num_samples {
                    let t = i as f32 / sr;
                    // ADSR envelope
                    let env = if t < 0.01 {
                        t / 0.01
                    } else if t < 0.1 {
                        1.0 - (t - 0.01) * 0.3 / 0.09
                    } else if t < duration_secs - 0.3 {
                        0.7
                    } else {
                        0.7 * (1.0 - (t - (duration_secs - 0.3)) / 0.3).max(0.0)
                    };

                    // Additive synthesis with all partials
                    let mut s = 0.0f32;
                    for (h, &amp) in partials.iter().enumerate() {
                        let cf = freq * (h + 1) as f32;
                        if cf >= sr * 0.5 {
                            break;
                        }
                        let phase = t * cf * std::f32::consts::TAU;
                        s += amp * phase.sin();
                    }

                    // Normalize
                    let partial_sum: f32 = partials.iter().sum();
                    let norm = if partial_sum > 0.01 {
                        1.0 / partial_sum
                    } else {
                        1.0
                    };

                    data.push(s * norm * env * 0.8);
                }

                self.samples.insert(
                    (key, midi_note),
                    Sample {
                        data,
                        sample_rate,
                        root_note: midi_note,
                    },
                );
            }
        }
    }

    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }
    pub fn has_samples(&self) -> bool {
        !self.samples.is_empty()
    }
}

/// Render a sample at a given playback rate (pitch shift).
/// Returns a mono buffer of the requested length.
pub fn render_sample(
    sample: &Sample,
    playback_rate: f64,
    output_len: usize,
    velocity: f32,
) -> Vec<f32> {
    let mut output = Vec::with_capacity(output_len);
    let mut pos = 0.0f64;

    for _ in 0..output_len {
        let idx = pos as usize;
        if idx + 1 >= sample.data.len() {
            break;
        }

        // Linear interpolation
        let frac = (pos - idx as f64) as f32;
        let s = sample.data[idx] * (1.0 - frac) + sample.data[idx + 1] * frac;
        output.push(s * velocity);

        pos += playback_rate;
    }

    // Pad with zeros if sample was too short
    output.resize(output_len, 0.0);
    output
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn freq_to_midi(freq: f32) -> u8 {
    ((12.0 * (freq / 440.0).log2() + 69.0).round() as u8).clamp(0, 127)
}

fn midi_to_freq(note: u8) -> f32 {
    440.0 * 2.0f32.powf((note as f32 - 69.0) / 12.0)
}

fn parse_sample_name(name: &str) -> Option<(InstrumentKey, u8)> {
    // Expected format: "piano_c4", "violin_a3", etc.
    let parts: Vec<&str> = name.split('_').collect();
    if parts.len() < 2 {
        return None;
    }

    let key = match parts[0].to_lowercase().as_str() {
        "piano" => InstrumentKey::Piano,
        "violin" | "cello" | "strings" => InstrumentKey::Strings,
        "flute" | "wind" => InstrumentKey::Wind,
        "guitar" | "harp" => InstrumentKey::Plucked,
        "epiano" | "bell" | "keys" => InstrumentKey::Keys,
        "pad" | "organ" => InstrumentKey::Pad,
        _ => return None,
    };

    // Parse note name: c4, a3, etc.
    let note_str = parts[1].to_lowercase();
    let note = parse_note_name(&note_str)?;
    Some((key, note))
}

fn parse_note_name(s: &str) -> Option<u8> {
    let bytes = s.as_bytes();
    if bytes.len() < 2 {
        return None;
    }

    let pitch_class = match bytes[0] {
        b'c' => 0,
        b'd' => 2,
        b'e' => 4,
        b'f' => 5,
        b'g' => 7,
        b'a' => 9,
        b'b' => 11,
        _ => return None,
    };

    let octave = (bytes[bytes.len() - 1] as char).to_digit(10)? as u8;
    Some((octave + 1) * 12 + pitch_class)
}

fn load_wav(path: &Path, expected_note: u8) -> Result<Sample, String> {
    let reader = hound::WavReader::open(path).map_err(|e| format!("WAV read error: {e}"))?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    let data: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max)
                .collect()
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .filter_map(|s| s.ok())
            .collect(),
    };

    // If stereo, take left channel
    let mono = if spec.channels == 2 {
        data.iter().step_by(2).copied().collect()
    } else {
        data
    };

    Ok(Sample {
        data: mono,
        sample_rate,
        root_note: expected_note,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freq_to_midi_a4() {
        assert_eq!(freq_to_midi(440.0), 69);
    }

    #[test]
    fn freq_to_midi_c4() {
        assert_eq!(freq_to_midi(261.63), 60);
    }

    #[test]
    fn parse_note_name_c4() {
        assert_eq!(parse_note_name("c4"), Some(60));
    }

    #[test]
    fn parse_sample_name_piano() {
        let (key, note) = parse_sample_name("piano_c4").unwrap();
        assert_eq!(key, InstrumentKey::Piano);
        assert_eq!(note, 60);
    }

    #[test]
    fn synthetic_samples_generate() {
        let mut lib = SampleLibrary::new();
        lib.generate_synthetic(44100);
        assert!(
            lib.sample_count() >= 20,
            "should generate samples: {}",
            lib.sample_count()
        );
    }

    #[test]
    fn get_sample_finds_nearest() {
        let mut lib = SampleLibrary::new();
        lib.generate_synthetic(44100);
        let result = lib.get_sample(Instrument::Piano, 440.0);
        assert!(result.is_some(), "should find piano sample for A4");
        let (_, rate) = result.unwrap();
        assert!(
            rate > 0.5 && rate < 2.0,
            "playback rate should be reasonable: {rate}"
        );
    }

    #[test]
    fn render_sample_correct_length() {
        let sample = Sample {
            data: (0..44100)
                .map(|i| (i as f32 / 44100.0 * 440.0 * std::f32::consts::TAU).sin())
                .collect(),
            sample_rate: 44100,
            root_note: 69,
        };
        let output = render_sample(&sample, 1.0, 1000, 0.7);
        assert_eq!(output.len(), 1000);
    }
}
