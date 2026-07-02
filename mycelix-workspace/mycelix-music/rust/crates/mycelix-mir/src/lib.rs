// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Music Information Retrieval (MIR)
//!
//! Advanced music analysis: chord detection, beat tracking,
//! structure segmentation, and source separation.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use thiserror::Error;

pub mod beat;
pub mod chord;
pub mod structure;
pub mod separation;

// Re-export commonly used types
pub use beat::BeatTracker;
pub use chord::ChordDetector;
pub use structure::StructureAnalyzer;

#[derive(Error, Debug)]
pub enum MirError {
    #[error("Analysis error: {0}")]
    AnalysisError(String),

    #[error("Invalid audio data: {0}")]
    InvalidAudio(String),

    #[error("Model error: {0}")]
    ModelError(String),

    #[error("Buffer too small: need {needed}, got {actual}")]
    BufferTooSmall { needed: usize, actual: usize },
}

pub type Result<T> = std::result::Result<T, MirError>;

/// Musical note names
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NoteName {
    C, Cs, D, Ds, E, F, Fs, G, Gs, A, As, B,
}

impl NoteName {
    pub fn from_pitch_class(pc: u8) -> Self {
        match pc % 12 {
            0 => NoteName::C,
            1 => NoteName::Cs,
            2 => NoteName::D,
            3 => NoteName::Ds,
            4 => NoteName::E,
            5 => NoteName::F,
            6 => NoteName::Fs,
            7 => NoteName::G,
            8 => NoteName::Gs,
            9 => NoteName::A,
            10 => NoteName::As,
            11 => NoteName::B,
            _ => unreachable!(),
        }
    }

    pub fn to_pitch_class(self) -> u8 {
        match self {
            NoteName::C => 0,
            NoteName::Cs => 1,
            NoteName::D => 2,
            NoteName::Ds => 3,
            NoteName::E => 4,
            NoteName::F => 5,
            NoteName::Fs => 6,
            NoteName::G => 7,
            NoteName::Gs => 8,
            NoteName::A => 9,
            NoteName::As => 10,
            NoteName::B => 11,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            NoteName::C => "C",
            NoteName::Cs => "C#",
            NoteName::D => "D",
            NoteName::Ds => "D#",
            NoteName::E => "E",
            NoteName::F => "F",
            NoteName::Fs => "F#",
            NoteName::G => "G",
            NoteName::Gs => "G#",
            NoteName::A => "A",
            NoteName::As => "A#",
            NoteName::B => "B",
        }
    }

    pub fn all() -> [NoteName; 12] {
        [
            NoteName::C, NoteName::Cs, NoteName::D, NoteName::Ds,
            NoteName::E, NoteName::F, NoteName::Fs, NoteName::G,
            NoteName::Gs, NoteName::A, NoteName::As, NoteName::B,
        ]
    }
}

/// Chord quality
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChordQuality {
    Major,
    Minor,
    Diminished,
    Augmented,
    Sus2,
    Sus4,
    Dominant7,
    Major7,
    Minor7,
    Diminished7,
    HalfDiminished7,
    Add9,
    Unknown,
}

impl ChordQuality {
    pub fn symbol(&self) -> &'static str {
        match self {
            ChordQuality::Major => "",
            ChordQuality::Minor => "m",
            ChordQuality::Diminished => "dim",
            ChordQuality::Augmented => "aug",
            ChordQuality::Sus2 => "sus2",
            ChordQuality::Sus4 => "sus4",
            ChordQuality::Dominant7 => "7",
            ChordQuality::Major7 => "maj7",
            ChordQuality::Minor7 => "m7",
            ChordQuality::Diminished7 => "dim7",
            ChordQuality::HalfDiminished7 => "m7b5",
            ChordQuality::Add9 => "add9",
            ChordQuality::Unknown => "?",
        }
    }

    /// Get intervals (semitones from root)
    pub fn intervals(&self) -> Vec<u8> {
        match self {
            ChordQuality::Major => vec![0, 4, 7],
            ChordQuality::Minor => vec![0, 3, 7],
            ChordQuality::Diminished => vec![0, 3, 6],
            ChordQuality::Augmented => vec![0, 4, 8],
            ChordQuality::Sus2 => vec![0, 2, 7],
            ChordQuality::Sus4 => vec![0, 5, 7],
            ChordQuality::Dominant7 => vec![0, 4, 7, 10],
            ChordQuality::Major7 => vec![0, 4, 7, 11],
            ChordQuality::Minor7 => vec![0, 3, 7, 10],
            ChordQuality::Diminished7 => vec![0, 3, 6, 9],
            ChordQuality::HalfDiminished7 => vec![0, 3, 6, 10],
            ChordQuality::Add9 => vec![0, 4, 7, 14],
            ChordQuality::Unknown => vec![0],
        }
    }
}

/// Chord representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chord {
    pub root: NoteName,
    pub quality: ChordQuality,
    pub bass: Option<NoteName>,
    pub confidence: f32,
}

impl Chord {
    pub fn new(root: NoteName, quality: ChordQuality) -> Self {
        Self {
            root,
            quality,
            bass: None,
            confidence: 1.0,
        }
    }

    pub fn with_bass(mut self, bass: NoteName) -> Self {
        self.bass = Some(bass);
        self
    }

    pub fn symbol(&self) -> String {
        let mut s = format!("{}{}", self.root.name(), self.quality.symbol());
        if let Some(bass) = self.bass {
            if bass != self.root {
                s.push('/');
                s.push_str(bass.name());
            }
        }
        s
    }

    /// Get pitch classes in the chord
    pub fn pitch_classes(&self) -> Vec<u8> {
        let root_pc = self.root.to_pitch_class();
        self.quality
            .intervals()
            .iter()
            .map(|&i| (root_pc + i) % 12)
            .collect()
    }
}

/// Beat information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Beat {
    /// Time in seconds
    pub time: f32,
    /// Beat strength (0-1)
    pub strength: f32,
    /// Beat number within bar (1-indexed)
    pub position: u8,
}

/// Bar/measure information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bar {
    /// Start time in seconds
    pub start: f32,
    /// End time in seconds
    pub end: f32,
    /// Time signature numerator
    pub beats_per_bar: u8,
    /// Beats in this bar
    pub beats: Vec<Beat>,
}

/// Musical key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Key {
    pub tonic: NoteName,
    pub is_major: bool,
    pub confidence: f32,
}

impl Key {
    pub fn symbol(&self) -> String {
        format!(
            "{} {}",
            self.tonic.name(),
            if self.is_major { "major" } else { "minor" }
        )
    }

    /// Get scale degrees
    pub fn scale(&self) -> Vec<u8> {
        let root = self.tonic.to_pitch_class();
        let intervals = if self.is_major {
            [0, 2, 4, 5, 7, 9, 11] // Major scale
        } else {
            [0, 2, 3, 5, 7, 8, 10] // Natural minor scale
        };
        intervals.iter().map(|&i| (root + i) % 12).collect()
    }
}

/// Segment type for structure analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SegmentType {
    Intro,
    Verse,
    PreChorus,
    Chorus,
    Bridge,
    Breakdown,
    Buildup,
    Drop,
    Outro,
    Instrumental,
    Unknown,
}

impl SegmentType {
    pub fn name(&self) -> &'static str {
        match self {
            SegmentType::Intro => "Intro",
            SegmentType::Verse => "Verse",
            SegmentType::PreChorus => "Pre-Chorus",
            SegmentType::Chorus => "Chorus",
            SegmentType::Bridge => "Bridge",
            SegmentType::Breakdown => "Breakdown",
            SegmentType::Buildup => "Buildup",
            SegmentType::Drop => "Drop",
            SegmentType::Outro => "Outro",
            SegmentType::Instrumental => "Instrumental",
            SegmentType::Unknown => "Unknown",
        }
    }
}

/// Song section/segment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub start: f32,
    pub end: f32,
    pub segment_type: SegmentType,
    pub label: String,
    pub confidence: f32,
}

/// Chromagram (pitch class distribution)
#[derive(Debug, Clone)]
pub struct Chromagram {
    data: Array2<f32>,
    hop_size: usize,
    sample_rate: u32,
}

impl Chromagram {
    pub fn new(num_frames: usize) -> Self {
        Self {
            data: Array2::zeros((12, num_frames)),
            hop_size: 512,
            sample_rate: 44100,
        }
    }

    /// Compute chromagram from audio
    pub fn from_audio(audio: &[f32], sample_rate: u32, hop_size: usize) -> Self {
        let frame_size = 4096;
        let num_frames = (audio.len() - frame_size) / hop_size + 1;
        let mut chroma = Self {
            data: Array2::zeros((12, num_frames.max(1))),
            hop_size,
            sample_rate,
        };

        // Process each frame
        for frame_idx in 0..num_frames {
            let start = frame_idx * hop_size;
            let end = (start + frame_size).min(audio.len());
            let frame = &audio[start..end];

            // Compute pitch class energies using FFT bins
            let pitch_energies = compute_pitch_class_energies(frame, sample_rate);

            for (pc, &energy) in pitch_energies.iter().enumerate() {
                chroma.data[[pc, frame_idx]] = energy;
            }
        }

        // Normalize each frame
        for frame_idx in 0..num_frames {
            let sum: f32 = (0..12).map(|pc| chroma.data[[pc, frame_idx]]).sum();
            if sum > 1e-6 {
                for pc in 0..12 {
                    chroma.data[[pc, frame_idx]] /= sum;
                }
            }
        }

        chroma
    }

    pub fn num_frames(&self) -> usize {
        self.data.ncols()
    }

    pub fn get_frame(&self, idx: usize) -> [f32; 12] {
        let mut frame = [0.0f32; 12];
        for (pc, f) in frame.iter_mut().enumerate() {
            *f = self.data[[pc, idx]];
        }
        frame
    }

    pub fn frame_to_time(&self, frame_idx: usize) -> f32 {
        (frame_idx * self.hop_size) as f32 / self.sample_rate as f32
    }
}

/// Compute pitch class energies from audio frame
fn compute_pitch_class_energies(frame: &[f32], sample_rate: u32) -> [f32; 12] {
    let mut energies = [0.0f32; 12];
    let n = frame.len();

    // Simple DFT for specific frequencies
    // In production, use FFT and map bins to pitch classes
    for octave in 1..=7 {
        for pc in 0..12 {
            // Frequency of this pitch class in this octave
            let midi_note = octave * 12 + pc as i32;
            let freq = 440.0 * 2.0_f32.powf((midi_note as f32 - 69.0) / 12.0);

            if freq > sample_rate as f32 / 2.0 {
                continue;
            }

            // Goertzel algorithm for single frequency
            let k = (freq * n as f32 / sample_rate as f32).round() as usize;
            if k == 0 || k >= n / 2 {
                continue;
            }

            let omega = 2.0 * PI * k as f32 / n as f32;
            let coeff = 2.0 * omega.cos();

            let mut s0 = 0.0;
            let mut s1 = 0.0;
            let mut s2;

            for &sample in frame {
                s2 = s1;
                s1 = s0;
                s0 = sample + coeff * s1 - s2;
            }

            let power = s0 * s0 + s1 * s1 - coeff * s0 * s1;
            energies[pc as usize] += power.max(0.0).sqrt();
        }
    }

    energies
}

/// Main MIR analyzer
pub struct MirAnalyzer {
    sample_rate: u32,
    hop_size: usize,
}

impl MirAnalyzer {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            hop_size: 512,
        }
    }

    /// Perform full analysis
    pub fn analyze(&self, audio: &[f32]) -> MirAnalysis {
        let chromagram = Chromagram::from_audio(audio, self.sample_rate, self.hop_size);

        MirAnalysis {
            tempo: self.estimate_tempo(audio),
            key: self.detect_key(&chromagram),
            chords: self.detect_chords(&chromagram),
            beats: self.detect_beats(audio),
            segments: self.segment_structure(audio, &chromagram),
        }
    }

    fn estimate_tempo(&self, audio: &[f32]) -> f32 {
        beat::BeatTracker::new(self.sample_rate).estimate_tempo(audio)
    }

    fn detect_key(&self, chromagram: &Chromagram) -> Key {
        chord::KeyDetector::new().detect(chromagram)
    }

    fn detect_chords(&self, chromagram: &Chromagram) -> Vec<(f32, Chord)> {
        chord::ChordDetector::new().detect_sequence(chromagram)
    }

    fn detect_beats(&self, audio: &[f32]) -> Vec<Beat> {
        beat::BeatTracker::new(self.sample_rate).detect_beats(audio)
    }

    fn segment_structure(&self, audio: &[f32], chromagram: &Chromagram) -> Vec<Segment> {
        structure::StructureAnalyzer::new(self.sample_rate).analyze(audio, chromagram)
    }
}

/// Complete MIR analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirAnalysis {
    pub tempo: f32,
    pub key: Key,
    pub chords: Vec<(f32, Chord)>,
    pub beats: Vec<Beat>,
    pub segments: Vec<Segment>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_note_names() {
        assert_eq!(NoteName::from_pitch_class(0), NoteName::C);
        assert_eq!(NoteName::from_pitch_class(12), NoteName::C);
        assert_eq!(NoteName::A.to_pitch_class(), 9);
    }

    #[test]
    fn test_chord() {
        let chord = Chord::new(NoteName::C, ChordQuality::Major);
        assert_eq!(chord.symbol(), "C");
        assert_eq!(chord.pitch_classes(), vec![0, 4, 7]);

        let chord = Chord::new(NoteName::A, ChordQuality::Minor);
        assert_eq!(chord.symbol(), "Am");
    }

    #[test]
    fn test_key() {
        let key = Key {
            tonic: NoteName::G,
            is_major: true,
            confidence: 0.9,
        };
        assert_eq!(key.symbol(), "G major");
    }
}
