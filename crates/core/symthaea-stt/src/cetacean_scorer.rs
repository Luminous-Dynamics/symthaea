// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cetacean Scorer - HDC-based Acoustic Pattern Scorer for Whale Vocalizations
//!
//! This module applies the "articulatory feature" paradigm to cetacean communication,
//! treating whale sounds as having their own "phonological" structure.
//!
//! # Cetacean Articulatory Units (CAUs)
//!
//! Just as human speech has:
//!   - Manner: Stop, Fricative, Nasal, etc.
//!   - Place: Bilabial, Alveolar, Velar, etc.
//!
//! Cetacean vocalizations have:
//!   - **Manner**: Click (impulsive), Whistle (tonal), Burst (broadband)
//!   - **Place**: Upsweep, Downsweep, Flat (frequency modulation contour)
//!
//! # Why This Matters
//!
//! HDC excels at **structural relations** (grammar), not bulk storage (vocabulary).
//! With only 3 manner classes × 3 place classes = 9 possible units, the space is:
//!   - Small enough to avoid bundling saturation
//!   - Rich enough to capture distinct call types
//!   - Orthogonal enough for HDC discrimination
//!
//! This is fundamentally different from the WhaleSentinel's click-timing approach.
//! Here we abstract whale sounds into discrete "articulatory" categories.

use crate::hdc::{BundleAccumulator, HV16};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

// ============================================================================
// CETACEAN ARTICULATORY UNITS
// ============================================================================

/// Manner of cetacean articulation (production mechanism)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CetaceanManner {
    /// Click - impulsive, transient sound
    Click,
    /// Whistle - tonal, sustained sound
    Whistle,
    /// Burst - broadband, noisy sound
    Burst,
}

/// Place of cetacean articulation (frequency modulation contour)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CetaceanPlace {
    /// Upsweep - frequency increases over time
    Upsweep,
    /// Downsweep - frequency decreases over time
    Downsweep,
    /// Flat - constant frequency
    Flat,
}

/// A Cetacean Articulatory Unit (CAU) - analogous to a phoneme
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CetaceanUnit {
    pub manner: CetaceanManner,
    pub place: CetaceanPlace,
}

impl CetaceanUnit {
    pub fn new(manner: CetaceanManner, place: CetaceanPlace) -> Self {
        Self { manner, place }
    }

    /// Create from manner and place indices (for ML output)
    pub fn from_indices(manner_idx: usize, place_idx: usize) -> Option<Self> {
        let manner = match manner_idx {
            0 => CetaceanManner::Click,
            1 => CetaceanManner::Whistle,
            2 => CetaceanManner::Burst,
            _ => return None,
        };
        let place = match place_idx {
            0 => CetaceanPlace::Upsweep,
            1 => CetaceanPlace::Downsweep,
            2 => CetaceanPlace::Flat,
            _ => return None,
        };
        Some(Self { manner, place })
    }

    /// Get a unique string representation for HDC seeding
    pub fn to_seed(&self) -> String {
        format!("CAU_{:?}_{:?}", self.manner, self.place)
    }

    /// Short notation (e.g., "CU" = Click+Upsweep)
    pub fn short_name(&self) -> String {
        let m = match self.manner {
            CetaceanManner::Click => "C",
            CetaceanManner::Whistle => "W",
            CetaceanManner::Burst => "B",
        };
        let p = match self.place {
            CetaceanPlace::Upsweep => "U",
            CetaceanPlace::Downsweep => "D",
            CetaceanPlace::Flat => "F",
        };
        format!("{m}{p}")
    }
}

impl std::fmt::Display for CetaceanUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.short_name())
    }
}

// ============================================================================
// ALL CETACEAN UNITS
// ============================================================================

/// All possible CAUs (3 manner × 3 place = 9 units)
pub const ALL_CAUS: [CetaceanUnit; 9] = [
    CetaceanUnit {
        manner: CetaceanManner::Click,
        place: CetaceanPlace::Upsweep,
    },
    CetaceanUnit {
        manner: CetaceanManner::Click,
        place: CetaceanPlace::Downsweep,
    },
    CetaceanUnit {
        manner: CetaceanManner::Click,
        place: CetaceanPlace::Flat,
    },
    CetaceanUnit {
        manner: CetaceanManner::Whistle,
        place: CetaceanPlace::Upsweep,
    },
    CetaceanUnit {
        manner: CetaceanManner::Whistle,
        place: CetaceanPlace::Downsweep,
    },
    CetaceanUnit {
        manner: CetaceanManner::Whistle,
        place: CetaceanPlace::Flat,
    },
    CetaceanUnit {
        manner: CetaceanManner::Burst,
        place: CetaceanPlace::Upsweep,
    },
    CetaceanUnit {
        manner: CetaceanManner::Burst,
        place: CetaceanPlace::Downsweep,
    },
    CetaceanUnit {
        manner: CetaceanManner::Burst,
        place: CetaceanPlace::Flat,
    },
];

/// Manner class names for display
pub const MANNER_NAMES: [&str; 3] = ["Click", "Whistle", "Burst"];

/// Place class names for display
pub const PLACE_NAMES: [&str; 3] = ["Upsweep", "Downsweep", "Flat"];

// ============================================================================
// CETACEAN SCORER
// ============================================================================

/// The Cetacean Scorer implements HDC-based grammar scoring for whale sequences.
///
/// Unlike the WhaleSentinel (which uses timing), this uses discrete articulatory
/// categories and HDC grammar memory - exactly where HDC excels.
pub struct CetaceanScorer {
    /// Random hypervectors for each CAU (9 units + silence + unknown)
    unit_basis: HashMap<String, HV16>,

    /// Grammar memory: superposition of valid transitions
    /// Learned from corpus of whale call sequences
    grammar_memory: HV16,

    /// Current context state: rolling history
    context_state: HV16,

    /// Manner-only basis (for coarse discrimination)
    manner_basis: HashMap<CetaceanManner, HV16>,

    /// Place-only basis (for modulation analysis)
    place_basis: HashMap<CetaceanPlace, HV16>,

    /// HDC score weight (vs local constraints) - reserved for weighted scoring
    #[allow(dead_code)]
    hdc_weight: f32,

    /// Training sequence count
    training_count: usize,
}

impl Default for CetaceanScorer {
    fn default() -> Self {
        Self::new()
    }
}

impl CetaceanScorer {
    /// Create a new Cetacean Scorer with random basis vectors
    pub fn new() -> Self {
        let mut unit_basis = HashMap::new();
        let mut manner_basis = HashMap::new();
        let mut place_basis = HashMap::new();

        // Generate basis vectors for all CAUs
        for cau in ALL_CAUS.iter() {
            let seed = cau.to_seed();
            let hv = HV16::random(&seed);
            unit_basis.insert(cau.short_name(), hv);
        }

        // Silence and Unknown
        unit_basis.insert("SIL".to_string(), HV16::random("CAU_SIL"));
        unit_basis.insert("UNK".to_string(), HV16::random("CAU_UNK"));

        // Manner-level basis (coarse grain)
        manner_basis.insert(CetaceanManner::Click, HV16::random("MANNER_Click"));
        manner_basis.insert(CetaceanManner::Whistle, HV16::random("MANNER_Whistle"));
        manner_basis.insert(CetaceanManner::Burst, HV16::random("MANNER_Burst"));

        // Place-level basis (frequency modulation)
        place_basis.insert(CetaceanPlace::Upsweep, HV16::random("PLACE_Upsweep"));
        place_basis.insert(CetaceanPlace::Downsweep, HV16::random("PLACE_Downsweep"));
        place_basis.insert(CetaceanPlace::Flat, HV16::random("PLACE_Flat"));

        Self {
            unit_basis,
            grammar_memory: HV16::random("cetacean_grammar_init"),
            context_state: HV16::zero(),
            manner_basis,
            place_basis,
            hdc_weight: 0.8,
            training_count: 0,
        }
    }

    /// Get the basis hypervector for a CAU
    pub fn get_basis(&self, unit: &CetaceanUnit) -> Option<&HV16> {
        self.unit_basis.get(&unit.short_name())
    }

    /// Get basis by short name
    pub fn get_basis_by_name(&self, name: &str) -> Option<&HV16> {
        self.unit_basis.get(name)
    }

    /// Reset context state (start new sequence)
    pub fn reset(&mut self) {
        self.context_state = HV16::zero();
    }

    /// Process a single unit, updating context and returning resonance score
    ///
    /// The context update uses permutation binding:
    /// H_t = Π(H_{t-1}) ⊕ Φ_current
    ///
    /// This creates an "infinite n-gram" where all history is compressed
    /// into a fixed-size vector.
    pub fn process(&mut self, unit: &CetaceanUnit) -> f32 {
        let phi = match self.get_basis(unit) {
            Some(hv) => *hv,
            None => return 0.0,
        };

        // Rotate context (permutation = temporal ordering)
        let rotated = self.context_state.rotate(1);

        // Bind with current unit (XOR = association)
        self.context_state = rotated.bind(&phi);

        // Score: how well does this context match learned grammar?
        self.grammar_memory.similarity(&self.context_state)
    }

    /// Score a complete sequence of CAUs
    pub fn score_sequence(&mut self, sequence: &[CetaceanUnit]) -> f32 {
        self.reset();
        let mut total_score = 0.0;
        for unit in sequence {
            total_score += self.process(unit);
        }
        total_score / sequence.len().max(1) as f32
    }

    /// Train grammar memory on a sequence
    ///
    /// Each position in the sequence contributes a context vector
    /// that gets bundled into the grammar memory.
    pub fn train_sequence(&mut self, sequence: &[CetaceanUnit]) {
        if sequence.is_empty() {
            return;
        }

        let mut acc = BundleAccumulator::new();

        // Add current grammar memory (weighted by training count)
        if self.training_count > 0 {
            acc.add_weighted(&self.grammar_memory, self.training_count as f32);
        }

        // Process sequence and accumulate context vectors
        self.reset();
        for unit in sequence {
            if let Some(phi) = self.get_basis(unit) {
                let rotated = self.context_state.rotate(1);
                self.context_state = rotated.bind(phi);
                acc.add(&self.context_state);
            }
        }

        // Update grammar memory
        self.grammar_memory = acc.finalize();
        self.training_count += sequence.len();
    }

    /// Train on multiple sequences
    pub fn train_sequences(&mut self, sequences: &[Vec<CetaceanUnit>]) {
        for seq in sequences {
            self.train_sequence(seq);
        }
    }

    /// Discriminate between two sequences - which is more "grammatical"?
    ///
    /// Returns positive if seq1 is better, negative if seq2 is better.
    pub fn discriminate(&mut self, seq1: &[CetaceanUnit], seq2: &[CetaceanUnit]) -> f32 {
        let score1 = self.score_sequence(seq1);
        let score2 = self.score_sequence(seq2);
        score1 - score2
    }

    /// Get manner-level resonance (coarse discrimination)
    pub fn manner_resonance(&self, manner: CetaceanManner) -> f32 {
        if let Some(hv) = self.manner_basis.get(&manner) {
            self.context_state.similarity(hv)
        } else {
            0.0
        }
    }

    /// Get place-level resonance
    pub fn place_resonance(&self, place: CetaceanPlace) -> f32 {
        if let Some(hv) = self.place_basis.get(&place) {
            self.context_state.similarity(hv)
        } else {
            0.0
        }
    }

    /// Export grammar memory to file
    pub fn save_grammar(&self, path: &Path) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        let bytes = self.grammar_memory.to_bytes();
        writer.write_all(&bytes)?;
        Ok(())
    }

    /// Load grammar memory from file
    pub fn load_grammar(&mut self, path: &Path) -> std::io::Result<()> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut bytes = [0u8; 256];
        reader.read_exact(&mut bytes)?;
        self.grammar_memory = HV16::from_bytes(&bytes);
        Ok(())
    }

    /// Get training statistics
    pub fn stats(&self) -> CetaceanScorerStats {
        CetaceanScorerStats {
            num_units: self.unit_basis.len(),
            training_count: self.training_count,
            grammar_density: self.grammar_memory.popcount() as f32 / 2048.0,
        }
    }
}

/// Statistics about the Cetacean Scorer
#[derive(Debug)]
pub struct CetaceanScorerStats {
    pub num_units: usize,
    pub training_count: usize,
    pub grammar_density: f32,
}

// ============================================================================
// ACOUSTIC FEATURE EXTRACTION FOR CETACEANS
// ============================================================================

/// Acoustic features for CAU classification
#[derive(Debug, Clone)]
pub struct CetaceanAcousticFeatures {
    /// Peak energy (for click detection)
    pub peak_energy: f32,
    /// Spectral centroid
    pub spectral_centroid: f32,
    /// Spectral flux (for transient detection)
    pub spectral_flux: f32,
    /// Frequency modulation slope (Hz/s)
    pub fm_slope: f32,
    /// Harmonicity (for whistle detection)
    pub harmonicity: f32,
    /// Duration (seconds)
    pub duration: f32,
}

impl CetaceanAcousticFeatures {
    /// Classify acoustic features into a CAU
    ///
    /// This is a simple rule-based classifier. A neural network
    /// (like the CfC for human speech) would be more robust.
    pub fn classify(&self) -> CetaceanUnit {
        // Manner classification (production mechanism)
        let manner = if self.spectral_flux > 0.8 && self.duration < 0.05 {
            // High flux + short = Click
            CetaceanManner::Click
        } else if self.harmonicity > 0.6 && self.duration > 0.1 {
            // High harmonicity + sustained = Whistle
            CetaceanManner::Whistle
        } else {
            // Everything else = Burst
            CetaceanManner::Burst
        };

        // Place classification (frequency modulation)
        let place = if self.fm_slope > 100.0 {
            // Rising pitch
            CetaceanPlace::Upsweep
        } else if self.fm_slope < -100.0 {
            // Falling pitch
            CetaceanPlace::Downsweep
        } else {
            // Relatively constant
            CetaceanPlace::Flat
        };

        CetaceanUnit::new(manner, place)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cau_creation() {
        let cau = CetaceanUnit::new(CetaceanManner::Click, CetaceanPlace::Flat);
        assert_eq!(cau.short_name(), "CF");
        assert_eq!(cau.to_seed(), "CAU_Click_Flat");
    }

    #[test]
    fn test_all_caus() {
        assert_eq!(ALL_CAUS.len(), 9);
        let names: Vec<_> = ALL_CAUS.iter().map(|c| c.short_name()).collect();
        assert!(names.contains(&"CU".to_string()));
        assert!(names.contains(&"WF".to_string()));
        assert!(names.contains(&"BD".to_string()));
    }

    #[test]
    fn test_scorer_basics() {
        let scorer = CetaceanScorer::new();
        let stats = scorer.stats();
        assert_eq!(stats.num_units, 11); // 9 CAUs + SIL + UNK
        assert_eq!(stats.training_count, 0);
    }

    #[test]
    fn test_grammar_training() {
        let mut scorer = CetaceanScorer::new();

        // Train on a pattern: Click-Flat → Whistle-Upsweep → Click-Flat
        let pattern = vec![
            CetaceanUnit::new(CetaceanManner::Click, CetaceanPlace::Flat),
            CetaceanUnit::new(CetaceanManner::Whistle, CetaceanPlace::Upsweep),
            CetaceanUnit::new(CetaceanManner::Click, CetaceanPlace::Flat),
        ];

        // Train multiple times
        for _ in 0..10 {
            scorer.train_sequence(&pattern);
        }

        // Same pattern should score higher than random
        let same_score = scorer.score_sequence(&pattern);

        // Different pattern
        let different = vec![
            CetaceanUnit::new(CetaceanManner::Burst, CetaceanPlace::Downsweep),
            CetaceanUnit::new(CetaceanManner::Burst, CetaceanPlace::Flat),
            CetaceanUnit::new(CetaceanManner::Burst, CetaceanPlace::Upsweep),
        ];
        let diff_score = scorer.score_sequence(&different);

        println!("Same pattern score: {}", same_score);
        println!("Different pattern score: {}", diff_score);

        // The trained pattern should have higher resonance
        // Note: HDC with small vocabulary should show clear discrimination
        assert!(
            same_score > diff_score,
            "Expected trained pattern to score higher: {} vs {}",
            same_score,
            diff_score
        );
    }

    #[test]
    fn test_discrimination() {
        let mut scorer = CetaceanScorer::new();

        // Sperm whale coda pattern (Click-Flat repeated)
        let sperm_coda = vec![
            CetaceanUnit::new(CetaceanManner::Click, CetaceanPlace::Flat),
            CetaceanUnit::new(CetaceanManner::Click, CetaceanPlace::Flat),
            CetaceanUnit::new(CetaceanManner::Click, CetaceanPlace::Flat),
        ];

        // Humpback whale song pattern (Whistle with sweeps)
        let humpback_song = vec![
            CetaceanUnit::new(CetaceanManner::Whistle, CetaceanPlace::Upsweep),
            CetaceanUnit::new(CetaceanManner::Whistle, CetaceanPlace::Downsweep),
            CetaceanUnit::new(CetaceanManner::Whistle, CetaceanPlace::Flat),
        ];

        // Train on sperm whale codas
        for _ in 0..20 {
            scorer.train_sequence(&sperm_coda);
        }

        // Discrimination should favor sperm whale pattern
        let disc = scorer.discriminate(&sperm_coda, &humpback_song);
        println!("Discrimination (sperm vs humpback): {}", disc);
        assert!(
            disc > 0.0,
            "Sperm whale pattern should score higher after training"
        );
    }

    #[test]
    fn test_feature_classification() {
        // Click-like features
        let click_features = CetaceanAcousticFeatures {
            peak_energy: 0.9,
            spectral_centroid: 5000.0,
            spectral_flux: 0.95,
            fm_slope: 0.0,
            harmonicity: 0.2,
            duration: 0.01,
        };
        let cau = click_features.classify();
        assert_eq!(cau.manner, CetaceanManner::Click);
        assert_eq!(cau.place, CetaceanPlace::Flat);

        // Whistle-like features
        let whistle_features = CetaceanAcousticFeatures {
            peak_energy: 0.5,
            spectral_centroid: 2000.0,
            spectral_flux: 0.1,
            fm_slope: 500.0,
            harmonicity: 0.85,
            duration: 2.0,
        };
        let cau = whistle_features.classify();
        assert_eq!(cau.manner, CetaceanManner::Whistle);
        assert_eq!(cau.place, CetaceanPlace::Upsweep);
    }
}
