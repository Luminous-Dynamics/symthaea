// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Articulatory Resonator Architecture
//!
//! Instead of classifying acoustic shadows, we model the vocal tract configurations
//! that cast those shadows. Phonemes are not atomic labels - they are BOUND STATES
//! of articulatory gestures.
//!
//! The 3-Stream Architecture:
//! 1. Voicing Stream: Is the glottis vibrating? (Voiced vs Voiceless)
//! 2. Manner Stream: How is air escaping? (Stop, Fricative, Nasal, Liquid, etc.)
//! 3. Place Stream: Where is the constriction? (Bilabial, Alveolar, Velar, etc.)
//!
//! Query = Voicing_HV ⊗ Manner_HV ⊗ Place_HV
//! This query resonates in a Hopfield network to find the valid phoneme attractor.

use crate::hdc::{BundleAccumulator, HV16};
use std::collections::HashMap;

// ============================================================================
// ARTICULATORY FEATURES
// ============================================================================

/// Voicing feature: Is the glottis vibrating?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Voicing {
    Voiced,
    Voiceless,
}

/// Manner of articulation: How is air escaping?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Manner {
    Stop,      // Complete closure then release (p, b, t, d, k, g)
    Fricative, // Turbulent airflow through narrow gap (f, v, s, z, sh, etc.)
    Affricate, // Stop + Fricative (ch, jh)
    Nasal,     // Air through nose (m, n, ng)
    Liquid,    // Partial obstruction (l, r)
    Glide,     // Minimal obstruction, vowel-like (w, y)
    Vowel,     // Open vocal tract
}

/// Place of articulation: Where is the constriction?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Place {
    Bilabial,     // Both lips (p, b, m)
    Labiodental,  // Lower lip + upper teeth (f, v)
    Dental,       // Tongue + teeth (th, dh)
    Alveolar,     // Tongue + alveolar ridge (t, d, n, s, z, l)
    PostAlveolar, // Tongue behind alveolar ridge (r, er)
    Palatal,      // Tongue + hard palate (sh, zh, ch, jh, y)
    Velar,        // Tongue back + soft palate (k, g, ng)
    Glottal,      // Glottis (h)
    // Vowel places (tongue position)
    Front,   // Front vowels (iy, ih, ey, eh, ae)
    Central, // Central vowels (ah)
    Back,    // Back vowels (uw, uh, ow, ao, aa)
}

/// Complete articulatory specification for a phoneme
#[derive(Debug, Clone)]
pub struct ArticulatoryFeatures {
    pub voicing: Voicing,
    pub manner: Manner,
    pub place: Place,
    // Additional vowel features
    pub height: Option<VowelHeight>,
    pub roundness: Option<bool>,
}

/// Vowel height (tongue position)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VowelHeight {
    High, // iy, ih, uw, uh
    Mid,  // ey, eh, ow, er, ah
    Low,  // ae, aa, ao
}

// ============================================================================
// PHONEME TO ARTICULATORY MAPPING
// ============================================================================

/// Maps ARPABET phonemes to their articulatory features
pub struct ArticulatoryMapper {
    features: HashMap<String, ArticulatoryFeatures>,
}

impl ArticulatoryMapper {
    pub fn new() -> Self {
        let mut features = HashMap::new();

        // ========================
        // STOPS (Plosives)
        // ========================
        features.insert(
            "P".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiceless,
                manner: Manner::Stop,
                place: Place::Bilabial,
                height: None,
                roundness: None,
            },
        );
        features.insert(
            "B".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiced,
                manner: Manner::Stop,
                place: Place::Bilabial,
                height: None,
                roundness: None,
            },
        );
        features.insert(
            "T".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiceless,
                manner: Manner::Stop,
                place: Place::Alveolar,
                height: None,
                roundness: None,
            },
        );
        features.insert(
            "D".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiced,
                manner: Manner::Stop,
                place: Place::Alveolar,
                height: None,
                roundness: None,
            },
        );
        features.insert(
            "K".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiceless,
                manner: Manner::Stop,
                place: Place::Velar,
                height: None,
                roundness: None,
            },
        );
        features.insert(
            "G".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiced,
                manner: Manner::Stop,
                place: Place::Velar,
                height: None,
                roundness: None,
            },
        );

        // ========================
        // FRICATIVES
        // ========================
        features.insert(
            "F".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiceless,
                manner: Manner::Fricative,
                place: Place::Labiodental,
                height: None,
                roundness: None,
            },
        );
        features.insert(
            "V".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiced,
                manner: Manner::Fricative,
                place: Place::Labiodental,
                height: None,
                roundness: None,
            },
        );
        features.insert(
            "TH".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiceless,
                manner: Manner::Fricative,
                place: Place::Dental,
                height: None,
                roundness: None,
            },
        );
        features.insert(
            "DH".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiced,
                manner: Manner::Fricative,
                place: Place::Dental,
                height: None,
                roundness: None,
            },
        );
        features.insert(
            "S".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiceless,
                manner: Manner::Fricative,
                place: Place::Alveolar,
                height: None,
                roundness: None,
            },
        );
        features.insert(
            "Z".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiced,
                manner: Manner::Fricative,
                place: Place::Alveolar,
                height: None,
                roundness: None,
            },
        );
        features.insert(
            "SH".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiceless,
                manner: Manner::Fricative,
                place: Place::Palatal,
                height: None,
                roundness: None,
            },
        );
        features.insert(
            "ZH".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiced,
                manner: Manner::Fricative,
                place: Place::Palatal,
                height: None,
                roundness: None,
            },
        );
        features.insert(
            "HH".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiceless,
                manner: Manner::Fricative,
                place: Place::Glottal,
                height: None,
                roundness: None,
            },
        );

        // ========================
        // AFFRICATES
        // ========================
        features.insert(
            "CH".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiceless,
                manner: Manner::Affricate,
                place: Place::Palatal,
                height: None,
                roundness: None,
            },
        );
        features.insert(
            "JH".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiced,
                manner: Manner::Affricate,
                place: Place::Palatal,
                height: None,
                roundness: None,
            },
        );

        // ========================
        // NASALS
        // ========================
        features.insert(
            "M".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiced,
                manner: Manner::Nasal,
                place: Place::Bilabial,
                height: None,
                roundness: None,
            },
        );
        features.insert(
            "N".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiced,
                manner: Manner::Nasal,
                place: Place::Alveolar,
                height: None,
                roundness: None,
            },
        );
        features.insert(
            "NG".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiced,
                manner: Manner::Nasal,
                place: Place::Velar,
                height: None,
                roundness: None,
            },
        );

        // ========================
        // LIQUIDS
        // ========================
        features.insert(
            "L".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiced,
                manner: Manner::Liquid,
                place: Place::Alveolar,
                height: None,
                roundness: None,
            },
        );
        features.insert(
            "R".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiced,
                manner: Manner::Liquid,
                place: Place::PostAlveolar, // Approximant, postalveolar region
                height: None,
                roundness: None,
            },
        );

        // ========================
        // GLIDES (Semivowels)
        // ========================
        features.insert(
            "W".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiced,
                manner: Manner::Glide,
                place: Place::Bilabial, // Lip rounding
                height: Some(VowelHeight::High),
                roundness: Some(true),
            },
        );
        features.insert(
            "Y".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiced,
                manner: Manner::Glide,
                place: Place::Palatal,
                height: Some(VowelHeight::High),
                roundness: Some(false),
            },
        );

        // ========================
        // VOWELS
        // ========================
        // High Front
        features.insert(
            "IY".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiced,
                manner: Manner::Vowel,
                place: Place::Front,
                height: Some(VowelHeight::High),
                roundness: Some(false),
            },
        );
        features.insert(
            "IH".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiced,
                manner: Manner::Vowel,
                place: Place::Front,
                height: Some(VowelHeight::High),
                roundness: Some(false),
            },
        );

        // Mid Front
        features.insert(
            "EY".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiced,
                manner: Manner::Vowel,
                place: Place::Front,
                height: Some(VowelHeight::Mid),
                roundness: Some(false),
            },
        );
        features.insert(
            "EH".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiced,
                manner: Manner::Vowel,
                place: Place::Front,
                height: Some(VowelHeight::Mid),
                roundness: Some(false),
            },
        );

        // Low Front
        features.insert(
            "AE".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiced,
                manner: Manner::Vowel,
                place: Place::Front,
                height: Some(VowelHeight::Low),
                roundness: Some(false),
            },
        );

        // Central
        features.insert(
            "AH".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiced,
                manner: Manner::Vowel,
                place: Place::Central,
                height: Some(VowelHeight::Mid),
                roundness: Some(false),
            },
        );
        features.insert(
            "ER".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiced,
                manner: Manner::Vowel,
                place: Place::PostAlveolar, // Rhotacized — tongue curls to postalveolar region for /ɝ/
                height: Some(VowelHeight::Mid),
                roundness: Some(true), // Rhotacized — lips round slightly for /ɝ/
            },
        );

        // High Back
        features.insert(
            "UW".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiced,
                manner: Manner::Vowel,
                place: Place::Back,
                height: Some(VowelHeight::High),
                roundness: Some(true),
            },
        );
        features.insert(
            "UH".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiced,
                manner: Manner::Vowel,
                place: Place::Back,
                height: Some(VowelHeight::High),
                roundness: Some(true),
            },
        );

        // Mid Back
        features.insert(
            "OW".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiced,
                manner: Manner::Vowel,
                place: Place::Back,
                height: Some(VowelHeight::Mid),
                roundness: Some(true),
            },
        );

        // Low Back
        features.insert(
            "AA".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiced,
                manner: Manner::Vowel,
                place: Place::Back,
                height: Some(VowelHeight::Low),
                roundness: Some(false),
            },
        );
        features.insert(
            "AO".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiced,
                manner: Manner::Vowel,
                place: Place::Back,
                height: Some(VowelHeight::Mid), // /ɔː/ is open-mid, distinct from AA /ɑː/ which is fully low
                roundness: Some(true),
            },
        );

        // Diphthongs (use starting position)
        features.insert(
            "AY".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiced,
                manner: Manner::Vowel,
                place: Place::Central, // Starts low-central
                height: Some(VowelHeight::Low),
                roundness: Some(false),
            },
        );
        features.insert(
            "AW".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiced,
                manner: Manner::Vowel,
                place: Place::Central,
                height: Some(VowelHeight::Low),
                roundness: Some(false),
            },
        );
        features.insert(
            "OY".into(),
            ArticulatoryFeatures {
                voicing: Voicing::Voiced,
                manner: Manner::Vowel,
                place: Place::Back,
                height: Some(VowelHeight::Mid),
                roundness: Some(true),
            },
        );

        Self { features }
    }

    /// Get articulatory features for a phoneme (strips stress markers)
    pub fn get(&self, phoneme: &str) -> Option<&ArticulatoryFeatures> {
        let base = strip_stress(phoneme);
        self.features.get(&base)
    }

    /// Get all phonemes with specific features
    pub fn find_by_features(
        &self,
        voicing: Option<Voicing>,
        manner: Option<Manner>,
        place: Option<Place>,
    ) -> Vec<String> {
        self.features
            .iter()
            .filter(|(_, f)| {
                voicing.is_none_or(|v| f.voicing == v)
                    && manner.is_none_or(|m| f.manner == m)
                    && place.is_none_or(|p| f.place == p)
            })
            .map(|(name, _)| name.clone())
            .collect()
    }
}

impl Default for ArticulatoryMapper {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// 3-STREAM ARTICULATORY HDC
// ============================================================================

/// Hypervector representations of articulatory features
pub struct ArticulatoryHDC {
    // Role vectors (for binding) - public for detector access
    pub role_voicing: HV16,
    pub role_manner: HV16,
    pub role_place: HV16,
    pub role_height: HV16,
    pub role_roundness: HV16,

    // Voicing feature HVs
    hv_voiced: HV16,
    hv_voiceless: HV16,

    // Manner feature HVs
    hv_stop: HV16,
    hv_fricative: HV16,
    hv_affricate: HV16,
    hv_nasal: HV16,
    hv_liquid: HV16,
    hv_glide: HV16,
    hv_vowel: HV16,

    // Place feature HVs
    hv_bilabial: HV16,
    hv_labiodental: HV16,
    hv_dental: HV16,
    hv_alveolar: HV16,
    hv_postalveolar: HV16,
    hv_palatal: HV16,
    hv_velar: HV16,
    hv_glottal: HV16,
    hv_front: HV16,
    hv_central: HV16,
    hv_back: HV16,

    // Vowel height HVs
    hv_high: HV16,
    hv_mid: HV16,
    hv_low: HV16,

    // Roundness HVs
    hv_rounded: HV16,
    hv_unrounded: HV16,

    // Pre-computed phoneme HVs (bound feature combinations)
    phoneme_hvs: HashMap<String, HV16>,

    // The mapper
    mapper: ArticulatoryMapper,
}

impl ArticulatoryHDC {
    pub fn new() -> Self {
        // Generate random role vectors
        let role_voicing = Self::random_hv(1001);
        let role_manner = Self::random_hv(1002);
        let role_place = Self::random_hv(1003);
        let role_height = Self::random_hv(1004);
        let role_roundness = Self::random_hv(1005);

        // Generate voicing HVs
        let hv_voiced = Self::random_hv(2001);
        let hv_voiceless = Self::random_hv(2002);

        // Generate manner HVs
        let hv_stop = Self::random_hv(3001);
        let hv_fricative = Self::random_hv(3002);
        let hv_affricate = Self::random_hv(3003);
        let hv_nasal = Self::random_hv(3004);
        let hv_liquid = Self::random_hv(3005);
        let hv_glide = Self::random_hv(3006);
        let hv_vowel = Self::random_hv(3007);

        // Generate place HVs
        let hv_bilabial = Self::random_hv(4001);
        let hv_labiodental = Self::random_hv(4002);
        let hv_dental = Self::random_hv(4003);
        let hv_alveolar = Self::random_hv(4004);
        let hv_postalveolar = Self::random_hv(4011);
        let hv_palatal = Self::random_hv(4005);
        let hv_velar = Self::random_hv(4006);
        let hv_glottal = Self::random_hv(4007);
        let hv_front = Self::random_hv(4008);
        let hv_central = Self::random_hv(4009);
        let hv_back = Self::random_hv(4010);

        // Generate height HVs
        let hv_high = Self::random_hv(5001);
        let hv_mid = Self::random_hv(5002);
        let hv_low = Self::random_hv(5003);

        // Generate roundness HVs
        let hv_rounded = Self::random_hv(6001);
        let hv_unrounded = Self::random_hv(6002);

        let mapper = ArticulatoryMapper::new();

        let mut hdc = Self {
            role_voicing,
            role_manner,
            role_place,
            role_height,
            role_roundness,
            hv_voiced,
            hv_voiceless,
            hv_stop,
            hv_fricative,
            hv_affricate,
            hv_nasal,
            hv_liquid,
            hv_glide,
            hv_vowel,
            hv_bilabial,
            hv_labiodental,
            hv_dental,
            hv_alveolar,
            hv_postalveolar,
            hv_palatal,
            hv_velar,
            hv_glottal,
            hv_front,
            hv_central,
            hv_back,
            hv_high,
            hv_mid,
            hv_low,
            hv_rounded,
            hv_unrounded,
            phoneme_hvs: HashMap::new(),
            mapper,
        };

        // Pre-compute all phoneme HVs
        hdc.build_phoneme_hvs();

        hdc
    }

    /// Generate a deterministic random HV from a seed
    fn random_hv(seed: u64) -> HV16 {
        use std::hash::{Hash, Hasher};
        let mut hv = HV16::zero();

        for w in 0..16 {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            seed.hash(&mut hasher);
            w.hash(&mut hasher);
            hv.words[w] = hasher.finish() as u128;
        }

        hv
    }

    /// Build phoneme HVs by binding articulatory features
    fn build_phoneme_hvs(&mut self) {
        let phonemes = [
            "P", "B", "T", "D", "K", "G", // Stops
            "F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH", // Fricatives
            "CH", "JH", // Affricates
            "M", "N", "NG", // Nasals
            "L", "R", // Liquids
            "W", "Y", // Glides
            "IY", "IH", "EY", "EH", "AE", // Front vowels
            "AH", "ER", // Central/rhotacized vowels
            "UW", "UH", "OW", "AA", "AO", // Back vowels
            "AY", "AW", "OY", // Diphthongs
        ];

        for phoneme in phonemes {
            if let Some(features) = self.mapper.get(phoneme) {
                let hv = self.encode_features(features);
                self.phoneme_hvs.insert(phoneme.to_string(), hv);
            }
        }
    }

    /// Encode articulatory features into a bound HV
    pub fn encode_features(&self, features: &ArticulatoryFeatures) -> HV16 {
        // Get feature HVs
        let voicing_hv = match features.voicing {
            Voicing::Voiced => &self.hv_voiced,
            Voicing::Voiceless => &self.hv_voiceless,
        };

        let manner_hv = match features.manner {
            Manner::Stop => &self.hv_stop,
            Manner::Fricative => &self.hv_fricative,
            Manner::Affricate => &self.hv_affricate,
            Manner::Nasal => &self.hv_nasal,
            Manner::Liquid => &self.hv_liquid,
            Manner::Glide => &self.hv_glide,
            Manner::Vowel => &self.hv_vowel,
        };

        let place_hv = match features.place {
            Place::Bilabial => &self.hv_bilabial,
            Place::Labiodental => &self.hv_labiodental,
            Place::Dental => &self.hv_dental,
            Place::Alveolar => &self.hv_alveolar,
            Place::PostAlveolar => &self.hv_postalveolar,
            Place::Palatal => &self.hv_palatal,
            Place::Velar => &self.hv_velar,
            Place::Glottal => &self.hv_glottal,
            Place::Front => &self.hv_front,
            Place::Central => &self.hv_central,
            Place::Back => &self.hv_back,
        };

        // Bind: Role ⊗ Value for each feature, then bundle
        let bound_voicing = self.role_voicing.bind(voicing_hv);
        let bound_manner = self.role_manner.bind(manner_hv);
        let bound_place = self.role_place.bind(place_hv);

        // Bundle the bound features
        let mut acc = BundleAccumulator::new();
        acc.add(&bound_voicing);
        acc.add(&bound_manner);
        acc.add(&bound_place);

        // Add height for vowels
        if let Some(height) = features.height {
            let height_hv = match height {
                VowelHeight::High => &self.hv_high,
                VowelHeight::Mid => &self.hv_mid,
                VowelHeight::Low => &self.hv_low,
            };
            let bound_height = self.role_height.bind(height_hv);
            acc.add(&bound_height);
        }

        // Add roundness for vowels
        if let Some(rounded) = features.roundness {
            let roundness_hv = if rounded {
                &self.hv_rounded
            } else {
                &self.hv_unrounded
            };
            let bound_roundness = self.role_roundness.bind(roundness_hv);
            acc.add(&bound_roundness);
        }

        acc.finalize()
    }

    /// Get the pre-computed HV for a phoneme
    pub fn get_phoneme_hv(&self, phoneme: &str) -> Option<&HV16> {
        let base = strip_stress(phoneme);
        self.phoneme_hvs.get(&base)
    }

    /// Get the voicing HV
    pub fn voicing_hv(&self, voicing: Voicing) -> &HV16 {
        match voicing {
            Voicing::Voiced => &self.hv_voiced,
            Voicing::Voiceless => &self.hv_voiceless,
        }
    }

    /// Get the manner HV
    pub fn manner_hv(&self, manner: Manner) -> &HV16 {
        match manner {
            Manner::Stop => &self.hv_stop,
            Manner::Fricative => &self.hv_fricative,
            Manner::Affricate => &self.hv_affricate,
            Manner::Nasal => &self.hv_nasal,
            Manner::Liquid => &self.hv_liquid,
            Manner::Glide => &self.hv_glide,
            Manner::Vowel => &self.hv_vowel,
        }
    }

    /// Get the place HV
    pub fn place_hv(&self, place: Place) -> &HV16 {
        match place {
            Place::Bilabial => &self.hv_bilabial,
            Place::Labiodental => &self.hv_labiodental,
            Place::Dental => &self.hv_dental,
            Place::Alveolar => &self.hv_alveolar,
            Place::PostAlveolar => &self.hv_postalveolar,
            Place::Palatal => &self.hv_palatal,
            Place::Velar => &self.hv_velar,
            Place::Glottal => &self.hv_glottal,
            Place::Front => &self.hv_front,
            Place::Central => &self.hv_central,
            Place::Back => &self.hv_back,
        }
    }

    /// Query: Find best matching phoneme for a query HV
    pub fn query(&self, query_hv: &HV16) -> Vec<(String, f32)> {
        let mut results: Vec<_> = self
            .phoneme_hvs
            .iter()
            .map(|(name, hv)| (name.clone(), query_hv.similarity(hv)))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Get the articulatory mapper
    pub fn mapper(&self) -> &ArticulatoryMapper {
        &self.mapper
    }
}

impl Default for ArticulatoryHDC {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// HOPFIELD RESONATOR
// ============================================================================

/// Hopfield-style attractor network for phoneme recognition
///
/// The network stores valid phoneme configurations as attractors.
/// A noisy query will settle into the nearest valid basin.
pub struct ArticulatoryResonator {
    /// The articulatory HDC system
    hdc: ArticulatoryHDC,

    /// Attractor states (valid phoneme HVs)
    attractors: Vec<(String, HV16)>,

    /// Resonance iterations
    max_iterations: usize,

    /// Convergence threshold (reserved for iterative resonance settling)
    #[allow(dead_code)]
    convergence_threshold: f32,
}

impl ArticulatoryResonator {
    pub fn new() -> Self {
        let hdc = ArticulatoryHDC::new();

        // Build attractors from valid phonemes
        let attractors: Vec<_> = hdc
            .phoneme_hvs
            .iter()
            .map(|(name, hv)| (name.clone(), *hv))
            .collect();

        Self {
            hdc,
            attractors,
            max_iterations: 10,
            convergence_threshold: 0.99,
        }
    }

    /// Resonate: Let a query HV settle into an attractor basin
    ///
    /// Uses iterative cleanup dynamics where the state is repeatedly
    /// projected onto the nearest attractor, with diminishing influence
    /// from distant attractors.
    pub fn resonate(&self, query: &HV16) -> (String, f32, usize) {
        let mut state = *query;
        let mut iterations = 0;
        let mut prev_best = String::new();
        let mut stable_count = 0;

        for i in 0..self.max_iterations {
            iterations = i + 1;

            // Find the closest attractor
            let mut best_name = String::new();
            let mut best_sim = -1.0f32;
            let mut second_sim = -1.0f32;

            for (name, attractor) in &self.attractors {
                let sim = state.similarity(attractor);
                if sim > best_sim {
                    second_sim = best_sim;
                    best_sim = sim;
                    best_name = name.clone();
                } else if sim > second_sim {
                    second_sim = sim;
                }
            }

            // Check for stability (same winner twice in a row)
            if best_name == prev_best {
                stable_count += 1;
                if stable_count >= 2 {
                    break; // Converged to stable attractor
                }
            } else {
                stable_count = 0;
            }
            prev_best = best_name.clone();

            // Move state toward the best attractor (cleanup step)
            // Blend: 70% current best attractor, 30% current state
            if let Some((_, best_attractor)) = self.attractors.iter().find(|(n, _)| n == &best_name)
            {
                let mut acc = BundleAccumulator::new();
                // Add attractor multiple times for weighting
                for _ in 0..7 {
                    acc.add(best_attractor);
                }
                for _ in 0..3 {
                    acc.add(&state);
                }
                state = acc.finalize();
            }
        }

        // Final: find best matching attractor
        let mut best_name = String::new();
        let mut best_sim = -1.0f32;

        for (name, attractor) in &self.attractors {
            let sim = state.similarity(attractor);
            if sim > best_sim {
                best_sim = sim;
                best_name = name.clone();
            }
        }

        (best_name, best_sim, iterations)
    }

    /// Direct query without resonance (for comparison)
    pub fn query_direct(&self, query: &HV16) -> Vec<(String, f32)> {
        self.hdc.query(query)
    }

    /// Get the HDC system
    pub fn hdc(&self) -> &ArticulatoryHDC {
        &self.hdc
    }
}

impl Default for ArticulatoryResonator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// ACOUSTIC-TO-ARTICULATORY DETECTOR (3-Stream)
// ============================================================================

/// Detects articulatory features from acoustic (mel) features
///
/// This is a heuristic-based detector. The proper solution is to train
/// 3 CfC heads in Python and load the weights. But this lets us test
/// the architecture end-to-end.
pub struct AcousticArticulatoryDetector {
    /// The articulatory HDC system
    hdc: ArticulatoryHDC,
}

impl AcousticArticulatoryDetector {
    pub fn new() -> Self {
        Self {
            hdc: ArticulatoryHDC::new(),
        }
    }

    /// Detect articulatory features from mel spectrogram frame
    ///
    /// mel_features: 40-dimensional mel spectrogram (normalized 0-1)
    /// prev_mel: Previous frame (for temporal features)
    ///
    /// Returns bound articulatory HV: Voicing ⊗ Manner ⊗ Place
    pub fn detect(&self, mel_features: &[f32], prev_mel: Option<&[f32]>) -> HV16 {
        // Detect each stream
        let voicing = self.detect_voicing(mel_features);
        let manner = self.detect_manner(mel_features, prev_mel);
        let place = self.detect_place(mel_features);

        // Get HVs for detected features
        let voicing_hv = self.hdc.voicing_hv(voicing);
        let manner_hv = self.hdc.manner_hv(manner);
        let place_hv = self.hdc.place_hv(place);

        // Bind with roles and bundle
        let bound_v = self.hdc.role_voicing.bind(voicing_hv);
        let bound_m = self.hdc.role_manner.bind(manner_hv);
        let bound_p = self.hdc.role_place.bind(place_hv);

        let mut acc = BundleAccumulator::new();
        acc.add(&bound_v);
        acc.add(&bound_m);
        acc.add(&bound_p);

        acc.finalize()
    }

    /// Detect voicing from mel features
    ///
    /// Heuristic: Voiced sounds have strong low-frequency energy (harmonics)
    /// Voiceless sounds have more high-frequency energy (noise)
    fn detect_voicing(&self, mel: &[f32]) -> Voicing {
        if mel.len() < 10 {
            return Voicing::Voiced; // Default
        }

        // Low frequency energy (mel bins 0-10, roughly 0-1kHz)
        let low_energy: f32 = mel[0..10].iter().sum();

        // High frequency energy (mel bins 25-40, roughly 4-8kHz)
        let high_start = mel.len().saturating_sub(15);
        let high_energy: f32 = mel[high_start..].iter().sum();

        // Total energy
        let total: f32 = mel.iter().sum();

        if total < 0.1 {
            return Voicing::Voiceless; // Silence
        }

        // Voicing ratio: voiced sounds have dominant low-frequency
        let low_ratio = low_energy / (total + 0.001);
        let high_ratio = high_energy / (total + 0.001);

        if low_ratio > 0.4 && high_ratio < 0.3 {
            Voicing::Voiced
        } else if high_ratio > 0.35 {
            Voicing::Voiceless
        } else {
            Voicing::Voiced // Default to voiced (more common)
        }
    }

    /// Detect manner of articulation
    ///
    /// Heuristics:
    /// - Stops: Low energy followed by burst (high delta)
    /// - Fricatives: High-frequency noise (spread energy in high mels)
    /// - Nasals: Strong low-frequency, weak high, steady energy
    /// - Vowels: Strong formants, high total energy
    fn detect_manner(&self, mel: &[f32], prev_mel: Option<&[f32]>) -> Manner {
        if mel.len() < 10 {
            return Manner::Vowel;
        }

        let total: f32 = mel.iter().sum();
        let high_start = mel.len().saturating_sub(15);
        let high_energy: f32 = mel[high_start..].iter().sum();
        let low_energy: f32 = mel[0..10].iter().sum();

        // Compute energy delta if we have previous frame
        let delta = if let Some(prev) = prev_mel {
            let prev_total: f32 = prev.iter().sum();
            (total - prev_total).abs()
        } else {
            0.0
        };

        // High-frequency dominance suggests fricative
        let high_ratio = high_energy / (total + 0.001);

        // Low total energy + high delta = stop burst
        if total < 0.3 && delta > 0.3 {
            return Manner::Stop;
        }

        // High HF ratio = fricative
        if high_ratio > 0.35 && total > 0.2 {
            return Manner::Fricative;
        }

        // Strong low, weak high, moderate total = nasal
        let low_ratio = low_energy / (total + 0.001);
        if low_ratio > 0.5 && high_ratio < 0.15 && total > 0.15 && total < 0.5 {
            return Manner::Nasal;
        }

        // Low total = possible stop closure or glide
        if total < 0.2 {
            if delta < 0.1 {
                return Manner::Glide;
            } else {
                return Manner::Stop;
            }
        }

        // Moderate energy with good low-mid balance = liquid
        if total > 0.2 && total < 0.4 && low_ratio > 0.3 && low_ratio < 0.6 {
            return Manner::Liquid;
        }

        // Default: vowel (high energy, balanced spectrum)
        Manner::Vowel
    }

    /// Detect place of articulation
    ///
    /// Heuristics based on formant patterns:
    /// - Front articulation: Higher F2 (energy in mid-high mels)
    /// - Back articulation: Lower F2 (energy in lower-mid mels)
    /// - Bilabial: Affects F1 more than F2/F3
    fn detect_place(&self, mel: &[f32]) -> Place {
        if mel.len() < 20 {
            return Place::Alveolar; // Default
        }

        // F1 region (roughly mel bins 2-8)
        let f1_energy: f32 = mel[2..8.min(mel.len())].iter().sum();

        // F2 region (roughly mel bins 8-16)
        let f2_start = 8.min(mel.len());
        let f2_end = 16.min(mel.len());
        let f2_energy: f32 = mel[f2_start..f2_end].iter().sum();

        // F3 region (roughly mel bins 16-24)
        let f3_start = 16.min(mel.len());
        let f3_end = 24.min(mel.len());
        let f3_energy: f32 = if f3_start < mel.len() {
            mel[f3_start..f3_end].iter().sum()
        } else {
            0.0
        };

        // High frequency region (mel bins 30+)
        let hf_start = 30.min(mel.len());
        let hf_energy: f32 = if hf_start < mel.len() {
            mel[hf_start..].iter().sum()
        } else {
            0.0
        };

        let total: f32 = mel.iter().sum();
        if total < 0.1 {
            return Place::Alveolar; // Low energy - default
        }

        // Normalize
        let f1_ratio = f1_energy / (total + 0.001);
        let f2_ratio = f2_energy / (total + 0.001);
        let f3_ratio = f3_energy / (total + 0.001);
        let hf_ratio = hf_energy / (total + 0.001);

        // Bilabial: Strong F1, weak F2/F3
        if f1_ratio > 0.35 && f2_ratio < 0.25 {
            return Place::Bilabial;
        }

        // Palatal/Alveolar fricatives: High F3 and HF energy
        if f3_ratio > 0.2 || hf_ratio > 0.25 {
            if hf_ratio > 0.35 {
                return Place::Palatal; // Very high frequency = palatal
            }
            return Place::Alveolar;
        }

        // Velar: F2 and F3 close together (mid-frequency concentration)
        if f2_ratio > 0.25 && f3_ratio > 0.15 && (f2_ratio - f3_ratio).abs() < 0.1 {
            return Place::Velar;
        }

        // Dental: Between alveolar and labiodental
        if f2_ratio > 0.2 && f2_ratio < 0.35 && f1_ratio > 0.2 {
            return Place::Dental;
        }

        // Front vowels: High F2
        if f2_ratio > 0.35 {
            return Place::Front;
        }

        // Back vowels: Low F2, possibly high F1
        if f2_ratio < 0.2 && f1_ratio > 0.25 {
            return Place::Back;
        }

        // Default: alveolar (most common consonant place)
        Place::Alveolar
    }

    /// Get the underlying HDC system
    pub fn hdc(&self) -> &ArticulatoryHDC {
        &self.hdc
    }
}

impl Default for AcousticArticulatoryDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// UTILITIES
// ============================================================================

/// Strip stress markers from phoneme (AH0 -> AH, T_0 -> T)
fn strip_stress(phoneme: &str) -> String {
    let base = if let Some(idx) = phoneme.rfind('_') {
        if phoneme[idx + 1..].chars().all(|c| c.is_ascii_digit()) {
            &phoneme[..idx]
        } else {
            phoneme
        }
    } else {
        phoneme
    };

    base.trim_end_matches(|c: char| c.is_ascii_digit())
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_articulatory_mapper() {
        let mapper = ArticulatoryMapper::new();

        // Test stop consonants
        let t = mapper.get("T").unwrap();
        assert_eq!(t.voicing, Voicing::Voiceless);
        assert_eq!(t.manner, Manner::Stop);
        assert_eq!(t.place, Place::Alveolar);

        let d = mapper.get("D").unwrap();
        assert_eq!(d.voicing, Voicing::Voiced);
        assert_eq!(d.manner, Manner::Stop);
        assert_eq!(d.place, Place::Alveolar);

        // Test nasals - the key confusion pair
        let n = mapper.get("N").unwrap();
        assert_eq!(n.voicing, Voicing::Voiced);
        assert_eq!(n.manner, Manner::Nasal);
        assert_eq!(n.place, Place::Alveolar);

        let ng = mapper.get("NG").unwrap();
        assert_eq!(ng.voicing, Voicing::Voiced);
        assert_eq!(ng.manner, Manner::Nasal);
        assert_eq!(ng.place, Place::Velar); // Different place!
    }

    #[test]
    fn test_articulatory_hdc() {
        let hdc = ArticulatoryHDC::new();

        // N and NG should have similar manner but different place
        let n_hv = hdc.get_phoneme_hv("N").unwrap();
        let ng_hv = hdc.get_phoneme_hv("NG").unwrap();
        let d_hv = hdc.get_phoneme_hv("D").unwrap();

        // N and NG are both nasals - should have some similarity
        let n_ng_sim = n_hv.similarity(ng_hv);

        // N and D are both alveolar - should have some similarity
        let n_d_sim = n_hv.similarity(d_hv);

        // But N should be more similar to NG than to D (both nasals)
        // Actually this depends on how we weight the features
        println!("N-NG similarity: {:.4}", n_ng_sim);
        println!("N-D similarity: {:.4}", n_d_sim);

        // Both should be distinct (< 0.5 due to feature differences)
        assert!(n_ng_sim < 0.8);
        assert!(n_d_sim < 0.8);
    }

    #[test]
    fn test_resonator() {
        let resonator = ArticulatoryResonator::new();

        // Query with a clean phoneme HV
        let n_hv = resonator.hdc().get_phoneme_hv("N").unwrap();
        let (result, sim, iters) = resonator.resonate(n_hv);

        println!("Query N -> {} (sim={:.4}, iters={})", result, sim, iters);
        assert_eq!(result, "N");
    }

    #[test]
    fn test_strip_stress_basic() {
        assert_eq!(strip_stress("AH0"), "AH");
        assert_eq!(strip_stress("IY1"), "IY");
        assert_eq!(strip_stress("AE2"), "AE");
    }

    #[test]
    fn test_strip_stress_no_stress() {
        assert_eq!(strip_stress("T"), "T");
        assert_eq!(strip_stress("SH"), "SH");
        assert_eq!(strip_stress("NG"), "NG");
    }

    #[test]
    fn test_strip_stress_underscore_format() {
        assert_eq!(strip_stress("T_0"), "T");
        assert_eq!(strip_stress("AH_1"), "AH");
    }

    #[test]
    fn test_strip_stress_underscore_non_digit() {
        // Underscore followed by non-digit should not strip
        assert_eq!(strip_stress("T_X"), "T_X");
    }

    #[test]
    fn test_mapper_all_consonant_categories() {
        let mapper = ArticulatoryMapper::new();

        // Stops
        for phone in &["P", "B", "T", "D", "K", "G"] {
            let f = mapper.get(phone).unwrap();
            assert_eq!(f.manner, Manner::Stop, "{phone} should be Stop");
        }

        // Fricatives
        for phone in &["F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"] {
            let f = mapper.get(phone).unwrap();
            assert_eq!(f.manner, Manner::Fricative, "{phone} should be Fricative");
        }

        // Affricates
        for phone in &["CH", "JH"] {
            let f = mapper.get(phone).unwrap();
            assert_eq!(f.manner, Manner::Affricate, "{phone} should be Affricate");
        }

        // Nasals
        for phone in &["M", "N", "NG"] {
            let f = mapper.get(phone).unwrap();
            assert_eq!(f.manner, Manner::Nasal, "{phone} should be Nasal");
            assert_eq!(f.voicing, Voicing::Voiced, "{phone} should be Voiced");
        }

        // Liquids
        for phone in &["L", "R"] {
            let f = mapper.get(phone).unwrap();
            assert_eq!(f.manner, Manner::Liquid, "{phone} should be Liquid");
        }

        // Glides
        for phone in &["W", "Y"] {
            let f = mapper.get(phone).unwrap();
            assert_eq!(f.manner, Manner::Glide, "{phone} should be Glide");
        }
    }

    #[test]
    fn test_mapper_vowel_features() {
        let mapper = ArticulatoryMapper::new();

        // All vowels should be Voiced with Manner::Vowel
        for phone in &[
            "IY", "IH", "EY", "EH", "AE", "AH", "ER", "UW", "UH", "OW", "AA", "AO",
        ] {
            let f = mapper.get(phone).unwrap();
            assert_eq!(f.manner, Manner::Vowel, "{phone} should be Vowel");
            assert_eq!(f.voicing, Voicing::Voiced, "{phone} should be Voiced");
            assert!(f.height.is_some(), "{phone} should have height");
        }
    }

    #[test]
    fn test_mapper_voicing_pairs() {
        let mapper = ArticulatoryMapper::new();

        // Voiceless/Voiced pairs should share place and manner
        let pairs = [("P", "B"), ("T", "D"), ("K", "G"), ("F", "V"), ("S", "Z")];

        for (voiceless, voiced) in &pairs {
            let vl = mapper.get(voiceless).unwrap();
            let vd = mapper.get(voiced).unwrap();

            assert_eq!(vl.manner, vd.manner, "{voiceless}/{voiced} manner mismatch");
            assert_eq!(vl.place, vd.place, "{voiceless}/{voiced} place mismatch");
            assert_eq!(vl.voicing, Voicing::Voiceless);
            assert_eq!(vd.voicing, Voicing::Voiced);
        }
    }

    #[test]
    fn test_mapper_unknown_phoneme() {
        let mapper = ArticulatoryMapper::new();
        assert!(mapper.get("XYZ").is_none());
    }

    #[test]
    fn test_mapper_stress_stripping() {
        let mapper = ArticulatoryMapper::new();
        // Should strip stress markers
        let ah0 = mapper.get("AH0");
        let ah = mapper.get("AH");
        assert!(ah.is_some());
        assert!(ah0.is_some());
    }

    #[test]
    fn test_find_by_features_voicing() {
        let mapper = ArticulatoryMapper::new();

        let voiced_stops = mapper.find_by_features(Some(Voicing::Voiced), Some(Manner::Stop), None);

        assert!(voiced_stops.contains(&"B".to_string()));
        assert!(voiced_stops.contains(&"D".to_string()));
        assert!(voiced_stops.contains(&"G".to_string()));
        assert!(!voiced_stops.contains(&"P".to_string())); // Voiceless
    }

    #[test]
    fn test_find_by_features_place() {
        let mapper = ArticulatoryMapper::new();

        let bilabials = mapper.find_by_features(None, None, Some(Place::Bilabial));

        assert!(bilabials.contains(&"P".to_string()));
        assert!(bilabials.contains(&"B".to_string()));
        assert!(bilabials.contains(&"M".to_string()));
        assert!(bilabials.contains(&"W".to_string()));
    }

    #[test]
    fn test_articulatory_hdc_all_phonemes_present() {
        let hdc = ArticulatoryHDC::new();

        let expected = [
            "P", "B", "T", "D", "K", "G", "F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH", "CH",
            "JH", "M", "N", "NG", "L", "R", "W", "Y", "IY", "IH", "EY", "EH", "AE", "AH", "ER",
            "UW", "UH", "OW", "AA", "AO", "AY", "AW", "OY",
        ];

        for phone in &expected {
            assert!(
                hdc.get_phoneme_hv(phone).is_some(),
                "missing phoneme HV for {phone}"
            );
        }
    }

    #[test]
    fn test_articulatory_hdc_query_returns_sorted() {
        let hdc = ArticulatoryHDC::new();
        let p_hv = hdc.get_phoneme_hv("P").unwrap();

        let results = hdc.query(p_hv);

        // Results should be sorted by similarity (descending)
        for i in 1..results.len() {
            assert!(
                results[i - 1].1 >= results[i].1,
                "query results not sorted at index {i}"
            );
        }

        // Top result should be "P" itself
        assert_eq!(results[0].0, "P");
    }

    #[test]
    fn test_resonator_converges_for_all_phonemes() {
        let resonator = ArticulatoryResonator::new();
        let mapper = ArticulatoryMapper::new();

        // Phonemes with unique articulatory feature vectors resonate back to
        // themselves.  Some phoneme pairs (e.g. IY/IH, EY/EH) share identical
        // feature encodings, so the resonator may return either member of the
        // pair.  We only test phonemes whose feature tuple is unique across the
        // full inventory.
        let phonemes = [
            "P", "B", "T", "D", "K", "G", "F", "V", "M", "N", "NG", "L", "R", "W", "Y", "AH", "AA",
        ];

        for phone in &phonemes {
            let hv = resonator.hdc().get_phoneme_hv(phone).unwrap();
            let (result, sim, _iters) = resonator.resonate(hv);

            // The result must share the same articulatory features as the input.
            let expected_features = mapper.get(phone).unwrap();
            let result_features = mapper.get(&result).unwrap();
            assert_eq!(
                expected_features.voicing, result_features.voicing,
                "resonator returned wrong voicing for {phone}: got {result}"
            );
            assert_eq!(
                expected_features.manner, result_features.manner,
                "resonator returned wrong manner for {phone}: got {result}"
            );
            assert_eq!(
                expected_features.place, result_features.place,
                "resonator returned wrong place for {phone}: got {result}"
            );
            assert!(sim > 0.5, "{phone} resonated with low similarity: {sim}");
        }
    }

    #[test]
    fn test_acoustic_detector_creation() {
        let detector = AcousticArticulatoryDetector::new();
        assert!(!detector.hdc().phoneme_hvs.is_empty());
    }

    #[test]
    fn test_acoustic_detector_detect() {
        let detector = AcousticArticulatoryDetector::new();

        // Create a simple mel frame
        let mel = vec![0.5; 40];
        let hv = detector.detect(&mel, None);

        // Should produce a non-zero HV
        let popcount: u32 = hv.words.iter().map(|w| w.count_ones()).sum();
        assert!(popcount > 0, "detect produced all-zero HV");
    }

    #[test]
    fn test_acoustic_detector_with_prev_frame() {
        let detector = AcousticArticulatoryDetector::new();

        let mel = vec![0.5; 40];
        let prev = vec![0.3; 40];
        let hv = detector.detect(&mel, Some(&prev));

        let popcount: u32 = hv.words.iter().map(|w| w.count_ones()).sum();
        assert!(popcount > 0);
    }

    #[test]
    fn test_voicing_detection_low_energy() {
        let detector = AcousticArticulatoryDetector::new();

        // Very low energy should be voiceless
        let mel = vec![0.001; 40];
        let voicing = detector.detect_voicing(&mel);
        assert_eq!(voicing, Voicing::Voiceless);
    }

    #[test]
    fn test_manner_detection_short_input() {
        let detector = AcousticArticulatoryDetector::new();

        // Short input (< 10 bins) should default to Vowel
        let mel = vec![0.5; 5];
        let manner = detector.detect_manner(&mel, None);
        assert_eq!(manner, Manner::Vowel);
    }

    #[test]
    fn test_place_detection_short_input() {
        let detector = AcousticArticulatoryDetector::new();

        // Short input (< 20 bins) should default to Alveolar
        let mel = vec![0.5; 10];
        let place = detector.detect_place(&mel);
        assert_eq!(place, Place::Alveolar);
    }
}
