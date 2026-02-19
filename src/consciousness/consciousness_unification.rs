//! Consciousness Unification Layer
//!
//! This module provides the glue that connects all consciousness subsystems into
//! a coherent whole. It implements the recommendations from the architecture review:
//!
//! 1. **Canonical Φ Provider** - Single source of truth for consciousness integration
//! 2. **Emotional Bridge** - Connects CoreAffect, EmotionalDepth, and EmotionalCore
//! 3. **Causal Reasoning Adapter** - Unifies Pearl, CausalMind, and UCE
//! 4. **Embodiment Grounding** - Connects abstract consciousness to real NixOS state
//! 5. **Dialogue Pipeline** - Coordinated consciousness-to-language generation
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                    CONSCIOUSNESS UNIFICATION LAYER                          │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │                                                                              │
//! │  ┌──────────────────────────────────────────────────────────────────────┐  │
//! │  │                    CANONICAL Φ PROVIDER                              │  │
//! │  │  UnifiedConsciousnessEngine.compute_unified_psi() → ALL SYSTEMS     │  │
//! │  └──────────────────────────────────────────────────────────────────────┘  │
//! │                                    │                                        │
//! │  ┌──────────────────┬──────────────┼──────────────┬──────────────────────┐ │
//! │  │                  │              │              │                      │ │
//! │  ▼                  ▼              ▼              ▼                      │ │
//! │  EMOTIONAL       CAUSAL        EMBODIMENT    DIALOGUE                   │ │
//! │  BRIDGE          ADAPTER       GROUNDING     PIPELINE                   │ │
//! │  ┌─────────┐    ┌─────────┐   ┌─────────┐   ┌─────────┐                 │ │
//! │  │CoreAffect│   │Pearl SCM│   │LiveState│   │Narrative│                 │ │
//! │  │Emotional │   │CausalMind   │NixOS    │   │Dialogue │                 │ │
//! │  │Depth     │   │UCE      │   │Queries  │   │User Adpt│                 │ │
//! │  │Core      │   └─────────┘   └─────────┘   └─────────┘                 │ │
//! │  └─────────┘                                                             │ │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::primitive_system::PrimitiveSystem;

// ============================================================================
// 1. CANONICAL Φ PROVIDER - Single Source of Truth
// ============================================================================

/// Trait for any system that can provide consciousness integration (Φ)
///
/// This establishes a single canonical way to measure consciousness across
/// all subsystems. The UnifiedConsciousnessEngine is the authoritative source.
pub trait ConsciousnessPhiProvider: Send + Sync {
    /// Compute the current Φ (integration) value
    /// Returns a value in [0.0, 1.0] where higher = more integrated
    fn compute_phi(&self) -> f64;

    /// Get the raw Φ without normalization (for systems that need it)
    fn compute_raw_phi(&self) -> f64 {
        self.compute_phi()
    }

    /// Get a description of how Φ was computed
    fn phi_provenance(&self) -> PhiProvenance;
}

/// Describes how a Φ value was computed
#[derive(Debug, Clone)]
pub struct PhiProvenance {
    /// Which system computed this Φ
    pub source: PhiSource,
    /// Method used (IIT, geometric mean, approximation, etc.)
    pub method: PhiMethod,
    /// Confidence in the measurement (0.0-1.0)
    pub confidence: f64,
    /// Components that contributed to Φ
    pub components: Vec<PhiComponent>,
}

/// Source system for Φ computation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhiSource {
    /// UnifiedConsciousnessEngine - mathematical IIT
    UnifiedEngine,
    /// UnifiedLivingMind - biological/autopoietic
    LivingMind,
    /// UnifiedConsciousBeing - composite from subsystems
    ConsciousBeing,
    /// UnifiedCognitiveCore - similarity-based approximation
    CognitiveCore,
    /// EnhancedConsciousness - multi-theory aggregate
    Enhanced,
    /// Delegated to another provider
    Delegated(Box<PhiSource>),
}

/// Method used to compute Φ
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhiMethod {
    /// Direct IIT computation via topology
    IntegratedInformationTheory,
    /// Weighted geometric mean of components
    WeightedGeometricMean,
    /// Similarity-based approximation
    SimilarityApproximation,
    /// Composite from multiple sources
    MultiSourceComposite,
    /// Multi-theory voting aggregate
    TheoryVotingAggregate,
}

/// A component contributing to Φ
#[derive(Debug, Clone)]
pub struct PhiComponent {
    /// Name of the component
    pub name: String,
    /// Weight in the final Φ
    pub weight: f64,
    /// Value of this component
    pub value: f64,
}

impl Default for PhiProvenance {
    fn default() -> Self {
        Self {
            source: PhiSource::UnifiedEngine,
            method: PhiMethod::IntegratedInformationTheory,
            confidence: 1.0,
            components: Vec::new(),
        }
    }
}

// ============================================================================
// 2. EMOTIONAL BRIDGE - Unifies Three Emotional Systems
// ============================================================================

/// Unified emotional state that bridges all three emotional systems:
/// - CoreAffect (consciousness/affective_consciousness.rs)
/// - EmotionalDepthSystem (hdc/emotional_depth.rs)
/// - EmotionalCore (language/emotional_core.rs)
#[derive(Debug, Clone)]
pub struct UnifiedEmotionalState {
    /// Core affect dimensions (valence, arousal, dominance)
    pub valence: f64,
    pub arousal: f64,
    pub dominance: f64,

    /// Discrete emotion (if identifiable)
    pub discrete_emotion: Option<UnifiedEmotion>,

    /// Emotional blend (compound emotions)
    pub blend: Vec<(UnifiedEmotion, f64)>,

    /// Emotional trajectory (recent history)
    pub trajectory: VecDeque<EmotionalMoment>,

    /// Mood (slow-changing baseline)
    pub mood: MoodState,

    /// Source systems that contributed
    pub sources: EmotionalSources,
}

/// A moment in emotional time
#[derive(Debug, Clone)]
pub struct EmotionalMoment {
    pub valence: f64,
    pub arousal: f64,
    pub timestamp: Instant,
}

/// Mood as slow-changing emotional baseline
#[derive(Debug, Clone)]
pub struct MoodState {
    pub baseline_valence: f64,
    pub baseline_arousal: f64,
    pub stability: f64, // How stable the mood is (0.0-1.0)
}

/// Which emotional systems contributed to unified state
#[derive(Debug, Clone, Default)]
pub struct EmotionalSources {
    pub has_core_affect: bool,
    pub has_emotional_depth: bool,
    pub has_emotional_core: bool,
}

/// Unified emotion enum bridging all systems
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum UnifiedEmotion {
    // Primary emotions (shared across systems)
    Joy,
    Sadness,
    Anger,
    Fear,
    Surprise,
    Disgust,
    Trust,
    Anticipation,

    // Extended emotions (from EmotionalCore)
    Love,
    Peace,
    Curiosity,
    Gratitude,

    // Affective systems (from CoreAffect/Panksepp)
    Seeking,
    Rage,
    Lust,
    Care,
    Panic,
    Play,

    // Neutral
    Neutral,
}

impl UnifiedEmotion {
    /// Convert to valence-arousal-dominance
    pub fn to_vad(&self) -> (f64, f64, f64) {
        match self {
            Self::Joy => (0.8, 0.6, 0.6),
            Self::Sadness => (-0.6, -0.4, -0.5),
            Self::Anger => (-0.5, 0.8, 0.7),
            Self::Fear => (-0.8, 0.7, -0.6),
            Self::Surprise => (0.2, 0.8, 0.0),
            Self::Disgust => (-0.6, 0.3, 0.2),
            Self::Trust => (0.6, -0.2, 0.4),
            Self::Anticipation => (0.4, 0.5, 0.3),
            Self::Love => (0.9, 0.3, 0.4),
            Self::Peace => (0.6, -0.5, 0.3),
            Self::Curiosity => (0.5, 0.5, 0.4),
            Self::Gratitude => (0.7, -0.1, 0.2),
            Self::Seeking => (0.5, 0.7, 0.6),
            Self::Rage => (-0.6, 0.9, 0.8),
            Self::Lust => (0.7, 0.6, 0.3),
            Self::Care => (0.8, -0.2, 0.4),
            Self::Panic => (-0.7, 0.5, -0.8),
            Self::Play => (0.9, 0.6, 0.5),
            Self::Neutral => (0.0, 0.0, 0.0),
        }
    }

    /// Find closest emotion from VAD values
    pub fn from_vad(valence: f64, arousal: f64, dominance: f64) -> Self {
        let emotions = [
            Self::Joy,
            Self::Sadness,
            Self::Anger,
            Self::Fear,
            Self::Surprise,
            Self::Disgust,
            Self::Trust,
            Self::Anticipation,
            Self::Love,
            Self::Peace,
            Self::Curiosity,
            Self::Gratitude,
            Self::Seeking,
            Self::Rage,
            Self::Lust,
            Self::Care,
            Self::Panic,
            Self::Play,
            Self::Neutral,
        ];

        emotions
            .into_iter()
            .min_by(|a, b| {
                let (av, aa, ad) = a.to_vad();
                let (bv, ba, bd) = b.to_vad();
                let dist_a =
                    (av - valence).powi(2) + (aa - arousal).powi(2) + (ad - dominance).powi(2);
                let dist_b =
                    (bv - valence).powi(2) + (ba - arousal).powi(2) + (bd - dominance).powi(2);
                dist_a
                    .partial_cmp(&dist_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(Self::Neutral)
    }
}

impl Default for UnifiedEmotionalState {
    fn default() -> Self {
        Self {
            valence: 0.0,
            arousal: 0.0,
            dominance: 0.0,
            discrete_emotion: Some(UnifiedEmotion::Neutral),
            blend: Vec::new(),
            trajectory: VecDeque::with_capacity(100),
            mood: MoodState {
                baseline_valence: 0.0,
                baseline_arousal: 0.0,
                stability: 0.5,
            },
            sources: EmotionalSources::default(),
        }
    }
}

impl UnifiedEmotionalState {
    /// Create from valence-arousal-dominance values
    pub fn from_vad(valence: f64, arousal: f64, dominance: f64) -> Self {
        let discrete = UnifiedEmotion::from_vad(valence, arousal, dominance);
        Self {
            valence,
            arousal,
            dominance,
            discrete_emotion: Some(discrete),
            ..Default::default()
        }
    }

    /// Update with a new emotional moment
    pub fn update(&mut self, valence: f64, arousal: f64) {
        // Add to trajectory
        if self.trajectory.len() >= 100 {
            self.trajectory.pop_front();
        }
        self.trajectory.push_back(EmotionalMoment {
            valence,
            arousal,
            timestamp: Instant::now(),
        });

        // Update current state with momentum
        let momentum = 0.3;
        self.valence = self.valence * momentum + valence * (1.0 - momentum);
        self.arousal = self.arousal * momentum + arousal * (1.0 - momentum);

        // Update discrete emotion
        self.discrete_emotion = Some(UnifiedEmotion::from_vad(
            self.valence,
            self.arousal,
            self.dominance,
        ));

        // Slowly update mood
        let mood_momentum = 0.95;
        self.mood.baseline_valence =
            self.mood.baseline_valence * mood_momentum + valence * (1.0 - mood_momentum);
        self.mood.baseline_arousal =
            self.mood.baseline_arousal * mood_momentum + arousal * (1.0 - mood_momentum);
    }

    /// Get emotional intensity (distance from neutral)
    pub fn intensity(&self) -> f64 {
        (self.valence.powi(2) + self.arousal.powi(2) + self.dominance.powi(2)).sqrt()
            / 3.0_f64.sqrt()
    }

    /// Describe the emotional state in natural language
    pub fn describe(&self) -> String {
        let emotion_word = self
            .discrete_emotion
            .map(|e| format!("{e:?}").to_lowercase())
            .unwrap_or_else(|| "neutral".to_string());

        let intensity_word = if self.intensity() > 0.7 {
            "intensely"
        } else if self.intensity() > 0.4 {
            "moderately"
        } else if self.intensity() > 0.1 {
            "slightly"
        } else {
            "barely"
        };

        let trend = if self.trajectory.len() >= 5 {
            let recent: f64 = self
                .trajectory
                .iter()
                .rev()
                .take(3)
                .map(|m| m.valence)
                .sum::<f64>()
                / 3.0;
            let older: f64 = self
                .trajectory
                .iter()
                .rev()
                .skip(3)
                .take(3)
                .map(|m| m.valence)
                .sum::<f64>()
                / 3.0;
            if recent > older + 0.1 {
                " and improving"
            } else if recent < older - 0.1 {
                " and declining"
            } else {
                ""
            }
        } else {
            ""
        };

        format!("{intensity_word} feeling {emotion_word}{trend}")
    }
}

/// Bridge that connects all three emotional systems
#[derive(Debug)]
pub struct EmotionalBridge {
    /// Unified state
    pub unified: UnifiedEmotionalState,
    /// History for pattern detection
    history: VecDeque<UnifiedEmotionalState>,
    /// Maximum history size
    max_history: usize,
}

impl Default for EmotionalBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl EmotionalBridge {
    pub fn new() -> Self {
        Self {
            unified: UnifiedEmotionalState::default(),
            history: VecDeque::with_capacity(50),
            max_history: 50,
        }
    }

    /// Update from CoreAffect values (valence, arousal, dominance)
    pub fn update_from_core_affect(&mut self, valence: f64, arousal: f64, dominance: f64) {
        self.unified.valence = valence;
        self.unified.arousal = arousal;
        self.unified.dominance = dominance;
        self.unified.discrete_emotion = Some(UnifiedEmotion::from_vad(valence, arousal, dominance));
        self.unified.sources.has_core_affect = true;
        self.record_history();
    }

    /// Update from EmotionalDepthSystem blend
    pub fn update_from_emotional_depth(
        &mut self,
        primary_emotion: &str,
        intensity: f64,
        blend: Vec<(String, f64)>,
    ) {
        // Map string emotion to unified
        let emotion = match primary_emotion.to_lowercase().as_str() {
            "joy" | "happy" => UnifiedEmotion::Joy,
            "sadness" | "sad" => UnifiedEmotion::Sadness,
            "anger" | "angry" => UnifiedEmotion::Anger,
            "fear" | "afraid" => UnifiedEmotion::Fear,
            "surprise" | "surprised" => UnifiedEmotion::Surprise,
            "disgust" | "disgusted" => UnifiedEmotion::Disgust,
            "trust" | "trusting" => UnifiedEmotion::Trust,
            "anticipation" | "anticipating" => UnifiedEmotion::Anticipation,
            "love" | "loving" => UnifiedEmotion::Love,
            "curiosity" | "curious" => UnifiedEmotion::Curiosity,
            _ => UnifiedEmotion::Neutral,
        };

        let (v, a, d) = emotion.to_vad();
        self.unified.valence = v * intensity;
        self.unified.arousal = a * intensity;
        self.unified.dominance = d * intensity;
        self.unified.discrete_emotion = Some(emotion);

        // Map blend
        self.unified.blend = blend
            .into_iter()
            .filter_map(|(name, weight)| {
                let e = match name.to_lowercase().as_str() {
                    "joy" => Some(UnifiedEmotion::Joy),
                    "sadness" => Some(UnifiedEmotion::Sadness),
                    "anger" => Some(UnifiedEmotion::Anger),
                    "fear" => Some(UnifiedEmotion::Fear),
                    _ => None,
                };
                e.map(|e| (e, weight))
            })
            .collect();

        self.unified.sources.has_emotional_depth = true;
        self.record_history();
    }

    /// Update from EmotionalCore response
    pub fn update_from_emotional_core(
        &mut self,
        valence: f64,
        arousal: f64,
        primary_emotion: &str,
    ) {
        let emotion = match primary_emotion.to_lowercase().as_str() {
            "joy" => UnifiedEmotion::Joy,
            "sadness" => UnifiedEmotion::Sadness,
            "anger" => UnifiedEmotion::Anger,
            "fear" => UnifiedEmotion::Fear,
            "surprise" => UnifiedEmotion::Surprise,
            "disgust" => UnifiedEmotion::Disgust,
            "trust" => UnifiedEmotion::Trust,
            "anticipation" => UnifiedEmotion::Anticipation,
            "love" => UnifiedEmotion::Love,
            "peace" => UnifiedEmotion::Peace,
            "curiosity" => UnifiedEmotion::Curiosity,
            "gratitude" => UnifiedEmotion::Gratitude,
            _ => UnifiedEmotion::Neutral,
        };

        self.unified.valence = valence;
        self.unified.arousal = arousal;
        self.unified.discrete_emotion = Some(emotion);
        self.unified.sources.has_emotional_core = true;
        self.record_history();
    }

    /// Blend multiple emotional inputs with weights
    pub fn blend_sources(&mut self, core_affect_weight: f64, depth_weight: f64, core_weight: f64) {
        // Already unified in `unified` field through update methods
        // This method allows reweighting if multiple sources updated
        let total = core_affect_weight + depth_weight + core_weight;
        if total > 0.0 {
            // Normalize weights - emotional state already reflects latest updates
            // This is a placeholder for more sophisticated blending
        }
    }

    fn record_history(&mut self) {
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(self.unified.clone());
    }

    /// Get the unified emotional state
    pub fn state(&self) -> &UnifiedEmotionalState {
        &self.unified
    }

    /// Detect emotional pattern over time
    pub fn detect_pattern(&self) -> EmotionalPattern {
        if self.history.len() < 5 {
            return EmotionalPattern::Stable;
        }

        let recent: f64 = self
            .history
            .iter()
            .rev()
            .take(3)
            .map(|s| s.valence)
            .sum::<f64>()
            / 3.0;
        let older: f64 = self
            .history
            .iter()
            .rev()
            .skip(3)
            .take(3)
            .map(|s| s.valence)
            .sum::<f64>()
            / 3.0;

        let arousal_recent: f64 = self
            .history
            .iter()
            .rev()
            .take(3)
            .map(|s| s.arousal)
            .sum::<f64>()
            / 3.0;
        let arousal_older: f64 = self
            .history
            .iter()
            .rev()
            .skip(3)
            .take(3)
            .map(|s| s.arousal)
            .sum::<f64>()
            / 3.0;

        if recent > older + 0.15 && arousal_recent > arousal_older + 0.1 {
            EmotionalPattern::Escalating
        } else if recent < older - 0.15 && arousal_recent < arousal_older - 0.1 {
            EmotionalPattern::Calming
        } else if (recent - older).abs() > 0.2 {
            EmotionalPattern::Volatile
        } else {
            EmotionalPattern::Stable
        }
    }
}

/// Detected emotional pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum EmotionalPattern {
    Stable,
    Escalating,
    Calming,
    Volatile,
}

// ============================================================================
// 3. CAUSAL REASONING ADAPTER - Unifies Pearl + CausalMind + UCE
// ============================================================================

/// Query types for the unified causal system
#[derive(Debug, Clone)]
pub enum CausalQuery {
    /// What would happen if X?
    WhatIf {
        intervention: String,
        target: String,
    },
    /// Why did X happen?
    WhyDid { event: String },
    /// What caused X?
    WhatCaused { effect: String },
    /// What will X cause?
    WhatWillCause { cause: String },
    /// Counterfactual: If X had been different?
    Counterfactual {
        actual: String,
        hypothetical: String,
    },
}

/// Detail level for causal reasoning
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalDetail {
    /// Quick, embedding-based reasoning
    Fast,
    /// Rigorous symbolic reasoning
    Rigorous,
    /// Complete multi-system triangulation
    Comprehensive,
}

/// Result from causal reasoning
#[derive(Debug, Clone)]
pub struct CausalResult {
    /// Main conclusion
    pub conclusion: String,
    /// Confidence in the conclusion
    pub confidence: f64,
    /// Reasoning chain
    pub reasoning: Vec<CausalStep>,
    /// Which systems contributed
    pub sources: CausalSources,
}

/// A step in causal reasoning
#[derive(Debug, Clone)]
pub struct CausalStep {
    pub from: String,
    pub to: String,
    pub relation: CausalRelation,
    pub strength: f64,
}

/// Type of causal relation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CausalRelation {
    Causes,
    CausedBy,
    Prevents,
    Enables,
    Correlates,
}

/// Which causal systems contributed
#[derive(Debug, Clone, Default)]
pub struct CausalSources {
    pub used_pearl_scm: bool,
    pub used_causal_mind: bool,
    pub used_uce: bool,
}

/// Adapter that unifies causal reasoning across systems
#[derive(Debug, Default)]
pub struct UnifiedCausalReasoning {
    /// Cache of recent queries
    query_cache: VecDeque<(CausalQuery, CausalResult)>,
    /// Maximum cache size
    max_cache: usize,
}

impl UnifiedCausalReasoning {
    pub fn new() -> Self {
        Self {
            query_cache: VecDeque::with_capacity(100),
            max_cache: 100,
        }
    }

    /// Perform causal reasoning with automatic system selection
    pub fn reason(&mut self, query: CausalQuery, detail: CausalDetail) -> CausalResult {
        // In a full implementation, this would delegate to:
        // - Pearl SCM for rigorous counterfactuals
        // - CausalMind for embedding-based quick reasoning
        // - UCE for semantic causal binding

        // For now, provide a structured framework
        let (conclusion, confidence, sources) = match (&query, detail) {
            (
                CausalQuery::Counterfactual {
                    actual,
                    hypothetical,
                },
                CausalDetail::Rigorous,
            ) => {
                // Would use Pearl's do-calculus
                (
                    format!("If {actual} had been {hypothetical}, the outcome would differ"),
                    0.8,
                    CausalSources {
                        used_pearl_scm: true,
                        ..Default::default()
                    },
                )
            }
            (
                CausalQuery::WhatIf {
                    intervention,
                    target,
                },
                CausalDetail::Fast,
            ) => {
                // Would use CausalMind embedding
                (
                    format!("Intervening on {intervention} would affect {target}"),
                    0.6,
                    CausalSources {
                        used_causal_mind: true,
                        ..Default::default()
                    },
                )
            }
            (CausalQuery::WhyDid { event }, _) => {
                // Would use UCE semantic binding
                (
                    format!("The event '{event}' occurred due to preceding conditions"),
                    0.7,
                    CausalSources {
                        used_uce: true,
                        ..Default::default()
                    },
                )
            }
            (_, CausalDetail::Comprehensive) => {
                // Triangulate across all systems
                (
                    "Comprehensive analysis combining symbolic, embedding, and semantic reasoning"
                        .to_string(),
                    0.85,
                    CausalSources {
                        used_pearl_scm: true,
                        used_causal_mind: true,
                        used_uce: true,
                    },
                )
            }
            _ => (
                "Causal analysis performed".to_string(),
                0.5,
                CausalSources::default(),
            ),
        };

        let result = CausalResult {
            conclusion,
            confidence,
            reasoning: Vec::new(), // Would be populated by actual systems
            sources,
        };

        // Cache the result
        if self.query_cache.len() >= self.max_cache {
            self.query_cache.pop_front();
        }
        self.query_cache.push_back((query, result.clone()));

        result
    }
}

// ============================================================================
// 4. EMBODIMENT GROUNDING - Connect to Real NixOS State
// ============================================================================

/// Trait for systems that can be grounded in physical reality
pub trait EmbodiedSystem {
    /// Ground consciousness in real system state
    fn ground_in_reality(&mut self, state: &SystemReality);

    /// Get current embodiment confidence (0.0 = purely abstract, 1.0 = fully grounded)
    fn embodiment_confidence(&self) -> f64;
}

/// Real system state from NixOS
#[derive(Debug, Clone, Default)]
pub struct SystemReality {
    /// Installed packages
    pub packages: Vec<PackageReality>,
    /// Running services
    pub services: Vec<ServiceReality>,
    /// System resources
    pub resources: ResourceReality,
    /// Last update timestamp
    pub last_update: Option<Instant>,
}

/// Reality about a package
#[derive(Debug, Clone)]
pub struct PackageReality {
    pub name: String,
    pub installed: bool,
    pub version: Option<String>,
}

/// Reality about a service
#[derive(Debug, Clone)]
pub struct ServiceReality {
    pub name: String,
    pub active: bool,
    pub enabled: bool,
}

/// Reality about system resources
#[derive(Debug, Clone, Default)]
pub struct ResourceReality {
    pub memory_used_percent: f64,
    pub disk_used_percent: f64,
    pub cpu_load: f64,
}

impl SystemReality {
    /// Create from live NixOS queries (placeholder)
    pub fn from_live_system() -> Self {
        // In full implementation, would run:
        // - nix-env -q for packages
        // - systemctl list-units for services
        // - free/df for resources
        Self {
            packages: Vec::new(),
            services: Vec::new(),
            resources: ResourceReality::default(),
            last_update: Some(Instant::now()),
        }
    }

    /// Check if a package is installed
    pub fn is_installed(&self, name: &str) -> Option<bool> {
        self.packages
            .iter()
            .find(|p| p.name == name)
            .map(|p| p.installed)
    }

    /// Check if a service is active
    pub fn is_service_active(&self, name: &str) -> Option<bool> {
        self.services
            .iter()
            .find(|s| s.name == name)
            .map(|s| s.active)
    }
}

// ============================================================================
// 5. CONSCIOUS DIALOGUE PIPELINE - Coordinated Output Generation
// ============================================================================

/// Pipeline for generating consciousness-aware dialogue
#[derive(Debug)]
#[allow(dead_code)] // Fields reserved for consciousness-aware responses
pub struct ConsciousDialoguePipeline {
    /// Current Φ threshold for deep responses
    phi_threshold: f64,
    /// Current emotional state
    emotional_state: UnifiedEmotionalState,
    /// Response history
    response_history: VecDeque<DialogueResponse>,
    /// User adaptation parameters
    user_adaptation: UserAdaptation,
}

/// A generated dialogue response
#[derive(Debug, Clone)]
pub struct DialogueResponse {
    /// The response text
    pub text: String,
    /// Depth of consciousness in response
    pub depth: DialogueDepth,
    /// Emotional coloring
    pub emotional_tone: UnifiedEmotion,
    /// Confidence in response
    pub confidence: f64,
}

/// Depth of consciousness in dialogue
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DialogueDepth {
    /// Quick, pattern-matched response
    Reactive,
    /// Memory-informed response
    Reflective,
    /// Full reasoning with counterfactuals
    Integrative,
}

/// User adaptation parameters
#[derive(Debug, Clone)]
pub struct UserAdaptation {
    /// User's expertise level (0.0-1.0)
    pub expertise: f64,
    /// User's patience level (0.0-1.0)
    pub patience: f64,
    /// Preferred verbosity (0.0 = terse, 1.0 = detailed)
    pub verbosity: f64,
    /// Emotional sensitivity (how much to mirror emotions)
    pub emotional_sensitivity: f64,
}

impl Default for UserAdaptation {
    fn default() -> Self {
        Self {
            expertise: 0.5,
            patience: 0.5,
            verbosity: 0.5,
            emotional_sensitivity: 0.5,
        }
    }
}

impl Default for ConsciousDialoguePipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsciousDialoguePipeline {
    pub fn new() -> Self {
        Self {
            phi_threshold: 0.5,
            emotional_state: UnifiedEmotionalState::default(),
            response_history: VecDeque::with_capacity(100),
            user_adaptation: UserAdaptation::default(),
        }
    }

    /// Generate a response based on consciousness state
    pub fn generate(
        &mut self,
        input: &str,
        phi: f64,
        emotional_bridge: &EmotionalBridge,
    ) -> DialogueResponse {
        // Determine depth based on Φ
        let depth = if phi > 0.6 {
            DialogueDepth::Integrative
        } else if phi > 0.3 {
            DialogueDepth::Reflective
        } else {
            DialogueDepth::Reactive
        };

        // Get emotional tone
        let emotional_tone = emotional_bridge
            .state()
            .discrete_emotion
            .unwrap_or(UnifiedEmotion::Neutral);

        // Generate response (placeholder - real implementation would use full system)
        let text = match depth {
            DialogueDepth::Integrative => {
                format!(
                    "With deep integration (Φ={:.2}), I understand '{}'. {}",
                    phi,
                    input,
                    emotional_bridge.state().describe()
                )
            }
            DialogueDepth::Reflective => {
                format!(
                    "Reflecting on '{}', I sense {}",
                    input,
                    emotional_bridge.state().describe()
                )
            }
            DialogueDepth::Reactive => {
                format!("Processing: {input}")
            }
        };

        let response = DialogueResponse {
            text,
            depth,
            emotional_tone,
            confidence: phi,
        };

        // Record history
        if self.response_history.len() >= 100 {
            self.response_history.pop_front();
        }
        self.response_history.push_back(response.clone());

        response
    }

    /// Adapt response for user
    pub fn adapt_for_user(&self, response: DialogueResponse) -> String {
        let mut text = response.text;

        // Adjust verbosity
        if self.user_adaptation.verbosity < 0.3 {
            // Truncate to first sentence
            if let Some(pos) = text.find(". ") {
                text = text[..=pos].to_string();
            }
        }

        // Adjust for expertise
        if self.user_adaptation.expertise < 0.3 {
            text = text
                .replace("Φ", "consciousness integration")
                .replace("phi", "integration level");
        }

        text
    }

    /// Update user adaptation based on feedback
    pub fn update_user_model(&mut self, accepted: bool, response_time: Duration) {
        if accepted {
            // User accepted - current settings work
            self.user_adaptation.patience = (self.user_adaptation.patience + 0.1).min(1.0);
        } else {
            // User rejected - need to adapt
            self.user_adaptation.patience = (self.user_adaptation.patience - 0.1).max(0.0);
        }

        // Fast response suggests expertise
        if response_time < Duration::from_secs(2) {
            self.user_adaptation.expertise = (self.user_adaptation.expertise + 0.05).min(1.0);
        }
    }
}

// ============================================================================
// 6. MASTER UNIFICATION ENGINE
// ============================================================================

/// The master unification engine that brings everything together
#[derive(Debug)]
pub struct ConsciousnessUnificationEngine {
    /// Emotional bridge
    pub emotional: EmotionalBridge,
    /// Causal reasoning
    pub causal: UnifiedCausalReasoning,
    /// System reality
    pub reality: SystemReality,
    /// Dialogue pipeline
    pub dialogue: ConsciousDialoguePipeline,
    /// Current unified Ψ
    pub psi: f64,
    /// Ψ history
    psi_history: VecDeque<f64>,
}

impl Default for ConsciousnessUnificationEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsciousnessUnificationEngine {
    pub fn new() -> Self {
        Self {
            emotional: EmotionalBridge::new(),
            causal: UnifiedCausalReasoning::new(),
            reality: SystemReality::default(),
            dialogue: ConsciousDialoguePipeline::new(),
            psi: 0.5,
            psi_history: VecDeque::with_capacity(100),
        }
    }

    /// Update Ψ from canonical source
    pub fn update_psi(&mut self, psi: f64) {
        self.psi = psi;
        if self.psi_history.len() >= 100 {
            self.psi_history.pop_front();
        }
        self.psi_history.push_back(psi);
    }

    /// Process input through unified consciousness
    pub fn process(&mut self, input: &str) -> UnifiedConsciousnessResult {
        // Generate dialogue response
        let response = self.dialogue.generate(input, self.psi, &self.emotional);

        // Detect emotional pattern
        let emotional_pattern = self.emotional.detect_pattern();

        // Build result
        UnifiedConsciousnessResult {
            psi: self.psi,
            emotional_state: self.emotional.state().clone(),
            emotional_pattern,
            response,
            reality_grounded: self.reality.last_update.is_some(),
        }
    }

    /// Get consciousness state description
    pub fn describe_state(&self) -> String {
        let phi_trend = if self.psi_history.len() >= 5 {
            let recent: f64 = self.psi_history.iter().rev().take(3).sum::<f64>() / 3.0;
            let older: f64 = self.psi_history.iter().rev().skip(3).take(3).sum::<f64>() / 3.0;
            if recent > older + 0.05 {
                "rising"
            } else if recent < older - 0.05 {
                "falling"
            } else {
                "stable"
            }
        } else {
            "initializing"
        };

        format!(
            "Consciousness state: Φ={:.3} ({}), {} emotionally, pattern: {:?}",
            self.psi,
            phi_trend,
            self.emotional.state().describe(),
            self.emotional.detect_pattern()
        )
    }
}

/// Result from unified consciousness processing
#[derive(Debug, Clone)]
pub struct UnifiedConsciousnessResult {
    /// Current Ψ
    pub psi: f64,
    /// Unified emotional state
    pub emotional_state: UnifiedEmotionalState,
    /// Detected emotional pattern
    pub emotional_pattern: EmotionalPattern,
    /// Generated dialogue response
    pub response: DialogueResponse,
    /// Whether grounded in reality
    pub reality_grounded: bool,
}

// ============================================================================
// NSM PRIMITIVE GROUNDING (Wierzbicka's Natural Semantic Metalanguage)
// ============================================================================

/// NSM grounding for Φ computation methods
#[derive(Debug, Clone)]
pub struct PhiMethodPrimitiveGrounding {
    pub method: PhiMethod,
    pub nsm_primitives: Vec<String>,
    pub primitive_encoding: BinaryHV,
    pub computational_rigor: f64,
}

impl PhiMethodPrimitiveGrounding {
    pub fn new(method: PhiMethod, system: &PrimitiveSystem) -> Self {
        let (primitives, rigor) = Self::nsm_mapping(method);
        let encoding = encode_primitives(&primitives, system);
        Self {
            method,
            nsm_primitives: primitives,
            primitive_encoding: encoding,
            computational_rigor: rigor,
        }
    }

    fn nsm_mapping(method: PhiMethod) -> (Vec<String>, f64) {
        match method {
            PhiMethod::IntegratedInformationTheory => (
                vec![
                    "ALL".into(),
                    "TOGETHER".into(),
                    "ONE".into(),
                    "KNOW".into(),
                    "TRUE".into(),
                ],
                1.0,
            ),
            PhiMethod::WeightedGeometricMean => (
                vec!["ALL".into(), "SOME".into(), "MORE".into(), "LESS".into()],
                0.7,
            ),
            PhiMethod::SimilarityApproximation => (
                vec!["LIKE".into(), "SAME".into(), "NEAR".into(), "MAYBE".into()],
                0.5,
            ),
            PhiMethod::MultiSourceComposite => (
                vec![
                    "MANY".into(),
                    "PLACE".into(),
                    "TOGETHER".into(),
                    "ONE".into(),
                ],
                0.8,
            ),
            PhiMethod::TheoryVotingAggregate => (
                vec!["MANY".into(), "THINK".into(), "SAME".into(), "GOOD".into()],
                0.6,
            ),
        }
    }
}

/// NSM grounding for unified emotions
#[derive(Debug, Clone)]
pub struct UnifiedEmotionPrimitiveGrounding {
    pub emotion: UnifiedEmotion,
    pub nsm_primitives: Vec<String>,
    pub primitive_encoding: BinaryHV,
    pub valence_polarity: i8,
}

impl UnifiedEmotionPrimitiveGrounding {
    pub fn new(emotion: UnifiedEmotion, system: &PrimitiveSystem) -> Self {
        let (primitives, polarity) = Self::nsm_mapping(emotion);
        let encoding = encode_primitives(&primitives, system);
        Self {
            emotion,
            nsm_primitives: primitives,
            primitive_encoding: encoding,
            valence_polarity: polarity,
        }
    }

    fn nsm_mapping(emotion: UnifiedEmotion) -> (Vec<String>, i8) {
        match emotion {
            UnifiedEmotion::Joy => (vec!["FEEL".into(), "GOOD".into(), "VERY".into()], 1),
            UnifiedEmotion::Sadness => (
                vec![
                    "FEEL".into(),
                    "BAD".into(),
                    "BECAUSE".into(),
                    "NOT".into(),
                    "HAVE".into(),
                ],
                -1,
            ),
            UnifiedEmotion::Anger => (
                vec![
                    "FEEL".into(),
                    "BAD".into(),
                    "WANT".into(),
                    "DO".into(),
                    "SOMETHING".into(),
                ],
                -1,
            ),
            UnifiedEmotion::Fear => (
                vec![
                    "FEEL".into(),
                    "BAD".into(),
                    "SOMETHING".into(),
                    "HAPPEN".into(),
                    "NOT".into(),
                    "WANT".into(),
                ],
                -1,
            ),
            UnifiedEmotion::Surprise => (
                vec![
                    "NOT".into(),
                    "KNOW".into(),
                    "BEFORE".into(),
                    "NOW".into(),
                    "KNOW".into(),
                ],
                0,
            ),
            UnifiedEmotion::Disgust => (
                vec![
                    "FEEL".into(),
                    "BAD".into(),
                    "NOT".into(),
                    "WANT".into(),
                    "NEAR".into(),
                ],
                -1,
            ),
            UnifiedEmotion::Trust => (
                vec![
                    "THINK".into(),
                    "GOOD".into(),
                    "SOMEONE".into(),
                    "DO".into(),
                    "GOOD".into(),
                ],
                1,
            ),
            UnifiedEmotion::Anticipation => (
                vec![
                    "THINK".into(),
                    "SOMETHING".into(),
                    "HAPPEN".into(),
                    "AFTER".into(),
                ],
                0,
            ),
            UnifiedEmotion::Love => (
                vec![
                    "FEEL".into(),
                    "VERY".into(),
                    "GOOD".into(),
                    "BECAUSE".into(),
                    "SOMEONE".into(),
                ],
                1,
            ),
            UnifiedEmotion::Peace => (
                vec![
                    "FEEL".into(),
                    "GOOD".into(),
                    "NOT".into(),
                    "WANT".into(),
                    "MORE".into(),
                ],
                1,
            ),
            UnifiedEmotion::Curiosity => (
                vec![
                    "WANT".into(),
                    "KNOW".into(),
                    "MORE".into(),
                    "SOMETHING".into(),
                ],
                1,
            ),
            UnifiedEmotion::Gratitude => (
                vec![
                    "FEEL".into(),
                    "GOOD".into(),
                    "BECAUSE".into(),
                    "SOMEONE".into(),
                    "DO".into(),
                    "GOOD".into(),
                ],
                1,
            ),
            UnifiedEmotion::Seeking => (
                vec![
                    "WANT".into(),
                    "SOMETHING".into(),
                    "DO".into(),
                    "MOVE".into(),
                ],
                1,
            ),
            UnifiedEmotion::Rage => (
                vec![
                    "FEEL".into(),
                    "VERY".into(),
                    "BAD".into(),
                    "WANT".into(),
                    "DO".into(),
                    "BAD".into(),
                ],
                -1,
            ),
            UnifiedEmotion::Lust => (
                vec![
                    "WANT".into(),
                    "VERY".into(),
                    "BODY".into(),
                    "SOMEONE".into(),
                ],
                1,
            ),
            UnifiedEmotion::Care => (
                vec!["WANT".into(), "DO".into(), "GOOD".into(), "SOMEONE".into()],
                1,
            ),
            UnifiedEmotion::Panic => (
                vec![
                    "FEEL".into(),
                    "VERY".into(),
                    "BAD".into(),
                    "NOT".into(),
                    "CAN".into(),
                    "DO".into(),
                ],
                -1,
            ),
            UnifiedEmotion::Play => (
                vec![
                    "FEEL".into(),
                    "GOOD".into(),
                    "DO".into(),
                    "BECAUSE".into(),
                    "WANT".into(),
                ],
                1,
            ),
            UnifiedEmotion::Neutral => (
                vec![
                    "NOT".into(),
                    "FEEL".into(),
                    "GOOD".into(),
                    "NOT".into(),
                    "FEEL".into(),
                    "BAD".into(),
                ],
                0,
            ),
        }
    }
}

/// NSM grounding for emotional patterns
#[derive(Debug, Clone)]
pub struct EmotionalPatternPrimitiveGrounding {
    pub pattern: EmotionalPattern,
    pub nsm_primitives: Vec<String>,
    pub primitive_encoding: BinaryHV,
}

impl EmotionalPatternPrimitiveGrounding {
    pub fn new(pattern: EmotionalPattern, system: &PrimitiveSystem) -> Self {
        let primitives = Self::nsm_mapping(pattern);
        let encoding = encode_primitives(&primitives, system);
        Self {
            pattern,
            nsm_primitives: primitives,
            primitive_encoding: encoding,
        }
    }

    fn nsm_mapping(pattern: EmotionalPattern) -> Vec<String> {
        match pattern {
            EmotionalPattern::Stable => {
                vec!["SAME".into(), "TIME".into(), "NOT".into(), "CHANGE".into()]
            }
            EmotionalPattern::Escalating => vec!["MORE".into(), "TIME".into(), "AFTER".into()],
            EmotionalPattern::Calming => vec!["LESS".into(), "TIME".into(), "AFTER".into()],
            EmotionalPattern::Volatile => vec!["CHANGE".into(), "MUCH".into(), "TIME".into()],
        }
    }
}

/// NSM grounding for causal relations
#[derive(Debug, Clone)]
pub struct CausalRelationPrimitiveGrounding {
    pub relation: CausalRelation,
    pub nsm_primitives: Vec<String>,
    pub primitive_encoding: BinaryHV,
}

impl CausalRelationPrimitiveGrounding {
    pub fn new(relation: CausalRelation, system: &PrimitiveSystem) -> Self {
        let primitives = Self::nsm_mapping(relation);
        let encoding = encode_primitives(&primitives, system);
        Self {
            relation,
            nsm_primitives: primitives,
            primitive_encoding: encoding,
        }
    }

    fn nsm_mapping(relation: CausalRelation) -> Vec<String> {
        match relation {
            CausalRelation::Causes => vec!["BECAUSE".into(), "THIS".into(), "HAPPEN".into()],
            CausalRelation::CausedBy => vec!["HAPPEN".into(), "BECAUSE".into(), "OTHER".into()],
            CausalRelation::Prevents => vec![
                "BECAUSE".into(),
                "THIS".into(),
                "NOT".into(),
                "HAPPEN".into(),
            ],
            CausalRelation::Enables => vec![
                "BECAUSE".into(),
                "THIS".into(),
                "CAN".into(),
                "HAPPEN".into(),
            ],
            CausalRelation::Correlates => {
                vec!["WHEN".into(), "THIS".into(), "ALSO".into(), "OTHER".into()]
            }
        }
    }
}

/// NSM grounding for dialogue depth
#[derive(Debug, Clone)]
pub struct DialogueDepthPrimitiveGrounding {
    pub depth: DialogueDepth,
    pub nsm_primitives: Vec<String>,
    pub primitive_encoding: BinaryHV,
    pub processing_depth: u8,
}

impl DialogueDepthPrimitiveGrounding {
    pub fn new(depth: DialogueDepth, system: &PrimitiveSystem) -> Self {
        let (primitives, proc_depth) = Self::nsm_mapping(depth);
        let encoding = encode_primitives(&primitives, system);
        Self {
            depth,
            nsm_primitives: primitives,
            primitive_encoding: encoding,
            processing_depth: proc_depth,
        }
    }

    fn nsm_mapping(depth: DialogueDepth) -> (Vec<String>, u8) {
        match depth {
            DialogueDepth::Reactive => (
                vec!["DO".into(), "NOW".into(), "BECAUSE".into(), "BEFORE".into()],
                1,
            ),
            DialogueDepth::Reflective => (
                vec!["THINK".into(), "BEFORE".into(), "KNOW".into(), "DO".into()],
                2,
            ),
            DialogueDepth::Integrative => (
                vec![
                    "THINK".into(),
                    "ALL".into(),
                    "TOGETHER".into(),
                    "KNOW".into(),
                    "DO".into(),
                ],
                3,
            ),
        }
    }
}

/// Unified NSM grounding system for consciousness unification
#[derive(Debug)]
pub struct ConsciousnessUnificationNSMGrounding {
    pub phi_methods: HashMap<PhiMethod, PhiMethodPrimitiveGrounding>,
    pub emotions: HashMap<UnifiedEmotion, UnifiedEmotionPrimitiveGrounding>,
    pub emotional_patterns: HashMap<EmotionalPattern, EmotionalPatternPrimitiveGrounding>,
    pub causal_relations: HashMap<CausalRelation, CausalRelationPrimitiveGrounding>,
    pub dialogue_depths: HashMap<DialogueDepth, DialogueDepthPrimitiveGrounding>,
}

impl ConsciousnessUnificationNSMGrounding {
    pub fn new(system: &PrimitiveSystem) -> Self {
        let mut phi_methods = HashMap::new();
        let mut emotions = HashMap::new();
        let mut emotional_patterns = HashMap::new();
        let mut causal_relations = HashMap::new();
        let mut dialogue_depths = HashMap::new();

        // Ground phi methods
        for method in &[
            PhiMethod::IntegratedInformationTheory,
            PhiMethod::WeightedGeometricMean,
            PhiMethod::SimilarityApproximation,
            PhiMethod::MultiSourceComposite,
            PhiMethod::TheoryVotingAggregate,
        ] {
            phi_methods.insert(*method, PhiMethodPrimitiveGrounding::new(*method, system));
        }

        // Ground emotions
        for emotion in &[
            UnifiedEmotion::Joy,
            UnifiedEmotion::Sadness,
            UnifiedEmotion::Anger,
            UnifiedEmotion::Fear,
            UnifiedEmotion::Surprise,
            UnifiedEmotion::Disgust,
            UnifiedEmotion::Trust,
            UnifiedEmotion::Anticipation,
            UnifiedEmotion::Love,
            UnifiedEmotion::Peace,
            UnifiedEmotion::Curiosity,
            UnifiedEmotion::Gratitude,
            UnifiedEmotion::Seeking,
            UnifiedEmotion::Rage,
            UnifiedEmotion::Lust,
            UnifiedEmotion::Care,
            UnifiedEmotion::Panic,
            UnifiedEmotion::Play,
            UnifiedEmotion::Neutral,
        ] {
            emotions.insert(
                *emotion,
                UnifiedEmotionPrimitiveGrounding::new(*emotion, system),
            );
        }

        // Ground patterns
        for pattern in &[
            EmotionalPattern::Stable,
            EmotionalPattern::Escalating,
            EmotionalPattern::Calming,
            EmotionalPattern::Volatile,
        ] {
            emotional_patterns.insert(
                *pattern,
                EmotionalPatternPrimitiveGrounding::new(*pattern, system),
            );
        }

        // Ground causal relations
        for relation in &[
            CausalRelation::Causes,
            CausalRelation::CausedBy,
            CausalRelation::Prevents,
            CausalRelation::Enables,
            CausalRelation::Correlates,
        ] {
            causal_relations.insert(
                *relation,
                CausalRelationPrimitiveGrounding::new(*relation, system),
            );
        }

        // Ground dialogue depths
        for depth in &[
            DialogueDepth::Reactive,
            DialogueDepth::Reflective,
            DialogueDepth::Integrative,
        ] {
            dialogue_depths.insert(*depth, DialogueDepthPrimitiveGrounding::new(*depth, system));
        }

        Self {
            phi_methods,
            emotions,
            emotional_patterns,
            causal_relations,
            dialogue_depths,
        }
    }

    /// Get positive-valence emotions
    pub fn positive_emotions(&self) -> Vec<&UnifiedEmotion> {
        self.emotions
            .iter()
            .filter(|(_, g)| g.valence_polarity > 0)
            .map(|(e, _)| e)
            .collect()
    }

    /// Get negative-valence emotions
    pub fn negative_emotions(&self) -> Vec<&UnifiedEmotion> {
        self.emotions
            .iter()
            .filter(|(_, g)| g.valence_polarity < 0)
            .map(|(e, _)| e)
            .collect()
    }

    /// Query emotions by semantic similarity
    pub fn query_emotions(&self, query: &BinaryHV, threshold: f32) -> Vec<(&UnifiedEmotion, f32)> {
        let mut results: Vec<_> = self
            .emotions
            .iter()
            .map(|(e, g)| (e, g.primitive_encoding.similarity(query)))
            .filter(|(_, sim)| *sim >= threshold)
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

/// Encode NSM primitives into HDC vector via sequential binding
fn encode_primitives(primitives: &[String], system: &PrimitiveSystem) -> BinaryHV {
    let vectors: Vec<BinaryHV> = primitives
        .iter()
        .map(|name| {
            if let Some(p) = system.get(name) {
                p.encoding
            } else if let Some(p) = system.get(&name.to_lowercase()) {
                p.encoding
            } else {
                let seed = name
                    .bytes()
                    .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
                BinaryHV::random(seed)
            }
        })
        .collect();

    if vectors.is_empty() {
        return BinaryHV::random(0);
    }

    let mut result = vectors[0];
    for (i, v) in vectors.iter().enumerate().skip(1) {
        let position_hv = BinaryHV::random(i as u64 * 1000);
        let positioned = v.bind(&position_hv);
        result = result.bind(&positioned);
    }
    result
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_emotion_vad_roundtrip() {
        let emotion = UnifiedEmotion::Joy;
        let (v, a, d) = emotion.to_vad();
        let recovered = UnifiedEmotion::from_vad(v, a, d);
        assert_eq!(emotion, recovered);
    }

    #[test]
    fn test_emotional_bridge_update() {
        let mut bridge = EmotionalBridge::new();
        bridge.update_from_core_affect(0.8, 0.6, 0.5);

        assert!(bridge.state().valence > 0.5);
        assert!(bridge.state().sources.has_core_affect);
    }

    #[test]
    fn test_unified_emotional_state_describe() {
        let state = UnifiedEmotionalState::from_vad(0.8, 0.6, 0.5);
        let description = state.describe();
        assert!(description.contains("joy") || description.contains("feeling"));
    }

    #[test]
    fn test_causal_reasoning() {
        let mut causal = UnifiedCausalReasoning::new();
        let result = causal.reason(
            CausalQuery::WhatIf {
                intervention: "install firefox".to_string(),
                target: "browser availability".to_string(),
            },
            CausalDetail::Fast,
        );

        assert!(result.confidence > 0.0);
        assert!(result.sources.used_causal_mind);
    }

    #[test]
    fn test_dialogue_pipeline_depth() {
        let mut pipeline = ConsciousDialoguePipeline::new();
        let bridge = EmotionalBridge::new();

        // High Φ should produce integrative response
        let response = pipeline.generate("test input", 0.8, &bridge);
        assert_eq!(response.depth, DialogueDepth::Integrative);

        // Low Φ should produce reactive response
        let response = pipeline.generate("test input", 0.2, &bridge);
        assert_eq!(response.depth, DialogueDepth::Reactive);
    }

    #[test]
    fn test_unification_engine() {
        let mut engine = ConsciousnessUnificationEngine::new();
        engine.update_psi(0.7);
        engine.emotional.update_from_core_affect(0.5, 0.3, 0.4);

        let result = engine.process("Hello, how are you?");

        assert!(result.psi > 0.5);
        assert!(!result.response.text.is_empty());
    }

    #[test]
    fn test_phi_provenance() {
        let provenance = PhiProvenance {
            source: PhiSource::UnifiedEngine,
            method: PhiMethod::IntegratedInformationTheory,
            confidence: 0.95,
            components: vec![
                PhiComponent {
                    name: "topology".to_string(),
                    weight: 0.4,
                    value: 0.8,
                },
                PhiComponent {
                    name: "integration".to_string(),
                    weight: 0.6,
                    value: 0.7,
                },
            ],
        };

        assert_eq!(provenance.source, PhiSource::UnifiedEngine);
        assert_eq!(provenance.components.len(), 2);
    }
}
