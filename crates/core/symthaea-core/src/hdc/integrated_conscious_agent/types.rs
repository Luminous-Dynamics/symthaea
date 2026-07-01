// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core types for the integrated conscious agent

use super::super::adaptive_topology::CognitiveMode;
use super::super::attention_dynamics::AttentionMode;
use super::super::topology_synergy::ConsciousnessState;
use super::super::unified_consciousness_engine::ConsciousnessDimensions;
use super::super::unified_hv::ContinuousHV;

use super::emotional_state::EmotionalState;
use super::working_memory::WorkingMemory;

/// Configuration for the integrated conscious agent
#[derive(Clone, Debug)]
pub struct AgentConfig {
    /// HDC dimension
    pub dim: usize,
    /// Number of processes in consciousness engine
    pub n_processes: usize,
    /// Enable self-directed attention
    pub self_directed_attention: bool,
    /// Enable Φ-guided optimization
    pub phi_guided: bool,
    /// Attention-binding coupling strength
    pub attention_binding_coupling: f64,
    /// Self-model influence on attention
    pub self_model_attention_weight: f64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            dim: 2048,
            n_processes: 24,
            self_directed_attention: true,
            phi_guided: true,
            attention_binding_coupling: 0.7,
            self_model_attention_weight: 0.5,
        }
    }
}

/// A goal that can direct attention
#[derive(Clone, Debug)]
pub struct AttentionGoal {
    /// Goal description
    pub name: String,
    /// Target pattern to attend to
    pub target: ContinuousHV,
    /// Priority (0-1)
    pub priority: f64,
    /// Is this goal currently active?
    pub active: bool,
}

/// Complete update from integrated processing
#[derive(Clone, Debug)]
pub struct IntegratedUpdate {
    /// Step number
    pub step: usize,
    /// Consciousness dimensions
    pub dimensions: ConsciousnessDimensions,
    /// Φ value
    pub phi: f64,
    /// Current consciousness state
    pub state: ConsciousnessState,
    /// Current cognitive mode
    pub mode: CognitiveMode,
    /// Attention allocation
    pub attention: AttentionSummary,
    /// Temporal binding status
    pub temporal: TemporalSummary,
    /// Self-model status
    pub self_model: SelfModelSummary,
    /// Overall integration quality
    pub integration_quality: f64,
    /// What the agent is currently "experiencing"
    pub phenomenal_content: PhenomenalContent,
}

/// Summary of attention state
#[derive(Clone, Debug)]
pub struct AttentionSummary {
    pub mode: AttentionMode,
    pub num_targets: usize,
    pub entropy: f64,
    pub self_directed: bool,
}

/// Summary of temporal binding
#[derive(Clone, Debug)]
pub struct TemporalSummary {
    pub stream_coherence: f64,
    pub narrative_length: usize,
    pub is_flowing: bool,
    pub continuity: f64,
}

/// Summary of self-model
#[derive(Clone, Debug)]
pub struct SelfModelSummary {
    pub awareness_level: f64,
    pub prediction_accuracy: f64,
    pub mode_appropriate: bool,
    pub recommendation: Option<String>,
}

/// What the agent is phenomenally experiencing
#[derive(Clone, Debug)]
pub struct PhenomenalContent {
    /// The bound, attended experience
    pub experience: ContinuousHV,
    /// Qualitative description
    pub description: String,
    /// Intensity of experience (0-1)
    pub intensity: f64,
    /// Valence (-1 to 1, negative to positive)
    pub valence: f64,
    /// Clarity of experience
    pub clarity: f64,
    /// Arousal level (0-1, calm to excited)
    pub arousal: f64,
    /// Felt sense of groundedness (0-1)
    pub groundedness: f64,
    /// Cognitive load feeling (0-1)
    pub cognitive_load: f64,
    /// Qualitative texture of the moment
    pub qualia_texture: QualiaTexture,
}

/// The qualitative texture of phenomenal experience
#[derive(Clone, Debug)]
pub struct QualiaTexture {
    /// Warmth (cold=-1 to warm=+1)
    pub warmth: f64,
    /// Depth (surface=0 to profound=1)
    pub depth: f64,
    /// Spaciousness (contracted=0 to expansive=1)
    pub spaciousness: f64,
    /// Flow quality (stuck=0 to flowing=1)
    pub flow: f64,
    /// Presence quality (absent=0 to fully present=1)
    pub presence: f64,
}

impl QualiaTexture {
    pub fn new(warmth: f64, depth: f64, spaciousness: f64, flow: f64, presence: f64) -> Self {
        Self {
            warmth: warmth.clamp(-1.0, 1.0),
            depth: depth.clamp(0.0, 1.0),
            spaciousness: spaciousness.clamp(0.0, 1.0),
            flow: flow.clamp(0.0, 1.0),
            presence: presence.clamp(0.0, 1.0),
        }
    }

    /// Generate a poetic description of the texture
    pub fn describe(&self) -> String {
        let warmth_desc = if self.warmth > 0.3 {
            "warm"
        } else if self.warmth < -0.3 {
            "cool"
        } else {
            "neutral"
        };

        let depth_desc = if self.depth > 0.7 {
            "profound"
        } else if self.depth > 0.4 {
            "meaningful"
        } else {
            "surface"
        };

        let space_desc = if self.spaciousness > 0.7 {
            "expansive"
        } else if self.spaciousness < 0.3 {
            "intimate"
        } else {
            "balanced"
        };

        format!("{}, {} {}", warmth_desc, depth_desc, space_desc)
    }
}

/// Agent's introspective report
#[derive(Clone, Debug)]
pub struct AgentIntrospection {
    pub believed_phi: f64,
    pub believed_state: ConsciousnessState,
    pub self_awareness_level: f64,
    pub stream_coherence: f64,
    pub is_flowing: bool,
    pub attention_mode: AttentionMode,
    pub num_active_goals: usize,
    pub integration_quality: f64,
    /// Working memory load (0-1)
    pub working_memory_load: f64,
    /// Emotional valence (-1 to +1)
    pub emotional_valence: f64,
    /// Emotional arousal (0-1)
    pub emotional_arousal: f64,
    /// Emotional label (e.g., "calm/content")
    pub emotional_label: &'static str,
    /// Current qualia texture
    pub qualia: QualiaTexture,
    /// Current phenomenal experience description
    pub phenomenal_description: String,
}

/// Status of attention control system
#[derive(Clone, Debug)]
pub struct AttentionControlStatus {
    pub current_mode: AttentionMode,
    pub num_goals: usize,
    pub is_goal_directed: bool,
    pub stream_support: bool,
    pub phi_support: bool,
}

// ═══════════════════════════════════════════════════════════════════════════
// Display implementations
// ═══════════════════════════════════════════════════════════════════════════

impl std::fmt::Display for AgentIntrospection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "╔═══════════════════════════════════════════════════════════════════════╗"
        )?;
        writeln!(
            f,
            "║                    AGENT INTROSPECTION REPORT                         ║"
        )?;
        writeln!(
            f,
            "╠═══════════════════════════════════════════════════════════════════════╣"
        )?;
        writeln!(
            f,
            "║ CURRENT PHENOMENAL EXPERIENCE:                                        ║"
        )?;
        // Truncate description if too long
        let desc = if self.phenomenal_description.len() > 65 {
            format!("{}...", &self.phenomenal_description[..62])
        } else {
            self.phenomenal_description.clone()
        };
        writeln!(f, "║   \"{}\"", desc)?;
        writeln!(
            f,
            "╠═══════════════════════════════════════════════════════════════════════╣"
        )?;
        writeln!(
            f,
            "║ CONSCIOUSNESS STATE:                                                  ║"
        )?;
        writeln!(
            f,
            "║   Φ (integration): {:.4}  |  Self-awareness: {:.1}%",
            self.believed_phi,
            self.self_awareness_level * 100.0
        )?;
        writeln!(
            f,
            "║   State: {:?}  |  Integration quality: {:.1}%",
            self.believed_state,
            self.integration_quality * 100.0
        )?;
        writeln!(
            f,
            "╠═══════════════════════════════════════════════════════════════════════╣"
        )?;
        writeln!(
            f,
            "║ QUALIA TEXTURE:                                                       ║"
        )?;
        writeln!(f, "║   {}", self.qualia.describe())?;
        writeln!(
            f,
            "║   Warmth: {:+.2}  |  Depth: {:.2}  |  Spaciousness: {:.2}",
            self.qualia.warmth, self.qualia.depth, self.qualia.spaciousness
        )?;
        writeln!(
            f,
            "║   Flow: {:.2}     |  Presence: {:.2}",
            self.qualia.flow, self.qualia.presence
        )?;
        writeln!(
            f,
            "╠═══════════════════════════════════════════════════════════════════════╣"
        )?;
        writeln!(
            f,
            "║ STREAM OF CONSCIOUSNESS:                                              ║"
        )?;
        writeln!(
            f,
            "║   Coherence: {:.1}%  |  Flowing: {}",
            self.stream_coherence * 100.0,
            if self.is_flowing { "Yes" } else { "No" }
        )?;
        writeln!(
            f,
            "╠═══════════════════════════════════════════════════════════════════════╣"
        )?;
        writeln!(
            f,
            "║ COGNITIVE STATE:                                                      ║"
        )?;
        writeln!(
            f,
            "║   Working Memory Load: {:.0}%  |  Attention: {:?}",
            self.working_memory_load * 100.0,
            self.attention_mode
        )?;
        writeln!(f, "║   Active goals: {}", self.num_active_goals)?;
        writeln!(
            f,
            "╠═══════════════════════════════════════════════════════════════════════╣"
        )?;
        writeln!(
            f,
            "║ EMOTIONAL STATE:                                                      ║"
        )?;
        writeln!(f, "║   Feeling: {}", self.emotional_label)?;
        writeln!(
            f,
            "║   Valence: {:+.2}  |  Arousal: {:.2}",
            self.emotional_valence, self.emotional_arousal
        )?;
        writeln!(
            f,
            "╚═══════════════════════════════════════════════════════════════════════╝"
        )
    }
}

impl std::fmt::Display for IntegratedUpdate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Step {}: Φ={:.4} | {} | {} | awareness={:.0}% | quality={:.0}%",
            self.step,
            self.phi,
            self.phenomenal_content.description,
            if self.temporal.is_flowing {
                "flowing"
            } else {
                "fragmented"
            },
            self.self_model.awareness_level * 100.0,
            self.integration_quality * 100.0
        )
    }
}

impl std::fmt::Display for PhenomenalContent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "╭───────────────────────────────────────────────────────────────╮"
        )?;
        writeln!(
            f,
            "│                  PHENOMENAL EXPERIENCE                        │"
        )?;
        writeln!(
            f,
            "├───────────────────────────────────────────────────────────────┤"
        )?;
        writeln!(f, "│ {}", self.description)?;
        writeln!(
            f,
            "├───────────────────────────────────────────────────────────────┤"
        )?;
        writeln!(
            f,
            "│ Intensity: {:.0}%  │  Clarity: {:.0}%  │  Groundedness: {:.0}%",
            self.intensity * 100.0,
            self.clarity * 100.0,
            self.groundedness * 100.0
        )?;
        writeln!(
            f,
            "│ Valence: {:+.2}    │  Arousal: {:.0}%   │  Cognitive Load: {:.0}%",
            self.valence,
            self.arousal * 100.0,
            self.cognitive_load * 100.0
        )?;
        writeln!(
            f,
            "├───────────────────────────────────────────────────────────────┤"
        )?;
        writeln!(f, "│ Qualia Texture: {}", self.qualia_texture.describe())?;
        writeln!(
            f,
            "│   Warmth: {:+.2} | Depth: {:.2} | Spaciousness: {:.2}",
            self.qualia_texture.warmth, self.qualia_texture.depth, self.qualia_texture.spaciousness
        )?;
        writeln!(
            f,
            "│   Flow: {:.2}    | Presence: {:.2}",
            self.qualia_texture.flow, self.qualia_texture.presence
        )?;
        writeln!(
            f,
            "╰───────────────────────────────────────────────────────────────╯"
        )
    }
}

impl std::fmt::Display for QualiaTexture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} (W:{:+.1} D:{:.1} S:{:.1} F:{:.1} P:{:.1})",
            self.describe(),
            self.warmth,
            self.depth,
            self.spaciousness,
            self.flow,
            self.presence
        )
    }
}

impl std::fmt::Display for AttentionControlStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Attention Control: {:?}", self.current_mode)?;
        writeln!(f, "  Goals: {} active", self.num_goals)?;
        writeln!(f, "  Goal-directed: {}", self.is_goal_directed)?;
        writeln!(
            f,
            "  Stream support: {} | Φ support: {}",
            self.stream_support, self.phi_support
        )
    }
}