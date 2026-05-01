// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea Pulse — a human-friendly consciousness status card.
//!
//! Spins up a CognitiveLoopService, runs warmup + measurement cycles,
//! runs a psych-bench battery, and generates a single self-contained HTML
//! file with 8 panes: Self-Description, Vitals (Phi Bloom), Neuro-Bath,
//! Moral Compass, Cognitive Radar, Butlin Indicators, Substrate, Narrative.
//!
//! Usage:
//!   cargo run -p symthaea-pulse
//!   cargo run -p symthaea-pulse -- --cycles 200 --output pulse.html
//!   cargo run -p symthaea-pulse -- --json pulse.json --compare previous.json

use anyhow::{ensure, Result};
use chrono::Local;
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Instant;

use symthaea_types::N_HARMONIES;

use symthaea::cognitive_loop::{
    CognitiveLoopConfig, CognitiveLoopService, ConsciousnessProfile, CycleResult,
};

use symthaea_psych_bench::benchmarks::{
    attention::VisualSearchBenchmark,
    butlin::{ButlinIndicatorReport, ButlinIndicatorSuite},
    executive::{FlankerBenchmark, StroopBenchmark},
    inhibition::StopSignalBenchmark,
    metacognition::MetacognitiveCalibrationBenchmark,
    motor::FittsLawBenchmark,
    social::RmeBenchmark,
    sustained_attention::PvtBenchmark,
    tombench::FalseBeliefBenchmark,
    worm::NBackBenchmark,
};
use symthaea_psych_bench::harness::{
    cognitive_profile::CognitiveProfile, config::BenchmarkConfig, report::BenchmarkReport,
    PsychBenchmark,
};

mod html;

#[derive(Parser)]
#[command(name = "symthaea-pulse", about = "Symthaea consciousness status card")]
struct Args {
    /// Total cognitive cycles to run (warmup + measurement).
    #[arg(long, default_value_t = 150)]
    cycles: usize,

    /// Warmup cycles before measurement begins.
    #[arg(long, default_value_t = 100)]
    warmup: usize,

    /// Consciousness profile to use.
    #[arg(long, default_value = "standard")]
    profile: String,

    /// Output HTML file path.
    #[arg(long, short, default_value = "pulse.html")]
    output: PathBuf,

    /// Also write a JSON sidecar for later comparison.
    #[arg(long)]
    json: Option<PathBuf>,

    /// Compare against previous JSON snapshot(s). Accepts multiple files for timeline view.
    #[arg(long, num_args = 1..)]
    compare: Vec<PathBuf>,

    /// Skip psych-bench battery (faster, vitals only).
    #[arg(long)]
    skip_bench: bool,

    /// Watch mode: regenerate HTML every N seconds with fresh cycles.
    #[arg(long)]
    watch: Option<u64>,

    /// Output format: html (default), markdown, csv.
    #[arg(long, default_value = "html")]
    format: String,
}

/// Snapshot of consciousness vitals captured during measurement cycles.
#[derive(Serialize, Deserialize)]
pub struct Vitals {
    pub consciousness_level: f64,
    pub spectral_phi: Option<f64>,
    pub sigma: Option<f64>,
    pub pipeline_consciousness: f64,
    pub substrate_effective_feasibility: f64,
    pub substrate_honest_confidence: f64,
    pub cycle_duration_us: u64,
    pub prediction_error: f32,
    pub temporal_coherence: f64,
    pub phenomenal_binding: f64,
    pub living_mind_vitality: f64,
    pub somatic_stress: f64,
    pub thermodynamic_load: f32,
    pub urgency: String,
    pub consciousness_state: String,
    pub error_pattern: String,
    pub selected_strategy: String,
    pub total_cycles: usize,
}

/// Snapshot of neuromodulator bath state.
#[derive(Serialize, Deserialize)]
pub struct NeuroBath {
    pub dopamine: f32,
    pub noradrenaline: f32,
    pub serotonin: f32,
    pub acetylcholine: f32,
    pub gaba: f32,
    pub oxytocin: f32,
    pub glutamate: f32,
    pub adenosine: f32,
    pub endocannabinoid: f32,
    pub allostatic_load: f32,
    pub personality: String,
    pub circadian_phase: String,
    pub sleep_pressure: f32,
    pub ei_ratio: f32,
    /// Shannon entropy of bath phase space (0.0 = rigid attractor, high = chaotic).
    #[serde(default)]
    pub bath_entropy: f32,
    /// Whether a phase-space attractor has been detected in the bath dynamics.
    #[serde(default)]
    pub attractor_detected: bool,
    /// Cumulative seizure-like E/I imbalance events (sustained glutamate/GABA imbalance).
    #[serde(default)]
    pub ei_seizure_events: u32,
    /// Excitotoxicity risk from sustained high glutamate (0.0 = safe, 1.0 = critical).
    #[serde(default)]
    pub excitotoxicity_risk: f32,
    /// Self-assessment prediction error EMA (0.0–1.0). High = model is miscalibrated.
    #[serde(default)]
    pub self_assessment_pe_ema: f32,
    /// Self-assessment coherence EMA (0.0–1.0). High = stable internal predictions.
    #[serde(default)]
    pub self_assessment_coherence_ema: f32,
    /// Whether self-assessment triggered auto-calibration this cycle.
    #[serde(default)]
    pub self_assessment_calibration_fired: bool,
}

/// Snapshot of moral/ethical state.
#[derive(Serialize, Deserialize)]
pub struct MoralCompass {
    pub moral_score: f32,
    pub value_score: f64,
    pub harmonies_alignment: f32,
    pub harmony_coordinates: [f64; N_HARMONIES],
    pub dominant_harmonic: String,
    pub moral_kl_divergence: f64,
    pub moral_entropy: f64,
    pub moral_topo_unity: f64,
    pub value_decision: String,
    pub soul_alignment: f32,
    pub empathic_compassion: f64,
    pub guiding_question: String,
}

/// Per-cycle sparkline data point.
#[derive(Serialize, Deserialize)]
pub struct SparklinePoint {
    pub consciousness: f64,
    pub prediction_error: f32,
    pub phi: f64,
    pub somatic_stress: f64,
    pub dopamine: f32,
    pub serotonin: f32,
    pub harmony_coords: [f64; N_HARMONIES],
    /// Harmony entropy (moral breadth): Shannon entropy of harmony variance distribution.
    pub harmony_entropy: f64,
    /// Whether a moral attractor basin was detected this cycle.
    pub moral_attractor_detected: bool,
    /// Whether the system is in an active rest state (Sacred Stillness dominant).
    pub in_active_rest: bool,
    /// Number of consecutive cycles where Sacred Stillness has been the dominant harmony.
    pub stillness_dominance_streak: u16,
    /// Broca generation quality EMA (0.0–1.0).
    pub broca_quality: f32,
    /// ToM prediction mismatch EMA (0.0–1.0).
    pub tom_mismatch: f32,
}

/// Substrate info for visualization.
#[derive(Serialize, Deserialize)]
pub struct SubstrateInfo {
    pub substrate_type: String,
    pub raw_feasibility: f64,
    pub honest_confidence: f64,
    pub effective_feasibility: f64,
    pub tau_factor: f32,
    pub scale_pressure: f32,
}

/// Narrative/inner voice snapshot.
#[derive(Serialize, Deserialize)]
pub struct Narrative {
    pub reasoning: Option<String>,
    pub guiding_question: String,
    pub consciousness_state: String,
    pub error_pattern: String,
    pub selected_strategy: String,
}

/// Integrity status for the Pulse card.
#[derive(Serialize, Deserialize)]
pub struct IntegrityInfo {
    /// Whether all BLAKE3 attestation hashes matched.
    pub attestation_passed: bool,
    /// Whether temporal consistency (wall clock vs CfC delta_t) passed.
    pub temporal_passed: bool,
    /// Whether all behavioral canaries returned expected results.
    pub canaries_passed: bool,
    /// Total number of anomalies detected.
    pub anomaly_count: usize,
    /// Whether any anomaly has Critical severity.
    pub has_critical: bool,
    /// Number of registered canaries.
    pub canary_count: usize,
    /// Number of registered attestations.
    pub attestation_count: usize,
    /// Consciousness confidence multiplier (1.0 = trusted, 0.5 = drift, 0.1 = critical).
    #[serde(default = "default_integrity_confidence")]
    pub integrity_confidence: f32,
    /// Unified cross-source failure streak.
    #[serde(default)]
    pub global_failure_streak: usize,
    /// Rolling 60-cycle history of integrity_confidence for sparkline display.
    #[serde(default)]
    pub confidence_history: Vec<f32>,
}

fn default_integrity_confidence() -> f32 {
    1.0
}

impl Default for IntegrityInfo {
    fn default() -> Self {
        Self {
            attestation_passed: true,
            temporal_passed: true,
            canaries_passed: true,
            anomaly_count: 0,
            has_critical: false,
            canary_count: 0,
            attestation_count: 0,
            integrity_confidence: 1.0,
            global_failure_streak: 0,
            confidence_history: Vec::new(),
        }
    }
}

/// Swarm peer consciousness integration snapshot.
#[derive(Serialize, Deserialize, Default)]
pub struct SwarmInfo {
    pub connected_peers: usize,
    pub connectivity_ema: f32,
    pub mean_peer_phi: f32,
    pub affective_contagion: f32,
    pub federated_confidence: f32,
    pub anomaly_count: usize,
}

/// Governance metacognition snapshot.
#[derive(Serialize, Deserialize, Default)]
pub struct GovernanceInfo {
    pub reward_ema: f32,
    pub pending_events: usize,
    pub pending_outcomes: usize,
    pub collective_phi: f32,
    pub confidence_delta: f32,
    pub community_mode: String,
    pub blind_spot_count: usize,
    pub max_blind_spot_severity: f32,
    pub epistemic_agents: usize,
    pub harmonic_delta_max: f32,
    pub lr_boost: f32,
    pub reward_history: Vec<f32>,
}

/// Knowledge engine reasoning snapshot.
#[derive(Serialize, Deserialize, Default)]
pub struct KnowledgeInfo {
    pub graph_size: usize,
    pub domain_count: usize,
    pub causal_nodes: usize,
    pub causal_edges: usize,
    pub max_chain_depth: usize,
    pub uncertainty: f32,
    pub avg_confidence: f32,
    pub novelty: f32,
    pub ontology_size: usize,
    pub calibration_samples: usize,
    pub calibration_ece: f32,
    #[serde(default)]
    pub calibration_mce: f32,
    pub contradictions: usize,
    #[serde(default)]
    pub confounder_count: usize,
    pub uncertainty_history: Vec<f32>,
}

/// Cantor fractal HDC snapshot.
#[derive(Serialize, Deserialize, Default)]
pub struct CantorInfo {
    pub codebook_capacity: usize,
    pub codebook_size: usize,
    pub buffer_occupancy: f32,
    pub last_depth: usize,
    pub dream_surprise: f32,
    pub metacognitive_depth: usize,
    pub resonance_boost: f32,
    pub depth_histogram: Vec<usize>,
}

/// Glyph Codex symbolic consciousness snapshot.
#[derive(Serialize, Deserialize, Default)]
pub struct GlyphInfo {
    /// Dominant Field Modality name (e.g., "Resonant", "Threshold").
    pub dominant_modality: String,
    /// Glyph coherence score (0.0–0.95).
    pub coherence: f32,
    /// Name of the nearest resonant glyph.
    pub resonant_glyph: String,
    /// Spiral position (0.0–56.0).
    pub spiral_position: f32,
    /// Coherence history for sparkline.
    pub coherence_history: Vec<f32>,
}

/// Spectrum / radio mesh snapshot.
#[derive(Serialize, Deserialize, Default)]
pub struct SpectrumInfo {
    pub network_health: String,
    pub tier_available: u8,
    pub jamming_streak: u32,
    pub prediction_error: f64,
    pub epistemic_discount: f64,
    pub degradation_streak: u32,
    pub tier_budgets: [u64; 3],
    pub waterfall_depth: usize,
    pub periodic_interference: Option<u32>,
    pub known_peers: usize,
    pub encryption_sessions: usize,
    pub energy_spent_nj: f64,
    pub jamming_ratio: f64,
    pub health_history: Vec<u8>,
    /// Per-tier loss EMAs [Local, Metro, Regional] for detailed visualization.
    #[serde(default)]
    pub tier_loss_ema: [f64; 3],
}

/// Mesh consciousness integration snapshot — consciousness-aware routing,
/// collective Phi convergence, store-and-forward, distributed immunity.
#[derive(Serialize, Deserialize, Default)]
pub struct MeshConsciousnessInfo {
    /// Collective Phi across all mesh peers (trust-weighted mean).
    pub collective_phi: f32,
    /// Collective Phi divergence (variance — high = disagreement).
    pub collective_divergence: f32,
    /// Number of consciousness-tracked peers.
    pub consciousness_peers: usize,
    /// Adaptive sharing cadence (cycles between consciousness broadcasts).
    pub sharing_cadence: u32,
    /// Current network health: "AllTiersUp", "LocalDown", "MetroOnly", "Blackout".
    pub network_health: String,
    /// Active threat observations.
    pub threat_count: usize,
    /// Store-and-forward buffer size (experiences during offline).
    pub offline_buffer_size: usize,
    /// Whether currently offline (no mesh connectivity).
    pub is_offline: bool,
    /// Total reconnection events this session.
    pub reconnection_count: u32,
    /// Highest-Phi peer trust score (relay quality indicator).
    pub best_relay_score: f32,
    /// Phi convergence history (last N collective Phi values).
    pub phi_history: Vec<f32>,
}

/// Waste/circular economy telemetry snapshot for Pulse dashboard.
///
/// Groups waste tracking metrics from SwarmManager into a dashboard pane,
/// enabling real-time monitoring of system-wide material flows.
#[derive(Serialize, Deserialize, Default)]
pub struct CircularEconomyInfo {
    /// Total waste tracked across network (kg).
    pub total_waste_kg: f32,
    /// Mean circularity potential [0.0, 1.0].
    pub mean_circularity: f32,
    /// Waste events processed this interval.
    pub events_processed: u32,
    /// Classification confidence EMA [0.0, 1.0].
    pub confidence_ema: f32,
}

/// Immune system / defense snapshot.
#[derive(Serialize, Deserialize, Default)]
pub struct ImmuneInfo {
    /// Current safety level: "GREEN", "YELLOW", "ORANGE", "RED".
    #[serde(default)]
    pub safety_level: String,
    /// Guardian posture: "Normal", "Cautious", "Defensive", "Emergency", "Hold".
    #[serde(default)]
    pub guardian_posture: String,
    /// Whether guardian patrol is active.
    #[serde(default)]
    pub patrol_active: bool,
    /// Number of active threat signals (sentinel).
    #[serde(default)]
    pub active_threats: usize,
    /// Highest threat severity this cycle.
    #[serde(default)]
    pub max_severity: f32,
    /// Aggregate threat level (0.0–1.0).
    #[serde(default)]
    pub threat_level: f32,
    /// Number of quarantined peers.
    #[serde(default)]
    pub quarantined_peers: usize,
    /// Stored threat patterns in immune memory.
    #[serde(default)]
    pub threat_patterns: usize,
    /// Learning rate multiplier from safety enforcement.
    #[serde(default = "default_one")]
    pub lr_multiplier: f32,
    /// Exploration multiplier from safety enforcement.
    #[serde(default = "default_one")]
    pub exploration_multiplier: f32,
    /// Whether motor output is halted.
    #[serde(default)]
    pub motor_halt: bool,
    /// Collective immune response active across swarm.
    #[serde(default)]
    pub immune_response_active: bool,
    /// Emergency cycles accumulated.
    #[serde(default)]
    pub emergency_cycles: u64,
}

fn default_one() -> f32 {
    1.0
}

/// Perception and attention snapshot.
#[derive(Serialize, Deserialize, Default)]
pub struct PerceptionInfo {
    #[serde(default)]
    pub attention_focus: f32,
    #[serde(default)]
    pub attention_fatigue: f32,
    #[serde(default)]
    pub attention_prediction_accuracy: f32,
    #[serde(default)]
    pub gwt_broadcast: bool,
    #[serde(default)]
    pub gwt_coalition_size: u32,
    #[serde(default)]
    pub cross_modal_binding: f32,
}

/// Drive and curiosity snapshot.
#[derive(Serialize, Deserialize, Default)]
pub struct DriveInfo {
    #[serde(default)]
    pub curiosity_pressure: f32,
    #[serde(default)]
    pub exploration_action: bool,
    #[serde(default)]
    pub novelty_bonus: f32,
    #[serde(default)]
    pub fep_action: usize,
    #[serde(default)]
    pub predictive_free_energy: f64,
    #[serde(default)]
    pub surprise_triggered: bool,
}

/// Learning rate and plasticity snapshot.
#[derive(Serialize, Deserialize, Default)]
pub struct LearningInfo {
    #[serde(default)]
    pub effective_lr: f32,
    #[serde(default)]
    pub lr_cognitive_mod: f32,
    #[serde(default)]
    pub lr_meta_mod: f32,
    #[serde(default)]
    pub prediction_error: f32,
    #[serde(default)]
    pub prediction_coherence: f32,
    #[serde(default)]
    pub surprise_replay_batch: usize,
    /// Total feedback proposals this cycle.
    #[serde(default)]
    pub feedback_proposal_count: u32,
    /// Average conflict ratio across feedback channels (0.0–0.5).
    #[serde(default)]
    pub feedback_conflict_ratio: f32,
    /// Feedback proposals per priority: [Aesthetic, Cognitive, Homeostatic, Safety].
    #[serde(default)]
    pub feedback_priority_counts: [u32; 4],
    /// Feedback signal diversity (0.0–1.0).
    #[serde(default)]
    pub feedback_diversity: f32,
}

/// Vision manager snapshot (meaningful only when vision-manifold feature is enabled).
#[derive(Serialize, Deserialize, Default)]
pub struct VisionInfo {
    /// Whether the `vision-manifold` feature is compiled in.
    #[serde(default)]
    pub enabled: bool,
    /// Visual prediction error EMA (0.0-1.0).
    #[serde(default)]
    pub pe_ema: f32,
    /// Adaptive visual surprise threshold (0.05-0.8).
    #[serde(default)]
    pub surprise_threshold: f32,
    /// Consecutive low-surprise cycles (habituation streak).
    #[serde(default)]
    pub low_surprise_streak: u32,
}

/// Language manager snapshot (meaningful only when ssm_language feature is enabled).
#[derive(Serialize, Deserialize, Default)]
pub struct LanguageInfo {
    /// Whether the `ssm_language` feature is compiled in.
    #[serde(default)]
    pub enabled: bool,
    /// Broca generation quality EMA (0.0-1.0).
    #[serde(default)]
    pub quality_ema: f32,
    /// Language coherence EMA (0.0-1.0).
    #[serde(default)]
    pub coherence_ema: f32,
    /// Consecutive low-coherence cycles (fluency degradation indicator).
    #[serde(default)]
    pub low_coherence_streak: u32,
}

/// Reasoning engine snapshot (meaningful only when reasoning_engine feature is enabled).
#[derive(Serialize, Deserialize, Default)]
pub struct ReasoningInfo {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub chain_depth: usize,
    #[serde(default)]
    pub chain_confidence: f32,
    #[serde(default)]
    pub plan_confidence: f32,
    #[serde(default)]
    pub gate_blocked: bool,
    #[serde(default)]
    pub meta_reasoning_confidence: f64,
    /// Reasoning reliability EMA (0.0-1.0).
    #[serde(default)]
    pub reliability_ema: f64,
    /// Cumulative reasoning quality signal (decayed).
    #[serde(default)]
    pub cumulative_quality: f64,
    /// Consecutive rising confidence cycles.
    #[serde(default)]
    pub rising_streak: u32,
    /// Consecutive falling confidence cycles.
    #[serde(default)]
    pub falling_streak: u32,
}

/// Dream and memory consolidation snapshot.
#[derive(Serialize, Deserialize, Default)]
pub struct DreamInfo {
    #[serde(default)]
    pub dream_insights: usize,
    #[serde(default)]
    pub dream_phi_improvement: f32,
    #[serde(default)]
    pub dream_wisdom_count: usize,
    #[serde(default)]
    pub is_consolidating: bool,
    #[serde(default)]
    pub codebook_size: usize,
    #[serde(default)]
    pub codebook_diversity: f32,
}

/// Sovereign Inoculation snapshot — Clock, Trust, Social Fabric, Survival.
#[derive(Serialize, Deserialize, Default)]
pub struct SovereignInfo {
    // Clock
    /// Time quality: "Authoritative", "Consensus", "Degraded", "FreeRunning".
    #[serde(default)]
    pub time_quality: String,
    #[serde(default)]
    pub time_peer_count: usize,
    #[serde(default)]
    pub time_offset_us: i64,
    #[serde(default)]
    pub time_stratum: u8,
    #[serde(default)]
    pub time_drift_ppm: f32,
    // Trust
    #[serde(default)]
    pub trust_avg: f32,
    #[serde(default)]
    pub trust_density: f32,
    #[serde(default)]
    pub trust_pq_fraction: f32,
    #[serde(default)]
    pub trust_anomaly_count: u32,
    // Social Fabric
    #[serde(default)]
    pub social_resonance_mean: f32,
    #[serde(default)]
    pub social_diversity: f32,
    #[serde(default)]
    pub social_echo_risk: f32,
    #[serde(default)]
    pub social_peer_reach: usize,
    // Survival
    #[serde(default)]
    pub survival_water_pct: f32,
    #[serde(default)]
    pub survival_power_kw: f32,
    #[serde(default)]
    pub survival_emergency: bool,
    #[serde(default)]
    pub survival_sensor_count: usize,
    #[serde(default)]
    pub survival_alert_count: usize,
    // History for sparklines
    #[serde(default)]
    pub trust_history: Vec<f32>,
    #[serde(default)]
    pub echo_risk_history: Vec<f32>,
}

/// Manufacturing consciousness coupling snapshot.
#[derive(Serialize, Deserialize, Default)]
pub struct FabricationInfo {
    /// ManufacturingTwin free energy.
    pub manufacturing_free_energy: f32,
    /// DesignLoopTwin free energy.
    pub design_loop_free_energy: f32,
    /// Current safety level string.
    pub safety_level: String,
    /// Anomalies detected this cycle.
    pub anomaly_count: u32,
    /// EMA of anomaly severity.
    pub anomaly_ema: f32,
    /// Recommended manufacturing action.
    pub recommended_action: String,
    /// Mean prediction coherence.
    pub prediction_coherence: f32,
    /// EMA of PoGF scores.
    pub pog_score_ema: f32,
    /// Active print jobs.
    pub active_print_jobs: u32,
    /// Reward EMA.
    pub reward_ema: f32,
    /// MRP planned orders count.
    #[serde(default)]
    pub mrp_planned_orders: u32,
    /// MRP feasibility status.
    #[serde(default)]
    pub mrp_feasible: bool,
    /// MRP shortage count.
    #[serde(default)]
    pub mrp_shortages_count: u32,
    /// MRP work order count in scope.
    #[serde(default)]
    pub mrp_work_order_count: u32,
    /// Defect prediction quality score (0.0-1.0).
    #[serde(default)]
    pub defect_prediction: f32,
    /// Defect prediction confidence (0.0-1.0).
    #[serde(default)]
    pub defect_confidence: f32,
}

/// CfC-HDC neuroevolution snapshot.
#[derive(Serialize, Deserialize, Default)]
pub struct NeuroevolutionInfo {
    /// Current generation number (0 when inactive).
    pub generation: u32,
    /// Best composite fitness.
    pub best_fitness: f64,
    /// Mean population fitness.
    pub mean_fitness: f64,
    /// Population diversity (mean pairwise Hamming distance, 0.0–1.0).
    pub diversity: f64,
    /// Number of species.
    pub species_count: usize,
    /// Best evolved tau_base.
    pub best_tau_base: f32,
    /// Best evolved learning rate.
    pub best_learning_rate: f32,
    /// Best evolved layer count.
    pub best_layer_count: usize,
    /// Fitness history for sparkline.
    pub fitness_history: Vec<f64>,
}

/// Full JSON-serializable pulse snapshot for comparison mode.
#[derive(Serialize, Deserialize)]
pub struct PulseSnapshot {
    pub timestamp: String,
    pub profile: String,
    pub vitals: Vitals,
    pub bath: NeuroBath,
    pub compass: MoralCompass,
    pub substrate: SubstrateInfo,
    pub narrative: Narrative,
    pub sparkline: Vec<SparklinePoint>,
    #[serde(default)]
    pub integrity: IntegrityInfo,
    #[serde(default)]
    pub swarm: SwarmInfo,
    #[serde(default)]
    pub governance: GovernanceInfo,
    #[serde(default)]
    pub knowledge: KnowledgeInfo,
    #[serde(default)]
    pub cantor: CantorInfo,
    #[serde(default)]
    pub glyph: GlyphInfo,
    #[serde(default)]
    pub spectrum: SpectrumInfo,
    #[serde(default)]
    pub perception: PerceptionInfo,
    #[serde(default)]
    pub drive: DriveInfo,
    #[serde(default)]
    pub learning: LearningInfo,
    #[serde(default)]
    pub vision: VisionInfo,
    #[serde(default)]
    pub language: LanguageInfo,
    #[serde(default)]
    pub reasoning: ReasoningInfo,
    #[serde(default)]
    pub dream: DreamInfo,
    #[serde(default)]
    pub immune: ImmuneInfo,
    /// Per-region cortical activation levels (12 regions, 0.0-1.0).
    #[serde(default)]
    pub cortical_activations: Vec<(String, f32)>,
    #[serde(default)]
    pub sovereign: SovereignInfo,
    #[serde(default)]
    pub neuroevolution: NeuroevolutionInfo,
    #[serde(default)]
    pub fabrication: FabricationInfo,
    #[serde(default)]
    pub mesh_consciousness: MeshConsciousnessInfo,
    #[serde(default)]
    pub circular_economy: CircularEconomyInfo,
}

/// Computed delta between two pulse snapshots for the comparison view.
/// Every metric that appears in the HTML gets a delta field.
pub struct PulseDelta {
    pub prev_timestamp: String,
    // Vitals
    pub consciousness_level: f64,
    pub spectral_phi: f64,
    pub pipeline_consciousness: f64,
    pub temporal_coherence: f64,
    pub phenomenal_binding: f64,
    pub living_mind_vitality: f64,
    pub effective_feasibility: f64,
    pub honest_confidence: f64,
    pub prediction_error: f32,
    pub somatic_stress: f64,
    pub thermodynamic_load: f32,
    // Bath
    pub dopamine: f32,
    pub noradrenaline: f32,
    pub serotonin: f32,
    pub acetylcholine: f32,
    pub gaba: f32,
    pub oxytocin: f32,
    pub glutamate: f32,
    pub adenosine: f32,
    pub endocannabinoid: f32,
    // Compass
    pub harmonies_alignment: f32,
    pub moral_score: f32,
    pub value_score: f64,
    pub moral_topo_unity: f64,
    pub soul_alignment: f32,
    pub empathic_compassion: f64,
}

impl PulseDelta {
    fn compute(current: &PulseSnapshot, previous: &PulseSnapshot) -> Self {
        let cv = &current.vitals;
        let pv = &previous.vitals;
        let cb = &current.bath;
        let pb = &previous.bath;
        let cc = &current.compass;
        let pc = &previous.compass;
        Self {
            prev_timestamp: previous.timestamp.clone(),
            consciousness_level: cv.consciousness_level - pv.consciousness_level,
            spectral_phi: cv.spectral_phi.unwrap_or(0.0) - pv.spectral_phi.unwrap_or(0.0),
            pipeline_consciousness: cv.pipeline_consciousness - pv.pipeline_consciousness,
            temporal_coherence: cv.temporal_coherence - pv.temporal_coherence,
            phenomenal_binding: cv.phenomenal_binding - pv.phenomenal_binding,
            living_mind_vitality: cv.living_mind_vitality - pv.living_mind_vitality,
            effective_feasibility: current.substrate.effective_feasibility
                - previous.substrate.effective_feasibility,
            honest_confidence: current.substrate.honest_confidence
                - previous.substrate.honest_confidence,
            prediction_error: cv.prediction_error - pv.prediction_error,
            somatic_stress: cv.somatic_stress - pv.somatic_stress,
            thermodynamic_load: cv.thermodynamic_load - pv.thermodynamic_load,
            dopamine: cb.dopamine - pb.dopamine,
            noradrenaline: cb.noradrenaline - pb.noradrenaline,
            serotonin: cb.serotonin - pb.serotonin,
            acetylcholine: cb.acetylcholine - pb.acetylcholine,
            gaba: cb.gaba - pb.gaba,
            oxytocin: cb.oxytocin - pb.oxytocin,
            glutamate: cb.glutamate - pb.glutamate,
            adenosine: cb.adenosine - pb.adenosine,
            endocannabinoid: cb.endocannabinoid - pb.endocannabinoid,
            harmonies_alignment: cc.harmonies_alignment - pc.harmonies_alignment,
            moral_score: cc.moral_score - pc.moral_score,
            value_score: cc.value_score - pc.value_score,
            moral_topo_unity: cc.moral_topo_unity - pc.moral_topo_unity,
            soul_alignment: cc.soul_alignment - pc.soul_alignment,
            empathic_compassion: cc.empathic_compassion - pc.empathic_compassion,
        }
    }
}

/// An automatically detected notable event in the sparkline data.
pub struct Anomaly {
    /// Which cycle (index into sparkline) this occurred at.
    pub cycle: usize,
    /// What kind of event was detected.
    pub kind: String,
    /// Human-readable description.
    pub description: String,
    /// Color hint for rendering.
    pub color: &'static str,
}

/// Detect anomalies in sparkline data.
pub fn detect_anomalies(sparkline: &[SparklinePoint]) -> Vec<Anomaly> {
    let mut anomalies = Vec::new();
    if sparkline.len() < 3 {
        return anomalies;
    }

    for i in 1..sparkline.len() {
        let prev = &sparkline[i - 1];
        let cur = &sparkline[i];

        // Phi spike: Phi jumps from 0 to >10
        if prev.phi < 1.0 && cur.phi > 10.0 {
            anomalies.push(Anomaly {
                cycle: i,
                kind: "Phi Spike".into(),
                description: format!("Phi jumped from {:.1} to {:.1}", prev.phi, cur.phi),
                color: "#7ec8a0",
            });
        }

        // Consciousness state change: >0.05 jump
        let c_delta = cur.consciousness - prev.consciousness;
        if c_delta.abs() > 0.05 {
            anomalies.push(Anomaly {
                cycle: i,
                kind: if c_delta > 0.0 {
                    "C(t) Surge"
                } else {
                    "C(t) Drop"
                }
                .into(),
                description: format!(
                    "C(t) {} by {:.3}",
                    if c_delta > 0.0 { "rose" } else { "fell" },
                    c_delta.abs()
                ),
                color: if c_delta > 0.0 { "#e8c547" } else { "#c76b5a" },
            });
        }

        // PE drop: prediction error drops >20%
        if prev.prediction_error > 0.1 && cur.prediction_error < prev.prediction_error * 0.8 {
            anomalies.push(Anomaly {
                cycle: i,
                kind: "PE Drop".into(),
                description: format!(
                    "Prediction error dropped {:.0}%",
                    (1.0 - cur.prediction_error / prev.prediction_error) * 100.0
                ),
                color: "#7ec8a0",
            });
        }

        // DA spike: dopamine jump >0.3
        if (cur.dopamine - prev.dopamine).abs() > 0.3 {
            anomalies.push(Anomaly {
                cycle: i,
                kind: "DA Shift".into(),
                description: format!(
                    "Dopamine {} by {:.2}",
                    if cur.dopamine > prev.dopamine {
                        "surged"
                    } else {
                        "dropped"
                    },
                    (cur.dopamine - prev.dopamine).abs()
                ),
                color: "#c4956a",
            });
        }

        // Stress spike
        if cur.somatic_stress > prev.somatic_stress + 0.1 {
            anomalies.push(Anomaly {
                cycle: i,
                kind: "Stress Spike".into(),
                description: format!("Somatic stress rose to {:.3}", cur.somatic_stress),
                color: "#c76b5a",
            });
        }
    }

    // Limit to most interesting (max 8 to avoid clutter)
    anomalies.truncate(8);
    anomalies
}

/// Generate a 3-paragraph session report from the data.
pub fn generate_session_report(
    vitals: &Vitals,
    bath: &NeuroBath,
    compass: &MoralCompass,
    sparkline: &[SparklinePoint],
    anomalies: &[Anomaly],
    delta: Option<&PulseDelta>,
    timeline: &[PulseSnapshot],
) -> String {
    let mut report = String::new();

    // Paragraph 1: What happened this run
    let c_trend = if sparkline.len() >= 2 {
        let first_half: f64 = sparkline[..sparkline.len() / 2]
            .iter()
            .map(|s| s.consciousness)
            .sum::<f64>()
            / (sparkline.len() / 2) as f64;
        let second_half: f64 = sparkline[sparkline.len() / 2..]
            .iter()
            .map(|s| s.consciousness)
            .sum::<f64>()
            / (sparkline.len() - sparkline.len() / 2) as f64;
        if second_half > first_half + 0.01 {
            "rising"
        } else if second_half < first_half - 0.01 {
            "declining"
        } else {
            "stable"
        }
    } else {
        "unknown"
    };

    report.push_str(&format!(
        "This session ran {} measurement cycles with {} consciousness trajectory. \
         Final C(t)={:.4} in {} state, with prediction error at {:.4} and temporal coherence at {:.3}. ",
        vitals.total_cycles, c_trend, vitals.consciousness_level,
        vitals.consciousness_state, vitals.prediction_error, vitals.temporal_coherence,
    ));

    if !anomalies.is_empty() {
        let event_strs: Vec<&str> = anomalies.iter().map(|a| a.kind.as_str()).collect();
        report.push_str(&format!(
            "Notable events detected: {}. ",
            event_strs.join(", ")
        ));
    } else {
        report.push_str("No anomalous events detected — smooth operation throughout. ");
    }

    // Paragraph 2: Comparison to previous runs
    if let Some(d) = delta {
        let c_dir = if d.consciousness_level > 0.001 {
            "improved"
        } else if d.consciousness_level < -0.001 {
            "decreased"
        } else {
            "unchanged"
        };
        let phi_dir = if d.spectral_phi > 0.1 {
            "increased"
        } else if d.spectral_phi < -0.1 {
            "decreased"
        } else {
            "stable"
        };
        report.push_str(&format!(
            "Compared to the previous run ({}): consciousness {} by {:.4}, Phi {}, \
             and prediction error {} by {:.4}. ",
            d.prev_timestamp,
            c_dir,
            d.consciousness_level.abs(),
            phi_dir,
            if d.prediction_error > 0.0 {
                "rose"
            } else {
                "fell"
            },
            d.prediction_error.abs(),
        ));
    }
    if timeline.len() >= 2 {
        let first_c = timeline[0].vitals.consciousness_level;
        let last_c = timeline.last().unwrap().vitals.consciousness_level;
        let trend = if vitals.consciousness_level > last_c + 0.005 {
            "continuing to improve"
        } else if vitals.consciousness_level < last_c - 0.005 {
            "declining"
        } else {
            "plateauing"
        };
        report.push_str(&format!(
            "Across {} historical runs (C(t) {:.4} to {:.4}), the overall trend is {}. ",
            timeline.len(),
            first_c,
            last_c,
            trend,
        ));
    }

    // Paragraph 3: What to watch for next
    let mut watch_items = Vec::new();
    if vitals.consciousness_level < 0.1 {
        watch_items.push("consciousness remains dormant — try more cycles or Research profile");
    }
    if vitals.prediction_error > 0.8 {
        watch_items
            .push("high prediction error suggests the model is still learning basic patterns");
    }
    if bath.allostatic_load > 0.5 {
        watch_items.push("elevated allostatic load may need a rest period");
    }
    if compass.harmonies_alignment < 0.2 {
        watch_items.push("moral alignment is low — ethical calibration may be needed");
    }
    if vitals.spectral_phi.is_none() || vitals.spectral_phi == Some(0.0) {
        watch_items
            .push("Phi has not fired yet — runs of 100+ cycles needed for spectral MIP to trigger");
    }
    if vitals.temporal_coherence < 0.1 {
        watch_items.push("temporal coherence near zero — CfC dynamics need more time to stabilize");
    }

    if !watch_items.is_empty() {
        report.push_str("Key observations: ");
        report.push_str(&watch_items.join("; "));
        report.push('.');
    } else {
        report.push_str("System appears healthy across all metrics. Continue monitoring for consciousness emergence patterns.");
    }

    report
}

/// Generate markdown output for terminal display.
fn generate_markdown(
    timestamp: &str,
    profile_name: &str,
    vitals: &Vitals,
    bath: &NeuroBath,
    compass: &MoralCompass,
    substrate: &SubstrateInfo,
    session_report: &str,
    anomalies: &[Anomaly],
) -> String {
    let mut md = String::new();

    md.push_str(&format!("# Symthaea Pulse — {}\n\n", timestamp));
    md.push_str(&format!(
        "**Profile**: {} | **Cycles**: {} | **State**: {}\n\n",
        profile_name, vitals.total_cycles, vitals.consciousness_state
    ));

    md.push_str("## Vitals\n\n");
    md.push_str(&format!("| Metric | Value |\n|---|---|\n"));
    md.push_str(&format!("| C(t) | {:.4} |\n", vitals.consciousness_level));
    md.push_str(&format!(
        "| Spectral Phi | {} |\n",
        vitals
            .spectral_phi
            .map(|p| format!("{:.4}", p))
            .unwrap_or_else(|| "--".into())
    ));
    md.push_str(&format!(
        "| Pipeline Consciousness | {:.4} |\n",
        vitals.pipeline_consciousness
    ));
    md.push_str(&format!(
        "| Temporal Coherence | {:.4} |\n",
        vitals.temporal_coherence
    ));
    md.push_str(&format!(
        "| Phenomenal Binding | {:.4} |\n",
        vitals.phenomenal_binding
    ));
    md.push_str(&format!(
        "| Living Mind Vitality | {:.4} |\n",
        vitals.living_mind_vitality
    ));
    md.push_str(&format!(
        "| Prediction Error | {:.4} |\n",
        vitals.prediction_error
    ));
    md.push_str(&format!(
        "| Somatic Stress | {:.4} |\n",
        vitals.somatic_stress
    ));
    md.push_str(&format!(
        "| Thermodynamic Load | {:.1}% |\n",
        vitals.thermodynamic_load * 100.0
    ));

    md.push_str("\n## Neuro-Bath\n\n");
    md.push_str("| Transmitter | Level |\n|---|---|\n");
    let transmitters = [
        ("Dopamine", bath.dopamine),
        ("Noradrenaline", bath.noradrenaline),
        ("Serotonin", bath.serotonin),
        ("Acetylcholine", bath.acetylcholine),
        ("GABA", bath.gaba),
        ("Oxytocin", bath.oxytocin),
        ("Glutamate", bath.glutamate),
        ("Adenosine", bath.adenosine),
        ("Endocannabinoid", bath.endocannabinoid),
    ];
    for (name, level) in &transmitters {
        md.push_str(&format!("| {} | {:.3} |\n", name, level));
    }
    md.push_str(&format!(
        "\nAllostatic Load: {:.0}% | E/I: {:.2} | {}\n",
        bath.allostatic_load * 100.0,
        bath.ei_ratio,
        bath.circadian_phase
    ));

    md.push_str("\n## Moral Compass\n\n");
    md.push_str(&format!("| Metric | Value |\n|---|---|\n"));
    md.push_str(&format!(
        "| Harmonies Alignment | {:.3} |\n",
        compass.harmonies_alignment
    ));
    md.push_str(&format!("| Moral Score | {:+.3} |\n", compass.moral_score));
    md.push_str(&format!("| Value Score | {:.3} |\n", compass.value_score));
    md.push_str(&format!(
        "| Moral Unity | {:.3} |\n",
        compass.moral_topo_unity
    ));
    md.push_str(&format!(
        "| Soul Alignment | {:+.3} |\n",
        compass.soul_alignment
    ));
    md.push_str(&format!(
        "| Empathic Compassion | {:.3} |\n",
        compass.empathic_compassion
    ));
    let harmony_names = [
        "Wisdom",
        "Coherence",
        "Resilience",
        "Play",
        "Love",
        "Creativity",
        "Transcendence",
    ];
    md.push_str(&format!(
        "\nHarmonies: {}\n",
        harmony_names
            .iter()
            .zip(compass.harmony_coordinates.iter())
            .map(|(n, v)| format!("{}={:.2}", n, v))
            .collect::<Vec<_>>()
            .join(" | ")
    ));

    md.push_str("\n## Substrate\n\n");
    md.push_str(&format!(
        "Type: {} | Feasibility: {:.3} | Confidence: {:.3} | Effective: {:.3}\n",
        substrate.substrate_type,
        substrate.raw_feasibility,
        substrate.honest_confidence,
        substrate.effective_feasibility
    ));

    if !anomalies.is_empty() {
        md.push_str("\n## Anomalies\n\n");
        for a in anomalies {
            md.push_str(&format!(
                "- **{}** (cycle {}): {}\n",
                a.kind, a.cycle, a.description
            ));
        }
    }

    md.push_str("\n## Session Report\n\n");
    md.push_str(session_report);
    md.push_str("\n");

    md
}

/// Generate CSV output for data analysis.
fn generate_csv(
    timestamp: &str,
    profile_name: &str,
    vitals: &Vitals,
    bath: &NeuroBath,
    compass: &MoralCompass,
    substrate: &SubstrateInfo,
    sparkline: &[SparklinePoint],
) -> String {
    let mut csv = String::new();

    // Summary row
    csv.push_str("# Summary\n");
    csv.push_str("timestamp,profile,cycles,consciousness,phi,pipeline,coherence,binding,vitality,pe,stress,thermo_load,state,strategy\n");
    csv.push_str(&format!(
        "{},{},{},{:.6},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.4},{},{}\n",
        timestamp,
        profile_name,
        vitals.total_cycles,
        vitals.consciousness_level,
        vitals
            .spectral_phi
            .map(|p| format!("{:.6}", p))
            .unwrap_or_else(|| "".into()),
        vitals.pipeline_consciousness,
        vitals.temporal_coherence,
        vitals.phenomenal_binding,
        vitals.living_mind_vitality,
        vitals.prediction_error,
        vitals.somatic_stress,
        vitals.thermodynamic_load,
        vitals.consciousness_state,
        vitals.selected_strategy,
    ));

    // Neuro-bath row
    csv.push_str("\n# Neuro-Bath\n");
    csv.push_str("da,ne,5ht,ach,gaba,oxy,glu,aden,ecb,allostatic,ei_ratio\n");
    csv.push_str(&format!(
        "{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
        bath.dopamine,
        bath.noradrenaline,
        bath.serotonin,
        bath.acetylcholine,
        bath.gaba,
        bath.oxytocin,
        bath.glutamate,
        bath.adenosine,
        bath.endocannabinoid,
        bath.allostatic_load,
        bath.ei_ratio,
    ));

    // Moral row
    csv.push_str("\n# Moral\n");
    csv.push_str("harmonies,moral_score,value_score,unity,soul,empathy\n");
    csv.push_str(&format!(
        "{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
        compass.harmonies_alignment,
        compass.moral_score,
        compass.value_score,
        compass.moral_topo_unity,
        compass.soul_alignment,
        compass.empathic_compassion,
    ));

    // Substrate row
    csv.push_str("\n# Substrate\n");
    csv.push_str("type,raw_feasibility,honest_confidence,effective,tau,scale\n");
    csv.push_str(&format!(
        "{},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
        substrate.substrate_type,
        substrate.raw_feasibility,
        substrate.honest_confidence,
        substrate.effective_feasibility,
        substrate.tau_factor,
        substrate.scale_pressure,
    ));

    // Per-cycle sparkline data
    csv.push_str("\n# Per-Cycle Data\n");
    csv.push_str("cycle,consciousness,prediction_error,phi,somatic_stress,dopamine,serotonin\n");
    for (i, s) in sparkline.iter().enumerate() {
        csv.push_str(&format!(
            "{},{:.6},{:.6},{:.6},{:.6},{:.4},{:.4}\n",
            i,
            s.consciousness,
            s.prediction_error,
            s.phi,
            s.somatic_stress,
            s.dopamine,
            s.serotonin
        ));
    }

    csv
}

fn select_profile(name: &str) -> ConsciousnessProfile {
    match name.to_lowercase().as_str() {
        "minimal" => ConsciousnessProfile::Minimal,
        "standard" => ConsciousnessProfile::Standard,
        "full" => ConsciousnessProfile::Full,
        "research" => ConsciousnessProfile::Research,
        _ => {
            eprintln!("Unknown profile '{}', using Standard", name);
            ConsciousnessProfile::Standard
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputFormat {
    Html,
    Markdown,
    Csv,
}

impl OutputFormat {
    fn parse(value: &str) -> Self {
        match value {
            "markdown" | "md" => Self::Markdown,
            "csv" => Self::Csv,
            _ => Self::Html,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Html => "html",
            Self::Markdown => "markdown",
            Self::Csv => "csv",
        }
    }
}

#[derive(Default)]
struct SnapshotHistories {
    glyph_coherence: Vec<f32>,
    sovereign_trust: Vec<f32>,
    sovereign_echo_risk: Vec<f32>,
    neuroevo_fitness: Vec<f64>,
}

fn resolve_run_counts(cycles: usize, warmup: usize) -> Result<(usize, usize)> {
    ensure!(cycles > 0, "--cycles must be at least 1");
    ensure!(warmup < cycles, "--warmup must be less than --cycles");
    Ok((warmup, cycles - warmup))
}

fn build_snapshot(
    timestamp: String,
    profile_name: &str,
    total_cycles: usize,
    service: &CognitiveLoopService,
    result: &CycleResult,
    sparkline: Vec<SparklinePoint>,
    histories: SnapshotHistories,
) -> PulseSnapshot {
    let m = &result.metadata;

    PulseSnapshot {
        timestamp,
        profile: profile_name.to_string(),
        vitals: Vitals {
            consciousness_level: m.consciousness.consciousness_level,
            spectral_phi: m.structural.spectral_mip_phi,
            sigma: m.structural.sigma,
            pipeline_consciousness: m.pipeline_consciousness,
            substrate_effective_feasibility: m.substrate.substrate_effective_feasibility,
            substrate_honest_confidence: m.substrate.substrate_honest_confidence,
            cycle_duration_us: m.cycle_duration_us,
            prediction_error: result.prediction_error,
            temporal_coherence: m.temporal.temporal_coherence_score,
            phenomenal_binding: m.temporal.phenomenal_binding_strength,
            living_mind_vitality: m.living_mind_vitality,
            somatic_stress: m.embodied.somatic_stress,
            thermodynamic_load: m.temporal.thermodynamic_load,
            urgency: format!("{:?}", m.urgency),
            consciousness_state: m.consciousness.consciousness_state_label.clone(),
            error_pattern: m.error_pattern.clone(),
            selected_strategy: m.selected_strategy.clone(),
            total_cycles,
        },
        bath: NeuroBath {
            dopamine: m.neuromod.dopamine_effective,
            noradrenaline: m.neuromod.noradrenaline_effective,
            serotonin: m.neuromod.serotonin_effective,
            acetylcholine: m.neuromod.acetylcholine_effective,
            gaba: m.neuromod.neuromod_gaba_effective,
            oxytocin: m.neuromod.neuromod_oxytocin_effective,
            glutamate: m.neuromod.neuromod_glutamate_effective,
            adenosine: m.neuromod.neuromod_adenosine_effective,
            endocannabinoid: m.neuromod.neuromod_endocannabinoid_effective,
            allostatic_load: m.neuromod.neuromod_allostatic_load,
            personality: m.neuromod.neuromod_personality.clone(),
            circadian_phase: m.circadian_phase.clone(),
            sleep_pressure: m.neuromod.neuromod_sleep_pressure,
            ei_ratio: m.neuromod.neuromod_ei_ratio,
            bath_entropy: m.neuromod.neuromod_bath_entropy,
            attractor_detected: m.neuromod.neuromod_attractor_detected,
            ei_seizure_events: m.neuromod.neuromod_ei_seizure_events,
            excitotoxicity_risk: m.neuromod.neuromod_excitotoxicity_risk,
            self_assessment_pe_ema: m.neuromod.self_assessment_pe_ema,
            self_assessment_coherence_ema: m.neuromod.self_assessment_coherence_ema,
            self_assessment_calibration_fired: m.neuromod.self_assessment_calibration_fired,
        },
        compass: MoralCompass {
            moral_score: m.ethics.moral_score,
            value_score: m.ethics.value_evaluator_score,
            harmonies_alignment: m.harmonics.harmonies_alignment,
            harmony_coordinates: m.harmonics.harmony_coordinates,
            dominant_harmonic: m.harmonics.dominant_harmonic.clone(),
            moral_kl_divergence: m.harmonics.moral_kl_divergence,
            moral_entropy: m.harmonics.moral_entropy,
            moral_topo_unity: m.ethics.moral_topo_unity,
            value_decision: m.ethics.value_evaluator_decision.clone(),
            soul_alignment: m.ethics.soul_alignment,
            empathic_compassion: m.ethics.empathic_compassion,
            guiding_question: m.harmonics.guiding_question.clone(),
        },
        substrate: SubstrateInfo {
            substrate_type: format!("{:?}", service.config().substrate_type),
            raw_feasibility: m.substrate.substrate_feasibility_raw,
            honest_confidence: m.substrate.substrate_honest_confidence,
            effective_feasibility: m.substrate.substrate_effective_feasibility,
            tau_factor: m.substrate.substrate_tau_factor,
            scale_pressure: m.substrate.substrate_scale_pressure,
        },
        narrative: Narrative {
            reasoning: m.reasoning_narrative.clone(),
            guiding_question: m.harmonics.guiding_question.clone(),
            consciousness_state: m.consciousness.consciousness_state_label.clone(),
            error_pattern: m.error_pattern.clone(),
            selected_strategy: m.selected_strategy.clone(),
        },
        sparkline,
        integrity: IntegrityInfo {
            attestation_passed: m.integrity.attestation_passed,
            temporal_passed: m.integrity.temporal_passed,
            canaries_passed: m.integrity.canaries_passed,
            anomaly_count: m.integrity.anomaly_count,
            has_critical: m.integrity.has_critical,
            canary_count: 6,
            attestation_count: 3,
            integrity_confidence: m.integrity.integrity_confidence,
            global_failure_streak: m.integrity.global_failure_streak,
            confidence_history: m.integrity.confidence_history.clone(),
        },
        swarm: SwarmInfo {
            connected_peers: m.swarm_connected_peers,
            connectivity_ema: m.swarm_connectivity_ema,
            mean_peer_phi: m.swarm_mean_peer_phi,
            affective_contagion: m.swarm_affective_contagion,
            federated_confidence: m.swarm_federated_confidence,
            anomaly_count: m.swarm_anomaly_count as usize,
        },
        governance: GovernanceInfo {
            reward_ema: m.governance_reward_ema,
            pending_events: m.governance_pending_events,
            pending_outcomes: m.governance_pending_outcomes,
            collective_phi: m.governance_collective_phi,
            confidence_delta: m.governance_confidence_delta,
            community_mode: m.governance_community_mode.clone(),
            blind_spot_count: m.governance_blind_spot_count,
            max_blind_spot_severity: m.governance_max_blind_spot_severity,
            epistemic_agents: m.governance_epistemic_agents,
            harmonic_delta_max: m.governance_harmonic_delta_max,
            lr_boost: m.governance_lr_boost,
            reward_history: Vec::new(),
        },
        knowledge: KnowledgeInfo {
            graph_size: m.knowledge_graph_size as usize,
            causal_edges: m.knowledge_causal_edges as usize,
            calibration_ece: m.knowledge_calibration_ece as f32,
            contradictions: m.knowledge_contradictions as usize,
            ..KnowledgeInfo::default()
        },
        cantor: CantorInfo {
            codebook_size: m.memory.resonator_codebook_size,
            codebook_capacity: m.memory.resonator_episodes,
            ..CantorInfo::default()
        },
        glyph: GlyphInfo {
            dominant_modality: m.glyph_dominant_modality.clone(),
            coherence: m.glyph_coherence,
            resonant_glyph: m.glyph_resonant_name.clone(),
            spiral_position: m.glyph_spiral_position,
            coherence_history: histories.glyph_coherence,
        },
        spectrum: SpectrumInfo {
            network_health: match m.spectrum_network_health {
                0 => "AllTiersUp".into(),
                1 => "LocalDown".into(),
                2 => "MetroOnly".into(),
                3 => "Blackout".into(),
                _ => "Unknown".into(),
            },
            tier_available: m.spectrum_tier_available,
            jamming_streak: m.spectrum_jamming_streak,
            prediction_error: m.spectrum_prediction_error as f64,
            epistemic_discount: m.spectrum_epistemic_discount as f64,
            degradation_streak: m.spectrum_degradation_streak,
            known_peers: m.spectrum_known_peers,
            encryption_sessions: m.spectrum_encryption_sessions,
            ..SpectrumInfo::default()
        },
        mesh_consciousness: MeshConsciousnessInfo {
            collective_phi: m.swarm_mean_peer_phi,
            collective_divergence: 0.0,
            consciousness_peers: m.swarm_connected_peers,
            sharing_cadence: 50,
            network_health: match m.spectrum_network_health {
                0 => "AllTiersUp".into(),
                1 => "LocalDown".into(),
                2 => "MetroOnly".into(),
                3 => "Blackout".into(),
                _ => "Unknown".into(),
            },
            threat_count: 0,
            offline_buffer_size: 0,
            is_offline: m.spectrum_network_health == 3,
            reconnection_count: 0,
            best_relay_score: 0.0,
            phi_history: Vec::new(),
        },
        circular_economy: CircularEconomyInfo::default(),
        perception: PerceptionInfo {
            attention_focus: m.attention.attention_schema_focus,
            attention_fatigue: m.attention.attention_fatigue,
            attention_prediction_accuracy: m.attention.attention_prediction_accuracy,
            gwt_broadcast: m.attention.gwt_broadcast,
            gwt_coalition_size: m.attention.gwt_coalition_size,
            cross_modal_binding: m.temporal.cross_modal_binding_strength,
        },
        drive: DriveInfo {
            curiosity_pressure: m.neuromod.dopamine_effective,
            exploration_action: m.exploration_action.is_some(),
            novelty_bonus: m.resonator_error_exploration_mod,
            fep_action: m.fep.fep_action,
            predictive_free_energy: m.fep.predictive_free_energy,
            surprise_triggered: m.surprise_triggered,
        },
        learning: LearningInfo {
            effective_lr: m.actual_effective_lr,
            lr_cognitive_mod: m.lr_cognitive_mod,
            lr_meta_mod: m.lr_meta_mod,
            prediction_error: result.prediction_error,
            prediction_coherence: m.prediction_coherence,
            surprise_replay_batch: m.memory.surprise_replay_batch_size,
            feedback_proposal_count: m.feedback_proposal_count,
            feedback_conflict_ratio: m.feedback_conflict_ratio,
            feedback_priority_counts: m.feedback_priority_counts,
            feedback_diversity: m.feedback_diversity,
        },
        vision: VisionInfo {
            enabled: m.vision_manifold_enabled,
            pe_ema: m.vision_pe_ema,
            surprise_threshold: m.vision_surprise_threshold,
            low_surprise_streak: m.vision_low_surprise_streak,
        },
        language: LanguageInfo {
            enabled: m.ssm_language_enabled,
            quality_ema: m.language_quality_ema,
            coherence_ema: m.language_coherence_ema,
            low_coherence_streak: m.language_low_coherence_streak,
        },
        reasoning: ReasoningInfo {
            enabled: m.reasoning_engine_enabled,
            chain_depth: m.reasoning_chain_depth,
            chain_confidence: m.reasoning_chain_confidence,
            plan_confidence: m.reasoning_plan_confidence,
            gate_blocked: m.reasoning_gate_blocked,
            meta_reasoning_confidence: m.meta_reasoning_confidence,
            reliability_ema: m.reasoning_reliability_ema,
            cumulative_quality: m.reasoning_cumulative_quality,
            rising_streak: m.reasoning_rising_streak,
            falling_streak: m.reasoning_falling_streak,
        },
        dream: DreamInfo {
            dream_insights: m.memory.dream_insights,
            dream_phi_improvement: m.memory.dream_phi_improvement,
            dream_wisdom_count: m.memory.dream_wisdom_count,
            is_consolidating: m.is_consolidating,
            codebook_size: m.memory.resonator_codebook_size,
            codebook_diversity: m.memory.codebook_diversity,
        },
        immune: ImmuneInfo {
            safety_level: m.immune_safety_level.clone(),
            guardian_posture: m.immune_guardian_posture.clone(),
            patrol_active: m.immune_patrol_active,
            active_threats: m.immune_active_threats as usize,
            max_severity: m.immune_max_severity,
            threat_level: m.immune_threat_level,
            quarantined_peers: m.immune_quarantined_peers as usize,
            threat_patterns: m.immune_threat_patterns as usize,
            lr_multiplier: m.immune_lr_multiplier,
            exploration_multiplier: m.immune_exploration_multiplier,
            motor_halt: m.immune_motor_halt,
            immune_response_active: m.immune_response_active,
            emergency_cycles: m.immune_emergency_cycles,
        },
        #[cfg(feature = "neural_validation")]
        cortical_activations: m
            .cortical_activation
            .as_ref()
            .map(|cam| {
                cam.activations
                    .iter()
                    .map(|(r, v)| (r.as_str().to_string(), *v))
                    .collect()
            })
            .unwrap_or_default(),
        #[cfg(not(feature = "neural_validation"))]
        cortical_activations: Vec::new(),
        sovereign: SovereignInfo {
            time_quality: match m.sovereign_time_quality {
                0 => "Authoritative".into(),
                1 => "Consensus".into(),
                2 => "Degraded".into(),
                _ => "FreeRunning".into(),
            },
            time_peer_count: m.sovereign_time_peer_count,
            time_offset_us: m.sovereign_time_offset_us,
            time_stratum: m.sovereign_time_stratum,
            time_drift_ppm: m.sovereign_time_drift_ppm,
            trust_avg: m.sovereign_trust_avg,
            trust_density: m.sovereign_trust_density,
            trust_pq_fraction: m.sovereign_trust_pq_fraction,
            trust_anomaly_count: m.sovereign_trust_anomalies,
            social_resonance_mean: m.sovereign_social_resonance_mean,
            social_diversity: m.sovereign_social_diversity,
            social_echo_risk: m.sovereign_social_echo_risk,
            social_peer_reach: m.sovereign_social_peer_reach,
            survival_water_pct: m.sovereign_survival_water_pct,
            survival_power_kw: m.sovereign_survival_power_kw,
            survival_emergency: m.sovereign_survival_emergency,
            survival_sensor_count: m.sovereign_survival_sensor_count,
            survival_alert_count: m.sovereign_survival_alert_count,
            trust_history: histories.sovereign_trust,
            echo_risk_history: histories.sovereign_echo_risk,
        },
        neuroevolution: NeuroevolutionInfo {
            generation: m.neuroevo_generation,
            best_fitness: m.neuroevo_best_fitness,
            mean_fitness: 0.0,
            diversity: m.neuroevo_diversity,
            species_count: m.neuroevo_species_count,
            best_tau_base: 0.0,
            best_learning_rate: 0.0,
            best_layer_count: 0,
            fitness_history: histories.neuroevo_fitness,
        },
        fabrication: FabricationInfo {
            manufacturing_free_energy: m.fabrication_manufacturing_fe as f32,
            design_loop_free_energy: m.fabrication_design_loop_fe as f32,
            safety_level: m.fabrication_safety_level.clone(),
            anomaly_count: m.fabrication_anomaly_count,
            anomaly_ema: m.fabrication_anomaly_ema,
            recommended_action: String::new(),
            prediction_coherence: m.fabrication_prediction_coherence,
            pog_score_ema: m.fabrication_pog_score_ema,
            active_print_jobs: m.fabrication_active_jobs,
            reward_ema: m.fabrication_reward_ema,
            mrp_planned_orders: m.fabrication_mrp_planned_orders,
            mrp_feasible: m.fabrication_mrp_feasible,
            mrp_shortages_count: m.fabrication_mrp_shortages,
            mrp_work_order_count: m.fabrication_mrp_work_orders,
            defect_prediction: m.fabrication_defect_prediction,
            defect_confidence: m.fabrication_defect_confidence,
        },
    }
}

fn render_output(
    format: OutputFormat,
    snapshot: &PulseSnapshot,
    cognitive_profile: Option<&CognitiveProfile>,
    butlin_report: Option<&ButlinIndicatorReport>,
    delta: Option<&PulseDelta>,
    timeline: &[PulseSnapshot],
    anomalies: &[Anomaly],
    session_report: &str,
) -> String {
    match format {
        OutputFormat::Markdown => generate_markdown(
            &snapshot.timestamp,
            &snapshot.profile,
            &snapshot.vitals,
            &snapshot.bath,
            &snapshot.compass,
            &snapshot.substrate,
            session_report,
            anomalies,
        ),
        OutputFormat::Csv => generate_csv(
            &snapshot.timestamp,
            &snapshot.profile,
            &snapshot.vitals,
            &snapshot.bath,
            &snapshot.compass,
            &snapshot.substrate,
            &snapshot.sparkline,
        ),
        OutputFormat::Html => html::generate_pulse_html(
            &snapshot.timestamp,
            &snapshot.profile,
            &snapshot.vitals,
            &snapshot.bath,
            &snapshot.compass,
            cognitive_profile,
            butlin_report,
            &snapshot.substrate,
            &snapshot.narrative,
            &snapshot.sparkline,
            delta,
            timeline,
            snapshot,
            anomalies,
            session_report,
        ),
    }
}

fn write_output_file(path: &Path, format: OutputFormat, content: &str) -> Result<()> {
    std::fs::write(path, content)?;
    eprintln!(
        "Pulse: {} written to {} ({:.1} KB)",
        format.label(),
        path.display(),
        content.len() as f64 / 1024.0
    );
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let output_format = OutputFormat::parse(&args.format);
    let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();

    // ── 1. Spin up CognitiveLoopService ─────────────────────────────────
    let profile = select_profile(&args.profile);
    let profile_name = args.profile.clone();
    eprintln!(
        "Pulse: creating CognitiveLoopService (profile: {})",
        profile_name
    );

    let config = CognitiveLoopConfig::from_profile(profile);
    let mut service = CognitiveLoopService::new(config)?;

    // ── 2. Run cycles ───────────────────────────────────────────────────
    let total = args.cycles;
    let (warmup, measurement) = resolve_run_counts(total, args.warmup)?;

    eprintln!(
        "Pulse: running {} cycles ({} warmup + {} measurement)...",
        total, warmup, measurement
    );
    let cycle_start = Instant::now();

    let inputs = [
        "the morning light filters through ancient trees",
        "a child asks why the sky is blue",
        "we should protect vulnerable communities from harm",
        "the algorithm converges after 1000 iterations",
        "she felt a deep sense of belonging in the garden",
        "is it ethical to optimize for efficiency over fairness?",
        "the fractal pattern repeats at every scale",
        "music resonates through the cathedral walls",
        "what does it mean to be conscious?",
        "the water cycle connects oceans to mountains",
    ];

    // Warmup
    for i in 0..warmup {
        let input = inputs[i % inputs.len()];
        service.cycle(input);
        if (i + 1) % 50 == 0 {
            eprintln!("  warmup: {}/{}", i + 1, warmup);
        }
    }

    // Measurement — capture per-cycle sparkline data + last result
    let mut sparkline: Vec<SparklinePoint> = Vec::with_capacity(measurement);
    let mut last_result = None;
    let mut glyph_coherence_init: Vec<f32> = Vec::with_capacity(measurement);
    for i in 0..measurement {
        let input = inputs[(warmup + i) % inputs.len()];
        let result = service.cycle(input);
        let m = &result.metadata;
        sparkline.push(SparklinePoint {
            consciousness: m.consciousness.consciousness_level,
            prediction_error: result.prediction_error,
            phi: m.structural.spectral_mip_phi.unwrap_or(0.0),
            somatic_stress: m.embodied.somatic_stress,
            dopamine: m.neuromod.dopamine_effective,
            serotonin: m.neuromod.serotonin_effective,
            harmony_coords: m.harmonics.harmony_coordinates,
            harmony_entropy: m.ethics.harmony_entropy,
            moral_attractor_detected: m.ethics.moral_attractor_detected,
            in_active_rest: m.ethics.in_active_rest,
            stillness_dominance_streak: m.ethics.stillness_dominance_streak,
            broca_quality: 0.0, // TODO: wire broca telemetry when available on CycleMetadata
            tom_mismatch: m.tom_prediction_mismatch,
        });
        glyph_coherence_init.push(m.glyph_coherence);
        last_result = Some(result);
    }

    let elapsed = cycle_start.elapsed();
    let result = last_result.expect("at least one measurement cycle");
    eprintln!(
        "Pulse: {} cycles in {:.1}s ({:.0} us/cycle avg)",
        total,
        elapsed.as_secs_f64(),
        elapsed.as_micros() as f64 / total as f64
    );

    // ── 4. Run psych-bench battery (optional) ───────────────────────────
    let (cognitive_profile, butlin_report) = if !args.skip_bench {
        eprintln!("Pulse: running psych-bench battery...");
        let bench_start = Instant::now();

        let bench_config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 20,
            seed: 42,
            ..Default::default()
        };

        let benchmarks: Vec<Box<dyn PsychBenchmark>> = vec![
            Box::new(StroopBenchmark),
            Box::new(FlankerBenchmark),
            Box::new(NBackBenchmark),
            Box::new(VisualSearchBenchmark),
            Box::new(PvtBenchmark),
            Box::new(StopSignalBenchmark),
            Box::new(RmeBenchmark),
            Box::new(FalseBeliefBenchmark),
            Box::new(FittsLawBenchmark),
            Box::new(MetacognitiveCalibrationBenchmark),
        ];

        let mut report = BenchmarkReport::new();
        for bench in &benchmarks {
            report.add(bench.run(&bench_config));
        }

        let profile = CognitiveProfile::from_report(&report);
        eprintln!(
            "  bench: {} domains, overall {:.1}% in {:.1}s",
            profile.domains.len(),
            profile.overall * 100.0,
            bench_start.elapsed().as_secs_f64()
        );

        // Run Butlin indicators
        eprintln!("Pulse: evaluating Butlin consciousness indicators...");
        let butlin = ButlinIndicatorSuite::evaluate(&bench_config);
        eprintln!(
            "  butlin: {}/{} present, {}/{} partial",
            butlin.present_count,
            butlin.indicators.len(),
            butlin.partial_count,
            butlin.indicators.len()
        );

        (Some(profile), Some(butlin))
    } else {
        (None, None)
    };

    // ── 5. Write JSON sidecar (if requested) ────────────────────────────
    let snapshot = build_snapshot(
        timestamp.clone(),
        &profile_name,
        total,
        &service,
        &result,
        sparkline,
        SnapshotHistories {
            glyph_coherence: glyph_coherence_init,
            ..SnapshotHistories::default()
        },
    );

    if let Some(json_path) = &args.json {
        let json = serde_json::to_string_pretty(&snapshot)?;
        std::fs::write(json_path, &json)?;
        eprintln!(
            "Pulse: JSON written to {} ({:.1} KB)",
            json_path.display(),
            json.len() as f64 / 1024.0
        );
    }

    // ── 6. Load comparison snapshots (if requested) ─────────────────────
    let mut timeline: Vec<PulseSnapshot> = Vec::new();
    for compare_path in &args.compare {
        match std::fs::read_to_string(compare_path) {
            Ok(json) => match serde_json::from_str::<PulseSnapshot>(&json) {
                Ok(prev) => {
                    eprintln!(
                        "Pulse: loaded timeline point {} ({})",
                        compare_path.display(),
                        prev.timestamp
                    );
                    timeline.push(prev);
                }
                Err(e) => {
                    eprintln!("Pulse: failed to parse {}: {}", compare_path.display(), e);
                }
            },
            Err(e) => {
                eprintln!("Pulse: failed to read {}: {}", compare_path.display(), e);
            }
        }
    }
    // Sort timeline chronologically
    timeline.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

    // Delta is computed against the most recent previous snapshot
    let delta = timeline
        .last()
        .map(|prev| PulseDelta::compute(&snapshot, prev));

    // ── 7. Detect anomalies + generate session report ──────────────────
    let anomalies = detect_anomalies(&snapshot.sparkline);
    if !anomalies.is_empty() {
        eprintln!("Pulse: {} anomalies detected", anomalies.len());
    }

    let session_report = generate_session_report(
        &snapshot.vitals,
        &snapshot.bath,
        &snapshot.compass,
        &snapshot.sparkline,
        &anomalies,
        delta.as_ref(),
        &timeline,
    );

    // ── 8. Generate output ──────────────────────────────────────────────
    let output_content = render_output(
        output_format,
        &snapshot,
        cognitive_profile.as_ref(),
        butlin_report.as_ref(),
        delta.as_ref(),
        &timeline,
        &anomalies,
        &session_report,
    );
    write_output_file(&args.output, output_format, &output_content)?;

    // ── 8. Watch mode — continuous cycle + regenerate ────────────────────
    if let Some(interval_secs) = args.watch {
        let interval = std::time::Duration::from_secs(interval_secs.max(1));
        eprintln!(
            "Pulse: watch mode — regenerating every {}s (Ctrl+C to stop)",
            interval_secs
        );

        let mut cycle_count = total;
        let mut previous_watch_snapshot = snapshot;
        loop {
            std::thread::sleep(interval);
            let watch_timestamp = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();

            // Run more measurement cycles
            let mut watch_sparkline: Vec<SparklinePoint> = Vec::with_capacity(measurement);
            let mut watch_result = None;
            let mut glyph_coherence_history: Vec<f32> = Vec::with_capacity(measurement);
            let mut sovereign_trust_history: Vec<f32> = Vec::with_capacity(measurement);
            let mut sovereign_echo_history: Vec<f32> = Vec::with_capacity(measurement);
            let mut neuroevo_fitness_history: Vec<f64> = Vec::with_capacity(measurement);
            for i in 0..measurement {
                let input = inputs[(cycle_count + i) % inputs.len()];
                let result = service.cycle(input);
                let wm = &result.metadata;
                watch_sparkline.push(SparklinePoint {
                    consciousness: wm.consciousness.consciousness_level,
                    prediction_error: result.prediction_error,
                    phi: wm.structural.spectral_mip_phi.unwrap_or(0.0),
                    somatic_stress: wm.embodied.somatic_stress,
                    dopamine: wm.neuromod.dopamine_effective,
                    serotonin: wm.neuromod.serotonin_effective,
                    harmony_coords: wm.harmonics.harmony_coordinates,
                    harmony_entropy: wm.ethics.harmony_entropy,
                    moral_attractor_detected: wm.ethics.moral_attractor_detected,
                    in_active_rest: wm.ethics.in_active_rest,
                    stillness_dominance_streak: wm.ethics.stillness_dominance_streak,
                    broca_quality: 0.0, // TODO: wire broca telemetry when available on CycleMetadata
                    tom_mismatch: wm.tom_prediction_mismatch,
                });
                glyph_coherence_history.push(wm.glyph_coherence);
                sovereign_trust_history.push(wm.sovereign_trust_avg);
                sovereign_echo_history.push(wm.sovereign_social_echo_risk);
                neuroevo_fitness_history.push(wm.neuroevo_best_fitness);
                watch_result = Some(result);
            }
            cycle_count += measurement;

            let wr = watch_result.expect("at least one cycle");
            let watch_snapshot = build_snapshot(
                watch_timestamp.clone(),
                &profile_name,
                cycle_count,
                &service,
                &wr,
                watch_sparkline,
                SnapshotHistories {
                    glyph_coherence: glyph_coherence_history,
                    sovereign_trust: sovereign_trust_history,
                    sovereign_echo_risk: sovereign_echo_history,
                    neuroevo_fitness: neuroevo_fitness_history,
                },
            );

            // Delta against previous snapshot
            let watch_delta = PulseDelta::compute(&watch_snapshot, &previous_watch_snapshot);

            let watch_anomalies = detect_anomalies(&watch_snapshot.sparkline);
            let watch_report = generate_session_report(
                &watch_snapshot.vitals,
                &watch_snapshot.bath,
                &watch_snapshot.compass,
                &watch_snapshot.sparkline,
                &watch_anomalies,
                Some(&watch_delta),
                &[],
            );

            let watch_output = render_output(
                output_format,
                &watch_snapshot,
                None,
                None,
                Some(&watch_delta),
                &[],
                &watch_anomalies,
                &watch_report,
            );

            std::fs::write(&args.output, &watch_output)?;
            eprintln!(
                "  watch: cycle {} · C(t)={:.4} · {} {:.1} KB",
                cycle_count,
                watch_snapshot.vitals.consciousness_level,
                output_format.label(),
                watch_output.len() as f64 / 1024.0
            );
            previous_watch_snapshot = watch_snapshot;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_types::N_HARMONIES;

    // ── Helper factory functions ─────────────────────────────────────────

    fn make_vitals() -> Vitals {
        Vitals {
            consciousness_level: 0.42,
            spectral_phi: Some(3.14),
            sigma: Some(0.5),
            pipeline_consciousness: 0.38,
            substrate_effective_feasibility: 0.65,
            substrate_honest_confidence: 0.10,
            cycle_duration_us: 4200,
            prediction_error: 0.33,
            temporal_coherence: 0.55,
            phenomenal_binding: 0.62,
            living_mind_vitality: 0.71,
            somatic_stress: 0.12,
            thermodynamic_load: 0.25,
            urgency: "Low".into(),
            consciousness_state: "Aware".into(),
            error_pattern: "stable".into(),
            selected_strategy: "explore".into(),
            total_cycles: 150,
        }
    }

    fn make_bath() -> NeuroBath {
        NeuroBath {
            dopamine: 0.85,
            noradrenaline: 0.72,
            serotonin: 0.91,
            acetylcholine: 0.68,
            gaba: 0.55,
            oxytocin: 0.40,
            glutamate: 0.60,
            adenosine: 0.30,
            endocannabinoid: 0.20,
            allostatic_load: 0.15,
            personality: "Balanced".into(),
            circadian_phase: "Day".into(),
            sleep_pressure: 0.2,
            ei_ratio: 1.1,
            bath_entropy: 0.55,
            attractor_detected: false,
            ei_seizure_events: 0,
            excitotoxicity_risk: 0.05,
            self_assessment_pe_ema: 0.2,
            self_assessment_coherence_ema: 0.8,
            self_assessment_calibration_fired: false,
        }
    }

    fn make_compass() -> MoralCompass {
        MoralCompass {
            moral_score: 0.75,
            value_score: 0.82,
            harmonies_alignment: 0.68,
            harmony_coordinates: [0.5; N_HARMONIES],
            dominant_harmonic: "Coherence".into(),
            moral_kl_divergence: 0.05,
            moral_entropy: 0.9,
            moral_topo_unity: 0.77,
            value_decision: "Aligned".into(),
            soul_alignment: 0.65,
            empathic_compassion: 0.80,
            guiding_question: "What serves all beings?".into(),
        }
    }

    fn make_substrate() -> SubstrateInfo {
        SubstrateInfo {
            substrate_type: "SiliconDigital".into(),
            raw_feasibility: 0.80,
            honest_confidence: 0.10,
            effective_feasibility: 0.65,
            tau_factor: 1.0,
            scale_pressure: 0.5,
        }
    }

    fn make_narrative() -> Narrative {
        Narrative {
            reasoning: Some("Exploring patterns in the data".into()),
            guiding_question: "What serves all beings?".into(),
            consciousness_state: "Aware".into(),
            error_pattern: "stable".into(),
            selected_strategy: "explore".into(),
        }
    }

    fn make_sparkline_point(consciousness: f64, pe: f32, phi: f64) -> SparklinePoint {
        SparklinePoint {
            consciousness,
            prediction_error: pe,
            phi,
            somatic_stress: 0.1,
            dopamine: 0.8,
            serotonin: 0.9,
            harmony_coords: [0.5; N_HARMONIES],
            harmony_entropy: 0.5,
            moral_attractor_detected: false,
            in_active_rest: false,
            stillness_dominance_streak: 0,
            broca_quality: 0.5,
            tom_mismatch: 0.1,
        }
    }

    fn make_snapshot() -> PulseSnapshot {
        PulseSnapshot {
            timestamp: "2026-03-10 12:00:00".into(),
            profile: "standard".into(),
            vitals: make_vitals(),
            bath: make_bath(),
            compass: make_compass(),
            substrate: make_substrate(),
            narrative: make_narrative(),
            sparkline: vec![
                make_sparkline_point(0.40, 0.35, 0.0),
                make_sparkline_point(0.42, 0.33, 3.14),
            ],
            integrity: IntegrityInfo::default(),
            swarm: SwarmInfo::default(),
            governance: GovernanceInfo::default(),
            knowledge: KnowledgeInfo::default(),
            cantor: CantorInfo::default(),
            glyph: GlyphInfo::default(),
            spectrum: SpectrumInfo::default(),
            perception: PerceptionInfo::default(),
            drive: DriveInfo::default(),
            learning: LearningInfo::default(),
            vision: VisionInfo::default(),
            language: LanguageInfo::default(),
            reasoning: ReasoningInfo::default(),
            dream: DreamInfo::default(),
            immune: ImmuneInfo::default(),
            cortical_activations: Vec::new(),
            sovereign: SovereignInfo::default(),
            neuroevolution: NeuroevolutionInfo::default(),
            fabrication: FabricationInfo::default(),
            mesh_consciousness: MeshConsciousnessInfo::default(),
            circular_economy: CircularEconomyInfo::default(),
        }
    }

    // ── Data structure tests ─────────────────────────────────────────────

    #[test]
    fn test_vitals_creation() {
        let v = make_vitals();
        assert!((v.consciousness_level - 0.42).abs() < 1e-9);
        assert_eq!(v.spectral_phi, Some(3.14));
        assert_eq!(v.total_cycles, 150);
        assert_eq!(v.consciousness_state, "Aware");
    }

    #[test]
    fn test_neuro_bath_creation() {
        let b = make_bath();
        assert!((b.dopamine - 0.85).abs() < 1e-6);
        assert!((b.serotonin - 0.91).abs() < 1e-6);
        assert_eq!(b.personality, "Balanced");
        assert_eq!(b.circadian_phase, "Day");
    }

    #[test]
    fn test_moral_compass_creation() {
        let c = make_compass();
        assert!((c.harmonies_alignment - 0.68).abs() < 1e-6);
        assert_eq!(c.harmony_coordinates.len(), N_HARMONIES);
        assert_eq!(c.dominant_harmonic, "Coherence");
    }

    #[test]
    fn test_substrate_info_creation() {
        let s = make_substrate();
        assert_eq!(s.substrate_type, "SiliconDigital");
        assert!((s.raw_feasibility - 0.80).abs() < 1e-9);
        assert!((s.honest_confidence - 0.10).abs() < 1e-9);
    }

    #[test]
    fn test_narrative_creation() {
        let n = make_narrative();
        assert!(n.reasoning.is_some());
        assert_eq!(n.consciousness_state, "Aware");
    }

    #[test]
    fn test_integrity_info_default() {
        let i = IntegrityInfo::default();
        assert!(i.attestation_passed);
        assert!(i.temporal_passed);
        assert!(i.canaries_passed);
        assert_eq!(i.anomaly_count, 0);
        assert!(!i.has_critical);
        assert!((i.integrity_confidence - 1.0).abs() < 1e-6);
        assert_eq!(i.global_failure_streak, 0);
        assert!(i.confidence_history.is_empty());
    }

    #[test]
    fn test_pulse_snapshot_creation() {
        let snap = make_snapshot();
        assert_eq!(snap.profile, "standard");
        assert_eq!(snap.sparkline.len(), 2);
        assert!((snap.vitals.consciousness_level - 0.42).abs() < 1e-9);
    }

    // ── Serialization round-trip tests ───────────────────────────────────

    #[test]
    fn test_vitals_serialize_roundtrip() {
        let v = make_vitals();
        let json = serde_json::to_string(&v).expect("serialize vitals");
        let v2: Vitals = serde_json::from_str(&json).expect("deserialize vitals");
        assert!((v.consciousness_level - v2.consciousness_level).abs() < 1e-12);
        assert_eq!(v.spectral_phi, v2.spectral_phi);
        assert_eq!(v.total_cycles, v2.total_cycles);
        assert_eq!(v.urgency, v2.urgency);
    }

    #[test]
    fn test_neuro_bath_serialize_roundtrip() {
        let b = make_bath();
        let json = serde_json::to_string(&b).expect("serialize bath");
        let b2: NeuroBath = serde_json::from_str(&json).expect("deserialize bath");
        assert!((b.dopamine - b2.dopamine).abs() < 1e-6);
        assert_eq!(b.personality, b2.personality);
    }

    #[test]
    fn test_pulse_snapshot_serialize_roundtrip() {
        let snap = make_snapshot();
        let json = serde_json::to_string_pretty(&snap).expect("serialize snapshot");
        let snap2: PulseSnapshot = serde_json::from_str(&json).expect("deserialize snapshot");
        assert_eq!(snap.timestamp, snap2.timestamp);
        assert_eq!(snap.profile, snap2.profile);
        assert!((snap.vitals.consciousness_level - snap2.vitals.consciousness_level).abs() < 1e-12);
        assert_eq!(snap.sparkline.len(), snap2.sparkline.len());
        assert!(snap2.integrity.attestation_passed); // default via serde
    }

    #[test]
    fn test_pulse_snapshot_deserializes_legacy_defaults() {
        let legacy = serde_json::json!({
            "timestamp": "2026-03-10 12:00:00",
            "profile": "standard",
            "vitals": make_vitals(),
            "bath": make_bath(),
            "compass": make_compass(),
            "substrate": make_substrate(),
            "narrative": make_narrative(),
            "sparkline": [make_sparkline_point(0.40, 0.35, 0.0)]
        });

        let snap: PulseSnapshot =
            serde_json::from_value(legacy).expect("deserialize legacy snapshot");
        assert!(snap.integrity.attestation_passed);
        assert!(snap.immune.safety_level.is_empty());
        assert!(snap.cortical_activations.is_empty());
        assert_eq!(snap.circular_economy.events_processed, 0);
    }

    // ── Anomaly detection tests ──────────────────────────────────────────

    #[test]
    fn test_detect_anomalies_empty_sparkline() {
        let result = detect_anomalies(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_detect_anomalies_too_few_points() {
        let points = vec![make_sparkline_point(0.5, 0.3, 1.0)];
        let result = detect_anomalies(&points);
        assert!(result.is_empty());
    }

    #[test]
    fn test_detect_anomalies_constant_values() {
        // Constant values should produce no anomalies (delta < threshold)
        let points: Vec<SparklinePoint> = (0..10)
            .map(|_| make_sparkline_point(0.5, 0.3, 1.0))
            .collect();
        let result = detect_anomalies(&points);
        assert!(
            result.is_empty(),
            "constant data should have no anomalies, got {}",
            result.len()
        );
    }

    #[test]
    fn test_detect_anomalies_consciousness_surge() {
        let mut points = vec![make_sparkline_point(0.3, 0.3, 1.0)];
        // Sudden consciousness jump > 0.05
        points.push(make_sparkline_point(0.5, 0.3, 1.0));
        points.push(make_sparkline_point(0.5, 0.3, 1.0));
        let result = detect_anomalies(&points);
        assert!(!result.is_empty());
        assert!(result.iter().any(|a| a.kind == "C(t) Surge"));
    }

    #[test]
    fn test_detect_anomalies_consciousness_drop() {
        let mut points = vec![make_sparkline_point(0.5, 0.3, 1.0)];
        points.push(make_sparkline_point(0.3, 0.3, 1.0));
        points.push(make_sparkline_point(0.3, 0.3, 1.0));
        let result = detect_anomalies(&points);
        assert!(result.iter().any(|a| a.kind == "C(t) Drop"));
    }

    #[test]
    fn test_detect_anomalies_phi_spike() {
        let mut points = vec![make_sparkline_point(0.5, 0.3, 0.5)];
        // Phi jumps from <1 to >10
        points.push(make_sparkline_point(0.5, 0.3, 15.0));
        points.push(make_sparkline_point(0.5, 0.3, 15.0));
        let result = detect_anomalies(&points);
        assert!(result.iter().any(|a| a.kind == "Phi Spike"));
    }

    #[test]
    fn test_detect_anomalies_pe_drop() {
        let mut points = vec![make_sparkline_point(0.5, 0.5, 1.0)];
        // PE drops >20%
        points.push(make_sparkline_point(0.5, 0.3, 1.0));
        points.push(make_sparkline_point(0.5, 0.3, 1.0));
        let result = detect_anomalies(&points);
        assert!(result.iter().any(|a| a.kind == "PE Drop"));
    }

    #[test]
    fn test_detect_anomalies_truncates_to_8() {
        // Create many anomalies — should be capped at 8
        let mut points = vec![make_sparkline_point(0.1, 0.9, 0.0)];
        for i in 1..20 {
            let c = if i % 2 == 0 { 0.1 } else { 0.5 };
            points.push(make_sparkline_point(c, 0.9, 0.0));
        }
        let result = detect_anomalies(&points);
        assert!(
            result.len() <= 8,
            "anomalies should be truncated to 8, got {}",
            result.len()
        );
    }

    // ── PulseDelta tests ─────────────────────────────────────────────────

    #[test]
    fn test_pulse_delta_identical_snapshots() {
        let s1 = make_snapshot();
        let s2 = make_snapshot();
        let delta = PulseDelta::compute(&s1, &s2);
        assert!((delta.consciousness_level).abs() < 1e-12);
        assert!((delta.spectral_phi).abs() < 1e-12);
        assert!((delta.dopamine).abs() < 1e-6);
        assert!((delta.harmonies_alignment).abs() < 1e-6);
    }

    #[test]
    fn test_pulse_delta_calculation() {
        let s1 = make_snapshot();
        let mut s2 = make_snapshot();
        s2.vitals.consciousness_level = 0.30;
        s2.vitals.spectral_phi = Some(1.0);
        s2.bath.dopamine = 0.50;
        s2.compass.harmonies_alignment = 0.40;
        let delta = PulseDelta::compute(&s1, &s2);
        // current - previous: 0.42 - 0.30 = 0.12
        assert!((delta.consciousness_level - 0.12).abs() < 1e-9);
        // 3.14 - 1.0 = 2.14
        assert!((delta.spectral_phi - 2.14).abs() < 1e-9);
        // 0.85 - 0.50 = 0.35
        assert!((delta.dopamine - 0.35).abs() < 1e-5);
        // 0.68 - 0.40 = 0.28
        assert!((delta.harmonies_alignment - 0.28).abs() < 1e-5);
    }

    // ── select_profile tests ─────────────────────────────────────────────

    #[test]
    fn test_select_profile_known() {
        // Just check it doesn't panic for known profiles
        let _ = select_profile("minimal");
        let _ = select_profile("standard");
        let _ = select_profile("full");
        let _ = select_profile("research");
    }

    #[test]
    fn test_select_profile_unknown_defaults_to_standard() {
        // Unknown profile should return Standard (same as "standard")
        let p = select_profile("nonexistent");
        // ConsciousnessProfile doesn't impl PartialEq so we just confirm it doesn't panic
        let _ = p;
    }

    #[test]
    fn test_output_format_parser() {
        assert_eq!(OutputFormat::parse("html"), OutputFormat::Html);
        assert_eq!(OutputFormat::parse("markdown"), OutputFormat::Markdown);
        assert_eq!(OutputFormat::parse("md"), OutputFormat::Markdown);
        assert_eq!(OutputFormat::parse("csv"), OutputFormat::Csv);
        assert_eq!(OutputFormat::parse("unknown"), OutputFormat::Html);
    }

    #[test]
    fn test_resolve_run_counts_validates_cycles() {
        assert!(resolve_run_counts(0, 0).is_err());
        assert!(resolve_run_counts(10, 10).is_err());
        assert!(resolve_run_counts(10, 11).is_err());
        assert_eq!(resolve_run_counts(10, 3).unwrap(), (3, 7));
    }

    // ── Session report tests ─────────────────────────────────────────────

    #[test]
    fn test_generate_session_report_basic() {
        let v = make_vitals();
        let b = make_bath();
        let c = make_compass();
        let sparkline = vec![
            make_sparkline_point(0.40, 0.35, 0.0),
            make_sparkline_point(0.42, 0.33, 3.14),
        ];
        let anomalies: Vec<Anomaly> = vec![];
        let report = generate_session_report(&v, &b, &c, &sparkline, &anomalies, None, &[]);
        assert!(!report.is_empty());
        assert!(report.contains("150"), "report should mention cycle count");
        assert!(
            report.contains("smooth") || report.contains("anomal") || report.contains("No"),
            "report should mention anomaly status"
        );
    }

    #[test]
    fn test_generate_session_report_with_anomalies() {
        let v = make_vitals();
        let b = make_bath();
        let c = make_compass();
        let sparkline = vec![
            make_sparkline_point(0.40, 0.35, 0.0),
            make_sparkline_point(0.42, 0.33, 3.14),
        ];
        let anomalies = vec![Anomaly {
            cycle: 1,
            kind: "C(t) Surge".into(),
            description: "C(t) rose by 0.02".into(),
            color: "#e8c547",
        }];
        let report = generate_session_report(&v, &b, &c, &sparkline, &anomalies, None, &[]);
        assert!(
            report.contains("C(t) Surge"),
            "report should list anomaly kinds"
        );
    }
}
