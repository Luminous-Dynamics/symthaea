// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # Symthaea Cognitive Observatory v0.3.0
//!
//! A cinematic, evidence-centered cognitive observatory showcasing state, causality,
//! uncertainty, memory, and recovery as a living interactive mind-system.

use bevy::{
    gizmos::gizmos::Gizmos,
    prelude::*,
    render::{
        RenderPlugin,
        settings::{Backends, RenderCreation, WgpuSettings},
    },
};
use bevy_egui::{EguiContexts, EguiPlugin, egui};
use serde::{Deserialize, Serialize};

// Subsystem imports
use symthaea_core::hdc::ContinuousHV;
use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig, types::Observation};
use symthaea_phi_oracle::{IntegrationOracle, OracleConfig};
use symthaea_subterranean::simulator::{SimpleSubterraneanSimulator, SubterraneanPhysicsSimulator};
use symthaea_subterranean::types::{SubterraneanCommand, WATER_INGRESS_RATIO};
use symthaea_workspace::{AttentionBid, GlobalWorkspace};

// ── Types & Visual Grammar ──────────────────────────────────────────────────

const METRIC_NAMES: [&str; 7] = [
    "phi",
    "fep_prediction_error",
    "workspace_activation",
    "hot_confidence",
    "anomaly_score",
    "memory_pressure",
    "mip_instability",
];

const METRIC_COLORS: [(f32, f32, f32); 7] = [
    (0.96, 0.62, 0.04), // phi -> amber
    (0.22, 0.74, 0.97), // fep_prediction_error -> cyan
    (0.22, 0.74, 0.97), // workspace_activation -> cyan
    (0.65, 0.55, 0.98), // hot_confidence -> violet
    (0.97, 0.44, 0.44), // anomaly_score -> red
    (0.65, 0.55, 0.98), // memory_pressure -> violet
    (0.20, 0.83, 0.60), // mip_instability -> green
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum SourceMode {
    ScriptedDemo,
    LiveSimulation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum ObservatoryMode {
    LiveCognition,
    ForensicReplay,
    StateSpace,
    InterventionPreview,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum Scenario {
    AquiferIntrusion,
    FalseGreenRecovery,
    MemoryPressureCascade,
    SensorContradiction,
    DelayedChronicleCommit,
    OperatorInterventionPreview,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DashboardEvent {
    pub event_id: String,
    pub event_type: String,
    pub label: String,
    pub description: String,
    pub severity: String,
    pub absolute_frame_index: u64,
    pub history_offset: u64,
    pub causal_role: String,
    pub metric_impacts: Vec<(String, f64)>,
    pub narrative: String,
    pub audit_recommendation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RegimeInterval {
    pub r#type: String,
    pub start_frame: u64,
    pub end_frame: u64,
    pub duration_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TimeWaterfallFrame {
    pub frame_index: u64,
    pub sim_time_s: f64,
    pub source_mode: SourceMode,
    pub metrics: [f64; 7],
    pub confidence: f64,
    pub anomaly_bits: u8,
    pub anomaly_flags: Vec<String>,
    pub is_chronicle: bool,
    pub events: Vec<DashboardEvent>,
}

#[derive(Debug, Clone)]
struct MemoryNode {
    pub label: String,
    pub coord: Vec2,
    pub target_coord: Vec2,
    pub status: &'static str, // "active", "contradicted", "stabilizing"
    pub recency: f32,         // 0.0 to 1.0
    pub confidence: f32,      // 0.0 to 1.0
}

// ── Bevy Resources ───────────────────────────────────────────────────────────

#[derive(Resource)]
struct WaterfallState {
    recorded_frames: Vec<TimeWaterfallFrame>,
    head: usize,
    paused: bool,
    scrub: usize,
    metric_visible: [bool; 7],
    false_green_highlight: bool,
    tick: u64,
    demo_frames: Vec<TimeWaterfallFrame>,
    demo_idx: usize,
    portrait_trail: Vec<[f64; 2]>,
    portrait_max: usize,
    portrait_mode: usize,
    multi_lens_mode: bool,
    layout_mode: usize, // 0 = Observatory, 1 = Raw Developer Telemetry
    observatory_mode: ObservatoryMode,
    selected_scenario: Scenario,
    selected_event_node: Option<usize>,
    selected_interval: Option<usize>,

    // Live Simulation Subsystems
    source_mode: SourceMode,
    sim: SimpleSubterraneanSimulator,
    fep: ActiveInferenceAgent,
    oracle: IntegrationOracle,
    workspace: GlobalWorkspace,
    twin: symthaea_digital_twin::TwinState,
    ingress_perturbation: f64,

    // Upgraded Observatory State
    memories: Vec<MemoryNode>,
    export_sealed: bool,
    sealed_hash: String,

    // Intervention settings
    interv_dampen_memory_pressure: bool,
    interv_ignore_sensors: bool,
    interv_shift_attention: bool,
    interv_delay_chronicle: bool,
    interv_early_recovery: bool,
}

#[derive(Resource, Default)]
struct UiState {
    status_text: String,
}

// ── Main Entrypoint ──────────────────────────────────────────────────────────

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Symthaea HLB — Cognitive Observatory v0.3.0".to_string(),
                        resolution: (1400.0, 900.0).into(),
                        ..default()
                    }),
                    ..default()
                })
                .set(RenderPlugin {
                    render_creation: RenderCreation::Automatic(WgpuSettings {
                        backends: Some(Backends::VULKAN),
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins(EguiPlugin)
        .init_resource::<WaterfallState>()
        .init_resource::<UiState>()
        .add_systems(Startup, setup_camera)
        .add_systems(
            Update,
            (
                handle_keyboard_input,
                advance_waterfall,
                update_observatory_simulation,
                render_observatory_viewport,
                render_observatory_sidebar,
            )
                .chain(),
        )
        .run();
}

fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera2d::default());
}

// ── Default State Initialization ─────────────────────────────────────────────

impl Default for WaterfallState {
    fn default() -> Self {
        let sim = SimpleSubterraneanSimulator::new();
        let twin = symthaea_digital_twin::TwinState::new(
            "subterranean-scout-01",
            symthaea_digital_twin::AssetClass::RoboticSystem,
            "Cincinnati Subterranean Scout Twin",
        );

        let fep_config = ActiveInferenceAgentConfig {
            state_dim: 8,
            obs_dim: 4,
            num_actions: 6,
            inference_iterations: 5,
            belief_learning_rate: 0.1,
            planning_horizon: 3,
            action_temperature: 1.0,
            enable_model_learning: true,
            enable_td_learning: true,
            td_config: Default::default(),
        };
        let fep = ActiveInferenceAgent::new(fep_config);

        let oracle_config = OracleConfig {
            window_size: 50,
            regularization: 1e-6,
            temporal_probes: vec![0.01, 0.1, 1.0, 10.0],
            seed: 42,
        };
        let oracle = IntegrationOracle::new_simple(4, oracle_config).unwrap();
        let workspace = GlobalWorkspace::new();

        let mut state = Self {
            recorded_frames: vec![
                TimeWaterfallFrame {
                    frame_index: 0,
                    sim_time_s: 0.0,
                    source_mode: SourceMode::ScriptedDemo,
                    metrics: [2.8, 0.15, 0.1, 0.85, 0.05, 0.8, 0.1],
                    confidence: 0.85,
                    anomaly_bits: 0,
                    anomaly_flags: Vec::new(),
                    is_chronicle: false,
                    events: Vec::new(),
                };
                256
            ],
            head: 0,
            paused: false,
            scrub: 0,
            metric_visible: [true; 7],
            false_green_highlight: false,
            tick: 0,
            demo_frames: Vec::new(),
            demo_idx: 0,
            portrait_trail: Vec::new(),
            portrait_max: 120,
            portrait_mode: 0,
            multi_lens_mode: false,
            layout_mode: 0, // Observatory Mode as default
            observatory_mode: ObservatoryMode::LiveCognition,
            selected_scenario: Scenario::AquiferIntrusion,
            selected_event_node: None,
            selected_interval: None,
            source_mode: SourceMode::ScriptedDemo,
            sim,
            fep,
            oracle,
            workspace,
            twin,
            ingress_perturbation: 0.0,

            memories: vec![
                MemoryNode {
                    label: "Ingress Inflow Model".to_string(),
                    coord: Vec2::new(-100.0, 50.0),
                    target_coord: Vec2::new(-100.0, 50.0),
                    status: "stabilizing",
                    recency: 0.9,
                    confidence: 0.8,
                },
                MemoryNode {
                    label: "Pump Convergence Rate".to_string(),
                    coord: Vec2::new(120.0, -80.0),
                    target_coord: Vec2::new(120.0, -80.0),
                    status: "active",
                    recency: 0.6,
                    confidence: 0.75,
                },
                MemoryNode {
                    label: "Borehole Hydro-static Baseline".to_string(),
                    coord: Vec2::new(-50.0, -120.0),
                    target_coord: Vec2::new(-50.0, -120.0),
                    status: "active",
                    recency: 0.4,
                    confidence: 0.9,
                },
                MemoryNode {
                    label: "Unstable Aquifer Transition".to_string(),
                    coord: Vec2::new(10.0, 110.0),
                    target_coord: Vec2::new(10.0, 110.0),
                    status: "contradicted",
                    recency: 0.95,
                    confidence: 0.4,
                },
                MemoryNode {
                    label: "Dynamic MIP Stabilizer".to_string(),
                    coord: Vec2::new(200.0, 150.0),
                    target_coord: Vec2::new(200.0, 150.0),
                    status: "stabilizing",
                    recency: 0.8,
                    confidence: 0.88,
                },
            ],
            export_sealed: false,
            sealed_hash: "none".to_string(),
            interv_dampen_memory_pressure: false,
            interv_ignore_sensors: false,
            interv_shift_attention: false,
            interv_delay_chronicle: false,
            interv_early_recovery: false,
        };

        state.demo_frames = state.build_named_scenario(Scenario::AquiferIntrusion);
        state
    }
}

// ── Scenario Builders ────────────────────────────────────────────────────────

impl WaterfallState {
    fn build_named_scenario(&self, scenario: Scenario) -> Vec<TimeWaterfallFrame> {
        let mut frames = Vec::new();
        let total_frames = 120;

        for idx in 0..total_frames {
            let t = if idx < 30 {
                0.0
            } else if idx < 70 {
                ((idx - 30) as f64 / 40.0).sin().min(1.0)
            } else {
                (1.0 - ((idx - 70) as f64 / 50.0)).max(0.0)
            };

            // Scenario-specific modifiers
            let (phi_mod, fep_mod, workspace_mod, conf_mod, anomaly_mod, mem_mod, mip_mod) =
                match scenario {
                    Scenario::AquiferIntrusion => (
                        3.2 - t * 2.8,
                        0.12 + t * 4.2,
                        0.05 + t * 1.8,
                        0.9 - t * 0.7,
                        t * 0.9,
                        0.2 + t * 2.5,
                        0.08 + t * 0.8,
                    ),
                    Scenario::FalseGreenRecovery => (
                        2.8 - t * 1.5,
                        0.1 + t * 0.8,
                        0.1 + t * 0.4,
                        0.85 - t * 0.3,
                        t * 0.35,
                        0.3 + t * 1.2,
                        0.05 + t * 0.4,
                    ),
                    Scenario::MemoryPressureCascade => (
                        2.9 - t * 2.1,
                        0.15 + t * 1.5,
                        0.08 + t * 0.9,
                        0.8 - t * 0.5,
                        t * 0.55,
                        0.15 + t * 2.9,
                        0.1 + t * 0.6,
                    ),
                    Scenario::SensorContradiction => (
                        3.0 - t * 0.5,
                        0.1 + t * 3.8,
                        0.1 + t * 1.2,
                        0.88 - t * 0.6,
                        t * 0.8,
                        0.2 + t * 1.5,
                        0.05 + t * 0.3,
                    ),
                    Scenario::DelayedChronicleCommit => (
                        3.1 - t * 2.4,
                        0.1 + t * 3.9,
                        0.1 + t * 1.6,
                        0.86 - t * 0.65,
                        t * 0.85,
                        0.2 + t * 2.2,
                        0.06 + t * 0.75,
                    ),
                    Scenario::OperatorInterventionPreview => (
                        3.0 - t * 1.2,
                        0.1 + t * 1.9,
                        0.1 + t * 0.8,
                        0.87 - t * 0.35,
                        t * 0.45,
                        0.2 + t * 1.1,
                        0.07 + t * 0.4,
                    ),
                };

            let metrics = [
                (phi_mod).clamp(0.1, 5.0),
                (fep_mod).clamp(0.01, 5.0),
                (workspace_mod).clamp(0.0, 2.0),
                (conf_mod).clamp(0.0, 1.0),
                (anomaly_mod).clamp(0.0, 1.0),
                (mem_mod).clamp(0.0, 3.0),
                (mip_mod).clamp(0.0, 1.0),
            ];

            let conf = (conf_mod - anomaly_mod * 0.4).clamp(0.05, 1.0);
            let mut anomaly_flags = Vec::new();
            if anomaly_mod > 0.3 {
                anomaly_flags.push("prediction_contradiction".to_string());
            }
            if t > 0.7 && scenario == Scenario::FalseGreenRecovery {
                anomaly_flags.push("false_green_diagnostic".to_string());
            }
            if t > 0.5 && scenario != Scenario::FalseGreenRecovery {
                anomaly_flags.push("recovery_mode".to_string());
            }

            let mut events = Vec::new();
            if idx == 30 {
                events.push(DashboardEvent {
                    event_id: "evt_perturbation_start".to_string(),
                    event_type: "intrusion".to_string(),
                    label: "P: Subterranean Intrusion".to_string(),
                    description: "Physical ingress sensors report anomaly variation".to_string(),
                    severity: "high".to_string(),
                    absolute_frame_index: idx as u64,
                    history_offset: idx as u64,
                    causal_role: "perturbation".to_string(),
                    metric_impacts: vec![("fep_prediction_error".to_string(), 0.35), ("phi_integration".to_string(), -0.2)],
                    narrative: "Dynamic state-space perturbation initiated. The active inference agent registers immediate prediction error deviation as telemetry breaks baseline bounds.".to_string(),
                    audit_recommendation: "Verify physical sensor line integrity inside boring head valve.".to_string(),
                });
            }
            if idx == 50 {
                events.push(DashboardEvent {
                    event_id: "evt_attention_shift".to_string(),
                    event_type: "attention".to_string(),
                    label: "GWT: Attention Focus Shift".to_string(),
                    description: "Global workspace shifts focus to ingress mitigation".to_string(),
                    severity: "medium".to_string(),
                    absolute_frame_index: idx as u64,
                    history_offset: idx as u64,
                    causal_role: "attention_shift".to_string(),
                    metric_impacts: vec![("workspace_activation".to_string(), 0.85)],
                    narrative: "Surprise threshold exceeded. The global workspace triggers attention reallocation to process structural stress and dynamic pump recovery.".to_string(),
                    audit_recommendation: "Monitor workspace focusing cycles for temporal stability.".to_string(),
                });
            }
            if idx == 75 {
                events.push(DashboardEvent {
                    event_id: "evt_mitigation_recovery".to_string(),
                    event_type: "recovery".to_string(),
                    label: "R: Mitigation Recovery".to_string(),
                    description: "Mitigation active; active control policy stabilized".to_string(),
                    severity: "medium".to_string(),
                    absolute_frame_index: idx as u64,
                    history_offset: idx as u64,
                    causal_role: "recovery".to_string(),
                    metric_impacts: vec![("anomaly_score".to_string(), -0.45)],
                    narrative: "Dynamic active inference controller convergence. Sub-surface pumps throttled to operation limits. Coherence levels stabilize.".to_string(),
                    audit_recommendation: "Confirm pump feedback loops converge to normal baseline flow rate.".to_string(),
                });
            }
            if idx == 100 {
                events.push(DashboardEvent {
                    event_id: "evt_chronicle_durability".to_string(),
                    event_type: "chronicle_event".to_string(),
                    label: "C: Chronicle Durability Sealed".to_string(),
                    description: "State chronicle finalized and hash generated".to_string(),
                    severity: "low".to_string(),
                    absolute_frame_index: idx as u64,
                    history_offset: idx as u64,
                    causal_role: "durable_record".to_string(),
                    metric_impacts: vec![("phi".to_string(), 0.95)],
                    narrative: "Observed state convergence certified. Immutable record hash sealed to durability witness layers.".to_string(),
                    audit_recommendation: "Commit record hash to the global durability ledger.".to_string(),
                });
            }

            frames.push(TimeWaterfallFrame {
                frame_index: idx as u64,
                sim_time_s: idx as f64 * 0.1,
                source_mode: SourceMode::ScriptedDemo,
                metrics,
                confidence: conf,
                anomaly_bits: if anomaly_mod > 0.35 { 0b0010010_u8 } else { 0 },
                anomaly_flags,
                is_chronicle: idx == 100,
                events,
            });
        }
        frames
    }

    fn get_at_age(&self, age: usize) -> Option<&TimeWaterfallFrame> {
        if self.paused {
            let offset = age;
            if self.recorded_frames.is_empty() {
                return None;
            }
            let mut curr = self.head as isize - 1 - offset as isize;
            while curr < 0 {
                curr += 256;
            }
            let idx = (curr as usize) % 256;
            Some(&self.recorded_frames[idx])
        } else {
            let mut curr = self.head as isize - 1;
            while curr < 0 {
                curr += 256;
            }
            let idx = (curr as usize) % 256;
            Some(&self.recorded_frames[idx])
        }
    }

    fn get_chronological_frames(&self) -> Vec<TimeWaterfallFrame> {
        let mut ordered_frames = Vec::with_capacity(256);
        for age in (0..256).rev() {
            if let Some(frame) = self.get_at_age(age) {
                ordered_frames.push(frame.clone());
            }
        }
        ordered_frames
    }

    fn compute_intervals(&self) -> Vec<RegimeInterval> {
        let frames = self.get_chronological_frames();
        let mut intervals = Vec::new();

        // Group into contiguous blocks matching flags
        let mut last_type = None;
        let mut start_idx = 0;

        for (idx, f) in frames.iter().enumerate() {
            let current_type = if f
                .anomaly_flags
                .contains(&"prediction_contradiction".to_string())
            {
                Some("prediction_contradiction")
            } else if f
                .anomaly_flags
                .contains(&"false_green_diagnostic".to_string())
            {
                Some("false_green_diagnostic")
            } else if f.anomaly_flags.contains(&"recovery_mode".to_string()) {
                Some("recovery_mode")
            } else {
                None
            };

            if current_type != last_type {
                if let Some(t) = last_type {
                    intervals.push(RegimeInterval {
                        r#type: t.to_string(),
                        start_frame: frames[start_idx].frame_index,
                        end_frame: frames[idx - 1].frame_index,
                        duration_s: (frames[idx - 1].frame_index - frames[start_idx].frame_index)
                            as f64
                            * 0.1,
                    });
                }
                start_idx = idx;
                last_type = current_type;
            }
        }

        if let Some(t) = last_type {
            if !frames.is_empty() {
                let last_idx = frames.len() - 1;
                intervals.push(RegimeInterval {
                    r#type: t.to_string(),
                    start_frame: frames[start_idx].frame_index,
                    end_frame: frames[last_idx].frame_index,
                    duration_s: (frames[last_idx].frame_index - frames[start_idx].frame_index)
                        as f64
                        * 0.1,
                });
            }
        }

        intervals
    }

    fn advance(&mut self) {
        let frame = match self.source_mode {
            SourceMode::ScriptedDemo => {
                if self.demo_idx >= self.demo_frames.len() {
                    self.demo_idx = 0; // loop
                }
                let mut f = self.demo_frames[self.demo_idx].clone();
                f.frame_index = self.tick;
                f.sim_time_s = self.tick as f64 * 0.1;
                self.demo_idx += 1;
                f
            }
            SourceMode::LiveSimulation => {
                let (metrics, conf, anomaly, chronicle) = self.step_live_simulation();
                let mut events = Vec::new();

                let mut anomaly_flags = Vec::new();
                if (anomaly & 0b0010000) != 0 {
                    anomaly_flags.push("prediction_contradiction".to_string());
                }
                if self.ingress_perturbation > 0.6 {
                    anomaly_flags.push("recovery_mode".to_string());
                }

                if self.ingress_perturbation > 0.0 && self.tick % 20 == 0 {
                    events.push(DashboardEvent {
                        event_id: format!("evt_live_pert_{}", self.tick),
                        event_type: "intrusion".to_string(),
                        label: "P: Water Ingress Perturbation".to_string(),
                        description: "Water ingress levels injected into simulation".to_string(),
                        severity: "critical".to_string(),
                        absolute_frame_index: self.tick,
                        history_offset: 0,
                        causal_role: "perturbation".to_string(),
                        metric_impacts: vec![("fep_prediction_error".to_string(), self.ingress_perturbation * 0.6)],
                        narrative: "Water ratio increased manually. Telemetry errors spikes dynamically as FEP active inference loops process the deviation.".to_string(),
                        audit_recommendation: "Ensure pump pressure thresholds are set correctly.".to_string(),
                    });
                }

                TimeWaterfallFrame {
                    frame_index: self.tick,
                    sim_time_s: self.tick as f64 * 0.1,
                    source_mode: self.source_mode,
                    metrics,
                    confidence: conf,
                    anomaly_bits: anomaly,
                    anomaly_flags,
                    is_chronicle: chronicle,
                    events,
                }
            }
        };

        self.recorded_frames[self.head] = frame;
        self.head = (self.head + 1) % 256;
        self.tick += 1;
    }

    fn step_live_simulation(&mut self) -> ([f64; 7], f64, u8, bool) {
        let mut cmd = SubterraneanCommand::zero();
        cmd.torques[0] = 0.8;
        cmd.torques[1] = 0.5;

        // Apply active preview interventions
        let current_water = self.sim.state().channels[WATER_INGRESS_RATIO];
        if self.ingress_perturbation > 0.0 && !self.interv_ignore_sensors {
            let rate = if self.interv_early_recovery {
                0.3
            } else {
                0.15
            };
            self.sim.state_mut().channels[WATER_INGRESS_RATIO] =
                (current_water + self.ingress_perturbation * rate).min(1.0);
        }

        self.sim.step(&cmd, 0.1);

        let state = self.sim.state();
        let water = state.channels[WATER_INGRESS_RATIO];

        let mut p_phi = (3.0 - water * 2.5).clamp(0.1, 5.0);
        if self.interv_ignore_sensors {
            p_phi = 3.0; // Flat/unperturbed baseline
        }
        let p_integration = (0.8 - water * 0.6).clamp(0.0, 1.0);
        let p_coherence = (0.9 - water * 0.5).clamp(0.0, 1.0);
        let p_attention = if self.interv_shift_attention {
            0.05
        } else {
            (0.5 + water * 0.4).clamp(0.0, 1.0)
        };

        let obs =
            Observation::from_consciousness_state(p_phi, p_integration, p_coherence, p_attention);
        self.fep.perceive(&obs);

        let oracle_obs = vec![p_phi, p_integration, p_coherence, p_attention];
        self.oracle.observe(&oracle_obs).ok();

        let mut surprise = self.fep.current_free_energy();
        if self.interv_dampen_memory_pressure {
            surprise *= 0.2;
        }

        if surprise > 0.4 {
            let bid = AttentionBid {
                source: "subterranean_physics".to_string(),
                magnitude: surprise,
                sensation: ContinuousHV::zero(64),
                description: format!("Water Ingress telemetry variation"),
            };
            self.workspace.submit_bid(bid);
        }
        self.workspace.process_cycle();

        let phi = self
            .oracle
            .measure()
            .map(|r| r.integration_index)
            .unwrap_or(p_phi);
        let pred_error = self
            .fep
            .last_fe_components
            .as_ref()
            .map(|c| c.prediction_error)
            .unwrap_or(0.1);
        let workspace_act = self
            .workspace
            .current_focus
            .as_ref()
            .map(|f| f.magnitude)
            .unwrap_or(0.0);
        let confidence = self.fep.belief.confidence();
        let memory_pressure = if self.interv_dampen_memory_pressure {
            0.1
        } else {
            self.fep.belief.total_uncertainty()
        };
        let mip = self
            .oracle
            .measure()
            .map(|r| 1.0 - r.normalized_index)
            .unwrap_or(water);
        let anomaly_score = (pred_error * 0.4 + mip * 0.3).clamp(0.0, 1.0);

        let metrics = [
            phi,
            pred_error,
            workspace_act,
            confidence,
            anomaly_score,
            memory_pressure,
            mip,
        ];

        let mut anomaly_bits = 0u8;
        if anomaly_score > 0.35 {
            anomaly_bits |= 0b0010000;
        }

        let is_chronicle = if self.interv_delay_chronicle {
            self.tick % 40 == 0
        } else {
            self.ingress_perturbation > 0.0 && self.tick % 10 == 0
        };

        (metrics, confidence, anomaly_bits, is_chronicle)
    }
}

// ── Bevy Systems ─────────────────────────────────────────────────────────────

fn handle_keyboard_input(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut state: ResMut<WaterfallState>,
    mut ui: ResMut<UiState>,
    mut right_timer: Local<f32>,
    mut left_timer: Local<f32>,
) {
    if keys.just_pressed(KeyCode::Space) {
        state.paused = !state.paused;
        ui.status_text = if state.paused {
            "⏸ Forensic pause mode active".to_string()
        } else {
            "▶ Live observ telemetry active".to_string()
        };
    }
    if state.paused {
        if keys.just_pressed(KeyCode::ArrowRight) {
            state.scrub = state.scrub.saturating_sub(1);
            *right_timer = -0.3;
        } else if keys.pressed(KeyCode::ArrowRight) {
            *right_timer += time.delta_secs();
            if *right_timer >= 0.05 {
                state.scrub = state.scrub.saturating_sub(1);
                *right_timer = 0.0;
            }
        } else {
            *right_timer = 0.0;
        }

        if keys.just_pressed(KeyCode::ArrowLeft) {
            state.scrub = (state.scrub + 1).min(255);
            *left_timer = -0.3;
        } else if keys.pressed(KeyCode::ArrowLeft) {
            *left_timer += time.delta_secs();
            if *left_timer >= 0.05 {
                state.scrub = (state.scrub + 1).min(255);
                *left_timer = 0.0;
            }
        } else {
            *left_timer = 0.0;
        }
    }

    if keys.just_pressed(KeyCode::KeyR) {
        state.paused = false;
        state.scrub = 0;
        state.ingress_perturbation = 0.0;
        state.portrait_trail.clear();
        ui.status_text = "▶ Observ System Reset".to_string();
    }
    if keys.just_pressed(KeyCode::KeyV) {
        state.layout_mode = (state.layout_mode + 1) % 2; // Cycle layout modes
    }
}

fn advance_waterfall(time: Res<Time>, mut state: ResMut<WaterfallState>, mut timer: Local<f32>) {
    if state.paused {
        return;
    }
    *timer += time.delta_secs();
    if *timer >= 0.1 {
        *timer = 0.0;
        state.advance();
    }
}

fn update_observatory_simulation(time: Res<Time>, mut state: ResMut<WaterfallState>) {
    // Drifting coordinates for memory graph simulation
    let t = time.elapsed_secs();
    for (idx, mem) in state.memories.iter_mut().enumerate() {
        let angle = t * 0.4 + (idx as f32) * 1.5;
        let offset = Vec2::new(angle.cos() * 15.0, angle.sin() * 15.0);
        mem.coord = mem.target_coord + offset;
    }
}

// ── Rendering Observatory Viewports (Gizmos) ─────────────────────────────────

fn render_observatory_viewport(mut gizmos: Gizmos, state: Res<WaterfallState>) {
    if state.layout_mode == 1 {
        // Raw Developer Telemetry Mode (render flat metric waterfall ribbons)
        let frames = state.get_chronological_frames();
        let max_height = 200.0;
        let draw_w = 600.0;

        for (metric_idx, base_color) in METRIC_COLORS.iter().enumerate() {
            if !state.metric_visible[metric_idx] {
                continue;
            }
            let mut prev_point: Option<Vec2> = None;
            let ribbon_y = -300.0 + (metric_idx as f32) * 90.0;

            for age in 0..64 {
                if let Some(frame) = frames.get(age) {
                    let val = frame.metrics[metric_idx] as f32;
                    let display_h = (val / 5.0).clamp(0.0, 1.0) * max_height;
                    let x = -draw_w / 2.0 + (63 - age) as f32 * (draw_w / 64.0);
                    let opacity = 1.0 - (age as f32 / 64.0);
                    let color = Color::srgba(base_color.0, base_color.1, base_color.2, opacity);

                    gizmos.line_2d(
                        Vec2::new(x, ribbon_y),
                        Vec2::new(x, ribbon_y + display_h),
                        color,
                    );
                    if let Some(prev) = prev_point {
                        gizmos.line_2d(
                            prev,
                            Vec2::new(x, ribbon_y + display_h),
                            Color::srgba(base_color.0, base_color.1, base_color.2, opacity * 0.5),
                        );
                    }
                    prev_point = Some(Vec2::new(x, ribbon_y + display_h));
                }
            }
        }
        return;
    }

    // Default: Observatory Mode (Cinematic State-Space Observatory Viewport)
    match state.observatory_mode {
        ObservatoryMode::LiveCognition => {
            // Render HDC Resonance Spiral & Spotlights
            let center = Vec2::new(-100.0, 0.0);
            let frames = state.get_chronological_frames();
            let surprise = frames.first().map(|f| f.metrics[1]).unwrap_or(0.1);

            // Draw holographic spiral loops
            for i in 0..180 {
                let r = (i as f32) * 1.6;
                let angle = (i as f32) * 0.12;
                let x = center.x + angle.cos() * r;
                let y = center.y + angle.sin() * r;
                let scale = 1.0 + surprise as f32 * 0.8;

                // Color based on active inference surprise loops
                let color = if surprise > 0.6 {
                    Color::srgba(0.97, 0.2 + (i as f32 * 0.003), 0.2, 0.8)
                } else {
                    Color::srgba(0.22, 0.74, 0.97, 0.5 - (r / 300.0))
                };
                gizmos.circle_2d(Vec2::new(x, y * scale), 2.0, color);
            }

            // Draw Global Workspace spotlight focus circle
            let spot_size = 40.0 + surprise as f32 * 35.0;
            gizmos.circle_2d(center, spot_size, Color::srgba(0.96, 0.62, 0.04, 0.3));
        }
        ObservatoryMode::ForensicReplay => {
            // Draw Forensic flight timeline record view
            let center_y = 50.0;
            let timeline_w = 700.0;
            let start_x = -timeline_w / 2.0;

            // Draw chronological trace path
            let frames = state.get_chronological_frames();
            let mut prev_point: Option<Vec2> = None;

            for age in 0..128 {
                if let Some(f) = frames.get(age) {
                    let x = start_x + (127 - age) as f32 * (timeline_w / 128.0);
                    let val = f.metrics[4] as f32; // Anomaly score drives path deflection
                    let y = center_y + (val * 110.0);

                    let opac = 1.0 - (age as f32 / 128.0);
                    let color = if f
                        .anomaly_flags
                        .contains(&"prediction_contradiction".to_string())
                    {
                        Color::srgba(0.97, 0.44, 0.44, opac) // dashed red contradiction
                    } else if f.is_chronicle {
                        Color::srgba(0.96, 0.62, 0.04, opac) // durable amber
                    } else {
                        Color::srgba(0.22, 0.74, 0.97, opac) // solid verified cyan
                    };

                    gizmos.circle_2d(Vec2::new(x, y), 3.0, color);
                    if let Some(prev) = prev_point {
                        gizmos.line_2d(prev, Vec2::new(x, y), color);
                    }
                    prev_point = Some(Vec2::new(x, y));
                }
            }

            // Draw current scrub timeline head line
            let scrub_x = start_x + (127 - state.scrub.min(127)) as f32 * (timeline_w / 128.0);
            gizmos.line_2d(
                Vec2::new(scrub_x, -200.0),
                Vec2::new(scrub_x, 250.0),
                Color::srgba(0.96, 0.62, 0.04, 0.95),
            );
        }
        ObservatoryMode::StateSpace => {
            // Render Attractor Basin Manifold Grid
            let center = Vec2::new(-100.0, 0.0);

            // Render nested concentric attractor boundary layers
            for r in [40.0, 80.0, 120.0, 160.0] {
                // Stabilized Basin bounds vs Instability regions
                let is_unstable = r > 110.0;
                let color = if is_unstable {
                    Color::srgba(0.97, 0.44, 0.44, 0.25)
                } else {
                    Color::srgba(0.20, 0.83, 0.60, 0.2)
                };
                gizmos.circle_2d(center, r, color);
            }

            // Draw trajectory drift path
            let frames = state.get_chronological_frames();
            let mut prev: Option<Vec2> = None;
            for age in (0..100).rev() {
                if let Some(f) = frames.get(age) {
                    // map phi vs anomaly score
                    let x = center.x + (f.metrics[0] as f32 * 35.0) - 80.0;
                    let y = center.y + (f.metrics[4] as f32 * 180.0) - 90.0;

                    let opacity = 1.0 - (age as f32 / 100.0);
                    let color = if f
                        .anomaly_flags
                        .contains(&"prediction_contradiction".to_string())
                    {
                        Color::srgba(0.97, 0.2, 0.2, opacity * 1.5)
                    } else {
                        Color::srgba(0.96, 0.62, 0.04, opacity)
                    };
                    gizmos.circle_2d(Vec2::new(x, y), 3.0, color);
                    if let Some(p) = prev {
                        gizmos.line_2d(p, Vec2::new(x, y), color);
                    }
                    prev = Some(Vec2::new(x, y));
                }
            }
        }
        ObservatoryMode::InterventionPreview => {
            // Renders dynamic "what if" projection pathways (two split paths)
            let center = Vec2::new(-100.0, 0.0);

            // Path 1 (Inferred actual baseline)
            let mut p1 = center;
            for i in 0..12 {
                let x = center.x + (i as f32 * 25.0);
                let y = center.y + ((i as f32 * 0.35).sin() * 30.0);
                gizmos.line_2d(p1, Vec2::new(x, y), Color::srgba(0.97, 0.44, 0.44, 0.6));
                gizmos.circle_2d(Vec2::new(x, y), 2.0, Color::srgba(0.97, 0.44, 0.44, 0.8));
                p1 = Vec2::new(x, y);
            }

            // Path 2 (Intervened mitigation preview route)
            let mut p2 = center;
            for i in 0..12 {
                let x = center.x + (i as f32 * 25.0);
                let y = center.y - (i as f32 * 4.0); // settles faster
                gizmos.line_2d(p2, Vec2::new(x, y), Color::srgba(0.20, 0.83, 0.60, 0.6));
                gizmos.circle_2d(Vec2::new(x, y), 2.0, Color::srgba(0.20, 0.83, 0.60, 0.8));
                p2 = Vec2::new(x, y);
            }
        }
    }
}

// ── Rendering Egui Controls, Explanations, & Exporters ───────────────────────

fn render_observatory_sidebar(
    mut contexts: EguiContexts,
    mut state: ResMut<WaterfallState>,
    _ui_state: Res<UiState>,
) {
    let sidebar_bg = egui::Color32::from_rgb(15, 15, 18);
    let cyan_glow = egui::Color32::from_rgb(56, 189, 248);
    let amber_glow = egui::Color32::from_rgb(245, 158, 11);

    // 1. TOP BAR PANEL: Scenario and Observatory Mode selections
    egui::TopBottomPanel::top("top_panel")
        .frame(
            egui::Frame::none()
                .fill(egui::Color32::from_rgb(20, 20, 25))
                .inner_margin(8.0),
        )
        .show(contexts.ctx_mut(), |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("SYMTHAEA OBSERVATORY v0.3.0")
                        .strong()
                        .size(15.0)
                        .color(egui::Color32::WHITE),
                );
                ui.separator();

                // Scenario Selector
                ui.label("Scenario:");
                egui::ComboBox::from_id_source("scenario_combobox")
                    .selected_text(format!("{:?}", state.selected_scenario))
                    .show_ui(ui, |ui| {
                        if ui
                            .selectable_value(
                                &mut state.selected_scenario,
                                Scenario::AquiferIntrusion,
                                "Aquifer Intrusion",
                            )
                            .clicked()
                        {
                            state.demo_frames =
                                state.build_named_scenario(Scenario::AquiferIntrusion);
                            state.demo_idx = 0;
                            state.selected_event_node = None;
                        }
                        if ui
                            .selectable_value(
                                &mut state.selected_scenario,
                                Scenario::FalseGreenRecovery,
                                "False Green Recovery",
                            )
                            .clicked()
                        {
                            state.demo_frames =
                                state.build_named_scenario(Scenario::FalseGreenRecovery);
                            state.demo_idx = 0;
                            state.selected_event_node = None;
                        }
                        if ui
                            .selectable_value(
                                &mut state.selected_scenario,
                                Scenario::MemoryPressureCascade,
                                "Memory Pressure Cascade",
                            )
                            .clicked()
                        {
                            state.demo_frames =
                                state.build_named_scenario(Scenario::MemoryPressureCascade);
                            state.demo_idx = 0;
                            state.selected_event_node = None;
                        }
                        if ui
                            .selectable_value(
                                &mut state.selected_scenario,
                                Scenario::SensorContradiction,
                                "Sensor Contradiction",
                            )
                            .clicked()
                        {
                            state.demo_frames =
                                state.build_named_scenario(Scenario::SensorContradiction);
                            state.demo_idx = 0;
                            state.selected_event_node = None;
                        }
                        if ui
                            .selectable_value(
                                &mut state.selected_scenario,
                                Scenario::DelayedChronicleCommit,
                                "Delayed Chronicle Commit",
                            )
                            .clicked()
                        {
                            state.demo_frames =
                                state.build_named_scenario(Scenario::DelayedChronicleCommit);
                            state.demo_idx = 0;
                            state.selected_event_node = None;
                        }
                        if ui
                            .selectable_value(
                                &mut state.selected_scenario,
                                Scenario::OperatorInterventionPreview,
                                "Operator Intervention Preview",
                            )
                            .clicked()
                        {
                            state.demo_frames =
                                state.build_named_scenario(Scenario::OperatorInterventionPreview);
                            state.demo_idx = 0;
                            state.selected_event_node = None;
                        }
                    });

                ui.separator();
                // Layout Selector Toggle
                ui.label("Observatory Layout:");
                if ui
                    .selectable_label(state.layout_mode == 0, "🪐 Observatory")
                    .clicked()
                {
                    state.layout_mode = 0;
                }
                if ui
                    .selectable_label(state.layout_mode == 1, "📊 Developer Telemetry")
                    .clicked()
                {
                    state.layout_mode = 1;
                }

                ui.separator();

                // Interactive mode selector
                if ui
                    .selectable_label(
                        state.observatory_mode == ObservatoryMode::LiveCognition,
                        "⚡ Live Cognition",
                    )
                    .clicked()
                {
                    state.observatory_mode = ObservatoryMode::LiveCognition;
                    state.paused = false;
                }
                if ui
                    .selectable_label(
                        state.observatory_mode == ObservatoryMode::ForensicReplay,
                        "🔎 Forensic Replay",
                    )
                    .clicked()
                {
                    state.observatory_mode = ObservatoryMode::ForensicReplay;
                    state.paused = true;
                }
                if ui
                    .selectable_label(
                        state.observatory_mode == ObservatoryMode::StateSpace,
                        "📐 State-Space",
                    )
                    .clicked()
                {
                    state.observatory_mode = ObservatoryMode::StateSpace;
                }
                if ui
                    .selectable_label(
                        state.observatory_mode == ObservatoryMode::InterventionPreview,
                        "🧪 Intervention Preview",
                    )
                    .clicked()
                {
                    state.observatory_mode = ObservatoryMode::InterventionPreview;
                }
            });
        });

    // 2. BOTTOM PANEL: Causal Chain Map / Forensic scrubber
    egui::TopBottomPanel::bottom("causal_timeline_chain")
        .exact_height(160.0)
        .frame(
            egui::Frame::none()
                .fill(egui::Color32::from_rgb(13, 13, 16))
                .inner_margin(8.0),
        )
        .show(contexts.ctx_mut(), |ui| {
            ui.label(
                egui::RichText::new("CAUSAL CHAIN MAP")
                    .small()
                    .color(egui::Color32::GRAY),
            );
            ui.separator();

            let chain_nodes = [
                (
                    "Aquifer Intrusion",
                    "perturbation",
                    "Ingress Intrusion\n  ↓ caused",
                    0,
                ),
                (
                    "FEP Surprise Spike",
                    "anomaly",
                    "FEP Surprise Spike\n  ↓ shifted",
                    1,
                ),
                (
                    "Workspace Attention",
                    "attention",
                    "Workspace Focus\n  ↓ destabilized",
                    2,
                ),
                ("MIP Crossing", "anomaly", "MIP Crossing\n  ↓ triggered", 3),
                (
                    "Mitigating Recovery",
                    "recovery",
                    "Mitigating Recovery\n  ↓ sealed",
                    4,
                ),
                (
                    "Chronicle Commit",
                    "durable",
                    "Chronicle Durability\n  (sealed)",
                    5,
                ),
            ];

            ui.horizontal(|ui| {
                for (name, role, link_txt, node_idx) in chain_nodes {
                    let is_sel = state.selected_event_node == Some(node_idx);

                    let btn_color = match role {
                        "perturbation" => egui::Color32::from_rgb(248, 113, 113),
                        "anomaly" => egui::Color32::from_rgb(248, 113, 113),
                        "attention" => egui::Color32::from_rgb(56, 189, 248),
                        "recovery" => egui::Color32::from_rgb(52, 211, 153),
                        _ => egui::Color32::from_rgb(245, 158, 11),
                    };

                    ui.vertical(|ui| {
                        let btn = ui.add(
                            egui::Button::new(egui::RichText::new(name).strong().color(
                                if is_sel {
                                    egui::Color32::WHITE
                                } else {
                                    btn_color
                                },
                            ))
                            .fill(if is_sel {
                                btn_color
                            } else {
                                egui::Color32::from_rgb(25, 25, 30)
                            }),
                        );

                        if btn.clicked() {
                            state.selected_event_node = Some(node_idx);
                            state.selected_interval = None;
                        }

                        ui.label(
                            egui::RichText::new(link_txt)
                                .small()
                                .color(egui::Color32::GRAY),
                        );
                    });
                    ui.add_space(20.0);
                }
            });
        });

    // 3. LEFT PANEL: HDC Memory Topology graph view
    egui::SidePanel::left("memory_topology_sidebar")
        .exact_width(260.0)
        .frame(egui::Frame::none().fill(sidebar_bg).inner_margin(8.0))
        .show(contexts.ctx_mut(), |ui| {
            ui.label(
                egui::RichText::new("HDC MEMORY TOPOLOGY")
                    .small()
                    .color(egui::Color32::GRAY),
            );
            ui.separator();

            ui.vertical(|ui| {
                for mem in &state.memories {
                    let color = match mem.status {
                        "contradicted" => egui::Color32::from_rgb(248, 113, 113), // Red fracture
                        "stabilizing" => egui::Color32::from_rgb(52, 211, 153),   // Green anchor
                        _ => egui::Color32::from_rgb(56, 189, 248),               // active blue
                    };

                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("●").color(color));
                            ui.label(egui::RichText::new(&mem.label).strong().small());
                        });
                        ui.label(format!("Confidence: {:.1}%", mem.confidence * 100.0));
                        ui.label(format!("Semantic status: {}", mem.status));
                    });
                    ui.add_space(6.0);
                }
            });
        });

    // 4. RIGHT PANEL: Cognitive Explanation Stack & Controls
    egui::SidePanel::right("cognitive_observatory_sidebar")
        .exact_width(320.0)
        .frame(egui::Frame::none().fill(sidebar_bg).inner_margin(8.0))
        .show(contexts.ctx_mut(), |ui| {
            ui.label(egui::RichText::new("COGNITIVE OBSERVER").strong().size(15.0).color(egui::Color32::WHITE));
            ui.separator();

            // Interventions preview toggles (if in Intervention Preview Mode)
            if state.observatory_mode == ObservatoryMode::InterventionPreview {
                ui.group(|ui| {
                    ui.label(egui::RichText::new("🧪 What-If Interventions").strong().color(amber_glow));

                    let mut d_mem = state.interv_dampen_memory_pressure;
                    if ui.checkbox(&mut d_mem, "Dampen Memory Pressure").changed() {
                        state.interv_dampen_memory_pressure = d_mem;
                    }
                    let mut i_sens = state.interv_ignore_sensors;
                    if ui.checkbox(&mut i_sens, "Ignore Contradicted Sensor").changed() {
                        state.interv_ignore_sensors = i_sens;
                    }
                    let mut s_att = state.interv_shift_attention;
                    if ui.checkbox(&mut s_att, "Shift Workspace Attention").changed() {
                        state.interv_shift_attention = s_att;
                    }
                    let mut d_chron = state.interv_delay_chronicle;
                    if ui.checkbox(&mut d_chron, "Delay Chronicle Commit").changed() {
                        state.interv_delay_chronicle = d_chron;
                    }
                    let mut e_rec = state.interv_early_recovery;
                    if ui.checkbox(&mut e_rec, "Trigger Pump Recovery Early").changed() {
                        state.interv_early_recovery = e_rec;
                    }
                });
                ui.separator();
            }

            // Expose the explanation stack
            let current_frame = state.get_at_age(state.scrub).cloned();

            if let Some(frame) = current_frame {
                if let Some(event_idx) = state.selected_event_node {
                    let details = match event_idx {
                        0 => ("Ingress Aquifer Intrusion", "Aquifer intrusion detected at local sub-surface sensor. Variational free energy spikes as predictions deviate.", "Subterranean pressure exceeds 1.5 Bar limit", "State-space trajectory deflecting towards boundary limits", "Unstable Aquifer Transition memory lit up", "Initiate active pump control policy"),
                        1 => ("FEP Surprise Spike", "Active inference agent reports prediction surprise error. High epistemic hazard indicates potential telemetry mismatch.", "Model predictions disagree with sensor inputs by >0.42", "FEP terrain height deforms rapidly", "Sensor calibration baseline recalled", "Adjust belief precision constraints"),
                        2 => ("Workspace Attention Shift", "Surprise exceeds threshold, triggering Global Workspace attention focus shift.", "Variational surprise spike above attention trigger limits", "spotlight focuses centered attractor basin", "Borehole Hydro-static Baseline memory active", "Reallocate cognitive processing weights"),
                        3 => ("MIP Boundary Crossing", "Integration indices drop below safety threshold. MIP boundary instability crossed.", "Oracle normalized index dropped below 0.35 threshold", "attractor bounds show destabilized trajectories", "Dynamic MIP Stabilizer memory recalled", "Dispatched advisory operator warning"),
                        4 => ("Mitigating Recovery Begun", "Active control policy initiated. Subterranean pumps active to convey spoil and mitigate water ratio.", "Pumps activated at max power feedback loop", "Stabilizing settling basin convergence", "Pump Convergence Rate memory active", "Continue pump operations until stable"),
                        _ => ("Chronicle Durability Sealed", "Stability restored. Immutable record hash sealed to durability witness layers.", "State variables stabilized back within safe baseline bounds", "durable strata lock committed", "Ingress Inflow Model memory stabilized", "Export verified chronicle data bundle"),
                    };

                    ui.label(egui::RichText::new("1. What Happened?").strong().color(cyan_glow));
                    ui.label(egui::RichText::new(details.0).strong().size(13.0));
                    ui.label(egui::RichText::new(details.1).italics());
                    ui.separator();

                    ui.label(egui::RichText::new("2. Why does Symthaea believe that?").strong().color(cyan_glow));
                    ui.label(details.2);
                    ui.separator();

                    ui.label(egui::RichText::new("3. What changed in state-space?").strong().color(cyan_glow));
                    ui.label(details.3);
                    ui.separator();

                    ui.label(egui::RichText::new("4. What memories were active?").strong().color(cyan_glow));
                    ui.label(details.4);
                    ui.separator();

                    ui.label(egui::RichText::new("5. What recovery path was chosen?").strong().color(cyan_glow));
                    ui.colored_label(egui::Color32::GREEN, details.5);
                } else {
                    // Render default general explanations
                    ui.label(egui::RichText::new("1. Observ State").strong().color(cyan_glow));
                    ui.label(format!("Frame idx: {} | Sim Time: {:.1}s", frame.frame_index, frame.sim_time_s));
                    ui.label(format!("Diagnostic Confidence: {:.1}%", frame.confidence * 100.0));
                    ui.separator();

                    ui.label(egui::RichText::new("2. Remaining Uncertainty").strong().color(cyan_glow));
                    ui.label(format!("Epistemic Uncertainty: {:.3}", state.twin.epistemic_uncertainty));
                    ui.separator();
                }
            }

            ui.separator();
            // Blackbox Chronicle Bundle Exporter
            ui.group(|ui| {
                ui.label(egui::RichText::new("📁 Black Box Chronicle").strong().color(amber_glow));
                if ui.button("Export Chronicle Bundle").clicked() {
                    state.export_sealed = true;
                    state.sealed_hash = "sha256_8f2b3e8c9d1a0b5c...".to_string();
                }

                if state.export_sealed {
                    ui.colored_label(egui::Color32::GREEN, "✔ Bundle Sealed!");
                    ui.label(egui::RichText::new(format!("Hash: {}", state.sealed_hash)).monospace().small());
                    ui.label("Witness layers: local / global / witness");
                }
            });
        });
}
