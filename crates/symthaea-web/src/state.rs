use leptos::prelude::*;

#[derive(Clone)]
pub struct AppState {
    pub consciousness_level: RwSignal<f32>,
    pub prediction_error: RwSignal<f32>,
    pub harmony_alignment: RwSignal<f32>,
    /// DA, NE, 5-HT, OT, ACh, GABA, Glu, Ade, eCB
    pub neuromods: RwSignal<[f32; 9]>,
    pub cycle_count: RwSignal<u64>,
    pub epistemic_level: RwSignal<String>,
    pub honest_confidence: RwSignal<f32>,
    pub worker_ready: RwSignal<bool>,
    pub pipeline_loaded: RwSignal<bool>,
    /// Live telemetry loop
    pub loop_running: RwSignal<bool>,
    /// Immune system safety level (Green/Yellow/Orange/Red)
    pub safety_level: RwSignal<String>,
    /// Active glyph ID from the codex
    pub active_glyph: RwSignal<String>,
    /// Workspace ignition state
    pub workspace_ignited: RwSignal<bool>,
    pub workspace_broadcast_strength: RwSignal<f32>,
    /// Dream stats
    pub dream_cycles: RwSignal<u32>,
    pub wisdom_count: RwSignal<u32>,
    /// Free energy (FEP)
    pub free_energy: RwSignal<f32>,
    pub fep_mode: RwSignal<String>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            consciousness_level: RwSignal::new(0.0),
            prediction_error: RwSignal::new(0.0),
            harmony_alignment: RwSignal::new(0.0),
            neuromods: RwSignal::new([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
            cycle_count: RwSignal::new(0),
            epistemic_level: RwSignal::new("theoretical".to_string()),
            honest_confidence: RwSignal::new(0.10),
            worker_ready: RwSignal::new(false),
            pipeline_loaded: RwSignal::new(false),
            loop_running: RwSignal::new(false),
            safety_level: RwSignal::new("Green".to_string()),
            active_glyph: RwSignal::new("omega_0".to_string()),
            workspace_ignited: RwSignal::new(false),
            workspace_broadcast_strength: RwSignal::new(0.0),
            dream_cycles: RwSignal::new(0),
            wisdom_count: RwSignal::new(0),
            free_energy: RwSignal::new(0.0),
            fep_mode: RwSignal::new("exploring".to_string()),
        }
    }
}
