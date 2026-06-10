// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
    /// Eight Harmonies scores [RC, PSF, IW, IP, UI, SR, EP, SS]
    pub harmony_scores: RwSignal<[f32; 8]>,
    /// Dominant harmony name
    pub dominant_harmony: RwSignal<String>,
    /// Factcheck bridge metrics (Broca→Mycelix knowledge verification)
    pub factcheck_accuracy: RwSignal<f32>,
    pub factcheck_verified: RwSignal<u32>,
    pub factcheck_suppressed: RwSignal<u32>,
    pub factcheck_pending: RwSignal<u32>,
    /// QOL trend data (rolling buffers fed by broadcast handler)
    pub trend_consciousness: RwSignal<Vec<f32>>,
    pub trend_harmony: RwSignal<Vec<f32>>,
    /// Active wellbeing profile
    pub wellbeing_profile: RwSignal<String>,
    /// Broca pipeline download progress: "idle" | "downloading 45%" | "verifying" | "loading" | "ready"
    pub pipeline_status: RwSignal<String>,
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
            harmony_scores: RwSignal::new([0.5; 8]),
            dominant_harmony: RwSignal::new("Resonant Coherence".to_string()),
            factcheck_accuracy: RwSignal::new(0.95),
            factcheck_verified: RwSignal::new(0),
            factcheck_suppressed: RwSignal::new(0),
            factcheck_pending: RwSignal::new(0),
            trend_consciousness: RwSignal::new(Vec::new()),
            trend_harmony: RwSignal::new(Vec::new()),
            wellbeing_profile: RwSignal::new("Default".to_string()),
            pipeline_status: RwSignal::new("idle".to_string()),
        }
    }
}
