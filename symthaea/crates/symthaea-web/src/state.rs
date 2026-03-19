use leptos::prelude::*;

#[derive(Clone)]
pub struct AppState {
    pub consciousness_level: RwSignal<f32>,
    pub prediction_error: RwSignal<f32>,
    pub harmony_alignment: RwSignal<f32>,
    /// DA, NE, 5-HT, OT
    pub neuromods: RwSignal<[f32; 4]>,
    pub cycle_count: RwSignal<u64>,
    pub epistemic_level: RwSignal<String>,
    pub honest_confidence: RwSignal<f32>,
    pub worker_ready: RwSignal<bool>,
    pub pipeline_loaded: RwSignal<bool>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            consciousness_level: RwSignal::new(0.0),
            prediction_error: RwSignal::new(0.0),
            harmony_alignment: RwSignal::new(0.0),
            neuromods: RwSignal::new([0.5, 0.5, 0.5, 0.5]),
            cycle_count: RwSignal::new(0),
            epistemic_level: RwSignal::new("theoretical".to_string()),
            honest_confidence: RwSignal::new(0.10),
            worker_ready: RwSignal::new(false),
            pipeline_loaded: RwSignal::new(false),
        }
    }
}
