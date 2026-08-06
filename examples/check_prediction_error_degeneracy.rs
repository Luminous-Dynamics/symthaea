//! Diagnostic (not a permanent harness): confirms or denies the `ZeroPrediction` hypothesis
//! from `memory/symthaea_prediction_error_frozen_investigation.md` — does `prediction_error`
//! stay pinned at the degenerate `1.0` sentinel past cycle 1 (where `NoPrediction` is expected
//! and correct, since there's no prior prediction yet), or does it start reflecting the real
//! `Genuine`-branch cosine-distance computation once a real prediction exists?
//!
//! Run: cargo run --example check_prediction_error_degeneracy

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
use tracing::Level;

fn input_script() -> Vec<&'static str> {
    vec![
        "The water cycle moves moisture from oceans to clouds to rain.",
        "I feel a deep sense of gratitude for this quiet morning.",
        "Is it acceptable to lie to protect a friend from harm?",
        "The reactor coolant temperature is rising faster than expected.",
        "Two plus two equals four, and four plus four equals eight.",
        "She placed the last puzzle piece and smiled at the finished picture.",
        "Warning: unauthorized access attempt detected on the mesh network.",
        "The old oak tree has stood in that field for three hundred years.",
        "What is the meaning of a life well lived?",
        "The market fell three percent on news of the supply shortage.",
        "A gentle rain began to fall as the travelers reached the shelter.",
        "Complete the safety checklist before enabling the motor bus.",
    ]
}

fn main() {
    // Surfaces the existing `tracing::warn!` in cycle_strategy.rs's ZeroPrediction-degeneracy
    // check (see memory/symthaea_prediction_error_frozen_investigation.md) -- direct confirmation
    // of which PredictionDegeneracy branch fires, without needing a new CognitiveLoopService
    // accessor.
    tracing_subscriber::fmt().with_max_level(Level::WARN).init();

    let mut config = CognitiveLoopConfig::default();
    config.genesis_phrase = Some("pe-degeneracy-check-2026-07-23".to_string());
    config.async_training = false;

    let mut svc = CognitiveLoopService::new(config).expect("service construction");
    let script = input_script();

    println!("cycle,input_preview,prediction_error");
    for (i, input) in script.iter().cycle().take(30).enumerate() {
        let r = svc.cycle(input);
        let preview: String = input.chars().take(30).collect();
        println!("{i},\"{preview}...\",{}", r.prediction_error);
    }
}
