// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-Architecture Support Status
//!
//! Shows which models are supported and validated for phenomenal detection.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example cross_architecture_status --features neural-bridge --release
//! ```

use symthaea::perception::multi_model_extractor::{ModelPreset, print_support_summary};

fn main() {
    print_support_summary();

    println!("════════════════════════════════════════════════════════════════");
    println!("   PHENOMENAL CORRIDOR PREDICTIONS");
    println!("════════════════════════════════════════════════════════════════\n");

    println!("Model            │ Layers │ Predicted Peak │ Depth");
    println!("─────────────────┼────────┼────────────────┼───────");

    for preset in [
        ModelPreset::BgeM3,
        ModelPreset::BertBase,
        ModelPreset::BertLarge,
        ModelPreset::RobertaBase,
        ModelPreset::XlmRobertaBase,
    ] {
        let config = preset.config();
        let corridor = preset.phenomenal_corridor_layer();
        let depth = (corridor as f64 / config.num_layers as f64) * 100.0;

        println!(
            "{:16} │ {:6} │ Layer {:8} │ {:.1}%",
            config
                .model_id
                .split('/')
                .next_back()
                .unwrap_or(&config.model_id),
            config.num_layers,
            corridor,
            depth
        );
    }

    println!("\n");
    println!("The phenomenal corridor hypothesis predicts the effect appears");
    println!("at approximately 92% network depth across all architectures.");
    println!();
}