// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Organoid Consciousness Ethics Framework Demo
//!
//! Grows a cerebral organoid over 200 developmental days while applying
//! the quantitative consciousness ethics framework at each step. Prints
//! tier transitions, required actions, and a final ethics report.
//!
//! Run: cargo run -p symthaea-cell-foundry --example organoid_ethics_demo

use symthaea_cell_foundry::consciousness_ethics_framework::{
    ConsciousnessEthicsFramework, EthicsTier,
};
use symthaea_cell_foundry::organoid_pipeline::{OrganoidPipeline, OrganoidPipelineConfig};

fn main() {
    println!("==========================================================");
    println!("   Organoid Consciousness Ethics Framework Demo");
    println!("==========================================================");
    println!("  Growing neural tissue with ethics monitoring enabled.");
    println!("  Precautionary mode: ON (thresholds lowered by 30%)");
    println!("==========================================================\n");

    let config = OrganoidPipelineConfig {
        initial_cells: 50,
        max_cells: 5_000,
        seed: 42,
        ethics_phi_threshold: 1.0, // We let the framework decide, not the pipeline
        target_day: 200,
        enable_lfp_recording: true,
        morphogenetic_steps_per_day: 3,
    };

    let mut pipeline = OrganoidPipeline::new(config);
    let mut framework = ConsciousnessEthicsFramework::new();
    let mut prev_tier = EthicsTier::Tier0;
    let mut last_assessment = None;

    for _day in 0..200 {
        let snapshot = match pipeline.step() {
            Some(s) => s,
            None => {
                println!("  Pipeline halted at day {}.", _day);
                break;
            }
        };

        let metrics = pipeline.digital_organoid().metrics();
        let assessment = framework.assess(metrics, snapshot.lfp.as_ref());

        if assessment.current_tier != prev_tier {
            println!(
                "  Day {:3}: Tier {:?} -> {:?}  (Phi={:.4}, risk={:.3})",
                snapshot.day,
                prev_tier,
                assessment.current_tier,
                assessment.phi_estimate,
                assessment.risk_score,
            );
            for action in &assessment.required_actions {
                println!("           Action: {:?}", action);
            }
            println!("           Rec: {:?}\n", assessment.recommendation);
            prev_tier = assessment.current_tier;
        }

        last_assessment = Some(assessment);
    }

    if let Some(assessment) = &last_assessment {
        println!("\n{}", framework.format_report(assessment));

        let trend = framework.trend_analysis();
        println!("Assessments recorded: {}", framework.history().len());
        println!("Final Phi slope:      {:.6}/day", trend.phi_slope);
        if let Some(d) = trend.days_to_tier4_estimate {
            println!("Est. days to Tier 4:  {}", d);
        }
    }
}
