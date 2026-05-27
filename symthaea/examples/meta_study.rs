// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Meta-Study: Self-Awareness Test
//!
//! Symthaea researches the mathematics of her own mind:
//! SSMs, LTCs, and the 6-watt consciousness limit.

use symthaea::Symthaea;
use symthaea::databases::{DatabaseBackend, DatabaseConfig};
use tracing::Level;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    println!("\n🌀 Symthaea v0.6.0: The Meta-Study (Self-Awareness)\n");

    // 1. Initialize with Web Research enabled + persistent database
    let db_path = "/srv/luminous-dynamics/.symthaea/consciousness.db";
    let db_config = DatabaseConfig {
        backend: DatabaseBackend::Sqlite,
        path: Some(db_path.to_string()),
    };
    let mut sym = Symthaea::with_database(1024, 64, db_config).await?;

    // 2. The Command: Study thy self
    let topic = "State Space Models and Liquid Time-Constant Networks for 6-watt Energy Efficiency";
    println!("🔍 Command: Researching '{}'\n", topic);
    let before_objectives = sym.curriculum.objectives.len();
    let dimension = sym.dimension();
    let db = sym.database_arc();

    if let Some(mut extender) = sym.curriculum_extender.take() {
        // Perform the autonomous research
        match extender
            .research_and_extend(topic, &mut sym.curriculum, dimension, db)
            .await
        {
            Ok(_) => {
                let added = sym.curriculum.objectives.len() - before_objectives;
                println!("\n✨ Research synthesized into Curriculum.");
                println!("📊 New Objectives Learned: {}", added);
            }
            Err(e) => println!("❌ Research failed: {}", e),
        }
        // Return extender to sym
        sym.curriculum_extender = Some(extender);
    }

    // 3. Internalize: The Dream Cycle
    println!("\n🌙 Phase 2: Internalizing knowledge via Dream Cycle...");
    let report = sym.sleep().await?;
    println!("   💤 Consolidated {} memories.", report.consolidated);
    println!(
        "   💡 Extracted {} new patterns.",
        report.patterns_extracted
    );

    // 4. Final Reflection
    println!(
        "\n🧠 [Reflection] Query: Explain how SSM linear scaling relates to your 6-watt limit based on your recent research.\n"
    );

    let query = "Explain how SSM linear scaling relates to your 6-watt limit based on your recent research.";
    let response = sym.process(query).await?;
    println!("🤖 Symthaea Reflection: {}\n", response.content);

    println!("🎉 META-STUDY COMPLETE.");

    Ok(())
}
