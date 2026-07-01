// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Curriculum Research Benchmark
//!
//! Measures the end-to-end pipeline: research → ingest → recall.

use std::time::Instant;

use symthaea::Symthaea;
use tracing::Level;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    let topic = std::env::var("SYMTHAEA_RESEARCH_BENCH_TOPIC")
        .unwrap_or_else(|_| "State Space Models and Liquid Time-Constant Networks".to_string());
    let query = std::env::var("SYMTHAEA_RESEARCH_BENCH_QUERY").unwrap_or_else(|_| {
        "Explain how SSM linear scaling relates to energy efficiency.".to_string()
    });

    println!("\n⏱  Symthaea Curriculum Research Benchmark\n");
    println!("🔍 Topic: {topic}");

    let mut sym = Symthaea::new(1024, 64).await?;
    let dimension = sym.dimension();

    let before = sym.curriculum.objectives.len();
    let db = sym.database_arc();
    let research_start = Instant::now();

    let summary = if let Some(mut extender) = sym.curriculum_extender.take() {
        let result = extender
            .research_and_extend(&topic, &mut sym.curriculum, dimension, db)
            .await;
        sym.curriculum_extender = Some(extender);
        result?
    } else {
        println!("❌ Web research module is not available.");
        return Ok(());
    };

    let research_duration = research_start.elapsed();
    let after = sym.curriculum.objectives.len();
    let new_count = after.saturating_sub(before);

    println!("✅ Research + ingest complete");
    println!("   ⏱  Duration: {:.2?}", research_duration);
    println!("   📚 Objectives added: {}", new_count);
    println!("   🧭 Confidence: {:.2}", summary.confidence);

    let save_start = Instant::now();
    if let Err(e) = sym.record_research(&topic, summary.objectives_added) {
        println!("⚠️  Failed to record research metadata: {}", e);
    }
    let save_duration = save_start.elapsed();
    println!("   💾 Save time: {:.2?}", save_duration);

    let recall_start = Instant::now();
    let response = sym.process(&query).await?;
    let recall_duration = recall_start.elapsed();

    println!("\n🧠 Recall query: {query}");
    println!("🤖 Response length: {} chars", response.content.len());
    println!("⏱  Recall duration: {:.2?}", recall_duration);

    println!("\n🎯 Total objectives: {}", sym.curriculum.objectives.len());

    Ok(())
}
