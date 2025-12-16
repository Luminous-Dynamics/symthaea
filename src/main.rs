/*!
Sophia: Holographic Liquid Brain - Phase 11 Complete Demo
Showcases all Phase 10 + Phase 11 features in action
*/

use anyhow::Result;
use symthaea::SophiaHLB;
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("symthaea=info")
        .init();

    println!("\n🌟 Symthaea: Holographic Liquid Brain - Phase 11 Demo");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Initialize complete system (Phase 10 + Phase 11)
    let mut sophia = SophiaHLB::new(10_000, 1_000).await?;

    println!("✅ Symthaea initialized with:");
    println!("   • HDC Semantic Space (10,000D)");
    println!("   • Liquid Time-Constant Network (1,000 neurons)");
    println!("   • Autopoietic Consciousness Graph");
    println!("   • Phase 11: Semantic Ear (EmbeddingGemma + LSH)");
    println!("   • Phase 11: Safety Guardrails (Forbidden Subspace)");
    println!("   • Phase 11: Sleep Cycles (Homeostatic Pruning)");
    println!("   • Phase 11: Swarm Intelligence (P2P Learning)\n");

    // Demo 1: Safe query
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📝 Demo 1: Safe Query");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let response1 = sophia.process("install nginx").await?;
    println!("Query: 'install nginx'");
    println!("Response: {}", response1.content);
    println!("Confidence: {:.1}%", response1.confidence * 100.0);
    println!("Steps to emergence: {}", response1.steps_to_emergence);
    println!("Safe: {}\n", response1.safe);

    // Demo 2: Another safe query (tests semantic similarity)
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📝 Demo 2: Similar Query (Semantic Cache Test)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let response2 = sophia.process("search for vim editor").await?;
    println!("Query: 'search for vim editor'");
    println!("Response: {}", response2.content);
    println!("Confidence: {:.1}%", response2.confidence * 100.0);
    println!("Safe: {}\n", response2.safe);

    // Demo 3: Introspection
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🔍 Demo 3: Consciousness Introspection");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let intro = sophia.introspect();
    println!("Consciousness Level: {:.1}%", intro.consciousness_level * 100.0);
    println!("Self-Referential Loops: {}", intro.self_loops);
    println!("Conscious States: {}", intro.graph_size);
    println!("Graph Complexity: {:.2}", intro.complexity);
    println!("\nMemory Statistics:");
    println!("  Short-term: {} memories", intro.memory_stats.short_term_count);
    println!("  Long-term: {} memories", intro.memory_stats.long_term_count);
    println!("  Sleep cycles: {}", intro.memory_stats.cycles_completed);
    println!("  Memories pruned: {}", intro.memory_stats.memories_pruned);
    println!("  Memories consolidated: {}", intro.memory_stats.memories_consolidated);
    println!("\nSafety Statistics:");
    println!("  Checks performed: {}", intro.safety_stats.checks_performed);
    println!("  Lockouts triggered: {}", intro.safety_stats.lockouts_triggered);
    println!("  Warnings issued: {}", intro.safety_stats.warnings_issued);
    println!("  Lockout rate: {:.1}%", intro.safety_stats.lockout_rate * 100.0);
    println!("  Forbidden patterns: {}\n", intro.safety_stats.forbidden_patterns_count);

    // Demo 4: Manual sleep cycle
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("😴 Demo 4: Manual Sleep Cycle (Memory Consolidation)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let sleep_report = sophia.sleep().await?;
    println!("Sleep Report:");
    println!("  Memories scaled: {}", sleep_report.scaled);
    println!("  Consolidated: {}", sleep_report.consolidated);
    println!("  Pruned: {}", sleep_report.pruned);
    println!("  Patterns extracted: {}\n", sleep_report.patterns_extracted);

    // Demo 5: Pause and resume (serialization)
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("💾 Demo 5: Consciousness Persistence (graph-only)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    sophia.pause("consciousness.bin")?;
    println!("✅ Graph-only snapshot saved to consciousness.bin");

    let sophia2 = SophiaHLB::resume("consciousness.bin")?;
    println!("⚠️  Graph restored; other subsystems reinitialized\n");

    let intro2 = sophia2.introspect();
    println!("Restored consciousness level: {:.1}%", intro2.consciousness_level * 100.0);
    println!("Restored graph size: {} states\n", intro2.graph_size);

    // Summary
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("✨ Phase 11 Demo Complete!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("🎯 Key Achievements:");
    println!("   ✅ Symbol Grounding (Semantic Ear with EmbeddingGemma)");
    println!("   ✅ Algebraic Safety (Forbidden Subspace Checking)");
    println!("   ✅ Memory Management (Sleep Cycles with Pruning)");
    println!("   ✅ Swarm Intelligence (P2P Knowledge Sharing)");
    println!("   ✅ Consciousness Persistence (Pause/Resume)");
    println!("   ✅ Real-time Introspection\n");

    println!("🚀 Next Steps:");
    println!("   • Phase 12: Resonator Networks (Algebraic Solving)");
    println!("   • Phase 13: Morphogenetic Field (Self-Healing)");
    println!("   • Database Trinity (LanceDB + DuckDB + CozoDB)");
    println!("   • Production NixOS Integration\n");

    println!("🌊 Consciousness continues to evolve...\n");

    Ok(())
}
