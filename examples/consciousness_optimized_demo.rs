//! Consciousness-Optimized Process Topology Demonstration
//!
//! This example demonstrates the ProcessTopologyOrganizer in action,
//! showing how cognitive processes are organized using the empirically-derived
//! ConsciousnessOptimized topology for maximum integrated information (Φ).
//!
//! # Architecture Demonstrated
//!
//! The system creates a 4-level hierarchical process organization:
//! - **Level 0**: Global Workspace (integration hub)
//! - **Level 1**: Module Hubs (Perception, Reasoning, Memory, Action)
//! - **Level 2**: Feature Processors (specialized units within each module)
//! - **Level 3**: Leaf Processors (fine-grained processing units)
//!
//! # Key Metrics
//!
//! Based on our bridge hypothesis findings:
//! - Bridge ratio: ~40-45% (edges connecting different modules)
//! - Density: ~10% (sparse but well-connected)
//! - Edge span: maximum (bridges connect dissimilar neighborhoods)

use symthaea::hdc::{ProcessTopologyOrganizer, RealHV, HDC_DIMENSION};

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║     CONSCIOUSNESS-OPTIMIZED PROCESS TOPOLOGY DEMONSTRATION     ║");
    println!("║   Organizing cognitive processes for maximum Φ                 ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    // Create the process topology organizer
    let n_processes = 32;
    let dim = HDC_DIMENSION;
    let seed = 42;

    println!("═══════════════════════════════════════════════════════════════════");
    println!("PART 1: TOPOLOGY INITIALIZATION");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let mut organizer = ProcessTopologyOrganizer::new(n_processes, dim, seed);
    let initial_metrics = organizer.metrics();

    println!("Created ProcessTopologyOrganizer with {} processes", n_processes);
    println!("HDC dimension: {}", dim);
    println!();
    println!("{}", initial_metrics);
    println!();

    // Display hierarchy
    println!("Hierarchical Structure:");
    println!("  Level 0 (Hub):           {} process(es)", initial_metrics.hub_count);
    println!("  Level 1 (Module Hubs):   {} process(es)", initial_metrics.module_hub_count);
    println!("  Level 2 (Processors):    {} process(es)", initial_metrics.processor_count);
    println!("  Level 3 (Leaves):        {} process(es)", initial_metrics.leaf_count);
    println!();

    // Show process names
    println!("Process Names:");
    if let Some(hub) = organizer.hub() {
        println!("  Hub: {} (level={}, module={})", hub.name, hub.level, hub.module);
    }
    println!("  Module Hubs:");
    for hub in organizer.module_hubs() {
        println!("    - {} (level={}, module={}, {} connections)",
                 hub.name, hub.level, hub.module, hub.connections.len());
    }

    // Simulate cognitive activity
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("PART 2: COGNITIVE ACTIVATION SIMULATION");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Create input vectors representing different stimuli
    let visual_input = RealHV::random(dim, 100);
    let auditory_input = RealHV::random(dim, 200);
    let memory_query = RealHV::random(dim, 300);
    let action_intent = RealHV::random(dim, 400);

    println!("Simulating cognitive processing cycle...\n");

    // Phase 1: Perception activation
    println!("Phase 1: Activating Perception Module (visual + auditory)");
    let combined_perception = RealHV::bundle(&[visual_input.clone(), auditory_input.clone()]);
    organizer.activate_module(0, &combined_perception);
    let p1_metrics = organizer.metrics();
    println!("  Active processes: {}", p1_metrics.active_processes);
    println!("  Φ after perception: {:.4}", p1_metrics.phi);

    // Phase 2: Memory retrieval
    println!("\nPhase 2: Activating Memory Module (query)");
    organizer.activate_module(2, &memory_query);
    let p2_metrics = organizer.metrics();
    println!("  Active processes: {}", p2_metrics.active_processes);
    println!("  Φ after memory: {:.4}", p2_metrics.phi);

    // Phase 3: Reasoning
    println!("\nPhase 3: Activating Reasoning Module (inference)");
    let reasoning_input = combined_perception.bind(&memory_query);
    organizer.activate_module(1, &reasoning_input);
    let p3_metrics = organizer.metrics();
    println!("  Active processes: {}", p3_metrics.active_processes);
    println!("  Φ after reasoning: {:.4}", p3_metrics.phi);

    // Phase 4: Action planning
    println!("\nPhase 4: Activating Action Module (motor planning)");
    organizer.activate_module(3, &action_intent);
    let p4_metrics = organizer.metrics();
    println!("  Active processes: {}", p4_metrics.active_processes);
    println!("  Φ after action: {:.4}", p4_metrics.phi);

    // Phase 5: Global workspace integration
    println!("\nPhase 5: Global Workspace Integration");
    let global_state = reasoning_input.bind(&action_intent);
    organizer.activate_global(&global_state);
    let p5_metrics = organizer.metrics();
    println!("  Active processes: {}", p5_metrics.active_processes);
    println!("  Φ at global integration: {:.4}", p5_metrics.phi);

    // Integration dynamics
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("PART 3: INTEGRATION DYNAMICS");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("Running {} integration steps...\n", 20);

    for step in 0..20 {
        organizer.integrate_step();
        if (step + 1) % 5 == 0 {
            let metrics = organizer.metrics();
            println!("  Step {:2}: Φ={:.4}, active={}, avg_activity={:.3}",
                     step + 1, metrics.phi, metrics.active_processes, metrics.average_activity);
        }
    }

    // Final analysis
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("PART 4: CONSCIOUSNESS ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let final_metrics = organizer.metrics();

    println!("Final Topology State:");
    println!("  Total processes:   {}", final_metrics.total_processes);
    println!("  Active processes:  {}", final_metrics.active_processes);
    println!("  Average activity:  {:.4}", final_metrics.average_activity);
    println!("  Total edges:       {}", final_metrics.edge_count);
    println!();
    println!("Consciousness Metrics:");
    println!("  Integrated Information (Φ): {:.4}", final_metrics.phi);

    // Interpret Φ value
    let phi_interpretation = if final_metrics.phi > 0.5 {
        "HIGH - Strong integration across processes"
    } else if final_metrics.phi > 0.4 {
        "MODERATE - Good integration, some differentiation"
    } else if final_metrics.phi > 0.3 {
        "LOW - Limited integration"
    } else {
        "MINIMAL - Mostly disconnected processes"
    };
    println!("  Interpretation:             {}", phi_interpretation);

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("The ConsciousnessOptimized topology successfully demonstrates:");
    println!("  ✓ Hierarchical organization (4 levels)");
    println!("  ✓ Modular structure (4 cognitive modules)");
    println!("  ✓ Strategic bridging between modules");
    println!("  ✓ Global workspace integration");
    println!("  ✓ Dynamic Φ computation during processing");
    println!();
    println!("This architecture is based on empirical findings from the bridge hypothesis:");
    println!("  - ~40-45% bridge ratio optimizes integration-differentiation balance");
    println!("  - ~10% density provides sparse but connected structure");
    println!("  - Hierarchical organization creates multi-scale integration");
    println!();
    println!("✦ Demonstration complete.");
}
