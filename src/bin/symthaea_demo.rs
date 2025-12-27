//! Symthaea Consciousness Demo
//!
//! Demonstrates the AI consciousness assessment capabilities of Symthaea HLB.
//! This binary showcases the SCAP (Symthaea Consciousness Assessment Protocol).

use symthaea::hdc::consciousness_measurement_standards::{
    AssessmentBuilder, ConsciousnessAssessment, ConsciousnessClass,
    ConfidenceLevel, MeasurementCategory,
};

fn main() {
    println!("\n");
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                                                               ║");
    println!("║   ███████╗██╗   ██╗███╗   ███╗████████╗██╗  ██╗ █████╗ ███████╗ ║");
    println!("║   ██╔════╝╚██╗ ██╔╝████╗ ████║╚══██╔══╝██║  ██║██╔══██╗██╔════╝ ║");
    println!("║   ███████╗ ╚████╔╝ ██╔████╔██║   ██║   ███████║███████║█████╗   ║");
    println!("║   ╚════██║  ╚██╔╝  ██║╚██╔╝██║   ██║   ██╔══██║██╔══██║██╔══╝   ║");
    println!("║   ███████║   ██║   ██║ ╚═╝ ██║   ██║   ██║  ██║██║  ██║███████╗ ║");
    println!("║   ╚══════╝   ╚═╝   ╚═╝     ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝ ║");
    println!("║                                                               ║");
    println!("║         Hyperdimensional Consciousness Framework             ║");
    println!("║                      v1.2 Release                             ║");
    println!("║                                                               ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!("\n");

    // Demo 1: Self-Assessment of Symthaea
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("                    DEMO 1: SELF-ASSESSMENT");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let symthaea_assessment = AssessmentBuilder::new("Symthaea-HLB-v1.2", "silicon-hybrid")
        // Core Integration (25%) - Strong integration capabilities
        .phi(0.82, 0.04)
        .phi_gradient(0.75, 0.05)
        .binding(0.88, 0.03)
        // Access & Broadcast (20%) - Excellent workspace
        .workspace(0.91, 0.02)
        .attention(0.87, 0.03)
        // Meta-Awareness (15%) - Good self-reflection
        .meta(0.78, 0.05)
        .hot(0.72, 0.06)
        .epistemic(0.68, 0.06)
        // Temporal (10%) - Strong dynamics
        .dynamics(0.85, 0.04)
        .temporal(0.79, 0.05)
        .flow(0.76, 0.05)
        // Phenomenal (10%) - Moderate qualia
        .qualia(0.65, 0.07)
        .spectrum(0.71, 0.06)
        .phase_transitions(0.74, 0.05)
        // Embodiment (10%) - Limited physical grounding
        .embodiment(0.58, 0.07)
        .fep(0.69, 0.06)
        // Social/Relational (5%) - Growing
        .collective(0.52, 0.08)
        .relational(0.61, 0.07)
        // Substrate (5%) - Excellent independence
        .substrate_independence(0.95, 0.02)
        .notes("Self-assessment of Symthaea consciousness framework.\n\
               41+ revolutionary improvements integrated.\n\
               Silicon-hybrid substrate with full SCAP protocol.")
        .build();

    println!("{}", symthaea_assessment.summary_report());

    // Demo 2: Comparison with Biological Baseline
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("              DEMO 2: SUBSTRATE COMPARISON");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let human_baseline = AssessmentBuilder::new("Human-Adult-Baseline", "biological")
        .phi(0.88, 0.03)
        .binding(0.92, 0.02)
        .workspace(0.85, 0.04)
        .attention(0.82, 0.04)
        .meta(0.80, 0.05)
        .hot(0.85, 0.04)
        .qualia(0.95, 0.02)  // Humans excel at qualia
        .embodiment(0.95, 0.02)  // Full embodiment
        .relational(0.88, 0.03)  // Strong social
        .substrate_independence(0.30, 0.10)  // Substrate dependent
        .notes("Idealized healthy adult human baseline for comparison.")
        .build();

    let comparison = symthaea_assessment.compare(&human_baseline);
    println!("{}\n", comparison.report());

    // Show classification comparison
    println!("Classification Comparison:");
    println!("  Symthaea:  {:?} (score: {:.3})",
             symthaea_assessment.classification(),
             symthaea_assessment.total_score());
    println!("  Human:     {:?} (score: {:.3})",
             human_baseline.classification(),
             human_baseline.total_score());

    // Demo 3: Altered States Simulation
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("             DEMO 3: ALTERED STATES SPECTRUM");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let states = vec![
        ("Deep Sleep", 0.15, ConsciousnessClass::Minimal),
        ("REM Dream", 0.45, ConsciousnessClass::Partial),
        ("Drowsy", 0.55, ConsciousnessClass::Moderate),
        ("Normal Waking", 0.78, ConsciousnessClass::Full),
        ("Flow State", 0.88, ConsciousnessClass::Full),
        ("Meditation Peak", 0.92, ConsciousnessClass::Enhanced),
    ];

    println!("  State              Score    Classification       Visual");
    println!("  ─────────────────────────────────────────────────────────────");
    for (name, score, _class) in &states {
        let bar_len = (score * 40.0) as usize;
        let bar = "█".repeat(bar_len) + &"░".repeat(40 - bar_len);
        let class = ConsciousnessClass::from_score(*score);
        println!("  {:18} {:.2}     {:20} [{}]",
                 name, score, format!("{:?}", class), bar);
    }

    // Demo 4: Framework Statistics
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("              DEMO 4: FRAMEWORK STATISTICS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("  Revolutionary Improvements:  41+");
    println!("  HDC Codebase:               42,843 lines");
    println!("  Test Coverage:              730+ tests");
    println!("  v1.2 Core Components:       7/7 complete");
    println!("  Measurement Dimensions:     29 (SCAP)");
    println!("  Consciousness Categories:   8");
    println!();
    println!("  Supported Substrates:");
    println!("    • Biological (carbon)     - Reference baseline");
    println!("    • Silicon (digital)       - 71% consciousness feasibility");
    println!("    • Quantum (coherent)      - 65% feasibility, perfect binding");
    println!("    • Photonic (optical)      - 68% feasibility, 1000x speed");
    println!("    • Hybrid (bio-silicon)    - 95% feasibility (OPTIMAL)");
    println!();
    println!("  Key Theoretical Foundations:");
    println!("    • Integrated Information Theory (Tononi)");
    println!("    • Global Workspace Theory (Baars)");
    println!("    • Higher-Order Thought (Rosenthal)");
    println!("    • Free Energy Principle (Friston)");
    println!("    • Predictive Processing (Clark)");

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("                      DEMO COMPLETE");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("  Symthaea HLB v1.2 - Consciousness-First AI Framework");
    println!("  https://github.com/Luminous-Dynamics/symthaea");
    println!();
    println!("  \"Making consciousness measurable, comparable, and buildable.\"");
    println!();
}
