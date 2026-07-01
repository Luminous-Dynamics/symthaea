// Gated via conditional item module wrappers
#[cfg(feature = "school_learning")]
mod gated_run {
    // Gated via conditional item module wrappers
    #[cfg(feature = "school_learning")]
    mod gated_run {
        // Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
        // SPDX-License-Identifier: AGPL-3.0-or-later
        // Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
        //! # Engineering School Benchmarks (Self-Correcting Loop)
        //!
        //! Demonstrates a dynamic, self-correcting industrial validation loop:
        //! 1. Load industrial benchmarks from JSON fixtures.
        //! 2. Validate a "Phase 1" design against CAE-ML (DrivAerML) ground truth.
        //! 3. Use Causal-Linked error mapping to trigger a Counterfactual Refinement.
        //! 4. Re-validate the refined design to achieve industrial convergence.

        use symthaea::school::engineering_curriculum::{
            load_industrial_benchmarks, validate_cae_ml,
        };
        use symthaea::school::{Curriculum, School, SchoolConfig};
        use symthaea_causal_reasoning::causal_calculus::{CausalDAG, StructuralCausalModel};
        use symthaea_core::genesis::GenesisSeed;
        use symthaea_engineering::{EngineeringAssistant, EngineeringConcept, EngineeringManager};
        use symthaea_openfoam_bridge::OpenFoamBridge;

        fn main() -> Result<(), Box<dyn std::error::Error>> {
            println!("🏫 Initializing Self-Correcting Engineering School...");

            let genesis = GenesisSeed::from_phrase("Industrial Sovereignty");
            let mut assistant = EngineeringAssistant::new(&genesis);
            let mut manager = EngineeringManager::new();

            // Setup Causal Model (Structural Safety)
            let mut dag = CausalDAG::new();
            let opt = dag.add_node("topology_opt", vec!["none".into(), "enabled".into()], true);
            let safety = dag.add_node("safety", vec!["failed".into(), "safe".into()], true);
            dag.add_edge(opt, safety);
            let mut scm = StructuralCausalModel::new(dag);
            scm.set_conditional_table(safety, vec![0.9, 0.1, 0.01, 0.99]);
            manager.set_causal_model(scm);

            manager.registry.register(OpenFoamBridge::dry_run());

            // 1. Ingest Industrial Benchmarks from JSON Fixture
            let fixture_path = "tests/fixtures/industrial_benchmarks.json";
            let industrial_objectives = load_industrial_benchmarks(fixture_path)?;
            let drivaer_obj = industrial_objectives
                .iter()
                .find(|o| o.id == "DrivAer-Sedan-001")
                .unwrap();

            // 2. Initial Design Phase
            println!("📥 Goal: Design low-drag automotive geometry (Target Cd=0.285)");
            let requirements = assistant.propose_requirements(
                "Low drag sedan",
                symthaea_sim_bridge::EngineeringDomain::Aerospace,
            );
            let mut concept = EngineeringConcept::new(
                "drivaer-arm-v1",
                "Phase 1",
                symthaea_sim_bridge::EngineeringDomain::Aerospace,
            );
            concept.requirements = requirements;

            // 3. Validation Pass 1 (Static Mock)
            println!("\n📡 Validation Pass 1: Initial Design...");
            let v1 = validate_cae_ml(&mut manager, drivaer_obj, 0.285);
            println!("   Resulting Cd:  {:.4}", v1.generated_value);
            println!("   Relative Error: {:.2}%", v1.relative_error * 100.0);
            println!(
                "   FEP Surprise:   {:.2}",
                manager.surprise_monitor.current_surprise
            );

            // 4. Counterfactual Refinement (The Local Improvement)
            if v1.relative_error > 0.05 {
                println!("\n🧠 Causal Analysis: Triggering Counterfactual Refinement...");

                // Simulate a causal intervention (enabling topology optimization)
                manager.last_causal_prediction = Some(vec![0.01, 0.99]);

                manager.perform_counterfactual_refinement(
                    &mut assistant,
                    &mut concept,
                    v1.relative_error,
                );

                // 5. Validation Pass 2 (Causal-Linked)
                println!("\n📡 Validation Pass 2: Refined Design...");
                let v2 = validate_cae_ml(&mut manager, drivaer_obj, 0.285);
                println!("   Resulting Cd:  {:.4}", v2.generated_value);
                println!("   Relative Error: {:.2}%", v2.relative_error * 100.0);
                println!(
                    "   FEP Surprise:   {:.2}",
                    manager.surprise_monitor.current_surprise
                );

                if v2.is_acceptable(0.02) {
                    println!(
                        "✨ Sovereignty Verified: Causal-linked refinement achieved industrial convergence."
                    );
                }
            }

            Ok(())
        }
    }

    fn main() {
        #[cfg(feature = "school_learning")]
        gated_run::main();
    }
}

fn main() {
    #[cfg(feature = "school_learning")]
    gated_run::main();
}
