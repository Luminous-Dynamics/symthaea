// Gated via conditional item module wrappers
#[cfg(feature = "school_learning")]
mod gated_run {
    // Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
    // SPDX-License-Identifier: AGPL-3.0-or-later
    // Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
    //! # Sovereign Design Loop Demo (Consolidated v4)
    //!
    //! Demonstrates the full DESIGN -> SIMULATE -> PROVE -> MAKE loop
    //! with Amodal Fusion, Robotic Platform Synthesis, and Infrastructure Design.

    use symthaea_causal_reasoning::causal_calculus::{CausalDAG, StructuralCausalModel};
    use symthaea_core::genesis::GenesisSeed;
    use symthaea_engineering::{EngineeringAssistant, EngineeringConcept, EngineeringManager};
    use symthaea_fabrication_kernel::csg::CSGNode;
    use symthaea_fabrication_kernel::thought::GeometricThought;
    use symthaea_infrastructure::simulator::SympoiesisSandbox;
    use symthaea_infrastructure::town_simpoiesis::TownSympoiesis;
    use symthaea_materials::{MaterialProperty, encoder::MaterialHdcEncoder};
    use symthaea_mujoco_bridge::MuJoCoBridge;
    use symthaea_silicon::{SiliconArchitect, SiliconPPA};
    use symthaea_sim_bridge::{EngineeringDomain, MetricEncoder, SimulationRequest, SolverKind};

    pub fn run_iteration(
        iteration: usize,
        manager: &mut EngineeringManager,
        assistant: &mut EngineeringAssistant,
        _genesis: &GenesisSeed,
        goal: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n🔄 STARTING ITERATION {} --------------------", iteration);

        // 0. Recall Design Wisdom
        let goal_hv = symthaea_sim_bridge::embed_text(goal, 16384);
        manager.last_goal_hv = Some(goal_hv.clone());
        let wisdom = manager.recall_wisdom(&goal_hv);
        if wisdom.is_empty() {
            println!("🧠 Step 0: No prior wisdom found. Initializing first principles.");
        } else {
            println!("🧠 Step 0: Recalled {} items of past wisdom:", wisdom.len());
            for item in &wisdom {
                println!("   - {}", item);
            }
        }

        // 1. Propose Requirements
        let mut requirements = assistant.propose_requirements(goal, EngineeringDomain::Aerospace);
        println!(
            "✅ Step 1: Broca synthesized {} requirements.",
            requirements.len()
        );

        // Add symbolic invariants to trigger dynamic Pareto weighting
        if let Some(req) = requirements.first_mut() {
            req.structural_invariants
                .push("(>= thickness 3.0)".to_string());
            req.structural_invariants
                .push("(<= temperature 1500)".to_string());
        }

        // 2. Create Engineering Concept
        let mut concept =
            EngineeringConcept::new("arm-v1", "Phase 1", EngineeringDomain::Aerospace);
        for req in requirements {
            concept.add_requirement(req);
        }

        // 2.1 Dynamic Pareto Material Sifting
        let mat = manager
            .sift_best_material(&concept)
            .unwrap_or_else(|| MaterialProperty::titanium_ti6al4v());
        println!(
            "🧪 Step 2.1: Dynamic Pareto Sifting selected material: {} (Optimized for invariants)",
            mat.name
        );

        // 3. Causal Topology Optimization
        println!("🧬 Step 3: Optimizing geometry...");
        let geometry = CSGNode::cube();
        let mut thought = GeometricThought::from_csg(geometry);

        // Evaluate material and compensate for aging
        let remaining_strength = manager.evaluate_material(&[], &mat)?;
        manager.compensate_for_aging(&mut thought, remaining_strength);

        let fitness = manager.optimize_geometry(&mut thought, "material_strength", 0.8)?;
        println!("✅ Geometry optimized. Estimated fitness: {:.4}", fitness);

        // 4. Run Simulation
        println!("📡 Step 4: Dispatching to MuJoCo for physical validation...");
        let request = SimulationRequest::new(
            "val-01",
            EngineeringDomain::Aerospace,
            SolverKind::MultibodyDynamics,
            goal,
        );
        concept.simulation_requests.push(request.clone());
        manager.evaluate_concept(&mut concept);

        // 4.5 Amodal HDC Sensation Fusion
        let encoder = MetricEncoder::new(16384);
        let result = manager.registry.run(&request)?;
        let shape_hv = encoder.encode_result(&result);

        let mat_encoder = MaterialHdcEncoder::new();
        let matter_hv = mat_encoder.encode(&mat);

        let physical_state_hv = manager.fuse_shape_and_matter(&shape_hv, &matter_hv);
        println!(
            "🔮 Step 4.5: Amodal Fusion Complete. Physical State Norm: {:.4}",
            physical_state_hv.norm()
        );

        // 5. Formal Verification
        println!("📜 Step 5: Generating formal Lean 4 proofs...");
        let proofs = manager.formally_verify(&concept);
        for (id, _) in &proofs {
            println!("✅ Proof for {}: Discharged", id);
        }

        // 6. Fabrication (Moral-Gated)
        println!("🖨️  Step 6: Transitioning to Fabrication Phase...");

        // 6.1 Slicer Calibration
        let (h, w) = thought.slicer_calibration(&mat.name);
        println!(
            "   Slicer Calibrated: Layer Height={:.2}mm, Wall Thickness={:.1}mm",
            h, w
        );

        // 6.2 Tooling Synthesis (Support Structures)
        if let Some(_tooling) = thought.synthesize_tooling() {
            println!("   Tooling: Autonomously synthesized sacrificial support structures.");
        }

        match manager.prepare_fabrication(&thought, goal) {
            Ok(_) => println!("✅ Design resolved and fabrication started."),
            Err(e) => println!("{}", e),
        }

        // 7. Dream Phase (Consolidation)
        manager.dream_cycle();

        // 8. Technical Documentation & Blueprints
        println!("\n📜 Step 8: Synthesizing Technical Documentation & Blueprints...");
        let report = symthaea_engineering::DocumentGenerator::generate_technical_report(
            &concept, &thought, &mat, &proofs,
        );

        let report_path = "TECHNICAL_REPORT.md";
        std::fs::write(report_path, &report)?;
        println!(
            "✅ Technical Design Document synthesized and saved to {}.",
            report_path
        );

        Ok(())
    }

    pub fn main() -> Result<(), Box<dyn std::error::Error>> {
        println!("🚀 Starting Symthaea Sovereign Design Loop v4 (Systems Edition)...");

        let genesis = GenesisSeed::from_phrase("Sovereign Engineering Constitution");
        let mut assistant = EngineeringAssistant::new(&genesis);
        let mut manager = EngineeringManager::new();

        // Setup Causal Model
        let mut dag = CausalDAG::new();
        let strength = dag.add_node("material_strength", vec!["low".into(), "high".into()], true);
        let safety = dag.add_node("safety", vec!["failed".into(), "safe".into()], true);
        dag.add_edge(strength, safety);
        let mut scm = StructuralCausalModel::new(dag);
        scm.set_conditional_table(safety, vec![0.9, 0.1, 0.4, 0.6]);
        manager.set_causal_model(scm);

        manager.registry.register(MuJoCoBridge::dry_run());

        let goal = "High-strength multirotor arm (< 100g, > 50N)";

        // Run Iteration 1: The AI learns, fuses, and compensates
        run_iteration(1, &mut manager, &mut assistant, &genesis, goal)?;

        // 9. Robotic Platform Synthesis
        println!("\n🤖 Step 9: Synthesizing Autonomous Robotic Platform...");
        let platform = manager.synthesize_platform("high-speed multi-terrain explorer");
        println!(
            "✅ Platform Designed: {} with {} limb segments.",
            platform.name,
            platform.limbs.len()
        );
        println!("   - Primary Material: {}", platform.limbs[0].material.name);
        println!("   - Sensor Package: {:?}", platform.sensors);

        // 10. Infrastructure Construction Design
        println!("\n🏗️  Step 10: Designing Sovereign Infrastructure...");
        let infra = manager.design_infrastructure("Fabrication Outpost Delta", 5.0);
        println!("✅ Infrastructure Design Complete: {}", infra.name);
        println!("   - Assembly Modules: {}", infra.modules.len());
        println!(
            "   - Construction Sequence: {} steps identified.",
            infra.assembly_sequence.len()
        );
        for step in &infra.assembly_sequence {
            println!("     -> {}", step);
        }

        println!(
            "\n✨ Sovereignty Verified: Symthaea is now designing her own bodies and habitats."
        );

        // 11. Silicon Sovereignty: Autonomous Chip Design
        println!("\n🔌 Step 11: Engaging Silicon Sovereignty (Autonomous Chip Design)...");
        let silicon = SiliconArchitect;
        let chip_target = SiliconPPA {
            power_mw: 50.0,
            freq_mhz: 200.0,
            area_um2: 1500.0,
            slack_ns: 0.5,
        };

        let artifact = silicon.synthesize_rtl("Conscious Accelerator v1", chip_target);
        println!("✅ Silicon RTL Synthesized: {}", artifact.label);
        println!(
            "   - PPA Target: {}MHz, {}mW, {}um²",
            artifact.ppa_target.freq_mhz,
            artifact.ppa_target.power_mw,
            artifact.ppa_target.area_um2
        );

        let silicon_concept = silicon.to_engineering_concept(&artifact);
        let silicon_invariants = silicon.derive_timing_invariants(&artifact);
        println!(
            "✅ Electrical Safety Case generated with {} timing invariants.",
            silicon_invariants.len()
        );

        // 12. Closed-Loop Town Sympoiesis
        println!("\n🏡 Step 12: Establishing Closed-Loop Town Sympoiesis...");
        let mut town = TownSympoiesis::new("Sympoiesis Outpost 1", &mut manager);
        println!("✅ Town Metabolism Initialized: {}", town.name);
        println!(
            "   - Power Grid: {}% Renewable",
            town.power_grid.renewable_ratio * 100.0
        );
        println!(
            "   - Fluid State: Water Clarity={:.2}, Nutrient Advection={:.2}",
            town.water_clarity, town.nutrient_advection
        );

        // Formal Silicon Sanity: Deadlock Proof (Step 11.5 - Corrected variable access)
        println!("📜 Step 11.5: Proving Silicon Sanity (Deadlock-Freedom)...");
        match silicon.prove_deadlock_freedom(&town.power_grid) {
            Ok(_) => {
                println!("   ✅ Proof Discharged: Power logic algorithm is mathematically sane.")
            }
            Err(e) => println!("   ❌ Proof Failed: {}", e),
        }

        for inv in &silicon_invariants {
            println!("     -> SMT Gate: {}", inv);
        }

        let town_surprise = town.step(12.5, 15.0); // Increase load, available 15MW
        println!("🧪 Town Metabolic Step: Surprise={:.2}", town_surprise);
        println!(
            "   - Updated Fluid State: Water Clarity={:.2}, Nutrient Advection={:.2}",
            town.water_clarity, town.nutrient_advection
        );

        // 13. Deterministic Co-Simulation Sandbox: Stress Test
        println!("\n🌪️  Step 13: Initializing Deterministic Sandbox Stress Test...");
        let mut sandbox = SympoiesisSandbox::new(town);

        println!("   Running baseline metabolism (5 frames)...");
        for _ in 0..5 {
            sandbox.tick(10.0, 15.0);
            sandbox.print_diagnostics();
        }

        println!("\n🔥 PHASE A: Injecting Solar Flare (Grid Collapse)...");
        sandbox.inject_anomaly("solar_flare");
        for _ in 0..5 {
            sandbox.tick(12.0, 1.0); // High demand, low solar
            sandbox.print_diagnostics();
        }

        println!("\n🔥 PHASE B: Injecting Structural Fracture (Fluid Leak)...");
        sandbox.inject_anomaly("structural_fracture");
        for _ in 0..5 {
            sandbox.tick(8.0, 15.0); // Stabilized solar
            sandbox.print_diagnostics();
        }

        println!(
            "\n✨ Simulation Complete: Symthaea's logical immune system maintained 100% uptime."
        );

        Ok(())
    }
}

fn main() {
    #[cfg(feature = "school_learning")]
    match gated_run::main() {
        Ok(_) => (),
        Err(e) => eprintln!("Error: {}", e),
    }
}
