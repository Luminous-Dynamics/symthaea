// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Geodesic Bridge — Bridges Broca's semantic nuclei to Geodesic's topological synthesis.
//!
//! Maps high-level abstract intent (Macro-HVs) to topological program skeletons.
//!
//! **Note on duplication**: an equivalent `GeodesicBridge` also exists in
//! `symthaea-broca-tools` (`symthaea-broca-tools/src/geodesic_bridge.rs`), which
//! has its own toolkit-facing consumers. This crate needs its own copy because
//! `LiquidMambaGenerator::synthesize_program()` is called from `symthaea-broca`'s
//! own `src/bin/broca_exercism_bench.rs` and `examples/hierarchical_reasoning.rs`
//! — `symthaea-broca` cannot depend on `symthaea-broca-tools` (that's the reverse
//! of the crate's established, correct toolkit→core dependency direction; see
//! `SYMTHAEA_IMPROVEMENT_PLAN_2026-07.md` Phase 2's `mamba-cpu` investigation for
//! the fuller version of this same structural constraint). Both copies depend
//! only on `symthaea_core`/`symthaea_geodesic`, so there's no cross-crate coupling
//! introduced by the duplication — just two independent, small implementations of
//! the same synthesis logic. Trims `synthesize_mujoco_model` and
//! `synthesize_leptos_dashboard` (broca-tools-only capabilities, unused here) to
//! keep this copy scoped to what `synthesize_program()` actually needs.

#![cfg(feature = "code-sheaf-eval")]

use anyhow::Result;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
use symthaea_geodesic::manifold::ProgramManifold;
use symthaea_geodesic::skeleton_synthesis::{ActiveInferenceResult, active_inference_synthesize};
use symthaea_geodesic::topology::BettiNumbers;

/// Orchestrates the transition from semantic thought to topological code.
#[derive(Clone)]
pub struct GeodesicBridge {
    pub manifold: ProgramManifold,
    /// Prototypical HVs for topological structures.
    pub proto_linear: ContinuousHV,
    pub proto_branch: ContinuousHV,
    pub proto_loop: ContinuousHV,
    pub proto_recursion: ContinuousHV,
}

impl GeodesicBridge {
    pub fn new(genesis: &GenesisSeed) -> Self {
        Self {
            manifold: ProgramManifold::new(),
            proto_linear: genesis.hv("topo-linear", 16384),
            proto_branch: genesis.hv("topo-branch", 16384),
            proto_loop: genesis.hv("topo-loop", 16384),
            proto_recursion: genesis.hv("topo-recursion", 16384),
        }
    }

    /// Synthesize a program skeleton from a semantic nucleus.
    pub fn synthesize_from_nucleus(
        &self,
        nucleus: &ContinuousHV,
        name: &str,
    ) -> Result<ActiveInferenceResult> {
        // 1. Infer topology from nucleus similarity to prototypes
        let sim_branch = nucleus.similarity(&self.proto_branch);
        let sim_loop = nucleus.similarity(&self.proto_loop);
        let sim_recurse = nucleus.similarity(&self.proto_recursion);

        let mut beta1 = 0;
        let mut hints = vec![name];

        if sim_recurse > 0.3 {
            beta1 = 1;
            hints.push("recursive");
        } else if sim_loop > 0.3 {
            beta1 = 1;
            hints.push("iterate");
        }

        if sim_branch > 0.3 {
            hints.push("conditional branch");
        }

        let target_betti = BettiNumbers {
            beta_0: 1,
            beta_1: beta1,
            beta_2: 0,
        };

        // 2. Run Active Inference Synthesis
        let mut result = active_inference_synthesize(
            &target_betti,
            &hints,
            &self.manifold,
            None, // No expected output HV yet
            10,   // Max iterations
        );

        // 3. Post-process: Fill defaults if manifold was empty (for demo stability)
        if result.emitted_code.is_none() {
            symthaea_geodesic::skeleton_synthesis::fill_skeleton_defaults_for_signature(
                &mut result.skeleton,
                Some(&format!("fn {}(items: &[i32]) -> i32", name)),
            );
            result.emitted_code = result.skeleton.emit_rust(1);
        }

        // 4. Wrap in WASM FFI Shims (The Hardening Layer)
        if let Some(ref mut code) = result.emitted_code {
            let safe_name = name.replace("-", "_");
            let wrapped_code = format!(
                r#"
#[unsafe(no_mangle)]
static mut HV_BUFFER: [f32; 65536] = [0.0; 65536];

#[unsafe(no_mangle)]
pub extern "C" fn get_hypervector_buffer_ptr() -> *mut f32 {{
    unsafe {{ HV_BUFFER.as_mut_ptr() }}
}}

#[unsafe(no_mangle)]
pub extern "C" fn {}(ptr: *mut f32, len: i32) {{
    let _slice = unsafe {{ std::slice::from_raw_parts_mut(ptr, len as usize) }};

    // --- Synthesized Logic Start ---
    {{
        {}
    }};
    // --- Synthesized Logic End ---
}}
"#,
                safe_name, code
            );
            *code = wrapped_code;
        }

        Ok(result)
    }
}
