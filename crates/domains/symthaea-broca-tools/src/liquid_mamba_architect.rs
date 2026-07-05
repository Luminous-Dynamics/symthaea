// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Liquid-Mamba Architect — self-modifying toolkit wrapper
//!
//! Composes a plain [`LiquidMambaGenerator`] (core generation, lives in
//! `symthaea-broca`) with the "self-authoring architect" bridges that used to
//! be fields directly on the generator: physical safety verification,
//! formal proof verification, P2P memetic propagation, self-authoring
//! codebase edits, self-evolution audit, and narrative-sovereignty proofs.
//!
//! Those bridges were previously fields on `LiquidMambaGenerator` itself, but
//! `symthaea-broca` cannot depend on `symthaea-broca-tools` (that's the
//! reverse of the crate's established, correct toolkit→core dependency
//! direction — see `SYMTHAEA_IMPROVEMENT_PLAN_2026-07.md` Phase 2). This type
//! is the toolkit-side home for that capability set instead: it holds the
//! generator via composition (has-a, not is-a) and re-exposes the same
//! methods callers used before, so the split is invisible to the "prove
//! coherence" / "verify physical safety" call sites, just relocated to the
//! crate that can actually own it.
//!
//! `substrate_rewriter`, `swarm_bridge`, `codebase_bridge`, `formal_bridge`,
//! and `cognitive_ledger` are exposed as plain public fields (their only
//! consumers were direct field access, e.g. `generator.cognitive_ledger.commit(...)`
//! — no wrapping method existed on the generator for these, so none is added
//! here either).

#[cfg(feature = "mamba-cpu")]
use symthaea_broca::ThoughtChunkSequence;
#[cfg(feature = "mamba-cpu")]
use symthaea_broca::liquid_mamba::LiquidMambaGenerator;
#[cfg(feature = "mamba-cpu")]
use symthaea_core::genesis::GenesisSeed;
#[cfg(feature = "mamba-cpu")]
use symthaea_core::hdc::ContinuousHV;

#[cfg(feature = "mamba-cpu")]
use crate::codebase_bridge::CodebaseBridge;
#[cfg(feature = "mamba-cpu")]
use crate::cognitive_ledger::CognitiveLedger;
#[cfg(feature = "mamba-cpu")]
use crate::formal_bridge::FormalBridge;
#[cfg(feature = "mamba-cpu")]
use crate::simulation_bridge::PhysicalVerifier;
#[cfg(feature = "mamba-cpu")]
use crate::sovereignty_bridge::{CoherenceProof, SovereigntyBridge};
#[cfg(feature = "mamba-cpu")]
use crate::substrate_rewriter::SubstrateRewriter;
#[cfg(feature = "mamba-cpu")]
use crate::swarm_bridge::SwarmBridge;

/// Wraps a [`LiquidMambaGenerator`] with the self-authoring architect toolkit.
#[cfg(feature = "mamba-cpu")]
pub struct LiquidMambaArchitect {
    /// The wrapped core generator. Public — most callers already had direct
    /// access to the generator's own API and should keep using it unchanged
    /// (`architect.generator.commit_weights()`, etc.).
    pub generator: LiquidMambaGenerator,
    /// Physical safety verification via SimBridge.
    pub physical_verifier: PhysicalVerifier,
    /// Formal proof verification (Z3/Lean).
    pub formal_bridge: FormalBridge,
    /// P2P memetic propagation via Iroh.
    pub swarm_bridge: SwarmBridge,
    /// Self-authoring codebase improvements.
    pub codebase_bridge: CodebaseBridge,
    /// Self-evolution audit trail.
    pub cognitive_ledger: CognitiveLedger,
    /// Proofs of coherence and narrative sovereignty (Mycelix ZKP).
    pub sovereignty_bridge: SovereigntyBridge,
    /// Direct source code modification.
    pub substrate_rewriter: SubstrateRewriter,
}

#[cfg(feature = "mamba-cpu")]
impl LiquidMambaArchitect {
    /// Wrap an already-constructed generator with the architect toolkit.
    ///
    /// `hdc_dim` and `genesis` are needed only to construct the toolkit
    /// bridges (matching what the generator's own construction used to
    /// need) — callers already have both in scope wherever they build a
    /// `LiquidMambaGenerator`.
    pub fn new(generator: LiquidMambaGenerator, hdc_dim: usize, genesis: &GenesisSeed) -> Self {
        Self {
            generator,
            physical_verifier: PhysicalVerifier::new(hdc_dim),
            formal_bridge: FormalBridge::new(),
            swarm_bridge: SwarmBridge::new(),
            codebase_bridge: CodebaseBridge::new("."),
            cognitive_ledger: CognitiveLedger::new(".").expect("cognitive ledger init"),
            sovereignty_bridge: SovereigntyBridge::new(genesis.timeline_id()),
            substrate_rewriter: SubstrateRewriter::new("."),
        }
    }

    /// Verify the physical safety of a synthesized tool logic using SimBridge.
    pub fn verify_physical_safety(
        &self,
        name: &str,
        intent_nucleus: &ContinuousHV,
    ) -> anyhow::Result<ContinuousHV> {
        self.physical_verifier
            .verify_tool_impact(name, intent_nucleus)
    }

    /// Generate a cryptographic 'Proof of Reason' for a completed monologue.
    pub fn prove_narrative_sovereignty(
        &self,
        sequence: &ThoughtChunkSequence,
    ) -> anyhow::Result<CoherenceProof> {
        let nucleus = self.generator.recursive_fold(sequence);

        // Extract the full trajectory of reasoning metrics
        let coherence_history: Vec<f32> = sequence
            .chunks
            .iter()
            .map(|c| c.confidence) // Per-chunk coherence snapshot
            .collect();

        let gap_history: Vec<f32> = sequence
            .chunks
            .iter()
            .map(|c| c.spectral_entropy) // True time-series telemetry
            .collect();

        self.sovereignty_bridge
            .prove_coherence(&coherence_history, &gap_history, &nucleus)
    }
}
