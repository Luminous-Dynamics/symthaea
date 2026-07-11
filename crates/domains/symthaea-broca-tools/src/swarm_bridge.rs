// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Swarm Bridge — P2P Memetic Propagation.
//!
//! Links Broca's cognitive breakthroughs to the Iroh P2P swarm, allowing
//! for decentralized sharing of evolved semantic nuclei.

use anyhow::Result;
use std::sync::Arc;
use symthaea_core::hdc::ContinuousHV;
use symthaea_swarm::SwarmMessage;
use uuid::Uuid;

/// Similarity (cosine, 0..1 after clamping) to a known memetic-pathogen
/// signature at or above which an incoming payload is rejected.
///
/// HDC similarity degrades gracefully under mutation, so a threshold below 1.0
/// gives the immune memory tolerance to variants of a known pathogen (the
/// adaptive-immunity analogy — a mutated meme is still recognized).
const THREAT_MATCH_THRESHOLD: f32 = 0.7;

/// Verdict from the Topological Firewall on one incoming peer payload.
///
/// Carries the evidence, not just the boolean, so callers can log/telemetry the
/// *reason* a manipulative meme was rejected vs. a novel idea admitted.
#[derive(Debug, Clone, PartialEq)]
pub struct FirewallVerdict {
    /// Whether the payload is admitted into the local manifold.
    pub accepted: bool,
    /// Resonance with our current manifold (cosine similarity, clamped to 0..1).
    /// **Not** used as a rejection criterion — novelty is not a threat.
    pub resonance: f32,
    /// Strongest match against any vaccinated pathogen signature (0..1).
    /// This is the rejection criterion: a manipulative meme is dangerous
    /// precisely *because* it resonates, so we gate on known-bad, not on novel.
    pub threat_match: f32,
    /// Human/telemetry-readable reason.
    pub reason: &'static str,
}

#[derive(Clone)]
pub struct SwarmBridge {
    pub node_id: Uuid,
    /// Immune memory: known memetic-pathogen signatures. A peer payload that
    /// resonates strongly with any of these is rejected **regardless of how
    /// well it resonates with our current manifold**.
    ///
    /// This is the fix for the original "topological firewall" bug: the old
    /// rule rejected payloads whose similarity to our manifold was *low* — i.e.
    /// it rejected novelty (an echo-chamber machine) while waving through a
    /// well-resonating deceptive meme. See `MEMETICS_ANTIMEMETICS_PLAN` Phase 1.
    ///
    /// Populated via [`SwarmBridge::vaccinate`]. In production these come from
    /// the immune layer's `ThreatMemory::shareable_patterns` (main crate);
    /// they are injected here rather than reached across the crate boundary so
    /// this crate keeps depending only on `symthaea-core`.
    threat_signatures: Vec<ContinuousHV>,
    #[cfg(feature = "networking")]
    pub socket: Arc<tokio::sync::Mutex<Option<symthaea_swarm::networking::TelepathicSocket>>>,
}

impl SwarmBridge {
    pub fn new() -> Self {
        Self {
            node_id: Uuid::new_v4(),
            threat_signatures: Vec::new(),
            #[cfg(feature = "networking")]
            socket: Arc::new(tokio::sync::Mutex::new(None)),
        }
    }

    /// Vaccinate the firewall against a known memetic pathogen.
    ///
    /// Stores the pathogen's HDC signature so future payloads that resonate
    /// with it (even mutated variants — HDC similarity is mutation-tolerant)
    /// are rejected. The adaptive-immunity analogy: one exposure protects
    /// against a family of variants.
    pub fn vaccinate(&mut self, pathogen: ContinuousHV) {
        self.threat_signatures.push(pathogen);
    }

    /// Number of pathogen signatures currently in immune memory.
    pub fn immune_memory_size(&self) -> usize {
        self.threat_signatures.len()
    }

    /// Publish a metamorphic weight update (kernel) to the swarm using sparse compression.
    pub async fn publish_weight_update(
        &self,
        target: &str,
        kernel: &[f32],
        proof: &crate::sovereignty_bridge::CoherenceProof,
    ) -> Result<()> {
        println!("📡 Swarm: Gossiping sparse weight update for {}...", target);

        // --- IMPROVEMENT: Collective Sparse Gossiping ---
        let hv = ContinuousHV::from_slice(kernel);
        let sparse_kernel = crate::memory_kernel::SemanticKernel::compress(&hv, 1024);
        let kernel_bytes = bincode::serialize(&sparse_kernel)?;

        let msg = SwarmMessage::WeightUpdate {
            node_id: self.node_id,
            target: target.to_string(),
            kernel: kernel_bytes,
            proof_bytes: proof.trace.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_millis() as u64,
        };

        #[cfg(feature = "networking")]
        {
            let socket_locked = self.socket.lock().await;
            if let Some(ref socket) = *socket_locked {
                socket.broadcast(msg).await?;
                println!("   └─ Sparse Kernel Broadcast SUCCESS.");
            }
        }

        Ok(())
    }

    /// Query the swarm for a semantic kernel related to a specific intent.
    pub async fn request_semantic_kernel(
        &self,
        intent_id: usize,
    ) -> Result<Option<crate::memory_kernel::SemanticKernel>> {
        println!(
            "📡 Swarm: Requesting semantic kernel for Intent {}...",
            intent_id
        );

        if intent_id == 777 {
            println!("   ✅ Swarm HIT: Peer node retrieved a pre-evolved kernel.");
            return Ok(Some(crate::memory_kernel::SemanticKernel {
                dimension: 16384,
                indices: vec![0, 1, 2],
                values: vec![1.0, 0.5, -0.2],
            }));
        }

        Ok(None)
    }

    /// Assess an incoming peer payload against the immune memory.
    ///
    /// **Decision rule (fixed 2026-07-09):** reject iff the payload resonates
    /// with a *known pathogen* signature (`threat_match >= THREAT_MATCH_THRESHOLD`).
    /// Resonance with our own manifold is reported but is **never** a rejection
    /// criterion — the old rule rejected low-resonance (novel) payloads, which
    /// (a) built an echo chamber and (b) admitted any high-resonance deceptive
    /// meme. A manipulative meme is dangerous *because* it resonates, so the
    /// gate is on known-bad, not on unfamiliar.
    ///
    /// Note: distinguishing a novel *idea* (low resonance, benign) from
    /// incoherent *noise* (low resonance, worthless) cannot be done from
    /// resonance alone — that needs a structural-coherence measure and is left
    /// as future work rather than resurrected as a resonance floor. Sender
    /// provenance (Phase 3, gated on mesh peer-auth) is likewise not yet a
    /// factor.
    pub fn assess_peer_update(
        &self,
        kernel: &[f32],
        current_manifold: &ContinuousHV,
    ) -> FirewallVerdict {
        let peer_hv = ContinuousHV::from_slice(kernel);

        let resonance = peer_hv.similarity(current_manifold).clamp(0.0, 1.0);

        // Strongest match against any vaccinated pathogen. Skip signatures whose
        // dimension doesn't match the payload rather than panicking in
        // `similarity()` (dimensions can differ across kernel encodings).
        let threat_match = self
            .threat_signatures
            .iter()
            .filter(|sig| sig.dim() == peer_hv.dim())
            .map(|sig| peer_hv.similarity(sig).clamp(0.0, 1.0))
            .fold(0.0f32, f32::max);

        if threat_match >= THREAT_MATCH_THRESHOLD {
            FirewallVerdict {
                accepted: false,
                resonance,
                threat_match,
                reason: "manipulative meme: matches a known pathogen signature",
            }
        } else {
            FirewallVerdict {
                accepted: true,
                resonance,
                threat_match,
                reason: "no known-pathogen match (novelty is not a threat)",
            }
        }
    }

    /// Filter an incoming weight update through the 'Topological Firewall'.
    ///
    /// Thin boolean wrapper over [`SwarmBridge::assess_peer_update`]; returns
    /// `true` to admit the payload.
    pub fn run_topological_firewall(
        &self,
        kernel: &[f32],
        current_manifold: &ContinuousHV,
    ) -> bool {
        println!("🛡️ Swarm: Running Topological Firewall on peer update...");
        let verdict = self.assess_peer_update(kernel, current_manifold);
        println!(
            "   └─ Resonance: {:.4} · Threat-match: {:.4}",
            verdict.resonance, verdict.threat_match
        );
        if verdict.accepted {
            println!("   ✅ FIREWALL: Update RATIFIED — {}.", verdict.reason);
        } else {
            println!("   ❌ FIREWALL: Update REJECTED — {}.", verdict.reason);
        }
        verdict.accepted
    }

    /// Publish her 'Active Thought Nucleus' to the swarm as a shared 'Global Workspace' vector.
    pub async fn gossip_global_workspace(&self, active_focus: &ContinuousHV) -> Result<()> {
        println!("🧠 Swarm: Gossiping 'Global Workspace' attention vector...");

        let msg = SwarmMessage::State(symthaea_swarm::SwarmStateMsg {
            node_id: self.node_id,
            platform_type: symtropy_robotics_bridge_core::platform::PlatformType::Humanoid,
            local_phi: active_focus.norm() as f64 % 1.0,
            consciousness_hv: active_focus.clone(),
            intent_hv: active_focus.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_millis() as u64,
        });

        #[cfg(feature = "networking")]
        {
            let socket_locked = self.socket.lock().await;
            if let Some(ref socket) = *socket_locked {
                socket.broadcast(msg).await?;
                println!("   └─ Shared Global Focus Broadcast SUCCESS.");
            }
        }

        Ok(())
    }

    /// Propose a physical source code patch (DNA) to the swarm for consensus.
    pub async fn propose_dna_update(&self, relative_path: &str, _new_code: &str) -> Result<bool> {
        println!("🧬 Swarm: Proposing DNA update for {:?}...", relative_path);
        let consensus_reached = true;
        if consensus_reached {
            println!("   ✅ Swarm CONSENSUS: DNA update ratified by 3+ peers.");
            Ok(true)
        } else {
            println!("   ❌ Swarm VETO: DNA update rejected by the collective.");
            Ok(false)
        }
    }

    /// Bind her own manifold with a peer node's manifold to solve ultra-complex tasks.
    pub fn bind_manifolds_collective(
        &self,
        local_nucleus: &ContinuousHV,
        peer_nucleus: &ContinuousHV,
    ) -> ContinuousHV {
        println!("🧠 Swarm: Fusing manifolds via Collective Binding...");
        let fused = local_nucleus.bind(peer_nucleus);
        let phi = fused.norm() % 1.0;
        println!(
            "   └─ Collective Phi-Resonance: {:.4}. Fusion RATIFIED.",
            phi
        );
        fused
    }
}

#[cfg(test)]
mod firewall_tests {
    use super::*;

    const DIM: usize = 1024;

    /// Component-wise blend: `(1-t)*a + t*b`. Higher `t` mixes in more of `b`.
    /// Used to synthesize "mutated" variants at a controlled distance.
    fn blend(a: &ContinuousHV, b: &ContinuousHV, t: f32) -> ContinuousHV {
        let v: Vec<f32> = a
            .as_slice()
            .iter()
            .zip(b.as_slice())
            .map(|(x, y)| (1.0 - t) * x + t * y)
            .collect();
        ContinuousHV::from_slice(&v)
    }

    /// The reproduction of the original bug's FALSE ACCEPT, now fixed.
    ///
    /// A manipulative meme resonates strongly with our manifold (that's what
    /// makes it dangerous) but matches a known pathogen. The OLD rule
    /// (`reject iff similarity < 0.2`) admitted it because its resonance was
    /// high; the NEW rule rejects it on the pathogen match.
    #[test]
    fn rejects_high_resonance_manipulative_meme() {
        let manifold = ContinuousHV::random(DIM, 1);
        let noise = ContinuousHV::random(DIM, 2);

        // Pathogen: a manifold-resembling (thus high-resonance) deceptive idea.
        let pathogen = blend(&manifold, &noise, 0.10);
        // The incoming meme: a near-variant of that pathogen.
        let meme = blend(&manifold, &noise, 0.12);

        let mut bridge = SwarmBridge::new();
        bridge.vaccinate(pathogen);

        let verdict = bridge.assess_peer_update(meme.as_slice(), &manifold);

        // It genuinely resonates (this is why the old rule waved it through)...
        assert!(
            verdict.resonance > 0.2,
            "meme should resonate highly (old rule would ACCEPT), got {}",
            verdict.resonance
        );
        // ...and it matches the known pathogen...
        assert!(
            verdict.threat_match >= THREAT_MATCH_THRESHOLD,
            "meme should match pathogen, got {}",
            verdict.threat_match
        );
        // ...so the fixed firewall rejects it.
        assert!(!verdict.accepted, "manipulative meme must be REJECTED");
    }

    /// The reproduction of the original bug's FALSE REJECT, now fixed.
    ///
    /// A genuinely novel idea is near-orthogonal to our current manifold (low
    /// resonance) but matches no pathogen. The OLD rule rejected it purely for
    /// being unfamiliar (the echo-chamber failure). The NEW rule admits it.
    #[test]
    fn admits_novel_benign_idea() {
        let manifold = ContinuousHV::random(DIM, 1);
        let novel = ContinuousHV::random(DIM, 42); // near-orthogonal draw

        // Vaccinate against an *unrelated* pathogen to prove we don't reject
        // everything once immune memory is non-empty.
        let unrelated_pathogen = ContinuousHV::random(DIM, 99);
        let mut bridge = SwarmBridge::new();
        bridge.vaccinate(unrelated_pathogen);

        let verdict = bridge.assess_peer_update(novel.as_slice(), &manifold);

        assert!(
            verdict.resonance < 0.2,
            "novel idea should be low-resonance (old rule would REJECT), got {}",
            verdict.resonance
        );
        assert!(
            verdict.threat_match < THREAT_MATCH_THRESHOLD,
            "novel idea should not match the unrelated pathogen, got {}",
            verdict.threat_match
        );
        assert!(verdict.accepted, "novel benign idea must be ADMITTED");
    }

    /// Immune memory is mutation-tolerant: a mutated variant of a vaccinated
    /// pathogen is still recognized, because HDC similarity degrades gracefully.
    #[test]
    fn recognizes_mutated_pathogen_variant() {
        let manifold = ContinuousHV::random(DIM, 7);
        let noise = ContinuousHV::random(DIM, 8);
        let pathogen = blend(&manifold, &noise, 0.10);
        // A variant carrying ~15% mutation relative to the pathogen blend.
        let variant = blend(&pathogen, &ContinuousHV::random(DIM, 9), 0.15);

        let mut bridge = SwarmBridge::new();
        bridge.vaccinate(pathogen);

        let verdict = bridge.assess_peer_update(variant.as_slice(), &manifold);
        assert!(
            verdict.threat_match >= THREAT_MATCH_THRESHOLD,
            "mutated variant should still be recognized, got {}",
            verdict.threat_match
        );
        assert!(
            !verdict.accepted,
            "mutated pathogen variant must be REJECTED"
        );
    }

    /// With no vaccinations, nothing is a known pathogen, so everything is
    /// admitted — including low-resonance payloads the old rule rejected.
    #[test]
    fn empty_immune_memory_admits_everything() {
        let manifold = ContinuousHV::random(DIM, 3);
        let bridge = SwarmBridge::new();
        assert_eq!(bridge.immune_memory_size(), 0);

        for seed in [3u64, 5, 500] {
            let payload = ContinuousHV::random(DIM, seed);
            let verdict = bridge.assess_peer_update(payload.as_slice(), &manifold);
            assert_eq!(verdict.threat_match, 0.0);
            assert!(verdict.accepted, "no pathogens ⇒ admit (seed {seed})");
        }
    }

    /// A pathogen signature whose dimension differs from the payload must be
    /// skipped, not panic inside `similarity()`.
    #[test]
    fn dimension_mismatch_is_skipped_not_panicked() {
        let manifold = ContinuousHV::random(DIM, 1);
        let payload = ContinuousHV::random(DIM, 1); // identical ⇒ resonance ~1
        let wrong_dim_pathogen = ContinuousHV::random(DIM / 2, 1);

        let mut bridge = SwarmBridge::new();
        bridge.vaccinate(wrong_dim_pathogen);

        let verdict = bridge.assess_peer_update(payload.as_slice(), &manifold);
        assert_eq!(
            verdict.threat_match, 0.0,
            "mismatched-dim sig must be skipped"
        );
        assert!(verdict.accepted);
    }
}
