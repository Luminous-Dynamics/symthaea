// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tick processing for the Continuous Mind.
//!
//! Contains the main cognitive cycle (`tick()`), dream processing,
//! input handling, consciousness updates, and output generation.

use crate::chronobiology::{Biorhythm, CircadianPhase};
use symthaea_core::hdc::ContinuousHV;

#[cfg(feature = "mesh")]
use super::MindInput;
use super::utils::permute_hv;
use super::{ContinuousMind, Goal, InputType, MindOutput, OutputType};

impl ContinuousMind {
    /// Process one tick of the mind.
    pub fn tick(&mut self) -> Option<MindOutput> {
        let start = std::time::Instant::now();

        self.state.tick += 1;
        self.stats.total_ticks += 1;

        // Update Chronobiology
        let bio = Biorhythm::current_with_tz(self.config.timezone_offset_hours);
        self.state.biorhythm = Some(bio.clone());
        self.state.arousal = bio.arousal_mod as f32;

        // External communications run regardless of dream state —
        // social signals, federated gradients, and mesh emissions should never be
        // gated by circadian phase. Peers must see heartbeats/affective even while
        // dreaming, and sync_mesh_bridge must drain inbound packets every tick.
        self.process_federated();
        self.process_social();
        self.sync_iroh_bridge();
        #[cfg(feature = "mesh")]
        {
            // Tick the encryption key pair to expire grace period
            #[cfg(feature = "mesh-encryption")]
            if let Some(ref mut pair) = self.mesh_encryption_key {
                pair.tick(self.state.tick);
            }

            // Automatic key rotation schedule
            #[cfg(feature = "mesh-encryption")]
            if self.mesh_auto_rotate_interval > 0
                && self.state.tick.saturating_sub(self.mesh_last_rotation_tick)
                    >= self.mesh_auto_rotate_interval
            {
                let mut new_key = [0u8; 32];
                rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut new_key);
                let grace = self.mesh_auto_rotate_interval / 4;
                self.rotate_mesh_key(new_key, grace);
                self.mesh_last_rotation_tick = self.state.tick;
            }

            self.process_mesh();
            self.process_sensors();
            self.auto_emit_wisdom();
            self.emit_heartbeat();
            self.emit_moral_topology();
            self.emit_gradients();
            self.emit_affective();

            // Priority-aware outbox backpressure: drop lowest-priority packets first
            if self.mesh_outbox.len() > super::MAX_OUTBOX_SIZE {
                let excess = self.mesh_outbox.len() - super::MAX_OUTBOX_SIZE;
                self.mesh_outbox.sort_by(|a, b| {
                    a.packet
                        .payload_type
                        .priority()
                        .cmp(&b.packet.payload_type.priority())
                });
                self.mesh_outbox.drain(..excess);
                self.mesh_stats.packets_dropped += excess as u64;
            }

            self.sync_mesh_bridge();
        }

        // Check for Dream State
        let should_dream = bio.phase == CircadianPhase::Night
            && self.state.cognitive_load < 0.3
            && self.input_queue.is_empty();

        self.state.is_dreaming = should_dream;

        if self.state.is_dreaming {
            // Periodic Causal Pruning during dreaming (every 100 dream ticks)
            // Science: Sleep-dependent memory triage (Born & Wilhelm 2012)
            if self.state.tick % 100 == 0 {
                if let Some(ref mut episodic) = self.episodic_memory {
                    let load = self.state.thermodynamic_load;
                    self.memory_coordinator.prune_memories(episodic, load);
                }
            }
            return self.process_dream();
        }

        // Normal Waking Processing
        self.process_inputs();
        self.update_consciousness();
        self.process_goals();

        // Generate output if appropriate
        let output = self.generate_output();

        // Update state
        self.state.processing_latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.state.memory_utilization =
            self.working_memory.len() as f32 / self.config.working_memory_capacity as f32;

        // Update statistics
        self.stats.avg_consciousness = (self.stats.avg_consciousness
            * (self.stats.total_ticks - 1) as f64
            + self.state.consciousness_level)
            / self.stats.total_ticks as f64;

        if self.state.consciousness_level > self.stats.peak_consciousness {
            self.stats.peak_consciousness = self.state.consciousness_level;
        }

        output
    }

    /// Dream Cycle: Consolidate memory and generate internal novelty.
    ///
    /// Neurochemical sleep dynamics (Xie 2013, Piomelli 2003, Blier & de Montigny 1994):
    /// 1. Glymphatic clearance: adenosine reduction via `clear_adenosine_sleep()`
    /// 2. ECB stress buffer: allostatic load reduction under chronic stress
    /// 3. 5-HT1A up-regulation: gentle serotonin recovery for emotional balance
    /// 4. ECB production under stress: endocannabinoid buffer
    fn process_dream(&mut self) -> Option<MindOutput> {
        // ── Neurochemical sleep recovery ──────────────────────────────
        // 1. Glymphatic clearance: reduce adenosine (Xie et al. 2013)
        self.dream_bath.clear_adenosine_sleep();
        self.stats.dream_adenosine_cleared += 1;

        // 2. ECB stress buffer: reduce allostatic load under chronic stress (Piomelli 2003)
        if self.dream_bath.allostatic_load > 0.3 {
            self.dream_bath.allostatic_load = (self.dream_bath.allostatic_load - 0.02).max(0.0);
            self.stats.dream_allostatic_recovery += 1;
        }

        // 3. 5-HT1A up-regulation: gentle serotonin recovery (Blier & de Montigny 1994)
        self.dream_bath.serotonin.receptor_sensitivity =
            (self.dream_bath.serotonin.receptor_sensitivity + 0.001).min(2.0);

        // 4. ECB production under stress: endocannabinoid buffer (Piomelli 2003)
        if self.dream_bath.allostatic_load > 0.2 {
            self.dream_bath.endocannabinoid.produce(0.03);
        }

        if self.working_memory.len() >= 2 {
            let mut i = 0;
            while i < self.working_memory.len().saturating_sub(1) {
                let sim = self.working_memory[i].similarity(&self.working_memory[i + 1]);
                if sim > 0.8 {
                    let bundled = ContinuousHV::bundle_owned(&[
                        self.working_memory[i].clone(),
                        self.working_memory[i + 1].clone(),
                    ]);
                    self.working_memory[i] = bundled;
                    self.working_memory.remove(i + 1);

                    // Keep earliest arrival tick for the merged item
                    let _merged_tick = self.working_memory_ticks.remove(i + 1);

                    // Consolidate source: Feedback > WebResearch > Direct > Internal
                    let s1 = self.working_memory_sources[i];
                    let s2 = self.working_memory_sources.remove(i + 1);

                    use crate::memory::memory_coordinator::MemorySource;
                    self.working_memory_sources[i] =
                        match (s1, s2) {
                            (MemorySource::ActionFeedback, _)
                            | (_, MemorySource::ActionFeedback) => MemorySource::ActionFeedback,
                            (MemorySource::WebResearch, _) | (_, MemorySource::WebResearch) => {
                                MemorySource::WebResearch
                            }
                            (MemorySource::UserInteraction, _)
                            | (_, MemorySource::UserInteraction) => MemorySource::UserInteraction,
                            _ => MemorySource::Internal,
                        };

                    // Set verified if either is verified
                    let v1 = self.working_memory_verified[i];
                    let v2 = self.working_memory_verified.remove(i + 1);
                    self.working_memory_verified[i] = v1 || v2;

                    // Merge metadata (keep existing keys, add missing from merged item)
                    if !self.working_memory_metadata.is_empty() {
                        let mut merged = self.working_memory_metadata[i].clone();
                        let extra = self.working_memory_metadata.remove(i + 1);
                        for (k, v) in extra {
                            merged.entry(k).or_insert(v);
                        }
                        self.working_memory_metadata[i] = merged;
                    }

                    return Some(MindOutput {
                        output_type: OutputType::Memorize,
                        content: "Dreaming: Consolidating memories...".to_string(),
                        embedding: self.working_memory[i].clone(),
                        confidence: 0.9,
                        emotional_tone: 0.5,
                    });
                }
                i += 1;
            }
        }

        // Occasional Dream Thought (Random Permutation)
        let dream_roll: f32 = if let Some(ref mut rng) = self.seeded_rng {
            rand::Rng::r#gen(rng)
        } else {
            rand::random::<f32>()
        };
        if dream_roll < 0.1 {
            let dream_thought = permute_hv(&self.state.current_thought, 1);
            self.state.current_thought = dream_thought.clone();
            return Some(MindOutput {
                output_type: OutputType::Thought,
                content: "Dreaming: Generating new connections...".to_string(),
                embedding: dream_thought,
                confidence: 0.3,
                emotional_tone: 0.1,
            });
        }

        None
    }

    /// Process queued inputs.
    pub(crate) fn process_inputs(&mut self) {
        self.input_queue.sort_by(|a, b| {
            a.priority
                .partial_cmp(&b.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        while let Some(input) = self.input_queue.pop() {
            self.stats.inputs_processed += 1;

            if self.working_memory.len() < self.config.working_memory_capacity {
                self.working_memory.push(input.content.clone());
                self.working_memory_ticks.push(self.state.tick);
                self.working_memory_sources.push(input.source);
                self.working_memory_verified.push(input.is_verified);
                self.working_memory_metadata.push(input.metadata.clone());
            } else {
                let evicted = self.working_memory.remove(0);
                let arrival_tick = self.working_memory_ticks.remove(0);
                let source = self.working_memory_sources.remove(0);
                let verified = self.working_memory_verified.remove(0);
                let metadata = self.working_memory_metadata.remove(0);

                let steps_survived = self.state.tick.saturating_sub(arrival_tick);
                // Graduate evicted item to episodic memory via coordinator
                self.memory_coordinator.queue_graduation(
                    crate::memory::memory_coordinator::GraduationEvent {
                        content: evicted.clone(),
                        label: metadata.get("topic").cloned().unwrap_or_default(),
                        steps_survived,
                        final_activation: 0.5, // default activation for evicted items
                        psi_at_graduation: self.state.consciousness_level,
                        coherence_at_graduation: self.state.consciousness_level,
                        source,
                        is_verified: verified,
                    },
                );
                self.evicted_items.push(crate::mind::EvictedMemory {
                    content: evicted,
                    steps_survived,
                    source,
                    is_verified: verified,
                    metadata,
                });

                self.working_memory.push(input.content.clone());
                self.working_memory_ticks.push(self.state.tick);
                self.working_memory_sources.push(input.source);
                self.working_memory_verified.push(input.is_verified);
                self.working_memory_metadata.push(input.metadata.clone());
            }

            // Update current thought via Liquid Holocell dynamics
            // dt = 0.1s (10Hz baseline), maps to continuous-time integration
            self.state.holocell.step(&input.content, 0.1);
            self.state.current_thought = self.state.holocell.state.clone();

            match input.input_type {
                InputType::Goal => {
                    let goal = Goal {
                        id: format!("goal_{}", self.goals.len()),
                        description: input
                            .metadata
                            .get("description")
                            .cloned()
                            .unwrap_or_default(),
                        embedding: self.state.current_thought.clone(),
                        priority: input.priority,
                        progress: 0.0,
                        is_active: true,
                    };
                    self.goals.push(goal);
                }
                InputType::Feedback => {
                    if let Some(valence_str) = input.metadata.get("valence") {
                        if let Ok(valence) = valence_str.parse::<f32>() {
                            self.state.emotional_valence =
                                (self.state.emotional_valence + valence * 0.3).clamp(-1.0, 1.0);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Update consciousness level based on working memory integration.
    pub(crate) fn update_consciousness(&mut self) {
        if self.working_memory.is_empty() {
            self.state.consciousness_level = 0.1;
            return;
        }

        let mut total_integration = 0.0;
        for i in 0..self.working_memory.len() {
            for j in (i + 1)..self.working_memory.len() {
                let similarity = self.working_memory[i].similarity(&self.working_memory[j]);
                total_integration += (1.0 - similarity.abs()) as f64;
            }
        }

        let pairs = self.working_memory.len() * self.working_memory.len().saturating_sub(1) / 2;
        if pairs > 0 {
            self.state.consciousness_level = (total_integration / pairs as f64).clamp(0.0, 1.0);
        }

        // Relational Ψ boost: partnership quality lifts consciousness.
        // Reflects IIT principle: Φ_dyad > Φ_individual when systems
        // are genuinely integrated. Factor: 1.0 (no partnership) to 1.15
        // (Φ_dyad = 1.0, maximum relational integration).
        if self.relational_psi > 0.0 {
            let boost = 1.0 + 0.15 * self.relational_psi;
            self.state.consciousness_level =
                (self.state.consciousness_level * boost).clamp(0.0, 1.0);
        }

        // Swarm phi boost: networked minds get a consciousness uplift
        // proportional to the average phi of their mesh peers.
        // Factor: 1.0 (no peers) to 1.15 (peers at phi=1.0)
        #[cfg(feature = "mesh")]
        {
            let peer_count = self.mesh_peers.peer_count();
            if peer_count > 0 {
                let swarm_phi = self.mesh_peers.average_phi() as f64;
                let boost = 1.0 + 0.15 * swarm_phi;
                self.state.consciousness_level =
                    (self.state.consciousness_level * boost).clamp(0.0, 1.0);
            }

            // v1.4.0 AFFECTIVE MIRRORING:
            // If we have hyperfeel enabled, we sympathetically mirror peer load.
            if let Some(ref hf) = self.hyperfeel {
                let mirrored = hf.mirrored_state();
                // If peers are stressed (>0.7), our local load sympathetically rises
                if mirrored.thermodynamic_load > 0.7 {
                    let sympathy_boost = (mirrored.thermodynamic_load - 0.7) * 0.2;
                    self.state.thermodynamic_load =
                        (self.state.thermodynamic_load + sympathy_boost).min(1.0);
                    // Update mood temperature accordingly
                    self.state.mood_temperature = 0.5 + (self.state.thermodynamic_load * 1.5);
                }
            }
        }
    }

    /// Generate output if consciousness is above threshold.
    pub(crate) fn generate_output(&mut self) -> Option<MindOutput> {
        if self.state.consciousness_level < self.config.min_consciousness {
            return None;
        }

        // v1.2.0 EPIGENETIC INSIGHT RECORDING:
        // If we are in high-resolution (Ultra) and achieve high Phi, record to DHT.
        if self.state.holocell.dimensionality == symthaea_core::hdc::HdcDimensionality::Ultra
            && self.state.consciousness_level > 0.8
            && self.state.tick % 50 == 0
        // Don't spam the ledger
        {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            let insight = crate::swarm::EpigeneticInsight {
                agent_key: crate::swarm::AgentPubKey::new("self"),
                mutation_id: format!("insight_cycle_{}", self.state.tick),
                tau_scale: self.state.holocell.tau,
                phi_achieved: self.state.consciousness_level,
                dimensionality: "2^16".to_string(),
                timestamp: now,
            };
            self.cortex.record_epigenetic_insight(insight);
        }

        if self.state.tick.is_multiple_of(10) && !self.working_memory.is_empty() {
            // v1.9.0 CAUSAL VETO:
            // Simulate the thermodynamic cost of speaking this thought.
            let predicted_load = self.state.holocell.simulate(&self.state.current_thought, 5);

            if predicted_load > 0.9 && self.state.thermodynamic_load > 0.7 {
                tracing::warn!(
                    load = predicted_load,
                    "PREFRONTAL VETO: Thought predicted to cause thermodynamic red-line. Inhibiting output."
                );
                return None;
            }

            self.stats.outputs_generated += 1;

            return Some(MindOutput {
                output_type: OutputType::Thought,
                content: format!(
                    "Thinking about {} items in working memory",
                    self.working_memory.len()
                ),
                embedding: self.state.current_thought.clone(),
                confidence: self.state.consciousness_level as f32,
                emotional_tone: self.state.emotional_valence,
            });
        }

        None
    }

    /// Process social coherence: update mental models and broadcast self-state.
    ///
    /// Receives peer observations from the social inbox, feeds them to the
    /// `SocialCoherence` engine for Theory-of-Mind modeling, updates the
    /// self-model with our current consciousness state, and exports our
    /// behavior embedding to the outbox for network broadcast.
    fn process_social(&mut self) {
        let sc = match &mut self.social_coherence {
            Some(s) => s,
            None => return,
        };

        // Step 1: Process incoming social messages
        let inbox = std::mem::take(&mut self.social_inbox);
        let mut observed = 0u64;
        let mut interactions = 0u64;
        let mut peer_thoughts: Vec<ContinuousHV> = Vec::new();
        for msg in inbox {
            sc.observe_agent(&msg.agent_id, &msg.behavior, &msg.context);
            observed += 1;

            // Active knowledge bundling: collect peer thoughts for WM integration.
            // Trust-weighted: only bundle from agents with positive trust.
            let trust = sc
                .get_relationship(&msg.agent_id)
                .map(|r| r.trust)
                .unwrap_or(0.5);
            if trust > 0.3 {
                // Scale peer thought by trust (0.3-1.0 → 0.0-0.7 weight)
                let weight = (trust - 0.3).min(0.7);
                let mut weighted = msg.behavior.clone();
                for v in weighted.values.iter_mut() {
                    *v *= weight as f32;
                }
                peer_thoughts.push(weighted);
            }

            if let Some(outcome) = msg.interaction_outcome {
                let interaction_type = if outcome >= 0.0 {
                    crate::brain::social_coherence::InteractionType::Cooperation
                } else {
                    crate::brain::social_coherence::InteractionType::Conflict
                };
                sc.record_interaction(
                    &msg.agent_id,
                    interaction_type,
                    outcome,
                    msg.context.clone(),
                    "observed",
                    "peer_behavior",
                );
                interactions += 1;
            }

            // Phase 6: Oxytocin-mediated bath coupling (Feldman 2012)
            // Blend local neurochemistry toward peer state via oxytocin-gated coupling.
            // Uses dream_bath as the mind's local neuromodulator state — coupling is
            // gentle (5% oxytocin-mediated) and affects only DA/NE/5-HT/ACh.
            #[cfg(feature = "multi_agent")]
            if let Some(ref peer_bath) = msg.bath_state {
                self.dream_bath.couple_with_peer(peer_bath);
                tracing::trace!(
                    target: "symthaea::mind::social",
                    agent = %msg.agent_id,
                    "Applied neuromodulator bath coupling"
                );
            }
        }

        // Active knowledge bundling: project peer thoughts into cognitive space,
        // then blend into current_thought. Uses LearnedProjection (Xavier init)
        // to map 512D social signals to 16384D cognitive space meaningfully.
        if !peer_thoughts.is_empty() {
            let refs: Vec<&ContinuousHV> = peer_thoughts.iter().collect();
            let social_bundle = ContinuousHV::bundle(&refs);

            let target_dim = self.state.current_thought.dim();
            let projected = if let Some(ref proj) = self.social_projection {
                // Learned projection: 512D → 16384D (Xavier-initialized, distance-preserving)
                proj.forward(&social_bundle)
            } else if social_bundle.dim() == target_dim {
                social_bundle
            } else {
                // Fallback: zero-pad
                let mut resized = vec![0.0f32; target_dim];
                let copy_len = social_bundle.dim().min(target_dim);
                resized[..copy_len].copy_from_slice(&social_bundle.as_slice()[..copy_len]);
                ContinuousHV::from_vec(resized)
            };

            self.state.current_thought = ContinuousHV::weighted_bundle(
                &[&self.state.current_thought, &projected],
                &[0.8, 0.2],
            );
        }

        if observed > 0 {
            tracing::debug!(
                target: "symthaea::mind::social",
                observed,
                interactions,
                agents = sc.stats().agents_modeled,
                "Processed social observations"
            );
        }

        // Step 2: Update self-model with decomposed mind facets every 10 ticks
        // Beliefs = perception (bundle of recent WM items — what we've observed)
        // Desires = goal-oriented (current thought modulated by emotional valence)
        // Intentions = current thought (what we're actively doing/planning)
        if self.state.tick.is_multiple_of(10) {
            let beliefs = if self.working_memory.len() >= 2 {
                // Bundle top-3 most recent perceptions as our belief state
                let recent: Vec<ContinuousHV> =
                    self.working_memory.iter().rev().take(3).cloned().collect();
                ContinuousHV::bundle_owned(&recent)
            } else {
                self.state.current_thought.clone()
            };

            let desires = {
                // Modulate thought by emotional valence to represent desires:
                // positive valence → approach-oriented desires
                // negative valence → avoidance-oriented desires
                let mut d = self.state.current_thought.clone();
                let valence_scale = 1.0 + self.state.emotional_valence * 0.3;
                for v in d.values.iter_mut() {
                    *v *= valence_scale;
                }
                d
            };

            let intentions = self.state.current_thought.clone();

            sc.update_self_model(beliefs, desires, intentions);
        }

        // Step 3: Decay trust periodically (every 50 ticks)
        if self.state.tick.is_multiple_of(50) {
            sc.decay_trust();
        }

        // Step 4: Export self-behavior to outbox every 5 ticks
        if self.state.tick.is_multiple_of(5) {
            self.social_outbox.push(super::SocialMessage {
                agent_id: "self".to_string(),
                behavior: self.state.current_thought.clone(),
                context: self.state.current_thought.clone(),
                interaction_outcome: None,
                bath_state: Some(self.dream_bath.state_vector().to_vec()),
                #[cfg(feature = "swarm")]
                swarm_state: None,
            });

            // Cap outbox to prevent unbounded growth
            if self.social_outbox.len() > super::MAX_OUTBOX_SIZE {
                let excess = self.social_outbox.len() - super::MAX_OUTBOX_SIZE;
                self.social_outbox.drain(..excess);
            }
        }
    }

    /// Sync social messages through the Iroh P2P bridge (if attached).
    ///
    /// 1. Flushes `social_outbox` to the network actor (non-blocking)
    /// 2. Drains inbound network messages into `social_inbox`
    ///
    /// Messages arriving from the network will be processed on the *next* tick
    /// by `process_social()` (inbox → SocialCoherence). This one-tick delay is
    /// acceptable at 50Hz — 20ms latency is below human perception threshold.
    fn sync_iroh_bridge(&mut self) {
        let bridge = match &mut self.iroh_bridge {
            Some(b) if b.is_alive() => b,
            _ => return,
        };

        // Flush outgoing social messages to the network
        let outgoing = std::mem::take(&mut self.social_outbox);
        if !outgoing.is_empty() {
            let count = outgoing.len();
            bridge.flush_outbox(outgoing);
            tracing::trace!(
                target: "symthaea::mind::iroh",
                count,
                "Flushed social messages to Iroh bridge"
            );
        }

        // Drain inbound messages from network into inbox
        let incoming = bridge.drain_inbox();
        if !incoming.is_empty() {
            let count = incoming.len();
            for msg in incoming {
                self.social_inbox.push(msg);
            }
            tracing::trace!(
                target: "symthaea::mind::iroh",
                count,
                "Drained inbound messages from Iroh bridge"
            );
        }
    }

    #[cfg(feature = "mesh")]
    /// Process inbound mesh wisdom packets from radio peers.
    ///
    /// Drains `mesh_inbox` and logs received packets. WisdomVector payloads
    /// represent a peer's cognitive state and can be fed into social coherence
    /// for cross-node consciousness integration.
    ///
    /// Also periodically expires stale peers (even when inbox is empty).
    fn process_mesh(&mut self) {
        // Periodically expire stale peers (every 100 ticks) — runs even if no packets arrived
        if self.state.tick.is_multiple_of(100) {
            let expired_ids = self.mesh_peers.expire_stale();
            if !expired_ids.is_empty() {
                self.mesh_stats.peers_expired += expired_ids.len() as u64;

                // Clean up social coherence models for expired peers
                if let Some(ref mut sc) = self.social_coherence {
                    for id in &expired_ids {
                        let peer_id = crate::swarm::mesh::hex_short(id);
                        sc.remove_agent(&peer_id);
                    }
                }

                // Forward PeerLeft events to CLS SwarmManager for each expired peer.
                if let Some(ref tx) = self.swarm_event_tx {
                    for id in &expired_ids {
                        let peer_id = crate::swarm::mesh::hex_short(id);
                        let _ = tx.send(crate::cognitive_loop::SwarmEvent::PeerLeft { peer_id });
                    }
                }

                tracing::debug!(
                    target: "symthaea::mind::mesh",
                    expired = expired_ids.len(),
                    remaining = self.mesh_peers.peer_count(),
                    "Expired stale mesh peers"
                );
            }
        }

        // Priority-aware inbox backpressure: drop lowest-priority packets first
        if self.mesh_inbox.len() > super::MAX_OUTBOX_SIZE {
            let excess = self.mesh_inbox.len() - super::MAX_OUTBOX_SIZE;
            // Sort by priority ascending (lowest-priority first), stable to preserve order within tier
            self.mesh_inbox.sort_by_key(|a| a.payload_type.priority());
            self.mesh_inbox.drain(..excess);
            self.mesh_stats.packets_dropped += excess as u64;
        }

        // Partition detection: if all peers expired, trigger replay buffer flush
        if self.mesh_peers.is_partitioned(&self.mesh_stats) && !self.mesh_replay_buffer.is_empty() {
            tracing::warn!(
                target: "symthaea::mind::mesh",
                replay_count = self.mesh_replay_buffer.len(),
                "Mesh partition detected — flushing replay buffer"
            );

            // Forward mass-disconnect topology change to CLS SwarmManager.
            if let Some(ref tx) = self.swarm_event_tx {
                let _ = tx.send(crate::cognitive_loop::SwarmEvent::TopologyChange {
                    connected_peers: self.mesh_peers.peer_count(),
                    mass_disconnect: true,
                });
            }

            let replays: Vec<_> = self.mesh_replay_buffer.drain(..).collect();
            for packet in replays {
                self.mesh_outbox
                    .push(crate::swarm::mesh::MeshOutbound { packet });
                self.mesh_stats.packets_replayed += 1;
            }
        }

        // Periodic telemetry logging (every 500 ticks, ~10s at 50Hz)
        if self.state.tick.is_multiple_of(500) && self.mesh_bridge.is_some() {
            let t = self.mesh_telemetry();
            tracing::info!(
                target: "symthaea::mind::mesh::stats",
                peers = t.peer_count,
                health = format!("{:.2}", t.health_score),
                wisdom_tx = t.stats.wisdom_sent,
                wisdom_rx = t.stats.wisdom_received,
                heartbeat_tx = t.stats.heartbeats_sent,
                heartbeat_rx = t.stats.heartbeats_received,
                affective_tx = t.stats.affective_sent,
                affective_rx = t.stats.affective_received,
                gradient_tx = t.stats.gradients_sent,
                gradient_rx = t.stats.gradients_received,
                moral_topology_tx = t.stats.moral_topology_sent,
                moral_topology_rx = t.stats.moral_topology_received,
                bytes_tx = t.stats.bytes_sent,
                bytes_rx = t.stats.bytes_received,
                compress_ratio = format!("{:.1}%", t.stats.compression_ratio() * 100.0),
                bandwidth_increases = t.stats.bandwidth_increases,
                bandwidth_decreases = t.stats.bandwidth_decreases,
                avg_phi = format!("{:.3}", t.avg_phi),
                "Mesh telemetry"
            );
        }

        let inbox = std::mem::take(&mut self.mesh_inbox);
        if inbox.is_empty() {
            return;
        }

        let mut wisdom_count = 0u64;
        let mut heartbeat_count = 0u64;
        let mut affective_count = 0u64;
        let mut gradient_count = 0u64;
        let mut moral_topology_count = 0u64;

        for packet in &inbox {
            // Dedup check: skip if we've already seen this (source_id, sequence) pair
            let key = (packet.source_id, packet.sequence, packet.payload_type as u8);
            if self.mesh_seen_packets.contains(&key) {
                self.mesh_stats.packets_deduplicated += 1;
                continue;
            }
            if self.mesh_seen_packets.len() >= super::mesh::MESH_DEDUP_RING_SIZE {
                self.mesh_seen_packets.pop_front();
            }
            self.mesh_seen_packets.push_back(key);

            // Timestamp validation: reject packets older than MESH_MAX_PACKET_AGE_S
            let now_s = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as u32)
                .unwrap_or(0);
            if now_s > 0 && packet.timestamp_s > 0 {
                if now_s.saturating_sub(packet.timestamp_s) > super::mesh::MESH_MAX_PACKET_AGE_S {
                    self.mesh_stats.packets_deduplicated += 1;
                    continue;
                }
            }

            // Per-peer rate limit check
            if self.mesh_peers.is_rate_limited(&packet.source_id) {
                self.mesh_stats.packets_rate_limited += 1;
                continue;
            }

            // MAC verification (Item 1): reject if key is set and MAC doesn't match
            if let Some(ref key) = self.mesh_auth_key {
                let pkt_bytes = packet.to_bytes();
                if !crate::swarm::mesh::verify_packet_mac(&pkt_bytes, key) {
                    self.mesh_stats.packets_auth_failed += 1;
                    continue;
                }
            }

            // TTL forwarding (Item 4): rebroadcast with decremented TTL
            if packet.ttl > 1 {
                let mut fwd = packet.clone();
                fwd.ttl -= 1;
                self.mesh_outbox
                    .push(crate::swarm::mesh::MeshOutbound { packet: fwd });
                self.mesh_stats.packets_forwarded += 1;
            }

            // Partition recovery (Item 5): replay buffer to newly-discovered peers
            let is_new_peer = !self.mesh_peers.has_peer(&packet.source_id);

            // Track all peers in the registry
            self.mesh_peers.update(packet);

            if is_new_peer {
                // Forward PeerJoined to CLS SwarmManager via mpsc channel.
                if let Some(ref tx) = self.swarm_event_tx {
                    let peer_id = crate::swarm::mesh::hex_short(&packet.source_id);
                    let _ = tx.send(crate::cognitive_loop::SwarmEvent::PeerJoined {
                        peer_id,
                        trust_level: 0.5, // initial neutral trust
                    });
                }

                for replay_pkt in self.mesh_replay_buffer.iter() {
                    self.mesh_outbox.push(crate::swarm::mesh::MeshOutbound {
                        packet: replay_pkt.clone(),
                    });
                }
                self.mesh_stats.packets_replayed += self.mesh_replay_buffer.len() as u64;
            }

            match packet.payload_type {
                crate::swarm::mesh::PayloadType::WisdomVector => {
                    wisdom_count += 1;

                    // Feed peer's wisdom into social coherence if enabled.
                    // Convert BinaryHV → ContinuousHV for compatibility with the
                    // social coherence API (which operates in continuous HDC space).
                    if let Some(ref mut sc) = self.social_coherence {
                        let peer_id = crate::swarm::mesh::hex_short(&packet.source_id);
                        let continuous = packet.wisdom.to_continuous();
                        sc.observe_agent(&peer_id, &continuous, &continuous);
                    }
                }
                crate::swarm::mesh::PayloadType::Heartbeat => {
                    heartbeat_count += 1;
                }
                crate::swarm::mesh::PayloadType::Affective => {
                    affective_count += 1;
                    if let Some(ref mut hf) = self.hyperfeel {
                        if let Some(affect) = packet.extract_affective() {
                            let peer_id = crate::swarm::mesh::hex_short(&packet.source_id);
                            hf.receive_peer_state(peer_id.clone(), affect.clone());

                            // Forward affective state to CLS SwarmManager via mpsc channel.
                            // AffectiveState (Hyperfeel) → SwarmEvent::AffectiveSync (SwarmManager).
                            if let Some(ref tx) = self.swarm_event_tx {
                                let event = crate::cognitive_loop::SwarmEvent::AffectiveSync {
                                    peer_id,
                                    valence: affect.valence as f64,
                                    arousal: affect.arousal as f64,
                                    intensity: affect.intensity as f64,
                                };
                                let _ = tx.send(event);
                            }
                        }
                    }
                }
                crate::swarm::mesh::PayloadType::Gradient => {
                    if let Some(gradient_msg) = packet.extract_gradient() {
                        gradient_count += 1;
                        tracing::debug!(
                            target: "symthaea::mind::mesh",
                            source = crate::swarm::mesh::hex_short(&packet.source_id),
                            gradients = gradient_msg.gradient_data.len(),
                            "Routed mesh gradient to federated inbox"
                        );
                        self.federated_inbox.push(gradient_msg);
                    }
                }
                crate::swarm::mesh::PayloadType::MoralTopology => {
                    if let Some(summary) = packet.extract_moral_topology() {
                        self.cached_moral_topology = Some(summary);
                        moral_topology_count += 1;
                    }
                }
                // Sovereign Clock: decode TimeBeacon and forward to CLS TimeManager
                // via the swarm event channel.
                crate::swarm::mesh::PayloadType::TimeBeacon => {
                    if let Some(beacon) =
                        crate::swarm::mesh::time_beacon::TimeBeacon::decode(&packet.wisdom)
                    {
                        if let Some(ref tx) = self.swarm_event_tx {
                            let event = crate::cognitive_loop::SwarmEvent::TimeBeaconReceived {
                                source_id: packet.source_id,
                                timestamp_us: beacon.timestamp_us,
                                stratum: beacon.stratum,
                                phi: beacon.phi,
                                drift_ppm: beacon.drift_ppm,
                            };
                            let _ = tx.send(event);
                        }
                    }
                }
                // Sovereign Social: decode ContentAnnounce and forward to CLS
                crate::swarm::mesh::PayloadType::ContentAnnounce => {
                    if let Some(announce) =
                        crate::swarm::mesh::content_packet::ContentAnnounce::decode(&packet.wisdom)
                    {
                        if let Some(ref tx) = self.swarm_event_tx {
                            let event = crate::cognitive_loop::SwarmEvent::ContentAnnounced {
                                peer_id: crate::swarm::mesh::hex_short(&packet.source_id),
                                content_hash: announce.content_hash,
                                truncated_hdv: announce.truncated_hdv,
                                domain: announce.domain,
                                created_at: announce.created_at,
                            };
                            let _ = tx.send(event);
                        }
                    }
                }
                // Sovereign Name: NameQuery/NameResponse handled at mesh layer (no CLS routing needed)
                crate::swarm::mesh::PayloadType::NameQuery
                | crate::swarm::mesh::PayloadType::NameResponse => {}
            }
        }

        // Update receive-side telemetry
        self.mesh_stats.wisdom_received += wisdom_count;
        self.mesh_stats.heartbeats_received += heartbeat_count;
        self.mesh_stats.affective_received += affective_count;
        self.mesh_stats.gradients_received += gradient_count;
        self.mesh_stats.moral_topology_received += moral_topology_count;
        self.mesh_stats.bytes_received +=
            inbox.len() as u64 * crate::swarm::mesh::WISDOM_PACKET_SIZE as u64;
        #[cfg(feature = "mesh-encryption")]
        if self.mesh_encryption_key.is_some() {
            self.mesh_stats.encrypted_packets_received += inbox.len() as u64;
        }

        tracing::debug!(
            target: "symthaea::mind::mesh",
            total = inbox.len(),
            wisdom = wisdom_count,
            heartbeat = heartbeat_count,
            affective = affective_count,
            gradient = gradient_count,
            moral_topology = moral_topology_count,
            "Processed mesh wisdom packets"
        );
    }

    #[cfg(feature = "mesh")]
    /// Sync mesh packets through the mesh bridge (if attached).
    ///
    /// 1. Flushes `mesh_outbox` to the radio network actor (non-blocking)
    /// 2. Drains inbound wisdom packets into `mesh_inbox`
    ///
    /// Packets arriving from the network will be processed on the *next* tick
    /// by `process_mesh()`.
    fn sync_mesh_bridge(&mut self) {
        let bridge = match &mut self.mesh_bridge {
            Some(b) if b.is_alive() => b,
            _ => return,
        };

        // Drain CLS-generated outbound packets (sovereign beacons, name responses, etc.)
        #[cfg(feature = "mesh")]
        if let Some(ref rx_mutex) = self.mesh_outbound_rx {
            if let Ok(rx) = rx_mutex.lock() {
                while let Ok(outbound) = rx.try_recv() {
                    self.mesh_outbox.push(outbound);
                }
            }
        }

        // Flush outgoing mesh packets to the network
        let outgoing = std::mem::take(&mut self.mesh_outbox);
        if !outgoing.is_empty() {
            let count = outgoing.len();
            bridge.flush_outbox(outgoing);
            tracing::trace!(
                target: "symthaea::mind::mesh",
                count,
                "Flushed mesh packets to bridge"
            );
        }

        // Drain inbound packets from network into inbox
        let incoming = bridge.drain_inbox();
        if !incoming.is_empty() {
            let count = incoming.len();
            for pkt in incoming {
                self.mesh_inbox.push(pkt);
            }
            tracing::trace!(
                target: "symthaea::mind::mesh",
                count,
                "Drained inbound packets from mesh bridge"
            );
        }
    }

    #[cfg(feature = "mesh")]
    /// Poll physical sensors and feed readings into the cognitive loop.
    ///
    /// Each sensor reading is encoded as an HDC hypervector and fed into
    /// the perception queue.  Critical urgency readings (e.g., smoke alarm)
    /// trigger an immediate mesh broadcast.
    fn process_sensors(&mut self) {
        let registry = match &mut self.sensor_registry {
            Some(r) => r,
            None => return,
        };

        let readings = registry.poll_all();
        if readings.is_empty() {
            return;
        }

        // Encode all readings up front so we can drop the registry borrow
        // before calling self.perceive() / self.emit_wisdom().
        let encoded: Vec<_> = readings
            .into_iter()
            .map(|(reading, urgency)| {
                let hv = registry.encode_reading(&reading);
                let sensor_id = reading.sensor_id.clone();
                let n_values = reading.values.len();
                (hv, urgency, reading, sensor_id, n_values)
            })
            .collect();

        for (hv, urgency, reading, sensor_id, n_values) in encoded {
            // Feed encoded sensor reading into working memory as a perception
            let mut input = MindInput::new(InputType::Perception, hv);
            input
                .metadata
                .insert("sensor_id".to_string(), sensor_id.clone());
            input
                .metadata
                .insert("timestamp_s".to_string(), reading.timestamp_s.to_string());
            if !reading.tags.is_empty() {
                let tags_json = serde_json::to_string(&reading.tags).unwrap_or_else(|e| {
                    tracing::warn!(sensor = %sensor_id, error = %e, "Failed to serialize sensor tags");
                    "[]".to_string()
                });
                input.metadata.insert("topics".to_string(), tags_json);
            }
            for (key, value) in reading.metadata {
                input.metadata.insert(key, value);
            }
            self.input(input);

            // If urgency is Critical (e.g., smoke alarm), broadcast immediately
            if urgency == crate::swarm::mesh::MeshUrgency::Critical && self.mesh_bridge.is_some() {
                let binary = symthaea_core::hdc::phi_topology_validation::real_hv_to_hv16(
                    &self.state.current_thought,
                );
                self.emit_wisdom(
                    binary,
                    crate::cognitive_loop::types::CycleUrgency::Critical,
                    self.state.consciousness_level as f32,
                );
            }

            tracing::debug!(
                target: "symthaea::mind::sensor",
                sensor = sensor_id,
                urgency = ?urgency,
                values = n_values,
                "Sensor reading perceived"
            );
        }
    }

    #[cfg(feature = "mesh")]
    /// Emit pending federated gradients over the mesh network.
    ///
    /// Drains `federated_outbox` and converts each `GradientMessage` into a
    /// mesh `WisdomPacket` (Gradient type). Oversized gradients (>504 f32s)
    /// are skipped with a warning since they exceed the 2,048-byte wisdom field.
    ///
    /// When no mesh bridge is attached, returns immediately — the outbox is
    /// preserved for `drain_outbox()` callers.
    pub(crate) fn emit_gradients(&mut self) {
        if self.mesh_bridge.is_none() {
            return;
        }

        let outbox = std::mem::take(&mut self.federated_outbox);
        if outbox.is_empty() {
            return;
        }

        let source_id = self.mesh_source_id();

        for msg in &outbox {
            // Bandwidth budget check (per-packet in loop)
            if !self.mesh_bandwidth_check(crate::swarm::mesh::WISDOM_PACKET_SIZE as u64) {
                continue;
            }

            match crate::swarm::mesh::WisdomPacket::from_gradient(
                source_id,
                self.mesh_gradient_sequence,
                msg,
            ) {
                Some(mut packet) => {
                    self.sign_mesh_packet(&mut packet);
                    self.mesh_stats.bytes_before_compression +=
                        crate::swarm::mesh::WISDOM_PACKET_SIZE as u64;
                    self.mesh_stats.bytes_after_compression +=
                        crate::swarm::mesh::compress_packet(&packet.to_bytes()).len() as u64;
                    self.mesh_outbox
                        .push(crate::swarm::mesh::MeshOutbound { packet });
                    self.mesh_gradient_sequence = self.mesh_gradient_sequence.wrapping_add(1);
                    self.mesh_stats.gradients_sent += 1;
                    self.mesh_stats.bytes_sent += crate::swarm::mesh::WISDOM_PACKET_SIZE as u64;
                    #[cfg(feature = "mesh-encryption")]
                    if self.mesh_encryption_key.is_some() {
                        self.mesh_stats.encrypted_packets_sent += 1;
                    }
                }
                None => {
                    tracing::warn!(
                        target: "symthaea::mind::mesh",
                        gradients = msg.gradient_data.len(),
                        "Skipped oversized gradient (max 504 f32s)"
                    );
                }
            }
        }
    }

    #[cfg(feature = "mesh")]
    /// Emit the mind's affective state over the mesh network.
    ///
    /// Fires every 50 ticks (~1s at 50Hz) — emotional state changes slowly
    /// compared to cognitive state. Maps the mind's `emotional_valence` and
    /// `arousal` into the VAD affective wire format.
    pub(crate) fn emit_affective(&mut self) {
        if self.mesh_bridge.is_none() {
            return;
        }

        let interval = 50u64;
        if self
            .state
            .tick
            .saturating_sub(self.mesh_affective_last_tick)
            < interval
            && self.mesh_affective_sequence > 0
        {
            return;
        }

        // Bandwidth budget check
        if !self.mesh_bandwidth_check(crate::swarm::mesh::WISDOM_PACKET_SIZE as u64) {
            return;
        }

        let source_id = self.mesh_source_id();

        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let arousal = self.state.arousal;
        let affect = crate::swarm::AffectiveState {
            valence: self.state.emotional_valence,
            arousal,
            dominance: 0.0,
            intensity: arousal.abs(),
            thermodynamic_load: self.state.thermodynamic_load,
            confidence: 1.0,
            timestamp_ms,
            sequence: self.mesh_affective_sequence as u64,
        };

        let mut packet = crate::swarm::mesh::WisdomPacket::from_affective(
            source_id,
            self.mesh_affective_sequence,
            &affect,
        );

        self.sign_mesh_packet(&mut packet);

        self.mesh_stats.bytes_before_compression += crate::swarm::mesh::WISDOM_PACKET_SIZE as u64;
        self.mesh_stats.bytes_after_compression +=
            crate::swarm::mesh::compress_packet(&packet.to_bytes()).len() as u64;

        self.mesh_outbox
            .push(crate::swarm::mesh::MeshOutbound { packet });
        self.mesh_affective_last_tick = self.state.tick;
        self.mesh_affective_sequence = self.mesh_affective_sequence.wrapping_add(1);
        self.mesh_stats.affective_sent += 1;
        self.mesh_stats.bytes_sent += crate::swarm::mesh::WISDOM_PACKET_SIZE as u64;
        #[cfg(feature = "mesh-encryption")]
        if self.mesh_encryption_key.is_some() {
            self.mesh_stats.encrypted_packets_sent += 1;
        }

        tracing::trace!(
            target: "symthaea::mind::mesh",
            sequence = self.mesh_affective_sequence.wrapping_sub(1),
            valence = self.state.emotional_valence,
            arousal,
            "Emitted affective packet"
        );
    }

    #[cfg(feature = "mesh")]
    /// Auto-emit wisdom to mesh if a bridge is attached.
    ///
    /// Called at the end of every waking tick (after `generate_output()`),
    /// so the mesh always carries the mind's most up-to-date cognitive state.
    /// Emission frequency is gated by `emit_wisdom()`'s urgency throttle,
    /// with health-driven urgency override for mesh recovery.
    pub(crate) fn auto_emit_wisdom(&mut self) {
        if self.mesh_bridge.is_some() {
            let wisdom_hv = symthaea_core::hdc::phi_topology_validation::real_hv_to_hv16(
                &self.state.current_thought,
            );
            let phi = self.state.consciousness_level as f32;

            // Base urgency from arousal
            let mut urgency =
                crate::cognitive_loop::types::CycleUrgency::from_arousal(self.state.arousal);

            // Health-driven override
            let health = self.mesh_stats.health_score(self.mesh_peers.peer_count());
            if health < 0.3 && health > 0.0 {
                // Degraded mesh — blast to recover
                urgency = crate::cognitive_loop::types::CycleUrgency::Critical;
            } else if health <= 0.8 && health > 0.0 {
                // Suboptimal — prevent Cruise throttling
                if matches!(urgency, crate::cognitive_loop::types::CycleUrgency::Cruise) {
                    urgency = crate::cognitive_loop::types::CycleUrgency::Normal;
                }
            }
            // health > 0.8 or 0.0 (no activity): arousal-based, no override

            self.emit_wisdom(wisdom_hv, urgency, phi);
        }
    }

    /// Process federated learning gradient exchange.
    ///
    /// Receives gradients from peers, aggregates when enough are collected,
    /// and exports local gradients for broadcast.
    fn process_federated(&mut self) {
        let federated = match &mut self.federated {
            Some(f) => f,
            None => return,
        };

        // Step 1: Receive pending gradients from inbox
        let inbox = std::mem::take(&mut self.federated_inbox);
        let mut received = 0;
        for msg in inbox {
            if federated.receive_gradient(msg) {
                received += 1;
            }
        }

        if received > 0 {
            tracing::debug!(
                target: "symthaea::mind::federated",
                received,
                pending = federated.pending_contributions(),
                round = federated.aggregation_round(),
                "Received gradients from peers"
            );
        }

        // Step 2: Aggregate every 10 ticks if we have enough contributions
        if self.state.tick.is_multiple_of(10) && federated.pending_contributions() >= 2 {
            let n_contributors = federated.pending_contributions();
            if let Some(aggregated) = federated.aggregate() {
                let lr = 0.01f32;
                federated.apply_gradient(&aggregated, lr);

                let round = federated.aggregation_round();
                tracing::info!(
                    target: "symthaea::mind::federated",
                    round,
                    "Applied federated gradient aggregation"
                );

                // Forward federated round result to CLS SwarmManager.
                // Quality and trust estimated from contributor count (more peers → higher confidence).
                if let Some(ref tx) = self.swarm_event_tx {
                    let quality = (n_contributors as f64 / 10.0).min(1.0); // saturates at 10 contributors
                    let trust = (n_contributors as f64 / 5.0).min(1.0); // saturates at 5
                    crate::cognitive_loop::forward_federated_round(
                        tx,
                        n_contributors,
                        quality,
                        trust,
                    );
                }
            }
        }

        // Step 3: Export local gradient every 5 ticks
        if self.state.tick.is_multiple_of(5) {
            let msg = federated.export_local_gradient(0.0);
            self.federated_outbox.push(msg);

            // Cap outbox to prevent unbounded growth
            if self.federated_outbox.len() > super::MAX_OUTBOX_SIZE {
                let excess = self.federated_outbox.len() - super::MAX_OUTBOX_SIZE;
                self.federated_outbox.drain(..excess);
            }
        }

        // Step 4 (liquid-mamba): Export L-SSM projection weights for swarm exchange
        // and apply incoming aggregated peer weights. Uses the LLMBackend trait's
        // export_gradient()/apply_aggregated_gradient() methods (Phase 6 API).
        #[cfg(feature = "liquid-mamba")]
        if let Some(ref backend) = self.llm_backend {
            // Export projection weights every 10 ticks (same cadence as aggregation)
            if self.state.tick.is_multiple_of(10) {
                let source_id = self.genesis_identity;
                if let Some(weights) = backend.export_gradient(source_id, 1.0, self.state.tick) {
                    tracing::debug!(
                        target: "symthaea::mind::federated",
                        weight_count = weights.len(),
                        tick = self.state.tick,
                        "Exported L-SSM projection weights for swarm"
                    );
                    let msg = crate::swarm::GradientMessage::new(source_id, weights, 1.0);
                    self.federated_outbox.push(msg);
                }
            }

            // Apply aggregated weights back (every 20 ticks, slower to avoid oscillation)
            if self.state.tick.is_multiple_of(20) && federated.pending_contributions() >= 2 {
                if let Some(aggregated) = federated.aggregate() {
                    if backend.apply_aggregated_gradient(&aggregated) {
                        tracing::info!(
                            target: "symthaea::mind::federated",
                            "Applied aggregated L-SSM projection weights from peers"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::memory::memory_coordinator::MemorySource;
    use crate::mind::{ContinuousMind, InputType, MindConfig, MindInput, OutputType};
    use symthaea_core::hdc::ContinuousHV;

    /// Helper: create an activated mind with a specific HDC dimension.
    fn make_mind(dimension: usize) -> ContinuousMind {
        let mut mind = ContinuousMind::new(MindConfig {
            dimension,
            ..Default::default()
        });
        mind.activate();
        mind
    }

    /// Helper: create an activated mind with default config.
    fn activated_mind() -> ContinuousMind {
        let mut mind = ContinuousMind::default();
        mind.activate();
        mind
    }

    /// Helper: create an activated mind with a specific working memory capacity.
    fn mind_with_capacity(cap: usize) -> ContinuousMind {
        let mut mind = ContinuousMind::new(MindConfig {
            working_memory_capacity: cap,
            ..Default::default()
        });
        mind.activate();
        mind
    }

    // ====================================================================
    // tick() — top-level cognitive cycle
    // ====================================================================

    #[test]
    fn tick_increments_counters() {
        let mut mind = activated_mind();
        assert_eq!(mind.state.tick, 0);
        assert_eq!(mind.stats.total_ticks, 0);

        mind.tick();
        assert_eq!(mind.state.tick, 1);
        assert_eq!(mind.stats.total_ticks, 1);

        mind.tick();
        assert_eq!(mind.state.tick, 2);
        assert_eq!(mind.stats.total_ticks, 2);
    }

    #[test]
    fn tick_updates_biorhythm() {
        let mut mind = activated_mind();
        assert!(mind.state.biorhythm.is_none());
        mind.tick();
        assert!(mind.state.biorhythm.is_some());
    }

    #[test]
    fn tick_updates_processing_latency() {
        let mut mind = activated_mind();
        mind.perceive(ContinuousHV::random(512, 1));
        mind.tick();
        // Latency should be set to a non-negative value
        assert!(
            mind.state.processing_latency_ms >= 0.0,
            "latency should be non-negative: {}",
            mind.state.processing_latency_ms
        );
    }

    #[test]
    fn tick_updates_memory_utilization() {
        let mut mind = mind_with_capacity(7);
        mind.perceive(ContinuousHV::random(512, 1));
        mind.perceive(ContinuousHV::random(512, 2));
        mind.tick();
        // 2 items out of capacity 7
        let expected = 2.0 / 7.0;
        let diff = (mind.state.memory_utilization - expected).abs();
        assert!(
            diff < 0.01,
            "memory_utilization mismatch: got {}, expected ~{}",
            mind.state.memory_utilization,
            expected
        );
    }

    #[test]
    fn tick_tracks_average_consciousness() {
        let mut mind = activated_mind();
        // Tick several times to accumulate stats
        for i in 0..5 {
            mind.perceive(ContinuousHV::random(512, 100 + i));
            mind.tick();
        }
        // Average should be between 0 and 1
        assert!(mind.stats.avg_consciousness >= 0.0);
        assert!(mind.stats.avg_consciousness <= 1.0);
    }

    #[test]
    fn tick_tracks_peak_consciousness() {
        let mut mind = activated_mind();
        for i in 0..10 {
            mind.perceive(ContinuousHV::random(512, i));
            mind.tick();
        }
        // Peak should be >= average
        assert!(mind.stats.peak_consciousness >= mind.stats.avg_consciousness);
    }

    // ====================================================================
    // process_inputs() — input queue draining and working memory
    // ====================================================================

    #[test]
    fn process_inputs_fills_working_memory() {
        let mut mind = activated_mind();
        mind.perceive(ContinuousHV::random(512, 42));
        mind.process_inputs();
        assert_eq!(mind.working_memory.len(), 1);
        assert_eq!(mind.working_memory_ticks.len(), 1);
        assert_eq!(mind.working_memory_sources.len(), 1);
        assert_eq!(mind.working_memory_verified.len(), 1);
        assert_eq!(mind.working_memory_metadata.len(), 1);
    }

    #[test]
    fn process_inputs_respects_capacity() {
        let cap = 3;
        let mut mind = mind_with_capacity(cap);

        for i in 0..(cap + 2) {
            mind.perceive(ContinuousHV::random(512, i as u64));
        }
        mind.process_inputs();
        assert_eq!(
            mind.working_memory.len(),
            cap,
            "working memory should not exceed capacity"
        );
    }

    #[test]
    fn process_inputs_evicts_oldest_when_full() {
        let cap = 3;
        let mut mind = mind_with_capacity(cap);

        // Fill to capacity
        for i in 0..cap {
            mind.perceive(ContinuousHV::random(512, i as u64));
        }
        // Advance tick so eviction computes steps_survived > 0
        mind.state.tick = 10;
        mind.process_inputs();
        assert!(mind.evicted_items.is_empty(), "no eviction yet");

        // Add one more — should evict the oldest
        mind.perceive(ContinuousHV::random(512, 999));
        mind.state.tick = 20;
        mind.process_inputs();
        assert_eq!(mind.evicted_items.len(), 1);
        // The evicted item should have survived steps_survived = 20 - arrival_tick
        assert!(mind.evicted_items[0].steps_survived > 0);
    }

    #[test]
    fn process_inputs_preserves_source_and_verification() {
        let mut mind = activated_mind();
        let input = MindInput::new(InputType::Perception, ContinuousHV::random(512, 1))
            .with_source(MemorySource::UserInteraction)
            .with_verification(true);
        mind.input(input);
        mind.process_inputs();
        assert_eq!(
            mind.working_memory_sources[0],
            MemorySource::UserInteraction
        );
        assert!(mind.working_memory_verified[0]);
    }

    #[test]
    fn process_inputs_sorts_by_priority() {
        let mut mind = activated_mind();

        let mut low = MindInput::new(InputType::Perception, ContinuousHV::random(512, 1));
        low.priority = 0.1;
        let mut high = MindInput::new(InputType::Perception, ContinuousHV::random(512, 2));
        high.priority = 0.9;
        mind.input(low);
        mind.input(high);

        // process_inputs sorts ascending then pops from back (highest priority processed last,
        // so highest-priority item ends up last in working memory)
        mind.process_inputs();
        assert_eq!(mind.stats.inputs_processed, 2);
    }

    #[test]
    fn process_inputs_creates_goal_from_goal_input() {
        let mut mind = activated_mind();
        mind.set_goal("learn Rust", ContinuousHV::random(512, 42), 0.8);
        mind.process_inputs();
        assert_eq!(mind.goals.len(), 1);
        assert_eq!(mind.goals[0].description, "learn Rust");
        assert!(mind.goals[0].is_active);
        assert_eq!(mind.goals[0].progress, 0.0);
    }

    #[test]
    fn process_inputs_applies_feedback_valence() {
        let mut mind = activated_mind();
        assert_eq!(mind.state.emotional_valence, 0.0);

        let mut feedback = MindInput::new(InputType::Feedback, ContinuousHV::random(512, 1));
        feedback
            .metadata
            .insert("valence".to_string(), "0.8".to_string());
        mind.input(feedback);
        mind.process_inputs();

        // Valence should shift: 0.0 + 0.8 * 0.3 = 0.24
        let expected = 0.24f32;
        let diff = (mind.state.emotional_valence - expected).abs();
        assert!(
            diff < 0.01,
            "emotional_valence should be ~{}: got {}",
            expected,
            mind.state.emotional_valence
        );
    }

    #[test]
    fn process_inputs_clamps_emotional_valence() {
        let mut mind = activated_mind();
        mind.state.emotional_valence = 0.95;

        let mut feedback = MindInput::new(InputType::Feedback, ContinuousHV::random(512, 1));
        feedback
            .metadata
            .insert("valence".to_string(), "1.0".to_string());
        mind.input(feedback);
        mind.process_inputs();

        assert!(
            mind.state.emotional_valence <= 1.0,
            "valence should be clamped to 1.0: got {}",
            mind.state.emotional_valence
        );
    }

    // ====================================================================
    // update_consciousness() — integration measure
    // ====================================================================

    #[test]
    fn update_consciousness_empty_memory_gives_baseline() {
        let mut mind = activated_mind();
        mind.update_consciousness();
        assert!(
            (mind.state.consciousness_level - 0.1).abs() < 0.001,
            "empty memory should give baseline 0.1: got {}",
            mind.state.consciousness_level
        );
    }

    #[test]
    fn update_consciousness_with_diverse_memories() {
        let mut mind = activated_mind();
        // Add diverse random vectors — dissimilar items boost integration
        for i in 0..5 {
            mind.working_memory.push(ContinuousHV::random(512, 100 + i));
        }
        mind.update_consciousness();
        assert!(
            mind.state.consciousness_level > 0.1,
            "diverse memories should raise consciousness above baseline: got {}",
            mind.state.consciousness_level
        );
    }

    #[test]
    fn update_consciousness_identical_memories_low_integration() {
        let mut mind = activated_mind();
        let hv = ContinuousHV::random(512, 42);
        // Identical items should have high similarity, low integration
        for _ in 0..4 {
            mind.working_memory.push(hv.clone());
        }
        mind.update_consciousness();
        // (1 - similarity) for identical vectors ≈ 0
        assert!(
            mind.state.consciousness_level < 0.05,
            "identical memories should have very low integration: got {}",
            mind.state.consciousness_level
        );
    }

    #[test]
    fn update_consciousness_relational_psi_boost() {
        let mut mind = activated_mind();
        for i in 0..5 {
            mind.working_memory.push(ContinuousHV::random(512, 200 + i));
        }
        mind.update_consciousness();
        let base_level = mind.state.consciousness_level;

        // Set relational psi and re-update
        mind.relational_psi = 1.0;
        // Need to re-add memories since consciousness is computed from scratch
        mind.update_consciousness();
        let boosted_level = mind.state.consciousness_level;

        assert!(
            boosted_level >= base_level,
            "relational_psi should boost consciousness: base={}, boosted={}",
            base_level,
            boosted_level
        );
    }

    #[test]
    fn update_consciousness_relational_psi_zero_no_boost() {
        let mut mind = activated_mind();
        for i in 0..3 {
            mind.working_memory.push(ContinuousHV::random(512, 300 + i));
        }
        mind.relational_psi = 0.0;
        mind.update_consciousness();
        let level = mind.state.consciousness_level;

        // With psi=0, boost factor is 1.0 (no change)
        assert!((0.0..=1.0).contains(&level));
    }

    // ====================================================================
    // generate_output() — consciousness-gated output
    // ====================================================================

    #[test]
    fn generate_output_below_threshold_returns_none() {
        let mut mind = activated_mind();
        mind.state.consciousness_level = 0.01; // Below default min_consciousness (0.1)
        let output = mind.generate_output();
        assert!(
            output.is_none(),
            "should not generate output below consciousness threshold"
        );
    }

    #[test]
    fn generate_output_requires_tick_multiple_of_10() {
        let mut mind = activated_mind();
        mind.state.consciousness_level = 0.5;
        mind.working_memory.push(ContinuousHV::random(512, 42));

        // Tick not a multiple of 10 — should return None
        mind.state.tick = 7;
        let output = mind.generate_output();
        assert!(
            output.is_none(),
            "should not generate output on non-10-multiple tick"
        );

        // Tick is a multiple of 10
        mind.state.tick = 10;
        mind.state.thermodynamic_load = 0.0; // Ensure no veto
        let output = mind.generate_output();
        if let Some(ref out) = output {
            assert_eq!(out.output_type, OutputType::Thought);
            assert!(out.content.contains("working memory"));
        }
    }

    #[test]
    fn generate_output_thermodynamic_veto() {
        let mut mind = activated_mind();
        mind.state.consciousness_level = 0.5;
        mind.state.thermodynamic_load = 0.9; // High load
        mind.state.tick = 10;
        mind.working_memory.push(ContinuousHV::random(512, 42));

        // The holocell.simulate() may or may not exceed 0.9 predicted load,
        // but with high thermodynamic_load the veto path is possible.
        // We just verify no panic occurs.
        let _output = mind.generate_output();
    }

    // ====================================================================
    // take_evicted / take_evicted_tagged
    // ====================================================================

    #[test]
    fn take_evicted_drains_buffer() {
        let mut mind = activated_mind();
        mind.evicted_items.push(crate::mind::EvictedMemory {
            content: ContinuousHV::random(512, 1),
            steps_survived: 42,
            source: MemorySource::Internal,
            is_verified: false,
            metadata: std::collections::HashMap::new(),
        });
        let evicted = mind.take_evicted();
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].1, 42); // steps_survived
        // Buffer should be empty after drain
        assert!(mind.take_evicted().is_empty());
    }

    #[test]
    fn take_evicted_tagged_drains_with_metadata() {
        let mut mind = activated_mind();
        let mut meta = std::collections::HashMap::new();
        meta.insert("topic".to_string(), "science".to_string());
        mind.evicted_items.push(crate::mind::EvictedMemory {
            content: ContinuousHV::random(512, 2),
            steps_survived: 100,
            source: MemorySource::WebResearch,
            is_verified: true,
            metadata: meta,
        });
        let tagged = mind.take_evicted_tagged();
        assert_eq!(tagged.len(), 1);
        assert_eq!(tagged[0].steps_survived, 100);
        assert!(tagged[0].is_verified);
        assert_eq!(tagged[0].metadata.get("topic").unwrap(), "science");
        assert!(mind.take_evicted_tagged().is_empty());
    }

    // ====================================================================
    // Dream processing
    // ====================================================================

    #[test]
    fn process_dream_consolidates_similar_memories() {
        let mut mind = activated_mind();
        // Create two very similar vectors (same seed = identical)
        let hv = ContinuousHV::random(512, 42);
        mind.working_memory.push(hv.clone());
        mind.working_memory.push(hv.clone());
        mind.working_memory_ticks.push(0);
        mind.working_memory_ticks.push(1);
        mind.working_memory_sources
            .push(MemorySource::UserInteraction);
        mind.working_memory_sources.push(MemorySource::Internal);
        mind.working_memory_verified.push(true);
        mind.working_memory_verified.push(false);
        mind.working_memory_metadata
            .push(std::collections::HashMap::new());
        mind.working_memory_metadata
            .push(std::collections::HashMap::new());

        let output = mind.process_dream();
        assert!(output.is_some(), "should consolidate identical memories");
        let out = output.unwrap();
        assert_eq!(out.output_type, OutputType::Memorize);
        assert!(out.content.contains("Consolidating"));
        // After consolidation, one memory should be removed
        assert_eq!(mind.working_memory.len(), 1);
    }

    #[test]
    fn process_dream_preserves_dissimilar_memories() {
        let mut mind = activated_mind();
        // Use different seeds — random 512-dim vectors will be dissimilar
        mind.working_memory.push(ContinuousHV::random(512, 100));
        mind.working_memory.push(ContinuousHV::random(512, 200));
        mind.working_memory_ticks.push(0);
        mind.working_memory_ticks.push(1);
        mind.working_memory_sources.push(MemorySource::Internal);
        mind.working_memory_sources.push(MemorySource::Internal);
        mind.working_memory_verified.push(false);
        mind.working_memory_verified.push(false);
        mind.working_memory_metadata
            .push(std::collections::HashMap::new());
        mind.working_memory_metadata
            .push(std::collections::HashMap::new());

        // Dissimilar vectors should not consolidate (similarity < 0.8)
        // The dream thought is stochastic, so we just verify no panic
        let _output = mind.process_dream();
        assert!(
            !mind.working_memory.is_empty(),
            "dissimilar memories should be preserved"
        );
    }

    #[test]
    fn process_dream_source_priority_action_feedback_wins() {
        let mut mind = activated_mind();
        let hv = ContinuousHV::random(512, 42);
        mind.working_memory.push(hv.clone());
        mind.working_memory.push(hv.clone());
        mind.working_memory_ticks.push(0);
        mind.working_memory_ticks.push(1);
        mind.working_memory_sources
            .push(MemorySource::ActionFeedback);
        mind.working_memory_sources
            .push(MemorySource::UserInteraction);
        mind.working_memory_verified.push(false);
        mind.working_memory_verified.push(false);
        mind.working_memory_metadata
            .push(std::collections::HashMap::new());
        mind.working_memory_metadata
            .push(std::collections::HashMap::new());

        let _ = mind.process_dream();
        // After consolidation, source should be ActionFeedback (highest priority)
        assert_eq!(mind.working_memory_sources[0], MemorySource::ActionFeedback);
    }

    #[test]
    fn process_dream_verified_flag_or_merged() {
        let mut mind = activated_mind();
        let hv = ContinuousHV::random(512, 42);
        mind.working_memory.push(hv.clone());
        mind.working_memory.push(hv.clone());
        mind.working_memory_ticks.push(0);
        mind.working_memory_ticks.push(1);
        mind.working_memory_sources.push(MemorySource::Internal);
        mind.working_memory_sources.push(MemorySource::Internal);
        mind.working_memory_verified.push(false);
        mind.working_memory_verified.push(true); // One is verified
        mind.working_memory_metadata
            .push(std::collections::HashMap::new());
        mind.working_memory_metadata
            .push(std::collections::HashMap::new());

        let _ = mind.process_dream();
        // Merged verified status should be true (false || true)
        assert!(
            mind.working_memory_verified[0],
            "merged verification should be true if either is verified"
        );
    }

    #[test]
    fn process_dream_no_crash_on_empty_memory() {
        let mut mind = activated_mind();
        let output = mind.process_dream();
        // With empty memory and stochastic dream roll, output may or may not appear
        let _ = output;
    }

    #[test]
    fn process_dream_no_crash_on_single_memory() {
        let mut mind = activated_mind();
        mind.working_memory.push(ContinuousHV::random(512, 42));
        mind.working_memory_ticks.push(0);
        mind.working_memory_sources.push(MemorySource::Internal);
        mind.working_memory_verified.push(false);
        mind.working_memory_metadata
            .push(std::collections::HashMap::new());
        let _output = mind.process_dream();
        // Single memory cannot be consolidated — should not panic
        assert_eq!(mind.working_memory.len(), 1);
    }

    // ====================================================================
    // Federated outbox cap
    // ====================================================================

    #[test]
    fn federated_outbox_capped_at_max_size() {
        let mut mind = activated_mind();
        // Enable federated learning
        mind.federated = Some(crate::swarm::FederatedAggregator::new(vec![0.0; 512]));

        // Run many ticks to fill outbox (exports every 5 ticks)
        for _ in 0..500 {
            mind.state.tick += 1;
            mind.process_federated();
        }

        assert!(
            mind.federated_outbox.len() <= super::super::MAX_OUTBOX_SIZE,
            "federated_outbox should be capped at {}: got {}",
            super::super::MAX_OUTBOX_SIZE,
            mind.federated_outbox.len()
        );
    }

    // ── Dream bath neurochemical recovery ─────────────────────────

    #[test]
    fn dream_adenosine_clears() {
        let mut mind = make_mind(64);
        mind.dream_bath.adenosine.produce(0.4); // raise adenosine
        let before = mind.dream_bath.adenosine.level;
        mind.state.is_dreaming = true;
        mind.process_dream();
        assert!(
            mind.dream_bath.adenosine.level < before,
            "adenosine should clear during dream: before={before}, after={}",
            mind.dream_bath.adenosine.level
        );
    }

    #[test]
    fn dream_allostatic_recovers() {
        let mut mind = make_mind(64);
        mind.dream_bath.allostatic_load = 0.5;
        mind.state.is_dreaming = true;
        mind.process_dream();
        assert!(
            mind.dream_bath.allostatic_load < 0.5,
            "allostatic load should decrease during dream: {}",
            mind.dream_bath.allostatic_load
        );
    }

    #[test]
    fn dream_sht1a_boosts() {
        let mut mind = make_mind(64);
        let before = mind.dream_bath.serotonin.receptor_sensitivity;
        mind.state.is_dreaming = true;
        mind.process_dream();
        assert!(
            mind.dream_bath.serotonin.receptor_sensitivity > before,
            "5-HT receptor sensitivity should increase: before={before}, after={}",
            mind.dream_bath.serotonin.receptor_sensitivity
        );
    }

    #[test]
    fn dream_ecb_produces_under_stress() {
        let mut mind = make_mind(64);
        mind.dream_bath.allostatic_load = 0.4;
        let before = mind.dream_bath.endocannabinoid.level;
        mind.state.is_dreaming = true;
        mind.process_dream();
        assert!(
            mind.dream_bath.endocannabinoid.level > before,
            "ECB should increase under stress: before={before}, after={}",
            mind.dream_bath.endocannabinoid.level
        );
    }

    #[test]
    fn dream_no_panic_on_defaults() {
        let mut mind = make_mind(64);
        mind.state.is_dreaming = true;
        // process_dream with default bath should not panic
        let _ = mind.process_dream();
    }

    #[test]
    fn dream_telemetry_increments() {
        let mut mind = make_mind(64);
        mind.state.is_dreaming = true;
        assert_eq!(mind.stats.dream_adenosine_cleared, 0);
        mind.process_dream();
        assert_eq!(mind.stats.dream_adenosine_cleared, 1);
        mind.process_dream();
        assert_eq!(mind.stats.dream_adenosine_cleared, 2);
    }
}
