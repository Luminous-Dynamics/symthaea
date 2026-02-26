//! Tick processing for the Continuous Mind.
//!
//! Contains the main cognitive cycle (`tick()`), dream processing,
//! input handling, consciousness updates, and output generation.

use crate::chronobiology::{Biorhythm, CircadianPhase};
use symthaea_core::hdc::ContinuousHV;

use super::utils::permute_hv;
use super::{ContinuousMind, Goal, InputType, MindInput, MindOutput, OutputType};

impl ContinuousMind {
    /// Process one tick of the mind.
    pub fn tick(&mut self) -> Option<MindOutput> {
        let start = std::time::Instant::now();

        self.state.tick += 1;
        self.stats.total_ticks += 1;

        // Update Chronobiology
        let bio = Biorhythm::current();
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
            self.process_mesh();
            self.process_sensors();
            self.auto_emit_wisdom();
            self.emit_heartbeat();
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
            if self.state.tick % 100 == 0 {
                let load = self.state.thermodynamic_load;
                self.memory_coordinator.prune_memories(&mut self.episodic_memory, load);
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
    fn process_dream(&mut self) -> Option<MindOutput> {
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
                    self.working_memory_sources[i] = match (s1, s2) {
                        (MemorySource::ActionFeedback, _) | (_, MemorySource::ActionFeedback) => MemorySource::ActionFeedback,
                        (MemorySource::WebResearch, _) | (_, MemorySource::WebResearch) => MemorySource::WebResearch,
                        (MemorySource::UserInteraction, _) | (_, MemorySource::UserInteraction) => MemorySource::UserInteraction,
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
            rand::Rng::gen(rng)
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

            // Update current thought via EMA blend (not bind — bind is multiply,
            // which is permanently zero when starting from zero vector).
            if self.state.current_thought.norm() < f32::EPSILON {
                // First perception: adopt input directly
                self.state.current_thought = input.content.clone();
            } else {
                // Exponential moving average: 70% retain, 30% new input
                self.state
                    .current_thought
                    .lerp_in_place(&input.content, 0.7, 0.3);
            }

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

        let pairs = self.working_memory.len() * (self.working_memory.len() - 1) / 2;
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
        }
    }

    /// Generate output if consciousness is above threshold.
    pub(crate) fn generate_output(&mut self) -> Option<MindOutput> {
        if self.state.consciousness_level < self.config.min_consciousness {
            return None;
        }

        if self.state.tick.is_multiple_of(10) && !self.working_memory.is_empty() {
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
        for msg in inbox {
            sc.observe_agent(&msg.agent_id, &msg.behavior, &msg.context);
            observed += 1;

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
            self.mesh_inbox
                .sort_by(|a, b| a.payload_type.priority().cmp(&b.payload_type.priority()));
            self.mesh_inbox.drain(..excess);
            self.mesh_stats.packets_dropped += excess as u64;
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
                bytes_tx = t.stats.bytes_sent,
                bytes_rx = t.stats.bytes_received,
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

        for packet in &inbox {
            // Dedup check: skip if we've already seen this (source_id, sequence) pair
            let key = (packet.source_id, packet.sequence, packet.payload_type as u8);
            if self.mesh_seen_packets.contains(&key) {
                self.mesh_stats.packets_deduplicated += 1;
                continue;
            }
            if self.mesh_seen_packets.len() >= super::MESH_DEDUP_RING_SIZE {
                self.mesh_seen_packets.remove(0);
            }
            self.mesh_seen_packets.push(key);

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
                            hf.receive_peer_state(peer_id, affect);
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
            }
        }

        // Update receive-side telemetry
        self.mesh_stats.wisdom_received += wisdom_count;
        self.mesh_stats.heartbeats_received += heartbeat_count;
        self.mesh_stats.affective_received += affective_count;
        self.mesh_stats.gradients_received += gradient_count;
        self.mesh_stats.bytes_received +=
            inbox.len() as u64 * crate::swarm::mesh::WISDOM_PACKET_SIZE as u64;

        tracing::debug!(
            target: "symthaea::mind::mesh",
            total = inbox.len(),
            wisdom = wisdom_count,
            heartbeat = heartbeat_count,
            affective = affective_count,
            gradient = gradient_count,
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
                let tags_json = serde_json::to_string(&reading.tags).unwrap_or_else(|_| "[]".to_string());
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
                    self.mesh_outbox
                        .push(crate::swarm::mesh::MeshOutbound { packet });
                    self.mesh_gradient_sequence = self.mesh_gradient_sequence.wrapping_add(1);
                    self.mesh_stats.gradients_sent += 1;
                    self.mesh_stats.bytes_sent += crate::swarm::mesh::WISDOM_PACKET_SIZE as u64;
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

        self.mesh_outbox
            .push(crate::swarm::mesh::MeshOutbound { packet });
        self.mesh_affective_last_tick = self.state.tick;
        self.mesh_affective_sequence = self.mesh_affective_sequence.wrapping_add(1);
        self.mesh_stats.affective_sent += 1;
        self.mesh_stats.bytes_sent += crate::swarm::mesh::WISDOM_PACKET_SIZE as u64;

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
            if let Some(aggregated) = federated.aggregate() {
                let lr = 0.01f32;
                federated.apply_gradient(&aggregated, lr);

                tracing::info!(
                    target: "symthaea::mind::federated",
                    round = federated.aggregation_round(),
                    "Applied federated gradient aggregation"
                );
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
    }
}
