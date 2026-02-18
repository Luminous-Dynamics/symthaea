//! Tick processing for the Continuous Mind.
//!
//! Contains the main cognitive cycle (`tick()`), dream processing,
//! input handling, consciousness updates, and output generation.

use crate::chronobiology::{Biorhythm, CircadianPhase};
use symthaea_core::hdc::ContinuousHV;

use super::utils::permute_hv;
use super::{ContinuousMind, Goal, InputType, MindOutput, OutputType};

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
        // social signals and federated gradients from peers should never be ignored.
        self.process_federated();
        self.process_social();
        self.sync_iroh_bridge();

        // Check for Dream State
        let should_dream = bio.phase == CircadianPhase::Night
            && self.state.cognitive_load < 0.3
            && self.input_queue.is_empty();

        self.state.is_dreaming = should_dream;

        if self.state.is_dreaming {
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
            } else {
                let evicted = self.working_memory.remove(0);
                let arrival_tick = self.working_memory_ticks.remove(0);
                let steps_survived = self.state.tick.saturating_sub(arrival_tick);
                self.evicted_items.push((evicted, steps_survived));
                self.working_memory.push(input.content.clone());
                self.working_memory_ticks.push(self.state.tick);
            }

            self.state.current_thought = self.state.current_thought.bind(&input.content);

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

        // Step 2: Update self-model with current mind state every 10 ticks
        if self.state.tick.is_multiple_of(10) {
            sc.update_self_model(
                self.state.current_thought.clone(),
                self.state.current_thought.clone(),
                self.state.current_thought.clone(),
            );
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
        }
    }
}
