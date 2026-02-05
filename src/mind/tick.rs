//! Tick processing for the Continuous Mind.
//!
//! Contains the main cognitive cycle (`tick()`), dream processing,
//! input handling, consciousness updates, and output generation.

use symthaea_core::hdc::RealHV;
use crate::chronobiology::{Biorhythm, CircadianPhase};

use super::{ContinuousMind, Goal, InputType, MindOutput, OutputType};
use super::utils::permute_hv;

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
        self.stats.avg_consciousness =
            (self.stats.avg_consciousness * (self.stats.total_ticks - 1) as f64
                + self.state.consciousness_level) / self.stats.total_ticks as f64;

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
                    let bundled = RealHV::bundle(&[
                        self.working_memory[i].clone(),
                        self.working_memory[i + 1].clone(),
                    ]);
                    self.working_memory[i] = bundled;
                    self.working_memory.remove(i + 1);
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
            a.priority.partial_cmp(&b.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        while let Some(input) = self.input_queue.pop() {
            self.stats.inputs_processed += 1;

            if self.working_memory.len() < self.config.working_memory_capacity {
                self.working_memory.push(input.content.clone());
            } else {
                self.working_memory.remove(0);
                self.working_memory.push(input.content.clone());
            }

            self.state.current_thought = self.state.current_thought.bind(&input.content);

            match input.input_type {
                InputType::Goal => {
                    let goal = Goal {
                        id: format!("goal_{}", self.goals.len()),
                        description: input.metadata.get("description").cloned().unwrap_or_default(),
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
                content: format!("Thinking about {} items in working memory", self.working_memory.len()),
                embedding: self.state.current_thought.clone(),
                confidence: self.state.consciousness_level as f32,
                emotional_tone: self.state.emotional_valence,
            });
        }

        None
    }
}
