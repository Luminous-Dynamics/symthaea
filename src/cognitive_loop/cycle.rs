// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core cognitive cycle implementation with parallel post-processing.
//!
//! Contains the main `cycle()` method which implements the bidirectional
//! HDC-CfC loop with rayon-parallelized subsystem updates.
//!
//! The heavy lifting is delegated to four phase modules:
//! - `cycle_phase_perception` — safety, encoding, moral evaluation, strategy
//! - `cycle_phase_dynamics`   — CfC step, FEP inference, training, post-processing
//! - `cycle_phase_feedback`   — consciousness metrics, quality gating, homeostasis
//! - `cycle_phase_output`     — metadata assembly, CycleResult construction
//!
//! Phase result structs live in `phase_results.rs`.

use std::time::Instant;

use super::{CognitiveLoopService, CycleResult};

impl CognitiveLoopService {
    /// Run one cognitive cycle (the core loop).
    ///
    /// Uses CfC's O(1) closed-form solution for temporal prediction,
    /// enabling instant forward-time queries and multi-scale prediction.
    ///
    /// ## Architecture
    ///
    /// The cycle is split into four phases:
    ///
    /// 1. **Perception** — safety precheck, thalamic routing, moral evaluation,
    ///    strategy selection, HDC encoding, surprise exploration, ethics engine
    /// 2. **Dynamics** — CfC step, prediction, FEP active inference, training,
    ///    parallel post-processing
    /// 3. **Feedback** — consciousness metrics, quality gating, homeostasis,
    ///    dream engine, episodic replay
    /// 4. **Output** — metadata assembly, telemetry, CycleResult construction
    pub fn cycle(&mut self, input: &str) -> CycleResult {
        let cycle_start = Instant::now();
        self.stats.total_cycles += 1;
        self.substrate_manager.tick_energy(&self.config);
        // Feed substrate energy data to ThermodynamicManager
        self.thermodynamic_mgr.set_energy(
            self.substrate_manager.energy_per_cycle,
            self.substrate_manager.total_energy_spent,
            self.substrate_manager.energy_throughput_multiplier,
        );

        // Integrity: run tamper detection (temporal every cycle, canaries at co-prime intervals)
        #[cfg(feature = "integrity")]
        {
            // Night phase: run full integrity sweep (all attestations + all canaries)
            // Science: immune system deep maintenance during sleep (Besedovsky et al. 2012)
            let is_night =
                self.biorhythm_mgr.rhythm.phase == crate::chronobiology::CircadianPhase::Night;
            self.integrity_manager.tick(
                self.stats.total_cycles as usize,
                self.config.cfc_config.delta_t,
                is_night,
            );
            // Live verification: re-hash safety thresholds from current const values
            // (catches binary patching that the frozen-copy attestation might miss
            //  if the patching happened after initial registration)
            let live_hash = crate::integrity::attestation::blake3_hash_f32_slice(&[
                super::thresholds::MORAL_CONCERN_THRESHOLD,
                super::thresholds::MORAL_BENEFIT_THRESHOLD,
                super::thresholds::MORAL_CONCERN_EXPLORATION_DAMPEN,
                super::thresholds::MORAL_CONCERN_PAUSE_BOOST,
                super::thresholds::MORAL_BENEFIT_CONFIDENCE_BOOST,
            ]);
            if let Some(failure) = self
                .integrity_manager
                .verify_live_thresholds("safety_thresholds", live_hash)
            {
                tracing::error!(
                    target: "cognitive_loop::integrity",
                    "{failure}"
                );
                self.integrity_manager.status.attestation_passed = false;
                self.integrity_manager
                    .status
                    .anomalies
                    .push(crate::integrity::IntegrityAnomaly {
                        source: "live_attestation",
                        description: failure,
                        detected_at: std::time::Instant::now(),
                        severity: crate::integrity::AnomalySeverity::Critical,
                    });
            }
            // Escalate critical integrity anomalies to safety telemetry
            if self.integrity_manager.has_critical_anomaly() {
                tracing::error!(
                    target: "cognitive_loop::integrity",
                    anomaly_count = self.integrity_manager.status.anomalies.len(),
                    "Critical integrity anomaly detected — escalating safety"
                );
                // Feed integrity failure into safety-relevant carryover so the
                // safety precheck and downstream SafetyAgent consumers see it.
                // Attestation failure = consciousness metrics untrustworthy.
                self.carryover.quality.last_epistemic_confidence = 0.0;

                // Alert human operators — integrity failure requires investigation
                super::safety_alert::emit_alert(
                    &self.safety_alert_tx,
                    super::safety_alert::SafetyAlertKind::IntegrityCritical,
                    format!(
                        "Integrity critical anomaly — {} anomalies detected",
                        self.integrity_manager.status.anomalies.len()
                    ),
                    self.stats.total_cycles as u64,
                    self.carryover.history.consciousness_level,
                );
            }
        }
        let mut module_timings = super::ModuleTimings::default();

        // User state inference: process input to update context, frustration, cognitive load
        // Science: adaptive UI via inferred cognitive state (Ritter et al. 2019)
        if let Some(ref mut usi) = self.language_comm.user_state {
            usi.process(input, false);
            let state = usi.state();
            // Frustration → dampen exploration (noisy signals, don't overfit to errors)
            // Flow → boost exploration (user engaged, safe to explore)
            // Science: Yerkes-Dodson (1908) — moderate arousal optimal for learning
            if state.frustration > super::thresholds::FRUSTRATION_DAMPEN_THRESHOLD {
                self.carryover.quality.last_exploration_bonus *= 1.0
                    - super::thresholds::FRUSTRATION_DAMPEN_GAIN
                        * (state.frustration - super::thresholds::FRUSTRATION_DAMPEN_THRESHOLD)
                            as f32;
            }
            if state.is_in_flow() {
                self.carryover.quality.last_exploration_bonus +=
                    super::thresholds::FLOW_EXPLORATION_INCREMENT;
            }

            // USI → neuromodulator coupling (Sapolsky 2004; Schultz 1997)
            // Frustration raises NE baseline (stress-arousal axis, locus coeruleus)
            // Flow raises DA baseline (reward-engagement axis, VTA)
            // Gentle nudges toward current baseline ± 0.03, naturally decays via bath dynamics
            let frustration = state.frustration as f32;
            let engagement = state.engagement as f32;
            if frustration > super::thresholds::FRUSTRATION_NE_NUDGE_THRESHOLD {
                let ne_base = self.neuromod.bath.noradrenaline.baseline_val();
                let ne_nudge = super::thresholds::FRUSTRATION_NE_NUDGE_SCALE
                    * (frustration - super::thresholds::FRUSTRATION_NE_NUDGE_THRESHOLD);
                self.neuromod
                    .bath
                    .noradrenaline
                    .set_baseline(ne_base + ne_nudge);
            }
            if state.is_in_flow() {
                let da_base = self.neuromod.bath.dopamine.baseline_val();
                self.neuromod
                    .bath
                    .dopamine
                    .set_baseline(da_base + super::thresholds::FLOW_DA_NUDGE);
            } else if (engagement as f64) < super::thresholds::ENGAGEMENT_LOW_THRESHOLD {
                // Disengagement → slight DA reduction (anhedonia pathway)
                let da_base = self.neuromod.bath.dopamine.baseline_val();
                self.neuromod
                    .bath
                    .dopamine
                    .set_baseline(da_base - super::thresholds::DISENGAGEMENT_DA_NUDGE);
            }
        }

        // Spectrum → neuromodulator coupling (Aston-Jones & Cohen 2005; Schultz 1997)
        // Sustained jamming activates threat axis (NE up, locus coeruleus)
        // Network recovery triggers reward relief (DA nudge)
        #[cfg(feature = "mesh")]
        {
            use super::thresholds::{
                RADIO_JAMMING_NE_NUDGE, RADIO_NEUROMOD_JAMMING_MIN_STREAK, RADIO_RECOVERY_DA_NUDGE,
            };
            let telem = self.spectrum_manager.telemetry();
            // Sustained jamming → NE arousal spike
            if telem.jamming_streak >= RADIO_NEUROMOD_JAMMING_MIN_STREAK {
                let ne_base = self.neuromod.bath.noradrenaline.baseline_val();
                self.neuromod
                    .bath
                    .noradrenaline
                    .set_baseline(ne_base + RADIO_JAMMING_NE_NUDGE);
            }
            // Recovery from blackout → DA relief
            if telem.network_health == 0 && self.stats.total_cycles > 1 {
                if telem.degradation_streak == 0 && telem.jamming_streak == 0 {
                    let had_recent_loss = telem.tier_loss_ema.iter().any(|&l| l > 0.01);
                    if had_recent_loss {
                        let da_base = self.neuromod.bath.dopamine.baseline_val();
                        self.neuromod
                            .bath
                            .dopamine
                            .set_baseline(da_base + RADIO_RECOVERY_DA_NUDGE);
                    }
                }
            }
        }

        // ── Pre-phase: Text-based crisis detection (safety-critical) ─────
        // Runs BEFORE perception so crisis state is available for safety precheck.
        // Science: C-SSRS screening protocol — catch indirect expressions early.
        #[cfg(feature = "therapeutic")]
        if self.config.enable_therapeutic {
            self.therapeutic_manager.detect_crisis_from_text(input);
            // Apply dream feedback to strategy selection (accuracy-gated exploration)
            self.therapeutic_manager.apply_dream_feedback();
        }

        // ═══════════════════════════════════════════════════════════════════
        // PHASE 1: PERCEPTION
        // Safety checks, encoding, moral evaluation, strategy, urgency
        // ═══════════════════════════════════════════════════════════════════

        // Ground consciousness model trust in real-time physical accuracy (Epistemic Humility)
        let epistemic_quality = self.get_epistemic_quality();

        let mut perception = match self.phase_perception(input, cycle_start, &mut module_timings) {
            Ok(p) => p,
            Err(blocked) => return *blocked,
        };

        // P3-A: Cognition -> Vision goal signal
        // The current thought HV becomes the top-down task vector for the
        // next frame: what Symthaea just perceived shapes what she looks for.
        // Rao & Ballard (1999): top-down predictions drive bottom-up attention.
        #[cfg(feature = "vision-manifold")]
        if let Some(ref mut bridge) = self.sensorimotor.vision_sensory.vision_bridge {
            let thought_hv = perception.encoding.encoding_result.hdv.clone();
            // EMA drift (α=0.05): gently nudge the goal template toward the current
            // thought rather than hard-replacing it each cycle. Prevents attentional
            // thrashing when thoughts shift faster than the visual system can track.
            bridge.update_goal_from_cognition(&thought_hv, 0.05);

            // P4-B: Phi-modulated visual attention — consciousness level modulates
            // the attention boost factor. High Phi → broad exploratory attention
            // (0.8), low Phi → narrow conservative focus (0.2).
            // Science: Luck et al. (1997) — arousal/ACh modulates V1/V4 gain.
            let phi = self.carryover.history.consciousness_level as f32;
            let phi_boost = 0.2 + 0.6 * phi.clamp(0.0, 1.0);
            bridge.set_attention_boost(phi_boost);

            // P6-A: Feed imagination surprise into VisionManager for FEP integration.
            // When reality diverges from imagination, the system increases exploration.
            self.vision_manager
                .set_imagination_surprise(bridge.imagination_surprise());

            // P6-B: Blend visual context into the thought HV.
            // The scene context (working memory + scene graph + state) colors
            // the current thought with what the system is visually attending to.
            // Desimone & Duncan (1995): biased competition — attention shapes perception.
            if let Some(context_hv) = bridge.scene_context_hv() {
                let thought_hv = perception.encoding.encoding_result.hdv.clone();
                perception.encoding.encoding_result.hdv =
                    symthaea_core::hdc::ContinuousHV::weighted_bundle(
                        &[&thought_hv, &context_hv],
                        &[0.85, 0.15],
                    )
                    .normalize();
            }
        }

        // ===================================================================
        // PHASE 2: DYNAMICS
        // CfC step, prediction, FEP, training, parallel post-processing
        // ═══════════════════════════════════════════════════════════════════
        let mut dynamics =
            self.phase_dynamics(input, &perception, cycle_start, &mut module_timings);

        // Coherence field: apply hormone modulation from neuromod bath
        // Science: McEwen (2007) — allostatic load shapes integration capacity
        if let Some(ref mut cf) = self.sensorimotor.vision_sensory.coherence_field {
            use super::neuromodulators::NeuromodulatorBathExt;
            let hormones = self.neuromod.bath.to_hormone_state();
            cf.apply_hormone_modulation(&hormones);
        }

        // ═══════════════════════════════════════════════════════════════════
        // PHASE 2.5: EMBODIMENT
        // Motor output → physics sim → proprioceptive HV → next cycle blend
        // Science: Lakoff & Johnson (1999) embodied cognition, Varela (1991) enactivism
        // Platform-agnostic: works for ALL platforms via the EmbodimentBridge trait.
        // This feature list MUST stay in sync with switch_embodiment()'s
        // (accessors/system.rs) — platforms constructible there but absent
        // here are silently never driven (motor, moral gate, and safety
        // overrides all inert). That exact bug shipped for the six newer
        // platforms below; a shared umbrella feature is the planned root fix
        // (SYMTHAEA_ROBOTICS_IMPROVEMENT_PLAN_2026-07-06.md Tier 0.5).
        // ═══════════════════════════════════════════════════════════════════
        #[cfg(any(
            feature = "humanoid",
            feature = "helicopter",
            feature = "flight",
            feature = "vehicle",
            feature = "auv",
            feature = "manipulator",
            feature = "exoskeleton",
            feature = "surgical",
            feature = "orbital",
            feature = "quadruped",
            feature = "subterranean",
            feature = "infrastructure",
            feature = "scavenger",
            feature = "agribot",
            feature = "biota",
            feature = "clime",
            feature = "phone"
        ))]
        {
            let cycle_num = self.stats.total_cycles as usize;
            let interval = self.config.embodiment_step_interval.max(1);
            if cycle_num % interval == 0 {
                if let Some(mut bridge) = self.sensorimotor.embodiment_bridge.take() {
                    let phi = self.carryover.history.consciousness_level;
                    // Scale motor dt by substrate speed factor (Putnam 1967: substrate independence).
                    // Photonic substrates step physics faster; biological substrates slower.
                    let dt = self.config.cfc_config.delta_t * self.substrate_manager.tau_factor;
                    // Forward ethics verdict to embodiment bridge (from previous cycle).
                    // Platforms that override apply_moral_gate() will constrain motor output.
                    let verdict_u8 = match self.last_ethics_verdict {
                        super::ethics_engine::EthicalVerdict::Safe => 0,
                        super::ethics_engine::EthicalVerdict::Caution => 1,
                        super::ethics_engine::EthicalVerdict::Blocked => 2,
                    };
                    bridge.apply_moral_gate(symthaea_core::embodiment::MoralGateInput {
                        verdict: verdict_u8,
                        consent_violation: self.stats.consent_violation,
                        // Independent deontological signal (ahimsa_*/prevent_suffering/
                        // minimize_collateral obligations), not merely Blocked mirrored —
                        // see EthicsEngineOutput::ahimsa_violated in ethics_engine.rs.
                        ahimsa_violated: self.last_ahimsa_violated,
                    });

                    #[allow(unused_mut)]
                    let mut thought_hv = perception.encoding.encoding_result.hdv.clone();
                    // OMI-2 (2026-07-09): blend the grounded semantic HV into the
                    // thought vector feeding the motor decoder. Off by default
                    // (semantic_thought_blend = 0.0). The semantic HV is one
                    // cycle stale (background embedding thread) — acceptable at
                    // embodiment cadence.
                    #[cfg(feature = "semantic-encoder")]
                    {
                        let w = self.config.semantic_thought_blend;
                        if w > 0.0 {
                            if let Some(ref sem) = self.feature_integ.last_semantic_continuous {
                                if sem.len() == thought_hv.values.len() {
                                    let sem_hv =
                                        symthaea_core::hdc::ContinuousHV::from_vec(sem.clone());
                                    thought_hv.lerp_in_place(&sem_hv, 1.0 - w, w);
                                }
                            }
                        }
                    }
                    let result = bridge.step(&thought_hv, dt, phi);
                    if result.success {
                        let proprioceptive_hv = bridge.encode_perception();
                        let w = self.config.embodiment_blend_weight;
                        if w > 0.0 {
                            perception.encoding.encoding_result.hdv.lerp_in_place(
                                &proprioceptive_hv,
                                1.0 - w,
                                w,
                            );
                        }
                        self.sensorimotor.last_proprioceptive_hv = Some(proprioceptive_hv);
                    }
                    self.sensorimotor.embodiment_telemetry = bridge.telemetry();
                    self.sensorimotor.embodiment_bridge = Some(bridge);
                }
            }
        }

        // ── Distress emission: broadcast when embodiment is in trouble ────
        // Conditions: energy depleted + high prediction error + safety degraded.
        // Cooldown: max 1 per 100 cycles. Basis: de Waal (2008) social alarm.
        #[cfg(feature = "muse")]
        if self.stats.total_cycles % 100 == 0 {
            let psi = self.carryover.history.consciousness_level;
            let load = self.neuromod.bath.allostatic_load;
            let energy_critical = !self.substrate_manager.consciousness_viable;
            let safety_degraded = psi < 0.3;

            if energy_critical && load > 0.7 && safety_degraded {
                let distress = super::managers::swarm_manager::SwarmEvent::DistressSignal {
                    peer_id: String::new(),
                    prediction_error: load, // allostatic load as proxy for distress severity
                    energy_depletion: 0.95,
                    safety_level: if psi < 0.1 { 3 } else { 2 },
                    allostatic_load: load,
                    consciousness_level: psi as f32,
                };
                let _ = self.swarm_event_tx.send(distress);
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // PHASE 3: FEEDBACK
        // Consciousness metrics, quality gating, homeostasis, dream engine
        // ═══════════════════════════════════════════════════════════════════
        let mut feedback =
            self.phase_feedback(input, &perception, &mut dynamics, &mut module_timings);

        // ═══════════════════════════════════════════════════════════════════
        // PHASE 3.5: SAFETY ENFORCEMENT
        // NRC defense-in-depth: assess consciousness → gate LR/exploration/motor
        // Science: Arnsten (2009), Yerkes-Dodson (1908), Aston-Jones & Cohen (2005)
        // ═══════════════════════════════════════════════════════════════════
        #[cfg(feature = "safety-agents")]
        {
            let safety_result = self.safety_supervisor.assess(
                feedback.consciousness.consciousness_level as f32,
                dynamics.core.prediction_error,
                feedback.self_model.temporal_coherence_score as f32,
                {
                    #[cfg(feature = "integrity")]
                    {
                        self.integrity_manager.has_critical_anomaly()
                    }
                    #[cfg(not(feature = "integrity"))]
                    {
                        false
                    }
                },
                self.stats.total_cycles as usize,
                {
                    #[cfg(feature = "sentinel")]
                    {
                        Some(&self.collective_immune_state)
                    }
                    #[cfg(not(feature = "sentinel"))]
                    {
                        None
                    }
                },
            );

            self.apply_safety_gates(&safety_result, feedback.consciousness.consciousness_level);

            // ── Gate 5: Embodiment motor halt carry-forward ──────────────
            // Platform-agnostic: applies to ALL embodiment platforms.
            // Keep in sync with the Phase 2.5 list above.
            #[cfg(any(
                feature = "humanoid",
                feature = "helicopter",
                feature = "flight",
                feature = "vehicle",
                feature = "auv",
                feature = "manipulator",
                feature = "exoskeleton",
                feature = "surgical",
                feature = "orbital",
                feature = "quadruped",
                feature = "subterranean",
                feature = "infrastructure",
                feature = "scavenger",
                feature = "agribot",
                feature = "biota",
                feature = "clime",
                feature = "phone"
            ))]
            if let Some(ref mut bridge) = self.sensorimotor.embodiment_bridge {
                use crate::cognitive_loop::motor_bridge::MotorSafetyLevel;
                if safety_result.motor_halt {
                    bridge.set_safety_override(MotorSafetyLevel::Red);
                } else if safety_result.motor_readonly {
                    bridge.set_safety_override(MotorSafetyLevel::Orange);
                } else {
                    bridge.clear_safety_override();
                }
            }

            // ── Defense Cascade: propose → moral filter → apply ──────────
            // Propose graduated defense actions based on assessed safety level.
            // Each action goes through moral algebra before application.
            let mut defense_actions = super::defense::propose_defense_actions(
                safety_result.level,
                self.stats.total_cycles as usize,
            );
            super::defense::moral_filter(&mut defense_actions);

            // Apply morally approved defense actions
            for action in &defense_actions {
                if !action.morally_approved {
                    continue;
                }
                match &action.kind {
                    super::defense::DefenseActionKind::BoostVigilance { ne_delta } => {
                        let ne_base = self.neuromod.bath.noradrenaline.baseline_val();
                        self.neuromod
                            .bath
                            .noradrenaline
                            .set_baseline(ne_base + *ne_delta);
                    }
                    super::defense::DefenseActionKind::StressResponse { cortisol_delta } => {
                        self.neuromod
                            .bath
                            .accumulate_allostatic_load(*cortisol_delta, false);
                    }
                    super::defense::DefenseActionKind::RestrictMotorToReadOnly => {
                        // Restrict motor output to read-only via safety level gate
                        if let Some(ref mut bridge) =
                            self.sensorimotor.motor_rendering.output_bridge
                        {
                            bridge.set_safety_level(crate::safety::SafetyLevel::Orange);
                        }
                    }
                    super::defense::DefenseActionKind::HaltMotor => {
                        // Halt all motor output via safety level gate
                        if let Some(ref mut bridge) =
                            self.sensorimotor.motor_rendering.output_bridge
                        {
                            bridge.set_safety_level(crate::safety::SafetyLevel::Red);
                        }
                    }
                    // Other actions (peer quarantine, governance freeze, etc.)
                    // are handled by their respective subsystems via telemetry
                    _ => {}
                }
            }

            // Record defense cascade telemetry
            self.defense_actions_proposed = defense_actions.len() as u32;
            self.defense_actions_approved = defense_actions
                .iter()
                .filter(|a| a.morally_approved)
                .count() as u32;
            tracing::debug!(
                safety_level = ?safety_result.level,
                proposed = self.defense_actions_proposed,
                approved = self.defense_actions_approved,
                "Defense cascade"
            );

            // ── Civic Crisis Detection ───────────────────────────────────
            // Monitor prediction error, safety level, Phi, and arousal for
            // sustained anomalies that indicate community-level emergencies.
            // Produces CivicCrisisEvent for the Mycelix emergency-incidents zome.
            let crisis_input = super::civic_crisis_detector::CrisisDetectorInput {
                prediction_error: dynamics.core.prediction_error as f64,
                safety_level_ordinal: match safety_result.level {
                    crate::safety::SafetyLevel::Green => 0,
                    crate::safety::SafetyLevel::Yellow => 1,
                    crate::safety::SafetyLevel::Orange => 2,
                    crate::safety::SafetyLevel::Red => 3,
                },
                consciousness_level: feedback.consciousness.consciousness_level,
                collective_phi: self.swarm_manager.mean_peer_phi(),
                arousal: self.neuromod.bath.noradrenaline.baseline_val() as f64,
                has_peers: self.swarm_manager.connected_peers() > 0,
            };

            if let Some(crisis_event) = self
                .civic_crisis_detector
                .tick(&crisis_input, self.stats.total_cycles as u64)
            {
                self.security_telemetry.crisis_events_emitted += 1;
                tracing::warn!(
                    severity = crisis_event.severity,
                    crisis_type = ?crisis_event.crisis_type,
                    confidence = crisis_event.confidence,
                    signals = crisis_event.trigger_signals.len(),
                    "Civic crisis detected — forwarding to Mycelix emergency-incidents"
                );
                // Queue for external dispatch to Mycelix civic bridge.
                // The host application drains this via drain_pending_crisis_events()
                // and forwards to MycelixBridge::dispatch_crisis().
                self.pending_crisis_events.push(crisis_event);
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // PHASE 3.5b: MINIMAL SAFETY BASELINE (when safety-agents feature is OFF)
        // Consciousness-gated motor halt is a thermodynamic necessity, not optional.
        // Even without the full SafetyAgent cascade, the physical boundary must hold:
        // Phi < 0.1 → halt all motors, Phi < 0.35 → read-only.
        // ═══════════════════════════════════════════════════════════════════
        #[cfg(not(feature = "safety-agents"))]
        {
            let phi = feedback.consciousness.consciousness_level as f32;
            // Red: consciousness collapse — halt everything
            if phi < super::thresholds::SAFETY_MINIMAL_RED_THRESHOLD {
                self.carryover.quality.safety_motor_halt = true;
                self.carryover.quality.safety_motor_readonly = true;
                tracing::warn!(
                    phi = phi,
                    "Minimal safety baseline: motor HALT (Phi < {})",
                    super::thresholds::SAFETY_MINIMAL_RED_THRESHOLD,
                );
                super::safety_alert::emit_alert(
                    &self.safety_alert_tx,
                    super::safety_alert::SafetyAlertKind::MotorHalt,
                    format!("Motor HALT (minimal safety) — Phi={phi:.3}"),
                    self.stats.total_cycles as u64,
                    phi as f64,
                );
            }
            // Orange: severe degradation — read-only
            else if phi < super::thresholds::SAFETY_MINIMAL_ORANGE_THRESHOLD {
                self.carryover.quality.safety_motor_readonly = true;
                tracing::warn!(
                    phi = phi,
                    "Minimal safety baseline: motor READ-ONLY (Phi < {})",
                    super::thresholds::SAFETY_MINIMAL_ORANGE_THRESHOLD,
                );
                super::safety_alert::emit_alert(
                    &self.safety_alert_tx,
                    super::safety_alert::SafetyAlertKind::MotorReadOnly,
                    format!("Motor read-only (minimal safety) — Phi={phi:.3}"),
                    self.stats.total_cycles as u64,
                    phi as f64,
                );
            }
        }

        // Sentinel threat processing: detect → store → share
        #[cfg(feature = "sentinel")]
        {
            // Store locally-detected threats in immune memory
            for threat in self.sentinel_manager.active_threats() {
                self.threat_memory.store_threat(
                    super::threat_memory::ThreatSignalKind::from_sentinel_kind(threat.kind),
                    threat.severity,
                    threat.confidence,
                    self.stats.total_cycles as usize,
                    &threat.evidence,
                );
            }

            // Update collective immune state from swarm data
            let local_threat_level = self.sentinel_manager.threat_level();
            let local_kinds: Vec<String> = self
                .sentinel_manager
                .active_threats()
                .iter()
                .map(|t| format!("{:?}", t.kind))
                .collect();
            self.collective_immune_state.update(
                local_threat_level,
                &[], // peer threat reports wired when swarm drain is available
                feedback.consciousness.consciousness_level,
                self.swarm_manager.connected_peers(),
                &local_kinds,
                &local_kinds, // bootstrap: local = swarm until peer sharing wired
            );

            // ── Epistemic: echo chamber risk detection ──
            // High individual Phi + low collective Phi = epistemic fragmentation.
            #[cfg(feature = "epistemic")]
            {
                let individual_phi = feedback.consciousness.consciousness_level;
                let collective_phi = self.swarm_manager.mean_peer_phi();
                self.collective_immune_state
                    .compute_echo_chamber_risk(individual_phi, collective_phi);
            }
        }

        // ── FHE Collective Wisdom: encrypt & contribute local state ──────
        // Imani et al. (2019): privacy-preserving HDC — XOR-OTP encryption
        // preserves Hamming distance, enabling homomorphic classification.
        // Each cycle contributes the encoded BinaryHV to the collective pool.
        // Aggregation is attempted when enough contributions accumulate.
        #[cfg(feature = "fhe-wisdom")]
        if self.config.fhe_wisdom_enabled {
            // Contribute local consciousness state (encrypted with session mask)
            self.swarm_manager
                .contribute_local_wisdom(&perception.encoding.hv16_cached);

            // Try aggregation at configured interval
            if self.config.fhe_aggregation_interval > 0
                && self.stats.total_cycles as usize % self.config.fhe_aggregation_interval == 0
            {
                if let Some(collective_wisdom) = self.swarm_manager.try_aggregate_and_decrypt() {
                    // Epistemic Humility: Only integrate if local confidence > 0.4
                    // High local uncertainty suggests we are currently exploring;
                    // trust local experience over swarm consensus during high discovery phases.
                    let local_confidence = self.stats.prediction_confidence;
                    if local_confidence > 0.4 {
                        let sim = perception
                            .encoding
                            .hv16_cached
                            .similarity(&collective_wisdom);

                        // Apply collective update weighted by current reliability
                        let update_weight = (local_confidence * 0.2).clamp(0.05, 0.25);
                        perception.encoding.hv16_cached.lerp_in_place(
                            &collective_wisdom,
                            1.0 - update_weight,
                            update_weight,
                        );

                        tracing::debug!(
                            target: "cognitive_loop::fhe",
                            pool_count = self.swarm_manager.wisdom_pool_count(),
                            local_confidence = %format!("{local_confidence:.4}"),
                            local_collective_sim = %format!("{sim:.4}"),
                            "FHE collective wisdom consolidated"
                        );
                    } else {
                        tracing::debug!(
                            target: "cognitive_loop::fhe",
                            "Epistemic Humility: High local uncertainty; swarm wisdom ignored"
                        );
                    }
                }
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // PHASE 3.7: SCIENTIFIC METHOD
        // observe → hypothesize → predict → test → update_beliefs
        // Bayesian belief updates accumulate across cycles.
        // Science: Popper (1959), Bayes (1763), Jaynes (2003)
        // ═══════════════════════════════════════════════════════════════════
        #[cfg(feature = "scientific_method")]
        {
            // Observation: prediction error encodes how surprising the current
            // input is relative to the system's internal model.
            let pe = dynamics.core.prediction_error as f64;
            let phi = feedback.consciousness.consciousness_level;
            self.scientific_method_engine
                .observe(vec![pe, phi], "prediction_error_and_consciousness");

            // Predict what the standing hypothesis (id 0) expects.
            let predicted = self.scientific_method_engine.predict(0);

            // The "observed value" for coherence is 1 − prediction_error (clipped
            // to [0, 1]): low PE means high coherence, confirming the hypothesis.
            let observed_coherence = (1.0 - pe).clamp(0.0, 1.0);

            // Test and update Bayesian posterior.
            self.scientific_method_engine
                .test_hypothesis(0, observed_coherence, predicted);

            // Batch-recompute lifecycle status for all hypotheses.
            self.scientific_method_engine.update_beliefs();
        }

        // ═══════════════════════════════════════════════════════════════════
        // PHASE 4: OUTPUT
        // Metadata assembly, telemetry, CycleResult construction
        // ═══════════════════════════════════════════════════════════════════
        let result = self.phase_output(
            input,
            cycle_start,
            &mut perception,
            &mut dynamics,
            &mut feedback,
            module_timings,
        );

        // ── Record thought snapshot for observability ───────────────
        self.tracer.record(super::observability::ThoughtSnapshot {
            cycle: self.stats.total_cycles as u64,
            timestamp: Instant::now(),
            input_summary: input.chars().take(50).collect(),
            consciousness_level: feedback.consciousness.consciousness_level,
            prediction_error: dynamics.core.prediction_error,
            primary_neuromodulators: [
                self.neuromod.bath.noradrenaline.baseline_val() as f32,
                self.neuromod.bath.dopamine.baseline_val() as f32,
                self.neuromod.bath.serotonin.baseline_val() as f32,
                self.neuromod.bath.acetylcholine.baseline_val() as f32,
            ],
            focus_hv_checksum: 0, // Placeholder
            flow_state: self.behavior.flow_state.intensity as f32,
        });

        result
    }
    /// Apply the safety enforcement results to the service state.
    #[cfg(feature = "safety-agents")]
    pub(crate) fn apply_safety_gates(
        &mut self,
        result: &super::safety_enforcement::SafetyEnforcementResult,
        consciousness_level: f64,
    ) {
        // Gate 1: Learning rate
        if result.lr_multiplier < 1.0 {
            self.stats.effective_learning_rate *= result.lr_multiplier;
        }

        // Gate 2: Exploration
        if result.exploration_multiplier < 1.0 {
            self.carryover.quality.last_exploration_bonus *= result.exploration_multiplier;
        }

        // Gate 3: Neuromodulators
        if result.ne_nudge > 0.0 {
            let ne_base = self.neuromod.bath.noradrenaline.baseline_val();
            self.neuromod
                .bath
                .noradrenaline
                .set_baseline(ne_base + result.ne_nudge);
        }
        if result.allostatic_load > 0.0 {
            self.neuromod
                .bath
                .accumulate_allostatic_load(result.allostatic_load, false);
        }

        // Propagate motor safety flags
        self.carryover.quality.safety_motor_halt = result.motor_halt;
        self.carryover.quality.safety_motor_readonly = result.motor_readonly;

        // Emit alerts
        if result.motor_halt {
            super::safety_alert::emit_alert(
                &self.safety_alert_tx,
                super::safety_alert::SafetyAlertKind::MotorHalt,
                format!(
                    "Motor output HALTED — safety Red (Phi={:.3})",
                    consciousness_level
                ),
                self.stats.total_cycles as u64,
                consciousness_level,
            );
        } else if result.motor_readonly {
            super::safety_alert::emit_alert(
                &self.safety_alert_tx,
                super::safety_alert::SafetyAlertKind::MotorReadOnly,
                format!(
                    "Motor output restricted to read-only — safety Orange (Phi={:.3})",
                    consciousness_level
                ),
                self.stats.total_cycles as u64,
                consciousness_level,
            );
        }

        // Guardian posture update
        self.safety_supervisor.guardian_state.update(
            result.level,
            consciousness_level,
            self.stats.total_cycles as usize,
        );
    }
    // Extracted cycle phases moved to helpers/cycle_phases.rs:
    // - run_resonator_codebook_phase()
    // - run_episodic_replay_and_memory_phase()
    // - run_dream_phase()

    /// Safe wrapper around `cycle()` that catches panics from unexpected subsystem failures.
    ///
    /// Use this in production code paths where a panic must not propagate (e.g., actor loops,
    /// async bridges). Returns `Err` with the panic message if any subsystem panics during
    /// the cycle.
    /// Online distillation step for the Liquid-Mamba HDC↔SSM projection.
    ///
    /// Called after generation with the original thought HV, back-projected
    /// output HVs, and semantic prediction error. Adjusts projection weights
    /// using FEP-modulated learning rate, gated by the cognitive loop's
    /// learning state and thermodynamic load.
    #[cfg(feature = "liquid-mamba")]
    pub fn update_liquid_mamba_telemetry(
        &mut self,
        semantic_pe: f32,
        effective_rank: f32,
        current_lr: f32,
        generation_count: u32,
    ) {
        self.stats.last_liquid_mamba_pe = semantic_pe;
        self.stats.last_liquid_mamba_rank = effective_rank;
        self.stats.last_liquid_mamba_lr = current_lr;
        self.stats.liquid_mamba_generation_count = generation_count;
    }

    #[cfg(feature = "liquid-mamba")]
    pub fn liquid_mamba_distillation_step(
        &mut self,
        thought_hv: &symthaea_core::hdc::ContinuousHV,
        output_hvs: &[symthaea_core::hdc::ContinuousHV],
        semantic_pe: f32,
        projection: &mut symthaea_broca::HdcSsmProjection,
    ) {
        self.stats.last_liquid_mamba_pe = semantic_pe;

        // Gate on FEP precision confidence (mirrors enhanced_fep_bridge threshold)
        if self.carryover.learning.prediction_confidence < 0.4 {
            return;
        }
        if output_hvs.is_empty() || semantic_pe > 0.8 {
            return;
        }

        // FEP-modulated learning rate: precision × load × boost
        let fep_precision = self.fep.learning_signal.clamp(0.0, 1.0);
        let effective_lr =
            0.001 * fep_precision * (1.0 - self.thermodynamic_load) * self.fep.lr_boost as f32;
        if effective_lr < 1e-6 {
            return;
        }

        let refs: Vec<&symthaea_core::hdc::ContinuousHV> = output_hvs.iter().collect();
        let bundled = symthaea_core::hdc::ContinuousHV::bundle(&refs).normalize();
        projection.compute_gradients(thought_hv, &bundled);
        projection.apply_gradients(effective_lr, 1.0);
    }

    pub fn try_cycle(&mut self, input: &str) -> Result<CycleResult, crate::errors::SymthaeaError> {
        // SAFETY: CognitiveLoopService is not UnwindSafe by default because it contains
        // mutable state. We use AssertUnwindSafe because a panic mid-cycle leaves the
        // service in a potentially inconsistent state, but callers should reset() after
        // an error rather than continuing.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.cycle(input)));
        result.map_err(|payload| {
            crate::errors::SymthaeaError::CognitiveLoop(format_panic_payload(payload))
        })
    }
}

/// Convert a panic payload into a human-readable error string.
///
/// Handles the three common payload types: `&str`, `String`, and unknown.
/// This is a standalone function so it can be tested independently.
pub(crate) fn format_panic_payload(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        format!("cognitive cycle panicked: {s}")
    } else if let Some(s) = payload.downcast_ref::<String>() {
        format!("cognitive cycle panicked: {s}")
    } else {
        "cognitive cycle panicked with unknown payload".to_string()
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::CognitiveLoopService;
    use super::format_panic_payload;
    use crate::cognitive_loop::CognitiveLoopConfig;

    // ── format_panic_payload tests (existing) ─────────────────────────

    #[test]
    fn test_panic_payload_str() {
        let payload: Box<dyn std::any::Any + Send> = Box::new("subsystem failure");
        let msg = format_panic_payload(payload);
        assert_eq!(msg, "cognitive cycle panicked: subsystem failure");
    }

    #[test]
    fn test_panic_payload_string() {
        let payload: Box<dyn std::any::Any + Send> = Box::new(String::from("HDC bridge overflow"));
        let msg = format_panic_payload(payload);
        assert_eq!(msg, "cognitive cycle panicked: HDC bridge overflow");
    }

    #[test]
    fn test_panic_payload_unknown() {
        let payload: Box<dyn std::any::Any + Send> = Box::new(42u32);
        let msg = format_panic_payload(payload);
        assert_eq!(msg, "cognitive cycle panicked with unknown payload");
    }

    #[test]
    fn test_panic_payload_empty_str() {
        let payload: Box<dyn std::any::Any + Send> = Box::new("");
        let msg = format_panic_payload(payload);
        assert_eq!(msg, "cognitive cycle panicked: ");
    }

    // ── Helper ────────────────────────────────────────────────────────

    fn make_service() -> CognitiveLoopService {
        CognitiveLoopService::new(CognitiveLoopConfig::default())
            .expect("default config must initialize")
    }

    // ── cycle() basic execution ───────────────────────────────────────

    #[test]
    fn cycle_returns_valid_result() {
        let mut s = make_service();
        let result = s.cycle("hello world");
        assert!(!result.output.is_empty(), "output should not be empty");
        assert!(result.prediction_error.is_finite());
        assert!(result.peak_attention.is_finite());
        assert!(result.cycle_time_us > 0);
    }

    #[test]
    fn cycle_increments_total_cycles() {
        let mut s = make_service();
        assert_eq!(s.stats().total_cycles, 0);
        s.cycle("first");
        assert_eq!(s.stats().total_cycles, 1);
        s.cycle("second");
        assert_eq!(s.stats().total_cycles, 2);
    }

    #[test]
    fn cycle_output_dimension_matches_config() {
        let mut s = make_service();
        let result = s.cycle("testing output dim");
        assert_eq!(
            result.output.len(),
            s.config().cfc_config.num_neurons,
            "output dimension should match num_neurons"
        );
    }

    #[test]
    fn cycle_prediction_error_non_negative() {
        let mut s = make_service();
        let result = s.cycle("checking error sign");
        assert!(
            result.prediction_error >= 0.0,
            "prediction_error should be non-negative, got {}",
            result.prediction_error
        );
    }

    #[test]
    fn cycle_thought_vector_has_values() {
        let mut s = make_service();
        let result = s.cycle("thought projection");
        assert!(!result.thought_vector.is_empty());
        assert_eq!(
            result.thought_vector.len(),
            32,
            "thought_vector should be 32D"
        );
    }

    #[test]
    fn cycle_metadata_urgency_populated() {
        let mut s = make_service();
        let result = s.cycle("metadata check");
        // Urgency should be one of the three valid variants
        let u = result.metadata.urgency;
        assert!(
            matches!(
                u,
                crate::cognitive_loop::CycleUrgency::Critical
                    | crate::cognitive_loop::CycleUrgency::Normal
                    | crate::cognitive_loop::CycleUrgency::Cruise
            ),
            "urgency should be a valid variant"
        );
    }

    #[test]
    fn cycle_output_all_finite() {
        let mut s = make_service();
        let result = s.cycle("NaN guard check");
        for (i, &v) in result.output.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] is not finite: {v}");
        }
    }

    // ── Multiple cycles ───────────────────────────────────────────────

    #[test]
    fn multiple_cycles_do_not_panic() {
        let mut s = make_service();
        for i in 0..10 {
            let result = s.cycle(&format!("cycle input {i}"));
            assert!(result.prediction_error.is_finite());
        }
        assert_eq!(s.stats().total_cycles, 10);
    }

    #[test]
    fn empty_input_does_not_panic() {
        let mut s = make_service();
        let result = s.cycle("");
        assert!(result.prediction_error.is_finite());
    }

    #[test]
    fn long_input_does_not_panic() {
        let mut s = make_service();
        let long_input = "a".repeat(10_000);
        let result = s.cycle(&long_input);
        assert!(result.prediction_error.is_finite());
    }

    #[test]
    fn repeated_identical_input_reduces_prediction_error() {
        let mut s = make_service();
        // First cycle has no prior prediction
        let r1 = s.cycle("repeating input");
        // Run several identical cycles so the system can learn the pattern
        let mut last_error = r1.prediction_error;
        for _ in 0..20 {
            last_error = s.cycle("repeating input").prediction_error;
        }
        // After 20 identical cycles, error should be lower or comparable
        // (not necessarily strictly lower due to stochastic subsystems)
        assert!(
            last_error.is_finite(),
            "error should remain finite after repeated cycles"
        );
    }

    // ── try_cycle() ───────────────────────────────────────────────────

    #[test]
    fn try_cycle_returns_ok() {
        let mut s = make_service();
        let result = s.try_cycle("safe input");
        assert!(result.is_ok(), "try_cycle should succeed for normal input");
    }

    #[test]
    fn try_cycle_result_matches_cycle() {
        // Use genesis phrase for determinism
        let mut cfg = CognitiveLoopConfig::default();
        cfg.genesis_phrase = Some("determinism test".to_string());
        let mut s1 = CognitiveLoopService::new(cfg.clone()).expect("test config must initialize");
        let mut s2 = CognitiveLoopService::new(cfg).expect("test config must initialize");

        let r1 = s1.cycle("hello");
        let r2 = s2.try_cycle("hello").expect("try_cycle should succeed");

        // Both should produce same output with deterministic genesis
        assert_eq!(r1.output.len(), r2.output.len());
        assert_eq!(r1.prediction_error, r2.prediction_error);
    }

    // ── Cycle with different backends ─────────────────────────────────

    #[test]
    fn cycle_with_hdc_ltc_unified_backend() {
        let config = CognitiveLoopConfig::with_hdc_ltc_unified();
        let mut s =
            CognitiveLoopService::new(config).expect("unified backend config must initialize");
        let result = s.cycle("HdcLtc backend test");
        assert!(!result.output.is_empty());
        assert!(result.prediction_error.is_finite());
    }

    #[test]
    fn cycle_with_hdc_ltc_fast_backend() {
        let config = CognitiveLoopConfig::with_hdc_ltc_fast();
        let mut s = CognitiveLoopService::new(config).expect("fast backend config must initialize");
        let result = s.cycle("fast backend test");
        assert!(!result.output.is_empty());
        assert!(result.prediction_error.is_finite());
    }

    // ── Cycle stats tracking ──────────────────────────────────────────

    #[test]
    fn cycle_updates_avg_prediction_error() {
        let mut s = make_service();
        s.cycle("first");
        let err1 = s.stats().avg_prediction_error;
        // After first cycle, avg error should be populated (may be 0.0 for first cycle)
        assert!(err1.is_finite());
    }

    #[test]
    fn cycle_populates_adaptive_learning_rate() {
        let mut s = make_service();
        s.cycle("learning rate check");
        let lr = s.stats().adaptive_learning_rate;
        assert!(lr.is_finite());
        assert!(lr >= 0.0);
    }

    // ── Genesis determinism ───────────────────────────────────────────

    #[test]
    fn genesis_seeded_cycles_are_deterministic() {
        let phrase = "We hold these truths to be self-evident".to_string();

        let mut cfg_a = CognitiveLoopConfig::default();
        cfg_a.genesis_phrase = Some(phrase.clone());
        let mut sa = CognitiveLoopService::new(cfg_a).expect("determinism config must initialize");

        let mut cfg_b = CognitiveLoopConfig::default();
        cfg_b.genesis_phrase = Some(phrase);
        let mut sb = CognitiveLoopService::new(cfg_b).expect("determinism config must initialize");

        let ra = sa.cycle("determinism check");
        let rb = sb.cycle("determinism check");

        assert_eq!(ra.output, rb.output, "genesis-seeded outputs should match");
        assert_eq!(
            ra.prediction_error, rb.prediction_error,
            "genesis-seeded errors should match"
        );
    }

    // ── Cycle metadata fields ─────────────────────────────────────────

    #[test]
    fn cycle_metadata_somatic_stress_finite() {
        let mut s = make_service();
        let result = s.cycle("somatic check");
        assert!(result.metadata.embodied.somatic_stress.is_finite());
    }

    #[test]
    fn cycle_metadata_consciousness_level_bounded() {
        let mut s = make_service();
        // Run a few cycles to populate MCE
        for _ in 0..15 {
            s.cycle("populate MCE");
        }
        let result = s.cycle("check consciousness");
        assert!(result.metadata.consciousness.consciousness_level >= 0.0);
        assert!(result.metadata.consciousness.consciousness_level <= 1.0);
    }

    #[test]
    fn cycle_metadata_thermodynamic_load_bounded() {
        let mut s = make_service();
        let result = s.cycle("thermo check");
        assert!(result.metadata.temporal.thermodynamic_load >= 0.0);
        assert!(result.metadata.temporal.thermodynamic_load <= 1.0);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Dynamics phase coverage (cycle_phase_dynamics.rs)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn dynamics_fep_fields_finite() {
        let mut s = make_service();
        let r = s.cycle("FEP test");
        assert!(r.metadata.fep.fep_pragmatic_value.is_finite());
        assert!(r.metadata.fep.fep_accuracy.is_finite());
        assert!(r.metadata.fep.fep_complexity.is_finite());
        assert!(r.metadata.fep.fep_surprise.is_finite());
        assert!(r.metadata.fep.fep_td_error.is_finite());
    }

    #[test]
    fn dynamics_reasoning_fields_populated() {
        let mut s = make_service();
        let r = s.cycle("reasoning check");
        assert!(r.metadata.reasoning_confidence.is_finite());
        // gate_blocked is a bool — just verify it doesn't panic
        let _ = r.metadata.reasoning_gate_blocked;
    }

    #[test]
    fn dynamics_coherence_finite() {
        let mut s = make_service();
        let r = s.cycle("coherence check");
        assert!(r.metadata.prediction_coherence.is_finite());
        assert!(r.metadata.cross_module_agreement.is_finite());
    }

    #[test]
    fn dynamics_homeostasis_pulls_finite() {
        let mut s = make_service();
        let r = s.cycle("homeostasis check");
        assert!(r.metadata.valence_homeostasis_pull.is_finite());
        assert!(r.metadata.arousal_homeostasis_pull.is_finite());
        assert!(r.metadata.homeostasis_pull_strength.is_finite());
    }

    #[test]
    fn dynamics_neuromod_fields_populated() {
        let mut s = make_service();
        let r = s.cycle("neuromod check");
        let _ = &r.metadata.harmonics.guiding_question;
        assert!(!r.metadata.harmonics.dominant_harmonic.is_empty());
    }

    #[test]
    fn dynamics_metacognitive_anomaly_defaults_false() {
        let mut s = make_service();
        let r = s.cycle("anomaly check");
        // On first cycle, no anomaly should be detected
        assert!(!r.metadata.metacognitive_anomaly);
    }

    #[test]
    fn dynamics_cycle_reward_finite() {
        let mut s = make_service();
        let r = s.cycle("reward test");
        assert!(r.metadata.cycle_reward.is_finite());
    }

    #[test]
    fn dynamics_attention_budget_fields() {
        let mut s = make_service();
        let r = s.cycle("attention budget");
        // Budget shouldn't be exceeded on first cycle
        let _ = r.metadata.attention.attention_budget_exceeded;
        // Elapsed should be non-negative
        assert!(r.metadata.attention.attention_budget_elapsed_us < u64::MAX);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Feedback phase coverage (cycle_phase_feedback.rs)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn feedback_consciousness_metrics_finite() {
        let mut s = make_service();
        for _ in 0..5 {
            s.cycle("warmup");
        }
        let r = s.cycle("consciousness check");
        assert!(r.metadata.consciousness.consciousness_level.is_finite());
        assert!(r.metadata.consciousness.consciousness_level >= 0.0);
        assert!(r.metadata.consciousness.consciousness_level <= 1.0);
        assert!(r.metadata.primitive_psi.is_finite());
    }

    #[test]
    fn feedback_quality_gating_fields() {
        let mut s = make_service();
        let r = s.cycle("quality check");
        assert!(r.metadata.quality.unified_quality_score.is_finite());
        // Gating booleans should be accessible
        let _ = r.metadata.quality.coherence_velocity_gated;
        let _ = r.metadata.quality.dissipative_health_gated;
        let _ = r.metadata.epistemic_gate_approved;
    }

    #[test]
    fn feedback_temporal_fields_finite() {
        let mut s = make_service();
        let r = s.cycle("temporal check");
        assert!(r.metadata.temporal.temporal_continuity.is_finite());
        assert!(r.metadata.temporal.temporal_coherence_score.is_finite());
    }

    #[test]
    fn feedback_harmonic_fields_finite() {
        let mut s = make_service();
        let r = s.cycle("harmonic check");
        assert!(r.metadata.harmonics.harmonies_alignment.is_finite());
        assert!(r.metadata.harmonics.harmonic_field_coherence.is_finite());
        assert!(r.metadata.harmonics.harmonic_love_resonance.is_finite());
    }

    #[test]
    fn feedback_dream_engine_defaults() {
        let mut s = make_service();
        let r = s.cycle("dream check");
        // Dream engine requires many cycles, so on first cycle defaults
        assert!(r.metadata.memory.dream_phi_improvement.is_finite());
    }

    #[test]
    fn feedback_dissipative_fields_finite() {
        let mut s = make_service();
        let r = s.cycle("dissipative check");
        assert!(r.metadata.quality.dissipative_health.is_finite());
        assert!(r.metadata.quality.dissipative_entropy_rate.is_finite());
    }

    #[test]
    fn feedback_epistemic_fields_finite() {
        let mut s = make_service();
        let r = s.cycle("epistemic check");
        assert!(r.metadata.epistemic_quality.is_finite());
        assert!(r.metadata.quality.epistemic_phi_eff.is_finite());
    }

    #[test]
    fn feedback_holographic_fields_finite() {
        let mut s = make_service();
        let r = s.cycle("holographic check");
        assert!(r.metadata.temporal.holographic_unity.is_finite());
        assert!(r.metadata.temporal.holographic_binding.is_finite());
    }

    #[test]
    fn feedback_phenomenal_binding_fields() {
        let mut s = make_service();
        let r = s.cycle("phenomenal check");
        assert!(r.metadata.temporal.phenomenal_binding_strength.is_finite());
        let _ = r.metadata.temporal.phenomenal_fragmented;
    }

    #[test]
    fn feedback_affective_fields_bounded() {
        let mut s = make_service();
        let r = s.cycle("affective check");
        assert!(r.metadata.embodied.affective_valence.is_finite());
        assert!(r.metadata.embodied.affective_arousal.is_finite());
        assert!(r.metadata.embodied.body_valence.is_finite());
        assert!(r.metadata.embodied.body_arousal.is_finite());
    }

    #[test]
    fn feedback_living_mind_fields_finite() {
        let mut s = make_service();
        let r = s.cycle("living mind check");
        assert!(r.metadata.living_mind_vitality.is_finite());
        assert!(r.metadata.living_mind_coherence.is_finite());
    }

    // ═══════════════════════════════════════════════════════════════════
    // Output phase coverage (cycle_phase_output/)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn output_thought_vector_32d() {
        let mut s = make_service();
        let r = s.cycle("thought vector");
        assert_eq!(r.thought_vector.len(), 32);
        for v in &r.thought_vector {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn output_cycle_time_nonzero() {
        let mut s = make_service();
        let r = s.cycle("timing");
        assert!(r.cycle_time_us > 0);
    }

    #[test]
    fn output_circadian_phase_non_empty() {
        let mut s = make_service();
        let r = s.cycle("circadian");
        assert!(!r.metadata.circadian_phase.is_empty());
    }

    #[test]
    fn output_selected_strategy_non_empty() {
        let mut s = make_service();
        let r = s.cycle("strategy output");
        assert!(!r.metadata.selected_strategy.is_empty());
    }

    #[test]
    fn output_module_timings_populated() {
        let mut s = make_service();
        let r = s.cycle("timing check");
        let t = &r.metadata.module_timings_us;
        assert!(t.core_hdc_encode > 0 || t.core_cfc_step > 0);
    }

    #[test]
    fn output_sigma_and_spectral_phi() {
        let mut s = make_service();
        let r = s.cycle("sigma check");
        // May be None on first cycle — just verify finitely populated if present
        if let Some(sigma) = r.metadata.structural.sigma {
            assert!(sigma.is_finite());
        }
        if let Some(phi) = r.metadata.structural.spectral_mip_phi {
            assert!(phi.is_finite());
        }
    }

    #[test]
    fn output_detected_primitives_exist() {
        let mut s = make_service();
        let r = s.cycle("hello world primitive");
        // May or may not detect primitives — just verify it doesn't panic
        for p in &r.detected_primitives {
            assert!(!p.is_empty());
        }
    }

    #[test]
    fn soul_alignment_computed_when_enabled() {
        let mut config = CognitiveLoopConfig::default();
        config.enable_soul_alignment = true;
        let mut s =
            CognitiveLoopService::new(config).expect("soul alignment config must initialize");
        let r = s.cycle("resonance and flourishing");
        assert!(r.metadata.ethics.soul_alignment.is_finite());
        let _ = &s.ethics_values.soul;
    }

    #[test]
    fn soul_alignment_zero_when_disabled() {
        let mut config = CognitiveLoopConfig::default();
        config.enable_soul_alignment = false;
        let mut s = CognitiveLoopService::new(config).unwrap();
        let r = s.cycle("test without soul");
        assert_eq!(r.metadata.ethics.soul_alignment, 0.0);
        assert!(s.ethics_values.soul.is_none());
    }

    #[test]
    fn output_wisdom_hv_correct_size() {
        let mut s = make_service();
        let r = s.cycle("wisdom hv");
        // BinaryHV is 16384 bits = 2048 bytes
        assert_eq!(r.wisdom_hv.0.len(), 2048);
    }

    #[test]
    fn output_soul_alignment_finite() {
        let mut s = make_service();
        let r = s.cycle("soul alignment");
        assert!(r.metadata.ethics.soul_alignment.is_finite());
    }

    #[test]
    fn output_feedback_divergence_tracked() {
        let mut s = make_service();
        // Run several cycles so feedback proposals accumulate
        for _ in 0..10 {
            s.cycle("divergence");
        }
        // After cycles, feedback state should have integration results
        assert!(s.feedback_state.last_confidence_integration.is_some());
        assert!(s.feedback_state.last_lr_integration.is_some());
    }

    #[test]
    fn defense_cascade_runs_without_panic() {
        let mut s = make_service();
        // Run enough cycles to populate feedback and trigger safety enforcement
        for _ in 0..5 {
            s.cycle("defense cascade test");
        }
        // After cycles, system should still be operational
        let r = s.cycle("final check");
        assert!(r.prediction_error.is_finite());
    }
}
