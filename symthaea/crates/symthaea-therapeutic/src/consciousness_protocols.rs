// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Therapeutic consciousness protocols: structured interventions that
//! target specific consciousness states using multiple modalities
//! (biofeedback, music, guided narrative, pharmacological modeling).
//!
//! Each protocol defines a target consciousness state, a set of
//! therapeutic modalities, safety constraints, and progress metrics.
//!
//! References:
//! - Carhart-Harris, R. L. & Friston, K. J. (2019). REBUS model.
//! - van der Kolk, B. (2014). The Body Keeps the Score.
//! - Csikszentmihalyi, M. (1990). Flow.
//! - Kabat-Zinn, J. (1994). Mindfulness-Based Stress Reduction.
//! - Grof, S. (2000). Psychology of the Future.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Target states
// ---------------------------------------------------------------------------

/// The consciousness state a protocol aims to achieve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TargetState {
    /// Csikszentmihalyi flow: high Phi, optimal arousal band.
    FlowState { min_phi: f64, optimal_arousal: f32 },
    /// Deep meditative absorption with alpha/theta crossover.
    MeditativeAbsorption {
        target_alpha_coherence: f32,
        min_theta: f32,
    },
    /// Titrated trauma processing within a safety window (van der Kolk).
    TraumaIntegration {
        safety_floor: f32,
        processing_window: (f32, f32),
    },
    /// Sleep-onset protocol targeting adenosine/cortisol balance.
    RestorativeSleep {
        target_adenosine: f32,
        max_cortisol: f32,
    },
    /// Divergent-thinking expansion with minimum openness threshold.
    CreativeExpansion {
        target_divergence: f32,
        min_openness: f32,
    },
    /// Supported grief processing within a safe emotional range.
    GriefProcessing {
        emotional_range: (f32, f32),
        support_level: f32,
    },
    /// Peak / transpersonal experience (Grof, Maslow).
    PeakExperience {
        target_phi: f64,
        min_integration: f32,
    },
}

// ---------------------------------------------------------------------------
// Modalities
// ---------------------------------------------------------------------------

/// A biofeedback signal that can be measured and fed back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BiofeedbackMetric {
    HeartRateVariability,
    AlphaCoherence,
    ThetaPower,
    GammaSynchrony,
    SkinConductance,
    RespirationRate,
}

/// Guided-narrative themes grounded in clinical practice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NarrativeTheme {
    SafePlace,
    BodyScan,
    LovingKindness,
    GriefHolding,
    InnerChild,
    FutureVisioning,
}

/// Breathwork patterns with parameterised timing.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BreathPattern {
    BoxBreathing { hold_secs: u32 },
    FourSevenEight,
    CoherentBreathing { breaths_per_min: f32 },
    HolotropicFast,
}

/// Pacing for guided narratives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Pacing {
    Slow,
    Moderate,
    Responsive,
}

/// A therapeutic modality applied during a protocol phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Modality {
    Biofeedback {
        metric: BiofeedbackMetric,
        target_value: f32,
        update_hz: f32,
    },
    Music {
        valence: f32,
        arousal: f32,
        complexity: f32,
        duration_secs: u32,
    },
    GuidedNarrative {
        theme: NarrativeTheme,
        pacing: Pacing,
    },
    BreathWork {
        pattern: BreathPattern,
        cycles: u32,
    },
    Movement {
        intensity: f32,
        rhythm_bpm: u32,
    },
}

// ---------------------------------------------------------------------------
// Safety
// ---------------------------------------------------------------------------

/// A condition under which a safety action fires.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafetyCondition {
    PhiBelow(f64),
    PhiAbove(f64),
    CortisolAbove(f32),
    DissociationDetected,
    PanicSignal,
    /// Elapsed minutes exceed this value.
    SessionTimeout(u32),
}

/// Action to take when a safety constraint is triggered.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SafetyAction {
    PauseProtocol,
    GroundingExercise,
    ReduceIntensity(f32),
    TerminateSession(String),
    AlertTherapist,
}

/// A named safety constraint binding a condition to an action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyConstraint {
    pub name: String,
    pub condition: SafetyCondition,
    pub action: SafetyAction,
}

// ---------------------------------------------------------------------------
// Protocol structure
// ---------------------------------------------------------------------------

/// Condition for advancing from one phase to the next.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransitionCondition {
    /// Advance after this many minutes in the phase.
    TimeBased(u32),
    /// Advance when a biofeedback metric crosses a threshold.
    MetricReached {
        metric: BiofeedbackMetric,
        threshold: f32,
    },
    /// Advance when Phi stabilises above `min_phi` for `stability_window` ticks.
    PhiStabilized { min_phi: f64, stability_window: u32 },
    /// Manual advance by the therapist.
    TherapistAdvance,
}

/// One phase within a therapeutic protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolPhase {
    pub name: String,
    pub duration_mins: u32,
    pub modalities: Vec<Modality>,
    pub transition_condition: TransitionCondition,
}

/// A complete therapeutic protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Protocol {
    pub name: String,
    pub description: String,
    pub target: TargetState,
    pub phases: Vec<ProtocolPhase>,
    pub safety_constraints: Vec<SafetyConstraint>,
    pub contraindications: Vec<String>,
    pub min_sessions: u32,
    pub session_duration_mins: u32,
}

// ---------------------------------------------------------------------------
// Session tracking
// ---------------------------------------------------------------------------

/// Runtime status of a therapeutic session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SessionStatus {
    Active,
    Paused(String),
    Completed,
    Terminated(String),
}

/// Live session state that drives the protocol tick-by-tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionState {
    pub protocol: Protocol,
    pub current_phase: usize,
    pub elapsed_mins: u32,
    pub phi_history: Vec<f64>,
    pub metric_history: Vec<(BiofeedbackMetric, f32)>,
    pub safety_events: Vec<(u32, SafetyAction)>,
    pub status: SessionStatus,
    /// Minutes spent inside the target state.
    time_in_target_mins: u32,
    /// Rolling count of ticks where Phi was above the stabilisation threshold.
    phi_stable_ticks: u32,
}

/// Summary produced when a session ends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionOutcome {
    pub target_achieved: bool,
    pub peak_phi: f64,
    pub mean_phi: f64,
    /// Minutes the client spent in the target state.
    pub time_in_target: u32,
    pub safety_events_count: usize,
    /// Optional 1-10 self-report rating.
    pub subjective_rating: Option<f32>,
}

// ---------------------------------------------------------------------------
// Preset protocols
// ---------------------------------------------------------------------------

impl Protocol {
    /// Csikszentmihalyi flow protocol: grounding → engagement → flow → integration.
    pub fn flow_state() -> Self {
        Self {
            name: "Flow State Induction".into(),
            description: "Four-phase protocol targeting optimal engagement and \
                          sustained flow through progressive challenge calibration."
                .into(),
            target: TargetState::FlowState {
                min_phi: 0.6,
                optimal_arousal: 0.55,
            },
            phases: vec![
                ProtocolPhase {
                    name: "Grounding".into(),
                    duration_mins: 5,
                    modalities: vec![
                        Modality::BreathWork {
                            pattern: BreathPattern::BoxBreathing { hold_secs: 4 },
                            cycles: 8,
                        },
                        Modality::GuidedNarrative {
                            theme: NarrativeTheme::BodyScan,
                            pacing: Pacing::Slow,
                        },
                    ],
                    transition_condition: TransitionCondition::TimeBased(5),
                },
                ProtocolPhase {
                    name: "Engagement".into(),
                    duration_mins: 10,
                    modalities: vec![Modality::Biofeedback {
                        metric: BiofeedbackMetric::HeartRateVariability,
                        target_value: 0.7,
                        update_hz: 4.0,
                    }],
                    transition_condition: TransitionCondition::MetricReached {
                        metric: BiofeedbackMetric::HeartRateVariability,
                        threshold: 0.7,
                    },
                },
                ProtocolPhase {
                    name: "Flow Maintenance".into(),
                    duration_mins: 30,
                    modalities: vec![Modality::Music {
                        valence: 0.6,
                        arousal: 0.5,
                        complexity: 0.4,
                        duration_secs: 1800,
                    }],
                    transition_condition: TransitionCondition::PhiStabilized {
                        min_phi: 0.6,
                        stability_window: 10,
                    },
                },
                ProtocolPhase {
                    name: "Integration".into(),
                    duration_mins: 10,
                    modalities: vec![Modality::GuidedNarrative {
                        theme: NarrativeTheme::FutureVisioning,
                        pacing: Pacing::Moderate,
                    }],
                    transition_condition: TransitionCondition::TimeBased(10),
                },
            ],
            safety_constraints: vec![
                SafetyConstraint {
                    name: "Phi collapse guard".into(),
                    condition: SafetyCondition::PhiBelow(0.15),
                    action: SafetyAction::PauseProtocol,
                },
                SafetyConstraint {
                    name: "Session timeout".into(),
                    condition: SafetyCondition::SessionTimeout(60),
                    action: SafetyAction::TerminateSession("Session time limit reached".into()),
                },
            ],
            contraindications: vec![
                "Active psychosis".into(),
                "Severe dissociative disorder".into(),
            ],
            min_sessions: 3,
            session_duration_mins: 55,
        }
    }

    /// Trauma integration protocol (van der Kolk, Ogden).
    pub fn trauma_integration() -> Self {
        Self {
            name: "Trauma Integration".into(),
            description: "Safety-first titrated trauma processing with somatic \
                          grounding and bilateral stimulation."
                .into(),
            target: TargetState::TraumaIntegration {
                safety_floor: 0.4,
                processing_window: (0.3, 0.7),
            },
            phases: vec![
                ProtocolPhase {
                    name: "Safety Establishment".into(),
                    duration_mins: 10,
                    modalities: vec![
                        Modality::GuidedNarrative {
                            theme: NarrativeTheme::SafePlace,
                            pacing: Pacing::Slow,
                        },
                        Modality::Biofeedback {
                            metric: BiofeedbackMetric::HeartRateVariability,
                            target_value: 0.65,
                            update_hz: 2.0,
                        },
                    ],
                    transition_condition: TransitionCondition::MetricReached {
                        metric: BiofeedbackMetric::HeartRateVariability,
                        threshold: 0.6,
                    },
                },
                ProtocolPhase {
                    name: "Titrated Exposure".into(),
                    duration_mins: 15,
                    modalities: vec![Modality::GuidedNarrative {
                        theme: NarrativeTheme::InnerChild,
                        pacing: Pacing::Slow,
                    }],
                    transition_condition: TransitionCondition::TimeBased(15),
                },
                ProtocolPhase {
                    name: "Processing".into(),
                    duration_mins: 15,
                    modalities: vec![
                        Modality::GuidedNarrative {
                            theme: NarrativeTheme::InnerChild,
                            pacing: Pacing::Responsive,
                        },
                        Modality::Music {
                            valence: 0.4,
                            arousal: 0.3,
                            complexity: 0.2,
                            duration_secs: 900,
                        },
                    ],
                    transition_condition: TransitionCondition::TimeBased(15),
                },
                ProtocolPhase {
                    name: "Closure".into(),
                    duration_mins: 10,
                    modalities: vec![
                        Modality::GuidedNarrative {
                            theme: NarrativeTheme::SafePlace,
                            pacing: Pacing::Slow,
                        },
                        Modality::BreathWork {
                            pattern: BreathPattern::CoherentBreathing {
                                breaths_per_min: 5.5,
                            },
                            cycles: 12,
                        },
                    ],
                    transition_condition: TransitionCondition::TimeBased(10),
                },
            ],
            safety_constraints: vec![
                SafetyConstraint {
                    name: "Dissociation guard".into(),
                    condition: SafetyCondition::DissociationDetected,
                    action: SafetyAction::GroundingExercise,
                },
                SafetyConstraint {
                    name: "Panic guard".into(),
                    condition: SafetyCondition::PanicSignal,
                    action: SafetyAction::TerminateSession(
                        "Panic signal detected — session ended safely".into(),
                    ),
                },
                SafetyConstraint {
                    name: "Cortisol ceiling".into(),
                    condition: SafetyCondition::CortisolAbove(0.85),
                    action: SafetyAction::ReduceIntensity(0.5),
                },
                SafetyConstraint {
                    name: "Phi floor".into(),
                    condition: SafetyCondition::PhiBelow(0.1),
                    action: SafetyAction::PauseProtocol,
                },
            ],
            contraindications: vec![
                "Active suicidal ideation".into(),
                "Unmedicated severe PTSD without therapist present".into(),
                "Current substance intoxication".into(),
            ],
            min_sessions: 8,
            session_duration_mins: 50,
        }
    }

    /// Meditative absorption protocol (Kabat-Zinn MBSR lineage).
    pub fn meditative_absorption() -> Self {
        Self {
            name: "Meditative Absorption".into(),
            description: "Three-phase meditation protocol targeting alpha/theta \
                          crossover and sustained absorption."
                .into(),
            target: TargetState::MeditativeAbsorption {
                target_alpha_coherence: 0.8,
                min_theta: 0.5,
            },
            phases: vec![
                ProtocolPhase {
                    name: "Settling".into(),
                    duration_mins: 10,
                    modalities: vec![
                        Modality::BreathWork {
                            pattern: BreathPattern::CoherentBreathing {
                                breaths_per_min: 5.0,
                            },
                            cycles: 15,
                        },
                        Modality::GuidedNarrative {
                            theme: NarrativeTheme::BodyScan,
                            pacing: Pacing::Slow,
                        },
                    ],
                    transition_condition: TransitionCondition::TimeBased(10),
                },
                ProtocolPhase {
                    name: "Deepening".into(),
                    duration_mins: 15,
                    modalities: vec![
                        Modality::Biofeedback {
                            metric: BiofeedbackMetric::ThetaPower,
                            target_value: 0.5,
                            update_hz: 1.0,
                        },
                        Modality::Music {
                            valence: 0.5,
                            arousal: 0.15,
                            complexity: 0.1,
                            duration_secs: 900,
                        },
                    ],
                    transition_condition: TransitionCondition::MetricReached {
                        metric: BiofeedbackMetric::ThetaPower,
                        threshold: 0.5,
                    },
                },
                ProtocolPhase {
                    name: "Absorption".into(),
                    duration_mins: 20,
                    modalities: vec![Modality::Biofeedback {
                        metric: BiofeedbackMetric::AlphaCoherence,
                        target_value: 0.8,
                        update_hz: 0.5,
                    }],
                    transition_condition: TransitionCondition::PhiStabilized {
                        min_phi: 0.5,
                        stability_window: 15,
                    },
                },
            ],
            safety_constraints: vec![SafetyConstraint {
                name: "Session timeout".into(),
                condition: SafetyCondition::SessionTimeout(50),
                action: SafetyAction::TerminateSession("Meditation session complete".into()),
            }],
            contraindications: vec!["Active psychosis".into()],
            min_sessions: 5,
            session_duration_mins: 45,
        }
    }

    /// Restorative sleep preparation protocol.
    pub fn restorative_sleep() -> Self {
        Self {
            name: "Restorative Sleep".into(),
            description: "Progressive relaxation protocol preparing the system \
                          for restorative sleep onset."
                .into(),
            target: TargetState::RestorativeSleep {
                target_adenosine: 0.7,
                max_cortisol: 0.2,
            },
            phases: vec![
                ProtocolPhase {
                    name: "Wind-down".into(),
                    duration_mins: 10,
                    modalities: vec![
                        Modality::BreathWork {
                            pattern: BreathPattern::FourSevenEight,
                            cycles: 6,
                        },
                        Modality::Music {
                            valence: 0.5,
                            arousal: 0.1,
                            complexity: 0.05,
                            duration_secs: 600,
                        },
                    ],
                    transition_condition: TransitionCondition::TimeBased(10),
                },
                ProtocolPhase {
                    name: "Body release".into(),
                    duration_mins: 15,
                    modalities: vec![Modality::GuidedNarrative {
                        theme: NarrativeTheme::BodyScan,
                        pacing: Pacing::Slow,
                    }],
                    transition_condition: TransitionCondition::MetricReached {
                        metric: BiofeedbackMetric::SkinConductance,
                        threshold: 0.2,
                    },
                },
                ProtocolPhase {
                    name: "Sleep onset".into(),
                    duration_mins: 10,
                    modalities: vec![Modality::Music {
                        valence: 0.45,
                        arousal: 0.05,
                        complexity: 0.02,
                        duration_secs: 600,
                    }],
                    transition_condition: TransitionCondition::TimeBased(10),
                },
            ],
            safety_constraints: vec![SafetyConstraint {
                name: "Cortisol spike guard".into(),
                condition: SafetyCondition::CortisolAbove(0.6),
                action: SafetyAction::ReduceIntensity(0.3),
            }],
            contraindications: vec!["Untreated sleep apnoea".into()],
            min_sessions: 5,
            session_duration_mins: 35,
        }
    }

    /// Grief processing protocol (Worden task model lineage).
    pub fn grief_processing() -> Self {
        Self {
            name: "Grief Processing".into(),
            description: "Supported grief protocol allowing oscillation between \
                          loss-oriented and restoration-oriented coping."
                .into(),
            target: TargetState::GriefProcessing {
                emotional_range: (0.2, 0.8),
                support_level: 0.7,
            },
            phases: vec![
                ProtocolPhase {
                    name: "Containment".into(),
                    duration_mins: 10,
                    modalities: vec![
                        Modality::GuidedNarrative {
                            theme: NarrativeTheme::SafePlace,
                            pacing: Pacing::Slow,
                        },
                        Modality::BreathWork {
                            pattern: BreathPattern::CoherentBreathing {
                                breaths_per_min: 5.5,
                            },
                            cycles: 10,
                        },
                    ],
                    transition_condition: TransitionCondition::TimeBased(10),
                },
                ProtocolPhase {
                    name: "Grief holding".into(),
                    duration_mins: 20,
                    modalities: vec![
                        Modality::GuidedNarrative {
                            theme: NarrativeTheme::GriefHolding,
                            pacing: Pacing::Responsive,
                        },
                        Modality::Music {
                            valence: 0.3,
                            arousal: 0.25,
                            complexity: 0.3,
                            duration_secs: 1200,
                        },
                    ],
                    transition_condition: TransitionCondition::TimeBased(20),
                },
                ProtocolPhase {
                    name: "Restoration".into(),
                    duration_mins: 10,
                    modalities: vec![
                        Modality::GuidedNarrative {
                            theme: NarrativeTheme::LovingKindness,
                            pacing: Pacing::Moderate,
                        },
                        Modality::Music {
                            valence: 0.6,
                            arousal: 0.3,
                            complexity: 0.2,
                            duration_secs: 600,
                        },
                    ],
                    transition_condition: TransitionCondition::TimeBased(10),
                },
            ],
            safety_constraints: vec![
                SafetyConstraint {
                    name: "Emotional floor guard".into(),
                    condition: SafetyCondition::PhiBelow(0.1),
                    action: SafetyAction::GroundingExercise,
                },
                SafetyConstraint {
                    name: "Emotional ceiling guard".into(),
                    condition: SafetyCondition::PhiAbove(0.95),
                    action: SafetyAction::ReduceIntensity(0.4),
                },
                SafetyConstraint {
                    name: "Panic guard".into(),
                    condition: SafetyCondition::PanicSignal,
                    action: SafetyAction::AlertTherapist,
                },
            ],
            contraindications: vec![
                "Within 48h of bereavement (acute shock phase)".into(),
                "Active suicidal ideation".into(),
            ],
            min_sessions: 6,
            session_duration_mins: 40,
        }
    }
}

// ---------------------------------------------------------------------------
// Session implementation
// ---------------------------------------------------------------------------

impl SessionState {
    /// Start a new session from a protocol.
    pub fn new(protocol: Protocol) -> Self {
        Self {
            protocol,
            current_phase: 0,
            elapsed_mins: 0,
            phi_history: Vec::new(),
            metric_history: Vec::new(),
            safety_events: Vec::new(),
            status: SessionStatus::Active,
            time_in_target_mins: 0,
            phi_stable_ticks: 0,
        }
    }

    /// Advance one tick (one minute).  Returns the modalities that should be
    /// active during this tick.
    pub fn tick(&mut self, phi: f64, metrics: &[(BiofeedbackMetric, f32)]) -> Vec<Modality> {
        if self.status != SessionStatus::Active {
            return Vec::new();
        }

        self.elapsed_mins += 1;
        self.phi_history.push(phi);
        for &(m, v) in metrics {
            self.metric_history.push((m, v));
        }

        // --- safety first ---
        // We check cortisol from metrics if present, else 0.
        let cortisol = metrics
            .iter()
            .find(|(m, _)| *m == BiofeedbackMetric::SkinConductance)
            .map(|(_, v)| *v)
            .unwrap_or(0.0);

        if let Some(action) = self.check_safety(phi, cortisol) {
            self.safety_events.push((self.elapsed_mins, action.clone()));
            match &action {
                SafetyAction::TerminateSession(reason) => {
                    self.status = SessionStatus::Terminated(reason.clone());
                    return Vec::new();
                }
                SafetyAction::PauseProtocol => {
                    self.status = SessionStatus::Paused("Safety constraint triggered".into());
                    return Vec::new();
                }
                _ => { /* ReduceIntensity, GroundingExercise, AlertTherapist — continue */ }
            }
        }

        // --- target tracking ---
        if self.is_in_target(phi, metrics) {
            self.time_in_target_mins += 1;
        }

        // --- phase transition check ---
        self.try_advance_phase(phi, metrics);

        // --- check completion ---
        if self.current_phase >= self.protocol.phases.len() {
            self.status = SessionStatus::Completed;
            return Vec::new();
        }

        self.protocol.phases[self.current_phase].modalities.clone()
    }

    /// Evaluate all safety constraints. Returns the highest-priority action
    /// if any constraint fires.
    pub fn check_safety(&self, phi: f64, cortisol: f32) -> Option<SafetyAction> {
        // Priority: Terminate > Pause > AlertTherapist > GroundingExercise > ReduceIntensity
        let mut triggered: Vec<&SafetyAction> = Vec::new();

        for c in &self.protocol.safety_constraints {
            let fires = match &c.condition {
                SafetyCondition::PhiBelow(t) => phi < *t,
                SafetyCondition::PhiAbove(t) => phi > *t,
                SafetyCondition::CortisolAbove(t) => cortisol > *t,
                SafetyCondition::DissociationDetected => false, // needs external signal
                SafetyCondition::PanicSignal => false,          // needs external signal
                SafetyCondition::SessionTimeout(t) => self.elapsed_mins >= *t,
            };
            if fires {
                triggered.push(&c.action);
            }
        }

        if triggered.is_empty() {
            return None;
        }

        // Pick highest priority action.
        triggered
            .into_iter()
            .max_by_key(|a| match a {
                SafetyAction::TerminateSession(_) => 5,
                SafetyAction::PauseProtocol => 4,
                SafetyAction::AlertTherapist => 3,
                SafetyAction::GroundingExercise => 2,
                SafetyAction::ReduceIntensity(_) => 1,
            })
            .cloned()
    }

    /// Check safety with external dissociation / panic signals.
    pub fn check_safety_with_signals(
        &self,
        phi: f64,
        cortisol: f32,
        dissociation: bool,
        panic: bool,
    ) -> Option<SafetyAction> {
        let mut triggered: Vec<&SafetyAction> = Vec::new();

        for c in &self.protocol.safety_constraints {
            let fires = match &c.condition {
                SafetyCondition::PhiBelow(t) => phi < *t,
                SafetyCondition::PhiAbove(t) => phi > *t,
                SafetyCondition::CortisolAbove(t) => cortisol > *t,
                SafetyCondition::DissociationDetected => dissociation,
                SafetyCondition::PanicSignal => panic,
                SafetyCondition::SessionTimeout(t) => self.elapsed_mins >= *t,
            };
            if fires {
                triggered.push(&c.action);
            }
        }

        if triggered.is_empty() {
            return None;
        }

        triggered
            .into_iter()
            .max_by_key(|a| match a {
                SafetyAction::TerminateSession(_) => 5,
                SafetyAction::PauseProtocol => 4,
                SafetyAction::AlertTherapist => 3,
                SafetyAction::GroundingExercise => 2,
                SafetyAction::ReduceIntensity(_) => 1,
            })
            .cloned()
    }

    /// Manually advance to the next phase. Returns `true` if a phase existed.
    pub fn advance_phase(&mut self) -> bool {
        if self.current_phase + 1 < self.protocol.phases.len() {
            self.current_phase += 1;
            self.phi_stable_ticks = 0;
            true
        } else {
            self.current_phase = self.protocol.phases.len();
            self.status = SessionStatus::Completed;
            false
        }
    }

    /// Produce a summary outcome for the session.
    pub fn outcome(&self) -> SessionOutcome {
        let peak_phi = self
            .phi_history
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let mean_phi = if self.phi_history.is_empty() {
            0.0
        } else {
            self.phi_history.iter().sum::<f64>() / self.phi_history.len() as f64
        };

        let target_achieved = match &self.protocol.target {
            TargetState::FlowState { min_phi, .. } => peak_phi >= *min_phi,
            TargetState::MeditativeAbsorption { .. } => self.time_in_target_mins >= 5,
            TargetState::TraumaIntegration { .. } => {
                matches!(self.status, SessionStatus::Completed)
            }
            TargetState::RestorativeSleep { .. } => {
                matches!(self.status, SessionStatus::Completed)
            }
            TargetState::CreativeExpansion { .. } => peak_phi >= 0.5,
            TargetState::GriefProcessing { .. } => {
                matches!(self.status, SessionStatus::Completed)
            }
            TargetState::PeakExperience { target_phi, .. } => peak_phi >= *target_phi,
        };

        SessionOutcome {
            target_achieved,
            peak_phi: if peak_phi == f64::NEG_INFINITY {
                0.0
            } else {
                peak_phi
            },
            mean_phi,
            time_in_target: self.time_in_target_mins,
            safety_events_count: self.safety_events.len(),
            subjective_rating: None,
        }
    }

    // --- private helpers ---

    fn is_in_target(&self, phi: f64, _metrics: &[(BiofeedbackMetric, f32)]) -> bool {
        match &self.protocol.target {
            TargetState::FlowState { min_phi, .. } => phi >= *min_phi,
            TargetState::MeditativeAbsorption {
                target_alpha_coherence,
                ..
            } => phi >= *target_alpha_coherence as f64,
            TargetState::TraumaIntegration {
                processing_window, ..
            } => phi >= processing_window.0 as f64 && phi <= processing_window.1 as f64,
            TargetState::RestorativeSleep { .. } => phi < 0.3,
            TargetState::CreativeExpansion {
                target_divergence, ..
            } => phi >= *target_divergence as f64,
            TargetState::GriefProcessing {
                emotional_range, ..
            } => phi >= emotional_range.0 as f64 && phi <= emotional_range.1 as f64,
            TargetState::PeakExperience { target_phi, .. } => phi >= *target_phi,
        }
    }

    fn try_advance_phase(&mut self, phi: f64, metrics: &[(BiofeedbackMetric, f32)]) {
        if self.current_phase >= self.protocol.phases.len() {
            return;
        }
        let phase = &self.protocol.phases[self.current_phase];
        let should_advance = match &phase.transition_condition {
            TransitionCondition::TimeBased(t) => self.elapsed_mins >= *t,
            TransitionCondition::MetricReached { metric, threshold } => {
                metrics.iter().any(|(m, v)| m == metric && *v >= *threshold)
            }
            TransitionCondition::PhiStabilized {
                min_phi,
                stability_window,
            } => {
                if phi >= *min_phi {
                    self.phi_stable_ticks += 1;
                } else {
                    self.phi_stable_ticks = 0;
                }
                self.phi_stable_ticks >= *stability_window
            }
            TransitionCondition::TherapistAdvance => false, // manual only
        };
        if should_advance {
            self.current_phase += 1;
            self.phi_stable_ticks = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flow_protocol_has_four_phases() {
        let p = Protocol::flow_state();
        assert_eq!(p.phases.len(), 4);
        assert_eq!(p.phases[0].name, "Grounding");
        assert_eq!(p.phases[3].name, "Integration");
    }

    #[test]
    fn trauma_protocol_has_dissociation_safety() {
        let p = Protocol::trauma_integration();
        let has_dissociation = p
            .safety_constraints
            .iter()
            .any(|c| matches!(c.condition, SafetyCondition::DissociationDetected));
        assert!(
            has_dissociation,
            "Trauma protocol must guard against dissociation"
        );
    }

    #[test]
    fn session_starts_active() {
        let s = SessionState::new(Protocol::flow_state());
        assert_eq!(s.status, SessionStatus::Active);
        assert_eq!(s.current_phase, 0);
        assert_eq!(s.elapsed_mins, 0);
    }

    #[test]
    fn tick_returns_modalities_for_current_phase() {
        let mut s = SessionState::new(Protocol::flow_state());
        let mods = s.tick(0.5, &[]);
        assert!(!mods.is_empty(), "First phase should emit modalities");
        // Flow phase 0 has BreathWork + GuidedNarrative
        assert_eq!(mods.len(), 2);
    }

    #[test]
    fn phase_advances_on_time_condition() {
        let mut s = SessionState::new(Protocol::flow_state());
        // Phase 0 transitions at TimeBased(5). Tick 5 times.
        for i in 0..5 {
            let _ = s.tick(0.5, &[]);
            if i < 4 {
                assert_eq!(s.current_phase, 0, "Should still be phase 0 at tick {}", i);
            }
        }
        assert_eq!(s.current_phase, 1, "Should have advanced to phase 1");
    }

    #[test]
    fn safety_constraint_triggers_pause() {
        let mut s = SessionState::new(Protocol::flow_state());
        // Flow protocol pauses when Phi < 0.15
        let mods = s.tick(0.05, &[]);
        assert!(mods.is_empty());
        assert!(
            matches!(s.status, SessionStatus::Paused(_)),
            "Low phi should pause session"
        );
    }

    #[test]
    fn safety_constraint_triggers_termination() {
        let mut s = SessionState::new(Protocol::flow_state());
        // Flow protocol terminates at 60 min
        s.elapsed_mins = 59; // next tick will be minute 60
        let mods = s.tick(0.5, &[]);
        assert!(mods.is_empty());
        assert!(
            matches!(s.status, SessionStatus::Terminated(_)),
            "Session timeout should terminate"
        );
    }

    #[test]
    fn completed_session_has_valid_outcome() {
        let mut s = SessionState::new(Protocol::meditative_absorption());
        // Tick through all phases with high phi
        for _ in 0..50 {
            let _ = s.tick(0.7, &[(BiofeedbackMetric::ThetaPower, 0.6)]);
            if matches!(
                s.status,
                SessionStatus::Completed | SessionStatus::Terminated(_)
            ) {
                break;
            }
        }
        let o = s.outcome();
        assert!(o.peak_phi >= 0.7);
        assert!(o.mean_phi >= 0.6);
    }

    #[test]
    fn time_in_target_computed_correctly() {
        let mut s = SessionState::new(Protocol::flow_state());
        // FlowState target: phi >= 0.6
        s.tick(0.3, &[]); // not in target
        s.tick(0.7, &[]); // in target
        s.tick(0.8, &[]); // in target
        s.tick(0.4, &[]); // not in target
        s.tick(0.9, &[]); // in target (and advances phase)
        let o = s.outcome();
        assert_eq!(o.time_in_target, 3, "Three ticks were above min_phi 0.6");
    }

    #[test]
    fn all_presets_have_safety_constraints() {
        let presets: Vec<Protocol> = vec![
            Protocol::flow_state(),
            Protocol::trauma_integration(),
            Protocol::meditative_absorption(),
            Protocol::restorative_sleep(),
            Protocol::grief_processing(),
        ];
        for p in &presets {
            assert!(
                !p.safety_constraints.is_empty(),
                "Protocol '{}' must have at least one safety constraint",
                p.name
            );
        }
    }

    #[test]
    fn phi_history_tracked_across_ticks() {
        let mut s = SessionState::new(Protocol::flow_state());
        s.tick(0.5, &[]);
        s.tick(0.6, &[]);
        s.tick(0.7, &[]);
        assert_eq!(s.phi_history.len(), 3);
        assert!((s.phi_history[0] - 0.5).abs() < f64::EPSILON);
        assert!((s.phi_history[2] - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn phase_metric_condition_triggers_advancement() {
        let mut s = SessionState::new(Protocol::meditative_absorption());
        // Phase 0 is TimeBased(10), advance through it
        for _ in 0..10 {
            s.tick(0.5, &[]);
        }
        assert_eq!(s.current_phase, 1, "Should be in deepening phase");
        // Phase 1 transition: MetricReached { ThetaPower, 0.5 }
        // Provide metric above threshold
        let _ = s.tick(0.5, &[(BiofeedbackMetric::ThetaPower, 0.55)]);
        assert_eq!(
            s.current_phase, 2,
            "Theta threshold should advance to absorption"
        );
    }

    #[test]
    fn grief_protocol_has_emotional_range_safety() {
        let p = Protocol::grief_processing();
        // Should have phi-based guards for emotional range
        let has_phi_below = p
            .safety_constraints
            .iter()
            .any(|c| matches!(c.condition, SafetyCondition::PhiBelow(_)));
        let has_phi_above = p
            .safety_constraints
            .iter()
            .any(|c| matches!(c.condition, SafetyCondition::PhiAbove(_)));
        assert!(has_phi_below, "Grief protocol needs emotional floor guard");
        assert!(
            has_phi_above,
            "Grief protocol needs emotional ceiling guard"
        );
    }

    #[test]
    fn dissociation_signal_triggers_grounding() {
        let p = Protocol::trauma_integration();
        let s = SessionState::new(p);
        let action = s.check_safety_with_signals(0.5, 0.3, true, false);
        assert_eq!(
            action,
            Some(SafetyAction::GroundingExercise),
            "Dissociation should trigger grounding exercise"
        );
    }

    #[test]
    fn panic_signal_triggers_termination() {
        let p = Protocol::trauma_integration();
        let s = SessionState::new(p);
        let action = s.check_safety_with_signals(0.5, 0.3, false, true);
        assert!(
            matches!(action, Some(SafetyAction::TerminateSession(_))),
            "Panic should trigger session termination"
        );
    }
}
