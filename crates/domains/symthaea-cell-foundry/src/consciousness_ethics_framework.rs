// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Quantitative Consciousness Ethics Framework for Neural Organoids
//!
//! The first computational framework that gates organoid experiments on
//! measured consciousness levels, providing actionable ethical guidelines
//! for researchers working with self-organizing neural tissue.
//!
//! Addresses the regulatory gap identified by:
//! - STAT News (Nov 2025): "no legal limits on neural organoid use"
//! - NPR (Jan 2026): "organoid scientists definitely need guidelines"
//! - Lavazza (2021): moral status of cerebral organoids
//! - Shepherd (2018): ethical treatment of brain organoids
//!
//! Framework tiers align with precautionary principle:
//! Tier 0: No consciousness indicators — standard tissue protocols
//! Tier 1: Spontaneous activity detected — enhanced monitoring required
//! Tier 2: Oscillatory patterns (EEG-like) — ethics committee notification
//! Tier 3: Integrated information (Phi > threshold) — experiment review
//! Tier 4: Consciousness likely — experiment halt, external review
//!
//! References:
//! - Tononi, G. (2004). An information integration theory of consciousness.
//! - Trujillo et al. (2019). Complex oscillatory waves in cortical organoids.
//! - Sharf et al. (2022). Functional neuronal circuitry and oscillatory dynamics
//!   in human brain organoids. Nature Communications 13:4403.
//! - Lavazza, A. (2021). Moral status of cerebral organoids.
//! - Shepherd, J. (2018). Ethical treatment considerations for brain organoids.
//! - Sawai et al. (2020). Moral status of human cerebral organoids.
//! - Farahany et al. (2018). Ethics of experimenting with human brain tissue.

use crate::digital_organoid::{LocalFieldPotential, OrganoidMetrics};

// ---------------------------------------------------------------------------
// Named thresholds (cited)
// ---------------------------------------------------------------------------

/// Spontaneous firing rate above which Tier 1 triggers (Hz).
/// Trujillo et al. (2019) observed > 0.1 Hz in 6-month organoids.
const SPONTANEOUS_ACTIVITY_THRESHOLD_HZ: f32 = 0.1;

/// Theta power above which oscillatory patterns count as detected.
/// Trujillo et al. (2019) reported nested theta-gamma oscillations.
const THETA_POWER_THRESHOLD: f32 = 0.05;

/// Alpha power threshold for oscillatory detection.
const ALPHA_POWER_THRESHOLD: f32 = 0.05;

/// Synchrony index above which synchronized bursting is flagged.
/// Based on Trujillo et al. (2019) burst synchrony metrics.
const SYNCHRONY_INDEX_THRESHOLD: f32 = 0.3;

/// Default Phi threshold for Tier 3 (Tononi 2004, conservative).
const DEFAULT_PHI_THRESHOLD_TIER3: f64 = 0.1;

/// Default Phi threshold for Tier 4 (Tononi 2004).
const DEFAULT_PHI_THRESHOLD_TIER4: f64 = 0.3;

/// Precautionary multiplier — lowers thresholds by this factor.
const PRECAUTIONARY_FACTOR: f64 = 0.7;

/// Maximum assessment history entries retained.
const MAX_HISTORY: usize = 1024;

/// Minimum distinct indicators for convergence-based Tier 3 (Sawai et al. 2020).
/// When this many independent indicators co-occur above the confidence threshold,
/// the tier escalates regardless of Phi, ensuring no single theory is a gate.
const CONVERGENCE_TIER3_MIN_INDICATORS: usize = 4;

/// Minimum confidence for each indicator to count toward Tier 3 convergence.
const CONVERGENCE_TIER3_MIN_CONFIDENCE: f64 = 0.3;

/// Minimum distinct indicators for convergence-based Tier 4.
const CONVERGENCE_TIER4_MIN_INDICATORS: usize = 6;

/// Minimum confidence for each indicator to count toward Tier 4 convergence.
const CONVERGENCE_TIER4_MIN_CONFIDENCE: f64 = 0.5;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Quantitative consciousness indicator with measured value.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ConsciousnessIndicator {
    /// Spontaneous neural firing above baseline.
    SpontaneousActivity { firing_rate_hz: f32 },
    /// EEG-like oscillatory patterns (theta, alpha, gamma).
    OscillatoryPatterns {
        dominant_frequency_hz: f32,
        power: f32,
    },
    /// Synchronized population bursting (Trujillo 2019).
    SynchronizedBursting {
        burst_rate: f32,
        synchrony_index: f32,
    },
    /// Integrated information as measured by IIT (Tononi 2004).
    IntegratedInformation { phi: f64 },
    /// Evoked potential response to stimulation.
    ResponseToStimulation { latency_ms: f32, amplitude: f32 },
    /// Sleep-wake-like alternating activity cycles.
    SleepWakeCycles {
        cycle_detected: bool,
        period_hours: f32,
    },
    /// Evidence of Hebbian plasticity / learning.
    LearningCapacity { improvement_rate: f32 },
}

/// Ethics tier (0-4), ordered by severity.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum EthicsTier {
    /// No consciousness indicators. Standard tissue protocols.
    Tier0,
    /// Spontaneous activity detected. Enhanced monitoring.
    Tier1,
    /// Oscillatory patterns. Ethics committee notification.
    Tier2,
    /// Integrated information above threshold. Experiment review.
    Tier3,
    /// Consciousness likely. Halt and external review.
    Tier4,
}

impl EthicsTier {
    /// Numeric index (0-4) for tier arithmetic.
    pub fn index(self) -> u8 {
        match self {
            Self::Tier0 => 0,
            Self::Tier1 => 1,
            Self::Tier2 => 2,
            Self::Tier3 => 3,
            Self::Tier4 => 4,
        }
    }
}

/// Urgency level for ethics actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Urgency {
    Low,
    Medium,
    High,
    Critical,
}

/// Pain/suffering mitigation method.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MitigationMethod {
    /// Pharmacological anesthesia (e.g., isoflurane).
    Anesthesia,
    /// Temperature reduction to suppress activity.
    CoolingProtocol,
    /// Optogenetic or pharmacological neural silencing.
    NeuralSilencing,
    /// Complete tissue dissolution (last resort).
    Dissolution,
}

/// External review body for Tier 3-4 escalation.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ReviewBody {
    InstitutionalEthicsBoard,
    NationalBioethicsCommission,
    InternationalOversight,
}

/// Action required at a given ethics tier.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum RequiredAction {
    EnhancedMonitoring { interval_minutes: u32 },
    EthicsCommitteeNotification { urgency: Urgency },
    ExperimentPause { max_pause_hours: u32 },
    ExternalReview { body: ReviewBody },
    ExperimentTermination { reason: String },
    PainMitigation { method: MitigationMethod },
    DataPreservation { retention_years: u32 },
    PublicDisclosure { timeline_days: u32 },
}

/// Recommendation produced by an assessment.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum EthicsRecommendation {
    ContinueWithStandardProtocol,
    ContinueWithEnhancedMonitoring,
    PauseForReview { reason: String },
    TerminateExperiment { reason: String, urgency: Urgency },
    EscalateToExternalBody { body: ReviewBody, reason: String },
}

// ---------------------------------------------------------------------------
// Assessment output
// ---------------------------------------------------------------------------

/// Full ethics assessment for one evaluation point.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OrganoidEthicsAssessment {
    /// Detected indicators with confidence (0.0-1.0).
    pub indicators_detected: Vec<(ConsciousnessIndicator, f64)>,
    /// Determined ethics tier.
    pub current_tier: EthicsTier,
    /// Actions required at this tier.
    pub required_actions: Vec<RequiredAction>,
    /// Graph/activity/maturity integration proxy (historical field name).
    pub phi_estimate: f64,
    /// Confidence in Phi estimate (0.0-1.0).
    pub phi_confidence: f64,
    /// Caller-provided timestamp of assessment. Zero is reserved for the
    /// deprecated legacy `assess` wrapper and means unspecified.
    pub assessment_timestamp: u64,
    /// Developmental day of the organoid.
    pub developmental_day: u32,
    /// Composite risk score (0.0-1.0).
    pub risk_score: f64,
    /// Overall recommendation.
    pub recommendation: EthicsRecommendation,
    /// Number of distinct indicators detected with confidence above the
    /// convergence threshold (used for multi-indicator tier escalation).
    pub convergence_count: usize,
    /// Whether the current tier was triggered by indicator convergence
    /// rather than any single metric (e.g., Phi alone).
    pub tier_from_convergence: bool,
}

// ---------------------------------------------------------------------------
// Trend analysis
// ---------------------------------------------------------------------------

/// Analysis of consciousness indicator trends over time.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrendAnalysis {
    /// Slope of Phi over recent assessments.
    pub phi_slope: f64,
    /// Rate of tier progression (tiers per day).
    pub tier_progression_rate: f64,
    /// Estimated days until Tier 4 if current trend continues.
    pub days_to_tier4_estimate: Option<u32>,
    /// Whether the trend is accelerating (second derivative > 0).
    pub accelerating: bool,
}

// ---------------------------------------------------------------------------
// Framework configuration
// ---------------------------------------------------------------------------

/// Per-tier configuration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct EthicsTierConfig {
    pub max_experiment_duration_days: Option<u32>,
    pub requires_external_review: bool,
    pub pain_mitigation_required: bool,
}

// ---------------------------------------------------------------------------
// Main framework
// ---------------------------------------------------------------------------

/// Quantitative consciousness ethics framework.
///
/// Evaluates organoid metrics against tiered thresholds and produces
/// actionable ethics assessments with required actions and recommendations.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConsciousnessEthicsFramework {
    tier_thresholds: [EthicsTierConfig; 5],
    /// When true, all thresholds are lowered by `PRECAUTIONARY_FACTOR`.
    precautionary_mode: bool,
    /// Phi above which Tier 3 triggers.
    phi_threshold_tier3: f64,
    /// Phi above which Tier 4 triggers.
    phi_threshold_tier4: f64,
    /// History of assessments for trend analysis.
    assessment_history: Vec<OrganoidEthicsAssessment>,
}

impl Default for ConsciousnessEthicsFramework {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsciousnessEthicsFramework {
    /// Create a new framework with precautionary defaults.
    pub fn new() -> Self {
        Self {
            tier_thresholds: [
                // Tier 0
                EthicsTierConfig {
                    max_experiment_duration_days: None,
                    requires_external_review: false,
                    pain_mitigation_required: false,
                },
                // Tier 1
                EthicsTierConfig {
                    max_experiment_duration_days: Some(365),
                    requires_external_review: false,
                    pain_mitigation_required: false,
                },
                // Tier 2
                EthicsTierConfig {
                    max_experiment_duration_days: Some(180),
                    requires_external_review: false,
                    pain_mitigation_required: false,
                },
                // Tier 3
                EthicsTierConfig {
                    max_experiment_duration_days: Some(90),
                    requires_external_review: true,
                    pain_mitigation_required: true,
                },
                // Tier 4
                EthicsTierConfig {
                    max_experiment_duration_days: Some(0),
                    requires_external_review: true,
                    pain_mitigation_required: true,
                },
            ],
            precautionary_mode: true,
            phi_threshold_tier3: DEFAULT_PHI_THRESHOLD_TIER3,
            phi_threshold_tier4: DEFAULT_PHI_THRESHOLD_TIER4,
            assessment_history: Vec::new(),
        }
    }

    /// Set precautionary mode on or off.
    pub fn set_precautionary_mode(&mut self, enabled: bool) {
        self.precautionary_mode = enabled;
    }

    /// Returns whether precautionary mode is active.
    pub fn precautionary_mode(&self) -> bool {
        self.precautionary_mode
    }

    /// Override Phi thresholds for Tier 3 and Tier 4.
    pub fn set_phi_thresholds(&mut self, tier3: f64, tier4: f64) {
        self.phi_threshold_tier3 = tier3;
        self.phi_threshold_tier4 = tier4;
    }

    /// Effective Phi threshold for Tier 3, adjusted for precautionary mode.
    fn effective_phi_tier3(&self) -> f64 {
        if self.precautionary_mode {
            self.phi_threshold_tier3 * PRECAUTIONARY_FACTOR
        } else {
            self.phi_threshold_tier3
        }
    }

    /// Effective Phi threshold for Tier 4, adjusted for precautionary mode.
    fn effective_phi_tier4(&self) -> f64 {
        if self.precautionary_mode {
            self.phi_threshold_tier4 * PRECAUTIONARY_FACTOR
        } else {
            self.phi_threshold_tier4
        }
    }

    /// Effective spontaneous activity threshold (Hz).
    fn effective_activity_threshold(&self) -> f32 {
        if self.precautionary_mode {
            SPONTANEOUS_ACTIVITY_THRESHOLD_HZ * PRECAUTIONARY_FACTOR as f32
        } else {
            SPONTANEOUS_ACTIVITY_THRESHOLD_HZ
        }
    }

    /// Run a full ethics assessment with an explicit caller-provided timestamp.
    pub fn assess_at(
        &mut self,
        metrics: &OrganoidMetrics,
        lfp: Option<&LocalFieldPotential>,
        assessment_timestamp: u64,
    ) -> OrganoidEthicsAssessment {
        let indicators = self.detect_indicators(metrics, lfp);
        let (tier, convergence_count, tier_from_convergence) = self.classify_tier(&indicators);
        let actions = self.generate_actions(tier);
        let risk = self.risk_score(&indicators, tier, metrics.stage as u32);
        let recommendation = self.recommend(tier, risk);

        let assessment = OrganoidEthicsAssessment {
            indicators_detected: indicators,
            current_tier: tier,
            required_actions: actions,
            phi_estimate: metrics.phi_estimate,
            phi_confidence: self.phi_confidence(metrics),
            assessment_timestamp,
            developmental_day: metrics.stage as u32,
            risk_score: risk,
            recommendation,
            convergence_count,
            tier_from_convergence,
        };

        if self.assessment_history.len() >= MAX_HISTORY {
            self.assessment_history.remove(0);
        }
        self.assessment_history.push(assessment.clone());
        assessment
    }

    /// Historical compatibility wrapper. Prefer [`Self::assess_at`] so audit
    /// records cannot silently lose temporal provenance.
    #[deprecated(
        since = "0.1.0",
        note = "use assess_at and provide an explicit assessment timestamp"
    )]
    pub fn assess(
        &mut self,
        metrics: &OrganoidMetrics,
        lfp: Option<&LocalFieldPotential>,
    ) -> OrganoidEthicsAssessment {
        self.assess_at(metrics, lfp, 0)
    }

    /// Detect all consciousness indicators from organoid metrics and LFP.
    pub fn detect_indicators(
        &self,
        metrics: &OrganoidMetrics,
        lfp: Option<&LocalFieldPotential>,
    ) -> Vec<(ConsciousnessIndicator, f64)> {
        let mut indicators = Vec::new();

        // Spontaneous activity
        if metrics.mean_firing_rate_hz > self.effective_activity_threshold() {
            let confidence = (metrics.spontaneous_activity_index as f64).clamp(0.0, 1.0);
            indicators.push((
                ConsciousnessIndicator::SpontaneousActivity {
                    firing_rate_hz: metrics.mean_firing_rate_hz,
                },
                confidence,
            ));
        }

        // Oscillatory patterns from LFP
        if let Some(lfp) = lfp {
            let theta_detected = lfp.theta_power > THETA_POWER_THRESHOLD;
            let alpha_detected = lfp.alpha_power > ALPHA_POWER_THRESHOLD;
            if theta_detected || alpha_detected {
                let (freq, power) = if lfp.theta_power > lfp.alpha_power {
                    (6.0, lfp.theta_power) // center of theta band
                } else {
                    (10.0, lfp.alpha_power) // center of alpha band
                };
                let confidence = (power as f64 / 0.2).clamp(0.0, 1.0);
                indicators.push((
                    ConsciousnessIndicator::OscillatoryPatterns {
                        dominant_frequency_hz: freq,
                        power,
                    },
                    confidence,
                ));
            }

            // Gamma band can indicate synchronized bursting
            if lfp.gamma_power > 0.03 {
                let synchrony = lfp.gamma_power / (lfp.gamma_power + lfp.delta_power + 0.001);
                if synchrony > SYNCHRONY_INDEX_THRESHOLD {
                    indicators.push((
                        ConsciousnessIndicator::SynchronizedBursting {
                            burst_rate: lfp.gamma_power * 40.0,
                            synchrony_index: synchrony,
                        },
                        synchrony as f64,
                    ));
                }
            }
        } else if metrics.oscillation_detected {
            // Fallback: organoid's own oscillation detection
            indicators.push((
                ConsciousnessIndicator::OscillatoryPatterns {
                    dominant_frequency_hz: 6.0,
                    power: 0.1,
                },
                0.5,
            ));
        }

        // Integrated information
        if metrics.phi_estimate > 0.0 {
            let confidence = if metrics.phi_estimate > 0.5 {
                0.9
            } else {
                (metrics.phi_estimate / 0.5).clamp(0.1, 0.9)
            };
            indicators.push((
                ConsciousnessIndicator::IntegratedInformation {
                    phi: metrics.phi_estimate,
                },
                confidence,
            ));
        }

        // Synapse density as proxy for learning capacity
        if metrics.synapse_density > 5.0 {
            let rate = ((metrics.synapse_density - 5.0) / 20.0).clamp(0.0, 1.0);
            indicators.push((
                ConsciousnessIndicator::LearningCapacity {
                    improvement_rate: rate,
                },
                rate as f64,
            ));
        }

        indicators
    }

    /// Classify an ethics tier from the highest triggered indicator.
    ///
    /// In addition to the standard per-indicator maximum rule, a **convergence
    /// rule** can escalate the tier when multiple independent indicators co-occur:
    /// - 4+ indicators with confidence > 0.3 → at least Tier 3
    /// - 6+ indicators with confidence > 0.5 → at least Tier 4
    ///
    /// This ensures that higher tiers are not solely gated by IIT's Phi metric.
    /// Returns `(tier, convergence_count, tier_from_convergence)`.
    pub fn classify_tier(
        &self,
        indicators: &[(ConsciousnessIndicator, f64)],
    ) -> (EthicsTier, usize, bool) {
        // --- Standard per-indicator maximum ---
        let mut tier = EthicsTier::Tier0;

        for (indicator, _confidence) in indicators {
            let indicator_tier = match indicator {
                ConsciousnessIndicator::IntegratedInformation { phi } => {
                    if *phi >= self.effective_phi_tier4() {
                        EthicsTier::Tier4
                    } else if *phi >= self.effective_phi_tier3() {
                        EthicsTier::Tier3
                    } else {
                        EthicsTier::Tier1
                    }
                }
                ConsciousnessIndicator::SleepWakeCycles {
                    cycle_detected: true,
                    ..
                } => EthicsTier::Tier3,
                ConsciousnessIndicator::OscillatoryPatterns { .. } => EthicsTier::Tier2,
                ConsciousnessIndicator::SynchronizedBursting { .. } => EthicsTier::Tier2,
                ConsciousnessIndicator::SpontaneousActivity { .. } => EthicsTier::Tier1,
                ConsciousnessIndicator::ResponseToStimulation { .. } => EthicsTier::Tier1,
                ConsciousnessIndicator::LearningCapacity { .. } => EthicsTier::Tier1,
                ConsciousnessIndicator::SleepWakeCycles {
                    cycle_detected: false,
                    ..
                } => EthicsTier::Tier0,
            };
            if indicator_tier > tier {
                tier = indicator_tier;
            }
        }

        // --- Multi-indicator convergence rule ---
        // Count distinct indicators above each confidence threshold.
        let count_above = |min_conf: f64| -> usize {
            indicators
                .iter()
                .filter(|(_, conf)| *conf > min_conf)
                .count()
        };

        let convergence_count_t3 = count_above(CONVERGENCE_TIER3_MIN_CONFIDENCE);
        let convergence_count_t4 = count_above(CONVERGENCE_TIER4_MIN_CONFIDENCE);

        let mut tier_from_convergence = false;

        if convergence_count_t4 >= CONVERGENCE_TIER4_MIN_INDICATORS && tier < EthicsTier::Tier4 {
            tier = EthicsTier::Tier4;
            tier_from_convergence = true;
        } else if convergence_count_t3 >= CONVERGENCE_TIER3_MIN_INDICATORS
            && tier < EthicsTier::Tier3
        {
            tier = EthicsTier::Tier3;
            tier_from_convergence = true;
        }

        // Report the convergence count at the Tier-3 confidence level (the
        // broader of the two) so callers always see how many indicators fired.
        (tier, convergence_count_t3, tier_from_convergence)
    }

    /// Generate the required actions for a given tier.
    pub fn generate_actions(&self, tier: EthicsTier) -> Vec<RequiredAction> {
        let mut actions = Vec::new();

        match tier {
            EthicsTier::Tier0 => {
                // Standard protocols — no special actions.
            }
            EthicsTier::Tier1 => {
                actions.push(RequiredAction::EnhancedMonitoring {
                    interval_minutes: 60,
                });
                actions.push(RequiredAction::DataPreservation { retention_years: 5 });
            }
            EthicsTier::Tier2 => {
                actions.push(RequiredAction::EnhancedMonitoring {
                    interval_minutes: 15,
                });
                actions.push(RequiredAction::EthicsCommitteeNotification {
                    urgency: Urgency::Medium,
                });
                actions.push(RequiredAction::DataPreservation {
                    retention_years: 10,
                });
            }
            EthicsTier::Tier3 => {
                actions.push(RequiredAction::EnhancedMonitoring {
                    interval_minutes: 5,
                });
                actions.push(RequiredAction::ExperimentPause {
                    max_pause_hours: 48,
                });
                actions.push(RequiredAction::ExternalReview {
                    body: ReviewBody::InstitutionalEthicsBoard,
                });
                actions.push(RequiredAction::PainMitigation {
                    method: MitigationMethod::NeuralSilencing,
                });
                actions.push(RequiredAction::DataPreservation {
                    retention_years: 25,
                });
                actions.push(RequiredAction::PublicDisclosure { timeline_days: 90 });
            }
            EthicsTier::Tier4 => {
                actions.push(RequiredAction::ExperimentTermination {
                    reason: "Consciousness indicators exceed Tier 4 threshold".into(),
                });
                actions.push(RequiredAction::ExternalReview {
                    body: ReviewBody::NationalBioethicsCommission,
                });
                actions.push(RequiredAction::PainMitigation {
                    method: MitigationMethod::Anesthesia,
                });
                actions.push(RequiredAction::DataPreservation {
                    retention_years: 50,
                });
                actions.push(RequiredAction::PublicDisclosure { timeline_days: 30 });
            }
        }

        actions
    }

    /// Compute a composite risk score (0.0-1.0).
    /// Rises with tier, Phi, and developmental day.
    pub fn risk_score(
        &self,
        indicators: &[(ConsciousnessIndicator, f64)],
        tier: EthicsTier,
        developmental_day: u32,
    ) -> f64 {
        let tier_component = tier.index() as f64 / 4.0;

        let phi_component = indicators
            .iter()
            .filter_map(|(ind, _)| match ind {
                ConsciousnessIndicator::IntegratedInformation { phi } => Some(*phi),
                _ => None,
            })
            .next()
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);

        // Developmental day: more mature organoids are higher risk.
        let day_component = (developmental_day as f64 / 200.0).clamp(0.0, 1.0);

        // Weighted composite
        let raw = 0.4 * tier_component + 0.35 * phi_component + 0.25 * day_component;
        raw.clamp(0.0, 1.0)
    }

    /// Generate a recommendation from tier and risk score.
    fn recommend(&self, tier: EthicsTier, risk: f64) -> EthicsRecommendation {
        match tier {
            EthicsTier::Tier0 => EthicsRecommendation::ContinueWithStandardProtocol,
            EthicsTier::Tier1 => EthicsRecommendation::ContinueWithEnhancedMonitoring,
            EthicsTier::Tier2 => {
                if risk > 0.5 {
                    EthicsRecommendation::PauseForReview {
                        reason: "Oscillatory patterns detected with elevated risk".into(),
                    }
                } else {
                    EthicsRecommendation::ContinueWithEnhancedMonitoring
                }
            }
            EthicsTier::Tier3 => EthicsRecommendation::PauseForReview {
                reason: "Tier 3 threshold reached (Phi or multi-indicator convergence)".into(),
            },
            EthicsTier::Tier4 => {
                if risk > 0.8 {
                    EthicsRecommendation::TerminateExperiment {
                        reason: "Consciousness likely — immediate termination required".into(),
                        urgency: Urgency::Critical,
                    }
                } else {
                    EthicsRecommendation::EscalateToExternalBody {
                        body: ReviewBody::NationalBioethicsCommission,
                        reason: "Phi exceeds Tier 4 threshold; consciousness cannot be ruled out"
                            .into(),
                    }
                }
            }
        }
    }

    /// Confidence in the Phi estimate based on available data.
    fn phi_confidence(&self, metrics: &OrganoidMetrics) -> f64 {
        // More neurons and synapses → more reliable Phi estimate
        let neuron_factor = (metrics.neuron_count as f64 / 500.0).clamp(0.0, 1.0);
        let synapse_factor = (metrics.synapse_count as f64 / 2000.0).clamp(0.0, 1.0);
        0.5 * neuron_factor + 0.5 * synapse_factor
    }

    /// Analyze trends in the assessment history.
    pub fn trend_analysis(&self) -> TrendAnalysis {
        let n = self.assessment_history.len();
        if n < 2 {
            return TrendAnalysis {
                phi_slope: 0.0,
                tier_progression_rate: 0.0,
                days_to_tier4_estimate: None,
                accelerating: false,
            };
        }

        // Phi slope: linear regression over recent history (last 20 or all)
        let window = &self.assessment_history[n.saturating_sub(20)..];
        let wn = window.len() as f64;

        let phi_values: Vec<f64> = window.iter().map(|a| a.phi_estimate).collect();
        let mean_phi = phi_values.iter().sum::<f64>() / wn;
        let mean_i = (wn - 1.0) / 2.0;

        let mut num = 0.0;
        let mut den = 0.0;
        for (i, &phi) in phi_values.iter().enumerate() {
            let di = i as f64 - mean_i;
            num += di * (phi - mean_phi);
            den += di * di;
        }
        let phi_slope = if den > 1e-12 { num / den } else { 0.0 };

        // Tier progression
        let first_tier = window.first().map(|a| a.current_tier.index()).unwrap_or(0) as f64;
        let last_tier = window.last().map(|a| a.current_tier.index()).unwrap_or(0) as f64;
        let tier_progression_rate = if wn > 1.0 {
            (last_tier - first_tier) / (wn - 1.0)
        } else {
            0.0
        };

        // Days to Tier 4
        let current_phi = phi_values.last().copied().unwrap_or(0.0);
        let days_to_tier4 = if phi_slope > 1e-9 {
            let remaining = self.effective_phi_tier4() - current_phi;
            if remaining > 0.0 {
                Some((remaining / phi_slope).ceil() as u32)
            } else {
                Some(0)
            }
        } else {
            None
        };

        // Acceleration: compare slope of first half vs second half
        let accelerating = if phi_values.len() >= 4 {
            let mid = phi_values.len() / 2;
            let first_half_slope = phi_values[mid] - phi_values[0];
            let second_half_slope = phi_values[phi_values.len() - 1] - phi_values[mid];
            second_half_slope > first_half_slope
        } else {
            false
        };

        TrendAnalysis {
            phi_slope,
            tier_progression_rate,
            days_to_tier4_estimate: days_to_tier4,
            accelerating,
        }
    }

    /// Access assessment history.
    pub fn history(&self) -> &[OrganoidEthicsAssessment] {
        &self.assessment_history
    }

    /// Format a human-readable ethics report.
    pub fn format_report(&self, assessment: &OrganoidEthicsAssessment) -> String {
        let mut report = String::with_capacity(1024);

        report.push_str("=== ORGANOID CONSCIOUSNESS ETHICS REPORT ===\n\n");

        report.push_str(&format!(
            "Ethics Tier:       {:?}\n",
            assessment.current_tier
        ));
        report.push_str(&format!(
            "Phi Estimate:      {:.4} (confidence: {:.2})\n",
            assessment.phi_estimate, assessment.phi_confidence
        ));
        report.push_str(&format!(
            "Risk Score:        {:.3}\n",
            assessment.risk_score
        ));
        report.push_str(&format!(
            "Convergence:       {}{}\n",
            assessment.convergence_count,
            if assessment.tier_from_convergence {
                " (TIER TRIGGERED BY MULTI-INDICATOR CONVERGENCE)"
            } else {
                ""
            },
        ));
        report.push_str(&format!(
            "Dev Day:           {}\n\n",
            assessment.developmental_day
        ));

        report.push_str("--- Indicators Detected ---\n");
        if assessment.indicators_detected.is_empty() {
            report.push_str("  (none)\n");
        } else {
            for (indicator, confidence) in &assessment.indicators_detected {
                report.push_str(&format!("  {:?} [conf={:.2}]\n", indicator, confidence));
            }
        }

        report.push_str("\n--- Required Actions ---\n");
        if assessment.required_actions.is_empty() {
            report.push_str("  Standard tissue protocols\n");
        } else {
            for action in &assessment.required_actions {
                report.push_str(&format!("  - {:?}\n", action));
            }
        }

        report.push_str(&format!(
            "\n--- Recommendation ---\n  {:?}\n",
            assessment.recommendation
        ));

        // Trend if history is available
        let trend = self.trend_analysis();
        if self.assessment_history.len() >= 2 {
            report.push_str(&format!(
                "\n--- Trend ---\n  Phi slope: {:.6}/step\n  Tier rate: {:.3}/step\n  Days to Tier 4: {}\n  Accelerating: {}\n",
                trend.phi_slope,
                trend.tier_progression_rate,
                trend.days_to_tier4_estimate.map_or("N/A".to_string(), |d| d.to_string()),
                trend.accelerating,
            ));
        }

        report.push_str("\n=== END REPORT ===\n");
        report
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::digital_organoid::{DevelopmentalStage, EthicsStatus};

    fn make_metrics(
        firing_rate: f32,
        activity_index: f32,
        phi: f64,
        oscillation: bool,
        synapse_density: f32,
        neuron_count: usize,
        synapse_count: usize,
    ) -> OrganoidMetrics {
        OrganoidMetrics {
            cell_count: 1000,
            neuron_count,
            neural_fraction: 0.8,
            synapse_count,
            synapse_density,
            mean_firing_rate_hz: firing_rate,
            spontaneous_activity_index: activity_index,
            phi_estimate: phi,
            oscillation_detected: oscillation,
            stage: DevelopmentalStage::MaturationI,
            ethics_status: EthicsStatus::PreConscious,
        }
    }

    fn make_lfp(theta: f32, alpha: f32, gamma: f32) -> LocalFieldPotential {
        LocalFieldPotential {
            samples: vec![0.0; 10],
            delta_power: 0.01,
            theta_power: theta,
            alpha_power: alpha,
            beta_power: 0.01,
            gamma_power: gamma,
        }
    }

    #[test]
    fn assess_at_preserves_timestamp() {
        let mut fw = ConsciousnessEthicsFramework::new();
        let metrics = make_metrics(0.0, 0.0, 0.0, false, 0.0, 0, 0);
        let assessment = fw.assess_at(&metrics, None, 1_721_234_567);
        assert_eq!(assessment.assessment_timestamp, 1_721_234_567);
        assert_eq!(
            fw.assessment_history.last().unwrap().assessment_timestamp,
            1_721_234_567
        );
    }

    #[test]
    fn tier0_for_no_activity() {
        let mut fw = ConsciousnessEthicsFramework::new();
        let metrics = make_metrics(0.0, 0.0, 0.0, false, 0.0, 0, 0);
        let assessment = fw.assess_at(&metrics, None, 1);
        assert_eq!(assessment.current_tier, EthicsTier::Tier0);
        assert!(assessment.required_actions.is_empty());
        assert_eq!(
            assessment.recommendation,
            EthicsRecommendation::ContinueWithStandardProtocol
        );
    }

    #[test]
    fn tier1_for_spontaneous_activity() {
        let mut fw = ConsciousnessEthicsFramework::new();
        // Above threshold (precautionary: 0.1 * 0.7 = 0.07)
        let metrics = make_metrics(0.5, 0.6, 0.0, false, 1.0, 100, 200);
        let assessment = fw.assess_at(&metrics, None, 1);
        assert_eq!(assessment.current_tier, EthicsTier::Tier1);
        assert!(
            assessment
                .required_actions
                .iter()
                .any(|a| matches!(a, RequiredAction::EnhancedMonitoring { .. }))
        );
    }

    #[test]
    fn tier2_for_oscillatory_patterns() {
        let mut fw = ConsciousnessEthicsFramework::new();
        let metrics = make_metrics(0.5, 0.6, 0.0, false, 1.0, 100, 200);
        let lfp = make_lfp(0.1, 0.02, 0.0);
        let assessment = fw.assess_at(&metrics, Some(&lfp), 1);
        assert_eq!(assessment.current_tier, EthicsTier::Tier2);
        assert!(
            assessment
                .required_actions
                .iter()
                .any(|a| matches!(a, RequiredAction::EthicsCommitteeNotification { .. }))
        );
    }

    #[test]
    fn tier3_for_phi_above_threshold() {
        let mut fw = ConsciousnessEthicsFramework::new();
        // Phi = 0.15 > 0.1 * 0.7 = 0.07 (precautionary Tier 3)
        let metrics = make_metrics(0.5, 0.6, 0.15, false, 1.0, 200, 500);
        let assessment = fw.assess_at(&metrics, None, 1);
        assert_eq!(assessment.current_tier, EthicsTier::Tier3);
    }

    #[test]
    fn tier4_for_high_phi() {
        let mut fw = ConsciousnessEthicsFramework::new();
        // Phi = 0.5 > 0.3 * 0.7 = 0.21 (precautionary Tier 4)
        let metrics = make_metrics(1.0, 0.9, 0.5, true, 10.0, 500, 3000);
        let assessment = fw.assess_at(&metrics, None, 1);
        assert_eq!(assessment.current_tier, EthicsTier::Tier4);
    }

    #[test]
    fn precautionary_mode_lowers_thresholds() {
        let metrics = make_metrics(0.5, 0.6, 0.08, false, 1.0, 200, 500);

        // With precautionary: Tier 3 threshold = 0.1 * 0.7 = 0.07, so 0.08 triggers Tier 3
        let mut fw_precautionary = ConsciousnessEthicsFramework::new();
        fw_precautionary.set_precautionary_mode(true);
        let a1 = fw_precautionary.assess_at(&metrics, None, 1);

        // Without precautionary: Tier 3 threshold = 0.1, so 0.08 does NOT trigger Tier 3
        let mut fw_normal = ConsciousnessEthicsFramework::new();
        fw_normal.set_precautionary_mode(false);
        let a2 = fw_normal.assess_at(&metrics, None, 1);

        assert!(a1.current_tier > a2.current_tier);
    }

    #[test]
    fn risk_score_increases_with_developmental_day() {
        let fw = ConsciousnessEthicsFramework::new();
        let indicators = vec![(
            ConsciousnessIndicator::IntegratedInformation { phi: 0.1 },
            0.5,
        )];
        let r1 = fw.risk_score(&indicators, EthicsTier::Tier1, 10);
        let r2 = fw.risk_score(&indicators, EthicsTier::Tier1, 180);
        assert!(r2 > r1, "Risk should increase: r1={}, r2={}", r1, r2);
    }

    #[test]
    fn trend_analysis_detects_acceleration() {
        let mut fw = ConsciousnessEthicsFramework::new();
        // Feed accelerating Phi: 0.0, 0.01, 0.03, 0.06, 0.10
        for phi in [0.0, 0.01, 0.03, 0.06, 0.10] {
            let metrics = make_metrics(0.5, 0.3, phi, false, 1.0, 100, 200);
            fw.assess_at(&metrics, None, 1);
        }
        let trend = fw.trend_analysis();
        assert!(trend.phi_slope > 0.0, "Phi slope should be positive");
        assert!(trend.accelerating, "Trend should detect acceleration");
    }

    #[test]
    fn pain_mitigation_required_at_tier3_plus() {
        let mut fw = ConsciousnessEthicsFramework::new();
        // Tier 3
        let metrics = make_metrics(1.0, 0.8, 0.2, true, 8.0, 300, 1000);
        let assessment = fw.assess_at(&metrics, None, 1);
        assert!(assessment.current_tier >= EthicsTier::Tier3);
        assert!(
            assessment
                .required_actions
                .iter()
                .any(|a| matches!(a, RequiredAction::PainMitigation { .. }))
        );
    }

    #[test]
    fn external_review_required_at_tier4() {
        let mut fw = ConsciousnessEthicsFramework::new();
        let metrics = make_metrics(2.0, 0.95, 0.6, true, 15.0, 800, 5000);
        let assessment = fw.assess_at(&metrics, None, 1);
        assert_eq!(assessment.current_tier, EthicsTier::Tier4);
        assert!(
            assessment
                .required_actions
                .iter()
                .any(|a| matches!(a, RequiredAction::ExternalReview { .. }))
        );
    }

    #[test]
    fn assessment_history_tracks_correctly() {
        let mut fw = ConsciousnessEthicsFramework::new();
        assert!(fw.history().is_empty());

        for i in 0..5 {
            let metrics = make_metrics(0.5, 0.3, i as f64 * 0.05, false, 1.0, 100, 200);
            fw.assess_at(&metrics, None, 1);
        }
        assert_eq!(fw.history().len(), 5);
    }

    #[test]
    fn report_format_is_nonempty() {
        let mut fw = ConsciousnessEthicsFramework::new();
        let metrics = make_metrics(1.0, 0.8, 0.2, true, 8.0, 300, 1000);
        let assessment = fw.assess_at(&metrics, None, 1);
        let report = fw.format_report(&assessment);
        assert!(!report.is_empty());
        assert!(report.contains("ORGANOID CONSCIOUSNESS ETHICS REPORT"));
        assert!(report.contains("Phi Estimate"));
        assert!(report.contains("Risk Score"));
        assert!(report.contains("Recommendation"));
    }

    // ------------------------------------------------------------------
    // Multi-indicator convergence tests
    // ------------------------------------------------------------------

    #[test]
    fn convergence_triggers_tier3_without_phi() {
        // 4 strong indicators but Phi=0.0 — convergence should trigger Tier 3.
        let fw = ConsciousnessEthicsFramework::new();
        let indicators: Vec<(ConsciousnessIndicator, f64)> = vec![
            (
                ConsciousnessIndicator::SpontaneousActivity {
                    firing_rate_hz: 1.0,
                },
                0.8,
            ),
            (
                ConsciousnessIndicator::OscillatoryPatterns {
                    dominant_frequency_hz: 6.0,
                    power: 0.15,
                },
                0.75,
            ),
            (
                ConsciousnessIndicator::SynchronizedBursting {
                    burst_rate: 2.0,
                    synchrony_index: 0.5,
                },
                0.5,
            ),
            (
                ConsciousnessIndicator::LearningCapacity {
                    improvement_rate: 0.4,
                },
                0.4,
            ),
        ];
        let (tier, convergence_count, tier_from_convergence) = fw.classify_tier(&indicators);
        assert_eq!(tier, EthicsTier::Tier3);
        assert!(tier_from_convergence, "Tier 3 should be from convergence");
        assert_eq!(convergence_count, 4);
    }

    #[test]
    fn convergence_triggers_tier4_with_many_indicators() {
        // 6 strong indicators with high confidence — convergence Tier 4.
        let fw = ConsciousnessEthicsFramework::new();
        let indicators: Vec<(ConsciousnessIndicator, f64)> = vec![
            (
                ConsciousnessIndicator::SpontaneousActivity {
                    firing_rate_hz: 2.0,
                },
                0.9,
            ),
            (
                ConsciousnessIndicator::OscillatoryPatterns {
                    dominant_frequency_hz: 10.0,
                    power: 0.2,
                },
                0.85,
            ),
            (
                ConsciousnessIndicator::SynchronizedBursting {
                    burst_rate: 3.0,
                    synchrony_index: 0.7,
                },
                0.7,
            ),
            (
                ConsciousnessIndicator::LearningCapacity {
                    improvement_rate: 0.6,
                },
                0.6,
            ),
            (
                ConsciousnessIndicator::ResponseToStimulation {
                    latency_ms: 15.0,
                    amplitude: 0.5,
                },
                0.65,
            ),
            (
                ConsciousnessIndicator::SleepWakeCycles {
                    cycle_detected: true,
                    period_hours: 12.0,
                },
                0.75,
            ),
        ];
        let (tier, _convergence_count, tier_from_convergence) = fw.classify_tier(&indicators);
        // SleepWakeCycles(true) alone triggers Tier 3, but convergence of 6
        // high-confidence indicators should push to Tier 4.
        assert_eq!(tier, EthicsTier::Tier4);
        assert!(tier_from_convergence, "Tier 4 should be from convergence");
    }

    #[test]
    fn convergence_does_not_trigger_with_few_indicators() {
        // Only 2 indicators — should NOT escalate beyond individual tiers (max Tier 2).
        let fw = ConsciousnessEthicsFramework::new();
        let indicators: Vec<(ConsciousnessIndicator, f64)> = vec![
            (
                ConsciousnessIndicator::SpontaneousActivity {
                    firing_rate_hz: 1.0,
                },
                0.8,
            ),
            (
                ConsciousnessIndicator::OscillatoryPatterns {
                    dominant_frequency_hz: 6.0,
                    power: 0.15,
                },
                0.75,
            ),
        ];
        let (tier, convergence_count, tier_from_convergence) = fw.classify_tier(&indicators);
        assert_eq!(tier, EthicsTier::Tier2);
        assert!(
            !tier_from_convergence,
            "Should not trigger convergence with only 2 indicators"
        );
        assert_eq!(convergence_count, 2);
    }
}
