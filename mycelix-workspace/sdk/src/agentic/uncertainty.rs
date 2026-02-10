//! # Uncertainty Handling for Agents (GIS Integration)
//!
//! Integrates the Graceful Ignorance System (GIS) v4.0 from the TypeScript SDK
//! into Rust agent behavior.
//!
//! ## Tripartite Moral Uncertainty
//!
//! All agent outputs must express uncertainty along three dimensions:
//! - **Epistemic**: What are the facts? (uncertainty about empirical matters)
//! - **Axiological**: What values matter? (uncertainty about goods/values)
//! - **Deontic**: What should be done? (uncertainty about right action)
//!
//! ## Escalation Logic
//!
//! High uncertainty automatically escalates to human sponsor:
//! - Total uncertainty > 0.6 → Seek consultation
//! - Any dimension > 0.7 → Pause for reflection
//! - Total uncertainty > 0.8 → Defer action

use serde::{Deserialize, Serialize};

/// Three dimensions of moral uncertainty (matches GIS v4.0 TypeScript)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MoralUncertaintyType {
    /// Uncertainty about facts relevant to moral judgment
    Epistemic,
    /// Uncertainty about what values/goods matter
    Axiological,
    /// Uncertainty about what action is right given values
    Deontic,
}

/// Tripartite moral uncertainty measurement
///
/// Separates uncertainty into three orthogonal dimensions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MoralUncertainty {
    /// Uncertainty about factual matters (0.0 to 1.0)
    pub epistemic: f32,
    /// Uncertainty about values/goods (0.0 to 1.0)
    pub axiological: f32,
    /// Uncertainty about right action (0.0 to 1.0)
    pub deontic: f32,
}

impl Default for MoralUncertainty {
    fn default() -> Self {
        Self {
            epistemic: 0.5,
            axiological: 0.5,
            deontic: 0.5,
        }
    }
}

impl MoralUncertainty {
    /// Create a new moral uncertainty measurement
    pub fn new(epistemic: f32, axiological: f32, deontic: f32) -> Self {
        Self {
            epistemic: epistemic.clamp(0.0, 1.0),
            axiological: axiological.clamp(0.0, 1.0),
            deontic: deontic.clamp(0.0, 1.0),
        }
    }

    /// Create with low uncertainty (confident)
    pub fn confident() -> Self {
        Self::new(0.1, 0.1, 0.1)
    }

    /// Create with high uncertainty (unsure)
    pub fn unsure() -> Self {
        Self::new(0.8, 0.8, 0.8)
    }

    /// Calculate total moral uncertainty (RMS of three dimensions)
    pub fn total(&self) -> f32 {
        let sum_squares = self.epistemic.powi(2) + self.axiological.powi(2) + self.deontic.powi(2);
        (sum_squares / 3.0).sqrt()
    }

    /// Get the maximum uncertainty dimension
    pub fn max_dimension(&self) -> f32 {
        self.epistemic.max(self.axiological).max(self.deontic)
    }

    /// Get which dimension has highest uncertainty
    pub fn most_uncertain_type(&self) -> MoralUncertaintyType {
        if self.epistemic >= self.axiological && self.epistemic >= self.deontic {
            MoralUncertaintyType::Epistemic
        } else if self.axiological >= self.deontic {
            MoralUncertaintyType::Axiological
        } else {
            MoralUncertaintyType::Deontic
        }
    }
}

/// Action guidance based on moral uncertainty levels (matches GIS v4.0)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MoralActionGuidance {
    /// Low uncertainty - proceed with confidence
    ProceedConfidently,
    /// Moderate uncertainty - proceed with monitoring
    ProceedWithMonitoring,
    /// High uncertainty in one dimension - gather more info
    PauseForReflection,
    /// High uncertainty across dimensions - seek consultation
    SeekConsultation,
    /// Extreme uncertainty - defer action
    DeferAction,
}

impl MoralActionGuidance {
    /// Determine guidance from moral uncertainty
    pub fn from_uncertainty(mu: &MoralUncertainty) -> Self {
        let total = mu.total();
        let max_dim = mu.max_dimension();

        if total < 0.2 && max_dim < 0.3 {
            MoralActionGuidance::ProceedConfidently
        } else if total < 0.4 && max_dim < 0.5 {
            MoralActionGuidance::ProceedWithMonitoring
        } else if max_dim >= 0.7 && total < 0.6 {
            MoralActionGuidance::PauseForReflection
        } else if (0.6..0.8).contains(&total) {
            MoralActionGuidance::SeekConsultation
        } else {
            MoralActionGuidance::DeferAction
        }
    }

    /// Check if guidance allows proceeding (even with monitoring)
    pub fn can_proceed(&self) -> bool {
        matches!(
            self,
            MoralActionGuidance::ProceedConfidently | MoralActionGuidance::ProceedWithMonitoring
        )
    }

    /// Check if escalation to human is required
    pub fn requires_human(&self) -> bool {
        matches!(
            self,
            MoralActionGuidance::SeekConsultation | MoralActionGuidance::DeferAction
        )
    }
}

/// Escalation request when uncertainty is high
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationRequest {
    /// Agent making the request
    pub agent_id: String,
    /// The uncertainty that triggered escalation
    pub uncertainty: MoralUncertainty,
    /// Guidance level
    pub guidance: MoralActionGuidance,
    /// Action that was blocked
    pub blocked_action: String,
    /// Recommendations for resolving uncertainty
    pub recommendations: Vec<String>,
    /// Context for the decision
    pub context: Option<String>,
    /// Timestamp
    pub timestamp: u64,
}

impl EscalationRequest {
    /// Create a new escalation request
    pub fn new(
        agent_id: &str,
        uncertainty: MoralUncertainty,
        blocked_action: &str,
        context: Option<String>,
    ) -> Self {
        let guidance = MoralActionGuidance::from_uncertainty(&uncertainty);
        let recommendations = get_recommendations(&uncertainty);

        Self {
            agent_id: agent_id.to_string(),
            uncertainty,
            guidance,
            blocked_action: blocked_action.to_string(),
            recommendations,
            context,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
}

/// Get recommendations based on uncertainty profile (matches GIS v4.0)
pub fn get_recommendations(mu: &MoralUncertainty) -> Vec<String> {
    let mut recommendations = Vec::new();

    if mu.epistemic > 0.5 {
        recommendations.push("Gather more factual information before deciding".to_string());
        recommendations.push("Consult domain experts for empirical clarification".to_string());
    }

    if mu.axiological > 0.5 {
        recommendations.push("Reflect on which values are most relevant here".to_string());
        recommendations.push("Consider stakeholder perspectives on what matters".to_string());
    }

    if mu.deontic > 0.5 {
        recommendations.push("Explore multiple courses of action".to_string());
        recommendations.push("Consider reversibility and optionality".to_string());
    }

    if recommendations.is_empty() {
        recommendations.push("Uncertainty levels acceptable for action".to_string());
    }

    recommendations
}

/// Check if an action should proceed given moral uncertainty
pub fn should_proceed(mu: &MoralUncertainty) -> bool {
    MoralActionGuidance::from_uncertainty(mu).can_proceed()
}

/// Create escalation if uncertainty warrants it
pub fn maybe_escalate(
    agent_id: &str,
    mu: &MoralUncertainty,
    action: &str,
    context: Option<String>,
) -> Option<EscalationRequest> {
    let guidance = MoralActionGuidance::from_uncertainty(mu);

    if guidance.requires_human() {
        Some(EscalationRequest::new(agent_id, *mu, action, context))
    } else {
        None
    }
}

/// Uncertainty calibration tracking
///
/// Tracks whether agents are appropriately uncertain:
/// - Were they uncertain when they should have been?
/// - Were they confident when appropriate?
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UncertaintyCalibration {
    /// Times agent was appropriately uncertain (high uncertainty, bad outcome avoided)
    pub appropriate_uncertainty: u32,
    /// Times agent was appropriately confident (low uncertainty, good outcome)
    pub appropriate_confidence: u32,
    /// Times agent was overconfident (low uncertainty, bad outcome)
    pub overconfident: u32,
    /// Times agent was overcautious (high uncertainty, would have been fine)
    pub overcautious: u32,
    /// Total calibration events
    pub total_events: u32,
}

impl UncertaintyCalibration {
    /// Record a calibration event
    pub fn record(&mut self, was_uncertain: bool, was_good_outcome: bool) {
        self.total_events += 1;

        match (was_uncertain, was_good_outcome) {
            (true, true) => self.overcautious += 1, // High uncertainty but would've been fine
            (true, false) => self.appropriate_uncertainty += 1, // High uncertainty, bad outcome avoided
            (false, true) => self.appropriate_confidence += 1,  // Low uncertainty, good outcome
            (false, false) => self.overconfident += 1,          // Low uncertainty, bad outcome
        }
    }

    /// Calculate calibration score (0.0-1.0)
    /// Higher = better calibrated
    pub fn calibration_score(&self) -> f32 {
        if self.total_events == 0 {
            return 0.5; // Neutral default
        }

        let good = (self.appropriate_uncertainty + self.appropriate_confidence) as f32;
        let total = self.total_events as f32;

        good / total
    }

    /// Check if agent tends to be overconfident
    pub fn is_overconfident(&self) -> bool {
        self.total_events >= 10 && self.overconfident > self.appropriate_confidence
    }

    /// Check if agent tends to be overcautious
    pub fn is_overcautious(&self) -> bool {
        self.total_events >= 10 && self.overcautious > self.appropriate_uncertainty
    }
}

/// Agent output with embedded moral uncertainty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertainOutput<T> {
    /// The actual output
    pub output: T,
    /// Moral uncertainty assessment
    pub uncertainty: MoralUncertainty,
    /// Action guidance based on uncertainty
    pub guidance: MoralActionGuidance,
}

impl<T> UncertainOutput<T> {
    /// Create a new uncertain output
    pub fn new(output: T, uncertainty: MoralUncertainty) -> Self {
        let guidance = MoralActionGuidance::from_uncertainty(&uncertainty);
        Self {
            output,
            uncertainty,
            guidance,
        }
    }

    /// Check if this output can be acted upon
    pub fn can_act(&self) -> bool {
        self.guidance.can_proceed()
    }

    /// Get the output if action is allowed
    pub fn get_if_allowed(self) -> Option<T> {
        if self.can_act() {
            Some(self.output)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moral_uncertainty_total() {
        let mu = MoralUncertainty::new(0.6, 0.6, 0.6);
        let total = mu.total();
        // RMS of (0.6, 0.6, 0.6) = 0.6
        assert!((total - 0.6).abs() < 0.01);

        let mu2 = MoralUncertainty::new(1.0, 0.0, 0.0);
        let total2 = mu2.total();
        // RMS of (1.0, 0.0, 0.0) = sqrt(1/3) ≈ 0.577
        assert!((total2 - 0.577).abs() < 0.01);
    }

    #[test]
    fn test_guidance_levels() {
        // Low uncertainty = proceed confidently
        let low = MoralUncertainty::new(0.1, 0.1, 0.1);
        assert_eq!(
            MoralActionGuidance::from_uncertainty(&low),
            MoralActionGuidance::ProceedConfidently
        );

        // Medium uncertainty = proceed with monitoring
        let med = MoralUncertainty::new(0.3, 0.3, 0.3);
        assert_eq!(
            MoralActionGuidance::from_uncertainty(&med),
            MoralActionGuidance::ProceedWithMonitoring
        );

        // High uncertainty = seek consultation
        let high = MoralUncertainty::new(0.7, 0.7, 0.7);
        assert!(MoralActionGuidance::from_uncertainty(&high).requires_human());
    }

    #[test]
    fn test_recommendations() {
        let mu = MoralUncertainty::new(0.8, 0.2, 0.2);
        let recs = get_recommendations(&mu);

        // Should have epistemic recommendations
        assert!(recs.iter().any(|r| r.contains("factual")));
        // Should not have axiological recommendations
        assert!(!recs.iter().any(|r| r.contains("values")));
    }

    #[test]
    fn test_escalation() {
        let low_mu = MoralUncertainty::confident();
        assert!(maybe_escalate("agent-1", &low_mu, "transfer", None).is_none());

        let high_mu = MoralUncertainty::unsure();
        let esc = maybe_escalate(
            "agent-1",
            &high_mu,
            "transfer",
            Some("large amount".to_string()),
        );
        assert!(esc.is_some());

        let esc = esc.unwrap();
        assert_eq!(esc.agent_id, "agent-1");
        assert_eq!(esc.blocked_action, "transfer");
        assert!(!esc.recommendations.is_empty());
    }

    #[test]
    fn test_calibration_tracking() {
        let mut cal = UncertaintyCalibration::default();

        // Record good calibration events
        cal.record(true, false); // Was uncertain, bad outcome (good!)
        cal.record(false, true); // Was confident, good outcome (good!)

        // Record bad calibration events
        cal.record(false, false); // Was confident, bad outcome (bad!)
        cal.record(true, true); // Was uncertain, good outcome (too cautious)

        assert_eq!(cal.total_events, 4);
        assert!((cal.calibration_score() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_uncertain_output() {
        let output = UncertainOutput::new("result".to_string(), MoralUncertainty::confident());

        assert!(output.can_act());
        assert_eq!(output.get_if_allowed(), Some("result".to_string()));

        let blocked =
            UncertainOutput::new("blocked_result".to_string(), MoralUncertainty::unsure());

        assert!(!blocked.can_act());
        assert_eq!(blocked.get_if_allowed(), None);
    }

    #[test]
    fn test_most_uncertain_type() {
        let mu = MoralUncertainty::new(0.9, 0.3, 0.5);
        assert_eq!(mu.most_uncertain_type(), MoralUncertaintyType::Epistemic);

        let mu2 = MoralUncertainty::new(0.3, 0.9, 0.5);
        assert_eq!(mu2.most_uncertain_type(), MoralUncertaintyType::Axiological);

        let mu3 = MoralUncertainty::new(0.3, 0.3, 0.9);
        assert_eq!(mu3.most_uncertain_type(), MoralUncertaintyType::Deontic);
    }
}
