//! Somatic manager: groups virtual body, user state inference, somatic error bridge,
//! thermodynamic load, mood temperature, and pain channel.

use crate::infrastructure::somatic_error_bridge::SomaticErrorBridge;

/// Consolidates the somatic-related CLS fields.
pub(crate) struct SomaticManager {
    /// Virtual body for embodied cognition.
    pub virtual_body: Option<super::virtual_body::VirtualBody>,

    /// User state inference for tracking external user.
    pub user_state: Option<crate::user_state_inference::UserStateInference>,

    /// Somatic error bridge for interoceptive signals.
    pub somatic_bridge: SomaticErrorBridge,

    /// Cumulative thermodynamic load (0.0-1.0).
    pub thermodynamic_load: f32,

    /// Current mood temperature scalar.
    pub mood_temperature: f32,

    /// Pain channel sender for downstream consumers.
    pub pain_tx: Option<crate::infrastructure::somatic_error_bridge::PainSender>,
}

impl SomaticManager {
    /// Create a new SomaticManager from its component fields.
    pub fn new(
        virtual_body: Option<super::virtual_body::VirtualBody>,
        user_state: Option<crate::user_state_inference::UserStateInference>,
        somatic_bridge: SomaticErrorBridge,
        pain_tx: Option<crate::infrastructure::somatic_error_bridge::PainSender>,
    ) -> Self {
        Self {
            virtual_body,
            user_state,
            somatic_bridge,
            thermodynamic_load: 0.0,
            mood_temperature: 0.5,
            pain_tx,
        }
    }
}
