use serde::{Deserialize, Serialize};

use crate::{NativeInteroceptiveModel, ViabilityChannel};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum InterventionKind {
    SetValue(f32),
    AddValue(f32),
}

impl InterventionKind {
    fn requested_value(self, before: f32) -> f32 {
        match self {
            Self::SetValue(value) => value,
            Self::AddValue(delta) => before + delta,
        }
    }

    fn validate(self) {
        let value = match self {
            Self::SetValue(value) | Self::AddValue(value) => value,
        };
        assert!(value.is_finite(), "intervention values must be finite");
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct InteroceptiveIntervention {
    pub channel: ViabilityChannel,
    pub kind: InterventionKind,
}

impl InteroceptiveIntervention {
    pub fn set(channel: ViabilityChannel, value: f32) -> Self {
        Self {
            channel,
            kind: InterventionKind::SetValue(value),
        }
    }

    pub fn add(channel: ViabilityChannel, delta: f32) -> Self {
        Self {
            channel,
            kind: InterventionKind::AddValue(delta),
        }
    }
}

/// Receipt for an explicit causal intervention.
///
/// Interventions reset measured velocity to zero so an exogenous state jump is
/// never silently reinterpreted as an endogenous trend by the kinematic forecast.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct InterventionRecord {
    pub cycle: u64,
    pub channel: ViabilityChannel,
    pub before: f32,
    pub requested: f32,
    pub after: f32,
    pub clamped: bool,
}

pub fn apply_intervention(
    model: &mut NativeInteroceptiveModel,
    intervention: InteroceptiveIntervention,
) -> InterventionRecord {
    intervention.kind.validate();

    let config = model.config();
    let cycle = model.cycle();
    let variable = model.state_mut().get_mut(intervention.channel);
    let before = variable.value();
    let requested = intervention.kind.requested_value(before);
    assert!(requested.is_finite(), "intervention target must be finite");
    let after = requested.clamp(config.min_value, config.max_value);

    variable.set_observation(after, 0.0);

    InterventionRecord {
        cycle,
        channel: intervention.channel,
        before,
        requested,
        after,
        clamped: after != requested,
    }
}
