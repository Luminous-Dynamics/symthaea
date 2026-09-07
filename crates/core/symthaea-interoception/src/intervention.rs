use serde::{de, Deserialize, Deserializer, Serialize};

use crate::{NativeInteroceptiveModel, ViabilityChannel};

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum InterventionKind {
    SetValue(f32),
    AddValue(f32),
}

#[derive(Deserialize)]
enum InterventionKindWire {
    SetValue(f32),
    AddValue(f32),
}

impl<'de> Deserialize<'de> for InterventionKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let kind = match InterventionKindWire::deserialize(deserializer)? {
            InterventionKindWire::SetValue(value) => Self::SetValue(value),
            InterventionKindWire::AddValue(value) => Self::AddValue(value),
        };
        kind.try_validate().map_err(de::Error::custom)?;
        Ok(kind)
    }
}

impl InterventionKind {
    fn requested_value(self, before: f32) -> f32 {
        match self {
            Self::SetValue(value) => value,
            Self::AddValue(delta) => before + delta,
        }
    }

    fn try_validate(self) -> Result<(), String> {
        let value = match self {
            Self::SetValue(value) | Self::AddValue(value) => value,
        };
        if !value.is_finite() {
            return Err("intervention values must be finite".into());
        }
        Ok(())
    }

    fn validate(self) {
        self.try_validate()
            .unwrap_or_else(|error| panic!("{error}"));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct InteroceptiveIntervention {
    pub channel: ViabilityChannel,
    pub kind: InterventionKind,
}

impl InteroceptiveIntervention {
    pub fn set(channel: ViabilityChannel, value: f32) -> Self {
        let kind = InterventionKind::SetValue(value);
        kind.validate();
        Self { channel, kind }
    }

    pub fn add(channel: ViabilityChannel, delta: f32) -> Self {
        let kind = InterventionKind::AddValue(delta);
        kind.validate();
        Self { channel, kind }
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
