use serde::{Deserialize, Serialize};

pub const CHANNEL_COUNT: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ViabilityChannel {
    ComputeReserve = 0,
    MemoryHeadroom = 1,
    ModelStability = 2,
    EpistemicResolution = 3,
    Integrity = 4,
    ActionEfficacy = 5,
    NoveltyBalance = 6,
    /// Reliability of action/environment coupling, without assuming that the
    /// relevant counterpart is another agent.
    InteractionReliability = 7,
}

impl ViabilityChannel {
    pub const ALL: [Self; CHANNEL_COUNT] = [
        Self::ComputeReserve,
        Self::MemoryHeadroom,
        Self::ModelStability,
        Self::EpistemicResolution,
        Self::Integrity,
        Self::ActionEfficacy,
        Self::NoveltyBalance,
        Self::InteractionReliability,
    ];

    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Stable machine-readable identifier for evidence exports.
    pub const fn stable_id(self) -> &'static str {
        match self {
            Self::ComputeReserve => "compute_reserve",
            Self::MemoryHeadroom => "memory_headroom",
            Self::ModelStability => "model_stability",
            Self::EpistemicResolution => "epistemic_resolution",
            Self::Integrity => "integrity",
            Self::ActionEfficacy => "action_efficacy",
            Self::NoveltyBalance => "novelty_balance",
            Self::InteractionReliability => "interaction_reliability",
        }
    }
}

/// One native viability channel.
///
/// Fields are private so construction-time invariants cannot later be bypassed
/// by mutating bounds, precision, importance, or the measured state directly.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ViabilityVariable {
    value: f32,
    preferred_low: f32,
    preferred_high: f32,
    viable_low: f32,
    viable_high: f32,
    precision: f32,
    velocity: f32,
    importance: f32,
}

impl ViabilityVariable {
    pub fn new(
        value: f32,
        preferred_low: f32,
        preferred_high: f32,
        viable_low: f32,
        viable_high: f32,
        precision: f32,
        importance: f32,
    ) -> Self {
        validate_observation(value, 0.0);
        assert!(preferred_low.is_finite());
        assert!(preferred_high.is_finite());
        assert!(viable_low.is_finite());
        assert!(viable_high.is_finite());
        assert!(precision.is_finite() && precision >= 0.0);
        assert!(importance.is_finite() && importance >= 0.0);
        assert!(viable_low <= preferred_low);
        assert!(preferred_low <= preferred_high);
        assert!(preferred_high <= viable_high);

        Self {
            value,
            preferred_low,
            preferred_high,
            viable_low,
            viable_high,
            precision,
            velocity: 0.0,
            importance,
        }
    }

    #[inline]
    pub const fn value(&self) -> f32 {
        self.value
    }

    #[inline]
    pub const fn velocity(&self) -> f32 {
        self.velocity
    }

    #[inline]
    pub const fn preferred_low(&self) -> f32 {
        self.preferred_low
    }

    #[inline]
    pub const fn preferred_high(&self) -> f32 {
        self.preferred_high
    }

    #[inline]
    pub const fn viable_low(&self) -> f32 {
        self.viable_low
    }

    #[inline]
    pub const fn viable_high(&self) -> f32 {
        self.viable_high
    }

    #[inline]
    pub const fn precision(&self) -> f32 {
        self.precision
    }

    #[inline]
    pub const fn importance(&self) -> f32 {
        self.importance
    }

    #[inline]
    pub fn preferred_midpoint(&self) -> f32 {
        0.5 * (self.preferred_low + self.preferred_high)
    }

    #[inline]
    pub fn is_preferred(&self) -> bool {
        self.value >= self.preferred_low && self.value <= self.preferred_high
    }

    #[inline]
    pub fn is_viable(&self) -> bool {
        self.value >= self.viable_low && self.value <= self.viable_high
    }

    /// Normalized deviation from the preferred interval.
    ///
    /// `0.0` means the state is inside its preferred interval, `1.0` means it
    /// has reached a viability boundary, and values above `1.0` mean that the
    /// boundary has been exceeded.
    pub fn normalized_deviation(&self) -> f32 {
        if self.value < self.preferred_low {
            let span = (self.preferred_low - self.viable_low).max(f32::EPSILON);
            (self.preferred_low - self.value) / span
        } else if self.value > self.preferred_high {
            let span = (self.viable_high - self.preferred_high).max(f32::EPSILON);
            (self.value - self.preferred_high) / span
        } else {
            0.0
        }
    }

    pub fn predicted(&self, dt: f32) -> Self {
        assert!(dt.is_finite() && dt >= 0.0);
        let mut next = *self;
        next.set_observation(self.value + self.velocity * dt, self.velocity);
        next
    }

    pub(crate) fn set_observation(&mut self, value: f32, velocity: f32) {
        validate_observation(value, velocity);
        self.value = value;
        self.velocity = velocity;
    }
}

fn validate_observation(value: f32, velocity: f32) {
    assert!(value.is_finite(), "viability values must be finite");
    assert!(velocity.is_finite(), "viability velocities must be finite");
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeInteroceptiveState {
    channels: [ViabilityVariable; CHANNEL_COUNT],
}

impl NativeInteroceptiveState {
    pub fn new(channels: [ViabilityVariable; CHANNEL_COUNT]) -> Self {
        Self { channels }
    }

    #[inline]
    pub fn get(&self, channel: ViabilityChannel) -> &ViabilityVariable {
        &self.channels[channel.index()]
    }

    #[inline]
    pub(crate) fn get_mut(&mut self, channel: ViabilityChannel) -> &mut ViabilityVariable {
        &mut self.channels[channel.index()]
    }

    #[inline]
    pub fn channels(&self) -> &[ViabilityVariable; CHANNEL_COUNT] {
        &self.channels
    }

    /// Build an experimental initial condition with an explicit observed value
    /// and measured velocity. Live state changes should use the model transition
    /// law or the intervention API instead.
    pub fn with_observation(
        mut self,
        channel: ViabilityChannel,
        value: f32,
        velocity: f32,
    ) -> Self {
        self.channels[channel.index()].set_observation(value, velocity);
        self
    }

    pub fn with_value(self, channel: ViabilityChannel, value: f32) -> Self {
        self.with_observation(channel, value, 0.0)
    }

    pub fn predicted(&self, dt: f32) -> Self {
        Self {
            channels: self.channels.map(|variable| variable.predicted(dt)),
        }
    }
}

impl Default for NativeInteroceptiveState {
    fn default() -> Self {
        let standard = || ViabilityVariable::new(0.75, 0.65, 0.85, 0.25, 1.0, 1.0, 1.0);
        let centered = || ViabilityVariable::new(0.50, 0.40, 0.60, 0.10, 0.90, 1.0, 1.0);

        Self::new([
            standard(),
            standard(),
            standard(),
            standard(),
            ViabilityVariable::new(0.90, 0.80, 1.00, 0.40, 1.00, 1.0, 1.5),
            standard(),
            centered(),
            standard(),
        ])
    }
}
