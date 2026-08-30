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
    SocialSecurity = 7,
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
        Self::SocialSecurity,
    ];

    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ViabilityVariable {
    pub value: f32,
    pub preferred_low: f32,
    pub preferred_high: f32,
    pub viable_low: f32,
    pub viable_high: f32,
    pub precision: f32,
    pub velocity: f32,
    pub importance: f32,
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
        assert!(value.is_finite());
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
        next.value = self.value + self.velocity * dt;
        next
    }
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
    pub fn get_mut(&mut self, channel: ViabilityChannel) -> &mut ViabilityVariable {
        &mut self.channels[channel.index()]
    }

    #[inline]
    pub fn channels(&self) -> &[ViabilityVariable; CHANNEL_COUNT] {
        &self.channels
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
