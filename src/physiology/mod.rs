//! Physiology Module - The Body of Sophia
//!
//! Week 4: The Physiology of Feeling
//!
//! The physiology layer sits beneath the neural layer (Actor Model) and provides
//! the "body" that regulates cognitive function through slow-moving chemical states.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────┐
//! │   Neural Layer (Actor Model)    │  Fast (milliseconds)
//! │  - Prefrontal Cortex            │  - Attention
//! │  - Motor Cortex                 │  - Goals
//! │  - Cerebellum                   │  - Skills
//! └─────────────┬───────────────────┘
//!               │ reads from
//!               ▼
//! ┌─────────────────────────────────┐
//! │   Chemical Layer (Endocrine)    │  Slow (minutes)
//! │  - Cortisol (Stress)            │  - Moods
//! │  - Dopamine (Reward)            │  - Arousal
//! │  - Acetylcholine (Focus)        │  - Valence
//! └─────────────────────────────────┘
//! ```
//!
//! ## The Four Systems
//!
//! 1. **Endocrine Core** (Days 1-3) - Moods
//!    - Cortisol, Dopamine, Acetylcholine
//!    - Slow-moving chemical regulation
//!
//! 2. **The Daemon** (Days 4-5) - Creativity
//!    - Spontaneous idea generation
//!    - Background processing
//!
//! 3. **The Hearth** (Days 6-7) - Metabolism
//!    - Finite energy budget
//!    - Fatigue and rest cycles
//!
//! 4. **The Chronos Lobe** (Days 3-4) - Time Perception
//!    - Subjective vs objective time
//!    - Emotional time dilation
//!    - Circadian rhythms
//!
//! 5. **Proprioception** (Days 5-7) - Hardware Awareness
//!    - Battery level affects energy capacity
//!    - CPU temperature creates stress
//!    - Disk space creates bloating
//!    - RAM usage affects cognition
//!
//! 6. **Coherence Field** (Week 6+) - Revolutionary Energy Model
//!    - Consciousness as integration (not commodity)
//!    - Connected work BUILDS coherence
//!    - Gratitude synchronizes systems
//!    - Replaces ATP scarcity with integration dynamics
//!
//! 7. **Social Coherence** (Week 11+) - Collective Intelligence
//!    - Multiple instances synchronize coherence
//!    - Coherence lending and borrowing
//!    - Collective learning and wisdom
//!
//! 8. **Larynx** (Week 12 Phase 2a) - Voice Output
//!    - Kokoro TTS with prosody modulation
//!    - Voice changes based on emotional state
//!    - Natural, expressive speech synthesis

pub mod endocrine;
pub mod emotional_reasoning;
pub mod hearth;
pub mod chronos;
pub mod proprioception;
pub mod coherence;
pub mod social_coherence;
#[cfg(feature = "audio")]
pub mod larynx;

pub use endocrine::{
    EndocrineConfig,
    EndocrineStats,
    EndocrineSystem,
    HormoneEvent,
    HormoneState,
    HormoneTrend,
    Trend,
};
pub use emotional_reasoning::{
    EmotionalReasoner,
    EmotionalState,
    Emotion,
};
pub use hearth::{
    ActionCost,
    EnergyState,
    HearthActor,
    HearthConfig,
    HearthStats,
};
pub use chronos::{
    ChronosActor,
    ChronosConfig,
    ChronosStats,
    TimeMode,
    TimeQuality,
    CircadianPhase,
};
pub use proprioception::{
    ProprioceptionActor,
    ProprioceptionConfig,
    ProprioceptionStats,
    BodySensation,
};
pub use coherence::{
    CoherenceField,
    CoherenceConfig,
    CoherenceStats,
    CoherenceState,
    CoherenceError,
    TaskComplexity,
    ScatterCause,      // Week 9: Scatter classification
    ScatterAnalysis,   // Week 9: Scatter diagnosis
};
pub use social_coherence::{
    CoherenceBeacon,         // Week 11: Broadcast coherence state
    SocialCoherenceField,    // Week 11: Synchronization & collective coherence
    CoherenceLoan,           // Week 11 Phase 2: Lending protocol
    CoherenceLendingProtocol, // Week 11 Phase 2: Coherence lending system
    ThresholdObservation,    // Week 11 Phase 3: Threshold tracking
    SharedKnowledge,         // Week 11 Phase 3: Collective knowledge pool
    CollectiveLearning,      // Week 11 Phase 3: Shared learning system
};
#[cfg(feature = "audio")]
pub use larynx::{
    LarynxActor,             // Week 12 Phase 2a: Voice synthesis
    LarynxConfig,            // Week 12 Phase 2a: Voice configuration
    LarynxStats,             // Week 12 Phase 2a: Voice statistics
    ProsodyParams,           // Week 12 Phase 2a: Prosody parameters
};
