//! Drive subsystems: emotion contagion, curiosity, and self-reflection.

mod emotion_contagion;
mod curiosity_drive;
mod self_reflection;

pub(crate) use emotion_contagion::EmotionContagion;
pub(crate) use curiosity_drive::{CuriosityDrive, ExplorationUpdate};
pub(crate) use self_reflection::{
    SelfReflection, SelfAssessment, Recommendation, RecommendationTarget,
    AdjustmentDirection, ReflectionThresholds, ReflectionSummary,
};
