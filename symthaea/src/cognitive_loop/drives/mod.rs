//! Drive subsystems: emotion contagion, curiosity, and self-reflection.

mod curiosity_drive;
mod emotion_contagion;
mod self_reflection;

pub(crate) use curiosity_drive::{CuriosityDrive, ExplorationUpdate};
pub(crate) use emotion_contagion::EmotionContagion;
pub(crate) use self_reflection::{
    AdjustmentDirection, Recommendation, RecommendationTarget, ReflectionSummary,
    ReflectionThresholds, SelfAssessment, SelfReflection,
};
