//! Affect domain benchmarks.
//!
//! - **ValenceClassification** — Classify stimuli into positive/negative/neutral valence
//! - **MoodCongruentRecall** — Test mood-congruent memory bias

pub mod mood_congruent_recall;
pub mod valence_classification;

pub use mood_congruent_recall::MoodCongruentRecallBenchmark;
pub use valence_classification::ValenceClassificationBenchmark;
