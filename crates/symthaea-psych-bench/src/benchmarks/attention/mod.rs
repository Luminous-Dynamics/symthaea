//! Attention domain benchmarks.
//!
//! - **AttentionalBlink** — Temporal attention limits in RSVP streams
//! - **VisualSearch** — Parallel vs serial attentional processing
//! - **MismatchNegativity** — Pre-attentive oddball detection

pub mod attentional_blink;
pub mod mismatch_negativity;
pub mod visual_search;

pub use attentional_blink::AttentionalBlinkBenchmark;
pub use mismatch_negativity::MismatchNegativityBenchmark;
pub use visual_search::VisualSearchBenchmark;
