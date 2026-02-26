//! Attention domain benchmarks.
//!
//! - **AttentionalBlink** — Temporal attention limits in RSVP streams
//! - **VisualSearch** — Parallel vs serial attentional processing

pub mod attentional_blink;
pub mod visual_search;

pub use attentional_blink::AttentionalBlinkBenchmark;
pub use visual_search::VisualSearchBenchmark;
