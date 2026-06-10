//! symthaea-logparse — Phase 1 spike for the MSP / IT-ops wedge.
//!
//! Goal: prove that Symthaea's HDC encoder produces useful semantic clusters
//! over real IT-ops logs (Windows Event Log + syslog). If clustering purity on
//! a labeled Evtx corpus exceeds 50%, the wedge thesis survives and Phase 2
//! (fleshing out `symthaea-support` + wiring `reasoning_engine` to a typed
//! SystemStateGraph) is justified. If it fails, the thesis dies cheaply here.
//!
//! See `memory/project_msp_wedge.md` for the full three-product ladder and
//! kill criteria.
//!
//! This crate intentionally has a thin surface:
//!   - `evtx_source` — Windows Event Log ingestion
//!   - `syslog_source` — RFC5424/RFC3164 syslog ingestion
//!   - `event` — normalized `LogEvent` all sources flatten into
//!   - `encoder` — HDC hypervector encoding (feature-gated on `hdc-encoder`)
//!   - `cluster` — HDBSCAN wrapper + purity metric

pub mod cluster;
pub mod encoder;
pub mod event;
pub mod evtx_source;
pub mod fixtures;
pub mod otrf_source;
pub mod probe;
pub mod syslog_source;

pub use event::{LogEvent, Severity, Source};

#[derive(Debug, thiserror::Error)]
pub enum LogParseError {
    #[error("evtx parse error: {0}")]
    Evtx(String),
    #[error("syslog parse error: {0}")]
    Syslog(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("encoder not enabled — build with --features hdc-encoder")]
    EncoderDisabled,
}

pub type Result<T> = std::result::Result<T, LogParseError>;
