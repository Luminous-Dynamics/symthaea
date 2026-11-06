//! Standalone compilation/test harness for the production Muse wire protocol
//! and genealogy verifier, kept outside the root Cargo workspace so it remains
//! runnable while unrelated workspace packages are quarantined.

extern crate self as symthaea_muse_protocol;

#[path = "../../../crates/domains/symthaea-muse-protocol/src/lib.rs"]
mod production_protocol;

pub use production_protocol::*;

#[path = "../../../crates/domains/symthaea-muse/src/genealogy.rs"]
pub mod genealogy;
