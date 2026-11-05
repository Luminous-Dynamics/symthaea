//! Standalone compilation/test harness for the production genealogy module.
//!
//! The production source is included directly. Only the small protocol DTO
//! dependency is represented by a local type-compatible stub because the
//! repository root workspace currently has an unrelated missing member.

#[path = "../../../crates/domains/symthaea-muse/src/genealogy.rs"]
pub mod genealogy;
