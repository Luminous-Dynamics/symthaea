//! Isolated test harness for the authoritative FEP observation-event module.
//!
//! This runner imports the production source by path. It does not copy or
//! reimplement the contract and does not require unrelated root workspace
//! members to resolve.

#![deny(unsafe_code)]

#[path = "../../../src/observation_event.rs"]
pub mod observation_event;
