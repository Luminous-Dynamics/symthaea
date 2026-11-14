#![forbid(unsafe_code)]

//! Standalone benchmark reference-test package.
//!
//! This crate is deliberately outside the production workspace. It contains no
//! production service API; its integration tests independently evaluate the
//! synthetic decision-minimal-evidence corpus and verify the frozen intervention
//! and mutation manifests.
