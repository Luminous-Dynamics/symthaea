//! Isolated test harness for the production connectome identity contract.
//!
//! Includes the checked-in library source directly. The unit tests therefore
//! exercise the exact implementation without resolving the root workspace.

#[path = "../../../crates/domains/symthaea-connectome-identity/src/lib.rs"]
pub mod connectome_identity;
