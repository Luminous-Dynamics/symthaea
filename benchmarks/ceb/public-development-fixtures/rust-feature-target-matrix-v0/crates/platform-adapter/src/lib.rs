//! Target adapter for the public CEB feature/target development fixture.

// Intentional development defect: both source modules are included regardless
// of target. Their imports are platform-specific and must be gated by cfg.
pub mod native;
