//! Native target modules for the public CEB feature/target development fixture.

// Intentional development defect: each module should be included only on its
// matching target. The eventual specification must declare the target set.
mod unix;
mod windows;

#[cfg(unix)]
pub use unix::platform_id;

#[cfg(windows)]
pub use windows::platform_id;
