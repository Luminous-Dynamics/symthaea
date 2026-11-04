//! Public API facade with feature- and target-conditional modules.

mod profile_api;

pub use profile_api::{Profile, get_profile};

pub fn build_info() -> &'static str {
    "ceb-repository-twin-v0"
}

#[cfg(feature = "audit")]
pub mod audit;

#[cfg(all(target_os = "windows", feature = "windows-native"))]
mod windows_native;
