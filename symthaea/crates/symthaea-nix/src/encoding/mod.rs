//! HDC Encoding for NixOS Structures
//!
//! Transforms NixOS data (option paths, packages, configs, system state,
//! user input) into hypervectors in a shared semantic space.
//!
//! All encoders share a `NixCodebook` to ensure vectors are comparable.

pub mod codebook;
pub mod config_encoder;
pub mod flake_graph_encoder;
pub mod option_encoder;
pub mod package_encoder;
pub mod semantic_search;
pub mod system_state_encoder;
pub mod user_input_encoder;

pub use codebook::NixCodebook;
pub use config_encoder::ConfigEncoder;
pub use flake_graph_encoder::{
    parse_flake_lock, EncodedFlakeGraph, FlakeDependencyEdge, FlakeGraphEncoder, FlakeInput,
};
pub use option_encoder::OptionEncoder;
pub use package_encoder::{PackageEncoder, PackageMetadata};
pub use semantic_search::{search_options, search_options_by_vector, OptionMatch};
pub use system_state_encoder::{ServiceState, SystemStateEncoder, SystemStateSnapshot};
pub use user_input_encoder::UserInputEncoder;
