//! Small, dependency-free reference algorithms for CROP-INTEL-001.
//!
//! The crate measures descriptive geometry and runs explicitly declared
//! decoders. It does not infer authorship, origin, intent, or semantics.

pub mod crop_flow;
pub mod decoder;
pub mod geometry;
pub mod grid;
pub mod statistics;

pub use crop_flow::{CropFlowDescriptor, CropFlowError, CropFlowSample, describe_crop_flow};
pub use decoder::{
    BitOrder, DecodeError, DecodeOutcome, DecoderSpec, PayloadEncoding, decode,
    encode_ascii7, run_tournament,
};
pub use geometry::{
    Bounds2, GeometryDescriptor, GeometryError, Point2, SymmetryScore, describe,
};
pub use grid::{
    BinaryGrid, GridDecodeOutcome, GridDecoderManifest, GridDecoderSpec, GridError, GridTransform,
    ManifestError, NullGridResult,
    TraversalOrder, margin_preserving_shuffle, run_grid_tournament,
};
pub use statistics::{wilson_interval, binary_entropy, upper_tail_max_p_value};
