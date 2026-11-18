//! Small, dependency-free reference algorithms for CROP-INTEL-001.
//!
//! The crate measures descriptive geometry and runs explicitly declared
//! decoders. It does not infer authorship, origin, intent, or semantics.

pub mod circle_layout;
pub mod crop_flow;
pub mod decoder;
pub mod geometry;
pub mod grid;
pub mod statistics;

pub use circle_layout::{
    audit_circle_measurement_sensitivity, class_circle_layout, count_centred_motifs,
    describe_circle_layout, parse_circle_csv, validate_circles, CircleClassing, CircleContact,
    CircleLayout, CircleLayoutError, CircleRecord, CircleSensitivityConfig,
    CircleSensitivitySummary,
};
pub use crop_flow::{CropFlowDescriptor, CropFlowError, CropFlowSample, describe_crop_flow, describe_crop_flow_about_center};
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
    standard_d4_ascii7_manifest,
};
pub use statistics::{wilson_interval, binary_entropy, upper_tail_max_p_value};
