// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Metric spatial-world foundations for Symthaea.
//!
//! This crate deliberately separates raw evidence, bounded admission,
//! runtime observations/beliefs, and persisted records. Coordinate frames,
//! evidence references, and clock domains are source- and generation-qualified
//! so correctly assigned identities do not collapse merely because source-local
//! identifiers are reused. Evidence references additionally bind immutable claim
//! content through a strict 256-bit digest. Metric geometry remains independent
//! from semantic HDC representations and rendering backends.
//!
//! Namespace values in this crate are ordinary identity labels. They do **not**
//! prove source ownership, authenticity, or globally unique allocation. A future
//! owning sensor/evidence adapter must assign and qualify namespace lineage before
//! data is promoted into the spatial runtime boundary.
//!
//! # Metric frame and pose convention
//!
//! Spatial-world metric frames are right-handed Cartesian frames expressed in
//! metres. `MetricPoint3` denotes a position; `MetricVector3` denotes a free
//! displacement/translation. Those semantics are intentionally distinct even
//! though both serialize as three finite metric components.
//!
//! `Pose3` is interpreted as `T_reference_from_local`:
//!
//! - translation is the vector from the reference-frame origin to the local-frame
//!   origin, expressed in reference-frame axes;
//! - quaternion order is `[w, x, y, z]`;
//! - quaternion rotation maps vectors from local-frame axes into reference-frame
//!   axes;
//! - `PoseUncertainty` rotational components are tangent perturbations consistent
//!   with that same local-to-reference convention.
//!
//! Adapters for left-handed, non-metric, camera-specific, or otherwise different
//! coordinate conventions must perform an explicit conversion before constructing
//! spatial-world state. This crate intentionally does not guess conventions from
//! raw numeric arrays.
//!
//! ```text
//! raw evidence reference
//!       !=
//! admitted observation evidence -> SpatialObservation<T>
//!                                    |
//!                                    v
//!                             SpatialObservationRecord<T>
//!                             (non-authorizing data)
//!
//! raw evidence reference
//!       !=
//! admitted belief support       -> SpatialBelief<T>
//!                                    |
//!                                    v
//!                             SpatialBeliefRecord<T>
//!                             (non-authorizing data)
//! ```
//!
//! V1 intentionally implements no public issuer for either admitted evidence
//! token. Those adapters must be bound to the canonical evidence authority rather
//! than recreated locally.

#![deny(unsafe_code)]
#![warn(missing_docs)]

/// Current closed-world serialized schema version for spatial observation/belief records.
pub const SPATIAL_WORLD_SCHEMA_VERSION: u16 = 1;

pub mod evidence;
pub mod geometry;
pub mod time;

pub use evidence::{
    AdmittedBeliefSupportEvidence, AdmittedObservationEvidence, EvidenceDigest,
    EvidenceDigestAlgorithm, EvidenceId, EvidenceNamespaceId, EvidenceRef, SpatialBelief,
    SpatialBeliefRecord, SpatialEvidenceKind, SpatialObservation, SpatialObservationRecord,
};
pub use geometry::{
    MetricPoint3, MetricVector3, Pose3, PoseEstimate, PoseUncertainty, ReferenceFrameId,
    ReferenceFrameNamespaceId, UnitQuaternion,
};
pub use time::{ClockDomainId, ClockInstant, ClockNamespaceId};

/// Validation failures at the spatial-world type boundary.
#[derive(Debug, thiserror::Error)]
pub enum SpatialValidationError {
    /// A stable identifier used the reserved zero value.
    #[error("{kind} identifier must be non-zero")]
    ZeroId {
        /// Human-readable identifier kind.
        kind: &'static str,
    },
    /// A serialized record used an unsupported closed-world schema version.
    #[error("unsupported spatial-world schema version {found}")]
    UnsupportedSchemaVersion {
        /// Unsupported version found on the wire.
        found: u16,
    },
    /// An evidence claim digest was not strict SHA-256/BLAKE3 256-bit hex.
    #[error("malformed spatial evidence digest")]
    MalformedDigest,
    /// A metric or uncertainty component was NaN or infinite.
    #[error("{field} must be finite, got {value}")]
    NonFinite {
        /// Name of the invalid field/component.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// A standard deviation was negative.
    #[error("{field} standard deviation must be >= 0, got {value}")]
    NegativeUncertainty {
        /// Name of the invalid uncertainty component.
        field: &'static str,
        /// Invalid negative value.
        value: f64,
    },
    /// A quaternion had effectively zero magnitude.
    #[error("quaternion norm is too small to normalize")]
    DegenerateQuaternion,
    /// A runtime belief was constructed without any admitted support.
    #[error("spatial belief requires at least one admitted support item")]
    EmptyBeliefSupport,
    /// One source- and generation-qualified evidence identity appeared more than once.
    #[error(
        "duplicate evidence reference namespace={namespace} id={id} generation={generation}"
    )]
    DuplicateEvidenceRef {
        /// Evidence namespace.
        namespace: u64,
        /// Source-local evidence ID.
        id: u64,
        /// Evidence-stream generation.
        generation: u64,
    },
}
