// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Orbital Mechanics Library for Mycelix-Space
//!
//! This library provides:
//! - TLE parsing and validation
//! - SGP4/SDP4 orbital propagation
//! - Covariance matrix propagation (uncertainty tracking)
//! - Conjunction analysis (collision probability)
//! - Coordinate transformations (ECI, ECEF, geodetic)
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Orbital State Model                          │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                 │
//! │  TLE Input ──────► SGP4 Elements ──────► State Vector           │
//! │       │                   │                    │                │
//! │       │                   │                    ▼                │
//! │       │                   │           ┌──────────────┐          │
//! │       │                   │           │ Position (3) │          │
//! │       │                   │           │ Velocity (3) │          │
//! │       │                   │           │ Covariance   │          │
//! │       │                   │           │   (6x6)      │          │
//! │       │                   │           └──────────────┘          │
//! │       │                   │                    │                │
//! │       │                   ▼                    ▼                │
//! │       │           ┌─────────────┐    ┌─────────────────┐        │
//! │       │           │ Propagator  │───►│ Future State    │        │
//! │       │           │ (SGP4/SDP4) │    │ + Uncertainty   │        │
//! │       │           └─────────────┘    └─────────────────┘        │
//! │       │                                       │                 │
//! │       ▼                                       ▼                 │
//! │  ┌──────────┐                        ┌───────────────┐          │
//! │  │ Validate │                        │ Conjunction   │          │
//! │  │ Checksum │                        │ Analysis      │          │
//! │  └──────────┘                        └───────────────┘          │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

pub mod cdm;
pub mod cdm_parser;
pub mod conjunction;
pub mod conjunction_network;
pub mod coordinates;
pub mod covariance;
pub mod fusion;
pub mod keplerian;
pub mod lambert;
pub mod orbit_determination;
pub mod propagator;
pub mod state;
pub mod tle;

pub use cdm::{
    CdmBuilder, CdmCovariance, CdmObjectMetadata, CdmRefFrame, CdmStateVector,
    ConjunctionDataMessage, Maneuverable,
};
pub use cdm_parser::{parse_cdm_kvn, CdmParseError};
pub use conjunction::{CollisionProbability, ConjunctionAssessment};
pub use covariance::CovarianceMatrix;
pub use fusion::{FusedEstimate, FusionPipeline, SensorMeasurement};
pub use keplerian::{
    collision_avoidance_maneuver, hohmann_transfer, ImpulsiveManeuver, KeplerianElements,
    KeplerianError,
};
pub use lambert::{solve_lambert, LambertSolution, TransferType};
pub use orbit_determination::{gauss_iod, ObservationRecord, ObservationType, OrbitDetermination};
pub use propagator::{PropagationError, Propagator};
pub use state::{OrbitalState, StateVector};
pub use tle::{TleParseError, TwoLineElement};

#[cfg(test)]
mod validation;
