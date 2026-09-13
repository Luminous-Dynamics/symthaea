// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reference-integrity evaluation for measured-boot assurance evidence.

#![deny(unsafe_code)]

mod evaluation;
mod measurements;
mod reference;
mod util;

pub use evaluation::*;
pub use measurements::*;
pub use reference::*;

pub const REFERENCE_INTEGRITY_SCOPE: &str =
    "signed_reference_assertions_over_verified_measured_boot_evidence_v1";
