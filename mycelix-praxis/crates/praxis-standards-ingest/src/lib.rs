// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # EduNet Standards Ingest
//!
//! CLI tool and library for fetching K-12 academic standards from the
//! [Common Standards Project](https://commonstandardsproject.com) API and
//! converting them into EduNet curriculum JSON files compatible with the
//! `knowledge_zome` data model.
//!
//! ## Usage
//!
//! ```bash
//! # List available jurisdictions (states)
//! edunet-standards-ingest list-jurisdictions --type state
//!
//! # List standard sets for a jurisdiction
//! edunet-standards-ingest list-sets <jurisdiction-id>
//!
//! # Fetch a standard set and output curriculum JSON
//! edunet-standards-ingest fetch <standard-set-id> -o output.json
//! ```

pub mod api_types;
pub mod bridge;
pub mod career_profile;
pub mod client;
pub mod converter;
pub mod higher_ed_types;
pub mod merge;
pub mod pathfind;
pub mod resources;
pub mod sources;
pub mod stats;
pub mod taxonomy;
