// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-culinary — chemically-honest gastronomy
//!
//! Phase 0 of `CULINARY_PLAN_2026-07-09.md`: a **flavor network** over the real
//! volatile-compound data of Ahn et al. (2011), operationalizing the food-pairing
//! hypothesis and — critically — **reproducing the paper's headline result as a
//! falsifiable ground-truth test** ([`tests/ahn_2011.rs`](../tests/ahn_2011.rs)).
//!
//! This is the honest core of the culinary pitch. It deliberately does **not**
//! include the parts that were metaphor rather than measurement (no Φ-of-a-dish,
//! no CfC-as-physics claim, no HDC as load-bearing — the science here is plain
//! set overlap over compound vectors). Later phases (spec validator, process
//! dynamics, active-inference palate) build on this.
//!
//! ## What it does
//!
//! An [`Ingredient`] is a sparse flavor vector — its set of volatile compounds.
//! [`Ingredient::shared_compounds`] / [`Ingredient::jaccard`] / [`Ingredient::cosine`]
//! score how much two ingredients overlap. [`delta_nc`] answers, for a whole
//! cuisine, whether its recipes pair overlapping ingredients more or less than a
//! frequency-conserving null — the Ahn 2011 result.
//!
//! ```
//! use symthaea_culinary::flavor_network::delta_nc_default;
//! // North American cooking pairs ingredients that SHARE compounds (ΔNc > 0);
//! // East Asian cooking AVOIDS shared compounds (ΔNc < 0).
//! assert!(delta_nc_default("NorthAmerican").unwrap().delta > 0.0);
//! assert!(delta_nc_default("EastAsian").unwrap().delta < 0.0);
//! ```
//!
//! No external dependencies; all data is embedded (see `data/PROVENANCE.md`).

pub mod balance;
pub mod data;
pub mod dynamics;
pub mod flavor_network;
pub mod ingredient;
pub mod kitchen;
pub mod nutrition;
pub mod palate;
pub mod presets;
pub mod reaction;
pub mod rng;
pub mod spec;
pub mod thermal;
pub mod thresholds;
pub mod validate;

pub use balance::{BalanceScore, TasteProfile, balance_score};
pub use dynamics::{NewtonCooling, emulsion_relative_viscosity, krieger_dougherty};
pub use flavor_network::{DeltaNc, delta_nc, delta_nc_default};
pub use ingredient::Ingredient;
pub use nutrition::NutrientProfile;
pub use reaction::{arrhenius_rate, maillard_feasible, q10, reaction_extent};
pub use spec::CulinarySpec;
pub use thermal::ThermalTrajectory;
pub use validate::{CulinaryViolation, validate};
