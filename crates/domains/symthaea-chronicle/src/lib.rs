// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-chronicle
//!
//! The formal, testable core of history-as-narrative: historical **events**,
//! **causal chains**, and **anachronism detection**. Fourth of the "hard"
//! knowledge domains (`symthaea/HARD_DOMAINS_PLAN_2026-07-07.md`).
//!
//! **Non-duplication:** rich temporal (Allen-interval) reasoning already exists
//! in the main crate (`src/consciousness/temporal/`); this crate uses only
//! minimal date comparison and should defer to that module when integrated. Its
//! genuinely new value is the *events + causation + anachronism* layer.
//!
//! **Scope note:** it computes ordering, causal reachability, and anachronism —
//! it does NOT judge historical significance or synthesize narrative (the
//! interpretive parts of history stay out of scope).
//!
//! Pure `std`, zero dependencies, no `symthaea-core` link.
//!
//! ## Example
//!
//! ```
//! use symthaea_chronicle::Chronicle;
//! let mut c = Chronicle::new();
//! c.event("press", 1440, None).event("reformation", 1517, None)
//!  .causation("press", "reformation").entity("napoleon", 1769, 1821);
//! assert!(c.causally_leads_to("press", "reformation"));
//! assert_eq!(c.is_anachronistic("napoleon", 2007), Some(true)); // smartphone!
//! ```

pub mod allen;
pub mod chronicle;

pub use allen::{AllenRelation, relation as allen_relation};
pub use chronicle::{Chronicle, Event, Lifespan};
