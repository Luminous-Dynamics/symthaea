// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Accessors for the memetic immune system (plan Phase 2).
//!
//! Exposes the live memetic-defense telemetry so callers/tests can observe
//! screening without reaching into the loop internals. Feature-gated behind
//! `social-fabric` (the same gate as the immune field itself).

#![cfg(feature = "social-fabric")]

use crate::cognitive_loop::CognitiveLoopService;
use symthaea_memetics::{FilteredItem, MemeticTelemetry, Ruleset, WardConfig};

impl CognitiveLoopService {
    /// Current memetic immune telemetry: memes seen/rejected/accepted, rolling
    /// mean resonance and contagion index, and immune-memory size.
    pub fn memetic_telemetry(&self) -> MemeticTelemetry {
        self.memetic_immune.telemetry()
    }

    /// Vaccinate the memetic immune system against a known pathogen signature,
    /// so future variants that resonate with it are rejected (mutation-tolerant).
    pub fn vaccinate_meme(&mut self, pathogen: symthaea_core::hdc::BinaryHV) {
        self.memetic_immune.vaccinate(pathogen);
    }

    /// Bulk-vaccinate against every entry in `ruleset` (Warded Node design,
    /// Phase 5a: a guardian's pre-trusted starting set of known-bad
    /// patterns). Returns the number of signatures applied. This crate does
    /// not verify the ruleset's provenance — see `symthaea_memetics::Ruleset`
    /// docs for why (no signature checking, no file I/O; both stay the host
    /// application's job).
    pub fn vaccinate_ruleset(&mut self, ruleset: &Ruleset) -> usize {
        self.memetic_immune.vaccinate_ruleset(ruleset)
    }

    /// Set this node's warded-node protective settings (Phase 1 of
    /// `WARDED_NODE_DESIGN_2026-07-11.md`: a posture floor). The effective
    /// memetic posture used for screening is then `max(derived_posture,
    /// floor)` — it can only get stricter than what psi/context would
    /// otherwise derive, never more lenient. Pass `WardConfig::default()` to
    /// clear (unwarded/adult behavior). Local to this node; no network effect.
    pub fn set_ward_config(&mut self, ward: WardConfig) {
        self.memetic_immune.set_ward_config(ward);
    }

    /// Current warded-node protective settings.
    pub fn ward_config(&self) -> WardConfig {
        self.memetic_immune.ward_config()
    }

    /// The `limit` most recent memetic-firewall rejections (newest first),
    /// for guardian transparency (Phase 2 of `WARDED_NODE_DESIGN_2026-07-11.md`
    /// — deliberately NOT a black box). Each entry carries why it was
    /// rejected but not the content itself; correlate via `meme_id`.
    pub fn memetic_filtered_log(&self, limit: usize) -> Vec<FilteredItem> {
        self.memetic_immune.filtered_log(limit)
    }

    /// Count of entries currently retained in the filtered-items audit log
    /// (bounded ring buffer — see `symthaea_memetics::FilteredItem` docs).
    pub fn memetic_filtered_log_len(&self) -> usize {
        self.memetic_immune.filtered_log_len()
    }
}
