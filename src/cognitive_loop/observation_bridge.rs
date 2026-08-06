// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Implements `symthaea_core::observation::CognitiveObservation` for `CognitiveLoopService`.
//!
//! Phase 2 of `SYMTHAEA_COGNITIVE_CORE_RECONCILIATION_PLAN_2026-07-28.md`. This is the
//! narrow, Tier-1-hosted alternative to driving `CognitiveLoopService`'s full public
//! method surface directly (the pattern `symthaea-psych-bench`/`symthaea-pulse` use
//! today). Migrating those existing consumers onto this trait is an explicit stretch
//! goal for this phase, not required -- see the plan doc.
//!
//! Every value returned here is pulled from an existing, already-real accessor or
//! field -- nothing here is a stub or placeholder.

use super::CognitiveLoopService;
use symthaea_core::observation::CognitiveObservation;

impl CognitiveObservation for CognitiveLoopService {
    fn state_dimensions(&self) -> usize {
        // Delegates to the existing real accessor (accessors/system.rs:90),
        // itself `self.config.cfc_config.input_dim` -- not reimplemented here.
        self.state_dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive_loop::CognitiveLoopConfig;

    fn test_service() -> CognitiveLoopService {
        CognitiveLoopService::new(CognitiveLoopConfig::default())
            .expect("default config must construct a service")
    }

    #[test]
    fn observation_trait_reports_real_service_state() {
        let service = test_service();

        // state_dim is a construction-time fact, so it must already be real
        // before any cycle has run.
        assert!(
            service.state_dimensions() > 0,
            "state_dimensions() must be a real positive dimension"
        );
        assert_eq!(
            service.state_dimensions(),
            service.state_dim(),
            "trait method must delegate to the existing real accessor, not reimplement it"
        );
    }
}
