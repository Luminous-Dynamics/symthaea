// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

#[cfg(test)]
mod tests {
    use crate::DreamEngine;
    use crate::crystallized_dream::CrystallizedAction;
    use symthaea_core::hdc::grid_encoder::GridTransform;

    #[test]
    fn test_crystallized_dreaming() {
        let mut engine = DreamEngine::<CrystallizedAction>::with_defaults();

        // 1. Record a high-surprise event where a rotation was helpful
        let state = vec![0.5; 64];
        let action = CrystallizedAction::single(GridTransform::Rotate90);
        let outcome = vec![0.8; 64];
        let surprise = 0.9;

        engine.record(&state, action, &outcome, surprise);

        // 2. Run a dream cycle
        // This will perturb the action (e.g. try Rotate90 + something else)
        // and simulate outcomes.
        let result = engine.dream().unwrap();

        println!("Dream Result: {:?}", result);

        // Should gain at least one simulation
        assert!(result.simulations_run > 0);
    }
}
