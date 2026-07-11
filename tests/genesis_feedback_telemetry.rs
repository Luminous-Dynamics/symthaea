// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

#[cfg(all(test, feature = "cell-foundry"))]
mod tests {
    use symthaea::cognitive_loop::genesis_bridge::InnateTraits;
    use symthaea_cell_foundry::types::DevelopmentalTelemetry;

    #[test]
    fn test_genesis_feedback_telemetry_flow() {
        let innate = InnateTraits::default();
        let mut telemetry = DevelopmentalTelemetry::default();

        // 1. Simulate high-stress period (Allostatic load = 0.8)
        let signal1 = innate.emit_developmental_signal(0.8, 0.4, 24.0);
        telemetry.record(&signal1);

        // 2. Simulate high-focus period (Integration peak = 0.9)
        let signal2 = innate.emit_developmental_signal(0.1, 0.9, 12.0);
        telemetry.record(&signal2);

        // 3. Verify telemetry aggregation
        assert_eq!(telemetry.signal_count, 2);
        assert!(telemetry.cumulative_stress > 0.0);
        assert!((telemetry.max_phi_peak - 0.9).abs() < 1e-6);

        // Cumulative stress should be (0.8 * 24) + (0.1 * 12) = 19.2 + 1.2 = 20.4
        assert!((telemetry.cumulative_stress - 20.4).abs() < 1e-6);
    }
}
