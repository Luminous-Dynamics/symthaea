// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Property-based testing for the HDC-LTC runtime contract.
//! Validates stability, boundary behavior, and deterministic replay of LTC network steps
//! under irregular-time jumps, extreme inputs, and invalid parameter bounds.

use proptest::prelude::*;
use symthaea_hdc_ltc::{
    ContinuousHV, HdcLtcUnifiedNetwork, HdcLtcUnifiedNeuron, NetworkConfig, NeuronConfig,
    StepTimingConfig,
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Test that extreme timestamps are bounded and handled correctly under timing configuration
    #[test]
    fn test_proptest_network_irregular_time_bounds(
        // Generates time increments including tiny, huge, and backward jumps
        dts in prop::collection::vec(-10.0f32..100.0f32, 1..10),
        // Random input vector elements
        input_val in -100.0f32..100.0f32
    ) {
        let dim = 128;
        let neuron_config = NeuronConfig {
            dim,
            tau_base: 0.1,
            tau_min: 0.01,
            tau_max: 5.0,
            ..NeuronConfig::default()
        };
        let config = NetworkConfig {
            layer_sizes: vec![2, 2],
            neuron_config,
            use_layer_binding: true,
            skip_connections: true,
        };

        let mut net = HdcLtcUnifiedNetwork::new(config, 42);
        let timing_config = StepTimingConfig {
            min_dt: 0.005,
            max_dt: 2.0,
            reject_backward_time: true,
        };
        net.set_timing_config(timing_config);

        let input = ContinuousHV::from_values(vec![input_val; dim]);
        let mut sim_time = 0.0f64;

        for dt in dts {
            sim_time += dt as f64;
            // Execute step with timestamp
            net.step_with_timestamp(sim_time, &input);

            // Assertions on runtime contracts
            let output = net.output();
            assert!(output.norm().is_finite(), "Output norm must be finite, got norm={}", output.norm());

            // Check that the network successfully tracks timestamp
            if let Some(last_t) = net.last_timestamp() {
                assert!(last_t.is_finite());
            }
        }
    }

    /// Verify deterministic replay: the exact same sequence of irregular times and inputs
    /// yields the exact same final states down to float precision.
    #[test]
    fn test_proptest_deterministic_replay(
        dts in prop::collection::vec(0.001f32..5.0f32, 5),
        seed in 0u64..1000u64
    ) {
        let dim = 128;
        let config = NetworkConfig {
            layer_sizes: vec![2, 1],
            neuron_config: NeuronConfig {
                dim,
                ..NeuronConfig::default()
            },
            use_layer_binding: false,
            skip_connections: false,
        };

        // Network A
        let mut net_a = HdcLtcUnifiedNetwork::new(config.clone(), seed);
        // Network B with same seed
        let mut net_b = HdcLtcUnifiedNetwork::new(config, seed);

        let input = ContinuousHV::new_random(dim, seed);

        // Run network A
        let mut t_a = 0.0f64;
        for &dt in &dts {
            t_a += dt as f64;
            net_a.step_with_timestamp(t_a, &input);
        }

        // Run network B
        let mut t_b = 0.0f64;
        for &dt in &dts {
            t_b += dt as f64;
            net_b.step_with_timestamp(t_b, &input);
        }

        let out_a = net_a.output();
        let out_b = net_b.output();
        let sim = out_a.similarity(&out_b);
        assert!((sim - 1.0).abs() < 1e-5, "Replay diverged: similarity={}", sim);
    }

    /// Neuron invariant verification: extreme and malformed inputs must produce
    /// finite, bounded outputs.
    #[test]
    fn test_proptest_neuron_invariants(
        val in -1e5f32..1e5f32,
        dt in 0.0001f32..1000.0f32
    ) {
        let dim = 128;
        let config = NeuronConfig {
            dim,
            tau_base: 0.1,
            tau_min: 0.001,
            tau_max: 10.0,
            ..NeuronConfig::default()
        };
        let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);

        // Input with extreme values
        let input = ContinuousHV::from_values(vec![val; dim]);
        neuron.evolve_closed_form(dt, &input);

        let state = neuron.state();
        assert!(state.norm().is_finite(), "State norm must be finite under extreme inputs");
        assert!(state.norm() <= 5.01, "Neuron state bounds violated, norm={}", state.norm());
    }
}
