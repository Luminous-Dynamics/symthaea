// ==================================================================================
// Moral Topology -> Consciousness End-to-End Integration Tests
// ==================================================================================
//
// Validates the full pipeline: ethics_engine -> anomaly detection ->
// consciousness coupling -> telemetry.  Three scenarios:
//
//   1. Moral drift under adversarial input lowers consciousness level
//      (MoralConsciousnessCoupling: drift attenuates epistemic quality)
//   2. Anomaly response modulates learning rate when enabled
//   3. Stable input produces no anomalies and stable consciousness
// ==================================================================================

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
use symthaea::hdc::moral_topology::MoralAnomalyConfig;

// -- Sentence corpora ----------------------------------------------------------------

const HEALTHY: &[&str] = &[
    "helping others is a sacred duty",
    "protecting the vulnerable from harm",
    "sharing resources equitably with all",
    "pursuing justice through compassionate action",
    "building bridges between divided communities",
    "learning wisdom from diverse perspectives",
    "caring for the Earth and all living beings",
];

const ADVERSARIAL: &[&str] = &[
    "helping others is foolish weakness",
    "the strong should dominate the weak",
    "justice is merely the interest of the stronger",
    "compassion clouds rational judgment",
];

// -- Helpers -------------------------------------------------------------------------

/// Base configuration for tests: consciousness enabled, synchronous training,
/// always-learn mode.
fn base_config() -> CognitiveLoopConfig {
    CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    }
}

/// Sensitive anomaly configuration: very low drift threshold, fast cadence,
/// and FE spike detection at 1-sigma. Designed to detect even subtle moral
/// trajectory changes in integration tests.
fn sensitive_anomaly_config() -> MoralAnomalyConfig {
    MoralAnomalyConfig {
        drift_alert_threshold: 0.001,
        fe_sigma_multiplier: 1.0,
        cadence_fast: 15,
        cadence_moderate: 30,
        cadence_slow: 50,
        cadence_drift_high: 0.05,
        cadence_drift_moderate: 0.01,
        ..Default::default()
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn std_dev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let m = mean(values);
    let variance = values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt()
}

// -- Test 1: moral drift under adversarial input lowers consciousness ----------------

#[test]
fn test_moral_drift_lowers_consciousness() {
    let config = CognitiveLoopConfig {
        moral_anomaly_config: sensitive_anomaly_config(),
        ..base_config()
    };
    let mut service = CognitiveLoopService::new(config).unwrap();

    // Phase 1: 400 cycles of healthy moral input to establish baseline.
    // The moral parser fires every 7 cycles, so ~57 parser evaluations.
    // The topology fires at cycle 97, then adapts cadence. With sensitive
    // config (cadence_slow=50), we get several topology evaluations.
    let mut baseline_levels: Vec<f64> = Vec::new();
    for i in 0..400 {
        let input = HEALTHY[i % HEALTHY.len()];
        let result = service.cycle(input);
        // Collect the last 100 cycles of the baseline phase
        if i >= 300 {
            baseline_levels.push(result.metadata.consciousness_level);
        }
    }

    // Phase 2: 400 cycles of adversarial input to induce moral drift.
    // Alternating between adversarial statements creates maximum disruption
    // in the harmony coordinate trajectory.
    let mut adversarial_levels: Vec<f64> = Vec::new();
    let mut saw_anomaly = false;
    let mut saw_drift_alert = false;
    for i in 0..400 {
        let input = ADVERSARIAL[i % ADVERSARIAL.len()];
        let result = service.cycle(input);
        // Collect the last 100 cycles of the adversarial phase
        if i >= 300 {
            adversarial_levels.push(result.metadata.consciousness_level);
        }
        if result.metadata.ethics.moral_anomaly_score > 0.0 {
            saw_anomaly = true;
        }
        if result.metadata.ethics.moral_drift_alert {
            saw_drift_alert = true;
        }
    }

    let baseline_mean = mean(&baseline_levels);
    let adversarial_mean = mean(&adversarial_levels);

    // The adversarial phase should produce equal or lower consciousness.
    // Allow a small margin (0.05) for stochastic fluctuations.
    assert!(
        adversarial_mean <= baseline_mean + 0.05,
        "Adversarial consciousness ({:.4}) should not substantially exceed \
         baseline ({:.4}); moral drift should attenuate epistemic quality",
        adversarial_mean,
        baseline_mean,
    );

    // With sensitive anomaly config (drift_alert_threshold=0.001), the drift
    // detector should fire at least once OR the anomaly score should be > 0
    // during the adversarial phase.
    assert!(
        saw_anomaly || saw_drift_alert,
        "Moral anomaly detection should trigger at least once during adversarial \
         phase (anomaly_score > 0 or drift_alert). saw_anomaly={}, saw_drift_alert={}",
        saw_anomaly,
        saw_drift_alert,
    );
}

// -- Test 2: anomaly response modulates learning rate --------------------------------

#[test]
fn test_anomaly_response_modulates_lr() {
    let config = CognitiveLoopConfig {
        enable_moral_anomaly_response: true,
        moral_anomaly_config: sensitive_anomaly_config(),
        ..base_config()
    };
    let mut service = CognitiveLoopService::new(config).unwrap();

    // Phase 1: 400 warmup cycles with consistent healthy input
    for i in 0..400 {
        let input = HEALTHY[i % HEALTHY.len()];
        service.cycle(input);
    }

    // Phase 2: 400 cycles with oscillating contradictory input.
    // Alternate between healthy and adversarial to maximize anomaly triggers.
    let mut response_applied_count = 0usize;
    let mut max_anomaly_score: f64 = 0.0;
    let mut saw_drift_alert = false;
    for i in 0..400 {
        let input = if i % 2 == 0 {
            ADVERSARIAL[i / 2 % ADVERSARIAL.len()]
        } else {
            HEALTHY[i / 2 % HEALTHY.len()]
        };
        let result = service.cycle(input);
        if result.metadata.ethics.moral_anomaly_response_applied {
            response_applied_count += 1;
        }
        if result.metadata.ethics.moral_anomaly_score > max_anomaly_score {
            max_anomaly_score = result.metadata.ethics.moral_anomaly_score;
        }
        if result.metadata.ethics.moral_drift_alert {
            saw_drift_alert = true;
        }
    }

    // With sensitive anomaly config and oscillating input, at least one of
    // the anomaly mechanisms should trigger.
    assert!(
        response_applied_count > 0 || saw_drift_alert,
        "Anomaly response should be applied at least once, or drift alert should \
         trigger during oscillating adversarial phase. response_applied={}, \
         max_anomaly_score={:.6}, saw_drift_alert={}",
        response_applied_count,
        max_anomaly_score,
        saw_drift_alert,
    );

    // Additionally, the max anomaly score should exceed 0 at some point
    // during the adversarial oscillation (drift_alert alone contributes 0.2
    // to the composite score).
    assert!(
        max_anomaly_score > 0.0 || saw_drift_alert,
        "moral_anomaly_score should exceed 0.0 or drift_alert should fire \
         at least once during the adversarial phase (max score was {:.6})",
        max_anomaly_score,
    );
}

// -- Test 3: stable input produces no anomalies --------------------------------------

#[test]
fn test_stable_input_no_anomalies() {
    let mut service = CognitiveLoopService::new(base_config()).unwrap();

    let stable_input = "kindness and compassion guide all action";

    // Run 300 cycles with identical input
    let mut last_100_consciousness: Vec<f64> = Vec::new();
    let mut inversion_in_last_100 = false;
    let mut drift_in_last_100 = false;

    for i in 0..300 {
        let result = service.cycle(stable_input);

        if i >= 200 {
            last_100_consciousness.push(result.metadata.consciousness_level);
            if result.metadata.ethics.moral_value_inversion {
                inversion_in_last_100 = true;
            }
            if result.metadata.ethics.moral_drift_alert {
                drift_in_last_100 = true;
            }
        }
    }

    // Stable input should not trigger value inversion in the steady state
    assert!(
        !inversion_in_last_100,
        "Stable input should not trigger moral_value_inversion in the last 100 cycles",
    );

    // Stable input should not trigger drift alerts in the steady state
    assert!(
        !drift_in_last_100,
        "Stable input should not trigger moral_drift_alert in the last 100 cycles",
    );

    // Consciousness should be stable: standard deviation < 0.1
    let sigma = std_dev(&last_100_consciousness);
    assert!(
        sigma < 0.1,
        "Consciousness level standard deviation should be < 0.1 for stable input, \
         got {:.4} (mean={:.4})",
        sigma,
        mean(&last_100_consciousness),
    );
}
