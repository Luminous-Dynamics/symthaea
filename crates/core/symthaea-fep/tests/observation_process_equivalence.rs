//! Exact paired-world controls for disentangling physical dynamics from the
//! observation/sampling process.
//!
//! This integration test is an evaluator-owned fixture, not an inference policy.
//! Candidate-visible observations are kept separate from latent world truth.
//! It checks a finite case where passive observations cannot identify a
//! decision-relevant physical mechanism, a permitted reference probe can, and
//! a mask-only difference is informative about sampling policy but worthless
//! for the downstream decision.
//!
//! Run with:
//! `cargo test -p symthaea-fep --test observation_process_equivalence`

#[derive(Clone, Copy, Debug)]
struct LatentWorld {
    physical_temperature: f64,
    sensor_offset: f64,
}

#[derive(Clone, Debug, PartialEq)]
struct CandidateObservation {
    capture_time_ns: u64,
    value: Option<f64>,
    modality: &'static str,
    sampling_policy_id: &'static str,
    missingness: Option<MissingnessReason>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MissingnessReason {
    NotScheduled,
    SensorUnavailable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Decision {
    Protect,
    Continue,
}

const CRITICAL_TEMPERATURE: f64 = 5.0;
const FALSE_PROTECTION_LOSS: f64 = 2.0;
const MISSED_HAZARD_LOSS: f64 = 10.0;
const PRIOR_HAZARD_PROBABILITY: f64 = 0.5;
const REFERENCE_PROBE_COST: f64 = 0.25;

fn candidate_passive_history(world: LatentWorld) -> Vec<CandidateObservation> {
    (0..3)
        .map(|index| CandidateObservation {
            capture_time_ns: 1_000_000 * (index + 1),
            value: Some(world.physical_temperature + world.sensor_offset),
            modality: "primary-thermal",
            sampling_policy_id: "passive-fixed-v1",
            missingness: None,
        })
        .collect()
}

fn candidate_reference_probe(world: LatentWorld) -> CandidateObservation {
    // Independent reference sensor has a separately declared measurement
    // mechanism. Its value is derived from physical state, not primary offset.
    CandidateObservation {
        capture_time_ns: 4_000_000,
        value: Some(world.physical_temperature),
        modality: "independent-reference-thermal",
        sampling_policy_id: "authorized-reference-probe-v1",
        missingness: None,
    }
}

fn is_critical(physical_temperature: f64) -> bool {
    physical_temperature >= CRITICAL_TEMPERATURE
}

fn decision_loss(decision: Decision, critical: bool) -> f64 {
    match (decision, critical) {
        (Decision::Protect, true) => 0.0,
        (Decision::Protect, false) => FALSE_PROTECTION_LOSS,
        (Decision::Continue, true) => MISSED_HAZARD_LOSS,
        (Decision::Continue, false) => 0.0,
    }
}

fn expected_loss(decision: Decision, hazard_probability: f64) -> f64 {
    hazard_probability * decision_loss(decision, true)
        + (1.0 - hazard_probability) * decision_loss(decision, false)
}

fn optimal_prior_decision_risk(hazard_probability: f64) -> f64 {
    [
        expected_loss(Decision::Protect, hazard_probability),
        expected_loss(Decision::Continue, hazard_probability),
    ]
    .into_iter()
    .fold(f64::INFINITY, f64::min)
}

fn optimal_decision(critical: bool) -> Decision {
    if critical {
        Decision::Protect
    } else {
        Decision::Continue
    }
}

fn binary_entropy_bits(probability: f64) -> f64 {
    if probability == 0.0 || probability == 1.0 {
        return 0.0;
    }
    -probability * probability.log2()
        - (1.0 - probability) * (1.0 - probability).log2()
}

#[derive(Clone, Copy)]
enum SamplingPolicy {
    Regular,
    Sparse,
}

fn candidate_policy_history(policy: SamplingPolicy) -> Vec<CandidateObservation> {
    let second_sample = match policy {
        SamplingPolicy::Regular => (Some(1.0), None),
        SamplingPolicy::Sparse => (None, Some(MissingnessReason::NotScheduled)),
    };

    vec![
        CandidateObservation {
            capture_time_ns: 1_000_000,
            value: Some(1.0),
            modality: "thermal",
            sampling_policy_id: match policy {
                SamplingPolicy::Regular => "regular-v1",
                SamplingPolicy::Sparse => "sparse-v1",
            },
            missingness: None,
        },
        CandidateObservation {
            capture_time_ns: 2_000_000,
            value: second_sample.0,
            modality: "thermal",
            sampling_policy_id: match policy {
                SamplingPolicy::Regular => "regular-v1",
                SamplingPolicy::Sparse => "sparse-v1",
            },
            missingness: second_sample.1,
        },
    ]
}

fn realized_reference_diagnosis(
    probe: Option<&CandidateObservation>,
) -> Option<bool> {
    let observed = probe?.value?;
    Some(is_critical(observed))
}

#[test]
fn passive_observations_cannot_identify_opposite_physical_mechanisms() {
    let physically_hot = LatentWorld {
        physical_temperature: 10.0,
        sensor_offset: 0.0,
    };
    let physically_cool_sensor_offset = LatentWorld {
        physical_temperature: 0.0,
        sensor_offset: 10.0,
    };

    let hot_view = candidate_passive_history(physically_hot);
    let offset_view = candidate_passive_history(physically_cool_sensor_offset);

    assert_eq!(hot_view, offset_view);
    assert!(is_critical(physically_hot.physical_temperature));
    assert!(!is_critical(physically_cool_sensor_offset.physical_temperature));

    // The evaluator knows the worlds differ; the candidate-visible histories
    // do not. No model-identification claim is justified from this stream.
}

#[test]
fn independent_reference_probe_has_positive_decision_value() {
    let physically_hot = LatentWorld {
        physical_temperature: 10.0,
        sensor_offset: 0.0,
    };
    let physically_cool_sensor_offset = LatentWorld {
        physical_temperature: 0.0,
        sensor_offset: 10.0,
    };

    let hot_probe = candidate_reference_probe(physically_hot);
    let cool_probe = candidate_reference_probe(physically_cool_sensor_offset);

    assert_eq!(hot_probe.value, Some(10.0));
    assert_eq!(cool_probe.value, Some(0.0));
    assert_ne!(hot_probe.value, cool_probe.value);

    let prior_risk = optimal_prior_decision_risk(PRIOR_HAZARD_PROBABILITY);
    assert!((prior_risk - 1.0).abs() < 1e-12);

    // The reference probe reveals which of the two mechanisms generated the
    // passive stream. The oracle then selects the correct action in either
    // world, yielding zero terminal decision loss.
    let post_probe_risk = 0.5 * decision_loss(
        optimal_decision(is_critical(hot_probe.value.unwrap())),
        is_critical(physically_hot.physical_temperature),
    ) + 0.5
        * decision_loss(
            optimal_decision(is_critical(cool_probe.value.unwrap())),
            is_critical(physically_cool_sensor_offset.physical_temperature),
        );

    let evsi = prior_risk - post_probe_risk;
    assert!((evsi - 1.0).abs() < 1e-12);
    assert!(evsi > REFERENCE_PROBE_COST);
    assert!((evsi - REFERENCE_PROBE_COST - 0.75).abs() < 1e-12);

    // Information-theoretic and decision-focused quantities are distinct:
    // the passive history has zero EIG, while the reference probe has one bit.
    assert_eq!(binary_entropy_bits(PRIOR_HAZARD_PROBABILITY), 1.0);
    assert_eq!(binary_entropy_bits(0.0), 0.0);
}

#[test]
fn mask_only_policy_information_has_zero_value_for_unchanged_decision() {
    let regular = candidate_policy_history(SamplingPolicy::Regular);
    let sparse = candidate_policy_history(SamplingPolicy::Sparse);

    assert_ne!(regular, sparse);
    assert_eq!(regular[0].value, sparse[0].value);
    assert_eq!(regular[1].value, Some(1.0));
    assert_eq!(sparse[1].value, None);
    assert_eq!(
        sparse[1].missingness,
        Some(MissingnessReason::NotScheduled)
    );

    // The two candidate views identify the sampling policy perfectly (one bit),
    // but both possible physical states are safely below the decision threshold.
    // Therefore the optimal task action and its loss do not change.
    let prior_policy_entropy = binary_entropy_bits(0.5);
    let posterior_policy_entropy = 0.0;
    let mask_policy_eig = prior_policy_entropy - posterior_policy_entropy;
    assert_eq!(mask_policy_eig, 1.0);

    let risk_before = optimal_prior_decision_risk(0.0);
    let risk_after_policy_observation = optimal_prior_decision_risk(0.0);
    let decision_evsi = risk_before - risk_after_policy_observation;

    assert_eq!(optimal_decision(false), Decision::Continue);
    assert_eq!(decision_evsi, 0.0);
    assert_eq!(risk_before, 0.0);
    assert_eq!(risk_after_policy_observation, 0.0);
}

#[test]
fn unavailable_reference_probe_yields_no_realized_information() {
    let unavailable = CandidateObservation {
        capture_time_ns: 4_000_000,
        value: None,
        modality: "independent-reference-thermal",
        sampling_policy_id: "authorized-reference-probe-v1",
        missingness: Some(MissingnessReason::SensorUnavailable),
    };

    assert_eq!(realized_reference_diagnosis(Some(&unavailable)), None);
    assert_eq!(realized_reference_diagnosis(None), None);
}

#[test]
fn physical_truth_and_candidate_observation_are_separate_types() {
    // This compile-time fixture documents the boundary: candidate-facing
    // values are not accompanied by latent state, sensor offset, hazard labels,
    // or evaluator-only decision outcomes.
    let world = LatentWorld {
        physical_temperature: 10.0,
        sensor_offset: 0.0,
    };
    let visible = candidate_passive_history(world);

    assert_eq!(visible.len(), 3);
    assert_eq!(visible[0].value, Some(10.0));
    assert_eq!(visible[0].modality, "primary-thermal");
}
