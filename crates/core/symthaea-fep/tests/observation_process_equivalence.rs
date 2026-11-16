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

fn posterior_hazard_probability(
    prior_hazard_probability: f64,
    likelihood_if_hazard: f64,
    likelihood_if_offset: f64,
) -> f64 {
    let hazard_mass = prior_hazard_probability * likelihood_if_hazard;
    let offset_mass = (1.0 - prior_hazard_probability) * likelihood_if_offset;
    hazard_mass / (hazard_mass + offset_mass)
}

fn binary_entropy_bits(probability: f64) -> f64 {
    if probability == 0.0 || probability == 1.0 {
        return 0.0;
    }
    -probability * probability.log2() - (1.0 - probability) * (1.0 - probability).log2()
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

fn realized_reference_diagnosis(probe: Option<&CandidateObservation>) -> Option<bool> {
    let probe = probe?;
    if probe.modality != "independent-reference-thermal"
        || probe.sampling_policy_id != "authorized-reference-probe-v1"
    {
        return None;
    }
    let observed = probe.value?;
    Some(is_critical(observed))
}

#[test]
fn identical_passive_likelihoods_leave_noisy_world_posterior_unchanged() {
    // Both physical mechanisms yield the same distribution for the primary
    // sensor, not merely the same single realised trace.
    let prior = PRIOR_HAZARD_PROBABILITY;
    let posterior_after_low = posterior_hazard_probability(prior, 0.2, 0.2);
    let posterior_after_high = posterior_hazard_probability(prior, 0.8, 0.8);

    assert!((posterior_after_low - prior).abs() < 1e-12);
    assert!((posterior_after_high - prior).abs() < 1e-12);
    assert_eq!(
        binary_entropy_bits(prior) - binary_entropy_bits(posterior_after_low),
        0.0
    );
}

#[test]
fn noisy_reference_probe_separates_eig_from_decision_evsi() {
    let prior = PRIOR_HAZARD_PROBABILITY;
    let probability_hot_if_hazard: f64 = 0.9;
    let probability_hot_if_sensor_offset: f64 = 0.1;
    let posterior_after_hot = posterior_hazard_probability(
        prior,
        probability_hot_if_hazard,
        probability_hot_if_sensor_offset,
    );
    let posterior_after_cool = posterior_hazard_probability(
        prior,
        1.0 - probability_hot_if_hazard,
        1.0 - probability_hot_if_sensor_offset,
    );

    assert!((posterior_after_hot - 0.9).abs() < 1e-12);
    assert!((posterior_after_cool - 0.1).abs() < 1e-12);

    let eig_bits = binary_entropy_bits(prior)
        - 0.5 * binary_entropy_bits(posterior_after_hot)
        - 0.5 * binary_entropy_bits(posterior_after_cool);
    assert!((eig_bits - (1.0 - binary_entropy_bits(0.1))).abs() < 1e-12);
    assert!(eig_bits > 0.5);

    let prior_risk = optimal_prior_decision_risk(prior);
    let expected_post_probe_risk = 0.5 * optimal_prior_decision_risk(posterior_after_hot)
        + 0.5 * optimal_prior_decision_risk(posterior_after_cool);
    let evsi = prior_risk - expected_post_probe_risk;

    assert!((prior_risk - 1.0).abs() < 1e-12);
    assert!((expected_post_probe_risk - 0.6).abs() < 1e-12);
    assert!((evsi - 0.4).abs() < 1e-12);
    // The probe has useful information, but much less task value than EIG.
    assert!(evsi < eig_bits);
}

#[test]
fn noisy_probe_is_rejected_when_its_cost_exceeds_evsi() {
    let posterior_after_hot = posterior_hazard_probability(0.5, 0.9, 0.1);
    let posterior_after_cool = posterior_hazard_probability(0.5, 0.1, 0.9);
    let prior_risk = optimal_prior_decision_risk(0.5);
    let expected_post_probe_risk = 0.5 * optimal_prior_decision_risk(posterior_after_hot)
        + 0.5 * optimal_prior_decision_risk(posterior_after_cool);
    let evsi = prior_risk - expected_post_probe_risk;

    let low_cost_net_value = evsi - 0.25;
    let high_cost_net_value = evsi - 0.5;
    assert!(low_cost_net_value > 0.0);
    assert!(high_cost_net_value < 0.0);
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
    let passive_posterior = posterior_hazard_probability(0.5, 1.0, 1.0);
    let passive_eig = binary_entropy_bits(0.5) - binary_entropy_bits(passive_posterior);
    assert_eq!(passive_posterior, 0.5);
    assert_eq!(passive_eig, 0.0);
    assert!(is_critical(physically_hot.physical_temperature));
    assert!(!is_critical(
        physically_cool_sensor_offset.physical_temperature
    ));

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
    let post_probe_risk =
        0.5 * decision_loss(
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
    // passive observations leave the posterior unchanged, while the independent
    // reference outcome identifies the mechanism in this finite oracle fixture.
    let posterior_after_hot_probe = posterior_hazard_probability(0.5, 1.0, 0.0);
    let posterior_after_cool_probe = posterior_hazard_probability(0.5, 0.0, 1.0);
    assert_eq!(posterior_after_hot_probe, 1.0);
    assert_eq!(posterior_after_cool_probe, 0.0);
    let reference_eig = binary_entropy_bits(0.5)
        - 0.5 * binary_entropy_bits(posterior_after_hot_probe)
        - 0.5 * binary_entropy_bits(posterior_after_cool_probe);
    assert_eq!(reference_eig, 1.0);
}

#[test]
fn mask_only_policy_information_has_zero_value_for_unchanged_decision() {
    let regular = candidate_policy_history(SamplingPolicy::Regular);
    let sparse = candidate_policy_history(SamplingPolicy::Sparse);

    assert_ne!(regular, sparse);
    assert_eq!(regular[0].value, sparse[0].value);
    assert_eq!(regular[1].value, Some(1.0));
    assert_eq!(sparse[1].value, None);
    assert_eq!(sparse[1].missingness, Some(MissingnessReason::NotScheduled));

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

    let primary_sensor_observation = CandidateObservation {
        capture_time_ns: 4_000_000,
        value: Some(10.0),
        modality: "primary-thermal",
        sampling_policy_id: "passive-fixed-v1",
        missingness: None,
    };
    assert_eq!(
        realized_reference_diagnosis(Some(&primary_sensor_observation)),
        None
    );
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
