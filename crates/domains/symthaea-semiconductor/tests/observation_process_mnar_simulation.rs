#![forbid(unsafe_code)]

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

use symthaea_semiconductor::benchmark::{
    ControllerAssumptionStateV1, FeedbackTrackingRecordV1, FeedbackTrackingStateV1,
    IndependentAuditObservationV1, K1QualificationRecordV1, K1QualificationStateV1,
    OBSERVATION_PROCESS_SCHEMA_V1, ObservationProcessInputV1, ObservationProcessRoundV1,
    ObservationRiskControllerV1, PhysicalAcquisitionRecordV1, PhysicalAcquisitionStateV1,
    PhysicalSelectionDesignKindV1, SelectionAuditDesignV1, SELECTION_AUDIT_DESIGN_SCHEMA_V1,
    evaluate_observation_process_v1,
};

const POPULATION_SIZE: usize = 512;
const SEED_REPLICATES: usize = 64;
const MAX_DELAY: u64 = 3;
const TRACKING_CAPACITY: usize = 2;
const AUDIT_PROBABILITY: f64 = 0.40;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MissingnessRegime {
    Mcar,
    MarByStratum,
    MnarByOutcome,
}

impl MissingnessRegime {
    fn name(self) -> &'static str {
        match self {
            Self::Mcar => "mcar",
            Self::MarByStratum => "mar-by-stratum",
            Self::MnarByOutcome => "mnar-by-outcome",
        }
    }
}

#[derive(Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
    }

    fn index(&mut self, upper_exclusive: usize) -> usize {
        (self.next_u64() % upper_exclusive as u64) as usize
    }
}

fn digest_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn digest_label(value: &str) -> String {
    digest_bytes(value.as_bytes())
}

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> String {
    digest_bytes(&serde_json::to_vec(value).unwrap())
}

#[derive(Clone, Copy)]
struct SeedBundle {
    latent_truth: u64,
    acquisition: u64,
    qualification: u64,
    feedback_delay: u64,
    proxy_tracking: u64,
    independent_audit: u64,
}

impl SeedBundle {
    fn from_root(root: u64) -> Self {
        let value = Self {
            latent_truth: root ^ 0x2D35_8DCC_AA6C_78A5,
            acquisition: root ^ 0xA076_1D64_78BD_642F,
            qualification: root ^ 0xE703_7ED1_A0B4_28DB,
            feedback_delay: root ^ 0x8EBC_6AF0_9C88_C6E3,
            proxy_tracking: root ^ 0x5899_65CC_7537_4CC3,
            independent_audit: root ^ 0x1D8E_4E27_C47D_124F,
        };
        let unique = BTreeSet::from([
            value.latent_truth,
            value.acquisition,
            value.qualification,
            value.feedback_delay,
            value.proxy_tracking,
            value.independent_audit,
        ]);
        assert_eq!(unique.len(), 6, "simulation random streams must have distinct seeds");
        value
    }

    fn manifest_sha256(self, root: u64) -> String {
        digest_label(&format!(
            "root={root};latent={};acquisition={};qualification={};delay={};proxy={};audit={}",
            self.latent_truth,
            self.acquisition,
            self.qualification,
            self.feedback_delay,
            self.proxy_tracking,
            self.independent_audit
        ))
    }
}

fn qualification_missing_probability(
    regime: MissingnessRegime,
    stratum: usize,
    true_loss: f64,
) -> f64 {
    match regime {
        MissingnessRegime::Mcar => 0.18,
        MissingnessRegime::MarByStratum => {
            if stratum == 0 { 0.08 } else { 0.32 }
        }
        MissingnessRegime::MnarByOutcome => {
            if true_loss > 0.5 { 0.58 } else { 0.08 }
        }
    }
}

fn operational_selection_probability(regime: MissingnessRegime, stratum: usize) -> f64 {
    if regime == MissingnessRegime::Mcar {
        0.65
    } else if stratum == 0 {
        0.45
    } else {
        0.82
    }
}

fn proxy_delay_survival(actual_delay: u64) -> f64 {
    // Proxy delay is uniform over {-1,0,1,2,3}.
    match actual_delay {
        0 => 0.8,
        1 => 0.6,
        2 => 0.4,
        3 => 0.2,
        _ => 0.0,
    }
}

fn make_row(
    round_id: usize,
    true_loss: f64,
    selected: bool,
    measurement_succeeded: bool,
    qualification_state: K1QualificationStateV1,
    label_revision: Option<u64>,
    payload_sha256: Option<String>,
    proxy_delay: Option<i64>,
    actual_delay: Option<u64>,
    slot_id: Option<u64>,
    tracking_state: FeedbackTrackingStateV1,
    tracking_end_round: Option<u64>,
    survival_probability: Option<f64>,
    controller_received_at: Option<i64>,
    controller_update_applied: bool,
    audit: Option<IndependentAuditObservationV1>,
    design_sha256: &str,
    selection_probability: f64,
) -> ObservationProcessRoundV1 {
    let round = round_id as u64;
    let event_time = 1_700_000_000_i64 + round_id as i64 * 100;
    let prediction_time = event_time + 1;
    let measured_at = event_time + 2;
    let label_available = event_time + 3;
    let report_available = event_time + 4;
    let measured = selected && measurement_succeeded;
    let acquired_state = if !selected {
        PhysicalAcquisitionStateV1::NotSelected
    } else if !measurement_succeeded {
        PhysicalAcquisitionStateV1::MeasurementUnavailable
    } else {
        PhysicalAcquisitionStateV1::Measured
    };
    let has_label = measured;
    let report_exists = has_label;
    let qualified = qualification_state == K1QualificationStateV1::Eligible;
    let loss_receipt = qualified.then(|| digest_label(&format!("qualified-loss-{round_id}")));
    let selected_probability = selection_probability;

    ObservationProcessRoundV1 {
        round_id: round,
        sample_id: format!("mc-{round_id}"),
        measurement_event_id: format!("mc-event-{round_id}"),
        prediction_receipt_sha256: digest_label(&format!("prediction-{round_id}")),
        interval_receipt_sha256: digest_label(&format!("interval-{round_id}")),
        model_artifact_sha256: digest_label("frozen-global-model-v1"),
        event_time_unix_seconds: event_time,
        prediction_time_unix_seconds: prediction_time,
        clock_domain_id: "synthetic-fab-clock".into(),
        acquisition: PhysicalAcquisitionRecordV1 {
            state: acquired_state,
            selection_design_kind: PhysicalSelectionDesignKindV1::RandomizedKnownInclusionProbability,
            selection_design_sha256: Some(design_sha256.into()),
            inclusion_probability: Some(selected_probability),
            acquisition_cost_units: selected.then_some(if measured { 1.0 } else { 0.2 }),
            selected_at_round: selected.then_some(round),
            measured_at_unix_seconds: measured.then_some(measured_at),
            operational_measurement_method_sha256: measured.then(|| digest_label("operational-gauge-v1")),
        },
        qualification: K1QualificationRecordV1 {
            state: if has_label {
                qualification_state
            } else {
                K1QualificationStateV1::NotApplicable
            },
            label_id: has_label.then(|| format!("mc-label-{round_id}")),
            label_revision,
            label_payload_sha256: payload_sha256,
            measurement_event_id: has_label.then(|| format!("mc-event-{round_id}")),
            label_available_at_unix_seconds: has_label.then_some(label_available),
            report_sha256: report_exists.then(|| digest_label(&format!("k1-report-{round_id}"))),
            report_available_at_unix_seconds: report_exists.then_some(report_available),
            qualified_loss: qualified.then_some(true_loss),
            qualified_loss_evidence_sha256: loss_receipt,
        },
        tracking: FeedbackTrackingRecordV1 {
            state: if measured {
                tracking_state
            } else {
                FeedbackTrackingStateV1::NotApplicable
            },
            proxy_delay_rounds: measured.then_some(proxy_delay).flatten(),
            actual_feedback_delay_rounds: measured.then_some(actual_delay).flatten(),
            controller_feedback_received_at_unix_seconds: if measured {
                controller_received_at
            } else {
                None
            },
            proxy_survival_probability: measured.then_some(survival_probability).flatten(),
            tracking_slot_id: measured.then_some(slot_id).flatten(),
            tracking_end_round: measured.then_some(tracking_end_round).flatten(),
            action_receipt_sha256: measured.then(|| digest_label(&format!("tracking-action-{round_id}"))),
            controller_update_applied: measured && controller_update_applied && qualified,
        },
        independent_audit: audit,
    }
}

struct SimulationOutput {
    input_sha256: String,
    report_sha256: String,
    truth: f64,
    qualified_risk: Option<f64>,
    audit_estimate: f64,
    audit_lower: f64,
    audit_upper: f64,
    partial_lower: f64,
    partial_upper: f64,
    max_occupancy: u64,
    capacity_denied: usize,
    missing_count: usize,
}

fn simulate(root_seed: u64, regime: MissingnessRegime) -> SimulationOutput {
    let seeds = SeedBundle::from_root(root_seed);
    let mut latent_rng = SplitMix64::new(seeds.latent_truth);
    let mut acquisition_rng = SplitMix64::new(seeds.acquisition);
    let mut qualification_rng = SplitMix64::new(seeds.qualification);
    let mut delay_rng = SplitMix64::new(seeds.feedback_delay);
    let mut proxy_rng = SplitMix64::new(seeds.proxy_tracking);
    let mut audit_rng = SplitMix64::new(seeds.independent_audit);

    let design_sha256 = digest_label(&format!("selection-design-{}", regime.name()));
    let audit_design = SelectionAuditDesignV1 {
        schema: SELECTION_AUDIT_DESIGN_SCHEMA_V1.into(),
        design_sha256: digest_label(&format!("audit-design-{}", regime.name())),
        randomization_provenance_sha256: seeds.manifest_sha256(root_seed),
        reference_method_sha256: digest_label("independent-audit-gauge-v1"),
        minimum_inclusion_probability: AUDIT_PROBABILITY,
        confidence_delta: 0.05,
        independent_of_operational_qualification: true,
    };

    let mut rows: Vec<ObservationProcessRoundV1> = Vec::with_capacity(POPULATION_SIZE);
    let mut active_slots: Vec<(usize, u64)> = Vec::new();
    let mut truth_sum = 0.0;

    for round_id in 0..POPULATION_SIZE {
        let stratum = (round_id / 32) % 2;
        let event_probability = if stratum == 0 { 0.05 } else { 0.30 };
        let true_loss = if latent_rng.unit() < event_probability { 1.0 } else { 0.0 };
        truth_sum += true_loss;

        let selection_probability = operational_selection_probability(regime, stratum);
        let selected = acquisition_rng.unit() < selection_probability;
        let measurement_succeeded = selected && acquisition_rng.unit() >= 0.02;
        let mut qualification_state = K1QualificationStateV1::NotApplicable;
        let mut label_revision = None;
        let mut payload_sha256 = None;

        if measurement_succeeded {
            let missing_probability =
                qualification_missing_probability(regime, stratum, true_loss);
            if qualification_rng.unit() < missing_probability {
                qualification_state = if qualification_rng.unit() < 0.5 {
                    K1QualificationStateV1::Blocked
                } else {
                    K1QualificationStateV1::Unknown
                };
            } else {
                qualification_state = K1QualificationStateV1::Eligible;
            }
            label_revision = Some(1);
            payload_sha256 = Some(digest_label(&format!(
                "mc-label-payload-{round_id}-revision-1-loss-{true_loss}"
            )));
        }

        let mut proxy_delay = None;
        let mut actual_delay = None;
        let mut slot_id = None;
        let mut tracking_state = FeedbackTrackingStateV1::NotApplicable;
        let mut tracking_end = None;
        let mut survival = None;
        let mut controller_received = None;
        let mut update_applied = false;

        if measurement_succeeded {
            // The bounded-delay tracker evicts expired slots before the new
            // admission; feedback arriving on its expiry round still occupies
            // the slot for that round, as specified by CA-OCRC.
            active_slots.retain(|(_, end_round)| *end_round >= round_id as u64);

            let proxy = [-1_i64, 0, 1, 2, 3][proxy_rng.index(5)];
            let actual = delay_rng.index((MAX_DELAY + 1) as usize) as u64;
            proxy_delay = Some(proxy);
            actual_delay = Some(actual);

            if proxy < 0 {
                tracking_state = FeedbackTrackingStateV1::ProxyNotTracked;
            } else if active_slots.len() >= TRACKING_CAPACITY {
                tracking_state = FeedbackTrackingStateV1::CapacityDenied;
            } else {
                let end = if actual <= proxy as u64 {
                    tracking_state = FeedbackTrackingStateV1::TrackedReceived;
                    survival = Some(proxy_delay_survival(actual));
                    controller_received = Some(
                        1_700_000_000_i64
                            + round_id as i64 * 100
                            + 4
                            + actual as i64 * 100,
                    );
                    update_applied = qualification_state == K1QualificationStateV1::Eligible;
                    round_id as u64 + actual
                } else {
                    tracking_state = FeedbackTrackingStateV1::ProxyExpired;
                    round_id as u64 + proxy as u64
                };
                slot_id = Some(round_id as u64 + 1);
                tracking_end = Some(end);
                active_slots.push((round_id, end));
            }
        }

        let is_missing = !measurement_succeeded
            || qualification_state != K1QualificationStateV1::Eligible;
        let audit = if is_missing && audit_rng.unit() < AUDIT_PROBABILITY {
            Some(IndependentAuditObservationV1 {
                round_id: round_id as u64,
                loss: true_loss,
                inclusion_probability: AUDIT_PROBABILITY,
                audit_measurement_receipt_sha256: digest_label(&format!(
                    "audit-receipt-{root_seed}-{round_id}"
                )),
            })
        } else {
            None
        };

        rows.push(make_row(
            round_id,
            true_loss,
            selected,
            measurement_succeeded,
            qualification_state,
            label_revision,
            payload_sha256,
            proxy_delay,
            actual_delay,
            slot_id,
            tracking_state,
            tracking_end,
            survival,
            controller_received,
            update_applied,
            audit,
            &design_sha256,
            selection_probability,
        ));
    }

    let mut sorted = rows.clone();
    sorted.sort_by_key(|row| row.round_id);
    let source_ledger_sha256 = json_sha256(&sorted);
    let input = ObservationProcessInputV1 {
        schema: OBSERVATION_PROCESS_SCHEMA_V1.into(),
        run_id: format!("mc-{}-{}", regime.name(), root_seed),
        population_manifest_sha256: digest_label(&format!("population-{}-{root_seed}", regime.name())),
        source_ledger_sha256,
        controller: ObservationRiskControllerV1::CapacityAwareOcrcBounded,
        controller_policy_sha256: Some(digest_label("bounded-ca-ocrc-policy-k2-dmax3")),
        delay_contract_sha256: Some(digest_label("synthetic-delay-contract-dmax3")),
        delay_contract_declared_and_enforced: true,
        fixed_delay_rounds: None,
        maximum_feedback_delay_rounds: Some(MAX_DELAY),
        tracking_capacity: Some(TRACKING_CAPACITY as u64),
        closeout_round: POPULATION_SIZE as u64 + MAX_DELAY + 1,
        independent_audit_design: Some(audit_design),
        rounds: rows,
    };
    let report = evaluate_observation_process_v1(&input).unwrap();
    let audit = report.population_risk.independent_audit_estimate.as_ref().unwrap();
    SimulationOutput {
        input_sha256: report.input_sha256,
        report_sha256: report.report_sha256,
        truth: truth_sum / POPULATION_SIZE as f64,
        qualified_risk: report.population_risk.qualified_subset_risk,
        audit_estimate: audit.population_risk_estimate,
        audit_lower: audit.hoeffding_lower,
        audit_upper: audit.hoeffding_upper,
        partial_lower: report.population_risk.partial_identification_lower,
        partial_upper: report.population_risk.partial_identification_upper,
        max_occupancy: report.maximum_observed_tracking_occupancy,
        capacity_denied: report.funnel.capacity_denied,
        missing_count: report.population_risk.missing_count,
    }
}

#[test]
fn mcar_mar_mnar_joint_observation_simulations_preserve_population_bounds_and_audit_coverage() {
    for regime in [
        MissingnessRegime::Mcar,
        MissingnessRegime::MarByStratum,
        MissingnessRegime::MnarByOutcome,
    ] {
        let mut audit_coverages = 0_usize;
        let mut audit_bias_sum = 0.0;
        let mut capacity_denials = 0_usize;

        for replicate in 0..SEED_REPLICATES {
            let seed = 0x5EED_0000_u64 + replicate as u64 * 7919;
            let result = simulate(seed, regime);
            assert!(
                result.partial_lower <= result.truth + 1e-12
                    && result.truth <= result.partial_upper + 1e-12,
                "{} seed {seed}: worst-case bound missed truth [{}, {}] vs {}",
                regime.name(),
                result.partial_lower,
                result.partial_upper,
                result.truth
            );
            if result.audit_lower <= result.truth + 1e-12
                && result.truth <= result.audit_upper + 1e-12
            {
                audit_coverages += 1;
            }
            assert!(result.max_occupancy <= TRACKING_CAPACITY as u64);
            assert!(result.missing_count > 0);
            assert!(result.qualified_risk.is_some());
            audit_bias_sum += result.audit_estimate - result.truth;
            capacity_denials += result.capacity_denied;
        }

        let mean_audit_bias = audit_bias_sum / SEED_REPLICATES as f64;
        assert!(
            audit_coverages >= (SEED_REPLICATES * 90).div_ceil(100),
            "{}: only {audit_coverages}/{SEED_REPLICATES} audit Hoeffding intervals contained population truth",
            regime.name()
        );
        assert!(
            mean_audit_bias.abs() < 0.025,
            "{}: audit HT mean bias {}",
            regime.name(),
            mean_audit_bias
        );
        assert!(capacity_denials > 0, "{}: capacity stress did not exercise a denial", regime.name());
    }
}

#[test]
fn fixed_seed_bundle_replays_identical_population_and_observation_receipts() {
    let first = simulate(0xC0FF_EE12, MissingnessRegime::MnarByOutcome);
    let second = simulate(0xC0FF_EE12, MissingnessRegime::MnarByOutcome);
    assert_eq!(first.input_sha256, second.input_sha256);
    assert_eq!(first.report_sha256, second.report_sha256);
}

#[test]
fn mnar_qualified_subset_can_false_reassure_while_truth_and_audit_remain_separate() {
    let mut truth_sum = 0.0;
    let mut qualified_sum = 0.0;
    let mut audit_sum = 0.0;
    for replicate in 0..SEED_REPLICATES {
        let seed = 0xA11D_0000_u64 + replicate as u64 * 6151;
        let result = simulate(seed, MissingnessRegime::MnarByOutcome);
        truth_sum += result.truth;
        qualified_sum += result.qualified_risk.unwrap_or(0.0);
        audit_sum += result.audit_estimate;
    }
    let n = SEED_REPLICATES as f64;
    let mean_truth = truth_sum / n;
    let mean_qualified = qualified_sum / n;
    let mean_audit = audit_sum / n;
    assert!(
        mean_qualified + 0.02 < mean_truth,
        "MNAR qualified subset should demonstrate false reassurance: qualified={mean_qualified}, truth={mean_truth}"
    );
    assert!(
        (mean_audit - mean_truth).abs() < 0.025,
        "independent audit mean should recover the population: audit={mean_audit}, truth={mean_truth}"
    );
}
