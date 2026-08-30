use std::collections::BTreeMap;

use symthaea_evidence_plane::{EvidenceCounters, Expectation, RunEvidence, RunId};
use symthaea_interoception::{
    InteroceptiveDrive, InteroceptiveStepReport, NativeInteroceptiveModel,
    NativeInteroceptiveState, ViabilityChannel,
};

fn counters_from_step(report: InteroceptiveStepReport) -> EvidenceCounters {
    let mut counters = EvidenceCounters::new();
    counters.record("driven_channels", report.driven_channels as f64);
    counters.record("restorative_channels", report.restorative_channels as f64);
    counters.record("clamped_channels", report.clamped_channels as f64);
    counters.record("changed_channels", report.changed_channels as f64);
    counters
}

fn declared(entries: &[(&str, Expectation)]) -> BTreeMap<String, Expectation> {
    entries
        .iter()
        .map(|(name, expectation)| ((*name).to_string(), *expectation))
        .collect()
}

#[test]
fn evidence_plane_distinguishes_passive_restorative_driven_and_clamped_arms() {
    let mut passive = NativeInteroceptiveModel::default();
    let passive_report = passive.step(InteroceptiveDrive::ZERO);
    let passive_evidence = RunEvidence::new(
        RunId::new("interoception-v0.1-passive"),
        &passive.config(),
        declared(&[
            ("driven_channels", Expectation::MustBeZero),
            ("restorative_channels", Expectation::MustBeZero),
            ("clamped_channels", Expectation::MustBeZero),
            ("changed_channels", Expectation::MustBeZero),
        ]),
        counters_from_step(passive_report),
    );
    passive_evidence.enforce();

    let mut perturbed = NativeInteroceptiveState::default();
    perturbed.get_mut(ViabilityChannel::ComputeReserve).value = 0.30;
    let mut restorative = NativeInteroceptiveModel::new(perturbed, Default::default());
    let restorative_report = restorative.step(InteroceptiveDrive::ZERO);
    let restorative_evidence = RunEvidence::new(
        RunId::new("interoception-v0.1-restorative"),
        &restorative.config(),
        declared(&[
            ("driven_channels", Expectation::MustBeZero),
            ("restorative_channels", Expectation::MustBePositive),
            ("clamped_channels", Expectation::MustBeZero),
            ("changed_channels", Expectation::MustBePositive),
        ]),
        counters_from_step(restorative_report),
    );
    restorative_evidence.enforce();

    let mut driven = NativeInteroceptiveModel::default();
    let drive = InteroceptiveDrive::ZERO.with_rate(ViabilityChannel::ComputeReserve, -0.01);
    let driven_report = driven.step(drive);
    let driven_evidence = RunEvidence::new(
        RunId::new("interoception-v0.1-driven"),
        &(driven.config(), drive),
        declared(&[
            ("driven_channels", Expectation::MustBePositive),
            ("restorative_channels", Expectation::MustBeZero),
            ("clamped_channels", Expectation::MustBeZero),
            ("changed_channels", Expectation::MustBePositive),
        ]),
        counters_from_step(driven_report),
    );
    driven_evidence.enforce();

    let mut clamped = NativeInteroceptiveModel::default();
    let clamp_drive = InteroceptiveDrive::ZERO.with_rate(ViabilityChannel::ComputeReserve, -2.0);
    let clamped_report = clamped.step(clamp_drive);
    let clamped_evidence = RunEvidence::new(
        RunId::new("interoception-v0.1-clamped"),
        &(clamped.config(), clamp_drive),
        declared(&[
            ("driven_channels", Expectation::MustBePositive),
            ("restorative_channels", Expectation::MustBeZero),
            ("clamped_channels", Expectation::MustBePositive),
            ("changed_channels", Expectation::MustBePositive),
        ]),
        counters_from_step(clamped_report),
    );
    clamped_evidence.enforce();
}
