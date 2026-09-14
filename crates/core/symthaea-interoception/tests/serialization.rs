use symthaea_interoception::{
    apply_intervention, AllostaticConfig, InteroceptiveDrive, InteroceptiveIntervention,
    InteroceptiveSnapshot, InterventionRecord, NativeInteroceptiveModel, ViabilityChannel,
};

#[test]
fn kinematic_snapshot_survives_json_round_trip() {
    let model = NativeInteroceptiveModel::default();
    let snapshot = InteroceptiveSnapshot::capture_kinematic(&model, AllostaticConfig::default());

    let encoded = serde_json::to_vec(&snapshot).expect("serialize kinematic snapshot");
    let decoded: InteroceptiveSnapshot =
        serde_json::from_slice(&encoded).expect("deserialize kinematic snapshot");

    assert_eq!(decoded, snapshot);
}

#[test]
fn dynamics_aware_snapshot_survives_json_round_trip() {
    let model = NativeInteroceptiveModel::default();
    let drive = InteroceptiveDrive::ZERO
        .with_rate(ViabilityChannel::ComputeReserve, -0.03)
        .with_rate(ViabilityChannel::EpistemicResolution, -0.01);
    let snapshot =
        InteroceptiveSnapshot::capture_with_drive(&model, drive, AllostaticConfig::default());

    let encoded = serde_json::to_vec(&snapshot).expect("serialize rollout snapshot");
    let decoded: InteroceptiveSnapshot =
        serde_json::from_slice(&encoded).expect("deserialize rollout snapshot");

    assert_eq!(decoded, snapshot);
}

#[test]
fn intervention_receipt_survives_json_round_trip() {
    let mut model = NativeInteroceptiveModel::default();
    let receipt = apply_intervention(
        &mut model,
        InteroceptiveIntervention::add(ViabilityChannel::Integrity, -0.2),
    );

    let encoded = serde_json::to_vec(&receipt).expect("serialize intervention receipt");
    let decoded: InterventionRecord =
        serde_json::from_slice(&encoded).expect("deserialize intervention receipt");

    assert_eq!(decoded, receipt);
}
