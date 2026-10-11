use symthaea_interoception::{
    AllostaticConfig, InteroceptiveSnapshot, NativeInteroceptiveModel,
    INTEROCEPTIVE_MODEL_SEMANTICS_VERSION, INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION,
};

#[test]
fn captured_snapshot_binds_schema_and_model_semantics_versions() {
    let model = NativeInteroceptiveModel::default();
    let snapshot = InteroceptiveSnapshot::capture_kinematic(&model, AllostaticConfig::default());

    assert_eq!(snapshot.schema_version, INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION);
    assert_eq!(
        snapshot.model_semantics_version,
        INTEROCEPTIVE_MODEL_SEMANTICS_VERSION
    );
}
