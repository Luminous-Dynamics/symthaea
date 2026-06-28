use symthaea_quantum_comp::{
    BindingProbeConfig, ExperimentMatrixConfig, ResearchBundle, RunPreset, known_schema_labels,
    preflight_binding_config, preflight_matrix_config,
};

#[test]
fn alpha7_preset_configs_are_preflight_clean_enough() {
    for preset in [
        RunPreset::Smoke,
        RunPreset::LocalResearch,
        RunPreset::PilotMatrix,
    ] {
        let binding = preflight_binding_config(&preset.binding_config());
        assert!(
            binding.can_run(),
            "binding preflight failed for {:?}: {}",
            preset,
            binding.to_text()
        );
        let matrix = preflight_matrix_config(&preset.matrix_config());
        assert!(
            matrix.can_run(),
            "matrix preflight failed for {:?}: {}",
            preset,
            matrix.to_text()
        );
    }
}

#[test]
fn alpha7_preflight_blocks_invalid_binding() {
    let cfg = BindingProbeConfig {
        dimension: 0,
        trials: 0,
        noise: -0.1,
        seed: 1,
        topology_threshold: 2.0,
    };
    let report = preflight_binding_config(&cfg);
    assert!(!report.can_run());
    assert!(report.to_text().contains("dimension-zero"));
}

#[test]
fn alpha7_schema_labels_are_stable() {
    let labels = known_schema_labels();
    assert!(labels.iter().any(|label| label.contains("binding_probe")));
    assert!(
        labels
            .iter()
            .all(|label| label.starts_with("symthaea.quantum_comp."))
    );
}

#[test]
fn alpha7_bundle_is_deterministic() {
    let a = ResearchBundle::new("bundle", "manifest", "result", "audit", "receipt");
    let b = ResearchBundle::new("bundle", "manifest", "result", "audit", "receipt");
    assert_eq!(a.bundle_fingerprint, b.bundle_fingerprint);
    assert!(a.to_markdown().contains("local bundle only"));
}

#[test]
fn alpha7_matrix_default_has_multiple_cells() {
    let cfg = ExperimentMatrixConfig::default();
    assert!(cfg.dimensions.len() > 1);
    assert!(cfg.noise_levels.len() > 1);
}
