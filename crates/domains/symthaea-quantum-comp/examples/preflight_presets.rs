use symthaea_quantum_comp::{
    RunPreset, preflight_binding_config, preflight_matrix_config, supported_preset_names,
};

fn main() {
    for name in supported_preset_names() {
        let preset = RunPreset::from_name(name).unwrap();
        println!("preset={}", preset.name());
        println!(
            "{}",
            preflight_binding_config(&preset.binding_config()).to_text()
        );
        println!(
            "{}",
            preflight_matrix_config(&preset.matrix_config()).to_text()
        );
    }
}
