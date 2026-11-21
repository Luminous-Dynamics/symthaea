//! Inspect the renderer-neutral Mycelial Presence model and tube mesh.

use symthaea_canvas::{
    EmbodimentSnapshot, PresenceMode, TubeTessellationConfig, build_mycelial_model,
    tessellate_mycelial_model,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let snapshot = EmbodimentSnapshot::default();
    let config = TubeTessellationConfig::default();

    for mode in [
        PresenceMode::Ambient,
        PresenceMode::Cognitive,
        PresenceMode::Conversational,
    ] {
        let model = build_mycelial_model(&snapshot, mode);
        let mesh = tessellate_mycelial_model(&model, config)?;
        println!(
            "mode={mode:?} filaments={} terminal={} vertices={} triangles={} valid={}",
            model.filament_count(),
            model.terminal_filament_count(),
            mesh.vertex_count(),
            mesh.triangle_count(),
            mesh.validate(),
        );
    }

    Ok(())
}
