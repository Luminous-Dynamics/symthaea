import re

with open('/srv/luminous-dynamics/symthaea/src/cognitive_loop/fep_module.rs', 'r') as f:
    fep_content = f.read()

# Remove the health retrieval since it doesn't exist
fep_content = fep_content.replace(
    'let health = bridge.actuator_health();',
    'let health = vec![1.0; bridge.num_actuators()]; // Default to full health'
)

with open('/srv/luminous-dynamics/symthaea/src/cognitive_loop/fep_module.rs', 'w') as f:
    f.write(fep_content)

# Fix symtropy-render-bridge
with open('/srv/luminous-dynamics/symtropy/crates/symtropy-render-bridge/src/material.rs', 'r') as f:
    mat = f.read()
mat = mat.replace('use bevy::shader::ShaderRef;', 'use bevy::render::render_resource::ShaderRef;')
with open('/srv/luminous-dynamics/symtropy/crates/symtropy-render-bridge/src/material.rs', 'w') as f:
    f.write(mat)

with open('/srv/luminous-dynamics/symtropy/crates/symtropy-render-bridge/src/inspector.rs', 'r') as f:
    insp = f.read()
insp = insp.replace('use crate::material::{PhiHeatmapMaterial, PhiHeatmapSettings};', '// use crate::material::{PhiHeatmapMaterial, PhiHeatmapSettings};')
with open('/srv/luminous-dynamics/symtropy/crates/symtropy-render-bridge/src/inspector.rs', 'w') as f:
    f.write(insp)

