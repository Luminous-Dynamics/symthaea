import re

with open('/srv/luminous-dynamics/symthaea/src/cognitive_loop/accessors/memory.rs', 'r') as f:
    mem_content = f.read()

# Fix .somatic_bridge.as_ref() missing as_ref issue - we should remove .as_ref() and the whole matching if .is_some() logic if it's not an Option anymore
mem_content = re.sub(r'if let Some\(bridge\) = self\n\s*\.sensorimotor\n\s*\.somatic_bridge\n\s*\.as_ref\(\) \{', '{ let bridge = &self.sensorimotor.somatic_bridge;', mem_content)
with open('/srv/luminous-dynamics/symthaea/src/cognitive_loop/accessors/memory.rs', 'w') as f:
    f.write(mem_content)

# Fix symtropy-render-bridge
with open('/srv/luminous-dynamics/symtropy/crates/symtropy-render-bridge/src/inspector.rs', 'r') as f:
    insp = f.read()

insp = insp.replace('mut query_material: Query<&mut Handle<PhiHeatmapMaterial>>', '/* mut query_material: Query<&mut Handle<PhiHeatmapMaterial>> */')
insp = insp.replace('mut materials: ResMut<Assets<PhiHeatmapMaterial>>', '/* mut materials: ResMut<Assets<PhiHeatmapMaterial>> */')

with open('/srv/luminous-dynamics/symtropy/crates/symtropy-render-bridge/src/inspector.rs', 'w') as f:
    f.write(insp)

