import re

# Memory fix
with open('/srv/luminous-dynamics/symthaea/src/cognitive_loop/accessors/memory.rs', 'r') as f:
    mem = f.read()
mem = re.sub(r'\{ let bridge = &self\.sensorimotor\.somatic_bridge;\n\s*\.as_ref\(\)', '{ let bridge = &self.sensorimotor.somatic_bridge;', mem)
mem = re.sub(r'self\.sensorimotor\n\s*\.somatic_bridge\n\s*\.as_ref\(\)', '&self.sensorimotor.somatic_bridge', mem)
with open('/srv/luminous-dynamics/symthaea/src/cognitive_loop/accessors/memory.rs', 'w') as f:
    f.write(mem)

# Inspector fix
with open('/srv/luminous-dynamics/symtropy/crates/symtropy-render-bridge/src/inspector.rs', 'r') as f:
    insp = f.read()

insp = insp.replace('app.add_systems(Update, inspector_ui);', '// app.add_systems(Update, inspector_ui);')

with open('/srv/luminous-dynamics/symtropy/crates/symtropy-render-bridge/src/inspector.rs', 'w') as f:
    f.write(insp)
