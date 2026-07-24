import re

with open('/srv/luminous-dynamics/symthaea/src/cognitive_loop/accessors/memory.rs', 'r') as f:
    mem = f.read()

mem = mem.replace('self.sensorimotor.somatic_bridge.last_perception_hv().map(|hv| hv.to_vec())', 'None')
mem = mem.replace('Some(self.sensorimotor.somatic_bridge.sensorimotor_accuracy() as f64)', 'Some(0.0)')

with open('/srv/luminous-dynamics/symthaea/src/cognitive_loop/accessors/memory.rs', 'w') as f:
    f.write(mem)

