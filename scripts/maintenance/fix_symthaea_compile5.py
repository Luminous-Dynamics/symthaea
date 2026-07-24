import re

# Memory fix: SomaticErrorBridge is not an Option, it's just the struct.
# Original code we want to fix is roughly:
# .somatic_bridge.and_then(|b| b.last_perception_hv().map(|hv| hv.to_vec()))
# .somatic_bridge.map(|b| b.sensorimotor_accuracy() as f64)

with open('/srv/luminous-dynamics/symthaea/src/cognitive_loop/accessors/memory.rs', 'r') as f:
    mem = f.read()

# Replace and_then
mem = re.sub(r'&self\.sensorimotor\.somatic_bridge\s*\.and_then\(\|b\| b\.last_perception_hv\(\)\.map\(\|hv\| hv\.to_vec\(\)\)\)', 
             'self.sensorimotor.somatic_bridge.last_perception_hv().map(|hv| hv.to_vec())', mem)

# Replace map
mem = re.sub(r'&self\.sensorimotor\.somatic_bridge\s*\.map\(\|b\| b\.sensorimotor_accuracy\(\) as f64\)', 
             'Some(self.sensorimotor.somatic_bridge.sensorimotor_accuracy() as f64)', mem)

with open('/srv/luminous-dynamics/symthaea/src/cognitive_loop/accessors/memory.rs', 'w') as f:
    f.write(mem)

