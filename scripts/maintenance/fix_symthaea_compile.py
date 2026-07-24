import re

with open('/srv/luminous-dynamics/symthaea/src/cognitive_loop/managers/learning_manager.rs', 'r') as f:
    content = f.read()

# Remove duplicate inject_federated_wisdom block
dup_block = r'''    pub fn inject_federated_wisdom\(&mut self, peer_id: String, hash: \[u8; 32\], value: f32\) -> bool \{
        if hash == self.last_peer_wisdom_hash \{
            return false;
        \}
        self.last_peer_wisdom_hash = hash;

        // Epistemic Humility: Don't trust peer wisdom blindly.
        // Integrate only a fraction based on trust \(here hardcoded humility factor 0.3\).
        let humility_factor = 0.3;
        let integrated_value = value \* humility_factor;

        self.federated_wisdom_acc \+= integrated_value;

        // High-value federated wisdom boosts local plasticity to facilitate integration
        let boost = \(integrated_value \* 0.05\).min\(0.1\);
        self.plasticity = \(self.plasticity \+ boost\).min\(Self::MAX_PLASTICITY\);

        tracing::debug!\(
            "🧪 Epistemic Humility: Integrated \{:.2\} from \{\} \(raw=\{:.2\}\)",
            integrated_value,
            peer_id,
            value
        \);
        true
    \}'''

content = re.sub(dup_block, '', content, count=1)

# Add missing fields
content = content.replace('error_trend: f32,', 'error_trend: f32,\n    federated_wisdom_acc: f32,\n    last_peer_wisdom_hash: [u8; 32],')
content = content.replace('error_trend: 0.0,', 'error_trend: 0.0,\n            federated_wisdom_acc: 0.0,\n            last_peer_wisdom_hash: [0; 32],')

with open('/srv/luminous-dynamics/symthaea/src/cognitive_loop/managers/learning_manager.rs', 'w') as f:
    f.write(content)

with open('/srv/luminous-dynamics/symthaea/src/cognitive_loop/accessors/memory.rs', 'r') as f:
    mem_content = f.read()

# Replace .embodiment_bridge with .somatic_bridge
mem_content = mem_content.replace('.embodiment_bridge', '.somatic_bridge')
with open('/srv/luminous-dynamics/symthaea/src/cognitive_loop/accessors/memory.rs', 'w') as f:
    f.write(mem_content)

with open('/srv/luminous-dynamics/symthaea/src/cognitive_loop/config/mod.rs', 'r') as f:
    cfg = f.read()

cfg = cfg.replace('let mut config = Self {', 'let config = Self {')
with open('/srv/luminous-dynamics/symthaea/src/cognitive_loop/config/mod.rs', 'w') as f:
    f.write(cfg)

