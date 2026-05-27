import os

target_file = "crates/symthaea-fep/src/agent.rs"
if not os.path.exists(target_file) and os.path.exists("agent.rs"):
    target_file = "agent.rs"

if os.path.exists(target_file):
    with open(target_file, "r") as f:
        content = f.read()

    # Define the epigenetic chromatin gating implementation
    epigenetic_impl = """
    /// Levin Epigenetic Regulation: Computes a sparse, chromatin-style element-wise 
    /// masking vector based on the agent's active operational archetype. Folds away 
    /// irrelevant sectors of the 16,384D vector space to drastically compress 
    /// combinatorial search bloat and protect specialized parameters from cross-talk.
    pub fn enforce_epigenetic_vector_masking(&self, latent_hv: &mut crate::hdc::unified_hv::ContinuousHV) {
        // Archetype Gating: If planning horizon is contracted (Stress/Reflex Mode),
        // mask out the upper multi-scale environmental tracking sectors of the vector
        if self.config.planning_horizon <= 1 {
            let half_dim = latent_hv.values.len() / 2;
            for i in half_dim..latent_hv.values.len() {
                // Mechanically silence long-term strategic dimensions (Chromatin folding)
                latent_hv.values[i] = 0.0;
            }
        }
    }

    /// Friston Niche Construction: Active Environmental Stigmergy."""

    if "enforce_epigenetic_vector_masking" not in content:
        lines = content.splitlines()
        updated = False
        
        for i, line in enumerate(lines):
            if "pub fn project_stigmergic_niche_marker" in line:
                lines.insert(i, epigenetic_impl)
                updated = True
                break
                
        if updated:
            content = "\n".join(lines) + "\n"
            with open(target_file, "w") as f:
                f.write(content)
            print("✔ Successfully deployed the Epigenetic Expression Gating Module.")
        else:
            print("Error: Could not locate project_stigmergic_niche_marker entry boundary.")
    else:
        print("Epigenetic gating logic is already active inside agent.rs.")
else:
    print("Error: Target agent script file could not be located in your workspace directories.")
