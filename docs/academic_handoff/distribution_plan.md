# Distribution Plan: Levin Lab Outreach Sequence

## 1. Goal
Initiate a collaborative research partnership with the Allen Discovery Center (Dr. Michael Levin) by providing a turn-key, open-source computational framework for morphogenetic active inference.

## 2. Outreach Assets
The following assets are located in `docs/academic_handoff/`:
- `README.md`: High-level engine overview.
- `levin_outreach_draft.md`: Formal email/correspondence pitch.
- `algebraic_unbinding_spec.md`: Mathematical foundation for state isolation.
- `planarian_tas_sim.rs`: Executable proof-of-concept simulation.

## 3. Implementation Sequence

### Phase 1: Intellectual Baseline Freeze
- **Tag Stable Release**: Create a `morphogenesis-v1.0` tag in the repository.
- **Verification**: Run `cargo check --workspace --all-targets --offline` to ensure no regressions.

### Phase 2: First Contact (Low-Friction)
- Send the `levin_outreach_draft.md` correspondence to Dr. Levin.
- Highlight the **Analytical ODE / CfC Advantage** for multi-scale biological modeling.
- Direct them to the `crates/symthaea-morphogenesis/examples/planarian_tas_sim.rs` file as an immediate "in-silico" demonstration.

### Phase 3: Technical Briefing
- If interest is confirmed, provide a live demonstration of the **Conformal Geometric HDC** growth tracking.
- Propose an initial data-sharing pilot: Mapping real-world *Xenopus* embryo bioelectric imaging data into the Symthaea organic mesh.

## 4. Key Value Propositions
- **O(D) Scale-Free Computing**: Resolving the combinatorial explosion of biological manifold modeling.
- **Topological Intent Reading**: Detecting "cryptic memory" as a verifiable mathematical invariant.
- **Implementation Independence**: A control layer that can migrate between silicon, hybrid organoids, and biological tissue.

## 5. Maintenance
- Ensure the `symthaea-morphogenesis` workspace crate remains isolated and robust during future monorepo upgrades.
- Continue refining the **Integrated Information (Phi) Bridge** to provide more granular causal emergence metrics.
