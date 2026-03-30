# Convergence: Connecting the Multiworld Sim to the Ecosystem

## The 5 Streams

### 1. Symthaea Consciousness Pipeline
**What changed**: `cycle_with_hv()` now feeds directly into the consciousness
equation (commit `e6d7a7c04`). Previously, phi=0.0 for all non-text inputs.
Consciousness is now modality-agnostic.

**Integration**: The sim's robotic fleet uses `MotorSafetyLevel::from_phi(phi)`
with thresholds Green(>0.6)/Yellow(>0.3)/Orange(>0.1)/Red(≤0.1). These are
the SAME thresholds in `symthaea-core/src/embodiment.rs:33-42`. Our robotics
module is already aligned — no code change needed, just documentation that
these numbers come from Symthaea's consciousness equation.

**Next**: When the sim's V2 engine (cohort + event-driven) is built, each
robot's Phi should be computed by Symthaea's actual consciousness equation
running on the robot's sensor state, not a scalar from power allocation.
This is the bridge between sim-level approximation and physics-level truth.

### 2. Symtropy Biological Validation
**What changed**: The game's physics produces real biological phenomena:
- Cooperative metabolic scaling (Kleiber's law)
- Consciousness threshold at 45% matches experimental 42%
- Information efficiency validated against biological systems
- 9D physics verified (bivector algebra up to D=9)

**Integration**: The multiworld sim uses the same thermodynamic principles
as Symtropy — energy conservation, entropy production, consciousness as
integrated information. The biological validation strengthens the sim's
theoretical foundation: if the physics produces correct scaling in a game
environment, it should produce correct scaling in a civilization environment.

**Next**: Add a "Symtropy validation" section to the technical report citing
the biological validation results as independent confirmation of the
thermodynamic model.

### 3. Honest Framing (Paper)
**What changed**: The paper reframed from "consciousness" to "functional
correlates of consciousness-theoretic predictions" (commit `063639a9d`,
following Phua 2025).

**Integration**: The multiworld sim should adopt the same honesty:
- We model "functional correlates of collective consciousness" — not
  actual collective consciousness
- ConsciousnessProfile is a governance metric, not a claim about sentience
- CollectivePhi measures organizational integration, not phenomenal experience
- Robot Phi measures computational coherence, not robot feelings

**Next**: Update TECHNICAL_REPORT.md with honest framing. Replace claims
like "consciousness-gated governance" with "consciousness-metric-gated
governance" in documentation.

### 4. Portal Domain Structure
**What changed**: 7 content domains complete with 41 tests:
- Hearth: kinship bonds (Partner/Child/Mentor/Companion), gratitude notes,
  stories, milestones
- Knowledge: claims with status, fact-check verdicts, prediction markets
- Finance, Commons, Health, Education, Governance all wired

**Integration**: The sim's social graph (cohort.rs NotableAgent connections)
maps directly to Hearth's kinship bonds. The knowledge system maps to
Knowledge claims. The project system maps to Governance proposals. The
TEND economy maps to Finance payments.

**Next**: When the portal visualizes simulation data, each domain has a
natural home:
- Hearth → named character relationships and grief propagation
- Knowledge → exploration discoveries and tech milestones
- Finance → TEND economy and trade balance
- Governance → independence movements and policy decisions
- Health → radiation exposure and medical facility tracking

### 5. ALIFE 2026 Deadline (April 12)
**What changed**: Symtropy paper submitted with biological validation.
LaTeX, figures, and reproduction guide complete.

**Integration**: The multiworld sim provides a different validation case
for the same physics — civilizational-scale emergence from thermodynamic
principles. The paper could reference the sim as "ongoing work applying
the same framework to multi-generational population dynamics" in the
Future Work section.

**Next**: Not blocking — the ALIFE paper is about Symtropy specifically.
But the multiworld sim strengthens the research program's credibility
as a whole.

## Alignment Verification

| Sim Component | Symthaea Source | Aligned? |
|---------------|----------------|----------|
| Robot Phi thresholds | `embodiment.rs:33-42` | YES (same values) |
| ConsciousnessProfile | `bridge-common` | YES (imported) |
| ConsciousnessTier | `bridge-common` | YES (imported) |
| PlanetaryBody | `bridge-common` | YES (imported) |
| MotorSafetyLevel | `embodiment.rs:24-29` | Conceptual (not imported) |
| Thermodynamic model | Symtropy validated | Theoretical alignment |
| Honest framing | Paper reframe | Needs documentation update |
| Portal domains | 7 domains complete | Ready for visualization |
