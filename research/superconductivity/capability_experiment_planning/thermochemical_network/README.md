# SC-THERMOCHEMICAL-NETWORK-0.1

**Status:** bounded synthetic stoichiometric and thermodynamic network auditor; no DFT computation or physical synthesis is performed.  
**Research issue:** [#12997](https://github.com/Luminous-Dynamics/symthaea/issues/12997)  
**Reaction selectivity parent:** [PR #12996](https://github.com/Luminous-Dynamics/symthaea/pull/12996)  
**Synthesis protocol/execution lineage:** [PR #12618](https://github.com/Luminous-Dynamics/symthaea/pull/12618)

## What changes

The reaction-selectivity auditor (#12996) accepts source-declared reaction-balance metadata. This workstream makes the next layer executable: it computes atom and charge conservation from a declared species-composition map, validates path connectivity, sums reaction \(\Delta G\) intervals on a common extent basis and standard state, and checks both stoichiometric and energetic cycle closure.

The binary is \`sc_thermochemical_network_audit\`, implemented in the isolated \`superconductivity-discovery-harness\`.

## Contract

### Species

The elemental composition map is authoritative for the arithmetic:
- stable species ID;
- formula string retained for human/source provenance but not independently parsed;
- explicit element-to-atom-count map;
- integer charge.

The auditor deliberately does not guess composition from a formula string. Correct arithmetic over a wrong upstream formula remains wrong chemistry, so species identity/composition still needs trusted provenance and review.

### Reactions

Each reaction declares:
- signed stoichiometric terms (negative = reactant, positive = product);
- a condition ID shared by the network;
- a reference reaction-extent basis ID and unit shared by the network;
- a standard-state identifier shared by the network;
- interval-valued reaction free energy and activation barrier;
- a source class.

For each reaction, the auditor computes:

\[
r_e = \sum_i \nu_i n_{i,e}, \qquad
r_q = \sum_i \nu_i q_i
\]

where \(n_{i,e}\) is the declared count of element \(e\) in species \(i\), \(q_i\) its charge and \(\nu_i\) its signed coefficient. All residuals must be within the frozen balance tolerance.

This is a computed conservation check over caller-declared species composition maps. It does not independently validate the source's structural phase assignment, formula identity, oxidation state, or whether a proposed reaction is chemically accessible.

### Paths

An ordered path is validated for:
- registered, unique reaction IDs;
- declared starting species consumed by its first reaction;
- a shared intermediate hand-off between each adjacent pair of reactions;
- final production of its declared target species.

Route \(\Delta G\) is the interval sum of the reaction intervals, under the assumption that every reaction is expressed on the exact same declared extent basis, unit and standard state. The barrier output is the maximum lower and maximum upper per-step barrier envelope. This is not an Arrhenius rate, a reaction time, or a synthesis-probability estimate.

### Cycles

For each oriented reaction cycle, the auditor computes both:
1. net species coefficient residuals across the entire cycle;
2. the signed interval sum of reaction \(\Delta G\).

An energetic interval that intersects the zero-closure tolerance is considered *consistent with zero within supplied uncertainty* only after the species ledger also closes. If the species coefficients do not close, cycle status is \`cycle_stoichiometry_does_not_close\`, regardless of the \(\Delta G\) sum.

The interval test is conservative but not a proof of thermodynamic correctness: input intervals may share correlated model errors, use approximate standard-state data or omit relevant species.

## Frozen synthetic network

The main fixture includes:
- precursor \(A_2B\);
- element \(C\);
- intermediate \(A_2BC\);
- target polymorph \(A_2BC_2\);
- competing polymorph \(A_2BC_2\);
- a two-step target route;
- a direct competing route;
- a polymorph-conversion reaction;
- one four-edge cycle.

The target route sums to \(\Delta G \in [-0.90,-0.70]\) eV per declared reference extent. Its maximum barrier envelope is \([0.9,1.2]\) eV. The competing route has \(\Delta G \in [-0.75,-0.60]\) eV. The cycle energy interval is \([-0.35,0.15]\) eV and intersects zero within the declared closure tolerance; the species ledger closes exactly.

These values and chemical symbols are synthetic examples. They are not material candidates, reaction prescriptions, DFT results or laboratory conditions.

## Adversarial suite

Registry: \`tools/superconductivity-discovery-harness/fixtures/thermochemical_network_suite_v0_1.json\`.

1. **Balanced reference network** — all reaction atom/charge balances close, target route connects, competitor route connects, cycle closes in species space and its free-energy interval includes zero.
2. **Unbalanced reaction** — the coefficient of \(C\) in the first reaction is deliberately corrupted. The auditor rejects the network and identifies the nonzero \(C\) residual.
3. **Thermodynamically inconsistent cycle** — the polymorph-conversion \(\Delta G\) is shifted so the cycle interval excludes zero. The auditor reports an energy inconsistency after verifying stoichiometric cycle closure.
4. **Disconnected route** — the target path uses a reaction that does not produce the intermediate consumed by its next step. The auditor rejects the route.

Additional unit tests mutate standard state, extent basis, cycle orientation and target species to enforce fail-closed behavior.

## Reproducible commands

\`\`\`bash
cargo test --locked --manifest-path tools/superconductivity-discovery-harness/Cargo.toml --bin sc_thermochemical_network_audit

cargo run --locked --manifest-path tools/superconductivity-discovery-harness/Cargo.toml --bin sc_thermochemical_network_audit -- \\
  --schema data/schemas/superconductivity_thermochemical_network_v0_1.schema.json \\
  --fixture tools/superconductivity-discovery-harness/fixtures/thermochemical_network_smoke_v0_1.json \\
  --out /tmp/sc-thermochemical-network.json
\`\`\`

The discovery workflow also runs the unbalanced and disconnected fail-closed negative controls and retains the inconsistent-cycle receipt.

## Relationship to the 2026 synthesis literature

Schlesinger et al., *Thermodynamic assessment of machine learning models for solid-state synthesis prediction* (Materials Horizons, 2026), evaluates machine-learning synthesizability scores against energy above hull and thermodynamic selectivity of enumerated reaction routes. It reports that learned models generally overpredict synthesis likelihood and motivates separate hull and reaction-selectivity diagnostics: https://doi.org/10.1039/D6MH00209A

Lambard's 2026 synthesis-first perspective emphasizes the protocol-to-phase-to-property gap and the value of machine-readable protocols, inverse reaction-pathway design and closed-loop refinement: https://doi.org/10.1088/2515-7639/ae6e72

This auditor is a deterministic finite-network contract, not an ML synthesizability predictor and not a reproduction of either paper. Prospective comparison should use identical species/reaction graphs, reference states, candidate routes, source/precursor holdouts and actual phase-realization outcomes.

## Claim ceiling

The auditor establishes only that the synthetic species ledger conserves the declared elements/charge, that paths connect under the declared species graph, and that interval sums/cycles are internally consistent under the supplied standard state and normalization. It does not calculate DFT, prove species identity, establish a real reaction mechanism, predict an actual reaction rate, demonstrate a synthesized phase, establish superconductivity, or authorize a laboratory procedure.
