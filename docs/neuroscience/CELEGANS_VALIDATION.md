# C. elegans Synthetic Topology Fixture (Legacy)

**Status:** synthetic graph fixture only — **not** a source-verified biological connectome, neural dynamics model, or closed-loop animal controller.

**Module:** `crates/core/symthaea-core/src/hdc/celegans_connectome.rs`

**Tracked remediation:** [SCI-CONNECTOMICS-033 / #9902](https://github.com/Luminous-Dynamics/symthaea/issues/9902)

---

## Executive summary

The current `CElegansConnectome::new()` constructor builds a deterministic, C. elegans-inspired synthetic graph. It does not load the published anatomical connectome.

This distinction is important. Passing the existing tests means the synthetic graph and its topology-analysis helpers are internally usable. It does not establish that the graph reproduces biological anatomy, neural signal propagation, locomotion, chemotaxis, or consciousness.

Do not use output from this module as biological connectome evidence or as the neural controller for an embodied animal.

## What the implementation actually constructs

The current fixture creates:

| Node class | Generated entries |
|---|---:|
| Sensory | 80 |
| Interneuron | 82 |
| Motor | 117 |
| **Total** | **279** |

Connectivity is generated using a fixed-seed pseudo-random procedure with hand-selected probabilities for broad connection classes. Generated edge weights are synthetic floating-point values.

The public field names need careful interpretation:

- `total_chemical_synapses` counts generated directed chemical edges, not measured synaptic contacts.
- `total_gap_junctions` counts generated bidirectional pairs in this fixture, not a source-derived biological contact count.
- The generated weight values are neither anatomical contact counts nor calibrated physiological synaptic efficacies.

The fixture omits the pharyngeal system and does not represent all 302 hermaphrodite neurons. A comment or reference to published biological counts elsewhere in the module does not change the actual generated data.

## What it can and cannot validate

| Capability | Current status |
|---|---|
| Deterministic synthetic graph construction | Implemented |
| Basic graph statistics and subnetwork extraction | Implemented |
| Conversion into the existing topology representation | Implemented |
| Spectral/algebraic-connectivity analysis | Implemented as a graph statistic |
| Exact source neuron identities and anatomical edges | **Not implemented** |
| Source synapse/contact counts and polarity | **Not implemented** |
| Measured effective connectivity / perturbation responses | **Not implemented** |
| Time-dependent neural dynamics | **Not implemented** |
| Sensory, proprioceptive and motor interfaces | **Not implemented** |
| Physical body and environment coupling | **Not implemented** |
| Held-out locomotion or chemotaxis validation | **Not implemented** |

The existing spectral/algebraic-connectivity values are not exact Integrated Information Theory Φ. They should be described by the mathematical quantity actually computed, with no consciousness claim inferred from them.

## Scientific context

C. elegans remains an unusually valuable organism because extensive whole-animal wiring diagrams and neural measurements are available. It is not accurate to describe it as the only organism with a mapped connectome. Other major resources include the adult Drosophila whole-brain reconstruction and the 2026 brain-and-ventral-nerve-cord reconstruction.

More importantly, anatomy and function are distinct evidence layers. Randi et al. (2023) measured perturbation-driven signal propagation across thousands of neuron pairs. Those responses can include multi-hop, neuromodulatory and extrasynaptic effects; they must not be inserted into an anatomical graph as if they were observed direct synapses.

## Required model separation

Future source-backed work must preserve these as distinct artifacts:

1. **Structural connectome:** source-identified neurons, chemical synapse contacts, gap junctions, direction, polarity, uncertainty and completeness.
2. **Effective coupling:** measured perturbation response, sign, amplitude, latency, uncertainty and experimental scope.
3. **Neural dynamics:** model equations, fitted parameters, state, solver and timestep.
4. **Sensorimotor mapping:** receptor inputs, interoception/proprioception, motor neurons and body effectors.
5. **Body/environment:** morphology, muscles, contact mechanics, substrate and environmental fields.
6. **Behavioural evidence:** measured trajectories and task outcomes with held-out subject, study and condition identities.

Unknown, unmeasured, disputed, not applicable and measured-zero must remain different states. Missing data must not become fabricated edges, weights or capabilities.

## Reference research and software

- White et al. (1986), original C. elegans nervous-system reconstruction.
- Varshney et al. (2011), structural properties of the C. elegans neuronal network.
- Cook et al. (2019), whole-animal connectomes of both C. elegans sexes: https://doi.org/10.1038/s41586-019-1352-7
- Randi et al. (2023), neural signal propagation atlas: https://www.nature.com/articles/s41586-023-06683-4
- BAAIWorm (2024), integrated brain/body/environment model: https://www.nature.com/articles/s43588-024-00738-w
- Lee (2026 preprint), whole-body c302 simulation with restricted interneuron stimulation: https://doi.org/10.64898/2026.09.06.749731
- Dorkenwald et al. (2024), adult Drosophila whole-brain connectome: https://www.nature.com/articles/s41586-024-07558-y
- Bates et al. (2026), adult female Drosophila brain-and-nerve-cord connectome: https://www.nature.com/articles/s41586-026-10735-w

These resources have different anatomical scopes, experimental methods, model assumptions and use terms. Pin exact artifacts and rights before importing or redistributing data.

## Existing test boundary

The existing tests cover deterministic fixture creation, broad neuron-type populations, synthetic connectivity statistics, topology conversion, subnetwork extraction, graph-statistic analysis and topology comparisons.

They do not currently test:
- exact source data ingestion;
- node/edge identity against an authoritative release;
- measured synapse counts or polarity;
- neural response to perturbation;
- body mechanics or sensory feedback;
- locomotion, food seeking or other animal behaviour.

## Next implementation steps

1. Keep the legacy graph available only as an explicitly named synthetic fixture.
2. Define a source-bound, versioned connectome artifact and independent import oracle.
3. Import one exact, rights-cleared C. elegans data release without filling unknowns.
4. Add a separate perturbation/effective-coupling representation.
5. Reproduce a pinned neural-dynamics and closed-loop behaviour baseline.
6. Connect only qualified model artifacts to Symtropy's embodied-controller adapter.

See [#9902](https://github.com/Luminous-Dynamics/symthaea/issues/9902) for the full remediation, source-integrity, benchmark and delivery contract.

---

*This document describes the current implementation boundary as audited on 2026-11-02. It does not report a biological validation result.*