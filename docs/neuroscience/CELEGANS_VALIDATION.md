# C. elegans Connectome Module — Evidence and Limitations

**Status:** synthetic graph fixture; not empirical connectome validation.
**Source module:** crates/core/symthaea-core/src/hdc/celegans_connectome.rs
**Scientific use currently supported:** development and testing of graph/HDC algorithms on a deterministic, representative topology.

## Critical correction

The living organism C. elegans has a well-studied nervous system. The current Symthaea constructor does **not** load its empirical connectome edge table.

The Rust implementation creates a representative topology of approximately 279 named neuron slots and generates connectivity with a deterministic pseudo-random number generator according to broad sensory → interneuron → motor motifs. The source code itself describes the result as representative and constructs edges using deterministic pseudo-random draws.

Therefore:

- It is not a complete 302-neuron empirical connectome.
- Its edge identities and synapse counts are not imported from White, Varshney, Cook, Brittin or WormWiring source files.
- The generated graph must not be called “biologically accurate” or ground truth.
- A Φ value computed from this fixture is a property of the constructed synthetic graph and the chosen Φ approximation. It is not a biological measurement or evidence that evolution optimizes for a particular information-theoretic quantity.

This distinction supersedes any earlier wording in this document or related README that described the embedded data as a complete biological connectome or reported synthetic fixture behavior as biological validation.

## Biological background (separate from the fixture)

C. elegans hermaphrodite connectomics is a valuable neuroscience resource. Published reconstructions and datasets differ in scope, animal, preparation, reconstruction and annotation. A research import must select one specific source release and cite it explicitly rather than merging counts or neuron identities across sources.

Relevant references include:

- White et al. (1986), *The structure of the nervous system of the nematode Caenorhabditis elegans*.
- Varshney et al. (2011), *Structural properties of the Caenorhabditis elegans neuronal network*.
- Cook et al. (2019), *Whole-animal connectomes of both Caenorhabditis elegans sexes*.
- The WormWiring resource, with its exact dataset/version and retrieval date.

## Current fixture structure

The implementation represents named neurons, broad neuron classes, directed chemical edges and undirected electrical-coupling pairs. The connectivity-generation routine uses deterministic random draws and motif-inspired rules to make repeatable graphs for tests.

This is useful for:
- Checking graph import/export and subnetwork extraction.
- Testing deterministic behavior under a fixed seed.
- Comparing synthetic graph families under identical node counts and degree constraints.
- Testing HDC encoding and graph-statistic code.
- Injecting known structural perturbations and verifying that an algorithm detects them.

It is not currently suitable for:
- Estimating biological synapse counts or edge weights.
- Comparing biological sex, developmental stage or individual variation.
- Inferring actual sensorimotor pathways or circuit function.
- Validating the biological meaning of Φ.
- Making claims about human brain structure or psychiatric disease.

## Required empirical-data import before biological claims

A future importer must:

1. Pin a versioned source archive, citation, licence and retrieval timestamp.
2. Store a cryptographic digest of every input file and a manifest of the exact data release.
3. Preserve source neuron identifiers, cell class, sex/animal identifier and unresolved/ambiguous identities.
4. Preserve chemical and electrical connection types separately, with their source-defined weights and units.
5. Define how left/right, partial reconstructions, duplicate labels and uncertain synapses are handled.
6. Validate counts and identifiers against the original source's own release documentation.
7. Keep uncertain edges as alternatives or intervals rather than inventing a single “correct” value.
8. Compare derived graph statistics against independently computed reference results.
9. Compare exact Φ calculations with an appropriate independent implementation only where system size and assumptions make exact computation feasible; label approximations explicitly.
10. Publish a reproduction command, input hashes, tests and output report.

A correct import is still not a clinical model. C. elegans and human in-vivo neuroimaging are different species, scales and measurement regimes.

## Test interpretation

Existing tests can establish that:
- The constructor returns a deterministic object.
- Neuron-index and edge data structures satisfy programmed invariants.
- Subnetwork extraction and graph calculations execute.
- The chosen synthetic fixture behaves consistently under a fixed construction procedure.

They cannot establish:
- Agreement with an empirical C. elegans connectome.
- General biological principles from a single synthetic graph.
- A valid biomarker of consciousness.
- Clinical prediction, treatment selection or treatment efficacy.

Any test output from the current constructor must therefore carry a label equivalent to **synthetic topology fixture**.

## Recommended benchmark ladder

When an empirical source importer exists, compare:

- Source reconstruction fidelity: node and edge precision/recall, unresolved identity rate, weight agreement and source-version compatibility.
- Topology robustness: statistic variance over source ambiguities, annotation alternatives and permitted preprocessing.
- Null-model controls: degree-preserving, spatial and class-constrained nulls where scientifically justified.
- HDC representation: retrieval / reconstruction error, stability across perturbations, memory and runtime against transparent graph baselines.
- Biological relevance: only prespecified, independently grounded circuit/task predictions with held-out evidence.

Do not declare superiority from a single Φ ratio or a single random-graph comparison.

## Intended-use boundary

This module is for basic research and software development. It is not a diagnostic, prognostic or treatment system. No patient-specific inference or clinical claim should be derived from these fixture outputs.
