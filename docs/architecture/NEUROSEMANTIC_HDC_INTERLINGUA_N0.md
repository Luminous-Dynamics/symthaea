# Neurosemantic HDC Interlingua N0

Status: experimental / synthetic evidence only

## Purpose

This experiment defines the first deterministic, versioned HDC representation for a `GroundedConceptGraph`.

It is intentionally narrower than semantic decoding. The tested claim is:

> Given a fixed training-derived codebook, a held-out synthetic concept graph whose atomic node and relation vocabulary is already present in that codebook can be encoded, quantized, transmitted as binary HDC frames, retrieved with the same codebook, and reconstructed with measured structural precision and recall.

It does **not** establish that the representation captures a person's meaning, that a neural decoder can produce the graph, or that a brain-mediated channel can carry it.

## Representation

The adapter uses two independently framed channels.

### Node channel

Each canonical node atom is encoded as:

`ROLE_NODE ⊗ NODE_ATOM`

The node atoms are keyed by:

`ConceptKind + sorted grounded_by identifiers`

Lexical labels and transport-local node identifiers are intentionally excluded.

The resulting node vectors are bundled in continuous space and then passed through the existing codec:

`ContinuousHV:f32 -> sign(value > 0) -> BinaryHV:bits`

### Edge channel

Each directed edge is encoded compositionally as:

`ROLE_SOURCE ⊗ SOURCE_ATOM ⊗ ROLE_RELATION ⊗ RELATION_ATOM ⊗ ROLE_TARGET ⊗ TARGET_ATOM`

The edge vectors are bundled in continuous space and then quantized by the same codec.

Keeping nodes and edges in separate HDC frames avoids relying on even-cardinality binary majority bundling across unrelated semantic roles.

## Codebook provenance

The codebook is not an implicit global mutable dictionary.

It is constructed from a training graph manifest and has a versioned descriptor containing:

- codebook algorithm revision;
- role-set revision;
- deterministic seed;
- HDC dimension;
- node-atom manifest hash;
- relation-atom manifest hash;
- training structural-manifest hash.

The atom vectors are derived deterministically from:

`seed + domain + canonical atom key`

using BLAKE3 to derive the seed for the existing deterministic `ContinuousHV::random` generator.

The codebook hash is the content hash of this descriptor. A receiver must reject a representation whose descriptor or codebook hash differs from the active codebook.

This is deliberately fail-closed: the decoder may not silently extend or substitute the codebook during evaluation.

## Held-out evaluation

The N0 lab constructs the codebook from four training graphs and evaluates two held-out graphs.

The held-out graphs reuse only atoms already present in the training codebook but combine them into graph configurations that were not present as complete training graphs.

For each held-out case the lab measures:

- node precision and recall;
- edge precision and recall;
- structural equivalence;
- minimum selected retrieval score;
- selection margin over the best unselected candidate;
- confidence MAE;
- expected graph size vs encoded representation size.

The benchmark also verifies that reordering graph collections produces an identical binary representation.

## Negative controls

The benchmark includes:

1. an unrelated random HDC query against the node and edge candidate spaces;
2. a directed-edge role-swap control;
3. a codebook mismatch control using a different deterministic seed.

The first two quantify accidental retrieval. The third checks that provenance mismatch is rejected before interpretation rather than producing plausible-looking output in the wrong semantic coordinate system.

## Evidence boundary

N0 is a representation and retrieval experiment.

It does not test:

- neural recording;
- neural decoding;
- semantic understanding;
- preservation of subjective or intended meaning;
- human participants;
- private-thought inference.

The correct progression is therefore:

`neural activity -> validated decoder -> grounded concept graph -> HDC interlingua -> authorized transport`

not:

`neural activity -> HDC packet -> assumed meaning`

A future N1/N2 evaluation must independently demonstrate the quality of the upstream decoder and then evaluate whether the grounded representation remains predictive on held-out data.

## Why this architecture

HDC/VSA systems conventionally obtain structured representations by assigning high-dimensional vectors to symbols and combining them with binding, bundling, permutation, and associative retrieval. Graph encoders similarly represent vertices/edges and combine their hypervectors into graph-level representations.

The Symthaea adapter follows that established representational pattern but keeps codebook provenance explicit and treats continuous-to-binary quantization as a separately measured transport boundary.

## Reproducibility

All benchmark cases use deterministic seeds and a fixed codebook descriptor. CI archives the machine-readable N0 report with the communication evidence bundle.

A future promotion should additionally require:

- preregistered datasets and splits;
- held-out identities where human data are used;
- independent replication;
- calibration and abstention reporting;
- negative controls against leakage;
- explicit data-class and inference-class consent;
- versioned hashes for model, code, codebook and split manifests.