# Neurosemantic HDC Grounded Identity N0

Status: experimental / synthetic representation evidence only

## Purpose

This tranche extends the closed-world N0 interlingua without redefining it.

The earlier adapter keyed HDC node atoms from:

`ConceptKind + sorted grounded_by identifiers`

That is deterministic and leakage-resistant, but it makes an observation identifier part of the representation identity. A new observation of an otherwise equivalent concept therefore becomes out-of-vocabulary.

The grounded-identity adapter separates:

1. **stable concept identity** — an opaque identifier inside an explicitly declared concept scheme;
2. **stable relation identity** — an opaque predicate identifier inside that same scheme;
3. **grounding provenance** — observation/unit/context identifiers associated with the local manifestation;
4. **HDC retrieval identity** — deterministic vectors derived from stable identities.

The central invariant is:

> A grounding observation may change without changing the stable HDC atom, but HDC must never invent or validate the mapping from an observation to a stable concept.

## Identity manifest

An `HdcOntologyManifest` is supplied by an upstream identity/ontology adapter.

It contains:

- a versioned `scheme_id`;
- a `mapping_provenance_hash`;
- local node -> stable concept bindings;
- local relation -> stable predicate bindings;
- concept kind declarations;
- grounding references.

The manifest hash is canonicalized over collection order, so deterministic reordering does not create a new identity artifact.

The `mapping_provenance_hash` is intentionally not interpreted by the HDC layer. It is a pointer to the separately validated authority that established the identity mapping.

This follows the useful separation represented by W3C PROV: an entity's identity and its provenance/derivation are related but distinct concerns. It also matches the SKOS model in which a concept has an identifier and may have multiple labels, including labels in different languages.

## Stable concept encoding

Node atom vectors are derived from:

`seed + "concept" + stable concept_id`

not from lexical labels, local node identifiers, or grounding identifiers.

Therefore:

`concept:agent/sender`

can be represented by the same HDC atom when one observation is grounded by `train-agent` and another by `heldout-agent`.

The grounding references remain in the source manifest and are restored only through that sidecar. Decoding now requires the exact source manifest whose canonical hash equals `source_manifest_hash`; the receiver manifest supplies only the receiver's local node/relation identities. Thus receiver-local remapping cannot silently replace the provenance attached to the transmitted representation.


This is not semantic equivalence discovery. A French label, an English label, an audio unit, and a sensor-derived representation become equivalent here only when an independently validated upstream mapping assigns them the same stable concept identity.

## Stable predicate encoding

Relations receive the same treatment.

A local relation such as `initiates`, `commence`, or another modality-specific label may map to the same stable predicate ID.

The HDC edge representation is then constructed from stable source concept ID, stable relation ID, and stable target concept ID, with the existing fixed target permutation preserving directedness.

## Compatibility and provenance

The codebook descriptor binds:

- adapter and algorithm revision;
- role revision;
- deterministic seed;
- HDC dimension;
- identity scheme;
- identity-mapping authority provenance hash;
- stable concept manifest hash;
- stable predicate manifest hash;
- training graph structural manifest hash.

The HDC representation additionally carries the exact sender `source_manifest_hash`.

These are deliberately different:

`codebook compatibility != grounding provenance`

A new observation can therefore use the same HDC coordinate system while carrying a different provenance manifest.

The authenticated transport boundary must protect both the representation and the source sidecar manifest referenced by `source_manifest_hash`. The decoder's equality check proves that it received the expected sidecar; it does not itself provide transport authentication, confidentiality, signatures, or revocation.

A codebook is therefore tied not just to a scheme label but to the versioned identity-mapping authority that supplied that scheme. A manifest carrying the same scheme ID but a different mapping-authority provenance hash is rejected before encoding or decoding.


## Open-world boundary

A frozen codebook is still an explicit operating domain.

Encoding a stable concept absent from the frozen codebook fails rather than synthesizing a nearest atom.

Decoding likewise uses an explicit minimum-score and minimum-margin policy. An input that does not clear those gates must abstain rather than being forced into the closest known identity.

This is important because nearest-neighbour retrieval always has a winner even for unrelated input.

## Receiver-side ambiguity

A receiver manifest must provide exactly one local node binding for each selected stable concept in the graph being reconstructed.

Multiple bindings for one selected concept are rejected rather than choosing an arbitrary grounding. That prevents the HDC decoder from silently turning provenance ambiguity into an identity decision.

A larger corpus may contain many observations of the same concept, but that corpus-level representation should be normalized into a graph-specific or observation-specific sidecar before decoding.

Receiver-side stable predicate rendering follows the same fail-closed rule:
the current representation carries the stable relation ID, so the receiver
manifest must resolve each selected predicate ID to exactly one local relation
label. Multiple local labels may be valid upstream synonyms, but without an
explicit preferred-rendering field the HDC representation cannot safely choose
one; ambiguous receiver predicate mappings are therefore rejected rather than
silently selecting a label.

## Composition/generalization matrix

A second executable N0 lab evaluates the frozen identity codebook under increasing
composition load rather than testing only one held-out graph.

The matrix uses eight cases:

| Case | Nodes | Edges | Distractor concepts | Stable edge triples seen during training |
| --- | ---: | ---: | ---: | ---: |
| load-3x2 | 3 | 2 | 7 | 0% |
| load-5x4 | 5 | 4 | 5 | 0% |
| load-7x6 | 7 | 6 | 3 | 0% |
| load-9x8 | 9 | 8 | 1 | 0% |
| topology-out-star | 7 | 6 | 3 | 0% |
| topology-in-star | 7 | 6 | 3 | 0% |
| topology-merge-branch | 7 | 6 | 3 | 0% |
| topology-cycle | 7 | 6 | 3 | 0% |

Every atom required by the held-out graphs is known to the frozen codebook, but
the complete stable source/relation/target edge triples are held out. This
distinguishes compositional reconstruction from memorizing previously observed
triples.

The decoder must clear the same conservative score/margin policy at every load
point. The benchmark therefore reports the worst node and edge selection
margins in addition to exact identity and structural metrics. A future
promotion can tighten this matrix further by increasing the number of concepts,
graph density, predicate inventory, and distractor population rather than
altering the acceptance rule after seeing outcomes.

The benchmark remains an N0 synthetic representation test. Passing it does not
demonstrate that an upstream neural decoder has correctly inferred a person's
meaning.

## Identity-confusion adversarial sweep


The `neurosemantic_hdc_grounded_identity_adversarial_n0` lab applies deterministic
random bit corruption independently to the node and edge HDC frames at:

`0%, 2%, 5%, 10%, 20%, 30%, 40%, 50%`

The acceptance rule is deliberately asymmetric:

- a corrupted frame may remain accepted only when the decoded stable identities
  are still exactly correct;
- a decoder error is treated as abstention;
- an accepted but incorrect stable identity is a hard failure.

This does not establish adversarial robustness in the cryptographic sense.
It is an N0 red-team boundary against a simpler and important failure mode:
random transport corruption turning into an apparently confident but incorrect
ontology identity.

The lab reports correct accepts, abstentions, and confident-wrong accepts
separately. CI requires zero confident-wrong accepts while retaining at least one
clean correct decode. The existing fixed `0.20 / 0.05` score/margin policy is not
relaxed to accommodate corruption.

A second deterministic sweep interpolates the clean binary node and edge frames
toward independent null vectors at clean weights from 1.0 down to 0.0, then
requantizes before decoding. This deliberately probes the neighborhood of the
open decision boundary rather than sampling only independent random flips. A
successful decode must still recover the exact stable identity/topology; any
incorrect acceptance is a hard failure and otherwise the decoder must abstain.

## Empirical calibration boundary


`HdcOntologyEmpiricalCalibration` provides a conservative empirical calibration
layer over the existing fixed decoder policy. It consumes only successful clean
calibration metrics and derives lower-quantile floors for selected score and
selection margin.

The calibrated policy is:

`calibrated threshold = max(existing conservative baseline, empirical clean floor)`

The calibration artifact is also bound to exactly one frozen codebook hash. Clean
metrics from a different decoder codebook cannot be mixed into the same calibration
artifact, even though mixing them would tend to make the resulting thresholds
more conservative. This keeps the empirical measurement attached to the exact
representation space it is meant to characterize.

Therefore calibration cannot silently weaken the current `0.20` score or `0.05`
margin boundaries.

The executable calibration N0 lab keeps the calibration and evaluation cases
disjoint, verifies exact reconstruction on the evaluation split, and separately
runs 4096 deterministic random binary null frames. CI requires every null sample
to abstain. The null generator is deliberately in the binary transport space so
this sweep measures the decoder's rejection boundary rather than conflating it
with continuous-to-binary quantization behavior.

This is intentionally **not** conformal prediction. The calibration layer makes
no distribution-free coverage, false-accept, or adversarial-robustness guarantee.
A future higher evidence tier can introduce a formally specified conformity score
and calibration/test protocol without changing the current fail-closed baseline.

## Reconstruction resource boundary

The identity-aware decoder has an explicit N0 resource ceiling independent of
the 1 MiB transport payload bound. A representation may request at most 256
decoded nodes and 2048 decoded edges. Before allocating edge candidates, the
decoder also checks the Cartesian candidate budget and fails closed above
1,000,000 candidates. This prevents a small authenticated frame from inducing
unbounded receiver-side combinatorial work.

These are implementation safety bounds, not evidence of scalability. Raising
them should require a new benchmark tranche that measures runtime, memory, and
retrieval margins at the larger operating point rather than silently widening
the accepted domain.

## Evidence execution provenance

Communication Evidence Gates checks that pull-request execution is performed
against the exact immutable PR head. The evidence bundle records both the
GitHub event SHA and the checked-out revision, and CI requires the checked-out
revision to equal the PR head on pull-request events. The uploaded artifact is
named from that exact PR head rather than from GitHub's synthetic merge SHA.

This separates three things that must not be conflated:

- the event's synthetic merge reference;
- the exact source revision being qualified;
- the archived evidence bundle produced by that execution.

## Evidence ladder

The accompanying executable N0 lab demonstrates:

- a codebook built from stable identities;
- held-out grounding identifiers with no codebook rebuild;
- changed local/lexical labels with the same stable identities;
- exact stable concept and predicate identity recovery under clean retrieval;
- unchanged HDC frames when only local grounding changes;
- deterministic rejection of an OOV stable identity;
- deterministic rejection of an incompatible identity scheme;
- deterministic rejection of an incompatible identity-authority revision;
- deterministic rejection of a tampered source manifest hash;
- deterministic rejection of an ambiguous receiver manifest.

The benchmark remains below neural decoding and semantic-understanding claims:

`upstream decoder -> grounded concept graph -> validated identity mapping -> HDC interlingua -> authorized transport`

not:

`neural activity -> HDC vector -> assumed concept meaning`

The HDC layer therefore demonstrates a representation contract, not a mind-reading capability.

## Relationship to the original N0

The original `symthaea.hdc.semantic-interlingua-v1` remains a useful closed-world baseline.

It answers:

> Can a fixed training-derived codebook preserve structure when the atomic grounding vocabulary itself is held fixed?

The grounded-identity adapter answers a different question:

> Can the same stable identity coordinates survive changes in observation grounding and local labels when an independent identity mapping says they refer to the same declared concept?

Keeping these as separate adapters prevents a benchmark improvement from being mistaken for a retroactive strengthening of the original evidence claim.

## External conceptual references

- W3C PROV-O: provenance distinguishes entities, activities, agents, and derivation relationships.
- W3C SKOS: concepts have stable identifiers and may have multiple labels; mappings between concept schemes should remain explicit.
- Sandbrink & Young (2026), *Advancing data protections for implantable brain-computer interfaces*: data protections should distinguish raw recordings, processed features, decoded inferences, and personalized models, with purpose and inference sensitivity.
- 2026 iBCI consent research further supports treating authorization as a lifecycle rather than a single undifferentiated permission.

No external ontology, RDF stack, or semantic authority is introduced by this tranche.
