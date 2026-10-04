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

The grounding references remain in the manifest and are restored only through that sidecar.

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
- stable concept manifest hash;
- stable predicate manifest hash;
- training graph structural manifest hash.

The HDC representation additionally carries the exact sender `source_manifest_hash`.

These are deliberately different:

`codebook compatibility != grounding provenance`

A new observation can therefore use the same HDC coordinate system while carrying a different provenance manifest.

The authenticated transport boundary must protect both the representation and any sidecar manifest referenced by `source_manifest_hash`. The HDC adapter does not substitute for packet authentication, confidentiality, signatures, or revocation.

## Open-world boundary

A frozen codebook is still an explicit operating domain.

Encoding a stable concept absent from the frozen codebook fails rather than synthesizing a nearest atom.

Decoding likewise uses an explicit minimum-score and minimum-margin policy. An input that does not clear those gates must abstain rather than being forced into the closest known identity.

This is important because nearest-neighbour retrieval always has a winner even for unrelated input.

## Receiver-side ambiguity

A receiver manifest must provide exactly one local node binding for each selected stable concept in the graph being reconstructed.

Multiple bindings for one selected concept are rejected rather than choosing an arbitrary grounding. That prevents the HDC decoder from silently turning provenance ambiguity into an identity decision.

A larger corpus may contain many observations of the same concept, but that corpus-level representation should be normalized into a graph-specific or observation-specific sidecar before decoding.

## Evidence ladder

The accompanying executable N0 lab demonstrates:

- a codebook built from stable identities;
- held-out grounding identifiers with no codebook rebuild;
- changed local/lexical labels with the same stable identities;
- exact stable concept and predicate identity recovery under clean retrieval;
- unchanged HDC frames when only local grounding changes;
- deterministic rejection of an OOV stable identity;
- deterministic rejection of an incompatible identity scheme;
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
