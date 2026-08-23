# Symthaea Cognitive Interchange Protocol (SCIP) v1

Status: **experimental / draft protocol**

Crate: `crates/bridges/symthaea-interlingua`

SCIP is a versioned interchange layer for moving grounded cognitive state between
Symthaea and heterogeneous peers such as LLMs, local sequence models, symbolic
systems, neuromorphic systems, and future machine agents.

It does **not** replace natural language, Xenia, or Symthaea's internal HDC
substrate. It gives them a shared semantic boundary.

## Core invariant

`GroundedConceptGraph` is the canonical semantic object.

An HDC vector is a compact associative **projection** of that graph. It is not
self-authenticating evidence of meaning. A receiver may treat an HDC payload as
grounded only when its `semantic_hash` resolves to the grounded graph from which
the projection was created.

This preserves the existing `symthaea-communication` distinction between
observing structure and making grounded claims about reference or intent.

## v1 envelope

A `CognitiveEnvelope` carries:

- exact SCIP protocol version;
- content-addressed `message_id`;
- optional parent message;
- negotiated semantic/wire profile;
- grounded graph, HDC projection, structured JSON, text, or semantic reference;
- calibrated confidence;
- provider/model provenance;
- evidence identifiers.

The envelope validates its own content identity. Mutation without refreshing the
ID is rejected.

## HDC profile

The default profile uses:

- dimension: `symthaea_core::hdc::HDC_DIMENSION` (16,384);
- container/binding: `ContinuousHV` with Hadamard binding;
- atom representation: deterministic bipolar `{-1,+1}` values;
- atom derivation: BLAKE3 XOF over namespace/domain/value;
- bundling: deterministic mean;
- namespace: `symthaea.scip.v1`.

Bipolar atoms are derived independently of model weights and independently of
`ContinuousHV::random`, so two implementations can reconstruct the same role
and concept atoms from the profile contract.

The profile fingerprint covers dimension, algebra, atom derivation, and
namespace. Peers must match the fingerprint before selecting HDC.

### What is encoded

Node projections currently bind/bundle:

- record type;
- node ID;
- concept kind;
- optional label;
- quantized confidence;
- grounding references.

Edge projections currently bind/bundle:

- record type;
- source;
- relation;
- target;
- quantized confidence;
- evidence references.

The exact grounded graph hash remains attached to the HDC payload, so
quantization in the associative projection cannot alter canonical semantics.

## Negotiation

Peers advertise:

- protocol versions;
- supported representations;
- HDC profiles;
- sparse HDC-delta support;
- exact graph-delta support;
- semantic-reference support.

v1 requires an exact common protocol version.

Preference order is:

1. matching HDC profile;
2. grounded concept graph;
3. structured JSON;
4. natural language.

A peer that advertises HDC but has a different codebook/profile does **not**
guess or reinterpret the vector. It falls back to the next shared
representation.

Negotiation also records the complete ordered `shared_representations` set.
The first entry is a backward-compatible preferred representation, not a mandate
that every message use the same transfer form. Per-message transfer planning may
choose any negotiated representation whose prerequisites are satisfied.

Sparse HDC deltas and exact graph deltas are independent capabilities:

- `sparse_hdc_deltas` requires an exact shared HDC profile;
- `exact_graph_deltas` requires bilateral advertisement and at least one shared
  exact grounded representation (`GroundedGraph` or canonical `StructuredJson`).

Therefore an HDC-preferred session can still use exact graph deltas when it also
shares a grounded representation. Conversely, an HDC-only session cannot claim
exact semantic graph-delta reconstruction merely because it supports sparse
vector updates.

The optional `exact_graph_deltas` field uses a fail-closed compatibility rule:
an older capability advertisement that omits the field is interpreted as not
advertising exact graph-delta support.

## Persistent sessions

### Exact semantic graph synchronization

`GraphDelta` is an exact edit set from one canonical grounded graph to another.
Both endpoints are content-addressed:

- the receiver must possess the exact `base_semantic_hash` graph;
- applying the delta to any other base fails;
- the reconstructed graph must equal the declared `target_semantic_hash` before
  it is accepted;
- node removals/upserts and edge removals/additions are canonicalized for stable
  wire-size measurement.

Negotiating `exact_graph_deltas` does **not** mean every message should use a
delta. The transfer planner must compare the actual canonical delta bytes with
the complete grounded-graph bytes and choose the smaller exact representation.
A very broad semantic change may legitimately fall back to a full graph.

Controlled Broca-to-SCIP evidence currently shows exact graph deltas smaller than
the full graph in all 11 measured mutation classes, with localized/medium
changes at 6.57–16.29% of full-graph size and one broad multi-field transition at
72.88%. This is evidence for capability negotiation, not a universal compression
claim.

### Associative HDC synchronization

`SparseHdcDelta` can represent component changes relative to a known HDC
payload. A delta is valid only when:

- base semantic hash matches;
- profile fingerprint matches;
- dimensions match;
- indices are strictly increasing and in bounds;
- values are finite.

SCIP does not assume that HDC deltas are smaller. Semantic changes may alter most
dimensions. `DeltaMetrics` reports the changed fraction and estimated bytes so a
session can choose a dense frame whenever a delta is not economical.

Sparse HDC synchronization updates an associative projection; it is not a
substitute for exact grounded semantic reconstruction.

### Semantic references

Semantic references require the receiver to possess the grounded graph
identified by the referenced semantic hash. A reference is the smallest exact
semantic transfer when that target is already cached, but cache possession must
be established by the surrounding session/synchronization protocol rather than
assumed from the reference itself.

## LLM compatibility

Most hosted LLM APIs currently expose text/token interfaces rather than latent
state interfaces. `LlmTextFallback` therefore provides a compatibility path:

`SCIP envelope -> verified grounded graph -> compact JSON + grounding policy -> LLM`

For HDC or reference envelopes, text fallback fails unless the caller provides
the grounded graph whose hash matches the payload.

Two modes are defined:

- `GroundedReasoning`: the model may infer, but must distinguish new inference
  from supplied grounded facts.
- `FaithfulTranslation`: the model translates without adding facts or
  conclusions.

A future native model adapter can replace the text step without changing the
canonical semantic contract.

## Security and epistemic threat model

SCIP v1 explicitly defends against these classes of semantic failure:

### Codebook confusion

A vector from another HDC algebra/namespace must never be interpreted under the
local profile. Negotiation and profile fingerprints reject this.

### Ungrounded vector injection

A high-similarity or syntactically valid HDC vector cannot assert its own
meaning. Grounding requires the matching canonical semantic hash and graph.

### Profile downgrade

Peers choose the strongest mutually supported representation by deterministic
preference. Applications should log fallback/downgrade decisions.

### Stale reference

A semantic reference that cannot be resolved is an error, not permission to
guess its contents.

### Delta poisoning

Exact graph deltas are bound to both base and target semantic hashes. Sparse HDC
deltas are bound to base semantic hash, HDC profile, and dimension. Applying
either delta type to an incompatible base is rejected.

### Replay and peer authentication

SCIP itself is a semantic protocol, not a secure transport. Replay prevention,
peer authentication, confidentiality, authorization, and channel integrity
belong to the transport/session layer. For Symthaea/Mycelix deployment, Xenia is
the intended integration boundary.

A future Xenia binding should include SCIP `message_id`, protocol/profile
fingerprints, parent/session identifiers, and transport transcript identity in
the authenticated transcript.

## What v1 does not claim

SCIP v1 does **not** claim:

- that HDC communication is always smaller than text;
- that HDC deltas are always sparse;
- that exact graph deltas are always smaller than full grounded graphs;
- that an LLM can natively consume SCIP HDC vectors;
- that cosine similarity establishes semantic truth;
- lossless decoding of arbitrary graphs from a lone HDC vector;
- consciousness, phenomenology, or subjective understanding by a peer;
- transport security or authorization.

Those are separate empirical or systems questions.

## Required evaluation

Before describing SCIP as more efficient than text, measure at least:

1. semantic hash preservation;
2. grounding/evidence preservation;
3. epistemic-confidence preservation;
4. projection similarity under independent implementations;
5. canonical graph bytes;
6. dense HDC bytes;
7. exact graph-delta bytes and operation counts;
8. sparse HDC-delta bytes and changed fraction;
9. model-specific token counts;
10. encode/decode/adapter latency;
11. task accuracy with text vs structured vs HDC/latent adapters;
12. profile/version/capability mismatch rejection;
13. corrupted/stale reference and wrong-base delta rejection.

Token counts must be measured with each concrete model tokenizer.

## Integration roadmap

### Phase A — protocol foundation

Current work includes:

- SCIP v1 contracts;
- canonical semantic hashing;
- deterministic grounded HDC projection;
- profile and representation negotiation;
- semantic references;
- exact content-addressed graph deltas and explicit capability negotiation;
- sparse HDC deltas;
- current-API LLM text fallback;
- fidelity/wire metrics.

### Phase B — Symthaea language integration

Add a small adapter around `LLMOrgan` / `LLMBackend` that accepts a SCIP
envelope and uses `LlmTextFallback` for current providers. Do not make the
language module the canonical semantic substrate.

### Phase C — Xenia session binding

Define a transport-neutral SCIP frame codec and then bind it to authenticated
Xenia sessions. Keep cryptography, identities, replay controls, and capability
authorization in Xenia.

The session layer also needs explicit cache/base acknowledgement so semantic
references and exact graph deltas are sent only when the receiver has confirmed
possession of the required content-addressed graph.

### Phase D — native latent adapters

Experimentally learn mappings:

`model latent state <-> SCIP HDC space`

Evaluate against text and grounded-graph baselines. A native adapter ships only
if it improves measured cost/latency/task performance without degrading
grounding or epistemic calibration.

## Design principle

The goal is not to make every intelligence think like Symthaea.

The goal is to let heterogeneous systems preserve their native architectures
while sharing a grounded, negotiated cognitive language.
