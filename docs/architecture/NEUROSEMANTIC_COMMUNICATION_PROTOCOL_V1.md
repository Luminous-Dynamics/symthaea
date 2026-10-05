# Neurosemantic Communication Protocol v1

Status: experimental protocol infrastructure

This document adds a conservative exchange layer to Symthaea for derived cognitive
representations. It does not claim that present-day BCIs can decode unrestricted
thoughts or write arbitrary mental states.

## Core idea

Transport the representation independently of the mechanism that produced it:

cognitive representation -> authorized packet -> peer -> reconstructed representation

This allows ordinary text, HDC semantic states, silent-speech outputs, sensorimotor
representations, and future neural adapters to share one exchange contract without
making any modality the universal ontology.

## What the protocol adds

The neurosemantic module defines:

- six routing channels: semantic, affective, spatial, temporal, procedural, sensory;
- directional permissions for read and write;
- purpose binding;
- explicit time-bounded consent leases;
- revocation and consent epochs;
- content-addressed packets;
- explicit sensitivity labels with consent-enforced read/write ceilings;
- derived-neural-feature payloads without a raw-neural payload type.

The default authority model is deny. Legacy/deserialized leases without explicit sensitivity
ceilings default to **Public**, preventing silent escalation.

## Why consent is part of the protocol

A future semantic BCI creates a security boundary that ordinary networking does not have:
a person can be physically connected to a system while granting only a small subset of
what that connection is allowed to read or write.

A grant is therefore specific to:

peer + purpose + channel + direction + sensitivity ceiling + time + consent epoch

Direction is from the **subject's perspective**:
- **Read** means subject -> peer.
- **Write** means peer -> subject.

The packet endpoints must match that direction exactly. A packet cannot elevate its own authority.

## Integrity model

Each packet carries a BLAKE3 hash for the payload and a second hash over a canonical copy
of the packet with the packet hash blanked.

This is integrity addressing, not authentication or confidentiality. Deployment still needs
authenticated peers and encrypted transport. Mycelix and Xenia are natural candidates
for that outer security layer.

## Scope boundary

This module does not implement:

- neural decoding;
- neural stimulation;
- brain-to-brain hardware;
- unrestricted thought decoding;
- identity inference from neural data;
- autonomous access to private cognition;
- claims of paranormal telepathy.

Future adapters must remain behind the existing evidence-gated communication pipeline.

## Evidence ladder

N0: synthetic semantic exchange

The repository now includes an executable N0 harness at
`crates/core/symthaea-communication/examples/neurosemantic_n0.rs`.

It exercises deterministic graph serialization round trips plus independent checks for:
- packet integrity/tamper detection;
- consent authorization;
- channel scoping;
- lease expiry;
- exact replay detection;
- sequence collision rejection;
- capability-gated expression.

N0 deliberately measures protocol correctness, not semantic understanding. It must not be
presented as evidence that a system can read thoughts.

N1: silent-speech bridge

A validated speech BCI feeds a semantic adapter. Evidence should include participant
holdouts, utterance holdouts, calibration, error characterization, and preregistered
criteria.

A 2025 Cell study reported real-time decoding of imagined sentences from motor cortex in
four participants. This supports a research direction, not general mind reading.

N2: conceptual communication

The decoder works on conceptual representations instead of requiring word-level
articulation. A 2026 Nature Reviews Bioengineering perspective identifies conceptual
decoding as a next-generation direction for communication neuroprostheses.

N3: bidirectional semantic presentation

A receiver obtains a semantic representation through text, audio, visual, haptic, or
clinically validated neural presentation. Read/write permissions remain separate.

N4: mediated brain-to-brain semantic communication

Two participants exchange information through an end-to-end neural pipeline. The scientific
claim should be communication through a mediated physical information channel, not
paranormal telepathy.

N4 evidence should require independent replication, held-out participants and sites,
information-transfer metrics, calibration, error bounds, replay resistance, and explicit
consent.

## Privacy boundary

Treat these as separate data products:

raw neural data != derived neural features != semantic representation != decoded claim

Each transition needs its own provenance and access policy.

The design principle is:

Never give the semantic layer more information than it needs to perform the task.

This aligns with current BCI privacy work, which highlights gaps around de-identification,
individual control, consent, misuse guardrails, and ownership of neural data.

## Symthaea and Mycelix

Symthaea should own:

signal -> representation -> semantic state -> reconstruction

Mycelix/Xenia should own:

identity -> authorization -> authenticated transport -> revocation -> audit

Symthaea should not decide who is authorized to access a representation, and the trust
layer should not decide what a neural signal means.

## Research position

Current literature supports progression from speech BCIs toward semantic and conceptual
communication, including small-cohort inner-speech decoding. It does not establish
unrestricted private-thought decoding.

Accordingly, telepathy is best treated as a user-facing metaphor for a future mediated
communication experience. The engineering target is content-addressed, consent-bound
transmission and reconstruction of structured cognitive representations.


## N0 interlingua benchmark

The standalone crate also exposes a deterministic interlingua benchmark at
`crates/core/symthaea-communication/examples/neurosemantic_interlingua_n0.rs`.

It compares a fixed grounded concept graph against controlled transformations:
- exact round trip;
- collection reordering;
- transport-local identifier renaming;
- edge deletion;
- duplicate nodes;
- edge duplication;
- lexical relabeling;
- confidence drift.

The benchmark reports node/edge precision and recall, confidence mean absolute error,
content hashes, and serialized sizes. Structural equivalence deliberately ignores
collection ordering, lexical labels, and transport-local node identifiers while retaining
grounding, node kind, and relation structure.

This is a **protocol/interlingua preservation benchmark**, not a semantic-understanding
benchmark. In particular, N0 does not establish that an HDC, BCI, neural decoder, or any
other future adapter preserved a person's intended meaning. A future representation adapter
must provide its own encoder/decoder evidence and can then reuse this benchmark contract.

## Resource and version boundaries

Protocol v1 is pinned by `NEUROSEMANTIC_PROTOCOL_VERSION = 1`. Unknown protocol versions are
rejected rather than silently interpreted.

Neurosemantic payloads are bounded to 1 MiB before hashing. Derived neural features must be
finite. These limits are defensive defaults, not claims about the maximum useful payload for
future hardware.

The replay tracker is bounded to 4096 active sender/recipient/lease/consent-epoch keys. The
public `observe_authorized` path validates the consent lease before consuming replay-tracker
state; the lower-level `observe` function is intended for callers that already performed
authorization.

Authentication, confidentiality, lease signing, revocation distribution, and durable audit
remain deployment responsibilities for the outer identity/transport layers.

## HDC integration boundary

The current Symthaea HDC stack has separate continuous and binary representations.
The semantic encoder produces a 16,384-dimensional `ContinuousHV`, while the current
semantic decoder consumes `BinaryHV`. A `BinaryHV::from_bipolar` conversion exists, but
this is a quantization boundary and must be measured independently from semantic decoding.

Therefore a future HDC neurosemantic adapter must report at least:
- encoder configuration and seed;
- continuous representation fidelity;
- continuous -> binary quantization loss;
- decoder reconstruction fidelity;
- codebook/prototype hash;
- structural interlingua metrics;
- corruption robustness.

No one of these layers, alone, constitutes evidence of human thought decoding.

The N0 graph has two distinct hashes:
- the ordinary graph hash, which is serialization/content addressing;
- the structural hash, which is canonicalized across collection order and transport-local
  identifiers while excluding confidence, because confidence is reported separately.
## Opt-in HDC codec

The communication crate exposes an opt-in `hdc-codec` feature that reuses Symthaea's
existing `ContinuousHV` and `BinaryHV` types. The codec performs:

`ContinuousHV -> sign quantization -> BinaryHV -> ContinuousHV`

The N0 lab reports full-dimension cosine similarity and sign disagreement. The cosine
calculation is implemented locally rather than using the global cognitive-stride-aware
HDC similarity helper, making the evidence independent of ambient stride configuration.
The compact binary representation is 2,048 bytes for 16,384 dimensions versus 65,536 bytes
for f32 continuous storage, a 32x storage reduction before protocol framing.

The codec does not invoke the current semantic decoder and does not claim semantic
reversibility. Quantization loss is an explicit measured boundary for later adapter work.

The codec also emits a versioned provenance descriptor containing the codec identifier,
representation types, quantizer rule, dimension, and optional encoder/codebook identifiers.
This prevents a future receiver from silently treating vectors produced by a different
encoder revision or codebook as interchangeable.

The N0 lab additionally sweeps deterministic binary corruption at 0%, 0.1%, 1%, 5%, and
10% nominal bit-flip probability and records binary similarity plus full-dimension cosine
similarity to the original continuous vector. These are transport/representation metrics
only; they are not semantic-accuracy metrics.

## Deterministic HDC semantic interlingua N0

The opt-in HDC track now has a deterministic semantic adapter documented in
`docs/architecture/NEUROSEMANTIC_HDC_INTERLINGUA_N0.md` and executable as
`neurosemantic_hdc_interlingua_n0`.

The adapter uses a versioned training-derived codebook and two independently framed
HDC channels: a role-marked node bundle and a compositional directed-edge bundle. Each
channel is composed in continuous space and then crosses the already measured
`ContinuousHV -> sign -> BinaryHV` codec boundary.

The N0 lab builds the codebook from four synthetic training graphs and evaluates two
held-out graphs. The held-out cases reuse only atoms present in the training manifest,
but test unseen graph combinations. Retrieval uses the same codebook and reports node
and edge precision/recall, structural equivalence, confidence MAE, selected-score
margins, and representation size.

The lab also includes unrelated-vector, role-swap, collection-reordering, and codebook-
mismatch controls. A codebook descriptor or hash mismatch is rejected before semantic
interpretation, preventing silent coordinate-system substitution.

This remains synthetic representation/retrieval evidence. It is not neural decoding,
semantic-understanding evidence, or evidence of preserving subjective intent.


## Machine-readable data and inference policy

The protocol now separates three different properties that must not be conflated:

- **data class** — what kind of cognitive artifact exists (raw neural recording, derived neural feature, semantic representation, decoded claim, personalized decoder/model state);
- **inference classes** — what information the artifact may expose or enable (signal pattern, unit pattern, linguistic content, semantic content, affective state, intent, identity);
- **transport sensitivity** — the network/data-handling sensitivity label already enforced by the consent lease.

Each neurosemantic packet carries a versioned data policy. A policy must explicitly name its data class, at least one inference class, and at least one permitted communication purpose.

Consent leases separately authorize data classes and inference classes for read and write directions. Legacy leases deserialize those permissions as empty sets, so they cannot silently acquire access to newly introduced cognitive data or inference classes.

Policy-bearing packets also bind the declared data class to the payload's intrinsic type. Typed semantic graphs/hypervectors authorize only `SemanticRepresentation`; typed decoded claims authorize only `DecodedClaim`; typed derived features authorize only `DerivedNeuralFeature`. The legacy opaque `StructuredRepresentation` variant remains deserializable but cannot cross a policy authorization boundary because its data class cannot be verified from the payload type alone.

Recent iBCI governance work likewise distinguishes raw recordings, processed features, decoded inferences, and personalized model parameters, while identifying conflated consent and weak misuse guardrails as important gaps. (see Sandbrink & Young, *Communications Medicine*, 27 July 2026, DOI 10.1038/s43856-026-01797-y; Young et al., *Device*, available 11 August 2026, DOI 10.1016/j.device.2026.101271).

### Downstream handling sovereignty

The v2 data policy also declares downstream handling constraints separately from transport sensitivity:

- an origin jurisdiction and an explicit destination-jurisdiction allow-list;
- a retention policy (`Ephemeral` or an explicit Unix-time expiry);
- an explicit secondary-use allow-list, empty by default;
- a concrete handling action at the enforcement boundary (`Transmit`, `Persist`, or `SecondaryUse(...)`).

Destination identifiers currently use two-character uppercase jurisdiction codes (for example `ZA` and `GB`) as an interoperable policy identifier. This is a representation constraint, not a legal determination of which law applies.

A packet being valid and consent-authorized does not imply that it may be persisted indefinitely, transferred to another jurisdiction, used for model training, commercial analytics, behavioral profiling, affective inference, or identity inference. Those downstream actions require an explicit matching handling policy. For the two inference-sensitive secondary-use classes, AffectiveInference additionally requires AffectiveState to be declared in the packet's inference-class set, and IdentityInference additionally requires Identity. This prevents a downstream-use flag from escalating the inference capability beyond the declared data product. The handling check is intentionally an additional gate rather than a replacement for identity, signed-consent, revocation, audit, or legal/policy provenance managed outside this crate.

This design responds directly to current neurotechnology governance concerns about permissive secondary-use clauses, inference-sensitive misuse, and the need to preserve individual control over downstream processing. UNESCO's 2025 Recommendation emphasizes mental privacy, freedom of thought, consent, and restrictions on coercive or surveillance uses; 2026 iBCI governance work likewise highlights secondary-use and inference-level risks.


## Serialized artifact trust boundary

Untrusted wire or persisted JSON must enter the protocol through the bounded constructors
`CognitiveConsentLease::from_json_bytes`, `NeurosemanticPacket::from_json_bytes`, or
`AuthorizedNeurosemanticMessage::from_json_bytes`. Each rejects oversized raw JSON before
`serde_json` materialization and then re-runs the relevant semantic, integrity, and
authorization checks. The protocol currently caps serialized artifacts at 1 MiB and
constructor/validator identifiers at 4096 bytes.

Direct unbounded deserialization remains outside the protocol trust contract. A packet
that is merely integrity-valid is not thereby authorization-valid: unknown or legacy
data-policy state remains unable to cross the consent boundary.

### Revocation and replay lifecycle

Consent revocation carries an explicit effective timestamp. A revoked lease without an
effective timestamp is invalid, while a future effective timestamp permits an explicitly
scheduled revocation. Authorization checks deny access at or after the effective time.
The N0 evidence exercises both sides of a scheduled revocation boundary: access is accepted
before the effective timestamp and rejected at the effective timestamp.

Replay protection is bounded by consent epoch, sender/recipient, and lease identity. The
replay tracker records lease expiry with each retained sequence state and can reclaim
expired entries. The N0 evidence explicitly observes a packet, detects an exact replay,
then verifies that the expired replay state is reclaimed at lease expiry. This is important
because a fixed-capacity replay table without lifecycle reclamation would turn normal
short-lived leases into a permanent resource-exhaustion path.

### External policy provenance

The handling policy carries an opaque reference to the externally authoritative policy or
consent record and a BLAKE3-256 digest of the exact record bytes (or a separately specified
canonical form). Symthaea binds both values into the content-addressed packet policy and
can verify supplied record bytes against the stored digest. This proves record-byte
binding, not issuer identity or authority. Mycelix or another designated policy authority
must resolve and authenticate the reference, signed consent, revocation state, and audit
history at the system boundary.

The N0 evidence also checks the negative case: a different record digest is rejected even
after the packet is re-hashed consistently. This keeps the provenance check distinct from
packet integrity and prevents a resolver from treating an arbitrary record as equivalent
merely because the packet itself remains internally consistent.

Thus there are deliberately separate gates:

1. packet integrity;
2. consent and data/inference authorization;
3. downstream handling policy;
4. external identity/policy provenance and revocation authority.

Passing one gate does not imply passage through the others.

### External policy provenance hardening

The handling policy now carries two distinct machine-readable provenance fields: a reference identifying the externally authoritative policy/consent record, and a domain-separated BLAKE3-256 digest over the exact provenance reference and exact record bytes (or a separately specified canonical form). The binding prevents a digest valid for one reference from being presented under another reference. Symthaea exposes an executable `verify_policy_provenance_bytes(...)` check that verifies the exact supplied record bytes against the stored reference+record binding, with the same bounded-artifact ceiling used elsewhere. This validates the record binding, but it does not authenticate the issuing authority, signature, revocation status, or legal applicability; those remain responsibilities of Mycelix or another designated policy authority. Recomputing the packet or provenance digest is therefore not an authority proof.

Changing this binding is a schema change (v5), so v4 handling artifacts fail closed rather than being silently upgraded. The v5 digest construction is domain-separated and length-delimited to avoid ambiguous concatenation.

Downstream handling calls should receive the resulting provenance-binding token. The token also fingerprints the complete packet handling-policy state, so mutating destination, retention, secondary-use, jurisdiction, or provenance fields after verification invalidates the token rather than allowing a stale authorization capability to survive. The token is intentionally not an issuer credential: it proves that the exact supplied record matches the packet's declared reference and digest, while the external policy/identity authority remains responsible for authenticating who issued that record and whether it is current.
