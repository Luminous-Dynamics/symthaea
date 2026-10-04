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
