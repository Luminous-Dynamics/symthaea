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
- explicit sensitivity labels;
- derived-neural-feature payloads without a raw-neural payload type.

The default authority model is deny.

## Why consent is part of the protocol

A future semantic BCI creates a security boundary that ordinary networking does not have:
a person can be physically connected to a system while granting only a small subset of
what that connection is allowed to read or write.

A grant is therefore specific to:

peer + purpose + channel + direction + time + consent epoch

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
