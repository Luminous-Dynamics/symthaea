# EKM-052 — Joint Trusted Restart Context

## Purpose

EKM-051 can construct a read-only quarantine, but it accepts the trusted restart-anchor tracker and trusted verifier checkpoint as two independent trusted inputs. EKM-052 removes that call-site ambiguity by binding both into one immutable trust epoch before quarantine admission.

The central invariant is:

**a restart candidate must be reviewed against one exact deployment trust context, not an arbitrary pairing of independently sourced anchor and verifier checkpoints.**

## `JointTrustedRestartContextV1`

Capture binds:

- deployment identifier;
- trust-domain identifier;
- context commit cycle;
- complete cloned `RestartAnchorTrackerV1` state;
- anchor sequence;
- anchor statement digest;
- anchored EKM-044 validation-receipt digest;
- anchored epistemic capture cycle;
- complete cloned `TrustedRestartVerifierStateV1`;
- verifier trust-snapshot sequence;
- verifier profile digest;
- verifier provenance digest;
- a domain-separated BLAKE3 context digest.

The commit cycle may not predate the anchored epistemic capture cycle.

`verify_integrity()` independently checks that the retained tracker/verifier objects still match the summaries committed into the context and that the context digest is reproducible.

## Contextualized quarantine

`ContextualizedEpistemicRestartQuarantineV2::validate_and_construct(...)` accepts one `JointTrustedRestartContextV1` and:

1. verifies the joint context;
2. rejects review before the context commit cycle;
3. calls the complete EKM-051 read-only quarantine admission path using the context's private anchor/verifier snapshots;
4. requires EKM-051's observed prior anchor sequence to equal the context anchor sequence;
5. requires EKM-051's observed prior verifier sequence to equal the context verifier trust-snapshot sequence;
6. binds the EKM-051 quarantine-admission digest to the joint-context digest and review cycle under a second domain-separated digest.

## Migration boundary

A verifier identity, implementation, version or configuration change is still rejected by EKM-050. EKM-052 does not turn those changes into ordinary continuity. A future migration protocol must explicitly establish a new trusted joint context.

## Persistence / attestation boundary

EKM-052 is an in-memory integrity contract. It does not itself prove that the joint context came from rollback-resistant storage or a hardware/transparency/witness trust root.

Deployment integration can later protect the context digest using the same external trust families already modeled by EKM-046:

- protected monotonic storage;
- signed checkpoints;
- witnessed checkpoints;
- hardware attestation;
- transparency checkpoints.

That binding remains a separate qualified tranche.

## Authority boundary

EKM-052 does **not**:

- mutate the anchor tracker;
- mutate trusted verifier state;
- hydrate writable epistemic support;
- reconstruct a writable revision history;
- activate or swap live state;
- authorize evidence/belief/causal/world-model/action mutation;
- perform file/network I/O;
- sign or manage keys.

Both the joint context and contextualized quarantine remain read-only. The contextualized quarantine hard-codes:

- `trusted_context_mutated = false`
- `writable_hydration_authorized = false`
- `activation_authorized = false`

## Qualification status

This tranche is stacked on EKM-051. GitHub Actions remains the executable authority; queued or cancelled jobs are not format, compile, Clippy, test, or runtime evidence.
