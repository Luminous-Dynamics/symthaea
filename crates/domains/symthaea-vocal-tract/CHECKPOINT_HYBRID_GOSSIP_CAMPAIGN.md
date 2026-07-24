# Checkpoint Hybrid Public Verification and Gossip Campaign

**Program:** Symthaea vocal-tract checkpoint evidence
**Series:** 21
**Preregistered:** 2026-07-20
**Status:** implementation complete; compilation and campaign execution pending in the full workspace

## Purpose

Series 20 made federation lifecycle evidence publicly verifiable with Ed25519,
an append-only transparency history, M-of-N authority endorsement, and independent
witness cosigning. Series 21 evaluates two remaining public-trust boundaries:

1. migration from classical-only signatures to a mandatory Ed25519 plus
   ML-DSA-65 verification policy; and
2. detection of transparency split views by observers obtaining signed heads
   from independently bound origins.

This campaign does not reinterpret a Series 20 signature as post-quantum evidence.
The Series 20 bundle must verify first. Series 21 adds a distinct hybrid overlay
whose post-quantum signatures cover the exact encoded Series 20 bundle.

## Frozen algorithms and dependencies

- Classical signature: Ed25519 through `ed25519-dalek = 2.2.0`.
- Post-quantum signature: ML-DSA-65, FIPS 204 category 3 parameter set, through
  `fips204 = 0.4.6`.
- Transcript digest and artifact identifiers: BLAKE3 with explicit application
  domains and length prefixes.
- Portable encoding: bounded `postcard` artifacts.

The `fips204` backend is a software reference implementation. This campaign does
not claim FIPS 140 validation, independent cryptographic audit, hardware key
protection, resistance to host compromise, or side-channel certification.

## Trust roles

The following roles must remain logically distinct:

- Series 20 federation and transparency authorities;
- hybrid publication signers;
- ML-DSA signing providers;
- transparency gossip observers;
- independently bound transparency origins; and
- the public verifier.

A software ML-DSA key is permitted only for the engineering lane. Production
signing may implement `CheckpointMlDsa65SigningProvider` using an HSM, KMS,
remote signing service, or isolated agent. Provider identity alone is not a
hardware-attestation claim.

## Hybrid migration policy

Each hybrid policy freezes:

- a nonzero policy identifier;
- at least two signer organizations;
- one Ed25519 and one ML-DSA-65 verifying key per signer;
- a quorum threshold;
- a policy validity interval; and
- an exact `hybrid_required_from_unix_seconds` transition time.

### Before the transition

Classical-only endorsements may verify for historical migration purposes.
Post-quantum endorsements, when present, must still be valid and bound to the
same signer and transcript.

### At and after the transition

Every counted endorsement must contain both signatures. A missing ML-DSA-65
signature is a downgrade failure, not a reduced-assurance pass. The campaign
must include a classical-only candidate that is reverified after the transition
and rejected specifically as `HybridDowngrade`.

### Positive hybrid lane

The positive lane must demonstrate:

- the embedded Series 20 bundle verifies;
- the hybrid policy is valid at the evaluation time;
- the exact Series 20 bundle digest is covered by both algorithms;
- at least two valid ML-DSA-65 signatures;
- at least two independent signer organizations; and
- no duplicate classical key, post-quantum key, or organization is counted.

## Transparency gossip policy

Each gossip policy freezes:

- the target transparency log identifier;
- the transparency authority key;
- at least two observer keys;
- a minimum observer quorum;
- a minimum organization count;
- a bounded statement age; and
- a policy validity interval.

Each observer statement binds:

- the policy digest;
- a nonzero origin identifier;
- a source binding for the independently hosted endpoint or mirror;
- the observer key;
- the exact authority-signed head digest; and
- the observation time.

Observer statements are authenticated before their heads participate in fork
classification.

## Gossip consistency lanes

### Equal-size heads

All valid heads for the same log and entry count must be byte-identical. Two
valid authority-signed heads with the same entry count and different digests are
a split view and must be rejected.

### Unequal-size heads

Every unequal-size observation must provide a valid append-only consistency
proof in the correct direction between the observation and the anchor head.
A proof for another log, reversed endpoints, missing entries, or altered roots
must fail.

### Positive gossip lane

The positive lane must demonstrate:

- at least two valid observer statements;
- at least two distinct origins and source bindings;
- at least two observer organizations;
- all equal-size heads match exactly; and
- every unequal-size head has a valid consistency path.

### Split-view negative lane

The portable Series 21 artifact must carry the actual conflicting candidate,
not a precomputed boolean. The public verifier reruns the candidate and accepts
the negative lane only when it fails specifically as `SplitViewDetected`.

## Portable public artifact

`CheckpointSeries21PublicVerificationBundle` contains:

- the positive hybrid bundle;
- the actual classical-only downgrade candidate;
- the transparency authority public key;
- the gossip policy;
- the positive gossip bundle; and
- the actual split-view candidate.

No federation MAC key, checkpoint encryption key, laboratory evidence key,
Ed25519 secret key, or ML-DSA secret key is included.

The strict evaluator must decode this bounded artifact and independently rerun:

1. Series 20 verification;
2. positive hybrid verification;
3. hybrid downgrade rejection;
4. positive gossip verification; and
5. split-view rejection.

## Required operational gates

Series 21 promotion requires all of the following:

- hybrid signature policy verified;
- at least two valid ML-DSA-65 signatures;
- at least two hybrid signer organizations;
- classical-only downgrade rejected after the cutoff;
- at least two gossip observations;
- at least two independently bound gossip origins;
- at least two gossip observer organizations;
- all gossip heads linked by equality or append-only consistency; and
- an authenticated equal-size split view rejected.

Missing lanes are `not_exercised`, not pass.

## Negative controls

The campaign must reject:

- zero, malformed, or wrong-length ML-DSA keys and signatures;
- mismatched Ed25519 and ML-DSA signer identities;
- duplicate signer keys or organizations;
- signatures over a different Series 20 bundle;
- classical-only endorsements after the hybrid cutoff;
- a post-quantum signature altered by one byte;
- stale observer statements;
- duplicate observer, origin, or source identities;
- observer statements signed by unknown keys;
- statements observed before the signed head was issued;
- same-size authority-signed heads with different roots;
- consistency proofs for a different log;
- reversed or incomplete consistency paths; and
- hand-built summaries substituted for the portable verification bundle.

## Promotion and non-claims

A passing Series 21 report supports the claim that the published lifecycle
artifact was verified under both Ed25519 and ML-DSA-65 after the migration
cutoff, and that a quorum of independent observers found a mutually consistent
transparency history while the verifier rejected a validly signed split-view
candidate.

It does not establish:

- hardware-rooted key custody;
- true distributed threshold signatures;
- post-quantum security of Ed25519 itself;
- secrecy of public artifacts;
- trusted or hardware-attested wall-clock time;
- independence of organizations beyond the authenticated bindings;
- global gossip coverage; or
- absence of a split view shown only to observers outside the campaign.
