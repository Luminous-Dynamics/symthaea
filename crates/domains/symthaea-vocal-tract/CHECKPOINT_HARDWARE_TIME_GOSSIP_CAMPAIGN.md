# Series 22 Hardware Custody, Trusted Time, and Durable Gossip Campaign

Status: preregistered engineering and deployment campaign

## Purpose

Series 21 established hybrid Ed25519 plus ML-DSA-65 public verification and
independent transparency-head gossip. Series 22 tests the next claims without
silently inferring them from valid signatures alone:

1. the counted hybrid signing keys were used through independently provisioned
   hardware-custody identities;
2. the publication time is bounded by multiple independent time authorities;
3. every authenticated gossip statement was delivered through multiple
   independently governed transport paths; and
4. every authenticated gossip statement received chained retention receipts
   from multiple independent archives.

The campaign does **not** treat a software-generated attestation as proof of HSM
certification. Promotion requires policy bindings and attestation keys that were
provisioned by the deployment authority from the intended hardware or isolated
signing service.

## Frozen artifacts

Before execution, publish and retain:

- the complete Series 21 public-verification bundle;
- the hardware-signing policy and all device records;
- the trusted-time policy and authority records;
- the gossip-transport policy;
- the gossip-archive policy;
- the verification Unix timestamp;
- the software-reference custody downgrade candidate;
- the stale trusted-time candidate; and
- SHA-256 or BLAKE3 digests of the complete Series 22 portable bundle.

Policy identifiers, public keys, organization bindings, device identities,
service bindings, repository bindings, network bindings, validity intervals,
counter floors, thresholds, and retention periods must be frozen before any
attestation or receipt is collected.

## Lane A: hybrid publication baseline

Run the complete Series 21 verifier first. The exact publication digest and
transparency anchor produced by that verifier become immutable inputs to every
Series 22 lane. Series 22 may not substitute another publication, hybrid policy,
transparency head, observer set, downgrade candidate, or split-view candidate.

Required result:

- every required Series 21 gate passes;
- the publication digest is nonzero;
- the gossip anchor digest is nonzero; and
- the Series 21 bundle and summary remain byte-for-byte unchanged while Series
  22 evidence is assembled.

## Lane B: hardware-custody evidence

Provision at least two signing devices controlled by at least two independent
organizations. Each device record binds:

- classical Ed25519 key ID;
- ML-DSA-65 key ID;
- separate attestation key ID;
- hardware model digest;
- firmware-policy digest;
- organization binding;
- minimum security level;
- minimum accepted signing counter; and
- validity interval.

For each counted hybrid endorsement, obtain an attestation over the exact
hybrid publication digest containing:

- device identity;
- classical and PQ key identities;
- measured boot digest;
- firmware and model digests;
- signing counter;
- issue time; and
- unique nonce.

Promotion requirements:

- at least two authenticated device attestations;
- at least two unique devices and organizations;
- policy security level `HardwareSecurityModule`;
- every signing counter meets its independently frozen floor; and
- no nonce or device identity is duplicated.

The campaign report must identify whether each provider was a software
reference, TEE, secure element, or HSM. A software reference is useful only for
contract tests and cannot satisfy promotion.

### Hardware negative

Construct a complete, otherwise-valid custody bundle whose policy explicitly
requires only `SoftwareReference`. The Series 22 verifier must validate the
candidate as internally authentic and then reject it for promotion as a custody
downgrade. A malformed candidate does not satisfy this negative.

## Lane C: externally witnessed time

Use at least two time services operated by at least two independent
organizations. Each statement binds the exact Series 21 publication digest,
policy digest, authority ID, witnessed time, uncertainty, issue time, and unique
nonce.

The verifier intersects all authenticated time intervals. Promotion requires:

- the authority threshold is met;
- at least two organizations and service bindings are represented;
- the verification timestamp lies inside the intersection;
- the intersection width is at most 10 seconds; and
- every statement is within the preregistered maximum age.

This is a bounded multi-authority time claim. It is not automatically a
hardware-rooted trusted-time claim; deployment documentation must state how each
time authority protects its clock and signing key.

### Stale-time negative

Create a time bundle that passes at its original issue time. Preserve the exact
signed statements until they exceed the maximum statement age. At the Series 22
verification timestamp, the verifier must reject the same otherwise-valid
bundle specifically because it is stale. Non-overlapping, unsigned, or malformed
statements do not satisfy this negative.

## Lane D: independent gossip transport

For every authenticated Series 21 gossip statement, collect delivery receipts
from at least two transports with distinct transport IDs, organizations, and
network bindings. Each receipt binds:

- the exact gossip-statement digest;
- origin ID;
- unique delivery ID;
- source and destination endpoint bindings;
- receive and delivery times; and
- the transport policy digest.

Promotion requirements:

- every statement has the required receipt quorum;
- no delivery ID is reused;
- transport signatures verify;
- origin IDs match the source statements;
- source and destination bindings differ;
- at least two organizations and network bindings participate; and
- the maximum authenticated delivery duration is no more than 60 seconds.

A single operator exposing two hostnames or two processes does not satisfy the
independent-network or independent-organization claims.

## Lane E: durable independent gossip archives

For every authenticated Series 21 gossip statement, collect receipts from at
least two independent archives. Each archive maintains its own append-only
receipt sequence. A receipt binds:

- the exact statement digest;
- archive and repository identity;
- sequence number;
- previous receipt digest;
- storage time; and
- retention-until time.

Promotion requirements:

- every statement has the required archive quorum;
- each archive chain starts at sequence one and zero predecessor;
- later receipts increment sequence by one and link the exact prior digest;
- no archive duplicates a receipt for one statement;
- at least two archive organizations participate; and
- the minimum retention time extends beyond the authenticated publication-time
  interval by the preregistered retention duration.

The code verifies authenticated retention commitments, not continuing physical
possession. Operational policy must periodically re-audit the repositories.

## Portable bundle and evaluator

The final artifact is encoded with
`symthaea.checkpoint-series22-public-verification-bundle.v1` and contains only
public verification material. Run:

```text
cargo run -p symthaea-vocal-tract \
  --example checkpoint_series22_hardware_time_archive_verifier -- \
  series22-public-bundle.postcard <verification-unix-seconds>
```

The evaluator must rerun Series 21 verification, the hardware positive and
software-custody negative, the trusted-time positive and stale negative, every
transport receipt, and every archive receipt. It then derives operational
evidence V12. No Series 22 metric may be supplied manually.

## Required negative controls

At minimum, execute and retain evidence for:

- wrong attestation key;
- wrong classical or ML-DSA key binding;
- firmware-policy digest mismatch;
- repeated device nonce;
- signing counter below the frozen floor;
- software-reference custody downgrade;
- wrong trusted-time subject digest;
- stale trusted-time statements;
- non-overlapping time intervals;
- duplicate time authority or service;
- delivery receipt for the wrong statement or origin;
- reused delivery ID;
- transport duration above the bound;
- missing transport receipt for one statement;
- archive receipt for the wrong statement;
- archive sequence gap, fork, or nonzero initial predecessor;
- archive retention below the policy floor; and
- missing archive receipt for one statement.

## Promotion gates

Series 22 promotion requires all V12 gates to pass:

- hardware custody policy;
- hardware attestation quorum;
- hardware organization diversity;
- HSM-level policy;
- signing-counter floors;
- custody downgrade rejection;
- trusted-time statement, organization, and service quorums;
- bounded trusted-time consensus interval;
- stale-time rejection;
- archive and archive-organization quorums;
- archive retention;
- transport, transport-organization, and network quorums; and
- bounded gossip delivery duration.

Missing evidence is `not_exercised`, not pass. A failure in any Series 21 gate
also blocks Series 22 promotion.

## Claims deliberately excluded

Series 22 does not claim:

- certification of any HSM or secure element;
- resistance to a compromised device attestation firmware;
- hardware-rooted or consensus time unless separately demonstrated;
- continuing archive possession after receipt issuance;
- anonymity or metadata confidentiality for gossip transport;
- availability under global network partition;
- distributed threshold ML-DSA; or
- independent execution of this campaign until real operators run it.
