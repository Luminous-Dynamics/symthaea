# Policy Governance Protocol

**Campaign XII — 2026-07-20**

## Purpose

Post-deployment learning creates a long-lived governance problem that cannot be solved by a one-time promotion gate. A learned controller may later be implicated in an incident, transferred to a different machine, superseded, quarantined across a fleet, or retired permanently. This protocol defines the bounded lifecycle and evidence rules for those transitions.

## Authority order

Policy governance never outranks physical safety. The effective ordering remains:

1. malformed hardware and sensor rejection;
2. physical hazards and return-reserve protection;
3. operator holds and recovery locks;
4. survivability envelopes and actuator isolation;
5. runtime invariants;
6. policy lifecycle authority;
7. baseline or canary nominal command.

A policy record can remove learned authority. It cannot authorize a command that any earlier layer rejected.

## Lifecycle

Each controller checkpoint receives a stable candidate identity and one lifecycle state:

- `Shadow`: evidence collection and training only;
- `Canary`: bounded nominal authority under the existing canary rollback gate;
- `Active`: current learned policy;
- `Quarantined`: no authority pending investigation or requalification;
- `Retired`: terminal state that cannot be reopened.

Activation of a new policy quarantines the previously active policy. A fail-closed registry may temporarily contain no active learned policy; the embodiment then retains baseline authority rather than selecting another learned policy implicitly.

## Incident response

Policy incidents are bounded, deployment-bound, and classified by severity and kind. Safety incidents immediately quarantine the implicated canary or active policy. Repeated safety incidents recommend fleet quarantine and retirement review. Incident evidence records policy identity, deployment identity, step, severity, cause, and resolution state.

## Fleet quarantine

Fleet bulletins require:

- externally established authentication;
- nonzero issuer, epoch, and sequence;
- bounded issuance and expiry steps;
- an incident-evidence digest;
- independent hardware-backed fleet safety and verification approvals;
- exact binding to policy identity and requested action.

The crate does not implement transport cryptography. It independently rejects stale epochs, replayed sequences, duplicate approvers, approval mismatches, and unauthenticated assertions.

## Policy transfer

A policy may be exported only from a locally originated active record. Transfer evidence binds:

- source and target deployment identities;
- controller architecture and dimensions;
- state schema;
- hardware family;
- calibration profile;
- candidate identity and epoch;
- source evidence digest;
- independent source-safety and receiving-verification approvals.

A valid transfer enters the receiving registry as `Shadow`. It never receives direct active authority and must repeat local non-regression, promotion, and canary gates.

## Retirement

Retirement is irreversible. It requires independent hardware-backed safety and governance approvals bound to one ceremony, policy identity, rationale, and incident digest. An active policy must first be replaced or quarantined; retirement cannot silently leave a moving platform without an explicit fallback.

## Evidence retention

Evidence classes have bounded record limits and minimum retention intervals:

- operational;
- learning;
- incident;
- certification.

Compacted evidence may be pruned only after its minimum interval. Legal-hold evidence is never pruned. Certification evidence cannot be released from hold by this crate.

## Persistence

Operational checkpoint schema v5 persists the complete policy-governance supervisor. Invalid registry, incident, transfer, fleet-bulletin, or retention state is rejected before live mutation. Older checkpoints restore with a fresh local baseline registry and no imported or canary authority.

## Release requirements

Campaign XII adds:

- `SUB-GOV-001` — explicit lifecycle and terminal retirement;
- `SUB-GOV-002` — incident-triggered quarantine;
- `SUB-GOV-003` — replay-resistant fleet quarantine;
- `SUB-GOV-004` — shadow-only policy transfer;
- `SUB-EVD-002` — bounded governance-evidence retention.

## Explicit non-claims

This crate does not claim:

- cryptographic authentication of fleet messages;
- secure transport or key management;
- legal sufficiency of retention periods;
- statistical proof that a transferred policy generalizes;
- fleet-wide consensus;
- automatic regulatory approval;
- safe physical deployment without hardware qualification.
