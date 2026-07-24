# Publication Recovery Re-entry and Incident Closure

## Scope

This contract governs the transition from an exceptional publication recovery into a new operational trust segment. It separates three claims that must not be collapsed:

1. **Recovery authority continuity** — the keys allowed to authorize exceptional branch selection have an append-only, dual-quorum rotation history.
2. **Post-recovery re-entry** — the selected branch has advanced to a fresh checkpoint witnessed under the recovered witness policy, on one exact catalog lineage.
3. **Operational incident closure** — the active recovery authorities and recovered witnesses have jointly authorized an operational closure decision after re-entry.

None of these claims proves universal canonicality, absence of withheld forks, unique human control of keys, or permanent incident resolution.

## Recovery-authority rotation

`CalibrationPublicationRecoveryAuthorityLedger` records an ordered authority-policy history. Genesis binds one authority policy to one exact catalog checkpoint. Every later epoch requires:

- a checkpoint later than the previous activation;
- a policy different from the outgoing policy;
- signatures meeting the outgoing authority threshold;
- signatures meeting the incoming authority threshold;
- one canonical rotation payload and set SHA-256;
- append-only epoch and rotation ordering.

This prevents a new recovery-authority set from silently replacing the old set. It does not prove that the outgoing and incoming authorities are organizationally independent unless external governance imposes that requirement.

## Post-recovery certification

`CalibrationPublicationPostRecoveryCertification` binds:

- the complete authorized incident-response package;
- a continuity bundle beginning at the selected recovery checkpoint;
- the recovered witness-policy anchor;
- the recovery-authority policy history active at the new head;
- the selected and new head checkpoint identities;
- the exact number of additional catalog events;
- a minimum required advance;
- a logical certification epoch.

Authenticated acceptance requires all configured verifiers to accept the incident response, witness history, head witnesses, gossip, quarantine, recovery authorization, and recovery-authority rotations.

A certification is rejected when the new head is not fresh, the recovered policy does not match the selected recovery, authority activation is absent from the same lineage, catalog advance is below policy, or any persisted identity is altered.

## Operational closure

`CalibrationPublicationIncidentClosureBundle` is a separate decision after accepted re-entry. It binds:

- the accepted post-recovery certification;
- an append-only quarantine ledger extending the recovery containment history;
- a closure policy;
- the active recovery-authority epoch;
- the active recovered-witness policy epoch;
- active witness and observer quarantines at the closure epoch;
- signatures from the active recovery authorities;
- signatures from the recovered witnesses.

The closure policy may require stricter signer thresholds than the active policies and may forbid active witness or observer quarantines.

Operational closure means that configured governance actors consider the selected branch fit for resumed publication operations. It does not erase incident evidence, establish fault, prove all forks are known, or prevent later evidence from reopening the incident.

## External trust boundary

The crate defines canonical payloads and verification traits. It does not:

- choose signature algorithms;
- manage private keys;
- establish authority or witness enrollment;
- prove key-holder independence;
- provide trusted time;
- prove that no hidden branch exists;
- determine legal, employment, or personal culpability.

The command-line tools invoke external verifier programs without a shell and pass verifier requests over standard input.

## Operator sequence

1. Create or verify the recovery-authority genesis ledger.
2. Rotate recovery authorities with outgoing and incoming authorization when required.
3. Produce a new checkpoint after the selected recovery checkpoint.
4. Build exact catalog lineage and recovered-policy continuity evidence.
5. Build and authenticate the post-recovery certification.
6. Evaluate quarantine state at the intended closure epoch.
7. Create a closure policy and canonical closure plan.
8. Obtain recovery-authority and recovered-witness signatures over the plan.
9. Build and authenticate the closure bundle.
10. Preserve all incident, recovery, re-entry, quarantine, and closure artifacts.

## Tools

- `evidence_publication_recovery_authority`
- `evidence_publication_post_recovery`
- `evidence_publication_incident_closure`

## Release requirement

Compilation, formatting, Clippy, and execution of the complete test suite remain mandatory in the canonical development environment. Static source inspection alone is not sufficient for release acceptance.
