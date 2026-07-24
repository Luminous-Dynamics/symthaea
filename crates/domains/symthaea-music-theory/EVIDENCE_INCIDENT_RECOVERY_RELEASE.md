# Publication Incident Containment and Recovery

## Purpose

This release layer provides a conservative response path when publication-catalog continuity evidence shows rollback, equivocation, or a fork and the normal outgoing witness quorum cannot safely authorize its own replacement.

It separates four claims that must not be collapsed:

1. **Incident evidence** — what signed checkpoint statements contradict one another.
2. **Containment** — which identities external governance has temporarily excluded from witness or observer evaluation.
3. **Recovery authorization** — which branch an external recovery authority and the incoming witness quorum jointly authorize as the starting point for a new trust segment.
4. **Portable response evidence** — one package binding the authorized recovery bundle to the recovered witness-policy anchor.

None of these claims proves intent, guilt, universal canonicality, absence of hidden forks, or independent control of signing keys.

## Incident reports

`CalibrationPublicationIncidentReport` is derived from an authenticated Series-19 continuity bundle containing explicit gossip conflict proofs.

The report distinguishes:

- **Direct signer contradiction** — one authenticated signer reports incompatible observations, allowing the key identity itself to be directly implicated.
- **Rollback observed** — one signer reports a lower catalog state after a higher state.
- **Branch conflict observed** — authenticated observers report incompatible catalog branches, but the observers are not automatically blamed for the authority-level conflict they revealed.

Every finding embeds the original conflict proof and derives its signer, observer, checkpoint, and witness-policy identities from that proof. Recomputing the finding or report SHA-256 cannot make altered attribution valid.

## Governed quarantine

`CalibrationPublicationQuarantineLedger` is an append-only, externally authenticated containment ledger.

A quarantine entry identifies:

- The affected key identity.
- Witness, observer, or combined scope.
- The exact incident report.
- Logical effective and optional expiry epochs.
- The externally governed reason.

A release is a new signed ledger decision; earlier containment history remains visible. Expired or released entries are excluded from the active evaluation.

Quarantine is a containment decision, not a declaration of fault. A reporting observer should not be quarantined merely because it exposed a branch conflict. Direct contradiction, separately established key compromise, quorum loss, or external administrative governance may justify containment according to the operator's policy.

## Recovery authorization

Normal witness-policy rotation requires the outgoing and incoming quorums. That path may become unavailable when the outgoing policy is compromised or unable to reach quorum.

`CalibrationPublicationRecoveryBundle` therefore binds:

- The exact authenticated incident report.
- The exact authenticated quarantine ledger.
- A disputed witness-policy epoch affected by the incident.
- An explicit catalog lineage beginning at a checkpoint in the incident history.
- The selected terminal checkpoint.
- A new incoming witness policy.
- A predeclared external recovery-authority policy.
- Distinct recovery-authority signatures.
- Distinct incoming-witness signatures.

Directly contradictory observer identities must be actively contained at the recovery epoch. Quarantined witness identities cannot appear in the incoming policy.

The resulting authorization means only:

> The configured external recovery authority and incoming witness quorum authorized this exact lineage and checkpoint as the beginning of a new witness-trust segment.

It does not prove that the selected branch was globally canonical or that withheld forks do not exist.

## Recovered policy anchor

`CalibrationPublicationRecoveredPolicyAnchor` creates a new witness-policy genesis bound to the exact recovery bundle and selected terminal checkpoint.

The recovered policy ledger intentionally starts a new trust segment. It does not rewrite the disputed policy history or remove the incident report.

Fresh checkpoint witnessing should occur under the recovered policy before operators treat later publication heads as normally witnessed.

## Portable response package

`CalibrationPublicationIncidentResponsePackage` binds:

- The complete authorized recovery bundle.
- The recovered witness-policy anchor.
- Mandatory machine-readable trust limitations.
- One canonical package SHA-256.

The structural audit validates nested hashes, limitations, recovery semantics, and anchor continuity. The authenticated audit additionally re-runs every configured external verifier for checkpoint witnesses, policy rotations, gossip, quarantine governance, recovery authorities, and incoming witnesses.

## Logical time

All incident, containment, and recovery epochs are caller-supplied logical epochs. The crate does not read wall-clock time, establish freshness, or choose an authoritative epoch source.

Operators must define how epochs are issued, persisted, and compared.

## External trust boundary

The crate defines verifier traits and a shell-free process adapter. It does not:

- Select signature algorithms.
- Store private keys.
- Enroll recovery authorities, governance signers, observers, or witnesses.
- Prove that key IDs correspond to independent humans or organizations.
- Verify certificate chains, hardware attestations, or transparency-log freshness.
- Establish the legal or organizational legitimacy of a recovery decision.

## Recommended operator sequence

1. Verify the conflict-bearing continuity bundle.
2. Build and independently audit the incident report.
3. Obtain externally governed quarantine decisions for directly contradictory or separately compromised identities.
4. Evaluate whether the normal outgoing-quorum rotation path remains safe and available.
5. When normal rotation is unavailable, choose an explicit lineage anchored in the incident history.
6. Obtain recovery-authority and incoming-witness signatures over the exact recovery plan bytes.
7. Build and verify the recovery bundle.
8. Build the recovered witness-policy anchor.
9. Build and verify the portable incident-response package.
10. Publish fresh checkpoint witnesses under the recovered policy and preserve the original incident evidence.

## Command-line tools

- `evidence_publication_incident`
- `evidence_publication_quarantine`
- `evidence_publication_recovery`
- `evidence_publication_incident_response`

Each tool accepts or emits JSON persistence records. Signing and verification programs remain external.
