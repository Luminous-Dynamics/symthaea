# Fabrication Federated Certification

Version 0.14 introduces a federation boundary above the durable single-gateway authority model.

## Authority chain

1. A durable gateway state is sealed and cryptographically identified.
2. Independent gateways endorse that exact state generation, predecessor, commit time, and digest.
3. Quorum verification counts only lifecycle-eligible, cryptographically valid, distinct gateway votes.
4. A persistent consensus tracker rejects rollback, generation gaps, same-generation forks, and gateway equivocation.
5. Disaster-recovery bundles retain a contiguous chain of sealed states and matching quorum evidence.
6. Signed operator commands are bound to one manifest, machine, timed session, printer job, sequence, and validity window.
7. Operator command tracking rejects replay, same-sequence substitution, time regression, and attempts to clear terminal cancellation or emergency-stop state.
8. Incidents are captured as immutable signed bundles and registered in an append-only resolution ledger.
9. Release-candidate evidence derives unresolved incidents from that ledger rather than trusting a caller-supplied claim.
10. Release certification binds source tree, fabrication manifest, governed replay, gateway replay, current gateway state, quorum, recovery bundle, trust snapshot, and unresolved incidents.

## Fail-closed rules

- An invalid gateway signature does not contribute to quorum.
- Two different states at one generation cannot both be accepted.
- A recovery bundle cannot skip generations, replace a predecessor, or remove retained evidence.
- A resume command cannot revive a cancelled or emergency-stopped execution.
- Release certification fails while any incident remains unresolved when the default policy is used.
- A valid certificate is still checked against the supplied underlying evidence before promotion.

## Non-claims

This layer does not implement a network consensus protocol, distributed clock synchronization, hardware-backed key storage, or automatic incident adjudication. It provides deterministic evidence contracts and capability boundaries for those systems to use.
