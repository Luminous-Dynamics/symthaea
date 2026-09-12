# symthaea-domain-awareness-evidence-depth

Typed candidate-evidence bindings from the evidence-depth qualification crucible into the canonical DomainAwareness safety contract.

A passing qualification can create candidates for:

- `DA-029` — verifier common-cause diversity
- `DA-030` — explicit atomic evidence-facet coverage

The bridge checks the reviewed positive and negative scenario matrix by exact scenario id. Missing, duplicated, or failed required controls fail closed.

The result is still only `CandidateEvidence`. It does not create a verified receipt, mutate obligation workflow state, establish deployment readiness, or grant physical authority.

```bash
cargo test -p symthaea-domain-awareness-evidence-depth
```
