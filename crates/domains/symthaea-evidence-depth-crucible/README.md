# symthaea-evidence-depth-crucible

Adversarial qualification for the verifier-diversity and atomic-coverage gates.

The crucible uses a small synthetic safety case that can genuinely become `Ready`, then applies positive and negative controls proving that:

- duplicate receipts from one verifier do not create verifier independence
- distinct verifier ids in one organization do not create organization diversity
- missing verifier profiles block a required diversity policy
- malformed duplicate verifier profiles fail invalid
- a parent receipt does not implicitly cover atomic facets
- coverage of one facet does not leak to a sibling facet
- repeated use of one facet evidence digest does not satisfy a distinct-evidence-object requirement
- explicitly independent verifiers and explicitly distinct facet artifacts can satisfy reviewed policies

The report is descriptive test evidence only and grants no physical authority.

```bash
cargo test -p symthaea-evidence-depth-crucible
```
