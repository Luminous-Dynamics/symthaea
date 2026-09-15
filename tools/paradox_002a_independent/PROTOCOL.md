# PARADOX-002A-I — Independent Census Verification v1

Authority: **MeasurementOnly / independent qualification companion**.

This verifier is frozen against qualified PARADOX-002A subject
`eb73527d05a913e79d1f05135ad6b06c1da8e2ee` and its whole-census commitment:

- records: `7168`
- bytes: `1830977`
- SHA-256: `49ff56a49ac730d960d7625b65ddb0139171fa3862d145a6deb3d68245c7e711`

The Python implementation owns only the independent parser and theorem. Rust may be used only to reproduce the exact already-qualified census byte stream. The Python verifier must never import Rust bindings, invoke `qualify_fixture`, derive rules by scraping Rust source, or consume a Rust-produced expected-label sidecar.

## Qualification sequence

1. Check out the exact verifier subject SHA.
2. In a separate worktree or exact-source checkout, reproduce `census.bin` using only the already-qualified `qualification_census` emitter from `eb73527d...`.
3. Before Python sees the corpus, require both exact byte count and exact SHA-256 above. A mismatch is `QualifiedCorpusReproductionMismatch` and terminates qualification.
4. Run `verify.py` with Python 3 standard library only and bind `--verifier-sha` to the exact verifier subject SHA.
5. The verifier parses all 7,168 fixture/report pairs, derives the manipulation theorem from fixture bytes, and compares every covered semantic field against the Rust report using exact discrete equality and exact IEEE-754 bit equality.
6. The verifier executes all 12 frozen semantic mutation controls, including a raw length-preserving option-tag corruption.
7. Run the verifier in two separate Python processes over the same `census.bin`; require byte-identical normalized receipts.
8. Preserve the receipt, exact corpus digest/length, Python version, verifier subject SHA, and logs as evidence. Artifact upload is archival convenience only; the digest gate remains authoritative.

No pull-request-triggered qualifier is added in this tranche while repository Actions backlog remains congested. A dedicated exact-subject qualifier should be added or dispatched only when intentionally prioritized under #3481.

## Claim boundary

A PASS establishes only independent cross-language agreement on the frozen synthetic PARADOX-002A manipulation plane and fail-closed rejection of the preregistered corruptions. It does not establish behavioral competence, metacognitive recruitment, ontology repair, consciousness, sentience, phenomenology, or superiority of any consciousness theory.
