# Subterra dependence audit v0.1

A small, standalone Rust reference for hierarchy-aware dataset split auditing.

## Why it exists

A new filename, case ID, reconstructed mesh or extracted feature vector does not create a new independent physical observation. Rows sharing a site, physical acquisition or declared common cause must stay within one Training, Calibration or Evaluation partition.

## Producer/consumer integration

The package consumes the versioned Symtropy observation-lineage manifest using strict Serde JSON deserialization. It validates unique record identity, source references, derivation-graph acyclicity, raw-versus-derived record shape, and exact inherited acquisition-lineage sets. Split assignments are supplied separately by the evaluator, converted to the existing audit types, and then checked for transitive leakage.

The six-record observation-lineage fixture is pinned to Symtropy branch `research/geobio-foundry-v0.2`, commit `db71f6a3075352f303a3f647ce74fe86029430dd`, fixture blob `348f4e8e3f4e817f033ea6cc89d43bb24a07c2f3`; the matching producer schema blob is `06c7dcb36975fbedc4810f121e84a670768dc5e0`. Three schema-conforming site-evidence fixtures from the same producer branch are mirrored beside it and their individual blob IDs are recorded below.

The site-evidence observation fixture blobs are:
- Site A: `714803b2b5654819d0e75c23c2b41f1ebf6f42fe`
- Site B: `2462e2b83e44f64d2f01a09315fcf6012b4ea27d`
- Site C: `094d91e870e30cd3a299bbabe3efc06122f0f2d8`

The consumer's observation cross-reference method verifies that every manifest record resolves to exactly one observation in the same site and that every site observation has exactly one lineage record. The input documents are projected to schema version, site ID and observation IDs; full schema validation remains the producer contract.

## Contract

- Automatically groups records by site and every physical acquisition lineage contributing to a record.
- Allows multiple derived records from one acquisition; it does not count them as separate dependency components.
- Accepts explicit shared-cause groups for instrument-calibration sessions, environmental episodes, processing runs, reference-label campaigns and other declared common causes.
- Uses transitive connected components: if A shares a cause with B and B shares a cause with C, all three remain in one split group.
- Rejects any group crossing a data-split boundary.
- Produces deterministic component and group summaries independent of row ordering.

## Important limits

A component is a conservative grouping boundary, not a proof that observations inside it have a known correlation coefficient or that different components are statistically independent. Undeclared common causes are invisible. Group IDs are caller assertions, not authenticated source evidence. Do not use a global instrument serial or software version as a grouping episode unless the benchmark intends to hold out that entire instrument or version.

This crate does not refit likelihoods, calculate confidence intervals, authenticate manifests or alter the existing calibration dataset. It is an upstream audit for a richer raw observation manifest. The existing likelihood-calibration package retains its stricter v0.1 one-case-per-site/action contract.

## Verification

Run with the Subterra standalone-package script:

`bash crates/domains/symthaea-subterranean/research/verify-subterra-reference-packages.sh`

There are 28 authored unit tests. Formatting, compilation, test execution and Clippy status must be reported separately.

## Paired dependency-component bootstrap

The crate also includes a deterministic paired percentile bootstrap over one already-aggregated baseline/candidate score pair per dependency component. It gives every component equal weight, sorts component IDs before resampling, records the seed and replicate count, requires at least 20 components and 200 replicates, and rejects duplicate/non-finite inputs.

For loss metrics, a negative candidate-minus-baseline interval favors the candidate. This is not proof of statistical independence: component grouping quality and the sampling frame remain assumptions. With fewer than 20–30 genuinely distinct components, treat interval estimates as exploratory and publish the component count prominently. The bootstrap does not correct selection bias, nonstationarity or missing common causes.

There are 47 authored tests in this package after site-observation cross-reference integration.
