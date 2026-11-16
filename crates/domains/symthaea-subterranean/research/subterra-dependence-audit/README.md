# Subterra dependence audit v0.1

A small, standalone Rust reference for hierarchy-aware dataset split auditing.

## Why it exists

A new filename, case ID, reconstructed mesh or extracted feature vector does not create a new independent physical observation. Rows sharing a site, physical acquisition or declared common cause must stay within one Training, Calibration or Evaluation partition.

## Contract

- Automatically groups records by site and physical acquisition lineage.
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

There are 15 authored unit tests. Formatting, compilation, test execution and Clippy status must be reported separately.

## Paired dependency-component bootstrap

The crate also includes a deterministic paired percentile bootstrap over one already-aggregated baseline/candidate score pair per dependency component. It gives every component equal weight, sorts component IDs before resampling, records the seed and replicate count, requires at least 10 components and 200 replicates, and rejects duplicate/non-finite inputs.

For loss metrics, a negative candidate-minus-baseline interval favors the candidate. This is not proof of statistical independence: component grouping quality and the sampling frame remain assumptions. With fewer than 20–30 genuinely distinct components, treat interval estimates as exploratory and publish the component count prominently. The bootstrap does not correct selection bias, nonstationarity or missing common causes.

There are 25 authored tests in this package after the bootstrap addition.
