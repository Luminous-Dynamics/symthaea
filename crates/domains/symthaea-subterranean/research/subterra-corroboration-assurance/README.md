# subterra-corroboration-assurance

A Rust consumer gate for deciding whether multiple observations can be treated as corroborating evidence under a specific, externally declared dependency/fault model.

## Paradigm

Count claims supported by independent evidence paths, not sensor modalities, vendor labels, robot IDs, model versions or receipt count. Different modalities mounted on the same robot can share power, timing, localization, compute, calibration and communications. Different robots can share a map, relay, fleet manager, training corpus or verifier lineage.

## Integration boundary

This crate consumes an `AdmittedInspectionOption` from `subterra-inspection-evidence-gate` and a Mycelix-produced transitive dependency closure using the existing LEX-NET-030 graph and QUAL-001E fault-domain profile. It does not create a second dependency graph.

The caller supplies the graph, profile, evidence-set, and assessment digests expected for this claim. The assessment supplies exact source and verifier dependency closures, completeness dispositions, a common-cause disposition and a verifier-separation disposition. Locally, the crate checks that the declared transitive claim-critical dependency identities do not overlap across evidence sources or between evidence and the independent verifier.

Only dependencies classified upstream as claim-critical belong in those lists. Shared context that is not a claim-critical failure path must not be inserted merely because it is shared; conversely, a dependency inventory must not omit a real common cause. Correctness remains conditional on the completeness and fidelity of the external dependency model.

## Admission

A portfolio is admitted only when:
- source, observation and calibration identities are unique;
- at least the policy-required number of sources are present;
- each source is bound to the exact asset, revision, frame and region;
- one source is exactly the observation/calibration/dataset anchor already admitted by the upstream evidence gate;
- all evidence and the external assessment are fresh under a frozen evaluation time;
- the graph, fault-domain profile, assessment and evidence-set digests match the expected context;
- every source and verifier has a complete, non-empty transitive dependency closure;
- every source has a purpose-matched rights receipt and live calibration and rights validity intervals;
- no claim-critical dependencies overlap across purportedly independent sources;
- no claim-critical dependency overlaps between acquisition and verifier;
- Mycelix's portfolio and verifier dispositions explicitly report no shared common cause / independent verification within the complete declared scope;
- the chosen fault semantics are supported by the declared finite model.

"Independent" means independent within this declared, complete fault model. It does not mean physically or metaphysically independent in the absence of an accurate inventory.

## Frozen benchmark manifest

`subterra-corroboration-benchmark.v0.1.json` defines the source/robot/modality/path-depth panel, 16 adversarial cases, metrics and research targets. It is a protocol manifest, not evidence that the cases have been executed or that physical independence has been demonstrated.

## Run

```sh
cargo fmt --manifest-path Cargo.toml -- --check
cargo test --manifest-path Cargo.toml --all-targets --offline
cargo run --manifest-path Cargo.toml --example synthetic_corroboration --offline
cargo clippy --manifest-path Cargo.toml --all-targets --offline -- -D warnings
```

## Claim ceiling

This crate does not authenticate Mycelix signatures, reconstruct the LEX-NET-030 graph, prove that the graph lists every real dependency, authenticate calibration/rights, establish POD/PFA or physical condition, or authorize intervention. A green synthetic run qualifies only this consumer contract on its declared fixtures.
