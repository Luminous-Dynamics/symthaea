# METHODE-Bench v0 method-fidelity contract auditor

Status: initial typed conformance implementation and deterministic negative controls. No AI agent has been evaluated and no scientific claim is qualified by these unit tests.

## Purpose

This package distinguishes:
- faithful reproduction of an independently reviewed method protocol;
- an explicitly declared exploratory deviation;
- work rejected before execution;
- unresolved/incomplete or untrusted execution evidence.

It deliberately keeps scientific result validation separate from method fidelity. A faithful method can produce an unsupported scientific result. A positive producer exit code or a self-reported success flag is never sufficient.

## Contents

- `src/lib.rs`: typed contract, receipt, audit report and 34 deterministic tests.
- `Cargo.toml` / `Cargo.lock`: standalone zero-dependency workspace so this slice does not depend on the monorepo workspace or its Cargo.lock.

## Checks implemented

The auditor binds:
- independently reviewed protocol identity and exact contract digest;
- dataset ID/version/raw digest and split-manifest digest;
- exact train/validation/test counts;
- preprocessing graph digest, fit scope, fitted split and fitted-record count;
- mandatory component implementation and configuration digests, with exact set equality;
- baseline identities/configuration digests;
- metric definition digests;
- budget-policy digest, required epochs and resource caps;
- exact reviewed output allowlist and output digests; output IDs are single-segment opaque identifiers, not file paths;
- independently pinned replay-verifier identity/implementation version, separate from the protocol reviewer; replay result bound to the reviewed contract and exact output artifact list.

A declared, non-critical mismatch is classified as `ExecutedWithDeclaredDeviation`, never as reference reproduction. An undeclared mismatch is unresolved. Resource overruns, incomplete execution, missing or extra output artifacts, path-like output IDs, invalid reference review or missing/failed independent replay remain unresolved even when the producer claims success.

## Run locally

```sh
cargo fmt --manifest-path tools/methode-fidelity-harness/Cargo.toml -- --check
cargo test --manifest-path tools/methode-fidelity-harness/Cargo.toml --locked
cargo clippy --manifest-path tools/methode-fidelity-harness/Cargo.toml --locked --all-targets -- -D warnings
```

## Limitations

This first slice validates typed in-memory receipt structures. It does not yet parse an external JSON protocol, execute scientific workloads, recompute raw artifact hashes, verify scientific ground truth, or replace #12471's method compiler. The caller must supply source/artifact and independent-review receipts from their owning systems. The exact output allowlist prevents unreviewed extra claim artifacts from being silently admitted. Output IDs are not filesystem paths; this check is not a filesystem sandbox. The pinned verifier identity is an anti-substitution check, not proof that the verifier actually ran or that it is isolated. This reference does not authenticate identities, signatures, processes or service principals. A production integration still requires a separately controlled verifier service/identity, authenticated receipts, OS/service isolation, and an audited serialization boundary before replay may be treated as independent.

## External benchmark alignment

- ASI-Bench B1/B2 separates full method/procedure guidance from method-specified implementation autonomy: https://asibench.apexin.ai/leaderboard/
- ABE-Ralph identifies silent dataset/budget reductions, component substitution and conclusions from resource-limited runs: https://arxiv.org/abs/2608.26753
- FrontierChallenge distinguishes partial progress from full deliverable completion and reports unsupported completion claims: https://arxiv.org/abs/2608.24979
- ScienceAgentBench includes expert-validated, paper-derived scientific tasks and a verified split: https://github.com/OSU-NLP-Group/ScienceAgentBench
