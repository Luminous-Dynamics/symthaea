# Science Benchmark Refresh

Protocol-first workstream for SCI-BENCH-REFRESH-001 (Issue #8771).

## Current contents

- `docs/research/SCI-BENCH-REFRESH-001.md`: benchmark-lifecycle protocol and competitive targets.
- `schema/generation-manifest.schema.json`: versioned JSON Schema for the v1 generation manifest.
- `schema/task-set.schema.json`: strict schema for evaluator-held tasks with separate policy-visible and evaluator-only fields.
- `schema/generator-config.schema.json`: versioned schema for deterministic synthetic task constructors.
- `fixtures/generation-manifest.example.json`: non-scoring placeholder manifest fixture.
- `fixtures/task-set.example.json`: eight-task, non-scoring task-set fixture aligned to the manifest's expected denominator and family counts.
- `fixtures/generator-config.example.json`: four-family synthetic generator configuration whose exact bytes are SHA-256-pinned by a real manifest.
- `src/lib.rs`, `src/generator.rs`, and `src/main.rs`: standalone Rust validator, commitment calculator, deterministic generator/replayer, task-set contract checker, and post-run reveal verifier.
- `tools/reference_replay.py`: standard-library-only, test-only independent semantic reconstruction used by CI to cross-check Rust task generation; it is not part of production CLI paths.

## Validate a manifest

```sh
cargo fmt --manifest-path research/science-bench-refresh/Cargo.toml -- --check
cargo test --locked --manifest-path research/science-bench-refresh/Cargo.toml
cargo run --locked --manifest-path research/science-bench-refresh/Cargo.toml -- \
  research/science-bench-refresh/fixtures/generation-manifest.example.json
```

The example should report `valid: true` and `manifest_declares_scoring_eligible: false`. The fixture must fail when the CLI is invoked with `--require-scoring` or `--require-sealed`.

## Commit–reveal protocol

The manifest carries three commitment digests:

- `protocol_commitment_sha256`: all frozen non-artifact fields in the manifest.
- `seed_commitment_sha256`: exact bytes of a high-entropy seed, bound to the protocol commitment.
- `task_set_commitment_sha256`: exact bytes of the complete evaluator task-set artifact, also bound to that protocol commitment.

The protocol context is a versioned Rust struct serialized with `serde_json`; it includes the generation ID, lane, purpose, generator repository/revision/config digest, evaluator version/source/boundary declaration, expected task count and family mixture, frozen policy/tool/prompt/config identities, budgets, analysis-plan digest, and reveal policy. It excludes the three commitment fields to prevent circularity. Any change to the frozen context changes its protocol commitment. The locked Rust/Serde serialization profile is part of this versioned contract.

Commitment bytes use SHA-256 with the domain prefix `symthaea.science_bench_refresh.commitment.v1\0`, a kind-specific domain (`protocol.v1\0`, `seed.v1\0`, or `task-set.v1\0`), and unsigned 64-bit big-endian length framing. Artifact commitments also include the protocol digest as context. Seed files must contain at least 32 bytes. Task-set commitments bind the exact raw bytes, so whitespace/ordering changes intentionally change the digest.

### Generate commitments

First create a draft manifest with all non-commitment fields frozen. The three commitment fields can be 64-zero placeholders during this step; do not use that draft for scoring.

```sh
cargo run --locked --manifest-path research/science-bench-refresh/Cargo.toml -- \
  protocol-commit draft-manifest.json

cargo run --locked --manifest-path research/science-bench-refresh/Cargo.toml -- \
  artifact-commit draft-manifest.json seed seed.bin

cargo run --locked --manifest-path research/science-bench-refresh/Cargo.toml -- \
  artifact-commit draft-manifest.json task-set evaluator-task-set.json
```

Copy the reported `protocol_commitment_sha256` into the manifest, and copy each artifact's reported `artifact_commitment_sha256` into its matching field. Because artifact commitments are computed from the non-commitment protocol context, computing them from the draft does not create a circular dependency. Before a real sealed run, the seed must be generated from a cryptographically secure source with at least 256 bits of entropy and kept outside the policy-visible environment. The committed task-set file must use `schema/task-set.schema.json`: every record has a unique `task_id`, a declared `family_id`, a `policy_input` object, and a separate `evaluator_only` object. A runner must pass only `policy_input` into the evaluated policy.

After submissions close and the reveal is authorized, verify the exact original bytes:

```sh
cargo run --locked --manifest-path research/science-bench-refresh/Cargo.toml -- \
  verify-reveal final-manifest.json revealed-seed.bin revealed-task-set.json
```

`verify-reveal` accepts only a structurally valid manifest declaring `purpose=qualified_run` and `lane=sealed_refresh`. It verifies the protocol, seed and exact task-set-byte commitments; it also checks task-set schema/version, matching generation ID, exact task count, unique task IDs, declared per-family counts, object-shaped inputs, and rejects known oracle-only field names recursively inside `policy_input`. It exits 0 only if commitments and task-set contract all pass, exits 5 if the revealed artifacts or task-set contract fail, and exits 2 for invalid inputs/manifest. Its report separates commitment-match flags from `task_set_contract_valid`; it never outputs seed bytes.

### Important limits

Commit–reveal proves that the revealed bytes match the frozen context. To establish that the commitment existed before outcomes were observed, independently anchor the protocol/manifest commitment before execution (for example, an append-only receipt or immutable commit) and retain that anchor with the run. A digest stored only in the final mutable manifest is not a trusted timestamp. Commit–reveal does **not** prove that the seed was secret before the run, that the runner prevented oracle access, that the declared generator actually produced the task set, or that the task set is scientifically valid. Those require separate runner isolation evidence, task regeneration from the revealed seed and pinned generator, and evaluator/duplicate/leakage controls.

The repository example is a **non-scoring fixture**: its revision says `NOT_IMPLEMENTED_YET` and its hashes are all zeros. It must never be reused for an actual experiment. The validator reports manifest-declared eligibility rather than pretending to certify execution isolation.

## Deterministic task regeneration

The standalone Rust module supports four bounded synthetic constructor families: known coordinate reparameterization, noisy linear-law recovery, common-cause/null discovery, and narrow-domain identifiability/abstention. It uses a deterministic SHA-256-derived stream keyed by the revealed seed, generation ID, family ID, task index, and field name; it does not depend on platform RNG state.

Useful commands:

```sh
cargo run --locked --manifest-path research/science-bench-refresh/Cargo.toml -- \
  generate-task-set MANIFEST.json REVEALED_SEED.bin GENERATOR_CONFIG.json OUTPUT_TASK_SET.json

cargo run --locked --manifest-path research/science-bench-refresh/Cargo.toml -- \
  verify-regeneration MANIFEST.json REVEALED_SEED.bin GENERATOR_CONFIG.json COMMITTED_TASK_SET.json
```

`generate-task-set` refuses a config whose exact bytes do not match `generator.configuration_sha256`, refuses a seed that does not match its context-bound commitment, requires a valid pinned sealed-refresh manifest, and checks generated task-family counts against the expected denominator. It intentionally permits an all-zero `task_set_commitment_sha256` placeholder during the precommit generation step; all other manifest fields and the frozen protocol commitment still validate, but the output is not admissible until the real task-set digest is inserted and `verify-reveal` / `verify-regeneration` pass. This removes the circular workflow trap where a task-set digest is needed before the task set exists. `verify-regeneration` reproduces the exact task-set bytes and compares them with the revealed artifact in addition to checking reveal commitments and the task-set contract. The crate's `build.rs` embeds the exact checked-out Git commit and records whether the worktree was clean; it checks any declared build revision against `git rev-parse HEAD`. Non-test task generation refuses a dirty build and refuses a manifest whose `generator.revision` differs from the compiled revision, while unit tests remain runnable in a dirty developer worktree. This closes the gap where a well-formed manifest could name one source commit while another source revision actually generated its tasks, within the limits of the runner's build provenance.

Task IDs are full SHA-256 digests over domain-separated seed/generation/family/index inputs, and generated rows are sorted by that opaque ID rather than grouped by family. The family label remains only in the evaluator-held outer row; the runner must forward only `policy_input` to the policy.

The task constructors are **synthetic protocol fixtures**, not a qualified physical/scientific benchmark. The v1 config locks each family to an audited policy-visible instruction template; arbitrary recipe instructions are rejected so the config cannot trivially inject answer text into a prompt. CI compares the generator to `tools/reference_replay.py`, an independent test-only Python implementation that reconstructs generated observations/targets from seed and config rather than calling the Rust generator itself. Both implementations run a fixed 64-seed sweep; the Rust test requires unique task IDs across 512 generated tasks, all four declared coordinate scales, all four slope bins, and non-degenerate noise residuals. This parity check reduces shared-code risk but still does not validate the scientific correctness of the generator's targets. A Rust semantic gate also checks transformed-law equivalence, declared noise bounds, and the common-cause null. For identifiability tasks it stores a second polynomial mean function with a root at every observed x and verifies that it differs by more than the declared noise amplitude at a separate out-of-domain challenge point. This is a concrete synthetic counterexample to unique global-law identification from finite observations, not evidence that any particular physical system is non-identifiable. They test determinism, family-count closure, input/oracle field separation, and reproducible generation. They do not establish that the task distribution is scientifically representative, that the answer labels are correct for a physical domain, or that the evaluated policy cannot access evaluator-only data. The `generator-config.example.json` is a recipe example; real scored manifests must pin the exact config bytes and immutable generator source revision.

## Current implementation boundary

The Rust validator checks strict manifest shape and unknown-field rejection, digest/revision syntax, protocol-context integrity, domain-separated seed/task commitments, exact-byte reveal verification, exact deterministic regeneration from seed/config bytes, task-set generation/denominator/family integrity, constructor-specific semantic consistency, duplicate task IDs/families, oracle/family-label canaries in policy-visible inputs, an ASCII letters/digits/underscore/hyphen key grammar enforced by both schema and Rust, and separator-/case-insensitive normalization, denominator and family-weight closure, lane/reveal-policy consistency, and separate manifest-declared profile fields. The independent Python tests also check the transformed-law equations, declared noise bounds, common-cause null, and the identifiability witness.

At the current draft stage, 58 Rust unit tests are authored across manifest/reveal validation, task-set contract checks, constructor-specific semantic checks, deterministic generation/replay, generator-instruction leakage controls, compiled-source pinning, and CLI argument handling. Eleven additional standard-library Python tests exercise the independent replay oracle, including its own 64-seed sweep and prompt-leakage rejection. They must not be described as passing until current-head exact CI completes. This authoring environment does not include a Rust toolchain, so no local compile/test/Clippy result is claimed.