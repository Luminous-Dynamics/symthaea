# Science Benchmark Refresh

Protocol-first workstream for SCI-BENCH-REFRESH-001 (Issue #8771).

## Current contents

- docs/research/SCI-EVIDENCE-CONTRACTS-001.md: evaluator-side evidence behavior contract ontology, metrics, and qualification plan (proposal; fixtures are non-scoring).
- src/evidence_behavior_contracts.rs: typed plan/report commitments, fail-closed case accounting, and per-family scorecards; CLI commands commit/verify both plan and report.
- schema/evidence-behavior-contract-plan.schema.json and schema/evidence-behavior-contract-report.schema.json: strict draft schemas for pre-run obligations and terminal adjudications.
- fixtures/evidence-behavior-contract-plan.example.json and fixtures/evidence-behavior-contract-report.example.json: seven-family illustrative fixtures with placeholder digests; not eligible for scoring.
- tools/evidence_behavior_contract_test_vectors.py: test-only end-to-end CLI fixture generator and hostile-input checks; emits synthetic accounting only, never an empirical model result.
- docs/research/SCI-JUDGE-ROBUSTNESS-001.md: evaluator repeatability, intervention-bias, position-swap and reference-agreement protocol.
- src/judge_robustness.rs: typed judge-audit plan/report commitments, complete repeated-observation accounting, per-family stability/consistency/gold-agreement metrics, and four CLI commands.
- schema/judge-robustness-plan.schema.json and schema/judge-robustness-report.schema.json plus non-scoring fixtures.
- tools/judge_robustness_test_vectors.py: synthetic CLI positive controls and hostile mutations; not empirical judge evidence.
- docs/research/SCI-REFERENCE-ASSURANCE-001.md: protocol for evidence-backed reference labels, independent adjudication, uncertainty preservation, source ancestry and external custody.
- docs/research/SCI-RUBRIC-EXPLOITABILITY-001.md: protocol for rubric validity, meaning-preserving/meaning-changing interventions, score gaming, adaptive Goodhart tests and human-alignment comparisons; no empirical result is claimed.
- schema/rubric-robustness-plan.schema.json: strict plan contract for meaning-preserving, surface-cue, meaning-changing and evidence-withdrawal interventions.
- tools/rubric_attack_vectors.py: deterministic synthetic paired rubric/candidate corpus generator; no judge execution or scoring.
- tools/test_rubric_attack_vectors.py: test-only checks for case coverage, digest binding and intervention semantics.
- schema/rubric-specification.schema.json and fixtures/rubric-specification.example.json: strict rubric-specification contract and a four-criterion, non-scoring example with explicit evidence rules, exclusions, priorities, hard vetoes, anchors, counterexamples and planned adversarial coverage.
- tools/rubric_specification_audit.py and tools/test_rubric_specification_audit.py: static rubric structure/coverage audit and hostile controls; reports declared coverage only, not empirical validity.
- src/reference_assurance.rs: strict typed reference-package commitment/verification, cross-record validation, disagreement accounting and explicit claim-boundary report.
- schema/reference-assurance-package.schema.json and fixtures/reference-assurance-package.example.json: strict v1 package schema and non-scoring example with intentionally preserved rater disagreement.
- tools/test_reference_assurance.py: independent standard-library Python commitment/structure oracle and adversarial unit tests; test-only, not a production validator.
- docs/research/SCI-METAMORPHIC-EVALUATOR-001.md: intervention-based evaluator consistency benchmark across invariance, directional sensitivity, locality, order equivariance, provenance non-duplication, cross-mode consistency and counterfactual relations.
- schema/metamorphic-evaluator-plan.schema.json and schema/metamorphic-evaluator-receipt.schema.json: strict preregistered pair-plan and observed-outcome receipt contracts.
- tools/metamorphic_evaluator_audit.py: deterministic paired-outcome relation checker; no judge execution or causal identification claim.
- tools/metamorphic_evaluator_fixture.py and tools/test_metamorphic_evaluator_audit.py: eight-pair/seven-family synthetic fixture generator and hostile relation/receipt controls.
- docs/research/SCI-FACTORIAL-JUDGE-BIAS-001.md: balanced 2^5 experimental design for candidate order, score-option order, criterion order, provenance cue and presentation format.
- src/judge_factorial.rs: strict full-factorial plan/receipt verification, exact condition/input binding, candidate-choice and score effects, per-criterion order effects, and all ten pairwise interactions.
- schema/judge-factorial-plan.schema.json, schema/judge-factorial-receipt.schema.json and schema/judge-factorial-report.schema.json: preregistered plan, ordinal-bound complete-denominator receipt and typed report contracts.
- tools/judge_factorial_fixture.py: deterministic synthetic 32-cell/two-repeat CLI fixture; no empirical judge run.
- docs/research/SCI-INTERVENTION-QUALITY-001.md: independent quality protocol for evaluator interventions: semantic validity, adjudication ambiguity, diversity and cost-effectiveness.
- src/intervention_quality.rs: census-based candidate and blinded-panel receipt validator, semantic validity precision/recall, raw pairwise agreement, Fleiss' kappa, unique test yield and cost metrics.
- schema/intervention-quality-plan.schema.json, schema/intervention-quality-receipt.schema.json and schema/intervention-quality-report.schema.json: strict plan, fixed-panel blinded adjudication and quality report contracts.
- tools/intervention_quality_fixture.py: deterministic seven-candidate synthetic census with valid, invalid and ambiguous panel outcomes; not a human study.
- docs/research/SCI-TARGET-DELTA-VERIFICATION-001.md: separately audits whether each intervention changed its declared target claim/scope in the intended direction without collateral material changes.
- src/intervention_target_delta.rs: exact parent-plan binding, blinded target-transition consensus, changed-scope comparison, collateral-change detection and combined global/target validity.
- schema/intervention-target-delta-plan.schema.json, schema/intervention-target-delta-receipt.schema.json and schema/intervention-target-delta-report.schema.json: strict target-delta plan, blinded adjudication receipt and claim-bounded report contracts.
- tools/intervention_target_delta_fixture.py: deterministic seven-candidate composed synthetic fixture; not a human study or semantic-truth result.

- `docs/research/SCI-BENCH-REFRESH-001.md`: benchmark-lifecycle protocol and competitive targets.
- `schema/generation-manifest.schema.json`: versioned JSON Schema for the v1 generation manifest.
- `schema/task-set.schema.json`: strict schema for evaluator-held tasks with separate policy-visible and evaluator-only fields.
- `schema/generator-config.schema.json`: versioned schema for deterministic synthetic task constructors.
- `fixtures/generation-manifest.example.json`: non-scoring placeholder manifest fixture.
- `fixtures/task-set.example.json`: eight-task, non-scoring task-set fixture aligned to the manifest's expected denominator and family counts.
- `fixtures/generator-config.example.json`: four-family synthetic generator configuration whose exact bytes are SHA-256-pinned by a real manifest.
- `src/lib.rs`, `src/generator.rs`, and `src/main.rs`: standalone Rust validator, commitment calculator, deterministic generator/replayer, task-set contract checker, and post-run reveal verifier.
- `tools/reference_replay.py`: standard-library-only, test-only independent semantic reconstruction used by CI to cross-check Rust task generation; it is not part of production CLI paths.

## Audit quality of generated evaluator interventions

This audit evaluates the mutation generator and semantic-verification pipeline, not the judge under test. V1 requires a full census of generated candidates and a fixed-size blinded human panel, blind to the declared intended relation as well as generator identity, pipeline disposition and evaluator outcome. It reports accepted-valid precision (with ambiguous accepts retained in the denominator), confirmed-invalid acceptance, ambiguity, valid-candidate recall, relation-stratified accepted/rejected/abstained outcomes, reviewer pairwise agreement, Fleiss' kappa, diversity and unique valid test yield per resource.

```sh
python3 research/science-bench-refresh/tools/intervention_quality_fixture.py /tmp/intervention-quality
cargo run --locked --quiet --manifest-path research/science-bench-refresh/Cargo.toml -- \
  analyze-intervention-quality /tmp/intervention-quality/intervention-quality-plan.example.json \
  /tmp/intervention-quality/intervention-quality-receipt.example.json
```

The fixture is synthetic and non-scoring. Panel labels are supplied declarations; the analyzer does not prove reviewer independence, source custody, semantic truth or downstream judge fault detection. Ambiguous candidates are never treated as valid. Sampled/adaptive selection is intentionally excluded from v1 to avoid selective-adjudication bias. See [SCI-INTERVENTION-QUALITY-001](../../docs/research/SCI-INTERVENTION-QUALITY-001.md).

## Verify intended target changes and collateral scope

This second-stage audit composes with SCI-INTERVENTION-QUALITY-001. The parent panel judges global meaning-preserving / meaning-changing / ambiguous status. This layer separately checks whether the declared target claim/scope changed in the expected direction, whether the exact changed-claim set matches, and whether material collateral changes occurred. Every target-plan entry inherits the parent baseline/variant input digests, and each reviewer receipt must echo those digests and the target identity.

```sh
python3 research/science-bench-refresh/tools/intervention_target_delta_fixture.py /tmp/target-delta
cargo run --locked --quiet --manifest-path research/science-bench-refresh/Cargo.toml -- \
  analyze-intervention-target-delta \
  /tmp/target-delta/intervention-quality-plan.example.json \
  /tmp/target-delta/intervention-quality-receipt.example.json \
  /tmp/target-delta/intervention-target-delta-plan.example.json \
  /tmp/target-delta/intervention-target-delta-receipt.example.json
```

The target plan is bound to the exact parent quality-plan bytes, commitment, analyzer revision, purpose and study ID. Reviewers are blinded to expected direction, allowed changed IDs, generator identity, intended relation, pipeline disposition and downstream evaluator outcome. A final valid intervention requires both global relation validity and target-delta validity. Any known invalid axis produces invalid; ambiguity on either axis remains ambiguous. Reviewer independence, competence, semantic truth, raw artifact custody and execution authenticity are not proven. Scoring eligibility remains false. See [SCI-TARGET-DELTA-VERIFICATION-001](../../docs/research/SCI-TARGET-DELTA-VERIFICATION-001.md).

## Analyze interacting judge nuisance factors

The Rust factorial analyzer checks a complete five-factor (2^5) design per semantic block: candidate order, score-option order, criterion order, provenance cue and presentation format. It requires every factor cell at every repeat, an exact execution-order permutation, exact condition IDs, input-digest echoes and execution ordinals that must match the committed per-block schedule. Missing and ineligible observations are never imputed.

```sh
python3 research/science-bench-refresh/tools/judge_factorial_fixture.py /tmp/judge-factorial
cargo run --locked --quiet --manifest-path research/science-bench-refresh/Cargo.toml -- \
  verify-judge-factorial-plan /tmp/judge-factorial/judge-factorial-plan.example.json
cargo run --locked --quiet --manifest-path research/science-bench-refresh/Cargo.toml -- \
  analyze-judge-factorial /tmp/judge-factorial/judge-factorial-plan.example.json \
  /tmp/judge-factorial/judge-factorial-receipt.example.json
```

The report contains main effects, per-criterion criterion-order shifts and all ten pairwise factor interactions. It also reports candidate score-gap and candidate-choice effects. All effect estimates use only fully completed 32-cell block/repeat units; incomplete repeats are excluded whole, with completed-but-excluded conditions separately counted while every planned cell remains in the denominator. The analyzer is descriptive only; it does not authenticate execution/randomization, prove raw prompt custody or semantic equivalence, estimate population effects, or permit scoring. See [SCI-FACTORIAL-JUDGE-BIAS-001](../../docs/research/SCI-FACTORIAL-JUDGE-BIAS-001.md).

## Audit intervention-based evaluator consistency

The metamorphic auditor checks whether paired evaluator outcomes obey predeclared invariance, directional sensitivity, locality, order-swap, provenance non-duplication, cross-mode and counterfactual relations. Receipts must echo the baseline/variant input digests committed by each plan pair; directional evidence pairs declare their exact expected variant decisions.

```sh
python3 research/science-bench-refresh/tools/metamorphic_evaluator_fixture.py /tmp/metamorphic-example
python3 research/science-bench-refresh/tools/metamorphic_evaluator_audit.py \
  /tmp/metamorphic-example/metamorphic-plan.example.json \
  /tmp/metamorphic-example/metamorphic-receipt.example.json
python3 -m unittest discover -s research/science-bench-refresh/tools -p 'test_metamorphic_evaluator_audit.py' -v
```

A structurally valid receipt can still fail a declared relation. Missing/extra/duplicate observations are rejected. The report does not authenticate evaluator execution or receipt provenance, identify causal effects, estimate generalization or grant scoring eligibility. See [SCI-METAMORPHIC-EVALUATOR-001](../../docs/research/SCI-METAMORPHIC-EVALUATOR-001.md).

## Audit a structured rubric specification

The static audit checks that each criterion defines what counts as evidence, what is insufficient, exclusions, applicability, priority, hard-veto semantics where needed, score anchors, counterexamples and mapped positive/negative/edge/adversarial tests.

```sh
python3 research/science-bench-refresh/tools/rubric_specification_audit.py \
  research/science-bench-refresh/fixtures/rubric-specification.example.json
python3 -m unittest discover -s research/science-bench-refresh/tools -p 'test_rubric_specification_audit.py' -v
```

A passing structural audit does not establish construct validity, human preference fit, empirical dimensionality, judge behavior or adversarial robustness. Those require blinded experiments and held-out human references.

## Generate rubric exploitability synthetic controls

The test-only generator emits a precommitted, non-scoring plan and a paired artifact corpus containing rubric paraphrase/order changes, verbosity and keyword attacks, confidence/citation/formatting theater, criterion removal/priority changes, and evidence withdrawal.

```sh
python3 research/science-bench-refresh/tools/rubric_attack_vectors.py /tmp/rubric-attacks
python3 -m unittest discover -s research/science-bench-refresh/tools -p 'test_rubric_attack_vectors.py' -v
```

The corpus does not call a judge or establish human preferences. Its expected relations are synthetic protocol controls; real scoring requires a blinded reference panel and frozen judge/rubric execution.

## Validate an evaluator-side reference package

The reference-assurance package validates atomic claims, declared temporal/scope metadata, source snapshots, byte-offset evidence spans, adjudicator declarations, initial votes, and separate resolution events. It preserves disagreement; no majority label is generated.

```sh
cargo run --locked --manifest-path research/science-bench-refresh/Cargo.toml -- \
  verify-reference-assurance-package \
  research/science-bench-refresh/fixtures/reference-assurance-package.example.json
```

To compute a package commitment before attaching `package_commitment_sha256`, provide the package fields without that field:

```sh
cargo run --locked --manifest-path research/science-bench-refresh/Cargo.toml -- \
  commit-reference-assurance-package-fields package-fields.json
```

A valid report is deliberately not a scoring qualification: adjudicator independence, source authenticity, source-span byte correspondence and runner execution remain unproven, and an external anchor is required. See [SCI-REFERENCE-ASSURANCE-001](../../docs/research/SCI-REFERENCE-ASSURANCE-001.md).

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

## Cross-set policy-input overlap audit

The overlap auditor is a conservative preflight for two task-set artifacts. It compares only each task's policy-visible input; it does not read evaluator targets to compute overlap.

Example:

```sh
cargo run --locked --manifest-path research/science-bench-refresh/Cargo.toml -- \
  audit-overlap development development-task-set.json refresh:generation-002 refreshed-task-set.json
```

Split labels must use a known role: `development`, `anchor`, or `refresh`; an optional colon suffix distinguishes versions/generations. The labels are caller-supplied metadata, not proof of runner isolation.

The report separates:
- **Shared task ID / exact overlap:** reused task IDs across the compared sets or deterministic canonical policy-input equality (ignores object-key order, JSON whitespace, task IDs, evaluator-only fields, and equivalent integral spellings such as 1 and 1.0). Either condition exits with code 6 and blocks admission.
- **Numeric near-duplicate candidates:** a conservative fingerprint that rounds numbers to three significant digits while preserving the remaining input structure/text. These are triage candidates, not proof of leakage; they exit with code 7 so a qualification pipeline pauses for review.
- **No detected overlap:** exits 0 when neither tier finds candidates.
- **Invalid inputs:** exits 2 for invalid labels, duplicate IDs, malformed task sets, or unsupported schema.

The report binds the raw input artifacts by SHA-256, reports deterministic digest groups and bounded task-ID samples, and never emits evaluator-only labels or target values. Candidate groups are bounded to a sample of 20 task IDs per side while preserving total pair counts.

This is intentionally a first, limited layer. It does not yet prove lineage independence, detect all semantic/unit-equivalent tasks, establish pretraining contamination status, or prove that a runner physically withheld evaluator-only data. Those remain separate gates in [SCI-BENCH-LEAKAGE-001 / #8842](https://github.com/Luminous-Dynamics/symthaea/issues/8842). Numeric candidate hits must be reviewed rather than silently counted as confirmed leakage.


**Ordering requirement:** this command accepts raw task-set JSON bytes and does not itself verify the associated manifest, seed commitment, or regenerated task bytes. First require the existing `verify-reveal` and `verify-regeneration` gates to pass independently for each set; then run `audit-overlap` across the relevant splits. A report saying no overlap is detected is not a standalone scoring authorization.


## Cross-split lineage ledger and ancestry audit

The lineage audit complements overlap fingerprints with explicit evaluator-side ancestry metadata. See `schema/lineage-ledger.schema.json` for the v1 sidecar format. Each ledger:
- binds the exact task-set bytes with `task_set_sha256`;
- covers every task ID exactly once, with no orphan or duplicate records;
- binds each row to the canonical digest of that task's policy-visible input;
- declares one or more typed lineage commitments for a parent world, source task, source item, or transform ancestor;
- records whether the task has never been exposed, was public development material, appeared in a prior scored refresh, or has unknown exposure.

A lineage commitment must describe a specific ancestry/world/source item. Do **not** use generic family names, schema IDs, template IDs, or corpus-version identifiers as lineage commitments; sharing those does not establish task leakage. Lineage digests are commitments, not independently authenticated facts. The ledger must remain evaluator-side, and a hash alone does not prove that the runner enforced this boundary.

After each task set has independently passed `verify-reveal` and `verify-regeneration`, compare the set and its matching ledger:

```sh
cargo run --locked --manifest-path research/science-bench-refresh/Cargo.toml -- \
  audit-lineage development-tasks.json development-lineage.json \
  refresh-tasks.json refresh-lineage.json
```

The command validates the ledger against the exact artifact bytes and task IDs, then compares typed ancestry commitments across split boundaries. Shared parent-world/source ancestry blocks admission even when task IDs and policy-visible inputs differ. A refresh record marked as anything other than `never_exposed` also blocks admission. Exit code 8 means shared declared lineage or exposed refresh tasks; exit 2 means invalid task-set/ledger evidence. Exit 0 means no conflict was detected under the supplied lineage records—it is not a general scoring authorization.

The deterministic report includes task-set and ledger digests, shared lineage commitment digests, bounded task-ID samples, and exposed-refresh counts. It does not include evaluator-only target values.

The ledger cannot independently prove the truthfulness of lineage declarations, full ancestry coverage, absence of unrecorded source overlap, pretraining contamination status, or runtime isolation. Those require independent generator/evaluator audits, controlled exposure logs, and evidence that the evaluated policy cannot read evaluator-side artifacts.


## Search-time exposure receipts

The search-time contamination work is tracked by [SCI-BENCH-SEARCH-EXPOSURE-001 / issue #8865](https://github.com/Luminous-Dynamics/symthaea/issues/8865). Research has shown that tool-using agents can retrieve benchmark metadata, question context, and even answers during inference; these are distinct contamination severities rather than ordinary general research performance. See the EMNLP 2026 Findings paper: https://aclanthology.org/2026.findings-emnlp.760/

This branch adds `schema/search-exposure-receipt.schema.json`, a Rust verifier in `src/search_exposure.rs`, and the `verify-search-receipt` CLI:

```sh
cargo run --locked --manifest-path research/science-bench-refresh/Cargo.toml -- \
  verify-search-receipt retrieval-receipt.json
```

Each receipt binds the run ID, generation, frozen policy revision, opaque task reference, declared network policy, sorted host allowlist when applicable, expected event count, final event digest, and a run-level receipt commitment. Events are ordinal and hash-chained. Query/request/response contents are not stored; only digests, normalized hosts, event kinds, policy decisions, and chain digests are recorded.

Supported network policies:
- `isolated`: any allowed network action is a violation; logged blocked attempts are acceptable.
- `audited_open_web`: allowed network actions are permitted but must be recorded with response digests.
- `controlled_allowlist`: allowed actions must target a hostname in the frozen, sorted host allowlist.

Exit codes: 0 means the chain and declared policy pass; 9 means the chain is intact but the declared network policy was violated; 10 means receipt integrity/count/commitment validation fails; 2 means malformed input. A valid hash chain plus policy compliance still reports that an external anchor is required.

**Security boundary:** a local digest chain detects ordinary event modification, deletion, and reordering against the declared chain. An attacker who can rewrite every event and recompute the final run commitment can produce a new internally consistent receipt. Before scoring, anchor the run/receipt commitment in an independent append-only channel and enforce network isolation in the runner itself. This verifier cannot prove all network channels were logged, authenticate the collector, or detect pretraining contamination.

Test-only vectors are generated by `tools/search_receipt_test_vectors.py`. The focused workflow exercises intact isolated receipts, allowed-but-policy-violating events, allowlist violations, and mutation/deletion/reordering controls. Authored tests are not passing evidence until exact-head CI finishes.


## Evaluator-side search exposure classification

The receipt verifier only audits the integrity and declared network policy of observable event logs. It does not classify whether retrieved content leaks benchmark information. This branch adds a distinct post-run match-report contract in `schema/search-exposure-match-report.schema.json` and Rust validator `src/search_exposure_match.rs`.

Example:

```sh
cargo run --locked --manifest-path research/science-bench-refresh/Cargo.toml -- \
  verify-search-matches retrieval-receipt.json evaluator-match-report.json
```

The match report binds to the exact receipt bytes and the receipt's run commitment, pins a matcher revision and decision-rules digest, and contains exactly one ordered record for every receipt event. The report fails closed on missing/orphaned event records, a modified report commitment, a mismatched receipt, invalid classification contracts, or an altered event digest.

Permitted classes:
- `benchmark_metadata`: retrieval found benchmark metadata that identifies the protected task/benchmark.
- `question_context`: retrieval found the held-out question or materially equivalent task context.
- `explicit_answer`: retrieval found protected answer/target material.
- `benign_related`: related content was judged non-leaking.
- `review_required`: evidence is ambiguous and requires human review.
- `no_match`: the matcher reports no protected match for a response-bearing event.
- `not_evaluable`: no response content exists to match, such as a blocked request.

For confirmed leakage classes, the record must carry a matcher method, a nonzero declared threshold, a score meeting that threshold, and digests for both the protected reference and the stored evidence artifact. Exact-digest classifications require a score of 1,000,000 and identical protected-reference/evidence digests. Raw queries, answer text, and retrieved content do not enter the verifier report.

Exit codes: 0 means event coverage and the evaluator-side report contract pass without a detected match or review requirement; 9 means the network policy was violated and no confirmed match took priority; 10 means receipt-integrity verification failed; 11 means the match-report commitment, receipt binding, event coverage, or classification contract failed; 12 means protected benchmark information was classified as exposed; 13 means manual review is required. The printed report distinguishes confirmed exposures from review-required events and keeps every event in the denominator.

**This is a report-contract verifier, not the semantic matcher itself.** An independent evaluator process must inspect protected benchmark metadata/context/answers against response artifacts, produce these classifications, and keep the evidence store/report out of the policy-visible environment. Hashes bind what was reported; they do not prove the matcher correctly recognized semantic equivalence. The matcher and thresholds require independent qualification. A valid report also does not establish absence of pretraining contamination or unlogged network channels.


## Exact protected retrieval matcher

The exact-match baseline is tracked under [SCI-BENCH-SEARCH-EXPOSURE-001 / #8865](https://github.com/Luminous-Dynamics/symthaea/issues/8865). It turns the prior report-contract verifier into an executable, deterministic first-pass matcher while keeping a hard boundary around what it can claim.

The evaluator-only protected index is defined by `schema/search-protected-artifact-index.schema.json`. Each record contains:
- an opaque reference-ID digest;
- a severity class (`benchmark_metadata`, `question_context`, or `explicit_answer`);
- SHA-256 of the exact protected artifact bytes.

The index records must be sorted by `(content_sha256, classification, reference_id_sha256)` and carry a domain-separated index commitment. The index, protected content, reference map, and report must never be mounted into the evaluated policy environment. Anchor the index commitment before the scored run through the frozen protocol/analysis-plan receipt or an independent append-only channel; a self-consistent hash generated after evaluation does not establish immutability.

To compute the typed commitment, create an `index-fields.json` containing only `generation_id`, `matcher_revision_sha256`, `decision_rules_sha256`, and the sorted `artifacts` array, then run:

```sh
cargo run --locked --manifest-path research/science-bench-refresh/Cargo.toml -- \
  commit-protected-index index-fields.json > index-commitment.json
commit="$(jq -r '.index_commitment_sha256' index-commitment.json)"
jq --arg commitment "$commit" '
  . + {
    schema: "symthaea.science_bench_refresh_protected_search_artifact_index.v1",
    index_commitment_sha256: $commitment
  }
' index-fields.json > protected-index.json
```

The exact matcher performs strict shape, digest, ordering, generation, and commitment checks on the final index. The commitment command computes the deterministic commitment; it does not independently certify that the protected artifact list is complete or authenticate when it was frozen.

Generate an exact-match report:

```sh
cargo run --locked --manifest-path research/science-bench-refresh/Cargo.toml -- \
  match-search-exact retrieval-receipt.json protected-index.json > match-report.json
```

The command verifies the receipt's internal chain, requires the index generation to match the receipt generation, validates the index schema/order/commitment, and creates one classification for every receipt event. It compares each response digest to the protected artifact content digests. If one response digest matches several protected classes, the most severe class wins in this order: explicit answer, question context, benchmark metadata. A deterministic opaque reference ID is used as the protected-reference handle.

`verify-search-exact-matches RECEIPT INDEX REPORT` recomputes the exact matcher output and rejects a report that was edited, relabeled, rebound to different receipt/index bytes, or generated under a different matcher/rules digest.

Exit behavior:
- 0: no exact protected match was detected; blocked/no-response events remain `not_evaluable`, not negative content matches.
- 12: one or more exact matches detected; inspect the emitted report and treat the run as contaminated for the applicable severity.
- 11: an exact report fails deterministic recomputation/binding.
- 2: malformed receipt/index or invalid index commitment/generation.
- 9/10/13 retain the downstream verifier's network-policy, receipt-integrity, and manual-review gates.

For privacy, the match report does not emit protected content or the protected content digest. `protected_reference_sha256` is the hash of an opaque reference identifier; `evidence_artifact_sha256` is a domain-separated event/evidence binding, not the raw response hash. The input receipt still contains its own response digest, so receipt publication and access controls need a separate privacy review.

**Claim limit:** this matcher catches byte-identical protected artifacts only. It does not detect a protected answer embedded within a larger web page, snippets with altered whitespace, paraphrases, semantic equivalents, cached derivatives, or unlogged tools. It is a high-precision positive control and a conservative baseline—not a completeness guarantee. The next matcher milestone should add separately measured canonicalized-span and semantic candidate tiers, with exact positives, benign-related negatives, blinded adjudication, and precision/recall reported independently. Do not merge the tiers into one unexplained leakage score.

## Counterfactual evidence intervention plans (v1)

The full-plan schema is `schema/counterfactual-evidence-intervention-plan.schema.json`; the commitment-input schema (the same object without `plan_commitment_sha256`) is `schema/counterfactual-evidence-intervention-plan-fields.schema.json`. The Rust module `src/counterfactual_evidence.rs` defines a strict pre-run contract for paired evidence interventions. The CLI is:

```sh
cargo run --locked --manifest-path research/science-bench-refresh/Cargo.toml -- \
  commit-evidence-intervention-plan plan-fields.json
cargo run --locked --manifest-path research/science-bench-refresh/Cargo.toml -- \
  verify-evidence-intervention-plan plan.json
```

The plan binds the generation, frozen policy revision, configuration, prompt, tools, resource budget, analysis plan, evaluator revision, decision-rule digest, and randomization commitment. The evaluator implementation and outcome-label rubric are fixed before the first trial; the post-run report must match those exact pins. Every task/world pair must include each declared condition exactly once, with unique trial IDs and a complete per-pair randomization-order permutation. Supported conditions are `natural_retrieval`, `network_isolated`, `gold_evidence_injected`, `gold_evidence_withheld`, `decoy_noise`, and `source_contradiction`. Policy inputs, evidence manifests, seeds, and task references are represented by digests/opaque IDs—not raw protected targets. Every condition within a pair must use a distinct, nonzero evidence-manifest digest; reusing a manifest across nominally different interventions fails closed. All-zero placeholder digests are rejected.

The typed commitment is domain-separated and recomputed by the Rust verifier. The focused CI fixture independently creates it in Python and checks Rust parity, complete-plan acceptance, incomplete-pair rejection even with a recomputed commitment, and mutated-commitment rejection.

**Claim boundary:** a valid result means only that the declared plan is structurally complete and its commitment matches. It does not prove that the plan was anchored before execution, that an external runner performed the interventions, that telemetry is complete, that a system causally used evidence, or that a scientific answer is correct. Keep evaluator-only labels and gold sources outside the policy environment; anchor the plan before the first trial in an independent append-only channel.

For interpretation, pre-register direction-specific outcomes rather than expecting every intervention to change the answer: evidence removal should trigger appropriate uncertainty/abstention when the answer becomes unsupported; a coherent update should change claims that depend on the updated fact; a contradiction should preserve the conflict and provenance; irrelevant decoys should not change a supported conclusion. Score retrieval, reading, evidence attribution, inference, abstention/calibration, and cost separately. Never collapse them into a single scalar that hides a safety-critical failure.

## Counterfactual evidence outcome receipts

The outcome receipt validates what the precommitted evaluator says happened after running a paired intervention plan. The report's evaluator revision and decision-rule digests must equal the values frozen in the plan; changing either after observing outcomes fails closed. It intentionally reuses the plan from `counterfactual_evidence.rs` and borrows full-denominator / missing-cell principles from the standalone paired-trial auditor in `research/scientific-tool-allocation-auditor`; it is not a replacement generic trial framework.

Schemas:
- `schema/evidence-outcome-report.schema.json`: complete evaluator-side report.
- `schema/evidence-outcome-report-fields.schema.json`: the typed commitment input before the report commitment is attached.
- `src/evidence_outcomes.rs`: exact-plan binding, outcome-axis/status invariants, deterministic condition summaries and resource coverage.

CLI:
```sh
cargo run --locked --manifest-path research/science-bench-refresh/Cargo.toml -- \
  commit-evidence-outcomes plan.json outcome-fields.json
cargo run --locked --manifest-path research/science-bench-refresh/Cargo.toml -- \
  verify-evidence-outcomes plan.json outcome-report.json
```

The first command validates the plan and complete outcome coverage, then emits a domain-separated typed SHA-256 commitment. Attach that value as `report_commitment_sha256` in the full report:

```sh
commit="$(cargo run --locked --manifest-path research/science-bench-refresh/Cargo.toml -- \
  commit-evidence-outcomes plan.json outcome-fields.json | jq -r '.report_commitment_sha256')"
jq --arg commitment "$commit" '. + {report_commitment_sha256: $commitment}' \
  outcome-fields.json > outcome-report.json
cargo run --locked --manifest-path research/science-bench-refresh/Cargo.toml -- \
  verify-evidence-outcomes plan.json outcome-report.json
```

The verifier requires one ordered record for every exact planned pair/trial/condition, including timeouts, blocked trials, invalid receipts, contaminated trials and trials never started. It fails closed on missing, extra, duplicated or mismatched records.

Keep these axes separate: terminal/execution disposition; answer/abstention status; task correctness; evidence use; claim support; abstention quality; confidence availability/source; and native resource counters. Confidence is probability in millionths, and null means *not recorded*, not zero. Per-condition summaries report category counts, confidence coverage and, for each resource field, both the number of observations and sum of observed values. Missing fields are not imputed as zero. No combined evidence/trust score or generalized-performance confidence interval is emitted.

Scored labels are disallowed on contaminated, budget-violating, failed, invalid, timed-out and not-started trials. They remain visible in the outcome report and in the planned denominator. `runner_execution_proven` remains false and `external_anchor_required` remains true: the report hash binds content only, not the authenticity of the collector or evaluator.

The validator deliberately does not compute calibration error or infer causal effects. Use the row-level pair/condition data in a separate preregistered analysis; distinguish fixed-benchmark accuracy from generalized accuracy and account for task/world dependence (for example, a well-specified GLMM when justified). Exact-head CI and independent runner evidence remain required.

## Preregistered paired evidence outcome analysis

The Rust module `src/paired_outcome_analysis.rs` consumes a validated intervention plan, a validated outcome report, and the exact analysis-plan artifact whose digest was frozen in the intervention plan. The CLI is:

```sh
cargo run --locked --manifest-path research/science-bench-refresh/Cargo.toml -- \
  analyze-paired-evidence-outcomes PLAN.json OUTCOME_REPORT.json ANALYSIS_PLAN.json
```

The analysis-plan schema is `schema/paired-outcome-analysis-plan.schema.json`; the verification/report envelope is `schema/paired-outcome-analysis-report.schema.json`. The plan freezes analyzer source revision, endpoint contrasts, directional hypotheses, multiplicity policy, pair-dependence policy and rationale digest. The expected-response endpoint additionally requires one evaluator-side target action per planned pair/condition. Never expose this artifact to the evaluated policy. The artifact must not be exposed to the evaluated policy. The analyzer rejects altered analysis-plan bytes or a different analyzer source revision.

Binary endpoints are strict correctness (correct=1; incorrect and partial=0; abstained or not-evaluated outcomes remain endpoint-ineligible), fully supported claims, relevant evidence use, and expected-response adherence. Resource endpoints preserve paired differences for wall clock, tool calls, experiments, tokens, and measured cost. They do not receive p-values by default. Every contrast retains the complete assigned pair denominator, eligible pair count, incomplete count and reasons, plus per-pair values/differences and specific missingness reasons (for example, a named terminal failure versus an unavailable evaluator axis). Failed, contaminated, budget-violating and not-started trials are never silently treated as incorrect answers or zero-cost trials.

For binary matched endpoints, the analyzer reports wins/losses/ties, paired risk difference, the exact two-sided McNemar conditional p-value, and a Wilson 95% interval for the win proportion among discordant pairs when the plan declares independent matched blocks and includes a precommitted rationale digest. If pair dependence is unknown, it still reports paired effects and denominators but suppresses p-values and confidence intervals. That interval is **not** a confidence interval for the risk difference. The exact-test output also preserves log10(p) and a numeric-floor flag, so extremely small p-values are not reported as zero after floating-point underflow; a floored p-value is a conservative numerical upper bound. Holm adjustment uses the full preregistered binary-contrast family size, including contrasts that become unestimable; missing contrasts cannot make surviving p-values appear more significant. Unadjusted p-values are available only under the explicitly exploratory multiplicity policy. Empty eligible-pair sets have no p-value. The report is deterministic and content-committed.

This estimates performance only on the fixed predeclared set. It does not estimate generalized accuracy, certify causality, authenticate evaluator labels or prove that the runner executed the declared intervention. NIST AI 800-3 motivates separating fixed-benchmark results from generalized performance and making uncertainty assumptions explicit. A GLMM/generalization model belongs in a separate analysis with a declared target population, item/task-family structure, assumptions and diagnostics—not as an implied property of a paired p-value.
