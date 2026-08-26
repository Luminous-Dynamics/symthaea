# SYM-ARCH-002A3 Benchmark Validity and Mutation Testing v1

**Status:** benchmark-integrity infrastructure  
**Scientific claim status:** none  
**Parent plan:** issue #55  
**Stacked dependency:** PR #57 (`research/sym-arch-002a-core-v1`)  

## Purpose

SYM-ARCH-002A3 makes benchmark validity a prerequisite for architecture evidence.

The intended ordering is:

1. define a typed `TaskProgram`;
2. execute its symbolic oracle over generated examples;
3. validate split identity, labels, support contracts, and leakage policy;
4. deliberately corrupt the benchmark in known ways;
5. require those corruptions to be detected;
6. only then permit architecture scores to enter a ClaimLedger.

A model result from a benchmark that fails this layer should be treated as **benchmark invalid / inconclusive**, not as evidence that the model succeeded or failed.

## Executable v1 oracle

A3 provides an executable interpreter for the v1 `RuleExpr` language:

- `Eq` / `Ne` compare named integer-valued features;
- `Parity` evaluates a named integer feature modulo a declared modulus;
- `Not`, `And`, `Or`, and `Xor` compose rules recursively.

Missing required features fail closed.

`symbolic_oracle_digest` domain-separates and hashes the exact serialized `RuleExpr`. A valid A3 task requires `TaskProgram.oracle_digest` to equal this executable-oracle digest, binding the stated oracle identity to the rule actually used for label checking.

This v1 contract intentionally supports only `RuleExpr`-grounded symbolic oracles. Richer external simulators/causal worlds should receive their own typed oracle contract rather than bypassing this check.

## Generated dataset identity

`GeneratedTaskDataset` binds:

- the exact `TaskProgram` digest;
- a training example set;
- an evaluation example set.

Each `ExampleRecord` has:

- a non-empty example id;
- a deterministic integer feature map (`BTreeMap`);
- explicit support tags;
- the expected oracle label.

Dataset hashing is set-like for example order: examples are canonicalized by id and support tags are sorted before hashing. Reordering a static train/evaluation set therefore does not manufacture a new dataset identity.

A separate **feature-only** digest deliberately ignores id, label, and support annotation so a train/eval duplicate cannot evade leakage detection by receiving a new id or label.

## Validity policy

`BenchmarkValidityPolicy::task_free_strict()` is the conservative default for latent-context/task-free tests. It rejects features named like explicit task/boundary leakage, including:

- `__task_id` / `task_id`;
- `__world_id` / `world_id`;
- `boundary_marker`;
- `time_to_switch`.

The policy also defaults to:

- feature-disjoint train/evaluation splits;
- declared support tags only;
- both classes represented in both splits.

Explicit-context experiments may construct a narrower forbidden-key policy, but doing so is an explicit benchmark-design decision rather than silent leakage.

## Fail-closed checks

`validate_generated_task` reports independent violation categories rather than one scalar validity score. v1 checks:

- TaskProgram structural validity;
- dataset ↔ TaskProgram digest binding;
- executable oracle ↔ oracle digest binding;
- non-empty splits;
- example structural validity;
- globally unique example ids;
- feature-identical train/eval leakage;
- forbidden task/world/boundary features;
- split support tags against TaskProgram declarations;
- expected labels against the symbolic oracle;
- observed positive/negative counts against TaskProgram declarations;
- class degeneracy within either split when required by policy.

Any violation makes the benchmark `invalid`.

## Mutation testing the validator

A3 includes deliberate benchmark corruptions:

1. **flip first evaluation label** — oracle/label integrity failure;
2. **leak first training feature assignment into evaluation** — split leakage;
3. **corrupt program digest** — provenance failure;
4. **inject forbidden task id** — hidden task-identity leakage;
5. **inject undeclared evaluation support** — split/support contract failure.

`mutation_detection_suite` first requires the unmodified benchmark to validate, then applies every standard corruption independently and requires each mutated benchmark to become invalid.

This tests the scientific instrument itself rather than assuming its failure paths work.

## What v1 does not certify

Passing A3 v1 does **not** prove that a benchmark is free of every shortcut.

In particular, v1 does not yet automatically detect:

- a non-reserved feature that happens to encode the label perfectly;
- a marginal-factor shortcut that solves a supposedly relational task;
- nearest-neighbor/template shortcuts;
- distributional artifacts that identify the split;
- semantic equivalence between syntactically different generated programs;
- temporal leakage embedded in otherwise legitimate timestamps;
- task identity hidden through a learned/encoded representation rather than an explicit feature key.

Those require the planned **construct-validity control ladder**: chance/majority, marginal predictors, nearest-neighbor/lookup, shuffled-relation negative controls, symbolic positive control, and eventually semantic/program-equivalence checks.

A3 v1 therefore supports the wording:

> structural/oracle benchmark integrity passed

not:

> all possible shortcut learning has been ruled out.

## Acceptance gate

The exact stacked PR head should pass:

1. rustfmt for A3 paths;
2. focused `experiment_validity` tests;
3. `cargo check -p symthaea-psych-bench --lib`;
4. nested boolean-oracle execution test;
5. executable-oracle digest binding test;
6. deterministic order-insensitive dataset identity test;
7. feature-only train/eval leakage test;
8. forbidden task-identity leakage test;
9. declared class-count test;
10. standard mutation suite with every corruption detected.

## Merge boundary

This PR is independent of A2 statistics and is stacked directly on #57. It may be reviewed in parallel with #58.

After #57 merges, retarget/rebase A3 to `main` without changing the validity contract. No architecture result is required for merge because this is scientific instrumentation, not evidence of an architecture advantage.
