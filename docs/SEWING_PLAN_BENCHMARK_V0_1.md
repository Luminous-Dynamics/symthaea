# SewingPlanBench v0.1 — typed garment assembly graph validation

## Goal

Evaluate whether a proposed garment assembly plan is topologically valid, traceable to exact pattern identities, compatible with declared machine capabilities, and safely handed to a human/scheduler when its assumptions are incomplete.

This benchmark complements [SEWING-PLAN-COMPILER-001 #12134](https://github.com/Luminous-Dynamics/symthaea/issues/12134) and the larger apparel digital thread. It is not a machine-control benchmark.

## Why the frontier matters

Park and Kim's 11 September 2026 study, *Development of an LLM-Based Multi-Agent System for Sewing Sequence Planning*, uses a semantic metadata schema for panels, edges and correspondences and a generation–verification–question–revision multi-agent loop. It studies 100 patterns and reports improvement over a single-agent configuration, while requiring expert intervention for some design-specific cases. This is a relevant SOTA comparator; Luminous should reproduce its task and evaluation conditions if source materials permit, not claim direct superiority from a different validator task.

Source: https://doi.org/10.1177/0887302X261487373

## Paradigm shift

Move from a generated list of sewing instructions to a **typed assembly contract**:

- pattern piece and edge identities are explicit;
- each joining interface names the exact participating edges;
- operation nodes carry predecessors and part/interface references;
- machine capabilities are declared by a versioned machine profile;
- unknown predecessors, missing edges, unsupported operations and cycles are not silently accepted;
- ambiguous or unmeasured construction decisions become questions/human review, not model guesses;
- the resulting DAG is handed to Symtropy for scheduling and process simulation only after structural qualification.

## Benchmark task ladder

### S0 — graph contract (synthetic)

- unique identities;
- valid pattern-piece/edge references;
- interface cardinality and units;
- predecessor identity and uniqueness;
- acyclic graph and deterministic topological ordering;
- machine-operation compatibility;
- exact plan/source brief/source pattern digest matching.

### S1 — garment semantics

- edge correspondence and orientation;
- notch sequence and seam length compatibility;
- grain direction and mirrored-piece detection;
- seam allowance/tolerance compatibility;
- irreversible step access (press/turn/finish before closure);
- operation-specific preconditions/postconditions;
- bounded rework loops with explicit terminal inspection.

### S2 — expert-reviewed assembly sequence

Compare expert patternmakers/sewing technologists, deterministic construction rules, single-LLM planner, multi-agent planner with clarification, and Luminous typed graph + human override. Keep pattern IDs disjoint across design/tuning/test.

### S3 — physical overshirt

Only after a named non-PPE overshirt pattern, fabric lot, machine profile, operators, inspection rubric and safety approval are frozen. Measure first-pass assembly acceptance, seam dimensions/quality, rework, elapsed time and task-specific physical failures.

## Metrics

| Metric | Definition |
|---|---|
| Reference validity | Fraction of part/edge/interface/operation references that resolve |
| Precedence validity | Fraction of dependency constraints satisfied; cycle/dead-end detection |
| Machine compatibility | Operation/profile compatibility, with unknown capability treated as unresolved |
| Stale-plan rejection | Exact-digest mismatch detection |
| Expert correction burden | Number and severity of human corrections per accepted plan |
| Clarification quality | Precision of questions that resolve true ambiguities; abstention on unsupported cases |
| First-pass assembly acceptance | Physically accepted garments / all attempted specimens |
| Rework burden | Rework events, material loss and additional time per accepted article |
| Evidence completeness | Required plan, machine, material, operator and inspection lineage present |

Do not use BLEU, semantic similarity or a fluent natural-language sequence as the correctness metric. Do not pool garment types, machine classes and fabrics without stratified results.

## Adversarial suite

Inject missing/duplicate pattern edges, orphan interfaces, duplicate operations, unknown predecessors, self-loops, multi-node cycles, unsupported machine operations, stale brief/pattern digests, manual operations with machine profiles, automated operations without profiles, missing seam allowances, contradictory notches, inaccessible pressing/turning steps, and unbounded rework.

Required behavior: explicit failure/unknown; no graph is passed by dropping a malformed component.

## Proposed targets (not achieved)

- 100% detection of injected identity/reference/digest/dependency adversarial cases.
- Zero invalid operation graphs admitted to Symtropy's scheduling handoff.
- Zero unknown machine capability treated as supported.
- 100% of accepted operation nodes trace to the exact source pattern edges and rule revision.
- At least 30% fewer expert corrections than a reproduced multi-agent comparator without lowering physical first-pass assembly acceptance.
- At least 20% fewer rework events than a matched sequence-planning baseline in the named overshirt scope.

Freeze stretch targets after task agreement and baseline/power analysis. A green synthetic graph validator is not evidence of physical sewing performance.

## Current P0 implementation

`schemas/sewing-assembly-plan.schema.json`, `examples/sewing-assembly-plan.overshirt.example.json`, and `tools/atelier-sewing-plan-core/` establish a versioned JSON contract, synthetic fixture and deterministic Rust graph validator. Eighteen Rust tests are authored, including adversarial cases for irreversible-step review/evidence, unsupported machine-profile basis and incomplete interface membership. They have not been compiled/executed yet. The first implementation validates structural references, digest binding, simple interface bounds, dependency validity and declared machine operation support. It does not yet validate geometric edge correspondence, seam-length/notch compatibility, operation access or physical construction.
