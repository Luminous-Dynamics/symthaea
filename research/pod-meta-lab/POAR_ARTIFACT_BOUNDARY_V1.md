# POAR Planner / Oracle Artifact Boundary v1

**Status:** design contract; implementation tracked by [Issue #9360](https://github.com/Luminous-Dynamics/symthaea/issues/9360)  
**Date:** 2026-10-31  
**Scope:** research-only synthetic POAR; not yet an externally deployable hidden benchmark.

## 1. Security and trust boundary

The current `GraphSuite` has two logically separated views:
- planner-visible `PlannerCase` values, available through `planner_cases()`;
- private oracle labels stored inside the same process.

That is a useful API invariant, **not a security boundary**. A planner with access to the same process, executable memory, or the oracle-bearing artifact can recover the labels. The current public generator/templates are also exposed to repository readers and future training corpora.

The confirmatory design therefore uses three files and two execution domains:

| Artifact | Contents | Reader |
|---|---|---|
| `planner.poar` | Cases, evidence metadata/assertions, query profile, proof alternatives and dependency graph; no hidden truth or expected answer | Planner / model runner |
| `oracle.poar.sealed` | Source-truth labels, gold support/refutation/contestation, gold evidence-path labels, generator seed and planner digest | Evaluator only |
| `predictions.poar` | Model dispositions, selected proof alternatives, ranked retrieval list, citations, model/config metadata; no oracle data | Evaluator only |

**Domain A (planner):** receives only `planner.poar`; no read permission, mount, environment variable, network endpoint, or log containing the oracle.  
**Domain B (evaluator):** receives the immutable planner artifact, sealed oracle, and a frozen prediction artifact after inference has ended. The evaluator must not run during inference and must not provide per-item oracle feedback to the planner.

The current generator still creates both files in one process, so this is **not** secure evaluator isolation. A later protected evaluation service or separately permissioned job must prevent the planner from receiving the oracle bytes, key, path, or endpoint.

## 2. Canonical record encoding

Version 1 uses a deterministic UTF-8, LF-terminated, tab-separated record stream (`POAR-TSV/1`) to avoid a runtime serialization dependency.

Canonical rules:
1. First record is the artifact header and contains format version, generator revision, and split/seed binding.
2. One record per line. Fields are separated by a single TAB. Embedded TAB, CR, LF, NUL, and all free-form UTF-8 string fields are encoded as lowercase hexadecimal UTF-8 bytes.
3. Enum values use a closed lowercase vocabulary defined by the schema. Integers are base-10 with no plus sign or leading zero (except `0`). Optional integer values use `-`.
4. Records are sorted lexicographically by their canonical encoded identity fields. Duplicate record identities are rejected.
5. Empty required identifiers, unknown record kinds, unknown enum values, trailing fields, missing fields, non-canonical hex, non-UTF-8 decoded strings, and dangling references fail closed.
6. Final file ends in exactly one LF. CRLF, BOM, and non-UTF-8 input are rejected.
7. Hash the exact raw bytes, not a parsed/re-serialized representation.

### Planner records (minimum set)

| Record | Fields after tag | Meaning |
|---|---|---|
| `HEADER` | schema, generator_git_sha, seed, split | Dataset identity |
| `CASE` | case_id_hex, valid_at, known_at, scope_hex, authority_hex | Query profile |
| `TARGET` | case_id_hex, claim_id_hex | Target claim |
| `CLAIM` | case_id_hex, claim_id_hex | Claim node |
| `ALT` | case_id_hex, claim_id_hex, alternative_id_hex, conclusion, min_direct_source_groups | OR alternative / AND rule |
| `REQ_E` | case_id_hex, claim_id_hex, alternative_id_hex, evidence_id_hex | Direct evidence premise |
| `REQ_C` | case_id_hex, claim_id_hex, alternative_id_hex, child_claim_id_hex, polarity | Dependent claim premise |
| `EVIDENCE` | case_id_hex, evidence_id_hex, source_id_hex, family_id_hex, independence_group_hex, subject_claim_hex, asserted_polarity, valid_from, valid_until, known_from, known_until, scope_hex, authority_hex, integrity, lifecycle | Planner-visible evidence metadata |

The planner artifact must not contain hidden source truth, gold dispositions, expected alternative IDs, oracle scores, or evaluator feedback. A source's asserted polarity is observable evidence metadata and is not treated as its hidden truth.

### Oracle records (minimum set)

| Record | Fields after tag | Meaning |
|---|---|---|
| `ORACLE_HEADER` | schema, planner_sha256, generator_git_sha, seed, split | Binds oracle to one exact planner artifact |
| `TRUTH` | case_id_hex, evidence_id_hex, truth_label | Hidden source content truth: supports assertion / refutes assertion / irrelevant |
| `GOLD_CLAIM` | case_id_hex, claim_id_hex, disposition | Oracle claim state: supported / refuted / contested / unknown |
| `GOLD_PATH` | case_id_hex, claim_id_hex, alternative_id_hex, ordered_or_set_witness_evidence_ids | Valid proof witness and provenance path |

Do not include source truth in the planner artifact, model prompt, planner logs, or prediction receipt.

### Prediction records (minimum set)

| Record | Fields after tag | Meaning |
|---|---|---|
| `PRED_HEADER` | schema, planner_sha256, model_id_hex, checkpoint_digest, prompt_digest, run_config_digest | Frozen evaluation identity |
| `DISPOSITION` | case_id_hex, claim_id_hex, disposition, selected_alternative_id_hex_or_dash | Model's claim decision |
| `RETRIEVED` | case_id_hex, rank, evidence_id_hex | Ranked retrieval output |
| `CITATION` | case_id_hex, evidence_id_hex | Evidence cited in final proof packet |

The evaluator rejects predictions bound to a different planner digest, duplicate case/claim IDs, duplicate ranks, rank gaps if the schema disallows them, unknown evidence IDs, and undeclared target claims.

## 3. Digest and run-receipt binding

Use SHA-256 over exact artifact bytes:
- planner artifact digest;
- sealed oracle digest (never expose to planner);
- prediction artifact digest;
- evaluator executable/source digest;
- generator source revision and schema version.

The evaluator emits a machine-readable run receipt with:
- POAR schema/generator/evaluator revisions;
- planner, oracle, and prediction SHA-256 digests;
- split and seed;
- model/checkpoint identity, quantization, tokenizer, prompt/config digest, sampling settings, tool versions, and inference budget;
- runtime container/image digest, operating system, CPU/GPU model, driver/runtime versions, peak memory, wall-clock time, token/tool counts, and energy measurement method or explicit unavailable status;
- evaluator exit status and individual metric values;
- contamination transparency card version and audit status.

Sign the receipt (e.g. Sigstore/cosign or an organizational key) only after successful evaluation. A hash establishes byte identity, not truth, independence, or contamination absence.

## 4. Process/job isolation requirements

The future qualification pipeline should have these phases:

1. **Generate:** deterministic generator creates planner and sealed oracle artifacts from a private seed; verifies internal references; computes raw SHA-256 digests.
2. **Publish planner:** only the planner artifact and non-sensitive manifest are made readable by the model runner.
3. **Infer:** model runner emits predictions; it cannot access the oracle artifact or evaluator endpoint. Freeze predictions before any score is revealed.
4. **Evaluate:** a separate protected job downloads planner, oracle, and predictions; verifies all digest bindings; executes the evaluator once per frozen prediction artifact.
5. **Report:** publish signed aggregate metrics and benchmark card; keep item-level oracle labels and detailed sealed artifacts private if the benchmark's threat model requires it.

A GitHub Actions artifact in a public repository is **not** a secret store. Do not upload `oracle.poar.sealed` as an ordinary public artifact. Use a private dataset repository, protected object storage, an encrypted artifact whose decryption key exists only in the evaluator job, or an independently operated evaluation service.

## 5. Evaluator invariants

Fail closed if:
- planner/oracle/prediction headers disagree on schema, generator, split, seed, or planner digest;
- any record identity is duplicate or any reference is dangling;
- an oracle label is missing for a generated evidence record or case target;
- a prediction cites an unknown or non-admissible source;
- a proof disposition names a path absent from the oracle;
- a source is out of valid-time / known-time, outside scope/authority, unverified/invalid, or retracted;
- the prediction artifact changes after evaluation begins;
- an unrecognized schema extension is supplied without a version bump.

Never collapse hard integrity failures into average accuracy. Preserve separate counts for false support/refutation, false closure, abstention quality, proof-path validity, source admissibility, temporal coverage/precision, retraction recall, unaffected-claim stability, and resource cost.

## 6. Current implementation gap

Implemented:
- Deterministic planner-only and oracle-only TSV encoders in `graph.rs`.
- `poar-graph-export` binary that writes the two files and binds the oracle header to the planner file's SHA-256 digest using the system `sha256sum` executable.
- Unit tests that assert hidden truth and gold claim/path records are absent from planner output and present in oracle output; non-canonical planner digests are rejected.
- Focused qualification workflow smoke step that verifies the digest binding and checks record separation in temporary runner storage, then deletes the oracle without uploading it.

Still not implemented:
- An independent artifact parser / evaluator process that consumes frozen predictions and the sealed oracle.
- Protected storage or permission boundaries that prevent the planner from reading oracle bytes.
- Signed benchmark run receipts and an externally auditable model evaluation pipeline.

The exporter currently emits both files within one process; it makes the data boundary explicit and testable but is **not itself a secure boundary**. This design document remains the target for the remaining work.

Issue [#9360](https://github.com/Luminous-Dynamics/symthaea/issues/9360) tracks implementation, tests, protected-job design, and acceptance evidence.
