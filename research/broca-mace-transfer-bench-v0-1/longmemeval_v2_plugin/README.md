# MACE LongMemEval-V2 official-harness bridge v0.1

**Status:** protocol adapter source and Rust JSONL sidecar; authored Rust tests remain unexecuted. No LongMemEval-V2 benchmark run or score is claimed.

## Scope

This is the first runnable-interface bridge for the official LongMemEval-V2 `Memory` API. It consists of:

- `mace_lmev2_memory.py`: a thin Python `Memory.insert/query` shim registered with the benchmark's `memory_modules.memory.register_memory`.
- `mace_lmev2_stdio.rs`: a persistent Rust JSONL sidecar that owns trajectory parsing, BM25 indexing, neighbor expansion and context-item construction.
- `longmemeval_v2.rs`: source-preserving trajectory/state adapter and screenshot-path/digest checks.

The first arm is **text BM25 retrieval with original screenshot evidence returned for selected states**. It accepts the official optional `query_image` argument but intentionally does not use the question-image pixels in ranking. It is not a multimodal retrieval result and must not be submitted or described as such.

## Pin the official benchmark contract

The adapter was checked against official upstream LongMemEval-V2 commit:

`xiaowu0162/LongMemEval-V2@2cc8c540bdb87fe6761629b585e727e1c4704520`

That revision defines:
- `Memory.insert(trajectory)`
- `Memory.query(query, query_image=None)`
- context items with exactly `type: "text"|"image"` and `value: string`
- harness-side validation and image loading
- an opaque per-query invocation ID, rather than question IDs, gold answers or evaluator labels, as the backend-visible query context.

Before a benchmark run, re-pin the official repository revision and verify the API has not changed.

## Build

From the Symthaea checkout:

```sh
cargo build --release \
  --manifest-path research/broca-mace-transfer-bench-v0-1/aml-adapter/Cargo.toml \
  --bin mace_lmev2_stdio
```

The resulting binary must be passed by absolute path as `sidecar_binary`.

## Install the thin plugin

Copy `mace_lmev2_memory.py` into the official checkout's `memory_modules/` directory. Add the following import alongside the built-in memory imports in that checkout's `memory_modules/memory.py`:

```python
from .mace_lmev2_memory import MaceLongMemEvalV2Memory  # noqa: E402,F401
```

The official harness then discovers `mace_lmev2_bm25_text` through its normal memory registry.

## Example memory configuration

```json
{
  "memory_type": "mace_lmev2_bm25_text",
  "memory_params": {
    "sidecar_binary": "/absolute/path/to/mace_lmev2_stdio",
    "source_screenshot_root": "/absolute/path/to/trajectory_screenshot_assets",
    "workspace_dir": "/absolute/path/to/mace_lmev2_workspace",
    "domain": "web",
    "run_id": "longmemeval-v2-web-small-run-001",
    "top_k": 8,
    "slice_radius": 1,
    "neighbor_radius": 1,
    "bm25_k1": 1.2,
    "bm25_b": 0.75
  }
}
```

Use a separate clean workspace and unique `run_id` for each benchmark execution. The configured screenshot root must contain all source screenshot files. `memory_params` must also provide a unique `run_id` for the exact benchmark execution. The bridge computes `backend_revision_sha256` from the exact sidecar executable bytes; this digest is passed into the Rust sidecar rather than supplied by the retrieval backend itself. The shim rejects absolute or relative screenshot references that resolve outside that root, copies selected source bytes into a private workspace, and records both the canonical source-trajectory digest and each copied screenshot's SHA-256.

For ingress, the plugin pins the configured screenshot root with a directory descriptor. It opens each intermediate directory and the final file relative to that descriptor using no-follow semantics; it rejects symlink replacement at both file and directory components. It reads through the opened file descriptor, requires a regular file, applies the byte cap while reading, and compares file identity, size, modification time and change time before/after the read. This closes the ordinary path-check/read symlink-swap window. The configured root itself must still be trusted, and the official reader's later open remains a separate boundary protected by the private evaluation workspace and sidecar pixel-digest verification.

The Rust sidecar checks screenshot path containment and verifies the copied screenshot bytes against the digest embedded at ingestion before returning an image context item. The official reader still performs its own image opening and context-token accounting.

## Evidence and privacy boundaries

- The sidecar accepts only `insert` and `query` JSONL messages; its stdout is reserved for one JSON response per request.
- The Python bridge serializes access to the persistent sidecar and registers cleanup at interpreter exit.
- Question IDs, evaluator functions, gold answers and hidden benchmark metadata are not passed to the sidecar.
- `query_image` is accepted by the official API shim but ignored by this text-only baseline.
- The backend returns absolute local image paths in official `{"type":"image","value":"..."}` context items.
- Screenshot source directories and the workspace should be immutable or access-controlled during a run. Digest verification detects post-ingest byte changes before context return; it does not make a mutable filesystem race-free.
- Stored memory state includes normalized trajectory JSON and copied image assets so the benchmark's save/load flow can replay inserts.

## Candidate-evidence integration

Each official query must carry the harness's opaque `query_invocation_id` from `Memory.get_query_context()`. The bridge uses this only as `candidate_id`; question IDs, categories, gold answers and evaluator labels remain unavailable to the backend.

The Rust sidecar now emits a canonical MACE candidate-evidence JSONL envelope alongside the official reader context:
- each BM25 text hit is a text candidate bound to the exact trajectory SHA-256 and the raw state SHA-256 parent IDs;
- the deterministic state-window/chunk digest is recorded as the derivation receipt;
- each attached screenshot is a separate image candidate and is rejected if it lacks its own pixel SHA-256;
- temporal neighbor expansion is represented on a separate retrieval channel with no inherited BM25 score;
- images repeated across overlapping windows are deduplicated by trajectory, state ordinal and image digest.

The envelope binds `run_id`, the opaque query invocation ID, exact query SHA-256, domain scope, backend identifier and the SHA-256 of the exact sidecar executable. The Python bridge checks these fields and persists the JSONL plus a receipt under `workspace_dir/candidate_evidence/` before returning context to the reader. Files are create-once, owner-only and invocation-keyed; a repeated invocation ID that produces different evidence is rejected. Empty-retrieval queries also receive a receipt with a zero candidate count.

This is structural provenance and reproducibility evidence, **not** issuer authentication, semantic truth, rights authorization or lifecycle qualification. The payload references are logical `lmev2://` resolver keys, not direct file paths; the original reader context continues to use the controlled local screenshot paths.

## Retrieval and accounting

The sidecar exposes build and search timings, hit counts and neighbor counts in its internal response. The official harness's measured `Memory.query` wall-clock includes RPC overhead, lazy index construction on the first query, screenshot digest verification and context assembly.

The existing lexical resource receipt counts source JSON/text and screenshot references; it does **not** claim full RSS, copied screenshot storage, allocator overhead, reader/judge cost or infrastructure TCO. Those must be collected by the benchmark runner.

## Validation boundary

The complete Rust adapter test command is:

```sh
cargo test --manifest-path research/broca-mace-transfer-bench-v0-1/aml-adapter/Cargo.toml
```

Eleven Python ingress/protocol tests are provided in `tests/test_mace_lmev2_memory.py` for byte-preserving image copy/digest binding, canonical original-trajectory preservation, absolute/traversal/symlink escape rejection, file and intermediate-directory symlink swaps after path validation, screenshot size limits, domain isolation, fail-closed behavior after a sidecar response is lost, run/query/backend-bound candidate evidence persistence, immutable invocation-ID conflict detection, and zero-result query receipts. An ambiguous RPC outcome poisons the sidecar process; the plugin refuses further operations until the memory is reloaded/reconciled instead of silently restarting an empty sidecar. The Rust sidecar has three binary tests for official context-item output, duplicate inserts and screenshot-byte substitution. The candidate-evidence module has thirteen protocol tests, and the V2 candidate adapter has five tests for text/image lineage, overlapping-image deduplication, missing digests, invalid score and wrong-domain rejection, plus neighbor expansion without BM25 score inheritance. Copy this test file into the official repository's `tests/` directory after installing the plugin. The prescribed command is:

```sh
python -m unittest tests.test_mace_lmev2_memory
```

The tests are authored but have not been executed here. Rust sidecar and the modified trajectory parser have authored source but no Rust compile/test/Clippy pass is claimed here.
- The Python shim has not been exercised against the official harness in this environment.
- The official V2 repository specifies Python 3.11, Qwen3.5-9B as the paper reader, Qwen3-Embedding-8B for embedding baselines, and gpt-5.2/medium as the default judge; these pins must be held constant for matched comparisons.
- No dataset download, official run, answer-accuracy result, LAFS value or SOTA comparison is claimed.
- Before qualification, run the official unit tests and privacy tests, add MACE-specific integration tests, and execute the official small split with the pinned reader, judge, prompts and resource limits.
