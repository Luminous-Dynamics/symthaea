# Broca MIRE reference harness

This dependency-free Rust harness implements the first executable slice of [issue #8380](https://github.com/Luminous-Dynamics/symthaea/issues/8380): deterministic memory-event fixtures, an evaluator-owned event log, simple reference systems, deliberately faulty controls, per-query outcomes, and a reproducible JSONL receipt.

It is **not** a model benchmark. It does not load Liquid Broca, HDC embeddings, Mamba, or CUDA. It checks that the fixture/scorer can detect selected classes of memory failure before coupling the tests to production retrieval APIs.

## Run

From the repository root, with a Rust toolchain installed:

```sh
cargo test --manifest-path research/broca-mire/Cargo.toml
cargo run --manifest-path research/broca-mire/Cargo.toml --release -- \
  --source-sha "$(git rev-parse HEAD)" \
  --seed 7 --episodes 64 --distractors 32 --capacity 16 \
  > mire-reference.jsonl
```

The source SHA is required. Use the exact commit being evaluated; a benchmark run without a source revision refuses to emit a result receipt. The command is dependency-free, needs no pretrained model, and does not use CUDA.

## Output

Output is deterministic newline-delimited JSON. It contains a run record, one record per evaluated query/system, and task-family summaries per system. Values are serialized as decimal strings to avoid consumer precision loss for 64-bit values.

The `fixture_fnv1a64` field is a stable, **non-cryptographic fixture fingerprint**, not a SHA-256 provenance or integrity proof. The exact source commit is recorded separately; replace or supplement the fingerprint with the repository's canonical cryptographic dataset digest when this fixture is integrated into the production benchmark pipeline.

## Systems

- `exact_versioned_reference`: unbounded reference map that rejects stale versions and retains deletion tombstones.
- `bounded_lru_versioned`: bounded LRU that preserves versions only while the key/tombstone remains resident. It makes the capacity-versus-tombstone-retention trade-off observable.
- `bounded_payload_persistent_ledger`: bounded payload cache plus a separate unbounded version/deletion metadata ledger. This is a diagnostic design arm: it should prevent stale-value resurrection, but may abstain after payload eviction. Its metadata is unbounded, so it is not a fair equal-total-memory result yet.
- `mutation_namespace_blind`: deliberately faulty control that drops the episode namespace; expected to leak same-key values across namespaces.
- `mutation_version_ignoring`: deliberately faulty control that accepts old versions; expected to fail stale-update and deletion-replay cells.

The two mutation controls are **not competitive baselines**. They exist to prove that the evaluator catches representative bugs. A control that fails as expected does not mean the harness is fully mutation-complete.

The two capacity-bounded designs intentionally expose different trade-offs: plain LRU can forget both payload and version/tombstone state, while the persistent-ledger arm retains control metadata but not necessarily payload. The latter's metadata grows with the number of unique keys and must be accounted for before any equal-memory claim. Its current purpose is to test correctness separation, not to assert that an unbounded ledger is a deployable design.

## Boundaries

This first slice covers retention under interference, stale update rejection, delete/replay behavior, and episode isolation. It does not establish vector retrieval quality, semantic memory utility, language quality, privacy erasure, secure deletion, throughput superiority, or state-of-the-art performance. Those require integration with the actual HDC store / MemoryBridge and labelled task-level evaluation. See [MIRE protocol PR #8377](https://github.com/Luminous-Dynamics/symthaea/pull/8377), [MemoryBridge semantics issue #8378](https://github.com/Luminous-Dynamics/symthaea/issues/8378), and [selective intervention issue #8392](https://github.com/Luminous-Dynamics/symthaea/issues/8392).
