# symthaea-proof-insights

A small, deterministic source-indexing utility for the Proof Insight Transfer research track.

This first version indexes catalog metadata only. It does **not** download manuscripts, extract mathematical proofs, claim a result is correct, or claim that a paper's headline theorem is formally verified.

## Run against a pinned OpenAI math checkout

Clone the public repository and check out the reviewed snapshot:

```bash
git clone https://github.com/openai/math.git /tmp/openai-math
git -C /tmp/openai-math checkout 1581d4cbdd69a0faf34b2089614fe2a814316a6a

cargo run -p symthaea-proof-insights -- \
  /tmp/openai-math \
  1581d4cbdd69a0faf34b2089614fe2a814316a6a \
  ./proof-insight-source-index.json
```

The working tree must be clean, and the checkout's `HEAD` must match the supplied expected commit. This avoids quietly indexing a moving `main` branch or locally edited catalog.

The frozen 372-family split is checked in at `data/benchmarks/proof-insight-transfer-v0/splits.json`. After generating the source index, compare its exact family assignments and partition digest against that artifact:

```bash
cargo run -p symthaea-proof-insights -- \
  --verify-splits \
  ./proof-insight-source-index.json \
  ./data/benchmarks/proof-insight-transfer-v0/splits.json
```

The verifier rejects source-revision, source-fingerprint, split-algorithm/count/digest, duplicate-ID, or family-assignment mismatches. Catalog paths appearing under multiple families also fail closed in the indexer. This confirms agreement with the frozen assignment, not that the split is free of cross-family mathematical-dependency leakage or pretraining contamination.

## Preregistered result analysis

The companion `analyze_pitb_results` binary implements the registered F-vs-validation-selected-baseline comparison. Its strict input contract, content-hash requirements, frozen split checks, validation-only baseline receipt, evidence fields, family-cluster bootstrap, output fields, and limitations are documented in [analysis-contract.md](../../data/benchmarks/proof-insight-transfer-v0/analysis-contract.md).

The analysis tool is deliberately separate from the source indexer and transfer-candidate validator. It will not create result rows, run Lean, or turn a curator annotation into a verified transfer. Only pass `semantic_gate_passed: true` for an arm after the exact target theorem, axiom/dependency policy, statement correspondence, applicable successor checks, and mutation controls have passed and their receipts are retained. Until then, there are no measured transfer results.

## What the JSON captures

- source commit and repository-relative SHA-256 fingerprints for `README.md`, `CONTENTS.md`, `lean/formalization.yaml`, the Comparator scope notes, every present local artifact linked from `CONTENTS.md`, formalization source files, and present main-result Lean/config artifacts;
- a cross-check between the manuscript/family counts declared in the upstream README and the actual family-header count parsed from `CONTENTS.md`; the indexer fails closed if the family counts disagree;
- result family identifiers/titles and repository-relative links from `CONTENTS.md`;
- a deterministic 70/15/15 train/validation/test assignment at the *related-result family* level, derived from SHA-256 ranking over a fixed seed and family ID, with the seed, algorithm, exact counts, and a canonical partition digest included in the output;
- formalization catalog entries, direct family-link matches, and explicit unmatched source IDs;
- a `catalogue_linked_artifacts` inventory preserving every distinct local path linked from `CONTENTS.md`: present files carry exact-byte SHA-256 and byte count, while missing paths remain explicit rows with null digest/size and summary counts;
- one sorted, unique `source_fingerprints` list across catalog metadata, present catalogue links, formalization sources and present main-result artifacts; conflicting fingerprints for one repository path fail closed;
- declared `status.main_results` records and the upstream manifest's stated scope/license label.

Family mapping is a direct source-path match. It is intentionally not semantic matching, and an unmatched item is preserved in the output instead of being silently dropped. The generated split is a *pre-screen assignment*, not a qualified benchmark: screen eligibility and verification status without changing assignments, and preserve both excluded and unknown families in the accounting. The split prevents direct file-level leakage inside a declared family; it does not prove there are no mathematical dependencies or model pretraining exposure across different families. Those risks require separate audit and disclosure. Missing formalization source files, missing declared main-result artifacts and missing local catalogue links are explicit in the JSON output. Artifact paths that escape the checkout or resolve through a symlink outside the checkout are rejected. The scope notes are hashed as evidence about what the catalog's verification setups establish; they are not interpreted as proof of the headline theorem.

## Checks

Unit tests cover family extraction, duplicate-family rejection, path normalization, and parsing a minimal manifest. On source format changes, the indexer fails closed for an unknown manifest version, empty source catalog, or absent family headers.

The authoring environment used for this initial remote change did not have a local Rust toolchain or direct GitHub network access, so **these Rust tests have not been executed here**. Do not describe this commit as compiled or tested until the tests run locally or in CI. The separate schema file likewise has only been JSON-parsed and structurally smoke-checked, not validated by an independent JSON Schema engine.
