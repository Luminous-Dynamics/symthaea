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

## What the JSON captures

- source commit and repository-relative SHA-256 fingerprints for `README.md`, `CONTENTS.md`, `lean/formalization.yaml`, the Comparator scope notes, referenced manuscript PDFs, and present main-result Lean/config artifacts;
- a cross-check between the manuscript/family counts declared in the upstream README and the actual family-header count parsed from `CONTENTS.md`; the indexer fails closed if the family counts disagree;
- result family identifiers/titles and repository-relative links from `CONTENTS.md`;
- a deterministic 70/15/15 train/validation/test assignment at the *related-result family* level, derived from SHA-256 ranking over a fixed seed and family ID, with the seed, algorithm, and exact counts included in the output;
- formalization catalog entries, direct family-link matches, and explicit unmatched source IDs;
- present/missing status and hashes for catalogued source artifacts;
- declared `status.main_results` records and the upstream manifest's stated scope/license label.

Family mapping is a direct source-path match. It is intentionally not semantic matching, and an unmatched item is preserved in the output instead of being silently dropped. The generated split is a *pre-screen assignment*, not a qualified benchmark: screen eligibility and verification status without changing assignments, and preserve both excluded and unknown families in the accounting. The split prevents direct file-level leakage inside a declared family; it does not prove there are no mathematical dependencies or model pretraining exposure across different families. Those risks require separate audit and disclosure. Missing formalization source files and missing declared main-result artifacts are also explicit in the summary. Artifact paths that escape the checkout or resolve through a symlink outside the checkout are rejected. The scope notes are hashed as evidence about what the catalog's verification setups establish; they are not interpreted as proof of the headline theorem.

## Checks

Unit tests cover family extraction, duplicate-family rejection, path normalization, and parsing a minimal manifest. On source format changes, the indexer fails closed for an unknown manifest version, empty source catalog, or absent family headers.

The authoring environment used for this initial remote change did not have a local Rust toolchain or direct GitHub network access, so **these Rust tests have not been executed here**. Do not describe this commit as compiled or tested until the tests run locally or in CI. The separate schema file likewise has only been JSON-parsed and structurally smoke-checked, not validated by an independent JSON Schema engine.
