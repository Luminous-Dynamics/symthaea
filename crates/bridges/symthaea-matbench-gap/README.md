# symthaea-matbench-gap

Pinned, network-free truth adapter for Matbench `matbench_expt_gap`.

The adapter does not download data. A caller supplies the compressed upstream artifact bytes; the official entry point accepts them only when their SHA-256 exactly matches the matminer metadata pin for `matbench_expt_gap`.

## Pinned upstream identity

- dataset: `matbench_expt_gap`
- expected rows: 4,604
- input: composition
- target: `gap expt`
- unit: eV
- URL: `https://ml.materialsproject.org/projects/matbench_expt_gap.json.gz`
- SHA-256: `783e7d1461eb83b00b2f2942da4b95fda5e58a0d1ae26b581c24cf8a82ca75b2`

These values are taken from matminer's dataset metadata. The parser also enforces matminer's split-dataframe shape (`data`, `columns`, `index`) and the exact two columns `composition` and `gap expt`.

## Composition handling

Matminer serializes composition-bearing data through Monty/pymatgen. The adapter therefore accepts either:

- a plain formula string; or
- a JSON object mapping element symbols to positive numeric amounts, matching `pymatgen.core.Composition.as_dict()` semantics.

Every composition is converted to a phase-insensitive atomic-fraction fingerprint. Fractions are normalized and conservatively quantized at 1e-9 before identity comparison. This intentionally treats polymorph labels with the same stoichiometry as overlapping chemistry for leakage detection.

The same fingerprinting is applied to `symthaea-bandgap::training_data::load_training_data()`. Thus a Matbench `SiC` composition collides with Symthaea training entries such as `SiC-4H`/`SiC-6H` even though their display strings differ.

Passing the overlap check establishes only that no composition fingerprint in the supplied truth artifact occurs in the currently exposed Symthaea band-gap training table. It does not prove absence of all historical prior knowledge used to tune hand-written baselines.

## What is checked

Before decompression, the official path checks the compressed artifact digest. After bounded decompression it checks:

- exactly 4,604 rows;
- exactly 4,604 index entries;
- exact column names/order;
- each row has exactly two values;
- valid composition representation;
- known element symbols;
- finite positive element amounts;
- finite, non-negative experimental gaps;
- unique dataframe indices;
- no duplicate normalized compositions.

The returned truth set preserves the official source URI and compressed-artifact digest. Canonical row order receives its own domain-separated digest.

## Licensing disclosure

The pinned matminer dataset metadata identifies provenance, source, artifact type, row count, URL, and hash but does not declare a dataset-specific license for `matbench_expt_gap`. The adapter records that as unknown rather than inventing one. Users remain responsible for complying with upstream dataset terms and citations.

## Dependency note

Gzip decoding uses Rust `flate2`. This is intentionally the only new third-party dependency in this bridge. The PR must not be considered qualified until the repository lockfile is reconciled and exact-head Cargo/Clippy/tests pass; an uncommitted resolver update is not evidence.

## Authority boundary

This crate verifies and parses benchmark truth. It performs no network access, model training, candidate ranking, experiment, material certification, novelty determination, procurement, or deployment action.
