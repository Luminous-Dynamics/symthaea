# AI-Newton public artifact audit — SCI-AINEWTON-001

**Status:** source/artifact inventory only; no upstream execution or paper reproduction claimed  
**Pinned upstream:** [Science-Discovery/AI-Newton](https://github.com/Science-Discovery/AI-Newton)  
**Commit:** `c143e865be42e067faf64cd8117cbeffd0fccfb6`  
**Commit date:** 2026-01-27  
**License:** MIT  
**Audit date:** 2026-11-18

## Why the version boundary matters

The inspected repository HEAD is dated **January 27, 2026**, 281 days before the paper's November 4, 2026 publication. No mapping from this commit's outputs to the paper's experiments or reported result table has been established.

The public files are useful for studying the implementation and its native artifact model, but they are not evidence that the paper's November 2026 runs have been reproduced.

## Pinned repository requirements

The upstream README states Linux, CUDA, Python 3.11.9+, Rust 1.84.0 and Maple 2024; it warns that a full example can take several days. The package metadata says Python >=3.10, while the README asks for >=3.11.9. The inspected root files do not pin an exact CUDA version, and the precise Maple dependency/use path has not yet been established.

The example script `test/examples/test_example_1.py` contains 38 `register_experiment` calls, configures 1,200 iterations and 32 threads, and registers mechanics / ground-laboratory / oscillation / celestial configurations. These are **source configuration declarations**, not a verified count of completed experimental trials.

## Inventory results

The manifest records exact Git blob SHAs and byte sizes from the pinned upstream commit.

| Artifact group | Inventory | Notes |
|---|---:|---|
| Numbered example folders | 12 | Numeric folder names are not treated as seed/trial IDs |
| Files in numbered folders | 84 | Seven files per folder |
| Human-operating example files | 5 | Explicitly labelled as a human example |
| Associated log files | 14 | Not one-to-one with the twelve folders |
| Numbered `*_memory.json` files | 386,923,512 bytes | Approximately 369 MiB |
| Associated log files | 120,415,658 bytes | Approximately 115 MiB |

Each numbered folder contains:
- `*_concept_grp_epoch_loss.txt`
- `*_general_model.json`
- `*_knowledge.txt`
- `*_memory.json`
- `*_object_model.json`
- `*_policy_nn.pth`
- `*_specific_model.json`

The log directory contains both `6.log`, `6(1).log` and `6(2).log`. Their relation to the twelve numbered folders is **unknown**. They may represent retries, alternate outputs, or different historical artifacts; the manifest does not guess.

The five files under `data/human/example_output/` are a separate upstream-labelled human-operating example and must not be pooled with autonomous test outputs.

## How to use the manifest

- `manifest.v0.json` pins the upstream commit, selected source files, all 84 numbered output artifacts, 14 logs and five human-example artifacts.
- `manifest.schema.json` defines the basic machine-readable contract.
- Git blob SHAs establish identity within the pinned upstream tree; they do not authenticate when/how an output was produced or whether it corresponds to a paper task.

The next importer should begin with small `*_knowledge.txt`, `*_object_model.json` and `*_specific_model.json` files. It should leave the large memory snapshots, general-model graphs and policy checkpoints opaque until their schemas and size limits are separately qualified.

## Required claim boundaries

1. A native `is conserved` or `is zero` label is an upstream status, **not** an independent mathematical proof.
2. A parsed knowledge file is not a count of independently validated physical laws.
3. Numeric directory names do not prove independent seeds, trials or replicates.
4. Imported output is not a reproduction; a reproduction requires the exact paper task, source, configuration, compute budget, stopping rule, scoring process and run provenance.
5. No AI-Newton vs Symthaea comparison or SOTA statement is made by this inventory.

Related work: Symthaea issue [#12808](https://github.com/Luminous-Dynamics/symthaea/issues/12808).
