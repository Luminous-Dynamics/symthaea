# NATS-Bench-MCU v2 export audit — Rust v0.1

Tracking: #9683 / PR #9707  
Dataset: NATS-Bench-MCU, Zenodo DOI 10.5281/zenodo.21644470, version 2.0  
Implementation: `symthaea-architecture-engineering::nats_bench_mcu_audit`

## Scope

The auditor validates the corrected *publication CSV export*. Zenodo v2 states that these CSVs already reflect the corrections, so this path does not merge the separate raw-artifact correction overlay.

The auditor is not a physical rerun and does not authenticate the download source. It computes SHA-256 over the exact CSV bytes supplied, then commits those digests and the derived audit facts into a domain-separated JSON receipt. The operator must still record the Zenodo download/version and compare the local files with the published record.

This v0.1 path does not parse raw PPK2 Parquet traces, verify the correction overlay, or validate the raw artifact archives. Those remain separate follow-ups.

## V2 invariants encoded

| Invariant | Contract |
|---|---:|
| Topology search-space size | 15,625 |
| Physically profiled rows | 14,281 |
| Flash-infeasible rows | 1,344 |
| Candidate indices | Every integer from 0 through 15,624 exactly once |
| Application-core flash | 1,024 KB, as stated by the dataset |
| Fixed firmware overhead | 277.6 KB |
| Published INT8 flatbuffer ceiling | 746.0 KB |
| Failure taxonomy | Exactly `infeasible` |
| Accuracy join | Exactly one row per measured architecture |

**Do not silently change the cutoff to 746.4 KB.** The arithmetic (1024 - 277.6 = 746.4) is recorded separately from the dataset's stated 746 KB deployability boundary. This validator uses the published boundary of 746.0 KB for classification and retains the underlying flash/firmware values in the receipt. It preserves the dataset's `KB` field naming and does not reinterpret the values as KiB.

## Required CSV columns

### hw_metrics.csv

Required columns:

- `arch_idx`, `arch_str`, `int8_kb`;
- `rom_kb`, `ram_kb`;
- `latency_s`, `current_uA`, `power_mW`, `energy_mJ`;
- `board`.

All hardware metric values used by v0.1 must be finite and positive. Additional source columns (for example, arena metrics, operator counts, memory-category JSON) are not rejected; their names are retained in the audit receipt so schema extensions are visible. They are not yet interpreted as objectives by this module.

### accuracies.csv

Required columns:

- `arch_idx`;
- `cifar10`, `cifar100`, `ImageNet16-120`.

Accuracy values must be finite percentages in [0, 100]. Indices must match the measured hardware index set exactly. Missing or extra accuracy records are not imputed.

### failures.csv

Required columns:

- `arch_idx`, `int8_kb`;
- exactly one of `stage` or `outcome`, with value exactly `infeasible`.

Infeasible model sizes must be strictly greater than 746.0 KB. A successful hardware row must be no greater than 746.0 KB.

## Pareto reconstruction

The module constructs two-objective fronts with accuracy maximized and one hardware cost minimized. It produces 12 fronts:

- CIFAR-10, CIFAR-100 and ImageNet16-120;
- each crossed with latency, energy, ROM and RAM.

Ties with identical cost and accuracy are retained. Each front in the receipt contains both its count and sorted architecture indices. This is not a joint four-cost/five-objective Pareto set; the complete candidate rows are returned separately so a later benchmark layer can reconstruct higher-dimensional fronts using explicit objective definitions and normalization.

## Receipt content

The deterministic receipt includes:

- dataset DOI and v2 version;
- exact measured and infeasible index lists;
- declared flash, firmware-overhead and model-ceiling values;
- SHA-256 of the exact input bytes for each CSV;
- names of extension columns observed in each table;
- all 12 Pareto-front index lists;
- a domain-separated SHA-256 over the canonical receipt with its digest field cleared.

This is a content-integrity receipt for the *inputs supplied to the parser*. It does not claim cryptographic provenance from Zenodo or authenticate the publisher.

## Command

From the repository root:

```bash
cargo run -p symthaea-architecture-engineering --bin nats_bench_mcu_audit -- \
  hw_metrics.csv accuracies.csv failures.csv audit-receipt.json
```

A successful command exit means the supplied CSVs satisfy the encoded structural contract. It does not mean the model binaries, raw power traces, compiler outputs, or physical devices have been independently verified.

## Authored negative controls

The Rust test module currently includes synthetic controls for:

- exhaustive partition and receipt digest construction;
- unknown extension-column visibility;
- exact 746.0 KB boundary acceptance;
- measured candidates above the published threshold;
- infeasible candidates at or below the threshold;
- duplicate architecture IDs;
- accuracy IDs substituted with an infeasible candidate;
- non-finite hardware objectives;
- failure-taxonomy substitution;
- unsupported Pareto dataset selection.

Tests are authored but have not been compiled or executed in this environment because Cargo/rustc are unavailable. No dataset row count or Pareto value has been empirically reproduced here.

## Source authority and stale repository documentation

The current Zenodo v2 record is the version authority. The public dataset GitHub README still contains version-1 counts (14,279 successful / 1,346 rejected) in parts of its historical overview, while Zenodo v2 explicitly updates the corrected partition to 14,281 / 1,344. The v2 export README records the corrected outcome taxonomy and boundary. Do not treat the repository's stale v1 summary as the v2 result.

Sources:

- Zenodo v2 record: https://zenodo.org/records/21644470
- Dataset code and export documentation: https://github.com/ankilab/NATS-Bench-MCU
- NATS-Bench-MCU benchmark paper: AutoML 2026 (linked from the Zenodo record)
