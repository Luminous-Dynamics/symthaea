# ARC-AGI-2 Dataset Path

This document describes how Symthaea consumes the ARC-AGI-2 public evaluation set. **The dataset is NOT vendored in this repository** — you must download it yourself from the upstream source.

## Upstream source

- Project page: <https://arcprize.org/>
- ARC-AGI-2 task page: <https://arcprize.org/arc-agi/2>
- GitHub (tasks, licensing, format): <https://github.com/fchollet/ARC-AGI>
  - ARC-AGI-2 lives in the same repository as ARC-AGI-1 under a sibling directory once released publicly. Check the repo README for the current canonical path; the layout has changed across releases.

## License

ARC-AGI-1 and ARC-AGI-2 are released by François Chollet and the ARC Prize organization under the terms stated in the upstream repo (historically Apache-2.0 for the 2019 release). **Verify the current license text in the upstream repo before using.** Symthaea's AGPL-3.0-or-later license applies only to this repository's code, not to the ARC dataset.

## Format

Each task is a JSON file:
```json
{
  "train": [{"input": [[...]], "output": [[...]]}, ...],
  "test":  [{"input": [[...]], "output": [[...]]}]
}
```
Grids are 2D arrays of integers 0–9 (color indices). Sizes are variable, typically ≤ 30×30.

ARC-AGI-2 public evaluation set contains 120 tasks. Two-attempt scoring convention (same as ARC Prize 2024–2026): each test input allows up to 2 predicted outputs, task scored correct if any prediction matches.

## Where to place the downloaded data

Symthaea reads the dataset path from the `SYMTHAEA_ARC2_DATA_DIR` environment variable. A typical layout:

```
<somewhere-on-your-disk>/arc-agi-2/
├── evaluation/
│   ├── 001.json
│   ├── 002.json
│   └── ...
└── README.md  (upstream)
```

Set the env var before running the benchmark:

```bash
export SYMTHAEA_ARC2_DATA_DIR=/absolute/path/to/arc-agi-2/evaluation
```

## Running the benchmark

The Phase-1 benchmark harness (to be added in Week 1–2 of the math/science completeness plan) will be at `examples/benchmark_arc_agi2.rs`. Planned invocation:

```bash
SYMTHAEA_ARC2_DATA_DIR=/path/to/arc-agi-2/evaluation \
  cargo run --release --example benchmark_arc_agi2 > arc_agi2_results.csv
```

The existing `examples/benchmark_arc_reasoning.rs` targets ARC-AGI-1 via the separate `SYMTHAEA_ARC_DATA_DIR` variable. Both benchmarks will coexist — the env var split keeps their datasets independent.

## Honesty disclaimer

The current Symthaea pipeline (`GridEncoder` + rule-vector similarity) scored single digits on ARC-AGI-1 training; ARC-AGI-2 is explicitly harder. The Phase-1 target is a **reproducible honest baseline CSV**, not a specific score. See `plans/2-please-make-precious-fairy.md` Workstream C for the measurement-first-then-decide strategy.

Frontier references (2026): Symbolica Agentica reports ≈85% on ARC-AGI-2, Gemini 3.1 Pro ≈77%; most non-agentic, non-synthesis approaches score well below 30%. Symthaea's rule-vector baseline is not expected to approach those figures without the stretch `grid_macro_discovery.rs` integration (Phase 1 go/no-go decision at Week 3 per the plan).

## Not vendored — why

ARC-AGI is actively maintained upstream; vendoring freezes a version and risks license drift. Downloading fresh on first use guarantees you run against the current official tasks.
