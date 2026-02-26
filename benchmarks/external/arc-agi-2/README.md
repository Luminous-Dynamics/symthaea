# ARC-AGI-2

ARC-AGI-2 evaluates generalization and efficiency on abstract reasoning tasks.

## Data Layout

Place the dataset here:

```
benchmarks/external/arc-agi-2/data/
  training/
  evaluation/
```

Each directory should contain JSON tasks with the standard ARC format.

## Fetch

```
./fetch.sh
```

This is a manual download step (see script output for instructions).

## Run

```
./run.sh
```

Results are written to:

```
benchmarks/external/results/arc-agi-2.json
```

## Notes

The runner uses `examples/benchmark_arc_reasoning.rs` and sets:

- `SYMTHAEA_ARC_DATA_DIR`
- `SYMTHAEA_ARC_RESULTS_PATH`
