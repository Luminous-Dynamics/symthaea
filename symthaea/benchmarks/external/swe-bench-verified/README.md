# SWE-bench Verified (Mini)

SWE-bench Verified evaluates real coding tasks using unit tests. We use the
Mini subset for CI-level cost and action-budget validation.

## Data Layout

Place the mini subset here:

```
benchmarks/external/swe-bench-verified/data/
  mini.json
```

## Fetch

```
./fetch.sh
```

This is a manual download step.

## Run

```
./run.sh
```

Results should be written to:

```
benchmarks/external/results/swe-bench-verified-mini.json
```

## Notes

- Run with ActionIR budgets enforced.
- Use the local LLM only for translation (no API calls).
- Set `SYMTHAEA_SWEBENCH_RUNNER` to point to an external harness command, or
  `SYMTHAEA_SWEBENCH_RESULT_JSON` to wrap precomputed results.
- For the official harness, set:
  - `SYMTHAEA_SWEBENCH_HARNESS_CMD`
  - `SYMTHAEA_SWEBENCH_HARNESS_ARGS`
  - `SYMTHAEA_SWEBENCH_HARNESS_RESULT_JSON` (optional file output)
