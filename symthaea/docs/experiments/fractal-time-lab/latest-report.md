# Symthaea Fractal Time Lab Benchmark Report

Version: `fractal-time-lab-v0.5`

Epistemic status: `EXPLORATORY_BENCHMARK_NOT_PHYSICAL_PROOF`

Configuration: `seed=42`, `trials=32`

## Summary

- Scorecards: 5
- All benchmark thresholds passed: `false`

## Scorecards

| Experiment | Score | Null mean | Null std | Effect size | Threshold | Passed |
|---|---:|---:|---:|---:|---:|---:|
| Hofstadter-HDC cross-scale similarity | 0.509644 | 0.533744 | 0.023547 | -1.023494 | 2.000000 | false |
| Persistent 2T response | 0.996717 | 0.004001 | 0.009206 | 107.837748 | 2.000000 | true |
| Multi-scale integration survival | 0.445500 | 0.426703 | 0.037333 | 0.503503 | 1.000000 | false |
| Box-covering integration survival | 0.739301 | 0.403818 | 0.085461 | 3.925576 | 0.500000 | true |
| Greedy box-dimension diagnostic | 0.602152 | 1.328111 | 0.241744 | -3.003007 | 0.500000 | false |

### Hofstadter-HDC cross-scale similarity

- Hypothesis: Related Harper slices preserve HDC similarity better than random/jittered/smooth spectra.
- Caveat: Exploratory: uses Harper slices and HDC quantization, not full experimental Hofstadter data.
- Passed: `false`

### Persistent 2T response

- Hypothesis: A DTC-like toy model shows persistent subharmonic response compared with damped and random controls.
- Caveat: Exploratory: classical Floquet surrogate, not a quantum many-body simulation.
- Passed: `true`

### Multi-scale integration survival

- Hypothesis: Hierarchical modular graphs preserve an EI/Phi proxy across spectral coarse-graining better than random controls.
- Caveat: Exploratory: EI/Phi proxy and spectral coarse-graining, not full IIT Phi.
- Passed: `false`

### Box-covering integration survival

- Hypothesis: A path/tree-like graph preserves EI/Phi proxy across greedy box covering differently than random graph controls.
- Caveat: Exploratory: greedy graph-radius boxes, not optimized minimum box covering.
- Passed: `true`

### Greedy box-dimension diagnostic

- Hypothesis: Tree-like graph has a stable greedy box-count scaling signal compared with random graph controls.
- Caveat: Exploratory: greedy cover dimension is heuristic and sensitive to graph family and radius range.
- Passed: `false`

## Claim Scope

Honest claims:

- This is an exploratory computational benchmark lab.
- The scorecards compare toy structural hypotheses against explicit null models.
- Passing a scorecard means the benchmark threshold was met for that seeded run.

Non-claims:

- This does not prove fractal time.
- This does not prove quantum consciousness.
- This does not simulate a physical quantum many-body time crystal.
- This does not compute full IIT Phi.
- This is not production scientific validation.
