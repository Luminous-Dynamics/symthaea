# HIST-CAUSAL-001 v0.9

Finite-sample composite-null model-adequacy reference.

## Files

- `composite_null_test.py`: NumPy-vectorized Monte Carlo rank p-values and intersection-union decision.
- `test_composite_null_test.py`: nine tests for rank validity, ties, coherent joint likelihood across probes, probability normalization, deterministic replay and fail-closed inputs.
- `manifest.json`: configuration and guarantee assumptions.
- `results.json`: aggregate development results.
- `../../v0.5/generator.py` and `../../v0.6/ood_falsification.py`: pinned synthetic recipe and OOD-mixture references.
- `../../../docs/research/HIST_CAUSAL_003_COMPOSITE_NULL_V09.md`: method, proof sketch, results, limits and benchmark comparisons.

## Run

From the repository root:

```bash
python -m unittest -v benchmarks/research/hist_causal_001/v0.9/test_composite_null_test.py
python benchmarks/research/hist_causal_001/v0.9/composite_null_test.py --seeds 24 --calibration 5000 --evaluation 2000 --n 1 2 5 10 20 --output results.json
```

NumPy is required for this research reference. Production implementation and independent verifier are tracked separately in HIST-CAUSAL-002 / #10081.

## Validity boundary

The intersection-union argument controls false `model_misspecified` declarations at nominal alpha for each declared in-family model only when recipe point parameters are correct and calibration/evaluation units are independent and exchangeable. Unknown nuisance parameters, persistent confounders and within-campaign dependence are not covered yet. A non-rejection does not prove a candidate model is true.
