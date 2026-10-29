# Validation

Research validation utilities and external-reference comparisons.

## PyPhi comparison scaffold — currently not qualified

`pyphi_crossvalidation.py` is an exploratory comparison scaffold, **not a qualified validation of IIT Phi**.

The current paths do not establish a common causal model: the HDC calculation uses node representations, while the PyPhi calculation constructs a transition-probability matrix from an OR-of-neighbours rule. The PyPhi formalism and package version are also not explicitly pinned by this script. A high correlation between these outputs would not make them equivalent measurements.

The script therefore reports descriptive values only and always marks the result `NOT_QUALIFIED`. Failed process runs, timeouts, invalid JSON, missing fields, and non-finite values remain failures; there are no topology-based numeric fallbacks.

### Usage

Run in the Python research shell:

```bash
nix develop .#python-research
python validation/pyphi_crossvalidation.py --output PYPHI_COMPARISON_REPORT.md
python -m unittest validation.test_pyphi_crossvalidation
```

### Current acceptance blockers

1. Define one explicit causal transition model, including its TPM and analysed state, and use it on both sides.
2. Pin the exact PyPhi package version and named IIT formalism/configuration.
3. Make the Rust adapter emit a verifiable model/state digest and a versioned metric contract.
4. Add deterministic known-result fixtures and independent reviewers for the mapping between the HDC quantity and the PyPhi quantity.
5. Only report a correlation when both finite series vary and at least three paired cases exist; correlation remains descriptive and cannot independently qualify the method.

Do not cite generated descriptive metrics as evidence that an HDC proxy is IIT Phi. This scaffold can help expose integration failures; a qualified scientific comparison requires the acceptance blockers above to be resolved first.
