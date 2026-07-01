# Validation

Cross-validation with external implementations.

## PyPhi Cross-Validation

`pyphi_crossvalidation.py` validates our Φ calculations against the reference PyPhi implementation.

### Usage

```bash
# Enter the Python research shell
nix develop .#python-research

# Run validation
python validation/pyphi_crossvalidation.py

# Run the lightweight package smoke lane
./scripts/ci_check_lanes.sh python-research

# Results output to results/pyphi_validation_results.csv
```

### What It Tests

1. **Φ values** - Compare our algebraic connectivity Φ with PyPhi's exact IIT Φ
2. **Topology consistency** - Verify topologies produce expected patterns
3. **Edge cases** - Test boundary conditions

### Results

See `results/pyphi_validation_results.csv` for comparison data.

Our implementation uses algebraic connectivity as a Φ proxy, which correlates with but differs from PyPhi's exact computation.
