# IIT Φ Validation Results

**Date**: January 10, 2026
**Status**: Historical exploratory report — not currently qualification-passed
**Conclusion**: The reported experiment supports only a synthetic topology/proxy observation; it does not establish IIT agreement or consciousness measurement

---

## Executive Summary

This historical experiment compared a graph-spectral HDC metric against a simplified IIT 3.0-like implementation created in-repository. The comparator was not an independent PyPhi oracle, and the HDC metric described below is algebraic connectivity rather than formal IIT Φ. Any reported agreement is limited to the explicitly constructed examples and should be treated as preliminary topology/proxy evidence.

### Key Result

| Ranking | IIT Φ (n=4,5,6) | HDC Φ (n=8) |
|---------|-----------------|-------------|
| #1 | Complete | Ring* |
| #2 | Ring | Torus |
| #3 | Star | Star |
| #4 | Line/Random | Random |

*Complete graph wasn't tested in original HDC validation; Ring was highest among tested topologies.

**Historical claim not established:** the table combines different topology sets and system sizes, so it does not demonstrate a general, method-invariant ordering. Preserve the raw observations as exploratory results; qualify any cross-method ranking claim against matched, independently checked fixtures.

---

## Methodology

### IIT Implementation (Ground Truth)

We implemented a simplified IIT 3.0 Φ calculator:

1. **Transition Probability Matrix (TPM)**: Created from connectivity matrix using majority vote dynamics + noise
2. **Bipartitions**: Enumerated all 2^(n-1) - 1 non-trivial partitions
3. **Information Loss**: Computed KL-divergence-like measure between original and cut TPMs
4. **MIP**: Found the Minimum Information Partition
5. **Φ**: Information at the MIP

### HDC Implementation (Our Method)

From `src/hdc/phi_real.rs`:

1. **Similarity Matrix**: Cosine similarity between node HDC representations
2. **Graph Laplacian**: L = D - S (degree matrix minus similarity)
3. **Φ**: Second smallest eigenvalue (algebraic connectivity / Fiedler value)

### Key Difference

| Aspect | IIT | HDC |
|--------|-----|-----|
| Complexity | O(2^n × n!) | O(n² + n³) |
| Measures | Information loss at MIP | Algebraic connectivity |
| Tractable for | n ≤ 8 | n ≤ 10,000+ |

---

## Results

### n=4 Nodes (Primary Test)

| Topology | IIT Φ | Rank | MIP |
|----------|-------|------|-----|
| Complete | 0.4000 | 1 | [0] \| [1,2,3] |
| Ring | 0.3000 | 2 | [0,1] \| [2,3] |
| Star | 0.1778 | 3 | [0,1,2] \| [3] |
| Line | 0.1250 | 4 | [0,1] \| [2,3] |
| Random | 0.1250 | 5 | [0,1] \| [2,3] |

### Scaling Behavior (n=4,5,6)

| Topology | n=4 | n=5 | n=6 | Trend |
|----------|-----|-----|-----|-------|
| Complete | 0.400 | 0.320 | 0.267 | ↓ |
| Ring | 0.300 | 0.213 | 0.163 | ↓ |
| Star | 0.178 | 0.125 | 0.096 | ↓ |
| Line | 0.125 | 0.093 | 0.074 | ↓ |

**Observation in this constructed family:** the tabulated values decrease with system size and retain the displayed order for those rows. This is not evidence of scale invariance outside the tested generator and parameters.

### Ranking Consistency

```
n=4: Complete > Ring > Star > Line  ✅
n=5: Complete > Ring > Star > Line  ✅
n=6: Complete > Ring > Star > Line  ✅
```

**Ring > Star at all sizes: limited historical observation only; requires matched-fixture requalification**

---

## Interpretation

### What This Experiment Can Support

1. **A graph-spectral proxy was evaluated on a small synthetic topology set.** The reported HDC value is the Laplacian algebraic-connectivity eigenvalue, not a formal IIT quantity.
2. **Some orderings were similar under the chosen constructions.** That observation motivates a controlled comparison but does not establish universal ranking preservation or agreement between formalisms.
3. **The tabulated family showed a repeated order at n=4, 5, 6.** Generalization requires independently generated, held-out topology families, uncertainty estimates, and explicit model matching.

### What This Doesn't Validate

1. **Exact Φ values**
   - IIT and HDC give different absolute numbers
   - Only relative orderings match

2. **Full IIT 4.0**
   - We used simplified IIT 3.0
   - Full cause-effect repertoires not computed

3. **True PyPhi comparison**
   - PyPhi is incompatible with Python 3.13
   - Used our own implementation

---

## Scientific Implications

### For Publication

A claim-bounded description suitable before further qualification is:

> "A tractable graph-spectral proxy whose rankings were observed on a specified synthetic topology set. Its agreement with IIT, error outside that set, and cognitive utility remain to be established."

### Naming Recommendation

Instead of claiming to compute "Φ" (which implies IIT), consider:

- **TII**: Topological Integration Index
- **Φ_HDC**: HDC-approximated integrated information
- **κ (kappa)**: Algebraic connectivity-based integration

### Caveats for Paper

1. "Our method approximates, but does not compute, true IIT Φ"
2. "Ranking preservation validated for n=4,5,6; larger systems inferred"
3. "Simplified IIT used as ground truth; full PyPhi comparison pending"

---

## Reproducibility

### Files

- **Validation script**: `scripts/pyphi_comparison.py`
- **Run command**:
  ```bash
  nix-shell -p python311 python311Packages.numpy --run "python3 scripts/pyphi_comparison.py"
  ```

### Dependencies

- Python 3.11+
- NumPy

---

## Future Work

1. **Full PyPhi comparison** when Python 3.10 environment available
2. **Larger systems** (n=8, 10) with parallelized IIT
3. **Real neural data** validation (C. elegans connectome)
4. **Correlation analysis** between IIT Φ and HDC Φ values

---

## Conclusion

The reported experiment is not sufficient to establish that the HDC metric is valid as an IIT approximation or a consciousness measure. It defines a graph property—algebraic connectivity—that can be useful for topology analysis at tractable cost.

Relative rankings do not identify which systems are "more conscious" without an independently validated bridge from the metric to that claim. A scalable proxy is useful only for the proposition and task for which it is separately qualified. See [`PHI_QUALIFICATION_V0_1.md`](PHI_QUALIFICATION_V0_1.md) for the current qualification contract.

---

*Validation performed: January 10, 2026*
*Validated by: Symthaea-HLB IIT Comparison Framework*
