# Audit of linked multi-stream e-process reference code

**Date:** 2026-10-30  
**Scope:** source-level reproduction audit only; not a criticism of the mathematical theorem.

## Reference identity

The 2026 EJS paper *Carefree multiple testing with e-processes* links the authors' repository at https://github.com/tavyrikov/runmax_eBH/.

Inspected repository main commit: 8166e058158a3703b3841e48a2da9a8248a86101.

Files inspected:
- counterexpample.py — Git blob 94d478da62d742d3b4f64a95fbcbb5d3fd43cebd
- simulation.py — Git blob deb37978c0db7e7f246787b074e15ba67c194c74

## Findings requiring correction before using it as a numerical oracle

### 1. The proof-counterexample script changes the martingale increments

The paper's theoretical two-null construction uses multiplicative increments:

- stream 1: 1/2, 2, 1/2, each with probability 1/3;
- stream 2: 1/2, 1/2, 2, each with probability 1/3.

Each marginal increment has expectation exactly one.

The linked counterexpample.py instead represents these in the log domain using -1 and +1, commenting that log(2) is approximately 1. This is not a harmless approximation for a martingale identity. Its marginal multiplicative expectation becomes

`(2 exp(-1) + exp(1))/3 = 1.1513469036...`

rather than one. Therefore the simplified process in that script is not a mean-one martingale/e-process under the claimed construction. The Symthaea reproduction uses exact ±log(2) (equivalently exact factors 1/2 and 2).

### 2. The normal-simulation script does not enforce the adjuster domain

The paper defines admissible adjusters on M in [1,infinity). The linked simulation computes cumulative e-values and their running maximum using:

`cmax_Ep = np.maximum.accumulate(E_cumulative, axis=1)`

but does not include the baseline value 1. When all observed cumulative e-values for a stream remain below one, cmax_Ep is also below one. Then A2(M)=sqrt(M)-1 becomes negative, so it is not a nonnegative e-value. A1's theorem also only applies on the stated domain; its displayed formula needs the continuous extension A1(1)=1/2 and stable evaluation near one.

The corrected reproduction must define

`M_t = max(1, sup_{s<=t} E_s)`

before applying either adjuster and evaluate in stable log space.

### 3. Paper text and linked script use different Monte Carlo counts

The paper describes T=2,000 observations and M=10,000 simulation repetitions. The linked simulation.py sets M=100,000. Both can be reported as separate profiles, but a reproduction must not silently conflate them.

### 4. The correlation matrix is not pinned by a single recorded seed

generate_correlation_matrix uses NumPy's global RNG for the exponential eigenvalue draw, then constructs a separate default_rng for scipy.stats.random_correlation.rvs. The resulting correlation matrix is not saved or hashed. The matrix should be serialized, SHA-256 hashed and recorded with the run so that paired comparisons use exactly the same dependency structure.

### 5. License boundary

No root README or LICENSE file was found in the inspected repository paths. This audit does not copy or package upstream code. Keep the repository as an external reference unless license terms are established separately.

## Interpretation

These are implementation/reproducibility caveats in the linked code. They do not refute the paper's theoretical results:

- the two-null counterexample in the paper is mathematically specified with exact 1/2 and 2 increments;
- the paper's adjuster theorem requires the running maximum to lie in [1,infinity), and applying an admissible adjuster restores an e-process;
- the published FDR-sup theorem remains a theoretical result under its stated assumptions.

The corrected Symthaea work should reproduce the 200-stream dependent-normal experiment independently, with explicit clipping, boundary handling, a fixed correlation matrix digest and both the 10k/100k repetition profiles.

Related tracker: https://github.com/Luminous-Dynamics/symthaea/issues/9113  
Base reproduction: https://github.com/Luminous-Dynamics/symthaea/pull/9110
