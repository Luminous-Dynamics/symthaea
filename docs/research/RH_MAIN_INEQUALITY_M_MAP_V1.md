# RH main inequality (M): source-grounded proof map v1

**Task:** map the main arithmetic inequality (M) to a small set of exact Lean declarations and conceptual roles.  
**Parent:** [#13293](https://github.com/Luminous-Dynamics/symthaea/issues/13293)  
**Graph baseline:** [RH_PROOF_GRAPH_BASELINE_V1](RH_PROOF_GRAPH_BASELINE_V1.md)  
**Implementation follow-on:** [#13295](https://github.com/Luminous-Dynamics/symthaea/issues/13295)  
**Machine map:** `RH_MAIN_INEQUALITY_M_MAP_V1.json`

## 1. The mathematical role of (M)

The guide states the main inequality on PDF page 32, equation (M):

[
\frac1n\sum_{u\in H}\sum_{t\in T}\sum_{j=1}^{m}
(P_{u,t,j}-M_{u,t,j})\leq\mathcal E,
]

with

[
\frac{\mathcal E}{B}
\leq x^{C\varepsilon}
\left(\frac{x}{m}+\frac{xm}{k}+m+\frac{k}{n^2}\right),
\qquad B=\frac{|T|mb\delta}{2}.
]

Here `P` is a smoothed prime-power divisibility sum and `M` is its main term, expressed using the logarithm of the cutoff scale and the mean logarithm of cell weights. The inequality averages over the translate (u), column (t), and theta-lift map index (j).

Conceptually, (M) compares two views of the same arithmetic object:
- **divisibility:** local inert-prime valuations of derivatives of theta-lift constructions contribute weighted prime powers;
- **size:** the complex / archimedean size of the corresponding lattice maps bounds the total divisibility;
- **averaging:** selected residue-class columns and a family of maps control the losses;
- **normalization:** the error has four competing costs, with the (x/m) and (m) terms balancing when (m\asymp\sqrt{x}).

The human guide then combines (M) with character-sum estimates B and C to obtain the one-sided bound for (G(x)) on long intervals; that analytic bound excludes zeros to the right of the critical line. The guide does not prove all of (M); it explicitly quotes several parts from Lean. See the PDF abstract and initial status note, Part II §4 and §6, and the Part III status table. <Link url="https://www.math.uwaterloo.ca/~mrubinst/alpoge_claude_grh/zeta-paper.pdf" title="Rubinstein working guide (PDF)"/>.

## 2. The first formal bridge we can now name

Source archive is pinned to the public mirror commit:

`leanprover-community/riemann-hypothesis @ 40e56bab026552e089d7afcfeff08d2defb9c005`

The mirror commit states it is a byte-for-byte copy of the published archive. The proof source has no license established here, so this Symthaea file records only metadata and source references—not copied Lean source.

### A. Formal prime-sum and main-term objects

File [3] in Rubinstein's source list:

`src/Formalism/SizeWindowBrackets/FirstPrimeNeSecondPrime.lean`

- `ASK` (lines 84–87): a weighted sum over inert primes and their powers, with a positive-part cutoff.
- `MAIN` (lines 89–90): the main term proportional to (a), including (log a-1-\gamma), the character logarithmic derivative and the mean log of the weights.
- `AskMainBalance` (lines 98–101) and `AskExpansion` (lines 133–139): formal relations connecting the prime sum and its character expansion.

Blob: `18deb7cb85d6f650a69f01fda6e0c789b9f04d89`.

### B. Per-translate bound

File [28]:

`src/Formalism/SizeWindowBrackets/ChainSlopeWindows/LevelOfCongruence.lean`

- `each_unit_bound_statement` (lines 175–197) gives the per-unit `PointwiseBoundVariant` under the required hypotheses.
- The guide explicitly identifies this as the statement used by the final theorem.
- The same file also contains `weighted_mean_bound_statement`, but do not confuse the two.

Blob: `e08edc360cdf8d5d95d82ef1b805a8ea1a6151ef`.

### C. Composition to a gap bound

File [5]:

`src/Development/AutomorphicArithmetic/DedekindLFunctions/LandauTheorems/ChTop.lean`

- `Deduction.ch_top_main_eq` (lines 11–35) identifies the abstract `Data.Main` expression with a sum of `MAIN` terms.
- `Deduction.ch_top` (lines 37–73) applies `each_unit_bound_statement`, combines size and error-window assumptions, and derives the `gapA` bound.

Blob: `ce06a34a9e0b1b332e10d3b7efb04a06e9ab77f5`.

This is the strongest source-inspected bridge we have so far: `ASK / MAIN` → per-unit bound → `gapA` / error bracket. It is not yet a complete mapping of every quoted hypothesis in (M), and final-root dependency membership should be checked against the external kernel graph.

## 3. Do not use nearby formulations as substitutes

The human guide makes two particularly important distinctions:

- [7] contains `MeanBoundSecondForm`, an averaged statement. The guide explicitly says this is **not** the statement used by the final theorem in that form; it identifies `each_unit_bound_statement` in [28] as the used per-translate statement.
- [6] contains a growth-to-RH formulation that is a parallel / earlier form, not the exact declaration form used by the final theorem.

This is a valuable adversarial benchmark: a system that retrieves the nearby theorem by name or topic but substitutes it for the actual used declaration should fail statement/dependency fidelity.

## 4. Other quoted ingredients and current status

The guide names the following formal inputs; their mathematical role needs an explicit review and exact dependency confirmation:

| Ingredient | Source file / declaration reference | Status in this map |
|---|---|---|
| Chowla–Selberg period | [16] `GlobalTheory.T.chowla_selberg_period` | Source inspected; role candidate |
| Digit/valuation arithmetic | [21] `StarArith` | Source inspected; role candidate |
| Height of a map / lattice slopes | [25] `htMap`, `SizeIneq`, `SlopeIneqMean` | Source inspected; role candidate |
| Galois cohomology / denominator passage | [8] and Part III §III.9 | Exact theorem role still to map |
| Local comparisons, Bessel and layer-cake inputs | [19] `GlobalTheoremInputs.lean` | Several source-level bundles inspected; exact quoted theorem map open |
| Theta-kernel integrality at height-two logarithm coordinates | Part III §III.4–III.5, hypothesis (B) | Explicitly quoted from Lean in the guide; exact declaration name not yet mapped |
| Trace/rank/pairing statements | Part III §III.8 | Exact declarations not yet mapped |
| Eleven size-bound arithmetic inputs | Part III §III.7 | Exact declaration-by-declaration map not yet completed |

The guide’s Part III status table explicitly separates what is proved in the working exposition, what is conditional on Lean statements, and what remains quoted. The working guide also states that the assembly of maps and weights and the full assembly of (M) had not yet been written up there.

## 5. A concrete comprehension benchmark task

Give Symthaea:
1. the exact theorem root and pinned source graph;
2. the human statement of (M);
3. the declarations `ASK`, `MAIN`, `each_unit_bound_statement`, `ch_top_main_eq`, and `ch_top`;
4. the nearby but unused / non-final `MeanBoundSecondForm` and parallel growth-to-RH statement.

Then ask it to:
- explain the prime-power table and positive-part cutoff;
- explain how the main term is computed;
- distinguish per-translate from averaged bounds;
- reconstruct the `ASK → MAIN → gapA` bridge;
- explain the four error terms and the (m\asymp\sqrt{x}) balance;
- list every quoted input and mark unresolved mathematical steps;
- reject substitution of `MeanBoundSecondForm` for the exact per-translate statement;
- identify which of these declarations are formally in the final theorem's proof dependency closure.

Score explanation accuracy, statement fidelity, dependency precision/recall, correct abstention, and human ability to reconstruct the argument after reading the explanation.

## 6. What remains before calling this an explanation

The current map is **source-inspected**, not expert-reviewed and not a complete graph extraction. It does not yet:
- map every named formal input in Part III §III.4–III.9;
- identify the exact Lean declaration for theta-kernel integrality hypothesis (B);
- trace all local valuation / Galois cohomology / trace-formula chains;
- replay the Lean proof or independently recheck any theorem;
- show that a human can understand the argument from the explanation.

The next high-value task is a bounded reconstruction of the `ASK / MAIN → each_unit_bound_statement → ch_top` chain, with an independently checked declaration graph and mathematician review.

## 7. Claim boundary

This is a source-grounded mapping of part of the formal architecture around (M). It is not a reconstruction of the full proof of (M), not a claim that every file named in the exposition is in the theorem's kernel closure, and not an independent proof of RH.
