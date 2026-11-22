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


## 8. The missing conceptual bridge: why the log-weight appears in M

The most useful part of Part II §5.1 is that it derives the main term (M) from ordinary prime summation, rather than treating the expression as an unexplained formal target.

Fix one table (t) and one cell (i=(v,s)). Define

[
S_i(y)=sum_{r\in i} c(r)(y/r-1)_+,
qquad
Psi_i(z)=sum_{substack{r\in i\\r\le z}}c(r).
]

Integrating the step function (Psi_i) gives

[
rac{S_i(y)}{y}=int_1^y Psi_i(z)rac{dz}{z^2}.
]

For fixed moduli, among primes not dividing (2qd), a given cell receives a raw density (1/(4n)): one half are nonsquares modulo (d), one half have the specified sign modulo (q), and (1/n) have the specified residue-pair column. The Chinese remainder theorem makes these conditions independent among invertible residue classes. The coefficient (c(r)=(1-chi_d(r))log p) equals (2log p) on the ordinary contributing primes, doubling the density in the weighted counting function. Hence the cell main density is

[
mu=rac{1}{2n}.
]

The fixed-modulus prime number theorem in progressions yields

[
Psi_i(z)=mu z+	ext{error}.
]

The constant

[
C_i=int_1^infty(Psi_i(z)-mu z)rac{dz}{z^2}
]

is convergent for fixed moduli. Subtracting the tail gives

[
S_i(y)=mu ylog y+C_i y+R_i(y),
qquad
R_i(y)=-yint_y^infty(Psi_i(z)-mu z)rac{dz}{z^2}.
]

The constants sum to

[
sum_i C_i=-1-gamma+c_{chi_d}.
]

The reason is that (sum_i S_i=S_1-S_{chi_d}): the ordinary smoothed von Mangoldt sum contributes the linear coefficient (-1-gamma), while the nonprincipal character sum contributes (-c_{chi_d}).

Now return to the weighted table:

[
P_{u,t,j}=sum_i w_i S_i(a/w_i).
]

Substituting the decomposition above, the logarithmic terms become

[
mu asum_ilog(a/w_i)
=alog a-alanglelog wangle,
qquad
langlelog wangle=rac{1}{2n}sum_ilog w_i.
]

The linear constants become (a(-1-gamma+c_{chi_d})). Consequently,

[
P_{u,t,j}-M_{u,t,j}=sum_i w_iR_i(a/w_i),
]

where

[
M_{u,t,j}
=aleft(log a-1-gamma+c_{chi_d}
-langlelog wangleight).
]

**This explains why the average logarithm of the cell weights is not decorative:** it is forced by the prime-density main term after rescaling each cell's cutoff from (a) to (a/w_i).

Important qualification: this argument uses fixed-modulus prime number theorems. Their constants may depend on (q,d,N). The uniform estimates needed while those moduli vary with (x) are a separate obligation, handled later by Estimate C. Do not use the fixed-modulus (O_A(z/log^A z)) as though it were already uniform.

## 9. The finite-group translation identity: how the table becomes one prime sum

The second conceptual bridge is exact combinatorics on the finite group (H).

For fixed (t) and a prime power (r), write (s=t[r]). For each (zin T_2), the condition (sin uT_2) determines exactly one translate:

[
u=sz^{-1}.
]

Therefore exactly (|T_2|) of the (n=|H|) translates select the column (s). With (delta=|T_2|/n) and row indicator (v(r)=1iffchi_q(r)=1),

[
rac1nsum_{uin H}
mathbf{1}_{{v(r)=1, t[r]in uT_2}}
=
deltamathbf{1}_{{chi_q(r)=1}}
=
rac{delta}{2}(1+chi_q(r)).
]

Multiplying by (1-chi_d(r)) gives exactly the coefficient in the single smoothed prime sum

[
S_{q,d}(x)=sum_{rle x}Lambda(r)
(1+chi_q(r))(1-chi_d(r))(x/r-1).
]

After multiplying by the common distinguished weight (b), the exact translation identity is

[
rac1nsum_usum_{i	ext{ distinguished at }u}
bS_i(a/b)=rac{delta b}{2}S_{q,d}(a/b).
]

The same identity applies to the residuals after subtracting their common logarithmic and linear centering terms. This is the mechanism by which the many-cell table average is transformed back into one prime sum with quadratic-character twists.

There are three separate claims here, and they must not be conflated:

1. **Exact translation counting:** the finite-group identity above.
2. **Approximation of distinguished weights by (b):** controlled by the small (Delta_i=w_i-b) bound.
3. **Discarding nondistinguished cells:** requires the uniform Estimate C bound because their cutoffs are much larger.

Only after both error estimates and the (a_t/bapprox x) adjustment does the exposition obtain

[
rac1nsum_{u,t,j}(P_{u,t,j}-M_{u,t,j})
=BR_{q,d}(x)+O_arepsilon(B(1+log x)).
]

Then (M) yields (R_{q,d}leq E/B+O_arepsilon(1+log x)), the one-sided Estimate A. Estimates B and C control the three character twists; the identity
(R_{q,d}=G+G_{chi_q}-G_{chi_d}-G_{chi_qchi_d})
gives the RH criterion.

**The source derivation is now mapped, but the exact formal Lean declarations for the cell-centering, translation identity and all uniformity steps still need extraction and review.** This is a conceptual reconstruction from Rubinstein's working guide, not a new Lean proof.


## 10. Finite translation-invariance control

A dependency-free Rust test now exhausts the finite translate-count identity on every nonempty subset (T_2) of cyclic groups (C_n) for (2\le n\le6).

For each target column (s) and each (z\in T_2), there is exactly one translate (u=s-z). Distinct (z)'s produce distinct (u)'s, so the number of translates selecting (s) is exactly (|T_2|). The test also checks the row indicator for (chi_q\in\{-1,+1\}):

[
2\,\#\{u:v=1,\ s\in u+T_2\}
=|T_2|(1+\chi_q).
]

The frozen test space contains 620 group/subset/target cases and 1,240 row-indicator checks. The test function is `translation_average_selects_each_column_exactly_cardinality_t2_times` in `scripts/research/proof_comprehension_oracle.rs`.

This is a finite sanity / mutation-resistant reference test for the combinatorial averaging identity. It is not a Lean proof of the general finite-group statement and does not validate the prime-density, analytic or theta-lift parts of the RH argument. Its exact-head execution is required before status is promoted.


## 11. Stronger non-use evidence and the limits of shared auxiliaries

The upstream v0.2.1 dataset's `data/aux-only-decls.tsv` contains:

`Formalism.RankGapBounds.WeightedMeanData.Data.Tterm`
→ `src/Formalism/RankGapBounds/MeanBoundSecondForm.lean:97–99`
→ `n_shared_aux_in_cone = 2`.

The upstream method defines this class carefully: a declaration's own constant is not in the kernel proof closure, but an auxiliary lemma/matcher generated by the declaration can be shared and used by another declaration. Therefore this is evidence that the generator declaration `MeanBoundSecondForm` itself is not in the final theorem's proof-term closure, while two generated shared auxiliary constants are used elsewhere.

This is not evidence that those two auxiliary constants are mathematically irrelevant. In particular, it does not replace the exact used statement `each_unit_bound_statement` identified in the working guide. The distinction between a theorem declaration and its generated shared helper must be preserved in the proof-to-concept graph.

## 12. Uniformity obligations after the exact translation identity

The guide explicitly states that higher prime powers and the one exceptional prime fit inside the fixed-modulus PNT error (O_A(z/(\log z)^A)). It also warns that this fixed-modulus error constant may depend on (q,d,N). Thus the fixed-modulus decomposition proves (R_i(y)=o(y)) with moduli fixed; it does not by itself prove the uniform bound required when the moduli move with (x).

The uniform-to-global bridge has three separate pieces:

- **Nondistinguished cells, Estimate C / (5.6) / (II.25):** for (y\geq \exp(x^{\varepsilon/3})/3), the character sums satisfy (|G_\chi(y)|\leq y\exp(-\kappa x^{\varepsilon/8})), with (kappa) independent of (x,q,d). A cell residual is a character combination whose Fourier coefficients have total absolute value 2, giving (|w_iR_i(a/w_i)|\leq2a\theta); the normalized aggregate is (o(1)).
- **Distinguished-weight perturbation, (II.26):** (w_i=b+\Delta_i), with (0\leq\Delta_i\ll b/(nx^2)), and (|w_iR_i(a/w_i)-bR_i(a/b)|\ll x\Delta_i). The normalized aggregate is (O(1/x)).
- **Cutoff shift:** (|a_t/b-x|\leq1), so the smoothed sum and centering terms vary by (O(1+\log x)).

Only after these three pieces are combined with exact centering (5.3) and translation (5.5) does the guide derive (5.7) and transfer (M) to Estimate A. The source map keeps these analytic obligations separate from the finite-group translation identity. Exact declaration names for Estimate C and the full dependency chain are still unresolved.
