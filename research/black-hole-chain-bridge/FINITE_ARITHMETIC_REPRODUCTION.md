# Finite Chain Theory arithmetic: independent count audit

**Status:** independent combinatorial reproduction of the finite record/weight arithmetic; no physical identification claimed.  
**Code:** [chain_alpha_count.rs](reference/chain_alpha_count.rs)  
**Parent:** [BH-CHAIN-TO-CONTINUUM-001](https://github.com/Luminous-Dynamics/symthaea/issues/12604)

## Source scope

The primary artifact is the author's [ChainAlpha.lean](https://alexwg.org/publications/ChainAlpha.lean), which states:
- exact read/record definitions and slot exchange;
- the qutrit–qubit object;
- four exchange-weight sector counts;
- readout dimension;
- exact radical readout value.

The file's own header says the theorem checks that its definitions produce the closed form and supporting finite facts, but does not check that the definitions are the right physics, the named literature theorems behind the read cap, the full unitary-group statements, or the physical identification with a constant of nature. The latter is conjectural and is not in the Lean file.

This Rust implementation is an independent, small combinatorial oracle for the slot/orbit arithmetic. It is not a port of the entire Lean proof.

## Reconstructed finite object

The object has a shared qutrit and three qubit arms, two read and one unread. The four slots of the two read links have traceless-basis dimensions

\[
[3^2-1,\ 2^2-1,\ 3^2-1,\ 2^2-1]=[8,3,8,3].
\]

Every product record places one basis letter in each of two distinct slots. Counting by slot pair:

| Slot pair type | Count |
|---|---:|
| Qutrit slots 0 and 2 | 8 × 8 = 64 |
| Qubit slots 1 and 3 | 3 × 3 = 9 |
| Four mixed slot pairs | 4 × 8 × 3 = 96 |
| **Product records** | **169** |

Read exchange maps slots 0↔2 and 1↔3. The independent enumerator finds:
- 11 fixed same-end assignments (8 qutrit-fixed plus 3 qubit-fixed);
- 31 non-fixed same-end exchange orbits;
- 48 mixed-end exchange orbits.

Each non-fixed orbit generates its symmetric and antisymmetric record combinations, while each fixed assignment contributes one record:

\[
11 + 2(31) + 2(48)=169.
\]

For the qubit far end, the finite Weyl phase is a sign. The three unordered pairs of the three traceless qubit letters contribute three commutator records, giving

\[
169+\binom{3}{2}=172.
\]

## Exchange-weight sectors

The Lean definitions assign a negative exchange sign when the transformed record returns with its own record sign. These are **weight-sector signs**, not simply the ordinary parity eigenvalues. The independent enumerator reproduces:

| (sigX, sigY) | Count | Weight radicand |
|---|---:|---:|
| (+1, +1) | 79 | 1 |
| (+1, −1) | 48 | 2 |
| (−1, +1) | 42 | 3 |
| (−1, −1) | 3 | 6 |
| **Total** | **172** | |

The weight law gives k=3 when the X exchange survives and k=2 when the Y exchange survives. Thus the exact radical numerator is

\[
79+48\sqrt 2+42\sqrt 3+3\sqrt 6.
\]

The readout dimension follows from two read links of dimension (3·2)^2=36 and one unread traceless link of dimension (3²−1)(2²−1)=24:

\[
N=36\cdot36\cdot24=31104.
\]

The resulting *finite-model number* is therefore

\[
\alpha_{\mathrm{cand}}=
\frac{79+48\sqrt2+42\sqrt3+3\sqrt6}{31104},
\qquad
\alpha_{\mathrm{cand}}^{-1}\approx137.03599918956616.
\]

This is the exact arithmetic consequence of the stated finite definitions. It is not evidence by itself that the quantity is the physical Thomson-limit fine-structure constant.

## What the Rust oracle checks

The file chain_alpha_count.rs independently:
- enumerates every two-slot assignment from [8,3,8,3];
- applies the exchange permutation and partitions assignments into fixed points and non-fixed orbits;
- constructs the four sector counts from the exchange-orbit classes;
- checks the qubit Weyl-sign condition and the three commutator-record count;
- derives the readout dimension and radical coefficients;
- evaluates the decimal reciprocal only after constructing the exact integer/radical expression.

The tests are intended to check the 169/172 totals, orbit decomposition, 79/48/42/3 sector counts, 31104 denominator, exact radical coefficients and decimal arithmetic.

## Explicit limits

This code does **not** independently reproduce:
- the full qutrit/qubit matrix basis, orthogonality and commutator expansions;
- the finite matching-group calculations and commutant theorem;
- the read-cap / selection theorem or its external literature assumptions;
- the complete theorem dependency graph or Lean kernel result;
- the physical interpretation of a readout as a photon propagator or alpha(0);
- any continuum limit, metric, curvature, horizon or gravitational dynamics.

Those remain separate proof and physics obligations. A shared abstract definition is not an independent physical derivation.

## Local test command and validation state

~~~sh
rustc --edition=2024 --test chain_alpha_count.rs -o /tmp/chain_alpha_count_tests
/tmp/chain_alpha_count_tests
~~~

**Current validation:** source committed and fetched back from the public branch. The count logic and exact weighted arithmetic were independently reproduced in a separate development-session calculation. Rust tests have been authored but not executed here because rustc/Cargo are unavailable. No Lean execution, theorem verification or physical identification is claimed.
