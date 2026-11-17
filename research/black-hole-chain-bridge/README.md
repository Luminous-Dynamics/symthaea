# Chain Theory → continuum geometry → black holes

**Status:** source audit and research protocol; no continuum-gravity derivation claimed.  
**Parent issue:** https://github.com/Luminous-Dynamics/symthaea/issues/12604  
**Broader claim registry:** https://github.com/Luminous-Dynamics/symthaea/issues/8923

## Primary sources

- [Fine-structure candidate paper](https://alexwg.org/publications/ChainTheoryAlpha.pdf), 23 October 2026.
- [Chain Theory announcement](https://theinnermostloop.substack.com/p/chain-theory), 24 October 2026.
- [Working compendium](https://alexwg.org/publications/ChainTheoryCompendium.pdf), 754-page draft.
- [Lean 4 artifact](https://alexwg.org/publications/ChainAlpha.lean).

## Initial claim-status audit

This repository must preserve the author's own distinctions between exact finite-model results, conditional mathematical results, open results and physical conjectures.

- The exact alpha-candidate algebraic computation is reproducible from the paper's definitions. The inverse is approximately 137.03599918956616.
- The identification of the qutrit–qubit link with a photon and the internal readout with physical alpha(0) is explicitly **Conjecture 1** in the paper. The paper states that the model supplies a residue/readout but not a momentum-transfer propagator; the zero-transfer pole enters through the conjecture.
- The Lean file explicitly limits its scope: it does not establish that the definitions are the right physics, prove the named literature theorems, cover the full unitary-group statements, or prove the identification with a constant of nature.
- Compendium §98.1 (around pp. 242–243) labels the tested horizon area-law result open. Its tested cut does not establish the expected proportionality; the physical correspondence is described as “by name”.
- Compendium §I.24.1 (around pp. 703–704) labels the proposed curvature relation open. It records disagreement between the washed graph Laplacian and the (2,2)-link-count second-difference operator.
- The horizon “field equation at the counting face” is also marked open in the compendium.

These are intake findings, not a verdict on the theory. Re-pin exact source revision and reproduce the finite constructions, including negative/open outcomes, before making broader statements.

## Critical scientific bridge

A finite relational/entanglement structure is not a Lorentzian spacetime merely because its operators resemble familiar geometric quantities.

Before connecting Chain Theory to Symtropy's metric solver, require:

1. A refinement family and a controlled finite-to-continuum scaling limit.
2. A physical causal order distinct from construction/docking order.
3. A local Lorentzian metric, signature and light cones.
4. Local Lorentz symmetry and diffeomorphism/covariance properties.
5. A connection and curvature tensor with invariant meaning.
6. A continuum action or dynamical law, matter coupling and conservation identities.
7. Convergence/error bounds across at least three refinement scales.
8. Universality or explicit dependence on graph choices and rewrite rules.
9. Black-hole horizon, entropy and dynamics derived from that same continuum theory.

A graph Laplacian is not automatically a curvature tensor. A Lorentz algebra is not automatically a Lorentzian metric. A graph cut is not automatically a causal horizon. A Page curve of a finite random circuit is not an evaporation derivation.

## Reuse existing work

- #8923 claim/dependency graph.
- #8919 shared generative grammar.
- #11631 exact finite-dimensional density matrices, entropy and Page/Haar comparisons (draft PR #11632).
- #11021 identifiability and active experiment design.
- Symtropy #2870 classical finite relational-grammar benchmark (structural analogy only).
- Symtropy #5811 matched GR/Weyl/quantum-information black-hole comparison.

## First experiments

1. Independently reproduce the finite alpha arithmetic and 172-record count, separately from the photon/alpha physical identification.
2. Independently implement the specified finite web, matching, link and read operations.
3. Check tensor-factor semantics, normalization, positivity, monogamy and no-cloning in small exact Hilbert spaces.
4. Reproduce the compendium's competing curvature operators and known disagreement.
5. Reproduce the horizon-cut area calculation including undefined/failing ratios.
6. Only if finite results are stable, construct a controlled refinement family and test whether any continuum metric/action emerges.

## Comparison with state of the art

Compare each layer against its actual specialist baselines:
- quantum information: Haar/Page, Hayden–Preskill, decoupling and operational recovery;
- entanglement/geometry: tensor networks, PEPS, holographic codes, Ryu–Takayanagi and quantum extremal surfaces;
- emergent gravitational dynamics: Jacobson-style entanglement equilibrium;
- discrete spacetime: causal sets and spin-network/loop approaches;
- continuum gravity: GR, GR-EFT, pure Weyl and explicitly named scalar/quadratic theories.

A win on a classical graph grammar task does not validate quantum axioms. A win on finite entropy curves does not derive a four-dimensional metric. Any advantage must be measured against a matched baseline with sealed holdouts and independent reproduction.

## Claim ceiling

No claim that Chain Theory derives the physical fine-structure constant, the Standard Model, Lorentzian spacetime, Einstein/Weyl equations, the Bekenstein–Hawking area law, Hawking radiation, or resolves the information paradox until the required proof/continuum/empirical bridges close.