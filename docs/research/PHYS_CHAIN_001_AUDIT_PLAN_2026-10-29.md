# PHYS-CHAIN-001 — Independent Chain Theory audit

**Status:** Research protocol and preliminary source audit. The independent 172-record reconstruction and Lean execution have **not** been completed.

Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/8921

## Scope

This is an independent scientific and mathematical audit of Alex D. Wissner-Gross, *A Parameter-Free Candidate for the Fine-Structure Constant from Entangled Qudits* (dated 23 October 2026), the first published result of Chain Theory.

- Paper: https://alexwg.org/publications/ChainTheoryAlpha.pdf
- Chain Theory announcement (24 October 2026): https://theinnermostloop.substack.com/p/chain-theory
- Lean 4 source, requiring Lean 4.35.0 per its header: https://alexwg.org/publications/ChainAlpha.lean
- 754-page working compendium: https://alexwg.org/publications/ChainTheoryCompendium.pdf
- CODATA 2022 values and status: https://physics.nist.gov/cuu/Reference/versioncon.shtml

The objective is neither to endorse nor dismiss the theory. It is to determine what follows from the stated definitions, how sensitive the result is to defensible modeling choices, and which additional empirical tests discriminate the physical interpretation from alternatives.

## Confirmed paper claims and boundaries

The paper gives the candidate

\[
\alpha_{\mathrm{cand}}=\frac{79+48\sqrt{2}+42\sqrt{3}+3\sqrt{6}}{31104},
\qquad
\alpha_{\mathrm{cand}}^{-1}=137.03599918956615502918\ldots
\]

The expression comes from the weighted readout of 172 records of a qutrit–qubit entangled-link construction. The paper conjectures that this link is the elementary link of the photon, thereby identifying the candidate with the Thomson-limit electromagnetic coupling \(\alpha(0)\).

The paper's formalization header explicitly says that the Lean 4 file:
- builds the record/sector calculation from the declared definitions and checks the closed form;
- does **not** establish that those definitions are the right physics;
- leaves the cited mathematical results behind the “read cap” as named hypotheses rather than formalizing them all;
- does **not** prove the identification with a constant of nature, which is the Photon Conjecture.

The paper's own supplementary section S9 lists alternatives that change the result. It says two alternatives land within one percent of the computed value: swapping the ends, and counting commutator records at exchange X alone. The paper argues that its independent model definitions exclude these alternatives without reference to the measured alpha. This audit must test that argument on its merits; their existence is not evidence of misconduct, and it is not by itself evidence that the candidate is selected post hoc.

## Preliminary numerical check — formula only

A high-precision decimal evaluation of the published closed-form expression gives:

| Quantity | Value |
|---|---:|
| Candidate alpha | 0.007297352563662260254797635588… |
| Candidate inverse alpha | 137.035999189566155029183744020… |

Comparisons using the paper's listed measurements and stated standard uncertainties:

| Reference | Reference inverse alpha | Candidate minus reference | Difference in stated standard uncertainties |
|---|---:|---:|---:|
| CODATA 2022 | 137.035999177(21) | +1.2566e-8 | +0.5984 sigma |
| Electron g−2 (2023) | 137.035999166(15) | +2.3566e-8 | +1.5711 sigma |
| Rubidium recoil (2020) | 137.035999206(11) | −1.6434e-8 | −1.4940 sigma |
| Cesium recoil (2018) | 137.035999046(27) | +1.4357e-7 | +5.3173 sigma |

These arithmetic checks independently evaluate the **published formula only**. They do not reproduce the record enumeration, prove the formula from the axiom, establish the Photon Conjecture, or constitute an independent experimental reanalysis. CODATA 2022 remains the latest CODATA recommended set listed by NIST as of this protocol date; its adjustment considers data available through 31 December 2022. See https://physics.nist.gov/cuu/Reference/versioncon.shtml and https://physics.nist.gov/cuu/pdf/JPCRD2022CODATA.pdf.

## Main falsifiable questions

1. Does an independent implementation derive the 169 product records plus three commutator records, their exchange sectors, weights, normalization, and final expression from the prose/definitions rather than copying the provided intermediate values?
2. Which results are definition-level mathematical consequences, which depend on cited hypotheses or unformalized results, and which are physical identifications?
3. Are all choices affecting the result independently motivated by the model before comparing with alpha? How much does a preregistered set of physically/mathematically admissible variants change the output?
4. Does the theory make additional predictions not used to choose the link/readout and not equivalent to fitting the alpha result? What observation could falsify each?
5. Which empirical comparisons are independent, and which share theoretical inputs or experimental systematics?

## Work plan

### WP1 — Independent reconstruction

- Rebuild the link algebra, records, exchange actions, joint eigenspaces, sector sizes, weight law, readout normalization, and algebraic expression in a small Rust implementation.
- Do not import the author's record table, intermediate counts, or final coefficients as inputs.
- Keep a separate tiny known-answer oracle or independently written evaluator for cross-checking the Rust version.
- Verify the exact algebraic expression independently of floating-point comparison, then use high-precision numerics as a secondary check.
- Bind source, toolchain, configuration, output files, and results with SHA-256 digests.

### WP2 — Lean reproducibility and proof boundary

- Download and archive the exact source bytes; record SHA-256 and Lean version.
- Execute the source with Lean 4.35.0 and retain complete stdout/stderr and exit status.
- Record declarations and axioms reported by Lean for the relevant theorems.
- Independently classify every load-bearing statement as: proved from definitions, conditional on a named hypothesis/external theorem, or conjectured physical identification.
- Do not describe this as a proof that nature has the value derived.

### WP3 — Choice-space and robustness audit

- Extract each variant in Supplement S9 and all further alternatives explicitly discussed in the compendium into a versioned table.
- For each variant, record the changed rule, derivation step, exact changed output, and reason offered for accepting/rejecting it.
- Freeze the variant set and independent admissibility criteria before computing numerical closeness to alpha.
- Include semantics-preserving controls (basis change, record permutation) and semantics-changing mutations (drop/duplicate a record, exchange-sign flip, single-sector weight mutation, normalization change).
- On each admissible choice, record whether it is independently rejected by the axioms/definitions or only has a worse alpha match. Report the complete denominator, not merely examples.
- Treat the “two within one percent” statement as a declared starting point to verify, not a full count of the alternative space.

### WP4 — Further prediction and measurement audit

- Map each additional result in the compendium to assumptions, formal status, derived observable, uncertainty, and an independently measurable prediction.
- Mark as non-independent any “prediction” that merely restates an input assumption or the target value used to motivate the interpretation.
- Retain the paper's cited cesium-recoil tension as a separate comparison; do not suppress it in a pooled consistency statement.
- Before any future measurement is unblinded, freeze the observable, numerical prediction, expected experimental uncertainty, analysis plan, and rejection region.

## Required mutations and negative controls

- Permuting record enumeration or changing a basis representation without changing semantics must preserve the canonical output.
- Changing one record's membership, exchange eigenvalue, sector weight, shared-qudit convention, or normalization must be detected and produce a changed result or a failing invariant.
- Mutating a claimed premise or input digest must invalidate its evidence receipt.
- A code-level theorem must not be accepted as evidence for an unformalized physical identification.
- An alternative must not be rejected only because its number disagrees with alpha; the admission rule must be assessed independently of that outcome.

## Acceptance criteria

- Independent reconstruction produces the same exact algebraic expression from the declared definitions.
- The Lean run is reproducible from pinned source bytes and toolchain, with exact artifacts and logs.
- The claim ledger distinguishes mathematical correctness, assumptions, formal proofs, numerical comparison, and empirical confirmation.
- The choice-space audit reports every preregistered variant and all exclusions.
- Every conclusion names the exact artifact revision and complete denominator.
- No unrun test is marked passing, and no queued job is treated as a pass.

## Claim ceiling

This document is a protocol plus a preliminary arithmetic check, not a completed independent audit. The arithmetic result establishes only that the published closed form evaluates to the published decimal. The physical identification remains a conjecture until supported by independent theoretical and empirical evidence.
