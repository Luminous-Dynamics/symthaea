# symthaea-chain-theory-audit

A std-only independent implementation of the combinatorial record
construction used in A. D. Wissner-Gross, A Parameter-Free Candidate for
the Fine-Structure Constant from Entangled Qudits (23 October 2026).

The independent implementation reconstructs the four slot dimensions
(8, 3, 8, 3), enumerates the two-slot product assignments, performs
read-exchange symmetrization and antisymmetrization, derives the three qubit
commutator records from integer 2x2 basis matrices, counts both exchange
signatures, emits all 172 row classifications in the author's CSV sort order,
and evaluates the resulting radical sum.

## Reproduction

Run:

    cargo test -p symthaea-chain-theory-audit
    cargo run --locked -p symthaea-chain-theory-audit --example chain_alpha_audit > /tmp/chain-alpha-rust.csv
    sed -n '/^record,construction,slot_kinds/,$p' /tmp/chain-alpha-rust.csv > /tmp/chain-alpha-rows.csv
    sed '/^#/d' crates/domains/symthaea-chain-theory-audit/tests/fixtures/author-records-a28db899.csv > /tmp/chain-alpha-author-rows.csv
    diff -u /tmp/chain-alpha-author-rows.csv /tmp/chain-alpha-rows.csv

The sector counts reproduce the author's published table:

| X signature | Y signature | Record count | Weight |
|---:|---:|---:|---:|
| +1 | +1 | 79 | 1 |
| +1 | -1 | 48 | sqrt(2) |
| -1 | +1 | 42 | sqrt(3) |
| -1 | -1 | 3 | sqrt(6) |
| Total | | 172 | |

There are 169 product records and three commutator records. The exact
symbolic expression represented by those counts is

    (79 + 48*sqrt(2) + 42*sqrt(3) + 3*sqrt(6)) / 31104

whose inverse is approximately 137.035999189566155.

## Primary-source grounding

The audited author source is pinned to commit
a28db8998bce5bd5ef11195c05f13c86cd9dac5c in
https://github.com/physical-superintelligence/chain-theory

Relevant author artefacts:

- papers/alpha/lean/ChainAlpha.lean
- papers/alpha/lean/README.md
- papers/alpha/anc/enumerate_records.py
- papers/alpha/anc/records.csv
- papers/alpha/anc/README.txt

The author publishes both the Lean 4.35.0 formalization and a standard-library
Python record enumerator. The Rust reproduction was written independently
from the documented slot, exchange and weight rules, including an
author-compatible lexicographic assignment ordering and stable
construction/slot-kind sort. Its tests assert all six contiguous CSV
classification bands, including construction, slot kind, X/Y signs, passed
ends and squared weight for all 172 ordinals. The example emits the complete
CSV rows so a byte-level comparison against the pinned author file can be run
once the Rust crate is compiled.

## What the formalization proves

The author’s Lean file checks that its Part 1 definitions produce the
published finite calculation: the product records, commutator records,
exchange eigen-records, sector counts and closed-form readout. Its main
arithmetic theorem is a theorem about those definitions.

The author’s own Lean README distinguishes that computation from the
external mathematical input used for the read-cap selection rule:

- The read cap is stated as a named hypothesis in the original formalization.
- The cited mathematical results are associated with Størmer, Woronowicz
  and Choi; the original ChainAlpha file does not re-prove all of those
  literature results from first principles.
- A separate public audit formalization reduces the read-cap statement, for
  the paper's readability predicate, to Woronowicz's theorem for positive
  maps M2 to M3. That external theorem is not proved inside that audit file.

Thus the record arithmetic can be exactly reproduced without the physical
interpretation or every mathematical premise becoming established.

## Physical interpretation remains conjectural

The paper identifies the qutrit-qubit link with a photon and its readout with
the zero-momentum electromagnetic coupling by a separate Photon Conjecture.
The record count does not construct:

- a momentum-dependent photon propagator or propagating pole;
- a QED interaction vertex and scattering amplitudes;
- a full renormalized QED calculation;
- an experimental mechanism connecting the defined read operation to
  all physically allowed distillation / LOCC operations.

The Lean file itself states that it checks the model computation, not whether
the definitions are the right physics, and that the photon identification is
a conjecture outside the file.

A separate public audit by Cody Mitchell (25 October 2026) reports a possible
tension between the paper's own unbroken generator / charge assignments and
the photon / electric-charge identification. That is an important objection
to investigate, not a result this crate independently adjudicates:

https://github.com/SproutSeeds/chain-theory-audit

The same audit reports a census of 1,492,992 alternatives formed from the
paper's explicitly listed choices, with the published value the only listed
1-sigma match. That is evidence about the enumerated alternative grammar,
not proof that every possible researcher choice or model grammar has been
exhausted.

## Measurement comparison

The example compares the candidate inverse coupling with CODATA 2022:
137.035999177(21), reporting about 0.60 of the adjusted standard uncertainty.
CODATA expanded the uncertainties of the six alpha input determinations by
a factor of 2.5 in that adjustment. The independent October audit reports
about 1.40 standard deviations against an unexpanded weighted mean of a
subset of those inputs. These comparisons are not independent validation.

The next regular CODATA adjustment is scheduled after its 31 December 2026
closing date, with publication expected in early 2027. Refresh the measurement
comparison only from the published adjustment and identified primary inputs.

## Separate physical-consistency audit (new)

The combinatorial reproduction and the physical interpretation are different
claims. The `physical_consistency` module adds exact integer checks for
small consequences of the Letter's stated representation, informed by the
independent audit by Cody Mitchell in
[SproutSeeds/chain-theory-audit](https://github.com/SproutSeeds/chain-theory-audit),
pinned here to commit
`95cd02da5955450397a848c25f8287c1a037d9f2` (25 October 2026).

The current checks are deliberately narrow:

1. **Unbroken generator.** Under the listed 5D diagonal assignments,
   `U/i = diag(1,1,-1,-1,0)`,
   `6Y = diag(3,3,-2,-2,-2)`, and
   `C8 = diag(0,0,1,1,-2)`, the exact identity is
   `U = (1/3)(6Y) - (1/3)C8`. Its weak-isospin coefficient is zero and its
   colour-generator coefficient is nonzero. Under these assignments it is
   not the Standard Model electric-charge generator `Q=T3+Y`.
2. **Pair-state charge.** The independent audit evaluates the two supported
   terms under its stated linked representation as `+2/3` and `-1/3`.
   A superposition of these two terms is not an eigenstate of that electric
   charge generator. This is a consistency tension if the same pair state is
   identified with a photon state.
3. **Weak-angle normalization.** Taking the compendium's conditional
   `cos²(theta_W)=1/2` and `alpha_em=alpha_Y cos²(theta_W)`, the candidate
   gives different implied `1/alpha_em` values under the usual Y, 2Y, 6Y,
   and GUT-normalized conventions. The independent audit reports that the
   photon identification would require a hypercharge scale of `sqrt(2)`,
   which is not one of those standard normalizations.
4. **Running-ladder distinction.** The readout dimension 31,104 is not an
   integer power of 319, the base used in the compendium's stated ladder.

These are conditional arithmetic and representation checks; they do not
rederive the whole gauge theory, prove that the independent auditor's reading
is complete, or establish a consensus verdict on Chain Theory. The external
audit also reports a census of 1,492,992 combinations of alternatives
explicitly listed in the Letter, with one value within the stated 1-sigma
CODATA window. That census does not include undocumented choices made during
the theory's development, so the broader look-elsewhere question remains open.

Run:
```
cargo test -p symthaea-chain-theory-audit
cargo run -p symthaea-chain-theory-audit --example chain_alpha_audit
```

A passing arithmetic test cannot promote the photon conjecture. The next
independent gate is to compare the generator, ray-stabilizer and Ward-identity
calculations against the author's full source definitions and a second
formalization, then preregister the physical predictions that differ from
Standard Model QED.

## Scientific non-claims

This crate does not show that:
- the model's definitions are the correct description of nature;
- the qutrit-qubit link is a photon;
- the calculated number is the electromagnetic coupling;
- the gauge-group calculation derives the complete Standard Model;
- the model provides QED propagators, vertices or scattering amplitudes.

The next milestones are per-record comparison with the author's CSV,
independent Lean build / axiom audit, read-cap premise review, explicit
charge-assignment consistency checks and preregistered comparison to new
independent measurements.
