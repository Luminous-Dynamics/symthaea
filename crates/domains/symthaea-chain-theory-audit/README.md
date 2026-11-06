# symthaea-chain-theory-audit

A std-only independent implementation of the combinatorial record construction
used in A. D. Wissner-Gross, *A Parameter-Free Candidate for the Fine-Structure
Constant from Entangled Qudits* (23 October 2026).

It reconstructs the four slot dimensions (8, 3, 8, 3), enumerates all
two-slot product assignments, symmetrizes/antisymmetrizes under the read
exchange, derives the three qubit commutator records from explicit integer
2x2 basis matrices, counts both exchange signatures, and evaluates the
resulting radical sum.

## Reproduction

```bash
cargo test -p symthaea-chain-theory-audit
cargo run -p symthaea-chain-theory-audit --example chain_alpha_audit
```

The exact symbolic result represented by the sector counts is

[
\alpha_{\mathrm{candidate}} =
\frac{79 + 48\sqrt{2} + 42\sqrt{3} + 3\sqrt{6}}{31104}.
]

Its inverse is approximately 137.035999189566. The denominator is reconstructed
from one unread-link traceless operator space of dimension 24 and two read-link
operator spaces of dimension 36 each.

## Independent scope

This is an independent combinatorial/numerical reproduction of the published
calculation under its stated slot, exchange, survival and weight definitions.
It does not copy or import the paper's Lean file. It does not independently
formalize the complete qutrit–qubit gauge-group derivation or the external
quantum-information theorems motivating the read cap.

The paper's Lean file itself proves that its *defined* `readableByReads`
predicate is equivalent to (d_1 d_2 \le 6). The remaining bridge is the
physical identification of those defined reads with all the distillation/LOCC
operations intended by the axiom; that is not established merely by the
arithmetic or by checking the internal definitions.

This crate does **not** show that:
- the model's definitions are the right description of nature;
- the qutrit–qubit link is a photon;
- the calculated number is the electromagnetic coupling;
- the gauge-group calculation yields all of the Standard Model;
- the model has a momentum-dependent photon propagator, QED vertex, or
  scattering amplitudes.

The author explicitly describes the physical identification with the photon
as a conjecture. The Lean file's header also says that it checks the model
calculation but not whether its definitions are the right physics.

## Experimental comparison

The comparison in the example uses the CODATA 2022 inverse fine-structure
constant, 137.035999177(21), as the latest regular CODATA adjustment
available at the time of this work. It is about 0.60 of that *adjusted
standard uncertainty* from the candidate. CODATA states that the six input
determinations for alpha in the 2022 adjustment required an uncertainty
expansion factor of 2.5 to bring their inconsistencies to an acceptable
level. Thus agreement with the combined adjusted value is a comparison,
not an independent validation of the model.

The next regular CODATA 2026 adjustment has a closing date of 31 December
2026, with results scheduled for early 2027. The benchmark should be refreshed
when the published adjustment and accepted input data become available.
