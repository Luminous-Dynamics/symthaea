# Chain Theory independent audit v0.1

Date: 2026-11-07
Status: primary-source reconstruction and independent combinatorial implementation; Rust execution pending.

## Result reproduced

Symthaea's new std-only crate reconstructs the published finite record
construction from the author's stated slot and exchange rules. The author
source is pinned to commit a28db8998bce5bd5ef11195c05f13c86cd9dac5c.

| Check | Author's published result | Independent Rust implementation |
|---|---:|---|
| Two-slot product assignments | 169 | test asserts 169 |
| Added commutator records | 3 | test asserts 3 |
| Total records | 172 | test asserts 172 |
| (+,+) exchange sector | 79 | test asserts 79 |
| (+,-) exchange sector | 48 | test asserts 48 |
| (-,+) exchange sector | 42 | test asserts 42 |
| (-,-) exchange sector | 3 | test asserts 3 |
| Readout dimension | 31,104 | test asserts 31,104 |
| Inverse-alpha candidate | 137.035999189566... | test asserts numeric value |

The author publishes an independent standard-library Python enumerator and
the row-level records.csv. The Rust implementation now emits the CSV row
schema in author-compatible stable order and tests all six contiguous
classification bands across all 172 row ordinals:

| Ordinals | Construction | Slot kind | X | Y | Passed ends | k | Count |
|---|---|---|---:|---:|---|---:|---:|
| R001–R048 | odd | mixed | +1 | +1 | none | 1 | 48 |
| R049–R079 | odd | same | +1 | +1 | none | 1 | 31 |
| R080–R127 | even | mixed | +1 | -1 | qubit | 2 | 48 |
| R128–R158 | even | same | -1 | +1 | qutrit | 3 | 31 |
| R159–R169 | exchange-fixed | same | -1 | +1 | qutrit | 3 | 11 |
| R170–R172 | commutator | bracket | -1 | -1 | both | 6 | 3 |

This establishes a category-by-category row-schema match to the published CSV
specification. It is not yet a byte-for-byte file comparison, because the
Rust example has not been compiled in this environment. The example now emits
all rows to make the final diff straightforward on a machine with the pinned
Rust toolchain.

## Formalization boundary

The author's Lean file checks that the formal Part 1 definitions generate
the finite record calculation and radical sum. Its own README explicitly
separates this result from whether the definitions describe nature.

The read-cap statement is used as a named mathematical input in the author's
selection analysis. A separate third-party Lean audit reduces that read-cap
statement, for the paper's specific readability predicate, to Woronowicz's
theorem for positive maps M2 to M3; that external theorem is not formalized
inside the audit file. This matters because an exact proof of arithmetic under
definitions is not the same as proving that the definitions are the right
physical model.

## Main physical gap

The Photon Conjecture identifies the qutrit-qubit link as a photon and the
candidate readout as the zero-momentum electromagnetic coupling. The finite
record calculation does not construct a momentum-dependent propagator,
photon pole, QED vertex or scattering amplitude. These are the primary
bridges required before the number can be interpreted as a derivation of
electromagnetism rather than a candidate scalar.

A third-party audit dated 25 October 2026 also reports a possible mismatch
between the paper's own unbroken generator / charge assignments and its
photon identification. This is not treated as settled here. It is a
concrete falsification target that should be independently reproduced from
the paper's equations before drawing conclusions.

The same third-party audit enumerates 1,492,992 alternatives from the
explicitly listed definition choices and reports only one 1-sigma match.
That bounds a defined alternative set; it does not eliminate unrecorded
researcher degrees of freedom or every possible model grammar.

## Measurement

Against CODATA 2022's adjusted inverse fine-structure constant
137.035999177(21), the candidate is about 0.60 of the adjusted standard
uncertainty away. CODATA expanded the uncertainties of its six alpha inputs
by a factor of 2.5. The third-party audit reports about 1.40 standard
deviations against an unexpanded weighted-mean comparison. These are
measurement comparisons, not validation of the photon conjecture.

The next regular CODATA adjustment is scheduled after 31 December 2026;
new comparisons should use the published adjustment and identified primary
measurements.

## Next experiments

1. Compare all 172 Rust records, signs and weights against the author's CSV.
2. Build and record the author's Lean file under its pinned Lean 4.35.0.
3. Independently audit the read-cap literature dependency and theorem scope.
4. Reconstruct the unbroken generator, charge assignments and Ward identities.
5. Test whether a covariant, momentum-dependent two-point function and
   physical pole can be derived without inserting the photon identification
   as an assumption.
6. Pre-register how future alpha measurements will score against the candidate.

## Reproducibility boundary

The Rust crate has not been compiled or tested in this environment because
the repository's Rust toolchain is not installed here. Its tests are
implementation intentions until run on the pinned toolchain.
