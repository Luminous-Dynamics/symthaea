# symthaea-connectome-identity

Source-bound identity joins for full square connectome matrices.

## Scope

This crate validates that:
- matrix row IDs, matrix column IDs and neuron annotation IDs form the same unique stable-ID set;
- declared matrix dimensions match the axis ID arrays;
- row/column orientation is explicit;
- reordered axes and annotation tables resolve through deterministic bijections;
- stimulus and keep/remove requests resolve through stable neuron IDs;
- duplicate, missing, extra, unknown or conflicting identities fail closed.

It emits two SHA-256 commitments: a canonical identity commitment invariant to source ordering, and a source-binding commitment that binds the exact row, column and annotation order.

The source-artifact digest strings are computed by the intake layer. This crate checks digest syntax and binds the references; it does not fetch or authenticate source bytes or inspect matrix numerical values.

## Non-goals

No silent ID normalization/coercion, biological identity inference, missing-neuron imputation, matrix payload parsing, connectivity correction, physiology, neural simulation or behaviour claims.

A valid identity join proves that the declared axes and annotations reconcile under the supplied source IDs and mapping protocol. It does not prove anatomical truth or function.

All unit-test data are synthetic fixtures. The unit oracle includes exact-order, permuted-axis, digest-stability, malformed identity, unknown target and keep/remove conflict controls.
