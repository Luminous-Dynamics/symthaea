# LQCD-021G — final-sample admission oracle

Independent standard-library execution for the negative-admission and campaign-invalidation semantics tracked by #2716, stacked on the exact staged campaign-subject oracle in #2820.

Exact executed subject SHA-256:

`371cb97c1afcf45b4da7a4407c862f9f079e5d7090d2ff0040b2b18bf85cabc1`

Canonical result SHA-256:

`d93f310f10b066ba8d06f2ba1441ca7f38a478e1b0ede0a82d09e8a7b5a0f3d6`

Synthetic admissible receipt digest:

`1f1d86092130bae9fbbbddbc330d67bbbcc0cc40287d97266f9ccb92a7c33962`

Exact expected 4000-configuration-set digest:

`dfd9c16460f3c8373dac0a682053a95f04cd190e6eb967b00a76a4b42f81964e`

## Admission theorem

The oracle freezes four typed outcomes:

- `Admissible`
- `NotAdmissible(reason_set)`
- `Inconclusive(missing_evidence)`
- `InfrastructureInvalid`

The baseline synthetic campaign contains exactly four chains × 1000 precommitted retained identities, exact frozen policy identity, complete required evidence, no hidden extension, no dropped chain, and no unverified replacement; it is admitted.

Required adversarial cases then prove:

- a failed topology/slow-mode gate is `NotAdmissible`;
- an incomplete measurement set is `NotAdmissible`;
- 3999/4000 is rejected as incomplete;
- post-start policy mutation is rejected;
- hidden extra configurations used after a discrepancy are rejected;
- an unverified replacement configuration is rejected;
- infrastructure failure remains a distinct `InfrastructureInvalid` outcome;
- final chain disagreement is rejected;
- deleting the disagreeing chain post hoc is rejected;
- genuinely missing required evidence is `Inconclusive`, not silently treated as pass.

A negative receipt remains content-addressed evidence. A scientifically redesigned successor receives a new campaign identity while the failed predecessor receipt remains unchanged.

## Scientific boundary

The fixtures are synthetic. This establishes final-admission semantics only. It does not establish real pilot thresholds, equilibrium, admissibility of any real β=6.0 final sample, or agreement with EHK.
