# symthaea-energy-native-dossier

Assembles Tier-1 energy-material dossiers directly from native campaign-bound evidence envelopes.

The caller no longer supplies generic dossier contributions independently. For every envelope, this crate derives:

- evidence dimension;
- candidate ID;
- exact envelope SHA-256 as `source_receipt_sha256`;
- exact generic `Prediction` stored in the envelope.

This removes a mismatch class where a valid source receipt could otherwise be paired with a different prediction, dimension, candidate ID, or receipt digest.

## External identity still matters

Native candidate/campaign lineage does **not** prove that an external Materials Project polymorph, hazard substance, recovery process, or manufacturing record actually refers to the intended candidate.

Therefore the existing #1942 `IdentityAssertion` ledger remains mandatory. Each assertion must bind the exact native-envelope SHA-256. The normal dossier assembler then applies its external identity checks unchanged.

## Partial evidence stays partial

The assembler accepts any unique subset of the seven Tier-1 envelope dimensions. Missing dimensions remain missing and the resulting screening assessment remains `Incomplete`.

No zero/default/heuristic values are synthesized.

## Revalidation

A native dossier revalidates:

- campaign manifest identity;
- candidate-version identity;
- screening-policy identity;
- underlying #1942 dossier integrity + digest;
- every native envelope against the frozen manifest;
- one-to-one envelope/contribution dimensions;
- exact envelope digest == contribution source receipt;
- exact envelope prediction == contribution prediction.

## Network-free CLI

`energy-native-dossier <input.json>`

Input contains the frozen campaign manifest, external identity assertions and native envelopes. Output contains the assembled native dossier and its SHA-256. Host-local input paths are not emitted.

## Authority boundary

Native dossier assembly proves consistency among the frozen campaign, native envelopes, generic predictions and identity ledger. It does not establish that external identity assertions are scientifically correct, nor does it grant synthesis, certification, manufacturing, procurement, investment, or deployment authority.
