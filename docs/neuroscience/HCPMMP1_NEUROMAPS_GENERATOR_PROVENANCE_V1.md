# HCP-MMP1 Lineage-B generator provenance v1

Status: **implementation-byte provenance for the Lineage-B candidate derivation**

The Lineage-B scientific commitment must identify not only the HCP/template inputs and Workbench toolchain, but also the exact Symthaea code bytes that parse GIFTI labels, enforce provenance, and orchestrate the transform.

## Implementation root

At runtime the derivation hashes exactly three executing source modules:

- `hcpmmp_neuromaps_common.py`;
- `hcpmmp_neuromaps_gifti.py`;
- `derive_hcpmmp1_neuromaps_lineage_b.py`.

Their SHA-256 map is canonically serialized and hashed again into `generator_implementation.digest`.

That aggregate digest is included in the scientific-input commitment and copied into each semantic output's source provenance as `generator_implementation_digest`.

A forgotten human version bump therefore cannot preserve the same scientific commitment after code bytes change.

## Execution stability

The derivation establishes input/tool identity before execution with #523's `verify_inputs()` boundary. After Workbench resampling and semantic normalization, it runs that byte verification again and also recomputes the generator implementation root.

Evidence is not emitted if:

- any scientific source/template file changed during execution;
- the Workbench executable or version-output identity changed;
- any of the three Symthaea generator modules changed.

## Evidence structure

`derivation-evidence.json` carries both the per-module SHA-256 map and its aggregate implementation digest. The semantic output source objects carry the aggregate digest and must bind it to the same evidence object.

This implementation root does not itself prove that the code is scientifically valid. Qualification of the code remains a separate CI/review theorem; the root answers only **which exact implementation bytes produced the candidate evidence**.

The retained-root verifier in #525 re-binds this implementation root under the external evidence-root authority boundary.
