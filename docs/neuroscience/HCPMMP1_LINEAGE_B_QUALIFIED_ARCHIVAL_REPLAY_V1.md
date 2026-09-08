# HCP-MMP1 Lineage-B Qualified Archival Replay v1

Status: **candidate cross-lineage compatibility theorem; hosted qualification pending**

## Purpose

Lineage-B now has two deliberately separate Git lineages:

```text
scientific generator / execution / custody lineage

and

archival verifier lineage (#525)
```

The separation is useful only if newer scientific-generator states can explicitly demonstrate that the independently qualified archival verifier still reconstructs their evidence contract.

A sibling PR existing is not ancestry, and a verifier that was qualified against one generator revision is not automatically qualified against all future revisions.

Therefore:

```text
QualifiedVerifierRoot
    + CurrentGeneratorWorld
    -> ExplicitCrossLineageReplay
```

not:

```text
QualifiedVerifierRoot
    -> UniversalFutureCompatibility
```

## Exact qualified verifier root

The source authority for this replay is the hosted-qualified #525 exact head:

```text
633376c58ff03973c50461451cb465c786dbad7c
```

Its dedicated `HCPMMP Lineage B Evidence Verifier` run `34268165716` passed exact-head checkout, syntax, all ten archival verifier contracts, non-execution/no-network ratchets, generator rebinding, authority ratchets, and CLI qualification.

This replay consumes that exact source root as an explicit external qualification input. It does not silently copy the verifier into the current scientific lineage.

## Replay subject

The initial replay subject is the Lineage-B generator/evidence world at the parent of this tranche:

```text
#977
3b99694b2570405614b0f6c0d41ab74ac6cba73f
```

That world intentionally differs from the historical generator seen by #525 because #977 changes `derive_hcpmmp1_neuromaps_lineage_b.py` to strengthen publication custody.

The replay must therefore prove it is genuinely crossing a generator revision rather than accidentally running the verifier against its own historical `derive.py`.

## Import topology

A full checkout of #525 placed first on `PYTHONPATH` would be wrong:

```text
#525 verifier
+ #525 historical derive/common/gifti
```

would only re-run #525 against itself.

The qualification workflow instead creates a verifier-only overlay:

```text
current exact subject checkout
        +
separate exact #525 checkout
        ↓
copy only:
  verify_hcpmmp1_neuromaps_lineage_b_evidence.py
  test_verify_hcpmmp1_neuromaps_lineage_b_evidence.py
        ↓
overlay data -> current subject data
        ↓
PYTHONPATH = overlay/scripts : current/scripts
```

The resulting imports must satisfy:

```text
verifier.__file__ -> qualified #525 overlay

derive.__file__  -> current subject scripts
common.__file__  -> current subject scripts
gifti.__file__   -> current subject scripts
```

The workflow fails if the overlay contains historical copies of the generator modules.

## Source binding

The workflow independently checks out the qualified verifier source at exactly:

```text
633376c58ff03973c50461451cb465c786dbad7c
```

and asserts that checkout's `HEAD` before constructing the overlay.

The copied verifier and test files must be byte-identical to that exact checkout.

The replay receipt records:

- current qualification subject SHA;
- exact qualified verifier Git SHA;
- SHA-256 of the verifier source actually replayed;
- SHA-256 of the verifier test source actually replayed;
- the current subject's generator implementation digest;
- contract count;
- narrow authority state.

## Cross-revision ratchet

The workflow requires the current subject's:

```text
scripts/derive_hcpmmp1_neuromaps_lineage_b.py
```

to differ byte-for-byte from the historical `derive.py` in the qualified #525 checkout.

This prevents the replay from receiving compatibility credit if it accidentally collapses back to the historical generator world.

## Replay theorem

All ten qualified #525 archival verifier contracts are executed with:

```text
VerifierImplementation = exact qualified #525 source
GeneratorImplementation = current subject source
Method/Area Data         = current subject data
```

If they all pass, the narrow theorem is:

```text
cross_lineage_contract_compatibility_established = true
```

for that exact pair of source roots.

This means the qualified verifier contract is structurally compatible with evidence generated according to the current Lineage-B schema/generator semantics exercised by those ten synthetic archival contracts.

It does **not** mean a real retained evidence bundle has been verified.

## Authority boundary

A successful replay may establish only:

```text
qualified_verifier_source_bound = true
cross_lineage_contract_compatibility_established = true
```

while requiring:

```text
real_evidence_bundle_verified = false
scientific_execution_qualified = false
atlas_correctness_established = false
fmq010_established = false
neural_alignment_established = false
consciousness_evidence = false
```

Core invariant:

```text
VerifierCompatibility != EvidenceVerification != ScientificValidity
```

The synthetic replay exercises the verifier contract against the newer generator world. It does not provide real HCP/BALSA bytes, execute Workbench, establish execution provenance, or verify a retained real evidence root.

## Why this belongs outside GeneratorImplementationRoot

This tranche changes only qualification workflow/documentation. It does not alter `derive`, `common`, `gifti`, snapshot logic, or transform execution.

Therefore the replay machinery belongs to the qualification envelope:

```text
QualifiedVerifierReplay -> QualificationEnvelopeRoot
```

not the scientific generator root.

A future verifier implementation change similarly does not create a new scientific result by itself; it creates a new qualification theorem that must be applied to the retained result.

## Relationship to snapshot/execution integration

This replay closes the current cross-lineage verifier topology gap, but the next scientific integration will deliberately change the generator/evidence world again by introducing snapshot-bound inputs and isolated Workbench execution.

That future tranche must replay the qualified archival verifier again after its coordinated schema/generator migration.

In particular it must not infer:

```text
#977 compatibility
    -> compatibility with future snapshot/executor generator
```

Compatibility is pair-specific:

```text
ReplayQualification = f(QualifiedVerifierRoot, SubjectGeneratorRoot, EvidenceSchema)
```

## Non-claims

This tranche does not establish:

- authorized HCP/BALSA acquisition;
- real Lineage-B execution;
- snapshot-integrated transform execution;
- same-host repeatability;
- path equivalence;
- cross-CPU equivalence;
- real retained evidence verification;
- atlas correctness;
- FMQ-010;
- neural alignment;
- consciousness evidence.
