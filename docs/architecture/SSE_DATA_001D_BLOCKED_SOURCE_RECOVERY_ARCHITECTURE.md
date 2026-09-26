# SSE-DATA-001D — Blocked-source recovery and exact-capture admission architecture

Status: architecture-only design subject

Date: 2026-09-26

Parents / related work:

- SSE-DATA-001A #6087 — benchmark custody architecture
- SSE-DATA-001B #6092 — frozen source manifest
- SSE-DATA-001C #6094 — exact source-capture receipt
- ION-001A #6072 — ion-transport evidence architecture

## Purpose

Define how a benchmark may recover from an unavailable, timed-out, licensed, relocated, or otherwise uncaptured primary scientific source without quietly substituting a convenient copy and then treating it as the original authority.

The motivating current state is `SSE-SRC-001`:

```text
paper / metadata verified
support repository pinned
primary conductivity dataset endpoint attempted
-> timeout
-> exact primary dataset bytes unresolved
```

This is a valid scientific-custody outcome. The pipeline must be able to remain blocked.

## Core theorem

```text
source unavailable
!= permission to reconstruct it

same-looking table
!= same artifact

same row count
!= same dataset

same numeric values
!= same provenance

support code
!= primary scientific data

paper table
!= full source dataset
```

Recovery succeeds only when the provenance relation between a newly obtained artifact and the intended source is itself evidence-bearing.

## Source recovery state machine

Suggested states:

```text
DeclaredSource
MetadataVerified
CaptureAttempted
PrimaryArtifactUnresolved
CandidateRecoveryArtifactFound
RecoveryProvenanceUnderReview
ExactPrimaryArtifactCaptured
AuthoritativeEquivalentCaptured
ConflictingCandidateArtifacts
UnavailableUnderCurrentRights
SupersededSourceGeneration
```

Do not collapse these into `available=true/false`.

## Recovery classes

At minimum distinguish:

### 1. Original endpoint retry

The exact declared source endpoint becomes reachable later.

Bind:

- original URL/identifier;
- prior failed-attempt receipt;
- retrieval timestamp as provenance;
- response metadata where available;
- exact obtained bytes/content hash;
- redirect chain where material;
- content type/size;
- license/terms state;
- relation to frozen source manifest.

A later successful fetch creates a new capture receipt lineage. It does not mutate the earlier failed receipt.

### 2. Author-declared canonical mirror

The authors/publisher explicitly identify a mirror, repository, DOI supplement, institutional archive, Zenodo/Figshare record, or equivalent as the source artifact.

Require evidence of the declaration/relationship.

```text
authoritative mirror relation
+ exact mirror artifact identity
-> candidate authoritative equivalent
```

Still preserve the fact that the bytes came from a different endpoint.

### 3. Publisher supplementary artifact

A publisher-hosted supplementary file may be admitted only if it can be linked to the exact article/version and exact data role.

A supplementary table that is merely a subset or transformed view must not silently become the full benchmark dataset.

### 4. Repository artifact with publication linkage

A Git repository may contain the primary data, support code, derived outputs, or all three.

Require path-level role classification:

```text
PrimaryScientificDataset
DerivedDataset
AnalysisIntermediate
TrainingDataset
CodeOnly
DocumentationOnly
UnknownRole
```

Repository membership alone does not establish `PrimaryScientificDataset`.

### 5. Author-provided artifact / hash

If an author provides exact data later, bind:

- communication/provenance receipt where permitted;
- exact bytes/hash;
- statement of relation to published source;
- version/date;
- license/allowed-use state.

The received artifact becomes a new capture lineage; never rewrite the historical unavailable state.

### 6. Third-party mirror

A third-party copy is **not** authoritative merely because filenames, row counts, or sample values match.

It may be retained as a `CandidateRecoveryArtifact` for reconciliation, but not admitted until provenance/equivalence is established under a declared profile.

## Explicitly rejected shortcuts

The following cannot satisfy primary-source recovery on their own:

- OCR of figures/tables;
- screenshots;
- values copied from the article prose;
- a Kaggle/Drive/random mirror with no provenance chain;
- an LLM-generated reconstruction;
- values scraped from citations of the original paper;
- support notebooks that contain only transformed training arrays;
- a later paper reprinting a subset;
- matching filename;
- matching row count;
- matching first/last rows;
- matching aggregate statistics.

Such artifacts may be useful secondary evidence but must retain that class.

## `RecoveryArtifactCandidateV1`

Conceptually bind:

```text
RecoveryArtifactCandidateV1 {
  intended_source_ref,
  candidate_artifact_identity,
  retrieval_origin,
  retrieval_receipt,
  claimed_relation,
  relation_evidence_refs,
  license_state,
  byte_size,
  content_digest,
  format,
  candidate_role,
  limitations,
}
```

No constructor should imply equivalence merely from content similarity.

## Provenance relation vocabulary

Prefer explicit relations such as:

```text
ExactOriginalEndpointArtifact
AuthorDeclaredCanonicalMirror
PublisherDeclaredSupplement
RepositoryDeclaredPrimaryDataset
AuthorProvidedPublishedVersion
ByteIdenticalToCapturedAuthoritativeArtifact
DeterministicallyDerivedFromAuthoritativeArtifact
ThirdPartyMirrorUnverified
SecondaryReproduction
PartialExtract
TransformedDerivative
UnknownRelation
ConflictingClaim
```

Only narrowly defined relations can advance toward authoritative capture.

## Byte identity versus semantic equivalence

The strongest simple case is byte identity:

```text
candidate bytes hash == authoritative captured bytes hash
```

but often the original cannot be fetched for direct comparison.

Then semantic/equivalence claims require stronger provenance evidence, not weaker hashing.

Do not define a generic tolerance-based equivalence over datasets and call it source identity.

```text
all numeric rows approximately equal
!= same scientific artifact
```

## Transform lineage

A deterministic conversion may be scientifically useful if its ancestry is explicit.

Example:

```text
exact XLSX source
+ exact parser
-> normalized CSV
```

The CSV is a `DeterministicallyDerivedFromAuthoritativeArtifact` child, not the original source.

A transformed artifact must bind:

- parent artifact;
- exact transformation implementation/profile;
- output digest;
- lossiness classification;
- excluded fields/rows;
- ordering behavior;
- unit transformations;
- warnings/refusals.

## Licensing / rights state

Scientific custody cannot ignore usage rights.

Track independently:

```text
PublicDomainOrEquivalent
OpenLicense
AcademicUseAllowed
CommercialUseRequiresPermission
AccessControlled
LicenseUnknown
UseNotPermittedUnderCurrentContext
```

License state does not determine scientific truth, but may determine whether Symthaea can store, redistribute, transform or use an artifact in a given workflow.

Do not mirror restricted bytes into the repository merely because they are technically downloadable.

## Retrieval receipts

A retrieval attempt is evidence even when it fails.

Bind where possible:

- requested identifier/URL;
- resolution/redirect outcome;
- status/failure class;
- retrieval environment/tool identity;
- response headers relevant to identity;
- observed content length/type when available;
- obtained digest when successful;
- bounded diagnostic;
- retry profile.

Possible outcomes:

```text
Captured
Timeout
DNSFailure
HTTPError
AccessDenied
AuthenticationRequired
LicenseBlocked
RedirectUnresolved
ContentChanged
UnexpectedContentType
PartialDownload
Indeterminate
```

Do not convert `Timeout` into `DoesNotExist`.

## Mutable endpoints

A stable URL is not an immutable artifact.

If the same endpoint later yields different bytes:

```text
same URL
+ changed bytes
-> new source artifact generation
```

Preserve both capture receipts.

If publication metadata cannot establish which generation corresponds to the published analysis, the source remains ambiguous.

## Versioned repositories

For Git-hosted scientific artifacts, prefer exact commit + tree/blob identity rather than branch names.

```text
repo@main
!= immutable source

repo@commit + path + blob
= exact external artifact identity
```

But even exact Git identity still needs role/provenance classification before scientific row admission.

## Primary versus derived data

A source bundle may contain:

- raw experimental observations;
- cleaned observations;
- deduplicated rows;
- descriptors/features;
- train/test arrays;
- normalized targets;
- model predictions.

Keep them distinct.

A benchmark that intends to represent experimental conductivity should not accidentally ingest model-ready transformed targets and then call them raw measurements.

## Reconciliation when multiple candidates exist

If two plausible source artifacts disagree, preserve the conflict.

Do not choose the more convenient one automatically.

Use dispositions such as:

```text
ByteIdentical
SemanticallyCompatibleWithDeclaredTransform
SubsetRelationship
SupersetRelationship
RowLevelConflict
MetadataConflict
VersionConflict
RoleConflict
InsufficientEvidenceToReconcile
```

The benchmark may remain blocked while conflict exists.

## Source-capture admission witness

A positive runtime/qualified witness should conceptually require:

```text
frozen source manifest entry
+ exact recovery artifact identity
+ accepted provenance relation
+ allowed-use/license state for the requested operation
+ successful integrity verification
+ no unresolved conflicting candidate with stronger/equal authority
-> CapturedSourceArtifactWitness
```

This witness establishes source custody only.

It cannot deserialize directly from stored JSON into live authority without replay/verification.

## Transition into parsing

Only a captured source artifact witness may enter the parser/normalization stage.

```text
CapturedSourceArtifactWitness
-> parser request
-> parse receipt
-> raw rows
```

not:

```text
paper URL
-> parsed benchmark rows
```

For the current Hargreaves conductivity dataset, this gate remains closed.

## Negative-result memory

Store failed capture attempts and rejected mirrors so future agents do not repeat the same unsafe shortcuts.

Examples:

- endpoint timed out under exact attempt profile;
- third-party mirror lacked provenance;
- support repo contained code but not licensed primary CSV;
- supplementary table was only a subset;
- candidate mirror disagreed with article row counts;
- artifact was post-publication transformed output.

A later legitimate recovery may supersede the block while preserving the earlier evidence.

## Required hostile fixtures

The first synthetic/reference corpus should include at least:

1. exact original endpoint succeeds with captured hash -> admissible source custody;
2. original endpoint timeout -> unresolved, not missing forever;
3. same URL later returns changed bytes -> new generation;
4. author-declared mirror with exact hash -> authoritative-equivalent candidate;
5. random mirror with same filename -> reject;
6. random mirror with same row count -> reject;
7. random mirror with matching first ten rows -> reject;
8. OCR reconstruction of paper table -> secondary reproduction only;
9. article supplementary file contains subset only -> no full-dataset promotion;
10. Git repo contains notebooks but no primary data -> support only;
11. Git repo contains primary data with publication linkage -> eligible exact capture;
12. Git branch moves after manifest freeze -> exact pinned commit remains stable;
13. author provides later corrected dataset -> new source generation, not rewrite;
14. two author/publisher artifacts conflict -> preserve conflict/block;
15. deterministic CSV conversion from exact XLSX -> derivative, not original;
16. lossy transformation hides missing rows -> reject as lossless derivative;
17. academic-use-only artifact requested for redistribution -> license blocks that action;
18. license unknown -> scientific metadata may be retained, restricted use remains blocked;
19. partial download has digest -> no complete-artifact witness;
20. unexpected HTML error page downloaded with `.csv` name -> reject by content/profile;
21. support training array numerically matches raw targets -> still transformed derivative;
22. later paper reproduces values -> secondary source, not original custody;
23. LLM reconstructs dataset from prose -> no source authority;
24. third-party artifact becomes acceptable only after author explicitly declares it canonical -> new relation evidence/new lineage;
25. capture PASS promoted directly to conductivity truth -> reject authority promotion.

## Suggested implementation train

```text
SSE-DATA-001D0
this architecture

SSE-DATA-001D1
synthetic recovery/capture corpus

SSE-DATA-001D2
independent known-answer custody validator

SSE-DATA-001D3
exact retrieval/capture adapter over reviewed source classes

SSE-DATA-001E
parser / source-row normalization after primary artifact capture

SSE-DATA-001F
subject / alias / leakage graph

SSE-DATA-001G
condition-bearing observation corpus
```

Do not make SSE-DATA source recovery a general web crawler. Generic retrieval/process primitives should be reused where qualified; this layer owns scientific source-custody semantics only.

## Qualification rule

Each executable capture/recovery subject receives exact-head qualification.

A capture PASS proves that the intended external artifact was obtained/identified under the exact provenance profile. It does not prove the data are scientifically correct, compatible, independent, representative, or suitable for a model benchmark.

## Claim ceiling

SSE-DATA-001D may establish an auditable provenance relation and exact artifact identity for a previously blocked scientific source. It does not establish ionic conductivity, material identity, transport-model validity, benchmark-row correctness, synthesis, battery performance, or discovery.
