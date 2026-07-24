# Muse Artist Capability & Publishing Integrity Design Specification

**Status:** Proposed  
**Audience:** Muse, Mycelix Music, DKG, product, UX, moderation, and publishing contributors  
**Primary goal:** Make Muse increase artist capability without becoming an autonomous content-spam pipeline  
**Related systems:** Muse Studio, Mycelix Music, Mycelix DKG, release tooling, provenance, rights, moderation  
**Suggested repository path:** `docs/design/MUSE_ARTIST_CAPABILITY_AND_PUBLISHING_INTEGRITY_DESIGN_SPEC.md`

---

## 1. Purpose

Muse should help artists:

- discover ideas;
- develop themes;
- explore alternatives;
- arrange and orchestrate;
- shape performance;
- understand structure;
- preserve provenance;
- prepare releases;
- publish with accurate credits and rights information.

Muse should not optimize for:

- one-click mass publication;
- infinite low-effort catalog generation;
- deceptive claims of human authorship;
- flooding public discovery with near-duplicate outputs;
- replacing artistic judgment with automatic release decisions.

The intended product relationship is:

> Muse proposes, reveals, transforms, and assists.  
> The artist selects, changes, performs, contextualizes, and takes responsibility.

Publishing must therefore be a deliberate transition from **generated material** to **artist-owned release intent**.

---

## 2. Key product decision

Muse should not directly publish to YouTube, Spotify, SoundCloud, or other external platforms through a one-click bulk pipeline.

Muse may provide:

- release preparation;
- validated metadata;
- provenance manifests;
- artwork and asset packaging;
- DDEX-compatible exports;
- distributor-ready bundles;
- external release-link attachment after publication.

External publishing should require a separate creator-controlled step or an approved distribution partner with appropriate safeguards.

Mycelix Music may support native publishing because it can enforce provenance, attribution, rate limits, duplicate detection, and catalog policy directly.

---

## 3. Can Muse detect an unedited Muse output?

### 3.1 Inside Muse and Mycelix

Yes, with high confidence.

Muse can identify an exact or near-exact generated artifact when it has access to the uploaded or published material and the original generation registry.

Detection methods include:

1. **Exact byte hash**
   - Detects an unchanged WAV, MIDI, MusicXML, score, or recipe.

2. **Normalized symbolic hash**
   - Detects the same composition after harmless file-level changes such as metadata removal, track ordering, or serialization differences.

3. **Transformation-aware symbolic fingerprint**
   - Detects a score that remains substantially identical after small note, timing, transposition, or orchestration changes.

4. **Perceptual audio fingerprint**
   - Detects the same render after transcoding, gain changes, trimming, or common compression.

5. **Recipe and lineage match**
   - Detects whether the work derives directly from a known Muse generation event.

6. **Signed export manifest**
   - Preserves an explicit provenance chain when the user exports through Muse.

Inside Mycelix Music, a new upload can be checked against the user's generated and exported artifacts before publication.

### 3.2 Outside the ecosystem

Only sometimes.

Muse cannot reliably know that a user published something elsewhere unless at least one of the following is true:

- the user links the external release;
- the external platform provides a suitable API or partnership;
- the work is publicly discoverable and Mycelix operates a compliant matching service;
- the release preserves a Muse provenance manifest or watermark;
- the work later returns to Mycelix for comparison.

If a user strips metadata, re-renders the MIDI with different instruments, changes tempo, edits the arrangement, or publishes to a platform Muse cannot inspect, certainty decreases.

The system must never claim universal detection.

Recommended language:

> Muse can verify exact and near-exact lineage when artifacts are available for comparison. It cannot guarantee detection across every external platform or after substantial transformation.

---

## 4. Detection confidence levels

Publishing integrity should use explicit confidence levels.

### Exact

Evidence:

- matching content hash;
- matching signed recipe;
- matching score hash;
- matching render identifier.

Meaning:

> This is the exact Muse artifact or canonical serialization.

### Equivalent

Evidence:

- normalized symbolic score is identical;
- only non-musical metadata differs;
- render differs but score is unchanged.

Meaning:

> This is musically the same composition with a different container or rendering.

### Near-derived

Evidence:

- high symbolic similarity;
- preserved motif, form, harmony, and event structure;
- limited edits or re-orchestration.

Meaning:

> This appears directly derived from a Muse artifact with modest transformation.

### Related

Evidence:

- shared motifs, recipe ancestry, or explicit lineage;
- meaningful structural change.

Meaning:

> This belongs to the same creative lineage but is not merely a duplicate.

### Unknown

Evidence is insufficient.

Meaning:

> Muse cannot determine whether the artifact derives from a known generation.

Confidence must be accompanied by the evidence used.

---

## 5. Do not compute a fake “human percentage”

Muse should not claim:

- 73% human;
- 27% AI;
- mostly human;
- authentically human.

Those labels imply a precision the system cannot justify.

Instead, show an objective **Contribution Record**.

Example:

- initial theme: Muse-generated;
- candidate selection: Tristan;
- development structure: Tristan, edited in Studio;
- orchestration: Muse proposal, modified by Tristan;
- piano performance: Tristan;
- lyrics: Tristan;
- mix and master: external DAW;
- final approval: Tristan;
- provenance: complete.

The contribution record describes what happened without pretending to measure creativity numerically.

---

## 6. Creative object states

Every Muse work should move through explicit maturity states.

### 6.1 Sketch

Default state for newly generated material.

Characteristics:

- private by default;
- incomplete metadata allowed;
- no public catalog placement;
- may contain direct Muse output;
- freely editable and exportable.

### 6.2 Study

A saved exploration or experiment.

Characteristics:

- may be shared by link;
- clearly labeled as a study;
- not treated as a finished artist release;
- may remain unedited;
- may include notes about what was explored.

### 6.3 Work in Progress

The creator has begun intentional development.

Requirements:

- named work;
- active author identity;
- at least one saved edit, performance, arrangement, annotation, or selection rationale;
- rights status acknowledged.

### 6.4 Release Candidate

A work intended for public release.

Requirements:

- complete credits;
- rights attestation;
- contribution record;
- selected canonical version;
- finished audio or score;
- duplicate and near-duplicate scan;
- full listen-through confirmation;
- publication intent;
- provenance manifest.

### 6.5 Published Work

A public Mycelix release.

Contains:

- canonical work identifier;
- release version;
- public contributors;
- rights and license claims;
- provenance root;
- recording/master distinction;
- public lineage;
- distribution links;
- dispute and correction mechanisms.

---

## 7. Native catalog classes

Mycelix Music should distinguish finished artist works from direct generative studies without moralizing either.

### Artist Work

Used when the creator has taken responsibility for the release and supplied a contribution record.

May include substantial Muse assistance.

Eligible for:

- standard artist catalog;
- discovery;
- licensing;
- monetization;
- editorial collections;
- release feeds.

### Muse Study

Used for direct or lightly edited Muse output presented as exploration.

Characteristics:

- clearly labeled;
- separate discovery surface;
- lower publication rate limits;
- not automatically placed in an artist's primary discography;
- monetization policy may differ;
- still preserves provenance and authorship claims.

### Adaptive Asset

Used for game, installation, application, or dynamic-media packages.

Contains:

- stems;
- transition rules;
- intensity states;
- looping regions;
- licenses;
- runtime provenance.

### Educational Example

Used for tutorials, analysis, demonstrations, and reproducible research.

These classes should be selected from verifiable workflow evidence and creator declaration, not an opaque score.

---

## 8. Publication gate

Publishing should be a deliberate workflow.

### 8.1 Required steps

1. **Choose canonical version**
2. **Listen through the complete work**
3. **Review duplicate and lineage results**
4. **Complete credits**
5. **Review contribution record**
6. **Declare rights and source permissions**
7. **Choose release class**
8. **Choose license and commercial terms**
9. **Review public provenance**
10. **Confirm publication intent**

The user must not be able to publish hundreds of generated candidates through one unattended action.

### 8.2 Full listen-through

For native Mycelix publication, require that the creator has played the final release candidate from beginning to end at least once after its last material edit.

This is not proof of artistic quality. It is a minimum review safeguard.

Accessibility alternatives must exist for creators who cannot audit through ordinary listening.

### 8.3 Material edit reset

A material edit resets final-review status.

Material edits include:

- score changes;
- arrangement changes;
- new render;
- different master;
- changed lyrics;
- altered duration;
- changed contributors;
- changed license.

Metadata spelling corrections do not require another full review.

---

## 9. Anti-spam controls

### 9.1 No bulk public publishing

Muse may batch-render or batch-export privately.

Public release creation must remain bounded and deliberate.

### 9.2 Rate limits

Rate limits may consider:

- account trust;
- creator verification;
- release class;
- duplicate rate;
- dispute history;
- catalog quality signals;
- human review where required.

Do not use payment alone to bypass integrity controls.

### 9.3 Duplicate suppression

Before publication, compare against:

- the user's own Muse generations;
- the user's existing releases;
- public Mycelix releases;
- known lineage branches;
- withdrawn duplicates where policy permits.

Outcomes:

- exact duplicate: block or require explicit replacement/version action;
- near duplicate: warn and request explanation;
- legitimate variation: attach lineage;
- uncertain: permit with disclosed uncertainty.

### 9.4 Catalog grouping

Multiple variations of the same work should normally appear under one work or project rather than flooding discovery as unrelated singles.

### 9.5 Release cooldown

A newly generated work should not move from generation to public release through a single accidental click.

A short review stage may be required, but the system should avoid arbitrary delays that punish legitimate rapid workflows.

### 9.6 Discovery throttling

Publishing and discovery are separate decisions.

A work may be publicly accessible without receiving immediate broad algorithmic distribution.

---

## 10. DKG provenance model

The DKG should distinguish creative objects and claims.

### 10.1 Core entities

- `Work`
- `WorkVersion`
- `Arrangement`
- `Score`
- `Performance`
- `Recording`
- `Master`
- `Release`
- `DistributionListing`
- `Contributor`
- `Contribution`
- `RightsClaim`
- `License`
- `GenerationEvent`
- `EditEvent`
- `SelectionEvent`
- `RenderEvent`
- `ExportEvent`
- `PublicationIntent`
- `PublicationReview`
- `Fingerprint`
- `SimilarityAssessment`
- `Dispute`
- `Correction`
- `Revocation`

### 10.2 Core relationships

- `derived_from`
- `version_of`
- `arranges`
- `performs`
- `records`
- `masters`
- `released_as`
- `distributed_as`
- `generated_by`
- `edited_by`
- `selected_by`
- `contributed_by`
- `asserted_by`
- `verified_by`
- `licensed_under`
- `supersedes`
- `corrects`
- `disputes`
- `revokes`
- `equivalent_to`
- `near_duplicate_of`

### 10.3 Claim states

A claim may be:

- asserted;
- co-signed;
- corroborated;
- verified;
- disputed;
- corrected;
- revoked.

The DKG proves who made a claim, when, and with what evidence. It does not magically prove copyright ownership.

---

## 11. Artifact fingerprints

Each artifact may have multiple fingerprints.

```rust
pub struct ArtifactFingerprints {
    pub byte_hash: Option<ContentHash>,
    pub normalized_score_hash: Option<ContentHash>,
    pub symbolic_fingerprint: Option<SymbolicFingerprint>,
    pub perceptual_audio_fingerprint: Option<AudioFingerprint>,
    pub recipe_hash: Option<ContentHash>,
    pub render_hash: Option<ContentHash>,
}
```

### 11.1 Symbolic fingerprint

May include:

- normalized note intervals;
- rhythm ratios;
- motif identity;
- form segmentation;
- harmony sequence;
- voice-leading profile;
- orchestration-independent structure.

It should support transposition-aware and tempo-aware comparison.

### 11.2 Audio fingerprint

Should be robust to:

- common codecs;
- loudness changes;
- small trims;
- sample-rate conversion.

It should not be presented as proof of composition ownership.

### 11.3 Privacy

Private sketches and unreleased works should not be added to a globally searchable public fingerprint registry.

Possible architecture:

- local/private fingerprints for private work;
- public fingerprints only after creator-approved publication;
- zero-knowledge or privacy-preserving matching may be explored later;
- clear retention and deletion behavior.

---

## 12. Signed export manifest

Every Muse export should optionally include a sidecar provenance manifest.

Example files:

- `work.wav`
- `work.mid`
- `work.musicxml`
- `work.muse-provenance.json`
- `work.credits.json`

The manifest may contain:

```rust
pub struct MuseExportManifest {
    pub work_id: WorkId,
    pub version_id: VersionId,
    pub generation_events: Vec<GenerationEventId>,
    pub edit_events: Vec<EditEventId>,
    pub contributors: Vec<ContributorClaim>,
    pub artifact_fingerprints: ArtifactFingerprints,
    pub license: Option<LicenseId>,
    pub exported_at: Timestamp,
    pub exporter_identity: AgentId,
    pub signature: Signature,
}
```

Removing the manifest should not make the audio unusable.

The manifest is evidence and convenience, not DRM.

---

## 13. Optional watermarking

Robust watermarking may help identify a specific exported render, but it must be optional, disclosed, and carefully bounded.

Do not:

- degrade audio;
- create hidden surveillance identifiers;
- encode personal identity without consent;
- treat watermark absence as evidence of wrongdoing;
- depend on watermarking as the sole provenance mechanism.

Preferred order:

1. signed manifest;
2. DKG lineage;
3. artifact fingerprints;
4. optional watermark for approved use cases.

---

## 14. Artist contribution workflow

Studio should encourage meaningful authorship through capability, not forced busywork.

Useful creator actions include:

- selecting and rejecting candidates;
- writing intent;
- editing form;
- changing harmony;
- developing motifs;
- arranging;
- orchestrating;
- performing;
- changing expression;
- writing lyrics;
- recording vocals or instruments;
- mixing and mastering;
- annotating meaning;
- choosing license and context;
- curating a journey or release.

Selection can be artistically meaningful, but it should be recorded honestly as selection rather than falsely described as composition.

### 14.1 Contribution record UI

The release builder should show:

- source materials;
- Muse generation events;
- manual changes;
- Studio transformations;
- external imports;
- performances;
- collaborators;
- mixing/mastering steps;
- final approval.

The creator may add context, but cannot delete immutable provenance events from the public release lineage. Private draft history may remain private unless required for a claim.

---

## 15. External publishing package

Muse should prepare, not automatically spam, external destinations.

A release package may contain:

- master WAV;
- alternate master;
- instrumental;
- stems where appropriate;
- MIDI;
- MusicXML;
- artwork;
- lyrics;
- credits;
- contributor roles;
- split sheet;
- rights attestation;
- DDEX metadata;
- provenance manifest;
- release notes;
- external identifier placeholders.

The creator or distributor performs the final external submission.

After release, the creator can attach:

- Spotify URL;
- Apple Music URL;
- SoundCloud URL;
- Bandcamp URL;
- YouTube URL;
- distributor receipt;
- ISRC;
- UPC/EAN.

These become `DistributionListing` nodes linked to the canonical Mycelix work.

---

## 16. UX language

Use language that supports artists.

Prefer:

- Draft
- Study
- Develop
- Arrange
- Perform
- Review
- Prepare release
- Contribution record
- Muse-assisted
- Provenance complete
- Needs review
- Similar to an existing version

Avoid:

- Generate content
- Auto-publish
- Humanize
- Human percentage
- Authenticity score
- Spam risk score shown as moral judgment
- AI-owned
- Fully original without evidence

---

## 17. Moderation and disputes

### 17.1 Publication review triggers

Possible triggers:

- exact duplicate of another creator's release;
- repeated near-duplicate publication;
- conflicting rights claims;
- missing contributor approval;
- suspicious external-source lineage;
- prohibited content;
- automated bulk behavior;
- false provenance statements.

### 17.2 Creator appeal

Creators must be able to:

- inspect the evidence;
- explain legitimate derivation;
- attach licenses;
- identify public-domain sources;
- correct credits;
- appeal moderation;
- publish as a version or study where appropriate.

### 17.3 Corrections

Public provenance must support correction without erasing history.

A corrected release should:

- supersede the prior claim;
- preserve the correction event;
- update current public status;
- retain dispute evidence under appropriate privacy controls.

---

## 18. Privacy boundaries

Private creative history can reveal:

- unfinished ideas;
- emotional intent;
- collaborators;
- listening habits;
- commercial plans;
- unreleased works.

Default rules:

- sketches remain local/private;
- private fingerprints are not public;
- contribution history is private until release;
- the creator chooses which evidence becomes public;
- public claims must include enough evidence to be meaningful;
- collaborators approve public credit;
- deletion and revocation semantics are explicit.

---

## 19. Architecture boundaries

Recommended services:

- `muse-generation-registry`
- `muse-artifact-fingerprint`
- `muse-contribution-ledger`
- `mycelix-release-builder`
- `mycelix-provenance-dkg`
- `mycelix-publication-policy`
- `mycelix-duplicate-detection`
- `mycelix-rights-claims`
- `mycelix-distribution-links`

Muse composition should not directly call external publishing APIs.

The release builder emits a reviewed release package and public provenance intent.

---

## 20. Suggested data types

```rust
pub enum CreativeState {
    Sketch,
    Study,
    WorkInProgress,
    ReleaseCandidate,
    Published,
    Withdrawn,
}
```

```rust
pub enum ReleaseClass {
    ArtistWork,
    MuseStudy,
    AdaptiveAsset,
    EducationalExample,
}
```

```rust
pub struct ContributionRecord {
    pub work_version: VersionId,
    pub contributions: Vec<ContributionEvent>,
    pub contributor_attestations: Vec<ContributorAttestation>,
    pub generation_lineage: Vec<GenerationEventId>,
    pub external_sources: Vec<ExternalSourceClaim>,
    pub final_approver: AgentId,
}
```

```rust
pub struct PublicationReview {
    pub candidate_version: VersionId,
    pub full_review_complete: bool,
    pub duplicate_assessment: SimilarityAssessment,
    pub credits_complete: bool,
    pub rights_attested: bool,
    pub contribution_record_complete: bool,
    pub provenance_manifest_ready: bool,
    pub release_class: ReleaseClass,
    pub warnings: Vec<PublicationWarning>,
}
```

---

## 21. Implementation phases

## P0 — Provenance-aware release preparation

- creative maturity states;
- exact artifact hashes;
- normalized MIDI/score hashes;
- contribution record;
- release candidate workflow;
- full listen-through confirmation;
- credits and rights attestation;
- signed export manifest;
- manual external distribution package;
- no one-click external publication.

## P1 — Native Mycelix publishing integrity

- Mycelix release builder;
- public DKG release lineage;
- exact and near-duplicate checks;
- release classes;
- catalog grouping;
- publication rate limits;
- correction and dispute flow;
- external release-link attachment.

## P2 — Advanced derivation detection

- symbolic similarity fingerprint;
- perceptual audio fingerprint;
- transformation-aware lineage suggestions;
- creator-confirmed equivalence;
- privacy-preserving private matching;
- distributor receipts and DDEX delivery evidence.

## P3 — Trusted distribution partnerships

- approved distributor integrations;
- explicit review checkpoints;
- partner-specific rate controls;
- signed delivery events;
- no autonomous bulk publishing;
- platform policy compliance.

---

## 22. Acceptance criteria

The system is ready for alpha when:

1. New generations default to Sketch.
2. Exact Muse artifacts can be detected on Mycelix upload.
3. Semantically identical MIDI files can be detected after metadata changes.
4. The system distinguishes exact, equivalent, near-derived, related, and unknown.
5. The release builder shows an objective contribution record.
6. The interface never reports a fake human/AI percentage.
7. Public release requires a canonical version, credits, rights attestation, and publication intent.
8. Material edits invalidate the prior final-review confirmation.
9. Direct Muse output can be labeled as a Muse Study without pretending it is a developed artist work.
10. Multiple variations are grouped under a work rather than flooding discovery.
11. Publishing and discovery eligibility are separate.
12. Private sketches are not entered into a public fingerprint registry.
13. Every public release has a DKG provenance root.
14. Creators can inspect and appeal duplicate or rights findings.
15. External release preparation does not automatically publish content.
16. The system clearly states that external detection is limited and cannot be universal.

---

## 23. Open questions

- Should Muse Studies be monetizable by default?
- What minimum evidence should distinguish an Artist Work from a Muse Study?
- How should pure curation be credited?
- Which symbolic similarity threshold is useful without creating false accusations?
- Should native releases require collaborator co-signatures before monetization?
- How long should private artifact fingerprints be retained?
- Which public-domain and traditional-material workflows need special handling?
- How should external DAW edits be imported into the contribution record?
- Can distributors preserve the Muse provenance manifest?
- Should users be able to publish a direct Muse output to their primary catalog if clearly labeled?
- What discovery policy best supports experimentation without rewarding catalog flooding?

---

## 24. Final design intent

Muse should make artists more capable, not make artistic responsibility disappear.

The system should help a creator move from:

> Here is something Muse generated.

to:

> Here is the work I selected, understood, shaped, performed, contextualized, credited, and chose to release.

Provenance should make that creative history visible without pretending that software can calculate authorship as a percentage or detect every external use.

The defining publication rule is:

> Generation may be immediate. Publication must be intentional.
