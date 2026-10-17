# Melothaea Rights-Aware Repertoire and Live Jam Partner v1

**Status:** product/engineering contract and research plan; not legal advice and not a claim that any specific use is lawful.  
**Core ambition:** a musician brings a song they are authorized to use; Melothaea understands enough to join the session; Symthaea listens and adapts to the human partner without silently turning every source into an unlicensed transcription, arrangement, recording, or training asset.

## 1. Paradigm shift: from song copying to rights-aware participation

The naive pipeline is “identify popular song → ingest recording → transcribe everything → generate a near-identical arrangement → save/export/share.” It is technically tempting but product-wise brittle. It collapses distinct rights, confuses a song with a particular master recording, accumulates derived assets, and cannot decide legality from a file path or recognition confidence.

The proposed paradigm is **repertoire-aware participation**:

1. Understand only the musical information needed to join: pulse, meter, likely key/chord function, form, phrase entrances, density and confidence.
2. Treat the human musician as an active partner. Hear their timing, cues, dynamics and choices; coordinate around them rather than replaying a static transcription.
3. Keep source recording, inferred musical-work content, newly generated performance and exported/recorded artifacts separate.
4. Use an authorized input route for the operation being performed; keep a provenance receipt and action-specific permissions.
5. When source or intended use is unclear, request a licensed file/catalog or fall back to a clean-room mode that composes from an allowed brief. Never present a model decision as a legal ruling.

The ambition is not to make Melothaea less musical. It is to make it more adaptive while reducing unnecessary copying and making permissions legible.

## 2. Rights are layered, not a single song-level boolean

A popular song may involve separately controlled layers:

- **Musical composition:** melody, harmony, lyrics and composition rights. A newly played cover does not use the original master recording, but it still performs the musical work.
- **Sound recording/master:** the specific released recording. Using its stems, samples or audio inside a product can involve rights beyond those for a new cover.
- **Arrangement/adaptation:** a changed arrangement, transcription, interpolation, remix or other adaptation may require additional permission depending on the facts and jurisdiction.
- **Performer/voice consent:** recording or imitating a particular performer raises issues distinct from composing a similar-feeling part.
- **Action and destination:** private rehearsal, public performance, making a cover recording, distributing audio/MIDI/notation, attaching music to video, and training a model are not interchangeable.
- **Platform terms:** access to a streaming service is not automatically permission to capture, analyse, download, transform or feed its content into another system.

So “local file,” “user supplied,” “public-domain composition,” “licensed recording,” and “permitted to publish” must not be treated as synonyms. Public-domain status of a composition does not by itself clear a modern recording of it.

## 3. Product modes

### A. Listen-along / clean-room jam

**Goal:** play with a human who has a track open in its normal licensed player, without Melothaea capturing or analysing that platform's audio.

- The user controls the player and the applicable service terms.
- Melothaea listens to live human instrument/microphone input, follows pulse, infers the active section, plays an original complementary part, and adapts to cues.
- No hidden loopback capture, stream scraping, downloading or routing platform audio through the analysis pipeline.
- Do not claim to have transcribed the protected recording when the system only followed live room/performance audio.
- Public-playback and venue obligations depend on where and how the music is played; excluding the original audio from Melothaea does not erase those obligations.

This is the lowest-friction first experience for users who want to jam now and do not need an exact transcription.

### B. Local analysis of an authorized source

**Goal:** the user selects an audio file or stems and asks for help understanding or rehearsing a piece.

- Ask intended use: local practice, transcription, arrangement, recording, publication or training.
- Record the asserted source basis and any supporting evidence. A local path or successful import is not proof of rights.
- Default to local processing, no source-audio upload and no persistent copy unless enabled for a specific purpose.
- Store minimal useful state. Ephemeral beat/chroma/section estimates are lower-retention than full stems or a note-for-note score, but are not automatically rights-exempt.
- Never show a “safe” badge solely because the file came from the user. Ambiguous rights or uses go to review.

### C. Licensed catalog / creator-authorized repertoire

Integrate only a rightsholder, publisher, distributor or provider whose agreement explicitly authorizes the relevant technical actions. The permission must cover each intended operation: audio analysis, transcription, arrangement, real-time performance, caching, recording, export, publication, territory, duration, attribution and any model-training use. An available streaming API is insufficient evidence by itself.

Permissions must be scope-limited and revocable. On expiry/revocation, block future use within the affected scope, stop refresh/caching and retain only records the applicable terms and law permit.

### D. Public-domain / open-license repertoire

Track composition and exact recording/stems separately. Record provenance, license identifier/version, attribution, territory if relevant, commercial-use/adaptation permissions and share-alike duties. Conflicting/incomplete evidence remains unresolved. Never infer a recording's public-domain status from the composition's age alone.

### E. Original-work fallback

When source/action cannot be cleared, preserve the musical goal without cloning the song. Accept a user-authored chord chart or a high-level brief such as “up-tempo minor-key funk with a sparse verse and a rising chorus.” Generate a distinct part or original piece. Do not promise that a style prompt, transformation or low similarity score guarantees non-infringement.

## 4. Action-scoped authorization contract

Do not implement one **copyright_safe: bool**. An eventual versioned repertoire-use request should contain:

- source identifier/type and whether the app received audio bytes, symbolic notation, live-performance audio or metadata only;
- user intent and destination;
- jurisdiction/territory and session context where known;
- separate evidence references for composition, master recording, platform/API terms, performer/voice consent and publication/venue/distribution permissions when relevant;
- allowed operations, evidence issuer, validity interval, revocation state and required attribution;
- processing mode: ephemeral local, persistent local, licensed service or cloud; retention choice;
- an audit event for gated export/recording/remote-processing actions, without copying source audio into the audit log.

The policy layer is **a documented-permission gate, not a legal oracle**. It should return one of:

- **AllowedByRecordedGrant:** every scope required by the product route is supported by current, verified evidence.
- **AllowedInExplicitLocalMode:** a narrowly scoped, user-confirmed local action is enabled by product policy, with no assertion that this overrides a legal or contractual restriction.
- **BlockedBySourceTerms:** relevant provider/source terms explicitly prohibit the operation.
- **NeedsRightsReview:** evidence is missing, ambiguous, expired, jurisdiction-sensitive or contradictory.

Do not infer fair use/fair dealing, compulsory licenses, personal-use rules or other statutory routes merely because no rightsholder has been found. Those require explicit, jurisdiction-aware legal policy approved for the product. User attestation must stay distinct from independently verified permission.

### Conservative default matrix

| Operation | Default product behavior |
|---|---|
| Open an external player under its normal terms | Open/link only; do not capture or analyse its protected audio |
| Analyse a user-selected local recording | Local-only; record the claimed basis; require scoped confirmation/evidence for the requested operation; review uncertain cases |
| Transcribe a full melody/lyrics or create a close arrangement | Require scope review; a personal copy does not automatically clear transcription/adaptation rights |
| Play a new live cover | Original master need not be used, but composition/performance and venue rules may apply |
| Record a cover for private rehearsal | Local and private by default; assess recording and retention rights separately |
| Publish/stream a cover or upload with video | Destination-specific rights review/licensing, including relevant composition and synchronization/recording rights |
| Export/share full-song MIDI, stems, tablature or notation | Disable unless the specific export and redistribution rights are recorded |
| Train/fine-tune on source audio/transcriptions | Disable unless explicit training permission and provider terms support it |
| Redistribute/embed the original master/stems | Disable without rights for the master and intended distribution |

This matrix intentionally errs toward asking for evidence. It is not a determination that a particular use is unlawful or lawful.

## 5. Provider and jurisdiction guardrails

### Streaming services

Spotify's current developer terms prohibit using Spotify Content to train or ingest into AI/ML models, and restrict modifying, creating derivative works from, or analysing Spotify Content. Do not build a “paste Spotify URL, capture stream, transcribe it” path. A metadata link, an explicitly authorized integration, a user-provided file and a captured stream are different sources with different permissions. Re-check terms before each integration/release.

### United States

The U.S. Copyright Office advises musicians not to assume they may freely use another artist's musical work or recording; no universal minimum duration automatically makes a use permission-free. Section 115 provides a limited compulsory-licensing route for certain phonorecord reproductions/distributions of musical works. It is not blanket permission to copy another artist's master, use any service's stream, make every arrangement or synchronize a cover to video. Match the actual action to the applicable right/license.

### South Africa

SAMRO describes public-performance rights and public-music-use licensing. CAPASSO describes mechanical/reproduction licensing and notes that rights in the master recording must be obtained separately from the label/producer. CAPASSO's physical licensing application also asks for permission for non-owned sound recordings and publisher permission for adaptations such as arrangements, translations, remixes and mashups.

The Constitutional Court delivered judgment on the Copyright Amendment Bill and Performers' Protection Amendment Bill on 26 June 2026. The judgment assessed the constitutionality of provisions in proposed amendments; do not treat it, by itself, as proof that every proposed amendment has commenced as operative legislation. Product decisions must check the latest enacted text, commencement and relevant licences before relying on any exception.

These are conservative product guardrails, not a legal opinion about any particular private jam/transcription. Outcomes depend on jurisdiction, source, exact use and distribution.

## 6. Live performance architecture

Keep the real-time musical system separate from the source-rights path:

1. **Input broker:** distinguish live room/instrument audio, user-local audio, licensed stems, symbolic score and metadata; assign each source a provenance record.
2. **Low-latency perception:** estimate beat phase, tempo, meter, downbeat, section, chord function, motif/phrase boundaries and confidence. Maintain alternate hypotheses instead of fabricating certainty.
3. **Compact musical state:** current pulse, harmony confidence, form position, density, human timing tendency, current role and cue history. Do not retain a permanent full transcription by default.
4. **Partner policy:** select a role—lock with bass/drums, comp, answer a phrase, leave space or support a transition—plus register, density, articulation and the next safe decision boundary. Observe the human's playing, not only the reference song.
5. **Separate gates:** musical constraints handle range, playability, collisions, stuck notes and intentional silence; rights gates handle data paths, export, recording, training and distribution. Passing one never implies passing the other.
6. **Rendering/control:** schedule against a shared clock; apply tempo/section changes at defined boundaries; provide an explicit, reversible stop/mute.
7. **Evidence receipt:** record software/model versions, configuration, event timing, confidence, cues, provenance and output hashes where appropriate. Do not store source audio or exact derived notation in public logs.
8. **Human control:** visible follow/lead/leave-space/stop controls, uncertainty indicators, hard stop, and simple corrections for key, tempo, section and role.

**Core shift:** the best partner is not the one that transcribes every note perfectly. It is the one that joins at the right moment, contributes musically, hears what the person does next and gracefully recovers when its interpretation is wrong.

## 7. Evaluation suite and initial targets

All targets below are **proposed research gates**, not measured results or universal standards. Calibrate them against strong baselines and freeze fixtures before making release claims.

### A. Music understanding from authorized audio

- Tempo absolute error and interval coverage by genre/tempo/confidence.
- Beat/downbeat F1 at declared tolerances (for example 50 ms and 100 ms), plus beat-phase-error distributions.
- Chord macro-F1 and calibrated confidence; “unknown/ambiguous” must be an acceptable answer.
- Section-boundary precision/recall with declared tolerance (for example one bar).
- Melody/instrument transcription precision, recall, onset error and pitch error only on datasets licensed for the exact use.
- Report clean stems and full-mix results separately; source-separation quality alone is not musicianship.

### B. Live duo/band coordination

Candidate gates to validate with expert partners:

- Stable-groove downbeat phase error: p95 ≤ 30 ms on a declared stable-tempo panel.
- After an intentional tempo/section change, re-lock within two bars in ≥ 95% of trials.
- Cue-to-musical-response latency no worse than one eighth-note at the declared tempo for discrete section/role cues; publish actual latency percentiles.
- Zero stuck notes, malformed events, unplanned resets or dropped sections in a 30-minute soak run.
- For requests such as “less busy,” “follow the guitarist” or “leave the vocal space,” at least 95% of actions affect the requested role/boundary, with a complete event receipt.
- One-action stop/correction by the musician; no hidden continuation after stop.

Timing must be role-aware: a bassist intentionally behind the kick is not necessarily wrong. Score phase locking, intended displacement, musical fit and listener judgment separately.

### C. Arrangement quality and player usefulness

- Harmony/motif preservation when explicitly requested.
- Counter-melody, voice leading, register occupancy and collision rate.
- Playability constraints for the selected instrument, including range and articulation; fret/bend constraints where modelled.
- Edit locality: changing one part/section preserves all other declared invariants.
- Useful variation across seeds without losing the brief.
- Recovery from uncertain/wrong chord, key or section estimates.
- Expert musicians' ratings of feel, timing, space, responsiveness, stylistic fit and usefulness in a real jam.

### D. Copyright and provenance controls

- 100% of audio, training, export and publication paths carry a source and intended-use classification.
- 100% of unsupported cloud uploads, training and redistribution cases are blocked or routed to rights review in an adversarial matrix.
- Zero source audio leaves ephemeral-local mode in deterministic network-egress tests.
- 100% of expired/revoked scopes are respected after revocation propagation.
- Zero public fixtures contain unlicensed commercial source audio, full song transcriptions, stems, lyrics or recognisable song-derived arrangements.
- User assertions, verified grants, public-domain evidence and unresolved status are never conflated in UI/receipt/code.
- Test false blocks as well as false permits. A compliance system that blocks everything is useless; an unauthorized release is a critical failure.

### E. Blind human evaluation

Use paired comparisons against a strong software baseline and expert human partners on the same cleared brief. Randomize order, normalize loudness, blind system identity, preregister exclusions/analysis, and rate timing, role fit, expressiveness, responsiveness, enjoyment and overall preference separately.

The current musicianship protocol sets a future target of at least 60% preference on a **named task/domain**, with the lower bound of a preregistered 95% confidence interval above 50%, followed by replication. This is a future goal, not a result. Include failure reports and listener subgroups; do not claim “better than artists” from one aggregate score.

## 8. Corpus and data governance

- Begin with synthetic fixtures, creator-owned compositions, explicit opt-in works and datasets whose licenses permit the exact analysis. Keep restricted sources out of Git.
- Use GMD only for drum-specific questions and appropriately licensed/authorized guitar/bass corpora for those roles.
- Build a licensed repertoire challenge set by obtaining permission for audio analysis, transcription and expected downstream uses. A popular-song test set without those permissions is not an acceptable default CI fixture.
- Split by source work, performer and session, never adjacent bars from the same recording alone.
- Store hashes/metadata and bounded metrics where useful; a hash establishes identity/provenance, not permission.
- Public examples should use original compositions, public-domain works plus verified recordings, or expressly licensed materials with required attribution.

## 9. Delivery sequence

**R0 — Safe listen-along:** live-room perception and an original adaptive part, with no external stream capture, no transcription export and no default audio retention.

**R1 — Source provenance record:** versioned rights evidence and operation-specific outcomes; UI separates user attestation from verified grants. No claim of automatically deciding fair use/fair dealing.

**R2 — Local authorized analysis:** source confirmation, ephemeral local processing, bounded storage, source deletion and network-egress regression tests.

**R3 — Licensed repertoire adapter:** integrate only where technical processing and output uses are expressly covered. Support revocation and territory/scope changes.

**R4 — Song-to-part arrangement:** convert authorized score/analysis into a playable, role-aware accompaniment; compare transcription-first with harmony/structure-first policies on matched tasks.

**R5 — Closed-loop partner:** test with drummers, guitarists and bassists across tempo changes, missed cues and intentional expressive displacement. Keep a hard stop and deterministic failure receipts.

**R6 — Recorded/published cover workflow:** distinct preflight for composition, recording, arrangement, venue, mechanical, synchronization and platform rights as relevant. Local-practice permission must never silently become upload permission.

**R7 — Measured release:** publish scorecards, exact corpus/license manifests, confidence intervals, false-permit/false-block rates and limitations. Queued/missing CI is not a pass.

## 10. Research references

- [Spotify Developer Terms](https://developer.spotify.com/terms) and [Developer Policy](https://developer.spotify.com/policy): platform-specific restrictions on using/analysing/ingesting Spotify Content.
- [U.S. Copyright Office — What Musicians Should Know About Copyright](https://www.copyright.gov/engage/musicians/): covers, rights, licensing, and no universal “few seconds” safe threshold.
- [17 U.S.C. §115 / U.S. Copyright Office Music Licensing Modernization](https://www.copyright.gov/music-modernization/115/): limited compulsory mechanical-licensing framework for covered uses.
- [SAMRO creator guidance](https://www.samro.org.za/creator): public-performance and mechanical-rights overview in South Africa.
- [CAPASSO music licensing](https://www.capasso.co.za/music-licensing/) and [licensing application](https://www.capasso.co.za/physical-licensing-application-form/): composition/mechanical use and separately cleared master recordings/adaptations.
- [Constitutional Court of South Africa — CCT306/24](https://www.concourt.org.za/index.php/judgement/647-ex-parte-president-of-the-republic-of-south-africa-in-re-constitutionality-of-the-copyright-amendment-bill-and-the-performers-protection-amendment-bill-cct306-24): June 2026 judgment on the referred amendment bills.

Re-check laws, official commencement notices, provider contracts and licence terms at implementation/release. This is a conservative product design contract and research plan—not an opinion that any particular user action is legal.
