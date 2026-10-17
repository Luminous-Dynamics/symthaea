# Melothaea Style Completion v1

**Status:** first implementation tranche; not a claim that all 152 styles are finished.  
**Scope:** the public Muse 152 registry, composer routes, compositional identity, and human-review gates.

## Baseline and honest completion measure

The catalog has 152 stable public IDs across 19 constellations. On this branch, 33 entries have a callable composer route, up from 29. The remaining 119 entries have no callable route.

A callable route is not the same as a finished or authentic style. Existing catalog statuses remain Developing or ExpertReviewRequired; no style is Foundation merely because it generates a score. Some existing entries route to a close related preset (for example, Chicago Blues to the general Blues engine or Motet to SacredChoral), so style-specific identity and blind listening still need to be established.

| Constellation | Callable routes | No route yet |
|---|---:|---:|
| Classical Lyric & Character | 7/8 | 1 |
| Baroque and Early Dance Forms | 1/8 | 7 |
| Developmental and Large Forms | 3/8 | 5 |
| Contrapuntal and Polyphonic | 3/8 | 5 |
| Ground, Ostinato and Transformation | 3/8 | 5 |
| Song and Narrative | 2/8 | 6 |
| Blues, Gospel and Soul | 1/8 | 7 |
| Jazz and Improvisatory Forms | 1/8 | 7 |
| Latin and Caribbean Cycles | 3/8 | 5 |
| European and North Atlantic Folk | 3/8 | 5 |
| Mediterranean, Middle Eastern and North African Modal | 1/8 | 7 |
| South Asian Raga and Tala | 1/8 | 7 |
| East and Southeast Asian Traditions | 0/8 | 8 |
| African and Diasporic Groove | 0/8 | 8 |
| Minimal, Process and Experimental | 1/8 | 7 |
| Ambient and Electronic Texture | 1/8 | 7 |
| Club and Beat Music | 0/8 | 8 |
| Pop and Rock | 0/8 | 8 |
| Dramatic, Screen and Stage | 2/8 | 6 |
| **Total** | **33/152** | **119/152** |

## Newly callable structural styles

This tranche adds four composer presets using existing, named structural engines instead of creating genre names with only a tempo label:

- **Rondo** — fixes the form pool to A–B–A–C–A, using the current Rondo realizer.
- **Theme and Variations** — fixes the form pool to the duration-preserving variation realizer.
- **Erosion** — fixes the ground-form pool to the existing progressive ground-loss realizer.
- **Lineage** — fixes the ground-form pool to the existing parent-to-descendant ground transformation realizer.

Each has a named catalogue mapping, a defined tempo band, and an exact-form regression test. Erosion and Lineage derive their starting instrumentation/motif/harmonic constraints from the Passacaglia preset and still need their own authored melodic DNA; they are structurally callable, not fully differentiated timbral or melodic identities.

The Style::ALL registry and the catalog's executable-route tests now cover the expanded 33. The known inert progression-field audit and intent-axis audit also include the ground-form additions. A style whose dedicated form owns its own harmonic/length logic must not be made to look complete by changing a progression field that the engine does not consume.

## Cultural and tradition-specific safety

The catalog no longer treats all unmapped styles in a culturally rooted constellation as review-free by default. Every entry in these constellations is marked as requiring knowledgeable musical review before any authenticity/foundation claim:

- Blues, Gospel and Soul;
- Jazz and Improvisatory Forms;
- Latin and Caribbean Cycles;
- European and North Atlantic Folk;
- Mediterranean, Middle Eastern and North African Modal;
- South Asian Raga and Tala;
- East and Southeast Asian Traditions;
- African and Diasporic Groove.

This is a review requirement, not a claim that every style is equally well understood or that a generated score is authentic. A culturally informed review may require specialist performers, instrument-specific constraints, tuning/raga/maqam knowledge, rhythmic-cycle correctness, phrasing, ensemble role, oral/aural transmission, and explicit terminology boundaries. The UI should prefer “inspired by” or a transparent hybrid label unless evidence justifies a stronger term. Review must be performed with the right practitioners; unit tests alone cannot establish it.

## Definition of done for each style

A style should only be promoted through these layers:

1. **Explicit identity:** structure/grammar, phrase behavior, harmonic system, rhythm, melodic language, ensemble role, performance dialect, production assumptions, and provenance for every borrowed tradition-specific claim.
2. **Executable route:** fixed and tested structural mechanisms; validated score; no empty, malformed, or out-of-range output; a clear map from the public catalog ID to the actual preset.
3. **Own signature:** tests show at least the declared core properties (for example, the rondo return, variation theme skeleton, repeated ground or cycle phase). A different enum value or tempo band is not enough.
4. **Determinism and controlled diversity:** same source/seed/version is reproducible; new seeds yield meaningful within-style variation; cross-style comparisons catch accidental duplicates and label bleed.
5. **Capability sensitivity:** tests identify which intent axes affect output and which are intentionally inert. A dead knob must be disclosed or wired into the actual engine.
6. **Fair baseline comparison:** compare matched lengths, loudness/renderer and task constraints to the existing baseline; retain negative controls; report confidence intervals and failed cases.
7. **Blind listening:** compare samples without labels, with experts and target listeners. Record task, sample identities, seed/config, reviewer expertise, outcomes and uncertainty. Do not promote to Foundation from implementation tests alone.
8. **Qualified review:** traditions requiring expert review cannot make authenticity/foundation claims without that evidence. User-created and open/public-domain corpora still need source and license provenance.

## Recommended remaining sequence

**Next: Classical and Baroque forms.** Build Minuet and Trio, Gavotte, Sarabande, Gigue, Allemande, Toccata, French Overture, Rondo-adjacent developmental forms and Elegy from actual formal/rhythmic differences, not aliases. Reuse existing form machinery when its invariants really match; add a new engine only when the structure cannot be represented faithfully.

**Then: song, blues and jazz.** Build verse/chorus, through-composed, sea shanty, art song/Lied, Delta/Country/Chicago blues variants, Gospel call-response, Soul/R&B/Neo-Soul, swing, bebop, cool/hard/modal/free jazz and jazz waltz. Require distinct groove/harmonic/phrase identity tests before making these callable under specific labels.

**Then: cycles and regional folk.** Build explicit cycle phase, instrument roles, and allowed subdivisions for son, salsa, mambo, cha-cha, samba, tango variants, Irish/Scottish/English/Nordic/Balkan/Klezmer traditions; acquire knowledgeable reviewers for culturally specific styles.

**Then: modal traditions and Asian/African styles.** Do not map maqam, makam, dastgah, raga/tala, gagaku, honkyoku, min'yo, guqin, sizhu, gugak, gamelan, piphat, Afrobeat, highlife, juju, soukous, amapiano, gqom, reggae or dub to Western functional-harmony substitutes and call them complete. Each needs domain-specific tuning/modes or cycle/ensemble models and informed evaluation.

**Then: process/electronic/club/pop/screen.** Complete phase music, additive process, spectral/aleatoric, dark ambient, drone, Berlin School, IDM/glitch/vaporwave/electroacoustic, house/deep house/techno/trance/DnB/UK garage/breakbeat/hip-hop, pop/indie/dream/synthpop/rock/prog/post-rock/folk-rock, musical theatre, noir/epic/horror/sci-fi and adaptive game-score states. Include arrangement, production and long-form tests as relevant; notes and tempo alone cannot finish these styles.

These batches are a prioritization, not permission to skip any of the 119 remaining catalog IDs.

## Verification status

This branch is the first style-completion tranche, not the final 152-style delivery. Four style routes and catalog coverage tests are authored. Rust formatting, compilation, unit tests and listening checks must pass on the exact final commit before this tranche is qualified. A queued, skipped, stale or missing CI result is not a pass.

No audio samples or third-party recordings are added to this branch. Listening and benchmark corpora must use original works, verified public-domain recordings, or explicitly licensed/authorized sources.
