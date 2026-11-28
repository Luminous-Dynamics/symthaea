# Southern African style profiles: Amapiano and Gqom — research tranche v0.1

**Status:** research-only profile package. No composer route, runtime profile, sample set, model training, human validation result or benchmark win is claimed.

**Base:** `main@771c288ea6c08bab72f15bdb975e8f3b082eceac`  
**Scope:** two source-grounded profile records for existing catalogue identities. This package deliberately does not add new `Style` enum variants or catalogue IDs.

## Why this tranche

The main-branch catalogue already names Amapiano (catalogue ID 109) and Gqom (catalogue ID 110), but the current composer enum has 29 executable variants and does not implement either identity. The catalogue/anatomy work in [#7417](https://github.com/Luminous-Dynamics/symthaea/issues/7417) establishes the right architecture: a tradition label is not a composer, and inherited constellation descriptions are not evidence of a style-specific grammar.

This tranche supplies style-specific, machine-readable research records that can be reviewed before implementation. It is a child research increment, not a replacement for:

- [#7417 — multidimensional profiles and hybrid grammar](https://github.com/Luminous-Dynamics/symthaea/issues/7417)
- [#7883 — open-ended musical range benchmark](https://github.com/Luminous-Dynamics/symthaea/issues/7883)
- [#10399 — musician-governed cross-cultural qualification](https://github.com/Luminous-Dynamics/symthaea/issues/10399)
- [#9397 — instrument idiom, orchestration and renderer-independent style identity](https://github.com/Luminous-Dynamics/symthaea/issues/9397)
- [#8164 / #8176 — style interaction protocol and initial probe](https://github.com/Luminous-Dynamics/symthaea/pull/8164)
- [#13908 / #13947 — property-relative identity across performance variation](https://github.com/Luminous-Dynamics/symthaea/issues/13908)
- [#10678 — current competitive frontier and comparison programme](https://github.com/Luminous-Dynamics/symthaea/pull/10678)

## Research inputs and bounded claims

### Amapiano

1. Eaby-Lomas, D. M. (2023), “Kwaito as history: Complicating contemporary historiographies,” *African Music*, 12(1), 21–36. [DOI: 10.21504/amj.v12i1.2529](https://doi.org/10.21504/amj.v12i1.2529). This situates Amapiano within a longer South African township-music history and discusses links with marabi, mbaqanga, bubblegum and kwaito. It treats histories and origins as multivocal and ambiguous.
2. “Understanding amapiano and the South African city through the music videos of Big Flexa and Bhebha” (2024). [DOI: 10.1080/02533952.2024.2526251](https://doi.org/10.1080/02533952.2024.2526251). Its discussion includes piano/keyboard, smooth-jazz and other genre influences, the log drum, extended instrumental openings, gradual entry of layers and local-language dissemination. These are candidate observations, not requirements for every Amapiano track.
3. “Ase trap tse ke pina tsa ko kasi — Sonics After the Struggle: Mapping Amapiano and Black Aesthetics in Post-Apartheid South Africa” (2026). [DOI: 10.1080/13696815.2026.2634896](https://doi.org/10.1080/13696815.2026.2634896). This provides newer scene and historical context. Read with practitioner discussion before making narrower production claims.

### Gqom

1. Eaby-Lomas, D. M. (2022), “Historicising gqom as a post-kwaito phenomenon,” *Journal of the Musical Arts in Africa*, 18(1), 101–116. [DOI: 10.2989/18121004.2021.2020964](https://doi.org/10.2989/18121004.2021.2020964). The article discusses Gqom's emergence, dissemination, production, gender discourse and party spaces, and describes rhythmic foregrounding, a raw/dark sound and slow-burn structure in the examples studied. It also notes that these traits have changed in newer releases.

The source set is sufficient to define research questions and avoid obvious generic approximations. It is not sufficient to freeze a universal grammar or claim community authority.

## Profile architecture

Each JSON file is a `research_profile_not_runtime_recipe`. It records identity and current catalogue reference, source-grounded historical context and claim limits, candidate identity axes, operational hypotheses that can be falsified, nearby styles that require contrastive review, explicit non-inferences, implementation readiness and missing evidence.

The records distinguish **literature-supported candidates** from **practitioner-confirmed facts**. Do not convert the current JSON into a runtime schema or fill unresolved values using an undocumented constellation default. The profile compiler in #7417 remains the schema owner.

## Key distinction hypotheses (to be tested, not assumed)

| Dimension | Amapiano research question | Gqom research question |
|---|---|---|
| Arrangement | How do extended instrumental openings and gradual layer entry vary across scenes/substyles? | How do gradual build, sparse/dense sections, breaks and energy arcs vary? |
| Rhythm | How do pulse, bass/log-drum motifs, percussion and syncopated layers interact? | How is rhythmic foregrounding realized without reducing Gqom to generic house/techno? |
| Pitch/harmony | When do keyboard voicings and jazz-influenced harmonic colours carry identity? | When are melody/harmony subordinate, and when are they prominent? |
| Production | How do piano, pads, bass timbres and production roles vary? | How do producers define and vary descriptors such as rawness and darkness? |
| Context | Which links to kwaito, bubblegum, mbaqanga and house are historically supported? | Which links to kwaito are accepted, and where is Gqom's distinct self-definition? |

The hypotheses must be revised when practitioner evidence or counterexamples disagree. Do not turn the table into a fixed recipe.

## Paradigm shifts and engineering implications

### 1. Genre identity is a set of attributed, versioned claims—not a label-to-pattern lookup

A profile should retain multiple histories, aliases and disagreements. Musical similarity does not prove historical lineage. A producer's or musician's account should be stored with attribution, scope and permission, not merged into a universal consensus.

### 2. Musical identity must survive permitted realization changes

Compose and evaluate the score, performance and renderer as separate causal layers. Same score/different renderer, same grammar/different ensemble, and same style/different performer should be controlled counterfactuals. Reuse #13908/#13947's property-relative FLIP/STAY design; do not add a second general-purpose evaluator.

### 3. A style profile is not a frozen loop

Model a constrained space of valid arrangements and performance variations. The objective is to preserve identity-bearing relationships while allowing within-style diversity. Fixed BPMs, one drum grid, one chord progression, one timbre or one template cannot be accepted as a universal genre definition without evidence.

### 4. A hybrid is a new recipe, not a historical claim

Use the existing layered compiler and compatibility graph. A hybrid should disclose its structural owner, donated dimensions, transformations, unresolved/unsupported dimensions and provenance. It must not be labelled as an authentic historical genre.

### 5. Cultural review is a distinct qualification boundary

Automated structure checks and deterministic renders can establish software properties. They cannot grant authenticity or community approval. Follow #10399's musician-governed, cohort-specific consent and attribution design.

## Execution plan

### M0 — research contract (this tranche)

- [x] Create separate Amapiano and Gqom research records.
- [x] Bind each candidate discriminator to its cited sources.
- [x] Record counterexamples, non-inferences and missing evidence.
- [x] Preserve current catalogue IDs without claiming executable routes.

### M1 — musician review and source-cleared capture

- Convene separate Amapiano and Gqom practitioner reviews; do not assume one panel can represent either tradition.
- Ask practitioners to nominate nearest-neighbour contrasts, valid variation ranges, local names and disputed boundaries.
- Separate annotation, internal evaluation, public release, model training and commercial-use permissions.
- Select sources by performer/producer/work/scene strata; no broad scraping or implicit training permission.
- Record source work, composition, recording, performer, producer, editor and rightsholder as separate identities where permitted.

**Exit:** a reviewed and consented vocabulary, a rights-cleared example manifest, explicit disagreements, and a preregistered pilot design.

### M2 — symbolic profile and grammar design

- Map practitioner-reviewed dimensions into the existing sparse profile architecture from #7417.
- Mark each field as observed, inherited with provenance, user-overridden, engine-derived or unresolved.
- Express hard constraints separately from typical tendencies.
- Write counterexample fixtures for nearest-neighbour confusion and within-style variation.
- Do not add an executable route until an actual engine owner and the relevant field-consumption path are identified.

**Exit:** versioned draft profiles, exact deterministic recipes, capability/unsupported-axis reports, and a fixture matrix.

### M3 — first executable composer slices

- Start with one narrow composition capability per identity, selected from the musician-reviewed evidence.
- Route through the actual grammar compiler and produce a real symbolic Score.
- Bind recipe/profile/engine/seed/score digests.
- Test that changing the target axis changes its intended musical observations and that protected non-target fields remain within predeclared tolerances.
- Preserve existing styles and serialized recipes; do not add new enum arms merely to make catalogue counts rise.
- Keep production fields unapplied until a qualified renderer adapter proves the effect.

**Exit:** exact-head Rust format/tests/Clippy plus independent symbolic oracle results. A green compile is not an artistic or cultural pass.

### M4 — performance and rendering

- Use #9397's instrument/renderer qualification; do not infer sound from a GM program name.
- Connect role-specific instrument assignments only where the relevant source profile, playability and renderer capability are verified.
- Measure actual rhythmic and expressive effects from rendered/performed output, not just requested settings.
- Keep native and SoundFont renderers as separate strata.

**Exit:** renderer-specific capability receipts and a blind listening pack; no authenticity promotion yet.

### M5 — held-out human qualification

- Use different source works/producers/performers across development and confirmatory panels.
- Use crossed or clustered analysis at source-work and performer levels; musical events/windows are not independent sample units.
- Blind system identity, randomize order and retain all candidate outputs, not only best-of-N winners.
- Report style identity, nearest-neighbour confusion, within-style diversity, target fidelity, non-target invariance and listener agreement separately.
- Require local practitioner review for cultural claims; retain disagreement and uncertainty.

**Exit:** a preregistered evidence package supporting exactly the claims it measured.

## Benchmark portfolio and targets

Reuse existing owners rather than duplicating their machinery:

| Question | Existing owner | Additional style-specific use |
|---|---|---|
| Are Amapiano and Gqom structurally different from generic dance controls? | #9397 / #7417 | Contrast grammar with same ensemble and renderer |
| Do combinations of style dimensions interact correctly? | #8164 / #8176 | Test Amapiano/Gqom profile controls and protected dimensions |
| Does identity survive performance variation? | #13703 / #13857 / #13908 / #13947 | Add source-cleared examples once rights and annotations exist |
| Can profiles recombine beyond memorized presets? | #7883 | Hold out profile-layer combinations and composition families |
| Are human and community judgments represented responsibly? | #10399 | Separate Amapiano and Gqom cohorts |
| Does the full system compare fairly with production tools? | #9453 / #10678 | Use a separate matched audio-generation track |

### Proposed benchmark tracks

1. **Symbolic identity:** rhythm/arrangement/pitch/grammar discriminators under a common instrument and renderer.
2. **Realization invariance:** same profile and target property across different performances, ensembles and renderers.
3. **Controlled edits:** change one requested profile dimension, measure target response and protected-feature drift.
4. **Within-profile diversity:** multiple valid outputs across held-out seeds and practitioner-approved variation envelopes.
5. **Nearest-neighbour separation:** Amapiano vs Gqom, Kwaito, deep/soulful house and generic house/techno controls, with comparisons chosen and reviewed by practitioners.
6. **Human review:** blinded pairwise identity, idiomaticity, variation and preference judgments; keep these distinct.

### Baselines

- unconditioned/default composer;
- nearest currently executable symbolic style, clearly disclosed as an approximation;
- generic electronic-dance control;
- ablated Amapiano/Gqom profile with one identity-bearing layer removed;
- practitioner-authored, rights-cleared reference/control examples;
- qualified external audio systems in a separate prompt-matched track.

For commercial systems, match prompt, duration, candidate budget, selection policy and available controls. Report unavailable controls as not comparable. Do not compare symbolic note-editability against systems that expose only audio.

### Targets (prospective, not achieved)

- 100% pass on deterministic structural and negative-control fixtures.
- Zero silent substitution of an unsupported style dimension.
- Zero false claims of renderer application when no qualified adapter exists.
- No held-out source-work, producer or performer leakage.
- Confirmatory style distinction must have a preregistered margin and uncertainty interval, not merely a point estimate or classifier accuracy.
- No profile graduates to cultural-authenticity status on automated metrics alone.

Do not set numeric human-performance targets before a rights-cleared pilot estimates variance, performer/source-work clustering and feasible statistical power.

## SOTA and industry comparison

- [MusicListenBench (22 November 2026)](https://arxiv.org/abs/2611.29375) is a key external anchor for distinguishing real musical changes from acoustic-only changes. Reproduce its official split and scorer before any direct claim; the property-relative extension must remain separate.
- [MIREX 2026 Symbolic Music Generation](https://futuremirex.com/mirex/wiki/2026%3ASymbolic_Music_Generation_Results) provides a public subjective comparison protocol. Its published study used a small prompt set, 12 participants and best-of-eight candidate selection; this is useful context, not sufficient evidence of broad genre mastery.
- [MIREX 2026 Music Evaluation via CMI-RewardBench](https://music-ir.org/mirex/wiki/2026%3AMusic_Evaluation_via_CMI-RewardBench_Results) is an evaluator-quality track, not an artistic-quality or authenticity test.
- [2026 music-generation survey](https://doi.org/10.1007/s10462-026-11582-x) identifies persistent challenges in musicality, consistency, controllability, representation and evaluation.

The industry strategy in #10678 and the instrumental preference programme in #9453 remain the owners for commercial frontier snapshots. This tranche does not reproduce their leaderboard data.

## Validation performed for this documentation tranche

- JSON parsing and required-key/schema-marker checks are performed before repository publication.
- Source URLs, DOI identifiers, profile IDs and catalogue references are recorded in the files.
- The package is documentation/research metadata only.
- No Rust compilation, benchmark execution, audio render, human review, rights clearance or CI pass is claimed.

## Immediate follow-on tasks

1. Fetch and review the full Amapiano and Gqom source texts with a South African practitioner/editor.
2. Turn the candidate axes into a review worksheet with separate questions for producer, performer, DJ and researcher roles.
3. Create a source-cleared example manifest only after rights review.
4. Reuse the current #13947 property-relative evaluator and #9397 factorial identity design for the pilot.
5. After the M1 exit evidence exists, make a narrow executable profile adapter PR against the then-current #7417 schema.
