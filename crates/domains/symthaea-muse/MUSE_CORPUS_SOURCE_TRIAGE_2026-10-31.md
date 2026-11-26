# Muse corpus source triage — 2026-10-31

**Status:** research-screened candidates; none is blanket-approved for ingestion, model training, redistribution, or commercial use.

This note is a source-discovery record for #9509 and the artifact-level rights gate in `corpus_registry.rs`. It is not legal advice. Dataset-level terms, item-level rights, composition copyright, edition rights, performance rights, sound-recording rights and metadata rights must remain distinct. A source being public on the internet, a file being MIDI, or a repository displaying a licence badge does not automatically clear all uses and jurisdictions.

## Shortlist

| Source | Official/source record | What it contributes | Observed terms / boundaries | Triage |
|---|---|---|---|---|
| MAESTRO v3.0.0 | [Google Magenta dataset page](https://magenta.tensorflow.org/datasets/maestro) | Paired high-resolution piano MIDI and audio; about 3 ms alignment; velocity and sustain/sostenuto/una-corda controller data; composer/title/performance metadata; composition-grouped train/validation/test split. | Dataset released under CC BY-NC-SA 4.0. Current v3 page reports 1,276 performances and 198.7 hours. It is almost entirely classical piano competition performance, not multi-instrument composition coverage. MIDI/audio are performances, not a substitute for independently sourced editions or all underlying rights. | **Priority for non-commercial alignment/performance research only**, after per-record rights/provenance review. Do not assume commercial training permission. |
| GigaMIDI v2.0.0 | [Hugging Face dataset card](https://huggingface.co/datasets/Metacreation/GigaMIDI) and [TISMIR 2025 paper](https://transactions.ismir.net/articles/10.5334/tismir.203) | Large symbolic multitrack collection, expressive/non-expressive performance annotations, loop boundaries, velocity and microtiming; curated human style subset; GM instrument taxonomy coverage. | Dataset card states CC BY-NC 4.0 and requires accepting access conditions. It reports 2,136,218 MIDI files, 6,891,738 tracks and 128 melodic GM plus 47 percussion programs. Source data aggregated from publicly available Zenodo/GitHub resources and the card points to a per-source PDF. | **Highest-priority broad instrument/loop research candidate under non-commercial research terms**, pending per-source manifest and upstream rights checks. Not cleared for commercial training. |
| GiantMIDI-Piano | [Official GitHub repository](https://github.com/bytedance/GiantMIDI-Piano), [dataset paper](https://doi.org/10.5334/tismir.80), [repository disclaimer](https://github.com/bytedance/GiantMIDI-Piano/blob/master/disclaimer.md) | Classical piano MIDI transcription at scale; metadata associated with work/composer names. | Repository states 10,855 MIDI files and 2,786 composers and labels the repository CC BY 4.0. It is transcribed from YouTube recordings; the repository directs users to agree to a disclaimer, was archived 2025-04-08, and the paper notes missing pitch spelling, beats, meter, key signatures and separation of notation from expressive performance. | **Useful secondary analysis corpus**, but not a score-ground-truth or commercial-audio clearance source. Per-record upstream recording/work provenance required. |
| IMSLP / Petrucci Music Library | [IMSLP licensing policy](https://imslp.org/wiki/CC), [public-domain guidance](https://imslp.org/wiki/Public_domain) | Primary score editions, historical notation, composer/work metadata and edition variation across periods and traditions. | IMSLP contains both public-domain and licensed works. Its guidance explicitly says public-domain status varies by country, warns that its guide is not a legal determination, and notes separate rules for sound recordings. It distinguishes public-domain underlying works from independently copyrightable critical editions, arrangements and recordings. | **High-value score discovery source**. Resolve rights on each actual item/edition and jurisdiction; never infer recording rights from a score's status. |

## Important version and scope facts

### MAESTRO v3

The official dataset page describes about 200 hours of piano performance audio/MIDI with approximately 3 ms alignment, including velocity and pedal controllers. Its v3.0.0 table reports 962 train, 137 validation and 177 test performances (1,276 total; 198.7 hours). The page publishes the audio/MIDI release digests:

- v3.0.0 audio+MIDI ZIP SHA-256: `6680fea5be2339ea15091a249fbd70e49551246ddbd5ca50f1b2352c08c95291`
- v3.0.0 MIDI-only ZIP SHA-256: `70470ee253295c8d2c71e6d9d4a815189e35c89624b76d22fce5a019d5dde12c`

The v3 split removes six chamber recordings that had appeared across train/test in earlier versions, so do not mix v2 and v3 split files. The page says same compositions performed by multiple contestants should not cross split boundaries. These properties make MAESTRO valuable for expressive-performance and score/audio alignment tests, but its piano-only scope limits its contribution to the 152-style and 42-instrument objectives.

### GigaMIDI

The current Hugging Face card explicitly points users to v2.0.0 and reports 2.1M+ symbolic files, over 9.2M non-expressive loops and 2.3M expressive loops. It covers all 128 GM melodic programs plus 47 percussion programs. The source-data section says the dataset aggregates public MIDI from sources including Zenodo and GitHub and provides a source-links PDF.

That makes it a promising *research discovery index*, particularly for instrument-role, loop, expressive timing and multi-track coverage. It also creates a critical provenance requirement: every original upstream item must retain its source URL, original licence/terms, attribution, upstream version, content digest, deduplication group and any restrictions. A CC BY-NC 4.0 dataset-card declaration does not erase the individual provenance of embedded items or grant commercial rights. The access agreement and non-commercial scope must be accepted/recorded before research use.

### GiantMIDI-Piano

The current repository states that its MIDI is transcribed from online piano recordings; its dataset paper reports 10,855 works and 38,700,838 transcribed notes. The paper itself lists important limitations: no pitch spellings, no beat/time/key signature, and no separation between score and expressive performance. The repository is read-only since April 8, 2025, lists CC BY 4.0 and separately asks users to agree to a disclaimer before download.

Treat the CC BY 4.0 repository label as the dataset publisher's licence assertion, not a universal clearance of all underlying YouTube recordings or the compositions performed. Prefer metadata/symbolic analysis only after its upstream work/recording provenance is reconstructed; do not use it as source-verified score truth.

### IMSLP

IMSLP is an index and hosting service with a mixed collection: public-domain material and material released under licences. Its policy says copyrights normally remain with creators unless expired or permission has been granted. The public-domain guidance warns that status varies by country and special rules apply to sound recordings. Its examples also distinguish a public-domain work from original content in a critical edition, arrangement, or recording.

For Muse, capture at least:
- underlying composition/work ID and author/death/publication evidence;
- the exact edition/publisher/engraving or MusicXML/MEI transcription;
- any arrangement, editorial, fingering or annotation layer;
- source jurisdiction and intended-use jurisdiction;
- separate performance and sound-recording records where recordings are involved;
- the source URL, file digest, item page, contributor and retrieval/version evidence.

## Ingestion and rights policy

1. A discovered source is `Discovered`, never automatically `RightsEvidenceChecked`.
2. Keep dataset release-level information and item-level artifact records. A corpus licence may govern the dataset compilation or files distributed by the curator, but item-level upstream restrictions still require review.
3. For each artifact, record each applicable rights layer separately: composition, score edition, performance, sound recording, audiovisual recording, annotation and metadata.
4. Record a grant against explicit jurisdiction(s) and intended uses (internal analysis, non-commercial research, model training, commercial training, derivatives, redistribution). Missing or unknown status fails closed.
5. Rights for one jurisdiction do not imply rights in another. A public-domain composition does not clear a modern recording or edition.
6. Keep split/derivative/duplicate groups by underlying work as well as recording and edition. Train/test split must be composer/work disjoint where practical, not merely file disjoint.
7. A rights clearance is not a cultural-protocol clearance. Record community/rights-holder requirements separately; uncertain or required-but-pending status blocks use.
8. No raw audio/MIDI is ingested into model training by this source triage.

## Research plan

- **P0: non-commercial score/performance alignment:** use MAESTRO v3 after item-level rights and data-protection review. Validate MIDI parser, alignment error, velocity/pedal interpretation, tempo map and score-to-performance normalization.
- **P0: broad symbolic/instrument discovery:** evaluate GigaMIDI v2 only in non-commercial research mode after obtaining its upstream source-links PDF and creating a per-source provenance ledger. Start with inventory/metadata statistics before processing MIDI bytes.
- **P1: historical score structure:** use IMSLP for item-level source discovery. Extract only items with a defensible edition and work status for the target jurisdiction/use. Include edition identifiers and scans, not just work titles.
- **P1: piano-only external validation:** GiantMIDI-Piano can provide auxiliary transcribed-performance statistics, but not score truth without reconstructing the underlying composition and edition. Avoid relying on its transcribed MIDI as a clean symbolic target.
- **P2: culturally specific and oral traditions:** seek community-led, consented archives, rights-holder permissions and contextual documentation; don't treat scraped labels or broad taxonomy tags as authentic ground truth.

## Benchmarks to build from these sources

- **Score/performance alignment:** median and P95 onset/duration error, unmatched note rate, pedal-event preservation, tempo-map residual, split-by-instrument/register/density and source OOD.
- **Instrument coverage:** per-instrument/per-technique coverage, precision/recall against audited labels, calibration and OOD performance. The MIREX 2026 instrument benchmark demonstrates that strong public-set performance need not transfer to a hidden set; do not accept OpenMIC-only scores as qualification.
- **Symbolic generation:** exact hard-constraint pass rate, long-range section/motif obligations, score edit locality, voice/part accounting, diversity/collapse, style separation and reference leakage.
- **Provenance:** every ingested artifact has an exact digest, source/release identity, required rights layers and eligibility receipt; zero unknown or unlicensed items in the training manifest.
- **Reproducibility:** deterministic split manifests, source digests, parser versions, compiler version, generation seed and objective evaluation config included in every report.

## Source links

- MAESTRO: https://magenta.tensorflow.org/datasets/maestro
- GigaMIDI: https://huggingface.co/datasets/Metacreation/GigaMIDI
- GigaMIDI TISMIR paper: https://transactions.ismir.net/articles/10.5334/tismir.203
- GiantMIDI-Piano: https://github.com/bytedance/GiantMIDI-Piano
- GiantMIDI-Piano paper: https://doi.org/10.5334/tismir.80
- GiantMIDI-Piano disclaimer: https://github.com/bytedance/GiantMIDI-Piano/blob/master/disclaimer.md
- IMSLP licensing: https://imslp.org/wiki/CC
- IMSLP public-domain guidance: https://imslp.org/wiki/Public_domain
- 2026 music-generation evaluation survey: https://link.springer.com/article/10.1007/s10462-026-11582-x
- MIREX 2026 instrument recognition: https://music-ir.org/mirex/wiki/2026%3AAudio_Instrument_Recognition_Results
