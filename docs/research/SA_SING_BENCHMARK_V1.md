# SA-SING-001 — South African multilingual singing evaluation pilot

**Date:** 11 November 2026  
**Status:** metadata contract and synthetic validation tests; no recordings ingested and no singing result claimed  
**Owner:** https://github.com/Luminous-Dynamics/symthaea/issues/11549

## Programme boundary

SA-SING-001 is a performer-consented, musician-governed singing qualification set for Muse. It reuses:
- #11154 ZA_LEX language-profile and phonological resources;
- #9509 artifact-level corpus rights registry;
- #10399 musician/community governance and cross-cultural annotation;
- VPTB/VCRB/VIA/VCCR/VMEA and the session ledger for independent pitch, control, correction, edit and preservation evidence.

It does not duplicate the rights registry and does not authorize collection, ingestion, training, cloning or public release.

## Resource audit

### South African spoken-language profiles

The existing ZA_LEX audit identifies six starting profiles: Afrikaans, South African English, Sesotho, Setswana, isiXhosa and isiZulu. They are pronunciation resources, not qualified singing models. Their phone inventories, click representation, tone, lexical coverage, regional variants and licence scopes need language-by-language validation.

South Africa has 11 spoken official languages. South African Sign Language is the 12th official language and must be treated as a distinct signed modality rather than coerced into the acoustic F0/phoneme track.

### External speech resources

- CSIR SAMA (SADiLaR) is a Level-0 catalogue record for orthographically transcribed broadband podcast speech in several South African languages, with XML segments and a CC BY 4.0 catalogue licence. It is not singing; item-level and upstream broadcast rights need review.
- Lwazi ASR catalogue metadata conflicts: its language table lists CC BY 3.0 while its narrative says CC BY 2.5. Resolve with upstream rights before any ingest. It is scripted telephonic speech, not singing.
- Swivuriso / ZA-African Next Voices covers seven South African languages and about 3,000 hours of speech. Its terms expressly prohibit TTS, voice cloning, voice synthesis and human-voice replication/generation. Keep it out of Muse singer training, timbre-reference training and synthesis.
- Lwazi III isiZulu TTS metadata reports CC BY-NC-SA 4.0. It is speech, non-commercially restricted and not a singing recording; it cannot support commercial singing generation by default.
- WAXAL public descriptions vary by release and language count. The audited release summary does not establish coverage of the South African target set. Pin the exact artifact/revision and inspect the language manifest before using it for South African speech-side tasks.

These resources may inform separately qualified speech/orthography/phonetic tasks. None is a substitute for a consented sung reference set.

## Rights and consent

Each case links to a rights-registry item and evaluation eligibility receipt. The registry must distinguish composition, arrangement, lyrics, performer, recording, annotation, voice/timbre reference, training, synthetic generation, commercial use, redistribution, retention and withdrawal. Approval for evaluation-only is not approval for model training, voice cloning or commercial output.

Pseudonymize performer, session and work-family identifiers in public manifests; keep identity mapping outside the repository. Obtain musician/native-speaker review and preserve dissent. Use contributor-approved self-description; do not infer culture, ethnicity, identity or consent from audio.

## Evaluation tracks

1. Lyric/phonetic realization: orthography, IPA phones, syllables, native adjudication, phone timing and switch boundaries.
2. Score pitch: score event map plus a separate independent measured F0 reference; report FFE/GPE/VDE, octave errors, vibrato and portamento.
3. Local edit preservation: target/protected region maps, VELB locality, neighboring-phone timing and seam checks.
4. Code-switch boundary: language spans + phone alignment + switch boundary labels.
5. Human perceptual quality: blinded native-speaking/music-expert panels; keep lyric intelligibility, naturalness, musicality, voice continuity and cultural acceptability distinct.

Do not infer tone from unmarked text. Do not treat score MIDI as sample-level F0 ground truth. Keep spoken-language code-switching and speech-vs-singing mode transitions as separate labels.

## Data split protocol

Use feasibility pilot, calibration, development, confirmatory and locked-test splits. Confirmatory evaluation must be disjoint by performer, song/work family and recording session; the source audio digest cannot appear in multiple splits. Near-duplicate arrangements/mixes must be audited. Power analysis must cluster by singer/session; frame-level independence is invalid.

The first data acquisition tranche should remain private and evaluation-only. Pilot sample counts will be determined with participating musicians and annotation capacity. A small pilot is only an engineering feasibility study, not a statistically generalizable benchmark or SOTA claim.

## Competitors and baselines

- LyricEditBench / YingMusic-Singer-Plus: six Chinese/English lyric edit operations; NC-SA dataset licence.
- UniVocal / SCSBench (ACL 2026): speech-versus-singing mode transitions, not multilingual language switches; useful mode-control baseline.
- SoulX-Singer-Eval: singing quality / lyric / melody / speaker similarity baselines where conditions match.
- MusicJudge: block-aligned lyric/pitch/rhythm comparison; not waveform locality or ground truth.
- Muse evaluators: independent pitch VPTB, acoustic response VCRB, calibration VNOC, interval VIA, edit locality VELB, correction VCCR, task edit VMEA, session ledger MMAE-TRACE.

Publish per-language × operation × singer/timbre condition × technique/register metrics, cluster uncertainty by singer/song/session, and keep automatic and human evaluation separate.

## Rust contract

The crate in tools/south-african-singing-benchmark:
- permits only sung-vocal sample modality and benchmark evaluation-only use;
- binds each case to a frozen rights-registry snapshot/item eligibility receipt;
- binds explicit language-profile digests and non-overlapping time spans;
- requires native-speaker reviewed lyric/phone annotations where claimed;
- requires independent measured F0 separate from score event references;
- requires the track-specific reference artifacts;
- validates performer/work/session/source split disjointness;
- emits counts by split/language/track only after all checks pass.

This validator checks metadata consistency and digest syntax. It cannot authenticate the rights registry or validate the sound content.

## Sources

- #9509 MUSE-CORPUS-001 rights-aware registry
- #10399 MEL-CULTURE-001 musician governance
- #11154 ZA_LEX language profiles
- #11425 independent singing F0
- #11549 SA-SING-001
- https://hdl.handle.net/20.500.12185/689
- https://huggingface.co/datasets/dsfsi/lwazi-asr-corpus-compressed
- https://huggingface.co/datasets/dsfsi-anv/za-african-next-voices
- https://aclanthology.org/2026.acl-long.1452/
