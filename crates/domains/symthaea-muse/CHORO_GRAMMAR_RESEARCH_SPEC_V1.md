# Choro Grammar Research Specification v1

**Status:** Research specification only. Choro remains outside the canonical catalog and has no composer route.

## Why Choro is a useful pilot

Moss, Souza and Rohrmeier's peer-reviewed corpus study analyzes 295 representative pieces from the Choro Songbook, transcribing chord symbols and formal structure. The published findings report that harmonic organization in regular 16- and 32-bar phrases relates to phrase hypermeter; chord-transition and key-transition relations differ within pieces; and older compositions use chords with fewer modifications than younger works. Those are source-reported findings, not Symthaea measurements.

Mário Sève's article on Choro phrasing and rhythmic/melodic considerations discusses a bounded historical setting in Rio de Janeiro from the second half of the nineteenth century through the first half of the twentieth, and examines polca-lundu, tango brasileiro, choro-amaxixado, and choro-sambado contexts. It discusses ensembles based around cavaquinho, violão and percussion. Those examples do not define one universal arrangement for all Choro.

## Hypotheses to test after license clearance

1. **Form/hypermeter:** For items explicitly annotated as regular 16- or 32-bar phrases, test whether harmonic event positions follow a pre-registered phrase-position model more strongly than a constrained within-piece shuffle.
2. **Harmonic transitions:** Test whether transitions depend on key/form context beyond aggregate chord frequency, with held-out-by-composition evaluation.
3. **Historical modification:** For reliably dated works, test the reported direction of relation between composition year and chord-modification frequency, publishing missing-year coverage and subgenre/songbook sensitivity.
4. **Rhythmic phrasing:** Not testable from chord/form labels alone. It requires separate rights-cleared score/onset evidence, source-defined subtype labels, and expert review.

The machine-readable JSON defines operationalization, candidate fields, null models, falsification conditions, and route gates. A deterministic synthetic-fixture harness checks three metric directions and malformed-input behavior without using the restricted corpus; these tests validate measurement machinery only, not Choro hypotheses. Numeric thresholds are intentionally not fabricated; they must be fixed before analysis after an eligibility and power review.

## Sampling dependence and generalization limits

The cited corpus study reports 295 pieces, 44,067 chord tokens, 682 unique chord types, 180 composers and 19 subgenres. It also reports 32 pieces by Pixinguinha and 31 by Jacob do Bandolim; those two composers alone account for 63/295 pieces (about 21.4%). These are **paper-reported counts**, not measurements made by this repository. The corpus has not been ingested here, and the songbook-based sampling frame is not equivalent to the full living Choro tradition. ([Moss, Souza & Rohrmeier, 2020](https://doi.org/10.1080/09298215.2020.1797109))

This concentration creates a generalization risk: a piece-held-out result can still be driven by composers appearing in both partitions. Before any authorized evaluation, keep each complete composition in one partition, prohibit token/bar-level leakage, and separately report a composer-grouped split where eligible coverage permits. Run leave-one-dominant-composer-out sensitivities for Pixinguinha and Jacob do Bandolim, and state the number of eligible independent compositions and composers. Report piece-weighted and composer-aggregated estimates where both are meaningful; cluster uncertainty at the independent unit appropriate to each hypothesis.

For the historical-modification hypothesis, record year coverage and overlap by composer, subgenre and songbook. Composition year must come from source provenance, not the publication or digitization date. If historical period and composer are too confounded for the registered question, the result is **not identifiable** from this sample; do not silently turn it into a causal claim. An underpowered or unestimable sensitivity must be reported as such, not omitted. The machine-readable rules in `sampling_and_dependence` make these conditions part of the pre-analysis contract.

## Corpus licensing and boundaries

The research repository is [DCMLab/choro, pinned to commit fe409c62deed50fc7f4b0d667863cd13ab230ff6](https://github.com/DCMLab/choro/tree/fe409c62deed50fc7f4b0d667863cd13ab230ff6). Its LICENSE identifies **CC BY-NC-SA 4.0**. The spec does not copy dataset rows, note sequences or transcriptions into Symthaea. Commercial analysis/training must not proceed on this dataset without an appropriate legal review and permission or a separate rights-cleared corpus. Public availability is not commercial authorization; rights in the underlying songbook material also require consideration.

## Exit gates

- Rights and permitted purpose documented before empirical corpus analysis.
- Source commit and license pinned; eligibility/missingness audit and hypotheses registered before outcomes are examined.
- Null models and held-out-by-composition splits fixed in advance.
- Candidate evidence compared with neighboring styles on compatible, rights-cleared reference material before claiming distinctness.
- Knowledgeable Choro musicians/researchers review the proposed grammar and limitations.
- Any eventual generator route is implemented and tested through the actual user composition path; symbolic results and renderer/instrument support are reported separately.

Passing the research contract does not itself create a composer route or justify a claim of authenticity.

## Why the preregistration contract is this explicit

The [Open Science Framework's registration guidance](https://help.osf.io/article/330-welcome-to-registrations) describes preregistration as a timestamped, read-only plan submitted before data collection or analysis, and recommends stating hypotheses and variables, statistical tests and decision criteria, exclusions, outcomes, contingencies and deviations. The Choro contract translates those principles into fields that can be inspected mechanically; freeze booleans alone are not sufficient. The gate checks the local plan bytes and recorded metadata but does **not** independently verify that an external registration URL is truthful, immutable or actually available.

The 2 June 2026 open-access survey [*Recent advances in music generation: methods, evaluation, and challenges*](https://doi.org/10.1007/s10462-026-11582-x) notes that no single objective metric adequately covers music-generation quality. It distinguishes symbolic outcomes such as tonality, rhythm and structure from audio-level properties such as timbre, transients and perceptual quality, and recommends complementary evaluation dimensions. Accordingly, this pilot keeps symbolic corpus findings, runtime behavior, rendered audio and practitioner/listener review as distinct evidence layers rather than collapsing them into one score.



The synthetic harness is run by `scripts/muse_choro_synthetic_harness.py --check` and tested by `tests/python/test_muse_choro_synthetic_harness.py`. Its report is explicitly labelled non-empirical; no corpus rows, audio or transcriptions are used.


## Rights-aware source intake: individual scores, not bulk corpus ingestion

The separate [Choro source-intake inventory](CHORO_SOURCE_INTAKE_V1.json) records six metadata-only score leads/collections: José Mauricio Braga's historical piano *Chorinho* (ca. 1918); Davi Rocha's contemporary *Choro mineiro* and 14-work *Choros Songbook* collection; Luciano Gallet's 1928 *12 Exercícios brasileiros* collection containing a movement titled *Chôrinho*; Norberto Macedo's *Choro p'ra Portinari* (1982), represented by a 2025 editor-typeset guitar score listed CC BY-SA 4.0; and *Chóra, chóra, choradô*, retained only as a possible Samba boundary/control item. Gallet and Macedo broaden composer/instrument coverage, but their piano-four-hands and guitar-solo sources respectively are not representative ensemble evidence.

**None is rights-cleared in this inventory.** These catalog pages are research leads, not legal opinions. Rights in the composition, score edition, digitization, performance recording, lyrics and annotations may differ. Verify the exact source file, attribution/license, territorial scope and intended use before acquisition, transcription or analysis. The historical piano score and contemporary single-composer collection are not representative samples by themselves. The inventory confirms zero downloads, zero transcriptions, and zero empirical analysis.

## Source-bound approved-input gate

`CHORO_APPROVED_INPUT_MANIFEST_V1.json` is intentionally empty and `CHORO_APPROVED_INPUT_APPROVAL_V1.json` is explicitly unapproved. The checker `scripts/muse_choro_input_manifest.py --check` reports the current blocked state; it does not authorize any work. A future source-bound experiment must require `--require-authorized --artifact-root <private-reviewed-files-dir>` before analysis. The manifest must also bind an exact SHA-256 of a separately registered analysis-plan JSON. The plan file must confirm frozen hypotheses, null models and thresholds against an immutable HTTPS registration record; both rights review and independent approval must post-date that registration. `CHORO_PRE_REGISTERED_ANALYSIS_PLAN_V1.json` is only a blocked template, and the general research specification does not itself satisfy this gate. A registered plan must contain concrete hypothesis and null-model records with tests, analysis/resampling units and falsification/preservation rules; source eligibility criteria; outcomes; effect/uncertainty and decision thresholds; multiplicity, missing-data, split and deviation policies; and mandatory sensitivity analyses. It must also enumerate exact source bindings (record/file URLs, artifact path, SHA-256, byte count and selected works) that match the approved-input manifest. Freeze flags alone do not satisfy the contract. The checker verifies file bytes, dates and recorded fields, not the scientific adequacy of the plan or the truth of an external registration record. Authorization requires exact on-disk SHA-256 and byte-count checks, a rights review bound to that artifact digest, explicit use/commercial/jurisdiction scope, separate decisions for the composition, edition/arrangement, digital file/scan and third-party material, plus an independent approval receipt hashing the exact manifest bytes.

This is a structural evidence gate, not a legal opinion: complete fields do not independently prove that a license is valid or that a reviewer reached the correct conclusion. Each collection entry must enumerate exact work/movement IDs and state its selection basis; the review-scope digest binds that selection to the exact artifact and intended use. Changing the selected movements, commercial status, jurisdictions or outputs invalidates the rights-review scope digest. Synthetic tests use only fabricated bytes and fixture evidence URLs. No downloaded score, transcription, restricted corpus row or empirical Choro measurement is included. Any future empirical runner must call this gate as a precondition; the current synthetic-only harness remains separate and does not ingest score sources.


## Clarification on the archived corpus's rights metadata

The paper's data-availability statement says the dataset is available for non-commercial use, while the [Zenodo v1.1 record](https://zenodo.org/records/3881347) displays Rights as **Other (Open)** with no further description. The associated [GitHub repository](https://github.com/DCMLab/choro) separately has a CC BY-NC-SA 4.0 license file. Do not treat the generic Zenodo label as a commercial license. The source-intake JSON records these as separate observations; this work has not downloaded or analyzed the dataset.
