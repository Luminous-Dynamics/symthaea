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


The synthetic harness is run by `scripts/muse_choro_synthetic_harness.py --check` and tested by `tests/python/test_muse_choro_synthetic_harness.py`. Its report is explicitly labelled non-empirical; no corpus rows, audio or transcriptions are used.


## Rights-aware source intake: individual scores, not bulk corpus ingestion

The separate [Choro source-intake inventory](CHORO_SOURCE_INTAKE_V1.json) records four metadata-only source leads: an IMSLP score of José Mauricio Braga's *Chorinho* (ca. 1918) labeled Public Domain; Davi Rocha's contemporary *Choro mineiro* score labeled CC BY 4.0; Rocha's *Choros Songbook* collection of 14 works labeled CC BY 4.0; and *Chóra, chóra, choradô*, listed as *Samba Rio-Grandense*, retained only as a possible boundary/control item.

**None is rights-cleared in this inventory.** These catalog pages are research leads, not legal opinions. Rights in the composition, score edition, digitization, performance recording, lyrics and annotations may differ. Verify the exact source file, attribution/license, territorial scope and intended use before acquisition, transcription or analysis. The historical piano score and contemporary single-composer collection are not representative samples by themselves. The inventory confirms zero downloads, zero transcriptions, and zero empirical analysis.


## Clarification on the archived corpus's rights metadata

The paper's data-availability statement says the dataset is available for non-commercial use, while the [Zenodo v1.1 record](https://zenodo.org/records/3881347) displays Rights as **Other (Open)** with no further description. The associated [GitHub repository](https://github.com/DCMLab/choro) separately has a CC BY-NC-SA 4.0 license file. Do not treat the generic Zenodo label as a commercial license. The source-intake JSON records these as separate observations; this work has not downloaded or analyzed the dataset.
