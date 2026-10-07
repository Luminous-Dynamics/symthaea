# UniMorph English 4.0 selected-record manifest

This document is the human-readable projection of `unimorph_eng_4_selection_manifest.json`. The JSON manifest is the machine-authoritative compiler-input selection; this Markdown document must remain an exact reviewable presentation of the same records and commitment.

## Frozen source

- Upstream repository: unimorph/eng
- Immutable commit: 66e0e9e8e2dcd196da081a25a48e5c1fe3d8b49b
- Git blob SHA: 8eae5ed242e87e50f6bd182133277f50fe93cef3
- Artifact BLAKE3-256: c4a677818237fb1060d2541272e2da1d5b6bfd2ae40df00b9187d1ae8566426f
- Artifact UTF-8 byte length: 18022905

## Selected records
Aggregate source-selection BLAKE3-256: 74585e2733a2d036cc396ebb5be77796d10b0de78814ad798731b2b96efbd726

This is the compiler's domain-separated BLAKE3 over the canonical JSON serialization of the seven source-slice objects in manifest order.


The selected byte ranges are exact ranges into the frozen artifact. Offsets are UTF-8 byte offsets and lengths include the terminal LF byte.

| Record ID | Source line | Byte offset | Byte length | Per-record BLAKE3 |
|---|---:|---:|---:|---|
| eng4:line:1 | 1 | 0 | 26 | d9374b6a84663b09917a521a865d73749386c059281bbd9ba3cbc92435f049f3 |
| eng4:line:2 | 2 | 26 | 32 | 907603894b6fe571ab7698f4e54e93dc2f0b707e210df91c9ae870b2c6449e67 |
| eng4:line:3 | 3 | 58 | 35 | 333a1e0f859e7be770e54645652a9c0da696a4ff3ae8ae50f8c113f93cbc2273 |
| eng4:line:4 | 4 | 93 | 27 | 3038dfd4dc2907b1d43cbd38f907673bec2ffc037903a65218e2642b0d5d0193 |
| eng4:line:5 | 5 | 120 | 34 | fe340586f544413839e9a00c7993f09f51f26cf92b609468fff47a311f8d2f3f |
| eng4:line:6 | 6 | 154 | 20 | 9db132698ce145c60901a0f287a93b531c9986efccefe4ed8ff94c0a4d272a03 |
| eng4:line:7 | 7 | 174 | 24 | 68a55886f3a052a603a873234c9a60a2cfcfbfff7b70eb6c25c04ce5c73fcf28 |

Selected records:

1. microtome\tmicrotomes\tN;PL
2. microtome\tmicrotomes\tV;PRS;3;SG
3. microtome\tmicrotoming\tV;V.PTCP;PRS
4. microtome\tmicrotomed\tV;PST
5. microtome\tmicrotomed\tV;V.PTCP;PST
6. eat\teats\tV;PRS;3;SG
7. eat\teating\tV;V.PTCP;PRS

## Deliberate exclusions

The next upstream rows are not selected for the current compiler contract because their transformations are outside the intentionally supported executable vocabulary. In particular, eat -> ate requires an irregular transformation and must remain a negative case rather than being guessed.

This manifest is a compiler-input manifest, not a claim that these seven rows are representative of English morphology or that the external corpus is complete/correct.