# GRAPE upstream source manifest v1

**Purpose:** pin the public reference implementation and all released anonymized feature bundles before any local reproduction.  
**Research benchmark:** #13469  
**Streaming adapter:** #13471 / draft PR #13478

## Paper identity

- Choi, Doyun; Lee, Cheonwoo; Han, Younghyun; Han, Kiwook; Park, Jinkook; Ryu, Sungyoon; Sohn, Younghoon; Yoo, Jaemin.
- *Streaming Virtual Metrology via Lightweight Adaptation under Temporal Distribution Shifts*.
- arXiv:2611.00806, submitted 2026-10-31; arXiv record says accepted at IEEE BigData 2026.
- Primary paper record: https://arxiv.org/abs/2611.00806
- DOI-style arXiv identifier: 10.48550/arXiv.2611.00806

## Repository identity

- Repository: https://github.com/cheonwoolee/GRAPE
- Default branch at inspection: main
- Pinned commit: `cce2d27a092abd9e397d7739e4beb6dfe0ae145e`
- Commit tree: `fdb1bd87fae7592ba844a417feec1836ab8b342e`
- Commit date: 2026-06-03T12:06:48Z
- Repository has one initial commit at the pinned revision; do not treat a moving main branch URL as the reproducibility pin.
- License: Apache-2.0, LICENSE blob `e6bbde2ae2423b2882f8274376341be227862a52` (9,270 bytes).
- README blob `acf5b5facfd2b5d2dd104aa4ba131c0a9ac3ad2f` (896 bytes).
- requirements blob `c2d7d761f29cffc60304d2c6c325f2f839eaeb34` (65 bytes).
- Requirements declare NumPy, pandas, PyTorch, scikit-learn and PyYAML; README requires Python 3.10+.

The IDs below are Git blob/tree object IDs as returned by GitHub's Git database API, **not SHA-256 hashes of downloaded files**. A reproduction tool must calculate and store SHA-256 after safely materializing each file. Do not substitute the Git blob ID for a SHA-256 content digest.

## Core implementation blobs

| Path | Git blob ID | Bytes |
|---|---|---:|
| `configs/config.yml` | `6ce6b68ae0e64c04ca1ba3c933c77c7de02c5706` | 477 |
| `src/evaluation/runner.py` | `3a6f1c42640563aa7bbd6e93168a4f20ccd07c0d` | 5,185 |
| `src/models/grape.py` | `f526d9d05f214434aa711980876cae7e148b9c6f` | 17,295 |
| `src/preprocessing/loader.py` | `3fc547c481ec76194119168870219174df34584e` | 5,718 |

## Released data bundle blobs

The public repository contains preprocessed/anonymized derived feature vectors, labels and context IDs. It does not contain the original fab sensor time-series or full historian events. All paths below are nested under the pinned commit.

| Dataset | Role | Git blob ID | Bytes |
|---|---|---|---:|
| D1 | context IDs | `d73eea8d8b4e40bdedfe5d23d796aa6f4f75c481` | 63,518 |
| D1 | train features | `71872e7582d7a60f121c4d70bdfc435f451812d4` | 89,761 |
| D1 | train labels | `cf91595958d7222f5de4fdf3b7d696f9b7053883` | 1,942 |
| D1 | validation features | `afd2fd48ce3bb8f70b3aef0f71ea288633fc7765` | 17,752 |
| D1 | validation labels | `96b5cd2388187b88f4753f51bf216a765345d9d1` | 502 |
| D1 | test features | `fde258d7c8e519643fa7231c0bbd49cc3d4654a0` | 35,752 |
| D1 | test labels | `03afe8adcbee209db85e8842854258aebe3a2fc1` | 862 |
| D1 | unseen features | `4826d3b8e5e08ecd7a2de89ec926313f9212d855` | 216,962 |
| D1 | unseen labels | `ef440691484c99f80fa0937e1129bf38481130b5` | 4,487 |
| D2 | context IDs | `8ff1d586ff790e3338c696921bc50a58a6a2d619` | 406,544 |
| D2 | train features | `0d0a4152b4a2b996b181e4906f4df2b57c332234` | 1,569,139 |
| D2 | train labels | `8f4da7b635c2783a64c44381901990e6e0a5d540` | 12,703 |
| D2 | validation features | `e2a0405f8782ac7e34c886fdb8fd1d3f8c53c6df` | 304,139 |
| D2 | validation labels | `ce2194d96348939d8c1be8215843cd51ad356a48` | 2,583 |
| D2 | test features | `c08b305cc3038a7cc298a1268adfa7fc7c283b93` | 608,139 |
| D2 | test labels | `ce093d23d88618616cf856087a77a6f8c462ca33` | 5,015 |
| D2 | unseen features | `f871d3291206fa8520c9aecbdaba9293ba70781b` | 3,317,139 |
| D2 | unseen labels | `aa8a459c9577e5638de90de86f6ba736bc4ed60c` | 26,687 |

## Protocol observations from pinned source

1. The runner's stream evaluator predicts each row and then calls the online update with that row's true label in the same loop. Its reported prequential setting is therefore a zero-delay label-arrival protocol.
2. One mutable model instance is carried across validation, test and unseen evaluation in the current runner path. Exact reproduction must preserve that state lifecycle; an isolated-split track must be labelled separately.
3. The unseen feature branch uses the first `nf` columns, while training/validation/test use the training-derived top-feature indices. This is an upstream code-path asymmetry. The faithful track must preserve and report it; a corrected feature-index track must use the declared training-selected indices and be reported separately.
4. The repository bundles extracted/anonymized feature arrays and labels, not original multi-rate raw sensor streams. It cannot independently validate asynchronous sensor alignment, actual metrology availability latency, source authenticity or equipment-event causality.
5. Pickle files must be treated as untrusted executable serialization. Do not load them in production runtimes or trusted CI jobs. Use a one-shot isolated conversion environment with no network/secrets, then export a safe, schema-validated, content-addressed format.

## Reproduction acceptance gate

Before reporting any comparison:
- verify the repository commit and tree against the pin;
- independently download and compute SHA-256 for all inputs;
- isolate and audit pickle conversion;
- record exact Python/library versions and deterministic settings;
- verify shapes, dtypes, NaN/Inf counts, sample ordering, label units and context-ID alignment;
- reconstruct the 25:5:10:60 chronological split from the authors' protocol and compare to supplied bundle splits;
- freeze train-only feature selection and preprocessing receipts;
- reproduce official zero-delay/state-carry behavior exactly;
- execute a separately named delayed-label track with explicit label-availability events;
- report D1 and D2 metrics separately and do not pool normalized values;
- report at least three seeds where the paper does, with paired uncertainty intervals;
- preserve the upstream unseen-index asymmetry in the faithful track and quantify the corrected-index delta in the corrected track.

No local dataset execution or paper-level reproduction is claimed by this source manifest.
