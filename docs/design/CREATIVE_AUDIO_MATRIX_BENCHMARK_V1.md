# Muse Creative Audio Matrix V1

Status: implementation and acceptance criteria are prospective until the example runs on an exact, recorded source revision. This matrix is an engineering integrity/reproducibility check, not a music-quality benchmark.

## Run

From the repository checkout:

    cargo run --locked -p symthaea-muse --example creative_audio_matrix -- <expected-git-sha>

The optional SHA is an assertion, not free-form provenance metadata. The program resolves git rev-parse HEAD from the crate directory, records the actual revision, captures rustc version and host OS/architecture, hashes the serialized cognitive state and renderer configuration, and exits non-zero if the supplied expected revision differs. It also fails closed if tracked or untracked working-tree changes exist, so HEAD alone cannot misrepresent a dirty checkout.

## Matrix

The harness composes a fixed seeded piece using the same duration, cognitive state, and generation budget across:

- Sample rates: 22,050; 44,100; 48,000; and 96,000 Hz.
- Output representations: mono signed 16-bit PCM, mono 32-bit float, and stereo 32-bit float.
- Repeats: render each of the 12 combinations twice with identical inputs.

Every case reports the renderer-configuration digest, symbolic-note digest, typed audio-buffer digest, dimensions, waveform audit, and whether the repeated render was identical. The receipt's audio digest hashes the exact typed AudioData representation; it is not a hash of exported WAV or FLAC bytes. The configuration digest hashes the exact JSON serialization of MuseConfig, not a claim of universal canonical JSON.

## Hard, falsifiable checks

1. Actual checkout revision matches the optional expected SHA.
2. Both renders in a case have identical symbolic note and audio-buffer digests.
3. Symbolic-note digests remain identical across all sample-rate and output-format combinations. A change here is unexpected because only rendering configuration should vary.
4. Every buffer has the expected frame count for its sample rate and declared duration.
5. Every waveform contains zero non-finite samples, zero samples outside normalized range, and is not effectively silent under the declared audit configuration.
6. Any mismatch is present in the JSON receipt and causes a non-zero process exit.

The silence threshold is a declared diagnostic, not a universal audibility standard. The matrix intentionally does not require waveform digests to match across sample rates or encodings: quantization, channel layout and sampling rate make byte-level equality inappropriate across those variants.

## What this does not establish

- It does not establish perceptual equivalence, musicality, timbral quality, originality, or listener preference.
- A non-silent signal is not necessarily a good or audible composition.
- Sample peak is not true peak; these checks do not establish EBU loudness-meter conformance or delivery loudness compliance.
- A fixed seed alone is not a reproducibility result. The runner's actual same-input comparison is the evidence; the Git SHA and explicit configuration/audio digests make that evidence attributable.
- Passing this matrix does not qualify all Muse code, all possible inputs, optional model files, or all platforms. In particular, if a future generator path reads external model/projection files, those inputs must be identified and hashed before a deterministic-reproduction claim is made for that path.

## Recommended next gates

First require 100% success on these 12 matrix cases, then repeat the matrix on supported operating systems/toolchains if cross-platform determinism is a product requirement. Keep musical acceptance separate: use blinded, matched-budget comparisons on held-out briefs, and report intent adherence, preference, diversity/coverage, novelty, and listenability independently. The prospective target of greater than 60% pairwise preference with a 95% confidence-interval lower bound above 50%, replicated on a held-out split, is a proposed research gate—not a result of this audio matrix.


## External benchmark alignment

The matrix above is the deterministic engineering gate, not the perceptual evaluation. The subsequent human study should align with current external practice without mistaking it for ground truth:

- MIREX 2026 Music Audio Generation separates expert listening dimensions (melody, harmony, key consistency, style consistency, arrangement/production); acoustic features are diagnostic, not used to rank systems: https://www.music-ir.org/mirex/wiki/2026%3AMusic_Audio_Generation
- MIREX 2026 Symbolic Music Generation uses blinded subjective ratings and lets systems generate eight continuations per test sample before selection: https://www.music-ir.org/mirex/wiki/2026%3ASymbolic_Music_Generation
- MIREX 2026 CMI-RewardBench evaluates reward/evaluation models by agreement with human pairwise preferences, reporting a 249-example core set and vocal/instrumental slices of a 618-example extended set. Its published table lists 73.9% core accuracy for the top entry but also 59.0% for an always-choose-B baseline; this is a practical reason to randomize candidate position and report position/label baselines and slices rather than assume empirical chance is 50%: https://music-ir.org/mirex/wiki/2026%3AMusic_Evaluation_via_CMI-RewardBench_Results
- A 2025 human-preference study evaluated 6,000 songs from 12 systems using 15,000 pairwise comparisons from 2,500 participants, illustrating why claims about automatic music metrics should be calibrated and evaluated on human judgments rather than self-reported by the metric: https://arxiv.org/abs/2506.19085

For Symthaea/Muse, keep two results separate: (1) **generation quality**—what blinded listeners prefer among matched-budget candidates; and (2) **evaluator validity**—how well an automated judge predicts held-out human choices, including calibration, balanced accuracy, and genre/voice/instrumental slices. Neither result implies the other.
