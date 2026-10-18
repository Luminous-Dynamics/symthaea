# Muse Creative Audio Matrix V1

Status: implementation and acceptance criteria are prospective until the example runs on an exact, recorded source revision. This matrix is an engineering integrity/reproducibility check, not a music-quality benchmark.

## Run

From the repository checkout:

    cargo run -p symthaea-muse --example creative_audio_matrix -- <expected-git-sha>

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
