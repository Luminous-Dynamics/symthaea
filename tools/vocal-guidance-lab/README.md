# Vocal guidance lab

An isolated, model-agnostic probe for two-objective singing-voice guidance: phoneme correctness and pitch alignment.

## What this is

The tool receives two same-dimensional finite gradient vectors exported from an external experiment, validates their source/checkpoint identifiers, measures their interaction, and computes a symmetric conflict-only projection. It writes a JSON receipt with the original and projected vectors, dot products, cosine similarities, norms, conflict classification and the decision whether a non-zero two-objective guidance direction remains.

For an original phoneme vector p and pitch vector m, the projection is:

    p' = p - dot(p, m) / ||m||² * m
    m' = m - dot(p, m) / ||p||² * p

Cooperative vectors are left unchanged. Conflicting vectors lose the components that directly oppose the other objective. The resulting vectors are not necessarily orthogonal; their post-projection cosine is measured and reported. Antiparallel vectors can project to zero; this is reported as an infeasible joint direction rather than a success. Missing objective gradients are not fabricated.

## What this is not

This is not a diffusion model, classifier, audio renderer, implementation of the AES paper, or evidence that guidance improves singing quality. The symmetric projection is a transparent research baseline. A later experiment must compare it against the paper's specified procedure and at least a no-projection baseline. The tool has no authority to decide that pronunciation or pitch constraints may be relaxed.

## Run

    cargo test --manifest-path tools/vocal-guidance-lab/Cargo.toml

    cargo run --manifest-path tools/vocal-guidance-lab/Cargo.toml -- \
      --input gradients.json \
      --output conflict-report.json

## Input contract

The JSON input must contain:

- schema vocal-guidance-gradients/v1;
- exact 40-character lowercase source Git SHA;
- non-empty model identifier;
- exact 64-character lowercase BLAKE3 checkpoint digest;
- same-length finite numeric phoneme and pitch gradient vectors.

The fixture used in unit tests is synthetic mathematics, not a classifier output or an audio-quality result.

## Research protocol

For any real model export, freeze the lyric/pronunciation plan, score, dataset split, model revision and checkpoint digest. Compare: no guidance; phoneme only; pitch only; naive joint guidance; the selected conflict-projection method; and guidance scale sweeps. Report original and projected objective alignment alongside PER, word accuracy, pitch error in cents, rest leakage, human naturalness and control-MOS. Keep objective regressions visible; do not select a winner from a weighted scalar alone.

Primary research lead: Li, Liu and Lu, “Singing Voice Synthesis via Guided Diffusion,” Journal of the Audio Engineering Society 74(11), 811–824 (10 November 2026), DOI 10.17743/jaes.2026.0303.
