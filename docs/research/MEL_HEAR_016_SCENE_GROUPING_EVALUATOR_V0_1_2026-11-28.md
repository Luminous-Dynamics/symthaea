# MEL-HEAR-016 — Auditory scene grouping evaluator v0.1

**Status:** standalone evaluator and schema contract. No source-separation model or real-audio grouping accuracy is claimed.

## Paradigm

The benchmark scores object-centric grouping of acoustic components into sources. A component may be a partial track, onset fragment, transient or localized time-frequency event. A polyphonic musical event can contain several source components; the evaluator does not force one event to equal one source.

Ground-truth source IDs and annotations are a separate evaluation-only input. The model-facing case and prediction records do not include reference labels. Ambiguous and not-estimable components are preserved as separate dispositions.

## Metrics

- Pairwise source co-membership precision, recall and F1.
- B-cubed precision, recall and F1, permutation-invariant to arbitrary source IDs.
- Explicit reference-cluster split and predicted-cluster merge counts/rates.
- Coverage and abstention counts.
- Ambiguous and not-estimable counts, with separate denominators.

Pairwise and B-cubed metrics are conditional on covered, resolved components. Coverage is always reported alongside them; abstention cannot improve the headline without exposing the coverage reduction. Components with ambiguous or not-estimable references are excluded from ordinary clustering denominators.

The evaluator computes pair counts from cluster intersections in linear time after indexing, rather than enumerating all component pairs.

## Claim boundary

A green evaluator test establishes metric correctness on synthetic known-answer partitions. It does not establish real-world auditory scene grouping, source separation quality, perceptual relevance, or annotation reliability. SDR/SI-SDR alone are not accepted as source-grouping evidence.

The next benchmark tranche should add controlled overlapping harmonics, missing fundamentals, onset synchrony/asynchrony, comodulation, masking, reverberation, stereo shifts and source motion. Compare classical clustering, NMF, pinned source-separation systems and foundation-model baselines. Hold out works, performers, instruments, rooms and recording chains.
