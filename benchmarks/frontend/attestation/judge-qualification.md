# SFEB evaluator and judge qualification

See `judge-qualification-v1.json` and its schema.

## Why this is separate

The executor attests which observations were captured. The independent oracle verifier derives machine-checkable outcomes. A learned or interaction-based judge may add valuable diagnostic and subjective evidence, but is not the trust root and cannot override critical deterministic gates.

## Evaluator arms

- J0: deterministic contract/security oracle.
- J1: code-only judge.
- J2: screenshot-only judge.
- J3: code + screenshot judge.
- J4: interaction-grounded computer-use judge.
- J5: bounded ensemble with J0 authority preserved.

## Qualification protocol

The plan uses 30 public development pairs, 150 comparative pairs and 500+ publication pairs, with at least 30% controller-only/hidden. Two independent human reviewers annotate each item; a third adjudicates disagreements. Metrics include per-category balanced accuracy, precision/recall, critical false-acceptance, Brier/ECE, human agreement, prompt-injection resistance, stale-observation detection, inconclusive rate, latency and cost.

Prospective promotion targets (not results): ≥0.85 balanced accuracy on held-out defects, ≥0.95 critical-defect recall, ECE ≤0.10 and ≥0.80 annotator agreement. Deterministic critical acceptance remains zero-tolerance and cannot be delegated to J1–J5.

## Research and rights

Arena's 2026 Interaction-as-a-Judge work is a useful secondary-evaluator design reference. The ArenaWebdevPreferences dataset card lists CC-BY-4.0 for user prompts, but model outputs remain under provider terms. ArenaWebdevCorrectness lists CC-BY-4.0. Both remain reference-only pending a separate asset, privacy and commercial-rights review; neither dataset is vendored here.

## Run status

This is a protocol and machine-readable study definition, not an executed judge evaluation or measured transfer result.
