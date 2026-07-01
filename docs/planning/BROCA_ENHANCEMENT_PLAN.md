# Broca & Coding Enhancement Plan (Phase 2.5)

This plan outlines the implementation of four key pillars to enhance Symthaea's coding capabilities, transitioning from an infrastructure proof-of-concept to a production-ready system.

## Pillar 1: Substrate Parity & Harvester Generalization
**Goal:** Prove substrate-independence by extending the distillation pipeline to Terraform (HCL) and Docker Compose.

- [x] Add generalized `examples/harvest_distillation.rs` while keeping the Nix-specific harvester available.
- [x] Use the existing `Substrate` abstraction for Nix, HCL, Docker Compose, and Rust harvests.
- [x] Add support for external prompt lists (`--prompt-file <path>`) to facilitate scaling.
- [ ] Add Python goldens before enabling Python harvests by default.
- [ ] Add yield telemetry for PASS-filtered pairs by substrate.

## Pillar 2: Unified Coding API
**Goal:** Simplify the developer experience for using Broca as a code generator.

- [ ] Add `BrocaGenerator::for_substrate(Substrate)` constructor.
- [ ] Automatically wire `for_nix_distillation` tokenizer and `LanguageGate`.
- [ ] Configure `EpistemicCubeGate` with substrate-specific defaults.

## Pillar 3: Epistemic Hallucination Suppression
**Goal:** Use Symthaea's unique epistemic signals to prevent hallucinated code options.

- [ ] Update `EpistemicCubeGate` to support a "Strict Code" mode.
- [ ] Implement logit-level suppression for attribute paths (Nix) or resource types (HCL) when epistemic certainty is low.
- [ ] Add an adversarial demo showing the gate suppressing `services.non_existent.enable`.

## Pillar 4: Synthetic Dataset Scaling
**Goal:** Grow the training corpus from 26 to 500+ PASS-filtered pairs.

- [ ] Create `scripts/generate_synthetic_prompts.py` (infrastructure for synthetic scaling).
- [ ] Implement a "Verification Loop" that only harvests pairs that pass the structural scorer.
- [ ] Add telemetry to track "Verification Yield" (Synthetic Prompts vs. PASS-filtered Pairs).

---
*Target Completion: Multi-step execution starting with Pillar 1.*
