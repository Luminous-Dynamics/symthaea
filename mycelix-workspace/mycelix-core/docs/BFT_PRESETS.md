# BFT Presets (Design Draft)

**Purpose**: Define named configurations for Byzantine Fault Tolerance that can be reused across Python, Rust, Holochain zomes, and SDKs, instead of scattering hard‑coded thresholds.

This is a design document only – no code currently consumes these presets.

---

## Goals

- Make BFT behaviour **explicit and tunable** for different deployments.
- Provide a small number of named presets that:
  - Choose aggregation defenses and parameters (Krum / Median / TrimmedMean / FedAvg).
  - Set PoGQ/MATL thresholds.
  - Encode expectations about adversary ratios and performance.
- Ensure Python (0TML), Rust (`fl-aggregator`, `mycelix-sdk`), and TypeScript (`@mycelix/sdk`) can all implement the same preset semantics.

---

## Proposed Presets

### 1. `fast_demo`

**Use case**: Interactive demos, local dev, education (not adversarial environments).

- Defense:
  - Primary: `FedAvg` (no Byzantine protection) or `Median` for modest robustness.
- Parameters:
  - `Krum` not used by default.
  - `TrimmedMean` not used by default.
  - PoGQ thresholds:
    - `pogq_low_threshold`: 0.2
    - `pogq_high_threshold`: 0.8
- Expectations:
  - Adversary ratio: ≤ 10–15% noisy or mildly malicious nodes.
  - Optimized for simplicity and speed over safety.

### 2. `balanced_33`

**Use case**: Typical federated scenarios where ≤ 33% of nodes may be Byzantine; want strong guarantees with reasonable latency.

- Defense:
  - Primary: `Median` or `TrimmedMean { beta ≈ 0.1–0.2 }`.
  - Krum used selectively for smaller networks or as a secondary check.
- Parameters (illustrative):
  - `TrimmedMean.beta`: 0.2
  - `Krum.f`: `floor(n * 0.3)` (subject to `n ≥ 2f + 3`).
  - PoGQ thresholds:
    - `pogq_low_threshold`: 0.3
    - `pogq_high_threshold`: 0.7
  - Adaptive threshold (`AdaptiveByzantineThreshold`):
    - `DEFAULT_THRESHOLD`: 0.30
    - `MIN_BYZANTINE_TOLERANCE`: 0.10
    - `MAX_BYZANTINE_TOLERANCE`: 0.45 (global cap)
- Expectations:
  - Adversary ratio: up to ~33% under IID and moderately non‑IID conditions.
  - This preset should match the “100% detection at ≤33%” regime documented in `docs/CRITICAL_REVIEW_AND_ROADMAP.md`.

### 3. `max_safety_45`

**Use case**: High‑stakes deployments where up to ~45% of nodes might be malicious or colluding; willing to trade off latency and throughput for safety.

- Defense:
  - Primary: `Median` or `TrimmedMean` with more aggressive trimming.
  - Krum / MultiKrum as a high‑cost, high‑safety option for smaller networks.
  - PoGQ/MATL classification must run on every round; no “skip detection” paths.
- Parameters (illustrative, to be calibrated from 0TML):
  - `TrimmedMean.beta`: 0.25–0.3 (depending on network size).
  - `Krum.f`: `floor(n * 0.45)` when used, with strict `n ≥ 2f + 3`.
  - PoGQ thresholds:
    - `pogq_low_threshold`: 0.4
    - `pogq_high_threshold`: 0.6
  - Adaptive threshold:
    - More conservative safety margin adjustments after confirmed attacks.
- Expectations:
  - Adversary ratio: up to ≈ 45% for well‑behaved scenarios (as demonstrated in 0TML experiments).
  - Explicitly document **failure modes**: if conditions exceed calibrated expectations, the system should `Escalate`/`Halt` rather than silently proceeding.

---

## Integration Points (Future Work)

These presets are not yet wired into code; possible places to introduce them:

- **Rust (`fl-aggregator`)**:
  - A `BftPreset` enum with methods to build `DefenseConfig`, PoGQ thresholds, and AdaptiveByzantineThreshold configs.
- **Rust SDK (`mycelix-sdk`)**:
  - Utility functions that map a preset name to MATL configuration.
- **Holochain FL zomes** (`mycelix_fl_core`):
  - `aggregate_gradients` input could accept either:
    - Explicit parameters (`method`, `f`, `beta`, thresholds), or
    - A `preset` string that selects one of the above profiles.
- **TypeScript SDK** (`@mycelix/sdk`):
  - Export the same preset names for clients/CLIs so developers can consistently choose BFT profiles.

---

## Alignment with Documentation

When presets are implemented, update:

- `docs/CRITICAL_REVIEW_AND_ROADMAP.md` – to state which presets were used for each validation.
- `docs/VALIDATED_CLAIMS.md` – to link claims to specific presets (e.g., “45% BFT with `max_safety_45` on 11‑node scenarios”).
- Holochain design docs – to clarify which preset is the default for on‑chain FL coordination.

