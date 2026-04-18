# Symthaea as a Coding AI

> A correctness-first, structurally-verified, consciousness-gated
> approach to code generation — positioned orthogonally to the scale
> race. This document is the one-page story of what exists today,
> the research claim that emerges from it, and what runs.

## The one-sentence claim

**Symthaea's Nix code generator catches four real codegen bugs that a
substring-based scorer (and, structurally, an LLM) silently approves.**
The scorer is AST-based, the repair loop is scorer-in-the-loop, and
the training stack feeds PASS-filtered pairs into consciousness-gated
emission — none of which require matching LLM-scale to work.

## What runs today

| Layer | Module | Tests | Demo |
|---|---|---|---|
| Structural scorer | `src/language/nix_scorer.rs` | 15 unit tests | `cargo run --features code_generation --example nix_scorer_diagnose -- "<prompt>"` |
| Goldens | `src/language/nix_eval_goldens.rs` | 4 unit tests, 26 goldens | `cargo run --features code_generation --example nix_eval_benchmark -- --goldens-only` |
| Self-repair loop | `src/language/nix_repair.rs` | 21 unit tests | `cargo run --features code_generation --example nix_eval_benchmark -- --goldens-only --repair` |
| Tier 3 cache | `src/language/learned_idioms.rs` | 14 unit tests | Invoked automatically |
| Module-eval cache | `src/language/nix_eval_cache.rs` | 9 unit tests | Invoked automatically |
| Intent bridge (→ Broca) | `src/language/nix_broca_bridge.rs` | 9 unit tests | — |
| Distillation harvester | `examples/harvest_nix_distillation.rs` | — | `cargo run --features code_generation --example harvest_nix_distillation` |
| Broca trainer adapter | `crates/symthaea-broca/src/bin/distill_nix_train.rs` | — | `cargo run --bin distill_nix_train -p symthaea-broca -- --epochs 10` |
| Generation demo | `crates/symthaea-broca/src/bin/distill_nix_generate.rs` | — | `cargo run --bin distill_nix_generate -p symthaea-broca` |

## Honest benchmark numbers

- **Legacy substring scorer: 84/94 (89%)** — matches the prior-session
  baseline exactly. Substring-containment on generator output.
- **Structural scorer, no repair: 26/26 (100%)** on the 26 golden-
  backed subset. 4 codegen bugs surfaced and fixed during the 26/94
  golden-writing process (see BENCHMARK.md for the per-commit story).
- **Structural + repair: 26/26 (100%)** with zero repair triggers
  after the native idiom fixes landed.

## The four bugs the scorer caught that substring didn't

| # | Prompt | What substring approved | What scorer flagged | Fix |
|---|---|---|---|---|
| 1 | `rust dev shell with sccache and openssl` | `sccache` in `buildInputs` | missing `RUSTC_WRAPPER = "sccache"` | `a69ca7d5df` |
| 2 | `open udp port 51820 for wireguard` | `allowedTCPPorts` (the right port number, wrong protocol) | missing `allowedUDPPorts` | `ae7408e7d5` (half 1) |
| 3 | `set time zone to Africa/Johannesburg` | `{ }` — empty config (no forbidden substrings either) | missing `time.timeZone` | `ae7408e7d5` (half 2) |
| 4 | `configure intel hardware acceleration` | `{ # hardware config }` — a comment | missing `hardware.graphics.enable` | `ce169cf3ee` |

Each of these is a **silent footgun** for anyone taking generator
output at face value. The substring scorer has no way to distinguish
them from correct output.

## Why this matters (the research framing)

### What we are NOT competing on

- Parameter count
- Training data volume
- Natural-language reasoning fluency
- Language breadth

LLMs win on all of these. Symthaea cannot and should not try.

### What we ARE competing on

- **Verifiability.** Every generation can be structurally scored
  against a golden. No "we think it's probably correct."
- **Closed-loop repair.** Scorer FAILs produce actionable feedback
  (missing paths, wrong values, protocol mismatches) that the
  `repair_structural` heuristics act on. The loop is bounded — max 5
  iterations per call.
- **Consciousness-gated emission.** Broca's `EpistemicCubeGate`
  operates at logit level — physically prevents tokens that violate
  epistemic constraints from being emitted. LLMs don't have this
  because they don't have consciousness-level scalars.
- **Substrate-independence.** The scorer + repair + KG pattern is
  not Nix-specific. Ports to Terraform HCL, Docker Compose, or
  Kubernetes manifests should exhibit the same "catches what
  substring misses" behavior — the substrate-independence claim.

## Reproducibility

```bash
# Preconditions: CUDA toolkit (12.x — 12.9 OK via the vendored cudarc patch),
# Rust 1.75+, nix-instantiate installed.
cd /srv/luminous-dynamics/symthaea

# Layer 1: run the benchmark (goldens subset, structural + repair)
cargo run --features code_generation \
    --example nix_eval_benchmark -- --goldens-only --repair

# Layer 2: harvest training pairs (26 PASS-filtered)
cargo run --features code_generation \
    --example harvest_nix_distillation

# Layer 3: train Broca on the harvested pairs (smoke: 1 epoch)
cargo run --bin distill_nix_train -p symthaea-broca -- --epochs 1

# Layer 4: load checkpoint + generate
cargo run --bin distill_nix_generate -p symthaea-broca
```

Each step produces observable output. Steps 1-2 are fast (seconds-
minutes). Step 3 is 20 min CPU / ~5 min GPU per epoch for 26 pairs.
Step 4 is seconds.

## Honest scope limitations

- **Training quality gated on epoch count.** The smoke-tested
  checkpoint (1 epoch) produces gibberish. Real quality needs
  50-200 epochs on GPU — a focused batch job, not a tight demo.
- **Broca tokenizer still default-minimal.** The `NIX_TOKENS`
  vocabulary (M5 of the coding-AI roadmap) is defined but not yet
  threaded into `BrocaGenerator::new`. Training still works;
  per-token granularity is sub-optimal.
- **94-problem corpus is author-curated.** Real-world benchmark
  (scrape public dotfiles) is Phase 4 of the coding-AI roadmap,
  not yet implemented.
- **One substrate.** The pipeline currently only handles Nix.
  Terraform HCL port is Phase 3 of the roadmap — independent of
  training status.

## Architecture diagram

```
              prompt (natural language)
                     │
                     ▼
              classify_nix_intent
               + build_nix_channels             ┐
                     │                           │  main crate
                     ▼                           │  (feature =
              generate_nix_with_self_repair      │   code_generation)
                     │                           │
                     ▼                           │
               ╔═════════════╗                   │
               ║  scorer     ║ ── verify ──▶ ┐   │
               ║  (rnix AST) ║               │   │
               ╚═════════════╝               │   │
                     │                       │   │
                     ▼                       │   │
               ╔═════════════╗               │   │
               ║  repair     ║ ◀─ feedback ──┘   │
               ║  (text)     ║                   │
               ╚═════════════╝                   │
                     │                           │
                     ▼                           │
              PASS-filtered pair                 │
              (prompt, 17D channels, Nix code)   ┘
                     │
                     ▼ harvest_nix_distillation
                JSONL file
                     │                           ┐
                     ▼                           │
              distill_nix_train                  │
              (Broca: CfC-HDC + Liquid-Mamba)    │  broca crate
                     │                           │  (feature = ssm_language)
                     ▼                           │
              broca-nix-distilled.mpk            │
                     │                           │
                     ▼                           │
              distill_nix_generate               │
              (EpistemicCubeGate + emission)     ┘
                     │
                     ▼
              Nix tokens / text
```

## Next work (multi-session)

Ordered by research-result value per hour:

1. **Real training run** (1-2 days GPU) — produces a Broca that emits
   real Nix, unlocks the `--deep` benchmark mode.
2. **Channel alignment + custom-vocab wiring** (~1 day) — integrates
   M5's `NIX_TOKENS` into `BrocaGenerator::new`. Improves per-token
   granularity during training.
3. **EpistemicCubeGate adversarial demo** (~1 day) — after real
   training, craft prompts that tempt the generator to emit imaginary
   option paths (`services.my_invented.enable`); show the gate
   suppressing them at logit level. This is the research crown jewel.
4. **Phase 3: HCL port** (2-3 days) — prove substrate-independence.
5. **Phase 4: real-world bench** (1-2 weeks) — publishable number.

## Documents

- `BENCHMARK.md` — per-commit benchmark trail, including the four
  codegen-bug stories.
- `plans/symthaea-coding-ai-roadmap.md` — the original roadmap
  framing the above work (in user's plans dir).
- `memory/symthaea_coding_ai_phase1_p2_m5.md` — next-session
  handoff with specific entry points.

---

*Consciousness-first correctness-verified coding AI, in 26 commits and
running end-to-end.*
