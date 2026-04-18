# Symthaea Nix Codegen — Benchmark Record

Honest measurement log. Each entry: date, commit, scorer used, result. Don't
amend historical rows — if numbers change, append a new row and note why.

## NixEval — 95-problem corpus (`src/language/nix_eval_corpus.rs`)

### 2026-04-18 — Structural scorer landed (`bcb2c3acd3`)

**Mode: `--goldens-only` (6 prompts, structural AST scorer)**

| # | Prompt | Verdict |
|---|---|---|
| 1 | set up postgresql with pgvector | PASS |
| 2 | enable nginx web server | PASS |
| 3 | enable redis cache server | PASS |
| 4 | enable docker and add my user to the docker group | PASS |
| 5 | set up a rust dev environment with rust-analyzer and mold | PASS |
| 6 | rust dev shell with sccache and openssl | FAIL (missing `RUSTC_WRAPPER` — real codegen gap) |

**Score: 5/6 (83%)**

### 2026-04-18 — RUSTC_WRAPPER codegen fix (same-day follow-on)

The 1/6 FAIL was a real generator defect the scorer surfaced. Fixed in
`src/language/nix_codegen.rs::emit_dev_shell` — when `sccache` is
requested, now emits `RUSTC_WRAPPER = "sccache";` alongside it. Prompt 6
flipped to PASS.

| # | Prompt | Verdict |
|---|---|---|
| 1–5 | (as above) | PASS |
| 6 | rust dev shell with sccache and openssl | PASS |

**Score: 6/6 (100%)**

Reproduce: `cargo run --features code_generation --example nix_eval_benchmark -- --goldens-only`

### Context

Prior session reported **84/94 (89%)** on the full corpus using legacy
substring-containment scoring. That number was an **upper bound on an
unknown true score** because:

- Substring scoring passes on `services.postgresql.enable = false; # pgvector`
  when the required substrings are `postgresql`, `enable`, `pgvector`
- No AST-based value check — `enable = true` vs `enable = false` was
  indistinguishable to the scorer
- Comments containing fake option paths could satisfy requirements

This session shipped the structural scorer (`src/language/nix_scorer.rs`)
and hand-wrote 6 golden references to calibrate. The 5/6 PASS rate on the
golden subset is the first **honest** number we have.

### Scorer evolution across commits

| Commit | Change | 6-prompt goldens score |
|---|---|---|
| `086cbc5a82` | Initial scorer, strict equality on all values | (n/a — not wired yet) |
| `3cca653f0a` | First end-to-end run, strict walker | 1/6 (17%) |
| `3cca653f0a` | Fixed: nested `services.x = { enable = ...; }` → `services.x.enable` | 4/6 (67%) |
| `bcb2c3acd3` | Added PackageList with subset semantics (extras OK) | 5/6 (83%) |

Each iteration was driven by `nix_scorer_diagnose` output — running the
scorer against one prompt, printing every attrpath extracted on both
sides, and figuring out why it failed. Two real bugs fixed in the
scorer (nested walker, list superset); one **real codegen gap** surfaced
(sccache without RUSTC_WRAPPER) which the scorer correctly catches.

### Full corpus (95 problems) — pending

The full benchmark requires more goldens. Backfilling happens in
`src/language/nix_eval_goldens.rs::golden_for`. When complete, run:

```
cargo run --release --features code_generation --example nix_eval_benchmark -- --structural
```

Expected trajectory from the plan (`plans/please-explore-deeper-research-logical-flurry.md`):

- Legacy substring: 84/95
- Structural (after all goldens backfilled): **projected 55–70/95**
- Structural + module-eval cache (P3): **projected 50–65/95**

The legacy score drop is where the real information lives — it tells
us which problems the substring scorer was over-counting.

## Follow-ups surfaced by the scorer

### ~~Codegen: dev-shell missing RUSTC_WRAPPER~~ RESOLVED

~~Prompt: `rust dev shell with sccache and openssl`~~

~~The generator puts `sccache` in `buildInputs` but doesn't wire the
`RUSTC_WRAPPER = "sccache";` env var.~~ Fixed 2026-04-18 in
`emit_dev_shell` — when sccache is requested, the env var is now
emitted. Prompt 6 now PASSes structural scoring.

**Value of this workflow:** the structural scorer surfaced a bug that
the substring scorer silently approved. A user taking that shell would
have had sccache installed but not wrapping compilation — silent
footgun. This is exactly the kind of defect an honest benchmark catches.

### Scorer: list semantics for non-package lists

`PackageList` subset semantics only fire on identifier-only lists.
Integer/string/expression lists stay Opaque, so extras in a firewall
port list would correctly fail. This is a deliberate boundary — lists
like `allowedTCPPorts = [ 80 443 8080 ]` with an extra 8080 should
probably fail the check (security-relevant) rather than be treated as
"extras OK". Revisit if real corpus growth shows this is too strict.

### Goldens not yet covering:

- Hardware configs (nvidia, amd, intel)
- Desktop environments (sway, kde, hyprland, gnome)
- Networking / firewall combinations
- Secrets / agenix / sops
- Home Manager integrations

Backfilling these is the P1 completion work — see plan §P1 for the
1-day budget estimate.
