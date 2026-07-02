# Xenia: Comprehensive Review & Improvement Plan (2026-07-02)

Scope: everything under `/srv/luminous-dynamics/xenia/` (the `xenia-wire` protocol
crate and `xenia-peer` application workspace, both vendored copies of standalone
public GitHub repos), the `sovereign-admin` (`xenia-admin`) Leptos console inside
`xenia-peer`, and the large amount of Xenia-related patch/archive churn sitting at
the monorepo root. Produced by four parallel audits (protocol crate, app workspace,
root-level patch churn, port-registry reconciliation).

## TL;DR

Xenia is a real, honestly-labeled, pre-alpha P2P remote-session project
(`xenia-wire` = AEAD wire protocol, `xenia-peer` = daemon + viewer app on top of
it). **Good news first**: unlike some other subsystems in this monorepo's recent
history, no fabricated results or inflated claims were found here — tests pass
(50+ in xenia-wire, 226 in xenia-peer), clippy is clean, `#![deny(unsafe_code)]`
holds in the wire crate, and several docs (ROADMAP.md, LEDGER_VERIFICATION_BOUNDARY.md)
are unusually disciplined about stating what is *not* proven. The actual problems
are process/hygiene: a stale `.gitignore` rule that no longer matches where the
repos live, ~19GB of pure disk waste from repeatedly archiving `target/` build
output, several already-merged patch files cluttering the monorepo root, and one
real documentation mislabel (the "Xenia Admin" port-registry entry).

## What Xenia actually is (for future sessions)

- **`xenia-wire`** (`xenia/xenia-wire/`) — standalone repo, published to crates.io,
  v0.2.0-alpha.3. Byte-level AEAD-sealed envelope protocol (ChaCha20-Poly1305,
  64-slot replay window, key rotation with grace period). No transport, no
  handshake, no PQC itself — "designed for ML-KEM-capable session keys supplied by
  a higher layer." Explicitly "PRE-ALPHA — DO NOT USE IN PRODUCTION."
- **`xenia-peer`** (`xenia/xenia-peer/`) — standalone repo, the application layer:
  a headless daemon (hosts a screen) and a viewer (CLI today, egui GUI at M4),
  built on `xenia-wire` + a real ML-KEM-768/Ed25519/HKDF handshake crate
  (`xenia-handshake`). Explicitly "PRE-ALPHA, M0 — NOT USABLE AS A PRODUCT YET."
  Real screen capture, input injection, consent UI, and video encode of real
  content are the documented hard blockers before anything resembling a product.
- **`xenia-admin`** (`xenia/xenia-peer/apps/sovereign-admin/`) — a Leptos CSR
  operator console for xenia-peer sessions: device inventory, session monitoring,
  a live client-side Ed25519 consent-ledger demo, governance/policy pages. This
  is the same thing CLAUDE.md/PORTS.md calls "Xenia Admin" at port 8134 — see
  **Finding D** below for why that entry is misleading as currently worded. It
  belongs to a separate "Mycelix Sovereign Suite" (Xenia + Pulse + Athena L1 +
  Identity, for PAM / high-security ops) that borrows the Mycelix name but is
  **not** part of the 16-cluster Mycelix fractal governance architecture
  described elsewhere in CLAUDE.md. `mycelix-sovereign/nixos-modules/xenia.nix`
  confirms port 8134 as the real, current admin-console port for this suite.

## Findings

### A. `xenia-wire` — healthy, two trivial doc fixes, one uncommitted-but-finished pass

- `cargo test --workspace`: all green (~18s), `cargo clippy --all-targets`: zero
  warnings. `#![deny(unsafe_code)]` (`src/lib.rs:91`). Envelope-size checks reject
  undersized input before parsing; nonce exhaustion is guarded rather than wrapped.
- **Bug**: `README.md:203` claims "6 deterministic hex fixtures" in `test-vectors/`;
  the directory actually documents 12 (01–12). One-line fix.
- **Bug**: `xenia-viewer-web/target/` was accidentally committed once
  (`0791bbc`) because the repo's `.gitignore` only has an anchored `/target`
  rule, not a bare `target` pattern. Someone has already deleted ~493 tracked
  artifact paths in the working tree (`git status` shows them as `D`), but
  **the `.gitignore` fix itself is still missing** — without it, the next
  `cargo build` in that subdir will re-stage hundreds of files.
- **Uncommitted, finished work**: `Cargo.toml`, `README.md`, `SECURITY.md`,
  `SPEC.md`, `src/lib.rs`, `.github/workflows/ci.yml` all carry one coherent,
  complete change — replacing "PQC-sealed" language with accurate "AEAD-sealed,
  designed for ML-KEM-capable handshakes" wording, plus a new `pqc-claims` CI
  job running `scripts/check-pqc-claims.sh` (already exists, executable). This
  should just be committed; it is not a stub.
- **Nice-to-have**: `open_frame_lz4` (`src/wire.rs:107-111`) decompresses via
  `decompress_size_prepended` on already-AEAD-verified plaintext (correct
  ordering — no pre-auth zip bomb), but the embedded size prefix itself isn't
  capped, so a valid-key sender could still trigger one large allocation
  post-auth. Low severity; worth a bound anyway.
- Unverifiable-from-repo: the empirical bandwidth numbers in the README
  (3.27–3.52× bincode, 2.12× LZ4, 4.7× WS-vs-QUIC HoL blocking) come from an
  external research stack (Holon-Soma) not present in this repo. Treat as
  reported, not independently reproduced here.

### B. `xenia-peer` — healthy, real crypto backing real docs, two stale paths

- `cargo test --workspace`: 226 passed, 0 failed (~3 min). `cargo clippy
  --all-targets`: zero errors, only style-level lints (a couple of
  collapsible-ifs in `apps/xenia-peer/src/m1_runtime.rs`, one
  `too_many_arguments` in `apps/xenia-peer/src/main.rs:1471`).
- Crate-by-crate reality check: `xenia-capture`'s `ScapCapture` backend (real,
  443 LOC, wired to PipeWire/portal via a forked `scap` crate) exists but is
  feature-gated off by default — so "synthetic-only today" is accurate for
  default builds, not because the real path doesn't exist. `xenia-inject` is a
  genuine stub (`NoopInjector`/`LoggingInjector` only), matching its own
  README table exactly — no overclaim. `xenia-ledger` (3,952 LOC, 58 passing
  tests, real Ed25519 hash-chain + tamper detection) and `xenia-handshake`
  (1,195 LOC, 26 tests, real ML-KEM-768 + Ed25519 + HKDF-SHA256, always-on, not
  feature-gated) are both real, not scaffolds.
- The ~15 docs under `docs/crypto/` are disproportionate in volume to the
  crypto code (~5,000 LOC total across ledger+handshake), but their *content*
  mostly documents boundaries and explicit non-claims
  (`docs/security/LEDGER_VERIFICATION_BOUNDARY.md` enumerates what the ledger
  does **not** prove: key custody, timestamp truth, UI honesty, host
  integrity, legal sufficiency) rather than inflating capability. This is the
  opposite of this house's more common failure mode and is worth naming as a
  good pattern to keep.
- **Bug**: `xenia-peer/crates/xenia-ledger/README.md`'s own "Status" section
  says "~480 LOC, 8 tests" — stale by roughly 8× on LOC and 7× on test count
  (undercounting, not overselling, but still wrong and worth refreshing).
- **Bug**: `apps/sovereign-admin/README.md` lines 19/48/82 reference
  `crates/xenia-admin` / `cd crates/xenia-admin`, but the actual directory is
  `apps/sovereign-admin` — a stale path from before a rename. Anyone following
  the README verbatim hits a `cd: no such directory` error.
- `ROADMAP.md` is unusually disciplined ("If this file disagrees with reality,
  the file is wrong"), and its hard-blockers list matches what code inspection
  actually shows. No milestones found marked done that don't hold up.

### C. `sovereign-admin` (`xenia-admin`) — scaffold, matches its own README's honesty

Confirmed to be the actual target of the CLAUDE.md/PORTS.md port-8134 entry (see
Finding D). Login/Devices/Policy pages are explicitly marked scaffold in its own
README; the live Ed25519 ledger demo is real and working. No further concerns
beyond the stale path noted in Finding B.

### D. CLAUDE.md / PORTS.md "Xenia Admin" entry — misleading, needs a wording fix

Current text (both files):

> `8134 | Xenia Admin (Mycelix Sovereign admin console, Leptos CSR) | admin.sovereign.mycelix.net | Scaffold`

This is accurate on port/domain (confirmed against `mycelix-sovereign/nixos-modules/xenia.nix`,
which defaults `adminConsolePort` to 8134 and cites this exact PORTS.md line) but
misleading in two ways: (1) it reads as if this is a frontend for the 16-cluster
Mycelix governance architecture described elsewhere in CLAUDE.md — it is not; it's
the operator console for the unrelated Xenia PAM/remote-support product, part of a
separate "Mycelix Sovereign Suite." (2) status "Scaffold" is now stale — the app
has grown a live ledger demo and governance/monitor pages beyond what "Scaffold"
implies alone.

**Recommended replacement:**

> `8134 | Xenia Admin (xenia-admin crate — operator console for the Xenia PAM/remote-support product in xenia/xenia-peer/apps/sovereign-admin/, part of the separate Mycelix Sovereign Suite, not a Mycelix governance-cluster frontend) | admin.sovereign.mycelix.net | Scaffold + live ledger demo`

### E. Root-level patch-train churn — real workflow, mostly stale, ~19GB of waste

The dozens of `xenia-*.patch` files and repeated `_archive/xenia-*-v1-v2-v3-v4-v5-v6-v7-*`
snapshots at the monorepo root are **not random noise**. `apply-xenia-pqc-patch-train.sh`
documents a real (if externally-authored) workflow: patches are prepared outside
this environment (one earlier-found patch note reads "prepared from an uploaded
repo archive and was not compiled in the ChatGPT container"), then applied here
into the real `xenia/xenia-wire` / `xenia/xenia-peer` working trees — the script
refuses to run unless both trees are clean, and it archives a pre-apply snapshot
on every invocation.

Problems found:
- **~19GB is pure waste**: several `_archive/xenia-*` dirs are full `target/`
  build-artifact snapshots (`xenia-peer-build-target-20260702T0830Z` alone is
  14GB; two more pairs are 1.6GB×2 and 916MB×2). These have zero audit value —
  build output is regenerable — and should simply be deleted.
- **Several root `.patch` files are already stale/merged.** Cross-checking
  content against the live repos shows some patches (e.g. touching
  `read_evidence_crypto_manifest_export_dir`, `EvidenceBundleSeal`, the `chrono`
  dependency) are already committed in `xenia-peer` — leftovers from an
  already-applied round that were never cleaned up.
- A closeout memo from the v1–v7 round
  (`_archive/xenia-closeout-memo-...-20260627T151026Z/closeout-summary.md`)
  explicitly states: *"This memo closes the v1-v7 patch series. Do not add
  another paper-only patch round before local Cargo/Nix validation or
  implementation work"* — with a **FAIL** result recorded because validation
  docs/scripts weren't in place at the time. Worth checking whether that
  guidance has been honored since, given the pattern repeated through at
  least 2026-07-02.
- **Stale `.gitignore` rule**: the monorepo's own `.gitignore` (lines 318-320)
  ignores `/xenia-wire/`, `/xenia-server/`, `/xenia-peer/` at the monorepo
  root — but the actual repos live nested under `/xenia/xenia-wire/` and
  `/xenia/xenia-peer/`. The rule stopped matching once the repos moved under a
  `xenia/` subdirectory, which is why `git status` currently shows the entire
  `xenia/` tree as untracked (`?? xenia/`) instead of cleanly ignored. The
  comment also references `memory/xenia_*_shipped.md` for context — no such
  memory file currently exists in this project's memory store, so that
  pointer is dangling too.
- No secrets found in any sampled patch file. Risk is low-but-real: the
  currently-dirty `xenia-wire` tree (Finding A) plus stale already-merged
  patches sitting at root is a footgun for a future manual `git apply` against
  a tree that's since diverged.

## Improvement Plan

**P0 — cheap, do first (no risk, mostly one-line fixes):**
1. Fix `.gitignore` (outer monorepo): update the "External public sibling
   repos" block to the real nested paths (`/xenia/xenia-wire/`,
   `/xenia/xenia-peer/`, or simply `/xenia/` if the whole tree should stay
   out of the monorepo) so `git status` stops showing it as untracked noise.
2. Fix CLAUDE.md + PORTS.md port-8134 line per Finding D's recommended wording.
3. In `xenia-wire`: add `xenia-viewer-web/target` (or de-anchor to a bare
   `target` pattern) to `.gitignore`, commit that alongside the already-deleted
   artifact paths; fix README.md:203 "6" → "12" fixtures; commit the
   already-finished PQC-wording pass (Cargo.toml/README/SECURITY/SPEC/lib.rs/ci.yml).
4. In `xenia-peer`: fix the stale `crates/xenia-admin` → `apps/sovereign-admin`
   path in `apps/sovereign-admin/README.md`; refresh `xenia-ledger/README.md`'s
   stale "~480 LOC, 8 tests" status line to current (3,952 LOC / 58 tests).

**P1 — low-risk cleanup (reclaims ~19GB, removes footguns):**
5. Delete the `target/`-snapshot archive directories under
   `_archive/xenia-*` (build output only, safely regenerable) — see Finding E
   for the specific dirs and sizes.
6. Reconcile the root-level `.patch` files against current repo state: delete
   ones already merged/stale; for any genuinely still-pending, move them into
   a `xenia/pending-patches/` (or similar) directory rather than the monorepo
   root, and add a root `.gitignore` rule for `/xenia-*.patch` and
   `/_patchwork/` so this doesn't recur as loose clutter.
7. In `xenia-wire`: cap the decompressed-size prefix in `open_frame_lz4`
   (`src/wire.rs:107-111`) as defense-in-depth against a valid-key sender
   triggering a large post-auth allocation.

**P2 — process / roadmap (needs a decision, not urgent):**
8. Decide whether the external-patch-train workflow (`apply-xenia-pqc-patch-train.sh`)
   is still the intended way to land changes into xenia-wire/xenia-peer. If
   yes: fix the archive step to snapshot only diffs + git-status output, never
   `target/`. If no: retire the script and document the replacement workflow.
9. Continue down `ROADMAP.md`'s existing hard-blocker sequencing (real capture
   backend enablement, input-injection backends, consent-ceremony UI, browser
   PQC handshake) — nothing in this review changes that priority order, it's
   just reaffirmed as accurate.
10. Consider a short note in CLAUDE.md clarifying that "Mycelix Sovereign
    Suite" (Xenia + Pulse + Athena L1 + Identity) is a distinct product line
    from the 16-cluster Mycelix fractal governance architecture, to prevent
    future sessions from conflating the two the way the port-registry entry
    did.

## What's notably *not* a problem here

Worth stating plainly given this house's history: no fabricated benchmark
results, no aspirational docs contradicted by code, clippy clean, tests green,
`unsafe_code` denied and enforced in the wire crate, and the security-boundary
docs (ROADMAP.md, LEDGER_VERIFICATION_BOUNDARY.md) actively under-claim rather
than over-claim in places. The issues found are entirely process/hygiene
(disk waste, stale doc paths, one mislabeled registry entry) — not integrity
issues in the code or its claims.
