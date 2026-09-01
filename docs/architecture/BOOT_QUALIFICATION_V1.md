# Spore Boot Qualification v1

Status: draft qualification ladder

## Principle

No decorative boot component is enabled on a physical host merely because it compiles. Qualification progresses from pure Rust semantics through VM lifecycle behavior to a reversible physical canary and repeated performance evidence.

## Q0 — architecture/static review

Required:

- authority boundaries documented;
- no presentation component required by boot/login/recovery;
- raw Linux diagnostics remain independently reachable;
- input adapter lifetime ends with early boot;
- telemetry and receipts remain privacy-minimized;
- stop timeout is bounded;
- no known systemd transaction cycle.

## Q1 — focused Rust/Nix lane

Run:

```bash
bash scripts/check-spore-boot-stack.sh
```

This checks formatting, all targets, tests, Clippy with warnings denied, the deterministic headless benchmark smoke case, and Nix module parsing when Nix is available.

Covered packages:

- `symthaea-boot-protocol`;
- `symthaea-boot-observer`;
- `symthaea-quicken-fb`;
- `symthaea-boot-control`;
- `symthaea-boot-input`.

## Q2 — NixOS VM

Run:

```bash
bash scripts/check-spore-boot-stack.sh --vm
```

The existing virtio-gpu VM is the first environment for lifecycle wiring. Expand VM assertions before enabling any new display-manager gate.

Required scenarios:

1. renderer starts on DRM-capable VM;
2. headless/no-DRM case skips without blocking boot;
3. renderer receives SIGTERM and exits within the configured stop bound;
4. post-DRM handoff receipt appears only after renderer release path;
5. missing/unwritable receipt does not block boot;
6. performance receipt is valid when enabled and absent when disabled;
7. observer absent;
8. observer restart/new lineage;
9. malformed/oversized telemetry;
10. display manager start/restart once a qualified trigger exists;
11. raw-log VT request once VT coordination exists;
12. recovery/rescue boot bypass.

## Q3 — physical canary

Do not replace the normal boot entry.

Use a dedicated NixOS specialisation or generation that enables the new Spore path while preserving:

- the previous generation;
- a diagnostic/no-Spore boot entry;
- rescue/emergency access;
- raw console availability.

Canary acceptance requires repeated clean boot, shutdown, suspend/resume, and display-manager restart on the actual GPU/driver combination.

A renderer crash, observer crash, missing receipt, malformed state file, or input adapter failure must not prevent login.

## Q4 — performance qualification

Enable opt-in renderer metrics and capture evidence with:

```bash
bash scripts/measure-spore-boot.sh <output-directory>
```

Compare otherwise-identical Spore ON/OFF specialisations using alternating blocks. Use distributions, not one boot.

At minimum inspect:

- whole boot time;
- graphical target timing;
- display-manager timing;
- DRM open;
- first frame;
- grow/render/blit p50/p95/p99;
- frame-work deadline misses;
- post-DRM release;
- time-to-session delta.

## Q5 — enablement

Only after Q0–Q4:

- enable typed telemetry by default for the Spore specialisation;
- qualify F1/F2/Esc integration;
- qualify the display-manager handoff trigger;
- consider making the specialisation the normal default.

Never remove the diagnostics/no-Spore escape path.

## Merge discipline

Keep the stack reviewable:

```text
protocol/policy
  -> systemd observer
  -> renderer telemetry adapter
  -> presentation control
  -> ephemeral input adapter
  -> renderer handoff contract
  -> performance baseline
  -> qualification
  -> measured optimization
  -> compatibility-preserving rename
```

Do not mix measured optimization and the `quicken-fb` rename. A rename should not change benchmark behavior.
