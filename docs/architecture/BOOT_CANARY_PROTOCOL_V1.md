# Spore Boot Canary Protocol v1

This protocol defines the only supported path from a qualified Spore build to persistent physical-host enablement.

## Principle

A visual boot feature is not qualified because it compiled, rendered correctly once, or reached the desktop once. Qualification demonstrates that presentation failures do not weaken bootability, recovery, or generation provenance.

The physical host remains disabled until every preceding gate is complete.

## Gate A — Exact source and build

Record and verify:

- exact Symthaea source commit;
- exact host-integration commit;
- committed `flake.lock` resolving that Symthaea commit;
- Rust toolchain and `Cargo.lock` used by the boot tools;
- successful declarative `spore-boot-tools` derivation;
- clean repository state for the candidate being qualified.

A moving branch is never a qualification input.

## Gate B — Renderer evidence

The exact live renderer path must produce reviewable artifacts for the lifecycle and Inoculation matrices. Generated concept art is design input only.

Before increasing fidelity, collect representative 1080p and 1440p renderer-cost evidence. Record p50/p95/max frame time and peak memory. Performance policy should be derived from measurements on representative hardware rather than invented in advance.

Reviewers explicitly check:

- no blank/near-blank scenarios;
- no accidental pixel-identical collapse between semantically distinct cases;
- readable sparse identity labels;
- no fake precision/progress;
- recovery states remain calm and factual;
- handoff frames are visually suitable for compositor transition.

## Gate C — VM fail-open matrix

At minimum exercise:

1. normal renderer and handoff;
2. renderer exits with failure;
3. renderer ignores SIGTERM until systemd enforces its bound;
4. state preparation fails / no receipt exists;
5. no usable DRM device;
6. corrupt or unsupported receipt;
7. progress source never appears;
8. renderer is killed mid-frame;
9. health observation succeeds but LKG blessing fails;
10. repeated boot of the same generation does not rotate `previous`;
11. a distinct generation transition rotates `previous` exactly once;
12. rollback restores prior Current/Previous/LKG semantics.

Every fault case must still make the independent graphical/diagnostic boot path available.

## Gate D — Live activation without boot blessing

Run `nixos-rebuild test` only after the VM matrix passes.

The host integration must distinguish `/run/current-system` from `/run/booted-system`. A live `test` or `switch` activation may change the former but must never promote that unbooted generation to Last Known Good or rotate the semantic Current/Previous boot roots.

The renderer must skip DRM acquisition when the display manager is already active.

During this gate verify:

- display manager remains active;
- no DRM takeover occurs;
- no new failed boot-critical units appear;
- Last Known Good still points at the actually booted generation;
- Current semantic root still points at `/run/booted-system`;
- a `bless-skipped-live-activation` or equivalent audit result is recorded when applicable.

## Gate E — Boot-only canary

Only now make the qualified candidate bootable for one reboot while retaining an immediately selectable known-good entry.

Before reboot record:

- bootloader entries and selected default;
- Current / Previous / Last-Known-Good roots;
- exact candidate and LKG store paths;
- boot-counting state where enabled;
- recovery instructions that do not depend on Spore.

On the canary boot verify:

- the candidate actually equals `/run/booted-system`;
- renderer starts only if a usable state receipt and DRM path exist;
- explicit compositor handoff completes within its bound;
- display manager and graphical session become usable;
- GPU/DRM state is healthy after handoff;
- Last Known Good advances only after the host health predicate succeeds;
- the recorded blessed generation exactly matches `/run/booted-system`.

Any uncertainty means the candidate is not blessed.

## Gate F — Persistent enablement

Persistent enablement is permitted only after the boot-only canary is reviewed and blessed.

Do not combine persistent enablement with a renderer feature change, protocol migration, Nix pin change, or recovery-policy change. Those require a new candidate lineage and qualification cycle.

## Abort conditions

Abort promotion and preserve the prior LKG if any of the following occur:

- compositor cannot acquire DRM cleanly after handoff;
- renderer exceeds its enforced lifetime;
- boot-state lineage is malformed or ambiguous;
- Current/Previous/LKG roots do not match their documented semantics;
- the candidate was only live-activated rather than actually booted;
- display manager starts and then repeatedly fails during the health window;
- evidence artifacts do not correspond to the exact source commit being installed.

## Governing availability property

> A machine with a broken or absent Spore presentation stack must be no less bootable than the same machine with Spore completely removed.

This property outranks animation fidelity, continuity, diagnostics convenience, and boot-time aesthetics.
