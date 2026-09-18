# Boot / Experience PR Train v1

The boot-experience program is split into independently revertible tranches. Qualification and architecture convergence are explicit gates; visual feature work does not bypass them.

1. `boot/observability-protocol-v1` — observability invariants, typed boot protocol, Nix presentation policy, executable protocol limits.
2. `spore/boot-ecology-v0.3` qualification freeze — finish exact-render evidence, renderer-cost evidence, declarative package validation, and companion QEMU fail-open qualification without adding another visual feature tranche.
3. `boot/protocol-ecology-convergence-v1` — adapt the qualified Boot Ecology factual inputs to `symthaea-boot-protocol`; remove duplicated live-observation types only after parity/golden tests exist. See `BOOT_ECOLOGY_CONVERGENCE_V1.md`.
4. `boot/observer-v1` — authoritative structured systemd/Nix target observation to normalized protocol events. The observer is disposable and never a boot dependency.
5. `boot/interactive-vt-v1` — explicit ambient/diagnostics/raw visibility controls with an independent raw VT.
6. `boot/diagnostics-ui-v1` — framebuffer-native structured diagnostics overlay consuming protocol state only.
7. `boot/health-escalation-v1` — robust delay/degradation/failure policy and automatic presentation escalation.
8. `boot/receipts-v1` — bounded boot receipts, historical timing baselines, compare/replay tooling with a strict privacy budget.
9. `boot/ambient-handoff-v1` — semantic boot-to-greeter/session handoff of the same qualified render plan and visual genome.
10. `boot/qualification-v1` — expanded VM fault injection, representative hardware qualification, `nixos-rebuild test`, boot-only canary, blessing, and only then persistent enablement.

## Merge discipline

Every PR must preserve `BOOT_OBSERVABILITY_INVARIANTS_V1.md` and satisfy all of the following:

- independently revertible;
- no new boot authority outside Linux/systemd/NixOS;
- no presentation component in a `Requires=` chain needed for login;
- malformed/missing protocol data degrades presentation only;
- raw diagnostics remain independently reachable;
- no physical-host enablement hidden inside an architecture or visual PR;
- no new visual fidelity work while the current qualification head is red or unreviewed.

The governing availability property is: **a broken or absent Spore presentation stack must be no less bootable than a machine with Spore completely removed.**
