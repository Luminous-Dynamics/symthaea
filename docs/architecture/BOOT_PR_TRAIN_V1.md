# Boot / Experience PR Train v1

1. `boot/observability-protocol-v1` — observability invariants, typed boot protocol, Nix presentation policy.
2. `boot/observer-v1` — authoritative structured systemd/Nix target observation to normalized events.
3. `boot/interactive-vt-v1` — explicit ambient/diagnostics/raw visibility controls with independent raw VT.
4. `boot/diagnostics-ui-v1` — framebuffer-native structured diagnostics overlay.
5. `boot/health-escalation-v1` — robust delay/degradation/failure policy and automatic presentation escalation.
6. `boot/receipts-v1` — bounded boot receipts, historical timing baselines, compare/replay tooling.
7. `boot/ambient-handoff-v1` — semantic boot-to-session ambient handoff.
8. `boot/qualification-v1` — VM/fault/hardware qualification and enablement gates.

Every PR is independently revertible and must preserve the invariants in `BOOT_OBSERVABILITY_INVARIANTS_V1.md`.
