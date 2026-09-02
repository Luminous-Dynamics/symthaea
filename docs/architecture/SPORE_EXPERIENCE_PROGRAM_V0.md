# Spore Experience Program v0

The long-term objective is not a replacement compositor or a themed wallpaper stack. Spore is an experience layer over mature Linux infrastructure, beginning with boot observability and lifecycle continuity.

## Program A — Lifecycle + Ambient

Boot, diagnostics, raw-log escape hatch, semantic handoff, lock, suspend/resume, and the ambient runtime.

## Program B — Continuity + Spaces

Persistent semantic workspaces, session restoration adapters, spatial memory, and per-Space policy.

## Program C — Intent + Attention

A command canvas that resolves human intent into typed actions, contextual search/actions, notification policy, Focus, Gaming, and Presentation modes.

## Program D — Trust + Reversibility

Permission visibility, causal explanations, meaningful system history, Nix change previews, undo/rollback, and recovery surfaces.

## Host strategy

Use Plasma/KWin and standardized Wayland/portal/PipeWire/systemd/Nix interfaces first. Do not build a compositor unless multiple hard product constraints cannot be expressed through supported host APIs and the compatibility burden is justified by concrete user benefit.

## Cross-program invariants

- Presentation never becomes authority.
- AI may propose typed actions; it never receives an unrestricted privileged shell.
- Core desktop operation remains complete offline.
- Optional Spore services fail independently and degrade calmly.
- User-visible operations should be reversible when the underlying platform permits it.
- Accessibility, reduced motion, privacy, and performance policy are first-class inputs rather than afterthoughts.
- Ambient consumers receive intentionally lossy context instead of raw surveillance-grade telemetry.
