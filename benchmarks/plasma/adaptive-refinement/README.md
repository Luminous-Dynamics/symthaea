# Response-aware memory-bounded spectral adaptation v0.1

This policy benchmark compares conventional spectral-tail triggers with future-response-aware triggers under a fixed peak-memory budget.

## Policy rule

Refine when either:
- spectral-tail measure / spectral-tail threshold >= 1; or
- the registered future-response upper bound / threshold >= 1.

A missing response estimate is not treated as zero risk: it triggers a conservative refinement attempt and remains UNKNOWN until a response measurement is available.

Before refinement, compare the projected peak memory (including padded arrays, FFT workspaces, split stages and temporary allocations) with the budget after reserved headroom. If the required allocation does not fit, return `BlockedByMemory` and `UnknownBlockedByMemory`. The policy must not call this a qualified stable state.

The Rust primitive is `decide_spectral_refinement`; its inputs include the spectral-tail measure, normalized future-response upper bound, projected peak memory, budget and reserve. The checked-in fixture exercises six deterministic manufactured cases.

## Relationship to the adaptive Fourier baseline

Cho & Russo (2026), arXiv:2607.12368, use heuristic high-k windows or boundary/global-peak tail triggers and Fourier zero-padding. Their method preserves discrete mass and L2 during padding, but their paper notes L2 decay at the GPU memory ceiling when unresolved filamentation continues while mass remains conserved.

This benchmark does not claim that tail triggers fail generally. It asks whether a future-QoI-aware signal can detect a low-energy but response-important feature earlier at the same memory ceiling, and whether the extra response estimation cost is worth it.

## Claim ceiling

These fixtures validate decision logic only. No spectral solver, kinetic trajectory, late-time echo or physical QoI has been evaluated. The projected peak allocation is only as reliable as the caller's memory model; it must include temporary buffers and solver stages.
