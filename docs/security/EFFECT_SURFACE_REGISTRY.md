# Effect Surface Registry v1

The machine-readable inventory lives at [effect-surface-registry.v1.json](effect-surface-registry.v1.json).

## Run the integrity check

From the repository root:

\`\`\`sh
cargo run -p xtask -- effect-surface-registry
\`\`\`

To make a local qualification pipeline reject this starter registry while it still contains unqualified surfaces:

\`\`\`sh
cargo run -p xtask -- effect-surface-registry --require-qualified
\`\`\`

The strict invocation is expected to fail today because none of the entries has exact qualification evidence. This is intentional: the command must not promote a source inventory to a qualification result.

## What the checker establishes

The Rust xtask command validates the v1 schema, uniqueness of surface IDs, repository-relative source paths, required claim-boundary fields, qualification-evidence shape, and whether every recorded source anchor occurs in its recorded file. It emits the inspected Git HEAD when available.

It does **not** establish that:

- the inventory includes every process, filesystem, network, service, device, FFI, generated, macro-expanded, plugin, shell, or indirect effect;
- the recorded symbol is reachable from an untrusted entrypoint;
- a listed gate is enforced at the true effect point;
- positive or negative controls were compiled or executed;
- an authorization was authentic, current, correctly scoped, or durable;
- the runtime is secure or the repository is free of bypasses.

## Current scope

This is an initial known-surface baseline audited against main commit c6355cc6ffe01843b841f6b9a8a09eabd2556deb on 2026-10-28, informed by #8656 and the linked per-surface issues. The entries are source-review records with explicit limitations. The inventory is intentionally incomplete and all listed surfaces remain unqualified or only partially guarded.

## Next qualification stage

1. Expand discovery beyond the initial known paths using Rust AST and effect-primitive scans, while reviewing macros, traits, FFI, subprocess wrappers, scripts, and generated code.
2. Split compound records (especially SimpleExecutor) until each independently governable effect family has a unique adapter/test seam.
3. For every surface, add instrumented positive and negative controls with actual side-effect witnesses.
4. Pin source tree, registry digest, toolchain, test plan, and raw result digests in an immutable receipt.
5. Fail a qualification gate when a new effect site is unregistered or a registered entrypoint changes without reviewed inventory treatment.
6. Treat an exact-head skipped, queued, missing, or non-executed lane as unavailable evidence, never as PASS.

The registry is a discovery and assurance input, not an authority source. Adding a row cannot authorize an effect.
