# SYM-WASM-000A Red Baseline

This document records the intended first qualification boundary for issue #4198.

At the parent subject `ec2cd846132eae5c6a5023618361d2de6ab30b9a`, `ActionIR::WasmSandbox` still creates `wasmtime::Engine::default()`, loads a caller-selected module path directly with `Module::from_file`, creates an unrestricted `Store<()>`, and sets no Wasmtime fuel or store limiter.

The accompanying static qualification lane is intentionally expected to report RED on this subject. That RED is evidence of the pre-repair state, not an infrastructure failure.

The repair claim is intentionally narrow:

- module resolution is forced through `SandboxRoot`;
- direct `Module::from_file` is removed from the real WasmSandbox arm;
- the engine explicitly enables fuel consumption;
- each invocation receives an explicit fuel budget;
- store memory/table/instance/memory-count limits are installed;
- no WASI capability is introduced.

A green static lane will still not establish the runtime adversarial theorem. Infinite-loop, memory-growth, malformed-module, missing-import, and host-survival tests belong to the next qualification layer.
