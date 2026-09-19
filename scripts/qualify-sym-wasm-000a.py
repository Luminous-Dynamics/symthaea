#!/usr/bin/env python3
"""SYM-WASM-000A static qualification canary.

This lane intentionally examines only the existing ActionIR::WasmSandbox source
boundary. It does not execute untrusted Wasm and it does not claim the full
sandbox theorem. The repair must make these source-level invariants true before
runtime adversarial tests are promoted.
"""
from pathlib import Path
import re
import sys

path = Path("src/action/mod.rs")
text = path.read_text(encoding="utf-8")

start = text.find("ActionIR::WasmSandbox {")
if start < 0:
    raise SystemExit("SYM-WASM-000A: WasmSandbox arm not found")

# Select the real-execution implementation, stopping before RunCommand.
real = text.find("ExecutionMode::Real =>", start)
end = text.find("ActionIR::RunCommand", real)
if real < 0 or end < 0:
    raise SystemExit("SYM-WASM-000A: real WasmSandbox implementation not found")
arm = text[real:end]

failures: list[str] = []

# No default engine: the executor must opt into fuel metering explicitly.
if "Engine::default()" in arm:
    failures.append("UNBOUNDED_DEFAULT_ENGINE")
if "consume_fuel(true)" not in arm:
    failures.append("FUEL_METERING_NOT_ENABLED")
if not re.search(r"set_fuel\s*\(", arm):
    failures.append("FUEL_BUDGET_NOT_SET")

# Module bytes must be resolved through SandboxRoot before loading.
if not re.search(r"sandbox\.validate\s*\(\s*module_path\s*\)", arm):
    failures.append("MODULE_PATH_NOT_SANDBOX_ROOTED")
if "Module::from_file" in arm:
    failures.append("DIRECT_MODULE_FROM_FILE")

# The store must be resource-limited, not Store<()>.
required_limit_tokens = [
    "StoreLimitsBuilder",
    ".memory_size(",
    ".instances(",
    ".tables(",
    ".memories(",
    ".trap_on_grow_failure(true)",
    "store.limiter(",
]
for token in required_limit_tokens:
    if token not in arm:
        failures.append(f"MISSING_LIMIT:{token}")

# This tranche must stay raw-Wasm-only: no ambient WASI imports.
if "wasmtime_wasi" in arm or "WasiCtx" in arm or "wasi:" in arm:
    failures.append("AMBIENT_WASI_INTRODUCED")

if failures:
    print("SYM-WASM-000A STATIC QUALIFICATION: RED")
    for failure in failures:
        print(f" - {failure}")
    sys.exit(1)

print("SYM-WASM-000A STATIC QUALIFICATION: PASS")
print("claim=ActionIR::WasmSandbox source boundary is sandbox-rooted and resource-bounded")
print("does_not_imply=runtime adversarial qualification, WASI safety, artifact authenticity, Spore component authority")
