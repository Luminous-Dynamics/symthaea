#!/usr/bin/env python3
"""Additional static canaries for the SYM-WASM-000A repair candidate.

The parent #4300 canary is frozen as the RED baseline. This child deliberately
adds stricter requirements without rewriting that historical subject.
"""
from pathlib import Path
import sys

text = Path("src/action/mod.rs").read_text(encoding="utf-8")
anchor = '"🧪 WASM Sandbox: Simulating verification of module {:?}::{}()"'
pos = text.find(anchor)
if pos < 0:
    raise SystemExit("SYM-WASM-000A-CANDIDATE: simulation anchor not found")
start = text.rfind("ActionIR::WasmSandbox {", 0, pos)
end = text.find("ActionIR::RunCommand", pos)
if start < 0 or end < 0:
    raise SystemExit("SYM-WASM-000A-CANDIDATE: execution arm boundaries not found")
arm = text[start:end]

required = {
    "SANDBOX_ROOTED_PATH": "sandbox.validate(module_path)",
    "MODULE_SIZE_LIMIT": "WASM_MODULE_MAX_BYTES",
    "BOUNDED_READ": ".take(WASM_MODULE_MAX_BYTES + 1)",
    "BINARY_ONLY_ADMISSION": "Module::from_binary",
    "FUEL_ENABLED": "config.consume_fuel(true)",
    "FUEL_SET": ".set_fuel(WASM_FUEL_BUDGET)",
    "MEMORY_SIZE_LIMIT": ".memory_size(WASM_MEMORY_LIMIT_BYTES)",
    "TABLE_ELEMENT_LIMIT": ".table_elements(WASM_TABLE_ELEMENT_LIMIT)",
    "INSTANCE_LIMIT": ".instances(1)",
    "TABLE_COUNT_LIMIT": ".tables(4)",
    "MEMORY_COUNT_LIMIT": ".memories(1)",
    "GROWTH_TRAPS": ".trap_on_grow_failure(true)",
    "STORE_LIMITER": "store.limiter(",
    "ZERO_IMPORTS": "Instance::new(&mut store, &module, &[])",
}

failures: list[str] = []
for name, token in required.items():
    if token not in arm:
        failures.append(f"MISSING_{name}")

forbidden = {
    "DEFAULT_ENGINE": "Engine::default()",
    "DIRECT_FILE_COMPILATION": "Module::from_file",
    "TEXT_WAT_ADMISSION": "Module::new",
    "WASI_HOST": "wasmtime_wasi",
    "WASI_CONTEXT": "WasiCtx",
}
for name, token in forbidden.items():
    if token in arm:
        failures.append(f"FORBIDDEN_{name}")

if failures:
    print("SYM-WASM-000A CANDIDATE QUALIFICATION: RED")
    for failure in failures:
        print(f" - {failure}")
    sys.exit(1)

print("SYM-WASM-000A CANDIDATE QUALIFICATION: PASS")
print("claim=bounded raw-Wasm source/admission boundary only")
print("module_max_bytes=16777216")
print("fuel=50000000")
print("memory_max_bytes=67108864")
print("table_elements=65536")
print("instances=1")
print("tables=4")
print("memories=1")
print("ambient_wasi=false")
print("does_not_imply=runtime adversarial qualification, compile hard-real-time bound, artifact authenticity, Spore authority")
