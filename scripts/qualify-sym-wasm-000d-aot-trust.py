#!/usr/bin/env python3
"""SYM-WASM-000D static AOT trust-boundary canary.

This intentionally does not execute or deserialize native Wasmtime artifacts.
It freezes source-level trust requirements that must hold before unsafe AOT
loading can be considered for qualification.
"""
from pathlib import Path
import sys

path = Path("crates/domains/symthaea-broca-tools/src/wasm_architect.rs")
text = path.read_text(encoding="utf-8")

load_start = text.find("fn load_verified_module(")
store_start = text.find("fn sandboxed_store(", load_start)
if load_start < 0 or store_start < 0:
    raise SystemExit("SYM-WASM-000D: load_verified_module boundary not found")
load = text[load_start:store_start]

compile_start = text.find("pub fn compile_to_wasm(")
fuel_start = text.find("const WASM_FUEL_BUDGET", compile_start)
if compile_start < 0 or fuel_start < 0:
    raise SystemExit("SYM-WASM-000D: compile_to_wasm boundary not found")
compile_body = text[compile_start:fuel_start]

failures: list[str] = []

# Embedded signer keys can prove self-consistency, not trust. Before unsafe
# deserialize there must be an explicit comparison/lookup against trusted host
# state rather than only signed.public_key.
if "verify_signature(&signed.bytes, &signed.signature, &signed.public_key)" in load:
    failures.append("EMBEDDED_KEY_SELF_AUTHORIZES_SIGNATURE")
if "Module::deserialize" in load and not any(
    token in load
    for token in (
        "trusted_public_key",
        "trusted_signer",
        "allowed_signer",
        "keypair.public_key",
        "trust_policy",
    )
):
    failures.append("UNSAFE_DESERIALIZE_WITHOUT_TRUSTED_SIGNER_PIN")

# serialized AOT bytes are not canonical raw Wasm source. A compatibility
# fallback must use separately retained raw bytes or fail closed.
if "Module::new(engine, &signed.bytes)" in load or "Module::from_binary(engine, &signed.bytes)" in load:
    failures.append("AOT_BYTES_REUSED_AS_RAW_WASM_FALLBACK")

# Portable/cache subject identity must be cryptographic; DefaultHasher is not a
# stable collision-resistant content identity.
if "DefaultHasher" in text and "fn compute_hash" in text:
    failures.append("NON_CRYPTOGRAPHIC_CACHE_IDENTITY")

# An API named compile_to_wasm must not ambiguously return either raw Wasm or a
# bincode envelope containing native AOT bytes.
if "return Ok(encoded);" in compile_body and "Ok(wasm_bytes)" in compile_body:
    failures.append("PORTABLE_AND_AOT_BYTES_SHARE_ONE_VEC_API")

if failures:
    print("SYM-WASM-000D AOT TRUST QUALIFICATION: RED")
    for failure in failures:
        print(f" - {failure}")
    sys.exit(1)

print("SYM-WASM-000D AOT TRUST QUALIFICATION: PASS")
print("claim=source-level AOT trust/identity boundary only")
print("does_not_imply=runtime unsafe-deserialize qualification, publisher authority, portable AOT safety")
