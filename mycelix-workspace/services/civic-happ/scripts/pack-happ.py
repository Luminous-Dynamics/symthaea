#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Pack civic-happ.happ without the wasm-bindgen externref transform.

hc 0.6.x adds wasm-bindgen externref transform imports during DNA packing,
but the holochain 0.6.x conductor doesn't provide those imports, causing
ModuleBuild errors. This script packs the hApp manually, preserving the
original WASM files unchanged.
"""
import gzip
import io
import sys
from pathlib import Path

try:
    import msgpack
except ImportError:
    print("Error: msgpack not installed. Run: pip install msgpack", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
WASM_DIR = ROOT / "target" / "wasm32-unknown-unknown" / "release"

ZOMES = {
    "integrity": [
        "civic_knowledge_integrity",
        "agent_reputation_integrity",
    ],
    "coordinator": [
        ("civic_knowledge", "civic_knowledge_coordinator", "civic_knowledge_integrity"),
        ("agent_reputation", "agent_reputation_coordinator", "agent_reputation_integrity"),
    ],
}


def main():
    # Read WASMs
    wasms = {}
    for name in ZOMES["integrity"]:
        path = WASM_DIR / f"{name}.wasm"
        if not path.exists():
            print(f"Error: {path} not found. Run cargo build first.", file=sys.stderr)
            sys.exit(1)
        wasms[f"{name}.wasm"] = path.read_bytes()

    for zome_name, crate_name, dep in ZOMES["coordinator"]:
        path = WASM_DIR / f"{crate_name}.wasm"
        if not path.exists():
            print(f"Error: {path} not found. Run cargo build first.", file=sys.stderr)
            sys.exit(1)
        wasms[f"{crate_name}.wasm"] = path.read_bytes()

    # DNA manifest
    dna_manifest = {
        "manifest_version": "0",
        "name": "civic",
        "integrity": {
            "network_seed": None,
            "properties": None,
            "zomes": [
                {"name": n, "hash": None, "path": f"{n}.wasm", "dependencies": None}
                for n in ZOMES["integrity"]
            ],
        },
        "coordinator": {
            "zomes": [
                {
                    "name": zome_name,
                    "hash": None,
                    "path": f"{crate_name}.wasm",
                    "dependencies": [{"name": dep}],
                }
                for zome_name, crate_name, dep in ZOMES["coordinator"]
            ]
        },
    }

    dna_bytes = gzip.compress(msgpack.packb({"manifest": dna_manifest, "resources": wasms}))

    # hApp manifest
    happ_manifest = {
        "manifest_version": "0",
        "name": "civic-happ",
        "description": "Civic AI hApp - Knowledge storage and agent reputation for Symthaea",
        "roles": [
            {
                "name": "civic",
                "provisioning": {"strategy": "create", "deferred": False},
                "dna": {
                    "path": "civic.dna",
                    "modifiers": {"network_seed": None, "properties": None},
                    "installed_hash": None,
                    "clone_limit": 0,
                },
            }
        ],
        "allow_deferred_memproofs": False,
    }

    happ_bytes = gzip.compress(
        msgpack.packb({"manifest": happ_manifest, "resources": {"civic.dna": dna_bytes}})
    )

    out = ROOT / "civic-happ.happ"
    out.write_bytes(happ_bytes)
    print(f"Wrote {out} ({len(happ_bytes)} bytes)")


if __name__ == "__main__":
    main()
