#!/usr/bin/env python3
"""Generate Rust element-metadata tables for all 118 real elements (Z=1-118).

Source data: ../src/element_data/reference/periodic_table.json, fetched via:
  curl -sL "https://raw.githubusercontent.com/Bowserinator/Periodic-Table-JSON/master/PeriodicTableJSON.json" \
    -o src/element_data/reference/periodic_table.json
Fetched directly with curl (not WebFetch, whose AI-summarization step was found
to corrupt dense numeric tables during Phase A.7 -- see the project memory note
`feedback_webfetch_summarization_corrupts_dense_tables.md`).

The source dataset has 119 entries; Z=119 (Ununennium) is a hypothetical,
undiscovered element with a placeholder/predicted mass and is excluded here --
this module covers strictly the 118 confirmed real elements.

Emits two Rust tables to stdout:
  1. `ELEMENT_METADATA`: a static array of `ElementMetadata` for `element_data.rs`.
  2. `ELEMENT_SYMBOLS_118` / a symbol_to_z match body, for splicing into
     `molecule.rs` (which currently hand-maintains a much shorter Z=1-20 table).

Usage: python3 generate_element_data.py > /tmp/element_data_generated.rs
Reviewed before splicing into source, not blindly piped in.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(HERE, "..", "src", "element_data", "reference", "periodic_table.json")

BLOCK_MAP = {"s": "S", "p": "P", "d": "D", "f": "F"}


def rust_float(x):
    return repr(float(x))


def main():
    with open(JSON_PATH) as f:
        d = json.load(f)
    elements = {e["number"]: e for e in d["elements"] if 1 <= e["number"] <= 118}
    assert len(elements) == 118, f"expected 118 elements, got {len(elements)}"

    print("// ── ElementMetadata table (element_data.rs) ──────────────────────")
    print("pub(crate) static ELEMENT_METADATA: [ElementMetadata; 118] = [")
    for z in range(1, 119):
        e = elements[z]
        symbol = e["symbol"]
        mass = rust_float(e["atomic_mass"])
        period = e["period"]
        block = BLOCK_MAP[e["block"]]
        print(
            f'    ElementMetadata {{ atomic_number: {z}, symbol: "{symbol}", '
            f"atomic_mass: {mass}, period: {period}, block: Block::{block} }},"
        )
    print("];")
    print()

    print("// ── ELEMENT_SYMBOLS (molecule.rs) ─────────────────────────────────")
    print('const ELEMENT_SYMBOLS: &[&str] = &[')
    print('    "",')
    for z in range(1, 119):
        print(f'    "{elements[z]["symbol"]}",')
    print("];")
    print()

    print("// ── symbol_to_z (molecule.rs) ──────────────────────────────────────")
    print("fn symbol_to_z(symbol: &str) -> Option<u8> {")
    print("    match symbol.to_uppercase().as_str() {")
    for z in range(1, 119):
        sym_upper = elements[z]["symbol"].upper()
        print(f'        "{sym_upper}" => Some({z}),')
    print("        _ => None,")
    print("    }")
    print("}")


if __name__ == "__main__":
    main()
