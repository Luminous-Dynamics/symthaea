#!/usr/bin/env python3
"""Generate Rust `shells_for_element` match arms for STO-3G, Z=1-54 (H-Xe).

Source data: ../src/basis/reference/sto3g_bse_h_xe.json, fetched via:
  curl -sL "https://www.basissetexchange.org/api/basis/sto-3g/format/json/" \
    -o src/basis/reference/sto3g_bse_h_xe.json
Fetched directly with curl (not WebFetch, whose AI-summarization step was found
to corrupt dense numeric tables during Phase A.7 -- see the project memory note
`feedback_webfetch_summarization_corrupts_dense_tables.md`).

Each BSE electron_shell's `angular_momentum` list length determines how many
`ShellData` entries it expands to, all sharing that shell's `exponents`:
  length 1 -> one ShellType::S entry (coefficients[0])
  length 2 -> ShellType::S (coefficients[0]) + ShellType::P (coefficients[1])
  length 3 -> ShellType::S + ShellType::P + ShellType::D (coefficients[0/1/2])

This mechanical, per-shell rule is required (not just convenient) because shell
*shapes* vary across blocks in ways a hand-picked small set of per-block
patterns can't cover: H-He is S-only; Li-Ar is S,SP,SP; K-Ca adds a 4th SP
shell; Sc-Zn adds a trailing standalone D shell; Ga-Kr's outer shell is a
combined SPD block; Rb-Sr is K-Ca's shape plus an SPD block; Y-Cd adds a
trailing standalone D on top of that; In-Xe has two SPD blocks. At least 8
distinct shapes across H-Xe, verified against the raw JSON before writing this
script.

Usage: python3 generate_sto3g.py > /tmp/sto3g_generated.rs
Then hand-splice the output into src/basis/sto3g.rs's shells_for_element match
(reviewed before committing, not blindly piped into the source file).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(HERE, "..", "src", "basis", "reference", "sto3g_bse_h_xe.json")

ELEMENT_NAMES = {
    1: "Hydrogen", 2: "Helium", 3: "Lithium", 4: "Beryllium", 5: "Boron",
    6: "Carbon", 7: "Nitrogen", 8: "Oxygen", 9: "Fluorine", 10: "Neon",
    11: "Sodium", 12: "Magnesium", 13: "Aluminum", 14: "Silicon", 15: "Phosphorus",
    16: "Sulfur", 17: "Chlorine", 18: "Argon", 19: "Potassium", 20: "Calcium",
    21: "Scandium", 22: "Titanium", 23: "Vanadium", 24: "Chromium", 25: "Manganese",
    26: "Iron", 27: "Cobalt", 28: "Nickel", 29: "Copper", 30: "Zinc",
    31: "Gallium", 32: "Germanium", 33: "Arsenic", 34: "Selenium", 35: "Bromine",
    36: "Krypton", 37: "Rubidium", 38: "Strontium", 39: "Yttrium", 40: "Zirconium",
    41: "Niobium", 42: "Molybdenum", 43: "Technetium", 44: "Ruthenium", 45: "Rhodium",
    46: "Palladium", 47: "Silver", 48: "Cadmium", 49: "Indium", 50: "Tin",
    51: "Antimony", 52: "Tellurium", 53: "Iodine", 54: "Xenon",
}

TYPE_MAP = {0: "S", 1: "P", 2: "D"}


def fmt(x):
    return repr(float(x))


def emit_element(z, el):
    lines = []
    name = ELEMENT_NAMES[z]
    header = f"        // ── {name} (Z={z}) "
    header += "─" * max(1, 64 - len(header))
    lines.append(header)
    lines.append(f"        {z} => vec![")
    for shell in el["electron_shells"]:
        am = shell["angular_momentum"]
        exps = shell["exponents"]
        coeffs = shell["coefficients"]
        exp_lits = ", ".join(fmt(e) for e in exps)
        for idx, a in enumerate(am):
            coeff_lits = ", ".join(fmt(c) for c in coeffs[idx])
            lines.append("            ShellData {")
            lines.append(f"                shell_type: ShellType::{TYPE_MAP[a]},")
            lines.append(f"                exponents: [{exp_lits}],")
            lines.append(f"                coefficients: [{coeff_lits}],")
            lines.append("            },")
    lines.append("        ],")
    lines.append("")
    return lines


def main():
    with open(JSON_PATH) as f:
        d = json.load(f)
    out = []
    for z in range(1, 55):
        el = d["elements"][str(z)]
        out.extend(emit_element(z, el))
    print("\n".join(out))


if __name__ == "__main__":
    main()
