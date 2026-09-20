#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Emit raw OQMD v1.7 Fe/Co/Zr standard-fit records as canonical-key NDJSON.

This adapter intentionally performs *source interpretation only*. It follows qmpy
1.4's public formation-energy list semantics by selecting fit="standard" and the
full non-empty Fe/Co/Zr composition space. It does not deduplicate, threshold by
energy/stability/property, canonicalize structures, or decide benchmark authority.
Those remain Rust-side responsibilities.
"""

import json
import logging
import logging.handlers
import math
import os
import sys

# qmpy 1.4's Django settings indexes this variable directly. The qualified local
# experiment uses a passwordless Unix-socket-only account, so bind the required
# setting explicitly rather than omitting it or introducing a secret-bearing input.
os.environ.setdefault("qmdb_v1_1_pswd", "")

# qmpy 1.4 constructs qmpy/logs/qmpy.log at import time. A reproducible package
# closure may be read-only, and diagnostics must not force a mutable source tree.
# Suppress only that file-handler construction during qmpy import, then restore the
# standard library class immediately. This changes logging side effects only; qmpy
# model/query semantics remain the reviewed source implementation.
_original_watched_file_handler = logging.handlers.WatchedFileHandler
logging.handlers.WatchedFileHandler = lambda *args, **kwargs: logging.NullHandler()
try:
    from qmpy.materials.formation_energy import FormationEnergy
finally:
    logging.handlers.WatchedFileHandler = _original_watched_file_handler

ALLOWED = {"Co", "Fe", "Zr"}
SCHEMA_VERSION = 1


def finite_text(value):
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite source numeric value")
    return repr(number)


def maybe_icsd_id(entry):
    path = entry.path or ""
    if "icsd" not in path.lower():
        return None
    leaf = path.rstrip("/").split("/")[-1]
    try:
        return str(int(leaf))
    except (TypeError, ValueError):
        return None


def structure_payload(calculation):
    structure = calculation.output if calculation is not None else None
    if structure is None:
        return None

    lattice = [
        [finite_text(structure.x1), finite_text(structure.x2), finite_text(structure.x3)],
        [finite_text(structure.y1), finite_text(structure.y2), finite_text(structure.y3)],
        [finite_text(structure.z1), finite_text(structure.z2), finite_text(structure.z3)],
    ]

    sites = []
    # Structure.sites is qmpy's source-native site abstraction. It may group
    # multiple partially occupied atoms at one fractional coordinate.
    for site in structure.sites:
        occupants = []
        for atom in site.atoms:
            occupants.append(
                {
                    "element": str(atom.element_id),
                    "occupancy": finite_text(atom.occupancy),
                    "oxidation_state": atom.ox,
                }
            )
        occupants.sort(
            key=lambda item: (
                item["element"],
                -10**9 if item["oxidation_state"] is None else int(item["oxidation_state"]),
                item["occupancy"],
            )
        )
        sites.append(
            {
                "fractional_coordinate": [
                    finite_text(site.x),
                    finite_text(site.y),
                    finite_text(site.z),
                ],
                "occupants": occupants,
            }
        )

    spacegroup = None
    if structure.spacegroup is not None:
        spacegroup = str(structure.spacegroup.hm)

    return {
        "lattice": lattice,
        "sites": sites,
        "spacegroup": spacegroup,
    }


def main():
    # qmpy's own public FormationEnergyList starts from fit="standard". Its
    # FormationEnergy.search helper then implements the source-native composition
    # space restriction without energy/property filtering.
    records = FormationEnergy.search(["Co", "Fe", "Zr"], fit="standard")
    records = records.order_by("entry_id", "id")

    for formation in records.iterator(chunk_size=512):
        if formation.entry is None or formation.composition is None:
            raise ValueError("standard formation-energy row lacks entry/composition")
        if formation.calculation is None:
            raise ValueError("standard formation-energy row lacks calculation")

        elements = sorted(str(element.symbol) for element in formation.composition.element_set.all())
        if not elements or any(element not in ALLOWED for element in elements):
            raise ValueError("qmpy composition search emitted an out-of-scope element set")

        duplicate_id = formation.entry.duplicate_of_id
        if duplicate_id == formation.entry_id:
            duplicate_id = None

        prototype = None
        if formation.entry.prototype is not None:
            prototype = str(formation.entry.prototype.name)

        row = {
            "schema_version": SCHEMA_VERSION,
            "formation_energy_id": int(formation.id),
            "entry_id": int(formation.entry_id),
            "duplicate_entry_id": None if duplicate_id is None else int(duplicate_id),
            "name": str(formation.composition.formula),
            "composition_formula": str(formation.composition.formula),
            "element_set": elements,
            "prototype": prototype,
            "natoms": int(formation.entry.natoms),
            "ntypes": int(formation.composition.ntypes),
            "delta_e_ev_atom": finite_text(formation.delta_e),
            "stability_ev_atom": finite_text(formation.stability),
            "band_gap_ev": finite_text(formation.calculation.band_gap),
            "calculation_id": int(formation.calculation_id),
            "calculation_label": None
            if formation.calculation.label is None
            else str(formation.calculation.label),
            "fit": str(formation.fit_id),
            "icsd_id": maybe_icsd_id(formation.entry),
            "structure": structure_payload(formation.calculation),
        }
        sys.stdout.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
