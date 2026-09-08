#!/usr/bin/env python3
"""Derive Symthaea ordered periodic-cell receipts from AiiDA StructureData.

This is a supplemental exporter for a completed `matter_aiida_qe_leaf_export.py`
capsule. It verifies that the leaf capsule and the live AiiDA CalcJob identify the
same process, then derives exact bit-level periodic-cell receipts from public
`aiida.orm.StructureData` values.

Supported in v1:
  * pw.x SCF: input structure
  * pw.x relax/vc-relax: input structure + output_structure

`ph.x` is intentionally refused: its structure is inherited through a parent
periodic calculation and needs an explicit provenance traversal rather than a
free-standing structure argument.

The exporter accepts only fully periodic, fully occupied, single-species kinds.
It does not canonicalize primitive/conventional-cell equivalence, space groups,
disorder, or partial occupancy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ADAPTER_NAME = "symthaea-aiida-structure-export"
ADAPTER_VERSION = "1"
LEAF_ADAPTER_NAME = "symthaea-aiida-qe-leaf-export"
LEAF_WIRE_SCHEMA = "symthaea.matter.crystal-solver-execution/v1"
ARTIFACT_SCHEMA = "symthaea.matter.aiida-ordered-cell-artifact/v1"
RECEIPT_SCHEMA = "symthaea.matter.ordered-periodic-cell-receipts/v1"
MAX_JSON_BYTES = 8 * 1024 * 1024

TASK_PERIODIC = "periodic_electronic_structure"
TASK_RELAX = "structural_relaxation"
TASK_LATTICE = "lattice_dynamics"

# IUPAC element symbols 1..118. AiiDA StructureData stores symbols, not atomic
# numbers; keeping this table local avoids depending on ASE/pymatgen internals.
ELEMENTS = (
    "",
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)
ATOMIC_NUMBER = {symbol: number for number, symbol in enumerate(ELEMENTS) if symbol}


class AdapterError(RuntimeError):
    """Fail-closed structure exporter error."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any) -> str:
    payload = _json_bytes(value)
    if len(payload) > MAX_JSON_BYTES:
        raise AdapterError(f"JSON artifact exceeds {MAX_JSON_BYTES} bytes: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = _sha256(payload)
    if _sha256(path.read_bytes()) != digest:
        raise AdapterError(f"persisted JSON bytes failed digest round-trip: {path}")
    return digest


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise AdapterError(f"missing JSON artifact: {path}")
    if path.stat().st_size > MAX_JSON_BYTES:
        raise AdapterError(f"JSON input exceeds {MAX_JSON_BYTES} bytes: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdapterError(f"cannot read JSON {path}: {error}") from error


def _prepare_empty_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise AdapterError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _yyyymmdd(dt: datetime) -> int:
    if dt.tzinfo is None:
        raise AdapterError("AiiDA datetime is timezone-naive")
    value = dt.astimezone(timezone.utc)
    return value.year * 10_000 + value.month * 100 + value.day


def _f64_bits(value: float) -> int:
    value = float(value)
    if not math.isfinite(value):
        raise AdapterError("structure contains a non-finite floating-point value")
    if value == 0.0:
        value = 0.0
    return struct.unpack(">Q", struct.pack(">d", value))[0]


def _det3(m: list[list[float]]) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _inverse3(m: list[list[float]]) -> list[list[float]]:
    det = _det3(m)
    if not math.isfinite(det) or det == 0.0:
        raise AdapterError("AiiDA StructureData cell is singular or invalid")
    inv_det = 1.0 / det
    return [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ]


def _fractional(cartesian: Iterable[float], inverse_cell: list[list[float]]) -> list[float]:
    r = [float(value) for value in cartesian]
    if len(r) != 3 or not all(math.isfinite(value) for value in r):
        raise AdapterError("AiiDA site position is not a finite three-vector")
    # AiiDA stores direct lattice vectors as rows: r = f * cell.
    return [sum(r[i] * inverse_cell[i][j] for i in range(3)) % 1.0 for j in range(3)]


def _kind_atomic_number(kind: Any) -> int:
    symbols = tuple(str(symbol) for symbol in getattr(kind, "symbols", ()))
    weights_raw = getattr(kind, "weights", None)
    weights = tuple(float(value) for value in weights_raw) if weights_raw is not None else (1.0,) * len(symbols)
    if len(symbols) != 1 or len(weights) != 1 or not math.isclose(weights[0], 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise AdapterError("ordered-cell v1 refuses alloy, vacancy, partial, or disordered AiiDA kinds")
    try:
        return ATOMIC_NUMBER[symbols[0]]
    except KeyError as error:
        raise AdapterError(f"unknown chemical symbol in AiiDA kind: {symbols[0]}") from error


def _derive_ordered_cell(structure: Any, source_role: str) -> tuple[dict[str, Any], dict[str, Any]]:
    pbc = tuple(bool(value) for value in getattr(structure, "pbc", ()))
    if pbc != (True, True, True):
        raise AdapterError(f"ordered-cell v1 requires fully periodic StructureData, got pbc={pbc}")

    cell = [[float(value) for value in row] for row in structure.cell]
    if len(cell) != 3 or any(len(row) != 3 for row in cell):
        raise AdapterError("AiiDA StructureData cell is not 3x3")
    if not all(math.isfinite(value) for row in cell for value in row):
        raise AdapterError("AiiDA StructureData cell contains non-finite values")
    inverse = _inverse3(cell)

    kinds = {str(kind.name): kind for kind in structure.kinds}
    if len(kinds) != len(structure.kinds):
        raise AdapterError("AiiDA StructureData contains duplicate kind names")

    sites: list[dict[str, Any]] = []
    occupied_coordinates: set[tuple[int, int, int]] = set()
    for site in structure.sites:
        kind_name = str(site.kind_name)
        if kind_name not in kinds:
            raise AdapterError(f"site references unknown AiiDA kind: {kind_name}")
        atomic_number = _kind_atomic_number(kinds[kind_name])
        fractional = _fractional(site.position, inverse)
        fractional_bits = tuple(_f64_bits(value) for value in fractional)
        if fractional_bits in occupied_coordinates:
            raise AdapterError("ordered-cell v1 refuses coincident fully occupied periodic sites")
        occupied_coordinates.add(fractional_bits)
        sites.append({"atomic_number": atomic_number, "fractional_bits": list(fractional_bits)})
    if not sites:
        raise AdapterError("AiiDA StructureData contains no sites")
    sites.sort(key=lambda site: (site["atomic_number"], tuple(site["fractional_bits"])))

    lattice_bits = [[_f64_bits(value) for value in row] for row in cell]
    artifact = {
        "schema_version": ARTIFACT_SCHEMA,
        "source_system": "AiiDA",
        "source_structure_uuid": str(structure.uuid),
        "source_role": source_role,
        "pbc": [True, True, True],
        "lattice_vectors_angstrom_bits": lattice_bits,
        "sites": sites,
        "interpretation": (
            "derived from public AiiDA StructureData values; ordered-cell v1 does not prove "
            "primitive/conventional-cell, origin-shift, symmetry, disorder, or partial-occupancy equivalence"
        ),
    }
    receipt_values = {
        "lattice_vectors_angstrom_bits": lattice_bits,
        "sites": sites,
    }
    return artifact, receipt_values


def _reference(evidence_id: str, digest: str, locator: str, subject: str, date: int) -> dict[str, Any]:
    return {
        "id": evidence_id,
        "role": "ArtifactContent",
        "content_sha256": digest,
        "locator": locator,
        "subject": subject,
        "claimed_utc_date": date,
        "issuer": ADAPTER_NAME,
    }


def _load_leaf_capsule(root: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    manifest = _read_json(root / "export-manifest.json")
    execution = _read_json(root / "execution.json")
    bundle = _read_json(root / "evidence-bundle.json")
    if not isinstance(manifest, dict) or not isinstance(execution, dict) or not isinstance(bundle, dict):
        raise AdapterError("leaf capsule JSON roots must be objects")
    adapter = manifest.get("adapter")
    if not isinstance(adapter, dict) or adapter.get("name") != LEAF_ADAPTER_NAME:
        raise AdapterError("leaf export manifest is not from the canonical AiiDA/QE leaf adapter")
    if execution.get("schema_version") != LEAF_WIRE_SCHEMA:
        raise AdapterError("leaf execution has unsupported or missing wire schema")
    source = manifest.get("source")
    if not isinstance(source, dict) or source.get("system") != "AiiDA" or not source.get("calcjob_uuid"):
        raise AdapterError("leaf export manifest has malformed AiiDA source identity")
    references = bundle.get("references")
    if not isinstance(references, list) or not references:
        raise AdapterError("leaf evidence bundle has no references")
    ids: set[str] = set()
    for reference in references:
        if not isinstance(reference, dict) or not str(reference.get("id", "")):
            raise AdapterError("leaf evidence bundle contains a malformed reference")
        evidence_id = str(reference["id"])
        if evidence_id in ids:
            raise AdapterError(f"leaf evidence bundle contains duplicate ID: {evidence_id}")
        ids.add(evidence_id)
    return manifest, execution, references


def _export(args: argparse.Namespace) -> None:
    from aiida import load_profile, orm  # public AiiDA API; deliberately lazy

    load_profile(args.profile) if args.profile else load_profile()
    leaf_root = Path(args.leaf_export).resolve()
    manifest, execution, existing_references = _load_leaf_capsule(leaf_root)
    source = manifest["source"]
    calcjob_uuid = str(source["calcjob_uuid"])

    node = orm.load_node(args.node)
    if not isinstance(node, orm.CalcJobNode):
        raise AdapterError("structure exporter requires the same AiiDA CalcJobNode as the leaf capsule")
    if str(node.uuid) != calcjob_uuid:
        raise AdapterError("live CalcJob UUID does not match the leaf export capsule")
    if not bool(node.is_finished_ok) or not bool(getattr(node, "is_sealed", False)):
        raise AdapterError("CalcJob must be sealed and finished successfully")

    task = str(execution.get("task", ""))
    if task == TASK_LATTICE:
        raise AdapterError("ph.x structure lineage must be inherited from an explicit parent periodic calculation")
    if task not in (TASK_PERIODIC, TASK_RELAX):
        raise AdapterError(f"unsupported leaf task for structure export: {task}")

    input_structure = getattr(node.inputs, "structure", None)
    if not isinstance(input_structure, orm.StructureData):
        raise AdapterError("pw.x leaf does not expose StructureData input")
    structures: list[tuple[str, Any]] = [("input", input_structure)]
    if task == TASK_RELAX:
        output_structure = getattr(node.outputs, "output_structure", None)
        if not isinstance(output_structure, orm.StructureData):
            raise AdapterError("relax/vc-relax leaf lacks StructureData output_structure")
        structures.append(("output", output_structure))

    out = Path(args.output).resolve()
    _prepare_empty_dir(out)
    new_references: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    seen_ids = {str(reference["id"]) for reference in existing_references}

    for role, structure in structures:
        artifact, values = _derive_ordered_cell(structure, role)
        artifact_path = out / "artifacts" / f"structure-{role}.json"
        digest = _write_json(artifact_path, artifact)
        evidence_id = f"aiida:{node.uuid}:structure:{role}:{digest[:16]}"
        if evidence_id in seen_ids:
            raise AdapterError(f"structure evidence ID collides with leaf capsule: {evidence_id}")
        seen_ids.add(evidence_id)
        new_references.append(
            _reference(
                evidence_id,
                digest,
                f"symthaea-export://{artifact_path.relative_to(out).as_posix()}",
                f"AiiDA StructureData {role} for CalcJob {node.uuid}",
                _yyyymmdd(structure.ctime),
            )
        )
        receipts.append(
            {
                "source_role": role,
                "source_structure_uuid": str(structure.uuid),
                "receipt": {
                    "structure_artifact_id": evidence_id,
                    **values,
                },
            }
        )

    _write_json(
        out / "ordered-cell-receipts.json",
        {
            "schema_version": RECEIPT_SCHEMA,
            "source_calcjob_uuid": str(node.uuid),
            "task": task,
            "receipts": receipts,
            "interpretation": "AiiDA-derived ordered-cell values; crystallographic equivalence remains limited to Symthaea ordered-cell v1",
        },
    )
    _write_json(out / "evidence-bundle.augmented.json", {"references": [*existing_references, *new_references]})
    _write_json(
        out / "structure-export-manifest.json",
        {
            "adapter": {"name": ADAPTER_NAME, "version": ADAPTER_VERSION},
            "source": {"system": "AiiDA", "calcjob_uuid": str(node.uuid), "task": task},
            "leaf_export_manifest_sha256": _sha256((leaf_root / "export-manifest.json").read_bytes()),
            "leaf_execution_sha256": _sha256((leaf_root / "execution.json").read_bytes()),
            "leaf_evidence_bundle_sha256": _sha256((leaf_root / "evidence-bundle.json").read_bytes()),
            "ordered_cell_receipts": "ordered-cell-receipts.json",
            "augmented_evidence_bundle": "evidence-bundle.augmented.json",
            "limitations": [
                "AiiDA StructureData values are trusted through the public ORM object; the Rust gate does not independently parse AiiDA storage",
                "ordered-cell v1 does not establish basis-transform, primitive/conventional-cell, origin-shift, symmetry, disorder, or partial-occupancy equivalence",
                "ph.x structure lineage is intentionally not inferred in v1",
            ],
        },
    )
    print(out)


def _self_test() -> None:
    class Kind:
        def __init__(self, name: str, symbol: str, weight: float = 1.0):
            self.name = name
            self.symbols = (symbol,)
            self.weights = (weight,)

    class Site:
        def __init__(self, kind_name: str, position: tuple[float, float, float]):
            self.kind_name = kind_name
            self.position = position

    class Structure:
        uuid = "fixture-structure"
        pbc = (True, True, True)
        cell = [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]]
        kinds = [Kind("Si", "Si")]
        sites = [Site("Si", (1.0, 1.0, 2.5))]

    artifact, receipt = _derive_ordered_cell(Structure(), "input")
    assert artifact["source_role"] == "input"
    assert receipt["sites"][0]["atomic_number"] == 14
    expected = [0.5, 0.25, 0.5]
    actual = [struct.unpack(">d", struct.pack(">Q", bits))[0] for bits in receipt["sites"][0]["fractional_bits"]]
    assert all(abs(left - right) < 1e-15 for left, right in zip(actual, expected))

    class Shifted(Structure):
        sites = [Site("Si", (5.0, -3.0, 2.5))]

    _, shifted = _derive_ordered_cell(Shifted(), "input")
    assert shifted["sites"][0]["fractional_bits"] == receipt["sites"][0]["fractional_bits"]

    class Skewed(Structure):
        cell = [[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [0.5, 0.5, 4.0]]
        # fractional (0.2, 0.3, 0.4) -> Cartesian (0.9, 1.1, 1.6)
        sites = [Site("Si", (0.9, 1.1, 1.6))]

    _, skewed = _derive_ordered_cell(Skewed(), "input")
    skewed_actual = [
        struct.unpack(">d", struct.pack(">Q", bits))[0]
        for bits in skewed["sites"][0]["fractional_bits"]
    ]
    assert all(abs(left - right) < 1e-14 for left, right in zip(skewed_actual, (0.2, 0.3, 0.4)))

    class Alloy(Structure):
        kinds = [Kind("mix", "Si", 0.5)]
        sites = [Site("mix", (0.0, 0.0, 0.0))]

    try:
        _derive_ordered_cell(Alloy(), "input")
    except AdapterError:
        pass
    else:
        raise AssertionError("partial/disordered occupancy must fail closed")
    print("AiiDA ordered-cell structure exporter self-test: PASS")


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="derive ordered-cell receipts from a matching AiiDA leaf capsule")
    export.add_argument("--profile")
    export.add_argument("--node", required=True)
    export.add_argument("--leaf-export", required=True)
    export.add_argument("--output", required=True)
    sub.add_parser("self-test", help="run dependency-free ordered-cell derivation invariants")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _arg_parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "export":
            _export(args)
        elif args.command == "self-test":
            _self_test()
        else:
            raise AdapterError(f"unsupported command: {args.command}")
        return 0
    except AdapterError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
