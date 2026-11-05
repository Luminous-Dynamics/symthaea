#!/usr/bin/env python3
"""Cross-engine CGAL mesh acceptance audit for engineering-fixtures-v0.3.

STEP import and tessellation use the same CadQuery/OCP family as fixture
creation. The resulting STL is then sent through OpenSCAD's CGAL
polyhedron/Boolean path using a geometrically disjoint subtractor, and the
resulting STL is checked with Trimesh.

This is a mesh-level cross-engine check, not STEP B-rep kernel
interoperability, native CAD history validation, or physical validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

import cadquery as cq
import trimesh
import OCP
from cadquery import importers

ROOT = Path(__file__).resolve().parent
EXPECTED_OPENSCAD_VERSION = "2021.01"
LINEAR_DEFLECTION_MM = 0.01
ANGULAR_DEFLECTION_RAD = 0.1
MAX_RELATIVE_BREP_TO_CGAL_MESH_VOLUME_ERROR = 6e-4
MAX_RELATIVE_INPUT_TO_CGAL_VOLUME_ERROR = 1e-5
MAX_BOUND_ERROR_MM = 0.01


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bbox_values(shape: cq.Shape) -> list[float]:
    box = shape.BoundingBox()
    return [
        float(box.xmin), float(box.ymin), float(box.zmin),
        float(box.xmax), float(box.ymax), float(box.zmax),
    ]


def mesh_bounds(mesh: trimesh.Trimesh) -> list[float]:
    return [
        float(mesh.bounds[0][0]), float(mesh.bounds[0][1]), float(mesh.bounds[0][2]),
        float(mesh.bounds[1][0]), float(mesh.bounds[1][1]), float(mesh.bounds[1][2]),
    ]


def run_openscad(
    openscad: str,
    mesh_path: Path,
    output_path: Path,
    bbox: list[float],
    timeout_seconds: int,
) -> dict[str, Any]:
    scad_path = mesh_path.with_suffix(".scad")
    # A far-away subtractor forces the imported STL through a CGAL difference
    # without changing the intended fixture solid.
    offset = max(abs(value) for value in bbox) + 10000.0
    import_literal = json.dumps(mesh_path.as_posix())
    scad_path.write_text(
        "difference() {\n"
        f"  import({import_literal});\n"
        f"  translate([{offset:.9f},{offset:.9f},{offset:.9f}]) cube([1,1,1]);\n"
        "}\n",
        encoding="utf-8",
    )
    command = [openscad, "-o", str(output_path), str(scad_path)]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AuditError(
            f"OpenSCAD timed out after {timeout_seconds}s: {mesh_path.name}"
        ) from exc

    log = (completed.stdout or "") + "\n" + (completed.stderr or "")
    require(
        completed.returncode == 0,
        f"OpenSCAD failed for {mesh_path.name}: {log[-3000:]}",
    )
    cache_lines = [
        line for line in log.splitlines()
        if "CGAL Polyhedrons in cache:" in line
    ]
    require(
        bool(cache_lines) and "1" in cache_lines[0].split(":", 1)[1],
        f"OpenSCAD did not report a CGAL polyhedron conversion: {mesh_path.name}: {log[-2000:]}",
    )
    require(
        re.search(r"Simple:\s+yes", log) is not None,
        f"OpenSCAD CGAL output was not simple: {mesh_path.name}: {log[-2000:]}",
    )
    require(
        output_path.is_file() and output_path.stat().st_size > 0,
        f"OpenSCAD produced no non-empty STL: {mesh_path.name}",
    )

    return {
        "command": command,
        "return_code": completed.returncode,
        "cgal_polyhedron_cache_line": cache_lines[0].strip(),
        "simple_line": next(
            (line.strip() for line in log.splitlines() if "Simple:" in line),
            None,
        ),
        "facets_line": next(
            (line.strip() for line in log.splitlines() if "Facets:" in line),
            None,
        ),
        "volumes_line": next(
            (line.strip() for line in log.splitlines() if "Volumes:" in line),
            None,
        ),
        "log": log,
    }


def audit_solid(
    step_path: Path,
    label: str,
    openscad: str,
    version_label: str,
    temp_root: Path,
    sequence: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    shape = importers.importStep(str(step_path)).val()
    require(shape.isValid(), f"OCP STEP import invalid: {label}")
    solids = shape.Solids()
    require(
        len(solids) == 1,
        f"Expected exactly one solid for {label}; found {len(solids)}",
    )
    solid = solids[0]
    brep_volume = float(solid.Volume())
    require(
        math.isfinite(brep_volume) and brep_volume > 0.0,
        f"Invalid B-rep volume: {label}",
    )

    vertices, faces = solid.tessellate(
        LINEAR_DEFLECTION_MM, ANGULAR_DEFLECTION_RAD
    )
    require(bool(vertices) and bool(faces), f"Empty OCP tessellation: {label}")
    input_mesh = trimesh.Trimesh(
        vertices=[vertex.toTuple() for vertex in vertices],
        faces=faces,
        process=True,
    )
    require(
        input_mesh.is_watertight and input_mesh.is_winding_consistent,
        f"OCP tessellation is not watertight/winding-consistent: {label}",
    )
    input_volume = abs(float(input_mesh.volume))
    require(
        math.isfinite(input_volume) and input_volume > 0.0,
        f"Invalid input mesh volume: {label}",
    )

    mesh_path = temp_root / f"{sequence:02d}-{step_path.stem}-input.stl"
    output_path = temp_root / f"{sequence:02d}-{step_path.stem}-cgal.stl"
    input_mesh.export(mesh_path)
    cgal_info = run_openscad(
        openscad, mesh_path, output_path, bbox_values(solid), timeout_seconds
    )

    output_mesh = trimesh.load_mesh(output_path, process=True)
    require(
        isinstance(output_mesh, trimesh.Trimesh),
        f"CGAL output did not load as one mesh: {label}",
    )
    require(
        output_mesh.is_watertight and output_mesh.is_winding_consistent,
        f"CGAL output mesh is not watertight/winding-consistent: {label}",
    )
    output_volume = abs(float(output_mesh.volume))
    require(
        math.isfinite(output_volume) and output_volume > 0.0,
        f"Invalid CGAL output volume: {label}",
    )
    output_bbox = mesh_bounds(output_mesh)
    expected_bbox = bbox_values(solid)
    bbox_error = max(abs(a - b) for a, b in zip(output_bbox, expected_bbox))
    input_to_output_rel = abs(input_volume - output_volume) / input_volume
    brep_to_output_rel = abs(brep_volume - output_volume) / brep_volume

    require(
        input_to_output_rel <= MAX_RELATIVE_INPUT_TO_CGAL_VOLUME_ERROR,
        f"Input/CGAL mesh volume changed beyond tolerance: {label}: {input_to_output_rel}",
    )
    require(
        brep_to_output_rel <= MAX_RELATIVE_BREP_TO_CGAL_MESH_VOLUME_ERROR,
        f"B-rep/CGAL mesh volume differs beyond tolerance: {label}: {brep_to_output_rel}",
    )
    require(
        bbox_error <= MAX_BOUND_ERROR_MM,
        f"B-rep/CGAL mesh bounds differ beyond tolerance: {label}: {bbox_error} mm",
    )

    return {
        "subject": label,
        "step_sha256": sha256(step_path),
        "input_stl_sha256": sha256(mesh_path),
        "cgal_output_stl_sha256": sha256(output_path),
        "brep_volume_mm3": brep_volume,
        "input_mesh_volume_mm3": input_volume,
        "cgal_output_mesh_volume_mm3": output_volume,
        "relative_input_to_cgal_volume_error": input_to_output_rel,
        "relative_brep_to_cgal_mesh_volume_error": brep_to_output_rel,
        "max_brep_to_cgal_bbox_error_mm": bbox_error,
        "input_mesh_faces": int(len(input_mesh.faces)),
        "cgal_output_mesh_faces": int(len(output_mesh.faces)),
        "cgal_output_watertight": bool(output_mesh.is_watertight),
        "cgal_output_winding_consistent": bool(output_mesh.is_winding_consistent),
        "openscad_version": version_label,
        "cgal_simple": True,
        "cgal_cache_line": cgal_info["cgal_polyhedron_cache_line"],
        "cgal_simple_line": cgal_info["simple_line"],
        "cgal_facets_line": cgal_info["facets_line"],
        "cgal_volumes_line": cgal_info["volumes_line"],
        "openscad_log_sha256": hashlib.sha256(
            cgal_info["log"].encode("utf-8")
        ).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an OpenSCAD/CGAL cross-engine audit on synthetic STEP-derived meshes."
    )
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=ROOT,
        help="Unpacked engineering-fixtures-v0.4 package root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON report path (default: <fixture-root>/openscad-cgal-mesh-audit.json)",
    )
    parser.add_argument("--openscad", default="openscad", help="OpenSCAD executable")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()

    fixture_root = args.fixture_root.resolve()
    require(
        (fixture_root / "cad").is_dir() and (fixture_root / "assembly").is_dir(),
        f"Not an engineering-fixtures-v0.4 package root: {fixture_root}",
    )
    executable = shutil.which(args.openscad)
    require(executable is not None, f"OpenSCAD executable not found: {args.openscad}")
    version_run = subprocess.run(
        [executable, "--version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
        shell=False,
    )
    version_label = (version_run.stdout + version_run.stderr).strip()
    require(version_run.returncode == 0, f"Cannot query OpenSCAD version: {version_label}")
    require(
        EXPECTED_OPENSCAD_VERSION in version_label,
        f"OpenSCAD version differs from frozen profile {EXPECTED_OPENSCAD_VERSION}: {version_label}",
    )

    step_paths = sorted((fixture_root / "cad").glob("*.step"))
    assembly_path = fixture_root / "assembly" / "ASM-P01_support-frame.step"
    require(
        len(step_paths) == 12 and assembly_path.is_file(),
        "Expected 12 CAD variants and one support-frame assembly STEP",
    )

    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="engineering-fixture-openscad-cgal-") as temporary:
        temp_root = Path(temporary)
        sequence = 0
        for step_path in step_paths:
            records.append(
                audit_solid(
                    step_path,
                    f"cad/{step_path.name}",
                    executable,
                    version_label,
                    temp_root,
                    sequence,
                    args.timeout_seconds,
                )
            )
            sequence += 1

        assembly = importers.importStep(str(assembly_path)).val()
        require(
            assembly.isValid() and len(assembly.Solids()) == 7,
            "Assembly STEP must contain seven valid solids",
        )
        for index, solid in enumerate(assembly.Solids(), start=1):
            # Export one assembly instance through a temporary STEP file so the
            # CGAL mesh test treats each article as a separate subject.
            local_step = temp_root / f"assembly-solid-{index}.step"
            cq.exporters.export(solid, str(local_step), exportType="STEP")
            records.append(
                audit_solid(
                    local_step,
                    f"assembly/ASM-P01_support-frame.step#solid-{index}",
                    executable,
                    version_label,
                    temp_root,
                    sequence,
                    args.timeout_seconds,
                )
            )
            sequence += 1

    max_input_to_cgal = max(
        record["relative_input_to_cgal_volume_error"] for record in records
    )
    max_brep_to_cgal = max(
        record["relative_brep_to_cgal_mesh_volume_error"] for record in records
    )
    max_bbox = max(
        record["max_brep_to_cgal_bbox_error_mm"] for record in records
    )
    report = {
        "schema_version": "openscad-cgal-mesh-audit-v0.1",
        "evidence_class": "synthetic_development_fixture_integrity",
        "status": "PASS_CROSS_ENGINE_CGAL_MESH_REEVALUATION_ONLY",
        "claim_ceiling": (
            "STEP import and tessellation use Open CASCADE via CadQuery OCP; "
            "OpenSCAD 2021.01 then routes each STL through a CGAL difference "
            "with a disjoint subtractor. This checks the mesh representation "
            "through another geometry engine. It is not native STEP B-rep "
            "import into a second CAD kernel, not native CAD feature-history "
            "validation, not AP242 conformance and not physical validation."
        ),
        "toolchain": {
            "cadquery": version("cadquery"),
            "cadquery_ocp_distribution": version("cadquery-ocp"),
            "opencascade_engine": getattr(OCP, "__version__", "unknown"),
            "trimesh": version("trimesh"),
            "openscad": version_label,
            "geometry_engine_after_tessellation": "CGAL through OpenSCAD",
        },
        "tessellation": {
            "linear_deflection_mm": LINEAR_DEFLECTION_MM,
            "angular_deflection_rad": ANGULAR_DEFLECTION_RAD,
        },
        "thresholds": {
            "max_relative_input_to_cgal_volume_error": MAX_RELATIVE_INPUT_TO_CGAL_VOLUME_ERROR,
            "max_relative_brep_to_cgal_mesh_volume_error": MAX_RELATIVE_BREP_TO_CGAL_MESH_VOLUME_ERROR,
            "max_brep_to_cgal_bbox_error_mm": MAX_BOUND_ERROR_MM,
        },
        "subject_count": len(records),
        "passed_count": sum(
            1 for record in records
            if record["cgal_simple"]
            and record["cgal_output_watertight"]
            and record["cgal_output_winding_consistent"]
        ),
        "max_relative_input_to_cgal_volume_error": max_input_to_cgal,
        "max_relative_brep_to_cgal_mesh_volume_error": max_brep_to_cgal,
        "max_brep_to_cgal_bbox_error_mm": max_bbox,
        "subjects": records,
    }
    output = (
        args.output
        or (fixture_root / "openscad-cgal-mesh-audit.json")
    ).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_keys = (
        "schema_version",
        "status",
        "subject_count",
        "passed_count",
        "max_relative_input_to_cgal_volume_error",
        "max_relative_brep_to_cgal_mesh_volume_error",
        "max_brep_to_cgal_bbox_error_mm",
        "toolchain",
    )
    print(json.dumps({key: report[key] for key in summary_keys}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
