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
    cache_match = re.search(r"^CGAL Polyhedrons in cache:\s*(\d+)\s*$", log, re.MULTILINE)
    require(
        cache_match is not None and int(cache_match.group(1)) == 1,
        f"Expected exactly one CGAL polyhedron conversion: {mesh_path.name}: {log[-2000:]}",
    )
    simple_match = re.search(r"^\s*Simple:\s*yes\s*$", log, re.MULTILINE)
    require(
        simple_match is not None,
        f"OpenSCAD CGAL output was not simple: {mesh_path.name}: {log[-2000:]}",
    )
    require(
        output_path.is_file() and output_path.stat().st_size > 0,
        f"OpenSCAD produced no non-empty STL: {mesh_path.name}",
    )

    return {
        "command": command,
        "return_code": completed.returncode,
        "cgal_polyhedron_cache_line": cache_match.group(0).strip(),
        "simple_line": simple_match.group(0).strip(),
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



def package_path(root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    require(not candidate.is_absolute(), f"Absolute package path forbidden: {relative_path}")
    require(".." not in candidate.parts, f"Parent traversal forbidden: {relative_path}")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    require(resolved.is_relative_to(resolved_root), f"Path escapes package root: {relative_path}")
    require(resolved.is_file(), f"Manifest artifact missing: {relative_path}")
    return resolved


def validate_manifest_digest(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        f"Invalid SHA-256 in materialization manifest for {label}",
    )
    return value


def load_manifest_subjects(root: Path) -> tuple[dict[str, Any], list[tuple[str, Path, str]], Path, str, str]:
    manifest_path = root / "materialization-manifest.json"
    require(manifest_path.is_file(), "materialization-manifest.json is required")
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"Invalid materialization manifest: {exc}") from exc
    require(isinstance(manifest, dict), "Materialization manifest root must be an object")
    schema_version = manifest.get("schema_version")
    require(
        isinstance(schema_version, str)
        and schema_version.startswith("engineering-fixture-materialization-v0."),
        f"Unsupported materialization manifest schema: {schema_version!r}",
    )

    cad_cases = manifest.get("cad_cases")
    require(isinstance(cad_cases, list) and len(cad_cases) == 6,
            "Manifest must contain exactly six CAD-HISTORY cases")
    subjects: list[tuple[str, Path, str]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for case in cad_cases:
        require(isinstance(case, dict), "CAD case must be an object")
        fixture_id = case.get("fixture_id")
        require(isinstance(fixture_id, str) and fixture_id not in seen_ids,
                f"Missing or duplicate fixture ID: {fixture_id!r}")
        seen_ids.add(fixture_id)
        variants = case.get("variants")
        require(isinstance(variants, list) and len(variants) == 2,
                f"{fixture_id} must declare exactly two variants")
        variant_names: set[str] = set()
        for variant in variants:
            require(isinstance(variant, dict), f"{fixture_id} variant must be an object")
            name = variant.get("variant")
            require(name in {"source", "edited"} and name not in variant_names,
                    f"{fixture_id} has missing/duplicate/unsupported variant {name!r}")
            variant_names.add(name)
            require(variant.get("valid_brep") is True and variant.get("solid_count") == 1,
                    f"{fixture_id}/{name} is not declared as one valid B-rep solid")
            rel = variant.get("artifact")
            require(isinstance(rel, str) and rel not in seen_paths,
                    f"Missing or duplicate CAD artifact path: {rel!r}")
            seen_paths.add(rel)
            expected = validate_manifest_digest(variant.get("sha256"), rel)
            artifact = package_path(root, rel)
            actual = sha256(artifact)
            require(actual == expected,
                    f"Manifest digest mismatch for {rel}: expected {expected}, got {actual}")
            subjects.append((rel, artifact, expected))
        require(variant_names == {"source", "edited"},
                f"{fixture_id} must declare source and edited variants")
    require(len(subjects) == 12, f"Expected 12 manifest-bound CAD variants; found {len(subjects)}")

    assembly = manifest.get("assembly")
    require(isinstance(assembly, dict), "Manifest assembly record is required")
    assembly_rel = assembly.get("artifact")
    require(isinstance(assembly_rel, str), "Manifest assembly artifact path is missing")
    assembly_digest = validate_manifest_digest(assembly.get("sha256"), assembly_rel)
    assembly_path = package_path(root, assembly_rel)
    require(sha256(assembly_path) == assembly_digest,
            f"Manifest digest mismatch for assembly artifact {assembly_rel}")
    require(assembly.get("valid_brep") is True
            and assembly.get("solid_count") == 7
            and assembly.get("physical_instance_count") == 7,
            "Assembly manifest must declare one valid seven-solid / seven-instance assembly")
    part_definitions = assembly.get("part_definitions")
    require(isinstance(part_definitions, list) and len(part_definitions) == 6,
            "Assembly manifest must declare six unique part definitions")
    for part in part_definitions:
        require(isinstance(part, dict) and part.get("valid_brep") is True,
                "Each part definition must be declared a valid B-rep")
        part_rel = part.get("artifact")
        expected = validate_manifest_digest(part.get("sha256"), str(part_rel))
        part_path = package_path(root, part_rel)
        require(sha256(part_path) == expected,
                f"Manifest digest mismatch for assembly part definition {part_rel}")

    manifest_digest = sha256(manifest_path)
    return manifest, subjects, assembly_path, assembly_digest, manifest_digest


def validate_audit_toolchain(manifest: dict[str, Any], openscad_arg: str) -> tuple[str, str, dict[str, str]]:
    declared = manifest.get("toolchain")
    require(isinstance(declared, dict), "Manifest toolchain profile is required")
    require(declared.get("cadquery") == "2.8.0",
            f"Manifest CadQuery version is not the audited pin: {declared.get('cadquery')!r}")
    require(declared.get("cadquery_ocp") == "7.9.3.1",
            f"Manifest Open CASCADE pin is not the audited pin: {declared.get('cadquery_ocp')!r}")
    python_version = sys.version.split()[0]
    require(declared.get("python") == python_version,
            f"Python version differs from manifest: expected {declared.get('python')}, got {python_version}")

    actual = {
        "python": python_version,
        "cadquery": version("cadquery"),
        "cadquery_ocp_distribution": version("cadquery-ocp"),
        "opencascade_module": getattr(OCP, "__version__", "unknown"),
        "trimesh": version("trimesh"),
    }
    require(actual["cadquery"] == "2.8.0", f"Unexpected CadQuery version: {actual['cadquery']}")
    require(actual["opencascade_module"] == "7.9.3.1",
            f"Unexpected Open CASCADE module version: {actual['opencascade_module']}")
    require(actual["trimesh"] == "4.11.1", f"Unexpected Trimesh version: {actual['trimesh']}")

    executable = shutil.which(openscad_arg)
    require(executable is not None, f"OpenSCAD executable not found: {openscad_arg}")
    result = subprocess.run(
        [executable, "--version"], capture_output=True, text=True,
        check=False, timeout=15, shell=False,
    )
    version_label = (result.stdout + result.stderr).strip()
    require(result.returncode == 0, f"Cannot query OpenSCAD version: {version_label}")
    require(version_label == "OpenSCAD version 2021.01",
            f"OpenSCAD version differs from frozen profile: {version_label}")
    actual["openscad"] = version_label
    return executable, version_label, actual

def audit_solid(
    step_path: Path,
    label: str,
    openscad: str,
    version_label: str,
    temp_root: Path,
    sequence: int,
    timeout_seconds: int,
    source_artifact: str | None = None,
    source_digest: str | None = None,
    manifest_digest: str | None = None,
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
    require(
        len(input_mesh.split(only_watertight=False)) == 1,
        f"OCP tessellation has multiple connected components: {label}",
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
    require(
        len(output_mesh.split(only_watertight=False)) == 1,
        f"CGAL output contains multiple connected components: {label}",
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
        "source_artifact": source_artifact or label,
        "source_artifact_sha256": source_digest or sha256(step_path),
        "source_manifest_sha256": manifest_digest,
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
    require(args.timeout_seconds > 0, "--timeout-seconds must be positive")

    fixture_root = args.fixture_root.resolve()
    manifest, cad_subjects, assembly_path, assembly_digest, manifest_digest = load_manifest_subjects(fixture_root)
    executable, version_label, toolchain = validate_audit_toolchain(manifest, args.openscad)

    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="engineering-fixture-openscad-cgal-") as temporary:
        temp_root = Path(temporary)
        sequence = 0
        for relative_path, step_path, expected_digest in cad_subjects:
            records.append(
                audit_solid(
                    step_path,
                    relative_path,
                    executable,
                    version_label,
                    temp_root,
                    sequence,
                    args.timeout_seconds,
                    source_artifact=relative_path,
                    source_digest=expected_digest,
                    manifest_digest=manifest_digest,
                )
            )
            sequence += 1

        assembly = importers.importStep(str(assembly_path)).val()
        require(
            assembly.isValid() and len(assembly.Solids()) == 7,
            "Assembly STEP must contain seven valid solids",
        )
        for index, solid in enumerate(assembly.Solids(), start=1):
            # The temporary STEP is a derived per-instance representation.
            # The parent assembly digest remains the source identity.
            local_step = temp_root / f"assembly-solid-{index}.step"
            cq.exporters.export(solid, str(local_step), exportType="STEP")
            records.append(
                audit_solid(
                    local_step,
                    f"{manifest['assembly']['fixture_id']}#solid-{index}",
                    executable,
                    version_label,
                    temp_root,
                    sequence,
                    args.timeout_seconds,
                    source_artifact=manifest["assembly"]["artifact"],
                    source_digest=assembly_digest,
                    manifest_digest=manifest_digest,
                )
            )
            sequence += 1

    require(len(records) == 19, f"Expected 19 mesh subjects, got {len(records)}")

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
        "schema_version": "openscad-cgal-mesh-audit-v0.2",
        "manifest_schema_version": manifest["schema_version"],
        "manifest_sha256": manifest_digest,
        "toolchain": toolchain,
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
            **toolchain,
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
    protected = {assembly_path.resolve(), (fixture_root / "materialization-manifest.json").resolve()}
    protected.update(path.resolve() for _, path, _ in cad_subjects)
    require(output not in protected, "Audit output cannot overwrite an input STEP or the manifest")
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
