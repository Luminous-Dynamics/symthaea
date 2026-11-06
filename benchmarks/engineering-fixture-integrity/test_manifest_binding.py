from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("audit_openscad_cgal_mesh.py")
SPEC = importlib.util.spec_from_file_location("audit_openscad_cgal_mesh", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ManifestBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "cad").mkdir()
        (self.root / "assembly").mkdir()

        self.cad_cases = []
        for case_index in range(1, 7):
            variants = []
            for variant_name in ("source", "edited"):
                rel = f"cad/CADH-P{case_index:02d}-{variant_name}.step"
                data = f"fixture-{case_index}-{variant_name}".encode()
                (self.root / rel).write_bytes(data)
                variants.append({
                    "variant": variant_name,
                    "artifact": rel,
                    "sha256": digest(data),
                    "valid_brep": True,
                    "solid_count": 1,
                })
            self.cad_cases.append({
                "fixture_id": f"CADH-P{case_index:02d}",
                "variants": variants,
            })

        assembly_rel = "assembly/ASM-P01_support-frame.step"
        assembly_data = b"assembly"
        (self.root / assembly_rel).write_bytes(assembly_data)
        self.part_definitions = []
        for part_index in range(1, 7):
            rel = f"assembly/part-{part_index}.step"
            data = f"part-{part_index}".encode()
            (self.root / rel).write_bytes(data)
            self.part_definitions.append({
                "artifact": rel,
                "sha256": digest(data),
                "valid_brep": True,
            })

        self.manifest = {
            "schema_version": "engineering-fixture-materialization-v0.4",
            "toolchain": {
                "python": "3.13.5",
                "cadquery": "2.8.0",
                "cadquery_ocp": "7.9.3.1",
            },
            "cad_cases": self.cad_cases,
            "assembly": {
                "fixture_id": "ASM-P01",
                "artifact": assembly_rel,
                "sha256": digest(assembly_data),
                "valid_brep": True,
                "solid_count": 7,
                "physical_instance_count": 7,
                "part_definitions": self.part_definitions,
            },
        }
        self.manifest_path = self.root / "materialization-manifest.json"
        self.write_manifest()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_manifest(self) -> None:
        self.manifest_path.write_text(
            json.dumps(self.manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    def test_valid_manifest_binds_all_twelve_cad_artifacts_and_assembly(self) -> None:
        manifest, subjects, assembly_path, assembly_digest, manifest_digest = (
            AUDIT.load_manifest_subjects(self.root)
        )
        self.assertEqual(len(subjects), 12)
        self.assertEqual(manifest["assembly"]["physical_instance_count"], 7)
        self.assertTrue(assembly_path.is_file())
        self.assertEqual(assembly_digest, self.manifest["assembly"]["sha256"])
        self.assertEqual(manifest_digest, digest(self.manifest_path.read_bytes()))

    def test_tampered_cad_artifact_fails_before_geometry_processing(self) -> None:
        artifact = self.root / self.cad_cases[0]["variants"][0]["artifact"]
        artifact.write_bytes(b"tampered")
        with self.assertRaisesRegex(AUDIT.AuditError, "Manifest digest mismatch"):
            AUDIT.load_manifest_subjects(self.root)

    def test_missing_cad_artifact_fails_closed(self) -> None:
        artifact = self.root / self.cad_cases[0]["variants"][0]["artifact"]
        artifact.unlink()
        with self.assertRaisesRegex(AUDIT.AuditError, "Manifest artifact missing"):
            AUDIT.load_manifest_subjects(self.root)

    def test_parent_path_traversal_is_rejected(self) -> None:
        self.cad_cases[0]["variants"][0]["artifact"] = "../outside.step"
        self.cad_cases[0]["variants"][0]["sha256"] = digest(b"outside")
        self.write_manifest()
        with self.assertRaisesRegex(AUDIT.AuditError, "Parent traversal forbidden"):
            AUDIT.load_manifest_subjects(self.root)

    def test_duplicate_cad_artifact_reference_is_rejected(self) -> None:
        first = self.cad_cases[0]["variants"][0]
        second = self.cad_cases[0]["variants"][1]
        second["artifact"] = first["artifact"]
        second["sha256"] = first["sha256"]
        self.write_manifest()
        with self.assertRaisesRegex(AUDIT.AuditError, "duplicate CAD artifact path"):
            AUDIT.load_manifest_subjects(self.root)

    def test_tampered_assembly_fails_digest_check(self) -> None:
        assembly_path = self.root / self.manifest["assembly"]["artifact"]
        assembly_path.write_bytes(b"tampered assembly")
        with self.assertRaisesRegex(AUDIT.AuditError, "Manifest digest mismatch for assembly"):
            AUDIT.load_manifest_subjects(self.root)

    def test_tampered_part_definition_fails_digest_check(self) -> None:
        part_path = self.root / self.part_definitions[0]["artifact"]
        part_path.write_bytes(b"tampered part")
        with self.assertRaisesRegex(AUDIT.AuditError, "Manifest digest mismatch for assembly part"):
            AUDIT.load_manifest_subjects(self.root)

    def test_instance_count_mismatch_is_rejected(self) -> None:
        self.manifest["assembly"]["physical_instance_count"] = 6
        self.write_manifest()
        with self.assertRaisesRegex(AUDIT.AuditError, "seven-solid / seven-instance"):
            AUDIT.load_manifest_subjects(self.root)

    def test_incomplete_cad_case_census_is_rejected(self) -> None:
        self.manifest["cad_cases"].pop()
        self.write_manifest()
        with self.assertRaisesRegex(AUDIT.AuditError, "exactly six CAD-HISTORY cases"):
            AUDIT.load_manifest_subjects(self.root)

    def test_symlink_escape_is_rejected(self) -> None:
        outside = self.root.parent / f"{self.root.name}-outside.step"
        link = self.root / "cad" / "outside-link.step"
        outside.write_bytes(b"outside artifact")
        try:
            link.symlink_to(outside)
            variant = self.cad_cases[0]["variants"][0]
            variant["artifact"] = "cad/outside-link.step"
            variant["sha256"] = digest(b"outside artifact")
            self.write_manifest()
            with self.assertRaisesRegex(AUDIT.AuditError, "Path escapes package root"):
                AUDIT.load_manifest_subjects(self.root)
        finally:
            if link.is_symlink():
                link.unlink()
            if outside.exists():
                outside.unlink()

    def test_duplicate_fixture_id_is_rejected(self) -> None:
        self.cad_cases[1]["fixture_id"] = self.cad_cases[0]["fixture_id"]
        self.write_manifest()
        with self.assertRaisesRegex(AUDIT.AuditError, "Missing or duplicate fixture ID"):
            AUDIT.load_manifest_subjects(self.root)

    def test_incomplete_source_edited_pair_is_rejected(self) -> None:
        self.cad_cases[0]["variants"].pop()
        self.write_manifest()
        with self.assertRaisesRegex(AUDIT.AuditError, "must declare exactly two variants"):
            AUDIT.load_manifest_subjects(self.root)

    def test_malformed_sha256_is_rejected(self) -> None:
        self.cad_cases[0]["variants"][0]["sha256"] = "not-a-sha256"
        self.write_manifest()
        with self.assertRaisesRegex(AUDIT.AuditError, "Invalid SHA-256"):
            AUDIT.load_manifest_subjects(self.root)

    def test_non_v04_manifest_schema_is_rejected(self) -> None:
        self.manifest["schema_version"] = "engineering-fixture-materialization-v0.1"
        self.write_manifest()
        with self.assertRaisesRegex(AUDIT.AuditError, "Expected exact manifest schema"):
            AUDIT.load_manifest_subjects(self.root)

    def test_duplicate_part_definition_path_is_rejected(self) -> None:
        self.part_definitions[1]["artifact"] = self.part_definitions[0]["artifact"]
        self.part_definitions[1]["sha256"] = self.part_definitions[0]["sha256"]
        self.write_manifest()
        with self.assertRaisesRegex(AUDIT.AuditError, "duplicate assembly part-definition artifact path"):
            AUDIT.load_manifest_subjects(self.root)

    def test_assembly_artifact_cannot_be_reused_as_part_definition(self) -> None:
        self.part_definitions[0]["artifact"] = self.manifest["assembly"]["artifact"]
        self.part_definitions[0]["sha256"] = self.manifest["assembly"]["sha256"]
        self.write_manifest()
        with self.assertRaisesRegex(AUDIT.AuditError, "duplicate assembly part-definition artifact path"):
            AUDIT.load_manifest_subjects(self.root)

    def test_output_guard_protects_part_definition_artifacts(self) -> None:
        _, subjects, assembly_path, _, _ = AUDIT.load_manifest_subjects(self.root)
        protected_part = self.root / self.part_definitions[0]["artifact"]
        with self.assertRaisesRegex(AUDIT.AuditError, "cannot overwrite an input STEP"):
            AUDIT.ensure_safe_output_path(
                protected_part,
                self.root,
                self.manifest,
                assembly_path,
                subjects,
            )

    def test_output_guard_rejects_symlink_to_input_part_definition(self) -> None:
        _, subjects, assembly_path, _, _ = AUDIT.load_manifest_subjects(self.root)
        protected_part = self.root / self.part_definitions[0]["artifact"]
        output_link = self.root / "audit-output.json"
        output_link.symlink_to(protected_part)
        try:
            with self.assertRaisesRegex(AUDIT.AuditError, "cannot overwrite an input STEP"):
                AUDIT.ensure_safe_output_path(
                    output_link,
                    self.root,
                    self.manifest,
                    assembly_path,
                    subjects,
                )
        finally:
            output_link.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
