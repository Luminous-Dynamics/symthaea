#!/usr/bin/env python3
"""Contract tests for compile_fsaverage5_glasser_map.py."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("compile_fsaverage5_glasser_map.py")
SPEC = importlib.util.spec_from_file_location("fsavg_compiler", SCRIPT)
assert SPEC and SPEC.loader
compiler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compiler)


AREA_ORDER = Path(__file__).resolve().parents[1] / "data" / "neuroscience" / "hcp_mmp1_area_order_v1.json"


def fake_source(name: str) -> dict[str, str]:
    return {
        "source_id": f"fixture:{name}",
        "source_version": "1",
        "source_digest": "sha256:" + ("1" if name == "left" else "2") * 64,
        "generator_id": "test-fixture",
        "generator_version": "1",
        "terms_reference": "test-only synthetic fixture",
    }


def surface_doc(hemisphere: str, areas: list[str]) -> dict:
    prefix = "L_" if hemisphere == "left" else "R_"
    labels: list[str | None] = [None] * compiler.VERTICES_PER_HEMISPHERE
    for idx, area in enumerate(areas):
        labels[idx] = prefix + area
    return {
        "schema": compiler.LABEL_SCHEMA,
        "space": "fsaverage5",
        "hemisphere": hemisphere,
        "vertex_count": compiler.VERTICES_PER_HEMISPHERE,
        "labels": labels,
        "source": fake_source(hemisphere),
    }


class CompilerContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.areas, _ = compiler.load_area_order(AREA_ORDER)
        self.lh = self.root / "lh.json"
        self.rh = self.root / "rh.json"
        self.write_json(self.lh, surface_doc("left", self.areas))
        self.write_json(self.rh, surface_doc("right", self.areas))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @staticmethod
    def write_json(path: Path, value: object) -> None:
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def test_compiles_complete_semantic_mapping(self) -> None:
        artifact = compiler.compile_artifact(self.lh, self.rh, AREA_ORDER)
        compiler.validate_artifact(artifact, AREA_ORDER, self.lh, self.rh)

        self.assertEqual(len(artifact["vertex_to_parcel"]), 20_484)
        self.assertEqual(artifact["vertex_to_parcel"][:180], list(range(1, 181)))
        self.assertEqual(
            artifact["vertex_to_parcel"][10_242 : 10_242 + 180],
            list(range(181, 361)),
        )
        self.assertEqual(artifact["qualification"]["parcel_vertex_counts"], [1] * 360)
        self.assertEqual(artifact["qualification"]["cross_hemisphere_violations"], 0)

    def test_deterministic_content_digest(self) -> None:
        first = compiler.compile_artifact(self.lh, self.rh, AREA_ORDER)
        second = compiler.compile_artifact(self.lh, self.rh, AREA_ORDER)
        self.assertEqual(first, second)
        self.assertEqual(first["content_digest"], second["content_digest"])

    def test_write_artifact_is_byte_deterministic(self) -> None:
        artifact = compiler.compile_artifact(self.lh, self.rh, AREA_ORDER)
        first = self.root / "first.json"
        second = self.root / "second.json"
        compiler.write_artifact(first, artifact)
        compiler.write_artifact(second, artifact)
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_malformed_source_input_is_rejected_cleanly(self) -> None:
        artifact = compiler.compile_artifact(self.lh, self.rh, AREA_ORDER)
        artifact["source_inputs"][0] = "not-an-object"
        with self.assertRaisesRegex(compiler.QualificationError, "source_input must be an object"):
            compiler.validate_artifact(artifact, AREA_ORDER)

    def test_missing_qualification_field_is_rejected_cleanly(self) -> None:
        artifact = compiler.compile_artifact(self.lh, self.rh, AREA_ORDER)
        del artifact["qualification"]["unknown_labels"]
        with self.assertRaisesRegex(compiler.QualificationError, "missing fields: unknown_labels"):
            compiler.validate_artifact(artifact, AREA_ORDER)

    def test_wrong_vertex_count_fails_closed(self) -> None:
        doc = surface_doc("left", self.areas)
        doc["labels"].pop()
        doc["vertex_count"] -= 1
        self.write_json(self.lh, doc)
        with self.assertRaisesRegex(compiler.QualificationError, "vertex_count must be 10242"):
            compiler.compile_artifact(self.lh, self.rh, AREA_ORDER)

    def test_cross_hemisphere_label_fails_closed(self) -> None:
        doc = surface_doc("left", self.areas)
        doc["labels"][0] = "R_V1"
        self.write_json(self.lh, doc)
        with self.assertRaisesRegex(compiler.QualificationError, "canonical L_ hemisphere prefix"):
            compiler.compile_artifact(self.lh, self.rh, AREA_ORDER)

    def test_unknown_area_fails_closed(self) -> None:
        doc = surface_doc("right", self.areas)
        doc["labels"][0] = "R_NOT_A_REAL_HCP_AREA"
        self.write_json(self.rh, doc)
        with self.assertRaisesRegex(compiler.QualificationError, "unknown HCP-MMP1 area"):
            compiler.compile_artifact(self.lh, self.rh, AREA_ORDER)

    def test_missing_parcel_coverage_fails_closed(self) -> None:
        doc = surface_doc("left", self.areas)
        doc["labels"][179] = None
        self.write_json(self.lh, doc)
        with self.assertRaisesRegex(compiler.QualificationError, "empty parcels"):
            compiler.compile_artifact(self.lh, self.rh, AREA_ORDER)

    def test_null_is_explicit_unassigned_vertex(self) -> None:
        artifact = compiler.compile_artifact(self.lh, self.rh, AREA_ORDER)
        self.assertIsNone(artifact["vertex_to_parcel"][180])
        self.assertIsNone(artifact["vertex_to_parcel"][10_242 + 180])
        self.assertEqual(
            artifact["qualification"]["unassigned_vertices_left"],
            10_242 - 180,
        )
        self.assertEqual(
            artifact["qualification"]["unassigned_vertices_right"],
            10_242 - 180,
        )

    def test_artifact_digest_tamper_is_rejected(self) -> None:
        artifact = compiler.compile_artifact(self.lh, self.rh, AREA_ORDER)
        artifact["content_digest"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(compiler.QualificationError, "content_digest mismatch"):
            compiler.validate_artifact(artifact, AREA_ORDER, self.lh, self.rh)

    def test_semantic_source_file_tamper_is_rejected(self) -> None:
        artifact = compiler.compile_artifact(self.lh, self.rh, AREA_ORDER)
        doc = surface_doc("left", self.areas)
        doc["source"]["source_version"] = "2"
        self.write_json(self.lh, doc)
        with self.assertRaisesRegex(compiler.QualificationError, "semantic label file digest mismatch"):
            compiler.validate_artifact(artifact, AREA_ORDER, self.lh, self.rh)

    def test_left_to_right_parcel_leakage_is_rejected(self) -> None:
        artifact = compiler.compile_artifact(self.lh, self.rh, AREA_ORDER)
        artifact["vertex_to_parcel"][0] = 181
        body = dict(artifact)
        del body["content_digest"]
        artifact["content_digest"] = compiler.sha256_digest(compiler.canonical_json_bytes(body))
        with self.assertRaisesRegex(compiler.QualificationError, "left vertex 0 maps to right parcel"):
            compiler.validate_artifact(artifact, AREA_ORDER, self.lh, self.rh)

    def test_unknown_input_field_is_rejected(self) -> None:
        doc = surface_doc("left", self.areas)
        doc["trust_me"] = True
        self.write_json(self.lh, doc)
        with self.assertRaisesRegex(compiler.QualificationError, "unknown fields: trust_me"):
            compiler.compile_artifact(self.lh, self.rh, AREA_ORDER)


if __name__ == "__main__":
    unittest.main()
