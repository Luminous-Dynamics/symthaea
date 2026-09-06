#!/usr/bin/env python3
"""Contract tests for compare_fsaverage5_glasser_maps.py."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AREA_ORDER = ROOT / "data" / "neuroscience" / "hcp_mmp1_area_order_v1.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compiler = load_module("compiler", Path(__file__).with_name("compile_fsaverage5_glasser_map.py"))
crosscheck = load_module("crosscheck", Path(__file__).with_name("compare_fsaverage5_glasser_maps.py"))


def source(tag: str, hemi: str) -> dict[str, str]:
    seed = (tag + hemi).encode("utf-8")
    return {
        "source_id": f"fixture:{tag}:{hemi}",
        "source_version": "1",
        "source_digest": compiler.sha256_digest(seed),
        "generator_id": f"generator:{tag}",
        "generator_version": "1",
        "terms_reference": "test-only synthetic fixture",
    }


def semantic_doc(tag: str, hemi: str, areas: list[str]) -> dict:
    prefix = "L_" if hemi == "left" else "R_"
    labels: list[str | None] = [None] * compiler.VERTICES_PER_HEMISPHERE
    # Two vertices per parcel allows tests to remove/change one while retaining coverage.
    for index, area in enumerate(areas):
        labels[index * 2] = prefix + area
        labels[index * 2 + 1] = prefix + area
    return {
        "schema": compiler.LABEL_SCHEMA,
        "space": "fsaverage5",
        "hemisphere": hemi,
        "vertex_count": compiler.VERTICES_PER_HEMISPHERE,
        "labels": labels,
        "source": source(tag, hemi),
    }


class CrosscheckContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.areas, _ = compiler.load_area_order(AREA_ORDER)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_json(self, name: str, value: object) -> Path:
        path = self.root / name
        path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return path

    def make_artifact(self, tag: str) -> dict:
        lh = self.write_json(f"{tag}-lh.json", semantic_doc(tag, "left", self.areas))
        rh = self.write_json(f"{tag}-rh.json", semantic_doc(tag, "right", self.areas))
        return compiler.compile_artifact(lh, rh, AREA_ORDER)

    def test_independent_metadata_can_have_exact_mapping_agreement(self) -> None:
        a = self.make_artifact("a")
        b = self.make_artifact("b")
        self.assertNotEqual(a["content_digest"], b["content_digest"])
        report = crosscheck.build_report(a, b)
        crosscheck.validate_report(report)
        self.assertTrue(report["qualification"]["exact_mapping_agreement"])
        self.assertFalse(report["qualification"]["self_comparison"])
        self.assertEqual(report["summary"]["disagreement_vertices"], 0)
        self.assertTrue(report["source_distinctness"]["metadata_distinctness_detected"])
        self.assertFalse(report["qualification"]["independence_established"])

    def test_self_comparison_is_explicit(self) -> None:
        a = self.make_artifact("a")
        report = crosscheck.build_report(a, a)
        self.assertTrue(report["qualification"]["self_comparison"])
        self.assertTrue(report["qualification"]["exact_mapping_agreement"])
        self.assertFalse(report["source_distinctness"]["metadata_distinctness_detected"])

    def test_parcel_mismatch_is_reported_by_vertex_and_parcel(self) -> None:
        a = self.make_artifact("a")
        b = self.make_artifact("b")
        # Swap one vertex from parcel 1 with one from parcel 2; coverage remains valid.
        b["vertex_to_parcel"][0] = 2
        b["vertex_to_parcel"][2] = 1
        b["qualification"]["parcel_vertex_counts"] = list(a["qualification"]["parcel_vertex_counts"])
        body = dict(b)
        del body["content_digest"]
        b["content_digest"] = compiler.sha256_digest(compiler.canonical_json_bytes(body))
        compiler.validate_artifact(b, AREA_ORDER)

        report = crosscheck.build_report(a, b)
        self.assertEqual(report["summary"]["parcel_mismatch"], 2)
        self.assertEqual(report["summary"]["disagreement_vertices"], 2)
        self.assertEqual([row["vertex"] for row in report["disagreements"]], [0, 2])
        parcel1 = report["parcels"][0]
        parcel2 = report["parcels"][1]
        self.assertEqual(parcel1["symmetric_difference_vertices"], 2)
        self.assertEqual(parcel2["symmetric_difference_vertices"], 2)

    def test_assignment_vs_unassigned_is_distinct(self) -> None:
        a = self.make_artifact("a")
        b = self.make_artifact("b")
        # Parcel 1 has two vertices; dropping one preserves non-zero coverage.
        b["vertex_to_parcel"][0] = None
        b["qualification"]["assigned_vertices_left"] -= 1
        b["qualification"]["unassigned_vertices_left"] += 1
        b["qualification"]["parcel_vertex_counts"][0] -= 1
        body = dict(b)
        del body["content_digest"]
        b["content_digest"] = compiler.sha256_digest(compiler.canonical_json_bytes(body))
        compiler.validate_artifact(b, AREA_ORDER)

        report = crosscheck.build_report(a, b)
        self.assertEqual(report["summary"]["assignment_vs_unassigned"], 1)
        self.assertEqual(report["disagreements"][0]["kind"], "assignment_vs_unassigned")

    def test_hemisphere_census_is_separate(self) -> None:
        a = self.make_artifact("a")
        b = self.make_artifact("b")
        b["vertex_to_parcel"][compiler.VERTICES_PER_HEMISPHERE] = None
        b["qualification"]["assigned_vertices_right"] -= 1
        b["qualification"]["unassigned_vertices_right"] += 1
        b["qualification"]["parcel_vertex_counts"][180] -= 1
        body = dict(b)
        del body["content_digest"]
        b["content_digest"] = compiler.sha256_digest(compiler.canonical_json_bytes(body))
        compiler.validate_artifact(b, AREA_ORDER)
        report = crosscheck.build_report(a, b)
        self.assertEqual(report["hemispheres"]["left"]["disagreement_vertices"], 0)
        self.assertEqual(report["hemispheres"]["right"]["disagreement_vertices"], 1)

    def test_report_write_is_byte_deterministic(self) -> None:
        report = crosscheck.build_report(self.make_artifact("a"), self.make_artifact("b"))
        first = self.root / "first-report.json"
        second = self.root / "second-report.json"
        crosscheck.write_report(first, report)
        crosscheck.write_report(second, report)
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_report_digest_tamper_is_rejected(self) -> None:
        report = crosscheck.build_report(self.make_artifact("a"), self.make_artifact("b"))
        report["content_digest"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(crosscheck.CrosscheckError, "content_digest mismatch"):
            crosscheck.validate_report(report)

    def test_input_artifact_must_validate_first(self) -> None:
        a = self.make_artifact("a")
        a["content_digest"] = "sha256:" + "0" * 64
        path = self.write_json("bad-artifact.json", a)
        with self.assertRaisesRegex(crosscheck.compiler.QualificationError, "content_digest mismatch"):
            crosscheck.load_validated(path, AREA_ORDER)

    def test_parcel_rows_cover_all_360(self) -> None:
        report = crosscheck.build_report(self.make_artifact("a"), self.make_artifact("b"))
        self.assertEqual(len(report["parcels"]), 360)
        self.assertEqual([row["parcel"] for row in report["parcels"]], list(range(1, 361)))

    def test_report_census_tamper_is_rejected_even_with_new_digest(self) -> None:
        report = crosscheck.build_report(self.make_artifact("a"), self.make_artifact("b"))
        report["summary"]["both_unassigned"] -= 1
        body = dict(report)
        del body["content_digest"]
        report["content_digest"] = compiler.sha256_digest(compiler.canonical_json_bytes(body))
        with self.assertRaisesRegex(crosscheck.CrosscheckError, "category census"):
            crosscheck.validate_report(report)

    def test_comparator_never_claims_independence(self) -> None:
        report = crosscheck.build_report(self.make_artifact("a"), self.make_artifact("b"))
        report["qualification"]["independence_established"] = True
        body = dict(report)
        del body["content_digest"]
        report["content_digest"] = compiler.sha256_digest(compiler.canonical_json_bytes(body))
        with self.assertRaisesRegex(crosscheck.CrosscheckError, "cannot establish source independence"):
            crosscheck.validate_report(report)


if __name__ == "__main__":
    unittest.main()
