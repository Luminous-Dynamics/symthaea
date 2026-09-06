#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("extract_fsaverage_hcpmmp1_semantic_labels.py")
SPEC = importlib.util.spec_from_file_location("extractor", SCRIPT)
assert SPEC and SPEC.loader
extractor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = extractor
SPEC.loader.exec_module(extractor)

N = extractor.FSAVERAGE_VERTICES_PER_HEMISPHERE
N5 = extractor.FSAVERAGE5_VERTICES_PER_HEMISPHERE
AREAS = [f"A{i:03d}" for i in range(1, 181)]


def pack_id(i: int) -> tuple[int, tuple[int, int, int, int]]:
    r = (i + 1) & 0xFF
    g = ((i + 1) >> 8) & 0xFF
    b = ((i + 1) >> 16) & 0xFF
    t = 0
    return r + (g << 8) + (b << 16), (r, g, b, t)


def bstr(text: str) -> bytes:
    raw = text.encode() + b"\x00"
    return struct.pack(">i", len(raw)) + raw


def make_annot(
    hemi: str = "left",
    *,
    version: int = 2,
    unknown_name: str = "???",
    wrong_name_at: int | None = None,
    missing_id_vertex: int | None = None,
    duplicate_vertex: bool = False,
    trailing: bytes = b"",
    omit_area: str | None = None,
) -> bytes:
    prefix = "L_" if hemi == "left" else "R_"
    names = [unknown_name] + [f"{prefix}{a}_ROI" for a in AREAS]
    ids_rgba = [pack_id(i) for i in range(len(names))]
    ids = [x[0] for x in ids_rgba]
    rgba = [x[1] for x in ids_rgba]
    if wrong_name_at is not None:
        names[wrong_name_at] = "X_BAD_ROI"

    vertex_ids = []
    for v in range(N):
        if v == 0:
            idx = 0
        else:
            idx = 1 + ((v - 1) % 180)
        if omit_area is not None and v < N5 and AREAS[idx - 1] == omit_area:
            idx = 1
        vertex_ids.append(ids[idx])
    if missing_id_vertex is not None:
        vertex_ids[missing_id_vertex] = 0x7F7E7D

    out = bytearray()
    out += struct.pack(">i", N)
    for v, label_id in enumerate(vertex_ids):
        vertex = v
        if duplicate_vertex and v == N - 1:
            vertex = N - 2
        out += struct.pack(">ii", vertex, label_id)
    out += struct.pack(">i", 1)

    if version == 1:
        out += struct.pack(">i", len(names))
        out += bstr("synthetic.ctab")
        for name, color in zip(names, rgba):
            out += bstr(name)
            out += struct.pack(">iiii", *color)
    elif version == 2:
        out += struct.pack(">i", -2)
        out += struct.pack(">i", len(names))
        out += bstr("synthetic.ctab")
        out += struct.pack(">i", len(names))
        for structure, (name, color) in enumerate(zip(names, rgba)):
            out += struct.pack(">i", structure)
            out += bstr(name)
            out += struct.pack(">iiii", *color)
    else:
        raise ValueError(version)
    out += trailing
    return bytes(out)


def write_area_order(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": extractor.AREA_SCHEMA,
                "atlas": "HCP-MMP1.0/Glasser360",
                "hemisphere_area_count": 180,
                "source": {
                    "repository": "synthetic/test",
                    "commit": "0" * 40,
                    "blob_sha": "1" * 40,
                    "path": "synthetic.xml",
                    "purpose": "test",
                },
                "areas": AREAS,
            }
        ),
        encoding="utf-8",
    )


def write_manifest(path: Path, lh: Path, rh: Path, *, bad_left_md5: bool = False) -> None:
    def md5(p: Path) -> str:
        return hashlib.md5(p.read_bytes(), usedforsecurity=False).hexdigest()

    path.write_text(
        json.dumps(
            {
                "schema": extractor.MANIFEST_SCHEMA,
                "lineage_id": "synthetic-lineage",
                "source_version": "v1",
                "atlas": "HCP-MMP1.0/Glasser360",
                "input_space": "fsaverage",
                "license": "test-only",
                "terms_reference": "https://example.invalid/terms",
                "acknowledgement_required": True,
                "files": {
                    "left": {
                        "url": "https://example.invalid/lh.annot",
                        "md5": "0" * 32 if bad_left_md5 else md5(lh),
                        "expected_vertices": N,
                    },
                    "right": {
                        "url": "https://example.invalid/rh.annot",
                        "md5": md5(rh),
                        "expected_vertices": N,
                    },
                },
                "provenance": {
                    "article_doi": "10.test/example",
                    "source_description": "synthetic contract fixture",
                    "downsample_rule": "first 10242 vertices",
                },
            }
        ),
        encoding="utf-8",
    )


class ExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.lh = self.root / "lh.annot"
        self.rh = self.root / "rh.annot"
        self.area = self.root / "areas.json"
        self.manifest = self.root / "lineage.json"
        self.lh.write_bytes(make_annot("left"))
        self.rh.write_bytes(make_annot("right"))
        write_area_order(self.area)
        write_manifest(self.manifest, self.lh, self.rh)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_version2_parses_and_extracts_nested_fsaverage5(self):
        doc = extractor.extract_semantic_labels(self.lh, "left", self.area, self.manifest)
        self.assertEqual(doc["vertex_count"], N5)
        self.assertIsNone(doc["labels"][0])
        self.assertEqual(doc["labels"][1], "L_A001")
        self.assertEqual(doc["labels"][180], "L_A180")
        self.assertEqual(len(doc["labels"]), N5)

    def test_version1_is_supported(self):
        self.lh.write_bytes(make_annot("left", version=1))
        write_manifest(self.manifest, self.lh, self.rh)
        doc = extractor.extract_semantic_labels(self.lh, "left", self.area, self.manifest)
        self.assertEqual(doc["labels"][2], "L_A002")

    def test_right_hemisphere_semantics(self):
        doc = extractor.extract_semantic_labels(self.rh, "right", self.area, self.manifest)
        self.assertEqual(doc["labels"][1], "R_A001")
        self.assertEqual(doc["labels"][180], "R_A180")

    def test_source_hash_mismatch_fails(self):
        write_manifest(self.manifest, self.lh, self.rh, bad_left_md5=True)
        with self.assertRaisesRegex(extractor.ExtractionError, "MD5 mismatch"):
            extractor.extract_semantic_labels(self.lh, "left", self.area, self.manifest)

    def test_wrong_hemisphere_name_fails(self):
        self.lh.write_bytes(make_annot("right"))
        write_manifest(self.manifest, self.lh, self.rh)
        with self.assertRaisesRegex(extractor.ExtractionError, "unexpected HCP-MMP1"):
            extractor.extract_semantic_labels(self.lh, "left", self.area, self.manifest)

    def test_unknown_color_table_name_is_not_unassigned(self):
        self.lh.write_bytes(make_annot("left", wrong_name_at=1))
        write_manifest(self.manifest, self.lh, self.rh)
        with self.assertRaisesRegex(extractor.ExtractionError, "unexpected HCP-MMP1"):
            extractor.extract_semantic_labels(self.lh, "left", self.area, self.manifest)

    def test_only_question_marks_becomes_unassigned(self):
        self.lh.write_bytes(make_annot("left", unknown_name="unknown"))
        write_manifest(self.manifest, self.lh, self.rh)
        with self.assertRaisesRegex(extractor.ExtractionError, "unexpected HCP-MMP1"):
            extractor.extract_semantic_labels(self.lh, "left", self.area, self.manifest)

    def test_all_180_areas_must_survive_fsaverage5_subset(self):
        self.lh.write_bytes(make_annot("left", omit_area="A180"))
        write_manifest(self.manifest, self.lh, self.rh)
        with self.assertRaisesRegex(extractor.ExtractionError, "empty HCP-MMP1 areas"):
            extractor.extract_semantic_labels(self.lh, "left", self.area, self.manifest)

    def test_missing_color_table_id_fails(self):
        self.lh.write_bytes(make_annot("left", missing_id_vertex=3))
        write_manifest(self.manifest, self.lh, self.rh)
        with self.assertRaisesRegex(extractor.ExtractionError, "missing from color table"):
            extractor.extract_semantic_labels(self.lh, "left", self.area, self.manifest)

    def test_duplicate_vertex_fails(self):
        self.lh.write_bytes(make_annot("left", duplicate_vertex=True))
        write_manifest(self.manifest, self.lh, self.rh)
        with self.assertRaisesRegex(extractor.ExtractionError, "duplicate vertex index"):
            extractor.extract_semantic_labels(self.lh, "left", self.area, self.manifest)

    def test_trailing_bytes_fail_closed(self):
        self.lh.write_bytes(make_annot("left", trailing=b"junk"))
        write_manifest(self.manifest, self.lh, self.rh)
        with self.assertRaisesRegex(extractor.ExtractionError, "trailing bytes"):
            extractor.extract_semantic_labels(self.lh, "left", self.area, self.manifest)

    def test_output_bytes_are_deterministic(self):
        doc1 = extractor.extract_semantic_labels(self.lh, "left", self.area, self.manifest)
        doc2 = extractor.extract_semantic_labels(self.lh, "left", self.area, self.manifest)
        self.assertEqual(extractor.canonical_json_bytes(doc1), extractor.canonical_json_bytes(doc2))

    def test_manifest_is_closed_world(self):
        doc = json.loads(self.manifest.read_text())
        doc["surprise"] = True
        self.manifest.write_text(json.dumps(doc))
        with self.assertRaisesRegex(extractor.ExtractionError, "unknown fields"):
            extractor.load_manifest(self.manifest)

    def test_bad_vertex_count_fails(self):
        data = bytearray(self.lh.read_bytes())
        data[:4] = struct.pack(">i", N5)
        self.lh.write_bytes(data)
        write_manifest(self.manifest, self.lh, self.rh)
        with self.assertRaisesRegex(extractor.ExtractionError, "vertex count"):
            extractor.extract_semantic_labels(self.lh, "left", self.area, self.manifest)


if __name__ == "__main__":
    unittest.main()
