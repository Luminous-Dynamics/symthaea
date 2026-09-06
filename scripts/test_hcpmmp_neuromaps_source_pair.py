#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import hcpmmp_neuromaps_common as c

METHOD = Path(__file__).parents[1] / "data/neuroscience/hcpmmp1_neuromaps_transform_method_v1.json"


def digest(ch: str) -> str:
    return "sha256:" + ch * 64


class SourcePairContractTests(unittest.TestCase):
    def run_doc(self) -> dict:
        method = c.load_method(METHOD)
        inputs = {
            role: {"path": f"/synthetic/{role}", "sha256": digest("a")}
            for role in method["required_inputs"]
        }
        inputs["hcp_left_dlabel"]["sha256"] = digest("1")
        inputs["hcp_right_dlabel"]["sha256"] = digest("2")
        return {
            "schema": c.RUN_SCHEMA,
            "method_manifest_digest": c.digest_file(METHOD),
            "execution_id": "synthetic-source-pair-test",
            "authorization_reference": "synthetic-only",
            "workbench": {
                "path": "/synthetic/wb_command",
                "sha256": digest("b"),
                "version_output_sha256": digest("c"),
            },
            "inputs": inputs,
        }

    def load(self, doc: dict) -> dict:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run.json"
            path.write_text(json.dumps(doc, sort_keys=True))
            return c.load_run(path, c.load_method(METHOD), METHOD)

    def test_exact_wn56_pair_identity_is_pinned(self):
        source = c.load_method(METHOD)["source_atlas"]
        self.assertEqual(source["scene_id"], "WN56")
        self.assertEqual(source["study_id"], "RVVG")
        self.assertEqual(source["left_file_id"], "npz0")
        self.assertEqual(source["right_file_id"], "pkN9")
        self.assertTrue(source["hemisphere_pair_required"])

    def test_distinct_hemisphere_roots_are_accepted(self):
        self.load(self.run_doc())

    def test_identical_hemisphere_roots_are_rejected(self):
        doc = self.run_doc()
        doc["inputs"]["hcp_right_dlabel"]["sha256"] = doc["inputs"]["hcp_left_dlabel"]["sha256"]
        with self.assertRaises(c.ContractError):
            self.load(doc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
