#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import st_lucia_stac_replay_v1_2 as r  # noqa: E402


def s2_item(item_id: str, when: str, bbox=None):
    return {
        "type": "Feature",
        "id": item_id,
        "collection": "sentinel-2-l2a",
        "bbox": [32.0, -28.5, 33.0, -27.5] if bbox is None else bbox,
        "properties": {"datetime": when, "eo:cloud_cover": 1.0},
        "assets": {
            "B03_10m": {},
            "B04_10m": {},
            "B08_10m": {},
            "B11_20m": {},
            "B12_20m": {},
        },
    }


class AcquisitionSetReplayTests(unittest.TestCase):
    def test_same_earliest_acquisition_keeps_all_tiles(self):
        items = [
            s2_item("tile-z", "2026-07-01T07:36:11.025000Z"),
            s2_item("tile-a", "2026-07-01T07:36:11.025000Z"),
            s2_item("later", "2026-07-03T07:36:11.025000Z"),
        ]
        selected, _audit, eligible = r.select_s2_acquisition_set(items)
        self.assertEqual([item["id"] for item in selected], ["tile-a", "tile-z"])
        self.assertEqual(eligible, ["tile-a", "tile-z", "later"])

    def test_later_clearer_scene_cannot_replace_earliest_acquisition(self):
        earliest = s2_item("earliest", "2026-07-01T07:36:11.025000Z")
        earliest["properties"]["eo:cloud_cover"] = 19.0
        later = s2_item("later", "2026-07-02T07:36:11.025000Z")
        later["properties"]["eo:cloud_cover"] = 0.0
        selected, _audit, _eligible = r.select_s2_acquisition_set([later, earliest])
        self.assertEqual([item["id"] for item in selected], ["earliest"])

    def test_missing_bbox_is_ineligible(self):
        item = s2_item("missing", "2026-07-01T07:36:11.025000Z")
        del item["bbox"]
        selected, audit, _eligible = r.select_s2_acquisition_set([item])
        self.assertEqual(selected, [])
        self.assertIn("missing-or-invalid-bbox", audit[0]["ineligibility_reasons"])

    def test_nonintersecting_bbox_is_ineligible(self):
        item = s2_item(
            "outside",
            "2026-07-01T07:36:11.025000Z",
            bbox=[10.0, -10.0, 11.0, -9.0],
        )
        selected, audit, _eligible = r.select_s2_acquisition_set([item])
        self.assertEqual(selected, [])
        self.assertIn("bbox-does-not-intersect-frozen-aoi", audit[0]["ineligibility_reasons"])

    def test_three_dimensional_stac_bbox_is_supported(self):
        item = s2_item(
            "3d",
            "2026-07-01T07:36:11.025000Z",
            bbox=[32.0, -28.5, 0.0, 33.0, -27.5, 100.0],
        )
        selected, _audit, _eligible = r.select_s2_acquisition_set([item])
        self.assertEqual([entry["id"] for entry in selected], ["3d"])


if __name__ == "__main__":
    unittest.main()
