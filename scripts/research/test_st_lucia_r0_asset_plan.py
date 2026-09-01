import json
import tempfile
import unittest
from pathlib import Path

from scripts.research import st_lucia_r0_asset_plan as plan


def _asset(href: str):
    return {"href": href, "type": "application/octet-stream", "roles": ["data"]}


def _s2_item(item_id: str):
    assets = {key: _asset(f"https://example.invalid/{item_id}/{key}") for key in plan.S2_SCIENCE_ASSETS + plan.S2_METADATA_ASSETS}
    assets["thumbnail"] = _asset(f"https://example.invalid/{item_id}/thumbnail")
    assets["Product"] = _asset(f"https://example.invalid/{item_id}/Product")
    return {"type": "Feature", "id": item_id, "collection": "sentinel-2-l2a", "properties": {}, "assets": assets}


def _s1_item(item_id: str):
    assets = {key: _asset(f"https://example.invalid/{item_id}/{key}") for key in plan.S1_SCIENCE_ASSETS + plan.S1_METADATA_ASSETS}
    assets["thumbnail"] = _asset(f"https://example.invalid/{item_id}/thumbnail")
    assets["Product"] = _asset(f"https://example.invalid/{item_id}/Product")
    return {"type": "Feature", "id": item_id, "collection": "sentinel-1-grd", "properties": {}, "assets": assets}


class AssetPlannerTests(unittest.TestCase):
    def test_asset_key_sets_never_include_preview_or_bulk_product(self):
        selected = set(plan.S2_SCIENCE_ASSETS + plan.S2_METADATA_ASSETS + plan.S1_SCIENCE_ASSETS + plan.S1_METADATA_ASSETS)
        self.assertTrue(plan.FORBIDDEN_KEYS.isdisjoint(selected))
        self.assertIn("SCL_20m", selected)
        self.assertIn("schema-calibration-vv", selected)

    def test_asset_entry_requires_https(self):
        item = _s2_item(plan.EXPECTED_S2_IDS[0])
        item["assets"]["B03_10m"]["href"] = "http://example.invalid/B03"
        with self.assertRaises(plan.PlanError):
            plan.asset_entry(item, "B03_10m", "science-payload")

    def test_missing_required_asset_fails_closed(self):
        item = _s2_item(plan.EXPECTED_S2_IDS[0])
        del item["assets"]["B12_20m"]
        with self.assertRaises(plan.PlanError):
            plan.asset_entry(item, "B12_20m", "science-payload")

    def test_forbidden_asset_cannot_be_requested(self):
        item = _s2_item(plan.EXPECTED_S2_IDS[0])
        with self.assertRaises(plan.PlanError):
            plan.asset_entry(item, "thumbnail", "science-payload")
        with self.assertRaises(plan.PlanError):
            plan.asset_entry(item, "Product", "science-payload")

    def test_index_rejects_conflicting_duplicate_selected_item(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _s2_item(plan.EXPECTED_S2_IDS[0])
            second = _s2_item(plan.EXPECTED_S2_IDS[0])
            second["properties"] = {"changed": True}
            (root / "page-0001.json").write_text(json.dumps({"features": [first]}), encoding="utf-8")
            (root / "page-0002.json").write_text(json.dumps({"features": [second]}), encoding="utf-8")
            with self.assertRaises(plan.PlanError):
                plan.index_items([root / "page-0001.json", root / "page-0002.json"])

    def test_deterministic_entries_cover_exact_frozen_counts(self):
        entries = []
        for item_id in plan.EXPECTED_S2_IDS:
            item = _s2_item(item_id)
            for key in plan.S2_SCIENCE_ASSETS:
                entries.append(plan.asset_entry(item, key, "science-payload"))
            for key in plan.S2_METADATA_ASSETS:
                entries.append(plan.asset_entry(item, key, "provenance-metadata"))
        s1 = _s1_item(plan.EXPECTED_S1_ID)
        for key in plan.S1_SCIENCE_ASSETS:
            entries.append(plan.asset_entry(s1, key, "science-payload"))
        for key in plan.S1_METADATA_ASSETS:
            entries.append(plan.asset_entry(s1, key, "calibration-provenance"))
        self.assertEqual(29, len(entries))
        self.assertFalse(any(row["asset_key"] in plan.FORBIDDEN_KEYS for row in entries))


if __name__ == "__main__":
    unittest.main()
