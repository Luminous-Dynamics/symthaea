import json
import tempfile
import unittest
from pathlib import Path

from scripts.research import st_lucia_r0_asset_plan as plan


def _asset(href: str):
    return {"href": href, "type": "application/octet-stream", "roles": ["data"]}


def _self_link(collection: str, item_id: str, host: str = plan.APPROVED_STAC_HOST):
    return {
        "rel": "self",
        "href": f"https://{host}/v1/collections/{collection}/items/{item_id}",
        "type": "application/geo+json",
    }


def _s3_href(item_id: str, key: str) -> str:
    return f"s3://{plan.APPROVED_S3_BUCKET}/test/{item_id}/{key}"


def _s2_item(item_id: str):
    collection = "sentinel-2-l2a"
    assets = {
        key: _asset(_s3_href(item_id, key))
        for key in plan.S2_SCIENCE_ASSETS + plan.S2_METADATA_ASSETS
    }
    assets["thumbnail"] = _asset(_s3_href(item_id, "thumbnail"))
    assets["Product"] = _asset(_s3_href(item_id, "Product"))
    return {
        "type": "Feature",
        "id": item_id,
        "collection": collection,
        "properties": {},
        "links": [_self_link(collection, item_id)],
        "assets": assets,
    }


def _s1_item(item_id: str):
    collection = "sentinel-1-grd"
    assets = {
        key: _asset(_s3_href(item_id, key))
        for key in plan.S1_SCIENCE_ASSETS + plan.S1_METADATA_ASSETS
    }
    assets["thumbnail"] = _asset(_s3_href(item_id, "thumbnail"))
    assets["Product"] = _asset(_s3_href(item_id, "Product"))
    return {
        "type": "Feature",
        "id": item_id,
        "collection": collection,
        "properties": {},
        "links": [_self_link(collection, item_id)],
        "assets": assets,
    }


class AssetPlannerTests(unittest.TestCase):
    def test_asset_key_sets_never_include_preview_or_bulk_product(self):
        selected = set(plan.S2_SCIENCE_ASSETS + plan.S2_METADATA_ASSETS + plan.S1_SCIENCE_ASSETS + plan.S1_METADATA_ASSETS)
        self.assertTrue(plan.FORBIDDEN_KEYS.isdisjoint(selected))
        self.assertIn("SCL_20m", selected)
        self.assertIn("schema-calibration-vv", selected)

    def test_realistic_eodata_s3_locator_is_preserved_and_decomposed(self):
        item = _s2_item(plan.EXPECTED_S2_IDS[0])
        href = "s3://eodata/Sentinel-2/MSI/L2A/2026/07/01/example.SAFE/GRANULE/example/IMG_DATA/R10m/B03_10m.jp2"
        item["assets"]["B03_10m"]["href"] = href
        entry = plan.asset_entry(item, "B03_10m", "science-payload")
        self.assertEqual(href, entry["stac_href"])
        self.assertEqual(href, entry["href"])
        self.assertEqual("s3", entry["access_method"])
        self.assertEqual(plan.APPROVED_S3_ENDPOINT, entry["s3_endpoint"])
        self.assertEqual("eodata", entry["s3_bucket"])
        self.assertEqual(
            "Sentinel-2/MSI/L2A/2026/07/01/example.SAFE/GRANULE/example/IMG_DATA/R10m/B03_10m.jp2",
            entry["s3_key"],
        )
        self.assertIsNone(entry["href_resolution_base"])

    def test_s3_wrong_bucket_fails_closed(self):
        item = _s2_item(plan.EXPECTED_S2_IDS[0])
        item["assets"]["B03_10m"]["href"] = "s3://evil-bucket/path/B03.jp2"
        with self.assertRaises(plan.PlanError):
            plan.asset_entry(item, "B03_10m", "science-payload")

    def test_s3_query_or_fragment_fails_closed(self):
        item = _s2_item(plan.EXPECTED_S2_IDS[0])
        item["assets"]["B03_10m"]["href"] = "s3://eodata/path/B03.jp2?x=1"
        with self.assertRaises(plan.PlanError):
            plan.asset_entry(item, "B03_10m", "science-payload")
        item["assets"]["B03_10m"]["href"] = "s3://eodata/path/B03.jp2#fragment"
        with self.assertRaises(plan.PlanError):
            plan.asset_entry(item, "B03_10m", "science-payload")

    def test_s3_empty_key_or_repeated_slash_fails_closed(self):
        item = _s2_item(plan.EXPECTED_S2_IDS[0])
        item["assets"]["B03_10m"]["href"] = "s3://eodata/"
        with self.assertRaises(plan.PlanError):
            plan.asset_entry(item, "B03_10m", "science-payload")
        item["assets"]["B03_10m"]["href"] = "s3://eodata//Sentinel-2/B03.jp2"
        with self.assertRaises(plan.PlanError):
            plan.asset_entry(item, "B03_10m", "science-payload")
        item["assets"]["B03_10m"]["href"] = "s3://eodata/Sentinel-2//B03.jp2"
        with self.assertRaises(plan.PlanError):
            plan.asset_entry(item, "B03_10m", "science-payload")

    def test_asset_entry_rejects_http(self):
        item = _s2_item(plan.EXPECTED_S2_IDS[0])
        item["assets"]["B03_10m"]["href"] = "http://stac.dataspace.copernicus.eu/B03"
        with self.assertRaises(plan.PlanError):
            plan.asset_entry(item, "B03_10m", "science-payload")

    def test_asset_entry_rejects_absolute_off_origin(self):
        item = _s2_item(plan.EXPECTED_S2_IDS[0])
        item["assets"]["B03_10m"]["href"] = "https://evil.invalid/B03"
        with self.assertRaises(plan.PlanError):
            plan.asset_entry(item, "B03_10m", "science-payload")

    def test_relative_href_resolves_against_unique_item_self_link(self):
        item = _s2_item(plan.EXPECTED_S2_IDS[0])
        item["assets"]["B03_10m"]["href"] = "./assets/B03_10m.tif"
        entry = plan.asset_entry(item, "B03_10m", "science-payload")
        self.assertEqual("./assets/B03_10m.tif", entry["stac_href"])
        self.assertEqual(
            f"https://{plan.APPROVED_STAC_HOST}/v1/collections/sentinel-2-l2a/items/assets/B03_10m.tif",
            entry["href"],
        )
        self.assertEqual("https", entry["access_method"])
        self.assertEqual(item["links"][0]["href"], entry["href_resolution_base"])
        self.assertIsNone(entry["s3_bucket"])

    def test_relative_href_without_self_link_fails_closed(self):
        item = _s2_item(plan.EXPECTED_S2_IDS[0])
        item["links"] = []
        item["assets"]["B03_10m"]["href"] = "./B03_10m.tif"
        with self.assertRaises(plan.PlanError):
            plan.asset_entry(item, "B03_10m", "science-payload")

    def test_relative_href_with_off_origin_self_link_fails_closed(self):
        item = _s2_item(plan.EXPECTED_S2_IDS[0])
        item["links"] = [_self_link("sentinel-2-l2a", item["id"], host="evil.invalid")]
        item["assets"]["B03_10m"]["href"] = "./B03_10m.tif"
        with self.assertRaises(plan.PlanError):
            plan.asset_entry(item, "B03_10m", "science-payload")

    def test_scheme_relative_off_origin_escape_fails_closed(self):
        item = _s2_item(plan.EXPECTED_S2_IDS[0])
        item["assets"]["B03_10m"]["href"] = "//evil.invalid/B03_10m.tif"
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
        self.assertTrue(all(row["access_method"] == "s3" for row in entries))
        self.assertTrue(all(row["s3_bucket"] == plan.APPROVED_S3_BUCKET for row in entries))
        self.assertTrue(all(row["s3_endpoint"] == plan.APPROVED_S3_ENDPOINT for row in entries))


if __name__ == "__main__":
    unittest.main()
