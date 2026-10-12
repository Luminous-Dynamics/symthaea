"""Offline tests for the catalog inventory utility; no network is used."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from datetime import date

from tools.water_leak_satellite.catalog_inventory import (
    CatalogError,
    date_range,
    inventory_collection,
    main,
    parse_bbox,
)


class FakeResponse:
    def __init__(self, value: dict):
        self.raw = json.dumps(value, separators=(",", ":")).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.raw


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.feature = {
            "type": "Feature",
            "id": "S1_TEST_001",
            "collection": "sentinel-1-grd",
            "bbox": [27.8, -26.3, 28.0, -26.1],
            "properties": {"datetime": "2025-01-01T04:00:00Z", "sar:instrument_mode": "IW"},
            "geometry": {"type": "Polygon", "coordinates": []},
            "assets": {},
            "links": [],
        }

    def test_bbox_validation(self):
        self.assertEqual(parse_bbox("27.7,-26.5,28.4,-25.8"), [27.7, -26.5, 28.4, -25.8])
        with self.assertRaises(Exception):
            parse_bbox("27,-26,28")
        with self.assertRaises(Exception):
            parse_bbox("28,-26,27,-25")

    def test_date_range_is_utc_and_inclusive(self):
        self.assertEqual(
            date_range(date(2025, 1, 1), date(2025, 1, 2)),
            "2025-01-01T00:00:00Z/2025-01-02T23:59:59Z",
        )

    def test_one_page_inventory_has_hashes_and_claim_ceiling(self):
        response = {"type": "FeatureCollection", "features": [self.feature], "links": []}
        with patch("tools.water_leak_satellite.catalog_inventory.urlopen", return_value=FakeResponse(response)) as mock_open:
            report = inventory_collection(
                endpoint="https://stac.dataspace.copernicus.eu/v1/search",
                collection="sentinel-1-grd", bbox=[27.7, -26.5, 28.4, -25.8],
                interval="2025-01-01T00:00:00Z/2025-01-02T23:59:59Z",
                limit=100, max_pages=10, timeout=2,
            )
        self.assertEqual(report["item_count"], 1)
        self.assertEqual(len(report["items"][0]["item_sha256"]), 64)
        self.assertEqual(len(report["page_records"][0]["response_sha256"]), 64)
        self.assertTrue(report["catalog_discovery_only"])
        self.assertEqual(mock_open.call_count, 1)

    def test_next_page_get_link_is_followed_and_duplicates_are_dropped(self):
        first = {
            "type": "FeatureCollection", "features": [self.feature],
            "links": [{"rel": "next", "href": "https://stac.dataspace.copernicus.eu/v1/search?token=page2", "method": "GET"}],
        }
        second = {
            "type": "FeatureCollection", "features": [self.feature], "links": [],
        }
        with patch("tools.water_leak_satellite.catalog_inventory.urlopen", side_effect=[FakeResponse(first), FakeResponse(second)]) as mock_open:
            report = inventory_collection(
                endpoint="https://stac.dataspace.copernicus.eu/v1/search",
                collection="sentinel-1-grd", bbox=[27.7, -26.5, 28.4, -25.8],
                interval="2025-01-01T00:00:00Z/2025-01-02T23:59:59Z",
                limit=1, max_pages=10, timeout=2,
            )
        self.assertEqual(mock_open.call_count, 2)
        self.assertEqual(report["page_count"], 2)
        self.assertEqual(report["item_count"], 1)
        self.assertEqual(report["duplicate_items_dropped"], 1)

    def test_rejects_external_next_link(self):
        response = {
            "type": "FeatureCollection", "features": [],
            "links": [{"rel": "next", "href": "https://example.com/steal", "method": "GET"}],
        }
        with patch("tools.water_leak_satellite.catalog_inventory.urlopen", return_value=FakeResponse(response)):
            with self.assertRaises(CatalogError):
                inventory_collection(
                    endpoint="https://stac.dataspace.copernicus.eu/v1/search",
                    collection="sentinel-1-grd", bbox=[27.7, -26.5, 28.4, -25.8],
                    interval="2025-01-01T00:00:00Z/2025-01-02T23:59:59Z",
                    limit=100, max_pages=10, timeout=2,
                )

    def test_main_writes_auditable_json_with_mock_catalog(self):
        response = {"type": "FeatureCollection", "features": [self.feature], "links": []}
        with tempfile.TemporaryDirectory() as tmp, patch(
            "tools.water_leak_satellite.catalog_inventory.urlopen", return_value=FakeResponse(response)
        ):
            out = Path(tmp) / "inventory.json"
            code = main([
                "--start", "2025-01-01", "--end", "2025-01-02",
                "--collection", "sentinel-1-grd", "--out", str(out),
            ])
            self.assertEqual(code, 0)
            document = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(document["collections"][0]["item_count"], 1)
            self.assertIn("overall_claim_ceiling", document)


if __name__ == "__main__":
    unittest.main()
