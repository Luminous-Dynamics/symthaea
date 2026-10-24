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
    _validate_next_url,
)
from tools.water_leak_satellite.verify_inventory import verify_inventory


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
        request_body = json.loads(mock_open.call_args.args[0].data)
        self.assertEqual(request_body["fields"]["exclude"], ["assets"])

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

    def test_limit_above_copernicus_cap_is_rejected(self):
        with self.assertRaises(SystemExit) as raised:
            main([
                "--start", "2025-01-01", "--end", "2025-01-02",
                "--limit", "101", "--out", "unused.json",
            ])
        self.assertEqual(raised.exception.code, 2)

    def test_atomic_write_failure_does_not_leave_partial_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "inventory.json"
            with patch("tools.water_leak_satellite.catalog_inventory.os.replace", side_effect=OSError("simulated replace failure")):
                from tools.water_leak_satellite.catalog_inventory import atomic_write_text
                with self.assertRaises(OSError):
                    atomic_write_text(target, '{"incomplete":')
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(tmp).glob(".inventory.json.*.tmp")), [])

    def test_page_limit_fails_closed_if_catalog_has_next_link(self):
        response = {
            "type": "FeatureCollection", "features": [],
            "links": [{"rel": "next", "href": "https://stac.dataspace.copernicus.eu/v1/search?token=page2", "method": "GET"}],
        }
        with patch("tools.water_leak_satellite.catalog_inventory.urlopen", return_value=FakeResponse(response)):
            with self.assertRaises(CatalogError):
                inventory_collection(
                    endpoint="https://stac.dataspace.copernicus.eu/v1/search",
                    collection="sentinel-1-grd", bbox=[27.7, -26.5, 28.4, -25.8],
                    interval="2025-01-01T00:00:00Z/2025-01-02T23:59:59Z",
                    limit=1, max_pages=1, timeout=2,
                )

    def test_post_next_link_uses_explicit_server_body(self):
        first = {
            "type": "FeatureCollection", "features": [self.feature],
            "links": [{
                "rel": "next", "href": "https://stac.dataspace.copernicus.eu/v1/search",
                "method": "POST",
                "body": {
                    "collections": ["sentinel-1-grd"], "bbox": [27.7, -26.5, 28.4, -25.8],
                    "datetime": "2025-01-01T00:00:00Z/2025-01-02T23:59:59Z",
                    "limit": 1, "next": "opaque-token",
                },
            }],
        }
        second = {"type": "FeatureCollection", "features": [], "links": []}
        with patch(
            "tools.water_leak_satellite.catalog_inventory.urlopen",
            side_effect=[FakeResponse(first), FakeResponse(second)],
        ) as mock_open:
            report = inventory_collection(
                endpoint="https://stac.dataspace.copernicus.eu/v1/search",
                collection="sentinel-1-grd", bbox=[27.7, -26.5, 28.4, -25.8],
                interval="2025-01-01T00:00:00Z/2025-01-02T23:59:59Z",
                limit=1, max_pages=10, timeout=2,
            )
        second_request = mock_open.call_args_list[1].args[0]
        self.assertEqual(json.loads(second_request.data)["next"], "opaque-token")
        self.assertEqual(report["page_count"], 2)

    def test_tokens_and_signed_asset_hrefs_are_not_retained(self):
        self.feature["links"] = [{
            "rel": "self",
            "href": "https://stac.dataspace.copernicus.eu/v1/collections/sentinel-1-grd/items/S1_TEST_001?sig=SELFSECRET",
        }]
        self.feature["assets"] = {
            "vv": {
                "href": "https://example.org/download.tif?token=ASSETSECRET",
                "type": "image/tiff",
                "roles": ["data"],
                "title": "VV",
            }
        }
        response = {"type": "FeatureCollection", "features": [self.feature], "links": []}
        with patch("tools.water_leak_satellite.catalog_inventory.urlopen", return_value=FakeResponse(response)):
            report = inventory_collection(
                endpoint="https://stac.dataspace.copernicus.eu/v1/search",
                collection="sentinel-1-grd", bbox=[27.7, -26.5, 28.4, -25.8],
                interval="2025-01-01T00:00:00Z/2025-01-02T23:59:59Z",
                limit=100, max_pages=10, timeout=2,
            )
        serialized = json.dumps(report)
        self.assertNotIn("SELFSECRET", serialized)
        self.assertNotIn("ASSETSECRET", serialized)
        item = report["items"][0]
        self.assertNotIn("feature", item)
        self.assertEqual(item["self_uri_redacted"], "https://stac.dataspace.copernicus.eu/v1/collections/sentinel-1-grd/items/S1_TEST_001")
        self.assertEqual(item["assets"]["vv"]["type"], "image/tiff")
        self.assertNotIn("href", item["assets"]["vv"])
        self.assertEqual(len(item["item_sha256"]), 64)

    def test_pagination_tokens_are_redacted_in_inventory_output(self):
        first = {
            "type": "FeatureCollection", "features": [],
            "links": [{
                "rel": "next",
                "href": "https://stac.dataspace.copernicus.eu/v1/search?token=PAGINATIONSECRET",
                "method": "GET",
            }],
        }
        second = {"type": "FeatureCollection", "features": [], "links": []}
        with patch("tools.water_leak_satellite.catalog_inventory.urlopen", side_effect=[FakeResponse(first), FakeResponse(second)]):
            report = inventory_collection(
                endpoint="https://stac.dataspace.copernicus.eu/v1/search",
                collection="sentinel-1-grd", bbox=[27.7, -26.5, 28.4, -25.8],
                interval="2025-01-01T00:00:00Z/2025-01-02T23:59:59Z",
                limit=100, max_pages=10, timeout=2,
            )
        serialized = json.dumps(report)
        self.assertNotIn("PAGINATIONSECRET", serialized)
        self.assertEqual(report["page_records"][1]["request_url_redacted"], "https://stac.dataspace.copernicus.eu/v1/search")
        self.assertEqual(len(report["page_records"][1]["request_url_sha256"]), 64)

    def test_rejects_catalog_url_with_userinfo_or_nonstandard_port(self):
        for href in (
            "https://user@stac.dataspace.copernicus.eu/v1/search",
            "https://stac.dataspace.copernicus.eu:444/v1/search",
        ):
            with self.subTest(href=href):
                with self.assertRaises(CatalogError):
                    _validate_next_url("https://stac.dataspace.copernicus.eu/v1/search", href)

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
            self.assertEqual(len(document["inventory_manifest_sha256"]), 64)
            self.assertEqual(len(document["collections"][0]["items"][0]["summary_sha256"]), 64)
            self.assertEqual(verify_inventory(document), [])
            self.assertIn("overall_claim_ceiling", document)


if __name__ == "__main__":
    unittest.main()
