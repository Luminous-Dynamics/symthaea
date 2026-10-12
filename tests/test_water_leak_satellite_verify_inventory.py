"""Offline tests for inventory artefact integrity verification."""
from __future__ import annotations
import hashlib
import unittest
from tools.water_leak_satellite.verify_inventory import canonical_json_bytes, verify_inventory


def digest(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def fixture():
    item = {
        "collection": "sentinel-1-grd",
        "id": "S1-ITEM-1",
        "self_uri_redacted": "https://stac.dataspace.copernicus.eu/v1/items/S1-ITEM-1",
        "item_sha256": "a" * 64,
        "bbox": [27.8, -26.3, 28.0, -26.1],
        "geometry": {"type": "Polygon", "coordinates": []},
        "properties": {"datetime": "2025-01-01T04:00:00Z", "sar:instrument_mode": "IW"},
        "asset_metadata_returned_in_response": False,
        "assets": {},
    }
    item["summary_sha256"] = digest(item)
    collection = {
        "collection": "sentinel-1-grd",
        "query": {
            "collections": ["sentinel-1-grd"],
            "bbox": [27.7, -26.5, 28.4, -25.8],
            "datetime": "2025-01-01T00:00:00Z/2025-01-01T23:59:59Z",
            "limit": 100,
            "fields": {"exclude": ["assets"]},
        },
        "page_count": 1,
        "page_records": [{
            "page": 1,
            "request_method": "POST",
            "request_url_redacted": "https://stac.dataspace.copernicus.eu/v1/search",
            "request_url_sha256": "b" * 64,
            "request_payload_sha256": "c" * 64,
            "response_sha256": "d" * 64,
            "feature_count": 1,
        }],
        "item_count": 1,
        "duplicate_items_dropped": 0,
        "items": [item],
        "catalog_discovery_only": True,
        "interpretation_limit": "metadata discovery only",
    }
    document = {
        "schema_version": 1,
        "generated_at_utc": "2026-10-12T10:00:00Z",
        "catalog_endpoint": "https://stac.dataspace.copernicus.eu/v1/search",
        "scope": {
            "bbox_wgs84": [27.7, -26.5, 28.4, -25.8],
            "start_date_inclusive": "2025-01-01",
            "end_date_inclusive": "2025-01-01",
        },
        "collections": [collection],
        "overall_claim_ceiling": "no leak claim",
    }
    document["inventory_manifest_sha256"] = digest(document)
    return document


class InventoryVerifierTests(unittest.TestCase):
    def test_valid_inventory_passes_integrity_check(self):
        self.assertEqual(verify_inventory(fixture()), [])

    def test_changed_summary_is_detected(self):
        document = fixture()
        document["collections"][0]["items"][0]["properties"]["datetime"] = "2025-02-01T00:00:00Z"
        errors = verify_inventory(document)
        self.assertTrue(any("summary_sha256" in error for error in errors))
        self.assertTrue(any("inventory_manifest_sha256" in error for error in errors))

    def test_changed_scope_is_detected_by_manifest(self):
        document = fixture()
        document["scope"]["bbox_wgs84"] = [27.0, -27.0, 29.0, -25.0]
        self.assertTrue(any("inventory_manifest_sha256" in error for error in verify_inventory(document)))

    def test_href_field_is_rejected_even_if_hashes_are_recomputed(self):
        document = fixture()
        document["collections"][0]["items"][0]["assets"]["vv"] = {"href": "https://example.invalid/x?token=secret"}
        item = document["collections"][0]["items"][0]
        item.pop("summary_sha256")
        item["summary_sha256"] = digest(item)
        document.pop("inventory_manifest_sha256")
        document["inventory_manifest_sha256"] = digest(document)
        self.assertTrue(any("href field" in error for error in verify_inventory(document)))

    def test_page_count_and_digest_shapes_are_checked(self):
        document = fixture()
        document["collections"][0]["page_count"] = 2
        document["collections"][0]["page_records"][0]["response_sha256"] = "not-a-hash"
        errors = verify_inventory(document)
        self.assertTrue(any("page_count" in error for error in errors))
        self.assertTrue(any("response_sha256" in error for error in errors))

    def test_href_outside_item_summary_is_rejected_after_manifest_rehash(self):
        document = fixture()
        document["pagination_extension"] = {
            "links": [{"href": "https://example.invalid/next?token=secret"}]
        }
        document.pop("inventory_manifest_sha256")
        document["inventory_manifest_sha256"] = digest(document)
        errors = verify_inventory(document)
        self.assertTrue(any("inventory: contains an href field" in error for error in errors))

    def test_page_feature_total_must_reconcile_with_items_and_duplicates(self):
        document = fixture()
        document["collections"][0]["page_records"][0]["feature_count"] = 9
        document.pop("inventory_manifest_sha256")
        document["inventory_manifest_sha256"] = digest(document)
        errors = verify_inventory(document)
        self.assertTrue(any("page feature total" in error for error in errors))

    def test_catalogue_query_must_match_declared_aoi_even_after_manifest_rehash(self):
        document = fixture()
        document["collections"][0]["query"]["bbox"] = [27.0, -27.0, 29.0, -25.0]
        document.pop("inventory_manifest_sha256")
        document["inventory_manifest_sha256"] = digest(document)
        errors = verify_inventory(document)
        self.assertTrue(any("query.bbox" in error for error in errors))

    def test_catalogue_query_must_match_declared_dates_even_after_manifest_rehash(self):
        document = fixture()
        document["collections"][0]["query"]["datetime"] = "2024-01-01T00:00:00Z/2024-01-01T23:59:59Z"
        document.pop("inventory_manifest_sha256")
        document["inventory_manifest_sha256"] = digest(document)
        errors = verify_inventory(document)
        self.assertTrue(any("query.datetime" in error for error in errors))

    def test_extremely_large_bbox_integer_fails_closed_without_overflow(self):
        document = fixture()
        document["scope"]["bbox_wgs84"][0] = 10 ** 10000
        errors = verify_inventory(document)
        self.assertTrue(any("scope.bbox_wgs84" in error for error in errors))

    def test_deep_nested_href_is_rejected_without_recursion_failure(self):
        document = fixture()
        nested = {}
        document["extension"] = nested
        for _ in range(1500):
            child = {}
            nested["child"] = child
            nested = child
        nested["href"] = "https://example.invalid/next?token=secret"
        errors = verify_inventory(document)
        self.assertTrue(any("inventory: contains an href field" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
