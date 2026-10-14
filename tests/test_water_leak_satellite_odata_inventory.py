"""Offline tests for the Copernicus OData cross-check; no network is used."""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from tools.water_leak_satellite.odata_inventory import (
    DEFAULT_ENDPOINT,
    ODataError,
    PROFILES,
    build_document,
    build_filter,
    inventory_query,
    main,
    parse_bbox,
    query_windows,
    validate_next_url,
    verify_inventory,
)


class FakeResponse:
    def __init__(self, value):
        self.raw = json.dumps(value, separators=(",", ":")).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.raw


class ODataInventoryTests(unittest.TestCase):
    def setUp(self):
        self.bbox = [27.7, -26.5, 28.4, -25.8]
        self.profile = PROFILES[0]
        self.product = {
            "Id": "odata-product-001",
            "Name": "S1A_IW_GRDH_1SDV_20250101T040000.SAFE",
            "ContentDate": {"Start": "2025-01-01T04:00:00.000Z", "End": "2025-01-01T04:00:25.000Z"},
            "GeoFootprint": {"type": "Polygon", "coordinates": [[[27.8, -26.3], [28.0, -26.3], [28.0, -26.1], [27.8, -26.1], [27.8, -26.3]]]},
            "Online": True,
            "ContentLength": 123456,
            "Checksum": {"Algorithm": "MD5", "Value": "abcd"},
        }

    def test_bbox_is_finite_ordered_wgs84(self):
        self.assertEqual(parse_bbox("27.7,-26.5,28.4,-25.8"), self.bbox)
        for invalid in ("1,2,3", "28,-26,27,-25", "nan,-26,28,-25", "0,-91,1,90"):
            with self.subTest(invalid=invalid), self.assertRaises(Exception):
                parse_bbox(invalid)

    def test_quarter_partition_uses_half_open_windows_covering_inclusive_dates(self):
        self.assertEqual(
            query_windows(date(2023, 11, 30), date(2025, 1, 2)),
            [
                (date(2023, 11, 30), date(2024, 1, 1)),
                (date(2024, 1, 1), date(2024, 4, 1)),
                (date(2024, 4, 1), date(2024, 7, 1)),
                (date(2024, 7, 1), date(2024, 10, 1)),
                (date(2024, 10, 1), date(2025, 1, 1)),
                (date(2025, 1, 1), date(2025, 1, 3)),
            ],
        )
        self.assertEqual(
            query_windows(date(2023, 11, 30), date(2025, 1, 2), window_months=12),
            [
                (date(2023, 11, 30), date(2024, 1, 1)),
                (date(2024, 1, 1), date(2025, 1, 1)),
                (date(2025, 1, 1), date(2025, 1, 3)),
            ],
        )
        with self.assertRaises(ODataError):
            query_windows(date(2025, 1, 1), date(2025, 1, 2), window_months=2)

    def test_filter_contains_closed_polygon_product_and_half_open_dates(self):
        text = build_filter(self.profile, self.bbox, date(2025, 1, 1), date(2025, 2, 1))
        self.assertIn("Collection/Name eq 'SENTINEL-1'", text)
        self.assertIn("Value eq 'IW_GRDH_1S'", text)
        self.assertIn("POLYGON((27.70000000 -26.50000000", text)
        self.assertTrue(text.rstrip().endswith("2025-02-01T00:00:00.000Z"))

    def test_single_page_inventory_and_integrity_verifier(self):
        response = {"value": [self.product]}
        with patch("tools.water_leak_satellite.odata_inventory.urlopen", return_value=FakeResponse(response)):
            query = inventory_query(
                profile=self.profile, bbox=self.bbox, start=date(2025, 1, 1),
                end_exclusive=date(2025, 1, 2), limit=100, max_pages=5, timeout=2,
            )
        self.assertEqual(query["item_count"], 1)
        self.assertEqual(query["page_records"][0]["feature_count"], 1)
        self.assertEqual(query["page_records"][0]["request_attempts"], 1)
        self.assertEqual(len(query["items"][0]["source_product_sha256"]), 64)
        self.assertEqual(len(query["items"][0]["summary_sha256"]), 64)
        doc = build_document(self.bbox, date(2025, 1, 1), date(2025, 1, 1), [query])
        # The standalone query is a deliberately partial fixture: the full-document
        # verifier requires both product profiles for every partition.
        self.assertEqual(verify_inventory(doc), [
            "queries: partitioned query set does not exactly cover both declared profiles and date scope"
        ])

    def test_transient_timeout_retries_are_bounded_and_recorded(self):
        response = {"value": [self.product]}
        with (
            patch(
                "tools.water_leak_satellite.odata_inventory.urlopen",
                side_effect=[TimeoutError("simulated timeout"), FakeResponse(response)],
            ) as open_mock,
            patch("tools.water_leak_satellite.odata_inventory.time.sleep") as sleep_mock,
        ):
            query = inventory_query(
                profile=self.profile,
                bbox=self.bbox,
                start=date(2025, 1, 1),
                end_exclusive=date(2025, 1, 2),
                limit=100,
                max_pages=5,
                timeout=2,
                max_retries=2,
            )
        self.assertEqual(open_mock.call_count, 2)
        sleep_mock.assert_called_once()
        self.assertEqual(query["item_count"], 1)
        self.assertEqual(query["page_records"][0]["request_attempts"], 2)

    def test_transient_timeout_exhaustion_fails_closed_not_as_zero_results(self):
        with (
            patch(
                "tools.water_leak_satellite.odata_inventory.urlopen",
                side_effect=[
                    TimeoutError("simulated timeout"),
                    TimeoutError("simulated timeout"),
                    TimeoutError("simulated timeout"),
                ],
            ) as open_mock,
            patch("tools.water_leak_satellite.odata_inventory.time.sleep"),
            self.assertRaisesRegex(ODataError, "after 3 attempt"),
        ):
            inventory_query(
                profile=self.profile,
                bbox=self.bbox,
                start=date(2025, 1, 1),
                end_exclusive=date(2025, 1, 2),
                limit=100,
                max_pages=5,
                timeout=2,
                max_retries=2,
            )
        self.assertEqual(open_mock.call_count, 3)

    def test_retry_count_above_policy_ceiling_is_rejected(self):
        with self.assertRaises(ODataError):
            inventory_query(
                profile=self.profile,
                bbox=self.bbox,
                start=date(2025, 1, 1),
                end_exclusive=date(2025, 1, 2),
                limit=100,
                max_pages=5,
                timeout=2,
                max_retries=6,
            )

    def test_duplicate_products_are_counted_and_dropped(self):
        response = {"value": [self.product], "@odata.nextLink": DEFAULT_ENDPOINT + "?%24skip=100"}
        second = {"value": [self.product]}
        with patch("tools.water_leak_satellite.odata_inventory.urlopen", side_effect=[FakeResponse(response), FakeResponse(second)]):
            query = inventory_query(
                profile=self.profile, bbox=self.bbox, start=date(2025, 1, 1),
                end_exclusive=date(2025, 1, 2), limit=1, max_pages=5, timeout=2,
            )
        self.assertEqual(query["item_count"], 1)
        self.assertEqual(query["duplicate_items_dropped"], 1)
        self.assertEqual(sum(p["feature_count"] for p in query["page_records"]), 2)

    def test_external_pagination_target_is_rejected(self):
        with self.assertRaises(ODataError):
            validate_next_url(DEFAULT_ENDPOINT, "https://example.invalid/steal")

    def test_wrong_host_nonstandard_port_and_wrong_path_are_rejected(self):
        for url in (
            "https://user@catalogue.dataspace.copernicus.eu/odata/v1/Products",
            "https://catalogue.dataspace.copernicus.eu:444/odata/v1/Products",
            "https://catalogue.dataspace.copernicus.eu/odata/v1/Users",
        ):
            with self.subTest(url=url), self.assertRaises(ODataError):
                validate_next_url(DEFAULT_ENDPOINT, url)

    def test_page_limit_fails_closed_if_next_link_remains(self):
        response = {"value": [], "@odata.nextLink": DEFAULT_ENDPOINT + "?%24skip=2"}
        with patch("tools.water_leak_satellite.odata_inventory.urlopen", return_value=FakeResponse(response)):
            with self.assertRaises(ODataError):
                inventory_query(
                    profile=self.profile, bbox=self.bbox, start=date(2025, 1, 1),
                    end_exclusive=date(2025, 1, 2), limit=1, max_pages=1, timeout=2,
                )

    def test_pagination_url_loop_is_rejected(self):
        response = {"value": [], "@odata.nextLink": DEFAULT_ENDPOINT + "?%24skip=2"}
        with patch("tools.water_leak_satellite.odata_inventory.urlopen", return_value=FakeResponse(response)):
            with self.assertRaises(ODataError):
                inventory_query(
                    profile=self.profile, bbox=self.bbox, start=date(2025, 1, 1),
                    end_exclusive=date(2025, 1, 2), limit=1, max_pages=5, timeout=2,
                )

    def test_main_writes_auditable_json_and_reverifies_without_network(self):
        def fake_open(request, timeout):
            params = parse_qs(urlparse(request.full_url).query)
            filter_text = params.get("$filter", [""])[0]
            product = dict(self.product)
            if "SENTINEL-2" in filter_text:
                product["Id"] = "odata-s2-001"
                product["Name"] = "S2A_MSIL2A_20250101T080000.SAFE"
            return FakeResponse({"value": [product]})

        with tempfile.TemporaryDirectory() as tmp, patch("tools.water_leak_satellite.odata_inventory.urlopen", side_effect=fake_open):
            out = Path(tmp) / "odata-inventory.json"
            code = main(["--start", "2025-01-01", "--end", "2025-01-01", "--out", str(out)])
            self.assertEqual(code, 0)
            document = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(len(document["queries"]), 2)
            self.assertEqual(verify_inventory(document), [])
            self.assertEqual(main(["--verify", str(out)]), 0)

    def test_invalid_bbox_type_fails_closed_without_exception(self):
        with patch("tools.water_leak_satellite.odata_inventory.urlopen", side_effect=[
            FakeResponse({"value": []}), FakeResponse({"value": []})
        ]):
            queries = [
                inventory_query(profile=p, bbox=self.bbox, start=date(2025, 1, 1),
                    end_exclusive=date(2025, 1, 2), limit=100, max_pages=5, timeout=2)
                for p in PROFILES
            ]
        doc = build_document(self.bbox, date(2025, 1, 1), date(2025, 1, 1), queries)
        doc["scope"]["bbox_wgs84"] = "malformed"
        from tools.water_leak_satellite.odata_inventory import sha256_json
        doc.pop("inventory_manifest_sha256")
        doc["inventory_manifest_sha256"] = sha256_json(doc)
        errors = verify_inventory(doc)
        self.assertTrue(any("scope.bbox_wgs84" in err for err in errors))

    def test_zero_product_results_form_a_valid_query_specific_inventory(self):
        with patch("tools.water_leak_satellite.odata_inventory.urlopen", side_effect=[
            FakeResponse({"value": []}), FakeResponse({"value": []})
        ]):
            queries = [
                inventory_query(profile=p, bbox=self.bbox, start=date(2025, 1, 1),
                    end_exclusive=date(2025, 1, 2), limit=100, max_pages=5, timeout=2)
                for p in PROFILES
            ]
        doc = build_document(self.bbox, date(2025, 1, 1), date(2025, 1, 1), queries)
        self.assertEqual(verify_inventory(doc), [])
        self.assertEqual([query["item_count"] for query in doc["queries"]], [0, 0])
        self.assertIn("does not establish", doc["overall_claim_ceiling"])

    def test_noncanonical_query_date_fails_closed(self):
        with patch("tools.water_leak_satellite.odata_inventory.urlopen", side_effect=[
            FakeResponse({"value": []}), FakeResponse({"value": []})
        ]):
            queries = [
                inventory_query(profile=p, bbox=self.bbox, start=date(2025, 1, 1),
                    end_exclusive=date(2025, 1, 2), limit=100, max_pages=5, timeout=2)
                for p in PROFILES
            ]
        doc = build_document(self.bbox, date(2025, 1, 1), date(2025, 1, 1), queries)
        doc["queries"][0]["start_date_inclusive"] = "20250101"
        from tools.water_leak_satellite.odata_inventory import sha256_json
        doc.pop("inventory_manifest_sha256")
        doc["inventory_manifest_sha256"] = sha256_json(doc)
        errors = verify_inventory(doc)
        self.assertTrue(any("canonical YYYY-MM-DD" in err for err in errors))

    def test_recorded_endpoint_and_page_paths_must_be_products_endpoint(self):
        with patch("tools.water_leak_satellite.odata_inventory.urlopen", side_effect=[
            FakeResponse({"value": []}), FakeResponse({"value": []})
        ]):
            queries = [
                inventory_query(profile=p, bbox=self.bbox, start=date(2025, 1, 1),
                    end_exclusive=date(2025, 1, 2), limit=100, max_pages=5, timeout=2)
                for p in PROFILES
            ]
        doc = build_document(self.bbox, date(2025, 1, 1), date(2025, 1, 1), queries)
        doc["catalogue_endpoint"] = "https://catalogue.dataspace.copernicus.eu/odata/v1/Users"
        doc["queries"][0]["page_records"][0]["request_url_redacted"] = "https://catalogue.dataspace.copernicus.eu/odata/v1/Users"
        from tools.water_leak_satellite.odata_inventory import sha256_json
        doc.pop("inventory_manifest_sha256")
        doc["inventory_manifest_sha256"] = sha256_json(doc)
        errors = verify_inventory(doc)
        self.assertTrue(any("catalogue_endpoint" in err for err in errors))
        self.assertTrue(any("request_url_redacted" in err for err in errors))

    def test_query_scope_tamper_is_detected_even_after_rehash(self):
        # Build an otherwise valid full document via fake responses.
        responses = [{"value": [dict(self.product, Id=f"p{i}", Name=f"n{i}")]} for i in range(2)]
        with patch("tools.water_leak_satellite.odata_inventory.urlopen", side_effect=[FakeResponse(x) for x in responses]):
            queries = [
                inventory_query(profile=p, bbox=self.bbox, start=date(2025, 1, 1),
                    end_exclusive=date(2025, 1, 2), limit=100, max_pages=5, timeout=2)
                for p in PROFILES
            ]
        doc = build_document(self.bbox, date(2025, 1, 1), date(2025, 1, 1), queries)
        doc["queries"][0]["filter"] = doc["queries"][0]["filter"].replace("27.70000000", "27.00000000")
        doc.pop("inventory_manifest_sha256")
        from tools.water_leak_satellite.odata_inventory import sha256_json
        doc["inventory_manifest_sha256"] = sha256_json(doc)
        self.assertTrue(any(".filter" in err for err in verify_inventory(doc)))


if __name__ == "__main__":
    unittest.main()
