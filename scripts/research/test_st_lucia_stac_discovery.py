#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import st_lucia_stac_discovery as d  # noqa: E402

BASE = "https://stac.dataspace.copernicus.eu/v1/test"


def s2_item(item_id: str, when: str, cloud: float = 5.0):
    return {
        "type": "Feature",
        "id": item_id,
        "collection": "sentinel-2-l2a",
        "properties": {"datetime": when, "eo:cloud_cover": cloud},
        "assets": {band: {} for band in d.REQUIRED_S2_BANDS},
    }


def s1_item(item_id: str, when: str, mode: str = "IW", pols=None):
    return {
        "type": "Feature",
        "id": item_id,
        "collection": "sentinel-1-grd",
        "properties": {
            "datetime": when,
            "sar:instrument_mode": mode,
            "sar:polarizations": ["VV", "VH"] if pols is None else pols,
        },
        "assets": {},
    }


def page(features, next_href=None):
    links = [] if next_href is None else [{"rel": "next", "href": next_href}]
    return {"type": "FeatureCollection", "features": features, "links": links}


def response(payload):
    return json.dumps(payload, sort_keys=True).encode(), {"x-test-server": "fixture"}


class DiscoveryTests(unittest.TestCase):
    def test_s2_selection_is_earliest_then_lexical_id(self):
        items = [
            s2_item("z-late", "2026-07-04T10:00:00Z"),
            s2_item("z-tie", "2026-07-02T10:00:00Z"),
            s2_item("a-tie", "2026-07-02T10:00:00Z"),
        ]
        selected, audit, ordered = d.select_s2(items)
        self.assertEqual(selected["id"], "a-tie")
        self.assertEqual(ordered, ["a-tie", "z-tie", "z-late"])
        self.assertEqual(sum(1 for row in audit if row["eligible"]), 3)

    def test_s2_half_open_end_is_enforced_locally(self):
        selected, audit, ordered = d.select_s2(
            [
                s2_item("july", "2026-07-31T23:59:59Z"),
                s2_item("august-boundary", "2026-08-01T00:00:00Z"),
            ]
        )
        self.assertEqual(selected["id"], "july")
        self.assertEqual(ordered, ["july"])
        boundary = next(row for row in audit if row["id"] == "august-boundary")
        self.assertIn("outside-half-open-time-window", boundary["ineligibility_reasons"])

    def test_s2_missing_required_band_metadata_is_not_eligible(self):
        item = s2_item("missing-band", "2026-07-02T00:00:00Z")
        del item["assets"]["B11"]
        selected, audit, ordered = d.select_s2([item])
        self.assertIsNone(selected)
        self.assertEqual(ordered, [])
        self.assertIn("missing-required-band-metadata:B11", audit[0]["ineligibility_reasons"])

    def test_s1_pairing_prefers_time_distance_then_earlier_then_id(self):
        s2_time = datetime(2026, 7, 10, 12, tzinfo=timezone.utc)
        items = [
            s1_item("later", "2026-07-10T13:00:00Z"),
            s1_item("z-earlier", "2026-07-10T11:00:00Z"),
            s1_item("a-earlier", "2026-07-10T11:00:00Z"),
        ]
        selected, _, ordered = d.select_s1(items, s2_time)
        self.assertEqual(selected["id"], "a-earlier")
        self.assertEqual(ordered, ["a-earlier", "z-earlier", "later"])

    def test_s1_requires_iw_and_vv_vh(self):
        s2_time = datetime(2026, 7, 10, 12, tzinfo=timezone.utc)
        items = [
            s1_item("ew", "2026-07-10T12:00:00Z", mode="EW"),
            s1_item("vv-only", "2026-07-10T12:00:00Z", pols=["VV"]),
        ]
        selected, audit, ordered = d.select_s1(items, s2_time)
        self.assertIsNone(selected)
        self.assertEqual(ordered, [])
        self.assertIn("not-iw-mode", audit[0]["ineligibility_reasons"])
        self.assertIn("missing-vv-vh", audit[1]["ineligibility_reasons"])

    def test_pagination_preserves_raw_pages_and_deduplicates_identical_items(self):
        first_url = f"{BASE}/page1"
        second_url = f"{BASE}/page2"
        first = page([s2_item("a", "2026-07-02T00:00:00Z")], second_url)
        second = page([
            s2_item("a", "2026-07-02T00:00:00Z"),
            s2_item("b", "2026-07-03T00:00:00Z"),
        ])
        payloads = {first_url: response(first), second_url: response(second)}

        def fetcher(request, timeout):
            self.assertGreater(timeout, 0)
            return payloads[request["url"]]

        with tempfile.TemporaryDirectory() as tmp:
            pages, evidence = d.exhaust_pages(
                d.initial_request(first_url), Path(tmp), fetcher=fetcher
            )
            items = d.deduplicated_items(pages)
            self.assertEqual([item["id"] for item in items], ["a", "b"])
            self.assertEqual(len(evidence), 2)
            self.assertEqual(evidence[0]["path"], "page-0001.json")
            self.assertEqual(evidence[0]["response_headers"]["x-test-server"], "fixture")
            self.assertTrue(evidence[0]["retrieved_at_utc"].endswith("Z"))
            self.assertTrue((Path(tmp) / "page-0001.json").exists())
            self.assertTrue((Path(tmp) / "page-0002.sha256").exists())

    def test_duplicate_id_metadata_conflict_fails_closed(self):
        first_url = f"{BASE}/page1"
        second_url = f"{BASE}/page2"
        payloads = {
            first_url: response(page([s2_item("same", "2026-07-02T00:00:00Z")], second_url)),
            second_url: response(page([s2_item("same", "2026-07-03T00:00:00Z")])),
        }

        def fetcher(request, timeout):
            return payloads[request["url"]]

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(d.DiscoveryError, "duplicate-id-metadata-conflict"):
                d.exhaust_pages(d.initial_request(first_url), Path(tmp), fetcher=fetcher)

    def test_pagination_cycle_fails_closed(self):
        first_url = f"{BASE}/page1"
        loop = page([], first_url)

        def fetcher(request, timeout):
            return response(loop)

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(d.DiscoveryError, "pagination request cycle"):
                d.exhaust_pages(d.initial_request(first_url), Path(tmp), fetcher=fetcher)

    def test_off_origin_pagination_is_rejected_before_fetch(self):
        first_url = f"{BASE}/page1"
        malicious = page([], "https://example.test/exfiltrate")
        calls = []

        def fetcher(request, timeout):
            calls.append(request["url"])
            return response(malicious)

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(d.DiscoveryError, "off-origin STAC URL rejected"):
                d.exhaust_pages(d.initial_request(first_url), Path(tmp), fetcher=fetcher)
        self.assertEqual(calls, [first_url])

    def test_initial_request_rejects_off_origin_url(self):
        with self.assertRaisesRegex(d.DiscoveryError, "off-origin STAC URL rejected"):
            d.initial_request("https://example.test/page")


if __name__ == "__main__":
    unittest.main()
