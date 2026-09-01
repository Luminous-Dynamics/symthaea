#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import st_lucia_stac_discovery as d  # noqa: E402


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


class DiscoveryTests(unittest.TestCase):
    def test_s2_selection_is_earliest_then_lexical_id(self):
        items = [
            s2_item("z-late", "2026-07-04T10:00:00Z"),
            s2_item("z-tie", "2026-07-02T10:00:00Z"),
            s2_item("a-tie", "2026-07-02T10:00:00Z"),
        ]
        selected, audit = d.select_s2(items)
        self.assertEqual(selected["id"], "a-tie")
        self.assertEqual(sum(1 for row in audit if row["eligible"]), 3)

    def test_s2_half_open_end_is_enforced_locally(self):
        selected, audit = d.select_s2(
            [
                s2_item("july", "2026-07-31T23:59:59Z"),
                s2_item("august-boundary", "2026-08-01T00:00:00Z"),
            ]
        )
        self.assertEqual(selected["id"], "july")
        boundary = next(row for row in audit if row["id"] == "august-boundary")
        self.assertIn("outside-half-open-time-window", boundary["ineligibility_reasons"])

    def test_s2_missing_required_band_metadata_is_not_eligible(self):
        item = s2_item("missing-band", "2026-07-02T00:00:00Z")
        del item["assets"]["B11"]
        selected, audit = d.select_s2([item])
        self.assertIsNone(selected)
        self.assertIn("missing-required-band-metadata:B11", audit[0]["ineligibility_reasons"])

    def test_s1_pairing_prefers_time_distance_then_earlier_then_id(self):
        s2_time = datetime(2026, 7, 10, 12, tzinfo=timezone.utc)
        items = [
            s1_item("later", "2026-07-10T13:00:00Z"),
            s1_item("z-earlier", "2026-07-10T11:00:00Z"),
            s1_item("a-earlier", "2026-07-10T11:00:00Z"),
        ]
        selected, _ = d.select_s1(items, s2_time)
        self.assertEqual(selected["id"], "a-earlier")

    def test_s1_requires_iw_and_vv_vh(self):
        s2_time = datetime(2026, 7, 10, 12, tzinfo=timezone.utc)
        items = [
            s1_item("ew", "2026-07-10T12:00:00Z", mode="EW"),
            s1_item("vv-only", "2026-07-10T12:00:00Z", pols=["VV"]),
        ]
        selected, audit = d.select_s1(items, s2_time)
        self.assertIsNone(selected)
        self.assertIn("not-iw-mode", audit[0]["ineligibility_reasons"])
        self.assertIn("missing-vv-vh", audit[1]["ineligibility_reasons"])

    def test_pagination_preserves_raw_pages_and_deduplicates_identical_items(self):
        first = page([s2_item("a", "2026-07-02T00:00:00Z")], "https://example.test/page2")
        second = page([
            s2_item("a", "2026-07-02T00:00:00Z"),
            s2_item("b", "2026-07-03T00:00:00Z"),
        ])
        payloads = {
            "https://example.test/page1": json.dumps(first, sort_keys=True).encode(),
            "https://example.test/page2": json.dumps(second, sort_keys=True).encode(),
        }

        def fetcher(request, timeout):
            self.assertGreater(timeout, 0)
            return payloads[request["url"]]

        with tempfile.TemporaryDirectory() as tmp:
            pages, evidence = d.exhaust_pages(
                d.initial_request("https://example.test/page1"),
                Path(tmp),
                fetcher=fetcher,
            )
            items = d.deduplicated_items(pages)
            self.assertEqual([item["id"] for item in items], ["a", "b"])
            self.assertEqual(len(evidence), 2)
            self.assertTrue((Path(tmp) / "page-0001.json").exists())
            self.assertTrue((Path(tmp) / "page-0002.sha256").exists())

    def test_duplicate_id_metadata_conflict_fails_closed(self):
        first_item = s2_item("same", "2026-07-02T00:00:00Z")
        second_item = s2_item("same", "2026-07-03T00:00:00Z")
        payloads = {
            "https://example.test/page1": json.dumps(
                page([first_item], "https://example.test/page2")
            ).encode(),
            "https://example.test/page2": json.dumps(page([second_item])).encode(),
        }

        def fetcher(request, timeout):
            return payloads[request["url"]]

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(d.DiscoveryError, "duplicate-id-metadata-conflict"):
                d.exhaust_pages(
                    d.initial_request("https://example.test/page1"),
                    Path(tmp),
                    fetcher=fetcher,
                )

    def test_pagination_cycle_fails_closed(self):
        loop = page([], "https://example.test/page1")

        def fetcher(request, timeout):
            return json.dumps(loop).encode()

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(d.DiscoveryError, "pagination request cycle"):
                d.exhaust_pages(
                    d.initial_request("https://example.test/page1"),
                    Path(tmp),
                    fetcher=fetcher,
                )


if __name__ == "__main__":
    unittest.main()
