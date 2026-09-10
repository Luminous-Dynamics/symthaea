#!/usr/bin/env python3
"""Regression tests for strict trusted-main GitHub provider-time parsing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import trusted_main_provider_time as provider_time  # noqa: E402


class ProviderTimeTests(unittest.TestCase):
    def test_whole_second_parses_to_exact_epoch_coordinate(self):
        instant = provider_time.parse_github_utc_instant("1970-01-01T00:00:00Z")
        self.assertEqual(instant.unix_nanos, 0)
        self.assertEqual(instant.canonical_text, "1970-01-01T00:00:00Z")
        self.assertEqual(instant.schema, provider_time.SCHEMA)

    def test_fractional_nanoseconds_are_integer_exact(self):
        instant = provider_time.parse_github_utc_instant("1970-01-01T00:00:00.123456789Z")
        self.assertEqual(instant.unix_nanos, 123_456_789)

    def test_fractional_digits_have_expected_scale(self):
        self.assertEqual(
            provider_time.parse_github_utc_instant("1970-01-01T00:00:00.1Z").unix_nanos,
            100_000_000,
        )
        self.assertEqual(
            provider_time.parse_github_utc_instant("1970-01-01T00:00:00.000000001Z").unix_nanos,
            1,
        )

    def test_pre_epoch_instants_order_without_float_conversion(self):
        instant = provider_time.parse_github_utc_instant("1969-12-31T23:59:59.999999999Z")
        self.assertEqual(instant.unix_nanos, -1)

    def test_leap_year_calendar_validity(self):
        provider_time.parse_github_utc_instant("2024-02-29T12:00:00Z")
        with self.assertRaises(provider_time.ProviderTimeError):
            provider_time.parse_github_utc_instant("2025-02-29T12:00:00Z")

    def test_impossible_month_day_hour_minute_second_reject(self):
        bad = (
            "2026-13-01T00:00:00Z",
            "2026-04-31T00:00:00Z",
            "2026-01-01T24:00:00Z",
            "2026-01-01T00:60:00Z",
            "2026-01-01T00:00:60Z",
        )
        for value in bad:
            with self.subTest(value=value):
                with self.assertRaises(provider_time.ProviderTimeError):
                    provider_time.parse_github_utc_instant(value)

    def test_timezone_offsets_are_not_canonical_provider_utc(self):
        for value in (
            "2026-09-10T06:54:39+00:00",
            "2026-09-10T08:54:39+02:00",
            "2026-09-10T06:54:39z",
        ):
            with self.subTest(value=value):
                with self.assertRaises(provider_time.ProviderTimeError):
                    provider_time.parse_github_utc_instant(value)

    def test_fraction_precision_is_bounded_to_nanoseconds(self):
        with self.assertRaises(provider_time.ProviderTimeError):
            provider_time.parse_github_utc_instant("2026-09-10T06:54:39.1234567890Z")

    def test_whitespace_and_control_characters_reject(self):
        for value in (
            " 2026-09-10T06:54:39Z",
            "2026-09-10T06:54:39Z ",
            "2026-09-10T06:54:39Z\n",
        ):
            with self.subTest(value=repr(value)):
                with self.assertRaises(provider_time.ProviderTimeError):
                    provider_time.parse_github_utc_instant(value)

    def test_provider_order_is_exact_at_fraction_boundary(self):
        self.assertEqual(
            provider_time.compare_provider_instants(
                "2026-09-10T06:54:39Z",
                "2026-09-10T06:54:39.000000001Z",
            ),
            -1,
        )
        self.assertEqual(
            provider_time.compare_provider_instants(
                "2026-09-10T06:54:39.1Z",
                "2026-09-10T06:54:39.100000000Z",
            ),
            0,
        )

    def test_order_crosses_day_boundary_correctly(self):
        self.assertEqual(
            provider_time.compare_provider_instants(
                "2026-09-09T23:59:59.999999999Z",
                "2026-09-10T00:00:00Z",
            ),
            -1,
        )

    def test_non_strings_reject(self):
        for value in (None, 0, True, {}, []):
            with self.subTest(value=value):
                with self.assertRaises(provider_time.ProviderTimeError):
                    provider_time.parse_github_utc_instant(value)


if __name__ == "__main__":
    unittest.main()
