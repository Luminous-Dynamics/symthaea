import unittest

from scripts.research import st_lucia_r0_s3_preflight as pre


def _asset(i: int):
    key = f"Sentinel-X/example/{i}.bin"
    return {
        "collection": "sentinel-test",
        "item_id": f"item-{i}",
        "asset_key": f"asset-{i}",
        "purpose": "science-payload",
        "access_method": "s3",
        "stac_href": f"s3://eodata/{key}",
        "s3_endpoint": pre.EXPECTED_ENDPOINT,
        "s3_bucket": pre.EXPECTED_BUCKET,
        "s3_key": key,
    }


def _plan():
    return {
        "schema": "symthaea-st-lucia-r0-asset-plan/v1",
        "tool_version": "1.2.0",
        "plan_sha256": pre.EXPECTED_PLAN_INTERNAL_SHA256,
        "approved_s3_endpoint": pre.EXPECTED_ENDPOINT,
        "approved_s3_bucket": pre.EXPECTED_BUCKET,
        "assets": [_asset(i) for i in range(pre.EXPECTED_ASSET_COUNT)],
    }


class PreflightTests(unittest.TestCase):
    def test_validate_plan_accepts_exact_29_s3_rows(self):
        assets = pre.validate_plan(_plan())
        self.assertEqual(29, len(assets))

    def test_validate_plan_rejects_wrong_bucket_or_locator(self):
        p = _plan()
        p["assets"][0]["s3_bucket"] = "evil"
        with self.assertRaises(pre.PreflightError):
            pre.validate_plan(p)
        p = _plan()
        p["assets"][0]["stac_href"] = "s3://evil/x"
        with self.assertRaises(pre.PreflightError):
            pre.validate_plan(p)

    def test_head_command_can_only_construct_head_object(self):
        cmd = pre.aws_head_command(_asset(1))
        self.assertEqual(["aws", "s3api", "head-object"], cmd[:3])
        self.assertNotIn("get-object", cmd)
        self.assertNotIn("cp", cmd)
        self.assertIn("--key", cmd)
        self.assertIn(pre.EXPECTED_ENDPOINT, cmd)

    def test_normalize_head_requires_nonnegative_length(self):
        got = pre.normalize_head({"ContentLength": 123, "ETag": '"abc"', "Metadata": {"secret": "omit"}})
        self.assertEqual(123, got["ContentLength"])
        self.assertEqual('"abc"', got["ETag"])
        self.assertNotIn("Metadata", got)
        with self.assertRaises(pre.PreflightError):
            pre.normalize_head({"ContentLength": -1})
        with self.assertRaises(pre.PreflightError):
            pre.normalize_head({"ETag": '"abc"'})

    def test_receipt_hash_is_deterministic(self):
        value = {"b": 2, "a": 1}
        self.assertEqual(pre.sha256_bytes(pre.canonical_json_bytes(value)), pre.sha256_bytes(pre.canonical_json_bytes({"a": 1, "b": 2})))


if __name__ == "__main__":
    unittest.main()
