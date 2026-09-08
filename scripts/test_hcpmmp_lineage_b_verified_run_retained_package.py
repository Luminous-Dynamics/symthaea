#!/usr/bin/env python3
from __future__ import annotations

import json
import unittest

import capture_hcpmmp1_neuromaps_lineage_b_qualified_run_v2 as qualified
import capture_hcpmmp1_neuromaps_lineage_b_verified_run as capture


class RetainedQualificationPackageTests(unittest.TestCase):
    def package(self):
        profile_raw = qualified.PROFILE_PATH.read_bytes()
        verification_raw = qualified.VERIFICATION_PATH.read_bytes()
        profile = capture.verify_profile_bytes(profile_raw)
        verification = capture.verify_retained_workbench_bytes(profile, verification_raw)
        return profile_raw, verification_raw, profile, verification

    def test_exact_retained_file_roots_and_canonical_bytes(self):
        profile_raw, verification_raw, profile, verification = self.package()
        self.assertEqual(capture.digest_bytes(profile_raw), qualified.QUALIFIED_PROFILE_FILE_SHA256)
        self.assertEqual(capture.digest_bytes(verification_raw), qualified.QUALIFIED_VERIFICATION_FILE_SHA256)
        self.assertEqual(profile_raw, capture.canonical_json_bytes(profile) + b"\n")
        self.assertEqual(verification_raw, capture.canonical_json_bytes(verification) + b"\n")

    def test_exact_hosted_qualification_source(self):
        _, _, profile, verification = self.package()
        source = profile["qualification_source"]
        self.assertEqual(source["pr"], 976)
        self.assertEqual(source["head_sha"], qualified.QUALIFIED_WORKBENCH_HEAD)
        self.assertEqual(source["workflow_run_id"], 34265087269)
        self.assertEqual(source["verification_file_sha256"], qualified.QUALIFIED_VERIFICATION_FILE_SHA256)
        self.assertEqual(
            source["independent_verification_archive_sha256"],
            "sha256:047060be35a90bdb0b6cf463b0b739fff506efefd4317bf04fc97b4a107506d7",
        )
        self.assertEqual(
            source["raw_evidence_archive_sha256"],
            "sha256:b7d30bd90c0943135bb3163502a8a9169aeb8d086a55f6f3d98d81c0947308a7",
        )
        self.assertEqual(verification["closure_digest"], qualified.QUALIFIED_CLOSURE_DIGEST)
        self.assertEqual(
            verification["program_content_sha256"],
            "sha256:ad461ffeef56a0d807617e41ca65e38a2c25f1ec47abaea560a0bf843613db88",
        )
        self.assertEqual(
            verification["version_output_sha256"],
            "sha256:d4e353408c9d76bc7c4ffe476e4161e0dc443add050ad499baa2cb93c84630c5",
        )

    def test_authority_is_exactly_narrow(self):
        _, _, profile, verification = self.package()
        self.assertEqual(verification["authority"], profile["required_verification_authority"])
        self.assertTrue(verification["authority"]["program_membership_verified"])
        self.assertTrue(verification["authority"]["invocation_profile_verified"])
        self.assertTrue(verification["authority"]["invocation_executed"])
        self.assertTrue(verification["authority"]["version_output_bound"])
        for key in (
            "same_host_repeatability_established",
            "path_equivalence_established",
            "cross_cpu_equivalence_established",
            "workbench_execution_qualified",
            "scientific_execution_qualified",
            "transform_executed",
            "atlas_correctness_established",
            "fmq010_established",
            "neural_alignment_established",
            "consciousness_evidence",
        ):
            self.assertFalse(verification["authority"][key], key)

    def test_pretty_printed_profile_is_not_the_qualified_profile(self):
        profile_raw, _, profile, _ = self.package()
        pretty = (json.dumps(profile, indent=2, sort_keys=True) + "\n").encode()
        self.assertNotEqual(pretty, profile_raw)
        self.assertNotEqual(capture.digest_bytes(pretty), qualified.QUALIFIED_PROFILE_FILE_SHA256)
        with self.assertRaises(capture.CaptureError):
            capture.verify_profile_bytes(pretty)


if __name__ == "__main__":
    unittest.main(verbosity=2)
