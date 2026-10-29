# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dependency-free tests for the Rust/Python binary digest contract."""

import unittest

from validation.iit_digest_contract import (
    analysis_request_digest,
    canonical_model_bytes,
    evaluation_digest,
    model_digest,
)


def fixture():
    # TPM rows for state indices [00, 10, 01, 11], unit 0 is the LSB.
    tpm = [
        [0.1, 0.2],
        [0.9, 0.2],
        [0.1, 0.8],
        [0.9, 0.8],
    ]
    # Source-major directed connectivity; one extra edge 0 -> 1 is permitted.
    connectivity = [
        [1, 1],
        [0, 1],
    ]
    return tpm, connectivity


class IitDigestContractTests(unittest.TestCase):
    def setUp(self):
        tpm, connectivity = fixture()
        self.model_bytes = canonical_model_bytes(tpm, connectivity)

    def test_model_digest_matches_rust_golden_vector(self):
        self.assertEqual(
            model_digest(self.model_bytes),
            "728eb74b44c4aec7896db360919b6efcb0f06e4b7eb487b2974c486c42a80b62",
        )

    def test_evaluation_digest_matches_rust_golden_vector(self):
        self.assertEqual(
            evaluation_digest(self.model_bytes, [1, 0]),
            "6188b51af953ac133cd3c4af0d628ceba73692ff36c2f37f6e05bdb1531214e2",
        )

    def test_analysis_request_digest_matches_rust_golden_vector(self):
        self.assertEqual(
            analysis_request_digest(
                self.model_bytes, [1, 0], [0, 1],
                "IIT_4_0_2023", "iit4_system_phi_s",
            ),
            "8b697dce9b5ad3786d4e15f7d5db39c563bce54ea3676976c07efea3f4698db0",
        )

    def test_state_formalism_subset_and_quantity_change_request_digest(self):
        base = analysis_request_digest(
            self.model_bytes, [1, 0], [0, 1],
            "IIT_4_0_2023", "iit4_system_phi_s",
        )
        variants = [
            analysis_request_digest(self.model_bytes, [0, 1], [0, 1], "IIT_4_0_2023", "iit4_system_phi_s"),
            analysis_request_digest(self.model_bytes, [1, 0], [0, 1], "IIT_4_0_2026", "iit4_system_phi_s"),
            analysis_request_digest(self.model_bytes, [1, 0], [0], "IIT_4_0_2023", "iit4_system_phi_s"),
            analysis_request_digest(self.model_bytes, [1, 0], [0, 1], "IIT_4_0_2023", "iit4_structure_big_phi"),
        ]
        self.assertEqual(len(set(variants + [base])), 5)

    def test_negative_zero_is_canonicalized(self):
        positive = canonical_model_bytes([[0.0], [1.0]], [[1]])
        negative = canonical_model_bytes([[-0.0], [1.0]], [[1]])
        self.assertEqual(positive, negative)

    def test_rejects_nonfinite_probabilities(self):
        for value in (float("nan"), float("inf"), float("-inf"), -0.01, 1.01):
            with self.subTest(value=value), self.assertRaises(ValueError):
                canonical_model_bytes([[value], [1.0]], [[1]])

    def test_rejects_connectivity_that_omits_tpm_influence(self):
        tpm, _ = fixture()
        with self.assertRaisesRegex(ValueError, "connectivity omits TPM influence"):
            canonical_model_bytes(tpm, [[1, 0], [1, 0]])

    def test_rejects_wrong_state_size_and_out_of_range_subset(self):
        with self.assertRaisesRegex(ValueError, "n_units binary bits"):
            evaluation_digest(self.model_bytes, [1])
        with self.assertRaisesRegex(ValueError, "model unit range"):
            analysis_request_digest(
                self.model_bytes, [1, 0], [0, 2],
                "IIT_4_0_2023", "iit4_system_phi_s",
            )

    def test_rejects_duplicate_or_empty_candidate_subsystems(self):
        for subset in ([], [0, 0]):
            with self.subTest(subset=subset), self.assertRaises(ValueError):
                analysis_request_digest(
                    self.model_bytes, [1, 0], subset,
                    "IIT_4_0_2023", "iit4_system_phi_s",
                )


if __name__ == "__main__":
    unittest.main()
