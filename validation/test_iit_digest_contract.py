# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dependency-free tests for the Rust/Python binary digest contract."""

import struct
import unittest

from validation.iit_digest_contract import (
    MODEL_MAGIC,
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

    def test_published_pyphi_conventions_tpm_matches_rust_golden_vectors(self):
        # Two-unit state-by-node TPM from PyPhi's conventions documentation.
        tpm = [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
            [0.7, 0.8],
        ]
        model_bytes = canonical_model_bytes(tpm, [[1, 1], [1, 1]])
        self.assertEqual(
            model_digest(model_bytes),
            "8e8000cfacf6631f5d229449662b260e07419d76b566d6a78214fe1f4b408894",
        )
        self.assertEqual(
            evaluation_digest(model_bytes, [1, 0]),
            "143ea0c7ae2676acb89adbd02f93d40a71ee3e6098bade2421a4a2105176eab8",
        )
        self.assertEqual(
            analysis_request_digest(
                model_bytes, [1, 0], [0, 1],
                "IIT_4_0_2023", "iit4_system_phi_s",
            ),
            "1b1d3a2f76139abfd2e5bf89da48376b6fac11420382844be813e6e7ac87f2fd",
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
        reversed_subset = analysis_request_digest(
            self.model_bytes, [1, 0], [1, 0],
            "IIT_4_0_2023", "iit4_system_phi_s",
        )
        self.assertEqual(base, reversed_subset)

    def test_negative_zero_is_canonicalized(self):
        positive = canonical_model_bytes([[0.0], [1.0]], [[1]])
        negative = canonical_model_bytes([[-0.0], [1.0]], [[1]])
        self.assertEqual(positive, negative)

    def test_rejects_nonfinite_probabilities(self):
        for value in (float("nan"), float("inf"), float("-inf"), -0.01, 1.01):
            with self.subTest(value=value), self.assertRaises(ValueError):
                canonical_model_bytes([[value], [1.0]], [[1]])

    def test_rejects_fractional_state_or_connectivity_bits(self):
        with self.assertRaisesRegex(ValueError, "binary bits"):
            evaluation_digest(self.model_bytes, [0.5, 1])
        tpm, _ = fixture()
        with self.assertRaisesRegex(ValueError, "binary entries"):
            canonical_model_bytes(tpm, [[1, 0.25], [0, 1]])

    def test_rejects_connectivity_that_omits_tpm_influence(self):
        tpm, _ = fixture()
        with self.assertRaisesRegex(ValueError, "connectivity omits TPM influence"):
            canonical_model_bytes(tpm, [[1, 0], [1, 0]])

    def test_raw_model_digest_rejects_invalid_encoded_probability(self):
        encoded = bytearray(self.model_bytes)
        offset = len(MODEL_MAGIC) + 4
        encoded[offset:offset + 8] = struct.pack("<d", 1.5)
        with self.assertRaisesRegex(ValueError, "invalid TPM probability"):
            model_digest(bytes(encoded))

    def test_raw_model_digest_rejects_nonbinary_connectivity(self):
        encoded = bytearray(self.model_bytes)
        n_units = 2
        offset = len(MODEL_MAGIC) + 4 + (1 << n_units) * n_units * 8
        encoded[offset] = 2
        with self.assertRaisesRegex(ValueError, "connectivity is not binary"):
            model_digest(bytes(encoded))

    def test_raw_model_digest_rejects_noncanonical_negative_zero(self):
        encoded = bytearray(self.model_bytes)
        offset = len(MODEL_MAGIC) + 4
        encoded[offset:offset + 8] = struct.pack("<d", -0.0)
        with self.assertRaisesRegex(ValueError, "non-canonical negative zero"):
            model_digest(bytes(encoded))

    def test_rejects_wrong_state_size_and_out_of_range_subset(self):
        with self.assertRaisesRegex(ValueError, "n_units integer/bool binary bits"):
            evaluation_digest(self.model_bytes, [1])
        for invalid_state in ([0.5, 1], [1.0, 0]):
            with self.subTest(state=invalid_state), self.assertRaisesRegex(
                ValueError, "n_units integer/bool binary bits"
            ):
                analysis_request_digest(
                    self.model_bytes, invalid_state, [0, 1],
                    "IIT_4_0_2023", "iit4_system_phi_s",
                )
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

    def test_rejects_unknown_or_incompatible_formalism_quantity_pairs(self):
        invalid_requests = [
            ("IIT_4_0_2023", "iit3_system_phi"),
            ("IIT_3_0", "iit4_system_phi_s"),
            ("IIT_4_0_2026", "not_a_quantity"),
            ("IIT_9_9", "iit4_system_phi_s"),
        ]
        for formalism, quantity in invalid_requests:
            with self.subTest(formalism=formalism, quantity=quantity), self.assertRaises(ValueError):
                analysis_request_digest(
                    self.model_bytes, [1, 0], [0, 1], formalism, quantity
                )

    def test_rejects_fractional_candidate_subsystem_index(self):
        with self.assertRaisesRegex(ValueError, "candidate subsystem indices must be integers"):
            analysis_request_digest(
                self.model_bytes, [1, 0], [0, 1.5],
                "IIT_4_0_2023", "iit4_system_phi_s",
            )


if __name__ == "__main__":
    unittest.main()
