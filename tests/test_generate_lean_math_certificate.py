"""Tests for the certificate-to-Lean proof bridge."""
from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from tools import verify_math_certificate as verifier
from tools.generate_lean_math_certificate import main, render_certificate
from tools.math_certificate_golden_vectors import (
    SYMFCER1_FACTOR_2,
    SYMPPRF1_PRIME_3,
)


def proof_two() -> verifier.PrimalityProof:
    return verifier.PrimalityProof(2, (verifier.PrimeNode(2, (), ()),))


def proof_three() -> verifier.PrimalityProof:
    return verifier.PrimalityProof(
        3,
        (
            verifier.PrimeNode(2, (), ()),
            verifier.PrimeNode(3, ((2, 1),), ((2, 2),)),
        ),
    )


def proof_five() -> verifier.PrimalityProof:
    return verifier.PrimalityProof(
        5,
        (
            verifier.PrimeNode(2, (), ()),
            verifier.PrimeNode(5, ((2, 2),), ((2, 2),)),
        ),
    )


class LeanCertificateBridgeTests(unittest.TestCase):
    def test_golden_vectors_generate_kernel_checkable_claims(self) -> None:
        prime_three = render_certificate(SYMPPRF1_PRIME_3)
        self.assertIn("theorem root_3_is_prime : Nat.Prime 3", prime_three)
        self.assertIn("prime_cert [small {2}, pock (3, 2, 2)]", prime_three)

        factor_two = render_certificate(SYMFCER1_FACTOR_2)
        self.assertIn("theorem factor_0_2_is_prime : Nat.Prime 2", factor_two)
        self.assertIn("theorem factorization_reconstructs : 2 = (2)", factor_two)
        self.assertIn("norm_num", factor_two)

    def test_combines_distinct_factor_witnesses_and_rechecks_them(self) -> None:
        # 2 is a primitive root modulo 13. The q=2 and q=3 witnesses are both 2;
        # isolated q-primary components multiply to the valid common root 11.
        proof = verifier.PrimalityProof(
            13,
            (
                verifier.PrimeNode(2, (), ()),
                verifier.PrimeNode(3, ((2, 1),), ((2, 2),)),
                verifier.PrimeNode(13, ((2, 2), (3, 1)), ((2, 2), (3, 2))),
            ),
        )
        encoded = verifier.encode_primality_proof(proof)
        lean_source = render_certificate(encoded)
        self.assertIn("pock (13, 11, 2 ^ 2 * 3)", lean_source)

    def test_factorization_emits_prime_proofs_and_exact_product_goal(self) -> None:
        bundle = verifier.CertifiedFactorization(
            verifier.Factorization(30, ((2, 1), (3, 1), (5, 1))),
            (proof_two(), proof_three(), proof_five()),
        )
        source = render_certificate(verifier.encode_certified_factorization(bundle))
        self.assertIn("theorem factor_0_2_is_prime : Nat.Prime 2", source)
        self.assertIn("theorem factor_1_3_is_prime : Nat.Prime 3", source)
        self.assertIn("pock (3, 2, 2)", source)
        self.assertIn("theorem factor_2_5_is_prime : Nat.Prime 5", source)
        self.assertIn("theorem factorization_reconstructs : 30 = (2 * 3 * 5)", source)

    def test_invalid_or_noncanonical_input_fails_closed(self) -> None:
        for data in (b"", b"not a certificate", SYMPPRF1_PRIME_3 + b"\\x00"):
            with self.subTest(data=data[:16]):
                with self.assertRaises(verifier.CertificateError):
                    render_certificate(data)

    def test_cli_writes_generated_source_from_golden_vector(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "GeneratedPrimeThree.lean"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main(
                    [
                        "--golden-vector",
                        "primality-3",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(status, 0, stdout.getvalue())
            self.assertIn("written", stdout.getvalue())
            self.assertIn("prime_cert [small {2}, pock (3, 2, 2)]", output.read_text())

if __name__ == "__main__":
    unittest.main()
