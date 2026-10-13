"""Tests for the independent standard-library certificate verifier."""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import verify_math_certificate as verifier
from tools.math_certificate_golden_vectors import SYMFCER1_FACTOR_2, SYMPPRF1_PRIME_3


def proof_two() -> verifier.PrimalityProof:
    return verifier.PrimalityProof(
        2,
        (verifier.PrimeNode(2, (), ()),),
    )


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


def proof_seventeen() -> verifier.PrimalityProof:
    return verifier.PrimalityProof(
        17,
        (
            verifier.PrimeNode(2, (), ()),
            verifier.PrimeNode(17, ((2, 4),), ((2, 3),)),
        ),
    )


def certificate_thirty() -> verifier.CertifiedFactorization:
    return verifier.CertifiedFactorization(
        verifier.Factorization(30, ((2, 1), (3, 1), (5, 1))),
        (proof_two(), proof_three(), proof_five()),
    )


class PrimalityProofTests(unittest.TestCase):
    def test_primality_proof_golden_vector_is_stable(self) -> None:
        # Hand-authored SYMPPRF1 encoding of p=3 with the base node p=2.
        golden = SYMPPRF1_PRIME_3
        self.assertEqual(verifier.encode_primality_proof(proof_three()), golden)
        self.assertEqual(verifier.decode_primality_proof(golden), proof_three())

    def test_valid_pocklington_proof_and_canonical_round_trip(self) -> None:
        proof = proof_seventeen()
        encoded = verifier.encode_primality_proof(proof)
        self.assertEqual(encoded[:8], b"SYMPPRF1")
        self.assertEqual(verifier.decode_primality_proof(encoded), proof)
        self.assertEqual(verifier.encode_primality_proof(verifier.decode_primality_proof(encoded)), encoded)

    def test_base_case_and_empty_factorization(self) -> None:
        self.assertEqual(verifier.decode_primality_proof(verifier.encode_primality_proof(proof_two())), proof_two())
        empty = verifier.CertifiedFactorization(verifier.Factorization(1, ()), ())
        encoded = verifier.encode_certified_factorization(empty)
        self.assertEqual(verifier.decode_certified_factorization(encoded), empty)

    def test_rejects_composite_claim_and_bad_witness(self) -> None:
        composite = verifier.PrimalityProof(
            15,
            (
                verifier.PrimeNode(2, (), ()),
                verifier.PrimeNode(3, ((2, 1),), ((2, 2),)),
                verifier.PrimeNode(7, ((2, 1), (3, 1)), ((2, 3), (3, 3))),
                verifier.PrimeNode(15, ((2, 1), (7, 1)), ((2, 2), (7, 2))),
            ),
        )
        with self.assertRaisesRegex(verifier.CertificateError, "Fermat congruence"):
            verifier.verify_primality_proof(composite)

        bad = proof_seventeen()
        root = verifier.PrimeNode(17, ((2, 4),), ((2, 1),))
        bad = verifier.PrimalityProof(17, (bad.certificates[0], root))
        with self.assertRaisesRegex(verifier.CertificateError, "witness base"):
            verifier.verify_primality_proof(bad)

    def test_rejects_unreachable_nodes_and_wrong_order(self) -> None:
        valid = proof_seventeen()
        extra = verifier.PrimeNode(3, ((2, 1),), ((2, 2),))
        with self.assertRaisesRegex(verifier.CertificateError, "strictly increasing|unreachable"):
            verifier.verify_primality_proof(
                verifier.PrimalityProof(17, (valid.certificates[0], extra, valid.certificates[1]))
            )
        with self.assertRaisesRegex(verifier.CertificateError, "strictly increasing"):
            verifier.verify_primality_proof(
                verifier.PrimalityProof(17, tuple(reversed(valid.certificates)))
            )

    def test_decoder_rejects_magic_truncation_trailing_and_oversize(self) -> None:
        encoded = verifier.encode_primality_proof(proof_seventeen())
        bad_magic = b"X" + encoded[1:]
        for data in (bad_magic, encoded[:-1], encoded + b"\x00", b"X" * (verifier.MAX_WIRE_BYTES + 1)):
            with self.subTest(length=len(data)):
                with self.assertRaises(verifier.CertificateError):
                    verifier.decode_primality_proof(data)

    def test_object_verification_and_encoding_enforce_resource_budgets_first(self) -> None:
        too_many_nodes = verifier.PrimalityProof(
            2,
            tuple(verifier.PrimeNode(2, (), ()) for _ in range(verifier.MAX_CERTIFICATES + 1)),
        )
        with self.assertRaisesRegex(verifier.CertificateError, "too many primality certificate nodes"):
            verifier.verify_primality_proof(too_many_nodes)
        with self.assertRaisesRegex(verifier.CertificateError, "too many certificate nodes"):
            verifier.encode_primality_proof(too_many_nodes)

        over_budget_node = verifier.PrimeNode(
            2,
            tuple((2, 1) for _ in range(verifier.MAX_PROOF_ENTRIES + 1)),
            (),
        )
        over_budget_entries = verifier.PrimalityProof(2, (over_budget_node,))
        with self.assertRaisesRegex(verifier.CertificateError, "too many proof entries"):
            verifier.verify_primality_proof(over_budget_entries)

    def test_decoder_checks_node_and_entry_budgets_before_looping(self) -> None:
        encoded = bytearray(verifier.encode_primality_proof(proof_seventeen()))
        encoded[16:20] = (verifier.MAX_CERTIFICATES + 1).to_bytes(4, "big")
        with self.assertRaisesRegex(verifier.CertificateError, "too many primality certificate nodes"):
            verifier.decode_primality_proof(bytes(encoded))

        encoded = bytearray(verifier.encode_primality_proof(proof_seventeen()))
        encoded[28:32] = (verifier.MAX_PROOF_ENTRIES + 1).to_bytes(4, "big")
        with self.assertRaisesRegex(verifier.CertificateError, "too many proof entries"):
            verifier.decode_primality_proof(bytes(encoded))


class CertifiedFactorizationTests(unittest.TestCase):
    def test_certified_factorization_wire_golden_vector_is_stable(self) -> None:
        certificate = verifier.CertifiedFactorization(
            verifier.Factorization(2, ((2, 1),)),
            (proof_two(),),
        )
        golden = SYMFCER1_FACTOR_2
        self.assertEqual(len(golden), 76)
        self.assertEqual(verifier.encode_certified_factorization(certificate), golden)
        self.assertEqual(verifier.decode_certified_factorization(golden), certificate)

    def test_valid_bundle_round_trip_and_exact_reconstruction(self) -> None:
        certificate = certificate_thirty()
        encoded = verifier.encode_certified_factorization(certificate)
        self.assertEqual(encoded[:8], b"SYMFCER1")
        self.assertEqual(verifier.decode_certified_factorization(encoded), certificate)
        self.assertEqual(verifier.encode_certified_factorization(verifier.decode_certified_factorization(encoded)), encoded)

    def test_rejects_wrong_product_missing_proof_and_mismatched_root(self) -> None:
        with self.assertRaisesRegex(verifier.CertificateError, "product mismatch"):
            verifier.verify_factorization(verifier.Factorization(31, ((2, 1), (3, 1), (5, 1))))
        with self.assertRaisesRegex(verifier.CertificateError, "proof count"):
            verifier.verify_certified_factorization(
                verifier.CertifiedFactorization(
                    verifier.Factorization(30, ((2, 1), (3, 1), (5, 1))),
                    (proof_two(), proof_three()),
                )
            )
        wrong_root = verifier.CertifiedFactorization(
            verifier.Factorization(30, ((2, 1), (3, 1), (5, 1))),
            (proof_two(), proof_five(), proof_five()),
        )
        with self.assertRaisesRegex(verifier.CertificateError, "does not match factor"):
            verifier.verify_certified_factorization(wrong_root)

    def test_decoder_rejects_tampered_and_over_budget_outer_fields(self) -> None:
        encoded = verifier.encode_certified_factorization(certificate_thirty())
        bad_magic = b"X" + encoded[1:]
        with self.assertRaises(verifier.CertificateError):
            verifier.decode_certified_factorization(bad_magic)

        wrong_value = bytearray(encoded)
        wrong_value[15] ^= 1
        with self.assertRaisesRegex(verifier.CertificateError, "product mismatch"):
            verifier.decode_certified_factorization(bytes(wrong_value))

        too_many_factors = bytearray(encoded)
        too_many_factors[16:20] = (verifier.MAX_FACTORS + 1).to_bytes(4, "big")
        with self.assertRaisesRegex(verifier.CertificateError, "too many factor entries"):
            verifier.decode_certified_factorization(bytes(too_many_factors))

        mismatched_proofs = bytearray(encoded)
        mismatched_proofs[56:60] = (2).to_bytes(4, "big")
        with self.assertRaisesRegex(verifier.CertificateError, "proof count"):
            verifier.decode_certified_factorization(bytes(mismatched_proofs))

    def test_decoder_rejects_truncation_trailing_and_bad_nested_proof(self) -> None:
        encoded = verifier.encode_certified_factorization(certificate_thirty())
        with self.assertRaisesRegex(verifier.CertificateError, "truncated"):
            verifier.decode_certified_factorization(encoded[:-1])
        with self.assertRaisesRegex(verifier.CertificateError, "trailing bytes"):
            verifier.decode_certified_factorization(encoded + b"\x00")

        bad_nested = bytearray(encoded)
        bad_nested[-1] = 1
        with self.assertRaises(verifier.CertificateError):
            verifier.decode_certified_factorization(bytes(bad_nested))

    def test_file_verifier_reports_format_and_value(self) -> None:
        proof_bytes = verifier.encode_primality_proof(proof_seventeen())
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "prime.cert"
            path.write_bytes(proof_bytes)
            self.assertEqual(
                verifier.verify_file(path),
                {"verified": True, "format": "SYMPPRF1", "value": 17, "certificate_nodes": 2},
            )

    def test_cli_emits_verified_json_for_valid_file(self) -> None:
        proof_bytes = verifier.encode_primality_proof(proof_seventeen())
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "prime.cert"
            path.write_bytes(proof_bytes)
            result = subprocess.run(
                [sys.executable, str(ROOT / "tools" / "verify_math_certificate.py"), str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout),
            {"verified": True, "format": "SYMPPRF1", "value": 17, "certificate_nodes": 2},
        )

def _reference_is_prime(value: int) -> bool:
    """Simple trial-division oracle, independent of the Pocklington checker."""
    if value < 2:
        return False
    divisor = 2
    while divisor <= value // divisor:
        if value % divisor == 0:
            return False
        divisor += 1
    return True


def _reference_factorization(value: int) -> tuple[tuple[int, int], ...]:
    """Factor a small positive integer using test-local trial division."""
    factors: list[tuple[int, int]] = []
    divisor = 2
    remaining = value
    while divisor <= remaining // divisor:
        if remaining % divisor == 0:
            exponent = 0
            while remaining % divisor == 0:
                remaining //= divisor
                exponent += 1
            factors.append((divisor, exponent))
        divisor += 1
    if remaining > 1:
        factors.append((remaining, 1))
    return tuple(factors)


def _small_domain_claim(
    value: int,
    verified_prime_cache: dict[int, verifier.PrimalityProof],
) -> verifier.PrimalityProof:
    """Build a Pocklington-shaped claim; composite roots deliberately use base 2."""
    is_prime = _reference_is_prime(value)
    if is_prime and value in verified_prime_cache:
        return verified_prime_cache[value]
    if value < 3:
        raise ValueError("the base proof for 2 must be supplied in the cache")

    predecessor_factors = _reference_factorization(value - 1)
    dependencies: dict[int, verifier.PrimeNode] = {}
    for prime, _ in predecessor_factors:
        child = _small_domain_claim(prime, verified_prime_cache)
        dependencies.update({node.prime: node for node in child.certificates})

    if is_prime:
        witnesses = []
        for prime, _ in predecessor_factors:
            witness = next(
                (
                    base
                    for base in range(2, value)
                    if pow(base, value - 1, value) == 1
                    and math.gcd((pow(base, (value - 1) // prime, value) - 1) % value, value) == 1
                ),
                None,
            )
            if witness is None:
                raise AssertionError(f"no Pocklington witness found for prime {value}, factor {prime}")
            witnesses.append((prime, witness))
    else:
        # A composite cannot satisfy all Pocklington conditions; use deterministic
        # candidate witnesses to test that the verifier rejects the claim.
        witnesses = [(prime, 2) for prime, _ in predecessor_factors]

    root = verifier.PrimeNode(value, predecessor_factors, tuple(witnesses))
    proof = verifier.PrimalityProof(
        value,
        tuple(sorted((*dependencies.values(), root), key=lambda node: node.prime)),
    )
    if is_prime:
        verified_prime_cache[value] = proof
    return proof


class ExhaustiveSmallDomainTests(unittest.TestCase):
    def test_acceptance_matches_trial_division_for_values_two_through_two_thousand(self) -> None:
        # The expected primality classification comes from a separate, deliberately
        # simple trial-division oracle—not from either certificate implementation.
        cache = {2: proof_two()}
        for value in range(2, 2001):
            with self.subTest(value=value):
                claim = _small_domain_claim(value, cache)
                if _reference_is_prime(value):
                    verifier.verify_primality_proof(claim)
                    encoded = verifier.encode_primality_proof(claim)
                    self.assertEqual(verifier.decode_primality_proof(encoded), claim)
                else:
                    with self.assertRaises(verifier.CertificateError):
                        verifier.verify_primality_proof(claim)

    def test_rejects_carmichael_number_after_fermat_condition_passes(self) -> None:
        # 561 is composite but Carmichael: base 2 satisfies Fermat's congruence,
        # so the checker must enforce Pocklington's per-factor GCD condition too.
        self.assertEqual(pow(2, 560, 561), 1)
        claim = _small_domain_claim(561, {2: proof_two()})
        with self.assertRaisesRegex(verifier.CertificateError, "Pocklington GCD condition failed"):
            verifier.verify_primality_proof(claim)


if __name__ == "__main__":
    unittest.main()
