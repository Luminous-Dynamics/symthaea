#!/usr/bin/env python3
"""Translate verified Symthaea certificates into Lean kernel-checkable claims.

This tool is a bounded proof bridge, not a formal proof of the Rust/Python
decoders. It verifies the exact bytes, emits the decoded wire structure and
bytes together, and asks Lean to prove that its independent canonical encoder
reproduces those bytes before proving the prime/factorization claims.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import verify_math_certificate as verifier
from tools.math_certificate_golden_vectors import (
    SYMFCER1_FACTOR_2,
    SYMPPRF1_PRIME_3,
    SYMPPRF1_PRIME_13_MULTI_FACTOR,
)

# The protocol verifier accepts up to 1 MiB. The formal bridge uses a lower
# bound because it emits every byte as a Lean numeral and checks it by reduction.
MAX_FORMAL_CERTIFICATE_BYTES = 65_536


def _combined_pocklington_witness(node: verifier.PrimeNode) -> int:
    """Combine per-factor witnesses into one root accepted by PrimeCert.

    For m = p - 1, raising a q-witness to m/q^e isolates its q-primary
    component. Those components have pairwise-coprime orders; their product is
    checked against every Pocklington congruence before any Lean is emitted.
    The generated PrimeCert proof and Lean kernel check remain authoritative.
    """
    p = node.prime
    if p <= 2:
        raise verifier.CertificateError("the base prime has no Pocklington witness")

    m = p - 1
    witness_by_factor = dict(node.witnesses)
    root = 1
    for q, exponent in node.predecessor_factors:
        witness = witness_by_factor.get(q)
        if witness is None:
            raise verifier.CertificateError(
                f"missing Pocklington witness for p={p}, factor={q}"
            )
        q_power = q**exponent
        if q_power <= 0 or m % q_power != 0:
            raise verifier.CertificateError(
                f"invalid predecessor factor power for p={p}, factor={q}"
            )
        component = pow(witness, m // q_power, p)
        root = (root * component) % p

    if not 1 < root < p:
        raise verifier.CertificateError(
            f"combined Pocklington witness for p={p} is outside 1 < a < p"
        )
    if pow(root, m, p) != 1:
        raise verifier.CertificateError(
            f"combined Pocklington witness fails Fermat congruence for p={p}"
        )
    for q, _ in node.predecessor_factors:
        residue = pow(root, m // q, p)
        if math.gcd((residue - 1) % p, p) != 1:
            raise verifier.CertificateError(
                f"combined Pocklington witness fails GCD condition for p={p}, q={q}"
            )
    return root


def _factor_expression(factors: tuple[tuple[int, int], ...]) -> str:
    if not factors:
        return "1"
    return " * ".join(
        str(prime) if exponent == 1 else f"{prime} ^ {exponent}"
        for prime, exponent in factors
    )


def _lean_pairs(pairs: tuple[tuple[int, int], ...]) -> str:
    return "[" + ", ".join(f"({left}, {right})" for left, right in pairs) + "]"


def _lean_bytes(data: bytes) -> str:
    return "[" + ", ".join(str(byte) for byte in data) + "]"


def _wire_proof_definition(
    proof: verifier.PrimalityProof,
    identifier: str,
) -> str:
    nodes = []
    for node in proof.certificates:
        nodes.append(
            "    { prime := "
            f"{node.prime}, predecessorFactors := "
            f"{_lean_pairs(node.predecessor_factors)}, witnesses := "
            f"{_lean_pairs(node.witnesses)} }}"
        )
    node_list = "[\n" + ",\n".join(nodes) + "\n  ]"
    return (
        f"def {identifier} : WirePrimalityProof :=\n"
        f"  {{ value := {proof.value},\n"
        f"    certificates := {node_list} }}"
    )


def _lean_prime_tactic(proof: verifier.PrimalityProof) -> str:
    if proof.value == 2:
        return "norm_num"
    if not proof.certificates or proof.certificates[0].prime != 2:
        raise verifier.CertificateError(
            f"proof for {proof.value} does not begin with the certified base 2"
        )
    if proof.certificates[-1].prime != proof.value:
        raise verifier.CertificateError(
            f"proof table for {proof.value} does not end at its root"
        )

    steps = ["small {2}"]
    for node in proof.certificates:
        if node.prime == 2:
            continue
        witness = _combined_pocklington_witness(node)
        factors = _factor_expression(node.predecessor_factors)
        steps.append(f"pock ({node.prime}, {witness}, {factors})")
    return "prime_cert [" + ", ".join(steps) + "]"


def _check_formal_size(data: bytes) -> None:
    if len(data) > MAX_FORMAL_CERTIFICATE_BYTES:
        raise verifier.CertificateError(
            "formal proof bridge limit is 65536 bytes; the protocol verifier "
            "supports larger inputs"
        )


def _render_primality_proof(data: bytes) -> list[str]:
    proof = verifier.decode_primality_proof(data)
    lines = [
        _wire_proof_definition(proof, "source_primality_proof"),
        f"def source_bytes : List Nat := {_lean_bytes(data)}",
        "",
        "theorem source_proof_fields_fit :",
        "    wirePrimalityProofFieldsFit source_primality_proof = true := by",
        "  decide",
        "",
        "theorem source_bytes_match_canonical_encoding :",
        "    encodeWirePrimalityProof source_primality_proof = source_bytes := by",
        "  decide",
        "",
        "theorem source_root_matches_claim :",
        f"    source_primality_proof.value = {proof.value} := by",
        "  decide",
        "",
        f"theorem root_{proof.value}_is_prime :",
        "    Nat.Prime source_primality_proof.value := by",
        "  rw [source_root_matches_claim]",
        f"  {_lean_prime_tactic(proof)}",
    ]
    return lines


def _render_factorization(data: bytes) -> list[str]:
    bundle = verifier.decode_certified_factorization(data)
    factorization = bundle.factorization
    proof_names = [
        f"source_factor_proof_{index}"
        for index in range(len(bundle.prime_proofs))
    ]
    lines: list[str] = []
    for name, proof in zip(proof_names, bundle.prime_proofs):
        lines.extend([_wire_proof_definition(proof, name), ""])

    factors = _lean_pairs(factorization.factors)
    proof_refs = "[" + ", ".join(proof_names) + "]"
    lines.extend(
        [
            "def source_certificate : WireCertifiedFactorization :=",
            "  { factorization :=",
            f"      {{ value := {factorization.value}, factors := {factors} }},",
            f"    primeProofs := {proof_refs} }}",
            f"def source_bytes : List Nat := {_lean_bytes(data)}",
            "",
            "theorem source_certificate_fields_fit :",
            "    wireCertifiedFactorizationFieldsFit source_certificate = true := by",
            "  decide",
            "",
            "theorem source_bytes_match_canonical_encoding :",
            "    encodeWireCertifiedFactorization source_certificate = source_bytes := by",
            "  decide",
            "",
        ]
    )

    for index, ((prime, _exponent), proof, name) in enumerate(
        zip(factorization.factors, bundle.prime_proofs, proof_names)
    ):
        root_match = f"source_factor_proof_{index}_root_matches_factor"
        lines.extend(
            [
                f"theorem {root_match} : {name}.value = {prime} := by",
                "  decide",
                "",
                f"theorem factor_{index}_{prime}_is_prime : Nat.Prime {name}.value := by",
                f"  rw [{root_match}]",
                f"  {_lean_prime_tactic(proof)}",
                "",
            ]
        )

    product = _factor_expression(factorization.factors)
    lines.extend(
        [
            "theorem factorization_reconstructs :",
            f"    source_certificate.factorization.value = ({product}) := by",
            "  norm_num [source_certificate]",
        ]
    )
    return lines


def render_certificate(data: bytes) -> str:
    """Validate bytes and generate a Lean proof tied to their canonical encoding."""
    _check_formal_size(data)
    lines = [
        "import SymthaeaMathCertificates",
        "",
        "open SymthaeaMathCertificates",
        "",
        "namespace SymthaeaGenerated",
        "",
    ]

    if data.startswith(verifier.PRIMALITY_MAGIC):
        lines.extend(_render_primality_proof(data))
    elif data.startswith(verifier.FACTORIZATION_MAGIC):
        lines.extend(_render_factorization(data))
    else:
        raise verifier.CertificateError("unrecognized certificate format")

    lines.extend(["", "end SymthaeaGenerated", ""])
    return "\n".join(lines)


def _read_limited(path: Path) -> bytes:
    with path.open("rb") as handle:
        data = handle.read(MAX_FORMAL_CERTIFICATE_BYTES + 1)
    _check_formal_size(data)
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a SYMPPRF1/SYMFCER1 certificate and emit Lean source that "
            "checks canonical re-encoding and the mathematical claims."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--certificate",
        type=Path,
        help="path to a binary SYMPPRF1 or SYMFCER1 certificate (up to 65536 bytes)",
    )
    source.add_argument(
        "--golden-vector",
        choices=("primality-3", "factorization-2", "prime-13-multifactor"),
        help="emit Lean for a fixed test fixture",
    )
    parser.add_argument("--output", required=True, type=Path, help="Lean source output path")
    args = parser.parse_args(argv)

    try:
        if args.golden_vector == "primality-3":
            data = SYMPPRF1_PRIME_3
        elif args.golden_vector == "factorization-2":
            data = SYMFCER1_FACTOR_2
        elif args.golden_vector == "prime-13-multifactor":
            data = SYMPPRF1_PRIME_13_MULTI_FACTOR
        else:
            data = _read_limited(args.certificate)

        _check_formal_size(data)
        lean_source = render_certificate(data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(lean_source, encoding="utf-8")
    except (verifier.CertificateError, OSError) as exc:
        print(json.dumps({"written": False, "error": str(exc)}), file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "written": True,
                "output": str(args.output),
                "source_bytes": len(data),
                "lean_bytes": len(lean_source.encode("utf-8")),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
