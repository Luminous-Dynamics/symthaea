#!/usr/bin/env python3
"""Translate verified Symthaea certificates into Lean kernel-checkable claims.

This tool is a proof bridge, not a formal proof of the Rust/Python decoders. It
first validates the canonical bytes with the independent Python verifier, then
constructs a Lean PrimeCert proof of each claimed prime and an exact Lean
arithmetic equality for the outer factorization.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from tools import verify_math_certificate as verifier
from tools.math_certificate_golden_vectors import (
    SYMFCER1_FACTOR_2,
    SYMPPRF1_PRIME_3,
)


def _combined_pocklington_witness(node: verifier.PrimeNode) -> int:
    """Combine per-factor witnesses into one root accepted by PrimeCert.

    For m = p - 1, raising a q-witness to m/q^e isolates its q-primary
    component. Those components have pairwise-coprime orders; their product is
    checked against every Pocklington congruence before any Lean is emitted.
    The Lean tactic then reconstructs the proof term and the kernel checks it.
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


def _lean_prime_theorem(
    proof: verifier.PrimalityProof,
    theorem_name: str,
) -> str:
    """Emit a theorem whose proof term is built by Lean's PrimeCert tactic."""
    if proof.value == 2:
        tactic = "by\n  norm_num"
    else:
        if not proof.certificates or proof.certificates[0].prime != 2:
            raise verifier.CertificateError(
                f"proof for {proof.value} does not begin with the certified base 2"
            )
        steps = ["small {2}"]
        for node in proof.certificates:
            if node.prime == 2:
                continue
            witness = _combined_pocklington_witness(node)
            factors = _factor_expression(node.predecessor_factors)
            steps.append(f"pock ({node.prime}, {witness}, {factors})")
        if proof.certificates[-1].prime != proof.value:
            raise verifier.CertificateError(
                f"proof table for {proof.value} does not end at its root"
            )
        tactic = "by\n  prime_cert [" + ", ".join(steps) + "]"
    return f"theorem {theorem_name} : Nat.Prime {proof.value} := {tactic}"


def render_certificate(data: bytes) -> str:
    """Validate canonical bytes and return a self-contained Lean source file."""
    if len(data) > verifier.MAX_WIRE_BYTES:
        raise verifier.CertificateError("certificate exceeds 1 MiB")

    lines = [
        "import PrimeCert",
        "",
        "namespace SymthaeaGenerated",
        "",
    ]

    if data.startswith(verifier.PRIMALITY_MAGIC):
        proof = verifier.decode_primality_proof(data)
        lines.append(_lean_prime_theorem(proof, f"root_{proof.value}_is_prime"))
    elif data.startswith(verifier.FACTORIZATION_MAGIC):
        bundle = verifier.decode_certified_factorization(data)
        factorization = bundle.factorization
        for index, ((prime, _exponent), proof) in enumerate(
            zip(factorization.factors, bundle.prime_proofs)
        ):
            lines.append(
                _lean_prime_theorem(proof, f"factor_{index}_{prime}_is_prime")
            )
            lines.append("")

        product = _factor_expression(factorization.factors)
        lines.append(
            "theorem factorization_reconstructs : "
            f"{factorization.value} = ({product}) := by\n  norm_num"
        )
    else:
        raise verifier.CertificateError("unrecognized certificate format")

    lines.extend(["", "end SymthaeaGenerated", ""])
    return "\n".join(lines)


def _read_limited(path: Path) -> bytes:
    with path.open("rb") as handle:
        data = handle.read(verifier.MAX_WIRE_BYTES + 1)
    if len(data) > verifier.MAX_WIRE_BYTES:
        raise verifier.CertificateError("certificate file exceeds 1 MiB")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a SYMPPRF1/SYMFCER1 certificate and emit Lean PrimeCert "
            "claims for kernel checking."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--certificate",
        type=Path,
        help="path to a binary SYMPPRF1 or SYMFCER1 certificate",
    )
    source.add_argument(
        "--golden-vector",
        choices=("primality-3", "factorization-2"),
        help="emit Lean for a fixed cross-language golden vector",
    )
    parser.add_argument("--output", required=True, type=Path, help="Lean source output path")
    args = parser.parse_args(argv)

    try:
        if args.golden_vector == "primality-3":
            data = SYMPPRF1_PRIME_3
        elif args.golden_vector == "factorization-2":
            data = SYMFCER1_FACTOR_2
        else:
            data = _read_limited(args.certificate)

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
