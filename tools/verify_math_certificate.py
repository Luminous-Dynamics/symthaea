#!/usr/bin/env python3
"""Standalone verifier for Symthaea SYMPPRF1 and SYMFCER1 math certificates.

This tool intentionally uses only the Python standard library. It does not import
Symthaea, invoke Cargo, or run a factorization/primality search to accept a proof.
It checks the serialized evidence directly with Pocklington's criterion.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

PRIMALITY_MAGIC = b"SYMPPRF1"
FACTORIZATION_MAGIC = b"SYMFCER1"
MAX_WIRE_BYTES = 1_048_576
MAX_CERTIFICATES = 4_096
MAX_PROOF_ENTRIES = 65_536
MAX_FACTORS = 4_096
MAX_PROOFS = 4_096
U32_MAX = (1 << 32) - 1
U64_MAX = (1 << 64) - 1


class CertificateError(ValueError):
    """Input is malformed or the claimed mathematical evidence is invalid."""


@dataclass(frozen=True)
class PrimeNode:
    prime: int
    predecessor_factors: tuple[tuple[int, int], ...]
    witnesses: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class PrimalityProof:
    value: int
    certificates: tuple[PrimeNode, ...]


@dataclass(frozen=True)
class Factorization:
    value: int
    factors: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class CertifiedFactorization:
    factorization: Factorization
    prime_proofs: tuple[PrimalityProof, ...]


class Reader:
    """Bounds-checked reader over an already size-limited byte string."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 0

    @property
    def remaining(self) -> int:
        return len(self.data) - self.offset

    def take(self, length: int) -> bytes:
        if length < 0:
            raise CertificateError("negative field length")
        end = self.offset + length
        if end > len(self.data):
            raise CertificateError("truncated input")
        out = self.data[self.offset:end]
        self.offset = end
        return out

    def u32(self) -> int:
        return struct.unpack(">I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack(">Q", self.take(8))[0]


def _pack_u32(value: int) -> bytes:
    if value < 0 or value > U32_MAX:
        raise CertificateError("u32 value out of range")
    return struct.pack(">I", value)


def _pack_u64(value: int) -> bytes:
    if value < 0 or value > U64_MAX:
        raise CertificateError("u64 value out of range")
    return struct.pack(">Q", value)


def _checked_product(value: int, factors: Sequence[tuple[int, int]]) -> int:
    if value < 1 or value > U64_MAX:
        raise CertificateError("factorization input must be in 1..=u64::MAX")
    if len(factors) > MAX_FACTORS:
        raise CertificateError("too many factor entries")
    product = 1
    previous = 1
    for prime, exponent in factors:
        if prime <= previous or prime > U64_MAX:
            raise CertificateError("factors must be strictly increasing positive u64 values")
        if exponent <= 0 or exponent > U32_MAX:
            raise CertificateError("factor exponents must be positive u32 values")
        previous = prime
        for _ in range(exponent):
            product *= prime
            if product > U64_MAX:
                raise CertificateError("factorization product overflows u64")
    if product != value:
        raise CertificateError(f"factorization product mismatch: expected {value}, got {product}")
    return product


def verify_primality_proof(proof: PrimalityProof) -> None:
    """Verify recursive Pocklington certificates without Miller-Rabin."""
    if proof.value < 2 or proof.value > U64_MAX:
        raise CertificateError("primality proof root must be in 2..=u64::MAX")
    nodes = proof.certificates
    if not nodes:
        raise CertificateError("proof table is empty")
    if len(nodes) > MAX_CERTIFICATES:
        raise CertificateError("too many primality certificate nodes")

    total_entries = 0
    for node in nodes:
        total_entries += len(node.predecessor_factors) + len(node.witnesses)
        if total_entries > MAX_PROOF_ENTRIES:
            raise CertificateError("too many proof entries")

    by_prime: dict[int, PrimeNode] = {}
    previous_node = 0
    for node in nodes:
        if node.prime <= previous_node:
            raise CertificateError("certificate nodes must be strictly increasing")
        if node.prime < 2 or node.prime > U64_MAX:
            raise CertificateError("certificate candidate outside u64 range")
        previous_node = node.prime
        by_prime[node.prime] = node

    if proof.value not in by_prime:
        raise CertificateError("proof table does not contain its root")

    verified: set[int] = set()
    for node in nodes:
        p = node.prime
        factors = node.predecessor_factors
        witnesses = node.witnesses

        if p == 2:
            if factors or witnesses:
                raise CertificateError("prime 2 base node must have no factors or witnesses")
            verified.add(p)
            continue
        if p < 3 or p % 2 == 0:
            raise CertificateError(f"non-base candidate {p} is not an odd integer >= 3")
        if len(factors) > MAX_PROOF_ENTRIES or len(witnesses) > MAX_PROOF_ENTRIES:
            raise CertificateError("too many factor or witness entries")

        product = 1
        previous_factor = 1
        for q, exponent in factors:
            if q <= previous_factor:
                raise CertificateError(f"predecessor factors for {p} are not strictly increasing")
            if exponent <= 0 or exponent > U32_MAX:
                raise CertificateError(f"predecessor exponent for {q} is not positive u32")
            if q not in verified:
                raise CertificateError(f"predecessor prime {q} was not proved before {p}")
            previous_factor = q
            for _ in range(exponent):
                product *= q
                if product > U64_MAX:
                    raise CertificateError(f"predecessor product for {p} overflows u64")

        if product != p - 1:
            raise CertificateError(f"predecessor product for {p} is {product}, expected {p - 1}")
        # Since the complete factorization of p-1 is included, F = p-1.
        if product * product <= p:
            raise CertificateError(f"Pocklington size condition F^2 > p fails for {p}")
        if len(witnesses) != len(factors):
            raise CertificateError(f"witness count does not match distinct factors for {p}")

        for (q, _), (witness_q, base) in zip(factors, witnesses):
            if q != witness_q:
                raise CertificateError(f"witness order does not match factor order for {p}")
            if base <= 1 or base >= p:
                raise CertificateError(f"witness base {base} outside 1 < a < {p}")
            if pow(base, p - 1, p) != 1:
                raise CertificateError(f"Fermat congruence failed for p={p}, base={base}")
            residue = pow(base, (p - 1) // q, p)
            if math.gcd((residue - 1) % p, p) != 1:
                raise CertificateError(f"Pocklington GCD condition failed for p={p}, q={q}")
        verified.add(p)

    reachable: set[int] = set()
    pending = [proof.value]
    while pending:
        p = pending.pop()
        if p in reachable:
            continue
        reachable.add(p)
        node = by_prime.get(p)
        if node is None:
            raise CertificateError(f"missing proof node for dependency {p}")
        pending.extend(q for q, _ in node.predecessor_factors)
    if reachable != set(by_prime):
        extras = sorted(set(by_prime) - reachable)
        raise CertificateError(f"unreachable proof nodes: {extras}")


def encode_primality_proof(proof: PrimalityProof) -> bytes:
    if len(proof.certificates) > MAX_CERTIFICATES:
        raise CertificateError("too many certificate nodes")
    total_entries = 0
    for node in proof.certificates:
        total_entries += len(node.predecessor_factors) + len(node.witnesses)
        if total_entries > MAX_PROOF_ENTRIES:
            raise CertificateError("too many proof entries")
    verify_primality_proof(proof)

    out = bytearray(PRIMALITY_MAGIC)
    out += _pack_u64(proof.value)
    out += _pack_u32(len(proof.certificates))
    for node in proof.certificates:
        out += _pack_u64(node.prime)
        out += _pack_u32(len(node.predecessor_factors))
        for q, exponent in node.predecessor_factors:
            out += _pack_u64(q)
            out += _pack_u32(exponent)
        out += _pack_u32(len(node.witnesses))
        for q, base in node.witnesses:
            out += _pack_u64(q)
            out += _pack_u64(base)
        if len(out) > MAX_WIRE_BYTES:
            raise CertificateError("primality proof exceeds 1 MiB")
    if len(out) > MAX_WIRE_BYTES:
        raise CertificateError("primality proof exceeds 1 MiB")
    return bytes(out)


def decode_primality_proof(data: bytes) -> PrimalityProof:
    if len(data) > MAX_WIRE_BYTES:
        raise CertificateError("primality proof exceeds 1 MiB")
    reader = Reader(data)
    if reader.take(8) != PRIMALITY_MAGIC:
        raise CertificateError("invalid SYMPPRF1 magic/version")
    value = reader.u64()
    count = reader.u32()
    if count > MAX_CERTIFICATES:
        raise CertificateError("too many primality certificate nodes")
    if count > reader.remaining // 16:
        raise CertificateError("truncated certificate table")

    nodes: list[PrimeNode] = []
    total_entries = 0
    for _ in range(count):
        prime = reader.u64()
        factor_count = reader.u32()
        total_entries += factor_count
        if total_entries > MAX_PROOF_ENTRIES:
            raise CertificateError("too many proof entries")
        if factor_count > reader.remaining // 12:
            raise CertificateError("truncated predecessor-factor list")
        factors = tuple((reader.u64(), reader.u32()) for _ in range(factor_count))

        witness_count = reader.u32()
        total_entries += witness_count
        if total_entries > MAX_PROOF_ENTRIES:
            raise CertificateError("too many proof entries")
        if witness_count > reader.remaining // 16:
            raise CertificateError("truncated witness list")
        witnesses = tuple((reader.u64(), reader.u64()) for _ in range(witness_count))
        nodes.append(PrimeNode(prime, factors, witnesses))

    if reader.remaining:
        raise CertificateError("trailing bytes after primality proof")
    proof = PrimalityProof(value, tuple(nodes))
    verify_primality_proof(proof)
    if encode_primality_proof(proof) != data:
        raise CertificateError("noncanonical SYMPPRF1 encoding")
    return proof


def verify_factorization(factorization: Factorization) -> None:
    _checked_product(factorization.value, factorization.factors)


def verify_certified_factorization(certificate: CertifiedFactorization) -> None:
    factors = certificate.factorization.factors
    proofs = certificate.prime_proofs
    if len(factors) > MAX_FACTORS:
        raise CertificateError("too many factor entries")
    if len(proofs) > MAX_PROOFS:
        raise CertificateError("too many nested prime proofs")
    if len(proofs) != len(factors):
        raise CertificateError("prime-proof count does not match factor count")
    verify_factorization(certificate.factorization)
    for (factor, _), proof in zip(factors, proofs):
        if proof.value != factor:
            raise CertificateError(f"prime proof root {proof.value} does not match factor {factor}")
        verify_primality_proof(proof)


def encode_certified_factorization(certificate: CertifiedFactorization) -> bytes:
    factors = certificate.factorization.factors
    proofs = certificate.prime_proofs
    if len(factors) > MAX_FACTORS:
        raise CertificateError("too many factor entries")
    if len(proofs) > MAX_PROOFS:
        raise CertificateError("too many nested prime proofs")
    verify_certified_factorization(certificate)

    out = bytearray(FACTORIZATION_MAGIC)
    out += _pack_u64(certificate.factorization.value)
    out += _pack_u32(len(factors))
    for prime, exponent in factors:
        out += _pack_u64(prime)
        out += _pack_u32(exponent)
    out += _pack_u32(len(proofs))
    for proof in proofs:
        proof_bytes = encode_primality_proof(proof)
        projected_length = len(out) + 4 + len(proof_bytes)
        if projected_length > MAX_WIRE_BYTES:
            raise CertificateError("certified factorization exceeds 1 MiB")
        out += _pack_u32(len(proof_bytes))
        out += proof_bytes
    if len(out) > MAX_WIRE_BYTES:
        raise CertificateError("certified factorization exceeds 1 MiB")
    return bytes(out)


def decode_certified_factorization(data: bytes) -> CertifiedFactorization:
    if len(data) > MAX_WIRE_BYTES:
        raise CertificateError("certified factorization exceeds 1 MiB")
    reader = Reader(data)
    if reader.take(8) != FACTORIZATION_MAGIC:
        raise CertificateError("invalid SYMFCER1 magic/version")
    value = reader.u64()
    factor_count = reader.u32()
    if factor_count > MAX_FACTORS:
        raise CertificateError("too many factor entries")
    if factor_count > reader.remaining // 12:
        raise CertificateError("truncated factor list")
    factors = tuple((reader.u64(), reader.u32()) for _ in range(factor_count))

    proof_count = reader.u32()
    if proof_count > MAX_PROOFS:
        raise CertificateError("too many nested prime proofs")
    if proof_count != factor_count:
        raise CertificateError("prime-proof count does not match factor count")
    if proof_count > reader.remaining // 4:
        raise CertificateError("truncated nested proof table")

    proofs: list[PrimalityProof] = []
    for _ in range(proof_count):
        length = reader.u32()
        if length > MAX_WIRE_BYTES:
            raise CertificateError("nested primality proof exceeds 1 MiB")
        proofs.append(decode_primality_proof(reader.take(length)))

    if reader.remaining:
        raise CertificateError("trailing bytes after certified factorization")
    certificate = CertifiedFactorization(Factorization(value, factors), tuple(proofs))
    verify_certified_factorization(certificate)
    if encode_certified_factorization(certificate) != data:
        raise CertificateError("noncanonical SYMFCER1 encoding")
    return certificate


def verify_file(path: Path) -> dict[str, object]:
    # Read at most limit+1 bytes so a hostile file cannot force an unbounded read.
    with path.open("rb") as handle:
        data = handle.read(MAX_WIRE_BYTES + 1)
    if len(data) > MAX_WIRE_BYTES:
        raise CertificateError("certificate file exceeds 1 MiB")
    if data.startswith(PRIMALITY_MAGIC):
        proof = decode_primality_proof(data)
        return {
            "verified": True,
            "format": "SYMPPRF1",
            "value": proof.value,
            "certificate_nodes": len(proof.certificates),
        }
    if data.startswith(FACTORIZATION_MAGIC):
        certificate = decode_certified_factorization(data)
        return {
            "verified": True,
            "format": "SYMFCER1",
            "value": certificate.factorization.value,
            "factor_count": len(certificate.factorization.factors),
            "prime_proof_count": len(certificate.prime_proofs),
        }
    raise CertificateError("unrecognized certificate format")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify Symthaea SYMPPRF1 or SYMFCER1 proof bytes using Python's standard library."
    )
    parser.add_argument("certificate", type=Path, help="path to a certificate file")
    args = parser.parse_args(argv)
    try:
        result = verify_file(args.certificate)
    except (CertificateError, OSError, struct.error) as exc:
        print(json.dumps({"verified": False, "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
