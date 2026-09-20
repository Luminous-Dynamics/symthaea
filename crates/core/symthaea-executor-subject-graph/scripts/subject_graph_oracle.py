#!/usr/bin/env python3
"""Independent stdlib SHA-256 oracle for EXEC-ID-001D3A canonical transcripts.

This script intentionally does not import or execute Rust. It reconstructs the
language-neutral graph transcript framing directly and checks frozen vectors.
The final match vector is a canonical transcript fixture only; it is not proof
of provider verification and cannot construct a live VerifiedExecutorBinding.
"""

from __future__ import annotations

import hashlib

SCHEMA = 1

NODE_DOMAIN = b"symthaea.executor.subject-graph.node.v1\0"
RELATION_DOMAIN = b"symthaea.executor.subject-graph.relation.v1\0"
POLICY_DOMAIN = b"symthaea.executor.subject-graph.policy.v1\0"
NODE_SET_DOMAIN = b"symthaea.executor.subject-graph.node-set.v1\0"
RELATION_SET_DOMAIN = b"symthaea.executor.subject-graph.relation-set.v1\0"
MATCH_DOMAIN = b"symthaea.executor.subject-graph.match-candidate.v1\0"

EXPECTED = {
    "workload_node": "04d28d04dcc1d9b9b7f5a59572fe17dc7f648f592c2eb18720af5f5f282f0455",
    "software_node": "5629ba7839e08f4e9d1cfb03f4c3c007749698a8b2dc7f45afd7a0f83986bfa3",
    "relation": "fd1176cf74330b36a23da8eb43c259631e9689f29bc194e8ed0b0ee486a7a554",
    "policy": "5eeb222be4f932efb6071e7b44ebe8c3226797ac1232a43690c0fca712fbe1d0",
    "node_set": "77234e7048d842a4065e83dede36ae71b78a754659ea836a2682dc66fa2eb25e",
    "relation_set": "925dfd62cfa84e127f42c822bd8767701158201abc6debc86501e1e9f9ef5c53",
    "match": "b64e2c851fb1d7b1535162d53b4062c676b3b3713e71b0dff8c4aa285876c54f",
}


def d(byte: int) -> bytes:
    return bytes([byte]) * 32


def u16(value: int) -> bytes:
    return value.to_bytes(2, "big")


def u32(value: int) -> bytes:
    return value.to_bytes(4, "big")


def digest(domain: bytes, *parts: bytes) -> bytes:
    h = hashlib.sha256()
    h.update(domain)
    for part in parts:
        h.update(part)
    return h.digest()


def node(
    dimension: int,
    subject: bytes,
    challenge: bytes,
    runtime: bytes,
    verifier: bytes,
    evidence: bytes,
) -> bytes:
    return digest(
        NODE_DOMAIN,
        u16(SCHEMA),
        u16(dimension),
        subject,
        challenge,
        runtime,
        verifier,
        evidence,
    )


def relation(
    relation_class: int,
    assurance: int,
    left: bytes,
    right: bytes,
    challenge: bytes,
    runtime: bytes,
    verifier: bytes,
    evidence: bytes,
) -> bytes:
    return digest(
        RELATION_DOMAIN,
        u16(SCHEMA),
        u16(relation_class),
        u16(assurance),
        left,
        right,
        challenge,
        runtime,
        verifier,
        evidence,
    )


def policy(minima: tuple[int, int, int, int]) -> bytes:
    parts = [u16(SCHEMA)]
    for relation_class, minimum in enumerate(minima):
        parts.extend((u16(relation_class), u16(minimum)))
    return digest(POLICY_DOMAIN, *parts)


def digest_set(domain: bytes, members: list[bytes]) -> bytes:
    members = sorted(members)
    return digest(domain, u16(SCHEMA), u32(len(members)), *members)


def match_candidate(
    challenge: bytes,
    requirement: bytes,
    runtime: bytes,
    relation_policy: bytes,
    workload_subject: bytes,
    workload_node: bytes,
    node_set: bytes,
    relation_set: bytes,
) -> bytes:
    return digest(
        MATCH_DOMAIN,
        u16(SCHEMA),
        challenge,
        requirement,
        runtime,
        relation_policy,
        workload_subject,
        workload_node,
        node_set,
        relation_set,
    )


def assert_vector(name: str, value: bytes) -> None:
    actual = value.hex()
    expected = EXPECTED[name]
    if actual != expected:
        raise SystemExit(f"{name}: expected {expected}, got {actual}")
    print(f"{name} {actual}")


def main() -> None:
    challenge = d(0x22)
    runtime = d(0x33)

    workload = node(2, d(0x11), challenge, runtime, d(0x44), d(0x55))
    software = node(3, d(0x66), challenge, runtime, d(0x77), d(0x88))
    edge = relation(1, 2, workload, software, challenge, runtime, d(0x99), d(0xAA))
    relation_policy = policy((1, 2, 2, 3))
    nodes = digest_set(NODE_SET_DOMAIN, [workload, software])
    relations = digest_set(RELATION_SET_DOMAIN, [edge])
    matched = match_candidate(
        challenge,
        d(0xBB),
        runtime,
        relation_policy,
        d(0x11),
        workload,
        nodes,
        relations,
    )

    assert_vector("workload_node", workload)
    assert_vector("software_node", software)
    assert_vector("relation", edge)
    assert_vector("policy", relation_policy)
    assert_vector("node_set", nodes)
    assert_vector("relation_set", relations)
    assert_vector("match", matched)


if __name__ == "__main__":
    main()
