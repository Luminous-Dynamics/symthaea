#!/usr/bin/env python3
"""Independent ChaCha8 + lattice stream-allocation oracle for LQCD-015.

Standard-library only. Imports no Symthaea or Rust RNG code.
"""

import argparse
import hashlib
import struct

MASK32 = 0xFFFF_FFFF
DOMAIN_GAUGE_TRANSITION = 1


def rotl32(x, n):
    return ((x << n) & MASK32) | (x >> (32 - n))


def qr(x, a, b, c, d):
    x[a] = (x[a] + x[b]) & MASK32
    x[d] ^= x[a]
    x[d] = rotl32(x[d], 16)
    x[c] = (x[c] + x[d]) & MASK32
    x[b] ^= x[c]
    x[b] = rotl32(x[b], 12)
    x[a] = (x[a] + x[b]) & MASK32
    x[d] ^= x[a]
    x[d] = rotl32(x[d], 8)
    x[c] = (x[c] + x[d]) & MASK32
    x[b] ^= x[c]
    x[b] = rotl32(x[b], 7)


def chacha8_block(seed, counter, stream):
    if len(seed) != 32:
        raise ValueError("seed must be exactly 32 bytes")
    constants = [0x61707865, 0x3320646E, 0x79622D32, 0x6B206574]
    key = list(struct.unpack("<8I", seed))
    state = constants + key + [
        counter & MASK32,
        (counter >> 32) & MASK32,
        stream & MASK32,
        (stream >> 32) & MASK32,
    ]
    work = state.copy()
    for _ in range(4):
        qr(work, 0, 4, 8, 12)
        qr(work, 1, 5, 9, 13)
        qr(work, 2, 6, 10, 14)
        qr(work, 3, 7, 11, 15)
        qr(work, 0, 5, 10, 15)
        qr(work, 1, 6, 11, 12)
        qr(work, 2, 7, 8, 13)
        qr(work, 3, 4, 9, 14)
    out = [(work[i] + state[i]) & MASK32 for i in range(16)]
    return struct.pack("<16I", *out)


def pack_stream_id(domain, ensemble_slot, replica, rank):
    if not 1 <= domain <= 255:
        raise ValueError("domain must fit nonzero u8")
    if not 0 <= ensemble_slot < (1 << 24):
        raise ValueError("ensemble_slot must fit 24 bits")
    if not 0 <= replica < (1 << 16):
        raise ValueError("replica must fit u16")
    if not 0 <= rank < (1 << 16):
        raise ValueError("rank must fit u16")
    return (domain << 56) | (ensemble_slot << 32) | (replica << 16) | rank


def open01_from_u64(x):
    k = (x >> 12) + 1
    denom = (1 << 52) + 1
    value = k / denom
    if not 0.0 < value < 1.0:
        raise AssertionError(value)
    return value


def self_test():
    # Published ChaCha8 256-bit all-zero key / 64-bit nonce test vector.
    expected_zero = bytes.fromhex(
        "3e00ef2f895f40d67f5bb8e81f09a5a1"
        "2c840ec3ce9a7f3b181be188ef711a1e"
        "984ce172b9216f419f445367456d5619"
        "314a42a3da86b001387bfdb80e0cfe42"
    )
    got_zero = chacha8_block(bytes(32), 0, 0)
    if got_zero != expected_zero:
        raise AssertionError((got_zero.hex(), expected_zero.hex()))

    seed = bytes(range(32))
    stream = pack_stream_id(DOMAIN_GAUGE_TRANSITION, 0x123456, 0x789A, 0xBCDE)
    expected = bytes.fromhex(
        "138cca6940325d71f9ab52ed1f9a9728"
        "3e01c5f699dc6a43545012c55c4e2237"
        "197bb7880bd322dc5f207b916e1b5009"
        "ead716b18bcd82a5259290fd33db2f9d"
    )
    got = chacha8_block(seed, 0, stream)
    if got != expected:
        raise AssertionError((got.hex(), expected.hex()))

    adjacent = pack_stream_id(DOMAIN_GAUGE_TRANSITION, 0x123456, 0x789A, 0xBCDF)
    if chacha8_block(seed, 0, adjacent) == got:
        raise AssertionError("adjacent stream fixture collided")

    u_min = open01_from_u64(0)
    u_max = open01_from_u64((1 << 64) - 1)
    for x in [0, 1, 0x123456789ABCDEF0, (1 << 64) - 2, (1 << 64) - 1]:
        a = open01_from_u64(x)
        b = open01_from_u64(((1 << 64) - 1) ^ x)
        if abs((a + b) - 1.0) > 2e-16:
            raise AssertionError((x, a, b, a + b))

    print("ok")
    print(f"zero_vector_sha256={hashlib.sha256(got_zero).hexdigest()}")
    print(f"packed_stream_id=0x{stream:016x}")
    print(f"qualification_block={got.hex()}")
    print(f"qualification_block_sha256={hashlib.sha256(got).hexdigest()}")
    print(f"open01_min={u_min:.17g}")
    print(f"open01_max={u_max:.17g}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("only --self-test is supported; this is a qualification oracle")
    self_test()


if __name__ == "__main__":
    main()
