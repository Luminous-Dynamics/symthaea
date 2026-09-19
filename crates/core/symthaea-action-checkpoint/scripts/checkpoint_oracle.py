#!/usr/bin/env python3
"""Independent canonical SHA-256 oracle for ACTION-RUNTIME-V2B.

Uses Python stdlib only and imports no Symthaea production code.
"""

from __future__ import annotations

import hashlib
import struct

DOMAIN = b"symthaea.action-checkpoint.v2\0"
EXPECTED = "3a756277007027ff06ccaf77c521220db8dd4ce4898c1e8922aa0000fc979efc"


def u16(value: int) -> bytes:
    return struct.pack(">H", value)


def u32(value: int) -> bytes:
    return struct.pack(">I", value)


def u64(value: int) -> bytes:
    return struct.pack(">Q", value)


def d(byte: int) -> bytes:
    return bytes([byte]) * 32


def risk(mutation: int, irreversible: int, disclosure: int, monetary: int) -> bytes:
    return b"".join(map(u64, (mutation, irreversible, disclosure, monetary)))


def fixture_preimage() -> bytes:
    out = bytearray(DOMAIN)
    out += u16(2)                         # checkpoint schema
    out += u64(7)                         # sequence
    out += b"\x01" + d(0x11)             # predecessor present + digest
    out += d(0x22)                        # checkpoint grant digest
    out += u16(2)                         # runtime snapshot schema
    out += d(0x22)                        # snapshot grant digest
    out += u32(5)                         # max uses
    out += risk(10, 2, 300, 4_000)       # account risk ceiling
    out += u32(2)                         # reservation count

    # Reservation 1 (BTreeMap key order).
    out += d(0x33)                        # map key
    out += d(0x33)                        # embedded reservation id
    out += d(0x44)                        # effect intent
    out += d(0x55)                        # attempt
    out += d(0x66)                        # effect binding
    out += risk(1, 0, 2, 3)
    out += b"\x00"                        # Reserved

    # Reservation 2.
    out += d(0x77)
    out += d(0x77)
    out += d(0x88)
    out += d(0x99)
    out += d(0xAA)
    out += risk(4, 1, 5, 6)
    out += b"\x01"                        # OutcomeUnknown
    return bytes(out)


def main() -> None:
    preimage = fixture_preimage()
    actual = hashlib.sha256(preimage).hexdigest()
    if len(preimage) != 565:
        raise SystemExit(f"unexpected canonical preimage length: {len(preimage)}")
    if actual != EXPECTED:
        raise SystemExit(f"checkpoint oracle mismatch: {actual} != {EXPECTED}")
    print(f"PASS checkpoint-v2 canonical vector {actual}")


if __name__ == "__main__":
    main()
