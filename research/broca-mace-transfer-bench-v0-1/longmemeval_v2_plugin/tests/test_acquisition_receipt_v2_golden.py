"""Independent Python implementation of the Rust V2 receipt golden vector."""
import hashlib
import struct


def _field(hasher, value):
    encoded = value.encode("utf-8")
    hasher.update(struct.pack(">Q", len(encoded)))
    hasher.update(encoded)


def test_rust_python_acquisition_v2_digest_golden_vector():
    h = hashlib.sha256()
    h.update(b"luminous:mace:retrieval-acquisition:v2\0")
    for value in ("r", "i", "d", "a" * 64, "a" * 64, "a" * 64):
        _field(h, value)
    h.update(b"\x01")
    _field(h, "b" * 64)
    h.update(b"\x01")  # candidates_returned
    h.update(struct.pack(">I", 1))  # tool_call_budget
    h.update(struct.pack(">I", 1))  # tool_calls_used
    h.update(struct.pack(">Q", 7))  # elapsed_ns
    h.update(struct.pack(">Q", 9))  # inspected_bytes
    h.update(b"\x01")  # inspected_bytes_complete
    h.update(struct.pack(">Q", 10))  # created_at_unix
    h.update(struct.pack(">Q", 1))  # operation count
    _field(h, "search")
    _field(h, "a" * 64)
    _field(h, "a" * 64)
    h.update(b"\x01")  # evidence_observed
    h.update(struct.pack(">Q", 7))
    h.update(struct.pack(">Q", 9))
    h.update(b"\x01")  # inspected_bytes_measured
    h.update(struct.pack(">Q", 1))  # evidence ID count
    _field(h, "e1")

    assert h.hexdigest() == "81d1dc85c850db8b4cbce896459e00e95b9d91b7bda8659b05f2d59252805f56"
