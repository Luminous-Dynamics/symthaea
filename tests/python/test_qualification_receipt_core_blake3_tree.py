import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "qualification_receipt_core_v1.py"
spec = importlib.util.spec_from_file_location("qualification_receipt_core_v1_tree", SCRIPT)
assert spec and spec.loader
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def test_reference_blake3_matches_official_4096_byte_tree_vector():
    # Official BLAKE3 test-vector input pattern: 0,1,...,250 repeating.
    data = bytes(index % 251 for index in range(4096))
    assert core.blake3_256(data).hex() == (
        "015094013f57a5277b59d8475c0501042c0b642e531b0a1c8f58d2163229e969"
    )
