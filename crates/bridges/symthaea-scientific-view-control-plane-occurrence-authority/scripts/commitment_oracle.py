#!/usr/bin/env python3
"""Independent SCI-014R2B store-authority identity oracle."""
import hashlib, struct

EXPECTED = {
    "authority_binding": "eba4dfa6626431008d986b01f026a5fdbabd034ed94baf8d7fc1c341df9e42e8",
    "qualified_commit": "686ef4d30dd20dcf19ed530a412542f094651d02d797e43024e7bcba0ac26d65",
    "historical_occurrence": "b2a1a5fa14083c083ba4e020f7410da140ece86eb4bdb471e2bdc97b4fc4c336",
}
STORE_BINDING = bytes.fromhex("cd4bf72e0bd17ea93149cb1feb270d58d0ef1471cade4d2733b7a6e390afcc05")
OCCURRENCE = bytes.fromhex("f6bfd1cafa34070c9d7634ce5482b95380b3b6a608a22da9966e34222a2ff3b0")

def u64(v): return struct.pack(">Q", v)
def raw(v): return u64(len(v)) + v
def text(v): return raw(v.encode("ascii"))
def c(v): return bytes((v,)) * 32
def digest(domain, *fields):
    h = hashlib.sha256(); h.update(raw(domain.encode("ascii")))
    for field in fields: h.update(field)
    return h.digest()

def build():
    binding = digest(
        "symthaea.science.view.control-plane.occurrence-store-authority-binding.v1",
        text("deployment/site01-a"), text("lunar/site01"), STORE_BINDING, u64(7),
        c(81), c(82), c(83), c(84),
    )
    qualified = digest(
        "symthaea.science.view.control-plane.store-authority-qualified-commit.v1",
        binding, OCCURRENCE, text("store-ref-1"),
    )
    historical = digest(
        "symthaea.science.view.control-plane.historical-committed-occurrence.v1",
        qualified, binding, OCCURRENCE,
    )
    return {
        "authority_binding": binding.hex(),
        "qualified_commit": qualified.hex(),
        "historical_occurrence": historical.hex(),
    }

def main():
    actual = build()
    if actual != EXPECTED:
        for k, v in EXPECTED.items():
            if actual.get(k) != v:
                print("FAIL", k, v, actual.get(k)); raise SystemExit(1)
    for k, v in actual.items(): print(k, v)
    print("SCI-014R2B store-authority vectors: PASS")

if __name__ == "__main__": main()
