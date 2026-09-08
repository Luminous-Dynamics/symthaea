#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS))

import workbench_nix_closure_identity as closure

ROOT = "/nix/store/0123456789abcdfghijklmnpqrsvwxyz-workbench"
LIB = "/nix/store/123456789abcdfghijklmnpqrsvwxyz0-lib"
DEP = "/nix/store/23456789abcdfghijklmnpqrsvwxyz01-dep"
OTHER = "/nix/store/3456789abcdfghijklmnpqrsvwxyz012-other"


def h(char: str) -> str:
    return "sha256:" + char * 64


def entries() -> list[dict]:
    return [
        {"path": ROOT, "nar_sha256": h("a"), "references": [DEP, LIB]},
        {"path": LIB, "nar_sha256": h("b"), "references": [DEP]},
        {"path": DEP, "nar_sha256": h("c"), "references": []},
    ]


class ClosureIdentityTests(unittest.TestCase):
    def test_valid_closed_graph_compiles_and_validates(self):
        identity = closure.compile_identity(ROOT, entries())
        self.assertEqual(identity["schema"], closure.SCHEMA)
        self.assertEqual(identity["root"], ROOT)
        self.assertEqual(identity["entry_count"], 3)
        self.assertEqual(identity, closure.validate_identity(identity))

    def test_input_and_reference_order_do_not_change_identity(self):
        a = closure.compile_identity(ROOT, entries())
        shuffled = list(reversed(entries()))
        shuffled[-1]["references"] = list(reversed(shuffled[-1]["references"]))
        b = closure.compile_identity(ROOT, shuffled)
        self.assertEqual(a, b)

    def test_root_must_be_present(self):
        with self.assertRaises(closure.ContractError):
            closure.compile_identity(OTHER, entries())

    def test_unreachable_orphan_entry_rejected(self):
        value = entries()
        value.append({"path": OTHER, "nar_sha256": h("d"), "references": []})
        with self.assertRaises(closure.ContractError):
            closure.compile_identity(ROOT, value)

    def test_duplicate_store_path_rejected(self):
        value = entries()
        value.append(copy.deepcopy(value[0]))
        with self.assertRaises(closure.ContractError):
            closure.compile_identity(ROOT, value)

    def test_duplicate_reference_rejected(self):
        value = entries()
        value[0]["references"].append(LIB)
        with self.assertRaises(closure.ContractError):
            closure.compile_identity(ROOT, value)

    def test_reference_outside_closure_rejected(self):
        value = entries()
        value[1]["references"].append(OTHER)
        with self.assertRaises(closure.ContractError):
            closure.compile_identity(ROOT, value)

    def test_self_reference_is_supported(self):
        value = [{"path": ROOT, "nar_sha256": h("a"), "references": [ROOT]}]
        identity = closure.compile_identity(ROOT, value)
        self.assertEqual(identity["entry_count"], 1)
        self.assertEqual(identity, closure.validate_identity(identity))

    def test_invalid_store_path_rejected(self):
        value = entries()
        value[0]["path"] = "/tmp/not-nix-store"
        with self.assertRaises(closure.ContractError):
            closure.compile_identity(ROOT, value)

    def test_store_path_with_control_character_rejected(self):
        value = entries()
        value[0]["path"] = ROOT + "\nforged"
        with self.assertRaises(closure.ContractError):
            closure.compile_identity(value[0]["path"], value)

    def test_invalid_nar_hash_rejected(self):
        value = entries()
        value[0]["nar_sha256"] = "sha256-NOT-CANONICAL"
        with self.assertRaises(closure.ContractError):
            closure.compile_identity(ROOT, value)

    def test_unknown_entry_field_rejected(self):
        value = entries()
        value[0]["qualified"] = True
        with self.assertRaises(closure.ContractError):
            closure.compile_identity(ROOT, value)

    def test_nar_content_change_changes_closure_digest(self):
        a = closure.compile_identity(ROOT, entries())
        value = entries()
        value[2]["nar_sha256"] = h("d")
        b = closure.compile_identity(ROOT, value)
        self.assertNotEqual(a["closure_digest"], b["closure_digest"])

    def test_reference_topology_change_changes_closure_digest(self):
        a = closure.compile_identity(ROOT, entries())
        value = entries()
        value[0]["references"] = [LIB]
        b = closure.compile_identity(ROOT, value)
        self.assertNotEqual(a["closure_digest"], b["closure_digest"])

    def test_validation_rejects_noncanonical_entry_order(self):
        identity = closure.compile_identity(ROOT, entries())
        identity["entries"] = list(reversed(identity["entries"]))
        with self.assertRaises(closure.ContractError):
            closure.validate_identity(identity)

    def test_validation_rejects_count_tamper(self):
        identity = closure.compile_identity(ROOT, entries())
        identity["entry_count"] += 1
        with self.assertRaises(closure.ContractError):
            closure.validate_identity(identity)

    def test_validation_rejects_boolean_count_for_singleton(self):
        identity = closure.compile_identity(
            ROOT,
            [{"path": ROOT, "nar_sha256": h("a"), "references": []}],
        )
        identity["entry_count"] = True
        with self.assertRaises(closure.ContractError):
            closure.validate_identity(identity)

    def test_validation_rejects_digest_tamper(self):
        identity = closure.compile_identity(ROOT, entries())
        identity["closure_digest"] = h("f")
        with self.assertRaises(closure.ContractError):
            closure.validate_identity(identity)

    def test_validation_rejects_unknown_identity_field(self):
        identity = closure.compile_identity(ROOT, entries())
        identity["closure_qualified"] = True
        with self.assertRaises(closure.ContractError):
            closure.validate_identity(identity)

    def test_validation_rejects_self_rehashed_noncanonical_payload(self):
        identity = closure.compile_identity(ROOT, entries())
        identity["entries"][0]["references"] = list(reversed(identity["entries"][0]["references"]))
        identity["closure_digest"] = closure._digest_payload(identity["root"], identity["entries"])
        with self.assertRaises(closure.ContractError):
            closure.validate_identity(identity)

    def test_duplicate_json_object_key_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "duplicate.json"
            path.write_text('{"path":"a","path":"b"}', encoding="utf-8")
            with self.assertRaises(closure.ContractError):
                closure.load(path)

    def test_cli_compile_and_verify_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            td = Path(temp)
            ep = td / "entries.json"
            ip = td / "identity.json"
            ep.write_text(json.dumps(entries()), encoding="utf-8")
            compiled = closure.compile_identity(ROOT, closure.load(ep))
            ip.write_text(json.dumps(compiled), encoding="utf-8")
            self.assertEqual(closure.main(["verify", "--identity", str(ip)]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
