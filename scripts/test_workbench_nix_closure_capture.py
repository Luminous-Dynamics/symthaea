#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import workbench_nix_closure_capture as cap
import workbench_nix_closure_identity as closure

ROOT = "/nix/store/0123456789abcdfghijklmnpqrsvwxyz-workbench"
DEP = "/nix/store/123456789abcdfghijklmnpqrsvwxyz0-lib"
REV = "9ae611a455b90cf061d8f332b977e387bda8e1ca"
NAR = "sha256-md8WlXOlfnIeHeOScMTTHFyf2d6iaTwPl2apR5EQ3P4="


def sri(byte: int) -> str:
    return "sha256-" + base64.b64encode(bytes([byte]) * 32).decode("ascii")


def path_info(extra: bool = False) -> bytes:
    rb = ROOT.rsplit("/", 1)[1]
    db = DEP.rsplit("/", 1)[1]
    info = {
        rb: {"narHash": sri(1), "references": [db]},
        db: {"narHash": sri(2), "references": []},
    }
    if extra:
        info[rb]["registrationTime"] = 123456
    return json.dumps({"version": 2, "storeDir": "/nix/store", "info": info}, separators=(",", ":")).encode()


def proc(argv: list[str], rc: int, out: bytes, err: bytes = b"") -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(argv, rc, out, err)


def fixture_files(td: Path) -> tuple[Path, Path]:
    profile = {
        "schema": cap.PROFILE_SCHEMA,
        "status": "candidate_profile_only",
        "qualification_platform": "x86_64-linux",
        "flake_selection": {"root_input": "nixpkgs", "locked_node": "nixpkgs_2", "rev": REV, "nar_hash": NAR},
        "nixpkgs_package": {"attribute": "connectome-workbench"},
        "execution_environment": {},
        "future_closure_receipt": {},
        "authority": {},
    }
    lock = {
        "nodes": {
            "root": {"inputs": {"nixpkgs": "nixpkgs_2"}},
            "nixpkgs_2": {"locked": {"owner": "NixOS", "repo": "nixpkgs", "rev": REV, "narHash": NAR, "type": "github"}},
        },
        "root": "root",
        "version": 7,
    }
    pp = td / "profile.json"
    lp = td / "flake.lock"
    pp.write_text(json.dumps(profile), encoding="utf-8")
    lp.write_text(json.dumps(lock), encoding="utf-8")
    return pp, lp


class CaptureContracts(unittest.TestCase):
    def run_capture(self, td: Path, *, platform=b"x86_64-linux", build_rc=0, build_out: bytes | None = None, info: bytes | None = None):
        pp, lp = fixture_files(td)
        sel = cap.derive_selection(pp, lp)
        realize_argv = cap.REALIZE_PREFIX + [sel["installable"]]
        outputs = [
            proc(cap.NIX_VERSION_ARGV, 0, b"nix (Nix) 2.33.6\n"),
            proc(cap.PLATFORM_ARGV, 0, platform),
            proc(realize_argv, build_rc, build_out if build_out is not None else (ROOT + "\n").encode()),
        ]
        if build_rc == 0 and (build_out is None or build_out.decode().splitlines() == [ROOT]):
            outputs.append(proc(cap.PATH_INFO_PREFIX + [ROOT], 0, info or path_info()))
        with mock.patch.object(cap, "run_command", side_effect=outputs):
            rc = cap.capture(pp, lp, td / "receipt")
        return rc, td / "receipt", pp, lp

    def test_selection_derives_exact_locked_installable(self):
        with tempfile.TemporaryDirectory() as tmp:
            pp, lp = fixture_files(Path(tmp))
            s = cap.derive_selection(pp, lp)
            self.assertEqual(s["installable"], f"github:NixOS/nixpkgs/{REV}#connectome-workbench")
            self.assertEqual(s["qualification_platform"], "x86_64-linux")

    def test_lock_substitution_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp); pp, lp = fixture_files(td)
            lock = json.loads(lp.read_text()); lock["nodes"]["nixpkgs_2"]["locked"]["rev"] = "0" * 40; lp.write_text(json.dumps(lock))
            with self.assertRaises(closure.ContractError): cap.derive_selection(pp, lp)

    def test_successful_capture_is_observed_but_unqualified(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, out, _, _ = self.run_capture(Path(tmp)); self.assertEqual(rc, 0)
            r = json.loads((out / "receipt.json").read_text())
            self.assertEqual(r["status"], cap.STATUS_SUCCESS)
            self.assertTrue(r["facts"]["canonical_closure_identity_compiled"])
            self.assertFalse(r["authority"]["closure_capture_qualified"])
            self.assertFalse(r["authority"]["fmq010_established"])
            self.assertTrue((out / "normalized/closure_identity.json").is_file())

    def test_root_is_derived_from_realization_not_supplied(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out, _, _ = self.run_capture(Path(tmp)); r = json.loads((out / "receipt.json").read_text())
            self.assertEqual(r["root"], ROOT)
            self.assertEqual(r["commands"]["realization"]["argv"][-1], r["selection"]["installable"])
            self.assertEqual(r["commands"]["path_info"]["argv"][-1], ROOT)

    def test_multi_root_realization_output_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, out, _, _ = self.run_capture(Path(tmp), build_out=(ROOT + "\n" + DEP + "\n").encode()); self.assertEqual(rc, 2)
            r = json.loads((out / "receipt.json").read_text())
            self.assertIsNone(r["root"]); self.assertIsNone(r["commands"]["path_info"])

    def test_realization_failure_is_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, out, _, _ = self.run_capture(Path(tmp), build_rc=1, build_out=b""); self.assertEqual(rc, 2)
            r = json.loads((out / "receipt.json").read_text())
            self.assertEqual(r["status"], cap.STATUS_FAILURE)
            self.assertEqual(r["commands"]["realization"]["exit_code"], 1)
            self.assertFalse(r["facts"]["realized_root_observed"])

    def test_platform_drift_fails_without_laundering_closure(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, out, _, _ = self.run_capture(Path(tmp), platform=b"aarch64-linux"); self.assertEqual(rc, 2)
            r = json.loads((out / "receipt.json").read_text())
            self.assertFalse(r["facts"]["canonical_closure_identity_compiled"])
            self.assertIsNone(r["normalized"]["closure_digest"])

    def test_path_info_v2_and_sri_normalization(self):
        entries = cap.parse_path_info_v2(path_info())
        identity = closure.compile_identity(ROOT, entries)
        self.assertEqual(identity["entry_count"], 2)
        self.assertEqual(entries[0]["nar_sha256"].split(":")[0], "sha256")

    def test_path_info_duplicate_key_rejected(self):
        raw = b'{"version":2,"version":2,"storeDir":"/nix/store","info":{}}'
        with self.assertRaises(closure.ContractError): cap.parse_path_info_v2(raw)

    def test_path_info_unknown_top_level_field_rejected(self):
        value = json.loads(path_info()); value["future"] = 1
        with self.assertRaises(closure.ContractError): cap.parse_path_info_v2(json.dumps(value).encode())

    def test_orphan_closure_entry_rejected(self):
        value = json.loads(path_info())
        value["info"]["23456789abcdfghijklmnpqrsvwxyz01-orphan"] = {"narHash": sri(3), "references": []}
        with self.assertRaises(closure.ContractError): closure.compile_identity(ROOT, cap.parse_path_info_v2(json.dumps(value).encode()))

    def test_unconsumed_nix_metadata_is_preserved_raw_but_not_scientific_identity(self):
        a, b = path_info(False), path_info(True)
        self.assertEqual(closure.compile_identity(ROOT, cap.parse_path_info_v2(a)), closure.compile_identity(ROOT, cap.parse_path_info_v2(b)))
        self.assertNotEqual(cap.digest_bytes(a), cap.digest_bytes(b))

    def test_noncanonical_sri_rejected(self):
        with self.assertRaises(closure.ContractError): cap.sri_sha256_to_hex("sha256-" + "A" * 43)

    def test_nix_version_shape_rejected(self):
        with self.assertRaises(closure.ContractError): cap.parse_nix_version(b"Nix 2.33.6\n")

    def test_output_is_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp); _, out, pp, lp = self.run_capture(td)
            with self.assertRaises(closure.ContractError): cap.capture(pp, lp, out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
