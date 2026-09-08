#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import os
import tempfile
import unittest
from pathlib import Path

import verify_workbench_nix_closure_capture as ver
import workbench_nix_closure_identity as closure

ROOT = "/nix/store/0123456789abcdfghijklmnpqrsvwxyz-workbench"
DEP = "/nix/store/123456789abcdfghijklmnpqrsvwxyz0-lib"
ORPHAN = "/nix/store/23456789abcdfghijklmnpqrsvwxyz01-orphan"
REV = "9ae611a455b90cf061d8f332b977e387bda8e1ca"
NAR = "sha256-md8WlXOlfnIeHeOScMTTHFyf2d6iaTwPl2apR5EQ3P4="


def sri(byte: int) -> str:
    return "sha256-" + base64.b64encode(bytes([byte]) * 32).decode("ascii")


def path_info(*, extra=False, orphan=False) -> bytes:
    rb = ROOT.rsplit("/", 1)[1]
    db = DEP.rsplit("/", 1)[1]
    info = {
        rb: {"narHash": sri(1), "references": [db]},
        db: {"narHash": sri(2), "references": []},
    }
    if extra:
        info[rb]["registrationTime"] = 123456
    if orphan:
        info[ORPHAN.rsplit("/", 1)[1]] = {"narHash": sri(3), "references": []}
    return json.dumps({"version": 2, "storeDir": "/nix/store", "info": info}, separators=(",", ":")).encode()


def fixture_sources(td: Path) -> tuple[Path, Path, Path, Path]:
    profile = {
        "schema": ver.PROFILE_SCHEMA,
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
    producer = td / "producer.py"
    normalizer = Path(closure.__file__).resolve()
    pp.write_text(json.dumps(profile), encoding="utf-8")
    lp.write_text(json.dumps(lock), encoding="utf-8")
    producer.write_text("# independent producer fixture\n", encoding="utf-8")
    return pp, lp, producer, normalizer


def write_stream(root: Path, rel: str, data: bytes) -> dict:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return {"path": rel, "byte_length": len(data), "sha256": ver.digest_bytes(data)}


def command(root: Path, name: str, argv: list[str], rc: int, stdout: bytes, stderr: bytes = b"") -> dict:
    out_rel, err_rel = ver.FIXED_STREAM_PATHS[name]
    return {
        "argv": argv,
        "exit_code": rc,
        "stdout": write_stream(root, out_rel, stdout),
        "stderr": write_stream(root, err_rel, stderr),
    }


def refresh_receipt(receipt: dict, root: Path) -> None:
    receipt["raw_observation_digest"] = ver.digest_bytes(ver.canonical_json_bytes(receipt["commands"]))
    receipt["capture_digest"] = ""
    receipt["capture_digest"] = ver.digest_bytes(
        ver.canonical_json_bytes({k: v for k, v in receipt.items() if k != "capture_digest"})
    )
    (root / "receipt.json").write_bytes(ver.canonical_json_bytes(receipt) + b"\n")


def success_receipt(td: Path, *, extra=False) -> tuple[Path, Path, Path, Path, Path, dict]:
    pp, lp, producer, normalizer = fixture_sources(td)
    receipt_dir = td / "receipt"
    receipt_dir.mkdir()
    selection = ver.derive_expected_selection(pp, lp)
    raw_path = path_info(extra=extra)
    commands = {
        "nix_version": command(receipt_dir, "nix_version", ver.NIX_VERSION_ARGV, 0, b"nix (Nix) 2.33.6\n"),
        "platform": command(receipt_dir, "platform", ver.PLATFORM_ARGV, 0, b"x86_64-linux"),
        "realization": command(receipt_dir, "realization", ver.REALIZE_PREFIX + [selection["installable"]], 0, (ROOT + "\n").encode()),
        "path_info": command(receipt_dir, "path_info", ver.PATH_INFO_PREFIX + [ROOT], 0, raw_path),
    }
    identity = closure.compile_identity(ROOT, ver.parse_path_info_v2(raw_path))
    identity_bytes = ver.canonical_json_bytes(identity) + b"\n"
    identity_rel = "normalized/closure_identity.json"
    p = receipt_dir / identity_rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(identity_bytes)
    receipt = {
        "schema": ver.SCHEMA,
        "status": ver.STATUS_SUCCESS,
        "selection": selection,
        "root": ROOT,
        "implementations": {"capture_sha256": ver.file_digest(producer), "normalizer_sha256": ver.file_digest(normalizer)},
        "commands": commands,
        "normalized": {
            "nix_version": "2.33.6",
            "platform": "x86_64-linux",
            "closure_identity_path": identity_rel,
            "closure_identity_sha256": ver.digest_bytes(identity_bytes),
            "closure_digest": identity["closure_digest"],
        },
        "facts": {
            "selection_revalidated": True,
            "nix_version_observed": True,
            "platform_observed": True,
            "realization_command_observed": True,
            "realized_root_observed": True,
            "path_info_observed": True,
            "canonical_closure_identity_compiled": True,
        },
        "authority": {key: False for key in ver.AUTHORITY_KEYS},
        "raw_observation_digest": "",
        "capture_digest": "",
    }
    refresh_receipt(receipt, receipt_dir)
    return receipt_dir, pp, lp, producer, normalizer, receipt


def failure_receipt(td: Path) -> tuple[Path, Path, Path, Path, Path, dict]:
    pp, lp, producer, normalizer = fixture_sources(td)
    receipt_dir = td / "receipt"
    receipt_dir.mkdir()
    (receipt_dir / "normalized").mkdir()
    selection = ver.derive_expected_selection(pp, lp)
    commands = {
        "nix_version": command(receipt_dir, "nix_version", ver.NIX_VERSION_ARGV, 0, b"nix (Nix) 2.33.6\n"),
        "platform": command(receipt_dir, "platform", ver.PLATFORM_ARGV, 0, b"x86_64-linux"),
        "realization": command(receipt_dir, "realization", ver.REALIZE_PREFIX + [selection["installable"]], 1, b"", b"build failed\n"),
        "path_info": None,
    }
    receipt = {
        "schema": ver.SCHEMA,
        "status": ver.STATUS_FAILURE,
        "selection": selection,
        "root": None,
        "implementations": {"capture_sha256": ver.file_digest(producer), "normalizer_sha256": ver.file_digest(normalizer)},
        "commands": commands,
        "normalized": {"nix_version": None, "platform": None, "closure_identity_path": None, "closure_identity_sha256": None, "closure_digest": None},
        "facts": {
            "selection_revalidated": True,
            "nix_version_observed": True,
            "platform_observed": True,
            "realization_command_observed": True,
            "realized_root_observed": False,
            "path_info_observed": False,
            "canonical_closure_identity_compiled": False,
        },
        "authority": {key: False for key in ver.AUTHORITY_KEYS},
        "raw_observation_digest": "",
        "capture_digest": "",
    }
    refresh_receipt(receipt, receipt_dir)
    return receipt_dir, pp, lp, producer, normalizer, receipt


def verify(args):
    root, pp, lp, producer, normalizer, _ = args
    return ver.verify_receipt(root, pp, lp, producer, normalizer)


class HostileReceiptVerifierContracts(unittest.TestCase):
    def test_valid_complete_receipt_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = verify(success_receipt(Path(tmp)))
            self.assertEqual(result["status"], "verified-complete-observation")
            self.assertTrue(result["authority"]["capture_receipt_verified"])
            self.assertFalse(result["authority"]["workbench_execution_qualified"])

    def test_valid_failure_receipt_verifies_as_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = verify(failure_receipt(Path(tmp)))
            self.assertEqual(result["status"], "verified-incomplete-observation")
            self.assertIsNone(result["closure_digest"])

    def test_raw_sidecar_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = success_receipt(Path(tmp)); (args[0] / "raw/path-info.stdout").write_bytes(b"tampered")
            with self.assertRaises(ver.VerificationError): verify(args)

    def test_self_rehashed_argv_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = success_receipt(Path(tmp)); r=args[-1]
            r["commands"]["realization"]["argv"][-1] += "-evil"
            refresh_receipt(r, args[0])
            with self.assertRaises(ver.VerificationError): verify(args)

    def test_sidecar_path_traversal_rejected_even_when_rehashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = success_receipt(Path(tmp)); r=args[-1]
            r["commands"]["nix_version"]["stdout"]["path"] = "../outside"
            refresh_receipt(r, args[0])
            with self.assertRaises(ver.VerificationError): verify(args)

    def test_sidecar_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            td=Path(tmp); args=success_receipt(td)
            target = td / "outside"; target.write_bytes(b"nix (Nix) 2.33.6\n")
            p=args[0] / "raw/nix-version.stdout"; p.unlink(); os.symlink(target, p)
            with self.assertRaises(ver.VerificationError): verify(args)

    def test_boolean_exit_code_laundering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=success_receipt(Path(tmp)); r=args[-1]; r["commands"]["platform"]["exit_code"] = False
            refresh_receipt(r,args[0])
            with self.assertRaises(ver.VerificationError): verify(args)

    def test_boolean_byte_length_laundering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=success_receipt(Path(tmp)); r=args[-1]; r["commands"]["platform"]["stdout"]["byte_length"] = True
            refresh_receipt(r,args[0])
            with self.assertRaises(ver.VerificationError): verify(args)

    def test_authority_escalation_rejected_after_rehash(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=success_receipt(Path(tmp)); r=args[-1]; r["authority"]["fmq010_established"] = True
            refresh_receipt(r,args[0])
            with self.assertRaises(ver.VerificationError): verify(args)

    def test_fact_tamper_rejected_after_rehash(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=success_receipt(Path(tmp)); r=args[-1]; r["facts"]["realized_root_observed"] = False
            refresh_receipt(r,args[0])
            with self.assertRaises(ver.VerificationError): verify(args)

    def test_profile_digest_substitution_rejected_after_rehash(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=success_receipt(Path(tmp)); r=args[-1]; r["selection"]["profile_sha256"] = "sha256:" + "0"*64
            refresh_receipt(r,args[0])
            with self.assertRaises(ver.VerificationError): verify(args)

    def test_realized_root_substitution_rejected_after_rehash(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=success_receipt(Path(tmp)); r=args[-1]; r["root"] = DEP
            refresh_receipt(r,args[0])
            with self.assertRaises(ver.VerificationError): verify(args)

    def test_normalized_identity_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=success_receipt(Path(tmp)); p=args[0]/"normalized/closure_identity.json"
            value=json.loads(p.read_text()); value["entry_count"] += 1; p.write_text(json.dumps(value))
            with self.assertRaises(ver.VerificationError): verify(args)

    def test_orphan_raw_closure_injection_rejected_even_when_rehashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=success_receipt(Path(tmp)); r=args[-1]; raw=path_info(orphan=True)
            p=args[0]/"raw/path-info.stdout"; p.write_bytes(raw)
            r["commands"]["path_info"]["stdout"]={"path":"raw/path-info.stdout","byte_length":len(raw),"sha256":ver.digest_bytes(raw)}
            refresh_receipt(r,args[0])
            with self.assertRaises((ver.VerificationError, closure.ContractError)): verify(args)

    def test_unconsumed_raw_metadata_can_change_without_changing_closure(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=success_receipt(Path(tmp), extra=True)
            result=verify(args)
            self.assertEqual(result["status"], "verified-complete-observation")

    def test_capture_digest_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=success_receipt(Path(tmp)); r=args[-1]; r["capture_digest"]="sha256:"+"f"*64
            (args[0]/"receipt.json").write_bytes(ver.canonical_json_bytes(r)+b"\n")
            with self.assertRaises(ver.VerificationError): verify(args)

    def test_unknown_receipt_field_rejected_even_when_rehashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=success_receipt(Path(tmp)); r=args[-1]; r["qualified"] = True
            refresh_receipt(r,args[0])
            with self.assertRaises(ver.VerificationError): verify(args)

    def test_unknown_command_field_rejected_even_when_rehashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=success_receipt(Path(tmp)); r=args[-1]; r["commands"]["platform"]["trusted"] = True
            refresh_receipt(r,args[0])
            with self.assertRaises(ver.VerificationError): verify(args)

    def test_duplicate_receipt_key_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=success_receipt(Path(tmp)); p=args[0]/"receipt.json"; raw=p.read_text()
            raw=raw.replace('"schema":', '"schema":"forged","schema":', 1); p.write_text(raw)
            with self.assertRaises(ver.VerificationError): verify(args)

    def test_failure_receipt_cannot_launder_normalized_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=failure_receipt(Path(tmp)); r=args[-1]; r["normalized"]["platform"]="x86_64-linux"
            refresh_receipt(r,args[0])
            with self.assertRaises(ver.VerificationError): verify(args)

    def test_extra_unbound_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=success_receipt(Path(tmp)); (args[0]/"raw/unbound.txt").write_text("x")
            with self.assertRaises(ver.VerificationError): verify(args)

    def test_noncanonical_receipt_serialization_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=success_receipt(Path(tmp)); p=args[0]/"receipt.json"
            value=json.loads(p.read_text()); p.write_text(json.dumps(value, indent=2) + "\n")
            with self.assertRaises(ver.VerificationError): verify(args)

    def test_normalizer_path_substitution_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=list(success_receipt(Path(tmp))); fake=Path(tmp)/"other-normalizer.py"
            fake.write_bytes(Path(closure.__file__).read_bytes()); args[4]=fake
            with self.assertRaises(ver.VerificationError): verify(tuple(args))


if __name__ == "__main__":
    unittest.main(verbosity=2)
