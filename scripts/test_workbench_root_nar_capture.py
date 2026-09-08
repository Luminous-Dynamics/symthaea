#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import workbench_root_nar_capture as producer
import verify_workbench_root_nar_capture as verifier

ROOT = "/nix/store/0123456789abcdfghijklmnpqrsvwxyz-workbench"
CLOSURE_DIGEST = "sha256:" + "a" * 64
CLOSURE_CAPTURE_DIGEST = "sha256:" + "b" * 64
ROOT_NAR_HASH = "sha256:" + "c" * 64


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def write_closure_receipt(root: Path, *, status: str = "observed-normalized-unqualified") -> Path:
    root.mkdir()
    receipt = {"status": status, "root": ROOT, "capture_digest": CLOSURE_CAPTURE_DIGEST}
    (root / "receipt.json").write_bytes(canonical(receipt))
    return root


def fake_runner(payload: bytes, rc: int = 0, stderr: bytes = b""):
    def run(argv: list[str], stdout_path: Path, stderr_path: Path) -> int:
        stdout_path.write_bytes(payload)
        stderr_path.write_bytes(stderr)
        return rc
    return run


def load_receipt(path: Path) -> dict:
    return json.loads((path / "receipt.json").read_text())


def rewrite_receipt(path: Path, mutate) -> dict:
    receipt = load_receipt(path)
    mutate(receipt)
    receipt["capture_digest"] = verifier.digest_bytes(
        verifier.canonical_json_bytes({k: v for k, v in receipt.items() if k != "capture_digest"})
    )
    (path / "receipt.json").write_bytes(verifier.canonical_json_bytes(receipt) + b"\n")
    return receipt


class CaptureTests(unittest.TestCase):
    def test_successful_capture_is_unqualified_and_create_only(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            closure = write_closure_receipt(base / "closure")
            out = base / "capture"
            self.assertEqual(producer.capture(closure, out, runner=fake_runner(b"nar-bytes")), 0)
            receipt = load_receipt(out)
            self.assertEqual(receipt["status"], producer.STATUS_SUCCESS)
            self.assertEqual(receipt["root"], ROOT)
            self.assertEqual(receipt["command"]["argv"], ["nix-store", "--dump", ROOT])
            self.assertTrue(all(value is False for value in receipt["authority"].values()))
            self.assertEqual((out / "raw/root.nar").read_bytes(), b"nar-bytes")
            with self.assertRaises(producer.CaptureError):
                producer.capture(closure, out, runner=fake_runner(b"other"))

    def test_failed_dump_is_retained_as_incomplete_observation(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            closure = write_closure_receipt(base / "closure")
            out = base / "capture"
            self.assertEqual(producer.capture(closure, out, runner=fake_runner(b"partial", 7, b"boom")), 2)
            receipt = load_receipt(out)
            self.assertEqual(receipt["status"], producer.STATUS_FAILURE)
            self.assertEqual(receipt["command"]["exit_code"], 7)
            self.assertEqual((out / "raw/root.nar").read_bytes(), b"partial")
            self.assertEqual((out / "raw/nix-store-dump.stderr").read_bytes(), b"boom")

    def test_incomplete_closure_receipt_cannot_select_dump_root(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            closure = write_closure_receipt(base / "closure", status="observation-incomplete-unqualified")
            with self.assertRaises(producer.CaptureError):
                producer.capture(closure, base / "capture", runner=fake_runner(b"x"))

    def test_noncanonical_root_rejected_before_runner(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            closure = write_closure_receipt(base / "closure")
            receipt = json.loads((closure / "receipt.json").read_text())
            receipt["root"] = "/tmp/workbench"
            (closure / "receipt.json").write_bytes(canonical(receipt))
            with self.assertRaises(producer.CaptureError):
                producer.capture(closure, base / "capture", runner=fake_runner(b"x"))


class DumpVerifierTests(unittest.TestCase):
    def make_capture(self, base: Path, payload: bytes = b"nar") -> Path:
        closure = write_closure_receipt(base / "closure")
        out = base / "capture"
        producer.capture(closure, out, runner=fake_runner(payload))
        return out

    def verify(self, out: Path, *, success: bool = True):
        return verifier.verify_dump_receipt(
            out,
            expected_root=ROOT,
            expected_closure_capture_digest=CLOSURE_CAPTURE_DIGEST,
            producer_path=Path(producer.__file__),
            require_success=success,
        )

    def test_valid_receipt_reconstructs_exact_subject(self):
        with tempfile.TemporaryDirectory() as td:
            out = self.make_capture(Path(td), b"abc")
            receipt, nar = self.verify(out)
            self.assertEqual(receipt["root"], ROOT)
            self.assertEqual(nar.read_bytes(), b"abc")

    def test_root_substitution_rejected_even_when_self_rehashed(self):
        with tempfile.TemporaryDirectory() as td:
            out = self.make_capture(Path(td))
            substitute = ROOT.replace("workbench", "other")
            rewrite_receipt(out, lambda r: (
                r.__setitem__("root", substitute),
                r["command"].__setitem__("argv", ["nix-store", "--dump", substitute]),
            ))
            with self.assertRaisesRegex(verifier.VerificationError, "root differs"):
                self.verify(out)

    def test_argv_mutation_rejected_when_self_rehashed(self):
        with tempfile.TemporaryDirectory() as td:
            out = self.make_capture(Path(td))
            rewrite_receipt(out, lambda r: r["command"].__setitem__("argv", ["nix-store", "--dump", "--", ROOT]))
            with self.assertRaisesRegex(verifier.VerificationError, "exact argv"):
                self.verify(out)

    def test_nar_sidecar_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            out = self.make_capture(Path(td), b"abc")
            (out / "raw/root.nar").write_bytes(b"abd")
            with self.assertRaisesRegex(verifier.VerificationError, "retained bytes"):
                self.verify(out)

    def test_stderr_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            out = self.make_capture(Path(td))
            (out / "raw/nix-store-dump.stderr").write_bytes(b"tamper")
            with self.assertRaisesRegex(verifier.VerificationError, "retained bytes"):
                self.verify(out)

    def test_authority_escalation_rejected_when_self_rehashed(self):
        with tempfile.TemporaryDirectory() as td:
            out = self.make_capture(Path(td))
            rewrite_receipt(out, lambda r: r["authority"].__setitem__("workbench_execution_qualified", True))
            with self.assertRaisesRegex(verifier.VerificationError, "escalation forbidden"):
                self.verify(out)

    def test_unknown_receipt_field_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            out = self.make_capture(Path(td))
            rewrite_receipt(out, lambda r: r.__setitem__("qualified", True))
            with self.assertRaisesRegex(verifier.VerificationError, "closed-world"):
                self.verify(out)

    def test_bool_exit_code_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            out = self.make_capture(Path(td))
            rewrite_receipt(out, lambda r: r["command"].__setitem__("exit_code", False))
            with self.assertRaisesRegex(verifier.VerificationError, "integer required"):
                self.verify(out)

    def test_extra_file_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            out = self.make_capture(Path(td))
            (out / "unbound.txt").write_text("x")
            with self.assertRaisesRegex(verifier.VerificationError, "inventory mismatch"):
                self.verify(out)

    def test_noncanonical_receipt_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            out = self.make_capture(Path(td))
            receipt = load_receipt(out)
            (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            with self.assertRaisesRegex(verifier.VerificationError, "canonical JSON"):
                self.verify(out)

    def test_failed_observation_can_be_verified_as_failed_but_not_promoted(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            closure = write_closure_receipt(base / "closure")
            out = base / "capture"
            producer.capture(closure, out, runner=fake_runner(b"partial", 5))
            receipt, _ = verifier.verify_dump_receipt(
                out,
                expected_root=ROOT,
                expected_closure_capture_digest=CLOSURE_CAPTURE_DIGEST,
                producer_path=Path(producer.__file__),
                require_success=False,
            )
            self.assertEqual(receipt["status"], producer.STATUS_FAILURE)
            with self.assertRaisesRegex(verifier.VerificationError, "successful dump required"):
                self.verify(out)

    def test_producer_implementation_substitution_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            out = self.make_capture(base)
            fake_source = base / "fake.py"
            fake_source.write_text("# different\n")
            with self.assertRaisesRegex(verifier.VerificationError, "producer source digest"):
                verifier.verify_dump_receipt(
                    out,
                    expected_root=ROOT,
                    expected_closure_capture_digest=CLOSURE_CAPTURE_DIGEST,
                    producer_path=fake_source,
                )

    def test_duplicate_json_key_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            out = self.make_capture(Path(td))
            data = (out / "receipt.json").read_text()
            data = data.replace('{"authority":', '{"schema":"duplicate","authority":', 1)
            (out / "receipt.json").write_text(data)
            with self.assertRaises(verifier.VerificationError):
                self.verify(out)


class ProjectionTests(unittest.TestCase):
    def test_root_projection_uses_exact_root_entry(self):
        identity = {
            "root": ROOT,
            "closure_digest": CLOSURE_DIGEST,
            "entries": [{"path": ROOT, "nar_sha256": ROOT_NAR_HASH, "references": []}],
        }
        self.assertEqual(verifier.project_verified_root(identity, CLOSURE_DIGEST), (ROOT, ROOT_NAR_HASH))

    def test_root_projection_rejects_digest_substitution(self):
        identity = {
            "root": ROOT,
            "closure_digest": CLOSURE_DIGEST,
            "entries": [{"path": ROOT, "nar_sha256": ROOT_NAR_HASH}],
        }
        with self.assertRaisesRegex(verifier.VerificationError, "closure digest mismatch"):
            verifier.project_verified_root(identity, "sha256:" + "d" * 64)

    def test_root_projection_rejects_missing_root_entry(self):
        identity = {"root": ROOT, "closure_digest": CLOSURE_DIGEST, "entries": []}
        with self.assertRaisesRegex(verifier.VerificationError, "exact root entry"):
            verifier.project_verified_root(identity, CLOSURE_DIGEST)


class PipelineTopologyTests(unittest.TestCase):
    def install_fake_modules(self, base: Path, *, target: dict) -> tuple[types.ModuleType, types.ModuleType, Path, Path]:
        closure_source = base / "fake_closure_verifier.py"
        membership_source = base / "fake_membership.py"
        closure_source.write_text("# closure verifier\n")
        membership_source.write_text("# membership verifier\n")

        closure_mod = types.ModuleType("verify_workbench_nix_closure_capture")
        closure_mod.__file__ = str(closure_source)
        closure_mod.verify_receipt = lambda *args: {
            "schema": "symthaea-workbench-nix-closure-capture-verification-v1",
            "status": "verified-complete-observation",
            "verifier_sha256": verifier.digest_file(closure_source),
            "capture_digest": CLOSURE_CAPTURE_DIGEST,
            "closure_digest": CLOSURE_DIGEST,
            "authority": {
                "capture_receipt_verified": True,
                "workbench_execution_qualified": False,
                "transform_executed": False,
                "fmq010_established": False,
                "neural_alignment_established": False,
                "consciousness_evidence": False,
            },
        }
        closure_mod.safe_file = lambda root, rel: Path(root) / rel
        closure_mod.closure = types.SimpleNamespace(validate_identity=lambda value: value)

        membership_mod = types.ModuleType("workbench_root_nar_membership")
        membership_mod.__file__ = str(membership_source)
        membership_mod.verify_membership = lambda path, expected, wanted: {
            "schema": "symthaea-workbench-root-nar-membership-v1",
            "status": "verified-nar-membership-only",
            "nar_sha256": expected,
            "target_path": wanted,
            "target": target,
            "authority": {
                "nar_bytes_match_verified_root": True,
                "target_membership_verified": True,
                "workbench_execution_qualified": False,
                "transform_executed": False,
                "fmq010_established": False,
                "neural_alignment_established": False,
                "consciousness_evidence": False,
            },
        }
        sys.modules[closure_mod.__name__] = closure_mod
        sys.modules[membership_mod.__name__] = membership_mod
        return closure_mod, membership_mod, closure_source, membership_source

    def pipeline(self, base: Path, target: dict) -> dict:
        _, _, closure_source, membership_source = self.install_fake_modules(base, target=target)
        closure_dir = write_closure_receipt(base / "closure")
        normalized = closure_dir / "normalized"
        normalized.mkdir()
        identity = {
            "root": ROOT,
            "closure_digest": CLOSURE_DIGEST,
            "entries": [{"path": ROOT, "nar_sha256": ROOT_NAR_HASH, "references": []}],
        }
        (normalized / "closure_identity.json").write_bytes(canonical(identity))
        capture_dir = base / "nar-capture"
        producer.capture(closure_dir, capture_dir, runner=fake_runner(b"synthetic-nar"))
        return verifier.verify_pipeline(
            closure_dir,
            capture_dir,
            base / "profile.json",
            base / "flake.lock",
            base / "closure-producer.py",
            base / "normalizer.py",
            closure_source,
            Path(producer.__file__),
            membership_source,
        )

    def test_regular_executable_promotes_only_program_membership_identity(self):
        with tempfile.TemporaryDirectory() as td:
            result = self.pipeline(Path(td), {
                "node_type": "regular", "executable": True, "content_length": 7,
                "content_sha256": "sha256:" + "e" * 64, "symlink_target_hex": None,
            })
            self.assertEqual(result["status"], "verified-root-nar-regular-executable-membership")
            self.assertTrue(result["authority"]["root_main_program_regular_executable_verified"])
            self.assertFalse(result["authority"]["workbench_execution_qualified"])
            self.assertFalse(result["authority"]["consciousness_evidence"])

    def test_symlink_membership_does_not_invent_executable_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            result = self.pipeline(Path(td), {
                "node_type": "symlink", "executable": None, "content_length": None,
                "content_sha256": None, "symlink_target_hex": "2e2e2f6c6962",
            })
            self.assertEqual(result["status"], "verified-root-nar-symlink-membership")
            self.assertTrue(result["authority"]["target_membership_verified"])
            self.assertFalse(result["authority"]["root_main_program_regular_executable_verified"])
            self.assertFalse(result["authority"]["symlink_resolution_verified"])


if __name__ == "__main__":
    unittest.main()
