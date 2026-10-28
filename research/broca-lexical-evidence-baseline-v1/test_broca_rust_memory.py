"""Offline protocol tests for broca_rust_memory.py using a fake JSONL worker.

Run with:
  python3 research/broca-lexical-evidence-baseline-v1/test_broca_rust_memory.py

These tests do not validate the Rust worker or official benchmark integration.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import threading
import types
import unittest
from unittest.mock import patch


RESEARCH_DIR = Path(__file__).resolve().parent
ADAPTER_PATH = RESEARCH_DIR / "broca_rust_memory.py"


class FakeMemoryBase:
    def __init__(self, memory_params):
        self.memory_params = dict(memory_params)
        self._context_local = threading.local()

    @property
    def memory_config(self):
        return {"memory_type": self.memory_type, "memory_params": dict(self.memory_params)}

    def set_query_context(self, *, query_invocation_id):
        self._context_local.query_context = {"query_invocation_id": query_invocation_id}

    def get_query_context(self):
        return dict(getattr(self._context_local, "query_context", {}))

    def clear_query_context(self):
        if hasattr(self._context_local, "query_context"):
            del self._context_local.query_context

    def save_memory(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "memory_config.json").write_text(
            json.dumps(self.memory_config, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._save_backend(output_dir)


def make_memory_package(package_name: str):
    package = types.ModuleType(package_name)
    package.__path__ = []
    sys.modules[package_name] = package

    memory_module = types.ModuleType(package_name + ".memory")
    memory_module.Memory = FakeMemoryBase
    memory_module.MemoryContextItem = dict

    def register_memory(cls):
        return cls

    def require(condition, message):
        if not condition:
            raise RuntimeError(message)

    memory_module.register_memory = register_memory
    memory_module.require = require
    sys.modules[package_name + ".memory"] = memory_module
    return package


def import_adapter(package_name: str):
    spec = importlib.util.spec_from_file_location(
        package_name + ".broca_rust",
        ADAPTER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load adapter module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.BrocaRustMemory


FAKE_WORKER = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import hashlib, json, os, sys, time

    mode = os.environ.get("BROCA_FAKE_WORKER_MODE", "normal")
    scope = None
    trajectory_ids = set()

    def emit(request_id, ok, result=None, error=None):
        response = {"request_id": request_id, "ok": ok}
        if ok:
            response["result"] = result or {}
        else:
            response["error"] = error or "fake worker error"
        if mode == "wrong_id":
            response["request_id"] = "mismatched"
        print(json.dumps(response, separators=(",", ":")), flush=True)

    for line in sys.stdin:
        try:
            request = json.loads(line)
            request_id = request.get("request_id")
            op = request.get("op")
            if op == "init":
                scope = request.get("scope_id")
                emit(request_id, True, {
                    "scope_id": scope,
                    "capabilities": {"text_query": True, "visual_query": False}
                })
            elif op == "insert":
                trajectory = request.get("trajectory", {})
                trajectory_id = trajectory.get("id")
                if trajectory_id in trajectory_ids:
                    emit(request_id, False, error="duplicate trajectory")
                else:
                    trajectory_ids.add(trajectory_id)
                    emit(request_id, True, {
                        "trajectory_id": trajectory_id,
                        "inserted_evidence_items": 1,
                        "source_sha256": "a" * 64
                    })
            elif op == "query":
                if mode == "slow_query":
                    time.sleep(1.0)
                value = "billing panel"
                digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
                if mode == "digest_mismatch":
                    digest = "0" * 64
                emit(request_id, True, {
                    "context": [{"type": "text", "value": value}],
                    "audit": [{
                        "evidence_id": "evidence-1",
                        "trajectory_id": next(iter(trajectory_ids), "trajectory-1"),
                        "source_pointer": "/states/0/text",
                        "source_sha256": "a" * 64,
                        "content_sha256": digest,
                        "bm25_score": 1.25
                    }],
                    "image_query_received_but_not_used": bool(request.get("query_image"))
                })
            else:
                emit(request_id, False, error="unsupported operation")
        except Exception as error:
            print(json.dumps({
                "request_id": None, "ok": False, "error": str(error)
            }), flush=True)
    """
)


class BrocaRustMemoryAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package_name = "_broca_adapter_fixture"
        make_memory_package(cls.package_name)
        cls.Adapter = import_adapter(cls.package_name)

    @classmethod
    def tearDownClass(cls):
        for module_name in list(sys.modules):
            if module_name == cls.package_name or module_name.startswith(cls.package_name + "."):
                sys.modules.pop(module_name, None)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.worker = self.root / "fake_worker.py"
        self.worker.write_text(FAKE_WORKER, encoding="utf-8")
        self.worker.chmod(0o755)
        self.instances = []

    def tearDown(self):
        for instance in self.instances:
            try:
                instance.close()
            except Exception:
                pass
        self.temp.cleanup()

    def make_adapter(self, *, timeout=2.0, line_limit=32 * 1024 * 1024, scope="web"):
        adapter = self.Adapter({
            "worker_path": str(self.worker),
            "scope_id": scope,
            "timeout_seconds": timeout,
            "max_jsonl_line_bytes": line_limit,
            "bm25_params": {"k1": 1.2, "b": 0.75, "max_items": 5},
        })
        self.instances.append(adapter)
        return adapter

    @staticmethod
    def trajectory(trajectory_id="trajectory-1"):
        return {
            "id": trajectory_id,
            "goal": "Open billing settings",
            "states": [{"url": "https://example.invalid/billing", "text": "Billing Settings page"}],
        }

    def test_insert_query_and_post_query_audit_round_trip(self):
        adapter = self.make_adapter()
        adapter.insert(self.trajectory())
        adapter.set_query_context(query_invocation_id="query-17")
        try:
            items = adapter.query("Where is Billing Settings?")
            self.assertEqual(items, [{"type": "text", "value": "billing panel"}])
            metadata = adapter.post_query_hook(
                query="Where is Billing Settings?",
                query_image=None,
                memory_context=items,
            )
        finally:
            adapter.clear_query_context()
        self.assertEqual(metadata["retrieval_type"], "lexical_bm25")
        self.assertEqual(metadata["audit"][0]["source_pointer"], "/states/0/text")

    def test_scope_is_required_and_restricted(self):
        with self.assertRaises(RuntimeError):
            self.make_adapter(scope="question-derived")

    def test_response_id_mismatch_fails_closed(self):
        with patch.dict(os.environ, {"BROCA_FAKE_WORKER_MODE": "wrong_id"}):
            with self.assertRaises(RuntimeError):
                self.make_adapter()

    def test_timeout_terminates_worker(self):
        with patch.dict(os.environ, {"BROCA_FAKE_WORKER_MODE": "slow_query"}):
            adapter = self.make_adapter(timeout=0.1)
            adapter.insert(self.trajectory())
            with self.assertRaises(TimeoutError):
                adapter.query("billing")

    def test_context_digest_mismatch_is_rejected(self):
        with patch.dict(os.environ, {"BROCA_FAKE_WORKER_MODE": "digest_mismatch"}):
            adapter = self.make_adapter()
            adapter.insert(self.trajectory())
            with self.assertRaises(RuntimeError):
                adapter.query("billing")

    def test_configured_line_bound_rejects_oversized_request(self):
        adapter = self.make_adapter(line_limit=1024)
        with self.assertRaises(RuntimeError):
            adapter.insert({
                "id": "huge",
                "goal": "x" * 2000,
                "states": [{"text": "billing"}],
            })

    def test_save_load_journal_replays_trajectory_and_checks_digest(self):
        adapter = self.make_adapter()
        adapter.insert(self.trajectory())
        saved = self.root / "saved"
        adapter.save_memory(saved)

        second = self.make_adapter()
        second._load_backend(saved)
        self.assertEqual(second._trajectory_count, 1)
        second.set_query_context(query_invocation_id="loaded-query")
        try:
            items = second.query("billing")
            self.assertEqual(items[0]["value"], "billing panel")
        finally:
            second.clear_query_context()

        journal = saved / "broca_trajectory_journal.jsonl"
        journal.write_text("{}\n", encoding="utf-8")
        third = self.make_adapter()
        with self.assertRaises(RuntimeError):
            third._load_backend(saved)


if __name__ == "__main__":
    unittest.main()
