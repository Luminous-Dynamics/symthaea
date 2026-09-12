#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

PLAN_SCHEMA = "ll009n.nasa-acquisition-plan.v1"
LOCK_SCHEMA = "ll009n.nasa-source-lock.v1"
TRANSPORT_SCHEMA = "ll009n.nasa-transport-receipt.v1"
VERIFY_SCHEMA = "ll009n.nasa-offline-verification.v1"
CHUNK_BYTES = 4 * 1024 * 1024


class NError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")) + "\n").encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: pathlib.Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
            count += len(chunk)
    return digest.hexdigest(), count


def safe_relpath(value: str) -> pathlib.PurePosixPath:
    if not isinstance(value, str) or not value:
        raise NError("artifact_path must be a non-empty string")
    path = pathlib.PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise NError(f"unsafe artifact_path {value!r}")
    return path


def canonical_host(value: str) -> str:
    try:
        return value.encode("idna").decode("ascii").lower().rstrip(".")
    except UnicodeError as exc:
        raise NError(f"invalid host {value!r}") from exc


def checked_url(value: str, allowed_hosts: set[str]) -> urllib.parse.ParseResult:
    if not isinstance(value, str) or not value:
        raise NError("source_url must be a non-empty string")
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme.lower() != "https":
        raise NError(f"source_url must use https: {value}")
    if parsed.username is not None or parsed.password is not None:
        raise NError("source_url must not contain credentials")
    if not parsed.hostname:
        raise NError(f"source_url missing host: {value}")
    if parsed.port not in (None, 443):
        raise NError(f"source_url uses unexpected port: {value}")
    if canonical_host(parsed.hostname) not in allowed_hosts:
        raise NError(f"source_url host not allowlisted: {parsed.hostname}")
    if parsed.fragment:
        raise NError(f"source_url fragment not allowed: {value}")
    return parsed


def validate_plan(plan: Any) -> tuple[set[str], list[dict[str, Any]]]:
    if not isinstance(plan, dict) or plan.get("schema_version") != PLAN_SCHEMA:
        raise NError(f"schema_version must be {PLAN_SCHEMA}")
    for key in ("study_id", "provider", "evidence_scope"):
        if not isinstance(plan.get(key), str) or not plan[key]:
            raise NError(f"missing {key}")
    hosts = plan.get("allowed_hosts")
    if not isinstance(hosts, list) or not hosts or not all(isinstance(x, str) and x for x in hosts):
        raise NError("allowed_hosts must be a non-empty string list")
    allowed_hosts = {canonical_host(x) for x in hosts}
    files = plan.get("files")
    if not isinstance(files, list) or not files:
        raise NError("files must be a non-empty list")

    ids: set[str] = set()
    paths: set[str] = set()
    urls: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, raw in enumerate(files):
        if not isinstance(raw, dict):
            raise NError(f"files[{index}] must be an object")
        entry = dict(raw)
        for key in ("source_id", "dataset_id", "role", "source_url", "artifact_path"):
            if not isinstance(entry.get(key), str) or not entry[key]:
                raise NError(f"files[{index}] missing {key}")
        source_id = entry["source_id"]
        rel = safe_relpath(entry["artifact_path"])
        parsed = checked_url(entry["source_url"], allowed_hosts)
        basename = pathlib.PurePosixPath(parsed.path).name
        if basename and basename != rel.name:
            raise NError(
                f"{source_id}: URL basename {basename!r} does not match artifact basename {rel.name!r}"
            )
        if source_id in ids:
            raise NError(f"duplicate source_id {source_id}")
        if entry["artifact_path"] in paths:
            raise NError(f"duplicate artifact_path {entry['artifact_path']}")
        if entry["source_url"] in urls:
            raise NError(f"duplicate source_url {entry['source_url']}")
        expected = entry.get("expected_sha256")
        if expected is not None and (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(ch not in "0123456789abcdef" for ch in expected)
        ):
            raise NError(f"{source_id}: expected_sha256 must be lowercase hex or null")
        required_for = entry.get("required_for")
        if not isinstance(required_for, list) or not required_for or not all(
            isinstance(x, str) and x for x in required_for
        ):
            raise NError(f"{source_id}: required_for must be a non-empty string list")
        ids.add(source_id)
        paths.add(entry["artifact_path"])
        urls.add(entry["source_url"])
        validated.append(entry)

    complete_sets = plan.get("promotion_sets")
    if not isinstance(complete_sets, list) or not complete_sets:
        raise NError("promotion_sets must be a non-empty list")
    for item in complete_sets:
        if not isinstance(item, dict) or not isinstance(item.get("set_id"), str):
            raise NError("each promotion set requires set_id")
        if not isinstance(item.get("claim_scope"), str) or not item["claim_scope"]:
            raise NError(f"promotion set {item.get('set_id')} requires claim_scope")
        required_ids = item.get("required_source_ids")
        if not isinstance(required_ids, list) or not required_ids:
            raise NError(f"promotion set {item.get('set_id')} requires required_source_ids")
        missing = [value for value in required_ids if value not in ids]
        if missing:
            raise NError(f"promotion set {item['set_id']} references unknown sources: {missing}")
    return allowed_hosts, validated


class RestrictedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, allowed_hosts: set[str]):
        super().__init__()
        self.allowed_hosts = allowed_hosts
        self.chain: list[str] = []

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urllib.parse.urljoin(req.full_url, newurl)
        checked_url(target, self.allowed_hosts)
        self.chain.append(target)
        return super().redirect_request(req, fp, code, msg, headers, target)


def parse_content_length(value: str | None, source_id: str) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise NError(f"{source_id}: malformed Content-Length") from exc
    if parsed < 0:
        raise NError(f"{source_id}: negative Content-Length")
    return parsed


def atomic_install_no_clobber(temp_path: pathlib.Path, destination: pathlib.Path) -> None:
    try:
        os.link(temp_path, destination)
    except FileExistsError as exc:
        raise NError(f"refusing to overwrite existing artifact {destination}") from exc
    except OSError as exc:
        raise NError(f"failed atomic no-clobber install for {destination}: {exc}") from exc
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def acquire_one(
    entry: dict[str, Any], root: pathlib.Path, allowed_hosts: set[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_id = entry["source_id"]
    checked_url(entry["source_url"], allowed_hosts)
    rel = safe_relpath(entry["artifact_path"])
    destination = root / pathlib.Path(*rel.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise NError(f"{source_id}: destination already exists; use offline verification instead")

    redirect_handler = RestrictedRedirectHandler(allowed_hosts)
    opener = urllib.request.build_opener(redirect_handler)
    request = urllib.request.Request(
        entry["source_url"],
        headers={
            "User-Agent": "Symthaea-LL009N-Evidence-Acquisition/1",
            "Accept": "application/octet-stream,*/*;q=0.1",
            "Accept-Encoding": "identity",
        },
        method="GET",
    )
    fd, temp_name = tempfile.mkstemp(prefix=f".{rel.name}.", suffix=".partial", dir=destination.parent)
    temp_path = pathlib.Path(temp_name)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with os.fdopen(fd, "wb") as output:
            try:
                response = opener.open(request, timeout=120)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                raise NError(f"{source_id}: download failed: {exc}") from exc
            with response:
                final_url = response.geturl()
                checked_url(final_url, allowed_hosts)
                status = getattr(response, "status", None)
                if status is not None and status != 200:
                    raise NError(f"{source_id}: expected HTTP 200, got {status}")
                content_encoding = response.headers.get("Content-Encoding")
                if content_encoding not in (None, "", "identity"):
                    raise NError(
                        f"{source_id}: refusing content-encoded transfer {content_encoding!r}; exact published bytes required"
                    )
                content_length = parse_content_length(response.headers.get("Content-Length"), source_id)
                for chunk in iter(lambda: response.read(CHUNK_BYTES), b""):
                    output.write(chunk)
                    digest.update(chunk)
                    byte_count += len(chunk)
                output.flush()
                os.fsync(output.fileno())
                if content_length is not None and byte_count != content_length:
                    raise NError(
                        f"{source_id}: transfer length mismatch: got {byte_count}, expected {content_length}"
                    )
                actual_sha256 = digest.hexdigest()
                expected = entry.get("expected_sha256")
                if expected is not None and actual_sha256 != expected:
                    raise NError(f"{source_id}: expected source hash mismatch")
                lock_entry = {
                    "source_id": source_id,
                    "dataset_id": entry["dataset_id"],
                    "role": entry["role"],
                    "source_url": entry["source_url"],
                    "artifact_path": entry["artifact_path"],
                    "sha256": actual_sha256,
                    "byte_size": byte_count,
                    "required_for": entry["required_for"],
                    "ll009l_binding": {
                        "path": entry["artifact_path"],
                        "sha256": actual_sha256,
                    },
                }
                transport_entry = {
                    **lock_entry,
                    "final_url": final_url,
                    "redirect_chain": list(redirect_handler.chain),
                    "http_status": status,
                    "content_length_header": content_length,
                    "content_type": response.headers.get("Content-Type"),
                    "content_encoding": content_encoding,
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                }
        atomic_install_no_clobber(temp_path, destination)
        return lock_entry, transport_entry
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def write_immutable(path: pathlib.Path, value: dict[str, Any]) -> None:
    payload = canonical_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise NError(f"refusing to overwrite differing immutable output {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def promotion_status(plan: dict[str, Any], acquired_ids: set[str]) -> list[dict[str, Any]]:
    result = []
    for item in plan["promotion_sets"]:
        required = list(item["required_source_ids"])
        missing = sorted(set(required) - acquired_ids)
        result.append(
            {
                "set_id": item["set_id"],
                "claim_scope": item["claim_scope"],
                "required_source_ids": required,
                "complete": not missing,
                "missing_source_ids": missing,
            }
        )
    return result


def acquire(plan_path: pathlib.Path, root: pathlib.Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = json.loads(plan_path.read_text())
    allowed_hosts, entries = validate_plan(plan)
    root.mkdir(parents=True, exist_ok=True)
    lock_entries = []
    transport_entries = []
    installed_paths: list[pathlib.Path] = []
    try:
        for entry in entries:
            locked, transport = acquire_one(entry, root, allowed_hosts)
            lock_entries.append(locked)
            transport_entries.append(transport)
            rel = safe_relpath(entry["artifact_path"])
            installed_paths.append(root / pathlib.Path(*rel.parts))
    except Exception:
        for path in reversed(installed_paths):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise

    ids = {entry["source_id"] for entry in lock_entries}
    plan_sha256 = sha256_file(plan_path)[0]
    lock = {
        "schema_version": LOCK_SCHEMA,
        "status": "exact_source_bytes_locked",
        "study_id": plan["study_id"],
        "provider": plan["provider"],
        "evidence_scope": plan["evidence_scope"],
        "plan_sha256": plan_sha256,
        "files": lock_entries,
        "promotion_sets": promotion_status(plan, ids),
        "promotion_policy": {
            "source_byte_identity_established": True,
            "offline_replay_required": True,
            "scientific_sufficiency_established": False,
            "ll009l_materialization_required": True,
            "ll009m_radial_uncertainty_required": True,
            "ll009k_horizon_reduction_required": True,
        },
        "non_claims": [
            "A source lock authenticates exact acquired bytes; it does not establish raster scientific sufficiency.",
            "Network metadata is transport provenance, not cryptographic identity.",
            "A complete horizon claim requires the promotion set declared by this plan and downstream LL-009L/M/K evidence.",
        ],
    }
    lock["receipt_sha256"] = sha256_bytes(canonical_bytes(lock))
    transport = {
        "schema_version": TRANSPORT_SCHEMA,
        "status": "network_transport_observed",
        "study_id": plan["study_id"],
        "plan_sha256": plan_sha256,
        "files": transport_entries,
        "non_claims": [
            "HTTP headers and redirects are recorded only as transport lineage.",
            "This receipt is not a substitute for exact SHA-256 source locking or offline replay.",
        ],
    }
    transport["receipt_sha256"] = sha256_bytes(canonical_bytes(transport))
    return lock, transport


def verify(lock_path: pathlib.Path, root: pathlib.Path) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text())
    if not isinstance(lock, dict) or lock.get("schema_version") != LOCK_SCHEMA:
        raise NError(f"lock schema must be {LOCK_SCHEMA}")
    files = lock.get("files")
    if not isinstance(files, list) or not files:
        raise NError("lock files missing")
    results = []
    seen_paths: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise NError("invalid lock file entry")
        rel_value = entry.get("artifact_path")
        digest = entry.get("sha256")
        size = entry.get("byte_size")
        if not isinstance(rel_value, str) or rel_value in seen_paths:
            raise NError("invalid or duplicate lock artifact_path")
        seen_paths.add(rel_value)
        if not isinstance(digest, str) or len(digest) != 64 or not isinstance(size, int) or size < 0:
            raise NError(f"invalid hash/size for {rel_value}")
        rel = safe_relpath(rel_value)
        path = root / pathlib.Path(*rel.parts)
        if not path.is_file():
            raise NError(f"missing locked file {rel_value}")
        actual_digest, actual_size = sha256_file(path)
        if actual_digest != digest:
            raise NError(f"offline hash mismatch for {rel_value}")
        if actual_size != size:
            raise NError(f"offline size mismatch for {rel_value}")
        results.append(
            {
                "source_id": entry["source_id"],
                "artifact_path": rel_value,
                "sha256": actual_digest,
                "byte_size": actual_size,
            }
        )
    lock_digest = sha256_file(lock_path)[0]
    result = {
        "schema_version": VERIFY_SCHEMA,
        "status": "pass",
        "study_id": lock["study_id"],
        "lock_file_sha256": lock_digest,
        "files": results,
        "exact_source_bytes_replayed": True,
        "scientific_sufficiency_established": False,
        "next_required_stage": "LL-009L native raster materialization under the pinned GIS toolchain",
    }
    result["receipt_sha256"] = sha256_bytes(canonical_bytes(result))
    return result


def self_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        data_root = root / "data"
        data_root.mkdir()
        payload_a = b"lunar-elevation\x00\x01"
        payload_b = b"uncertainty\x02\x03"
        (data_root / "a.tif").write_bytes(payload_a)
        (data_root / "b.tif").write_bytes(payload_b)
        lock = {
            "schema_version": LOCK_SCHEMA,
            "study_id": "self-test",
            "files": [
                {
                    "source_id": "a",
                    "artifact_path": "a.tif",
                    "sha256": hashlib.sha256(payload_a).hexdigest(),
                    "byte_size": len(payload_a),
                },
                {
                    "source_id": "b",
                    "artifact_path": "b.tif",
                    "sha256": hashlib.sha256(payload_b).hexdigest(),
                    "byte_size": len(payload_b),
                },
            ],
        }
        lock_path = root / "lock.json"
        lock_path.write_bytes(canonical_bytes(lock))
        first = verify(lock_path, data_root)
        second = verify(lock_path, data_root)
        if canonical_bytes(first) != canonical_bytes(second):
            raise NError("self-test deterministic replay failed")
        (data_root / "a.tif").write_bytes(payload_a + b"tamper")
        try:
            verify(lock_path, data_root)
        except NError as exc:
            if "mismatch" not in str(exc):
                raise
        else:
            raise NError("self-test expected tamper detection")

        plan = {
            "schema_version": PLAN_SCHEMA,
            "study_id": "self-test",
            "provider": "NASA",
            "evidence_scope": "test",
            "allowed_hosts": ["pgda.gsfc.nasa.gov"],
            "files": [
                {
                    "source_id": "x",
                    "dataset_id": "d",
                    "role": "elevation",
                    "source_url": "https://pgda.gsfc.nasa.gov/products/1/x.tif",
                    "artifact_path": "x.tif",
                    "expected_sha256": None,
                    "required_for": ["near-field"],
                }
            ],
            "promotion_sets": [
                {
                    "set_id": "near-field",
                    "claim_scope": "near field only",
                    "required_source_ids": ["x"],
                }
            ],
        }
        validate_plan(plan)
        unsafe = json.loads(json.dumps(plan))
        unsafe["files"][0]["artifact_path"] = "../x.tif"
        try:
            validate_plan(unsafe)
        except NError:
            pass
        else:
            raise NError("self-test expected path traversal rejection")
        bad_host = json.loads(json.dumps(plan))
        bad_host["files"][0]["source_url"] = "https://example.com/x.tif"
        try:
            validate_plan(bad_host)
        except NError:
            pass
        else:
            raise NError("self-test expected host rejection")


def main() -> int:
    parser = argparse.ArgumentParser(description="LL-009N exact NASA source-byte acquisition and replay")
    sub = parser.add_subparsers(dest="command", required=True)

    acquire_parser = sub.add_parser("acquire", help="download exact sources and emit immutable receipts")
    acquire_parser.add_argument("--plan", required=True, type=pathlib.Path)
    acquire_parser.add_argument("--artifact-root", required=True, type=pathlib.Path)
    acquire_parser.add_argument("--lock-output", required=True, type=pathlib.Path)
    acquire_parser.add_argument("--transport-output", required=True, type=pathlib.Path)

    verify_parser = sub.add_parser("verify", help="offline replay exact locked bytes")
    verify_parser.add_argument("--lock", required=True, type=pathlib.Path)
    verify_parser.add_argument("--artifact-root", required=True, type=pathlib.Path)
    verify_parser.add_argument("--output", required=True, type=pathlib.Path)

    sub.add_parser("self-test", help="run deterministic fail-closed logic tests")
    args = parser.parse_args()

    try:
        if args.command == "acquire":
            lock, transport = acquire(args.plan, args.artifact_root)
            write_immutable(args.lock_output, lock)
            write_immutable(args.transport_output, transport)
        elif args.command == "verify":
            result = verify(args.lock, args.artifact_root)
            write_immutable(args.output, result)
        else:
            self_test()
            print("LL-009N self-test: PASS")
    except (NError, json.JSONDecodeError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
