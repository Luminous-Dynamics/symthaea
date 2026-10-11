#!/usr/bin/env python3
"""Inspect or checksum-verify one file from a published Zenodo record.

Metadata-only by default. Downloads are capped, checksum-checked, never extracted,
and promoted from a temporary file only after size and checksum verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

USER_AGENT = "LuminousDynamics-ProteinResearch/1.0"
ALLOWED_ZENODO_HOSTS = {"zenodo.org", "www.zenodo.org", "api.zenodo.org"}
DEFAULT_TIMEOUT_SECONDS = 30
MAX_METADATA_BYTES = 25 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024


class FetchError(RuntimeError):
    """Expected failure when remote metadata or file integrity is not acceptable."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_final_response_url(response: Any, action: str) -> None:
    """Reject unexpected HTTP redirects when the response exposes its final URL."""
    geturl = getattr(response, "geturl", None)
    if not callable(geturl):  # urllib responses expose geturl; lightweight test doubles may not.
        return
    final_url = geturl()
    parsed = urlparse(final_url)
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.hostname not in ALLOWED_ZENODO_HOSTS
    ):
        raise FetchError(f"refusing {action} after redirect to a non-Zenodo or non-HTTPS URL: {final_url!r}")


def request_bytes(url: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            validate_final_response_url(response, "metadata retrieval")
            raw = response.read(MAX_METADATA_BYTES + 1)
            if len(raw) > MAX_METADATA_BYTES:
                raise FetchError(f"metadata response exceeded {MAX_METADATA_BYTES} bytes")
            return raw
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise FetchError(f"request failed for {url}: {exc}") from exc


def fetch_record_metadata(record_id: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> tuple[dict[str, Any], bytes]:
    if not re.fullmatch(r"[0-9]+", record_id):
        raise FetchError("record ID must contain decimal digits only")
    url = f"https://zenodo.org/api/records/{record_id}"
    raw = request_bytes(url, timeout)
    try:
        metadata = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FetchError(f"Zenodo record response is not valid JSON: {exc}") from exc
    if not isinstance(metadata, dict) or not isinstance(metadata.get("files"), list):
        raise FetchError("Zenodo response lacks a record object with a files array")
    return metadata, raw


def parse_file_metadata(record: dict[str, Any], filename: str) -> dict[str, Any]:
    if not filename or filename in {".", ".."} or "/" in filename or "\\" in filename:
        raise FetchError("filename must be an exact basename, not a path")
    matches = [item for item in record["files"] if isinstance(item, dict) and item.get("key", item.get("filename")) == filename]
    if len(matches) != 1:
        raise FetchError(f"expected exactly one file named {filename!r}; found {len(matches)}")
    item = matches[0]
    size = item.get("size", item.get("filesize"))
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise FetchError("Zenodo file metadata has no valid non-negative integer size")
    checksum = item.get("checksum")
    if not isinstance(checksum, str) or ":" not in checksum:
        raise FetchError("Zenodo file metadata has no supported algorithm:digest checksum")
    algorithm, digest = checksum.split(":", 1)
    algorithm = algorithm.lower()
    expected_lengths = {"md5": 32, "sha256": 64}
    if algorithm not in expected_lengths or not re.fullmatch(rf"[0-9a-fA-F]{{{expected_lengths[algorithm]}}}", digest):
        raise FetchError(f"unsupported or malformed checksum metadata: {checksum!r}")
    links = item.get("links")
    if not isinstance(links, dict):
        raise FetchError("Zenodo file metadata has no links object")
    download_url = links.get("content") or links.get("download")
    if not isinstance(download_url, str):
        raise FetchError("Zenodo file metadata has no content/download URL")
    parsed = urlparse(download_url)
    if parsed.scheme != "https" or parsed.username is not None or parsed.password is not None or parsed.hostname not in {"zenodo.org", "www.zenodo.org", "api.zenodo.org"}:
        raise FetchError("refusing a download URL not hosted on HTTPS Zenodo")
    return {
        "key": filename,
        "size": size,
        "checksum_algorithm": algorithm,
        "checksum_value": digest.lower(),
        "download_url": download_url,
    }


def inspect_file(record_id: str, record: dict[str, Any], record_metadata_sha256: str, file_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "Zenodo Records API",
        "record_id": record_id,
        "record_url": record.get("links", {}).get("self_html") if isinstance(record.get("links"), dict) else None,
        "doi": record.get("doi"),
        "conceptdoi": record.get("conceptdoi"),
        "version": record.get("metadata", {}).get("version") if isinstance(record.get("metadata"), dict) else None,
        "metadata_retrieved_at": datetime.now(timezone.utc).isoformat(),
        "record_metadata_sha256": record_metadata_sha256,
        **file_info,
    }


def download_verified(
    file_info: dict[str, Any],
    output_dir: Path,
    max_bytes: int,
    timeout: int = 60,
    force: bool = False,
    source_metadata: dict[str, Any] | None = None,
    record_metadata_raw: bytes | None = None,
) -> dict[str, Any]:
    size = file_info["size"]
    if max_bytes < 1:
        raise FetchError("--max-bytes must be a positive integer for downloads")
    if size > max_bytes:
        raise FetchError(f"refusing download: metadata size {size} bytes exceeds --max-bytes={max_bytes}")
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / file_info["key"]
    sidecar_path = output_dir / (file_info["key"] + ".zenodo.json")
    if final_path.exists() and not force:
        raise FetchError(f"destination already exists; refusing to overwrite: {final_path} (use --force only after review)")
    if sidecar_path.exists() and not force:
        raise FetchError(f"verification sidecar already exists without an authorized overwrite: {sidecar_path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{file_info['key']}.", suffix=".part", dir=output_dir)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    digest = hashlib.new(file_info["checksum_algorithm"])
    byte_count = 0
    request = Request(file_info["download_url"], headers={"User-Agent": USER_AGENT, "Accept": "application/octet-stream"})
    try:
        try:
            with urlopen(request, timeout=timeout) as response, temporary_path.open("wb") as out:
                validate_final_response_url(response, "file download")
                while True:
                    chunk = response.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    byte_count += len(chunk)
                    if byte_count > size or byte_count > max_bytes:
                        raise FetchError("download exceeded the metadata size or configured byte cap; partial file discarded")
                    digest.update(chunk)
                    out.write(chunk)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise FetchError(f"file download failed: {exc}") from exc
        if byte_count != size:
            raise FetchError(f"download size mismatch: expected {size}, received {byte_count}")
        observed = digest.hexdigest()
        if observed.lower() != file_info["checksum_value"].lower():
            raise FetchError(
                f"publisher checksum mismatch ({file_info['checksum_algorithm']}): "
                f"expected {file_info['checksum_value']}, received {observed}"
            )
        # Keep a local SHA-256 even when Zenodo publishes only an MD5 checksum.
        local_sha256 = hashlib.sha256()
        with temporary_path.open("rb") as verified_file:
            for chunk in iter(lambda: verified_file.read(CHUNK_BYTES), b""):
                local_sha256.update(chunk)

        metadata_snapshot_name = None
        metadata_snapshot_sha256 = None
        if record_metadata_raw is not None:
            metadata_snapshot_sha256 = sha256_bytes(record_metadata_raw)
            source_id = (source_metadata or {}).get("record_id", "unknown-record")
            metadata_snapshot = output_dir / f"zenodo-record-{source_id}-{metadata_snapshot_sha256[:12]}.json"
            if metadata_snapshot.exists():
                if sha256_bytes(metadata_snapshot.read_bytes()) != metadata_snapshot_sha256:
                    raise FetchError(f"existing metadata snapshot hash mismatch: {metadata_snapshot}")
            else:
                meta_fd, meta_temp_name = tempfile.mkstemp(prefix=".zenodo-metadata.", suffix=".part", dir=output_dir)
                try:
                    with os.fdopen(meta_fd, "wb") as meta_out:
                        meta_out.write(record_metadata_raw)
                    os.replace(meta_temp_name, metadata_snapshot)
                finally:
                    Path(meta_temp_name).unlink(missing_ok=True)
            metadata_snapshot_name = metadata_snapshot.name

        os.replace(temporary_path, final_path)
        sidecar = {
            "verification_status": "publisher_checksum_and_size_verified",
            "source_record_id": (source_metadata or {}).get("record_id"),
            "source_doi": (source_metadata or {}).get("doi"),
            "source_version": (source_metadata or {}).get("version"),
            "source_record_metadata_sha256": metadata_snapshot_sha256,
            "source_metadata_snapshot": metadata_snapshot_name,
            "download_url": file_info["download_url"],
            "publisher_checksum_algorithm": file_info["checksum_algorithm"],
            "publisher_checksum_value": file_info["checksum_value"],
            "expected_size_bytes": size,
            "downloaded_size_bytes": byte_count,
            "local_sha256": local_sha256.hexdigest(),
            "output_filename": final_path.name,
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }
        sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"path": str(final_path), **sidecar}
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-id", required=True, help="decimal Zenodo record ID; for PeptoneBench use 17306061")
    parser.add_argument("--filename", required=True, help="exact file basename, such as PeptoneDBs.tar.gz or Predictions.tar.gz")
    parser.add_argument("--download", action="store_true", help="download after showing metadata; default is metadata-only")
    parser.add_argument("--output-dir", type=Path, default=None, help="destination directory; required with --download")
    parser.add_argument("--max-bytes", type=int, default=None, help="hard download size cap; required with --download")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="metadata request timeout in seconds")
    parser.add_argument("--force", action="store_true", help="replace an existing destination only after the new copy verifies")
    args = parser.parse_args(argv)
    try:
        record, record_metadata_raw = fetch_record_metadata(args.record_id, args.timeout)
        metadata_record = inspect_file(
            args.record_id, record, sha256_bytes(record_metadata_raw), parse_file_metadata(record, args.filename)
        )
        print(json.dumps(metadata_record, indent=2, sort_keys=True))
        if not args.download:
            print("METADATA_ONLY: no dataset bytes were downloaded", file=sys.stderr)
            return 0
        if args.output_dir is None or args.max_bytes is None:
            raise FetchError("--download requires both --output-dir and --max-bytes")
        result = download_verified(
            metadata_record,
            args.output_dir,
            args.max_bytes,
            timeout=max(args.timeout, 60),
            force=args.force,
            source_metadata=metadata_record,
            record_metadata_raw=record_metadata_raw,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (FetchError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: unexpected acquisition failure; no unverified archive was accepted: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
