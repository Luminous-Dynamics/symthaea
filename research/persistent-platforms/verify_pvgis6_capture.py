"""Verify a previously captured PVGIS 6 raw-evidence bundle without network access.

This checks internal byte/digest/metadata consistency only. It does not establish
that PVGIS authored the files, that the endpoint schema is scientifically suitable,
or that any observation is correctly interpreted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

from capture_pvgis6 import (
    API_BASE,
    API_HOST,
    CaptureError,
    build_request_url,
    parse_json_bytes,
    sha256_bytes,
)

_REQUIRED_FILES = {
    "parameters.input.json",
    "request.json",
    "openapi.json",
    "raw-response.json",
    "capture.json",
}


def _read_regular_file(directory: Path, name: str) -> bytes:
    path = directory / name
    if path.is_symlink() or not path.is_file():
        raise CaptureError(f"required evidence file is missing or not a regular file: {name}")
    return path.read_bytes()


def _digest_record(metadata: dict, section: str, filename: str, payload: bytes) -> None:
    record = metadata.get(section)
    if not isinstance(record, dict) or record.get("file") != filename:
        raise CaptureError(f"capture metadata has invalid file binding for {section}")
    if record.get("sha256") != sha256_bytes(payload):
        raise CaptureError(f"SHA-256 mismatch for {filename}")
    if record.get("bytes") != len(payload):
        raise CaptureError(f"byte-length mismatch for {filename}")


def verify_bundle(bundle_dir: str) -> dict:
    """Fail closed on missing, changed, or internally inconsistent bundle evidence."""
    directory = Path(bundle_dir).expanduser().absolute()
    if directory.is_symlink() or not directory.is_dir():
        raise CaptureError("bundle path must be an existing non-symlink directory")
    present = {entry.name for entry in directory.iterdir()}
    if present != _REQUIRED_FILES:
        missing = sorted(_REQUIRED_FILES - present)
        extra = sorted(present - _REQUIRED_FILES)
        raise CaptureError(f"bundle file set mismatch; missing={missing}, extra={extra}")

    raw = {name: _read_regular_file(directory, name) for name in _REQUIRED_FILES}
    parameters = parse_json_bytes(raw["parameters.input.json"], "parameter input")
    request = parse_json_bytes(raw["request.json"], "request manifest")
    capture = parse_json_bytes(raw["capture.json"], "capture manifest")
    openapi = parse_json_bytes(raw["openapi.json"], "OpenAPI document")
    parse_json_bytes(raw["raw-response.json"], "raw endpoint response")
    if not isinstance(parameters, dict) or not isinstance(request, dict):
        raise CaptureError("parameter input and request manifest must be JSON objects")
    if not isinstance(capture, dict) or not isinstance(openapi, dict):
        raise CaptureError("capture manifest and OpenAPI document must be JSON objects")

    if request.get("schema_version") != "pvgis6-request-capture-v1":
        raise CaptureError("unsupported request-manifest schema version")
    if request.get("api_base") != API_BASE or request.get("method") != "GET":
        raise CaptureError("request manifest API base or method mismatch")
    if request.get("accept") != "application/json":
        raise CaptureError("request manifest Accept value mismatch")
    endpoint = request.get("endpoint_path")
    if not isinstance(endpoint, str):
        raise CaptureError("request manifest endpoint path is missing")
    expected_url = build_request_url(endpoint, parameters)
    if request.get("query_parameters") != parameters or request.get("request_url") != expected_url:
        raise CaptureError("request manifest does not match exact parameter input")
    if request.get("parameter_input_sha256") != sha256_bytes(raw["parameters.input.json"]):
        raise CaptureError("parameter-input digest mismatch")

    if capture.get("schema_version") != "pvgis6-raw-capture-v1":
        raise CaptureError("unsupported capture-manifest schema version")
    _digest_record(capture, "parameter_input", "parameters.input.json", raw["parameters.input.json"])
    _digest_record(capture, "request_manifest", "request.json", raw["request.json"])
    _digest_record(capture, "openapi", "openapi.json", raw["openapi.json"])
    _digest_record(capture, "source_response", "raw-response.json", raw["raw-response.json"])

    response = capture.get("source_response")
    openapi_record = capture.get("openapi")
    if not isinstance(response, dict) or not isinstance(openapi_record, dict):
        raise CaptureError("capture response records must be objects")
    for label, record in (("source response", response), ("OpenAPI", openapi_record)):
        if record.get("http_status") != 200:
            raise CaptureError(f"{label} status is not HTTP 200")
        content_type = record.get("content_type")
        if not isinstance(content_type, str) or content_type.split(";", 1)[0].strip().lower() != "application/json":
            raise CaptureError(f"{label} content type is not application/json")
        final_url_value = record.get("final_url")
        if not isinstance(final_url_value, str):
            raise CaptureError(f"{label} final URL is missing or not text")
        final_url = urlsplit(final_url_value)
        if final_url.scheme != "https" or final_url.hostname != API_HOST or final_url.username or final_url.password:
            raise CaptureError(f"{label} final URL is outside official PVGIS HTTPS host")

    if response.get("requested_url") != expected_url:
        raise CaptureError("captured endpoint requested URL does not match request manifest")
    if openapi_record.get("url") != f"{API_BASE}/openapi.json":
        raise CaptureError("OpenAPI URL does not match pinned API base")
    # The capture client permits redirects only to the same official HTTPS host.
    # The final URLs are checked above; they may therefore differ from requested URLs.

    return {
        "valid": True,
        "bundle_path": str(directory),
        "file_count": len(_REQUIRED_FILES),
        "request_url": expected_url,
        "digests": {
            name: sha256_bytes(raw[name])
            for name in sorted(_REQUIRED_FILES)
        },
        "claim_boundary": (
            "Internal consistency only; source authenticity, schema suitability, "
            "scientific interpretation, and derivation correctness are not established."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify internal consistency of a PVGIS 6 raw capture bundle")
    parser.add_argument("bundle_dir", help="Directory created by capture_pvgis6.py")
    args = parser.parse_args()
    try:
        result = verify_bundle(args.bundle_dir)
    except (OSError, CaptureError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
