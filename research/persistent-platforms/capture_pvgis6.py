"""Capture exact PVGIS 6 JSON and OpenAPI bytes without normalizing scientific data.

This is an evidence-acquisition helper, not an irradiance parser or solar-resource
profile generator. A later transform must inspect the captured endpoint's schema,
units, aggregation semantics, and source metadata before building a profile.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

API_BASE = "https://photovoltaic-geographical-information-system.ec.europa.eu/api/v6"
API_HOST = "photovoltaic-geographical-information-system.ec.europa.eu"
OPENAPI_URL = f"{API_BASE}/openapi.json"
USER_AGENT = "symthaea-pvgis6-capture/1.0"
MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_ALLOWED_ENDPOINT_FAMILIES = {
    "performance", "power", "meteorological", "solar-position", "coverage", "topography"
}
_PARAMETER_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


class CaptureError(ValueError):
    """A fail-closed capture input or response error."""


def sha256_bytes(payload: bytes) -> str:
    """Digest exact bytes without decoding or reserializing them."""
    if not isinstance(payload, bytes):
        raise CaptureError("payload must be bytes")
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value) -> bytes:
    """Serialize metadata using the documented Python canonical JSON policy."""
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CaptureError(f"value is not finite canonical JSON data: {exc}") from exc


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CaptureError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str):
    raise CaptureError(f"non-standard JSON numeric constant is forbidden: {value}")


def parse_json_bytes(payload: bytes, label: str):
    """Validate a JSON document but never replace its original bytes."""
    if not isinstance(payload, bytes):
        raise CaptureError(f"{label} must be bytes")
    try:
        return json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise CaptureError(f"{label} is not strict UTF-8 JSON: {exc}") from exc


def validate_endpoint_path(endpoint: str) -> str:
    """Accept a relative path within a documented PVGIS 6 endpoint family."""
    if not isinstance(endpoint, str) or not endpoint:
        raise CaptureError("endpoint must be a non-empty relative path")
    if endpoint.startswith("/") or "?" in endpoint or "#" in endpoint or "\\" in endpoint:
        raise CaptureError("endpoint must be relative and contain no query, fragment, or backslash")
    parts = endpoint.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise CaptureError("endpoint path contains an empty or traversal segment")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", part) for part in parts):
        raise CaptureError("endpoint path contains unsupported characters")
    if parts[0] not in _ALLOWED_ENDPOINT_FAMILIES:
        raise CaptureError(f"endpoint family must be one of {sorted(_ALLOWED_ENDPOINT_FAMILIES)}")
    return endpoint


def validate_query_parameters(parameters) -> dict:
    """Validate a simple scalar query map and reject implicit defaults."""
    if not isinstance(parameters, dict) or not parameters:
        raise CaptureError("query parameters must be a non-empty JSON object")
    validated = {}
    for name, value in parameters.items():
        if not isinstance(name, str) or not _PARAMETER_NAME.fullmatch(name):
            raise CaptureError(f"invalid query parameter name: {name!r}")
        if isinstance(value, bool):
            validated[name] = value
        elif isinstance(value, int):
            validated[name] = value
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise CaptureError(f"query parameter {name} must be finite")
            validated[name] = value
        elif isinstance(value, str) and value:
            validated[name] = value
        else:
            raise CaptureError(
                f"query parameter {name} must be a non-empty string, finite number, or boolean"
            )
    return validated


def query_scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise CaptureError("query numbers must be finite")
        return str(value)
    if isinstance(value, str) and value:
        return value
    raise CaptureError("unsupported query parameter value")


def build_request_url(endpoint: str, parameters: dict) -> str:
    """Build a deterministic HTTPS URL with query names sorted lexically."""
    endpoint = validate_endpoint_path(endpoint)
    parameters = validate_query_parameters(parameters)
    pairs = [(name, query_scalar(parameters[name])) for name in sorted(parameters)]
    return f"{API_BASE}/{endpoint}?{urlencode(pairs)}"


def validate_payload_size(payload: bytes, limit: int = MAX_RESPONSE_BYTES) -> None:
    if len(payload) > limit:
        raise CaptureError(f"response exceeds the {limit}-byte capture limit")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class _SameHostHTTPSRedirectHandler(HTTPRedirectHandler):
    """Refuse redirects away from the official HTTPS PVGIS host."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urlsplit(newurl)
        if target.scheme != "https" or target.hostname != API_HOST or target.username or target.password:
            raise HTTPError(req.full_url, code, "refused redirect outside official HTTPS PVGIS host", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _fetch_json(url: str, *, timeout: float, max_bytes: int) -> dict:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != API_HOST or parsed.username or parsed.password:
        raise CaptureError("request URL must use official PVGIS HTTPS host")
    request = Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT}, method="GET")
    opener = build_opener(_SameHostHTTPSRedirectHandler())
    try:
        with opener.open(request, timeout=timeout) as response:
            status = response.status
            headers = list(response.getheaders())
            final_url = response.geturl()
            content_type = response.headers.get("Content-Type", "")
            payload = response.read(max_bytes + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise CaptureError(f"PVGIS request failed: {exc}") from exc

    validate_payload_size(payload, max_bytes)
    final = urlsplit(final_url)
    if final.scheme != "https" or final.hostname != API_HOST or final.username or final.password:
        raise CaptureError("final response URL is outside the official HTTPS PVGIS host")
    if status != 200:
        raise CaptureError(f"PVGIS returned HTTP {status}; no successful capture was written")
    if content_type.split(";", 1)[0].strip().lower() != "application/json":
        raise CaptureError(f"expected application/json response, got {content_type!r}")
    parse_json_bytes(payload, "PVGIS response")
    return {
        "payload": payload,
        "status": status,
        "headers": [[str(key), str(value)] for key, value in headers],
        "content_type": content_type,
        "final_url": final_url,
        "retrieved_at_utc": _now_utc(),
    }


def _write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def capture(endpoint: str, parameters_bytes: bytes, output_dir: str, *, timeout: float = 45.0,
            max_bytes: int = MAX_RESPONSE_BYTES) -> Path:
    """Acquire OpenAPI + raw endpoint response and publish a new evidence bundle.

    The destination must not exist. Both JSON documents are archived byte-for-byte;
    metadata is written separately and does not overwrite or normalize source bytes.
    """
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise CaptureError("timeout must be a finite positive number")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise CaptureError("max_bytes must be a positive integer")
    validate_payload_size(b"", max_bytes)
    parameters = parse_json_bytes(parameters_bytes, "parameter file")
    parameters = validate_query_parameters(parameters)
    request_url = build_request_url(endpoint, parameters)

    destination = Path(output_dir).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise CaptureError(f"destination already exists; refusing to overwrite: {destination}")
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)

    started_at = _now_utc()
    openapi_capture = _fetch_json(OPENAPI_URL, timeout=float(timeout), max_bytes=max_bytes)
    if not isinstance(parse_json_bytes(openapi_capture["payload"], "OpenAPI specification"), dict):
        raise CaptureError("OpenAPI specification root must be an object")
    source_capture = _fetch_json(request_url, timeout=float(timeout), max_bytes=max_bytes)

    request_manifest = {
        "schema_version": "pvgis6-request-capture-v1",
        "api_base": API_BASE,
        "endpoint_path": endpoint,
        "method": "GET",
        "accept": "application/json",
        "query_parameters": parameters,
        "request_url": request_url,
        "parameter_input_sha256": sha256_bytes(parameters_bytes),
    }
    request_bytes = canonical_json_bytes(request_manifest)
    capture_manifest = {
        "schema_version": "pvgis6-raw-capture-v1",
        "capture_started_at_utc": started_at,
        "capture_completed_at_utc": _now_utc(),
        "api_maturity_note": (
            "PVGIS 6 is documented as a public beta/testing API; do not infer production stability."
        ),
        "request_manifest": {
            "file": "request.json",
            "sha256": sha256_bytes(request_bytes),
            "bytes": len(request_bytes),
        },
        "parameter_input": {
            "file": "parameters.input.json",
            "sha256": sha256_bytes(parameters_bytes),
            "bytes": len(parameters_bytes),
        },
        "openapi": {
            "url": OPENAPI_URL,
            "final_url": openapi_capture["final_url"],
            "file": "openapi.json",
            "sha256": sha256_bytes(openapi_capture["payload"]),
            "bytes": len(openapi_capture["payload"]),
            "content_type": openapi_capture["content_type"],
            "http_status": openapi_capture["status"],
            "response_headers": openapi_capture["headers"],
            "retrieved_at_utc": openapi_capture["retrieved_at_utc"],
        },
        "source_response": {
            "requested_url": request_url,
            "final_url": source_capture["final_url"],
            "file": "raw-response.json",
            "sha256": sha256_bytes(source_capture["payload"]),
            "bytes": len(source_capture["payload"]),
            "content_type": source_capture["content_type"],
            "http_status": source_capture["status"],
            "response_headers": source_capture["headers"],
            "retrieved_at_utc": source_capture["retrieved_at_utc"],
        },
        "evidence_boundary": (
            "Raw acquisition only. No field mapping, unit conversion, aggregation, daylight "
            "derivation, or solar-resource profile generation has been performed."
        ),
    }
    capture_bytes = canonical_json_bytes(capture_manifest)

    temporary = Path(tempfile.mkdtemp(prefix=".pvgis6-capture-", dir=parent))
    try:
        _write_new(temporary / "parameters.input.json", parameters_bytes)
        _write_new(temporary / "request.json", request_bytes)
        _write_new(temporary / "openapi.json", openapi_capture["payload"])
        _write_new(temporary / "raw-response.json", source_capture["payload"])
        _write_new(temporary / "capture.json", capture_bytes)
        if destination.exists() or destination.is_symlink():
            raise CaptureError(f"destination appeared during capture; refusing overwrite: {destination}")
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Archive exact PVGIS 6 JSON and OpenAPI bytes; does not build a solar profile."
    )
    parser.add_argument("--endpoint", required=True, help="Relative documented path, e.g. performance/broadband")
    parser.add_argument("--params-json", required=True, help="UTF-8 JSON file containing a non-empty scalar query object")
    parser.add_argument("--outdir", required=True, help="New output directory; existing paths are never overwritten")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--max-bytes", type=int, default=MAX_RESPONSE_BYTES)
    args = parser.parse_args()
    try:
        parameters_bytes = Path(args.params_json).read_bytes()
        destination = capture(args.endpoint, parameters_bytes, args.outdir, timeout=args.timeout, max_bytes=args.max_bytes)
    except (OSError, CaptureError) as exc:
        parser.error(str(exc))
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
