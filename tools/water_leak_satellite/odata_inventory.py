#!/usr/bin/env python3
"""Create and verify a metadata-only Copernicus OData inventory.

This is an independent cross-check for the STAC inventory, not a raster
download, scene-suitability assessment, or leak classifier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlparse
from urllib.request import Request, urlopen
from typing import Any

try:
    from .strict_json import loads as strict_json_loads
except ImportError:  # Direct script execution from the repository root.
    from strict_json import loads as strict_json_loads

DEFAULT_ENDPOINT = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
ALLOWED_HOST = "catalogue.dataspace.copernicus.eu"
USER_AGENT = "LuminousDynamics-WaterLeak-ODataInventory/0.1"
PROFILES = (
    {"profile_id": "sentinel-1-grd", "collection": "SENTINEL-1", "product_type": "IW_GRDH_1S"},
    {"profile_id": "sentinel-2-l2a", "collection": "SENTINEL-2", "product_type": "S2MSI2A"},
)
SELECT_FIELDS = "Id,Name,ContentDate,GeoFootprint,Online,ContentLength,Checksum"
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
RETRYABLE_HTTP_STATUSES = (408, 429, 500, 502, 503, 504)
MAX_RETRIES = 5
RETRY_AFTER_CAP_SECONDS = 10.0
SUPPORTED_WINDOW_MONTHS = (1, 3, 12)


class ODataError(RuntimeError):
    """A response or query could not be safely interpreted."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_json(value: Any) -> str:
    try:
        return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError):
        return ""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _finite_number(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def parse_bbox(value: str) -> list[float]:
    try:
        bbox = [float(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bbox must contain four comma-separated numbers") from exc
    if len(bbox) != 4:
        raise argparse.ArgumentTypeError("bbox must be west,south,east,north")
    west, south, east, north = bbox
    if (
        not all(math.isfinite(v) for v in bbox)
        or not (-180 <= west < east <= 180)
        or not (-90 <= south < north <= 90)
    ):
        raise argparse.ArgumentTypeError("bbox bounds must be finite and ordered WGS84 west,south,east,north")
    return bbox


def parse_date(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise argparse.ArgumentTypeError("date must use canonical YYYY-MM-DD")
    return parsed


def query_windows(
    start: date,
    end_inclusive: date,
    window_months: int = 3,
) -> list[tuple[date, date]]:
    """Partition inclusive dates into calendar-aligned half-open month windows."""
    if end_inclusive < start:
        raise ODataError("end date must be on or after start date")
    if window_months not in SUPPORTED_WINDOW_MONTHS:
        raise ODataError(f"window_months must be one of {SUPPORTED_WINDOW_MONTHS}")
    end_exclusive = end_inclusive + timedelta(days=1)
    windows: list[tuple[date, date]] = []
    cursor = start
    while cursor < end_exclusive:
        next_boundary_month = ((cursor.month - 1) // window_months + 1) * window_months + 1
        if next_boundary_month > 12:
            boundary = date(cursor.year + 1, 1, 1)
        else:
            boundary = date(cursor.year, next_boundary_month, 1)
        stop = min(boundary, end_exclusive)
        windows.append((cursor, stop))
        cursor = stop
    return windows


def _fmt_datetime(value: date) -> str:
    return f"{value.isoformat()}T00:00:00.000Z"


def build_filter(profile: dict[str, str], bbox: list[float], start: date, end_exclusive: date) -> str:
    """Build a closed, SRID=4326 AOI polygon and a half-open acquisition-time filter."""
    if end_exclusive <= start:
        raise ODataError("query end_exclusive must be after start")
    west, south, east, north = bbox
    ring = (
        f"{west:.8f} {south:.8f}, "
        f"{east:.8f} {south:.8f}, "
        f"{east:.8f} {north:.8f}, "
        f"{west:.8f} {north:.8f}, "
        f"{west:.8f} {south:.8f}"
    )
    spatial = f"OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(({ring}))')"
    product = (
        "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
        f"and att/OData.CSC.StringAttribute/Value eq '{profile['product_type']}')"
    )
    return (
        f"Collection/Name eq '{profile['collection']}' and {product} and {spatial} "
        f"and ContentDate/Start ge {_fmt_datetime(start)} "
        f"and ContentDate/Start lt {_fmt_datetime(end_exclusive)}"
    )


def _redacted_url(url: str) -> str | None:
    try:
        parsed = urlparse(url)
        port = parsed.port
    except (TypeError, ValueError):
        return None
    if (
        parsed.scheme != "https"
        or parsed.hostname != ALLOWED_HOST
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or parsed.fragment
    ):
        return None
    return f"https://{ALLOWED_HOST}{parsed.path or '/'}"


def validate_next_url(current_url: str, href: str) -> str:
    resolved = urljoin(current_url, href)
    parsed = urlparse(resolved)
    if (
        _redacted_url(resolved) is None
        or parsed.path.rstrip("/") != "/odata/v1/Products"
    ):
        raise ODataError("OData pagination returned a non-approved URL/path")
    return resolved


def _contains_href(value: Any) -> bool:
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, nested in current.items():
                if str(key).lower() == "href":
                    return True
                if isinstance(nested, (dict, list)):
                    stack.append(nested)
        elif isinstance(current, list):
            stack.extend(nested for nested in current if isinstance(nested, (dict, list)))
    return False


def _retry_delay(retry_index: int, headers: Any = None) -> float:
    """Return bounded Retry-After or exponential backoff delay in seconds."""
    retry_after = None
    if headers is not None:
        try:
            retry_after = float(headers.get("Retry-After", ""))
        except (TypeError, ValueError):
            retry_after = None
    if retry_after is None or not math.isfinite(retry_after):
        retry_after = min(0.5 * (2 ** retry_index), 4.0)
    return min(max(retry_after, 0.0), RETRY_AFTER_CAP_SECONDS)


def request_json(
    url: str,
    timeout: float,
    max_retries: int = 2,
) -> tuple[bytes, dict[str, Any], int]:
    """Fetch and strictly parse one page, retrying only bounded transient failures."""
    if _redacted_url(url) is None:
        raise ODataError("OData request URL is not on the approved HTTPS host")
    if type(max_retries) is not int or not 0 <= max_retries <= MAX_RETRIES:
        raise ODataError(f"max_retries must be an integer from 0 to {MAX_RETRIES}")

    for attempt_index in range(max_retries + 1):
        request = Request(
            url,
            headers={"Accept": "application/json", "User-Agent": USER_AGENT},
            method="GET",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read()
        except HTTPError as exc:
            excerpt = exc.read(300).decode("utf-8", errors="replace")
            if exc.code in RETRYABLE_HTTP_STATUSES and attempt_index < max_retries:
                time.sleep(_retry_delay(attempt_index, exc.headers))
                continue
            raise ODataError(
                f"OData HTTP {exc.code} after {attempt_index + 1} attempt(s); "
                f"response excerpt: {excerpt}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            if attempt_index < max_retries:
                time.sleep(_retry_delay(attempt_index))
                continue
            raise ODataError(
                f"OData request failed after {attempt_index + 1} attempt(s) "
                f"({type(exc).__name__}): {exc}"
            ) from exc

        try:
            parsed = strict_json_loads(raw)
        except (UnicodeDecodeError, ValueError, RecursionError) as exc:
            raise ODataError("OData returned malformed, ambiguous, or non-standard JSON") from exc
        if not isinstance(parsed, dict) or not isinstance(parsed.get("value"), list):
            raise ODataError("OData response must be an object containing a value array")
        return raw, parsed, attempt_index + 1

    raise ODataError("OData request retries exhausted unexpectedly")


def summarize_product(product: Any, profile: dict[str, str]) -> dict[str, Any]:
    if not isinstance(product, dict):
        raise ODataError("OData product must be a JSON object")
    product_id = product.get("Id")
    name = product.get("Name")
    if not isinstance(product_id, str) or not product_id or not isinstance(name, str) or not name:
        raise ODataError("OData product is missing string Id or Name")
    summary = {
        "profile_id": profile["profile_id"],
        "collection": profile["collection"],
        "product_type": profile["product_type"],
        "id": product_id,
        "name": name,
        "content_date": product.get("ContentDate"),
        "geo_footprint": product.get("GeoFootprint"),
        "online": product.get("Online"),
        "content_length": product.get("ContentLength"),
        "checksum": product.get("Checksum"),
        "source_product_sha256": sha256_json(product),
    }
    summary["summary_sha256"] = sha256_json(summary)
    return summary


def inventory_query(
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    profile: dict[str, str],
    bbox: list[float],
    start: date,
    end_exclusive: date,
    limit: int,
    max_pages: int,
    timeout: float,
    max_retries: int = 2,
) -> dict[str, Any]:
    if _redacted_url(endpoint) is None or urlparse(endpoint).path.rstrip("/") != "/odata/v1/Products":
        raise ODataError(f"endpoint must be HTTPS on {ALLOWED_HOST}/odata/v1/Products")
    if not 1 <= limit <= 100:
        raise ODataError("page limit must be from 1 to 100")
    if max_pages < 1 or not math.isfinite(timeout) or timeout <= 0:
        raise ODataError("max_pages and timeout must be positive and finite")
    if type(max_retries) is not int or not 0 <= max_retries <= MAX_RETRIES:
        raise ODataError(f"max_retries must be an integer from 0 to {MAX_RETRIES}")

    filter_text = build_filter(profile, bbox, start, end_exclusive)
    params = {
        "$filter": filter_text,
        "$orderby": "ContentDate/Start asc",
        "$select": SELECT_FIELDS,
        "$top": str(limit),
    }
    current_url = endpoint + "?" + urlencode(params)
    page_records: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_urls: set[str] = set()
    duplicate_count = 0

    while True:
        if current_url in seen_urls:
            raise ODataError("OData pagination repeated a URL; refusing a loop")
        if len(page_records) >= max_pages:
            raise ODataError(f"page limit ({max_pages}) reached while nextLink may remain; query is incomplete")
        seen_urls.add(current_url)
        raw, response, request_attempts = request_json(current_url, timeout, max_retries)
        features = response["value"]
        page_records.append({
            "page": len(page_records) + 1,
            "request_url_redacted": _redacted_url(current_url),
            "request_url_sha256": sha256_bytes(current_url.encode("utf-8")),
            "response_sha256": sha256_bytes(raw),
            "request_attempts": request_attempts,
            "feature_count": len(features),
        })
        for product in features:
            summary = summarize_product(product, profile)
            product_id = summary["id"]
            if product_id in seen_ids:
                duplicate_count += 1
                continue
            seen_ids.add(product_id)
            items.append(summary)

        next_link = response.get("@odata.nextLink")
        if next_link is None:
            break
        if not isinstance(next_link, str) or not next_link:
            raise ODataError("OData nextLink must be a non-empty string")
        current_url = validate_next_url(current_url, next_link)

    return {
        "query_id": f"{profile['profile_id']}:{start.isoformat()}:{end_exclusive.isoformat()}",
        "profile_id": profile["profile_id"],
        "collection": profile["collection"],
        "product_type": profile["product_type"],
        "start_date_inclusive": start.isoformat(),
        "end_date_exclusive": end_exclusive.isoformat(),
        "filter": filter_text,
        "page_count": len(page_records),
        "page_records": page_records,
        "item_count": len(items),
        "duplicate_items_dropped": duplicate_count,
        "items": items,
        "catalogue_discovery_only": True,
    }


def verify_inventory(document: Any) -> list[str]:
    errors: list[str] = []
    if _contains_href(document):
        errors.append("inventory: href key is forbidden anywhere in the saved artifact")
    if not isinstance(document, dict):
        return ["inventory: root must be an object"]
    required = {
        "schema_version", "generated_at_utc", "catalogue_endpoint", "scope", "queries",
        "request_policy", "overall_claim_ceiling", "inventory_manifest_sha256",
    }
    if set(document) != required:
        errors.append("inventory: root keys do not match the declared OData inventory shape")

    manifest = document.get("inventory_manifest_sha256")
    unsigned = dict(document)
    unsigned.pop("inventory_manifest_sha256", None)
    if not isinstance(manifest, str) or not SHA256_RE.fullmatch(manifest):
        errors.append("inventory_manifest_sha256: missing or invalid SHA-256")
    elif sha256_json(unsigned) != manifest:
        errors.append("inventory_manifest_sha256: digest does not match document")

    if document.get("schema_version") != "water-leak-odata-inventory/v2":
        errors.append("schema_version: unsupported OData inventory schema")
    endpoint = document.get("catalogue_endpoint", "")
    if (
        _redacted_url(endpoint) is None
        or urlparse(endpoint).path.rstrip("/") != "/odata/v1/Products"
    ):
        errors.append("catalogue_endpoint: must be the query-free approved OData Products endpoint")
    scope = document.get("scope")
    if not isinstance(scope, dict):
        errors.append("scope: must be an object")
        return errors
    window_months = scope.get("window_months")
    if type(window_months) is not int or window_months not in SUPPORTED_WINDOW_MONTHS:
        errors.append(f"scope.window_months: must be one of {SUPPORTED_WINDOW_MONTHS}")
        window_months = 3

    policy = document.get("request_policy")
    if not isinstance(policy, dict) or set(policy) != {
        "timeout_seconds", "max_retries", "retryable_http_statuses",
        "backoff_strategy", "retry_after_cap_seconds",
    }:
        errors.append("request_policy: missing or unexpected retry/timeout fields")
        max_retries = -1
    else:
        timeout_seconds = policy.get("timeout_seconds")
        if not _finite_number(timeout_seconds) or timeout_seconds <= 0:
            errors.append("request_policy.timeout_seconds: must be finite and positive")
        max_retries = policy.get("max_retries")
        if type(max_retries) is not int or not 0 <= max_retries <= MAX_RETRIES:
            errors.append(f"request_policy.max_retries: must be an integer from 0 to {MAX_RETRIES}")
            max_retries = -1
        if policy.get("retryable_http_statuses") != list(RETRYABLE_HTTP_STATUSES):
            errors.append("request_policy.retryable_http_statuses: does not match the bounded transient status list")
        if policy.get("backoff_strategy") != "bounded-exponential":
            errors.append("request_policy.backoff_strategy: unsupported strategy")
        if policy.get("retry_after_cap_seconds") != RETRY_AFTER_CAP_SECONDS:
            errors.append("request_policy.retry_after_cap_seconds: does not match the supported cap")

    bbox = scope.get("bbox_wgs84")
    bbox_valid = (
        isinstance(bbox, list)
        and len(bbox) == 4
        and all(_finite_number(v) for v in bbox)
    )
    if not bbox_valid:
        errors.append("scope.bbox_wgs84: must contain four finite numeric bounds")
    elif not (-180 <= bbox[0] < bbox[2] <= 180 and -90 <= bbox[1] < bbox[3] <= 90):
        errors.append("scope.bbox_wgs84: WGS84 bounds are not ordered")
        bbox_valid = False
    try:
        start = date.fromisoformat(scope.get("start_date_inclusive", ""))
        end = date.fromisoformat(scope.get("end_date_inclusive", ""))
        if start.isoformat() != scope.get("start_date_inclusive") or end.isoformat() != scope.get("end_date_inclusive"):
            raise ValueError("noncanonical date")
        if end < start:
            raise ValueError("reversed date range")
    except (TypeError, ValueError):
        errors.append("scope: start/end must be ordered canonical YYYY-MM-DD dates")
        return errors

    queries = document.get("queries")
    if not isinstance(queries, list):
        errors.append("queries: must be a list")
        return errors

    expected = []
    for profile in PROFILES:
        for window_start, window_end in query_windows(start, end, window_months):
            expected.append((profile, window_start, window_end))
    actual_keys = []
    for query_index, query in enumerate(queries):
        prefix = f"queries[{query_index}]"
        if not isinstance(query, dict):
            errors.append(f"{prefix}: must be an object")
            continue
        profile = next((p for p in PROFILES if p["profile_id"] == query.get("profile_id")), None)
        if profile is None:
            errors.append(f"{prefix}.profile_id: unsupported profile")
            continue
        q_start_text = query.get("start_date_inclusive", "")
        q_end_text = query.get("end_date_exclusive", "")
        try:
            q_start = date.fromisoformat(q_start_text)
            q_end = date.fromisoformat(q_end_text)
        except (TypeError, ValueError):
            errors.append(f"{prefix}: invalid query dates")
            continue
        if q_start.isoformat() != q_start_text or q_end.isoformat() != q_end_text:
            errors.append(f"{prefix}: query dates must use canonical YYYY-MM-DD")
        actual_keys.append((profile["profile_id"], q_start, q_end))
        if query.get("collection") != profile["collection"] or query.get("product_type") != profile["product_type"]:
            errors.append(f"{prefix}: collection/product type do not match the declared profile")
        if query.get("query_id") != f"{profile['profile_id']}:{q_start.isoformat()}:{q_end.isoformat()}":
            errors.append(f"{prefix}.query_id: does not match profile and query dates")
        if q_end <= q_start or q_start < start or q_end > end + timedelta(days=1):
            errors.append(f"{prefix}: query interval falls outside the declared scope")
        expected_filter = build_filter(profile, bbox, q_start, q_end) if bbox_valid and q_end > q_start else None
        if query.get("filter") != expected_filter:
            errors.append(f"{prefix}.filter: does not match declared profile, AOI and query dates")

        pages = query.get("page_records")
        if not isinstance(pages, list) or not pages:
            errors.append(f"{prefix}.page_records: must be a non-empty list")
            pages = []
        if type(query.get("page_count")) is not int or query.get("page_count") != len(pages):
            errors.append(f"{prefix}.page_count: does not match page_records length")
        page_total = 0
        page_total_valid = True
        for page_index, page in enumerate(pages):
            pfx = f"{prefix}.page_records[{page_index}]"
            if not isinstance(page, dict):
                errors.append(f"{pfx}: must be an object")
                page_total_valid = False
                continue
            if type(page.get("page")) is not int or page.get("page") != page_index + 1:
                errors.append(f"{pfx}.page: numbering must be consecutive from 1")
            for digest_name in ("request_url_sha256", "response_sha256"):
                digest = page.get(digest_name)
                if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
                    errors.append(f"{pfx}.{digest_name}: invalid SHA-256")
            if _redacted_url_is_safe_redacted(page.get("request_url_redacted")) is False:
                errors.append(f"{pfx}.request_url_redacted: must be query-free HTTPS URL on the approved host")
            attempts = page.get("request_attempts")
            if type(attempts) is not int or not 1 <= attempts <= max_retries + 1:
                errors.append(f"{pfx}.request_attempts: must be from 1 to max_retries + 1")
            count = page.get("feature_count")
            if type(count) is not int or count < 0:
                errors.append(f"{pfx}.feature_count: must be non-negative integer")
                page_total_valid = False
            else:
                page_total += count

        items = query.get("items")
        if not isinstance(items, list):
            errors.append(f"{prefix}.items: must be a list")
            items = []
        if type(query.get("item_count")) is not int or query.get("item_count") != len(items):
            errors.append(f"{prefix}.item_count: does not match items length")
        dup = query.get("duplicate_items_dropped")
        if type(dup) is not int or dup < 0:
            errors.append(f"{prefix}.duplicate_items_dropped: must be non-negative integer")
        elif page_total_valid and type(query.get("item_count")) is int and page_total != query["item_count"] + dup:
            errors.append(f"{prefix}: page feature total must equal item_count plus duplicates dropped")

        seen_ids = set()
        for item_index, item in enumerate(items):
            pfx = f"{prefix}.items[{item_index}]"
            if not isinstance(item, dict):
                errors.append(f"{pfx}: must be an object")
                continue
            if item.get("profile_id") != profile["profile_id"] or item.get("collection") != profile["collection"] or item.get("product_type") != profile["product_type"]:
                errors.append(f"{pfx}: profile/collection/product type mismatch")
            item_id = item.get("id")
            if not isinstance(item_id, str) or not item_id:
                errors.append(f"{pfx}.id: missing string identifier")
            elif item_id in seen_ids:
                errors.append(f"{pfx}.id: duplicate product ID")
            else:
                seen_ids.add(item_id)
            for digest_name in ("source_product_sha256", "summary_sha256"):
                digest = item.get(digest_name)
                if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
                    errors.append(f"{pfx}.{digest_name}: invalid SHA-256")
            unsigned_item = dict(item)
            unsigned_item.pop("summary_sha256", None)
            if sha256_json(unsigned_item) != item.get("summary_sha256"):
                errors.append(f"{pfx}.summary_sha256: digest does not match summary")
            if not isinstance(item.get("name"), str) or not item.get("name"):
                errors.append(f"{pfx}.name: missing product name")

    actual_keys = sorted(actual_keys, key=lambda v: (v[0], v[1], v[2]))
    expected_keys = sorted([(p["profile_id"], s, e) for p, s, e in expected], key=lambda v: (v[0], v[1], v[2]))
    if actual_keys != expected_keys:
        errors.append("queries: partitioned query set does not exactly cover both declared profiles and date scope")
    return errors


def _redacted_url_is_safe_redacted(value: Any) -> bool:
    if not isinstance(value, str) or "?" in value or "#" in value:
        return False
    parsed = urlparse(value)
    return (
        _redacted_url(value) == value
        and parsed.path.rstrip("/") == "/odata/v1/Products"
    )


def build_document(
    bbox: list[float],
    start: date,
    end: date,
    queries: list[dict[str, Any]],
    *,
    window_months: int = 3,
    timeout_seconds: float = 30.0,
    max_retries: int = 2,
) -> dict[str, Any]:
    if window_months not in SUPPORTED_WINDOW_MONTHS:
        raise ODataError(f"window_months must be one of {SUPPORTED_WINDOW_MONTHS}")
    if not _finite_number(timeout_seconds) or timeout_seconds <= 0:
        raise ODataError("timeout_seconds must be finite and positive")
    if type(max_retries) is not int or not 0 <= max_retries <= MAX_RETRIES:
        raise ODataError(f"max_retries must be an integer from 0 to {MAX_RETRIES}")
    document: dict[str, Any] = {
        "schema_version": "water-leak-odata-inventory/v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "catalogue_endpoint": DEFAULT_ENDPOINT,
        "scope": {
            "bbox_wgs84": bbox,
            "start_date_inclusive": start.isoformat(),
            "end_date_inclusive": end.isoformat(),
            "study_area_status": "reconnaissance_only_unless_separately_approved",
            "window_months": window_months,
        },
        "queries": queries,
        "request_policy": {
            "timeout_seconds": timeout_seconds,
            "max_retries": max_retries,
            "retryable_http_statuses": list(RETRYABLE_HTTP_STATUSES),
            "backoff_strategy": "bounded-exponential",
            "retry_after_cap_seconds": RETRY_AFTER_CAP_SECONDS,
        },
        "overall_claim_ceiling": (
            "OData catalogue discovery only. Product listing/footprint intersection does not establish "
            "raster suitability, continuous coverage, leak evidence, model performance or commercial-use rights."
        ),
    }
    document["inventory_manifest_sha256"] = sha256_json(document)
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", type=parse_bbox, default=[27.7, -26.5, 28.4, -25.8])
    parser.add_argument("--start", type=parse_date)
    parser.add_argument("--end", type=parse_date)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2, help=f"bounded retries after transient failures (0-{MAX_RETRIES})")
    parser.add_argument("--window-months", type=int, choices=SUPPORTED_WINDOW_MONTHS, default=3)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify", type=Path, help="verify a previously saved OData inventory without network access")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.verify:
        try:
            document = strict_json_loads(args.verify.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError, RecursionError) as exc:
            print(f"FAIL: cannot read strict UTF-8 JSON inventory: {exc}", file=sys.stderr)
            return 2
        errors = verify_inventory(document)
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            print("FAIL: OData inventory verification failed", file=sys.stderr)
            return 1
        print("PASS: OData manifest, scope/query binding, page counts and item summaries verified.")
        print("This does not authenticate the provider or establish imagery suitability/leak truth.")
        return 0

    if args.start is None or args.end is None or args.out is None:
        parser.error("--start, --end and --out are required unless --verify is used")
    if not 1 <= args.limit <= 100:
        parser.error("--limit must be from 1 to 100")
    if args.max_pages < 1:
        parser.error("--max-pages must be >= 1")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if not 0 <= args.retries <= MAX_RETRIES:
        parser.error(f"--retries must be from 0 to {MAX_RETRIES}")
    if args.end < args.start:
        parser.error("--end must be on or after --start")
    try:
        queries = []
        for profile in PROFILES:
            for window_start, window_end in query_windows(args.start, args.end, args.window_months):
                queries.append(inventory_query(
                    profile=profile,
                    bbox=args.bbox,
                    start=window_start,
                    end_exclusive=window_end,
                    limit=args.limit,
                    max_pages=args.max_pages,
                    timeout=args.timeout,
                    max_retries=args.retries,
                ))
        document = build_document(
            args.bbox,
            args.start,
            args.end,
            queries,
            window_months=args.window_months,
            timeout_seconds=args.timeout,
            max_retries=args.retries,
        )
        atomic_write_text(args.out, json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    except (ODataError, OSError, TypeError, ValueError, RecursionError) as exc:
        print(f"OData inventory failed closed: {exc}", file=sys.stderr)
        return 2
    counts = ", ".join(f"{q['profile_id']}:{q['start_date_inclusive']}..{q['end_date_exclusive']}={q['item_count']}" for q in queries)
    print(f"wrote {args.out}; metadata inventory only; {counts}")
    print(f"inventory_manifest_sha256={document['inventory_manifest_sha256']}")
    return 0


def atomic_write_text(path: Path, text: str) -> None:
    import os
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{path.name}.",
            suffix=".tmp", dir=path.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            import os
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
