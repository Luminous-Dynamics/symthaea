#!/usr/bin/env python3
"""Execute the preregistered St. Lucia Sentinel product discovery.

Catalogue discovery only: this tool never downloads raster assets or previews.
Raw STAC pages are retained byte-for-byte and hashed before local selection.

Python 3 standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

SCHEMA = "symthaea-st-lucia-stac-discovery/v1"
TOOL_VERSION = "1.1.0"
STAC_ROOT = "https://stac.dataspace.copernicus.eu/v1"
STAC_HOST = "stac.dataspace.copernicus.eu"
BBOX = (32.3166667, -28.1500000, 32.6166667, -27.8500000)
S2_START = datetime(2026, 7, 1, tzinfo=timezone.utc)
S2_END_EXCLUSIVE = datetime(2026, 8, 1, tzinfo=timezone.utc)
MAX_CLOUD = 20.0
PAIR_WINDOW = timedelta(hours=72)
REQUIRED_S2_BANDS = ("B03", "B04", "B08", "B11", "B12")
USER_AGENT = "symthaea-st-lucia-discovery/1.1"

Json = dict[str, Any]
Request = dict[str, Any]
FetchResult = tuple[bytes, dict[str, str]]
Fetcher = Callable[[Request, float], FetchResult]


class DiscoveryError(RuntimeError):
    pass


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_time(value: str) -> datetime:
    if not value:
        raise DiscoveryError("missing STAC datetime")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise DiscoveryError(f"STAC datetime lacks timezone: {value}")
    return parsed.astimezone(timezone.utc)


def iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_stac_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != STAC_HOST:
        raise DiscoveryError(f"off-origin STAC URL rejected: {url}")
    if parsed.username or parsed.password or parsed.port not in (None, 443):
        raise DiscoveryError(f"noncanonical STAC authority rejected: {url}")
    if not parsed.path.startswith("/v1/"):
        raise DiscoveryError(f"STAC URL outside /v1 rejected: {url}")
    if parsed.fragment:
        raise DiscoveryError(f"STAC URL fragment rejected: {url}")
    return url


def build_items_url(
    collection: str,
    start: datetime,
    end: datetime,
    *,
    cloud_limit: float | None = None,
) -> str:
    params: list[tuple[str, str]] = [
        ("bbox", ",".join(f"{v:.7f}" for v in BBOX)),
        ("datetime", f"{iso_z(start)}/{iso_z(end)}"),
        ("limit", "100"),
    ]
    if cloud_limit is not None:
        params.extend(
            [
                ("filter-lang", "cql2-text"),
                ("filter", f"eo:cloud_cover<={cloud_limit:g}"),
            ]
        )
    url = f"{STAC_ROOT}/collections/{collection}/items?{urllib.parse.urlencode(params)}"
    return _validate_stac_url(url)


def initial_request(url: str) -> Request:
    return {"method": "GET", "url": _validate_stac_url(url), "body": None, "headers": {}}


def _http_fetch(request: Request, timeout: float) -> FetchResult:
    method = str(request.get("method", "GET")).upper()
    url = _validate_stac_url(str(request["url"]))
    body_obj = request.get("body")
    data = None if body_obj is None else canonical_json_bytes(body_obj)
    headers = {"Accept": "application/geo+json, application/json", "User-Agent": USER_AGENT}
    headers.update({str(k): str(v) for k, v in request.get("headers", {}).items()})
    if data is not None:
        headers.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as response:  # nosec B310: URL is pinned above
        raw = response.read()
        response_headers = {key.lower(): value for key, value in response.headers.items()}
    return raw, dict(sorted(response_headers.items()))


def _next_request(page: Json) -> Request | None:
    for link in page.get("links", []):
        if link.get("rel") != "next":
            continue
        href = link.get("href")
        if not isinstance(href, str) or not href:
            raise DiscoveryError("STAC next link has no href")
        href = urllib.parse.urljoin(STAC_ROOT + "/", href)
        _validate_stac_url(href)
        method = str(link.get("method", "GET")).upper()
        if method not in {"GET", "POST"}:
            raise DiscoveryError(f"unsupported STAC pagination method: {method}")
        body = link.get("body") if method == "POST" else None
        headers = link.get("headers") or {}
        if not isinstance(headers, dict):
            raise DiscoveryError("STAC next-link headers must be an object")
        return {"method": method, "url": href, "body": body, "headers": headers}
    return None


def _request_evidence(request: Request) -> Json:
    body = request.get("body")
    return {
        "method": str(request.get("method", "GET")).upper(),
        "url": str(request["url"]),
        "body": body,
        "body_sha256": None if body is None else sha256_hex(canonical_json_bytes(body)),
    }


def exhaust_pages(
    request: Request,
    output_dir: Path,
    *,
    timeout: float = 60.0,
    fetcher: Fetcher = _http_fetch,
) -> tuple[list[Json], list[Json]]:
    """Fetch every STAC page, preserving raw bytes and rejecting item conflicts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pages: list[Json] = []
    page_evidence: list[Json] = []
    seen_items: dict[str, bytes] = {}
    seen_requests: set[bytes] = set()
    current = request

    for index in range(1, 10_001):
        request_view = _request_evidence(current)
        request_key = canonical_json_bytes(request_view)
        if request_key in seen_requests:
            raise DiscoveryError("pagination request cycle detected")
        seen_requests.add(request_key)

        retrieved_at = now_utc()
        raw, response_headers = fetcher(current, timeout)
        digest = sha256_hex(raw)
        page_path = output_dir / f"page-{index:04d}.json"
        page_path.write_bytes(raw)
        (output_dir / f"page-{index:04d}.sha256").write_text(
            f"{digest}  {page_path.name}\n", encoding="utf-8"
        )

        try:
            page = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DiscoveryError(f"invalid STAC JSON on page {index}: {exc}") from exc
        if not isinstance(page, dict) or not isinstance(page.get("features", []), list):
            raise DiscoveryError(f"invalid STAC FeatureCollection on page {index}")

        for item in page.get("features", []):
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                raise DiscoveryError(f"page {index} contains item without string id")
            item_id = item["id"]
            canonical = canonical_json_bytes(item)
            previous = seen_items.get(item_id)
            if previous is not None and previous != canonical:
                raise DiscoveryError(f"duplicate-id-metadata-conflict: {item_id}")
            seen_items[item_id] = canonical

        pages.append(page)
        page_evidence.append(
            {
                "page": index,
                "path": page_path.name,
                "sha256": digest,
                "byte_len": len(raw),
                "retrieved_at_utc": retrieved_at,
                "request": request_view,
                "response_headers": response_headers,
                "feature_count": len(page.get("features", [])),
            }
        )

        nxt = _next_request(page)
        if nxt is None:
            break
        current = nxt
    else:
        raise DiscoveryError("pagination exceeded 10,000 pages")

    return pages, page_evidence


def deduplicated_items(pages: list[Json]) -> list[Json]:
    by_id: dict[str, Json] = {}
    canonical_by_id: dict[str, bytes] = {}
    for page in pages:
        for item in page.get("features", []):
            item_id = item["id"]
            canonical = canonical_json_bytes(item)
            previous = canonical_by_id.get(item_id)
            if previous is not None and previous != canonical:
                raise DiscoveryError(f"duplicate-id-metadata-conflict: {item_id}")
            canonical_by_id[item_id] = canonical
            by_id[item_id] = item
    return [by_id[item_id] for item_id in sorted(by_id)]


def _band_tokens(item: Json) -> set[str]:
    tokens: set[str] = set()
    for key, asset in (item.get("assets") or {}).items():
        upper = str(key).upper().replace("-", "_")
        for required in REQUIRED_S2_BANDS:
            if required in upper:
                tokens.add(required)
        if isinstance(asset, dict):
            for band in asset.get("eo:bands", []) or []:
                if isinstance(band, dict):
                    for field in ("name", "common_name"):
                        value = band.get(field)
                        if isinstance(value, str):
                            upper_value = value.upper().replace("-", "_")
                            for required in REQUIRED_S2_BANDS:
                                if required in upper_value:
                                    tokens.add(required)
    return tokens


def s2_eligibility(item: Json) -> list[str]:
    reasons: list[str] = []
    if item.get("collection") != "sentinel-2-l2a":
        reasons.append("wrong-collection")
    props = item.get("properties") or {}
    try:
        acquired = parse_time(str(props.get("datetime", "")))
        if not (S2_START <= acquired < S2_END_EXCLUSIVE):
            reasons.append("outside-half-open-time-window")
    except DiscoveryError:
        reasons.append("missing-or-invalid-datetime")
    cloud = props.get("eo:cloud_cover")
    if not isinstance(cloud, (int, float)) or isinstance(cloud, bool):
        reasons.append("missing-cloud-cover")
    elif float(cloud) > MAX_CLOUD:
        reasons.append("cloud-cover-too-high")
    missing = sorted(set(REQUIRED_S2_BANDS) - _band_tokens(item))
    if missing:
        reasons.append("missing-required-band-metadata:" + ",".join(missing))
    return reasons


def summarize_item(item: Json, reasons: list[str]) -> Json:
    props = item.get("properties") or {}
    return {
        "id": item["id"],
        "collection": item.get("collection"),
        "datetime": props.get("datetime"),
        "eo:cloud_cover": props.get("eo:cloud_cover"),
        "sar:instrument_mode": props.get("sar:instrument_mode"),
        "sar:polarizations": props.get("sar:polarizations"),
        "asset_keys": sorted((item.get("assets") or {}).keys()),
        "eligible": not reasons,
        "ineligibility_reasons": reasons,
        "item_sha256": sha256_hex(canonical_json_bytes(item)),
    }


def select_s2(items: list[Json]) -> tuple[Json | None, list[Json], list[str]]:
    audited: list[tuple[Json, list[str]]] = [(item, s2_eligibility(item)) for item in items]
    eligible = [item for item, reasons in audited if not reasons]
    eligible.sort(key=lambda item: (parse_time(item["properties"]["datetime"]), item["id"]))
    selected = eligible[0] if eligible else None
    summaries = [summarize_item(item, reasons) for item, reasons in audited]
    summaries.sort(key=lambda row: row["id"])
    return selected, summaries, [item["id"] for item in eligible]


def s1_eligibility(item: Json, s2_time: datetime) -> list[str]:
    reasons: list[str] = []
    if item.get("collection") != "sentinel-1-grd":
        reasons.append("wrong-collection")
    props = item.get("properties") or {}
    try:
        acquired = parse_time(str(props.get("datetime", "")))
        if abs(acquired - s2_time) > PAIR_WINDOW:
            reasons.append("outside-pair-window")
    except DiscoveryError:
        reasons.append("missing-or-invalid-datetime")
    if str(props.get("sar:instrument_mode", "")).upper() != "IW":
        reasons.append("not-iw-mode")
    pols_raw = props.get("sar:polarizations") or []
    pols = {str(value).upper() for value in pols_raw} if isinstance(pols_raw, list) else set()
    if not {"VV", "VH"}.issubset(pols):
        reasons.append("missing-vv-vh")
    return reasons


def select_s1(items: list[Json], s2_time: datetime) -> tuple[Json | None, list[Json], list[str]]:
    audited: list[tuple[Json, list[str]]] = [
        (item, s1_eligibility(item, s2_time)) for item in items
    ]
    eligible = [item for item, reasons in audited if not reasons]
    eligible.sort(
        key=lambda item: (
            abs((parse_time(item["properties"]["datetime"]) - s2_time).total_seconds()),
            parse_time(item["properties"]["datetime"]),
            item["id"],
        )
    )
    selected = eligible[0] if eligible else None
    summaries = [summarize_item(item, reasons) for item, reasons in audited]
    summaries.sort(key=lambda row: row["id"])
    return selected, summaries, [item["id"] for item in eligible]


def snapshot_digest(page_evidence: list[Json]) -> str:
    # Deliberately excludes retrieval timestamps and response headers so the same
    # exact raw pages under the same requests have the same catalogue snapshot id.
    view = [
        {
            "page": row["page"],
            "sha256": row["sha256"],
            "byte_len": row["byte_len"],
            "request": row["request"],
        }
        for row in page_evidence
    ]
    return sha256_hex(canonical_json_bytes(view))


def _write_receipt(path: Path, receipt: Json) -> None:
    body = dict(receipt)
    body.pop("receipt_sha256", None)
    body["receipt_sha256"] = sha256_hex(canonical_json_bytes(body))
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_discovery(protocol_path: Path, output_dir: Path, timeout: float) -> Json:
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = now_utc()
    protocol_bytes = protocol_path.read_bytes()
    receipt: Json = {
        "schema": SCHEMA,
        "tool_version": TOOL_VERSION,
        "started_at_utc": started_at,
        "protocol_path": protocol_path.as_posix(),
        "protocol_sha256": sha256_hex(protocol_bytes),
        "stac_root": STAC_ROOT,
        "bbox_wgs84": list(BBOX),
        "s2": {},
        "s1": {},
    }

    try:
        s2_url = build_items_url(
            "sentinel-2-l2a", S2_START, S2_END_EXCLUSIVE, cloud_limit=MAX_CLOUD
        )
        s2_pages, s2_page_evidence = exhaust_pages(
            initial_request(s2_url), output_dir / "s2", timeout=timeout
        )
        s2_items = deduplicated_items(s2_pages)
        selected_s2, s2_audit, s2_eligible_ids = select_s2(s2_items)
        receipt["s2"] = {
            "query_url": s2_url,
            "snapshot_sha256": snapshot_digest(s2_page_evidence),
            "pages": s2_page_evidence,
            "candidate_audit": s2_audit,
            "eligible_candidate_ids_in_selection_order": s2_eligible_ids,
            "selected_item_id": None if selected_s2 is None else selected_s2["id"],
            "status": "no-eligible-item-visible" if selected_s2 is None else "selected",
        }

        if selected_s2 is None:
            receipt["s1"] = {"status": "not-run-no-selected-s2"}
        else:
            s2_time = parse_time(selected_s2["properties"]["datetime"])
            s1_start = s2_time - PAIR_WINDOW
            s1_end = s2_time + PAIR_WINDOW
            s1_url = build_items_url("sentinel-1-grd", s1_start, s1_end)
            s1_pages, s1_page_evidence = exhaust_pages(
                initial_request(s1_url), output_dir / "s1", timeout=timeout
            )
            s1_items = deduplicated_items(s1_pages)
            selected_s1, s1_audit, s1_eligible_ids = select_s1(s1_items, s2_time)
            receipt["s1"] = {
                "query_url": s1_url,
                "paired_to_s2_datetime": iso_z(s2_time),
                "snapshot_sha256": snapshot_digest(s1_page_evidence),
                "pages": s1_page_evidence,
                "candidate_audit": s1_audit,
                "eligible_candidate_ids_in_selection_order": s1_eligible_ids,
                "selected_item_id": None if selected_s1 is None else selected_s1["id"],
                "status": "no-eligible-item-visible" if selected_s1 is None else "selected",
            }
        receipt["status"] = "complete"
        receipt["completed_at_utc"] = now_utc()
    except Exception as exc:
        receipt["status"] = "catalogue-query-failed"
        receipt["error_type"] = type(exc).__name__
        receipt["error"] = str(exc)
        receipt["completed_at_utc"] = now_utc()
        _write_receipt(output_dir / "discovery_receipt.json", receipt)
        raise

    _write_receipt(output_dir / "discovery_receipt.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("docs/research/ST_LUCIA_SENTINEL_PILOT_V1.md"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)

    try:
        receipt = run_discovery(args.protocol, args.out, args.timeout)
    except Exception as exc:
        print(f"discovery failed: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": receipt["status"],
                "s2": receipt["s2"].get("selected_item_id"),
                "s1": receipt["s1"].get("selected_item_id"),
                "receipt": str(args.out / "discovery_receipt.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
