#!/usr/bin/env python3
"""Create a reproducible metadata-only inventory of Copernicus STAC scenes.

This utility inventories scene metadata. It does not download imagery, prove
that imagery is suitable for a leak study, or classify any location as a leak.
Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

try:
    from .strict_json import loads as strict_json_loads
except ImportError:  # Support direct execution as a script from the repository root.
    from strict_json import loads as strict_json_loads

DEFAULT_ENDPOINT = "https://stac.dataspace.copernicus.eu/v1/search"
ALLOWED_HOST = "stac.dataspace.copernicus.eu"
USER_AGENT = "LuminousDynamics-WaterLeak-CatalogInventory/0.1"
COLLECTIONS = ("sentinel-1-grd", "sentinel-2-l2a")
DEFAULT_BBOX = (27.7, -26.5, 28.4, -25.8)  # Reconnaissance only; not an approved study AOI.


class CatalogError(RuntimeError):
    """A catalog response or query could not be safely interpreted."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """Publish the complete JSON artifact atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{path.name}.",
            suffix=".tmp", dir=path.parent, delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def parse_bbox(value: str) -> list[float]:
    try:
        bbox = [float(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bbox must contain four comma-separated numbers") from exc
    if len(bbox) != 4:
        raise argparse.ArgumentTypeError("bbox must be west,south,east,north")
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180):
        raise argparse.ArgumentTypeError("bbox longitude bounds must satisfy -180 <= west < east <= 180")
    if not (-90 <= south < north <= 90):
        raise argparse.ArgumentTypeError("bbox latitude bounds must satisfy -90 <= south < north <= 90")
    return bbox


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def date_range(start: date, end: date) -> str:
    if end < start:
        raise CatalogError("end date must be on or after start date")
    begin = datetime.combine(start, time.min, tzinfo=timezone.utc)
    finish = datetime.combine(end, time.max.replace(microsecond=0), tzinfo=timezone.utc)
    return f"{begin.isoformat().replace('+00:00', 'Z')}/{finish.isoformat().replace('+00:00', 'Z')}"


def _redacted_catalog_uri(current_url: str, href: str) -> str | None:
    """Return a credential-free, query-free URI for provenance, or None if untrusted."""
    try:
        resolved = urljoin(current_url, href)
        parsed = urlparse(resolved)
        port = parsed.port
    except (TypeError, ValueError):
        return None
    if (
        parsed.scheme != "https"
        or parsed.hostname != ALLOWED_HOST
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
    ):
        return None
    # Query parameters can contain opaque pagination tokens or signed URLs.
    # Keep host/path for discovery, and bind the complete URI separately by SHA-256.
    return f"https://{ALLOWED_HOST}{parsed.path or '/'}"


def _validate_next_url(current_url: str, href: str) -> str:
    resolved = urljoin(current_url, href)
    if _redacted_catalog_uri(current_url, href) is None:
        raise CatalogError("catalog pagination returned a non-approved URL")
    return resolved


SAFE_PROPERTY_KEYS = (
    "datetime", "start_datetime", "end_datetime", "created", "updated",
    "platform", "constellation", "instruments", "mission", "eo:cloud_cover",
    "sar:instrument_mode", "sar:polarizations", "sat:relative_orbit",
    "sar:product_type", "sar:frequency_band", "sar:center_frequency",
    "processing:level", "s1:resolution", "s1:product_timeliness",
    "s2:mgrs_tile", "proj:epsg", "view:incidence_angle", "view:off_nadir",
    "sar:looks_range", "sar:looks_azimuth",
)
SAFE_ASSET_KEYS = ("type", "title", "roles", "gsd", "eo:bands", "raster:bands", "file:size")


def summarize_item_feature(
    feature: dict[str, Any], collection: str, current_url: str
) -> dict[str, Any]:
    """Keep research-relevant STAC metadata without signed asset URLs or raw links."""
    item_id = feature.get("id")
    if not isinstance(item_id, str) or not item_id:
        raise CatalogError("catalog feature is missing a non-empty string id")
    properties = feature.get("properties", {})
    if not isinstance(properties, dict):
        raise CatalogError(f"catalog item {item_id!r} has invalid properties")
    raw_links = feature.get("links", [])
    if not isinstance(raw_links, list):
        raise CatalogError(f"catalog item {item_id!r} has invalid links")
    self_uri = None
    for link in raw_links:
        if isinstance(link, dict) and link.get("rel") == "self" and isinstance(link.get("href"), str):
            self_uri = _redacted_catalog_uri(current_url, link["href"])
            if self_uri is not None:
                break

    raw_assets = feature.get("assets", {})
    if not isinstance(raw_assets, dict):
        raise CatalogError(f"catalog item {item_id!r} has invalid assets")
    asset_summaries: dict[str, dict[str, Any]] = {}
    for asset_name, asset in sorted(raw_assets.items()):
        if not isinstance(asset_name, str) or not isinstance(asset, dict):
            raise CatalogError(f"catalog item {item_id!r} has invalid asset entry")
        asset_summaries[asset_name] = {
            key: asset[key] for key in SAFE_ASSET_KEYS if key in asset
        }

    summary = {
        "collection": collection,
        "id": item_id,
        "self_uri_redacted": self_uri,
        "item_sha256": sha256_bytes(canonical_json_bytes(feature)),
        "bbox": feature.get("bbox"),
        "geometry": feature.get("geometry"),
        "properties": {key: properties[key] for key in SAFE_PROPERTY_KEYS if key in properties},
        # Hrefs intentionally omitted: remote catalogues may supply signed/query-bearing URLs.
        "asset_metadata_returned_in_response": bool(raw_assets),
        "assets": asset_summaries,
    }
    summary["summary_sha256"] = sha256_bytes(canonical_json_bytes(summary))
    return summary


def request_json(url: str, method: str, payload: dict[str, Any] | None, timeout: float) -> tuple[bytes, dict[str, Any]]:
    data = canonical_json_bytes(payload) if method.upper() == "POST" and payload is not None else None
    headers = {
        "Accept": "application/geo+json, application/json",
        "User-Agent": USER_AGENT,
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as exc:
        excerpt = exc.read(300).decode("utf-8", errors="replace")
        raise CatalogError(f"catalog HTTP {exc.code}; response excerpt: {excerpt}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise CatalogError(f"catalog request failed ({type(exc).__name__}): {exc}") from exc
    try:
        parsed = strict_json_loads(raw)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise CatalogError("catalog returned malformed, ambiguous, or non-standard JSON") from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("features"), list):
        raise CatalogError("catalog response did not contain a STAC FeatureCollection with a features array")
    return raw, parsed


def inventory_collection(
    *, endpoint: str, collection: str, bbox: list[float], interval: str,
    limit: int, max_pages: int, timeout: float,
) -> dict[str, Any]:
    parsed_endpoint = urlparse(endpoint)
    if parsed_endpoint.scheme != "https" or parsed_endpoint.hostname != ALLOWED_HOST:
        raise CatalogError(f"endpoint must be HTTPS on {ALLOWED_HOST}")

    query: dict[str, Any] = {
        "collections": [collection],
        "bbox": bbox,
        "datetime": interval,
        "limit": limit,
        # The first pass inventories scene metadata only. Exclude bulky image-asset
        # hrefs so this phase is smaller and cannot persist signed download URLs.
        "fields": {"exclude": ["assets"]},
    }
    current_url = endpoint
    method = "POST"
    payload: dict[str, Any] | None = query
    pages: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    duplicate_count = 0

    while True:
        if len(pages) >= max_pages:
            raise CatalogError(
                f"page limit ({max_pages}) reached while a next-page link may remain; result is incomplete"
            )
        raw, response = request_json(current_url, method, payload, timeout)
        feature_list = response["features"]
        page_number = len(pages) + 1
        pages.append({
            "page": page_number,
            "request_method": method,
            "request_url_redacted": _redacted_catalog_uri(current_url, current_url),
            "request_url_sha256": sha256_bytes(current_url.encode("utf-8")),
            "request_payload_sha256": sha256_bytes(canonical_json_bytes(payload)) if payload is not None else None,
            "response_sha256": sha256_bytes(raw),
            "feature_count": len(feature_list),
        })

        for feature in feature_list:
            if not isinstance(feature, dict):
                raise CatalogError("catalog feature was not a JSON object")
            item_id = feature.get("id")
            item_collection = feature.get("collection", collection)
            if not isinstance(item_id, str) or not item_id:
                raise CatalogError("catalog feature is missing a non-empty string id")
            if not isinstance(item_collection, str) or not item_collection:
                raise CatalogError(f"catalog item {item_id!r} has an invalid collection id")
            key = (item_collection, item_id)
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            items.append(summarize_item_feature(feature, item_collection, current_url))

        next_links = [link for link in response.get("links", []) if isinstance(link, dict) and link.get("rel") == "next"]
        if not next_links:
            break
        if len(next_links) != 1:
            raise CatalogError("catalog response had zero or multiple ambiguous next links")
        next_link = next_links[0]
        href = next_link.get("href")
        if not isinstance(href, str) or not href:
            raise CatalogError("catalog next link has no href")
        current_url = _validate_next_url(current_url, href)
        method = str(next_link.get("method", "GET")).upper()
        if method not in ("GET", "POST"):
            raise CatalogError(f"unsupported pagination method: {method}")
        if method == "POST":
            next_payload = next_link.get("body")
            if not isinstance(next_payload, dict):
                raise CatalogError("POST pagination link has no JSON body; refusing to guess query state")
            payload = next_payload
        else:
            payload = None

    return {
        "collection": collection,
        "query": query,
        "page_count": len(pages),
        "page_records": pages,
        "item_count": len(items),
        "duplicate_items_dropped": duplicate_count,
        "items": items,
        "catalog_discovery_only": True,
        "interpretation_limit": (
            "A catalog item or intersecting footprint is not evidence that the scene is fit for leak analysis; "
            "review acquisition geometry, radiometry, temporal alignment, ground labels and data rights."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", type=parse_bbox, default=list(DEFAULT_BBOX),
                        help="WGS84 bbox west,south,east,north (default is reconnaissance only, not an approved AOI)")
    parser.add_argument("--start", type=parse_date, required=True, help="inclusive UTC date YYYY-MM-DD")
    parser.add_argument("--end", type=parse_date, required=True, help="inclusive UTC date YYYY-MM-DD")
    parser.add_argument("--collection", choices=COLLECTIONS, action="append", dest="collections",
                        help="repeat to select collections; default is Sentinel-1 GRD and Sentinel-2 L2A")
    parser.add_argument("--limit", type=int, default=100, help="catalog page size (1–500; default 100)")
    parser.add_argument("--max-pages", type=int, default=100, help="fail closed if more pages are needed")
    parser.add_argument("--timeout", type=float, default=45.0, help="per-request timeout in seconds")
    parser.add_argument("--out", type=Path, required=True, help="output JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= 100:
        parser.error("--limit must be between 1 and the Copernicus STAC maximum of 100")
    if args.max_pages < 1:
        parser.error("--max-pages must be >= 1")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.end < args.start:
        parser.error("--end must be on or after --start")

    selected = args.collections or list(COLLECTIONS)
    interval = date_range(args.start, args.end)
    document = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "catalog_endpoint": DEFAULT_ENDPOINT,
        "scope": {
            "bbox_wgs84": args.bbox,
            "start_date_inclusive": args.start.isoformat(),
            "end_date_inclusive": args.end.isoformat(),
            "study_area_status": "reconnaissance_only_unless_separately_approved",
        },
        "collections": [],
        "overall_claim_ceiling": (
            "This report inventories catalog metadata only. It establishes no useful scene coverage, "
            "leak-detection performance, leak location, water saved, or commercial-use rights."
        ),
    }
    try:
        for collection in selected:
            document["collections"].append(inventory_collection(
                endpoint=DEFAULT_ENDPOINT,
                collection=collection,
                bbox=args.bbox,
                interval=interval,
                limit=args.limit,
                max_pages=args.max_pages,
                timeout=args.timeout,
            ))
        # Bind the complete published inventory, including compact summary hashes.
        document["inventory_manifest_sha256"] = sha256_bytes(canonical_json_bytes(document))
        output = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        atomic_write_text(args.out, output)
    except (CatalogError, OSError) as exc:
        print(f"catalog inventory failed closed: {exc}", file=sys.stderr)
        return 2

    counts = ", ".join(f"{entry['collection']}={entry['item_count']}" for entry in document["collections"])
    print(f"wrote {args.out} ({counts}); metadata inventory only, not leak evidence")
    print(f"inventory_manifest_sha256={document['inventory_manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
