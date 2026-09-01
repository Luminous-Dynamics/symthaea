#!/usr/bin/env python3
"""Build a zero-download, content-addressed R0 Sentinel asset acquisition plan.

The planner consumes only the frozen St. Lucia discovery/replay evidence and emits the
exact scientific/provenance asset hrefs that may be acquired next. It performs no
network access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

Json = dict[str, Any]

SCHEMA = "symthaea-st-lucia-r0-asset-plan/v1"
TOOL_VERSION = "1.1.0"
APPROVED_STAC_HOST = "stac.dataspace.copernicus.eu"

EXPECTED_DISCOVERY_FILE_SHA256 = (
    "bd1c91e4cb92bb6fe51c0b2ec819d7b5a87530b307dbdb4e35631f92f750efe0"
)
EXPECTED_REPLAY_FILE_SHA256 = (
    "50d25f28b8dd2b1c8787b763d0fdb7ddc668a3a6ab3ae50e5ab79ab0934369bc"
)
EXPECTED_DISCOVERY_INTERNAL_SHA256 = (
    "16552ad878560bd8df8242c5b4a3966ca3829f4daafea5e6738e8de8b3b60e85"
)
EXPECTED_REPLAY_INTERNAL_SHA256 = (
    "79ce4a14e95e0e7894c8c0684f7e4c6e4344e5bddb804b8381a9476a8de5c29a"
)
EXPECTED_S2_SNAPSHOT_SHA256 = (
    "41fec8ace5d07c0ab32c71ab352a5dfe2c77b750ee56e96aeed6afc3b1f28f25"
)
EXPECTED_S1_SNAPSHOT_SHA256 = (
    "3f29e3309527ff469ca1fcaa2cfeeec1c2574af2cbe5c6886285a7c8d98cefc7"
)

EXPECTED_S2_IDS = [
    "S2C_MSIL2A_20260701T073611_N0512_R092_T36JVP_20260701T122756",
    "S2C_MSIL2A_20260701T073611_N0512_R092_T36JVQ_20260701T122756",
]
EXPECTED_S1_ID = (
    "S1C_IW_GRDH_1SDV_20260630T031023_20260630T031048_008329_0107B6_765A_COG"
)

S2_SCIENCE_ASSETS = ["B03_10m", "B04_10m", "B08_10m", "B11_20m", "B12_20m", "SCL_20m"]
S2_METADATA_ASSETS = ["safe_manifest", "product_metadata", "granule_metadata", "datastrip_metadata"]
S1_SCIENCE_ASSETS = ["vv", "vh"]
S1_METADATA_ASSETS = [
    "safe_manifest",
    "schema-calibration-vv",
    "schema-calibration-vh",
    "schema-noise-vv",
    "schema-noise-vh",
    "schema-product-vv",
    "schema-product-vh",
]

FORBIDDEN_KEYS = {"thumbnail", "Product"}


class PlanError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def load_json(path: Path) -> Json:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PlanError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PlanError(f"expected JSON object: {path}")
    return value


def require_exact_file_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise PlanError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def verify_page_sidecars(directory: Path) -> list[Path]:
    pages = sorted(directory.glob("page-*.json"))
    if not pages:
        raise PlanError(f"no retained STAC pages found in {directory}")
    for page in pages:
        sidecar = page.with_suffix(".sha256")
        if not sidecar.is_file():
            raise PlanError(f"missing SHA-256 sidecar for {page}")
        tokens = sidecar.read_text(encoding="utf-8").strip().split()
        if not tokens:
            raise PlanError(f"empty SHA-256 sidecar: {sidecar}")
        expected = tokens[0].lower()
        if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
            raise PlanError(f"invalid SHA-256 sidecar value: {sidecar}")
        actual = sha256_file(page)
        if actual != expected:
            raise PlanError(f"retained STAC page hash mismatch: {page}")
    return pages


def index_items(page_paths: list[Path]) -> dict[str, Json]:
    result: dict[str, Json] = {}
    canonical: dict[str, bytes] = {}
    for path in page_paths:
        page = load_json(path)
        features = page.get("features")
        if not isinstance(features, list):
            raise PlanError(f"STAC page lacks feature list: {path}")
        for item in features:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                raise PlanError(f"invalid STAC feature in {path}")
            item_id = item["id"]
            encoded = canonical_json_bytes(item)
            previous = canonical.get(item_id)
            if previous is not None and previous != encoded:
                raise PlanError(f"duplicate selected item with conflicting metadata: {item_id}")
            canonical[item_id] = encoded
            result[item_id] = item
    return result


def checked_approved_https(url: str, label: str) -> str:
    try:
        parsed = urlparse(url)
        port = parsed.port
    except ValueError as exc:
        raise PlanError(f"{label} has invalid URL syntax") from exc

    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise PlanError(f"{label} is not absolute HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise PlanError(f"{label} must not contain userinfo")
    if parsed.hostname.lower().rstrip(".") != APPROVED_STAC_HOST:
        raise PlanError(f"{label} is off approved CDSE STAC origin")
    if port not in (None, 443):
        raise PlanError(f"{label} uses a non-HTTPS-default port")
    if parsed.fragment:
        raise PlanError(f"{label} must not contain a URL fragment")
    return url


def item_self_href(item: Json) -> str:
    item_id = item.get("id")
    links = item.get("links")
    if not isinstance(links, list):
        raise PlanError(f"item {item_id} has no STAC links array for relative href resolution")

    self_hrefs: list[str] = []
    for link in links:
        if not isinstance(link, dict) or link.get("rel") != "self":
            continue
        href = link.get("href")
        if not isinstance(href, str) or not href:
            raise PlanError(f"item {item_id} has an invalid self link")
        if href not in self_hrefs:
            self_hrefs.append(href)

    if len(self_hrefs) != 1:
        raise PlanError(f"item {item_id} must have exactly one unique self link for relative href resolution")

    return checked_approved_https(self_hrefs[0], f"item {item_id} self href")


def resolve_asset_href(item: Json, asset_key: str, asset: Any) -> tuple[str, str, str | None]:
    item_id = item.get("id")
    if not isinstance(asset, dict):
        raise PlanError(f"asset {item_id}/{asset_key} is not an object")
    raw_href = asset.get("href")
    if not isinstance(raw_href, str) or not raw_href:
        raise PlanError(f"asset {item_id}/{asset_key} has no href")

    parsed = urlparse(raw_href)
    if parsed.scheme or parsed.netloc:
        resolved = checked_approved_https(raw_href, f"asset {item_id}/{asset_key} href")
        return raw_href, resolved, None

    base = item_self_href(item)
    resolved = urljoin(base, raw_href)
    checked_approved_https(resolved, f"asset {item_id}/{asset_key} resolved href")
    return raw_href, resolved, base


def asset_entry(item: Json, asset_key: str, purpose: str) -> Json:
    item_id = item["id"]
    if asset_key in FORBIDDEN_KEYS:
        raise PlanError(f"forbidden asset key requested: {asset_key}")
    assets = item.get("assets")
    if not isinstance(assets, dict) or asset_key not in assets:
        raise PlanError(f"required asset missing: {item_id}/{asset_key}")
    asset = assets[asset_key]
    raw_href, resolved_href, resolution_base = resolve_asset_href(item, asset_key, asset)
    return {
        "collection": item.get("collection"),
        "item_id": item_id,
        "item_sha256": sha256_bytes(canonical_json_bytes(item)),
        "asset_key": asset_key,
        "purpose": purpose,
        "stac_href": raw_href,
        "href": resolved_href,
        "href_resolution_base": resolution_base,
        "media_type": asset.get("type") if isinstance(asset, dict) else None,
        "roles": asset.get("roles") if isinstance(asset, dict) else None,
        "title": asset.get("title") if isinstance(asset, dict) else None,
    }


def build_plan(original_dir: Path, replay_path: Path) -> Json:
    discovery_path = original_dir / "discovery_receipt.json"
    require_exact_file_hash(discovery_path, EXPECTED_DISCOVERY_FILE_SHA256, "original discovery receipt")
    require_exact_file_hash(replay_path, EXPECTED_REPLAY_FILE_SHA256, "acquisition-set receipt")

    discovery = load_json(discovery_path)
    replay = load_json(replay_path)

    if discovery.get("receipt_sha256") != EXPECTED_DISCOVERY_INTERNAL_SHA256:
        raise PlanError("original discovery internal receipt digest mismatch")
    if replay.get("receipt_sha256") != EXPECTED_REPLAY_INTERNAL_SHA256:
        raise PlanError("acquisition-set internal receipt digest mismatch")
    if discovery.get("s2", {}).get("snapshot_sha256") != EXPECTED_S2_SNAPSHOT_SHA256:
        raise PlanError("S2 catalogue snapshot mismatch")
    if discovery.get("s1", {}).get("snapshot_sha256") != EXPECTED_S1_SNAPSHOT_SHA256:
        raise PlanError("S1 catalogue snapshot mismatch")

    s2_ids = replay.get("selected_s2_acquisition_item_ids")
    s1_id = replay.get("selected_s1_item_id")
    if s2_ids != EXPECTED_S2_IDS:
        raise PlanError(f"unexpected S2 acquisition support set: {s2_ids!r}")
    if s1_id != EXPECTED_S1_ID:
        raise PlanError(f"unexpected S1 pair: {s1_id!r}")
    if replay.get("network_access") != "forbidden-and-unused":
        raise PlanError("replay receipt does not attest network-forbidden execution")

    s2_pages = verify_page_sidecars(original_dir / "s2")
    s1_pages = verify_page_sidecars(original_dir / "s1")
    s2_items = index_items(s2_pages)
    s1_items = index_items(s1_pages)

    entries: list[Json] = []
    for item_id in EXPECTED_S2_IDS:
        item = s2_items.get(item_id)
        if item is None:
            raise PlanError(f"selected S2 item absent from frozen pages: {item_id}")
        for key in S2_SCIENCE_ASSETS:
            entries.append(asset_entry(item, key, "science-payload"))
        for key in S2_METADATA_ASSETS:
            entries.append(asset_entry(item, key, "provenance-metadata"))

    item = s1_items.get(EXPECTED_S1_ID)
    if item is None:
        raise PlanError(f"selected S1 item absent from frozen pages: {EXPECTED_S1_ID}")
    for key in S1_SCIENCE_ASSETS:
        entries.append(asset_entry(item, key, "science-payload"))
    for key in S1_METADATA_ASSETS:
        entries.append(asset_entry(item, key, "calibration-provenance"))

    entries.sort(key=lambda row: (str(row["collection"]), row["item_id"], row["asset_key"]))

    plan: Json = {
        "schema": SCHEMA,
        "tool_version": TOOL_VERSION,
        "stage": "r0-pre-download-asset-plan",
        "network_access": "forbidden-and-unused",
        "download_permitted_by_this_receipt": False,
        "claim_boundary": "exact asset selection and href freezing only; no asset bytes fetched",
        "approved_stac_host": APPROVED_STAC_HOST,
        "original_discovery_receipt_file_sha256": EXPECTED_DISCOVERY_FILE_SHA256,
        "original_discovery_receipt_internal_sha256": EXPECTED_DISCOVERY_INTERNAL_SHA256,
        "acquisition_set_receipt_file_sha256": EXPECTED_REPLAY_FILE_SHA256,
        "acquisition_set_receipt_internal_sha256": EXPECTED_REPLAY_INTERNAL_SHA256,
        "s2_catalogue_snapshot_sha256": EXPECTED_S2_SNAPSHOT_SHA256,
        "s1_catalogue_snapshot_sha256": EXPECTED_S1_SNAPSHOT_SHA256,
        "selected_s2_item_ids": EXPECTED_S2_IDS,
        "selected_s1_item_id": EXPECTED_S1_ID,
        "assets": entries,
    }
    plan["plan_sha256"] = sha256_bytes(canonical_json_bytes(plan))
    return plan


def write_plan(plan: Json, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-discovery", type=Path, required=True)
    parser.add_argument("--acquisition-set-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        plan = build_plan(args.original_discovery, args.acquisition_set_receipt)
        write_plan(plan, args.out)
    except Exception as exc:
        print(f"asset plan failed: {exc}", file=__import__("sys").stderr)
        return 2

    print(
        json.dumps(
            {
                "status": "complete",
                "assets": len(plan["assets"]),
                "plan_sha256": plan["plan_sha256"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
