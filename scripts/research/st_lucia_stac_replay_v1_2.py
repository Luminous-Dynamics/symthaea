#!/usr/bin/env python3
"""Replay the St. Lucia catalogue selection against the frozen v1.1 pages.

This tool performs no network access. It verifies the exact original discovery
receipt and raw-page hashes, then applies the post-discovery/pre-asset
acquisition-set amendment without mutating the original evidence directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import st_lucia_stac_discovery as v1

SCHEMA = "symthaea-st-lucia-stac-acquisition-set-amendment/v1"
TOOL_VERSION = "1.2.0"
ORIGINAL_HEAD = "71ef19e6a35e02b631bfd3ba69b5e781decb70c0"
EXPECTED_PROTOCOL_SHA256 = "55d16c7d29b03030b2e53ad93b3e679035ce13169c9fdf1c40ac86549dbebd41"
EXPECTED_ORIGINAL_RECEIPT_INTERNAL_SHA256 = "16552ad878560bd8df8242c5b4a3966ca3829f4daafea5e6738e8de8b3b60e85"
EXPECTED_ORIGINAL_RECEIPT_FILE_SHA256 = "bd1c91e4cb92bb6fe51c0b2ec819d7b5a87530b307dbdb4e35631f92f750efe0"
EXPECTED_S2_SNAPSHOT_SHA256 = "41fec8ace5d07c0ab32c71ab352a5dfe2c77b750ee56e96aeed6afc3b1f28f25"
EXPECTED_S2_PAGE_SHA256 = "c233f9e85450705f7173f5af16eee226189df8263f6af71a8000f7f773a39fd0"
EXPECTED_S1_SNAPSHOT_SHA256 = "3f29e3309527ff469ca1fcaa2cfeeec1c2574af2cbe5c6886285a7c8d98cefc7"
EXPECTED_S1_PAGE_SHA256 = "ad7932be016575c61a58dea736d45a9331a8b93a3aed39ea381bb0dacae04aac"
EXPECTED_ORIGINAL_S2_ITEM_ID = "S2C_MSIL2A_20260701T073611_N0512_R092_T36JVP_20260701T122756"
EXPECTED_S2_ACQUISITION_SET = (
    "S2C_MSIL2A_20260701T073611_N0512_R092_T36JVP_20260701T122756",
    "S2C_MSIL2A_20260701T073611_N0512_R092_T36JVQ_20260701T122756",
)
EXPECTED_S1_ITEM_ID = "S1C_IW_GRDH_1SDV_20260630T031023_20260630T031048_008329_0107B6_765A_COG"

Json = dict[str, Any]


class ReplayError(RuntimeError):
    pass


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def verify_internal_receipt(receipt: Json) -> None:
    claimed = receipt.get("receipt_sha256")
    if claimed != EXPECTED_ORIGINAL_RECEIPT_INTERNAL_SHA256:
        raise ReplayError("unexpected original receipt internal digest")
    body = dict(receipt)
    body.pop("receipt_sha256", None)
    actual = sha256_hex(canonical_json_bytes(body))
    if actual != claimed:
        raise ReplayError("original receipt internal digest mismatch")


def _sidecar_digest(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    token = text.split(maxsplit=1)[0] if text else ""
    if len(token) != 64 or any(ch not in "0123456789abcdef" for ch in token):
        raise ReplayError(f"invalid SHA-256 sidecar: {path}")
    return token


def verify_and_load_pages(root: Path, modality: str, section: Json) -> list[Json]:
    rows = section.get("pages")
    if not isinstance(rows, list) or not rows:
        raise ReplayError(f"{modality}: original receipt has no page evidence")

    modality_dir = root / modality
    expected_json_names = [str(row.get("path")) for row in rows]
    actual_json_names = sorted(path.name for path in modality_dir.glob("page-*.json"))
    if sorted(expected_json_names) != actual_json_names:
        raise ReplayError(f"{modality}: retained page set differs from original receipt")

    expected_sidecars = sorted(name.replace(".json", ".sha256") for name in expected_json_names)
    actual_sidecars = sorted(path.name for path in modality_dir.glob("page-*.sha256"))
    if expected_sidecars != actual_sidecars:
        raise ReplayError(f"{modality}: retained SHA-256 sidecar set differs from original receipt")

    pages: list[Json] = []
    for row in rows:
        name = str(row["path"])
        page_path = modality_dir / name
        raw = page_path.read_bytes()
        actual = sha256_hex(raw)
        claimed = str(row.get("sha256", ""))
        if actual != claimed:
            raise ReplayError(f"{modality}: raw page digest mismatch for {name}")
        sidecar = _sidecar_digest(page_path.with_suffix(".sha256"))
        if sidecar != actual:
            raise ReplayError(f"{modality}: SHA-256 sidecar mismatch for {name}")
        if len(raw) != row.get("byte_len"):
            raise ReplayError(f"{modality}: byte length mismatch for {name}")
        try:
            page = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ReplayError(f"{modality}: invalid retained JSON for {name}") from exc
        if not isinstance(page, dict) or not isinstance(page.get("features", []), list):
            raise ReplayError(f"{modality}: retained page is not a STAC FeatureCollection")
        pages.append(page)

    if v1.snapshot_digest(rows) != section.get("snapshot_sha256"):
        raise ReplayError(f"{modality}: snapshot digest does not reproduce")
    return pages


def _item_bbox(item: Json) -> tuple[float, float, float, float] | None:
    raw = item.get("bbox")
    if not isinstance(raw, list) or len(raw) not in (4, 6):
        return None
    try:
        values = [float(value) for value in raw]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    if len(values) == 4:
        west, south, east, north = values
    else:
        west, south, _min_z, east, north, _max_z = values
    if west > east or south > north:
        return None
    return west, south, east, north


def bbox_intersects_frozen_aoi(item: Json) -> bool:
    bbox = _item_bbox(item)
    if bbox is None:
        return False
    west, south, east, north = bbox
    aoi_west, aoi_south, aoi_east, aoi_north = v1.BBOX
    return not (
        east < aoi_west
        or west > aoi_east
        or north < aoi_south
        or south > aoi_north
    )


def amended_s2_eligibility(item: Json) -> list[str]:
    reasons = list(v1.s2_eligibility(item))
    if _item_bbox(item) is None:
        reasons.append("missing-or-invalid-bbox")
    elif not bbox_intersects_frozen_aoi(item):
        reasons.append("bbox-does-not-intersect-frozen-aoi")
    return reasons


def select_s2_acquisition_set(items: list[Json]) -> tuple[list[Json], list[Json], list[str]]:
    audited: list[tuple[Json, list[str]]] = [
        (item, amended_s2_eligibility(item)) for item in items
    ]
    eligible = [item for item, reasons in audited if not reasons]
    eligible.sort(key=lambda item: (v1.parse_time(item["properties"]["datetime"]), item["id"]))
    if not eligible:
        selected: list[Json] = []
    else:
        earliest = v1.parse_time(eligible[0]["properties"]["datetime"])
        selected = [
            item
            for item in eligible
            if v1.parse_time(item["properties"]["datetime"]) == earliest
        ]
        selected.sort(key=lambda item: item["id"])
    summaries = [v1.summarize_item(item, reasons) for item, reasons in audited]
    summaries.sort(key=lambda row: row["id"])
    return selected, summaries, [item["id"] for item in eligible]


def write_receipt(path: Path, receipt: Json) -> None:
    body = dict(receipt)
    body.pop("receipt_sha256", None)
    body["receipt_sha256"] = sha256_hex(canonical_json_bytes(body))
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def replay(original_dir: Path, out_path: Path) -> Json:
    original_receipt_path = original_dir / "discovery_receipt.json"
    original_raw = original_receipt_path.read_bytes()
    if sha256_hex(original_raw) != EXPECTED_ORIGINAL_RECEIPT_FILE_SHA256:
        raise ReplayError("original discovery receipt file digest mismatch")
    original = json.loads(original_raw)
    if not isinstance(original, dict):
        raise ReplayError("original discovery receipt is not an object")
    verify_internal_receipt(original)

    if original.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise ReplayError("unexpected protocol digest")
    if original.get("schema") != v1.SCHEMA or original.get("tool_version") != "1.1.0":
        raise ReplayError("unexpected original discovery schema/tool version")
    if original.get("status") != "complete":
        raise ReplayError("original discovery was not complete")

    s2_section = original.get("s2") or {}
    s1_section = original.get("s1") or {}
    if s2_section.get("snapshot_sha256") != EXPECTED_S2_SNAPSHOT_SHA256:
        raise ReplayError("unexpected S2 catalogue snapshot")
    if s1_section.get("snapshot_sha256") != EXPECTED_S1_SNAPSHOT_SHA256:
        raise ReplayError("unexpected S1 catalogue snapshot")
    if s2_section.get("selected_item_id") != EXPECTED_ORIGINAL_S2_ITEM_ID:
        raise ReplayError("unexpected original S2 item selection")
    if s1_section.get("selected_item_id") != EXPECTED_S1_ITEM_ID:
        raise ReplayError("unexpected original S1 item selection")

    s2_pages = verify_and_load_pages(original_dir, "s2", s2_section)
    s1_pages = verify_and_load_pages(original_dir, "s1", s1_section)
    if len(s2_section["pages"]) != 1 or s2_section["pages"][0]["sha256"] != EXPECTED_S2_PAGE_SHA256:
        raise ReplayError("unexpected frozen S2 raw-page lineage")
    if len(s1_section["pages"]) != 1 or s1_section["pages"][0]["sha256"] != EXPECTED_S1_PAGE_SHA256:
        raise ReplayError("unexpected frozen S1 raw-page lineage")

    s2_items = v1.deduplicated_items(s2_pages)
    selected_s2, s2_audit, eligible_s2_ids = select_s2_acquisition_set(s2_items)
    selected_s2_ids = [item["id"] for item in selected_s2]
    if tuple(selected_s2_ids) != EXPECTED_S2_ACQUISITION_SET:
        raise ReplayError("amended S2 acquisition support differs from frozen expected set")

    s2_time = v1.parse_time(selected_s2[0]["properties"]["datetime"])
    if any(v1.parse_time(item["properties"]["datetime"]) != s2_time for item in selected_s2):
        raise ReplayError("selected S2 support is not one acquisition instant")

    s1_items = v1.deduplicated_items(s1_pages)
    selected_s1, s1_audit, eligible_s1_ids = v1.select_s1(s1_items, s2_time)
    if selected_s1 is None or selected_s1["id"] != EXPECTED_S1_ITEM_ID:
        raise ReplayError("S1 deterministic pairing did not reproduce")

    receipt: Json = {
        "schema": SCHEMA,
        "tool_version": TOOL_VERSION,
        "amendment_stage": "post-discovery-pre-asset",
        "network_access": "forbidden-and-unused",
        "original_source_head": ORIGINAL_HEAD,
        "original_protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "original_discovery_receipt_internal_sha256": EXPECTED_ORIGINAL_RECEIPT_INTERNAL_SHA256,
        "original_discovery_receipt_file_sha256": EXPECTED_ORIGINAL_RECEIPT_FILE_SHA256,
        "s2_catalogue_snapshot_sha256": EXPECTED_S2_SNAPSHOT_SHA256,
        "s1_catalogue_snapshot_sha256": EXPECTED_S1_SNAPSHOT_SHA256,
        "original_s2_selected_item_id": EXPECTED_ORIGINAL_S2_ITEM_ID,
        "selected_s2_acquisition_datetime": v1.iso_z(s2_time),
        "selected_s2_acquisition_item_ids": selected_s2_ids,
        "eligible_s2_candidate_ids_in_original_order": eligible_s2_ids,
        "s2_candidate_audit": s2_audit,
        "selected_s1_item_id": selected_s1["id"],
        "eligible_s1_candidate_ids_in_selection_order": eligible_s1_ids,
        "s1_candidate_audit": s1_audit,
        "status": "complete",
        "claim_boundary": "source-support selection only; no asset bytes or imagery inspected",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_receipt(out_path, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-discovery", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        receipt = replay(args.original_discovery, args.out)
    except Exception as exc:
        print(f"replay failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": receipt["status"],
        "s2_acquisition_items": receipt["selected_s2_acquisition_item_ids"],
        "s1": receipt["selected_s1_item_id"],
        "receipt": str(args.out),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
