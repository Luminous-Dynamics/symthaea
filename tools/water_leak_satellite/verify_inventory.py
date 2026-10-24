#!/usr/bin/env python3
"""Verify consistency/integrity hashes in a saved water-leak scene inventory.

A PASS verifies only that stored JSON matches its own manifest/summary hashes
and declared redaction/count invariants. It does not authenticate the provider,
prove the catalog response was genuine, or establish scene/leak truth.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

try:
    from .strict_json import loads as strict_json_loads
except ImportError:  # Support direct execution as a script from the repository root.
    from strict_json import loads as strict_json_loads

SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_json(value: Any) -> str:
    try:
        encoded = canonical_json_bytes(value)
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError):
        # Invalid or pathologically nested JSON-like values must fail verification,
        # rather than terminating the verifier with an exception.
        return ""
    return hashlib.sha256(encoded).hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _valid_iso_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def _finite_number(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _contains_href_key(value: Any) -> bool:
    # Iterative traversal avoids recursion-depth failure on malformed artifacts.
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


def verify_inventory(document: Any) -> list[str]:
    errors: list[str] = []
    # Enforce the no-href rule across the entire saved artifact, including
    # top-level extension fields and pagination metadata—not only item summaries.
    if _contains_href_key(document):
        errors.append("inventory: contains an href field; signed/download URLs must not be stored")
    if not isinstance(document, dict):
        return ["inventory: root must be an object"]

    recorded_manifest = document.get("inventory_manifest_sha256")
    if not is_sha256(recorded_manifest):
        errors.append("inventory_manifest_sha256: missing or invalid SHA-256")
    else:
        unsigned_document = dict(document)
        unsigned_document.pop("inventory_manifest_sha256", None)
        if sha256_json(unsigned_document) != recorded_manifest:
            errors.append("inventory_manifest_sha256: digest does not match document")

    schema_version = document.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        errors.append("schema_version: expected integer 1")
    scope = document.get("scope")
    if not isinstance(scope, dict):
        errors.append("scope: must be an object")
    else:
        bbox = scope.get("bbox_wgs84")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(not _finite_number(value) for value in bbox)
        ):
            errors.append("scope.bbox_wgs84: must contain four finite numeric bounds")
        elif not (-180 <= bbox[0] < bbox[2] <= 180 and -90 <= bbox[1] < bbox[3] <= 90):
            errors.append("scope.bbox_wgs84: bounds must be ordered WGS84 west,south,east,north")

        start_date = scope.get("start_date_inclusive")
        end_date = scope.get("end_date_inclusive")
        if not _valid_iso_date(start_date) or not _valid_iso_date(end_date):
            errors.append("scope: start/end dates must be canonical YYYY-MM-DD dates")
        elif end_date < start_date:
            errors.append("scope: end_date_inclusive precedes start_date_inclusive")

    collections = document.get("collections")
    if not isinstance(collections, list):
        errors.append("collections: must be a list")
        return errors

    seen_items: set[tuple[str, str]] = set()
    for collection_index, collection in enumerate(collections):
        prefix = f"collections[{collection_index}]"
        if not isinstance(collection, dict):
            errors.append(f"{prefix}: must be an object")
            continue
        name = collection.get("collection")
        if not isinstance(name, str) or not name:
            errors.append(f"{prefix}.collection: missing collection identifier")

        query = collection.get("query")
        if not isinstance(query, dict):
            errors.append(f"{prefix}.query: must preserve the catalogue query")
        else:
            if not isinstance(name, str) or query.get("collections") != [name]:
                errors.append(f"{prefix}.query.collections: must match the inventoried collection")
            if not isinstance(scope, dict) or query.get("bbox") != scope.get("bbox_wgs84"):
                errors.append(f"{prefix}.query.bbox: does not match declared inventory scope")
            if isinstance(scope, dict):
                start_date = scope.get("start_date_inclusive")
                end_date = scope.get("end_date_inclusive")
                if _valid_iso_date(start_date) and _valid_iso_date(end_date) and end_date >= start_date:
                    expected_interval = f"{start_date}T00:00:00Z/{end_date}T23:59:59Z"
                    if query.get("datetime") != expected_interval:
                        errors.append(f"{prefix}.query.datetime: does not match declared scope dates")
            limit = query.get("limit")
            if type(limit) is not int or not 1 <= limit <= 100:
                errors.append(f"{prefix}.query.limit: must be an integer from 1 to 100")

        pages = collection.get("page_records")
        if not isinstance(pages, list):
            errors.append(f"{prefix}.page_records: must be a list")
            pages = []
        if type(collection.get("page_count")) is not int or collection.get("page_count") != len(pages):
            errors.append(f"{prefix}.page_count: does not match page_records length")

        page_feature_total = 0
        page_feature_counts_valid = True
        for page_index, page in enumerate(pages):
            pfx = f"{prefix}.page_records[{page_index}]"
            if not isinstance(page, dict):
                errors.append(f"{pfx}: must be an object")
                page_feature_counts_valid = False
                continue
            if type(page.get("page")) is not int or page.get("page") != page_index + 1:
                errors.append(f"{pfx}.page: page numbers must be consecutive starting at 1")
            if page.get("request_method") not in ("GET", "POST"):
                errors.append(f"{pfx}.request_method: must be GET or POST")
            feature_count = page.get("feature_count")
            if type(feature_count) is not int or feature_count < 0:
                errors.append(f"{pfx}.feature_count: must be a non-negative integer")
                page_feature_counts_valid = False
            else:
                page_feature_total += feature_count
            for digest_name in ("response_sha256", "request_url_sha256"):
                if not is_sha256(page.get(digest_name)):
                    errors.append(f"{pfx}.{digest_name}: missing or invalid SHA-256")
            payload_digest = page.get("request_payload_sha256")
            if payload_digest is not None and not is_sha256(payload_digest):
                errors.append(f"{pfx}.request_payload_sha256: invalid SHA-256")
            redacted_url = page.get("request_url_redacted")
            if not isinstance(redacted_url, str) or "?" in redacted_url or "#" in redacted_url:
                errors.append(f"{pfx}.request_url_redacted: must be a query-free URL")

        items = collection.get("items")
        if not isinstance(items, list):
            errors.append(f"{prefix}.items: must be a list")
            items = []
        if type(collection.get("item_count")) is not int or collection.get("item_count") != len(items):
            errors.append(f"{prefix}.item_count: does not match items length")
        duplicate_count = collection.get("duplicate_items_dropped")
        if type(duplicate_count) is not int or duplicate_count < 0:
            errors.append(f"{prefix}.duplicate_items_dropped: must be a non-negative integer")
        elif page_feature_counts_valid and type(collection.get("item_count")) is int:
            if page_feature_total != collection["item_count"] + duplicate_count:
                errors.append(
                    f"{prefix}: page feature total must equal unique item_count plus duplicate_items_dropped"
                )

        for item_index, item in enumerate(items):
            pfx = f"{prefix}.items[{item_index}]"
            if not isinstance(item, dict):
                errors.append(f"{pfx}: must be an object")
                continue
            item_collection, item_id = item.get("collection"), item.get("id")
            if not isinstance(item_collection, str) or not item_collection or not isinstance(item_id, str) or not item_id:
                errors.append(f"{pfx}: missing item collection or ID")
            else:
                if isinstance(name, str) and item_collection != name:
                    errors.append(f"{pfx}.collection: item collection does not match parent collection")
                pair = (item_collection, item_id)
                if pair in seen_items:
                    errors.append(f"{pfx}: duplicate collection/ID pair")
                seen_items.add(pair)

            if not is_sha256(item.get("item_sha256")):
                errors.append(f"{pfx}.item_sha256: missing or invalid source-representation digest")
            summary_digest = item.get("summary_sha256")
            if not is_sha256(summary_digest):
                errors.append(f"{pfx}.summary_sha256: missing or invalid SHA-256")
            else:
                unsigned_item = dict(item)
                unsigned_item.pop("summary_sha256", None)
                if sha256_json(unsigned_item) != summary_digest:
                    errors.append(f"{pfx}.summary_sha256: digest does not match summary")
            self_uri = item.get("self_uri_redacted")
            if self_uri is not None and (not isinstance(self_uri, str) or "?" in self_uri or "#" in self_uri):
                errors.append(f"{pfx}.self_uri_redacted: must be a query-free URI")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path, help="saved catalogue inventory JSON")
    args = parser.parse_args(argv)
    try:
        document = strict_json_loads(args.inventory.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, RecursionError) as exc:
        print(f"FAIL: cannot read valid strict UTF-8 JSON inventory: {exc}", file=sys.stderr)
        return 2
    errors = verify_inventory(document)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print("FAIL: inventory integrity verification failed", file=sys.stderr)
        return 1
    print("PASS: manifest, compact summary hashes, page digest shapes and redaction invariants verified.")
    print("This does not authenticate the data source or establish scene suitability/leak truth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
