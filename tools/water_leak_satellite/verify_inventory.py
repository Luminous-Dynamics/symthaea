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
import re
import sys
from pathlib import Path
from typing import Any

SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _contains_href_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(str(key).lower() == "href" or _contains_href_key(nested) for key, nested in value.items())
    if isinstance(value, list):
        return any(_contains_href_key(item) for item in value)
    return False


def verify_inventory(document: Any) -> list[str]:
    errors: list[str] = []
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
    if not isinstance(document.get("scope"), dict):
        errors.append("scope: must be an object")

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
        pages = collection.get("page_records")
        if not isinstance(pages, list):
            errors.append(f"{prefix}.page_records: must be a list")
            pages = []
        if type(collection.get("page_count")) is not int or collection.get("page_count") != len(pages):
            errors.append(f"{prefix}.page_count: does not match page_records length")

        for page_index, page in enumerate(pages):
            pfx = f"{prefix}.page_records[{page_index}]"
            if not isinstance(page, dict):
                errors.append(f"{pfx}: must be an object")
                continue
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

        for item_index, item in enumerate(items):
            pfx = f"{prefix}.items[{item_index}]"
            if not isinstance(item, dict):
                errors.append(f"{pfx}: must be an object")
                continue
            item_collection, item_id = item.get("collection"), item.get("id")
            if not isinstance(item_collection, str) or not item_collection or not isinstance(item_id, str) or not item_id:
                errors.append(f"{pfx}: missing item collection or ID")
            else:
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
            if _contains_href_key(item):
                errors.append(f"{pfx}: contains an href field; signed/download URLs must not be stored")
            self_uri = item.get("self_uri_redacted")
            if self_uri is not None and (not isinstance(self_uri, str) or "?" in self_uri or "#" in self_uri):
                errors.append(f"{pfx}.self_uri_redacted: must be a query-free URI")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path, help="saved catalogue inventory JSON")
    args = parser.parse_args(argv)
    try:
        document = json.loads(args.inventory.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"FAIL: cannot read valid UTF-8 JSON inventory: {exc}", file=sys.stderr)
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
