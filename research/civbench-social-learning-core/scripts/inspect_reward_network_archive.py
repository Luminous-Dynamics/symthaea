#!/usr/bin/env python3
"""Inspect Reward Network III Zenodo archive without exporting raw data rows.

This is a research-only ingestion helper. It inventories names, sizes, CSV
headers, small JSON key structures and README headings. It does not copy raw
participant/move records into the output report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import PurePosixPath
from typing import Any

EXPECTED_MD5 = "55fbf9a0ffbc2f605da1792226471711"
MAX_JSON_INSPECT_BYTES = 5 * 1024 * 1024
MAX_HEADERS = 256


def digest_file(path: str) -> dict[str, Any]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    size = 0
    with open(path, "rb") as source:
        while chunk := source.read(1024 * 1024):
            size += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
    return {
        "size_bytes": size,
        "md5": md5.hexdigest(),
        "sha256": sha256.hexdigest(),
    }


def safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def csv_header(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> list[str] | None:
    try:
        with archive.open(info, "r") as stream:
            text = io.TextIOWrapper(stream, encoding="utf-8-sig", errors="replace", newline="")
            delimiter = "\t" if info.filename.lower().endswith(".tsv") else ","
            row = next(csv.reader(text, delimiter=delimiter), None)
            if row is None:
                return []
            return [column.strip() for column in row[:MAX_HEADERS]]
    except (OSError, UnicodeError, csv.Error):
        return None


def json_shape(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> dict[str, Any] | None:
    if info.file_size > MAX_JSON_INSPECT_BYTES:
        return {"inspection": "skipped_size_limit", "size_bytes": info.file_size}
    try:
        with archive.open(info, "r") as stream:
            payload = json.load(stream)
        if isinstance(payload, dict):
            return {"root_type": "object", "keys": sorted(map(str, payload.keys()))[:MAX_HEADERS]}
        if isinstance(payload, list):
            first = payload[0] if payload else None
            return {
                "root_type": "array",
                "length": len(payload),
                "first_item_type": type(first).__name__ if first is not None else None,
                "first_item_keys": sorted(map(str, first.keys()))[:MAX_HEADERS]
                if isinstance(first, dict)
                else [],
            }
        return {"root_type": type(payload).__name__}
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {"inspection": "unreadable_or_nonstandard_json"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    args = parser.parse_args()

    file_digest = digest_file(args.archive)
    if file_digest["md5"] != EXPECTED_MD5:
        raise SystemExit(
            f"MD5 mismatch: expected {EXPECTED_MD5}, got {file_digest['md5']}"
        )

    with zipfile.ZipFile(args.archive) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        unsafe = [info.filename for info in infos if not safe_member(info.filename)]
        if unsafe:
            raise SystemExit(f"unsafe archive paths: {unsafe[:10]}")

        extensions = Counter(
            PurePosixPath(info.filename).suffix.lower() or "[no extension]"
            for info in infos
        )
        top_level = sorted(
            {
                PurePosixPath(info.filename).parts[0]
                for info in infos
                if PurePosixPath(info.filename).parts
            }
        )

        entries: list[dict[str, Any]] = []
        csv_schemas: list[dict[str, Any]] = []
        json_schemas: list[dict[str, Any]] = []
        readmes: list[str] = []

        for info in sorted(infos, key=lambda item: item.filename):
            entry = {
                "path": info.filename,
                "size_bytes": info.file_size,
                "compressed_size_bytes": info.compress_size,
            }
            suffix = PurePosixPath(info.filename).suffix.lower()
            if suffix in {".csv", ".tsv"}:
                headers = csv_header(archive, info)
                if headers is not None:
                    csv_schemas.append({"path": info.filename, "headers": headers})
            elif suffix == ".json":
                shape = json_shape(archive, info)
                if shape is not None:
                    json_schemas.append({"path": info.filename, **shape})
            elif PurePosixPath(info.filename).name.lower().startswith("readme"):
                try:
                    with archive.open(info, "r") as stream:
                        text = stream.read(16_384).decode("utf-8", errors="replace")
                    headings = [
                        line.strip()[:200]
                        for line in text.splitlines()
                        if line.lstrip().startswith("#")
                    ][:40]
                    readmes.extend([f"{info.filename}: {heading}" for heading in headings])
                except OSError:
                    pass

            entries.append(entry)

    report = {
        "schema_version": "0.1.0",
        "source": {
            "zenodo_record": "https://zenodo.org/records/18547122",
            "doi": "10.5281/zenodo.18547122",
            "version": "v2.0",
            "filename": "center-for-humans-and-machines/reward-network-iii-v2.0.zip",
            "expected_md5": EXPECTED_MD5,
            "verified_file": file_digest,
        },
        "inspection": {
            "archive_integrity": "verified",
            "raw_participant_or_move_rows_exported": False,
            "file_count": len(entries),
            "top_level_entries": top_level,
            "extension_counts": dict(sorted(extensions.items())),
            "files": entries,
            "csv_tsv_headers": csv_schemas,
            "json_shapes": json_schemas,
            "readme_headings": readmes,
        },
        "limitations": [
            "This report inventories archive structure and field names; it is not a reproduction of the paper's analysis.",
            "CSV headers and JSON keys are metadata only; no participant or move-level row values are exported.",
            "Network orientation, tie-breaking, path legality and hidden-level constraints still require source-code review.",
            "The archive license and per-file data-use terms must be preserved from the source metadata before any redistribution.",
        ],
    }

    with open(args.output_json, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2, ensure_ascii=False)
        output.write("\n")

    lines = [
        "# Reward Network III v2.0 archive inventory",
        "",
        "- Archive integrity: verified against Zenodo MD5.",
        f"- Archive size: {file_digest['size_bytes']} bytes.",
        f"- SHA-256: `{file_digest['sha256']}`.",
        f"- Files: {len(entries)}.",
        "- Raw participant/move rows exported: **no**.",
        "",
        "## Top-level entries",
        "",
        *[f"- `{name}`" for name in top_level],
        "",
        "## File extensions",
        "",
        *[f"- `{suffix}`: {count}" for suffix, count in sorted(extensions.items())],
        "",
        "## README headings",
        "",
        *[f"- {heading}" for heading in readmes],
        "",
        "## CSV/TSV field names",
        "",
    ]
    for item in csv_schemas:
        lines.append(f"### `{item['path']}`")
        lines.append("")
        lines.append(", ".join(f"`{header}`" for header in item["headers"]))
        lines.append("")
    lines.extend([
        "## JSON structure",
        "",
    ])
    for item in json_schemas:
        lines.append(f"- `{item['path']}`: {json.dumps({k: v for k, v in item.items() if k != 'path'}, ensure_ascii=False)}")
    lines.extend(["", "## Limitations", "", *[f"- {note}" for note in report["limitations"]], ""])
    with open(args.output_markdown, "w", encoding="utf-8") as output:
        output.write("\n".join(lines))

    print(json.dumps({
        "file_count": len(entries),
        "sha256": file_digest["sha256"],
        "md5": file_digest["md5"],
        "csv_schema_count": len(csv_schemas),
        "json_schema_count": len(json_schemas),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
