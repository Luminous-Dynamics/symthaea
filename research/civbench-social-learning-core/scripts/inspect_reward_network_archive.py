#!/usr/bin/env python3
"""Inspect Reward Network III Zenodo archive without exporting raw data rows.

This is a research-only ingestion helper. It inventories names, sizes, CSV
headers, small JSON key structures and README headings. It does not copy raw
participant/move records into the output report.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import io
import json
import os
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


def read_lfs_pointer(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> dict[str, Any] | None:
    """Return Git LFS pointer metadata, or None for a normal archive blob."""
    if info.file_size > 1024:
        return None
    try:
        raw = archive.read(info)
        lines = raw.decode("ascii").splitlines()
    except (OSError, UnicodeError):
        return None
    if not lines or lines[0].strip() != "version https://git-lfs.github.com/spec/v1":
        return None
    fields = {}
    for line in lines[1:]:
        key, separator, value = line.partition(" ")
        if separator:
            fields[key.strip()] = value.strip()
    oid = fields.get("oid", "")
    size = fields.get("size", "")
    if not oid.startswith("sha256:") or len(oid) != 71 or not size.isdigit():
        raise ValueError(f"malformed Git LFS pointer at {info.filename}")
    return {"oid_sha256": oid.removeprefix("sha256:"), "declared_size_bytes": int(size)}

MAX_SOURCE_TEXT_BYTES = 64 * 1024
SOURCE_SNAPSHOT_ALLOWLIST = {
    "README.md",
    "abm/README.md",
    "abm/abm.ipynb",
    "abm/array_to_df.py",
    "abm/model.py",
    "algorithm/README.md",
    "algorithm/params/seed_0.yml",
    "algorithm/params/seed_1.yml",
    "algorithm/params/seed_2.yml",
    "algorithm/dqn/config_type.py",
    "algorithm/dqn/dqn_exp.py",
    "algorithm/dqn/dqn_agent.py",
    "algorithm/dqn/environment_vect.py",
    "algorithm/dqn/logger.py",
    "algorithm/dqn/memory.py",
    "algorithm/dqn/nn.py",
    "common/generate/generation.py",
    "analysis/README.md",
    "data/README.md",
    "statistics/README.md",
    ".gitattributes",
    ".lfsconfig",
    ".gitlab-ci.yml",
    "setup/ci/.gitlab-ci-backend.yml",
    "setup/ci/.gitlab-ci-frontend.yml",
    "setup/ci/.gitlab-ci-streamlit.yml",
}
NOTEBOOK_SNAPSHOT_MAX_BYTES = 1_500_000

def notebook_source_snapshot(raw: bytes, info: zipfile.ZipInfo) -> dict[str, Any]:
    """Retain notebook markdown headings and code cells, never cell outputs."""
    notebook = json.loads(raw.decode("utf-8"))
    extracted_cells = []
    for index, cell in enumerate(notebook.get("cells", [])):
        cell_type = cell.get("cell_type")
        source = cell.get("source", [])
        source_text = "".join(source) if isinstance(source, list) else str(source)
        if cell_type == "code":
            extracted_cells.append({
                "cell_index": index,
                "cell_type": "code",
                "execution_count": cell.get("execution_count"),
                "source": source_text,
            })
        elif cell_type == "markdown":
            headings = [
                line.strip() for line in source_text.splitlines()
                if line.lstrip().startswith("#")
            ]
            if headings:
                extracted_cells.append({
                    "cell_index": index,
                    "cell_type": "markdown_headings",
                    "headings": headings,
                })
    return {
        "path": info.filename,
        "size_bytes": info.file_size,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "notebook_metadata": {
            "nbformat": notebook.get("nbformat"),
            "nbformat_minor": notebook.get("nbformat_minor"),
            "kernelspec": notebook.get("metadata", {}).get("kernelspec"),
        },
        "extracted_cells": extracted_cells,
        "outputs_included": False,
    }

def source_snapshot(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> dict[str, Any] | None:
    parts = PurePosixPath(info.filename).parts
    relative_path = "/".join(parts[1:])
    if relative_path not in SOURCE_SNAPSHOT_ALLOWLIST:
        return None
    if relative_path.endswith(".ipynb"):
        if info.file_size > NOTEBOOK_SNAPSHOT_MAX_BYTES:
            return {"path": info.filename, "inspection": "notebook_skipped_size_limit", "size_bytes": info.file_size}
        try:
            raw = archive.read(info)
            if raw.startswith(b"version https://git-lfs.github.com/spec/v1"):
                return None
            return notebook_source_snapshot(raw, info)
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {"path": info.filename, "inspection": "unreadable_notebook"}
    if info.file_size > MAX_SOURCE_TEXT_BYTES:
        return {"path": info.filename, "inspection": "skipped_size_limit", "size_bytes": info.file_size}
    try:
        raw = archive.read(info)
        if raw.startswith(b"version https://git-lfs.github.com/spec/v1"):
            return None
        return {
            "path": info.filename,
            "size_bytes": info.file_size,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "content": raw.decode("utf-8", errors="replace"),
        }
    except OSError:
        return {"path": info.filename, "inspection": "unreadable_source_file"}

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
        lfs_objects: list[dict[str, Any]] = []
        research_sources: list[dict[str, Any]] = []

        for info in sorted(infos, key=lambda item: item.filename):
            entry = {
                "path": info.filename,
                "size_bytes": info.file_size,
                "compressed_size_bytes": info.compress_size,
            }
            pointer = read_lfs_pointer(archive, info)
            if pointer is not None:
                entry["storage_kind"] = "git_lfs_pointer"
                entry["lfs_oid_sha256"] = pointer["oid_sha256"]
                entry["lfs_declared_size_bytes"] = pointer["declared_size_bytes"]
                lfs_objects.append({"path": info.filename, **pointer})
            else:
                entry["storage_kind"] = "archive_blob"

            snapshot = source_snapshot(archive, info)
            if snapshot is not None:
                research_sources.append(snapshot)

            suffix = PurePosixPath(info.filename).suffix.lower()
            if pointer is None and suffix in {".csv", ".tsv"}:
                headers = csv_header(archive, info)
                if headers is not None:
                    csv_schemas.append({"path": info.filename, "headers": headers})
            elif pointer is None and suffix == ".json":
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
        "receipt": {
            "retrieved_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
            "workflow_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "source_commit": os.environ.get("GITHUB_SHA"),
            "inventory_tool": "inspect_reward_network_archive.py",
            "inventory_tool_version": "0.1.0",
            "python_version": sys.version.split()[0],
        },
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
            "git_lfs_pointer_count": len(lfs_objects),
        "research_source_snapshot_count": len(research_sources),
            "git_lfs_objects": lfs_objects,
            "research_source_snapshots": research_sources,
        },
        "limitations": [
            "This report inventories archive structure and field names; it is not a reproduction of the paper's analysis.",
            "CSV headers and JSON keys are metadata only; no participant or move-level row values are exported.",
            "Git LFS pointer files are identified by OID and declared size but are not hydrated by this inventory step; underlying LFS data rows are not inspected.",
            "Research source snapshots include only an explicit allowlist of README, ABM notebook code/markdown headings, ABM source, DQN source, configuration and CI/LFS endpoint metadata files; notebook cell outputs are excluded and no participant-level raw or processed data files are included.",
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
        f"- Git LFS pointer files: {len(lfs_objects)} (objects not hydrated by this inventory step).",
        f"- Allowlisted paper/ABM/DQN source snapshots: {len(research_sources)}.",
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
        "git_lfs_pointer_count": len(lfs_objects),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
