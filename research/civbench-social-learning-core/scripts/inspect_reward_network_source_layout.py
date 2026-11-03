#!/usr/bin/env python3
"""Inspect only allowlisted source/configuration files in the Zenodo ZIP."""
from __future__ import annotations
import argparse, hashlib, json, re, zipfile
from typing import Any

ALLOWLIST = (
    "README.md", ".gitattributes", ".gitlab-ci.yml", ".lfsconfig",
    "abm/README.md", "analysis/README.md", "algorithm/README.md",
    "experiment/README.md",
)
MAX_TEXT_BYTES = 32000
SECRET_LINE = re.compile(
    r"(?i)(token|password|secret|private[_-]?key|client[_-]?secret)\s*[:=]\s*\S+"
)
URL_RE = re.compile(r"https?://[^\s\"'<>]+")
LFS_RE = re.compile(r"(?i)(git[- ]?lfs|lfs|objects/batch|gitlab|github)")

def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def find_member(names: list[str], relative: str) -> str | None:
    matches = [n for n in names if n == relative or n.endswith("/" + relative)]
    if not matches:
        return None
    # Zenodo's GitHub source ZIP has one repository-root prefix. For generic
    # names such as README.md, prefer the shortest path (root file) rather than
    # confusing it with nested README files. Scoped names such as abm/README.md
    # remain unambiguous after the suffix match.
    shortest_depth = min(name.count("/") for name in matches)
    shortest = [name for name in matches if name.count("/") == shortest_depth]
    if len(shortest) != 1:
        raise RuntimeError(f"ambiguous root-level archive path for {relative}: {shortest}")
    return shortest[0]

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    args = parser.parse_args()
    files: list[dict[str, Any]] = []
    urls: set[str] = set()
    markers: list[str] = []
    lfs_config_found = False
    with zipfile.ZipFile(args.archive) as z:
        names = z.namelist()
        for relative in ALLOWLIST:
            member = find_member(names, relative)
            if member is None:
                continue
            raw = z.read(member)
            lfs_config_found |= relative == ".lfsconfig"
            txt = raw[:MAX_TEXT_BYTES].decode("utf-8", errors="replace")
            safe_lines = []
            for line in txt.splitlines():
                if SECRET_LINE.search(line):
                    safe_lines.append("[REDACTED: credential-like assignment]")
                    continue
                safe_lines.append(line[:500])
                if LFS_RE.search(line):
                    markers.append(relative + ": " + line[:240])
                urls.update(URL_RE.findall(line))
            files.append({
                "path": member, "size_bytes": len(raw), "sha256": digest(raw),
                "truncated": len(raw) > MAX_TEXT_BYTES, "text": "\n".join(safe_lines),
            })
    report = {
        "schema_version": "0.1.0",
        "source": {"zenodo_record": "https://zenodo.org/records/18547122", "version": "v2.0"},
        "inspection": {
            "allowlist": list(ALLOWLIST),
            "lfs_config_file_present": lfs_config_found,
            "files": files,
            "configuration_markers": sorted(set(markers)),
            "urls_found": sorted(urls),
            "participant_data_extracted": False,
            "network_and_solution_data_extracted": False,
            "model_weights_extracted": False,
        },
        "limitations": [
            "Only explicitly allowlisted small text files are inspected.",
            "A URL in a README or CI file is not proof that it hosts Git LFS objects.",
            "No Git LFS object is downloaded by this probe.",
        ],
    }
    with open(args.output_json, "w", encoding="utf-8") as out:
        json.dump(report, out, indent=2, ensure_ascii=False)
        out.write("\n")
    lines = [
        "# Reward Network III v2.0 source-layout inspection", "",
        "LFS config file present: " + str(lfs_config_found),
        "Participant/move rows extracted: no",
        "Networks/solutions/weights extracted: no", "",
        "## Allowlisted files", "",
    ]
    for item in files:
        lines.extend([
            "### " + item["path"], "",
            "Bytes: " + str(item["size_bytes"]),
            "SHA-256: " + item["sha256"],
            "Truncated: " + str(item["truncated"]), "",
            item["text"], "",
        ])
    lines.extend([
        "## Git LFS / hosting markers", "",
        *markers, "",
        "## URLs", "",
        *sorted(urls), "",
        "## Limitations", "",
        *report["limitations"], "",
    ])
    with open(args.output_markdown, "w", encoding="utf-8") as out:
        out.write("\n".join(lines))
    print(json.dumps({
        "allowlisted_files": len(files),
        "lfs_config_file_present": lfs_config_found,
        "urls_found": len(urls),
        "participant_data_extracted": False,
    }, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
