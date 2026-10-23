#!/usr/bin/env python3
"""Freeze qualified EPI-AUTH-FLOW R5R5-R2 output into a byte-bound R6 inventory.

The exact R5 measurement programs are replayed from their frozen Git commit against
the unchanged production source tree. Their stdout must byte-match the outputs from
the successful qualifier artifact before any inventory can pass.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SCHEMA = "epi-auth-flow-001-r6-byte-inventory-v1"
SCOPE = "measurement-only-byte-bound-authority-flow-inventory"
SOURCE = "adb69f11fa8068b019cc5bb598d0c7726a197fc9"
SOURCE_TREE = "35a6c5fdba319556af9bb487838734f67c8ac0d6"
R5_PRODUCT = "9d7f3cebc90ed69553d0487bd9f312c2259f5fdf"
R5_PRODUCT_TREE = "b801476b789ea3e5c534b351a1bd4c96fbb9aeca"
R5_QUAL = "9c4501835500b4c90c4f975b41c63a885d5f972b"
R5_QUAL_TREE = "3d5c2f8f1a1fc1f47975aeb97a72fc7e3154d8db"
RUN_ID = 35616089981
JOB_ID = 106387046876
ARTIFACT_ID = 10684849865
ARTIFACT_ZIP_SHA = "e2389e8dc1700898e5d956055cc7862a14f7f723398027491c2c9cce6e64415a"
DISCOVERY_SHA = "07cc24071bd84cb277dd91a668f1da96b29e5c4c37322b83b10b3ff195870504"
CONSUMER_SHA = "26ed61bb6f4cd49ac1673b3a96e95285772417586728eb657f244e64f36fdea2"
DISCOVERY_SCRIPT = "scripts/discover_epistemic_authority_flow.py"
CONSUMER_SCRIPT = "scripts/audit_epistemic_expression_consumers.py"
EXPECTED_SURFACE_COUNT = 17
EXPECTED_MEMBERSHIPS = 193
EXPECTED_UNIQUE_PATHS = 108
EXPECTED_WITNESSES = [
    "coding_confidence_to_generic_llm_context",
    "formal_verification_is_separate_optional_property",
    "generic_llm_context_is_internal_not_admission",
    "generic_llm_context_to_system_prompt",
    "verified_generation_overbroad_guarantee_vocabulary",
    "verified_generation_property_split",
    "verified_generation_real_execution_fail_closed",
]
EXPECTED_PATH_DIGESTS = {
    "broca_cube_gate_surface": "e4c0570c59fc0268ec882324b0a3f373f1576d59fdc5edda6ac15543d5d94e3f",
    "broca_cube_sink": "5475d66ac33c90b7f17f397062175942835bd7e209b7827e4d25b785fb801124",
    "broca_gate_control_surface": "5b69d7754991130ead91eb80708e7adfa65989e63ec6d3768676b5b08cb6e6cb",
    "broca_ordinal_gate_surface": "0d496dfcec701f7bff020dc9270b1da84b9621a0437fa021f35b406b2d2d4b35",
    "epistemic_code_bypass_surface": "875125920d877ffd48f82c5cf5d2d4cb3deaeb63426559a828dc78232f67eee6",
    "epistemic_ordinal_surface": "1eef368fbffeacaf68e4dd08bf13b7ed7f77c0bfe632ab406c77edf1e36834ba",
    "epistemic_status_definition": "fa40b7043725a0dc5228590544d2f0fee38215f1aa2380eb43642fad1f2e8de9",
    "legacy_empirical_authority": "e02bc004d7f2b3e76d6dae653a26466a8cfe1722216b72d65d59bbfc8c85fac4",
    "legacy_high_axis_authority": "baffdc6b55b9d46347dee3c90f6d56fc68b0b4f173c2623123b0c3fe4997436a",
    "llm_prompt_authority_surface": "3d6f3c8db5b3081271a1431c2f54d46ad4b9f1a3ae11f11368de41d6f2ad14ff",
    "numeric_cube_transport": "0ad6403b118d6fe496daf9f6e0cb09a48e5a52d99021a0770232e641d039f755",
    "ordinal_bypass_status_surface": "1356194d0833470e0d6cb92afd51db7185e1069a23db361445cc54cda6e6686b",
    "scalar_cube_collapse": "de9fd15d61fc7c480914fb7150d4064ff3e3b11bdba577ea1de79aa55ee30510",
    "strict_code_gate_surface": "9cb9b94e14b25969fbb70a1056288c186a1ed49d48728774560d00857199b216",
    "typed_cube_definition": "224fcf171a0ab7686c65bbd4b3117cb5985a1038c46794d628c3a8cd66326d2a",
    "typed_high_axis_authority": "6088b51a6a450a287c81f03ae529377ac69eb1996ca96392ec34bec19dae32d4",
    "typed_strong_empirical_authority": "6088b51a6a450a287c81f03ae529377ac69eb1996ca96392ec34bec19dae32d4",
}
PRODUCT_DELTA = [
    "docs/research/EPI_AUTH_FLOW_001_R5R5_R2_DISCOVERY.md",
    "scripts/audit_epistemic_expression_consumers.py",
    "scripts/discover_epistemic_authority_flow.py",
]
QUAL_DELTA = [".github/workflows/qual-epi-auth-flow-001-r5r5-r2.yml"]
ROOT = Path(__file__).resolve().parents[1]


def run(*args: str, cwd: Path | None = None, binary: bool = False) -> str | bytes:
    p = subprocess.run(
        args,
        cwd=cwd or ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return p.stdout if binary else p.stdout.decode().strip()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def path_digest(paths: list[str]) -> str:
    return sha256("\n".join(paths).encode())


def replay(script_path: str, expected_sha: str) -> bytes:
    script = run("git", "show", f"{R5_PRODUCT}:{script_path}", binary=True)
    assert isinstance(script, bytes)
    scripts_dir = ROOT / "scripts"
    fd, tmp_name = tempfile.mkstemp(prefix=".r6-replay-", suffix=".py", dir=scripts_dir)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(script)
        out = run(sys.executable, tmp_name, cwd=ROOT, binary=True)
        assert isinstance(out, bytes)
        assert sha256(out) == expected_sha, (script_path, sha256(out), expected_sha)
        return out
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def parse_discovery(text: str) -> tuple[dict[str, dict], str]:
    surfaces: dict[str, dict] = {}
    schema = scope = lexical = result = None
    for line in text.splitlines():
        if line.startswith("schema="):
            schema = line.split("=", 1)[1]
        elif line.startswith("authority_scope="):
            scope = line.split("=", 1)[1]
        elif line.startswith("lexical_surface_semantics="):
            lexical = line.split("=", 1)[1]
        elif line.startswith("SURFACE "):
            _, name, count_field, digest_field = line.split()
            surfaces[name] = {
                "count": int(count_field.split("=", 1)[1]),
                "digest": digest_field.split("=", 1)[1],
                "paths": [],
            }
        elif line.startswith("FILE "):
            _, name, path = line.split(" ", 2)
            surfaces[name]["paths"].append(path)
        elif line.startswith("result="):
            result = line.split("=", 1)[1]
    assert schema == "epi-auth-flow-001-r5r5-discovery-v1"
    assert scope == "measurement-only-producer-bypass-status-ordinal-cube-prompt-gates-and-controls"
    assert result == "PASS_DISCOVERY"
    assert len(surfaces) == EXPECTED_SURFACE_COUNT
    assert set(surfaces) == set(EXPECTED_PATH_DIGESTS)
    for name, surface in surfaces.items():
        paths = surface["paths"]
        assert paths == sorted(paths), name
        assert len(paths) == len(set(paths)) == surface["count"], name
        assert path_digest(paths) == surface["digest"] == EXPECTED_PATH_DIGESTS[name], name
    return surfaces, lexical or "<not-emitted>"


def parse_consumers(text: str) -> tuple[list[str], str]:
    schema = scope = lexical = result = None
    witnesses: list[str] = []
    for line in text.splitlines():
        if line.startswith("schema="):
            schema = line.split("=", 1)[1]
        elif line.startswith("authority_scope="):
            scope = line.split("=", 1)[1]
        elif line.startswith("lexical_surface_semantics="):
            lexical = line.split("=", 1)[1]
        elif line.startswith("WITNESS "):
            witnesses.append(line.split(" ", 1)[1])
        elif line.startswith("result="):
            result = line.split("=", 1)[1]
    assert schema == "epi-auth-flow-001-r5r5-r2-production-witness-v1"
    assert scope == "measurement-only-exact-production-consumer-witnesses"
    assert lexical == "conservative-file-level-membership-may-include-embedded-cfg-test-code"
    assert result == "PASS_CONSUMER_WITNESS"
    assert witnesses == sorted(EXPECTED_WITNESSES)
    return witnesses, lexical


def content_digest(entries: list[dict], domain: str) -> str:
    h = hashlib.sha256()
    h.update(domain.encode() + b"\0")
    for entry in entries:
        row = (
            entry["path"]
            + "\0"
            + entry["git_blob_id"]
            + "\0"
            + str(entry["bytes"])
            + "\0"
            + entry["sha256"]
            + "\n"
        )
        h.update(row.encode())
    return h.hexdigest()


def main() -> int:
    assert run("git", "rev-parse", SOURCE) == SOURCE
    assert run("git", "rev-parse", f"{SOURCE}^{{tree}}") == SOURCE_TREE
    assert run("git", "rev-parse", R5_PRODUCT) == R5_PRODUCT
    assert run("git", "rev-parse", f"{R5_PRODUCT}^{{tree}}") == R5_PRODUCT_TREE
    assert run("git", "rev-parse", R5_QUAL) == R5_QUAL
    assert run("git", "rev-parse", f"{R5_QUAL}^{{tree}}") == R5_QUAL_TREE
    assert run("git", "rev-parse", f"{R5_PRODUCT}^") == SOURCE
    assert run("git", "rev-parse", f"{R5_QUAL}^") == R5_PRODUCT

    product_delta = run("git", "diff", "--name-only", SOURCE, R5_PRODUCT).splitlines()
    qual_delta = run("git", "diff", "--name-only", R5_PRODUCT, R5_QUAL).splitlines()
    assert sorted(product_delta) == sorted(PRODUCT_DELTA)
    assert sorted(qual_delta) == QUAL_DELTA

    discovery_bytes = replay(DISCOVERY_SCRIPT, DISCOVERY_SHA)
    consumer_bytes = replay(CONSUMER_SCRIPT, CONSUMER_SHA)
    surfaces, lexical = parse_discovery(discovery_bytes.decode())
    witnesses, witness_lexical = parse_consumers(consumer_bytes.decode())

    memberships = sum(surface["count"] for surface in surfaces.values())
    paths = sorted({path for surface in surfaces.values() for path in surface["paths"]})
    assert memberships == EXPECTED_MEMBERSHIPS
    assert len(paths) == EXPECTED_UNIQUE_PATHS

    entries: dict[str, dict] = {}
    for path in paths:
        oid = run("git", "rev-parse", f"{SOURCE}:{path}")
        data = run("git", "cat-file", "blob", f"{SOURCE}:{path}", binary=True)
        assert isinstance(data, bytes)
        entries[path] = {
            "path": path,
            "git_blob_id": oid,
            "bytes": len(data),
            "sha256": sha256(data),
        }

    print(f"schema={SCHEMA}")
    print(f"authority_scope={SCOPE}")
    print(f"source_commit={SOURCE}")
    print(f"source_tree={SOURCE_TREE}")
    print(f"qualified_product_commit={R5_PRODUCT}")
    print(f"qualified_product_tree={R5_PRODUCT_TREE}")
    print(f"qualified_qualifier_commit={R5_QUAL}")
    print(f"qualified_qualifier_tree={R5_QUAL_TREE}")
    print(f"workflow_run_id={RUN_ID}")
    print(f"workflow_job_id={JOB_ID}")
    print(f"artifact_id={ARTIFACT_ID}")
    print(f"artifact_zip_sha256={ARTIFACT_ZIP_SHA}")
    print(f"qualified_discovery_sha256={DISCOVERY_SHA}")
    print(f"qualified_consumer_sha256={CONSUMER_SHA}")
    print(f"lexical_surface_semantics={lexical}")
    print(f"consumer_lexical_surface_semantics={witness_lexical}")

    surface_summary: dict[str, dict] = {}
    for name in sorted(surfaces):
        surface = surfaces[name]
        surface_entries = [entries[path] for path in surface["paths"]]
        digest = content_digest(surface_entries, f"epi-auth-flow-r6-surface:{name}")
        print(
            f"SURFACE {name} count={surface['count']} "
            f"path_digest={surface['digest']} content_digest={digest}"
        )
        for entry in surface_entries:
            print(
                f"FILE {name} {entry['path']} blob={entry['git_blob_id']} "
                f"bytes={entry['bytes']} sha256={entry['sha256']}"
            )
        surface_summary[name] = {
            "count": surface["count"],
            "path_digest": surface["digest"],
            "content_digest": digest,
        }

    for witness in witnesses:
        print(f"WITNESS {witness}")

    union_entries = [entries[path] for path in paths]
    union_digest = content_digest(union_entries, "epi-auth-flow-r6-union")
    print(
        f"UNION count={len(paths)} memberships={memberships} "
        f"content_digest={union_digest}"
    )
    summary = {
        "schema": SCHEMA,
        "authority_scope": SCOPE,
        "result": "PASS_INVENTORY",
        "source_commit": SOURCE,
        "source_tree": SOURCE_TREE,
        "surface_count": len(surfaces),
        "surface_memberships": memberships,
        "unique_paths": len(paths),
        "union_content_digest": union_digest,
        "surfaces": surface_summary,
        "witness_groups": witnesses,
    }
    print("SUMMARY " + json.dumps(summary, sort_keys=True, separators=(",", ":")))
    print("result=PASS_INVENTORY")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, subprocess.CalledProcessError, UnicodeDecodeError, OSError) as exc:
        print(
            f"result=FAIL_INVENTORY error={type(exc).__name__}:{exc}",
            file=sys.stderr,
        )
        raise
