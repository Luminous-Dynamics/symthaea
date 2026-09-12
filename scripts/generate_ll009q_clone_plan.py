#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import urllib.parse
from typing import Any

FAMILY_SCHEMA = "ll009q.clone-family.v1"
N_PLAN_SCHEMA = "ll009n.nasa-acquisition-plan.v1"
DEFAULT_REQUIRED_SET = "site01-clone-ensemble-source-bytes"


class QPlanError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")) + "\n"
    ).encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def safe_relpath(value: str) -> pathlib.PurePosixPath:
    if not isinstance(value, str) or not value:
        raise QPlanError("relative path must be a non-empty string")
    path = pathlib.PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise QPlanError(f"unsafe relative path {value!r}")
    return path


def canonical_host(value: str) -> str:
    try:
        return value.encode("idna").decode("ascii").lower().rstrip(".")
    except UnicodeError as exc:
        raise QPlanError(f"invalid host {value!r}") from exc


def validate_https_url(value: str, allowed_hosts: set[str]) -> None:
    if not isinstance(value, str) or not value:
        raise QPlanError("URL must be a non-empty string")
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme.lower() != "https":
        raise QPlanError(f"URL must use HTTPS: {value}")
    if parsed.username is not None or parsed.password is not None:
        raise QPlanError("URL credentials are not allowed")
    if not parsed.hostname:
        raise QPlanError("URL host required")
    if parsed.port not in (None, 443):
        raise QPlanError("unexpected URL port")
    if canonical_host(parsed.hostname) not in allowed_hosts:
        raise QPlanError(f"host not allowlisted: {parsed.hostname}")
    if parsed.fragment:
        raise QPlanError("URL fragments are not allowed")


def validate_family(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != FAMILY_SCHEMA:
        raise QPlanError(f"schema_version must be {FAMILY_SCHEMA}")
    data = dict(value)
    for key in (
        "study_id",
        "provider",
        "dataset_id",
        "source_page",
        "source_url_directory",
        "artifact_directory",
        "source_id_prefix",
        "filename_prefix",
        "filename_suffix",
        "role",
        "interpretation",
        "interpretation_basis",
    ):
        if not isinstance(data.get(key), str) or not data[key]:
            raise QPlanError(f"missing {key}")

    hosts = data.get("allowed_hosts")
    if (
        not isinstance(hosts, list)
        or not hosts
        or not all(isinstance(item, str) and item for item in hosts)
    ):
        raise QPlanError("allowed_hosts must be a non-empty string list")
    allowed_hosts = {canonical_host(item) for item in hosts}
    validate_https_url(data["source_page"], allowed_hosts)
    validate_https_url(data["source_url_directory"], allowed_hosts)

    directory_url = urllib.parse.urlparse(data["source_url_directory"])
    if not directory_url.path.endswith("/"):
        raise QPlanError("source_url_directory must end in '/'")

    artifact_dir = safe_relpath(data["artifact_directory"])
    if artifact_dir.name in (".", ""):
        raise QPlanError("artifact_directory invalid")

    for key in ("first_index", "last_index", "index_width", "member_count"):
        if not isinstance(data.get(key), int) or isinstance(data[key], bool):
            raise QPlanError(f"{key} must be integer")
    first_index = data["first_index"]
    last_index = data["last_index"]
    width = data["index_width"]
    member_count = data["member_count"]
    if first_index < 0 or last_index < first_index:
        raise QPlanError("invalid clone index range")
    if width < 1 or width > 8:
        raise QPlanError("index_width invalid")
    expected_count = last_index - first_index + 1
    if member_count != expected_count:
        raise QPlanError(
            f"member_count {member_count} does not equal inclusive index range {expected_count}"
        )
    if len(str(last_index)) > width:
        raise QPlanError("index_width too small for last_index")

    if "/" in data["filename_prefix"] or "/" in data["filename_suffix"]:
        raise QPlanError("filename prefix/suffix must not contain '/'")
    if not data["filename_suffix"].endswith(".tif"):
        raise QPlanError("V1 clone filename suffix must end in .tif")
    if data["interpretation"] not in {
        "additive_z_error_to_nominal_ldem",
        "full_clone_surface",
        "unknown",
    }:
        raise QPlanError("unsupported interpretation")

    expected_hashes = data.get("expected_sha256_by_index", {})
    if not isinstance(expected_hashes, dict):
        raise QPlanError("expected_sha256_by_index must be an object")
    for raw_index, digest in expected_hashes.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise QPlanError("expected hash index must be integer-like") from exc
        if index < first_index or index > last_index:
            raise QPlanError(f"expected hash index out of range: {index}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
        ):
            raise QPlanError(f"invalid SHA-256 for clone index {index}")

    data["_allowed_hosts"] = allowed_hosts
    return data


def member_filename(family: dict[str, Any], index: int) -> str:
    return (
        family["filename_prefix"]
        + f"{index:0{family['index_width']}d}"
        + family["filename_suffix"]
    )


def expected_digest(family: dict[str, Any], index: int) -> str | None:
    hashes = family.get("expected_sha256_by_index", {})
    return hashes.get(str(index), hashes.get(f"{index:0{family['index_width']}d}"))


def expand_family(value: Any) -> dict[str, Any]:
    family = validate_family(value)
    allowed_hosts = family.pop("_allowed_hosts")
    files: list[dict[str, Any]] = []
    source_ids: list[str] = []
    seen_names: set[str] = set()

    for index in range(family["first_index"], family["last_index"] + 1):
        token = f"{index:0{family['index_width']}d}"
        filename = member_filename(family, index)
        if filename in seen_names:
            raise QPlanError(f"duplicate expanded filename {filename}")
        seen_names.add(filename)
        source_url = urllib.parse.urljoin(family["source_url_directory"], filename)
        validate_https_url(source_url, allowed_hosts)
        if pathlib.PurePosixPath(urllib.parse.urlparse(source_url).path).name != filename:
            raise QPlanError("expanded URL basename mismatch")
        artifact_path = str(safe_relpath(f"{family['artifact_directory']}/{filename}"))
        source_id = f"{family['source_id_prefix']}-{token}"
        source_ids.append(source_id)
        files.append(
            {
                "source_id": source_id,
                "dataset_id": family["dataset_id"],
                "role": family["role"],
                "source_url": source_url,
                "artifact_path": artifact_path,
                "expected_sha256": expected_digest(family, index),
                "required_for": [DEFAULT_REQUIRED_SET],
                "ensemble_member_index": index,
                "ensemble_member_token": token,
            }
        )

    if len(files) != family["member_count"]:
        raise QPlanError("expanded member count drift")
    if len({item["source_id"] for item in files}) != len(files):
        raise QPlanError("duplicate expanded source_id")
    if len({item["artifact_path"] for item in files}) != len(files):
        raise QPlanError("duplicate expanded artifact_path")
    if len({item["source_url"] for item in files}) != len(files):
        raise QPlanError("duplicate expanded source_url")

    family_for_hash = dict(family)
    family_hash = sha256_bytes(canonical_bytes(family_for_hash))
    plan = {
        "schema_version": N_PLAN_SCHEMA,
        "study_id": family["study_id"],
        "provider": family["provider"],
        "evidence_scope": (
            "Exact source-byte acquisition for the LL-009Q Site01 statistical "
            "terrain-clone ensemble; this plan establishes byte identity only."
        ),
        "allowed_hosts": sorted(allowed_hosts),
        "source_pages": [
            {
                "dataset_id": family["dataset_id"],
                "url": family["source_page"],
                "published_semantics": (
                    "NASA PGDA publishes 100 Site01 clones for uncertainty studies; "
                    "Barker et al. describe a statistical ensemble with approximately "
                    "the same error properties as the LDEM."
                ),
                "caveat": (
                    "Finite ensemble members are empirical statistical evidence, not "
                    "deterministic upper bounds on all physically possible terrain."
                ),
            }
        ],
        "files": files,
        "promotion_sets": [
            {
                "set_id": DEFAULT_REQUIRED_SET,
                "claim_scope": (
                    f"Exact-byte completeness for all {family['member_count']} declared "
                    "clone members only; no horizon/visibility claim."
                ),
                "required_source_ids": source_ids,
            }
        ],
        "ll009q_family": {
            "family_schema": FAMILY_SCHEMA,
            "family_sha256": family_hash,
            "member_count": family["member_count"],
            "first_index": family["first_index"],
            "last_index": family["last_index"],
            "index_width": family["index_width"],
            "interpretation": family["interpretation"],
            "interpretation_basis": family["interpretation_basis"],
        },
        "downstream_requirements": [
            "Offline replay of every clone byte under the LL-009N verifier.",
            "Exact alignment of every clone raster with the nominal Site01 LDEM.",
            "LL-009P site state and nominal LDEM source binding.",
            "LL-009Q all-admitted-pixel horizon scan for every ensemble realization.",
            "Explicit finite-ensemble quantile/max semantics before downstream visibility use."
        ],
        "non_claims": [
            "A generated acquisition plan is not a source lock.",
            "Unknown clone SHA-256 values remain null until the source bytes are acquired.",
            "The clone ensemble is statistical evidence and is not a hard physical terrain bound.",
            "LL-009Q does not close unresolved terrain between raster support points."
        ],
    }
    return plan


def write_immutable(path: pathlib.Path, value: dict[str, Any]) -> None:
    payload = canonical_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise QPlanError(f"refusing to overwrite differing output {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def self_test() -> None:
    family = {
        "schema_version": FAMILY_SCHEMA,
        "study_id": "study-q",
        "provider": "NASA Goddard PGDA / LRO-LOLA",
        "dataset_id": "pgda-site01-clones",
        "source_page": "https://pgda.gsfc.nasa.gov/products/78",
        "allowed_hosts": ["pgda.gsfc.nasa.gov"],
        "source_url_directory": "https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/Site01/Clones/",
        "artifact_directory": "nasa/pgda/product78/Site01/Clones",
        "source_id_prefix": "site01-clone",
        "filename_prefix": "Site01_final_adj_5mpp_",
        "filename_suffix": "_err.tif",
        "first_index": 1,
        "last_index": 100,
        "index_width": 4,
        "member_count": 100,
        "role": "terrain_error_realization_m",
        "interpretation": "additive_z_error_to_nominal_ldem",
        "interpretation_basis": "synthetic test",
        "expected_sha256_by_index": {},
    }
    plan = expand_family(family)
    assert len(plan["files"]) == 100
    assert plan["files"][0]["source_url"].endswith(
        "/Site01_final_adj_5mpp_0001_err.tif"
    )
    assert plan["files"][-1]["source_url"].endswith(
        "/Site01_final_adj_5mpp_0100_err.tif"
    )
    assert plan["promotion_sets"][0]["required_source_ids"][0] == "site01-clone-0001"
    assert plan["promotion_sets"][0]["required_source_ids"][-1] == "site01-clone-0100"
    assert all(item["expected_sha256"] is None for item in plan["files"])

    bad = json.loads(json.dumps(family))
    bad["member_count"] = 99
    try:
        expand_family(bad)
    except QPlanError as exc:
        assert "member_count" in str(exc)
    else:
        raise QPlanError("self-test expected member-count rejection")

    bad = json.loads(json.dumps(family))
    bad["source_url_directory"] = "https://example.com/Clones/"
    try:
        expand_family(bad)
    except QPlanError as exc:
        assert "allowlisted" in str(exc)
    else:
        raise QPlanError("self-test expected host rejection")

    bad = json.loads(json.dumps(family))
    bad["artifact_directory"] = "../escape"
    try:
        expand_family(bad)
    except QPlanError as exc:
        assert "unsafe" in str(exc)
    else:
        raise QPlanError("self-test expected traversal rejection")

    first = canonical_bytes(plan)
    second = canonical_bytes(expand_family(family))
    assert first == second
    print("LL-009Q clone-plan self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Expand LL-009Q clone family into an LL-009N acquisition plan"
    )
    parser.add_argument("--family")
    parser.add_argument("--output")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        if not args.family or not args.output:
            raise QPlanError("--family and --output are required")
        family_path = pathlib.Path(args.family)
        family = json.loads(family_path.read_text())
        plan = expand_family(family)
        write_immutable(pathlib.Path(args.output), plan)
        print(json.dumps(plan, sort_keys=True, indent=2))
        return 0
    except (OSError, json.JSONDecodeError, QPlanError) as exc:
        raise SystemExit(f"LL-009Q clone-plan failure: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
