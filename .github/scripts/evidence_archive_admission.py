#!/usr/bin/env python3
"""Fail-closed admission for deterministic evidence tar.gz archives.

This module is intentionally stdlib-only. It validates a frozen JSON profile and
an archive's transport shape, tar headers, member types, metadata, and resource
bounds before copying any admitted file bytes into a fresh output directory.

Admission is a transport/security theorem only. It does not establish scientific
validity, signer authority, evidence positivity, or any support tier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
from typing import Any

PROFILE_SCHEMA_V1 = "butlin-evidence-archive-admission-profile-v1"


class ArchiveAdmissionError(RuntimeError):
    """Raised when a profile or archive fails closed admission."""


def _require_exact_keys(obj: dict[str, Any], expected: set[str], context: str) -> None:
    observed = set(obj)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ArchiveAdmissionError(
            f"{context} keys mismatch: missing={missing} extra={extra}"
        )


def _require_positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ArchiveAdmissionError(f"{field} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ArchiveAdmissionError(f"{field} must be a non-negative integer")
    return value


def _validate_posix_path(path: str, context: str) -> tuple[str, ...]:
    if not isinstance(path, str) or not path:
        raise ArchiveAdmissionError(f"{context} path must be a non-empty string")
    if path.startswith("/") or "\\" in path or "//" in path or "\x00" in path:
        raise ArchiveAdmissionError(f"unsafe {context} path {path!r}")
    parts = tuple(path.split("/"))
    if any(part in ("", ".", "..") for part in parts):
        raise ArchiveAdmissionError(f"unsafe {context} path component in {path!r}")
    pure = PurePosixPath(path)
    if pure.is_absolute() or tuple(pure.parts) != parts:
        raise ArchiveAdmissionError(f"non-canonical {context} path {path!r}")
    return parts


def load_profile(path: Path) -> dict[str, Any]:
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArchiveAdmissionError(f"invalid archive profile: {error}") from error

    if not isinstance(profile, dict):
        raise ArchiveAdmissionError("archive profile must be a JSON object")

    _require_exact_keys(
        profile,
        {
            "schema",
            "profile_id",
            "root",
            "canonical_uid",
            "canonical_gid",
            "canonical_mtime",
            "max_archive_bytes",
            "max_member_bytes",
            "max_total_unpacked_bytes",
            "members",
        },
        "profile",
    )

    if profile["schema"] != PROFILE_SCHEMA_V1:
        raise ArchiveAdmissionError(
            f"wrong profile schema {profile['schema']!r}; expected {PROFILE_SCHEMA_V1!r}"
        )
    if not isinstance(profile["profile_id"], str) or not profile["profile_id"]:
        raise ArchiveAdmissionError("profile_id must be a non-empty string")

    root_parts = _validate_posix_path(profile["root"], "profile root")
    if len(root_parts) != 1:
        raise ArchiveAdmissionError("profile root must be exactly one path component")

    for field in ("canonical_uid", "canonical_gid", "canonical_mtime"):
        profile[field] = _require_nonnegative_int(profile[field], field)
    for field in (
        "max_archive_bytes",
        "max_member_bytes",
        "max_total_unpacked_bytes",
    ):
        profile[field] = _require_positive_int(profile[field], field)

    members = profile["members"]
    if not isinstance(members, list) or not members:
        raise ArchiveAdmissionError("members must be a non-empty list")

    seen: set[str] = set()
    root_spec_count = 0
    normalized_members: list[dict[str, Any]] = []
    for index, member in enumerate(members):
        context = f"members[{index}]"
        if not isinstance(member, dict):
            raise ArchiveAdmissionError(f"{context} must be an object")
        kind = member.get("kind")
        if kind == "directory":
            _require_exact_keys(member, {"path", "kind", "mode"}, context)
        elif kind == "file":
            _require_exact_keys(
                member,
                {"path", "kind", "mode", "allow_empty", "max_bytes"},
                context,
            )
        else:
            raise ArchiveAdmissionError(f"{context}.kind must be 'directory' or 'file'")

        member_path = member["path"]
        parts = _validate_posix_path(member_path, context)
        if parts[0] != profile["root"]:
            raise ArchiveAdmissionError(
                f"{context} is outside root {profile['root']!r}: {member_path!r}"
            )
        if member_path in seen:
            raise ArchiveAdmissionError(f"duplicate profile member {member_path!r}")
        seen.add(member_path)

        mode = _require_nonnegative_int(member["mode"], f"{context}.mode")
        if mode > 0o7777:
            raise ArchiveAdmissionError(f"{context}.mode exceeds permission-bit range")

        normalized = dict(member)
        normalized["mode"] = mode
        if kind == "directory":
            if member_path == profile["root"]:
                root_spec_count += 1
            elif len(parts) == 1:
                raise ArchiveAdmissionError(f"unexpected top-level directory {member_path!r}")
        else:
            if not isinstance(member["allow_empty"], bool):
                raise ArchiveAdmissionError(f"{context}.allow_empty must be boolean")
            max_bytes = _require_positive_int(member["max_bytes"], f"{context}.max_bytes")
            if max_bytes > profile["max_member_bytes"]:
                raise ArchiveAdmissionError(
                    f"{context}.max_bytes exceeds profile max_member_bytes"
                )
            normalized["max_bytes"] = max_bytes

        normalized_members.append(normalized)

    if root_spec_count != 1:
        raise ArchiveAdmissionError("profile must contain exactly one root directory member")

    profile["members"] = normalized_members
    return profile


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_output_dir(output_dir: Path) -> Path:
    if not output_dir.is_absolute():
        raise ArchiveAdmissionError("output directory must be absolute")
    if output_dir.is_symlink():
        raise ArchiveAdmissionError("output directory must not be a symlink")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ArchiveAdmissionError("output path exists and is not a directory")
        if any(output_dir.iterdir()):
            raise ArchiveAdmissionError("output directory must be empty")
    else:
        output_dir.mkdir(parents=True, mode=0o700)
    return output_dir.resolve(strict=True)


def admit_archive(profile: dict[str, Any], archive: Path, output_dir: Path) -> dict[str, Any]:
    if not archive.is_absolute():
        raise ArchiveAdmissionError("archive path must be absolute")
    try:
        archive_lstat = archive.lstat()
    except OSError as error:
        raise ArchiveAdmissionError(f"cannot stat archive: {error}") from error
    if archive.is_symlink() or not archive.is_file():
        raise ArchiveAdmissionError("archive must be a non-symlink regular file")
    archive_bytes = archive_lstat.st_size
    if archive_bytes <= 0 or archive_bytes > profile["max_archive_bytes"]:
        raise ArchiveAdmissionError(
            f"archive outside compressed-size bound: {archive_bytes} bytes"
        )

    output_root = _prepare_output_dir(output_dir)
    expected = {member["path"]: member for member in profile["members"]}

    try:
        tf = tarfile.open(archive, mode="r:gz")
    except (OSError, tarfile.TarError) as error:
        raise ArchiveAdmissionError(f"cannot open tar.gz archive: {error}") from error

    total_unpacked = 0
    try:
        members = tf.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ArchiveAdmissionError("duplicate archive member path")
        if set(names) != set(expected):
            missing = sorted(set(expected) - set(names))
            extra = sorted(set(names) - set(expected))
            raise ArchiveAdmissionError(
                f"archive member mismatch: missing={missing} extra={extra}"
            )

        admitted: list[tuple[tarfile.TarInfo, dict[str, Any], tuple[str, ...]]] = []
        for member in members:
            spec = expected[member.name]
            parts = _validate_posix_path(member.name, "archive member")

            if member.pax_headers:
                raise ArchiveAdmissionError(
                    f"extended PAX metadata is not allowed: {member.name!r}"
                )
            if getattr(member, "sparse", None):
                raise ArchiveAdmissionError(f"sparse member is not allowed: {member.name!r}")
            if member.uid != profile["canonical_uid"]:
                raise ArchiveAdmissionError(f"wrong uid for {member.name!r}")
            if member.gid != profile["canonical_gid"]:
                raise ArchiveAdmissionError(f"wrong gid for {member.name!r}")
            if int(member.mtime) != profile["canonical_mtime"]:
                raise ArchiveAdmissionError(f"wrong mtime for {member.name!r}")
            if (member.mode & 0o7777) != spec["mode"]:
                raise ArchiveAdmissionError(f"wrong mode for {member.name!r}")

            if spec["kind"] == "directory":
                if not member.isdir():
                    raise ArchiveAdmissionError(
                        f"expected directory but observed another type: {member.name!r}"
                    )
                if member.size != 0:
                    raise ArchiveAdmissionError(
                        f"directory member has non-zero size: {member.name!r}"
                    )
            else:
                if not member.isfile():
                    raise ArchiveAdmissionError(
                        f"expected regular file but observed another type: {member.name!r}"
                    )
                if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                    raise ArchiveAdmissionError(
                        f"forbidden special member type: {member.name!r}"
                    )
                if member.size == 0 and not spec["allow_empty"]:
                    raise ArchiveAdmissionError(f"empty member forbidden: {member.name!r}")
                if member.size < 0 or member.size > spec["max_bytes"]:
                    raise ArchiveAdmissionError(
                        f"member outside size bound: {member.name!r} size={member.size}"
                    )
                total_unpacked += member.size
                if total_unpacked > profile["max_total_unpacked_bytes"]:
                    raise ArchiveAdmissionError("archive exceeds total unpacked-size bound")

            admitted.append((member, spec, parts))

        # Create only profile-declared directories, shallowest first.
        for member, spec, parts in sorted(admitted, key=lambda item: len(item[2])):
            if spec["kind"] != "directory":
                continue
            destination = output_root.joinpath(*parts)
            destination.mkdir(mode=spec["mode"], parents=False, exist_ok=False)

        # Copy bytes only from already-admitted regular members. Never call
        # TarFile.extract()/extractall(), so link/device semantics cannot escape.
        for member, spec, parts in admitted:
            if spec["kind"] != "file":
                continue
            destination = output_root.joinpath(*parts)
            parent = destination.parent.resolve(strict=True)
            if output_root != parent and output_root not in parent.parents:
                raise ArchiveAdmissionError(
                    f"resolved extraction parent escaped output root: {member.name!r}"
                )
            source = tf.extractfile(member)
            if source is None:
                raise ArchiveAdmissionError(f"could not read member: {member.name!r}")
            try:
                with source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                os.chmod(destination, spec["mode"])
            except OSError as error:
                raise ArchiveAdmissionError(
                    f"failed controlled extraction of {member.name!r}: {error}"
                ) from error
    finally:
        tf.close()

    return {
        "schema": "butlin-evidence-archive-admission-result-v1",
        "profile_id": profile["profile_id"],
        "archive_sha256": _sha256_file(archive),
        "archive_bytes": archive_bytes,
        "member_count": len(expected),
        "total_unpacked_bytes": total_unpacked,
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        profile = load_profile(args.profile.resolve(strict=True))
        result = admit_archive(profile, args.archive, args.output_dir)
    except (ArchiveAdmissionError, OSError) as error:
        print(f"evidence archive admission failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
