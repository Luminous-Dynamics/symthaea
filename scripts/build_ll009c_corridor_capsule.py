#!/usr/bin/env python3
"""Build an immutable LL-009C corridor evidence capsule.

LL-009C is an evidence-composition tool. It delegates semantic bundle closure
to the LL-009B validator, then copies the exact validated evidence bytes into
an immutable capsule directory with canonical indexes and SHA-256 receipts.

It does not validate the underlying physics and does not authorize launch.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
from typing import Any, Iterable

CAPSULE_INDEX_SCHEMA = "ll009c.corridor-capsule-index.v1"
CAPSULE_RECEIPT_SCHEMA = "ll009c.corridor-capsule-receipt.v1"


class CapsuleError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")) + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CapsuleError(f"expected JSON object: {path}")
    return value


def nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def safe_component(value: str, label: str) -> str:
    if not nonempty(value):
        raise CapsuleError(f"{label} must be non-empty")
    if any(
        ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._"
        for ch in value
    ):
        raise CapsuleError(f"{label} contains unsafe characters: {value!r}")
    if value in {".", ".."}:
        raise CapsuleError(f"{label} cannot be {value!r}")
    return value


def safe_manifest_path(value: str) -> PurePosixPath:
    if not nonempty(value):
        raise CapsuleError("artifact path must be non-empty")
    if "\\" in value:
        raise CapsuleError(f"artifact path must use POSIX separators: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise CapsuleError(f"artifact path escapes/ambiguous: {value!r}")
    return path


def resolve_under(root: Path, rel: PurePosixPath) -> Path:
    root_abs = root.resolve()
    candidate = (root_abs / Path(*rel.parts)).resolve()
    try:
        candidate.relative_to(root_abs)
    except ValueError as exc:
        raise CapsuleError(f"artifact path escapes root: {rel}") from exc
    return candidate


def load_ll009b_validator(path: Path) -> Any:
    if not path.is_file():
        raise CapsuleError(f"LL-009B validator not found: {path}")
    spec = importlib.util.spec_from_file_location("ll009b_validator_for_ll009c", path)
    if spec is None or spec.loader is None:
        raise CapsuleError(f"cannot load LL-009B validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "validate_bundle"):
        raise CapsuleError("LL-009B validator does not expose validate_bundle")
    return module


def tree_manifest(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not root.exists():
        return records
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        records.append(
            {
                "path": rel,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def aggregate_digest(records: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for record in sorted(records, key=lambda item: item["path"]):
        h.update(record["path"].encode("utf-8"))
        h.update(b"\0")
        h.update(record["sha256"].encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def write_exact(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def compare_trees(a: Path, b: Path) -> bool:
    return tree_manifest(a) == tree_manifest(b)


def build_capsule(
    bundle_path: Path,
    schema_path: Path,
    artifact_root: Path,
    output_dir: Path,
    validator_path: Path,
) -> dict[str, Any]:
    if not bundle_path.is_file():
        raise CapsuleError(f"missing bundle manifest: {bundle_path}")
    if not schema_path.is_file():
        raise CapsuleError(f"missing bundle schema: {schema_path}")
    if not artifact_root.is_dir():
        raise CapsuleError(f"artifact root is not a directory: {artifact_root}")
    if output_dir.exists() and not output_dir.is_dir():
        raise CapsuleError(f"output path exists and is not a directory: {output_dir}")

    validator = load_ll009b_validator(validator_path)
    try:
        ll009b_receipt = validator.validate_bundle(
            bundle_path, schema_path, artifact_root
        )
    except Exception as exc:
        raise CapsuleError(f"LL-009B validation failed: {exc}") from exc

    bundle = load_json(bundle_path)
    for field in (
        "bundle_id",
        "study_id",
        "frame_contract_id",
        "epoch_contract_id",
        "requested_promotion",
    ):
        if not nonempty(bundle.get(field)):
            raise CapsuleError(f"bundle missing {field}")

    bundle_id = safe_component(bundle["bundle_id"], "bundle_id")
    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise CapsuleError("bundle artifacts must be non-empty")

    with tempfile.TemporaryDirectory(prefix="ll009c-") as tmp:
        stage = Path(tmp) / "capsule"
        stage.mkdir(parents=True)

        source_bundle_target = stage / "source" / "bundle_manifest.json"
        source_schema_target = stage / "source" / "bundle_schema.json"
        write_exact(source_bundle_target, bundle_path.read_bytes())
        write_exact(source_schema_target, schema_path.read_bytes())

        ll009b_payload = canonical_json_bytes(ll009b_receipt)
        ll009b_target = stage / "validation" / "ll009b_receipt.json"
        write_exact(ll009b_target, ll009b_payload)

        copied_artifacts: list[dict[str, Any]] = []
        seen_roles: set[str] = set()
        for entry in artifacts:
            if not isinstance(entry, dict):
                raise CapsuleError("artifact entries must be objects")
            role = safe_component(str(entry.get("role", "")), "artifact role")
            if role in seen_roles:
                raise CapsuleError(f"duplicate artifact role: {role}")
            seen_roles.add(role)

            rel = safe_manifest_path(str(entry.get("path", "")))
            source = resolve_under(artifact_root, rel)
            if not source.is_file():
                raise CapsuleError(f"artifact {role}: missing file {source}")

            expected = str(entry.get("sha256", "")).lower()
            actual = sha256_file(source)
            if expected != actual:
                raise CapsuleError(
                    f"artifact {role}: source sha256 mismatch "
                    f"expected={expected} actual={actual}"
                )

            basename = safe_component(rel.name, f"artifact {role} basename")
            target = stage / "artifacts" / role / basename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            copied = sha256_file(target)
            if copied != actual:
                raise CapsuleError(f"artifact {role}: copied bytes changed")

            copied_artifacts.append(
                {
                    "role": role,
                    "source_path": rel.as_posix(),
                    "capsule_path": target.relative_to(stage).as_posix(),
                    "lineage_id": entry["lineage_id"],
                    "evidence_class": entry["evidence_class"],
                    "synthetic_only": bool(entry.get("synthetic_only", False)),
                    "bytes": target.stat().st_size,
                    "sha256": copied,
                }
            )

        index = {
            "schema_version": CAPSULE_INDEX_SCHEMA,
            "capsule_id": bundle_id,
            "study_id": bundle["study_id"],
            "frame_contract_id": bundle["frame_contract_id"],
            "epoch_contract_id": bundle["epoch_contract_id"],
            "requested_promotion": bundle["requested_promotion"],
            "promotion_status": ll009b_receipt.get("status", "unknown"),
            "source_bundle_sha256": sha256_file(bundle_path),
            "source_schema_sha256": sha256_file(schema_path),
            "ll009b_receipt_sha256": sha256_bytes(ll009b_payload),
            "ll009b_validator_sha256": sha256_file(validator_path),
            "ll009c_builder_sha256": sha256_file(Path(__file__).resolve()),
            "artifacts": sorted(copied_artifacts, key=lambda item: item["role"]),
            "non_claims": [
                "Capsule closure is provenance/referential evidence only.",
                "A promoted capsule is not launch, site, catcher, or navigation qualification.",
                "Operational release authority remains outside LL-009C.",
            ],
        }
        index_payload = canonical_json_bytes(index)
        write_exact(stage / "capsule_index.json", index_payload)

        pre_receipt_files = tree_manifest(stage)
        receipt = {
            "schema_version": CAPSULE_RECEIPT_SCHEMA,
            "capsule_id": bundle_id,
            "study_id": bundle["study_id"],
            "requested_promotion": bundle["requested_promotion"],
            "capsule_index_sha256": sha256_bytes(index_payload),
            "content_aggregate_sha256": aggregate_digest(pre_receipt_files),
            "file_count_excluding_receipt": len(pre_receipt_files),
            "status": "pass",
            "non_claim": (
                "LL-009C pass means immutable evidence composition only; "
                "it does not authorize physical launch."
            ),
        }
        write_exact(stage / "capsule_receipt.json", canonical_json_bytes(receipt))

        for item in copied_artifacts:
            target = stage / item["capsule_path"]
            if sha256_file(target) != item["sha256"]:
                raise CapsuleError(f"artifact changed during capsule build: {item['role']}")

        if output_dir.exists():
            if compare_trees(stage, output_dir):
                return receipt
            raise CapsuleError(
                f"refusing to overwrite differing immutable capsule {output_dir}; "
                "choose a new capsule/lineage path"
            )

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(stage, output_dir)
        if not compare_trees(stage, output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
            raise CapsuleError("capsule copy verification failed")

    return receipt


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="ll009c-selftest-") as tmp:
        root = Path(tmp)
        scripts = root / "scripts"
        scripts.mkdir()
        validator = scripts / "validator.py"
        validator.write_text(
            "def validate_bundle(bundle_path, schema_path, artifact_root):\n"
            "    import json\n"
            "    b=json.load(open(bundle_path))\n"
            "    return {'status':'pass','bundle_id':b['bundle_id'],"
            "'requested_promotion':b['requested_promotion']}\n",
            encoding="utf-8",
        )

        schema = root / "schema.json"
        schema.write_text('{"schema_version":"test"}\n', encoding="utf-8")
        artifact_root = root / "artifacts"
        artifact_root.mkdir()
        evidence = artifact_root / "terrain.json"
        evidence.write_text('{"terrain":"synthetic"}\n', encoding="utf-8")
        digest = sha256_file(evidence)

        bundle = root / "bundle.json"
        bundle.write_text(
            json.dumps(
                {
                    "schema_version": "ll009b.corridor-bundle.v1",
                    "bundle_id": "capsule-test-v1",
                    "study_id": "study-1",
                    "frame_contract_id": "frame-1",
                    "epoch_contract_id": "epoch-1",
                    "requested_promotion": "research_bundle",
                    "artifacts": [
                        {
                            "role": "terrain_pack_receipt",
                            "path": "terrain.json",
                            "sha256": digest,
                            "evidence_class": "synthetic",
                            "lineage_id": "terrain-v1",
                            "study_id": "study-1",
                            "frame_contract_id": "frame-1",
                            "epoch_contract_id": "epoch-1",
                            "synthetic_only": True,
                        }
                    ],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        out = root / "out" / "capsule-test-v1"
        first = build_capsule(bundle, schema, artifact_root, out, validator)
        second = build_capsule(bundle, schema, artifact_root, out, validator)
        assert first == second
        assert (out / "capsule_index.json").is_file()
        assert (out / "capsule_receipt.json").is_file()

        (out / "artifacts" / "terrain_pack_receipt" / "terrain.json").write_text(
            '{"terrain":"tampered"}\n', encoding="utf-8"
        )
        try:
            build_capsule(bundle, schema, artifact_root, out, validator)
        except CapsuleError:
            pass
        else:
            raise AssertionError("differing existing capsule must fail closed")

        bad_bundle = load_json(bundle)
        bad_bundle["bundle_id"] = "bad-path"
        bad_bundle["artifacts"][0]["path"] = "../terrain.json"
        bad_path = root / "bad.json"
        bad_path.write_bytes(canonical_json_bytes(bad_bundle))
        try:
            build_capsule(
                bad_path, schema, artifact_root, root / "bad-out", validator
            )
        except CapsuleError:
            pass
        else:
            raise AssertionError("path traversal must fail closed")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("configs/lunar_transport/ll009b_corridor_bundle_schema.json"),
    )
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--validator",
        type=Path,
        default=Path("scripts/validate_ll009b_corridor_bundle.py"),
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        print("LL-009C capsule self-test: PASS")
        return 0
    if args.bundle is None or args.artifact_root is None or args.output_dir is None:
        raise CapsuleError(
            "--bundle, --artifact-root, and --output-dir are required "
            "unless --self-test is used"
        )
    receipt = build_capsule(
        args.bundle,
        args.schema,
        args.artifact_root,
        args.output_dir,
        args.validator,
    )
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CapsuleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
