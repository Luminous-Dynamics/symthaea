#!/usr/bin/env python3
"""Bind LL-009D content-addressed contracts to an immutable LL-009C corridor capsule.

LL-009E preserves the complete LL-009C capsule byte-for-byte inside a higher-level
immutable envelope, alongside the exact canonical study/frame/epoch contract bytes
that define the bundle IDs. It is provenance/reproducibility tooling only and does
not authorize launch, site selection, or operational use.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable

OUTER_INDEX_SCHEMA = "ll009e.contract-bound-capsule-index.v1"
OUTER_RECEIPT_SCHEMA = "ll009e.contract-bound-capsule-receipt.v1"
BINDING_RECEIPT_SCHEMA = "ll009e.contract-binding-receipt.v1"
CONTRACT_MANIFEST_SCHEMA = "ll009d.contract-manifest.v1"


class EnvelopeError(RuntimeError):
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
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EnvelopeError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EnvelopeError(f"expected JSON object: {path}")
    return value


def tree_manifest(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    return [
        {
            "path": p.relative_to(root).as_posix(),
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }
        for p in sorted(x for x in root.rglob("*") if x.is_file())
    ]


def aggregate_digest(records: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for record in sorted(records, key=lambda item: item["path"]):
        h.update(record["path"].encode("utf-8"))
        h.update(b"\0")
        h.update(record["sha256"].encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def compare_trees(a: Path, b: Path) -> bool:
    return tree_manifest(a) == tree_manifest(b)


def load_module(path: Path, name: str) -> Any:
    if not path.is_file():
        raise EnvelopeError(f"required tool not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise EnvelopeError(f"cannot load tool module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_canonical_file(path: Path, value: dict[str, Any], label: str) -> None:
    expected = canonical_json_bytes(value)
    actual = path.read_bytes()
    if actual != expected:
        raise EnvelopeError(f"{label} is not canonical LL-009D JSON: {path}")


def validate_contract_record(kind: str, path: Path) -> dict[str, Any]:
    record = load_json(path)
    expected_schema = f"ll009d.{kind}-contract.v1"
    if record.get("schema_version") != expected_schema:
        raise EnvelopeError(
            f"{kind} contract schema must be {expected_schema}: {path}"
        )
    contract = record.get("contract")
    if not isinstance(contract, dict):
        raise EnvelopeError(f"{kind} contract missing contract object")
    semantic_payload = canonical_json_bytes(contract)
    digest = sha256_bytes(semantic_payload)
    if record.get("semantic_sha256") != digest:
        raise EnvelopeError(f"{kind} contract semantic_sha256 mismatch")
    expected_id = f"{kind}-{digest}"
    if record.get("contract_id") != expected_id:
        raise EnvelopeError(
            f"{kind} contract_id mismatch expected={expected_id} "
            f"actual={record.get('contract_id')}"
        )
    require_canonical_file(path, record, f"{kind} contract")
    return {
        "kind": kind,
        "contract_id": expected_id,
        "semantic_sha256": digest,
        "file_sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "filename": path.name,
    }


def validate_contract_dir(contract_dir: Path) -> dict[str, Any]:
    if not contract_dir.is_dir():
        raise EnvelopeError(f"contract directory not found: {contract_dir}")
    records: dict[str, dict[str, Any]] = {}
    for kind in ("study", "frame", "epoch"):
        path = contract_dir / f"{kind}_contract.json"
        if not path.is_file():
            raise EnvelopeError(f"missing {kind} contract: {path}")
        records[kind] = validate_contract_record(kind, path)

    manifest_path = contract_dir / "contract_manifest.json"
    if not manifest_path.is_file():
        raise EnvelopeError(f"missing contract manifest: {manifest_path}")
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != CONTRACT_MANIFEST_SCHEMA:
        raise EnvelopeError(
            f"contract manifest schema must be {CONTRACT_MANIFEST_SCHEMA}"
        )
    require_canonical_file(manifest_path, manifest, "contract manifest")

    expected = {
        "study_id": records["study"]["contract_id"],
        "frame_contract_id": records["frame"]["contract_id"],
        "epoch_contract_id": records["epoch"]["contract_id"],
        "study_semantic_sha256": records["study"]["semantic_sha256"],
        "frame_semantic_sha256": records["frame"]["semantic_sha256"],
        "epoch_semantic_sha256": records["epoch"]["semantic_sha256"],
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise EnvelopeError(
                f"contract manifest {field} mismatch expected={value} "
                f"actual={manifest.get(field)}"
            )

    return {
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_bytes": manifest_path.stat().st_size,
        "records": records,
    }


def validate_bundle_ids(bundle_path: Path, contracts: dict[str, Any]) -> dict[str, str]:
    bundle = load_json(bundle_path)
    expected = {
        "study_id": contracts["records"]["study"]["contract_id"],
        "frame_contract_id": contracts["records"]["frame"]["contract_id"],
        "epoch_contract_id": contracts["records"]["epoch"]["contract_id"],
    }
    for field, value in expected.items():
        if bundle.get(field) != value:
            raise EnvelopeError(
                f"bundle {field} does not match LL-009D contract "
                f"expected={value} actual={bundle.get(field)}"
            )
    bundle_id = bundle.get("bundle_id")
    if not isinstance(bundle_id, str) or not bundle_id.strip():
        raise EnvelopeError("bundle missing non-empty bundle_id")
    return {
        "bundle_id": bundle_id,
        "study_id": expected["study_id"],
        "frame_contract_id": expected["frame_contract_id"],
        "epoch_contract_id": expected["epoch_contract_id"],
        "requested_promotion": str(bundle.get("requested_promotion", "")),
    }


def copy_verified_contracts(contract_dir: Path, target: Path) -> list[dict[str, Any]]:
    target.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    for filename in (
        "study_contract.json",
        "frame_contract.json",
        "epoch_contract.json",
        "contract_manifest.json",
    ):
        source = contract_dir / filename
        destination = target / filename
        shutil.copyfile(source, destination)
        before = sha256_file(source)
        after = sha256_file(destination)
        if before != after:
            raise EnvelopeError(f"contract copy changed bytes: {filename}")
        copied.append(
            {
                "path": f"contracts/{filename}",
                "bytes": destination.stat().st_size,
                "sha256": after,
            }
        )
    return copied


def build_envelope(
    bundle_path: Path,
    schema_path: Path,
    artifact_root: Path,
    contract_dir: Path,
    output_dir: Path,
    ll009c_builder_path: Path,
    ll009b_validator_path: Path,
) -> dict[str, Any]:
    contracts = validate_contract_dir(contract_dir)
    ids = validate_bundle_ids(bundle_path, contracts)
    ll009c = load_module(ll009c_builder_path, "ll009c_for_ll009e")
    if not hasattr(ll009c, "build_capsule"):
        raise EnvelopeError("LL-009C tool does not expose build_capsule")

    with tempfile.TemporaryDirectory(prefix="ll009e-") as tmp:
        stage = Path(tmp) / "envelope"
        stage.mkdir(parents=True)
        inner = stage / "ll009c"
        try:
            inner_receipt = ll009c.build_capsule(
                bundle_path,
                schema_path,
                artifact_root,
                inner,
                ll009b_validator_path,
            )
        except Exception as exc:
            raise EnvelopeError(f"LL-009C capsule build failed: {exc}") from exc

        inner_before = tree_manifest(inner)
        if not inner_before:
            raise EnvelopeError("LL-009C produced an empty capsule")
        inner_aggregate = aggregate_digest(inner_before)

        copied_contracts = copy_verified_contracts(contract_dir, stage / "contracts")

        binding = {
            "schema_version": BINDING_RECEIPT_SCHEMA,
            "status": "pass",
            "bundle_id": ids["bundle_id"],
            "study_id": ids["study_id"],
            "frame_contract_id": ids["frame_contract_id"],
            "epoch_contract_id": ids["epoch_contract_id"],
            "requested_promotion": ids["requested_promotion"],
            "contract_manifest_sha256": contracts["manifest_sha256"],
            "inner_capsule_aggregate_sha256": inner_aggregate,
            "inner_capsule_receipt": inner_receipt,
            "non_claim": (
                "Contract binding establishes research provenance only; it is not "
                "site selection, corridor qualification, navigation qualification, "
                "or release authority."
            ),
        }
        binding_payload = canonical_json_bytes(binding)
        (stage / "contract_binding_receipt.json").write_bytes(binding_payload)

        inner_after = tree_manifest(inner)
        if inner_after != inner_before:
            raise EnvelopeError("LL-009C inner capsule changed while binding contracts")

        index = {
            "schema_version": OUTER_INDEX_SCHEMA,
            "bundle_id": ids["bundle_id"],
            "study_id": ids["study_id"],
            "frame_contract_id": ids["frame_contract_id"],
            "epoch_contract_id": ids["epoch_contract_id"],
            "requested_promotion": ids["requested_promotion"],
            "inner_capsule_aggregate_sha256": inner_aggregate,
            "contract_manifest_sha256": contracts["manifest_sha256"],
            "contract_files": copied_contracts,
            "contract_binding_receipt_sha256": sha256_bytes(binding_payload),
            "ll009c_builder_sha256": sha256_file(ll009c_builder_path),
            "ll009b_validator_sha256": sha256_file(ll009b_validator_path),
            "ll009e_builder_sha256": sha256_file(Path(__file__).resolve()),
            "non_claims": [
                "LL-009E binds exact study definitions to an immutable research capsule.",
                "It does not establish corridor safety, site selection, economic superiority, or launch authority.",
            ],
        }
        index_payload = canonical_json_bytes(index)
        (stage / "ll009e_index.json").write_bytes(index_payload)

        pre_receipt = tree_manifest(stage)
        receipt = {
            "schema_version": OUTER_RECEIPT_SCHEMA,
            "status": "pass",
            "bundle_id": ids["bundle_id"],
            "study_id": ids["study_id"],
            "ll009e_index_sha256": sha256_bytes(index_payload),
            "content_aggregate_sha256": aggregate_digest(pre_receipt),
            "file_count_excluding_receipt": len(pre_receipt),
            "non_claim": (
                "LL-009E pass means contract-bound immutable evidence composition "
                "only; it never authorizes a physical launch."
            ),
        }
        (stage / "ll009e_receipt.json").write_bytes(canonical_json_bytes(receipt))

        if tree_manifest(inner) != inner_before:
            raise EnvelopeError("inner LL-009C capsule was mutated by LL-009E")

        if output_dir.exists():
            if compare_trees(stage, output_dir):
                return receipt
            raise EnvelopeError(
                f"refusing to overwrite differing immutable envelope {output_dir}; "
                "choose a new lineage/output path"
            )

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(stage, output_dir)
        if not compare_trees(stage, output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
            raise EnvelopeError("LL-009E envelope copy verification failed")

    return receipt


def make_test_contract(kind: str, contract: dict[str, Any]) -> dict[str, Any]:
    digest = sha256_bytes(canonical_json_bytes(contract))
    return {
        "schema_version": f"ll009d.{kind}-contract.v1",
        "contract_id": f"{kind}-{digest}",
        "semantic_sha256": digest,
        "contract": contract,
    }


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="ll009e-selftest-") as tmp:
        root = Path(tmp)
        contracts_dir = root / "contracts"
        contracts_dir.mkdir()

        semantic = {
            "study": {"name": "synthetic-study", "architectures": ["rover", "ballistic"]},
            "frame": {"body_fixed": "FRAME_A", "inertial": "J2000"},
            "epoch": {"timescale": "TDB", "epoch": "synthetic-0"},
        }
        records = {
            kind: make_test_contract(kind, semantic[kind])
            for kind in ("study", "frame", "epoch")
        }
        for kind, record in records.items():
            (contracts_dir / f"{kind}_contract.json").write_bytes(
                canonical_json_bytes(record)
            )
        manifest = {
            "schema_version": CONTRACT_MANIFEST_SCHEMA,
            "input_filename": "synthetic.json",
            "input_sha256": "0" * 64,
            "study_id": records["study"]["contract_id"],
            "frame_contract_id": records["frame"]["contract_id"],
            "epoch_contract_id": records["epoch"]["contract_id"],
            "study_semantic_sha256": records["study"]["semantic_sha256"],
            "frame_semantic_sha256": records["frame"]["semantic_sha256"],
            "epoch_semantic_sha256": records["epoch"]["semantic_sha256"],
            "non_claim": "synthetic",
        }
        (contracts_dir / "contract_manifest.json").write_bytes(
            canonical_json_bytes(manifest)
        )

        artifact_root = root / "artifacts"
        artifact_root.mkdir()
        evidence = artifact_root / "terrain.json"
        evidence.write_text('{"synthetic":true}\n', encoding="utf-8")

        bundle = {
            "schema_version": "ll009b.corridor-bundle.v1",
            "bundle_id": "synthetic-envelope",
            "study_id": records["study"]["contract_id"],
            "frame_contract_id": records["frame"]["contract_id"],
            "epoch_contract_id": records["epoch"]["contract_id"],
            "requested_promotion": "research_bundle",
            "artifacts": [],
        }
        bundle_path = root / "bundle.json"
        bundle_path.write_bytes(canonical_json_bytes(bundle))
        schema_path = root / "schema.json"
        schema_path.write_text('{"schema_version":"synthetic"}\n', encoding="utf-8")

        ll009b = root / "validator.py"
        ll009b.write_text(
            "def validate_bundle(bundle_path, schema_path, artifact_root):\n"
            "    import json\n"
            "    b=json.load(open(bundle_path))\n"
            "    return {'status':'pass','bundle_id':b['bundle_id'],"
            "'requested_promotion':b['requested_promotion']}\n",
            encoding="utf-8",
        )
        ll009c = root / "capsule_builder.py"
        ll009c.write_text(
            "from pathlib import Path\n"
            "import hashlib, json\n"
            "def build_capsule(bundle_path, schema_path, artifact_root, output_dir, validator_path):\n"
            "    output_dir.mkdir(parents=True, exist_ok=False)\n"
            "    b=json.load(open(bundle_path))\n"
            "    payload=(json.dumps({'bundle_id':b['bundle_id'],'status':'pass'},"
            "sort_keys=True,indent=2)+\"\\n\").encode()\n"
            "    (output_dir/'capsule_index.json').write_bytes(payload)\n"
            "    receipt={'status':'pass','capsule_index_sha256':hashlib.sha256(payload).hexdigest()}\n"
            "    (output_dir/'capsule_receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\\n')\n"
            "    return receipt\n",
            encoding="utf-8",
        )

        out = root / "out" / "synthetic-envelope"
        first = build_envelope(
            bundle_path,
            schema_path,
            artifact_root,
            contracts_dir,
            out,
            ll009c,
            ll009b,
        )
        second = build_envelope(
            bundle_path,
            schema_path,
            artifact_root,
            contracts_dir,
            out,
            ll009c,
            ll009b,
        )
        assert first == second
        assert (out / "ll009c" / "capsule_receipt.json").is_file()
        assert (out / "contracts" / "study_contract.json").is_file()
        assert (out / "ll009e_receipt.json").is_file()

        bad_bundle = json.loads(json.dumps(bundle))
        bad_bundle["study_id"] = "study-" + "f" * 64
        bad_bundle_path = root / "bad-bundle.json"
        bad_bundle_path.write_bytes(canonical_json_bytes(bad_bundle))
        try:
            build_envelope(
                bad_bundle_path,
                schema_path,
                artifact_root,
                contracts_dir,
                root / "bad-out",
                ll009c,
                ll009b,
            )
        except EnvelopeError:
            pass
        else:
            raise AssertionError("bundle/contract ID mismatch must fail")

        study_path = contracts_dir / "study_contract.json"
        original = study_path.read_bytes()
        tampered = load_json(study_path)
        tampered["contract"]["name"] = "tampered"
        study_path.write_bytes(canonical_json_bytes(tampered))
        try:
            validate_contract_dir(contracts_dir)
        except EnvelopeError:
            pass
        else:
            raise AssertionError("semantic mutation under stale ID must fail")
        study_path.write_bytes(original)

        (out / "contracts" / "study_contract.json").write_text(
            '{"tampered":true}\n', encoding="utf-8"
        )
        try:
            build_envelope(
                bundle_path,
                schema_path,
                artifact_root,
                contracts_dir,
                out,
                ll009c,
                ll009b,
            )
        except EnvelopeError:
            pass
        else:
            raise AssertionError("differing existing envelope must fail closed")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("configs/lunar_transport/ll009b_corridor_bundle_schema.json"),
    )
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--contracts-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--ll009c-builder",
        type=Path,
        default=Path("scripts/build_ll009c_corridor_capsule.py"),
    )
    parser.add_argument(
        "--ll009b-validator",
        type=Path,
        default=Path("scripts/validate_ll009b_corridor_bundle.py"),
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        print("LL-009E contract-bound capsule self-test: PASS")
        return 0
    required = {
        "--bundle": args.bundle,
        "--artifact-root": args.artifact_root,
        "--contracts-dir": args.contracts_dir,
        "--output-dir": args.output_dir,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise EnvelopeError(f"missing required arguments: {', '.join(missing)}")
    receipt = build_envelope(
        args.bundle,
        args.schema,
        args.artifact_root,
        args.contracts_dir,
        args.output_dir,
        args.ll009c_builder,
        args.ll009b_validator,
    )
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EnvelopeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
