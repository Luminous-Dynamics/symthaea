#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA = "symthaea.vart-benchmark-firewall.v1"
HEX = set("0123456789abcdef")


class Reject(RuntimeError):
    pass


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise Reject(f"missing manifest: {path}") from exc
    except json.JSONDecodeError as exc:
        raise Reject(f"invalid JSON: {exc}") from exc


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise Reject(msg)


def sha_list(value: Any, name: str) -> list[str]:
    require(isinstance(value, list), f"{name} must be an array")
    out: list[str] = []
    for item in value:
        require(
            isinstance(item, str)
            and len(item) == 64
            and all(c in HEX for c in item),
            f"{name} contains non-sha256 value",
        )
        out.append(item)
    require(len(out) == len(set(out)), f"{name} contains duplicates")
    return out


def empty_secret_list(obj: dict[str, Any], key: str) -> None:
    value = obj.get(key)
    require(isinstance(value, list), f"vart.{key} must be an array")
    require(not value, f"HIDDEN_BENCHMARK_SECRET_EXPOSED: vart.{key}")


def outside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return False
    except ValueError:
        return True


def verify(manifest_path: Path, repo_root: Path) -> dict[str, Any]:
    data = read_json(manifest_path)
    require(isinstance(data, dict) and data.get("schema") == SCHEMA, "unexpected firewall schema")
    require(data.get("status") in {"prospective_committed", "frozen"}, "firewall is not prospectively committed")
    require(data.get("claim_authorized") is False, "firewall manifest cannot authorize a scientific claim")

    dev_tag = data.get("development_domain_tag")
    vart_tag = data.get("evaluation_domain_tag")
    require(isinstance(dev_tag, str) and dev_tag, "development domain tag missing")
    require(isinstance(vart_tag, str) and vart_tag, "evaluation domain tag missing")
    require(dev_tag != vart_tag, "DOMAIN_SEPARATION_FAILURE: DEVART and VART tags are identical")

    dev = data.get("devart")
    vart = data.get("vart")
    reuse = data.get("reuse_policy")
    require(isinstance(dev, dict) and isinstance(vart, dict) and isinstance(reuse, dict), "firewall sections missing")
    require(dev.get("used_as_development_feedback") is True, "DEVART domain must be explicitly development-visible")
    require(vart.get("revealed") is False, "PRELAUNCH_REVEAL_FORBIDDEN: hidden VART material already revealed")
    require(reuse.get("spent_commitments_may_not_return_to_hidden_confirmatory_use") is True, "spent benchmark reuse must be forbidden")

    dev_fixtures = set(sha_list(dev.get("fixture_commitments_sha256"), "devart.fixture_commitments_sha256"))
    dev_seeds = set(sha_list(dev.get("seed_commitments_sha256"), "devart.seed_commitments_sha256"))
    vart_fixtures = set(sha_list(vart.get("fixture_commitments_sha256"), "vart.fixture_commitments_sha256"))
    vart_seeds = set(sha_list(vart.get("seed_commitments_sha256"), "vart.seed_commitments_sha256"))
    prior_fixtures = set(sha_list(reuse.get("prior_vart_fixture_commitments_sha256"), "reuse_policy.prior_vart_fixture_commitments_sha256"))
    prior_seeds = set(sha_list(reuse.get("prior_vart_seed_commitments_sha256"), "reuse_policy.prior_vart_seed_commitments_sha256"))

    require(dev_fixtures.isdisjoint(vart_fixtures), "DEVART_VART_FIXTURE_OVERLAP")
    require(dev_seeds.isdisjoint(vart_seeds), "DEVART_VART_SEED_OVERLAP")
    require(vart_fixtures.isdisjoint(prior_fixtures), "SPENT_VART_FIXTURE_REUSE")
    require(vart_seeds.isdisjoint(prior_seeds), "SPENT_VART_SEED_REUSE")

    fixture_count = vart.get("fixture_count")
    seed_count = vart.get("seed_count")
    require(isinstance(fixture_count, int) and fixture_count > 0, "vart.fixture_count must be positive")
    require(isinstance(seed_count, int) and seed_count > 0, "vart.seed_count must be positive")
    require(fixture_count == len(vart_fixtures), "VART_FIXTURE_COUNT_MISMATCH")
    require(seed_count == len(vart_seeds), "VART_SEED_COUNT_MISMATCH")

    for key in ("generator_policy_sha256", "scoring_contract_sha256", "custodian_receipt_sha256"):
        sha_list([vart.get(key)], f"vart.{key}")

    hidden_root_raw = vart.get("hidden_material_root")
    require(isinstance(hidden_root_raw, str) and hidden_root_raw, "hidden_material_root missing")
    hidden_root = Path(hidden_root_raw).expanduser()
    require(hidden_root.is_absolute(), "hidden_material_root must be absolute")
    require(outside(hidden_root, repo_root), "HIDDEN_MATERIAL_INSIDE_REPOSITORY")

    for key in ("plaintext_fixture_ids", "plaintext_seeds", "expected_solutions", "target_defects", "trap_labels"):
        empty_secret_list(vart, key)

    return {
        "verdict": "VART_BENCHMARK_FIREWALL_PASS",
        "firewall_id": data.get("firewall_id"),
        "devart_fixture_count": len(dev_fixtures),
        "vart_fixture_count": len(vart_fixtures),
        "vart_seed_count": len(vart_seeds),
        "domain_separation": "PASS",
        "development_overlap": "PASS",
        "spent_benchmark_reuse": "PASS",
        "prelaunch_secret_exposure": "PASS",
        "claim_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the Symthaea DEVART/VART benchmark firewall")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = verify(args.manifest.resolve(), args.repo_root.resolve())
    except (Reject, OSError, ValueError) as exc:
        payload = {"verdict": "VART_BENCHMARK_FIREWALL_REJECT", "detail": str(exc), "claim_authorized": False}
        if args.json:
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        else:
            print(f"REJECT: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    else:
        print(result["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
