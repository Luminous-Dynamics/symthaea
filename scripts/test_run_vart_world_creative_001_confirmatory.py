#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

RUNNER = Path(__file__).with_name("run_vart_world_creative_001_confirmatory.py").resolve()
EXPERIMENT_ID = "VART-WORLD-CREATIVE-001"


def sh(*argv: str, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def git(repo: Path, *args: str) -> str:
    p = sh("git", "-C", str(repo), *args, cwd=repo)
    assert p.returncode == 0, p.stderr
    return p.stdout.strip()


def dump(path: Path, obj: object) -> str:
    path.write_text(json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory() -> dict:
    trials: list[dict] = []
    fixtures = ["ordinary", "PrettyTrap", "LocalOptimum", "HiddenDependency", "DelayedConsequence", "CounterfactualDecoy", "Path", "Plaza"]
    for i, fixture in enumerate(fixtures):
        cluster = f"{i+1:064x}"[-64:]
        for pidx, policy in enumerate(["full_symthaea", "random_valid", "heuristic"]):
            lineage = f"{1000 + i*10 + pidx:064x}"[-64:]
            trials.append({"trial_id": f"A-{i}-{policy}-r0", "subcampaign": "001A", "policy": policy,
                           "fixture": fixture, "seed": 100000+i, "world_cluster_sha256": cluster,
                           "world_lineage_sha256": lineage, "revision_index": 0})
            if policy == "full_symthaea":
                for revision in (1, 2, 3):
                    trials.append({"trial_id": f"A-{i}-{policy}-r{revision}", "subcampaign": "001A", "policy": policy,
                                   "fixture": fixture, "seed": 100000+i, "world_cluster_sha256": cluster,
                                   "world_lineage_sha256": lineage, "revision_index": revision})
    for i in range(8):
        cluster = f"{9000+i:064x}"[-64:]
        for pidx, policy in enumerate(["full_symthaea", "no_reality_ledger_context"]):
            trials.append({"trial_id": f"B-{i}-{policy}-r0", "subcampaign": "001B", "policy": policy,
                           "fixture": "MemoryTrap", "seed": 200000+i, "world_cluster_sha256": cluster,
                           "world_lineage_sha256": f"{9100+i*10+pidx:064x}"[-64:], "revision_index": 0})
    assert len(trials) == 64
    # Deterministic frozen permutation for the test. The production inventory supplies its own preregistered order.
    permuted = trials[::2] + trials[1::2]
    for order, row in enumerate(permuted):
        row["run_order"] = order
    return {"schema": "symthaea.vart-world-creative-001.confirmatory-inventory.v3",
            "experiment_id": EXPERIMENT_ID, "trials": permuted, "trial_ids": [r["trial_id"] for r in permuted]}


def freeze(inv_sha: str) -> dict:
    return {"schema": "symthaea.vart-world-creative-001.confirmatory-freeze.v3",
            "experiment_id": EXPERIMENT_ID, "frozen": True, "trial_inventory_sha256": inv_sha,
            "analysis_contract_sha256": None, "metric_definition_set_sha256": None,
            "constraints": {"scalar_world_quality_forbidden": True, "zero_peeking_enforced": True},
            "claim": {"authorized": False}}


with tempfile.TemporaryDirectory(prefix="vart-confirmatory-runner-") as td:
    root = Path(td)
    subject = root / "subject"
    subject.mkdir()
    git(subject, "init", "-q")
    git(subject, "config", "user.email", "test@example.invalid")
    git(subject, "config", "user.name", "VART test")

    emitter = subject / "emit_trial.py"
    emitter.write_text(
        """#!/usr/bin/env python3
import argparse,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--trial-id',required=True); p.add_argument('--output-root',required=True); a=p.parse_args()
if os.environ.get('FAIL_TRIAL_ID') == a.trial_id:
    print('synthetic outcome must remain private', file=sys.stdout); raise SystemExit(9)
t=Path(a.output_root)/'trials'/a.trial_id; t.mkdir(parents=True)
(t/'manifest.json').write_text(json.dumps({'trial_id':a.trial_id})+'\\n')
print('synthetic outcome magnitude=123.456')
""", encoding="utf-8")
    git(subject, "add", "emit_trial.py")
    git(subject, "commit", "-qm", "test subject")
    subject_head = git(subject, "rev-parse", "HEAD")
    subject_tree = git(subject, "rev-parse", "HEAD^{tree}")

    instrument_root = RUNNER.parent.parent
    instrument_head = git(instrument_root, "rev-parse", "HEAD")
    instrument_tree = git(instrument_root, "rev-parse", "HEAD^{tree}")

    inv = root / "inventory.json"
    frz = root / "freeze.json"
    inv_obj = inventory()
    inv_sha = dump(inv, inv_obj)
    freeze_obj = freeze(inv_sha)
    analysis = root / "analysis.json"
    metrics = root / "metrics.json"
    analysis.write_text('{}\n', encoding="utf-8")
    metrics.write_text('{}\n', encoding="utf-8")
    freeze_obj["analysis_contract_sha256"] = hashlib.sha256(analysis.read_bytes()).hexdigest()
    freeze_obj["metric_definition_set_sha256"] = hashlib.sha256(metrics.read_bytes()).hexdigest()
    freeze_sha = dump(frz, freeze_obj)

    fake_verifier = root / "fake_verifier.py"
    fake_verifier.write_text("import json; print(json.dumps({'verdict':'ACCEPT'}))\n", encoding="utf-8")

    def make_config(evidence: Path) -> Path:
        cfg = root / f"config-{evidence.name}.json"
        dump(cfg, {
            "schema": "symthaea.vart-world-creative-001.confirmatory-run.v1",
            "experiment_id": EXPERIMENT_ID,
            "runtime_root": str(subject),
            "evidence_root": str(evidence),
            "freeze_path": str(frz),
            "trial_inventory_path": str(inv),
            "expected_freeze_sha256": freeze_sha,
            "expected_subject_source": {"head": subject_head, "tree": subject_tree},
            "expected_instrument_source": {"head": instrument_head, "tree": instrument_tree},
            "qualified_verifier_path": str(fake_verifier),
            "contract_inputs": {"analysis_contract": str(analysis), "metric_definitions": str(metrics)},
            "runtime_argv": [sys.executable, str(emitter), "--trial-id", "{trial_id}", "--output-root", "{output_root}"],
            "claim_authorized": False,
        })
        return cfg

    # Dry run is side-effect-free and validates all frozen identities/schedule.
    evidence = root / "evidence-success"
    cfg = make_config(evidence)
    p = sh(sys.executable, str(RUNNER), str(cfg), "--dry-run", cwd=instrument_root)
    assert p.returncode == 0, (p.stdout, p.stderr)
    dry = json.loads(p.stdout)
    assert dry["verdict"] == "CONFIRMATORY_DRY_RUN_READY"
    assert dry["trial_count"] == 64
    assert not evidence.exists()

    # Full synthetic run seals all 64. Runtime outcome text must not appear on the console.
    p = sh(sys.executable, str(RUNNER), str(cfg), cwd=instrument_root)
    assert p.returncode == 0, (p.stdout, p.stderr)
    assert "magnitude=123.456" not in p.stdout
    assert (evidence / "_orchestrator" / "CONFIRMATORY_CAMPAIGN_RECEIPT.json").is_file()
    assert len(list((evidence / "trials").iterdir())) == 64
    final = json.loads(p.stdout.splitlines()[-1])
    assert final["verdict"] == "CONFIRMATORY_CAMPAIGN_SEALED_AND_VERIFIED"
    assert final["claim_authorized"] is False

    # Injected operational crash: stop at the frozen trial, preserve abort evidence, and do not retry.
    evidence_fail = root / "evidence-fail"
    cfg_fail = make_config(evidence_fail)
    first = sorted(inv_obj["trials"], key=lambda r: r["run_order"])[0]["trial_id"]
    env = os.environ.copy(); env["FAIL_TRIAL_ID"] = first
    p = sh(sys.executable, str(RUNNER), str(cfg_fail), cwd=instrument_root, env=env)
    assert p.returncode == 2
    assert "magnitude=123.456" not in p.stdout
    abort = json.loads((evidence_fail / "_orchestrator" / "CAMPAIGN_ABORT_RECEIPT.json").read_text())
    assert abort["failed_trial_id"] == first
    assert abort["attempted_trial_count"] == 1
    assert abort["automatic_retry"] is False

print("PASS: confirmatory runner dry-run + 64-trial seal + zero-peeking + fail-closed no-retry abort")
