#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

RUNNER = Path(__file__).with_name("run_vart_world_creative_001_frozen_analysis.py")
EXPERIMENT_ID = "VART-WORLD-CREATIVE-001"
RECEIPT_REL = "_orchestrator/CONFIRMATORY_CAMPAIGN_RECEIPT.json"


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(obj) + b"\n")


def closure(root: Path) -> tuple[str, int]:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel == RECEIPT_REL:
            continue
        rows.append({"path": rel, "sha256": sha(path)})
    return hashlib.sha256(canonical(rows)).hexdigest(), len(rows)


def fixture(base: Path, name: str) -> tuple[Path, Path, Path, str, str, str]:
    root = base / f"evidence-{name}"
    root.mkdir()
    analysis_contract = root / "analysis_contract.json"
    metrics = root / "metric_definitions.json"
    dump(analysis_contract, {"schema": "analysis-test", "frozen": True})
    dump(metrics, {"schema": "metrics-test", "frozen": True})
    freeze = root / "confirmatory_freeze.json"
    dump(
        freeze,
        {
            "experiment_id": EXPERIMENT_ID,
            "frozen": True,
            "analysis_contract_sha256": sha(analysis_contract),
            "metric_definition_set_sha256": sha(metrics),
        },
    )
    (root / "trials").mkdir()
    (root / "trials" / "sealed.bin").write_bytes(b"sealed-evidence")
    cl, count = closure(root)
    receipt = root / RECEIPT_REL
    dump(
        receipt,
        {
            "schema": "symthaea.vart-world-creative-001.confirmatory-campaign-receipt.v1",
            "experiment_id": EXPERIMENT_ID,
            "status": "sealed",
            "freeze_sha256": sha(freeze),
            "trial_count": 64,
            "evidence_closure_sha256": cl,
            "evidence_file_count": count,
            "zero_peeking": True,
            "automatic_retry": False,
            "claim_authorized": False,
        },
    )

    program = base / f"analysis-{name}.py"
    program.write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "out=Path(os.environ['VART_ANALYSIS_OUTPUT_ROOT'])\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "(out/'result.json').write_text(json.dumps({'h1':'reported','h2':'reported','h3':'reported'})+'\\n')\n"
        "print('UNBLINDED_RESULTS_WRITTEN')\n",
        encoding="utf-8",
    )
    output = base / f"analysis-output-{name}"
    anchor = base / f"analysis-anchor-{name}.json"
    config = base / f"config-{name}.json"
    dump(
        config,
        {
            "schema": "symthaea.vart-world-creative-001.frozen-analysis-run.v1",
            "experiment_id": EXPERIMENT_ID,
            "evidence_root": str(root),
            "expected_freeze_sha256": sha(freeze),
            "expected_campaign_receipt_sha256": sha(receipt),
            "analysis_program_path": str(program),
            "expected_analysis_program_sha256": sha(program),
            "analysis_output_root": str(output),
            "preunblind_anchor_path": str(anchor),
            "analysis_argv": [sys.executable, "{analysis_program}"],
            "claim_authorized": False,
        },
    )
    return root, program, config, sha(freeze), sha(receipt), sha(program)


def run(config: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUNNER), str(config), *extra],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


with tempfile.TemporaryDirectory(prefix="vart-frozen-analysis-test-") as td:
    base = Path(td)

    # A1: complete sealed inputs dry-run without creating anchor/output.
    root, program, config, freeze_sha, receipt_sha, program_sha = fixture(base, "good")
    proc = run(config, "--dry-run")
    assert proc.returncode == 0, proc.stderr
    dry = json.loads(proc.stdout)
    assert dry["verdict"] == "ANALYSIS_DRY_RUN_READY"
    assert dry["freeze_sha256"] == freeze_sha
    assert dry["campaign_receipt_sha256"] == receipt_sha
    assert dry["analysis_program_sha256"] == program_sha
    cfg = json.loads(config.read_text())
    assert not Path(cfg["preunblind_anchor_path"]).exists()
    assert not Path(cfg["analysis_output_root"]).exists()

    # A2: actual run anchors before unblinding and seals result exactly once.
    proc = run(config)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["verdict"] == "FROZEN_ANALYSIS_EXECUTED_AND_SEALED"
    anchor = Path(cfg["preunblind_anchor_path"])
    output = Path(cfg["analysis_output_root"])
    assert anchor.is_file()
    assert (output / "result.json").is_file()
    assert (output / "ANALYSIS_RECEIPT.json").is_file()
    assert "UNBLINDED_RESULTS_WRITTEN" in (output / "analysis.stdout.txt").read_text()
    again = run(config)
    assert again.returncode == 2

    # A3: sealed evidence mutation is rejected before unblinding.
    root2, _, config2, _, _, _ = fixture(base, "tamper")
    (root2 / "trials" / "sealed.bin").write_bytes(b"changed-after-seal")
    proc = run(config2, "--dry-run")
    assert proc.returncode == 2
    assert "evidence closure changed" in proc.stderr

    # A4: analysis executable substitution is rejected before unblinding.
    _, program3, config3, _, _, _ = fixture(base, "program")
    program3.write_text(program3.read_text() + "# mutation\n", encoding="utf-8")
    proc = run(config3, "--dry-run")
    assert proc.returncode == 2
    assert "analysis program digest mismatch" in proc.stderr

print("PASS: frozen analysis pre-anchor, one-shot seal, evidence tamper rejection, program substitution rejection")
