#!/usr/bin/env python3
"""Build deterministic transparent REL-005A qualification capsule v2.

Authority: AttestationInputOnly. Packages already-authoritative metric-free
receipts plus two audit receipts. It does not interpret scientific metrics.
"""
from __future__ import annotations
import argparse, hashlib, io, json, pathlib, tarfile, tempfile

REQUIRED = {
    "predicate-contract-receipt.json",
    "execution-v3-receipt.json",
    "observation-seal-v3.json",
    "comparison-only-qualification-receipt.json",
    "qualification-input-manifest.json",
    "qualification-assembly-receipt.json",
    "qualification-only.json",
    "sealed-manifest-replay-receipt.json",
    "authority-receipt-extraction.json",
}
GENERATED = "qualification-capsule-manifest.json"


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.as_posix().encode())


def validate(root: pathlib.Path) -> dict[str, pathlib.Path]:
    by: dict[str, pathlib.Path] = {}
    for path in files(root):
        rel = path.relative_to(root)
        if len(rel.parts) != 1:
            raise ValueError(f"nested capsule input forbidden: {rel}")
        if path.name in by:
            raise ValueError(f"duplicate capsule basename: {path.name}")
        by[path.name] = path
    if set(by) != REQUIRED:
        raise ValueError(f"capsule census mismatch: got={sorted(by)} expected={sorted(REQUIRED)}")
    return by


def manifest(stage: pathlib.Path, by: dict[str, pathlib.Path]) -> pathlib.Path:
    value = {
        "schema": "symthaea.rel.qualification-capsule-manifest.v2",
        "authority": "AttestationInputOnly",
        "relation": "REL-005A",
        "canonicalization": {
            "compression": "none", "format": "ustar", "sort": "bytewise basename ascending",
            "mtime": 0, "uid": 0, "gid": 0, "uname": "", "gname": "",
            "file_mode": "0644", "pax_headers": False,
        },
        "files": [
            {"basename": n, "byte_length": by[n].stat().st_size, "sha256": sha(by[n])}
            for n in sorted(by, key=lambda s: s.encode())
        ],
        "audit_receipts_included": [
            "sealed-manifest-replay-receipt.json", "authority-receipt-extraction.json"
        ],
        "raw_observation_included": False,
        "execution_logs_included": False,
        "detailed_predicate_values_included": False,
        "claims": {
            "attestation_created": False, "qualification_completed": False,
            "rel_005a_qualified": False, "scientific_pass": False, "scientific_fail": False,
        },
    }
    out = stage / GENERATED
    out.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return out


def add(tf: tarfile.TarFile, path: pathlib.Path, name: str) -> None:
    data = path.read_bytes(); info = tarfile.TarInfo(name=name)
    info.size=len(data); info.mtime=0; info.uid=0; info.gid=0; info.uname=""; info.gname=""; info.mode=0o644
    tf.addfile(info, io.BytesIO(data))


def build(root: pathlib.Path, output: pathlib.Path) -> dict[str, object]:
    source = validate(root)
    with tempfile.TemporaryDirectory() as td:
        stage = pathlib.Path(td); staged: dict[str,pathlib.Path] = {}
        for name in sorted(source, key=lambda s:s.encode()):
            target=stage/name; target.write_bytes(source[name].read_bytes()); staged[name]=target
        manifest(stage, staged)
        names=sorted([*staged, GENERATED], key=lambda s:s.encode())
        with tarfile.open(output, "w", format=tarfile.USTAR_FORMAT) as tf:
            for name in names: add(tf, stage/name, name)
    return {
        "schema": "symthaea.rel.attestation-input-receipt.v2",
        "authority": "AttestationInputOnly",
        "capsule_path": output.name,
        "capsule_sha256": sha(output),
        "capsule_byte_length": output.stat().st_size,
        "member_count": len(REQUIRED)+1,
        "members": names,
        "audit_transparency": True,
        "claims": {
            "attestation_created": False, "qualification_completed": False,
            "rel_005a_qualified": False, "scientific_pass": False, "scientific_fail": False,
        },
    }


def self_test() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as td:
        root=pathlib.Path(td); a=root/'a'; b=root/'b'; a.mkdir(); b.mkdir()
        ordered=sorted(REQUIRED,key=lambda s:s.encode())
        for i,n in enumerate(ordered): (a/n).write_text(f"{n}\n{i}\n")
        for i,n in reversed(list(enumerate(ordered))): (b/n).write_text(f"{n}\n{i}\n")
        ra=build(a,root/'a.tar'); rb=build(b,root/'b.tar')
        if ra['capsule_sha256'] != rb['capsule_sha256']: raise ValueError('determinism failure')
        missing=root/'missing'; missing.mkdir()
        for n in ordered[:-1]: (missing/n).write_text('x\n')
        try: build(missing,root/'missing.tar')
        except ValueError: pass
        else: raise ValueError('missing audit/input accepted')
    return {
        "schema":"symthaea.rel.attestation-input-self-test.v2",
        "authority":"AttestationContractOnly",
        "deterministic_archive":True,
        "audit_receipts_required":True,
        "missing_member_rejected":True,
    }


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--input-dir',type=pathlib.Path); p.add_argument('--output',type=pathlib.Path); p.add_argument('--self-test',action='store_true'); a=p.parse_args()
    if a.self_test: print(json.dumps(self_test(),indent=2,sort_keys=True)); return
    if a.input_dir is None or a.output is None: raise SystemExit('--input-dir and --output required')
    print(json.dumps(build(a.input_dir,a.output),indent=2,sort_keys=True))

if __name__=='__main__': main()
