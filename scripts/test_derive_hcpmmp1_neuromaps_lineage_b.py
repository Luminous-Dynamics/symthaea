#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import stat
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("derive_hcpmmp1_neuromaps_lineage_b.py")
SPEC = importlib.util.spec_from_file_location("lineage_b", MODULE_PATH)
assert SPEC and SPEC.loader
m = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m
SPEC.loader.exec_module(m)


def sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def bd(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def areas() -> list[str]:
    return [f"A{i:03d}" for i in range(1, 181)]


def gifti(hemi: str, encoding: str = "ASCII", *, mask: bool = False) -> bytes:
    if mask:
        names = ["???", "cortex"]
        values = [1] * m.VERTICES_PER_HEMISPHERE
        for i in range(0, m.VERTICES_PER_HEMISPHERE, 997):
            values[i] = 0
    else:
        prefix = "L_" if hemi == "left" else "R_"
        names = ["???"] + [f"{prefix}{x}_ROI" for x in areas()]
        values = [1 + (i % 180) for i in range(m.VERTICES_PER_HEMISPHERE)]
        for i in range(0, m.VERTICES_PER_HEMISPHERE, 991):
            values[i] = 0
    if encoding == "ASCII":
        payload = " ".join(map(str, values))
    else:
        raw = struct.pack("<" + "i" * len(values), *values)
        if encoding == "GZipBase64Binary":
            raw = zlib.compress(raw)
        payload = base64.b64encode(raw).decode()
    labels = "\n".join(f'<Label Key="{i}">{name}</Label>' for i, name in enumerate(names))
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<GIFTI Version="1.0" NumberOfDataArrays="1"><LabelTable>{labels}</LabelTable>
<DataArray Intent="NIFTI_INTENT_LABEL" DataType="NIFTI_TYPE_INT32" ArrayIndexingOrder="RowMajorOrder"
Dimensionality="1" Dim0="{m.VERTICES_PER_HEMISPHERE}" Encoding="{encoding}" Endian="LittleEndian"
ExternalFileName="" ExternalFileOffset="0"><Data>{payload}</Data></DataArray></GIFTI>'''.encode()


def method() -> dict:
    return json.loads((Path(__file__).parents[1] / "data/neuroscience/hcpmmp1_neuromaps_transform_method_v1.json").read_text())


def area_doc(order: list[str] | None = None) -> dict:
    return {
        "schema": m.AREA_SCHEMA, "atlas": "HCP-MMP1.0/Glasser360", "hemisphere_area_count": 180,
        "source": {"repository": "synthetic/test", "commit": "0" * 40, "blob_sha": "1" * 40, "path": "x", "purpose": "test"},
        "areas": order or areas(),
    }


class Tests(unittest.TestCase):
    def fixture(self, td: Path):
        mp = td / "method.json"; mp.write_text(json.dumps(method(), sort_keys=True))
        ap = td / "areas.json"; ap.write_text(json.dumps(area_doc(), sort_keys=True))
        lf, rf = td / "left.label.gii", td / "right.label.gii"
        lf.write_bytes(gifti("left", "GZipBase64Binary")); rf.write_bytes(gifti("right", "GZipBase64Binary"))
        wb = td / "wb_command"
        wb.write_text(r'''#!/bin/sh
if [ "$1" = "-version" ]; then
  printf 'Connectome Workbench fake-v1\n'
  exit 0
fi
if [ "$1" = "-label-resample" ]; then
  out="$6"
else
  for out do :; done
fi
case "$out" in
  *left*) cp "$FAKE_LEFT_GII" "$out" ;;
  *) cp "$FAKE_RIGHT_GII" "$out" ;;
esac
''')
        wb.chmod(wb.stat().st_mode | stat.S_IXUSR)
        inputs = {}
        for role in method()["required_inputs"]:
            p = td / role
            p.write_bytes(gifti("left" if "left" in role else "right", mask=True) if "fsaverage10k_" in role and "medialwall" in role else ("synthetic-" + role).encode())
            inputs[role] = {"path": str(p), "sha256": sha(p)}
        run = {
            "schema": m.RUN_SCHEMA, "method_manifest_digest": sha(mp), "execution_id": "run-a",
            "authorization_reference": "synthetic-only",
            "workbench": {"path": str(wb), "sha256": sha(wb), "version_output_sha256": bd(b"Connectome Workbench fake-v1\n")},
            "inputs": inputs,
        }
        rp = td / "run.json"; rp.write_text(json.dumps(run, sort_keys=True))
        return mp, rp, ap, lf, rf

    def test_method_boundary(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"m.json"; p.write_text(json.dumps(method()))
            d=m.load_method_manifest(p)
            self.assertFalse(d["source_atlas"]["automatic_acquisition_permitted"])
            self.assertFalse(d["independence_contract"]["independence_established_by_this_manifest"])

    def test_authority_escalation_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            d=method(); d["independence_contract"]["independence_established_by_this_manifest"]=True
            p=Path(t)/"m.json"; p.write_text(json.dumps(d))
            with self.assertRaises(m.DerivationError): m.load_method_manifest(p)

    def test_ascii_base64_compressed_gifti(self):
        with tempfile.TemporaryDirectory() as t:
            td=Path(t)
            for enc in ("ASCII","Base64Binary","GZipBase64Binary"):
                p=td/f"{enc}.gii"; p.write_bytes(gifti("left",enc))
                v, table=m.parse_label_gifti(p); self.assertEqual(len(v),10242); self.assertEqual(table[1],"L_A001_ROI")

    def test_external_payload_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"x.gii"; p.write_bytes(gifti("left").replace(b'ExternalFileName=""',b'ExternalFileName="x.bin"'))
            with self.assertRaises(m.DerivationError): m.parse_label_gifti(p)

    def test_wrong_vertex_count_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"x.gii"; p.write_bytes(gifti("left").replace(b'Dim0="10242"',b'Dim0="10241"'))
            with self.assertRaises(m.DerivationError): m.parse_label_gifti(p)

    def test_semantic_rules(self):
        aset=set(areas())
        self.assertIsNone(m.normalize_label("???","left",aset))
        with self.assertRaises(m.DerivationError): m.normalize_label("unknown","left",aset)
        with self.assertRaises(m.DerivationError): m.normalize_label("R_A001_ROI","left",aset)

    def test_missing_role_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            td=Path(t); mp,rp,_,_,_=self.fixture(td); d=json.loads(rp.read_text()); d["inputs"].pop(next(iter(d["inputs"]))); rp.write_text(json.dumps(d))
            with self.assertRaises(m.DerivationError): m.load_run_manifest(rp,m.load_method_manifest(mp),mp)

    def test_hash_mismatch_fails_before_transform(self):
        with tempfile.TemporaryDirectory() as t:
            td=Path(t); mp,rp,ap,lf,rf=self.fixture(td); d=json.loads(rp.read_text()); role=next(iter(d["inputs"])); Path(d["inputs"][role]["path"]).write_text("tampered")
            old=os.environ.copy(); os.environ["FAKE_LEFT_GII"]=str(lf); os.environ["FAKE_RIGHT_GII"]=str(rf)
            try:
                with self.assertRaises(m.DerivationError): m.derive(mp,rp,ap,td/"o")
            finally: os.environ.clear(); os.environ.update(old)

    def test_full_derivation_deterministic_non_authorizing_and_masked(self):
        with tempfile.TemporaryDirectory() as t:
            td=Path(t); mp,rp,ap,lf,rf=self.fixture(td); old=os.environ.copy(); os.environ["FAKE_LEFT_GII"]=str(lf); os.environ["FAKE_RIGHT_GII"]=str(rf)
            try:
                a,b=td/"a",td/"b"; e1=m.derive(mp,rp,ap,a); e2=m.derive(mp,rp,ap,b)
            finally: os.environ.clear(); os.environ.update(old)
            self.assertEqual((a/"left.semantic.json").read_bytes(),(b/"left.semantic.json").read_bytes())
            self.assertEqual(e1["content_digest"],e2["content_digest"]); self.assertFalse(e1["independence"]["independence_established"])
            self.assertIsNone(json.loads((a/"left.semantic.json").read_text())["labels"][997])
            m.validate_evidence(a)

    def test_execution_metadata_not_in_scientific_commitment(self):
        with tempfile.TemporaryDirectory() as t:
            td=Path(t); mp,rp,ap,lf,rf=self.fixture(td); old=os.environ.copy(); os.environ["FAKE_LEFT_GII"]=str(lf); os.environ["FAKE_RIGHT_GII"]=str(rf)
            try:
                a=td/"a"; e1=m.derive(mp,rp,ap,a); d=json.loads(rp.read_text()); d["execution_id"]="run-b"; d["authorization_reference"]="different-note"; rp.write_text(json.dumps(d,sort_keys=True)); b=td/"b"; e2=m.derive(mp,rp,ap,b)
            finally: os.environ.clear(); os.environ.update(old)
            self.assertEqual(e1["scientific_input_commitment"],e2["scientific_input_commitment"])
            self.assertNotEqual(e1["run_manifest_digest"],e2["run_manifest_digest"])

    def test_area_order_is_in_scientific_commitment(self):
        with tempfile.TemporaryDirectory() as t:
            td=Path(t); mp,rp,ap,lf,rf=self.fixture(td); old=os.environ.copy(); os.environ["FAKE_LEFT_GII"]=str(lf); os.environ["FAKE_RIGHT_GII"]=str(rf)
            try:
                e1=m.derive(mp,rp,ap,td/"a"); rev=list(reversed(areas())); ap.write_text(json.dumps(area_doc(rev),sort_keys=True)); e2=m.derive(mp,rp,ap,td/"b")
            finally: os.environ.clear(); os.environ.update(old)
            self.assertNotEqual(e1["scientific_input_commitment"],e2["scientific_input_commitment"])

    def test_workbench_version_digest_bound(self):
        with tempfile.TemporaryDirectory() as t:
            td=Path(t); mp,rp,ap,lf,rf=self.fixture(td); d=json.loads(rp.read_text()); d["workbench"]["version_output_sha256"]="sha256:"+"0"*64; rp.write_text(json.dumps(d)); old=os.environ.copy(); os.environ["FAKE_LEFT_GII"]=str(lf); os.environ["FAKE_RIGHT_GII"]=str(rf)
            try:
                with self.assertRaises(m.DerivationError): m.derive(mp,rp,ap,td/"o")
            finally: os.environ.clear(); os.environ.update(old)

    def test_evidence_tamper_rejected_even_when_outer_digest_rehashed(self):
        with tempfile.TemporaryDirectory() as t:
            td=Path(t); mp,rp,ap,lf,rf=self.fixture(td); old=os.environ.copy(); os.environ["FAKE_LEFT_GII"]=str(lf); os.environ["FAKE_RIGHT_GII"]=str(rf)
            try: out=td/"o"; m.derive(mp,rp,ap,out)
            finally: os.environ.clear(); os.environ.update(old)
            p=out/"derivation-evidence.json"; d=json.loads(p.read_text()); d["independence"]["independence_established"]=True; payload=dict(d); payload.pop("content_digest"); d["content_digest"]=m.digest_bytes(m.canonical_json_bytes(payload)); p.write_bytes(m.canonical_json_bytes(d)+b"\n")
            with self.assertRaises(m.DerivationError): m.validate_evidence(out)


if __name__ == "__main__": unittest.main(verbosity=2)
