#!/usr/bin/env python3
import json, shutil, tempfile, unittest
from pathlib import Path
import benchmark, verify_run

class ArtifactVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.run_dir=Path(cls.tmp.name)/"valid"
        import subprocess,sys
        subprocess.run([sys.executable,str(Path(benchmark.__file__)),"--out",str(cls.run_dir),"--dev-cases","500","--test-cases-per-regime","100"],check=True,capture_output=True,text=True)
    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()
    def clone(self,name):
        p=Path(self.__class__.tmp.name)/name
        if p.exists(): shutil.rmtree(p)
        shutil.copytree(self.__class__.run_dir,p)
        return p
    def test_valid_artifact(self):
        ok,errors=verify_run.verify(self.__class__.run_dir)
        self.assertTrue(ok,errors)
    def test_digest_tamper_rejected(self):
        p=self.clone("digest_bad")
        with (p/"per_case_policy_outcomes.jsonl").open("a") as f: f.write("{}\n")
        ok,errors=verify_run.verify(p)
        self.assertFalse(ok); self.assertTrue(any("SHA-256 mismatch" in e for e in errors))
    def test_recomputed_digest_cannot_hide_summary_mismatch(self):
        p=self.clone("summary_bad")
        f=p/"per_case_policy_outcomes.jsonl"
        rows=f.read_text().splitlines()
        r=json.loads(rows[0]); r["task_success"]=1-r["task_success"]; rows[0]=json.dumps(r,separators=(",",":"),sort_keys=True)
        f.write_text("\n".join(rows)+"\n")
        import hashlib
        rec=json.loads((p/"receipt.json").read_text()); summ=json.loads((p/"summary.json").read_text())
        digest=hashlib.sha256(f.read_bytes()).hexdigest(); rec["outcome_sha256"]=digest; summ["artifact_sha256"]=digest
        (p/"receipt.json").write_text(json.dumps(rec,indent=2,sort_keys=True)+"\n")
        (p/"summary.json").write_text(json.dumps(summ,indent=2,sort_keys=True)+"\n")
        rec=json.loads((p/"receipt.json").read_text()); rec["summary_sha256"]=hashlib.sha256((p/"summary.json").read_bytes()).hexdigest()
        (p/"receipt.json").write_text(json.dumps(rec,indent=2,sort_keys=True)+"\n")
        ok,errors=verify_run.verify(p)
        self.assertFalse(ok); self.assertTrue(any("summary mismatch" in e for e in errors))
    def test_missing_policy_row_rejected_even_after_digest_repin(self):
        p=self.clone("matrix_bad")
        f=p/"per_case_policy_outcomes.jsonl"
        lines=f.read_text().splitlines(); lines=lines[1:]
        f.write_text("\n".join(lines)+"\n")
        import hashlib
        rec=json.loads((p/"receipt.json").read_text()); summ=json.loads((p/"summary.json").read_text())
        digest=hashlib.sha256(f.read_bytes()).hexdigest()
        rec["outcome_sha256"]=digest; summ["artifact_sha256"]=digest
        (p/"summary.json").write_text(json.dumps(summ,indent=2,sort_keys=True)+"\n")
        rec["summary_sha256"]=hashlib.sha256((p/"summary.json").read_bytes()).hexdigest()
        (p/"receipt.json").write_text(json.dumps(rec,indent=2,sort_keys=True)+"\n")
        ok,errors=verify_run.verify(p)
        self.assertFalse(ok); self.assertTrue(any("row count mismatch" in e or "policy matrix incomplete" in e for e in errors))
    def test_config_tamper_rejected(self):
        p=self.clone("config_bad")
        with (p/"config.json").open("a") as f: f.write("\n# altered config\n")
        ok,errors=verify_run.verify(p)
        self.assertFalse(ok); self.assertTrue(any("config SHA-256 mismatch" in e for e in errors))

if __name__=="__main__": unittest.main(verbosity=2)
