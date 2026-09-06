#!/usr/bin/env python3
from __future__ import annotations
import contextlib, io, json, stat, sys, tempfile, unittest
from pathlib import Path
SCRIPTS=Path(__file__).parent;sys.path.insert(0,str(SCRIPTS))
import capture_hcpmmp1_neuromaps_lineage_b_run as capture
from hcpmmp_neuromaps_common import ContractError,REQUIRED_INPUT_ROLES
METHOD=Path(__file__).parents[1]/'data/neuroscience/hcpmmp1_neuromaps_transform_method_v1.json'
class Tests(unittest.TestCase):
    def fixture(self,td):
        inputs={}
        for i,role in enumerate(sorted(REQUIRED_INPUT_ROLES)):
            p=td/f"{i:02d}-{role}";p.write_bytes(f"{role}-bytes".encode());inputs[role]=p
        log=td/"wb.log";wb=td/"wb_command"
        wb.write_text("#!/bin/sh\n"+f"printf '%s\\n' \"$*\" >> '{log}'\n"+"if [ \"$1\" != \"-version\" ] || [ \"$#\" -ne 1 ]; then exit 77; fi\nprintf 'Workbench fake 1.0\\n'\n")
        wb.chmod(wb.stat().st_mode|stat.S_IXUSR)
        return inputs,wb,log
    def items(self,inputs): return [f"{r}={inputs[r]}" for r in sorted(inputs)]
    def cap(self,inputs,wb,e="r1",a="note"): return capture.capture_manifest(METHOD,wb,self.items(inputs),e,a)
    def test_valid(self):
        with tempfile.TemporaryDirectory() as t:
            i,w,_=self.fixture(Path(t));d=self.cap(i,w);self.assertEqual(set(d["inputs"]),REQUIRED_INPUT_ROLES)
    def test_version_only(self):
        with tempfile.TemporaryDirectory() as t:
            i,w,l=self.fixture(Path(t));self.cap(i,w);self.assertEqual(l.read_text().splitlines(),["-version"])
    def test_missing(self):
        with tempfile.TemporaryDirectory() as t:
            i,w,_=self.fixture(Path(t));i.pop(next(iter(i)))
            with self.assertRaises(ContractError): self.cap(i,w)
    def test_duplicate(self):
        with tempfile.TemporaryDirectory() as t:
            i,w,_=self.fixture(Path(t));x=self.items(i);x.append(x[0])
            with self.assertRaises(ContractError): capture.capture_manifest(METHOD,w,x,"x","y")
    def test_unknown(self):
        with tempfile.TemporaryDirectory() as t:
            i,w,_=self.fixture(Path(t));x=self.items(i);x[-1]=f"bad={Path(t)/'x'}"
            with self.assertRaises(ContractError): capture.capture_manifest(METHOD,w,x,"x","y")
    def test_identical_hcp(self):
        with tempfile.TemporaryDirectory() as t:
            i,w,_=self.fixture(Path(t));i["hcp_right_dlabel"].write_bytes(i["hcp_left_dlabel"].read_bytes())
            with self.assertRaises(ContractError): self.cap(i,w)
    def test_directory(self):
        with tempfile.TemporaryDirectory() as t:
            i,w,_=self.fixture(Path(t));i["fsaverage10k_left_sphere"]=Path(t)
            with self.assertRaises(ContractError): self.cap(i,w)
    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"x";p.write_text("keep")
            with self.assertRaises(ContractError): capture.write_new(p,{"x":1})
            self.assertEqual(p.read_text(),"keep")
    def test_atomic_output_is_restrictive(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"x";out=capture.write_new(p,{"x":1})
            self.assertEqual(out,p);self.assertEqual(stat.S_IMODE(p.stat().st_mode),0o600)
    def test_cli_stdout_is_digest_only_receipt(self):
        with tempfile.TemporaryDirectory() as t:
            td=Path(t);i,w,_=self.fixture(td);out=td/"run.json"
            argv=["--method-manifest",str(METHOD),"--workbench",str(w),"--execution-id","run-secret","--authorization-reference","authorization-secret","--output",str(out)]
            for role in sorted(i): argv.extend(["--input",f"{role}={i[role]}"])
            buf=io.StringIO()
            with contextlib.redirect_stdout(buf): self.assertEqual(capture.main(argv),0)
            receipt=json.loads(buf.getvalue())
            self.assertEqual(set(receipt),{"profile","run_manifest_file_sha256"})
            self.assertNotIn("authorization-secret",buf.getvalue());self.assertNotIn(str(next(iter(i.values()))),buf.getvalue());self.assertTrue(out.is_file())
    def test_metadata_separate(self):
        with tempfile.TemporaryDirectory() as t:
            i,w,_=self.fixture(Path(t));a=self.cap(i,w,"a","na");b=self.cap(i,w,"b","nb")
            self.assertEqual(a["inputs"],b["inputs"]);self.assertEqual(a["workbench"],b["workbench"]);self.assertNotEqual(a["execution_id"],b["execution_id"])
    def test_input_mutation_during_version_probe_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            td=Path(t);i,w,_=self.fixture(td);victim=i["fsaverage10k_left_sphere"]
            w.write_text("#!/bin/sh\nprintf 'mutated' >> '%s'\nprintf 'v\\n'\n" % victim);w.chmod(w.stat().st_mode|stat.S_IXUSR)
            with self.assertRaises(ContractError): self.cap(i,w)
    def test_workbench_mutation_during_probe_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            td=Path(t);i,w,_=self.fixture(td)
            w.write_text("#!/bin/sh\nprintf '# mutation\\n' >> \"$0\"\nprintf 'v\\n'\n");w.chmod(w.stat().st_mode|stat.S_IXUSR)
            with self.assertRaises(ContractError): self.cap(i,w)
    def test_bad_version(self):
        with tempfile.TemporaryDirectory() as t:
            i,w,_=self.fixture(Path(t));w.write_text("#!/bin/sh\nexit 9\n");w.chmod(w.stat().st_mode|stat.S_IXUSR)
            with self.assertRaises(ContractError): self.cap(i,w)
if __name__=="__main__": unittest.main(verbosity=2)
