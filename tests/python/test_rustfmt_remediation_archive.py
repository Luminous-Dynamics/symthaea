#!/usr/bin/env python3
"""Regression tests for the durable Rustfmt remediation archive."""
from __future__ import annotations
import hashlib,json,pathlib,subprocess,sys,tempfile,unittest
ROOT=pathlib.Path(__file__).resolve().parents[2]
VERIFY=ROOT/'scripts'/'verify-rustfmt-remediation-archive.py'
ARCHIVE=ROOT/'docs/release/evidence/assure-linux-ima-rustfmt-remediation-archive.v1.json'
DERIVATION=ROOT/'docs/release/evidence/assure-linux-ima-rustfmt-remediation-derived.v1.json'
EXPECTED_ARCHIVE_ID='sha256:22764eb4b2e505e0e0b58c26047de0cdb9d97af5d7d2f903e7bb991236b7f6c6'
EXPECTED_SHA256='030ded9703fbd0e6dd9e93eb8bc39736bf84ea4cc4e5b39d140b9399d7dd6ae7'
EXPECTED_BLOB='2ee7b299599570180fe2a0ca506399fdbbded4a7'
def run(*args,check=True):
 r=subprocess.run(list(args),text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 if check and r.returncode: raise AssertionError(f'{args!r}\n{r.stdout}\n{r.stderr}')
 return r
class ArchiveTests(unittest.TestCase):
 def test_production_archive_reconstructs_and_cross_checks(self):
  with tempfile.TemporaryDirectory() as raw:
   out=pathlib.Path(raw)/'lib.rs'
   r=run(sys.executable,str(VERIFY),'--archive-manifest',str(ARCHIVE),'--repository-root',str(ROOT),'--derivation-manifest',str(DERIVATION),'--output',str(out))
   receipt=json.loads(r.stdout)
   self.assertEqual(receipt['verification'],'PASS')
   self.assertEqual(receipt['authority'],'VerificationOnly')
   self.assertEqual(receipt['qualification_result'],'NOT_ESTABLISHED')
   self.assertEqual(receipt['archive_id'],EXPECTED_ARCHIVE_ID)
   self.assertEqual(receipt['reconstructed_sha256'],EXPECTED_SHA256)
   self.assertEqual(receipt['reconstructed_git_blob_sha1'],EXPECTED_BLOB)
   data=out.read_bytes()
   self.assertEqual(hashlib.sha256(data).hexdigest(),EXPECTED_SHA256)
   self.assertEqual(hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest(),EXPECTED_BLOB)
 def test_chunk_mutation_is_rejected(self):
  with tempfile.TemporaryDirectory() as raw:
   root=pathlib.Path(raw); archive=json.loads(ARCHIVE.read_text());
   for item in archive['payload']['chunks']:
    src=ROOT/item['path']; dst=root/item['path']; dst.parent.mkdir(parents=True,exist_ok=True); dst.write_bytes(src.read_bytes())
   first=root/archive['payload']['chunks'][0]['path']; data=bytearray(first.read_bytes()); data[0]^=1; first.write_bytes(data)
   manifest=root/'archive.json'; manifest.write_text(json.dumps(archive))
   r=run(sys.executable,str(VERIFY),'--archive-manifest',str(manifest),'--repository-root',str(root),check=False)
   self.assertNotEqual(r.returncode,0); self.assertIn('chunk 0',r.stderr)
 def test_manifest_mutation_is_rejected(self):
  with tempfile.TemporaryDirectory() as raw:
   archive=json.loads(ARCHIVE.read_text()); archive['payload']['reconstructed']['bytes']+=1
   manifest=pathlib.Path(raw)/'archive.json'; manifest.write_text(json.dumps(archive))
   r=run(sys.executable,str(VERIFY),'--archive-manifest',str(manifest),'--repository-root',str(ROOT),check=False)
   self.assertNotEqual(r.returncode,0); self.assertIn('archive_id mismatch',r.stderr)
if __name__=='__main__': unittest.main(verbosity=2)
