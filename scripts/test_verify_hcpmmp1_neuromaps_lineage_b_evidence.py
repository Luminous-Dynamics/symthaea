#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,sys,tempfile,unittest
from pathlib import Path
SCRIPTS=Path(__file__).parent;sys.path.insert(0,str(SCRIPTS))
import derive_hcpmmp1_neuromaps_lineage_b as lineage
import verify_hcpmmp1_neuromaps_lineage_b_evidence as verifier

def sha_bytes(data:bytes)->str:return 'sha256:'+hashlib.sha256(data).hexdigest()
def sha_file(path:Path)->str:return sha_bytes(path.read_bytes())
def canonical(value)->bytes:return lineage.canonical_json_bytes(value)
def area_doc()->dict:
 areas=[f'A{i:03d}' for i in range(1,181)]
 return {'schema':lineage.AREA_SCHEMA,'atlas':'HCP-MMP1.0/Glasser360','hemisphere_area_count':180,'source':{'repository':'synthetic/test','commit':'0'*40,'blob_sha':'1'*40,'path':'x','purpose':'test'},'areas':areas}
def method_doc()->dict:return json.loads((Path(__file__).parents[1]/'data/neuroscience/hcpmmp1_neuromaps_transform_method_v1.json').read_text())

class VerifierTests(unittest.TestCase):
 def fixture(self,td:Path):
  method_path=td/'method.json';method_path.write_text(json.dumps(method_doc(),sort_keys=True))
  area_path=td/'areas.json';areas=area_doc();area_path.write_text(json.dumps(areas,sort_keys=True))
  run_inputs={role:{'path':f'/synthetic/{role}','sha256':'sha256:'+hashlib.sha256(role.encode()).hexdigest()} for role in method_doc()['required_inputs']}
  run={'schema':lineage.RUN_SCHEMA,'method_manifest_digest':sha_file(method_path),'execution_id':'synthetic-run-1','authorization_reference':'synthetic-test-only','workbench':{'path':'/synthetic/wb_command','sha256':'sha256:'+'a'*64,'version_output_sha256':'sha256:'+'b'*64},'inputs':run_inputs}
  run_path=td/'run.json';run_path.write_text(json.dumps(run,sort_keys=True))
  method=lineage.load_method_manifest(method_path);run_loaded=lineage.load_run_manifest(run_path,method,method_path);generator=lineage.generator_implementation();commitment=lineage.commitment(method_path,run_loaded,area_path,run['workbench']['version_output_sha256'],generator['digest'])
  out=td/'out';out.mkdir();area_names=areas['areas'];left_labels=[f'L_{area_names[i%180]}' for i in range(lineage.VERTICES)];right_labels=[f'R_{area_names[i%180]}' for i in range(lineage.VERTICES)]
  def semantic(hemi,labels):return {'schema':lineage.OUTPUT_SCHEMA,'space':'fsaverage5','hemisphere':hemi,'vertex_count':lineage.VERTICES,'labels':labels,'source':{'source_id':f"{method['lineage_id']}:{hemi}",'source_version':'v1','source_digest':commitment,'generator_id':lineage.GENERATOR_ID,'generator_version':lineage.GENERATOR_VERSION,'generator_implementation_digest':generator['digest'],'terms_reference':method['terms_reference']}}
  left_path=out/'left.semantic.json';right_path=out/'right.semantic.json';left_path.write_bytes(canonical(semantic('left',left_labels))+b'\n');right_path.write_bytes(canonical(semantic('right',right_labels))+b'\n')
  evidence={'schema':lineage.EVIDENCE_SCHEMA,'lineage_id':method['lineage_id'],'execution_id':run['execution_id'],'authorization_reference':run['authorization_reference'],'method_manifest_digest':sha_file(method_path),'run_manifest_digest':sha_file(run_path),'area_order_digest':sha_file(area_path),'scientific_input_commitment':commitment,'generator_implementation':generator,'workbench':{'sha256':run['workbench']['sha256'],'version_output_sha256':run['workbench']['version_output_sha256']},'outputs':{'left_semantic_sha256':sha_file(left_path),'right_semantic_sha256':sha_file(right_path)},'independence':{**method['independence_contract'],'independence_established':False,'status':'requires_external_provenance_review'}}
  evidence['content_digest']=lineage.digest_bytes(canonical(evidence));(out/'derivation-evidence.json').write_bytes(canonical(evidence)+b'\n')
  return method_path,run_path,area_path,out,evidence
 def verify(self,f):
  method_path,run_path,area_path,out,e=f;return verifier.verify_bundle(out,method_path,run_path,area_path,e['content_digest'])
 def rewrite(self,out,doc):
  payload=dict(doc);payload.pop('content_digest',None);doc['content_digest']=lineage.digest_bytes(canonical(payload));(out/'derivation-evidence.json').write_bytes(canonical(doc)+b'\n');return doc
 def test_valid_bundle(self):
  with tempfile.TemporaryDirectory() as t:self.assertFalse(self.verify(self.fixture(Path(t)))['independence']['independence_established'])
 def test_external_retained_root_required(self):
  with tempfile.TemporaryDirectory() as t:
   mp,rp,ap,out,e=self.fixture(Path(t))
   with self.assertRaises(lineage.DerivationError):verifier.verify_bundle(out,mp,rp,ap,'sha256:'+'0'*64)
 def test_unknown_independence_field_rejected_even_with_new_root(self):
  with tempfile.TemporaryDirectory() as t:
   mp,rp,ap,out,_=self.fixture(Path(t));p=out/'derivation-evidence.json';d=json.loads(p.read_text());d['independence']['execution_independence_verified']=True;d=self.rewrite(out,d)
   with self.assertRaises(lineage.DerivationError):verifier.verify_bundle(out,mp,rp,ap,d['content_digest'])
 def test_noncanonical_semantic_label_rejected_even_with_new_root(self):
  with tempfile.TemporaryDirectory() as t:
   mp,rp,ap,out,_=self.fixture(Path(t));left=out/'left.semantic.json';d=json.loads(left.read_text());d['labels'][1]='L_NOT_A_CANONICAL_AREA';left.write_bytes(canonical(d)+b'\n');e=json.loads((out/'derivation-evidence.json').read_text());e['outputs']['left_semantic_sha256']=sha_file(left);e=self.rewrite(out,e)
   with self.assertRaises(lineage.DerivationError):verifier.verify_bundle(out,mp,rp,ap,e['content_digest'])
 def test_output_source_commitment_mismatch_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   mp,rp,ap,out,_=self.fixture(Path(t));left=out/'left.semantic.json';d=json.loads(left.read_text());d['source']['source_digest']='sha256:'+'f'*64;left.write_bytes(canonical(d)+b'\n');e=json.loads((out/'derivation-evidence.json').read_text());e['outputs']['left_semantic_sha256']=sha_file(left);e=self.rewrite(out,e)
   with self.assertRaises(lineage.DerivationError):verifier.verify_bundle(out,mp,rp,ap,e['content_digest'])
 def test_run_manifest_rebinding_required(self):
  with tempfile.TemporaryDirectory() as t:
   mp,rp,ap,out,e=self.fixture(Path(t));run=json.loads(rp.read_text());run['execution_id']='different-run';rp.write_text(json.dumps(run,sort_keys=True))
   with self.assertRaises(lineage.DerivationError):verifier.verify_bundle(out,mp,rp,ap,e['content_digest'])
 def test_workbench_root_mismatch_rejected_even_with_new_root(self):
  with tempfile.TemporaryDirectory() as t:
   mp,rp,ap,out,_=self.fixture(Path(t));e=json.loads((out/'derivation-evidence.json').read_text());e['workbench']['sha256']='sha256:'+'c'*64;e=self.rewrite(out,e)
   with self.assertRaises(lineage.DerivationError):verifier.verify_bundle(out,mp,rp,ap,e['content_digest'])
 def test_output_digest_mismatch_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   mp,rp,ap,out,e=self.fixture(Path(t));(out/'left.semantic.json').write_bytes(b'{}\n')
   with self.assertRaises(lineage.DerivationError):verifier.verify_bundle(out,mp,rp,ap,e['content_digest'])
 def test_self_hash_corruption_rejected_even_if_external_root_matches_field(self):
  with tempfile.TemporaryDirectory() as t:
   mp,rp,ap,out,_=self.fixture(Path(t));p=out/'derivation-evidence.json';d=json.loads(p.read_text());d['execution_id']='tampered-without-rehash';p.write_bytes(canonical(d)+b'\n')
   with self.assertRaises(lineage.DerivationError):verifier.verify_bundle(out,mp,rp,ap,d['content_digest'])
 def test_generator_file_map_tamper_rejected_after_rehash(self):
  with tempfile.TemporaryDirectory() as t:
   mp,rp,ap,out,_=self.fixture(Path(t));e=json.loads((out/'derivation-evidence.json').read_text());e['generator_implementation']['files']['derive']='sha256:'+'e'*64;e=self.rewrite(out,e)
   with self.assertRaises(lineage.DerivationError):verifier.verify_bundle(out,mp,rp,ap,e['content_digest'])
if __name__=='__main__':unittest.main(verbosity=2)
