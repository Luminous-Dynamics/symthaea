#!/usr/bin/env python3
import copy, hashlib, json, subprocess, tempfile
from pathlib import Path

CMP=Path('scripts/compare-lqcd-portable-hosted-receipts.py')
PD=b'symthaea.lqcd.portable-execution-receipt.v1\0'
SD=b'symthaea.lqcd.execution-semantics.v1\0'
SEM=(
    'subject_sha','subject_tree_sha','base_sha','base_tree_sha','verifier_sha','verifier_tree_sha',
    'qualification_profile_id','recipe_semantics_sha256','command_argv','cargo_lock_sha256',
    'manifest_set_sha256','rust_toolchain_sha256','rustc_identity','cargo_identity','clippy_identity',
    'target_triple','numerical_profile_id','environment_equivalence_id')

def canon(x): return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def h(s): return hashlib.sha256(s.encode()).hexdigest()
def sh(d,x): return hashlib.sha256(d+canon(x)).hexdigest()
def portable():
    r={
      'schema':'symthaea.lqcd.portable-execution-receipt.v1','authority_class':'PortableExecutionCandidate','provider_kind':'portable','provider_instance':'nix-fixture',
      'subject_sha':'1'*40,'subject_tree_sha':'2'*40,'base_sha':'3'*40,'base_tree_sha':'4'*40,'verifier_sha':'5'*40,'verifier_tree_sha':'6'*40,
      'qualification_profile_id':'lqcd-particle-physics-focused-v2','recipe_semantics_sha256':h('recipe'),
      'command_argv':['cargo','clippy','--locked','-p','symthaea-particle-physics','--all-targets','--','-D','warnings'],
      'cargo_lock_sha256':h('lock'),'manifest_set_sha256':h('manifest'),'rust_toolchain_sha256':h('toolchain'),
      'rustc_identity':'rustc 1.96.0 fixture','cargo_identity':'cargo 1.96.0 fixture','clippy_identity':'clippy 1.96.0 fixture',
      'target_triple':'x86_64-unknown-linux-gnu','numerical_profile_id':'non-numerical-clippy-v1','environment_equivalence_id':'nix-env-v1',
      'clean_before':True,'clean_after':True,'pre_tree_sha':'2'*40,'post_tree_sha':'2'*40,'dependency_resolution':'locked',
      'environment_profile_sha256':h('env'),'os_kernel_arch':'linux/x86_64/fixture','started_ns':1,'finished_ns':2,
      'gates':[{'gate_id':'subject-command','exit_status':0,'stdout_sha256':h('out'),'stderr_sha256':h('err')}],
    }
    r['semantic_sha256']=sh(SD,{k:r[k] for k in SEM}); r['receipt_sha256']=sh(PD,r); return r

def hosted():
    required=['verifier_binding','subject_binding','base_binding','governance','cargo_metadata','format','tests','clippy','postflight_subject_immutability','postflight_verifier_immutability']
    return {
      'schema_version':'symthaea.focused-positive-receipt.v2',
      'qualification_profile':{'profile_id':'lqcd-particle-physics-focused-v2','recipe_semantics_sha256':h('recipe'),'required_gates':required},
      'verifier_authority':{'checked_out_commit_sha':'5'*40,'checked_out_tree_sha':'6'*40,'expected_commit_sha':'5'*40,'expected_tree_sha':'6'*40,'declared_inputs_sha256':h('vinputs'),'source_state':'exact_verifier_clean'},
      'subject':{'class':'immutable_exact_head_package_focused','repository':'Luminous-Dynamics/symthaea','checked_out_commit_sha':'1'*40,'checked_out_tree_sha':'2'*40,'expected_head_sha':'1'*40,'qualification_base_sha':'3'*40,'provider_event_sha':'7'*40,'declared_inputs_sha256':h('sinputs'),'source_state':'exact_raw_head_clean'},
      'toolchain':{'rustc':'rustc 1.96.0 fixture','cargo':'cargo 1.96.0 fixture','rustfmt':'rustfmt 1.96.0 fixture','clippy':'clippy 1.96.0 fixture'},
      'attempt_sha256':h('attempt'),'gates':{x:'PASS' for x in required},
      'authority_boundary':'candidate package conformance only; full CI remains workspace authority',
    }

def run(p,hst,claimed=None):
    with tempfile.TemporaryDirectory() as td:
        pp=Path(td)/'p.json'; hp=Path(td)/'h.json'
        pp.write_text(json.dumps(p,sort_keys=True,separators=(',',':'))+'\n')
        hp.write_text(json.dumps(hst,sort_keys=True,indent=2)+'\n')
        digest=hashlib.sha256(hp.read_bytes()).hexdigest() if claimed is None else claimed
        x=subprocess.run([str(CMP),str(pp),str(hp),'--hosted-sha256',digest],capture_output=True,text=True)
        return x.returncode,json.loads(x.stdout)

def main():
    p=portable(); hst=hosted(); rc,core=run(p,hst)
    assert rc==0 and core['classification']=='CoreEquivalentExtendedBindingUnavailable'
    assert core['core_equivalent'] is True and core['cross_provider_corroborated'] is False
    shifts={}
    mutations={
      'subject':('subject','checked_out_commit_sha','8'*40,'SubjectShift'),
      'verifier':('verifier_authority','checked_out_commit_sha','9'*40,'VerifierShift'),
      'recipe':('qualification_profile','recipe_semantics_sha256',h('recipe2'),'RecipeShift'),
      'rustc':('toolchain','rustc','rustc changed','RustcShift'),
    }
    for name,(section,key,value,expected) in mutations.items():
        q=copy.deepcopy(hst); q[section][key]=value
        if name=='verifier': q[section]['expected_commit_sha']=value
        if name=='subject': q[section]['expected_head_sha']=value
        rc,out=run(p,q); assert rc==0 and out['classification']==expected,(name,rc,out)
        shifts[name]=out['classification']
    rc,bad_digest=run(p,hst,'0'*64)
    assert rc==1 and bad_digest['classification']=='HostedArtifactDigestMismatch'
    q=copy.deepcopy(hst); q['gates']['clippy']='FAIL'; rc,bad_gate=run(p,q)
    assert rc==1 and bad_gate['classification']=='HostedRequiredGateNotPass:clippy'
    q=copy.deepcopy(p); q['authority_class']='HostedExactHeadQualified'
    q['receipt_sha256']=sh(PD,{k:v for k,v in q.items() if k!='receipt_sha256'})
    rc,bad_port=run(q,hst)
    assert rc==1 and bad_port['classification']=='PortableAuthorityInvalid'
    print(json.dumps({
      'comparator_subject_sha256':hashlib.sha256(CMP.read_bytes()).hexdigest(),
      'core_positive':core,'shift_controls':shifts,
      'bad_hosted_artifact_digest':bad_digest['classification'],
      'hosted_nonpass_gate':bad_gate['classification'],
      'portable_self_promotion':bad_port['classification'],
      'full_cross_provider_corroboration_authorized':False,
      'real_hosted_receipt_compared':False,
      'real_rust_execution_performed':False,
      'real_beta6_campaign_authorized':False,
    },sort_keys=True,separators=(',',':')))
if __name__=='__main__': main()
