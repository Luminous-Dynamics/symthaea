import importlib.util, sys
from pathlib import Path
import pytest
S=Path(__file__).resolve().parents[2]/"scripts"
def load(n):
    p=importlib.util.spec_from_file_location(n,S/f"{n}.py"); assert p and p.loader
    m=importlib.util.module_from_spec(p); p.loader.exec_module(m); sys.modules[n]=m; return m
train=load("integration_train_manifest"); catalog=load("integration_train_catalog")
profile=load("qualification_profile"); subject=load("qualification_subject")
admission=load("qualification_admission_v3"); attempts=load("qualification_attempt_v3")
attempt_subject=load("qualification_attempt_subject"); select=load("qualification_pass_selection")
def sid(c): return "sha256:"+c*64
def rid(c): return "git-blob-sha1:"+c*40
def mk_profile():
    x={"schema":profile.SCHEMA,"profile_name":"research.integrity.focused.v1","required_recipe_ids":[rid("1"),rid("2")],"owned_surface_rules":["owns:research-integrity"],"required_cross_cutting_rules":["strict-clippy-required","workspace-lock-current"],"fallback_policy":"FULL_CI_REQUIRED_ON_UNKNOWN_OR_AMBIGUOUS","non_claims":["does not authorize merge"]}; x["profile_id"]=profile.compute_profile_id(x); return x
def mk_subject():
    x={"schema":subject.SCHEMA,"kind":"GitCommit","repository":"Luminous-Dynamics/symthaea","object_format":"sha1","source_commit":"a"*40,"source_tree":"b"*40}; x["subject_id"]=subject.compute_subject_id(x); return x
def mk_adm(p):
    x={"schema":admission.SCHEMA,"subject":mk_subject(),"qualification_profile":{"name":p["profile_name"],"profile_id":p["profile_id"],"profile_branch":"evidence/profile","profile_path":"docs/profile.json"},"reason":"qualify","evidence_refs":["issue:986"],"non_claims":["does not establish PASS"]}; n=admission.normalize_request(x,verify_declared_ids=False); x["admission_id"]=n["admission_id"]; x["admission_subject_id"]=n["admission_subject_id"]; return x
def reg(a,p,r,seq=1,closure="c",env="d"):
    x={"schema":attempts.REGISTRATION_SCHEMA,"admission_subject_id":a["admission_subject_id"],"qualification_subject_id":a["subject"]["subject_id"],"qualification_profile_id":p["profile_id"],"recipe_id":r,"input_closure_id":sid(closure),"qualification_environment_id":sid(env),"attempt_sequence":seq,"non_claims":["attempt sequence is not external chronology"]}; n=attempts.normalize_registration(x,verify_declared_id=False); x["attempt_registration_id"]=n["attempt_registration_id"]; return x
def obs(r,terminal="Passed",started=True,ref="run"):
    x={"schema":attempts.OBSERVATION_SCHEMA,"attempt_registration_id":r["attempt_registration_id"],"execution_started":started,"terminal_disposition":terminal,"provider_ref":{"provider":"github-actions","attempt_ref":ref},"evidence_refs":["artifact:evidence"],"non_claims":["provider metadata is provenance only"]}; n=attempts.normalize_observation(x,verify_declared_id=False); x["attempt_observation_id"]=n["attempt_observation_id"]; return x
def campaign(closure2="c"):
    p=mk_profile(); a=mk_adm(p); r1=reg(a,p,rid("1")); r2=reg(a,p,rid("2"),closure=closure2); o1=obs(r1,ref="one"); o2=obs(r2,ref="two"); chosen={rid("1"):o1["attempt_observation_id"],rid("2"):o2["attempt_observation_id"]}; cross=[{"rule":"strict-clippy-required","evidence_refs":["evidence:clippy"]},{"rule":"workspace-lock-current","evidence_refs":["evidence:lock"]}]; return p,a,[r1,r2],[o1,o2],chosen,cross
def build(c):
    p,a,rs,os,ch,cr=c; return select.build_pass_selection(admission=a,profile=p,registrations=rs,observations=os,selected_observation_ids_by_recipe=ch,cross_cutting_evidence=cr)
def test_exact_recipe_and_cross_cutting_coverage_passes():
    p,_,_,_,_,_=campaign(); x=build(campaign()); assert [i["recipe_id"] for i in x["selected_recipe_attempts"]]==p["required_recipe_ids"]
def test_missing_recipe_selection_fails():
    p,a,rs,os,ch,cr=campaign(); ch.pop(rid("2"))
    with pytest.raises(train.TrainManifestError,match="exactly equal"): select.build_pass_selection(admission=a,profile=p,registrations=rs,observations=os,selected_observation_ids_by_recipe=ch,cross_cutting_evidence=cr)
def test_cross_cutting_coverage_fails_closed():
    p,a,rs,os,ch,cr=campaign(); cr=cr[1:]
    with pytest.raises(train.TrainManifestError,match="exactly cover"): select.build_pass_selection(admission=a,profile=p,registrations=rs,observations=os,selected_observation_ids_by_recipe=ch,cross_cutting_evidence=cr)
def test_mixed_closure_fails_v1_selection():
    with pytest.raises(train.TrainManifestError,match="common input_closure"): build(campaign("9"))
def test_failed_attempt_cannot_be_selected_for_positive_support():
    p,a,rs,os,ch,cr=campaign(); os[0]=obs(rs[0],"RecipeFailed",True,"fail"); ch[rid("1")]=os[0]["attempt_observation_id"]
    with pytest.raises(train.TrainManifestError,match="executed Passed"): select.build_pass_selection(admission=a,profile=p,registrations=rs,observations=os,selected_observation_ids_by_recipe=ch,cross_cutting_evidence=cr)
