import importlib.util, sys
from pathlib import Path
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
def fixture():
    p={"schema":profile.SCHEMA,"profile_name":"one-recipe.v1","required_recipe_ids":[rid("1")],"owned_surface_rules":["owns:test"],"required_cross_cutting_rules":["workspace-lock-current"],"fallback_policy":"FULL_CI_REQUIRED_ON_UNKNOWN_OR_AMBIGUOUS","non_claims":["does not authorize merge"]}; p["profile_id"]=profile.compute_profile_id(p)
    s={"schema":subject.SCHEMA,"kind":"GitCommit","repository":"Luminous-Dynamics/symthaea","object_format":"sha1","source_commit":"a"*40,"source_tree":"b"*40}; s["subject_id"]=subject.compute_subject_id(s)
    a={"schema":admission.SCHEMA,"subject":s,"qualification_profile":{"name":p["profile_name"],"profile_id":p["profile_id"],"profile_branch":"evidence/profile","profile_path":"docs/profile.json"},"reason":"qualify","evidence_refs":["issue:986"],"non_claims":["does not establish PASS"]}; n=admission.normalize_request(a,verify_declared_ids=False); a["admission_id"]=n["admission_id"]; a["admission_subject_id"]=n["admission_subject_id"]
    def reg(seq):
        x={"schema":attempts.REGISTRATION_SCHEMA,"admission_subject_id":a["admission_subject_id"],"qualification_subject_id":s["subject_id"],"qualification_profile_id":p["profile_id"],"recipe_id":rid("1"),"input_closure_id":sid("c"),"qualification_environment_id":sid("d"),"attempt_sequence":seq,"non_claims":["attempt sequence is not external chronology"]}; q=attempts.normalize_registration(x,verify_declared_id=False); x["attempt_registration_id"]=q["attempt_registration_id"]; return x
    def obs(r,term,ref):
        x={"schema":attempts.OBSERVATION_SCHEMA,"attempt_registration_id":r["attempt_registration_id"],"execution_started":True,"terminal_disposition":term,"provider_ref":{"provider":"github-actions","attempt_ref":ref},"evidence_refs":["artifact:evidence"],"non_claims":["provider metadata is provenance only"]}; q=attempts.normalize_observation(x,verify_declared_id=False); x["attempt_observation_id"]=q["attempt_observation_id"]; return x
    return p,a,reg,obs
def test_retry_to_green_does_not_erase_earlier_failure():
    p,a,reg,obs=fixture(); passed_reg=reg(1); passed=obs(passed_reg,"Passed","pass")
    args=dict(admission=a,profile=p,selected_observation_ids_by_recipe={rid("1"):passed["attempt_observation_id"]},cross_cutting_evidence=[{"rule":"workspace-lock-current","evidence_refs":["evidence:lock"]}])
    clean=select.build_pass_selection(registrations=[passed_reg],observations=[passed],**args)
    failed_reg=reg(2); failed=obs(failed_reg,"RecipeFailed","fail")
    with_failure=select.build_pass_selection(registrations=[passed_reg,failed_reg],observations=[passed,failed],**args)
    assert clean["attempt_history_id"]!=with_failure["attempt_history_id"]
    assert clean["pass_selection_id"]!=with_failure["pass_selection_id"]
    assert clean["selected_recipe_attempts"]==with_failure["selected_recipe_attempts"]
