import importlib.util
import sys
from pathlib import Path
import pytest

SCRIPTS=Path(__file__).resolve().parents[2]/"scripts"

def load(name):
    spec=importlib.util.spec_from_file_location(name,SCRIPTS/f"{name}.py")
    assert spec and spec.loader
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); sys.modules[name]=mod
    return mod

train=load("integration_train_manifest")
catalog=load("integration_train_catalog")
profile=load("qualification_profile")
subject=load("qualification_subject")
admission=load("qualification_admission_v3")
routing=load("qualification_routing")
coverage=load("qualification_route_admission")


def oid(c): return c*40
def ident(c): return "sha256:"+c*64

def subj(commit="a",tree="b"):
    raw={"schema":subject.SCHEMA,"kind":"GitCommit","repository":"Luminous-Dynamics/symthaea","source_commit":oid(commit),"source_tree":oid(tree)}
    out=subject.normalize_subject(raw,verify_declared_id=False); raw["subject_id"]=out["subject_id"]; return raw

def adm(profile_id=None,reason="qualify",branch="evidence/profile-v1",path="docs/profile.json",source=None):
    raw={"schema":admission.SCHEMA,"subject":source or subj(),"qualification_profile":{"name":"research.integrity.focused.v1","profile_id":profile_id or ident("c"),"profile_branch":branch,"profile_path":path},"reason":reason,"evidence_refs":["issue:986"],"non_claims":["does not authorize merge"]}
    out=admission.normalize_request(raw,verify_declared_ids=False); raw["admission_id"]=out["admission_id"]; raw["admission_subject_id"]=out["admission_subject_id"]; return raw

def route(ids=None,disposition=None,source=None,recipe="d"):
    src=source or subj(); ids=ids if ids is not None else [ident("c")]; disposition=disposition or routing.FOCUSED
    raw={"schema":routing.SCHEMA,"subject_id":src["subject_id"],"source_commit":src["source_commit"],"base_commit":oid("e"),"merge_base":oid("f"),"changed_paths_sha256":ident("a"),"router_recipe_id":"git-blob-sha1:"+oid(recipe),"disposition":disposition,"required_profile_ids":ids,"reason":"complete diff maps to exact profile set"}
    out=routing.normalize_decision(raw,verify_declared_id=False); raw["decision_id"]=out["decision_id"]; return raw


def test_v3_is_generic_git_subject_not_train_catalog_ontology():
    item=admission.normalize_request(adm(),require_ids=True)
    assert "catalog_id" not in item and "target_train" not in item
    assert item["subject"]["kind"]=="GitCommit"


def test_request_provenance_changes_but_work_identity_does_not():
    first=adm(reason="one",branch="evidence/a",path="docs/a.json")
    second=adm(reason="two",branch="evidence/b",path="docs/b.json")
    assert first["admission_id"]!=second["admission_id"]
    assert first["admission_subject_id"]==second["admission_subject_id"]


def test_changed_profile_or_commit_changes_work_identity():
    first=adm()
    changed_profile=adm(profile_id=ident("9"))
    changed_commit=adm(source=subj(commit="9",tree="b"))
    assert first["admission_subject_id"]!=changed_profile["admission_subject_id"]
    assert first["admission_subject_id"]!=changed_commit["admission_subject_id"]


def test_subject_declared_identity_mismatch_fails_closed():
    raw=subj(); raw["subject_id"]=ident("0")
    with pytest.raises(train.TrainManifestError): subject.normalize_subject(raw,require_id=True)


def test_routing_profile_set_semantics_fail_closed():
    with pytest.raises(train.TrainManifestError): route([],routing.FOCUSED)
    with pytest.raises(train.TrainManifestError): route([ident("c")],routing.FULL)
    with pytest.raises(train.TrainManifestError): route([ident("c"),ident("c")],routing.FOCUSED)


def test_router_recipe_changes_decision_identity():
    assert route(recipe="d")["decision_id"]!=route(recipe="9")["decision_id"]


def test_exact_profile_set_coverage_is_order_independent():
    p1,p2=ident("1"),ident("2")
    r=route([p1,p2])
    a1,a2=adm(profile_id=p1),adm(profile_id=p2)
    left=coverage.build_coverage(r,[a1,a2]); right=coverage.build_coverage(r,[a2,a1])
    assert left==right
    assert left["required_profile_ids"]==[p1,p2]


def test_missing_extra_and_duplicate_profiles_fail_coverage():
    p1,p2,p3=ident("1"),ident("2"),ident("3")
    r=route([p1,p2])
    with pytest.raises(train.TrainManifestError,match="mismatch"): coverage.build_coverage(r,[adm(profile_id=p1)])
    with pytest.raises(train.TrainManifestError,match="mismatch"): coverage.build_coverage(r,[adm(profile_id=p1),adm(profile_id=p2),adm(profile_id=p3)])
    with pytest.raises(train.TrainManifestError,match="duplicate"): coverage.build_coverage(r,[adm(profile_id=p1),adm(profile_id=p1),adm(profile_id=p2)])


def test_full_ci_route_cannot_be_laundered_into_focused_coverage():
    r=route([],routing.FULL)
    with pytest.raises(train.TrainManifestError,match="full-CI"): coverage.build_coverage(r,[adm()])


def test_subject_substitution_fails_coverage():
    r=route()
    wrong=adm(source=subj(commit="9",tree="8"))
    with pytest.raises(train.TrainManifestError,match="routing subject"): coverage.build_coverage(r,[wrong])
