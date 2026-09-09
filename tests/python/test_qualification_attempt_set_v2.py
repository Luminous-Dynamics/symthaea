import importlib.util, sys
from pathlib import Path
import pytest

SCRIPTS=Path(__file__).resolve().parents[2]/"scripts"
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); assert spec and spec.loader
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
train=load("integration_train_manifest",SCRIPTS/"integration_train_manifest.py")
sys.modules["integration_train_manifest"]=train
attempts=load("qualification_attempt_set_v2",SCRIPTS/"qualification_attempt_set_v2.py")

def ident(c): return "sha256:"+c*64
def sha(c): return c*40
def progress(terminal="FormattingFailed"):
    fail=attempts.FAILURE_STAGE.get(terminal); out=[]
    for name in attempts.THEOREMS:
        if terminal=="Passed": d="Passed"
        elif fail is not None:
            i=attempts.THEOREMS.index(name); f=attempts.THEOREMS.index(fail)
            d="Passed" if i<f else "Failed" if i==f else "NotExecuted"
        else: d="NotExecuted"
        out.append({"name":name,"disposition":d})
    return out

def observation(subject="a",terminal="FormattingFailed",run=3):
    raw={
      "schema":attempts.OBSERVATION_SCHEMA,
      "qualification_profile":attempts.PROFILE,
      "provider":"github-actions","repository":"Luminous-Dynamics/symthaea",
      "provider_attempt":{"workflow_name":"Resource Stack Qualification","run_id":run,"run_number":run,"job_id":100+run,"pull_request":833},
      "subject":{"admission_subject_id":ident(subject),"admission_request_id":ident("b"),"requested_head_sha":sha("c"),"requested_base_sha":sha("d"),"checked_out_sha":sha("e")},
      "execution_environment":{"runner_os":"ubuntu-24.04.4","runner_image":"ubuntu-24.04","runner_image_version":"20260831.293.1","rustc":"1.96.0 (ac68faa20 2026-05-25)","target":"x86_64-unknown-linux-gnu"},
      "execution":{"started_at":"2026-09-09T10:07:31Z","completed_at":"2026-09-09T10:07:50Z","terminal_disposition":terminal},
      "theorem_progress":progress(terminal),
      "failure_evidence_refs":[] if terminal=="Passed" else ["github-actions:run:34215218821/job:102025268120"],
      "non_claims":["does not establish later theorems"],
    }
    raw["observation_id"]=attempts.compute_observation_id(raw)
    return raw

def test_observation_id_is_constructively_recomputed():
    raw=observation(); attempts.normalize(raw)
    raw["execution_environment"]["runner_image_version"]="changed"
    with pytest.raises(train.TrainManifestError,match="observation_id"): attempts.normalize(raw)

def test_theorem_order_is_canonical():
    raw=observation(); raw["theorem_progress"][0],raw["theorem_progress"][1]=raw["theorem_progress"][1],raw["theorem_progress"][0]
    with pytest.raises(train.TrainManifestError,match="expected"): attempts.normalize(raw,False)

def test_terminal_and_progress_must_agree():
    raw=observation(); raw["theorem_progress"][3]["disposition"]="Passed"
    with pytest.raises(train.TrainManifestError,match="inconsistent"): attempts.normalize(raw,False)

def test_same_subject_set_is_order_independent():
    a=observation(run=3); b=observation(run=4)
    assert attempts.build_attempt_set([a,b])==attempts.build_attempt_set([b,a])

def test_different_subjects_fail_closed():
    with pytest.raises(train.TrainManifestError,match="different admission subjects"):
        attempts.build_attempt_set([observation("a"),observation("f",run=4)])

def test_duplicate_provider_attempt_fails_closed():
    a=observation(); b=observation(); b["subject"]["checked_out_sha"]=sha("f"); b["observation_id"]=attempts.compute_observation_id(b)
    with pytest.raises(train.TrainManifestError,match="duplicate provider"): attempts.build_attempt_set([a,b])

def test_pass_requires_all_lanes_and_no_failure_evidence():
    raw=observation(terminal="Passed"); attempts.normalize(raw)
    raw["failure_evidence_refs"]=["unexpected"]
    with pytest.raises(train.TrainManifestError,match="Passed cannot"): attempts.normalize(raw,False)

def test_source_changing_repair_is_not_same_subject_set():
    with pytest.raises(train.TrainManifestError): attempts.build_attempt_set([observation("a"),observation("b",run=4)])
