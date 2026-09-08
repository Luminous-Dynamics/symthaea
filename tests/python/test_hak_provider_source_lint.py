import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

spec = importlib.util.spec_from_file_location(
    "hak_provider_source_lint", SCRIPTS / "hak_provider_source_lint.py"
)
assert spec and spec.loader
hak = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hak)

CAPSULE_PATH = ROOT / "docs/architecture/hak/evidence/real/hak007-run-34225891059.capsule.json"
PROJECTION_PATH = ROOT / "docs/architecture/hak/evidence/source/hak007-run-34225891059.provider-projection.json"
OBS_PATHS = [
    ROOT / "docs/architecture/hak/evidence/source/hak007-run-34225891059.workflow-run.observation.json",
    ROOT / "docs/architecture/hak/evidence/source/hak007-run-34225891059.jobs.observation.json",
    ROOT / "docs/architecture/hak/evidence/source/hak007-run-34225891059.job-102059802264.steps.observation.json",
]


def load(path):
    return json.loads(path.read_text())


def fixture():
    return load(CAPSULE_PATH), [load(path) for path in OBS_PATHS], load(PROJECTION_PATH)


def validate(capsule, observations, projection):
    hak.validate_provider_projection(
        capsule,
        observations,
        projection,
        capsule_repo_path="docs/architecture/hak/evidence/real/hak007-run-34225891059.capsule.json",
    )


def redigest_observation(doc):
    doc["normalized_payload_digest"] = hak.compute_payload_digest(doc["normalized_payload"])
    doc["observation_digest"] = hak.compute_observation_digest(doc)


def redigest_projection(doc):
    doc["projection_digest"] = hak.compute_projection_digest(doc)


def test_real_source_projection_valid():
    validate(*fixture())


def test_normalized_provider_payload_tampering_rejected():
    capsule, observations, projection = fixture()
    observations[0]["normalized_payload"]["status"] = "queued"
    with pytest.raises(hak.ProviderSourceLintError):
        validate(capsule, observations, projection)


def test_provider_observation_digest_tampering_rejected():
    capsule, observations, projection = fixture()
    observations[0]["observation_digest"] = "sha256:" + "0" * 64
    with pytest.raises(hak.ProviderSourceLintError):
        validate(capsule, observations, projection)


def test_provider_source_reference_mismatch_rejected():
    capsule, observations, projection = fixture()
    observations[0]["source_ref"] += ":other"
    redigest_observation(observations[0])
    with pytest.raises(hak.ProviderSourceLintError):
        validate(capsule, observations, projection)


def test_cryptographic_assurance_inflation_requires_attestation():
    capsule, observations, projection = fixture()
    observations[0]["assurance_class"] = "CryptographicallyAttested"
    observations[0]["provider_binding_ref"] = "provider:binding"
    redigest_observation(observations[0])
    projection["declared_source_assurance"] = "CryptographicallyAttested"
    projection["source_observations"][0]["observation_digest"] = observations[0]["observation_digest"]
    redigest_projection(projection)
    with pytest.raises(hak.ProviderSourceLintError):
        validate(capsule, observations, projection)


def test_raw_response_retention_contradiction_rejected():
    capsule, observations, projection = fixture()
    observations[0]["retention"]["raw_response_digest"] = "sha256:" + "1" * 64
    redigest_observation(observations[0])
    with pytest.raises(hak.ProviderSourceLintError):
        validate(capsule, observations, projection)


def test_projection_target_capsule_digest_mismatch_rejected():
    capsule, observations, projection = fixture()
    projection["target"]["capsule_digest"] = "sha256:" + "2" * 64
    redigest_projection(projection)
    with pytest.raises(hak.ProviderSourceLintError):
        validate(capsule, observations, projection)


def test_projection_source_observation_digest_mismatch_rejected():
    capsule, observations, projection = fixture()
    projection["source_observations"][0]["observation_digest"] = "sha256:" + "3" * 64
    redigest_projection(projection)
    with pytest.raises(hak.ProviderSourceLintError):
        validate(capsule, observations, projection)


def test_projection_missing_target_leaf_rejected():
    capsule, observations, projection = fixture()
    projection["mappings"].pop()
    redigest_projection(projection)
    with pytest.raises(hak.ProviderSourceLintError):
        validate(capsule, observations, projection)


def test_projection_duplicate_target_leaf_rejected():
    capsule, observations, projection = fixture()
    projection["mappings"].append(deepcopy(projection["mappings"][0]))
    redigest_projection(projection)
    with pytest.raises(hak.ProviderSourceLintError):
        validate(capsule, observations, projection)


def test_same_valued_cross_scope_source_substitution_rejected():
    capsule, observations, projection = fixture()
    mapping = next(item for item in projection["mappings"] if item["target_path"] == "provider")
    mapping["source_observation_id"] = observations[1]["observation_id"]
    redigest_projection(projection)
    with pytest.raises(hak.ProviderSourceLintError):
        validate(capsule, observations, projection)


def test_projection_source_value_mismatch_rejected():
    capsule, observations, projection = fixture()
    mapping = next(item for item in projection["mappings"] if item["target_path"] == "status")
    mapping["source_path"] = "normalized_payload.conclusion"
    redigest_projection(projection)
    with pytest.raises(hak.ProviderSourceLintError):
        validate(capsule, observations, projection)


def test_projection_digest_tampering_rejected():
    capsule, observations, projection = fixture()
    projection["projection_digest"] = "sha256:" + "4" * 64
    with pytest.raises(hak.ProviderSourceLintError):
        validate(capsule, observations, projection)


def test_step_observation_targets_wrong_job_rejected():
    capsule, observations, projection = fixture()
    step_obs = observations[2]
    step_obs["resource_identity"]["job_id"] += 1
    step_obs["source_ref"] = (
        "github-actions:Luminous-Dynamics/symthaea:"
        "workflow-run/34225891059:attempt/1:job/102059802265:steps"
    )
    redigest_observation(step_obs)
    binding = next(
        item for item in projection["source_observations"]
        if item["observation_id"] == step_obs["observation_id"]
    )
    binding["observation_digest"] = step_obs["observation_digest"]
    redigest_projection(projection)
    with pytest.raises(hak.ProviderSourceLintError):
        validate(capsule, observations, projection)


def test_observation_window_inversion_rejected():
    capsule, observations, projection = fixture()
    observations[0]["observation_window"]["after_or_at"] = "2026-09-08T20:38:00Z"
    observations[0]["observation_window"]["before_or_at"] = "2026-09-08T20:37:41Z"
    redigest_observation(observations[0])
    binding = projection["source_observations"][0]
    binding["observation_digest"] = observations[0]["observation_digest"]
    redigest_projection(projection)
    with pytest.raises(hak.ProviderSourceLintError):
        validate(capsule, observations, projection)


def test_normalization_replayability_overclaim_rejected():
    capsule, observations, projection = fixture()
    observations[0]["normalization"]["replayability"] = "ReplayableFromRetainedRawResponse"
    redigest_observation(observations[0])
    binding = projection["source_observations"][0]
    binding["observation_digest"] = observations[0]["observation_digest"]
    redigest_projection(projection)
    with pytest.raises(hak.ProviderSourceLintError):
        validate(capsule, observations, projection)
