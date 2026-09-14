#!/usr/bin/env python3
"""Independent final-sample admission oracle for LQCD-021G."""
import hashlib, json, copy

ORACLE_ID = "lqcd_021g_final_sample_admission_oracle_v1"
FINAL_CAMPAIGN_DIGEST = "497caad8930c3f3067b97928797ea1794055aaa0c4c947e8aeb503102cb2d0a8"
FROZEN_POLICY_DIGEST = hashlib.sha256(b"synthetic-frozen-final-admission-policy-v1").hexdigest()
EXPECTED_CHAINS = tuple(f"chain-{i:02d}" for i in range(4))
EXPECTED_PER_CHAIN = 1000

def h(b): return hashlib.sha256(b).hexdigest()

def expected_config_ids():
    return tuple(
        f"{chain}:retained-{ordinal:04d}"
        for chain in EXPECTED_CHAINS
        for ordinal in range(EXPECTED_PER_CHAIN)
    )

EXPECTED_CONFIGS = expected_config_ids()
EXPECTED_SET = frozenset(EXPECTED_CONFIGS)

REQUIRED_EVIDENCE = (
    "restart_continuity",
    "stationarity",
    "autocorrelation_block_adequacy",
    "center_mobility",
    "topology_slow_mode_adequacy",
    "numerical_integrity",
    "measurement_completeness",
    "environment_lineage_consistency",
)

def canonical_digest(obj):
    return h(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())

def baseline_evidence():
    return {
        "campaign_digest": FINAL_CAMPAIGN_DIGEST,
        "frozen_policy_digest": FROZEN_POLICY_DIGEST,
        "runtime_policy_digest": FROZEN_POLICY_DIGEST,
        "production_started": True,
        "infrastructure_valid": True,
        "retained_config_ids": list(EXPECTED_CONFIGS),
        "extra_unplanned_config_ids": [],
        "dropped_chain_ids": [],
        "corruption_replacements_without_lineage": [],
        "hidden_sample_extension": False,
        "restart_continuity": True,
        "stationarity": True,
        "autocorrelation_block_adequacy": True,
        "center_mobility": True,
        "topology_slow_mode_adequacy": True,
        "numerical_integrity": True,
        "measurement_completeness": True,
        "environment_lineage_consistency": True,
        "chain_agreement": True,
        "missing_required_evidence": [],
    }

def admit(e):
    if not e.get("infrastructure_valid", False):
        return {"disposition":"InfrastructureInvalid","reasons":["InfrastructureInvalid"]}

    if e.get("campaign_digest") != FINAL_CAMPAIGN_DIGEST:
        return {"disposition":"NotAdmissible","reasons":["CampaignSubjectMismatch"]}

    reasons = []
    if e.get("runtime_policy_digest") != e.get("frozen_policy_digest"):
        reasons.append("PostStartPolicyMutation")
    if e.get("frozen_policy_digest") != FROZEN_POLICY_DIGEST:
        reasons.append("FrozenPolicyIdentityMismatch")
    if e.get("hidden_sample_extension"):
        reasons.append("HiddenSequentialSampleExtension")
    if e.get("extra_unplanned_config_ids"):
        reasons.append("UnexpectedExtraConfigurations")
    if e.get("dropped_chain_ids"):
        reasons.append("PostHocChainDeletion")
    if e.get("corruption_replacements_without_lineage"):
        reasons.append("UnverifiedConfigurationReplacement")

    observed = e.get("retained_config_ids")
    if observed is None:
        return {"disposition":"Inconclusive","reasons":["MissingRetainedConfigurationSet"]}
    observed_set = frozenset(observed)
    if len(observed) != len(observed_set):
        reasons.append("DuplicateConfigurationIdentity")
    missing = EXPECTED_SET - observed_set
    unexpected = observed_set - EXPECTED_SET
    if missing:
        reasons.append("RetainedSetIncomplete")
    if unexpected:
        reasons.append("RetainedSetIdentityMismatch")
    if len(observed) != 4000:
        reasons.append("RetainedCountNotExactly4000")

    if not e.get("chain_agreement", True):
        reasons.append("FinalChainDisagreement")

    if reasons:
        return {"disposition":"NotAdmissible","reasons":sorted(set(reasons))}

    missing_evidence = list(e.get("missing_required_evidence", []))
    for field in REQUIRED_EVIDENCE:
        if field not in e:
            missing_evidence.append(field)
    if missing_evidence:
        return {"disposition":"Inconclusive","reasons":["MissingEvidence:" + x for x in sorted(set(missing_evidence))]}

    failed = [field for field in REQUIRED_EVIDENCE if e[field] is False]
    if failed:
        return {"disposition":"NotAdmissible","reasons":["FailedGate:" + x for x in sorted(failed)]}

    return {"disposition":"Admissible","reasons":[]}

def receipt(e):
    decision = admit(e)
    payload = {
        "oracle_id": ORACLE_ID,
        "campaign_digest": e.get("campaign_digest"),
        "frozen_policy_digest": e.get("frozen_policy_digest"),
        "expected_retained_count": 4000,
        "expected_config_set_digest": canonical_digest(list(EXPECTED_CONFIGS)),
        "evidence_digest": canonical_digest(e),
        "decision": decision,
    }
    payload["receipt_digest"] = canonical_digest(payload)
    return payload

def mutate(base, **changes):
    out = copy.deepcopy(base)
    out.update(changes)
    return out

def main():
    base = baseline_evidence()
    ok = receipt(base)
    assert ok["decision"]["disposition"] == "Admissible"

    cases = {}

    slow = mutate(base, topology_slow_mode_adequacy=False)
    cases["slow_mode_failure"] = admit(slow)
    assert cases["slow_mode_failure"] == {
        "disposition":"NotAdmissible",
        "reasons":["FailedGate:topology_slow_mode_adequacy"],
    }

    missing_shard = mutate(base, measurement_completeness=False)
    cases["missing_measurement_shard"] = admit(missing_shard)
    assert cases["missing_measurement_shard"]["disposition"] == "NotAdmissible"

    cfg3999 = copy.deepcopy(base)
    cfg3999["retained_config_ids"] = cfg3999["retained_config_ids"][:-1]
    cases["3999_of_4000"] = admit(cfg3999)
    assert cases["3999_of_4000"]["disposition"] == "NotAdmissible"
    assert "RetainedCountNotExactly4000" in cases["3999_of_4000"]["reasons"]

    policy_mutation = mutate(base, runtime_policy_digest=h(b"mutated-policy-after-production-start"))
    cases["post_start_policy_mutation"] = admit(policy_mutation)
    assert "PostStartPolicyMutation" in cases["post_start_policy_mutation"]["reasons"]

    hidden_extra = copy.deepcopy(base)
    hidden_extra["retained_config_ids"].append("chain-00:retained-1000")
    hidden_extra["extra_unplanned_config_ids"] = ["chain-00:retained-1000"]
    hidden_extra["hidden_sample_extension"] = True
    cases["hidden_extra_after_discrepancy"] = admit(hidden_extra)
    assert cases["hidden_extra_after_discrepancy"]["disposition"] == "NotAdmissible"

    replacement = copy.deepcopy(base)
    replacement["retained_config_ids"][-1] = "chain-03:replacement-next-ordinal"
    replacement["corruption_replacements_without_lineage"] = ["chain-03:replacement-next-ordinal"]
    cases["unverified_replacement"] = admit(replacement)
    assert cases["unverified_replacement"]["disposition"] == "NotAdmissible"

    infra = mutate(base, infrastructure_valid=False)
    cases["infrastructure_failure"] = admit(infra)
    assert cases["infrastructure_failure"]["disposition"] == "InfrastructureInvalid"

    disagree = mutate(base, chain_agreement=False)
    cases["chain_disagreement"] = admit(disagree)
    assert cases["chain_disagreement"]["disposition"] == "NotAdmissible"

    dropped = copy.deepcopy(disagree)
    dropped["dropped_chain_ids"] = ["chain-03"]
    dropped["retained_config_ids"] = [x for x in dropped["retained_config_ids"] if not x.startswith("chain-03:")]
    cases["delete_disagreeing_chain"] = admit(dropped)
    assert "PostHocChainDeletion" in cases["delete_disagreeing_chain"]["reasons"]

    missing_policy_evidence = copy.deepcopy(base)
    del missing_policy_evidence["center_mobility"]
    cases["missing_required_evidence"] = admit(missing_policy_evidence)
    assert cases["missing_required_evidence"]["disposition"] == "Inconclusive"

    failed_receipt = receipt(slow)
    failed_digest_before = failed_receipt["receipt_digest"]
    successor_campaign = h(b"beta6-final-campaign-v2-redesign")
    supersession = {
        "superseded_campaign": FINAL_CAMPAIGN_DIGEST,
        "preserved_failed_receipt": failed_digest_before,
        "successor_campaign": successor_campaign,
    }
    assert failed_receipt["receipt_digest"] == failed_digest_before
    assert successor_campaign != FINAL_CAMPAIGN_DIGEST

    result = {
        "oracle_id": ORACLE_ID,
        "admissible_receipt_digest": ok["receipt_digest"],
        "expected_config_set_digest": ok["expected_config_set_digest"],
        "expected_retained_count": 4000,
        "cases": cases,
        "supersession": supersession,
        "claim_boundary": {
            "final_admission_semantics_established": True,
            "real_final_sample_admissible": False,
            "pilot_thresholds_established": False,
            "ehk_agreement_established": False,
        },
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    print("ok")
    print("result_sha256=" + h(canonical))
    print(canonical.decode())

if __name__ == "__main__":
    main()
