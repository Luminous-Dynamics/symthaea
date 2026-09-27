#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import pathlib
import sys

EXPECTED_SHA256 = "87cbe6e38b84aaeecfa188997774b64aec209b314f158b2f7867ee9f34b29665"
EXPECTED_ORDER = [f"B{i:02d}" for i in range(1, 24)]
EXPECTED_CLASSES = [
    "OfflinePerceptionBenchmark",
    "OfflinePredictionForecastBenchmark",
    "OpenLoopPlanningBenchmark",
    "ClosedLoopSimulationBenchmark",
    "ScenarioConformanceBenchmark",
    "ComputeLatencyThroughputBenchmark",
    "SyntheticToHilDomainGapBenchmark",
    "PhysicalTestbedBenchmark",
    "OperationalOutcomeComparison",
    "ReferenceDatasetOnly",
    "ExternalStandardScenarioProfile",
]
EXPECTED_DISPOSITIONS = ["PreferredFirst", "Candidate", "ReferenceOnly", "BenchmarkGap", "Deferred"]
TOP_KEYS = {
    "schema","issue","source_kind","planning_snapshot_date","claim_ceiling",
    "dispositions","rights_rule","benchmark_classes","order","benchmarks"
}
ROW_KEYS = {
    "id","mode","benchmark","class","version","task","status","evaluator","rights",
    "exposure","disposition","claim_ceiling","source_ref"
}
RIGHT_KEYS = {"training","local_evaluation","redistribution","commercial"}
ALLOWED_RIGHT_VALUES = {
    "NonCommercialOnly","RestrictedByDatasetTerms","NotAdmittedByDefault",
    "RestrictedByWaymaxLicense","UnknownNeedsReview",
    "SubjectToCARLAAndLeaderboardTerms","NotApplicable",
    "SpecificationUseSubjectToTerms","SpecificationTermsApply",
    "BenchmarkSpecificRules","NotPrimaryTrainingDataset",
    "ResearchNonCommercialProfile","ReferenceUseNeedsReview","ISOtermsApply",
    "NonCommercialOrResearchRestrictedProfileNeedsExactReview","NeedsExactReview",
}
EXPECTED_CEILINGS = {
"B01":"BenchmarkSpecificPerceptionPerformanceOnly",
"B02":"BenchmarkSpecificMotionForecastPerformanceOnly",
"B03":"LoggedDatasetE2EResearchPerformanceOnly",
"B04":"ClosedLoopSimulationPerformanceOnly",
"B05":"TaskSpecificNuScenesPerformanceOnly",
"B06":"TaskSpecificMotionForecastPerformanceOnly",
"B07":"PlanningBenchmarkPerformanceOnly",
"B08":"CARLAClosedLoopSimulationPerformanceOnly",
"B09":"ScenarioSpecificMotionPlanningPerformanceOnly",
"B10":"ScenarioRepresentationOrConformanceOnly",
"B11":"ScenarioCorpusEvaluationOnly",
"B12":"AutomotiveMLComputePerformanceOnly",
"B13":"ExactPublishedODDOutcomeComparisonOnly",
"B14":"RailPerceptionAndOdometryBenchmarkOnly",
"B15":"OfflineTelemetryTaskPerformanceOnly",
"B16":"ScenarioProcessReferenceOnly",
"B17":"ODDOEReferenceOnly",
"B18":"NoClaimUntilExactSourceSelected",
"B19":"NoExternalBenchmarkClaim",
"B20":"SpacecraftPoseBenchmarkOnly",
"B21":"SyntheticToHILPoseDomainGapOnly",
"B22":"PhysicalTestPrecedentOnly",
"B23":"NoUniversalExternalBenchmarkClaim",
}

class ValidationError(Exception):
    pass

def req(cond, msg):
    if not cond:
        raise ValidationError(msg)

def index_rows(doc):
    rows = doc["benchmarks"]
    req(isinstance(rows, list), "benchmarks must be list")
    ids = [r.get("id") if isinstance(r, dict) else None for r in rows]
    req(ids == EXPECTED_ORDER, f"benchmark row IDs/order mismatch: {ids}")
    req(len(set(ids)) == len(ids), "duplicate benchmark ID")
    return {r["id"]: r for r in rows}

def validate(doc):
    req(isinstance(doc, dict), "root must be object")
    req(set(doc) == TOP_KEYS, f"top-level keys mismatch: {sorted(set(doc) ^ TOP_KEYS)}")
    req(doc["schema"] == "transport-bench-ext-001a-registry-v1", "schema mismatch")
    req(doc["issue"] == 6257, "issue mismatch")
    req(doc["source_kind"] == "external-benchmark-registry-reference", "source kind mismatch")
    req(doc["planning_snapshot_date"] == "2026-09-27", "snapshot mismatch")
    req(doc["claim_ceiling"] == "ArchitectureRegistryOnly", "registry claim ceiling mismatch")
    req(doc["dispositions"] == EXPECTED_DISPOSITIONS, "dispositions mismatch")
    req(doc["benchmark_classes"] == EXPECTED_CLASSES, "benchmark classes mismatch")
    req(doc["order"] == EXPECTED_ORDER, "order mismatch")
    req(
        doc["rights_rule"] ==
        "Unknown or unclear rights remain UnknownNeedsReview; this registry is not legal advice and does not itself admit use.",
        "rights rule mismatch",
    )
    by_id = index_rows(doc)

    for bid in EXPECTED_ORDER:
        row = by_id[bid]
        req(set(row) == ROW_KEYS, f"{bid}: row keys mismatch")
        for key in ("mode","benchmark","class","version","task","status","evaluator","exposure","disposition","claim_ceiling","source_ref"):
            req(isinstance(row[key], str) and row[key].strip(), f"{bid}: empty/non-string {key}")
        req(row["class"] in EXPECTED_CLASSES, f"{bid}: unknown benchmark class")
        req(row["disposition"] in EXPECTED_DISPOSITIONS, f"{bid}: unknown disposition")
        req(row["claim_ceiling"] == EXPECTED_CEILINGS[bid], f"{bid}: claim ceiling mismatch")
        req(isinstance(row["rights"], dict) and set(row["rights"]) == RIGHT_KEYS, f"{bid}: rights keys mismatch")
        for rk, rv in row["rights"].items():
            req(isinstance(rv, str) and rv in ALLOWED_RIGHT_VALUES, f"{bid}: invalid rights value {rk}={rv}")

    for bid in ("B01","B02","B03"):
        r = by_id[bid]["rights"]
        req(r["training"] == "NonCommercialOnly", f"{bid}: Waymo training must remain non-commercial")
        req(r["local_evaluation"] == "NonCommercialOnly", f"{bid}: Waymo eval must remain non-commercial")
        req(r["redistribution"] == "RestrictedByDatasetTerms", f"{bid}: Waymo redistribution restriction lost")
        req(r["commercial"] == "NotAdmittedByDefault", f"{bid}: Waymo commercial use escalated")

    r = by_id["B04"]
    req(r["benchmark"] == "Waymax", "B04 benchmark identity mismatch")
    req(r["rights"]["training"] == "NonCommercialOnly", "B04 training rights escalation")
    req(r["rights"]["local_evaluation"] == "NonCommercialOnly", "B04 eval rights escalation")
    req(r["rights"]["redistribution"] == "RestrictedByWaymaxLicense", "B04 redistribution mismatch")
    req(r["rights"]["commercial"] == "NotAdmittedByDefault", "B04 commercial use escalation")

    r = by_id["B08"]
    req(r["benchmark"] == "CARLA Autonomous Driving Leaderboard", "B08 identity mismatch")
    req(r["version"] == "Leaderboard 2.1", "B08 version mismatch")
    req(r["class"] == "ClosedLoopSimulationBenchmark", "B08 must remain simulation benchmark")

    r = by_id["B12"]
    req(r["benchmark"] == "MLPerf Automotive", "B12 identity mismatch")
    req(r["version"] == "v0.5", "B12 version mismatch")
    req(r["class"] == "ComputeLatencyThroughputBenchmark", "B12 must remain compute benchmark")

    r = by_id["B14"]
    req(r["benchmark"] == "RAIL-BENCH", "B14 identity mismatch")
    req(r["class"] == "OfflinePerceptionBenchmark", "B14 class mismatch")
    req("visual odometry" in r["task"], "B14 task census weakened")

    for bid in ("B16","B17"):
        req(by_id[bid]["status"] == "UnderDevelopmentNotPublishedStandard", f"{bid}: AWI publication status escalated")
        req(by_id[bid]["class"] == "ExternalStandardScenarioProfile", f"{bid}: AWI class mismatch")

    for bid in ("B19","B23"):
        r = by_id[bid]
        req(r["status"] == "BenchmarkGap", f"{bid}: benchmark gap status lost")
        req(r["disposition"] == "BenchmarkGap", f"{bid}: benchmark gap disposition lost")
        req(r["evaluator"] == "None", f"{bid}: fake evaluator added")
        req(all(v == "NotApplicable" for v in r["rights"].values()), f"{bid}: gap acquired fake rights")

    req(by_id["B20"]["class"] == "OfflinePerceptionBenchmark", "B20 must remain pose/perception benchmark")
    req(by_id["B21"]["class"] == "SyntheticToHilDomainGapBenchmark", "B21 must remain synthetic-to-HIL benchmark")
    req(by_id["B22"]["disposition"] == "ReferenceOnly", "B22 must remain reference-only")
    return True

def load_and_check(path):
    raw = pathlib.Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    req(actual == EXPECTED_SHA256, f"raw corpus SHA-256 mismatch: {actual}")
    doc = json.loads(raw.decode("utf-8"))
    canonical = json.dumps(doc, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    req(raw == canonical, "registry bytes are not canonical compact JSON")
    validate(doc)
    return doc

def mutate(doc, fn):
    x = copy.deepcopy(doc)
    fn(x)
    return x

def expect_reject(name, doc):
    try:
        validate(doc)
    except (ValidationError, KeyError, TypeError, ValueError):
        return
    raise AssertionError(f"mutation unexpectedly accepted: {name}")

def run_mutations(doc):
    tests = []
    def add(name, fn):
        tests.append((name, mutate(doc, fn)))

    add("schema", lambda d: d.__setitem__("schema", "evil-v2"))
    add("issue", lambda d: d.__setitem__("issue", 1))
    add("source_kind", lambda d: d.__setitem__("source_kind", "performance-authority"))
    add("snapshot", lambda d: d.__setitem__("planning_snapshot_date", "2099-01-01"))
    add("registry_authority", lambda d: d.__setitem__("claim_ceiling", "PhysicalAuthority"))
    add("drop_top", lambda d: d.pop("rights_rule"))
    add("extra_top", lambda d: d.__setitem__("trusted", True))
    add("duplicate_order", lambda d: d["order"].__setitem__(1, "B01"))
    add("missing_order", lambda d: d["order"].pop())
    add("reorder", lambda d: d["order"].__setitem__(slice(0,2), ["B02","B01"]))
    add("duplicate_row", lambda d: d["benchmarks"].__setitem__(1, copy.deepcopy(d["benchmarks"][0])))
    add("missing_row", lambda d: d["benchmarks"].pop())
    add("unknown_class", lambda d: d["benchmarks"][0].__setitem__("class", "UniversalSafetyBenchmark"))
    add("unknown_disposition", lambda d: d["benchmarks"][0].__setitem__("disposition", "BestBenchmark"))
    add("empty_mode", lambda d: d["benchmarks"][0].__setitem__("mode", ""))
    add("empty_benchmark", lambda d: d["benchmarks"][0].__setitem__("benchmark", ""))
    add("empty_version", lambda d: d["benchmarks"][0].__setitem__("version", ""))
    add("empty_task", lambda d: d["benchmarks"][0].__setitem__("task", ""))
    add("empty_evaluator", lambda d: d["benchmarks"][0].__setitem__("evaluator", ""))
    add("rights_missing", lambda d: d["benchmarks"][0]["rights"].pop("commercial"))
    add("rights_extra", lambda d: d["benchmarks"][0]["rights"].__setitem__("deployment", "Allowed"))
    add("waymo_training_commercial", lambda d: d["benchmarks"][0]["rights"].__setitem__("training", "CommercialAllowed"))
    add("waymo_commercial", lambda d: d["benchmarks"][1]["rights"].__setitem__("commercial", "CommercialAllowed"))
    add("waymo_redistribution", lambda d: d["benchmarks"][2]["rights"].__setitem__("redistribution", "OpenRedistribution"))
    add("waymax_commercial", lambda d: d["benchmarks"][3]["rights"].__setitem__("commercial", "CommercialAllowed"))
    add("carla_physical", lambda d: d["benchmarks"][7].__setitem__("class", "PhysicalTestbedBenchmark"))
    add("carla_old_version", lambda d: d["benchmarks"][7].__setitem__("version", "Leaderboard 2.0"))
    add("mlperf_version", lambda d: d["benchmarks"][11].__setitem__("version", "v0.6"))
    add("mlperf_driving_quality", lambda d: d["benchmarks"][11].__setitem__("claim_ceiling", "DrivingQuality"))
    add("rail_autonomy", lambda d: d["benchmarks"][13].__setitem__("claim_ceiling", "AutomatedTrainOperation"))
    add("awi25927_published", lambda d: d["benchmarks"][15].__setitem__("status", "PublishedStandard"))
    add("awi25930_published", lambda d: d["benchmarks"][16].__setitem__("status", "PublishedStandard"))
    add("auv_fake_benchmark", lambda d: d["benchmarks"][18].__setitem__("disposition", "PreferredFirst"))
    add("launch_fake_benchmark", lambda d: d["benchmarks"][22].__setitem__("status", "CurrentLeaderboard"))
    add("speed_onorbit", lambda d: d["benchmarks"][19].__setitem__("claim_ceiling", "OnOrbitRendezvousQualified"))
    add("speedplus_physical", lambda d: d["benchmarks"][20].__setitem__("class", "PhysicalTestbedBenchmark"))
    add("speedplus_mission", lambda d: d["benchmarks"][20].__setitem__("claim_ceiling", "MissionAuthority"))
    add("rpod_preferred", lambda d: d["benchmarks"][21].__setitem__("disposition", "PreferredFirst"))
    add("rights_rule_permissive", lambda d: d.__setitem__("rights_rule", "Unknown rights are allowed"))
    add("unknown_right_promoted", lambda d: d["benchmarks"][4]["rights"].__setitem__("commercial", "CommercialAllowed"))

    for name, candidate in tests:
        expect_reject(name, candidate)
    return len(tests)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("registry")
    p.add_argument("--mutations", action="store_true")
    args = p.parse_args()
    try:
        doc = load_and_check(args.registry)
        count = run_mutations(doc) if args.mutations else 0
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print(f"PASS: 23 registry entries; {count} adversarial mutations" if args.mutations else "PASS: 23 registry entries")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
