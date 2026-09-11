#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os, pathlib, platform, sqlite3, subprocess, sys, tempfile, time

LAB = pathlib.Path(__file__).with_name("continuity-fenced-resource-lab.py")
ORACLE = pathlib.Path(__file__).with_name("continuity-actuation-enforcement-campaign-oracle.py")
RECORD_DOMAIN = b"symthaea.continuity.fenced-resource-lab-observation.v1\0"
COMPLETE_DOMAIN = b"symthaea.continuity.fenced-resource-lab-complete-set.v1\0"

OBLIGATIONS = (
    ("boundary_identity", "static_implementation_inspection"),
    ("same_boundary_checks_and_mutates", "atomic_check_and_actuate_scenario"),
    ("durable_monotonic_fence", "crash_restart_scenario"),
    ("reject_stale_generation", "stale_holder_scenario"),
    ("reject_replay", "replay_scenario"),
    ("reject_deny_disposition", "deny_scenario"),
    ("emergency_stop_dominates", "emergency_stop_scenario"),
    ("one_use_permit_consumption", "one_use_consumption_scenario"),
    ("crash_recovery_preserves_fence", "crash_restart_scenario"),
)

def hx(n: int) -> str: return (bytes([n]) * 32).hex()
def now_ms() -> int: return time.time_ns() // 1_000_000
def sha(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def sha_text(text: str) -> str: return sha(text.encode("utf-8"))
def canonical(obj) -> bytes: return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")

def run(*args: object, expect: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run([sys.executable, str(LAB), *map(str, args)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != expect:
        raise AssertionError(f"rc={result.returncode} expected={expect}\nstdout={result.stdout}\nstderr={result.stderr}")
    return result

def parsed(result): return json.loads(result.stdout)
def save(path, obj): path.write_text(json.dumps(obj, sort_keys=True), encoding="utf-8")
def require_deny(result, code):
    if code not in result.stderr: raise AssertionError(f"missing deny {code!r}: {result.stderr!r}")

def observation_id(obligation: str, basis: str, evidence) -> str:
    h = hashlib.sha256(); h.update(RECORD_DOMAIN)
    for value in (obligation, basis):
        raw = value.encode("utf-8"); h.update(len(raw).to_bytes(2, "little")); h.update(raw)
    h.update(hashlib.sha256(canonical(evidence)).digest())
    return h.hexdigest()

def main() -> int:
    campaign_start = now_ms()
    observed = {}
    def mark(name, evidence): observed[name] = (now_ms(), evidence)

    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td); db = root / "lab.sqlite"
        rid, bid, pid = hx(1), hx(2), hx(3)
        run("init", "--db", db, "--resource-id", rid, "--backend-id", bid, "--profile-id", pid)

        token1 = parsed(run("issue", "--db", db, "--disposition", "permit", "--challenge", hx(10)))
        token2 = parsed(run("issue", "--db", db, "--disposition", "permit", "--challenge", hx(11)))
        p1, p2 = root / "t1.json", root / "t2.json"; save(p1, token1); save(p2, token2)
        stale1 = run("actuate", "--db", db, "--token", p1, "--delta", 5, "--operation-digest", hx(20), expect=2)
        require_deny(stale1, "actuate:stale_generation"); mark("reject_stale_generation", {"stderr": stale1.stderr})

        applied2 = parsed(run("actuate", "--db", db, "--token", p2, "--delta", 5, "--operation-digest", hx(21)))
        assert applied2["value"] == 5; mark("same_boundary_checks_and_mutates", applied2)
        replay = run("actuate", "--db", db, "--token", p2, "--delta", 1, "--operation-digest", hx(22), expect=2)
        require_deny(replay, "actuate:replay"); mark("reject_replay", {"stderr": replay.stderr})
        snap2 = parsed(run("snapshot", "--db", db))
        assert snap2["value"] == 5 and snap2["current_generation"] == 2 and len(snap2["consumed_tokens"]) == 1

        token3 = parsed(run("issue", "--db", db, "--disposition", "permit", "--challenge", hx(12)))
        p3 = root / "t3.json"; save(p3, token3)
        run("actuate", "--db", db, "--token", p3, "--delta", 7, "--operation-digest", hx(23), "--crash-before-commit", expect=91)
        snap_after_crash = parsed(run("snapshot", "--db", db))
        assert snap_after_crash["value"] == 5 and snap_after_crash["current_generation"] == 3 and len(snap_after_crash["consumed_tokens"]) == 1
        applied3 = parsed(run("actuate", "--db", db, "--token", p3, "--delta", 7, "--operation-digest", hx(23)))
        assert applied3["value"] == 12
        mark("crash_recovery_preserves_fence", {"after_crash": snap_after_crash, "retry": applied3})

        token4 = parsed(run("issue", "--db", db, "--disposition", "permit", "--challenge", hx(13)))
        p4_commit = root / "t4-commit.json"; save(p4_commit, token4)
        run("actuate", "--db", db, "--token", p4_commit, "--delta", 1, "--operation-digest", hx(24), "--crash-after-commit", expect=92)
        snap_after_commit_crash = parsed(run("snapshot", "--db", db))
        assert snap_after_commit_crash["value"] == 13 and snap_after_commit_crash["current_generation"] == 4 and len(snap_after_commit_crash["consumed_tokens"]) == 3
        post_commit_replay = run("actuate", "--db", db, "--token", p4_commit, "--delta", 1, "--operation-digest", hx(31), expect=2)
        require_deny(post_commit_replay, "actuate:replay")

        policy_deny = parsed(run("issue", "--db", db, "--disposition", "deny", "--reason", "policy_deny", "--challenge", hx(14)))
        p4 = root / "policy-deny.json"; save(p4, policy_deny)
        deny_attempt = run("actuate", "--db", db, "--token", p4, "--delta", 1, "--operation-digest", hx(30), expect=2)
        require_deny(deny_attempt, "actuate:deny_disposition")
        mark("reject_deny_disposition", {"stderr": deny_attempt.stderr})

        token5 = parsed(run("issue", "--db", db, "--disposition", "permit", "--challenge", hx(15)))
        p5 = root / "t5.json"; save(p5, token5)
        emergency = parsed(run("issue", "--db", db, "--disposition", "deny", "--reason", "emergency_stop", "--challenge", hx(16)))
        p6 = root / "emergency.json"; save(p6, emergency)
        stale_after_stop = run("actuate", "--db", db, "--token", p5, "--delta", 1, "--operation-digest", hx(25), expect=2)
        require_deny(stale_after_stop, "actuate:stale_generation")
        emergency_attempt = run("actuate", "--db", db, "--token", p6, "--delta", 1, "--operation-digest", hx(26), expect=2)
        require_deny(emergency_attempt, "actuate:emergency_stop")
        sticky = run("issue", "--db", db, "--disposition", "permit", "--challenge", hx(17), expect=2)
        require_deny(sticky, "issue:emergency_stop_sticky")
        mark("emergency_stop_dominates", {"stale": stale_after_stop.stderr, "deny": emergency_attempt.stderr, "sticky": sticky.stderr})

        other = root / "other.sqlite"
        run("init", "--db", other, "--resource-id", hx(4), "--backend-id", bid, "--profile-id", pid)
        parsed(run("issue", "--db", other, "--disposition", "permit", "--challenge", hx(20)))
        wrong_resource = run("actuate", "--db", other, "--token", p1, "--delta", 1, "--operation-digest", hx(27), expect=2)
        require_deny(wrong_resource, "actuate:resource_mismatch")

        other_backend = root / "other-backend.sqlite"
        run("init", "--db", other_backend, "--resource-id", rid, "--backend-id", hx(5), "--profile-id", pid)
        parsed(run("issue", "--db", other_backend, "--disposition", "permit", "--challenge", hx(18)))
        wrong_backend = run("actuate", "--db", other_backend, "--token", p1, "--delta", 1, "--operation-digest", hx(28), expect=2)
        require_deny(wrong_backend, "actuate:backend_mismatch")

        other_profile = root / "other-profile.sqlite"
        run("init", "--db", other_profile, "--resource-id", rid, "--backend-id", bid, "--profile-id", hx(6))
        parsed(run("issue", "--db", other_profile, "--disposition", "permit", "--challenge", hx(19)))
        wrong_profile = run("actuate", "--db", other_profile, "--token", p1, "--delta", 1, "--operation-digest", hx(29), expect=2)
        require_deny(wrong_profile, "actuate:profile_mismatch")
        mark("boundary_identity", {"resource_id": rid, "backend_id": bid, "profile_id": pid, "wrong_resource": wrong_resource.stderr, "wrong_backend": wrong_backend.stderr, "wrong_profile": wrong_profile.stderr})

        final = parsed(run("snapshot", "--db", db))
        assert final["value"] == 13 and final["current_generation"] == 7 and final["emergency_stop"] is True
        assert len(final["consumed_tokens"]) == 3
        mark("one_use_permit_consumption", {"consumed_tokens": final["consumed_tokens"]})
        mark("durable_monotonic_fence", {"generation_2": snap2["current_generation"], "generation_3": snap_after_crash["current_generation"], "generation_7": final["current_generation"]})

        campaign_end = max(now_ms(), max(ts for ts, _ in observed.values()))
        record_ids = []
        records = []
        bases = {}
        for obligation, basis in OBLIGATIONS:
            ts, evidence = observed[obligation]
            record_id = observation_id(obligation, basis, evidence)
            record_ids.append(record_id); bases[obligation] = basis
            records.append({"obligation": obligation, "record_id": record_id, "observed_at_unix_ms": ts})

        complete_set_id = sha(COMPLETE_DOMAIN + b"".join(bytes.fromhex(x) for x in record_ids))
        lab_bytes = LAB.read_bytes(); harness_bytes = pathlib.Path(__file__).read_bytes()
        backend_impl = sha(lab_bytes)
        boundary_impl = sha(b"sqlite-begin-immediate-check-and-mutate.v1\0" + bytes.fromhex(backend_impl))
        one_use = sha(b"sqlite-consumed-token-primary-key.v1\0" + bytes.fromhex(backend_impl))
        scenario_suite = sha_text("\n".join(f"{o}:{b}" for o,b in OBLIGATIONS))
        env = sha_text(json.dumps({"python": sys.version, "sqlite": sqlite3.sqlite_version, "platform": platform.platform()}, sort_keys=True))
        toolchain = sha_text(json.dumps({"implementation": sys.implementation.name, "version": list(sys.version_info[:3]), "cache_tag": sys.implementation.cache_tag}, sort_keys=True))
        manifest = {
            "schema": "symthaea-continuity-actuation-enforcement-campaign-manifest-v1",
            "complete_set_id": complete_set_id,
            "enforcement_profile_id": pid,
            "authentication_profile_id": sha_text("contained-simulation-authentication-profile-unqualified-v1"),
            "backend_id": bid,
            "backend_implementation_digest": backend_impl,
            "backend_generation": 1,
            "boundary_implementation_digest": boundary_impl,
            "one_use_mechanism_digest": one_use,
            "enforcement_profile_generation": 1,
            "campaign_nonce": sha_text(f"{campaign_start}:{os.getpid()}:{root}"),
            "harness_implementation_digest": sha(harness_bytes),
            "scenario_suite_manifest_digest": scenario_suite,
            "environment_manifest_digest": env,
            "topology_dependency_manifest_digest": sha_text("single-process-sqlite-single-resource-boundary-v1"),
            "hardware_firmware_manifest_digest": sha_text("software-simulation-no-physical-hardware-v1"),
            "toolchain_realization_digest": toolchain,
            "started_at_unix_ms": campaign_start,
            "ended_at_unix_ms": campaign_end,
            "records": records,
        }
        manifest_path = root / "campaign.json"; save(manifest_path, manifest)
        oracle = subprocess.run([sys.executable, str(ORACLE), str(manifest_path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if oracle.returncode != 0:
            raise AssertionError(f"campaign oracle rejected lab manifest\nstdout={oracle.stdout}\nstderr={oracle.stderr}")
        oracle_summary = json.loads(oracle.stdout)
        assert oracle_summary["obligation_count"] == 9

        print(json.dumps({
            "status": "PASS",
            "final_generation": final["current_generation"],
            "final_value": final["value"],
            "consumed_count": len(final["consumed_tokens"]),
            "campaign_manifest_sha256": sha(canonical(manifest)),
            "campaign_oracle": oracle_summary,
            "obligation_bases": bases,
            "properties": [
                "stale_generation_rejected", "one_use_replay_rejected", "committed_state_survives_reopen",
                "crash_before_commit_is_atomic", "crash_after_commit_is_durable", "emergency_stop_dominates", "deny_token_cannot_mutate",
                "cross_resource_token_rejected", "nine_obligation_campaign_shape_accepted",
            ],
        }, sort_keys=True))
    return 0

if __name__ == "__main__": raise SystemExit(main())
