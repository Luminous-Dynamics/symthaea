#!/usr/bin/env python3
from __future__ import annotations
import json, pathlib, subprocess, sys, tempfile

LAB = pathlib.Path(__file__).with_name("continuity-fenced-resource-lab.py")

def hx(n: int) -> str:
    return (bytes([n]) * 32).hex()

def run(*args: object, expect: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run([sys.executable, str(LAB), *map(str, args)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != expect:
        raise AssertionError(f"rc={result.returncode} expected={expect}\nstdout={result.stdout}\nstderr={result.stderr}")
    return result

def parsed(result: subprocess.CompletedProcess[str]):
    return json.loads(result.stdout)

def save(path: pathlib.Path, obj) -> None:
    path.write_text(json.dumps(obj, sort_keys=True), encoding="utf-8")

def require_deny(result: subprocess.CompletedProcess[str], code: str) -> None:
    if code not in result.stderr:
        raise AssertionError(f"missing deny {code!r}: {result.stderr!r}")

def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td); db = root / "lab.sqlite"
        rid, bid, pid = hx(1), hx(2), hx(3)
        run("init", "--db", db, "--resource-id", rid, "--backend-id", bid, "--profile-id", pid)

        token1 = parsed(run("issue", "--db", db, "--disposition", "permit", "--challenge", hx(10)))
        token2 = parsed(run("issue", "--db", db, "--disposition", "permit", "--challenge", hx(11)))
        p1, p2 = root / "t1.json", root / "t2.json"; save(p1, token1); save(p2, token2)

        denied = run("actuate", "--db", db, "--token", p1, "--delta", 5, "--operation-digest", hx(20), expect=2)
        require_deny(denied, "actuate:stale_generation")

        applied2 = parsed(run("actuate", "--db", db, "--token", p2, "--delta", 5, "--operation-digest", hx(21)))
        assert applied2["value"] == 5
        replay = run("actuate", "--db", db, "--token", p2, "--delta", 1, "--operation-digest", hx(22), expect=2)
        require_deny(replay, "actuate:replay")
        snap = parsed(run("snapshot", "--db", db))
        assert snap["value"] == 5 and snap["current_generation"] == 2 and len(snap["consumed_tokens"]) == 1

        token3 = parsed(run("issue", "--db", db, "--disposition", "permit", "--challenge", hx(12)))
        p3 = root / "t3.json"; save(p3, token3)
        run("actuate", "--db", db, "--token", p3, "--delta", 7, "--operation-digest", hx(23), "--crash-before-commit", expect=91)
        snap = parsed(run("snapshot", "--db", db))
        assert snap["value"] == 5 and snap["current_generation"] == 3 and len(snap["consumed_tokens"]) == 1
        applied3 = parsed(run("actuate", "--db", db, "--token", p3, "--delta", 7, "--operation-digest", hx(23)))
        assert applied3["value"] == 12

        deny_token = parsed(run("issue", "--db", db, "--disposition", "deny", "--reason", "emergency_stop", "--challenge", hx(13)))
        p4 = root / "deny.json"; save(p4, deny_token)
        stale = run("actuate", "--db", db, "--token", p3, "--delta", 1, "--operation-digest", hx(24), expect=2)
        require_deny(stale, "actuate:stale_generation")
        deny_attempt = run("actuate", "--db", db, "--token", p4, "--delta", 1, "--operation-digest", hx(25), expect=2)
        require_deny(deny_attempt, "actuate:emergency_stop")
        sticky = run("issue", "--db", db, "--disposition", "permit", "--challenge", hx(14), expect=2)
        require_deny(sticky, "issue:emergency_stop_sticky")

        other = root / "other.sqlite"
        run("init", "--db", other, "--resource-id", hx(4), "--backend-id", bid, "--profile-id", pid)
        parsed(run("issue", "--db", other, "--disposition", "permit", "--challenge", hx(15)))
        wrong_resource = run("actuate", "--db", other, "--token", p1, "--delta", 1, "--operation-digest", hx(26), expect=2)
        require_deny(wrong_resource, "actuate:resource_mismatch")

        final = parsed(run("snapshot", "--db", db))
        assert final["value"] == 12 and final["current_generation"] == 4 and final["emergency_stop"] is True
        assert len(final["consumed_tokens"]) == 2
        print(json.dumps({
            "status": "PASS",
            "final_generation": final["current_generation"],
            "final_value": final["value"],
            "consumed_count": len(final["consumed_tokens"]),
            "properties": [
                "stale_generation_rejected",
                "one_use_replay_rejected",
                "committed_state_survives_reopen",
                "crash_before_commit_is_atomic",
                "emergency_stop_dominates",
                "deny_token_cannot_mutate",
                "cross_resource_token_rejected",
            ],
        }, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
