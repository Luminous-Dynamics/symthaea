#!/usr/bin/env python3
"""Exact-head, hash-addressed qualification capsule runner.

The capsule records executor/environment identity. A LOCAL_NIX pass does not
become a GITHUB_HOSTED pass merely because the same profile is used.
"""
from __future__ import annotations
import argparse, hashlib, os, platform, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "symthaea-qualification-capsule-v1"
PROFILE_MAGIC = "SYMTHAEA_QUALIFICATION_PROFILE_V1"
CAPSULE_DOMAIN = b"SYMTHAEA_QUALIFICATION_CAPSULE_V1\0"
EXECUTORS = {"LOCAL_NIX", "GITHUB_HOSTED", "SELF_HOSTED_CI"}
SAFE_ENV = (
    "PATH", "RUSTFLAGS", "RUSTDOCFLAGS", "CARGO_ENCODED_RUSTFLAGS",
    "CARGO_HOME", "CARGO_TARGET_DIR", "RUSTUP_TOOLCHAIN", "NIX_PATH",
    "IN_NIX_SHELL", "GITHUB_ACTIONS", "RUNNER_ENVIRONMENT",
)
SENSITIVE_MARKERS = (
    "TOKEN", "SECRET", "PASSWORD", "PASSWD", "API_KEY", "ACCESS_KEY",
    "PRIVATE_KEY", "CREDENTIAL", "COOKIE", "SESSION",
)
SENSITIVE_EXACT = {
    "SSH_AUTH_SOCK", "SSH_ASKPASS", "GIT_ASKPASS", "GPG_AGENT_INFO",
    "NIX_CONFIG", "NETRC",
}

def child_env():
    """Copy the build environment while removing common credential channels."""
    clean = {}
    removed = []
    for name, value in os.environ.items():
        upper = name.upper()
        sensitive = (
            upper in SENSITIVE_EXACT
            or any(marker in upper for marker in SENSITIVE_MARKERS)
            or upper.startswith(("AWS_", "AZURE_", "GOOGLE_"))
        )
        if sensitive:
            removed.append(name)
        else:
            clean[name] = value
    clean["CARGO_TERM_COLOR"] = "never"
    clean["RUST_BACKTRACE"] = "1"
    return clean, tuple(sorted(removed))

STEP_ARGV = {
    "metadata": ("cargo", "metadata", "--locked", "--format-version", "1"),
    "fmt-statistics": ("cargo", "fmt", "-p", "symthaea-statistics", "--", "--check"),
    "test-statistics": ("cargo", "test", "--locked", "-p", "symthaea-statistics"),
    "doc-statistics": ("cargo", "test", "--locked", "-p", "symthaea-statistics", "--doc"),
    "clippy-statistics": ("cargo", "clippy", "--locked", "-p", "symthaea-statistics",
                          "--all-targets", "--", "-D", "warnings"),
    "wasm-statistics": ("cargo", "check", "--locked", "-p", "symthaea-statistics",
                        "--target", "wasm32-unknown-unknown"),
    "fmt-epidemiology": ("cargo", "fmt", "-p", "symthaea-epidemiology", "--", "--check"),
    "test-epidemiology": ("cargo", "test", "--locked", "-p", "symthaea-epidemiology"),
    "doc-epidemiology": ("cargo", "test", "--locked", "-p", "symthaea-epidemiology", "--doc"),
    "clippy-epidemiology": ("cargo", "clippy", "--locked", "-p", "symthaea-epidemiology",
                            "--all-targets", "--", "-D", "warnings"),
    "wasm-epidemiology": ("cargo", "check", "--locked", "-p", "symthaea-epidemiology",
                          "--target", "wasm32-unknown-unknown"),
}
CONTRACTS = {
    "statistics-active-test-surface-v1": (
        "metadata", "fmt-statistics", "test-statistics", "doc-statistics", "wasm-statistics"),
    "statistics-core-v1": (
        "metadata", "fmt-statistics", "test-statistics", "doc-statistics",
        "clippy-statistics", "wasm-statistics"),
    "epidemiology-surveillance-v1": (
        "metadata", "fmt-epidemiology", "test-epidemiology", "doc-epidemiology",
        "clippy-epidemiology", "wasm-epidemiology"),
}

class CapsuleError(RuntimeError):
    pass

def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def sha(data: bytes):
    return hashlib.sha256(data).hexdigest()

def file_sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def proc(argv, cwd, timeout=None):
    env, _ = child_env()
    return subprocess.run(list(argv), cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=timeout, check=False, env=env)

def text(argv, cwd):
    p = proc(argv, cwd)
    if p.returncode:
        raise CapsuleError(f"{list(argv)!r} failed ({p.returncode}): "
                           f"{p.stderr.decode('utf-8','replace').strip()}")
    return p.stdout.decode("utf-8", "strict").strip()

def origin_repo(value: str):
    value = value.strip()
    for prefix in ("https://github.com/", "http://github.com/", "ssh://git@github.com/",
                   "git://github.com/"):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    else:
        if value.startswith("git@github.com:"):
            value = value[len("git@github.com:"):]
        else:
            return None
    if value.endswith(".git"):
        value = value[:-4]
    parts = value.split("/")
    return value if len(parts) == 2 and all(parts) else None

def parse_profile(path: Path):
    raw = path.read_bytes()
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as e:
        raise CapsuleError("profile must be UTF-8") from e
    if not lines or lines[0] != PROFILE_MAGIC:
        raise CapsuleError(f"profile must begin with {PROFILE_MAGIC}")
    scalars, hashes, steps = {}, [], []
    allowed = {"profile", "repository", "rust_channel", "timeout_seconds"}
    for n, line in enumerate(lines[1:], 2):
        if not line or line.startswith("#"):
            continue
        if "\t" in line or "\r" in line or "=" not in line:
            raise CapsuleError(f"profile line {n}: invalid canonical key=value line")
        key, value = line.split("=", 1)
        if not key or not value:
            raise CapsuleError(f"profile line {n}: empty key/value")
        if key in allowed:
            if key in scalars:
                raise CapsuleError(f"profile line {n}: duplicate {key}")
            scalars[key] = value
        elif key == "hash":
            hashes.append(value)
        elif key == "step":
            steps.append(value)
        else:
            raise CapsuleError(f"profile line {n}: unknown key {key!r}")
    missing = allowed - scalars.keys()
    if missing:
        raise CapsuleError(f"profile missing: {', '.join(sorted(missing))}")
    name = scalars["profile"]
    if name not in CONTRACTS:
        raise CapsuleError(f"unknown profile {name!r}")
    if tuple(steps) != CONTRACTS[name]:
        raise CapsuleError("profile step sequence differs from runner's reviewed contract")
    if not hashes or len(hashes) != len(set(hashes)):
        raise CapsuleError("hash paths must be nonempty and unique")
    try:
        timeout = int(scalars["timeout_seconds"])
    except ValueError as e:
        raise CapsuleError("timeout_seconds must be an integer") from e
    if timeout <= 0:
        raise CapsuleError("timeout_seconds must be positive")
    return {**scalars, "timeout_seconds": timeout, "hashes": tuple(hashes),
            "steps": tuple(steps), "raw": raw, "sha256": sha(raw)}

def outside(output: Path, root: Path):
    o, r = output.resolve(), root.resolve()
    try:
        common = Path(os.path.commonpath((str(o), str(r))))
    except ValueError:
        return
    if common == r:
        raise CapsuleError("output must be outside the Git worktree")

def failure(output: Path, profile_path: Path, status: str, reason: str):
    output.mkdir(parents=True, exist_ok=False)
    (output / "PROFILE").write_bytes(profile_path.read_bytes() if profile_path.is_file() else b"")
    (output / "SOURCE").write_text(f"schema={SCHEMA}\nstatus={status}\n", encoding="utf-8")
    (output / "ENVIRONMENT").write_text(
        f"python={platform.python_version()}\nplatform={platform.platform()}\n", encoding="utf-8")
    (output / "RESULTS").write_text(
        f"final_status={status}\nreason_sha256={sha(reason.encode())}\n", encoding="utf-8")
    (output / "failure.txt").write_text(reason + "\n", encoding="utf-8")
    finish_digest(output)

def finish_digest(output: Path):
    payload = sorted(p.relative_to(output).as_posix() for p in output.rglob("*")
                     if p.is_file() and p.name not in {"SHA256SUMS", "CAPSULE.sha256"})
    sums = "".join(f"{file_sha(output / rel)}  {rel}\n" for rel in payload).encode()
    (output / "SHA256SUMS").write_bytes(sums)
    digest = sha(CAPSULE_DOMAIN + sums)
    (output / "CAPSULE.sha256").write_text(digest + "\n", encoding="ascii")
    return digest

def execute(args):
    root = Path(text(("git", "rev-parse", "--show-toplevel"), Path.cwd())).resolve()
    profile_path, output = Path(args.profile).resolve(), Path(args.output).resolve()
    if output.exists():
        raise CapsuleError(f"output already exists: {output}")
    outside(output, root)
    try:
        profile = parse_profile(profile_path)
    except Exception as e:
        failure(output, profile_path, "FAIL_PROFILE", str(e))
        return 2

    expected = args.expected_head.lower()
    if len(expected) != 40 or any(c not in "0123456789abcdef" for c in expected):
        failure(output, profile_path, "FAIL_SOURCE_IDENTITY", "expected HEAD must be 40 hex")
        return 2
    actual = text(("git", "rev-parse", "HEAD"), root)
    tree = text(("git", "rev-parse", "HEAD^{tree}"), root)
    dirty = text(("git", "status", "--porcelain=v1", "--untracked-files=all"), root)
    repo = origin_repo(text(("git", "remote", "get-url", "origin"), root))
    errors = []
    if actual != expected: errors.append(f"HEAD mismatch: {actual}")
    if repo != profile["repository"]: errors.append(f"repository mismatch: {repo}")
    if dirty: errors.append("worktree is dirty")
    for rel in profile["hashes"]:
        if not (root / rel).is_file(): errors.append(f"missing required file: {rel}")
    if errors:
        failure(output, profile_path, "FAIL_SOURCE_IDENTITY", "; ".join(errors))
        return 2

    rustc, cargo = proc(("rustc", "-vV"), root), proc(("cargo", "-Vv"), root)
    if rustc.returncode or cargo.returncode:
        failure(output, profile_path, "FAIL_ENVIRONMENT", "rustc/cargo version capture failed")
        return 2
    release = next((x.split(": ",1)[1] for x in rustc.stdout.decode(
        "utf-8","replace").splitlines() if x.startswith("release: ")), None)
    if release != profile["rust_channel"]:
        failure(output, profile_path, "FAIL_ENVIRONMENT",
                f"Rust release mismatch: expected {profile['rust_channel']}, got {release}")
        return 2

    output.mkdir(parents=True)
    logs = output / "logs"; logs.mkdir()
    (output / "PROFILE").write_bytes(profile["raw"])
    source = [
        f"schema={SCHEMA}", "source_status=EXACT_HEAD_CLEAN", f"repository={repo}",
        f"head={actual}", f"tree={tree}", f"profile_sha256={profile['sha256']}",
        f"runner_sha256={file_sha(Path(__file__).resolve())}",
    ]
    source += [f"file_sha256.{rel}={file_sha(root / rel)}" for rel in profile["hashes"]]
    (output / "SOURCE").write_text("\n".join(source)+"\n", encoding="utf-8")

    nix = "UNAVAILABLE"
    if shutil.which("nix"):
        n = proc(("nix","--version"), root, timeout=30)
        if n.returncode == 0: nix = n.stdout.decode("utf-8","replace").strip()
    _, removed_env = child_env()
    env = [
        f"executor={args.executor}", f"python={platform.python_version()}",
        f"platform={platform.platform()}", f"machine={platform.machine()}",
        f"rustc_stdout_sha256={sha(rustc.stdout)}", f"cargo_stdout_sha256={sha(cargo.stdout)}",
        f"nix_version_sha256={sha(nix.encode())}",
        f"sanitized_env.removed_count={len(removed_env)}",
        f"sanitized_env.removed_names_sha256={sha(chr(0).join(removed_env).encode())}",
    ]
    for name in SAFE_ENV:
        value = os.environ.get(name)
        env.append(f"env.{name}.present={int(value is not None)}")
        if value is not None: env.append(f"env.{name}.sha256={sha(value.encode('utf-8','surrogateescape'))}")
    (output / "ENVIRONMENT").write_text("\n".join(env)+"\n", encoding="utf-8")
    (logs/"000-rustc.stdout").write_bytes(rustc.stdout); (logs/"000-rustc.stderr").write_bytes(rustc.stderr)
    (logs/"001-cargo.stdout").write_bytes(cargo.stdout); (logs/"001-cargo.stderr").write_bytes(cargo.stderr)
    (logs/"002-nix-version.stdout").write_text(nix+"\n", encoding="utf-8")

    deadline = time.monotonic() + profile["timeout_seconds"]
    results = [f"schema={SCHEMA}", f"profile={profile['profile']}",
               f"executor={args.executor}", f"started_at={now()}"]
    final = "PASS_EXACT_HEAD"
    for i, step in enumerate(profile["steps"], 1):
        argv = STEP_ARGV[step]; remain = deadline - time.monotonic()
        if remain <= 0:
            results += [f"step.{i:03d}.id={step}", f"step.{i:03d}.status=TIMEOUT_NOT_STARTED"]
            final = "FAIL_COMMAND"; break
        outp, errp = logs/f"{100+i:03d}-{step}.stdout", logs/f"{100+i:03d}-{step}.stderr"
        started = now()
        try:
            p = proc(argv, root, timeout=remain); code=p.returncode; status="PASS" if code==0 else "FAIL"
            outp.write_bytes(p.stdout); errp.write_bytes(p.stderr)
        except subprocess.TimeoutExpired as e:
            outp.write_bytes(e.stdout or b""); errp.write_bytes(e.stderr or b"")
            code, status = 124, "TIMEOUT"
        results += [
            f"step.{i:03d}.id={step}",
            f"step.{i:03d}.argv_sha256={sha(chr(0).join(argv).encode())}",
            f"step.{i:03d}.started_at={started}", f"step.{i:03d}.ended_at={now()}",
            f"step.{i:03d}.exit_code={code}", f"step.{i:03d}.status={status}",
            f"step.{i:03d}.stdout_sha256={file_sha(outp)}",
            f"step.{i:03d}.stderr_sha256={file_sha(errp)}",
        ]
        if status != "PASS": final="FAIL_COMMAND"; break
    results += [f"final_status={final}", f"ended_at={now()}"]
    (output/"RESULTS").write_text("\n".join(results)+"\n", encoding="utf-8")
    digest = finish_digest(output)
    print(f"{final}: {output}\nCAPSULE_SHA256={digest}")
    return 0 if final == "PASS_EXACT_HEAD" else 1

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--executor", required=True, choices=sorted(EXECUTORS))
    p.add_argument("--output", required=True)
    a=p.parse_args()
    try: return execute(a)
    except CapsuleError as e:
        print(f"qualification capsule error: {e}", file=sys.stderr); return 2

if __name__ == "__main__":
    raise SystemExit(main())
