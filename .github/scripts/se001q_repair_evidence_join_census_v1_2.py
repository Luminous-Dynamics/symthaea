#!/usr/bin/env python3
import argparse, hashlib, json, re, zipfile
from pathlib import Path, PurePosixPath

DOMAIN = "symthaea.se001q.repair-evidence-join-census.v1.2"
SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def die(msg):
    raise SystemExit(msg)


def load(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), raw


def reject_authority(value, path="$"):
    if isinstance(value, dict):
        for key, child in value.items():
            here = f"{path}.{key}"
            if key == "repair_authority":
                die(f"forbidden repair_authority field at {here}")
            if key == "repair_authority_claim" and child != "NONE":
                die(f"repair authority violation at {here}")
            if key == "qualification_claim" and child != "NONE":
                die(f"qualification authority violation at {here}")
            if key == "sufficient_for_repair_grant" and child is not False:
                die(f"repair-grant sufficiency violation at {here}")
            reject_authority(child, here)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            reject_authority(child, f"{path}[{i}]")


def non_authorizing(obj, what):
    authority = obj.get("authority")
    if not isinstance(authority, dict):
        die(f"{what} authority object missing")
    if (
        authority.get("sufficient_for_repair_grant") is not False
        or authority.get("qualification_claim") != "NONE"
        or authority.get("repair_authority_claim") != "NONE"
    ):
        die(f"{what} authority tuple mismatch")


def require_sha(value, what):
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        die(f"invalid {what}")
    return value


def safe_rel(path):
    if not isinstance(path, str) or not path or "\\" in path:
        return False
    p = PurePosixPath(path)
    return not p.is_absolute() and ".." not in p.parts and "." not in p.parts and str(p) == path


def verify_content_addressed(obj, schema, id_key, what):
    if obj.get("schema") != schema:
        die(f"{what} schema mismatch")
    ident = obj.get("identity")
    if not isinstance(ident, dict):
        die(f"{what} identity missing")
    if obj.get(id_key) != digest(canonical(ident)):
        die(f"{what} content-addressed identity mismatch")
    return ident


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", required=True)
    ap.add_argument("--prior-join", required=True)
    ap.add_argument("--diagnostic-source-archive", required=True)
    ap.add_argument("--output", required=True)
    ns = ap.parse_args()

    contract, contract_bytes = load(ns.contract)
    prior, _ = load(ns.prior_join)
    reject_authority(contract)
    reject_authority(prior)
    if (
        contract.get("schema") != "symthaea.se001q.repair-evidence-join-census-contract.v1.2"
        or contract.get("domain") != DOMAIN
    ):
        die("bad census contract")
    non_authorizing(contract, "census contract")

    pi = verify_content_addressed(
        prior,
        contract["required_prior_join_schema"],
        "repair_evidence_join_id",
        "prior join",
    )
    non_authorizing(pi, "prior join")
    if pi.get("result") != contract["required_prior_join_result"]:
        die("prior join result mismatch")
    if pi.get("subject_sha") != contract["expected_subject_sha"]:
        die("prior join subject mismatch")
    source = pi.get("diagnostic_source")
    if not isinstance(source, dict):
        die("prior join diagnostic source missing")
    if source.get("artifact_zip_sha256") != contract["expected_source_artifact_zip_sha256"]:
        die("prior join source artifact mismatch")
    if source.get("evidence_manifest_id") != contract["expected_source_evidence_manifest_id"]:
        die("prior join source manifest mismatch")

    archive = Path(ns.diagnostic_source_archive)
    archive_bytes = archive.read_bytes()
    archive_sha = digest(archive_bytes)
    if archive_sha != contract["expected_source_artifact_zip_sha256"]:
        die("source archive digest mismatch")

    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        names = [i.filename for i in infos if not i.is_dir()]
        if len(names) != len(set(names)):
            die("duplicate ZIP member name")
        if "evidence/manifest.json" not in names:
            die("evidence manifest missing")

        manifest = json.loads(zf.read("evidence/manifest.json"))
        mi = dict(manifest)
        manifest_id = mi.pop("manifest_id", None)
        if manifest_id != digest(canonical(mi)):
            die("evidence manifest content-addressed identity mismatch")
        if manifest_id != contract["expected_source_evidence_manifest_id"]:
            die("evidence manifest ID mismatch")
        if manifest.get("subject_sha") != contract["expected_subject_sha"]:
            die("evidence manifest subject mismatch")

        entries = manifest.get("files")
        if not isinstance(entries, list) or len(entries) != contract["expected_source_evidence_file_count"]:
            die("evidence manifest file count mismatch")

        manifest_map = {}
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                die(f"manifest entry {idx} malformed")
            rel = entry.get("path")
            if not safe_rel(rel):
                die(f"unsafe manifest path at entry {idx}")
            if rel in manifest_map:
                die(f"duplicate manifest path: {rel}")
            manifest_map[rel] = require_sha(entry.get("sha256"), f"manifest digest for {rel}")

        evidence_names = [n for n in names if n.startswith("evidence/")]
        for n in evidence_names:
            rel = n[len("evidence/") :]
            if not safe_rel(rel):
                die(f"unsafe evidence ZIP member: {n}")

        expected_names = {"evidence/manifest.json"} | {f"evidence/{p}" for p in manifest_map}
        if set(evidence_names) != expected_names:
            missing = sorted(expected_names - set(evidence_names))
            extra = sorted(set(evidence_names) - expected_names)
            die(f"evidence ZIP census mismatch missing={missing} extra={extra}")

        census = []
        for rel in sorted(manifest_map):
            raw = zf.read(f"evidence/{rel}")
            actual = digest(raw)
            if actual != manifest_map[rel]:
                die(f"manifest digest mismatch: {rel}")
            census.append({"path": rel, "sha256": actual, "bytes": len(raw)})

    census_digest = digest(canonical(census))
    identity = {
        "domain": DOMAIN,
        "contract_sha256": digest(contract_bytes),
        "prior_repair_evidence_join_id": prior["repair_evidence_join_id"],
        "subject_sha": pi["subject_sha"],
        "source_artifact_zip_sha256": archive_sha,
        "source_artifact_bytes": len(archive_bytes),
        "source_evidence_manifest_id": manifest_id,
        "source_evidence_file_count": len(census),
        "source_evidence_census_sha256": census_digest,
        "source_lock_sha256": pi["source_lock_sha256"],
        "generated_lock_sha256": pi["generated_lock_sha256"],
        "result": contract["result"],
        "authority": {
            "meaning": contract["authority"]["meaning"],
            "sufficient_for_repair_grant": False,
            "qualification_claim": "NONE",
            "repair_authority_claim": "NONE",
        },
    }
    output = {
        "schema": DOMAIN,
        "repair_evidence_census_join_id": digest(canonical(identity)),
        "identity": identity,
    }
    reject_authority(output)
    non_authorizing(identity, "census join")

    p = Path(ns.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "schema": DOMAIN,
                "result": identity["result"],
                "repair_evidence_census_join_id": output["repair_evidence_census_join_id"],
                "prior_repair_evidence_join_id": identity["prior_repair_evidence_join_id"],
                "source_evidence_file_count": identity["source_evidence_file_count"],
                "source_evidence_census_sha256": identity["source_evidence_census_sha256"],
                "sufficient_for_repair_grant": False,
                "qualification_claim": "NONE",
                "repair_authority_claim": "NONE",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
