import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
MANIFEST_PATH = ROOT / "docs" / "operations" / "qualification-landing-source-manifest-v1.json"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


landing = load("qualification_landing_manifest_v1")


def git(repo, *args):
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout.strip()


def source_repo(root):
    repo = root / "source"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "QUAL test")
    git(repo, "config", "user.email", "qual@example.invalid")
    (repo / "source.txt").write_text("frozen qualification source\n", encoding="utf-8")
    git(repo, "add", "source.txt")
    git(repo, "commit", "-q", "-m", "source")
    commit = git(repo, "rev-parse", "HEAD")
    blob = git(repo, "rev-parse", f"{commit}:source.txt")
    return repo, commit, blob


def entry(
    role,
    commit,
    blob,
    *,
    status,
    domain,
    deps=None,
    external=None,
    goldens=None,
    migration=None,
    path="source.txt",
):
    return {
        "logical_role": role,
        "source_branch": "historical/source",
        "source_commit_sha": commit,
        "source_path": path,
        "source_blob_id": blob,
        "semantic_schema_or_domain": domain,
        "normative_status": status,
        "required_entry_roles": sorted(deps or []),
        "required_external_primitives": sorted(external or []),
        "golden_vector_refs": sorted(goldens or []),
        "claim_ceiling": ["fixture claim ceiling only"],
        "migration_or_equivalence_ref": migration,
        "notes": [],
    }


def manifest(commit, blob):
    return {
        "schema": landing.SCHEMA,
        "qualification_era_target": landing.ERA,
        "normative_framing_role": "normative_framing",
        "external_primitives": ["execution-lineage:test"],
        "entries": [
            entry(
                "normative_framing",
                commit,
                blob,
                status="NormativeCandidate",
                domain="symthaea.qualification-framing.v1",
                goldens=["test_framing"],
            ),
            entry(
                "test_framing",
                commit,
                blob,
                status="TestVector",
                domain="qualification-framing-test-vector",
            ),
        ],
    }


def sort_entries(value):
    value["entries"] = sorted(value["entries"], key=lambda item: item["logical_role"])


class LandingManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo, self.commit, self.blob = source_repo(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_repository_manifest_is_structurally_valid(self):
        value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        normalized = landing.validate_manifest(value, verify_git=False)
        self.assertEqual(len(normalized["entries"]), 38)
        receipt_status = {
            item["logical_role"]: item["normative_status"]
            for item in normalized["entries"]
            if item["semantic_schema_or_domain"] == landing.RECEIPT_V1_DOMAIN
        }
        self.assertEqual(
            receipt_status,
            {
                "positive_receipt_core_v1_python": "HistoricalOnly",
                "positive_receipt_core_v1_rust": "HistoricalOnly",
            },
        )

    def test_exact_git_commit_path_blob_binding(self):
        normalized = landing.validate_manifest(
            manifest(self.commit, self.blob), repo=self.repo, verify_git=True
        )
        self.assertEqual(normalized["entries"][0]["source_blob_id"], self.blob)

    def test_blob_substitution_is_rejected(self):
        value = manifest(self.commit, self.blob)
        value["entries"][0]["source_blob_id"] = "0" * 40
        with self.assertRaisesRegex(landing.ManifestError, "source blob mismatch"):
            landing.validate_manifest(value, repo=self.repo, verify_git=True)

    def test_missing_source_path_is_rejected(self):
        value = manifest(self.commit, self.blob)
        value["entries"][0]["source_path"] = "missing.txt"
        with self.assertRaisesRegex(landing.ManifestError, "git rev-parse"):
            landing.validate_manifest(value, repo=self.repo, verify_git=True)

    def test_parallel_normative_framing_is_rejected(self):
        value = manifest(self.commit, self.blob)
        value["entries"] += [
            entry(
                "other_framing",
                self.commit,
                self.blob,
                status="NormativeCandidate",
                domain="symthaea.other-qualification-framing.v1",
                goldens=["test_other_framing"],
            ),
            entry(
                "test_other_framing",
                self.commit,
                self.blob,
                status="TestVector",
                domain="other-framing-test-vector",
            ),
        ]
        sort_entries(value)
        with self.assertRaisesRegex(landing.ManifestError, "exactly one normative framing candidate"):
            landing.validate_manifest(value, verify_git=False)

    def test_receipt_v1_cannot_be_promoted_to_normative(self):
        value = manifest(self.commit, self.blob)
        value["entries"].append(
            entry(
                "receipt_v1",
                self.commit,
                self.blob,
                status="NormativeCandidate",
                domain=landing.RECEIPT_V1_DOMAIN,
                goldens=["test_framing"],
            )
        )
        sort_entries(value)
        with self.assertRaisesRegex(
            landing.ManifestError, "historical receipt-core V1 cannot be normative"
        ):
            landing.validate_manifest(value, verify_git=False)

    def test_legacy_status_requires_migration_reference(self):
        for status in ("HistoricalOnly", "CompatibilityAdapter"):
            with self.subTest(status=status):
                value = manifest(self.commit, self.blob)
                value["entries"].append(
                    entry(
                        "legacy",
                        self.commit,
                        self.blob,
                        status=status,
                        domain="symthaea.legacy.v1",
                    )
                )
                sort_entries(value)
                with self.assertRaisesRegex(
                    landing.ManifestError, "requires migration_or_equivalence_ref"
                ):
                    landing.validate_manifest(value, verify_git=False)

    def test_dependency_cycle_is_rejected(self):
        value = manifest(self.commit, self.blob)
        value["entries"][0]["required_entry_roles"] = ["test_framing"]
        value["entries"][1]["required_entry_roles"] = ["normative_framing"]
        with self.assertRaisesRegex(landing.ManifestError, "dependency cycle"):
            landing.validate_manifest(value, verify_git=False)

    def test_normative_candidate_requires_golden_vectors(self):
        value = manifest(self.commit, self.blob)
        value["entries"][0]["golden_vector_refs"] = []
        with self.assertRaisesRegex(landing.ManifestError, "requires golden vectors"):
            landing.validate_manifest(value, verify_git=False)

    def test_undeclared_external_primitive_is_rejected(self):
        value = manifest(self.commit, self.blob)
        value["entries"][0]["required_external_primitives"] = ["execution-lineage:unknown"]
        with self.assertRaisesRegex(landing.ManifestError, "undeclared external primitive"):
            landing.validate_manifest(value, verify_git=False)

    def test_generic_entry_cannot_depend_on_domain_owned_role(self):
        value = manifest(self.commit, self.blob)
        value["entries"].append(
            entry(
                "scientific_interpretation",
                self.commit,
                self.blob,
                status="DomainOwned",
                domain="symthaea.phys.claim-interpretation.v1",
            )
        )
        value["entries"][0]["required_entry_roles"] = ["scientific_interpretation"]
        sort_entries(value)
        with self.assertRaisesRegex(landing.ManifestError, "may not depend on DomainOwned"):
            landing.validate_manifest(value, verify_git=False)

    def test_unrepresented_local_import_is_rejected(self):
        repo = Path(self.tmp.name) / "imports"
        repo.mkdir()
        git(repo, "init", "-q")
        git(repo, "config", "user.name", "QUAL test")
        git(repo, "config", "user.email", "qual@example.invalid")
        (repo / "scripts").mkdir()
        (repo / "scripts" / "main.py").write_text("import hidden_dependency\n", encoding="utf-8")
        (repo / "scripts" / "hidden_dependency.py").write_text("VALUE = 1\n", encoding="utf-8")
        (repo / "vector.txt").write_text("vector\n", encoding="utf-8")
        git(repo, "add", ".")
        git(repo, "commit", "-q", "-m", "imports")
        commit = git(repo, "rev-parse", "HEAD")
        main_blob = git(repo, "rev-parse", f"{commit}:scripts/main.py")
        vector_blob = git(repo, "rev-parse", f"{commit}:vector.txt")

        value = {
            "schema": landing.SCHEMA,
            "qualification_era_target": landing.ERA,
            "normative_framing_role": "normative_framing",
            "external_primitives": ["execution-lineage:test"],
            "entries": [
                entry(
                    "normative_framing",
                    commit,
                    main_blob,
                    status="NormativeCandidate",
                    domain="symthaea.qualification-framing.v1",
                    goldens=["test_framing"],
                    path="scripts/main.py",
                ),
                entry(
                    "test_framing",
                    commit,
                    vector_blob,
                    status="TestVector",
                    domain="qualification-framing-test-vector",
                    path="vector.txt",
                ),
            ],
        }

        with self.assertRaisesRegex(landing.ManifestError, "unrepresented local import"):
            landing.validate_manifest(value, repo=repo, verify_git=True)

    def test_declared_import_dependency_must_match_source(self):
        repo = Path(self.tmp.name) / "declared-imports"
        repo.mkdir()
        git(repo, "init", "-q")
        git(repo, "config", "user.name", "QUAL test")
        git(repo, "config", "user.email", "qual@example.invalid")
        (repo / "scripts").mkdir()
        (repo / "scripts" / "main.py").write_text("import dependency\n", encoding="utf-8")
        (repo / "scripts" / "dependency.py").write_text("VALUE = 1\n", encoding="utf-8")
        (repo / "vector.txt").write_text("vector\n", encoding="utf-8")
        git(repo, "add", ".")
        git(repo, "commit", "-q", "-m", "imports")
        commit = git(repo, "rev-parse", "HEAD")
        main_blob = git(repo, "rev-parse", f"{commit}:scripts/main.py")
        dep_blob = git(repo, "rev-parse", f"{commit}:scripts/dependency.py")
        vector_blob = git(repo, "rev-parse", f"{commit}:vector.txt")

        value = {
            "schema": landing.SCHEMA,
            "qualification_era_target": landing.ERA,
            "normative_framing_role": "normative_framing",
            "external_primitives": ["execution-lineage:test"],
            "entries": [
                entry(
                    "dependency",
                    commit,
                    dep_blob,
                    status="CompatibilityAdapter",
                    domain="historical-module:dependency",
                    migration="issue:#3770",
                    path="scripts/dependency.py",
                ),
                entry(
                    "normative_framing",
                    commit,
                    main_blob,
                    status="NormativeCandidate",
                    domain="symthaea.qualification-framing.v1",
                    goldens=["test_framing"],
                    path="scripts/main.py",
                ),
                entry(
                    "test_framing",
                    commit,
                    vector_blob,
                    status="TestVector",
                    domain="qualification-framing-test-vector",
                    path="vector.txt",
                ),
            ],
        }

        with self.assertRaisesRegex(landing.ManifestError, "required_entry_roles must equal"):
            landing.validate_manifest(value, repo=repo, verify_git=True)

        value["entries"][1]["required_entry_roles"] = ["dependency"]
        normalized = landing.validate_manifest(value, repo=repo, verify_git=True)
        self.assertEqual(normalized["entries"][1]["required_entry_roles"], ["dependency"])

    def test_dependency_blob_version_skew_is_rejected(self):
        repo = Path(self.tmp.name) / "version-skew"
        repo.mkdir()
        git(repo, "init", "-q")
        git(repo, "config", "user.name", "QUAL test")
        git(repo, "config", "user.email", "qual@example.invalid")
        (repo / "scripts").mkdir()
        (repo / "scripts" / "main.py").write_text("import dependency\n", encoding="utf-8")
        (repo / "scripts" / "dependency.py").write_text("VALUE = 1\n", encoding="utf-8")
        (repo / "vector.txt").write_text("vector\n", encoding="utf-8")
        git(repo, "add", ".")
        git(repo, "commit", "-q", "-m", "importer snapshot")
        importer_commit = git(repo, "rev-parse", "HEAD")
        main_blob = git(repo, "rev-parse", f"{importer_commit}:scripts/main.py")
        vector_blob = git(repo, "rev-parse", f"{importer_commit}:vector.txt")

        (repo / "scripts" / "dependency.py").write_text("VALUE = 2\n", encoding="utf-8")
        git(repo, "add", "scripts/dependency.py")
        git(repo, "commit", "-q", "-m", "later dependency")
        later_commit = git(repo, "rev-parse", "HEAD")
        later_dep_blob = git(repo, "rev-parse", f"{later_commit}:scripts/dependency.py")

        value = {
            "schema": landing.SCHEMA,
            "qualification_era_target": landing.ERA,
            "normative_framing_role": "normative_framing",
            "external_primitives": ["execution-lineage:test"],
            "entries": [
                entry(
                    "dependency",
                    later_commit,
                    later_dep_blob,
                    status="CompatibilityAdapter",
                    domain="historical-module:dependency",
                    migration="issue:#3770",
                    path="scripts/dependency.py",
                ),
                entry(
                    "normative_framing",
                    importer_commit,
                    main_blob,
                    status="NormativeCandidate",
                    domain="symthaea.qualification-framing.v1",
                    deps=["dependency"],
                    goldens=["test_framing"],
                    path="scripts/main.py",
                ),
                entry(
                    "test_framing",
                    importer_commit,
                    vector_blob,
                    status="TestVector",
                    domain="qualification-framing-test-vector",
                    path="vector.txt",
                ),
            ],
        }

        with self.assertRaisesRegex(landing.ManifestError, "dependency blob version skew"):
            landing.validate_manifest(value, repo=repo, verify_git=True)

    def test_canonical_bytes_are_stable(self):
        value = manifest(self.commit, self.blob)
        left = landing.validate_manifest(copy.deepcopy(value), verify_git=False)
        right = landing.validate_manifest(copy.deepcopy(value), verify_git=False)
        self.assertEqual(landing.canonical_bytes(left), landing.canonical_bytes(right))


if __name__ == "__main__":
    unittest.main()
