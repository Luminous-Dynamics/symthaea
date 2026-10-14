"""Integrity checks for the metadata-only Choro source intake."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INTAKE_PATH = ROOT / "crates/domains/symthaea-muse/CHORO_SOURCE_INTAKE_V1.json"
SPEC_PATH = ROOT / "crates/domains/symthaea-muse/CHORO_GRAMMAR_RESEARCH_SPEC_V1.json"


class ChoroSourceIntakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.intake = json.loads(INTAKE_PATH.read_text(encoding="utf-8"))
        cls.spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))

    def test_summary_counts_match_source_records(self) -> None:
        sources = self.intake["sources"]
        summary = self.intake["summary"]
        self.assertEqual(summary["source_records"], len(sources))
        self.assertEqual(
            summary["records_pending_rights_review"],
            sum(s["status"] == "candidate_not_rights_cleared" for s in sources),
        )
        self.assertEqual(summary["records_rights_cleared"], 0)

    def test_research_spec_intake_snapshot_tracks_inventory(self) -> None:
        summary = self.intake["summary"]
        declared = self.spec["alternative_source_intake"]
        self.assertEqual(declared["path"], "crates/domains/symthaea-muse/CHORO_SOURCE_INTAKE_V1.json")
        self.assertEqual(declared["schema"], self.intake["schema"])
        for key in (
            "source_records",
            "records_rights_cleared",
            "records_pending_rights_review",
            "files_downloaded",
            "score_content_transcribed",
            "empirical_analysis_performed",
            "status",
        ):
            with self.subTest(field=key):
                self.assertEqual(declared[key], summary[key])

    def test_source_leads_remain_unapproved_and_content_free(self) -> None:
        summary = self.intake["summary"]
        self.assertEqual(summary["files_downloaded"], 0)
        self.assertEqual(summary["score_content_transcribed"], 0)
        self.assertEqual(summary["empirical_analysis_performed"], 0)
        self.assertGreaterEqual(len(self.intake["sources"]), 6)
        for source in self.intake["sources"]:
            with self.subTest(source=source["id"]):
                self.assertEqual(source["status"], "candidate_not_rights_cleared")
                self.assertEqual(source["commercial_use_status"], "not_approved")
                self.assertTrue(source["unresolved"])

    def test_scope_and_provenance_are_recorded_for_every_lead(self) -> None:
        source_ids = [source["id"] for source in self.intake["sources"]]
        self.assertEqual(len(source_ids), len(set(source_ids)))
        for source in self.intake["sources"]:
            with self.subTest(source=source["id"]):
                self.assertTrue(source["source_url"].startswith("https://"))
                self.assertTrue(source["source_evidence_kind"])
                self.assertTrue(source["scope"])
                self.assertTrue(source["fit"])

    def test_candidate_pool_covers_multiple_composers(self) -> None:
        composers = {source["composer"] for source in self.intake["sources"]}
        self.assertGreaterEqual(len(composers), 5)

    def test_gallet_collection_is_not_misrepresented_as_ensemble_evidence(self) -> None:
        source = next(
            s for s in self.intake["sources"]
            if s["id"] == "imslp-12-exercicios-brasileiros-gallet-1928"
        )
        self.assertIn("piano four-hands", source["scope"])
        self.assertIn("not representative of ensemble Choro", source["fit"])
        self.assertEqual(source["commercial_use_status"], "not_approved")

    def test_macedo_guitar_score_retains_editor_and_sharealike_caveats(self) -> None:
        source = next(
            s for s in self.intake["sources"]
            if s["id"] == "imslp-choro-pra-portinari-macedo-1982"
        )
        self.assertIn("guitar solo", source["scope"])
        self.assertIn("ShareAlike", source["source_stated_rights"])
        self.assertEqual(source["commercial_use_status"], "not_approved")
        self.assertEqual(source["status"], "candidate_not_rights_cleared")

    def test_reference_corpus_rights_signals_stay_separate(self) -> None:
        corpus = self.intake["restricted_reference_corpus"]
        self.assertIn("Other (Open)", corpus["zenodo_record_rights_display"])
        self.assertEqual(corpus["repository_license_stated"], "CC BY-NC-SA 4.0")
        self.assertIn("non-commercial", corpus["published_paper_data_availability"])
        self.assertEqual(
            corpus["status"],
            "non_commercial_reference_not_approved_for_commercial_analysis",
        )


if __name__ == "__main__":
    unittest.main()
