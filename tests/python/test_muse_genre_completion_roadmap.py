"""Regression tests for complete Muse genre roadmap coverage."""
from __future__ import annotations
import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"scripts"))
from muse_genre_completion_roadmap import ROADMAP_RELATIVE,WORKSTREAMS,build_roadmap,parse_grammar_foci
from muse_style_coverage import CoverageError
class GenreCompletionRoadmapTests(unittest.TestCase):
 def test_every_catalog_id_has_a_completion_track(self):
  r=build_roadmap(ROOT); e=r["entries"]; self.assertEqual(len(e),160); self.assertEqual([x["id"] for x in e],list(range(1,161))); self.assertEqual(len({x["name"] for x in e}),160); self.assertTrue(all(x["candidate_engine_track"] and x["minimum_evidence_gates"] for x in e)); self.assertEqual(len(r["workstreams"]),20); self.assertEqual({x["constellation"] for x in r["workstreams"]},set(WORKSTREAMS))
 def test_southern_african_styles_are_expert_gated_and_unrouted(self):
  r=build_roadmap(ROOT); rows=[e for e in r["entries"] if e["constellation"]=="SouthernAfricanUrbanFolkDance"]; self.assertEqual(len(rows),8); self.assertTrue(all(e["catalog_status"]=="ResearchExpertReviewRequired" and e["requires_expert_review"] and e["current_composer_route"] is None and e["route_state"]=="research_only_unrouted" for e in rows)); self.assertTrue(all(any("review" in g.lower() for g in e["minimum_evidence_gates"]) for e in rows))
 def test_future_candidates_stay_outside_catalog(self):
  r=build_roadmap(ROOT); q=r["expansion_candidates"]; names={e["name"].casefold() for e in r["entries"]}; self.assertGreaterEqual(len(q),20); self.assertTrue(all(e["catalog_id"] is None and e["state"]=="proposal_only_not_in_catalog" and e["name"].casefold() not in names for e in q))
 def test_global_expansion_queue_is_source_linked_and_stays_proposal_only(self):
  r=build_roadmap(ROOT); q=r["expansion_candidates"]; names={e["name"].casefold() for e in r["entries"]}
  self.assertEqual(len(q),60); self.assertEqual(len({e["name"].casefold() for e in q}),len(q))
  expected={"Benga","Chimurenga","Mbalax","Calypso","Cumbia","Vallenato","Bachata","Raï","Dangdut","Luk Thung","Mor Lam","K-pop","J-pop","Bongo Flava","Azonto","Banku Music"}
  self.assertTrue(expected <= {e["name"] for e in q})
  self.assertTrue(all(e["sources"] and all(s.startswith("https://") for s in e["sources"]) for e in q))
  self.assertTrue(all(e["catalog_id"] is None and e["state"]=="proposal_only_not_in_catalog" and e["name"].casefold() not in names and e["implementation_claim"] for e in q))
 def test_every_expansion_candidate_has_an_explicit_source_assessment(self):
  r=build_roadmap(ROOT); q=r["expansion_candidates"]
  self.assertEqual(len(q),60)
  allowed={"tradition_specific_context_present","broad_scholarly_context_only","journalism_lead_only","curated_reference_or_repository_only","discovery_only"}
  self.assertEqual({e["source_evidence_state"] for e in q},set(r["summary"]["candidate_source_evidence_states"]))
  self.assertTrue(all(e["source_assessments"] and e["source_evidence_state"] in allowed for e in q))
  self.assertTrue(all(all(x["url"].startswith("https://") and x["source_kind"] and x["scope"] and x["supports"] and x["does_not_establish"] for x in e["source_assessments"]) for e in q))
  self.assertTrue(all("does not prove" in e["source_evidence_limit"] or "none of these labels proves" in e["source_evidence_limit"] for e in q))
  directory_only=[e for e in q if e["source_evidence_state"]=="discovery_only"]
  self.assertTrue(directory_only); self.assertTrue(all("directory-only" in e["next_gate"] for e in directory_only))
  self.assertTrue(all(e["catalog_id"] is None and e["state"]=="proposal_only_not_in_catalog" for e in q))
 def test_choro_pilot_is_bounded_falsifiable_and_license_gated(self):
  path=ROOT/"crates/domains/symthaea-muse/CHORO_GRAMMAR_RESEARCH_SPEC_V1.json"
  spec=json.loads(path.read_text(encoding="utf-8"))
  self.assertEqual(spec["schema"],"muse-genre-specific-research-spec-v1")
  self.assertEqual(spec["identity"]["name"],"Choro")
  self.assertIsNone(spec["identity"]["catalog_id"]); self.assertIsNone(spec["identity"]["composer_route"])
  self.assertFalse(spec["decision_boundary"]["implementation_started"])
  self.assertFalse(spec["decision_boundary"]["runtime_capability_claimed"])
  self.assertFalse(spec["decision_boundary"]["source_data_copied_into_symthaea"])
  license=spec["license_and_data_handling"]
  self.assertEqual(license["repository_license"],"CC BY-NC-SA 4.0")
  self.assertFalse(license["commercial_use_authorized_by_this_spec"])
  self.assertTrue(any("permission" in x.lower() or "legal review" in x.lower() for x in license["before_empirical_analysis"]))
  hs=spec["hypotheses"]; self.assertGreaterEqual(len(hs),4); self.assertEqual(len({h["id"] for h in hs}),len(hs))
  self.assertTrue(all(h["state"] in {"proposed_falsifiable_hypothesis_not_tested","research_question_not_testable_from_chord_form_corpus_alone"} for h in hs))
  self.assertTrue(all(h.get("evidence_basis") or h.get("required_additional_evidence") for h in hs))
  self.assertTrue(all(h.get("operationalization") and h.get("falsification_or_no_pass") and h.get("null_model") is not None for h in hs if h["state"]=="proposed_falsifiable_hypothesis_not_tested"))
  self.assertTrue(any("held-out-by-composition" in x for x in spec["route_promotion_gates"]))
  self.assertTrue(any("knowledgeable Choro" in x for x in spec["route_promotion_gates"]))
 def test_choro_source_intake_is_metadata_only_and_per_work_rights_gated(self):
  path=ROOT/"crates/domains/symthaea-muse/CHORO_SOURCE_INTAKE_V1.json"
  intake=json.loads(path.read_text(encoding="utf-8"))
  self.assertEqual(intake["schema"],"muse-choro-source-intake-v1")
  summary=intake["summary"]; self.assertEqual(summary["source_records"],4)
  self.assertEqual(summary["records_rights_cleared"],0); self.assertEqual(summary["records_pending_rights_review"],4)
  self.assertEqual(summary["files_downloaded"],0); self.assertEqual(summary["score_content_transcribed"],0); self.assertEqual(summary["empirical_analysis_performed"],0)
  self.assertEqual(len(intake["sources"]),4)
  self.assertTrue(all(s["status"]=="candidate_not_rights_cleared" and s["commercial_use_status"]=="not_approved" and s["source_url"].startswith("https://") and s["unresolved"] for s in intake["sources"]))
  self.assertTrue(any("Samba Rio-Grandense" in s["title"] and "boundary" in s["scope"] for s in intake["sources"]))
  self.assertTrue(all(any("rights" in gate.lower() or "license" in gate.lower() for gate in intake["global_gates"]) for _ in [0]))
  bulk=intake["restricted_reference_corpus"]
  self.assertEqual(bulk["zenodo_record_rights_display"],"Other (Open) — no further description.")
  self.assertEqual(bulk["repository_license_stated"],"CC BY-NC-SA 4.0")
  self.assertTrue("non-commercial" in bulk["published_paper_data_availability"].lower())
  self.assertEqual(bulk["status"],"non_commercial_reference_not_approved_for_commercial_analysis")
  spec=json.loads((ROOT/"crates/domains/symthaea-muse/CHORO_GRAMMAR_RESEARCH_SPEC_V1.json").read_text(encoding="utf-8"))
  self.assertEqual(spec["alternative_source_intake"]["records_rights_cleared"],0)
  self.assertEqual(spec["license_and_data_handling"]["zenodo_record_rights_display"],"Other (Open) — no further description.")
  self.assertEqual(spec["alternative_source_intake"]["path"],"crates/domains/symthaea-muse/CHORO_SOURCE_INTAKE_V1.json")
 def test_focus_parser_fails_closed_on_missing_entry(self):
  s=(ROOT/"crates/domains/symthaea-muse-protocol/src/catalog.rs").read_text(encoding="utf-8"); self.assertEqual(len(parse_grammar_foci(s,160)),160); s=s.replace('        "South African bubblegum pop song",\n',"",1)
  with self.assertRaisesRegex(CoverageError,"cardinality mismatch|does not match"): parse_grammar_foci(s,160)
 def test_checked_in_manifest_is_source_bound(self):
  r=build_roadmap(ROOT); self.assertEqual(json.loads((ROOT/ROADMAP_RELATIVE).read_text(encoding="utf-8")),r)
if __name__=="__main__": unittest.main()
