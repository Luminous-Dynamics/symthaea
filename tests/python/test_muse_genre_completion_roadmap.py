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
 def test_focus_parser_fails_closed_on_missing_entry(self):
  s=(ROOT/"crates/domains/symthaea-muse-protocol/src/catalog.rs").read_text(encoding="utf-8"); self.assertEqual(len(parse_grammar_foci(s,160)),160); s=s.replace('        "South African bubblegum pop song",\n',"",1)
  with self.assertRaisesRegex(CoverageError,"cardinality mismatch|does not match"): parse_grammar_foci(s,160)
 def test_checked_in_manifest_is_source_bound(self):
  r=build_roadmap(ROOT); self.assertEqual(json.loads((ROOT/ROADMAP_RELATIVE).read_text(encoding="utf-8")),r)
if __name__=="__main__": unittest.main()
