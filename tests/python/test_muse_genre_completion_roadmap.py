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
 def test_focus_parser_fails_closed_on_missing_entry(self):
  s=(ROOT/"crates/domains/symthaea-muse-protocol/src/catalog.rs").read_text(encoding="utf-8"); self.assertEqual(len(parse_grammar_foci(s,160)),160); s=s.replace('        "South African bubblegum pop song",\n',"",1)
  with self.assertRaisesRegex(CoverageError,"cardinality mismatch|does not match"): parse_grammar_foci(s,160)
 def test_checked_in_manifest_is_source_bound(self):
  r=build_roadmap(ROOT); self.assertEqual(json.loads((ROOT/ROADMAP_RELATIVE).read_text(encoding="utf-8")),r)
if __name__=="__main__": unittest.main()
