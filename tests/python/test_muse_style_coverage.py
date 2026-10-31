"""Contract tests for the dependency-free Muse style coverage audit."""

from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from muse_style_coverage import (  # noqa: E402
    BASELINE_RELATIVE,
    CATALOG_RELATIVE,
    STYLE_RELATIVE,
    CoverageError,
    build_current_report,
    check_baseline,
    frozen_baseline,
    git_blob_sha,
    parse_catalog,
    parse_style_contract,
    print_summary,
)

CATALOG_FIXTURE = r'''
// This commented invocation is deliberately not a catalog row:
// style!(99, "Comment only", Z, "Default", Developing)
const EXAMPLE: &str = "style!(98, \"String only\", Z)";
macro_rules! style { ($id:literal) => { $id }; }
pub const CATALOG: [CanonicalStyle; 3] = [
    style!(1, "Default", A, "Default", Developing),
    style!(2, "Display Label", "Display Name", B, "Other", ExpertReviewRequired),
    style!(3, "Research Only", C),
];
'''

STYLE_FIXTURE = r'''
pub enum Style {
    /// Default route.
    Default,
    Other,
}
impl Style {
    pub const ALL: [Style; 2] = [
        Style::Default,
        Style::Other,
    ];
}
'''


class ParseCatalogTests(unittest.TestCase):
    def test_parses_supported_macro_shapes_and_ignores_comments_strings(self) -> None:
        research_review_fixture = CATALOG_FIXTURE.replace(
            'style!(3, "Research Only", C),',
            'style!(3, "Research Only", C, ResearchExpertReviewRequired),',
        )
        entries = parse_catalog(research_review_fixture)
        self.assertEqual([entry["id"] for entry in entries], [1, 2, 3])
        self.assertEqual(entries[1]["name"], "Display Label")
        self.assertEqual(entries[1]["display_name"], "Display Name")
        self.assertEqual(entries[1]["constellation"], "B")
        self.assertEqual(entries[1]["composer_style"], "Other")
        self.assertEqual(entries[1]["status"], "ExpertReviewRequired")
        self.assertTrue(entries[1]["requires_expert_review"])
        self.assertEqual(entries[2]["composer_style"], None)
        self.assertEqual(entries[2]["status"], "ResearchExpertReviewRequired")
        self.assertTrue(entries[2]["requires_expert_review"])

    def test_rejects_duplicate_ids(self) -> None:
        duplicate = CATALOG_FIXTURE.replace(
            'style!(3, "Research Only", C),',
            'style!(1, "Research Only", C),',
        )
        with self.assertRaisesRegex(CoverageError, "Duplicate style catalog IDs"):
            parse_catalog(duplicate)

    def test_rejects_gaps_in_stable_ids(self) -> None:
        gap = CATALOG_FIXTURE.replace(
            'style!(3, "Research Only", C),',
            'style!(4, "Research Only", C),',
        )
        with self.assertRaisesRegex(CoverageError, "contiguous"):
            parse_catalog(gap)

    def test_parses_enum_and_style_all_as_an_exact_ordered_contract(self) -> None:
        variants, all_names, declared_count = parse_style_contract(STYLE_FIXTURE)
        self.assertEqual(variants, ["Default", "Other"])
        self.assertEqual(all_names, variants)
        self.assertEqual(declared_count, 2)

    def test_rejects_mismatched_style_all(self) -> None:
        mismatch = STYLE_FIXTURE.replace(
            "Style::Other,",
            "Style::Default,",
            1,
        )
        with self.assertRaisesRegex(CoverageError, "duplicate variants|match enum variants"):
            parse_style_contract(mismatch)


class CoverageReportTests(unittest.TestCase):
    def make_root(self, root: Path) -> None:
        catalog_path = root / CATALOG_RELATIVE
        style_path = root / STYLE_RELATIVE
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        style_path.parent.mkdir(parents=True, exist_ok=True)
        catalog_path.write_text(CATALOG_FIXTURE, encoding="utf-8")
        style_path.write_text(STYLE_FIXTURE, encoding="utf-8")

    def test_research_expert_review_status_remains_unrouted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_root(root)
            catalog_path = root / CATALOG_RELATIVE
            source = catalog_path.read_text(encoding="utf-8").replace(
                'style!(3, "Research Only", C),',
                'style!(3, "Research Only", C, ResearchExpertReviewRequired),',
            )
            catalog_path.write_text(source, encoding="utf-8")
            report = build_current_report(root)
            self.assertEqual(report["summary"]["composer_routed_entries"], 2)
            self.assertEqual(report["summary"]["unrouted_catalog_entries"], 1)
            self.assertEqual(report["summary"]["status_counts"]["ResearchExpertReviewRequired"], 1)
            self.assertIsNone(report["entries"][2]["composer_style"])
            self.assertTrue(report["entries"][2]["requires_expert_review"])

    def test_report_counts_catalog_routes_and_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_root(root)
            report = build_current_report(root)
            self.assertEqual(report["summary"]["catalog_entries"], 3)
            self.assertEqual(report["summary"]["constellations"], 3)
            self.assertEqual(report["summary"]["composer_routed_entries"], 2)
            self.assertEqual(report["summary"]["unrouted_catalog_entries"], 1)
            self.assertEqual(
                report["summary"]["status_counts"],
                {"Developing": 1, "ExpertReviewRequired": 1, "Research": 1},
            )
            self.assertEqual(report["source"]["catalog_blob_sha"], git_blob_sha(
                (root / CATALOG_RELATIVE).read_bytes()
            ))

    def test_report_identifies_executable_styles_without_catalog_routes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_root(root)
            style_path = root / STYLE_RELATIVE
            source = STYLE_FIXTURE.replace(
                "    Other,\n}",
                "    Other,\n    NotCataloged,\n}",
            ).replace(
                "        Style::Other,\n",
                "        Style::Other,\n        Style::NotCataloged,\n",
            ).replace(
                "pub const ALL: [Style; 2]",
                "pub const ALL: [Style; 3]",
            )
            style_path.write_text(source, encoding="utf-8")

            report = build_current_report(root)
            self.assertEqual(
                report["summary"]["executable_styles_without_catalog_route"],
                1,
            )
            self.assertEqual(
                report["coverage_analysis"]["unreferenced_executable_styles"],
                ["NotCataloged"],
            )

            output = StringIO()
            with redirect_stdout(output):
                print_summary(report)
            self.assertIn(
                "Executable Style variants without catalog route: NotCataloged",
                output.getvalue(),
            )

    def test_report_exposes_zero_partial_and_full_constellation_routing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_root(root)
            catalog_path = root / CATALOG_RELATIVE
            source = catalog_path.read_text(encoding="utf-8").replace(
                'style!(3, "Research Only", C),',
                'style!(3, "Research Only", B),\n'
                '    style!(4, "Unrouted Family", D),',
            )
            catalog_path.write_text(source, encoding="utf-8")

            analysis = build_current_report(root)["coverage_analysis"]
            by_constellation = {
                item["constellation"]: item
                for item in analysis["by_constellation"]
            }
            self.assertEqual(analysis["zero_route_constellations"], ["D"])
            self.assertEqual(analysis["partial_route_constellations"], ["B"])
            self.assertEqual(analysis["fully_routed_constellations"], ["A"])
            self.assertEqual(by_constellation["B"]["catalog_entries"], 2)
            self.assertEqual(by_constellation["B"]["composer_routed_entries"], 1)
            self.assertEqual(by_constellation["B"]["unrouted_catalog_ids"], [3])
            self.assertEqual(by_constellation["D"]["routed_catalog_ids"], [])
            self.assertIn("not evidence", analysis["interpretation"])

    def test_summary_prints_actionable_routing_gaps_without_overclaiming(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_root(root)
            catalog_path = root / CATALOG_RELATIVE
            source = catalog_path.read_text(encoding="utf-8").replace(
                'style!(3, "Research Only", C),',
                'style!(3, "Research Only", B),\n'
                '    style!(4, "Unrouted Family", D),',
            )
            catalog_path.write_text(source, encoding="utf-8")
            report = build_current_report(root)

            output = StringIO()
            with redirect_stdout(output):
                print_summary(report)
            rendered = output.getvalue()
            self.assertIn("B: 1/2 routed; unrouted catalog IDs: 3", rendered)
            self.assertIn("Zero-route constellations: D", rendered)
            self.assertIn("Partial-route constellations: B", rendered)
            self.assertIn("no claim about musical distinctness", rendered)

    def test_report_fails_closed_on_duplicate_composer_routes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_root(root)
            catalog_path = root / CATALOG_RELATIVE
            source = catalog_path.read_text(encoding="utf-8").replace(
                '"Other", ExpertReviewRequired',
                '"Default", ExpertReviewRequired',
            )
            catalog_path.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(CoverageError, "same composer route"):
                build_current_report(root)

    def test_report_fails_closed_on_unknown_composer_route(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_root(root)
            catalog_path = root / CATALOG_RELATIVE
            source = catalog_path.read_text(encoding="utf-8").replace(
                '"Default", Developing',
                '"Missing", Developing',
            )
            catalog_path.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(CoverageError, "missing from Style::ALL"):
                build_current_report(root)

    def test_baseline_check_detects_source_drift_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_root(root)
            report = build_current_report(root)
            baseline_path = root / BASELINE_RELATIVE
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_path.write_text(
                json.dumps(frozen_baseline(report, "2026-10-12"), indent=2),
                encoding="utf-8",
            )
            self.assertEqual(check_baseline(root, baseline_path)["entries"], report["entries"])
            catalog_path = root / CATALOG_RELATIVE
            catalog_path.write_text(
                catalog_path.read_text(encoding="utf-8").replace(
                    '"Display Label", "Display Name"',
                    '"Display Label Revised", "Display Name"',
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CoverageError, "baseline is stale"):
                check_baseline(root, baseline_path)


if __name__ == "__main__":
    unittest.main()
