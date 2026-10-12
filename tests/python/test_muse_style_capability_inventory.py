"""Tests for the source-declared Muse runtime capability inventory."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from muse_style_capability_inventory import _runtime_report  # noqa: E402
from muse_style_coverage import CATALOG_RELATIVE, CoverageError, STYLE_RELATIVE, git_blob_sha  # noqa: E402
from muse_style_capability_inventory import REALIZATION_RELATIVE  # noqa: E402
from muse_style_capability_inventory import GRAMMAR_RELATIVE  # noqa: E402
from muse_style_capability_inventory import CAUSAL_EVIDENCE_RELATIVE  # noqa: E402

CATALOG_FIXTURE = r'''
style!(1, "Default", A, "Default", Developing),
style!(2, "Other", B, "Other", Developing),
style!(3, "Research Only", C),
'''


STYLE_FIXTURE = r'''
pub enum Style {
    Default,
    Other,
}
impl Style {
    pub const ALL: [Style; 2] = [
        Style::Default,
        Style::Other,
    ];

    pub fn grammar_family(self) -> crate::grammar::GrammarFamily {
        use crate::grammar::GrammarFamily::*;
        match self {
            Style::Default => PeriodSentence,
            Style::Other => GrooveCycle,
        }
    }

    pub fn supported_intent_axes(self) -> &'static [&'static str] {
        const ALL: &[&str] = &["valence", "arousal", "energy", "bars"];
        const LIMITED: &[&str] = &["arousal", "bars"];
        const NONE_LIVE: &[&str] = &[];
        match self {
            Style::Default => ALL,
            Style::Other => LIMITED,
        }
    }

    pub fn meter(self) -> u8 {
        match self {
            Style::Other => 3,
            _ => 4,
        }
    }

    pub fn tempo_range(self) -> (f32, f32) {
        match self {
            Style::Default => (60.0, 150.0),
            Style::Other => (80.0, 130.0),
        }
    }

    pub fn spec(self) -> crate::spec::CompositionSpec {
        match self {
            Style::Default => CompositionSpec { name: "default".into() },
            Style::Other => CompositionSpec { name: "other".into() },
        }
    }
}
'''

CAUSAL_EVIDENCE_FIXTURE = r'''
fn causal_sensitivity_matrix() { /* fixture representing the census source */ }
'''


GRAMMAR_FIXTURE = r'''
pub fn compose_with_grammar_plan(profile: GrammarProfile, intent: &Intent, spec: &CompositionSpec) -> Realization {
    let realized = match profile.family {
        GrammarFamily::PeriodSentence => {
            let (score, form) = crate::composer::compose_with_spec_and_form_and_grammar(intent, spec, Some(profile));
            build_period(score, form)
        }
        GrammarFamily::GrooveCycle => {
            let realized = crate::groove_cycle::realize_groove_cycle(intent, spec);
            build_groove(realized)
        }
        family => {
            let (score, form) = crate::composer::compose_with_spec_and_form_and_grammar(intent, spec, Some(profile));
            build_compatibility(family, score, form)
        }
    };
    realized
}
'''


REALIZATION_FIXTURE = r'''
pub(crate) fn instruments_for(style: Style, seed: u64) -> Pool {
    // Comment containing fake token must not create a false mapping:
    // Style::Phantom => &[(Instrument::Fake, Instrument::Fake, Instrument::Fake)],
    let pool = match style {
        Style::Default => &[
            (Instrument::Piano,
             Instrument::Flute,
             Instrument::Cello,
            ),
        ],
        Style::Other => &[
            (Instrument::Sitar, Instrument::Pad, Instrument::Bass,),
            (Instrument::Sitar, Instrument::Harp, Instrument::Bass,),
        ],
    };
    pool[(seed % pool.len() as u64) as usize]
}
'''


class RuntimeInventoryTests(unittest.TestCase):
    def make_root(self, root: Path) -> None:
        catalog_path = root / CATALOG_RELATIVE
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        catalog_path.write_text(CATALOG_FIXTURE, encoding="utf-8")
        style_path = root / STYLE_RELATIVE
        realization_path = root / REALIZATION_RELATIVE
        style_path.parent.mkdir(parents=True, exist_ok=True)
        realization_path.parent.mkdir(parents=True, exist_ok=True)
        style_path.write_text(STYLE_FIXTURE, encoding="utf-8")
        realization_path.write_text(REALIZATION_FIXTURE, encoding="utf-8")
        grammar_path = root / GRAMMAR_RELATIVE
        grammar_path.parent.mkdir(parents=True, exist_ok=True)
        grammar_path.write_text(GRAMMAR_FIXTURE, encoding="utf-8")
        evidence_path = root / CAUSAL_EVIDENCE_RELATIVE
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(CAUSAL_EVIDENCE_FIXTURE, encoding="utf-8")

    def test_reports_source_declared_controls_and_instrument_pools(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_root(root)
            report = _runtime_report(root)
            profiles = {item["style"]: item for item in report["profiles"]}

            self.assertEqual(report["summary"]["catalog_entries"], 3)
            self.assertEqual(report["summary"]["catalog_entries_with_runtime_profile"], 2)
            self.assertEqual(report["summary"]["catalog_entries_without_runtime_profile"], 1)
            self.assertEqual(report["summary"]["executable_styles"], 2)
            self.assertEqual(report["summary"]["styles_with_grammar_family"], 2)
            self.assertEqual(report["summary"]["styles_with_composition_spec"], 2)
            self.assertEqual(report["summary"]["styles_with_instrument_pool"], 2)
            self.assertEqual(
                report["summary"]["intent_axis_count_distribution"],
                {"2": 1, "4": 1},
            )
            self.assertEqual(report["summary"]["styles_with_zero_intent_axes"], [])
            self.assertEqual(
                report["summary"]["styles_with_partial_intent_axes"],
                ["Other"],
            )
            self.assertEqual(profiles["Default"]["grammar_family"], "PeriodSentence")
            self.assertEqual(profiles["Default"]["meter_beats"], 4)
            self.assertEqual(profiles["Default"]["tempo_range_bpm"], [60.0, 150.0])
            self.assertEqual(
                profiles["Default"]["supported_intent_axes"],
                ["valence", "arousal", "energy", "bars"],
            )
            self.assertEqual(profiles["Other"]["meter_beats"], 3)
            self.assertEqual(profiles["Other"]["instrument_pool"]["ensemble_choices"], 2)
            self.assertEqual(
                profiles["Other"]["instrument_pool"]["instrument_names"],
                ["Bass", "Harp", "Pad", "Sitar"],
            )
            self.assertIn("does not execute the causal census", report["interpretation"])

            self.assertEqual(
                report["source"]["intent_axis_evidence_path"],
                CAUSAL_EVIDENCE_RELATIVE.as_posix(),
            )
            self.assertEqual(
                report["source"]["intent_axis_evidence_blob_sha"],
                git_blob_sha((root / CAUSAL_EVIDENCE_RELATIVE).read_bytes()),
            )
            self.assertTrue(report["intent_axis_evidence"]["required_marker_present"])
            self.assertFalse(report["intent_axis_evidence"]["census_executed_by_inventory"])

    def test_zero_intent_axis_styles_are_explicit_in_summary(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_root(root)
            style_path = root / STYLE_RELATIVE
            style_path.write_text(
                STYLE_FIXTURE.replace(
                    "Style::Other => LIMITED,",
                    "Style::Other => NONE_LIVE,",
                ),
                encoding="utf-8",
            )
            report = _runtime_report(root)

            self.assertEqual(
                report["summary"]["intent_axis_count_distribution"],
                {"0": 1, "4": 1},
            )
            self.assertEqual(report["summary"]["styles_with_zero_intent_axes"], ["Other"])
            other = next(item for item in report["profiles"] if item["style"] == "Other")
            self.assertEqual(other["supported_intent_axes"], [])
            self.assertEqual(other["intent_axis_count"], 0)

    def test_catalog_rows_join_to_runtime_profiles_by_composer_route(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_root(root)
            report = _runtime_report(root)
            rows = {row["id"]: row for row in report["catalog_entries"]}

            self.assertEqual(rows[1]["runtime_capability"]["style"], "Default")
            self.assertEqual(rows[2]["runtime_capability"]["style"], "Other")
            self.assertIsNone(rows[3]["composer_style"])
            self.assertIsNone(rows[3]["runtime_capability"])
            self.assertEqual(
                report["source"]["catalog_blob_sha"],
                git_blob_sha((root / CATALOG_RELATIVE).read_bytes()),
            )

    def test_reports_grammar_dispatch_modes_separately(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_root(root)
            report = _runtime_report(root)
            profiles = {item["style"]: item for item in report["profiles"]}

            self.assertEqual(
                profiles["Default"]["grammar_dispatch"]["mode"],
                "period_sentence_pipeline",
            )
            self.assertEqual(
                profiles["Other"]["grammar_dispatch"]["mode"],
                "dedicated_engine",
            )
            self.assertEqual(
                profiles["Other"]["grammar_dispatch"]["entrypoint"],
                "groove_cycle:crate::groove_cycle::realize_groove_cycle",
            )
            self.assertEqual(
                report["summary"]["grammar_dispatch_modes"],
                {"dedicated_engine": 1, "period_sentence_pipeline": 1},
            )

    def test_fails_closed_on_unrecognized_dedicated_grammar_engine(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_root(root)
            grammar_path = root / GRAMMAR_RELATIVE
            grammar_path.write_text(
                GRAMMAR_FIXTURE.replace(
                    "crate::groove_cycle::realize_groove_cycle",
                    "crate::new_engine::realize_new_engine",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CoverageError, "unrecognized dedicated realizer"):
                _runtime_report(root)

    def test_classifies_families_using_the_dispatch_fallback(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_root(root)
            style_path = root / STYLE_RELATIVE
            style_path.write_text(
                STYLE_FIXTURE.replace(
                    "Style::Other => GrooveCycle",
                    "Style::Other => Developmental",
                ),
                encoding="utf-8",
            )
            report = _runtime_report(root)
            profiles = {item["style"]: item for item in report["profiles"]}

            self.assertEqual(
                profiles["Other"]["grammar_dispatch"]["mode"],
                "compatibility_fallback",
            )
            self.assertEqual(
                report["summary"]["styles_using_compatibility_fallback"],
                ["Other"],
            )
            self.assertEqual(
                report["summary"]["grammar_dispatch_modes"],
                {"compatibility_fallback": 1, "period_sentence_pipeline": 1},
            )

    def test_fails_closed_when_grammar_dispatcher_has_no_compatibility_fallback(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_root(root)
            grammar_path = root / GRAMMAR_RELATIVE
            grammar_path.write_text(
                GRAMMAR_FIXTURE.replace(
                    "        family => {\n"
                    "            let (score, form) = crate::composer::compose_with_spec_and_form_and_grammar(intent, spec, Some(profile));\n"
                    "            build_compatibility(family, score, form)\n"
                    "        }\n",
                    "",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CoverageError, "no compatibility fallback arm"):
                _runtime_report(root)

    def test_fails_closed_when_census_evidence_marker_is_missing(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_root(root)
            evidence_path = root / CAUSAL_EVIDENCE_RELATIVE
            evidence_path.write_text(
                "fn unrelated_report() {}",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                CoverageError,
                "Causal intent-axis evidence source is missing",
            ):
                _runtime_report(root)

    def test_fails_closed_when_grammar_mapping_omits_executable_style(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_root(root)
            style_path = root / STYLE_RELATIVE
            style_path.write_text(
                STYLE_FIXTURE.replace(
                    "            Style::Other => GrooveCycle,",
                    "",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CoverageError, "grammar_family does not cover"):
                _runtime_report(root)

    def test_fails_closed_when_instrument_pool_omits_executable_style(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_root(root)
            realization_path = root / REALIZATION_RELATIVE
            realization_path.write_text(
                REALIZATION_FIXTURE.replace(
                    "        Style::Other => &[\n"
                    "            (Instrument::Sitar, Instrument::Pad, Instrument::Bass,),\n"
                    "            (Instrument::Sitar, Instrument::Harp, Instrument::Bass,),\n"
                    "        ],",
                    "",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CoverageError, "instruments_for does not cover"):
                _runtime_report(root)


if __name__ == "__main__":
    unittest.main()
