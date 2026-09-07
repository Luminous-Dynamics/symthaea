import unittest

from scripts.ci.qualification_scope import QualificationMode, resolve_scope


class QualificationScopeTests(unittest.TestCase):
    def test_draft_pull_request_is_the_only_iteration_path(self) -> None:
        for action in ("opened", "synchronize", "reopened", "converted_to_draft"):
            with self.subTest(action=action):
                decision = resolve_scope(
                    event_name="pull_request",
                    ref="refs/pull/123/merge",
                    payload={"action": action, "pull_request": {"draft": True}},
                )
                self.assertEqual(decision.mode, QualificationMode.ITERATION)
                self.assertFalse(decision.run_full)
                self.assertFalse(decision.fail_safe)

    def test_ready_pull_request_runs_full(self) -> None:
        decision = resolve_scope(
            event_name="pull_request",
            ref="refs/pull/123/merge",
            payload={"action": "ready_for_review", "pull_request": {"draft": False}},
        )
        self.assertEqual(decision.mode, QualificationMode.FULL_PREMERGE)
        self.assertTrue(decision.run_full)
        self.assertFalse(decision.fail_safe)

    def test_nondraft_synchronize_runs_full_again(self) -> None:
        decision = resolve_scope(
            event_name="pull_request",
            ref="refs/pull/123/merge",
            payload={"action": "synchronize", "pull_request": {"draft": False}},
        )
        self.assertEqual(decision.mode, QualificationMode.FULL_PREMERGE)
        self.assertTrue(decision.run_full)

    def test_missing_draft_field_fails_to_full(self) -> None:
        decision = resolve_scope(
            event_name="pull_request",
            ref="refs/pull/123/merge",
            payload={"action": "synchronize", "pull_request": {}},
        )
        self.assertEqual(decision.mode, QualificationMode.FULL_PREMERGE)
        self.assertTrue(decision.run_full)
        self.assertTrue(decision.fail_safe)

    def test_wrong_draft_type_fails_to_full(self) -> None:
        decision = resolve_scope(
            event_name="pull_request",
            ref="refs/pull/123/merge",
            payload={"action": "synchronize", "pull_request": {"draft": "true"}},
        )
        self.assertEqual(decision.mode, QualificationMode.FULL_PREMERGE)
        self.assertTrue(decision.run_full)
        self.assertTrue(decision.fail_safe)

    def test_manual_dispatch_is_always_full(self) -> None:
        decision = resolve_scope(
            event_name="workflow_dispatch",
            ref="refs/heads/feature/x",
            payload={},
        )
        self.assertEqual(decision.mode, QualificationMode.MANUAL_FULL)
        self.assertTrue(decision.run_full)

    def test_schedule_preserves_full_regression_surface(self) -> None:
        decision = resolve_scope(
            event_name="schedule",
            ref="refs/heads/main",
            payload={"schedule": "0 4 * * 0"},
        )
        self.assertEqual(decision.mode, QualificationMode.SCHEDULED)
        self.assertTrue(decision.run_full)

    def test_main_push_is_full(self) -> None:
        decision = resolve_scope(
            event_name="push",
            ref="refs/heads/main",
            payload={},
        )
        self.assertEqual(decision.mode, QualificationMode.MAIN)
        self.assertTrue(decision.run_full)

    def test_unexpected_push_ref_fails_to_full(self) -> None:
        decision = resolve_scope(
            event_name="push",
            ref="refs/heads/feature/unexpected",
            payload={},
        )
        self.assertEqual(decision.mode, QualificationMode.FULL_PREMERGE)
        self.assertTrue(decision.run_full)
        self.assertTrue(decision.fail_safe)

    def test_unknown_event_fails_to_full(self) -> None:
        decision = resolve_scope(
            event_name="repository_dispatch",
            ref="refs/heads/main",
            payload={},
        )
        self.assertEqual(decision.mode, QualificationMode.FULL_PREMERGE)
        self.assertTrue(decision.run_full)
        self.assertTrue(decision.fail_safe)

    def test_github_outputs_are_stable_lowercase_booleans(self) -> None:
        decision = resolve_scope(
            event_name="pull_request",
            ref="refs/pull/1/merge",
            payload={"pull_request": {"draft": True}},
        )
        output = decision.github_outputs().splitlines()
        self.assertIn("mode=iteration", output)
        self.assertIn("run_full=false", output)
        self.assertIn("fail_safe=false", output)


if __name__ == "__main__":
    unittest.main()
