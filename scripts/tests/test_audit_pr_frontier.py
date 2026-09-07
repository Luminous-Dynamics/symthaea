import unittest

from scripts.audit_pr_frontier import (
    Pull,
    PullAudit,
    annotate_lineages,
    infer_parent_relations,
    is_workflow_patch_capsule,
    looks_self_declared_qualification_only,
)


def pull(
    number: int,
    *,
    base_ref: str,
    base_sha: str,
    head_ref: str,
    head_sha: str,
    title: str = "",
    body: str = "",
    draft: bool = True,
) -> Pull:
    return Pull(
        number=number,
        title=title,
        draft=draft,
        base_ref=base_ref,
        base_sha=base_sha,
        head_ref=head_ref,
        head_sha=head_sha,
        html_url=f"https://example.invalid/{number}",
        body=body,
        created_at="",
        updated_at="",
    )


class ParentRelationTests(unittest.TestCase):
    def test_exact_parent_requires_branch_and_sha_equality(self) -> None:
        pulls = [
            pull(
                1,
                base_ref="main",
                base_sha="m1",
                head_ref="feature/a",
                head_sha="a1",
            ),
            pull(
                2,
                base_ref="feature/a",
                base_sha="a1",
                head_ref="feature/b",
                head_sha="b1",
            ),
        ]
        relations = infer_parent_relations(pulls, "main")
        self.assertEqual(relations[1].kind, "default_branch_root")
        self.assertEqual(relations[2].kind, "exact_open_parent")
        self.assertEqual(relations[2].parent_number, 1)

    def test_matching_base_branch_with_sha_drift_is_not_exact(self) -> None:
        pulls = [
            pull(
                1,
                base_ref="main",
                base_sha="m1",
                head_ref="feature/a",
                head_sha="a2",
            ),
            pull(
                2,
                base_ref="feature/a",
                base_sha="a1",
                head_ref="feature/b",
                head_sha="b1",
            ),
        ]
        relation = infer_parent_relations(pulls, "main")[2]
        self.assertEqual(relation.kind, "base_ref_sha_drift")
        self.assertEqual(relation.parent_number, 1)
        self.assertIn("a1", relation.reason or "")
        self.assertIn("a2", relation.reason or "")

    def test_ambiguous_head_branch_is_not_ancestry_proof(self) -> None:
        pulls = [
            pull(
                1,
                base_ref="main",
                base_sha="m1",
                head_ref="feature/shared",
                head_sha="a1",
            ),
            pull(
                2,
                base_ref="main",
                base_sha="m1",
                head_ref="feature/shared",
                head_sha="a2",
            ),
            pull(
                3,
                base_ref="feature/shared",
                base_sha="a2",
                head_ref="feature/child",
                head_sha="c1",
            ),
        ]
        relation = infer_parent_relations(pulls, "main")[3]
        self.assertEqual(relation.kind, "ambiguous_base")
        self.assertIsNone(relation.parent_number)


class LineageTests(unittest.TestCase):
    def test_multi_level_depth_and_root(self) -> None:
        pulls = [
            pull(
                10,
                base_ref="main",
                base_sha="m1",
                head_ref="a",
                head_sha="a1",
            ),
            pull(
                11,
                base_ref="a",
                base_sha="a1",
                head_ref="b",
                head_sha="b1",
            ),
            pull(
                12,
                base_ref="b",
                base_sha="b1",
                head_ref="c",
                head_sha="c1",
            ),
        ]
        relations = infer_parent_relations(pulls, "main")
        audits = {
            item.number: PullAudit(item, relations[item.number]) for item in pulls
        }
        annotate_lineages(audits)
        self.assertEqual(audits[10].depth, 0)
        self.assertEqual(audits[11].depth, 1)
        self.assertEqual(audits[12].depth, 2)
        self.assertEqual(audits[12].root_number, 10)


class QualificationCapsuleTests(unittest.TestCase):
    def test_strict_workflow_patch_capsule(self) -> None:
        self.assertTrue(
            is_workflow_patch_capsule(
                [
                    ".github/workflows/fep-learning-mode-exclusivity-v1.yml",
                    "docs/release/evidence/fep-learning-mode-exclusivity-v1.patch",
                ]
            )
        )

    def test_product_source_prevents_capsule_classification(self) -> None:
        self.assertFalse(
            is_workflow_patch_capsule(
                [
                    ".github/workflows/x.yml",
                    "docs/release/evidence/x.patch",
                    "crates/core/example/src/lib.rs",
                ]
            )
        )

    def test_workflow_without_patch_is_not_capsule(self) -> None:
        self.assertFalse(is_workflow_patch_capsule([".github/workflows/x.yml"]))

    def test_textual_claim_is_only_a_heuristic(self) -> None:
        candidate = pull(
            20,
            base_ref="main",
            base_sha="m1",
            head_ref="qualification/x",
            head_sha="x1",
            body="Qualification-only companion; no product source is materialized.",
        )
        self.assertTrue(looks_self_declared_qualification_only(candidate))
        self.assertFalse(is_workflow_patch_capsule(["crates/core/x/src/lib.rs"]))


if __name__ == "__main__":
    unittest.main()
