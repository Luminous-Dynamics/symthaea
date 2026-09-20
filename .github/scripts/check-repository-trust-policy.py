#!/usr/bin/env python3
"""Verify Symthaea's minimum repository-root trust contract.

The verifier binds three things without conflating them:

1. the declared local trust policy;
2. the importable GitHub ruleset recipe intended to realize that policy;
3. GitHub's effective active rules for `main`.

A PASS here proves only the machine-readable minimum that the public effective
rules endpoint exposes. Bypass identities and destructive negative tests remain
out-of-band evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

POLICY_SCHEMA = "symthaea-repository-trust-policy-v1"
EXPECTED_REPOSITORY = "Luminous-Dynamics/symthaea"
EXPECTED_BRANCH = "main"
EXPECTED_RECIPE = ".github/rulesets/main-trusted-root-v1.json"
EXPECTED_RECIPE_NAME = "Symthaea main trusted root v1"
REQUIRED_RULE_TYPES = {"pull_request", "deletion", "non_fast_forward"}
EXPECTED_MERGE_METHODS = ["merge", "squash", "rebase"]

EXIT_INVALID = 2
EXIT_POLICY_FAIL = 3


class PolicyInputError(ValueError):
    pass


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyInputError(f"{path}: {exc}") from exc


def validate_policy(policy: Any) -> dict[str, Any]:
    if not isinstance(policy, dict):
        raise PolicyInputError("policy root must be an object")
    if policy.get("schema") != POLICY_SCHEMA:
        raise PolicyInputError(
            f"policy schema must be {POLICY_SCHEMA!r}, "
            f"observed={policy.get('schema')!r}"
        )
    if policy.get("repository") != EXPECTED_REPOSITORY:
        raise PolicyInputError("policy repository identity drifted")
    if policy.get("branch") != EXPECTED_BRANCH:
        raise PolicyInputError("policy branch identity drifted")
    if policy.get("ruleset_recipe") != EXPECTED_RECIPE:
        raise PolicyInputError("policy ruleset recipe identity drifted")

    required = policy.get("minimum_active_rule_types")
    if not isinstance(required, list):
        raise PolicyInputError("minimum_active_rule_types must be a list")
    if set(required) != REQUIRED_RULE_TYPES or len(required) != len(REQUIRED_RULE_TYPES):
        raise PolicyInputError(
            "minimum_active_rule_types must be exactly "
            "pull_request,deletion,non_fast_forward"
        )

    solo = policy.get("solo_maintainer_core")
    if not isinstance(solo, dict):
        raise PolicyInputError("solo_maintainer_core must be an object")
    for key in (
        "require_pull_request",
        "block_branch_deletion",
        "block_force_push",
    ):
        if solo.get(key) is not True:
            raise PolicyInputError(f"solo_maintainer_core.{key} must remain true")
    if solo.get("required_approving_review_count_min") != 0:
        raise PolicyInputError(
            "v1 solo-maintainer core must keep approval minimum at zero"
        )
    if solo.get("require_code_owner_review") is not False:
        raise PolicyInputError(
            "v1 must not require self-impossible CODEOWNER approval"
        )

    independent = policy.get("independent_review_tier")
    if not isinstance(independent, dict):
        raise PolicyInputError("independent_review_tier must be an object")
    if (
        independent.get("state")
        != "deferred-until-second-trusted-maintainer-or-team"
    ):
        raise PolicyInputError("independent review tier state drifted")
    if independent.get("future_require_code_owner_review") is not True:
        raise PolicyInputError("future CODEOWNER-review intent drifted")

    out_of_band = policy.get("out_of_band_requirements")
    if not isinstance(out_of_band, list) or len(out_of_band) < 4:
        raise PolicyInputError(
            "out_of_band_requirements must preserve the four v1 checks"
        )

    return policy


def validate_rules_shape(rules: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(rules, list):
        raise PolicyInputError(f"{label} must be a JSON array")

    validated: list[dict[str, Any]] = []
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise PolicyInputError(
                f"{label} rule at index {index} is not an object"
            )
        rule_type = rule.get("type")
        if not isinstance(rule_type, str) or not rule_type:
            raise PolicyInputError(
                f"{label} rule at index {index} has no valid type"
            )
        validated.append(rule)
    return validated


def evaluate_rules(
    policy: dict[str, Any],
    rules: list[dict[str, Any]],
) -> tuple[list[str], list[str], int, bool]:
    observed_types = sorted({str(rule["type"]) for rule in rules})
    required = list(policy["minimum_active_rule_types"])
    missing = sorted(set(required) - set(observed_types))

    max_required_approvals = 0
    code_owner_review_active = False

    for rule in rules:
        if rule["type"] != "pull_request":
            continue

        parameters = rule.get("parameters")
        if parameters is None:
            continue
        if not isinstance(parameters, dict):
            raise PolicyInputError(
                "pull_request rule parameters must be an object when present"
            )

        approvals = parameters.get("required_approving_review_count", 0)
        if not isinstance(approvals, int) or isinstance(approvals, bool) or approvals < 0:
            raise PolicyInputError(
                "required_approving_review_count must be a non-negative integer"
            )
        max_required_approvals = max(max_required_approvals, approvals)

        code_owner = parameters.get("require_code_owner_review", False)
        if not isinstance(code_owner, bool):
            raise PolicyInputError("require_code_owner_review must be boolean")
        code_owner_review_active = code_owner_review_active or code_owner

    return (
        observed_types,
        missing,
        max_required_approvals,
        code_owner_review_active,
    )


def require_bool(parameters: dict[str, Any], key: str, expected: bool) -> None:
    observed = parameters.get(key)
    if observed is not expected:
        raise PolicyInputError(
            f"ruleset recipe pull_request.{key} must be {expected!r}, "
            f"observed={observed!r}"
        )


def validate_recipe(
    policy: dict[str, Any],
    recipe: Any,
) -> dict[str, Any]:
    if not isinstance(recipe, dict):
        raise PolicyInputError("ruleset recipe root must be an object")

    if recipe.get("name") != EXPECTED_RECIPE_NAME:
        raise PolicyInputError("ruleset recipe name drifted")
    if recipe.get("target") != "branch":
        raise PolicyInputError("ruleset recipe target must remain 'branch'")
    if recipe.get("enforcement") != "active":
        raise PolicyInputError("ruleset recipe enforcement must remain 'active'")
    if "bypass_actors" in recipe:
        raise PolicyInputError(
            "ruleset recipe must not encode bypass_actors; bypass is reviewed "
            "out-of-band because GitHub import/export omits that authority"
        )

    conditions = recipe.get("conditions")
    if not isinstance(conditions, dict):
        raise PolicyInputError("ruleset recipe conditions must be an object")
    ref_name = conditions.get("ref_name")
    if not isinstance(ref_name, dict):
        raise PolicyInputError("ruleset recipe ref_name condition missing")
    if ref_name.get("include") != ["refs/heads/main"]:
        raise PolicyInputError(
            "ruleset recipe must target exactly refs/heads/main"
        )
    if ref_name.get("exclude") != []:
        raise PolicyInputError("ruleset recipe ref exclusions must remain empty")

    rules = validate_rules_shape(recipe.get("rules"), "ruleset recipe rules")
    observed, missing, approvals, code_owner = evaluate_rules(policy, rules)

    if missing:
        raise PolicyInputError(
            "ruleset recipe is missing minimum rule types: " + ",".join(missing)
        )
    if set(observed) != REQUIRED_RULE_TYPES or len(rules) != len(REQUIRED_RULE_TYPES):
        raise PolicyInputError(
            "ruleset recipe must contain exactly one each of "
            "pull_request,deletion,non_fast_forward"
        )
    if approvals != 0 or code_owner:
        raise PolicyInputError(
            "ruleset recipe would activate independent review in solo-maintainer v1"
        )

    pull_rules = [rule for rule in rules if rule["type"] == "pull_request"]
    if len(pull_rules) != 1:
        raise PolicyInputError(
            "ruleset recipe must contain exactly one pull_request rule"
        )

    parameters = pull_rules[0].get("parameters")
    if not isinstance(parameters, dict):
        raise PolicyInputError(
            "ruleset recipe pull_request parameters must be an object"
        )

    methods = parameters.get("allowed_merge_methods")
    if methods != EXPECTED_MERGE_METHODS:
        raise PolicyInputError(
            "ruleset recipe allowed_merge_methods must remain "
            f"{EXPECTED_MERGE_METHODS!r}, observed={methods!r}"
        )

    require_bool(parameters, "dismiss_stale_reviews_on_push", False)
    require_bool(parameters, "require_code_owner_review", False)
    require_bool(parameters, "require_last_push_approval", False)
    require_bool(parameters, "required_review_thread_resolution", False)

    approvals = parameters.get("required_approving_review_count")
    if approvals != 0 or isinstance(approvals, bool):
        raise PolicyInputError(
            "ruleset recipe required_approving_review_count must remain 0"
        )

    return recipe


def verify_effective_rules(
    policy: dict[str, Any],
    rules: list[dict[str, Any]],
) -> int:
    observed, missing, approvals, code_owner_active = evaluate_rules(policy, rules)

    if missing:
        print("repository_trust_minimum_rules=FAIL")
        print(f"missing_rule_types={','.join(missing)}")
        print("repository_trust_overall=NOT_ESTABLISHED")
        return EXIT_POLICY_FAIL

    if approvals > 0 or code_owner_active:
        print("repository_trust_solo_compatibility=FAIL")
        print(f"required_approving_review_count_observed={approvals}")
        print(
            "require_code_owner_review_observed="
            f"{str(code_owner_active).lower()}"
        )
        print(
            "reason=independent review became active while v1 tier is deferred"
        )
        print("repository_trust_overall=NOT_ESTABLISHED")
        return EXIT_POLICY_FAIL

    print("repository_trust_readback=PASS")
    print("repository_trust_minimum_rules=PASS")
    print("repository_trust_solo_compatibility=PASS")
    print("ruleset_recipe_coherence=PASS")
    print(
        "required_rule_types="
        + ",".join(policy["minimum_active_rule_types"])
    )
    print("observed_rule_types=" + ",".join(observed))
    print("independent_review_tier=DEFERRED")
    print("bypass_identity_verification=OUT_OF_BAND_REQUIRED")
    print("negative_mutation_tests=OUT_OF_BAND_REQUIRED")
    print(
        "repository_trust_overall="
        "PARTIAL_OUT_OF_BAND_EVIDENCE_REQUIRED"
    )
    return 0


def make_self_test_policy() -> dict[str, Any]:
    return {
        "schema": POLICY_SCHEMA,
        "repository": EXPECTED_REPOSITORY,
        "branch": EXPECTED_BRANCH,
        "ruleset_recipe": EXPECTED_RECIPE,
        "minimum_active_rule_types": [
            "pull_request",
            "deletion",
            "non_fast_forward",
        ],
        "solo_maintainer_core": {
            "require_pull_request": True,
            "block_branch_deletion": True,
            "block_force_push": True,
            "required_approving_review_count_min": 0,
            "require_code_owner_review": False,
        },
        "independent_review_tier": {
            "state": "deferred-until-second-trusted-maintainer-or-team",
            "future_require_code_owner_review": True,
        },
        "out_of_band_requirements": ["a", "b", "c", "d"],
    }


def make_self_test_recipe() -> dict[str, Any]:
    return {
        "name": EXPECTED_RECIPE_NAME,
        "target": "branch",
        "enforcement": "active",
        "conditions": {
            "ref_name": {
                "include": ["refs/heads/main"],
                "exclude": [],
            }
        },
        "rules": [
            {"type": "deletion"},
            {"type": "non_fast_forward"},
            {
                "type": "pull_request",
                "parameters": {
                    "allowed_merge_methods": EXPECTED_MERGE_METHODS,
                    "dismiss_stale_reviews_on_push": False,
                    "require_code_owner_review": False,
                    "require_last_push_approval": False,
                    "required_approving_review_count": 0,
                    "required_review_thread_resolution": False,
                },
            },
        ],
    }


def self_test() -> int:
    policy = validate_policy(make_self_test_policy())
    validate_recipe(policy, make_self_test_recipe())

    safe = validate_rules_shape(
        [
            {
                "type": "pull_request",
                "parameters": {
                    "required_approving_review_count": 0,
                    "require_code_owner_review": False,
                },
            },
            {"type": "deletion"},
            {"type": "non_fast_forward"},
        ],
        "self-test effective rules",
    )
    observed, missing, approvals, code_owner = evaluate_rules(policy, safe)
    assert set(observed) == REQUIRED_RULE_TYPES
    assert missing == []
    assert approvals == 0
    assert code_owner is False

    missing_force_push = validate_rules_shape(
        [{"type": "pull_request"}, {"type": "deletion"}],
        "self-test missing-force-push rules",
    )
    assert evaluate_rules(policy, missing_force_push)[1] == [
        "non_fast_forward"
    ]

    self_deadlock = validate_rules_shape(
        [
            {
                "type": "pull_request",
                "parameters": {
                    "required_approving_review_count": 1,
                    "require_code_owner_review": True,
                },
            },
            {"type": "deletion"},
            {"type": "non_fast_forward"},
        ],
        "self-test self-deadlock rules",
    )
    _, missing, approvals, code_owner = evaluate_rules(policy, self_deadlock)
    assert missing == []
    assert approvals == 1
    assert code_owner is True

    bad_recipe = make_self_test_recipe()
    bad_recipe["rules"] = [
        rule
        for rule in bad_recipe["rules"]
        if rule["type"] != "non_fast_forward"
    ]
    try:
        validate_recipe(policy, bad_recipe)
    except PolicyInputError:
        pass
    else:
        raise AssertionError("recipe missing non_fast_forward was accepted")

    print("repository_trust_self_test=PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--rules", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    if args.policy is None or args.recipe is None or args.rules is None:
        parser.error(
            "--policy, --recipe, and --rules are required unless --self-test "
            "is used"
        )

    try:
        policy = validate_policy(load_json(args.policy))
        validate_recipe(policy, load_json(args.recipe))
        rules = validate_rules_shape(
            load_json(args.rules),
            "effective rules response",
        )
    except PolicyInputError as exc:
        print("repository_trust_readback=INVALID")
        print(f"reason={exc}")
        return EXIT_INVALID

    return verify_effective_rules(policy, rules)


if __name__ == "__main__":
    sys.exit(main())
