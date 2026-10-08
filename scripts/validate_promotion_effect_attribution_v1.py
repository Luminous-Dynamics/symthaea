#!/usr/bin/env python3
"""Independent oracle for provider effect observation versus causal attribution.

claim_ceiling=deterministic semantic classification only
promotion_authority=false
"""

from dataclasses import dataclass
from enum import Enum


class Attribution(Enum):
    ESTABLISHED = "established"
    UNESTABLISHED = "unestablished"


class Source(Enum):
    DIRECT_PROVIDER_RESULT = "direct_provider_result"
    DURABLE_SUBJECT_OBSERVATION = "durable_subject_observation"


@dataclass(frozen=True)
class EffectObservation:
    source: Source
    attribution: Attribution
    local_promotion_operation_id: str | None
    provider_operation_uuid: str | None
    expected_pr_head_sha: str
    observed_merge_commit: str

    def validate(self) -> None:
        assert self.expected_pr_head_sha
        assert self.observed_merge_commit

        if self.source is Source.DIRECT_PROVIDER_RESULT:
            assert self.attribution is Attribution.ESTABLISHED
            assert self.local_promotion_operation_id
            assert self.provider_operation_uuid
        else:
            assert self.attribution is Attribution.UNESTABLISHED
            assert self.provider_operation_uuid is None


def direct_provider_merge() -> EffectObservation:
    result = EffectObservation(
        Source.DIRECT_PROVIDER_RESULT,
        Attribution.ESTABLISHED,
        "OP-1",
        "uuid-1",
        "H1",
        "M1",
    )
    result.validate()
    return result


def enqueued_then_durable_merge() -> EffectObservation:
    result = EffectObservation(
        Source.DURABLE_SUBJECT_OBSERVATION,
        Attribution.UNESTABLISHED,
        None,
        None,
        "H1",
        "M1",
    )
    result.validate()
    return result


def expired_uuid_then_durable_merge() -> EffectObservation:
    return enqueued_then_durable_merge()


def already_merged_retry() -> EffectObservation:
    return enqueued_then_durable_merge()


def another_actor_merge() -> EffectObservation:
    result = EffectObservation(
        Source.DURABLE_SUBJECT_OBSERVATION,
        Attribution.UNESTABLISHED,
        None,
        None,
        "H1",
        "M2",
    )
    result.validate()
    return result


def test_direct_provider_merge_is_causal() -> None:
    result = direct_provider_merge()
    assert result.attribution is Attribution.ESTABLISHED
    assert result.provider_operation_uuid == "uuid-1"


def test_enqueued_merge_is_effect_only() -> None:
    result = enqueued_then_durable_merge()
    assert result.attribution is Attribution.UNESTABLISHED
    assert result.provider_operation_uuid is None


def test_expired_uuid_does_not_backfill_causality() -> None:
    result = expired_uuid_then_durable_merge()
    assert result.attribution is Attribution.UNESTABLISHED


def test_already_merged_retry_is_not_causal() -> None:
    result = already_merged_retry()
    assert result.attribution is Attribution.UNESTABLISHED


def test_other_actor_merge_is_not_causal() -> None:
    result = another_actor_merge()
    assert result.attribution is Attribution.UNESTABLISHED


def test_direct_result_without_local_binding_is_invalid() -> None:
    result = EffectObservation(
        Source.DIRECT_PROVIDER_RESULT,
        Attribution.ESTABLISHED,
        None,
        "uuid-1",
        "H1",
        "M1",
    )
    try:
        result.validate()
    except AssertionError:
        return
    raise AssertionError("direct provider result without local operation binding was accepted")


def test_direct_result_without_uuid_is_invalid() -> None:
    result = EffectObservation(
        Source.DIRECT_PROVIDER_RESULT,
        Attribution.ESTABLISHED,
        "OP-1",
        None,
        "H1",
        "M1",
    )
    try:
        result.validate()
    except AssertionError:
        return
    raise AssertionError("direct provider result without UUID was accepted")


TESTS = [
    test_direct_provider_merge_is_causal,
    test_enqueued_merge_is_effect_only,
    test_expired_uuid_does_not_backfill_causality,
    test_already_merged_retry_is_not_causal,
    test_other_actor_merge_is_not_causal,
    test_direct_result_without_local_binding_is_invalid,
    test_direct_result_without_uuid_is_invalid,
]


if __name__ == "__main__":
    for test in TESTS:
        test()
        print("PASS", test.__name__)
    print("PromotionEffectAttributionV1 model: PASS")
    print("claim_ceiling=deterministic semantic classification only")
    print("promotion_authority=false")
