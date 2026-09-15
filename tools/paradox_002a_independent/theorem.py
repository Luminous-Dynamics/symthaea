"""Independent PARADOX-002A manipulation theorem.

This module derives semantics from parsed fixture bytes. It does not import,
execute, or translate the Rust qualifier.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Iterable

from format import (
    EvidenceEvent, F64, Fixture, MATCHED_BUDGET, OracleReport, Pair,
)

# Frozen wire values.
WORLD_EVIDENCE = 1
SELF_PREDICTION = 2
CONTEXT_SIGNAL = 3
PROPOSITION = 1
NEGATION = 2

COMMIT = 1
COMMIT_CONDITIONALLY = 2
ABSTAIN_PRESERVE_PLURALITY = 3
REFLEXIVE_UPDATE = 4
REVISE_CONTEXT = 5

CURRENT_REPRESENTATION = 1
EXPLICIT_CONTEXT = 2
IRREDUCIBLE = 3
REFLEXIVE = 4
REQUIRES_REPRESENTATION_REVISION = 5

KIND_PREDICTION_ERROR = 0
KIND_EVIDENCE_CONTRADICTION = 3
KIND_SELF_REFERENTIAL = 5
KIND_ONTOLOGY_FAILURE = 6

RESOLUTION_STABLE = 0
RESOLUTION_PERSISTENT_UNRESOLVED = 2
RESOLUTION_RESOLVED_WITHOUT_REVISION = 3
RESOLUTION_IRREDUCIBLE = 5

EVIDENCE_NEITHER = 0
EVIDENCE_PROP = 1
EVIDENCE_NEG = 2
EVIDENCE_BOTH = 3


class TheoremError(ValueError):
    pass


@dataclass
class Snapshot:
    proposition: bool
    negation: bool
    has_both: bool
    independent_fault_domains: bool

    @property
    def dominant(self) -> int | None:
        if self.proposition and not self.negation:
            return PROPOSITION
        if self.negation and not self.proposition:
            return NEGATION
        return None


@dataclass
class Derived:
    external_surprise: F64
    internal_disagreement: bool
    final_unresolved_disagreement: bool
    conflict_persistence: F64
    independent_conflict_sources: bool
    self_referential: bool
    explicit_context_resolution: bool
    ontology_failure: bool
    irreducible: bool
    conflict_slots: int
    expected_response: int
    resolution_class: int
    target_polarity: int | None
    kind: int
    support_proposition: F64
    support_negation: F64
    resolution_state: int
    uncertainty: F64
    evidence_polarity: int
    integration_coherence: F64
    conflict_load: F64
    candidate_recruitment_index: F64


def _require(value: bool, message: str) -> None:
    if not value:
        raise TheoremError(message)


def _validate_fixture_structure(fixture: Fixture) -> None:
    agent = fixture.agent_view
    _require(agent.budget == MATCHED_BUDGET, "matched resource budget violated")
    _require(len(agent.events) == 4, "fixture must contain exactly four events")
    _require(sum(event.polarity is not None for event in agent.events) == 3, "fixture must contain exactly three claim events")
    _require(sum(event.polarity is None for event in agent.events) == 1, "fixture must contain exactly one auxiliary event")

    for index, event in enumerate(agent.events):
        _require(event.slot == index, "event slots must be contiguous and ordered")
        value = event.reliability.value
        _require(math.isfinite(value) and 0.0 <= value <= 1.0, "event reliability must be finite and in [0,1]")
        if event.role == CONTEXT_SIGNAL:
            _require(event.polarity is None, "context signal must not carry claim polarity")
            _require(event.supersedes_slot is None, "context signal must not supersede a claim")
            _require(not event.caused_by_self_prediction, "context signal must not claim causal self-reference")
        else:
            _require(event.polarity is not None, "claim role must carry claim polarity")

    sources = {event.source_id for event in agent.events}
    _require(len(sources) == 2, "fixture must contain exactly two source identities")

    if fixture.condition == 6:
        _require(all(event.visible_context is None for event in agent.events), "C6 must not disclose visible context")


def _snapshot(events: list[EvidenceEvent], active: list[bool]) -> Snapshot:
    prop_domains: list[int] = []
    neg_domains: list[int] = []
    for enabled, event in zip(active, events):
        if not enabled:
            continue
        if event.polarity == PROPOSITION:
            prop_domains.append(event.fault_domain_id)
        elif event.polarity == NEGATION:
            neg_domains.append(event.fault_domain_id)
    proposition = bool(prop_domains)
    negation = bool(neg_domains)
    both = proposition and negation
    independent = (not both) or any(p != n for p in prop_domains for n in neg_domains)
    return Snapshot(proposition, negation, both, independent)


def _first_world_polarity(events: list[EvidenceEvent]) -> int | None:
    for event in events:
        if event.role == WORLD_EVIDENCE and event.polarity is not None:
            return event.polarity
    return None


def _polarity_in_context(events: list[EvidenceEvent], active: list[bool], belongs: Iterable[bool]) -> int | None:
    prop = False
    neg = False
    for enabled, in_context, event in zip(active, belongs, events):
        if not enabled or not in_context:
            continue
        prop |= event.polarity == PROPOSITION
        neg |= event.polarity == NEGATION
    if prop and not neg:
        return PROPOSITION
    if neg and not prop:
        return NEGATION
    return None


def _visible_context_resolution(fixture: Fixture, active: list[bool]) -> int | None:
    target = None
    for event in fixture.agent_view.events:
        if event.role == CONTEXT_SIGNAL:
            target = event.visible_context
            break
    if target is None:
        return None
    belongs = [event.visible_context == target for event in fixture.agent_view.events]
    return _polarity_in_context(fixture.agent_view.events, active, belongs)


def _hidden_context_resolution(fixture: Fixture, active: list[bool]) -> int | None:
    target = fixture.truth.target_context
    if target is None:
        return None
    belongs = [context == target for context in fixture.truth.hidden_claim_contexts]
    return _polarity_in_context(fixture.agent_view.events, active, belongs)


def _evidence_polarity(prop: bool, neg: bool) -> int:
    if prop and neg:
        return EVIDENCE_BOTH
    if prop:
        return EVIDENCE_PROP
    if neg:
        return EVIDENCE_NEG
    return EVIDENCE_NEITHER


def derive(fixture: Fixture) -> Derived:
    _validate_fixture_structure(fixture)
    events = fixture.agent_view.events
    active = [False] * 4
    conflict_slots = 0
    independent_conflict_sources = True

    for index, event in enumerate(events):
        if event.supersedes_slot is not None:
            _require(event.supersedes_slot < index, "supersession must reference an earlier slot")
            active[event.supersedes_slot] = False
        if event.polarity is not None:
            active[index] = True
        snapshot = _snapshot(events, active)
        if snapshot.has_both:
            conflict_slots += 1
            independent_conflict_sources = independent_conflict_sources and snapshot.independent_fault_domains

    final = _snapshot(events, active)
    internal = conflict_slots > 0
    final_unresolved = final.has_both
    persistence = conflict_slots / len(events)
    first_world = _first_world_polarity(events)
    surprise = 0.0 if first_world is None or first_world == fixture.agent_view.prior_expectation else 1.0

    self_ref = (
        any(event.role == SELF_PREDICTION and event.polarity is not None for event in events)
        and any(event.role == WORLD_EVIDENCE and event.caused_by_self_prediction for event in events)
    )
    visible_target = _visible_context_resolution(fixture, active)
    hidden_target = _hidden_context_resolution(fixture, active)
    explicit = visible_target is not None
    ontology = (
        final_unresolved
        and fixture.truth.latent_context_is_causal
        and not fixture.truth.context_dimension_available
        and hidden_target is not None
    )
    irreducible = final_unresolved and not explicit and not ontology and not self_ref

    if self_ref:
        expected = REFLEXIVE_UPDATE
        resolution_class = REFLEXIVE
        target = final.dominant
    elif ontology:
        expected = REVISE_CONTEXT
        resolution_class = REQUIRES_REPRESENTATION_REVISION
        target = hidden_target
    elif explicit:
        expected = COMMIT_CONDITIONALLY
        resolution_class = EXPLICIT_CONTEXT
        target = visible_target
    elif irreducible:
        expected = ABSTAIN_PRESERVE_PLURALITY
        resolution_class = IRREDUCIBLE
        target = None
    else:
        expected = COMMIT
        resolution_class = CURRENT_REPRESENTATION
        target = final.dominant

    uncertainty_value = {
        CURRENT_REPRESENTATION: 0.1,
        EXPLICIT_CONTEXT: 0.25,
        IRREDUCIBLE: 1.0,
        REFLEXIVE: 0.75,
        REQUIRES_REPRESENTATION_REVISION: 0.8,
    }[resolution_class]

    if fixture.condition in (0, 1):
        kind = KIND_PREDICTION_ERROR
    elif fixture.condition in (2, 3, 4):
        kind = KIND_EVIDENCE_CONTRADICTION
    elif fixture.condition == 5:
        kind = KIND_SELF_REFERENTIAL
    else:
        kind = KIND_ONTOLOGY_FAILURE

    if resolution_class == CURRENT_REPRESENTATION:
        resolution_state = RESOLUTION_RESOLVED_WITHOUT_REVISION if internal else RESOLUTION_STABLE
    elif resolution_class == EXPLICIT_CONTEXT:
        resolution_state = RESOLUTION_RESOLVED_WITHOUT_REVISION
    elif resolution_class == IRREDUCIBLE:
        resolution_state = RESOLUTION_IRREDUCIBLE
    else:
        resolution_state = RESOLUTION_PERSISTENT_UNRESOLVED

    internal_float = 1.0 if internal else 0.0
    self_float = 1.0 if self_ref else 0.0
    support_prop = 1.0 if final.proposition else 0.0
    support_neg = 1.0 if final.negation else 0.0
    coherence = 1.0 - internal_float
    conflict_load = internal_float * (0.5 + 0.5 * persistence)
    candidate = (internal_float + uncertainty_value + persistence + self_float) / 4.0

    result = Derived(
        F64.from_float(surprise), internal, final_unresolved, F64.from_float(persistence),
        independent_conflict_sources, self_ref, explicit, ontology, irreducible,
        conflict_slots, expected, resolution_class, target, kind,
        F64.from_float(support_prop), F64.from_float(support_neg), resolution_state,
        F64.from_float(uncertainty_value), _evidence_polarity(final.proposition, final.negation),
        F64.from_float(coherence), F64.from_float(conflict_load), F64.from_float(candidate),
    )
    _validate_condition(fixture.condition, result)
    return result


def _exact(value: F64, target: float) -> bool:
    return value.bits == F64.from_float(target).bits


def _validate_condition(condition: int, d: Derived) -> None:
    if d.internal_disagreement:
        _require(d.independent_conflict_sources, "conflicting evidence must span independent fault domains")
    if condition == 0:
        _require(_exact(d.external_surprise, 0.0) and not d.internal_disagreement and not d.final_unresolved_disagreement and d.expected_response == COMMIT, "C0 contract violated")
    elif condition == 1:
        _require(_exact(d.external_surprise, 1.0) and not d.internal_disagreement and not d.final_unresolved_disagreement and d.expected_response == COMMIT, "C1 contract violated")
    elif condition == 2:
        _require(_exact(d.external_surprise, 1.0) and d.internal_disagreement and not d.final_unresolved_disagreement and 0.0 < d.conflict_persistence.value < 0.5 and d.expected_response == COMMIT, "C2 contract violated")
    elif condition == 3:
        _require(_exact(d.external_surprise, 1.0) and d.internal_disagreement and d.final_unresolved_disagreement and d.conflict_persistence.value >= 0.5 and d.explicit_context_resolution and d.expected_response == COMMIT_CONDITIONALLY and d.target_polarity is not None, "C3 contract violated")
    elif condition == 4:
        _require(_exact(d.external_surprise, 1.0) and d.internal_disagreement and d.final_unresolved_disagreement and d.conflict_persistence.value >= 0.5 and d.irreducible and d.expected_response == ABSTAIN_PRESERVE_PLURALITY, "C4 contract violated")
    elif condition == 5:
        _require(_exact(d.external_surprise, 1.0) and d.internal_disagreement and d.final_unresolved_disagreement and d.self_referential and d.expected_response == REFLEXIVE_UPDATE, "C5 contract violated")
    elif condition == 6:
        _require(_exact(d.external_surprise, 1.0) and d.internal_disagreement and d.final_unresolved_disagreement and not d.explicit_context_resolution and d.ontology_failure and d.expected_response == REVISE_CONTEXT and d.target_polarity is not None, "C6 contract violated")
    else:
        raise TheoremError(f"unknown condition {condition}")


def _eq_bits(name: str, derived: F64, actual: F64) -> None:
    if derived.bits != actual.bits:
        raise TheoremError(f"report mismatch for {name}: {derived.bits:#018x} != {actual.bits:#018x}")


def _eq(name: str, derived: object, actual: object) -> None:
    if derived != actual:
        raise TheoremError(f"report mismatch for {name}: {derived!r} != {actual!r}")


def compare_report(d: Derived, report: OracleReport) -> None:
    _eq_bits("external_surprise", d.external_surprise, report.external_surprise)
    _eq("internal_disagreement", d.internal_disagreement, report.internal_disagreement)
    _eq("final_unresolved_disagreement", d.final_unresolved_disagreement, report.final_unresolved_disagreement)
    _eq_bits("conflict_persistence", d.conflict_persistence, report.conflict_persistence)
    _eq("independent_conflict_sources", d.independent_conflict_sources, report.independent_conflict_sources)
    _eq("self_referential", d.self_referential, report.self_referential)
    _eq("explicit_context_resolution", d.explicit_context_resolution, report.explicit_context_resolution)
    _eq("ontology_failure", d.ontology_failure, report.ontology_failure)
    _eq("irreducible", d.irreducible, report.irreducible)
    _eq("conflict_slots", d.conflict_slots, report.conflict_slots)
    _eq("expected_response", d.expected_response, report.expected_response)
    _eq("resolution_class", d.resolution_class, report.resolution_class)
    _eq("target_polarity", d.target_polarity, report.target_polarity)

    obs = report.observatory
    _eq("observatory.kind", d.kind, obs.kind)
    _eq_bits("observatory.support_proposition", d.support_proposition, obs.support_proposition)
    _eq_bits("observatory.support_negation", d.support_negation, obs.support_negation)
    _eq("observatory.resolution", d.resolution_state, obs.resolution)
    _eq_bits("observatory.external_surprise", d.external_surprise, obs.external_surprise)
    _eq_bits("observatory.internal_disagreement", F64.from_float(1.0 if d.internal_disagreement else 0.0), obs.internal_disagreement)
    _eq_bits("observatory.uncertainty", d.uncertainty, obs.uncertainty)
    _eq_bits("observatory.persistence", d.conflict_persistence, obs.persistence)
    _eq_bits("observatory.self_referential_relevance", F64.from_float(1.0 if d.self_referential else 0.0), obs.self_referential_relevance)
    _eq("observatory.evidence_polarity", d.evidence_polarity, obs.evidence_polarity)
    _eq_bits("observatory.integration_coherence", d.integration_coherence, obs.integration_coherence)
    _eq_bits("observatory.conflict_load", d.conflict_load, obs.conflict_load)
    _eq_bits("observatory.candidate_recruitment_index", d.candidate_recruitment_index, obs.candidate_recruitment_index)


def summary_record(fixture: Fixture, d: Derived) -> bytes:
    payload = {
        "condition": fixture.condition,
        "seed": fixture.seed,
        "trial": fixture.trial_index,
        "surprise_bits": f"{d.external_surprise.bits:016x}",
        "historical_conflict": d.internal_disagreement,
        "final_conflict": d.final_unresolved_disagreement,
        "persistence_bits": f"{d.conflict_persistence.bits:016x}",
        "independent": d.independent_conflict_sources,
        "self_ref": d.self_referential,
        "explicit_context": d.explicit_context_resolution,
        "ontology_failure": d.ontology_failure,
        "irreducible": d.irreducible,
        "conflict_slots": d.conflict_slots,
        "response": d.expected_response,
        "resolution_class": d.resolution_class,
        "target": d.target_polarity,
        "kind": d.kind,
        "support_p_bits": f"{d.support_proposition.bits:016x}",
        "support_n_bits": f"{d.support_negation.bits:016x}",
        "resolution_state": d.resolution_state,
        "uncertainty_bits": f"{d.uncertainty.bits:016x}",
        "evidence_polarity": d.evidence_polarity,
        "coherence_bits": f"{d.integration_coherence.bits:016x}",
        "conflict_load_bits": f"{d.conflict_load.bits:016x}",
        "candidate_bits": f"{d.candidate_recruitment_index.bits:016x}",
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"


def verify_pair(pair: Pair) -> tuple[Derived, bytes]:
    derived = derive(pair.fixture)
    compare_report(derived, pair.report)
    return derived, summary_record(pair.fixture, derived)
