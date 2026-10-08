#!/usr/bin/env python3
"""RH-006 set-origin/representation semantic micro-gate."""

from dataclasses import dataclass


class Origin:
    OBSERVATIONAL_EQUIVALENCE = "observational-equivalence-derived"
    MAINTAINED_RESTRICTION = "maintained-restriction-region"
    SENSITIVITY = "sensitivity-analysis-region"
    UNKNOWN = "unknown"


class Representation:
    WITNESS = "witness-only"
    INNER = "inner-approximation"
    OUTER = "outer-approximation"
    SHARP = "sharp"


@dataclass(frozen=True)
class Certificate:
    representation: str
    origin: str

    def permits_identified_set_claim(self) -> bool:
        return (
            self.origin == Origin.OBSERVATIONAL_EQUIVALENCE
            and self.representation != Representation.WITNESS
        )

    def permits_stable_identified_decision(self) -> bool:
        return self.permits_identified_set_claim() and self.representation in {
            Representation.OUTER,
            Representation.SHARP,
        }

    def permits_unstable_identified_decision(self) -> bool:
        return self.permits_identified_set_claim() and self.representation in {
            Representation.INNER,
            Representation.SHARP,
        }


def main() -> None:
    sensitivity = Certificate(Representation.SHARP, Origin.SENSITIVITY)
    observed_outer = Certificate(
        Representation.OUTER, Origin.OBSERVATIONAL_EQUIVALENCE
    )
    observed_inner = Certificate(
        Representation.INNER, Origin.OBSERVATIONAL_EQUIVALENCE
    )
    witness = Certificate(Representation.WITNESS, Origin.OBSERVATIONAL_EQUIVALENCE)

    assert not sensitivity.permits_identified_set_claim()
    assert not sensitivity.permits_stable_identified_decision()
    assert not sensitivity.permits_unstable_identified_decision()

    assert observed_outer.permits_identified_set_claim()
    assert observed_outer.permits_stable_identified_decision()
    assert not observed_outer.permits_unstable_identified_decision()

    assert observed_inner.permits_identified_set_claim()
    assert not observed_inner.permits_stable_identified_decision()
    assert observed_inner.permits_unstable_identified_decision()

    assert not witness.permits_identified_set_claim()
    assert not witness.permits_stable_identified_decision()
    assert not witness.permits_unstable_identified_decision()

    print(
        {
            "schema": "rh006-set-provenance-microtest/v1",
            "sensitivity_sharp_is_identified": False,
            "observed_outer_stable": True,
            "observed_inner_unstable": True,
            "witness_neither": True,
            "status": "research-diagnostic-only",
        }
    )


if __name__ == "__main__":
    main()
