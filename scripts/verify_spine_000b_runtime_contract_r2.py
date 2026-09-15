#!/usr/bin/env python3
"""Static consistency verifier for SPINE-000B runtime receipt contract R2.

Measurement-only. This does not execute cognition and does not qualify runtime
influence. It proves that the preregistered R2 contract names the current
production seams accurately before instrumentation is implemented.
"""

from pathlib import Path

SOURCE = Path("src/cognitive_loop/subsystem_trait.rs")
CONTRACT = Path("docs/research/SPINE_000B_RUNTIME_RECEIPT_CONTRACT_R2.md")


def fail(msg: str) -> None:
    raise SystemExit(f"SPINE-000B-R2 CONTRACT FAIL: {msg}")


def require(text: str, needle: str, where: str) -> None:
    if needle not in text:
        fail(f"missing {needle!r} in {where}")


def main() -> int:
    if not SOURCE.is_file():
        fail(f"missing production subject {SOURCE}")
    if not CONTRACT.is_file():
        fail(f"missing contract {CONTRACT}")

    source = SOURCE.read_text(encoding="utf-8")
    contract = CONTRACT.read_text(encoding="utf-8")

    # Production facts the R2 contract explicitly relies on.
    require(source, "if !output.is_neutral()", str(SOURCE))
    require(source, "self.outputs.push((name, output));", str(SOURCE))
    require(source, "if health.is_faulted(name)", str(SOURCE))
    require(source, "Some(SubsystemOutput::NEUTRAL)", str(SOURCE))
    require(source, "std::panic::catch_unwind", str(SOURCE))

    # Contract truth-table and applicability requirements.
    for needle in (
        "EXECUTED_NEUTRAL",
        "emitted = true",
        "admitted = false",
        "PANICKED_CAUGHT",
        "NotApplicable(NotAdmitted)",
        "RuntimeTelemetryEnvelope",
        "StateApplicationReceipt",
        "CycleIntegrationReceipt",
        "SubsystemExecutionReceipt",
        "state_changed",
        "duration_ns",
        "measurement-only",
        "LOAD_BEARING",
    ):
        require(contract, needle, str(CONTRACT))

    # Prevent the two important attribution mistakes from returning silently.
    require(
        contract,
        "`n_contributors` is canonical metadata and MUST be compared",
        str(CONTRACT),
    )
    require(
        contract,
        "it is **not** itself an influence channel",
        str(CONTRACT),
    )
    require(
        contract,
        "SPINE-000B MUST NOT conclude",
        str(CONTRACT),
    )
    require(
        contract,
        "subsystem S caused destination D to change",
        str(CONTRACT),
    )

    # Timing must be explicitly outside the canonical semantic replay surface.
    require(
        contract,
        "MUST NOT participate in canonical receipt hashes",
        str(CONTRACT),
    )

    print("SPINE-000B runtime contract R2 static consistency: PASS")
    print("authority=measurement-only")
    print("runtime_evidence_claimed=false")
    print("causal_load_claimed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
