"""Restart-safe privacy budget and immutable-output contract tests."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from tools.energy_telemetry.qualified_output import (
    MechanismAttestationError,
    MechanismAttestationPolicy,
    QualifiedOutputVerifier,
    attestation_signing_message,
    canonical_attestation_bytes,
)
from tools.energy_telemetry.durable_privacy_ledger import (
    MAX_COHORT_PARTICIPANTS,
    MAX_DP_OUTPUT_BYTES,
    DurablePrivacyLedger,
    DurablePrivacyPolicyError,
    EventIdentityConflict,
)
from tools.energy_telemetry.store_forward import QueueCapacityError, TelemetryQueueError


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def people(start=0, count=20):
    return [f"local-only-participant-{i:04d}" for i in range(start, start + count)]


class DurablePrivacyLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database = str(Path(self.temp.name) / "privacy.sqlite3")
        self.key = b"private-tokenization-key-for-tests-32bytes!"
        self.now = dt("2026-10-13T06:00:00Z")
        self.ledger = DurablePrivacyLedger(
            self.database, participant_hmac_key=self.key,
        )
        # Stable test-only key. Never use these bytes as a deployment trust root.
        self.attestation_private_key = ed25519.Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
        self.attestation_public_key = self.attestation_private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.attestation_policy = MechanismAttestationPolicy(
            trusted_public_key=self.attestation_public_key,
            qualification_reference="test-qualification-v1",
            library_name="test-reviewed-library",
            library_version="1.2.3",
            mechanism="discrete_laplace",
            mechanism_id="integer-energy-sum-v1",
            privacy_unit="household",
            adjacency="add_remove_one_participant",
            contribution_policy_id="energy-cap-v1",
            privacy_policy_id="dp-policy-v1",
            max_contribution_limit_wh=1000,
            expected_feature_flags=(),
        )
        self.attestation_verifier = QualifiedOutputVerifier(self.attestation_policy)
        self.a = dt("2026-10-13T05:00:00Z")
        self.b = dt("2026-10-13T05:15:00Z")
        self.c = dt("2026-10-13T05:30:00Z")
        self.d = dt("2026-10-13T05:45:00Z")
        self.e = dt("2026-10-13T06:00:00Z")

    def tearDown(self):
        self.temp.cleanup()

    def reopen(self, **kwargs):
        kwargs.setdefault("participant_hmac_key", self.key)
        return DurablePrivacyLedger(self.database, **kwargs)

    def reserve(self, release="a", *, query="b", participants=None, start=None, end=None,
                reserved=None, recipient="approved_research", mechanism="discrete_laplace",
                epsilon=0.5, delta=0.0, adjacency="add_remove_one_participant",
                contribution_limit=1000, contribution_policy="energy-cap-v1",
                privacy_policy="dp-policy-v1", floor=None, ledger=None):
        target = self.ledger if ledger is None else ledger
        return target.reserve_release(
            release_id=release * 32,
            query_sha256=query * 64,
            participant_tokens=people() if participants is None else participants,
            window_start_utc=self.a if start is None else start,
            window_end_utc=self.b if end is None else end,
            reserved_at_utc=self.now if reserved is None else reserved,
            recipient_class=recipient,
            mechanism=mechanism,
            epsilon=epsilon,
            delta=delta,
            adjacency=adjacency,
            contribution_limit_wh=contribution_limit,
            contribution_policy_id=contribution_policy,
            privacy_policy_id=privacy_policy,
            minimum_cohort_size=floor,
        )

    def test_reservation_survives_restart_and_raw_participant_tokens_are_not_stored(self):
        result = self.reserve()
        self.assertEqual(result.status, "reserved")
        self.assertEqual(result.cohort_size, 20)
        reopened = self.reopen()
        self.assertEqual(reopened.reservation_count, 1)
        self.assertEqual(reopened.participant_spend(people()[0]), (0.5, 0.0, 1))
        backing_files = [Path(self.database)]
        backing_files.extend(
            Path(path) for path in (self.database + "-wal", self.database + "-shm")
            if Path(path).exists()
        )
        raw_storage = b"".join(path.read_bytes() for path in backing_files)
        self.assertNotIn(people()[0].encode(), raw_storage)
        self.assertEqual(reopened.integrity_check(), "ok")

    def test_exact_reservation_retry_is_idempotent_and_spends_budget_once(self):
        first = self.reserve()
        second = self.reserve(ledger=self.reopen())
        self.assertEqual(first.receipt_sha256, second.receipt_sha256)
        self.assertEqual(second.status, "reserved")
        self.assertEqual(self.ledger.reservation_count, 1)
        self.assertEqual(self.ledger.participant_spend(people()[0]), (0.5, 0.0, 1))

    def test_reused_release_id_with_different_query_fails_closed(self):
        self.reserve()
        with self.assertRaises(EventIdentityConflict):
            self.reserve(query="c")
        self.assertEqual(self.ledger.reservation_count, 1)

    def test_overlapping_windows_for_shared_participant_are_rejected(self):
        self.reserve()
        with self.assertRaisesRegex(DurablePrivacyPolicyError, "overlapping release window"):
            self.reserve(
                release="c", query="d",
                start=dt("2026-10-13T05:10:00Z"),
                end=dt("2026-10-13T05:20:00Z"),
            )
        self.assertEqual(self.ledger.reservation_count, 1)

    def test_disjoint_cohorts_can_use_overlapping_windows(self):
        self.reserve(participants=people(0, 20))
        second = self.reserve(
            release="c", query="d", participants=people(20, 20),
            start=dt("2026-10-13T05:05:00Z"),
            end=dt("2026-10-13T05:20:00Z"),
        )
        self.assertEqual(second.status, "reserved")
        self.assertEqual(self.ledger.reservation_count, 2)

    def test_cumulative_epsilon_budget_is_enforced(self):
        self.reserve(epsilon=1.0)
        self.reserve(release="c", query="d", start=self.b, end=self.c, epsilon=1.0)
        with self.assertRaisesRegex(DurablePrivacyPolicyError, "cumulative epsilon"):
            self.reserve(release="e", query="f", start=self.c, end=self.d, epsilon=0.01)
        self.assertEqual(self.ledger.participant_spend(people()[0]), (2.0, 0.0, 2))

    def test_cumulative_delta_budget_is_enforced(self):
        self.reserve(epsilon=0.5, delta=1e-6, mechanism="gaussian")
        self.reserve(release="c", query="d", start=self.b, end=self.c, epsilon=0.5, delta=1e-6, mechanism="gaussian")
        with self.assertRaisesRegex(DurablePrivacyPolicyError, "cumulative delta"):
            self.reserve(release="e", query="f", start=self.c, end=self.d, epsilon=0.5, delta=1e-7, mechanism="gaussian")
        self.assertEqual(self.ledger.participant_spend(people()[0]), (1.0, 2e-6, 2))

    def test_small_cohort_is_suppressed(self):
        with self.assertRaisesRegex(DurablePrivacyPolicyError, "below the required minimum"):
            self.reserve(participants=people(count=19))
        self.assertEqual(self.ledger.reservation_count, 0)

    def test_research_ledger_rejects_unsupported_recipient_and_mechanism(self):
        for kwargs in (
            {"recipient": "grid_operations"},
            {"mechanism": "made_up_mechanism"},
            {"adjacency": "undefined_adjacency"},
            {"contribution_policy": ""},
            {"privacy_policy": ""},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(DurablePrivacyPolicyError):
                    self.reserve(**kwargs)
        self.assertEqual(self.ledger.reservation_count, 0)

    def test_commit_output_is_immutable_and_survives_restart(self):
        reservation = self.reserve()
        output = b'{"cohort_energy_wh_noisy":4200,"schema":"v1"}'
        committed = self.ledger._commit_output_storage(
            release_id=reservation.release_id, output_bytes=output, committed_at_utc=self.now
        )
        self.assertFalse(committed.duplicate)
        retry = self.reopen()._commit_output_storage(
            release_id=reservation.release_id, output_bytes=output, committed_at_utc=self.now
        )
        self.assertTrue(retry.duplicate)
        self.assertEqual(retry.output_sha256, committed.output_sha256)
        self.assertEqual(self.reopen()._committed_output_storage(reservation.release_id).output_bytes, output)
        reservation_retry = self.reopen().reserve_release(
            release_id="a" * 32,
            query_sha256="b" * 64,
            participant_tokens=people(),
            window_start_utc=self.a,
            window_end_utc=self.b,
            reserved_at_utc=self.now,
            recipient_class="approved_research",
            mechanism="discrete_laplace",
            epsilon=0.5,
            delta=0.0,
            adjacency="add_remove_one_participant",
            contribution_limit_wh=1000,
            contribution_policy_id="energy-cap-v1",
            privacy_policy_id="dp-policy-v1",
        )
        self.assertEqual(reservation_retry.status, "output_committed")
        self.assertEqual(self.ledger.participant_spend(people()[0]), (0.5, 0.0, 1))

    def test_second_output_for_same_reservation_is_rejected(self):
        result = self.reserve()
        self.ledger._commit_output_storage(
            release_id=result.release_id, output_bytes=b'{"noisy":1}', committed_at_utc=self.now
        )
        with self.assertRaises(EventIdentityConflict):
            self.ledger._commit_output_storage(
                release_id=result.release_id, output_bytes=b'{"noisy":2}', committed_at_utc=self.now
            )

    def test_output_cannot_be_committed_without_budget_reservation(self):
        with self.assertRaisesRegex(DurablePrivacyPolicyError, "without a reservation"):
            self.ledger._commit_output_storage(
                release_id="f" * 32, output_bytes=b'{"noisy":1}', committed_at_utc=self.now
            )

    def test_committed_output_digest_is_rechecked(self):
        result = self.reserve()
        self.ledger._commit_output_storage(
            release_id=result.release_id, output_bytes=b'{"noisy":1}', committed_at_utc=self.now
        )
        with sqlite3.connect(self.database) as db:
            db.execute(
                "UPDATE privacy_releases SET output_sha256 = ? WHERE release_id = ?",
                ("0" * 64, result.release_id),
            )
        with self.assertRaisesRegex(TelemetryQueueError, "digest mismatch"):
            self.ledger._committed_output_storage(result.release_id)

    def test_changed_participant_hmac_key_is_rejected_after_restart(self):
        self.reserve()
        with self.assertRaisesRegex(TelemetryQueueError, "participant-key configuration mismatch"):
            self.reopen(participant_hmac_key=b"different-secret-key-with-at-least-32-bytes")

    def test_record_limit_fails_closed(self):
        path = str(Path(self.temp.name) / "limited.sqlite3")
        small = DurablePrivacyLedger(path, participant_hmac_key=self.key, max_reservations=1)
        self.reserve(ledger=small)
        with self.assertRaises(QueueCapacityError):
            self.reserve(release="c", query="d", start=self.b, end=self.c, ledger=small)
        self.assertEqual(small.reservation_count, 1)

    def test_concurrent_duplicate_reservations_spend_budget_once(self):
        workers = 8
        barrier = threading.Barrier(workers)
        results = []
        errors = []
        lock = threading.Lock()

        def reserve_worker():
            try:
                ledger = self.reopen()
                barrier.wait(timeout=5)
                result = self.reserve(ledger=ledger)
                with lock:
                    results.append(result)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=reserve_worker) for _ in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        self.assertTrue(all(not thread.is_alive() for thread in threads), "thread did not terminate")
        self.assertEqual(errors, [])
        self.assertEqual(len(results), workers)
        self.assertEqual(len({r.receipt_sha256 for r in results}), 1)
        self.assertEqual(self.ledger.reservation_count, 1)
        self.assertEqual(self.ledger.participant_spend(people()[0]), (0.5, 0.0, 1))

    def test_concurrent_distinct_releases_cannot_overspend_shared_participant_budget(self):
        workers = 5
        barrier = threading.Barrier(workers)
        results = []
        rejected = []
        errors = []
        lock = threading.Lock()
        base = datetime(2026, 10, 13, 4, 0, tzinfo=timezone.utc)

        def worker(index):
            try:
                ledger = self.reopen()
                start = base + timedelta(minutes=10 * index)
                end = start + timedelta(minutes=5)
                barrier.wait(timeout=5)
                result = self.reserve(
                    release="abcde"[index],
                    query="12345"[index],
                    participants=people(),
                    start=start,
                    end=end,
                    epsilon=0.5,
                    delta=0.0,
                    mechanism="discrete_laplace",
                    ledger=ledger,
                )
                with lock:
                    results.append(result)
            except DurablePrivacyPolicyError as exc:
                with lock:
                    rejected.append(str(exc))
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        self.assertTrue(all(not thread.is_alive() for thread in threads), "thread did not terminate")
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 4)
        self.assertEqual(len(rejected), 1)
        self.assertIn("cumulative epsilon budget exceeded", rejected[0])
        self.assertEqual(self.ledger.reservation_count, 4)
        self.assertEqual(self.ledger.participant_spend(people()[0]), (2.0, 0.0, 4))

    def test_participant_tokens_cannot_be_passed_as_one_string(self):
        with self.assertRaisesRegex(DurablePrivacyPolicyError, "iterable of local tokens"):
            self.reserve(participants="this-is-not-a-participant-list")
        self.assertEqual(self.ledger.reservation_count, 0)

    def test_nonfinite_and_extreme_budget_configurations_fail_closed(self):
        for value in (float("inf"), float("nan"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(DurablePrivacyPolicyError, "finite numeric"):
                    DurablePrivacyLedger(
                        str(Path(self.temp.name) / f"bad-{str(value)}.sqlite3"),
                        participant_hmac_key=self.key,
                        max_epsilon_per_participant=value,
                    )
        with self.assertRaisesRegex(DurablePrivacyPolicyError, "exceeds the configured release limit"):
            DurablePrivacyLedger(
                str(Path(self.temp.name) / "giant-budget.sqlite3"),
                participant_hmac_key=self.key,
                max_epsilon_per_participant=10 ** 1000,
            )
        with self.assertRaisesRegex(ValueError, "positive SQLite integer"):
            DurablePrivacyLedger(
                str(Path(self.temp.name) / "bad-record-limit.sqlite3"),
                participant_hmac_key=self.key,
                max_reservations=1 << 100,
            )

    def test_mechanism_and_delta_must_be_consistent(self):
        with self.assertRaisesRegex(DurablePrivacyPolicyError, "discrete_laplace requires delta=0"):
            self.reserve(delta=1e-6)
        with self.assertRaisesRegex(DurablePrivacyPolicyError, "gaussian mechanism requires positive delta"):
            self.reserve(mechanism="gaussian", delta=0.0)
        self.assertEqual(self.ledger.reservation_count, 0)

    def test_malformed_recipient_mechanism_and_adjacency_types_fail_closed(self):
        for kwargs in (
            {"recipient": []},
            {"mechanism": []},
            {"adjacency": {}},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(DurablePrivacyPolicyError):
                    self.reserve(**kwargs)
        self.assertEqual(self.ledger.reservation_count, 0)

    def test_output_cannot_be_committed_before_release_window_ends(self):
        reservation = self.reserve(
            release="c", query="d",
            start=dt("2026-10-13T05:45:00Z"),
            end=dt("2026-10-13T06:02:00Z"),
            reserved=self.now,
        )
        with self.assertRaisesRegex(DurablePrivacyPolicyError, "before its release window ends"):
            self.ledger._commit_output_storage(
                release_id=reservation.release_id,
                output_bytes=b'{"noisy":4200}',
                committed_at_utc=dt("2026-10-13T06:01:00Z"),
            )
        self.assertEqual(self.ledger.reservation_count, 1)
        with self.assertRaisesRegex(DurablePrivacyPolicyError, "no committed output"):
            self.ledger._committed_output_storage(reservation.release_id)

    def test_output_commit_time_cannot_precede_reservation(self):
        reservation = self.reserve()
        with self.assertRaisesRegex(DurablePrivacyPolicyError, "cannot precede"):
            self.ledger._commit_output_storage(
                release_id=reservation.release_id,
                output_bytes=b'{"noisy":1}',
                committed_at_utc=dt("2026-10-13T05:59:59Z"),
            )
        self.assertEqual(self.ledger.reservation_count, 1)
        with self.assertRaisesRegex(DurablePrivacyPolicyError, "no committed output"):
            self.ledger._committed_output_storage(reservation.release_id)


    def test_unbounded_participant_iterable_is_bounded_before_budget_write(self):
        consumed = 0

        def unbounded_participants():
            nonlocal consumed
            while True:
                consumed += 1
                yield f"generated-local-participant-{consumed}"

        with self.assertRaisesRegex(
            DurablePrivacyPolicyError,
            f"exceeds hard maximum of {MAX_COHORT_PARTICIPANTS}",
        ):
            self.reserve(participants=unbounded_participants())

        self.assertEqual(consumed, MAX_COHORT_PARTICIPANTS + 1)
        self.assertEqual(self.ledger.reservation_count, 0)

    def make_signed_attestation(self, reservation, output_bytes, **overrides):
        context = self.ledger.reservation_context(reservation.release_id)
        fields = {
            "schema_version": 1,
            "release_id": context.release_id,
            "reservation_receipt_sha256": context.receipt_sha256,
            "query_sha256": context.query_sha256,
            "window_start_utc": context.window_start_utc,
            "window_end_utc": context.window_end_utc,
            "recipient_class": context.recipient_class,
            "mechanism": context.mechanism,
            "library_name": "test-reviewed-library",
            "library_version": "1.2.3",
            "mechanism_id": "integer-energy-sum-v1",
            "feature_flags": [],
            "privacy_unit": "household",
            "adjacency": context.adjacency,
            "contribution_limit_wh": context.contribution_limit_wh,
            "sensitivity_wh": context.contribution_limit_wh,
            "contribution_policy_id": context.contribution_policy_id,
            "privacy_policy_id": context.privacy_policy_id,
            "epsilon_nano": context.epsilon_nano,
            "delta_pico": context.delta_pico,
            "cohort_size": context.cohort_size,
            "qualification_reference": "test-qualification-v1",
            "execution_id": "e" * 32,
            "signer_key_sha256": hashlib.sha256(self.attestation_public_key).hexdigest(),
            "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
        }
        fields.update(overrides)
        raw = canonical_attestation_bytes(fields)
        signature = self.attestation_private_key.sign(attestation_signing_message(raw))
        return raw, signature

    def verify_signed_output(self, reservation, output_bytes, attestation=None, signature=None, verifier=None):
        if attestation is None or signature is None:
            attestation, signature = self.make_signed_attestation(reservation, output_bytes)
        return (self.attestation_verifier if verifier is None else verifier).verify_and_commit(
            self.ledger,
            release_id=reservation.release_id,
            output_bytes=output_bytes,
            attestation_bytes=attestation,
            signature=signature,
            committed_at_utc=self.now,
        )

    def test_unqualified_low_level_output_cannot_be_reloaded_as_attested(self):
        reservation = self.reserve()
        self.ledger._commit_output_storage(
            release_id=reservation.release_id,
            output_bytes=b'{"value":1}',
            committed_at_utc=self.now,
        )
        with self.assertRaisesRegex(DurablePrivacyPolicyError, "no retained signed attestation"):
            self.attestation_verifier.load_committed_output(
                self.ledger, release_id=reservation.release_id
            )

    def test_stored_attestation_tampering_is_detected_after_restart(self):
        reservation = self.reserve()
        output = b'{"value":1}'
        raw, signature = self.make_signed_attestation(reservation, output)
        self.verify_signed_output(reservation, output, raw, signature)
        with sqlite3.connect(self.database) as db:
            db.execute(
                "UPDATE privacy_release_attestations SET attestation_bytes = ? WHERE release_id = ?",
                (b"{}", reservation.release_id),
            )
        with self.assertRaisesRegex(TelemetryQueueError, "attestation evidence failed integrity"):
            self.attestation_verifier.load_committed_output(
                self.reopen(), release_id=reservation.release_id
            )

    def test_verifier_rejects_malformed_deployment_policy_types(self):
        base = {
            "trusted_public_key": self.attestation_public_key,
            "qualification_reference": "test-qualification-v1",
            "library_name": "test-reviewed-library",
            "library_version": "1.2.3",
            "mechanism": "discrete_laplace",
            "mechanism_id": "integer-energy-sum-v1",
            "privacy_unit": "household",
            "adjacency": "add_remove_one_participant",
            "contribution_policy_id": "energy-cap-v1",
            "privacy_policy_id": "dp-policy-v1",
            "max_contribution_limit_wh": 1000,
            "expected_feature_flags": (),
        }
        for change in ({"mechanism": []}, {"privacy_unit": []}, {"expected_feature_flags": ["contrib"]}):
            with self.subTest(change=change):
                candidate = dict(base)
                candidate.update(change)
                with self.assertRaises(MechanismAttestationError):
                    QualifiedOutputVerifier(MechanismAttestationPolicy(**candidate))

    def test_verifier_rejects_oversized_output_before_hashing_or_commit(self):
        with self.assertRaisesRegex(MechanismAttestationError, "at most 1 MiB"):
            self.attestation_verifier.verify_and_commit(
                self.ledger,
                release_id="a" * 32,
                output_bytes=b"x" * (MAX_DP_OUTPUT_BYTES + 1),
                attestation_bytes=b"{}",
                signature=b"x" * 64,
                committed_at_utc=self.now,
            )
        self.assertEqual(self.ledger.reservation_count, 0)

    def test_no_public_unqualified_output_commit_api_is_exposed(self):
        self.assertFalse(hasattr(self.ledger, "commit_output"))
        self.assertFalse(hasattr(self.ledger, "committed_output"))
        self.assertTrue(hasattr(self.ledger, "_commit_output_storage"))

    def test_signed_mechanism_attestation_binds_reservation_and_output(self):
        reservation = self.reserve()
        output = b'{"energy_wh_noisy":4012,"schema":"v1"}'
        raw, signature = self.make_signed_attestation(reservation, output)
        committed = self.verify_signed_output(reservation, output, raw, signature)
        self.assertFalse(committed.duplicate)
        self.assertEqual(committed.output_bytes, output)
        self.assertEqual(committed.output_sha256, hashlib.sha256(output).hexdigest())
        replay = self.verify_signed_output(reservation, output, raw, signature)
        self.assertTrue(replay.duplicate)
        self.assertEqual(self.ledger._committed_output_storage(reservation.release_id).output_bytes, output)
        reopened = self.reopen()
        recovered = self.attestation_verifier.load_committed_output(
            reopened, release_id=reservation.release_id
        )
        self.assertTrue(recovered.duplicate)
        self.assertEqual(recovered.output_bytes, output)

    def test_unsigned_or_tampered_attestation_cannot_commit_output(self):
        reservation = self.reserve()
        output = b'{"energy_wh_noisy":4012}'
        raw, signature = self.make_signed_attestation(reservation, output)
        damaged = bytearray(signature)
        damaged[0] ^= 1
        with self.assertRaisesRegex(MechanismAttestationError, "signature is invalid"):
            self.verify_signed_output(reservation, output, raw, bytes(damaged))
        self.assertEqual(self.ledger.reservation_count, 1)
        with self.assertRaisesRegex(DurablePrivacyPolicyError, "no committed output"):
            self.ledger._committed_output_storage(reservation.release_id)

    def test_attestation_for_different_output_cannot_be_replayed(self):
        reservation = self.reserve()
        original = b'{"energy_wh_noisy":4012}'
        raw, signature = self.make_signed_attestation(reservation, original)
        with self.assertRaisesRegex(MechanismAttestationError, "output digest"):
            self.verify_signed_output(reservation, b'{"energy_wh_noisy":4013}', raw, signature)
        with self.assertRaisesRegex(DurablePrivacyPolicyError, "no committed output"):
            self.ledger._committed_output_storage(reservation.release_id)

    def test_attestation_must_match_exact_reservation_fields(self):
        mismatch_cases = (
            {"release_id": "f" * 32},
            {"reservation_receipt_sha256": "0" * 64},
            {"query_sha256": "0" * 64},
            {"epsilon_nano": 500_000_001},
            {"delta_pico": 1},
            {"cohort_size": 21},
            {"adjacency": "replace_one_participant"},
            {"contribution_limit_wh": 2000},
            {"contribution_policy_id": "changed-cap-v2"},
            {"privacy_policy_id": "changed-policy-v2"},
            {"recipient_class": "public_reporting"},
            {"window_start_utc": "2026-10-13T04:00:00.000000Z"},
            {"window_end_utc": "2026-10-13T05:20:00.000000Z"},
        )
        for mismatch in mismatch_cases:
            with self.subTest(mismatch=mismatch):
                case_index = mismatch_cases.index(mismatch)
                case_id = "0123456789abc"[case_index]
                reservation = self.reserve(
                    release=case_id,
                    query=case_id,
                    participants=people((case_index + 1) * 20, 20),
                )
                output = b'{"value":1}'
                raw, signature = self.make_signed_attestation(reservation, output, **mismatch)
                with self.assertRaises(MechanismAttestationError):
                    self.verify_signed_output(reservation, output, raw, signature)
                with self.assertRaises(DurablePrivacyPolicyError):
                    self.ledger._committed_output_storage(reservation.release_id)

    def test_unapproved_library_version_constructor_privacy_unit_and_reference_fail(self):
        policy_changes = (
            {"library_name": "other-library"},
            {"library_version": "9.9.9"},
            {"mechanism_id": "unreviewed-sampler"},
            {"privacy_unit": "site"},
            {"qualification_reference": "unreviewed-qualification"},
            {"signer_key_sha256": "0" * 64},
        )
        for index, change in enumerate(policy_changes):
            with self.subTest(change=change):
                reservation = self.reserve(
                    release=chr(97 + index),
                    query=chr(97 + index),
                    participants=people((index + 30) * 20, 20),
                )
                output = b'{"value":1}'
                raw, signature = self.make_signed_attestation(reservation, output, **change)
                with self.assertRaises(MechanismAttestationError):
                    self.verify_signed_output(reservation, output, raw, signature)

    def test_reservation_cannot_exceed_deployment_qualified_contribution_cap(self):
        reservation = self.reserve(contribution_limit=2000)
        output = b'{"value":1}'
        raw, signature = self.make_signed_attestation(reservation, output)
        with self.assertRaisesRegex(MechanismAttestationError, "exceeds deployment-qualified maximum"):
            self.verify_signed_output(reservation, output, raw, signature)
        with self.assertRaisesRegex(DurablePrivacyPolicyError, "no committed output"):
            self.ledger._committed_output_storage(reservation.release_id)

    def test_unapproved_feature_flags_fail_closed(self):
        reservation = self.reserve()
        output = b'{"value":1}'
        raw, signature = self.make_signed_attestation(reservation, output, feature_flags=["contrib"])
        with self.assertRaisesRegex(MechanismAttestationError, "feature_flags"):
            self.verify_signed_output(reservation, output, raw, signature)

    def test_wrong_pinned_trust_key_fails_even_for_valid_other_signature(self):
        reservation = self.reserve()
        output = b'{"value":1}'
        raw, signature = self.make_signed_attestation(reservation, output)
        other_private = ed25519.Ed25519PrivateKey.from_private_bytes(bytes(reversed(range(32))))
        other_public = other_private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        other_verifier = QualifiedOutputVerifier(MechanismAttestationPolicy(
            trusted_public_key=other_public,
            qualification_reference="test-qualification-v1",
            library_name="test-reviewed-library",
            library_version="1.2.3",
            mechanism="discrete_laplace",
            mechanism_id="integer-energy-sum-v1",
            privacy_unit="household",
            adjacency="add_remove_one_participant",
            contribution_policy_id="energy-cap-v1",
            privacy_policy_id="dp-policy-v1",
            max_contribution_limit_wh=1000,
            expected_feature_flags=(),
        ))
        with self.assertRaisesRegex(MechanismAttestationError, "signature is invalid"):
            self.verify_signed_output(reservation, output, raw, signature, verifier=other_verifier)

    def test_noncanonical_json_and_duplicate_keys_are_rejected(self):
        reservation = self.reserve()
        output = b'{"value":1}'
        raw, signature = self.make_signed_attestation(reservation, output)
        with self.assertRaisesRegex(MechanismAttestationError, "canonical form"):
            self.verify_signed_output(reservation, output, raw + b" ", signature)
        duplicate = b'{"schema_version":1,"schema_version":1}'
        with self.assertRaisesRegex(MechanismAttestationError, "duplicate JSON key"):
            self.verify_signed_output(reservation, output, duplicate, signature)

    def test_malformed_and_extra_attestation_fields_are_rejected(self):
        reservation = self.reserve()
        output = b'{"value":1}'
        raw, signature = self.make_signed_attestation(reservation, output, untrusted_extra_field="x")
        with self.assertRaisesRegex(MechanismAttestationError, "field set mismatch"):
            self.verify_signed_output(reservation, output, raw, signature)
        with self.assertRaisesRegex(MechanismAttestationError, "valid UTF-8 JSON"):
            self.verify_signed_output(reservation, output, b"not-json", signature)

    def test_sensitivity_must_match_integer_energy_bound(self):
        reservation = self.reserve()
        output = b'{"value":1}'
        raw, signature = self.make_signed_attestation(reservation, output, sensitivity_wh=999)
        with self.assertRaisesRegex(MechanismAttestationError, "sensitivity_wh"):
            self.verify_signed_output(reservation, output, raw, signature)

    def test_attestation_verification_requires_a_real_reservation(self):
        output = b'{"value":1}'
        # Build a structurally sound, signed receipt for an identifier never reserved.
        fake = type("FakeReservation", (), {
            "release_id": "f" * 32,
        })()
        raw, signature = self.make_signed_attestation(self.reserve(), output, release_id="f" * 32)
        with self.assertRaisesRegex(DurablePrivacyPolicyError, "without a reservation"):
            self.attestation_verifier.verify_and_commit(
                self.ledger, release_id=fake.release_id, output_bytes=output,
                attestation_bytes=raw, signature=signature, committed_at_utc=self.now,
            )

if __name__ == "__main__":
    unittest.main()
