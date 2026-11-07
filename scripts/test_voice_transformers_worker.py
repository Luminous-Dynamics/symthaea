#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model-free contract tests for the offline Transformers worker."""
from __future__ import annotations

import json
import math
import unittest

from voice_asr_score import _hash_tree
from voice_transformers_worker import (
    LOCALE,
    handle_request,
    parse_request,
    preserved_text,
    response_envelope,
    transcribe_observation,
    validate_observation,
    validate_local_model_snapshot,
    hash_local_model_snapshot,
)


def observation(samples=None, **audio_overrides):
    audio = {"sample_rate_hz": 16000, "channels": 1}
    audio.update(audio_overrides)
    return {
        "id": "fixture-content-addressed-id",
        "modality": {"Audio": audio},
        "samples": [0.0, 0.1, -0.1] if samples is None else samples,
    }


class FakeFeatures:
    def __init__(self):
        self.moves = []

    def to(self, **kwargs):
        self.moves.append(kwargs)
        return self


class FakeProcessor:
    def __init__(self, transcript="Môre"):
        self.transcript = transcript
        self.calls = []
        self.features = FakeFeatures()
        self.decode_calls = []

    def __call__(self, samples, **kwargs):
        self.calls.append((samples, kwargs))
        return {"input_features": self.features}

    def batch_decode(self, generated, **kwargs):
        self.decode_calls.append((generated, kwargs))
        return [self.transcript]


class NoFeaturesProcessor(FakeProcessor):
    def __call__(self, samples, **kwargs):
        return {}


class FakeModel:
    def __init__(self, generated=None):
        self.generated = ["fake-token-ids"] if generated is None else generated
        self.calls = []

    def generate(self, features, **kwargs):
        self.calls.append((features, kwargs))
        return self.generated


class FakeTorch:
    class _Inference:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    @staticmethod
    def inference_mode():
        return FakeTorch._Inference()


class FakeNumpy:
    float32 = "float32"

    @staticmethod
    def asarray(values, dtype=None):
        return ("array", list(values), dtype)


class TransformersWorkerTests(unittest.TestCase):
    def test_duplicate_json_keys_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
            parse_request('{"operation":"transcribe","request_id":"1","request_id":"2"}')

    def test_nonstandard_json_numbers_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-standard JSON"):
            parse_request('{"operation":"transcribe","request_id":"1","value":NaN}')

    def test_non_object_missing_id_and_unknown_operation_rejected(self):
        with self.assertRaisesRegex(ValueError, "JSON object"):
            parse_request("[]")
        with self.assertRaisesRegex(ValueError, "request_id"):
            parse_request('{"operation":"transcribe"}')
        with self.assertRaisesRegex(ValueError, "unknown"):
            parse_request('{"operation":"mystery","request_id":"1"}')

    def test_accepts_valid_transcribe_request_shape(self):
        request = parse_request(json.dumps({
            "operation": "transcribe", "request_id": "req-1", "observation": observation()
        }))
        self.assertEqual(set(request), {"operation", "request_id", "observation"})

    def test_request_operation_requires_exact_fields(self):
        with self.assertRaisesRegex(ValueError, "request fields mismatch"):
            parse_request('{"operation":"transcribe","request_id":"req-1"}')
        with self.assertRaisesRegex(ValueError, "request fields mismatch"):
            parse_request('{"operation":"capabilities","request_id":"req-1","observation":{}}')
        with self.assertRaisesRegex(ValueError, "target_language"):
            parse_request('{"operation":"translate_speech","request_id":"req-1","observation":{},"target_language":" "}')

    def test_accepts_valid_16khz_mono_observation(self):
        self.assertEqual(validate_observation(observation(), max_audio_seconds=60), [0.0, 0.1, -0.1])

    def test_rejects_non_audio_wrong_rate_and_channels(self):
        value = observation()
        value["modality"] = {"Text": {"language": "af", "script": "Latn"}}
        with self.assertRaisesRegex(ValueError, "Audio"):
            validate_observation(value, max_audio_seconds=60)
        with self.assertRaisesRegex(ValueError, "16 kHz"):
            validate_observation(observation(sample_rate_hz=44100), max_audio_seconds=60)
        with self.assertRaisesRegex(ValueError, "mono"):
            validate_observation(observation(channels=2), max_audio_seconds=60)
        with self.assertRaisesRegex(ValueError, "mono"):
            validate_observation(observation(channels=True), max_audio_seconds=60)
        with self.assertRaisesRegex(ValueError, "16 kHz"):
            validate_observation(observation(sample_rate_hz=True), max_audio_seconds=60)
        extra = observation()
        extra["modality"]["Audio"]["unexpected"] = "ignored field"
        with self.assertRaisesRegex(ValueError, "fields do not match"):
            validate_observation(extra, max_audio_seconds=60)

    def test_rejects_empty_oversized_and_bad_samples(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            validate_observation(observation([]), max_audio_seconds=60)
        with self.assertRaisesRegex(ValueError, "duration"):
            validate_observation(observation([0.0] * 16001), max_audio_seconds=1)
        for bad in (math.nan, math.inf, -math.inf, 1.01, -1.01, True, "0.5"):
            with self.subTest(bad=repr(bad)):
                with self.assertRaisesRegex(ValueError, "sample"):
                    validate_observation(observation([bad]), max_audio_seconds=60)

    def test_local_model_snapshot_requires_real_directory_and_regular_files(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            root.mkdir()
            (root / "config.json").write_text("{}", encoding="utf-8")
            self.assertEqual(validate_local_model_snapshot(str(root)), str(root.resolve()))
            digest, count, total_bytes = hash_local_model_snapshot(str(root))
            self.assertEqual((digest, count, total_bytes), _hash_tree(root))
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
            self.assertEqual(count, 1)
            self.assertEqual(total_bytes, 2)
            alias = Path(tmp) / "link"
            alias.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "not a symlink"):
                validate_local_model_snapshot(str(alias))
            (root / "external-link").symlink_to(Path(tmp) / "outside.bin")
            with self.assertRaisesRegex(ValueError, "symlink"):
                validate_local_model_snapshot(str(root))

    def test_transcription_uses_model_config_prompt_without_language_override(self):
        processor, model = FakeProcessor("Môre!"), FakeModel()
        text = transcribe_observation(
            observation(),
            processor=processor,
            model=model,
            torch_module=FakeTorch,
            numpy_module=FakeNumpy,
            device="cpu",
            dtype="float32",
            max_audio_seconds=60,
        )
        self.assertEqual(text["original"], "Môre!")
        self.assertIsNone(text["normalized"])
        self.assertEqual(text["primary_language"], "af")
        self.assertEqual(text["script"], "Latn")
        self.assertEqual(processor.calls[0][1], {"sampling_rate": 16000, "return_tensors": "pt"})
        self.assertEqual(processor.features.moves[0], {"device": "cpu", "dtype": "float32"})
        self.assertEqual(model.calls[0][1], {
            "num_beams": 1, "no_repeat_ngram_size": 3, "repetition_penalty": 1.2
        })
        self.assertNotIn("language", model.calls[0][1])

    def test_empty_hypothesis_is_preserved_explicitly(self):
        processor, model = FakeProcessor(""), FakeModel()
        text = transcribe_observation(
            observation(),
            processor=processor,
            model=model,
            torch_module=FakeTorch,
            numpy_module=FakeNumpy,
            device="cpu",
            dtype="float32",
            max_audio_seconds=60,
        )
        self.assertEqual(text, {
            "original": "", "normalized": "", "primary_language": "af", "script": "Latn", "spans": []
        })
        self.assertEqual(preserved_text(""), text)

    def test_missing_features_and_bad_decode_fail_closed(self):
        processor, model = NoFeaturesProcessor(), FakeModel()
        with self.assertRaisesRegex(ValueError, "input_features"):
            transcribe_observation(observation(), processor=processor, model=model,
                torch_module=FakeTorch, numpy_module=FakeNumpy, device="cpu", dtype="float32", max_audio_seconds=60)
        processor, model = FakeProcessor(), FakeModel()
        processor.batch_decode = lambda *args, **kwargs: []
        with self.assertRaisesRegex(ValueError, "decoded batch"):
            transcribe_observation(observation(), processor=processor, model=model,
                torch_module=FakeTorch, numpy_module=FakeNumpy, device="cpu", dtype="float32", max_audio_seconds=60)

    def test_response_envelope_requires_exactly_one_result_or_error(self):
        good = response_envelope("worker", "req-1", {"kind": "languages", "value": []})
        self.assertEqual(good["protocol_version"], 1)
        self.assertEqual(good["request_id"], "req-1")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            response_envelope("worker", "req-1")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            response_envelope("worker", "req-1", {"kind": "languages", "value": []}, "error")

    def test_identify_language_does_not_claim_configured_locale_as_detection(self):
        envelope = handle_request(
            {"operation": "identify_language", "request_id": "req-1"},
            provider_id="test-worker", processor=None, model=None,
            torch_module=None, numpy_module=None, device="cpu", dtype="float32", max_audio_seconds=60,
        )
        self.assertEqual(envelope["result"], {"kind": "languages", "value": []})
        self.assertIsNone(envelope["error"])

    def test_unsupported_operations_return_errors_not_fake_capabilities(self):
        for operation in ("translate", "translate_speech", "capabilities"):
            with self.subTest(operation=operation):
                envelope = handle_request(
                    {"operation": operation, "request_id": "req-1"},
                    provider_id="test-worker", processor=None, model=None,
                    torch_module=None, numpy_module=None, device="cpu", dtype="float32", max_audio_seconds=60,
                )
                self.assertIsNone(envelope["result"])
                self.assertIsInstance(envelope["error"], str)

    def test_request_failures_are_reported_as_protocol_errors(self):
        envelope = handle_request(
            {"operation": "transcribe", "request_id": "req-1", "observation": observation([math.nan])},
            provider_id="test-worker", processor=None, model=None,
            torch_module=FakeTorch, numpy_module=FakeNumpy, device="cpu", dtype="float32", max_audio_seconds=60,
        )
        self.assertIsNone(envelope["result"])
        self.assertIn("audio sample", envelope["error"])


if __name__ == "__main__":
    unittest.main()
