#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""DiffSinger placeholder-voice worker for voice::diffsinger's JSONL bridge.

Speaks the fail-closed request/response protocol `src/voice/diffsinger.rs`
already implements on the Rust side (spawns this as a child process,
one JSON line per request on stdin, one JSON line per response on stdout).
This script does the actual singing-voice-synthesis inference; the Rust
side never does model I/O itself -- "provisioning and model download are
deliberately separate from runtime" (see diffsinger.rs's own module doc).

Why Python, not the `ort` Rust crate: the whole ONNX tensor pipeline
(linguistic encoder -> variance predictor -> acoustic diffusion -> vocoder)
was already built and verified working in Python for SING-15
(SYMTHAEA_SINGING_PLAN_2026-07-18.md) using onnxruntime directly -- no
PyTorch needed, every stage is pre-exported ONNX. Porting that to
untested `ort` crate tensor APIs (bool/scalar tensors in particular)
would mean re-verifying correctness inside multi-hour Rust rebuilds
(this monorepo's shared-tree contention made even small `cargo test`
runs take 45min-2h+ on 2026-07-24); reusing the already-proven Python
path was the pragmatic call, not a language preference.

Placeholder voice, not Symthaea's final voice: see the voicebank's own
terms-of-service.md before use (attribution required when publishing,
commercial use needs author approval, no training another model on its
outputs). The voicebank directory is located via --voicebank or the
SYMTHAEA_DIFFSINGER_VOICEBANK env var; this script never downloads
anything itself.

Run via scripts/diffsinger_worker.sh (handles the onnxruntime/numpy
environment via nix-shell) rather than invoking this file directly.
"""
import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

HEAD_FRAMES = 8
TAIL_FRAMES = 8
SAMPLE_RATE = 44100
HOP_SIZE = 512
FRAME_MS = 1000.0 * HOP_SIZE / SAMPLE_RATE  # ~11.61 ms
PROTOCOL_VERSION = 1
PROVIDER_ID = "diffsinger-local"

# IPA -> ARPAbet-style phone (matches src/voice/diffsinger_phonemes.rs's
# now-removed Rust table 1:1 -- kept here since this is the only remaining
# copy after the Rust ONNX path was replaced by this worker).
IPA_TO_ARPABET = {
    "aɪ": ["ay"], "aʊ": ["aw"], "eɪ": ["ey"], "oʊ": ["ow"], "ɔɪ": ["oy"],
    "iː": ["iy"], "uː": ["uw"], "ɑː": ["aa"], "ɔː": ["ao"], "ɜː": ["er"],
    "æ": ["ae"], "ɑ": ["aa"], "ɒ": ["aa"], "ɔ": ["ao"], "ə": ["ax"],
    "ɛ": ["eh"], "ɪ": ["ih"], "ʊ": ["uh"], "ʌ": ["ah"], "i": ["iy"],
    "u": ["uw"], "a": ["ah"],
    "tʃ": ["ch"], "dʒ": ["jh"], "ð": ["dh"], "ŋ": ["ng"], "ɹ": ["r"],
    "ʃ": ["sh"], "ʒ": ["zh"], "θ": ["th"],
    "b": ["b"], "d": ["d"], "f": ["f"], "ɡ": ["g"], "g": ["g"], "h": ["hh"],
    "j": ["y"], "k": ["k"], "l": ["l"], "m": ["m"], "n": ["n"], "p": ["p"],
    "s": ["s"], "t": ["t"], "v": ["v"], "w": ["w"], "z": ["z"],
    "ks": ["k", "s"],
}


def ipa_to_arpabet(ipa: str):
    return IPA_TO_ARPABET.get(ipa, ["ax"])


def hz_to_tone(hz: float) -> float:
    return 69.0 + 12.0 * math.log2(hz / 440.0)


def ms_to_frames(durations_ms):
    frames = []
    accumulated = 0.0
    previous = 0
    for d in durations_ms:
        accumulated += d
        frame = int(accumulated / FRAME_MS + 0.5)
        frames.append(frame - previous)
        previous = frame
    return frames


def load_emb(path: Path, hidden_size: int = 384) -> np.ndarray:
    data = path.read_bytes()
    assert len(data) == hidden_size * 4, f"{path} is {len(data)} bytes, expected {hidden_size*4}"
    return np.array(struct.unpack(f"<{hidden_size}f", data), dtype=np.float32)


class DiffSingerModel:
    """Loads all four ONNX stages once; synthesize() is called per request."""

    def __init__(self, root: Path):
        self.root = root
        self.phoneme_ids = json.loads((root / "dsmain" / "phonemes.json").read_text())
        self.language_ids = json.loads((root / "dsmain" / "languages.json").read_text())
        self.speaker_embed_main = load_emb(root / "dsmain" / "Standard.emb")
        self.speaker_embed_variance = load_emb(root / "dsvariance" / "Standard.emb")

        opts = ["CPUExecutionProvider"]
        self.linguistic = ort.InferenceSession(str(root / "dsvariance" / "linguistic.onnx"), providers=opts)
        self.variance = ort.InferenceSession(str(root / "dsvariance" / "variance.onnx"), providers=opts)
        self.acoustic = ort.InferenceSession(str(root / "dsmain" / "acoustic.onnx"), providers=opts)
        vocoder_dir = root / "dsvocoder"
        vocoder_path = next(p for p in vocoder_dir.iterdir() if p.suffix == ".onnx")
        self.vocoder = ort.InferenceSession(str(vocoder_path), providers=opts)

    def phoneme_token(self, phoneme: str):
        return self.phoneme_ids.get(phoneme)

    def language_id(self, lang: str) -> int:
        return self.language_ids.get(lang, 0)

    def build_score(self, performance: dict):
        """Walk a serialized VocalPerformance into (tokens, languages,
        ph_dur, f0) arrays, SP-padded, matching diffsinger_singing.rs's
        (now-removed) design: proportional per-syllable phoneme timing from
        natural_duration_s, melisma notes held on the syllable's last
        phoneme."""
        arpabet_phonemes = []
        durations_ms = []
        target_hz = []

        for syllable in performance["syllables"]:
            phonemes = syllable["phonemes"]
            total_natural_ms = max(sum(p["natural_duration_s"] * 1000.0 for p in phonemes), 1.0)
            note_ms = max(syllable["note"]["duration"] * 1000.0, 1.0)
            last_arpabet = "ax"
            for phoneme in phonemes:
                for arpabet in ipa_to_arpabet(phoneme["ipa"]):
                    arpabet_phonemes.append(arpabet)
                    share = (phoneme["natural_duration_s"] * 1000.0) / total_natural_ms
                    durations_ms.append(max(share * note_ms, 1.0))
                    target_hz.append(syllable["note"]["frequency"])
                    last_arpabet = arpabet
            for melisma_note in syllable["melisma_notes"]:
                arpabet_phonemes.append(last_arpabet)
                durations_ms.append(max(melisma_note["duration"] * 1000.0, 1.0))
                target_hz.append(melisma_note["frequency"])

        if not arpabet_phonemes:
            return None

        tokens = [self.phoneme_token("SP")]
        languages = [0]
        for arpabet in arpabet_phonemes:
            namespaced = f"en/{arpabet}"
            token = self.phoneme_token(namespaced)
            if token is None:
                raise ValueError(f"voicebank phonemes.json has no entry for '{namespaced}'")
            tokens.append(token)
            languages.append(self.language_id("en"))
        tokens.append(self.phoneme_token("SP"))
        languages.append(0)

        head_ms = FRAME_MS * HEAD_FRAMES
        tail_ms = FRAME_MS * TAIL_FRAMES
        padded_durations_ms = [head_ms] + durations_ms + [tail_ms]
        ph_dur = ms_to_frames(padded_durations_ms)

        padded_hz = [target_hz[0]] + target_hz + [target_hz[-1]]
        f0 = []
        for hz, frames in zip(padded_hz, ph_dur):
            f0.extend([hz] * max(frames, 0))

        return (
            np.array([tokens], dtype=np.int64),
            np.array([languages], dtype=np.int64),
            np.array([ph_dur], dtype=np.int64),
            np.array([f0], dtype=np.float32),
        )

    def synthesize(self, performance: dict):
        score = self.build_score(performance)
        if score is None:
            return np.array([], dtype=np.float32), SAMPLE_RATE
        tokens, languages, ph_dur, f0 = score
        total_frames = int(ph_dur.sum())

        (encoder_out, _x_masks) = self.linguistic.run(
            None, {"tokens": tokens, "languages": languages, "ph_dur": ph_dur}
        )

        tone_cents = np.array(
            [[hz_to_tone(hz) * 100.0 for hz in f0[0]]], dtype=np.float32
        )
        zeros = np.zeros((1, total_frames), dtype=np.float32)
        ones = np.ones((1, total_frames), dtype=np.float32)
        spk_embed_var = np.tile(self.speaker_embed_variance, (1, total_frames, 1)).astype(np.float32)
        retake = np.ones((1, total_frames, 3), dtype=bool)

        breathiness_pred, voicing_pred, tension_pred = self.variance.run(
            None,
            {
                "encoder_out": encoder_out,
                "ph_dur": ph_dur,
                "pitch": tone_cents,
                "breathiness": zeros,
                "voicing": zeros,
                "tension": zeros,
                "retake": retake,
                "spk_embed": spk_embed_var,
                "steps": np.array(10, dtype=np.int64),
            },
        )

        spk_embed_main = np.tile(self.speaker_embed_main, (1, total_frames, 1)).astype(np.float32)
        (mel,) = self.acoustic.run(
            None,
            {
                "tokens": tokens,
                "languages": languages,
                "durations": ph_dur,
                "f0": f0,
                "breathiness": breathiness_pred,
                "voicing": voicing_pred,
                "tension": tension_pred,
                "gender": zeros,
                "velocity": ones,
                "spk_embed": spk_embed_main,
                "steps": np.array(20, dtype=np.int64),
            },
        )

        (waveform,) = self.vocoder.run(None, {"mel": mel, "f0": f0})
        return waveform[0].astype(np.float32), SAMPLE_RATE


def locate_voicebank(cli_arg: str | None) -> Path:
    if cli_arg:
        path = Path(cli_arg)
    elif os.environ.get("SYMTHAEA_DIFFSINGER_VOICEBANK"):
        path = Path(os.environ["SYMTHAEA_DIFFSINGER_VOICEBANK"])
    else:
        path = Path.home() / ".cache" / "symthaea" / "diffsinger-voicebank"
    if not (path / "dsmain" / "acoustic.onnx").exists():
        raise SystemExit(
            f"No DiffSinger voicebank found at {path} (dsmain/acoustic.onnx missing). "
            "This worker never downloads a voicebank itself -- see "
            "diffsinger_engine module docs / --voicebank / SYMTHAEA_DIFFSINGER_VOICEBANK."
        )
    return path


def main():
    voicebank_arg = None
    if len(sys.argv) > 1 and sys.argv[1] == "--voicebank":
        voicebank_arg = sys.argv[2]
    root = locate_voicebank(voicebank_arg)
    model = DiffSingerModel(root)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("request_id")
            if request.get("protocol_version") != PROTOCOL_VERSION:
                raise ValueError(f"unsupported protocol_version {request.get('protocol_version')}")
            if request.get("operation") != "render":
                raise ValueError(f"unsupported operation {request.get('operation')!r}")
            samples, sample_rate = model.synthesize(request["performance"])
            response = {
                "protocol_version": PROTOCOL_VERSION,
                "provider_id": PROVIDER_ID,
                "request_id": request_id,
                "result": {"samples": samples.tolist(), "sample_rate": sample_rate},
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 -- must always answer, never crash the pipe
            response = {
                "protocol_version": PROTOCOL_VERSION,
                "provider_id": PROVIDER_ID,
                "request_id": request_id,
                "result": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        sys.stdout.write(json.dumps(response))
        sys.stdout.write("\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
