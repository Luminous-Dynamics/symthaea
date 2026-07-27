#!/usr/bin/env python3
"""
Diagnostic for the Step 1 negative result: confirms the CTC-blank-dominance
finding is a property of the ACOUSTIC MODEL's emissions (wav2vec2-espeak was
trained on spoken, not sung, audio), not an artifact of the generic Viterbi
forced_align() routine. Prints each target phone's best raw log-probability
frames across a sustained-note slice -- if the model itself never assigns
meaningful probability to the phone anywhere in the slice, the alignment
algorithm cannot do better than a 1-frame spike no matter how it's applied.

Usage: python3 probe_blank_dominance.py <wav_path> <t0> <t1> <token1> [<token2> ...]
"""
import json
import sys

import numpy as np
import torch
import torchaudio
import wave
from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"


def main():
    wav_path, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    tokens = sys.argv[4:] or ["oʊ", "ɹ", "<pad>"]

    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        wf.setpos(int(t0 * sr))
        raw = wf.readframes(int((t1 - t0) * sr))
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        pcm = pcm.reshape(-1, n_channels).mean(axis=1)
    wav = torch.from_numpy(pcm).unsqueeze(0)
    wav = torchaudio.functional.resample(wav, sr, 16000)

    vocab = json.load(open(hf_hub_download(MODEL_ID, "vocab.json")))
    fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    model.eval()
    with torch.no_grad():
        inputs = fe(wav.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt")
        logp = torch.log_softmax(model(inputs.input_values).logits, dim=-1).squeeze(0).numpy()

    frame_dur = (t1 - t0) / logp.shape[0]
    for tok in tokens:
        idx = vocab[tok]
        series = logp[:, idx]
        top = np.argsort(-series)[:8]
        pairs = sorted([(round(t0 + f * frame_dur, 3), round(float(series[f]), 2)) for f in top])
        print(f"{tok}: top-8 frames (time, logprob): {pairs}")


if __name__ == "__main__":
    main()
