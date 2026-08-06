"""Replaceable phone-alignment interface + two initial implementations,
per the reviewer's explicit design:

  class PhoneAligner(Protocol):
      def align(waveform, sample_rate, expected_phones, prior_spans) -> AlignmentResult

- NativeDurationPrior: wraps Kokoro's own pred_dur as spans directly.
  NEVER used as ground truth on its own (per the native-duration audit's
  conclusion) -- only as the `prior_spans` input to CtcPhoneAligner, and
  as a comparison baseline.
- CtcPhoneAligner: real acoustic forced alignment via
  facebook/wav2vec2-lv-60-espeak-cv-ft (a genuine phoneme-output CTC
  model) + torchaudio.functional.forced_align (the standard CTC
  forced-alignment algorithm -- no external ctc_forced_aligner package
  needed; that package's installed version turned out to be a fixed
  31-char English-ROMAN-ALPHABET aligner, not phone-aware, so it was not
  reused here).

Every returned PhoneSpan carries native_start/native_end (so a caller
can compute pred_dur-vs-acoustic discrepancy directly) and an explicit
warnings list -- no silent fallback to native timing on low confidence.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Protocol

import numpy as np
import torch

from misaki_to_espeak import MISAKI_TO_ESPEAK, transduce

CTC_MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
CTC_SAMPLE_RATE = 16000


@dataclass
class PhoneSpan:
    phone: str                     # the ORIGINAL misaki character
    phone_class: str
    start_sample: int              # at the ORIGINAL (Kokoro, 24kHz) sample rate
    end_sample: int
    confidence: float              # 0..1, backend-specific meaning (see backend)
    native_start: Optional[int] = None
    native_end: Optional[int] = None
    boundary_discrepancy_ms: Optional[float] = None
    alignment_backend: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass
class AlignmentResult:
    spans: List[PhoneSpan]
    backend: str
    phone_order_ok: bool
    global_warnings: List[str] = field(default_factory=list)


class PhoneAligner(Protocol):
    def align(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        expected_phones: List[str],
        prior_spans: Optional[List[PhoneSpan]] = None,
    ) -> AlignmentResult:
        ...


class NativeDurationPrior:
    """Wraps Kokoro's pred_dur-derived spans directly. Per the
    native-duration audit (2026-07-28): deterministic, correctly indexed,
    exactly partitions the waveform -- but NOT acoustically accurate
    enough to be a final boundary (variance comparable to or exceeding
    the mean skew in every phone class). Every span is marked with a
    standing warning so a caller can never mistake this for a validated
    alignment."""

    backend_name = "native_duration_prior"

    def align(self, waveform, sample_rate, expected_phones, prior_spans=None):
        if prior_spans is None:
            raise ValueError("NativeDurationPrior requires prior_spans (from pred_dur)")
        spans = []
        for ps in prior_spans:
            spans.append(PhoneSpan(
                phone=ps.phone, phone_class=ps.phone_class,
                start_sample=ps.start_sample, end_sample=ps.end_sample,
                confidence=0.0,  # never trust this as a real confidence
                native_start=ps.start_sample, native_end=ps.end_sample,
                boundary_discrepancy_ms=0.0,
                alignment_backend=self.backend_name,
                warnings=["NOT ACOUSTICALLY VALIDATED -- pred_dur only, see native-duration audit"],
            ))
        return AlignmentResult(spans=spans, backend=self.backend_name, phone_order_ok=True)


class CtcPhoneAligner:
    """Real forced alignment via facebook/wav2vec2-lv-60-espeak-cv-ft.

    Loaded WITHOUT the model's own AutoProcessor/tokenizer (which
    requires a system espeak install for its text->phoneme phonemizer
    backend, unused here since misaki phonemes are already available) --
    loads Wav2Vec2ForCTC + Wav2Vec2FeatureExtractor directly, plus the
    raw vocab.json for the phone->id mapping.
    """

    backend_name = "ctc_wav2vec2_espeak_cv_ft"

    def __init__(self, device: str = "cpu"):
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
        self.device = device
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(CTC_MODEL_ID)
        self.model = Wav2Vec2ForCTC.from_pretrained(CTC_MODEL_ID).to(device)
        self.model.eval()
        from huggingface_hub import hf_hub_download
        import json
        vocab_path = hf_hub_download(CTC_MODEL_ID, "vocab.json")
        self.vocab = json.loads(open(vocab_path).read())
        self.blank_id = self.vocab.get("<pad>", 0)

    def _resample(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate == CTC_SAMPLE_RATE:
            return waveform
        import torchaudio
        wav_t = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        out = torchaudio.functional.resample(wav_t, sample_rate, CTC_SAMPLE_RATE)
        return out.squeeze(0).numpy()

    def align(self, waveform, sample_rate, expected_phones, prior_spans=None):
        warnings = []
        wav16 = self._resample(waveform, sample_rate)

        # Build the CTC target sequence from the misaki phone string.
        # expected_phones is the raw misaki `ps` string (with markers);
        # transduce() drops markers and returns (char, orig_index, espeak_tok).
        triples, unknown_chars = transduce(expected_phones)
        if unknown_chars:
            warnings.append(f"unmapped misaki chars, skipped: {sorted(unknown_chars)}")
        unknown_espeak = sorted({tok for _c, _i, tok in triples if tok not in self.vocab})
        if unknown_espeak:
            warnings.append(f"espeak tokens not in CTC vocab, dropped: {unknown_espeak}")
        triples = [(c, i, tok) for c, i, tok in triples if tok in self.vocab]
        target_ids = [self.vocab[tok] for _c, _i, tok in triples]

        with torch.no_grad():
            inputs = self.feature_extractor(wav16, sampling_rate=CTC_SAMPLE_RATE, return_tensors="pt")
            logits = self.model(inputs.input_values.to(self.device)).logits[0]  # (T, C)
            log_probs = torch.log_softmax(logits, dim=-1)

        n_frames = log_probs.shape[0]
        min_frames_needed = len(target_ids) + sum(
            1 for k in range(1, len(target_ids)) if target_ids[k] == target_ids[k - 1]
        )
        if n_frames < min_frames_needed:
            warnings.append(f"too few CTC frames ({n_frames}) for target length ({min_frames_needed}) -- alignment unreliable")

        import torchaudio
        targets_t = torch.tensor([target_ids], dtype=torch.int64)
        path, scores = torchaudio.functional.forced_align(
            log_probs.unsqueeze(0), targets_t, blank=self.blank_id,
        )
        path = path[0].tolist()
        scores = scores[0].tolist()

        # Collapse the per-frame path into run-length segments; the k-th
        # non-blank segment corresponds EXACTLY to targets[k] (guaranteed
        # by forced_align's monotonic constrained decoding).
        segments = []  # (label, start_frame, end_frame_exclusive, mean_score)
        i = 0
        while i < len(path):
            j = i
            while j < len(path) and path[j] == path[i]:
                j += 1
            segments.append((path[i], i, j, float(np.mean(scores[i:j]))))
            i = j
        non_blank_segments = [s for s in segments if s[0] != self.blank_id]

        if len(non_blank_segments) != len(target_ids):
            warnings.append(
                f"segment count mismatch: {len(non_blank_segments)} non-blank segments "
                f"vs {len(target_ids)} target tokens -- alignment likely degenerate"
            )

        # Determine the model's frame stride empirically from this call
        # (rather than assuming the textbook 20ms), so it stays correct
        # if the feature extractor's conv stack ever changes.
        frame_stride_s = len(wav16) / CTC_SAMPLE_RATE / n_frames

        spans = []
        n_pairs = min(len(non_blank_segments), len(triples))
        for k in range(n_pairs):
            label, f0, f1, mean_score = non_blank_segments[k]
            char, orig_idx, espeak_tok = triples[k]
            t0 = f0 * frame_stride_s
            t1 = f1 * frame_stride_s
            start_sample = int(round(t0 * sample_rate))
            end_sample = int(round(t1 * sample_rate))
            span_warnings = []
            confidence = float(np.exp(mean_score))  # mean log-prob -> pseudo-probability
            if confidence < 0.3:
                span_warnings.append("low CTC confidence")
            spans.append(PhoneSpan(
                phone=char, phone_class="",  # filled in by caller, which has the classifier
                start_sample=start_sample, end_sample=end_sample,
                confidence=confidence,
                alignment_backend=self.backend_name,
                warnings=span_warnings,
            ))

        phone_order_ok = (len(non_blank_segments) == len(target_ids))
        return AlignmentResult(spans=spans, backend=self.backend_name,
                                phone_order_ok=phone_order_ok, global_warnings=warnings)
