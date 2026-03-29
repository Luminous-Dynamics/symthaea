#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Proof of Consciousness (PoC) - Unified Consciousness State Scoring

Integrates three validated Sentinels into a combined consciousness score:
- EmotionSentinel (Proof of Joy): Emotional valence and arousal
- SleepSentinel (Proof of Rest): Sleep stage and quality
- MeditationSentinel (Proof of Focus): Meditation depth and stability

The PoC score provides a holistic measure of consciousness state.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List, Tuple
from scipy import signal


class ConsciousnessState(Enum):
    """High-level consciousness states."""
    DEEP_SLEEP = "deep_sleep"          # N3 sleep
    LIGHT_SLEEP = "light_sleep"        # N1, N2 sleep
    REM = "rem"                        # REM sleep
    DROWSY = "drowsy"                  # Transition state
    RELAXED = "relaxed"                # Calm wakefulness
    FOCUSED = "focused"                # Active concentration
    MEDITATIVE = "meditative"          # Deep meditation
    FLOW = "flow"                      # Optimal performance
    STRESSED = "stressed"              # High arousal negative


@dataclass
class EmotionScore:
    """Emotion detection results."""
    valence: float       # -1 (negative) to +1 (positive)
    arousal: float       # 0 (calm) to 1 (excited)
    confidence: float    # 0 to 1

    @property
    def quadrant(self) -> str:
        """Return emotion quadrant (Russell's circumplex)."""
        if self.valence >= 0 and self.arousal >= 0.5:
            return "excited_positive"  # Happy, excited
        elif self.valence >= 0 and self.arousal < 0.5:
            return "calm_positive"     # Relaxed, content
        elif self.valence < 0 and self.arousal >= 0.5:
            return "excited_negative"  # Angry, anxious
        else:
            return "calm_negative"     # Sad, bored


@dataclass
class SleepScore:
    """Sleep stage detection results."""
    stage: int           # 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM
    confidence: float    # 0 to 1
    delta_power: float   # Relative delta power
    sleep_quality: float # Composite quality score 0-1

    @property
    def stage_name(self) -> str:
        names = ["Wake", "N1", "N2", "N3", "REM"]
        return names[self.stage] if 0 <= self.stage <= 4 else "Unknown"


@dataclass
class MeditationScore:
    """Meditation state detection results."""
    depth: float         # 0 (surface) to 1 (deep)
    stability: float     # 0 (fluctuating) to 1 (stable)
    alpha_power: float   # Relative alpha power
    theta_power: float   # Relative theta power
    meditation_index: float  # (alpha + theta) / beta

    @property
    def state(self) -> str:
        if self.depth > 0.7:
            return "deep"
        elif self.depth > 0.4:
            return "moderate"
        else:
            return "light"


@dataclass
class ProofOfConsciousness:
    """Combined consciousness proof."""
    emotion: Optional[EmotionScore]
    sleep: Optional[SleepScore]
    meditation: Optional[MeditationScore]
    timestamp: float

    # Derived scores
    overall_state: ConsciousnessState
    consciousness_level: float  # 0 (deep sleep) to 1 (peak awareness)
    wellbeing_score: float      # 0 (poor) to 1 (optimal)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "state": self.overall_state.value,
            "consciousness_level": self.consciousness_level,
            "wellbeing_score": self.wellbeing_score,
            "emotion": {
                "valence": self.emotion.valence if self.emotion else None,
                "arousal": self.emotion.arousal if self.emotion else None,
            },
            "sleep": {
                "stage": self.sleep.stage_name if self.sleep else None,
                "quality": self.sleep.sleep_quality if self.sleep else None,
            },
            "meditation": {
                "depth": self.meditation.depth if self.meditation else None,
                "index": self.meditation.meditation_index if self.meditation else None,
            },
        }


def band_power(data: np.ndarray, srate: float, fmin: float, fmax: float) -> float:
    """Compute band power using Welch's method."""
    nperseg = min(int(4 * srate), len(data))
    if nperseg < 8:
        return 0.0
    f, psd = signal.welch(data, srate, nperseg=nperseg)
    idx = (f >= fmin) & (f <= fmax)
    return np.trapz(psd[idx], f[idx]) if idx.sum() > 1 else 0


class EmotionSentinel:
    """Proof of Joy - Emotion detection from EEG."""

    def __init__(self):
        self.baseline_asymmetry = 0.0
        self.calibrated = False

    def calibrate(self, baseline_data: np.ndarray, srate: float,
                  left_channels: List[int], right_channels: List[int]):
        """Calibrate with resting baseline."""
        self.baseline_asymmetry = self._compute_asymmetry(
            baseline_data, srate, left_channels, right_channels
        )
        self.calibrated = True

    def _compute_asymmetry(self, data: np.ndarray, srate: float,
                           left_channels: List[int], right_channels: List[int]) -> float:
        """Compute frontal alpha asymmetry."""
        left_alpha = np.mean([
            band_power(data[ch], srate, 8, 13) for ch in left_channels
        ])
        right_alpha = np.mean([
            band_power(data[ch], srate, 8, 13) for ch in right_channels
        ])
        # Log transform for asymmetry index
        return np.log(right_alpha + 1e-10) - np.log(left_alpha + 1e-10)

    def analyze(self, data: np.ndarray, srate: float,
                left_channels: List[int] = [0],
                right_channels: List[int] = [1]) -> EmotionScore:
        """Analyze emotional state from EEG."""
        asymmetry = self._compute_asymmetry(data, srate, left_channels, right_channels)

        # Valence from frontal asymmetry (positive = right > left = positive emotion)
        valence = np.tanh(asymmetry - self.baseline_asymmetry)

        # Arousal from beta/alpha ratio
        alpha = np.mean([band_power(data[ch], srate, 8, 13) for ch in range(min(2, len(data)))])
        beta = np.mean([band_power(data[ch], srate, 13, 30) for ch in range(min(2, len(data)))])
        theta = np.mean([band_power(data[ch], srate, 4, 8) for ch in range(min(2, len(data)))])

        arousal_raw = beta / (alpha + theta + 1e-10)
        arousal = min(1.0, arousal_raw / 2.0)  # Normalize to 0-1

        # Confidence based on signal quality
        total_power = alpha + beta + theta
        confidence = min(1.0, total_power / 1e-8) if total_power > 0 else 0.5

        return EmotionScore(
            valence=float(valence),
            arousal=float(arousal),
            confidence=float(confidence)
        )


class SleepSentinel:
    """Proof of Rest - Sleep stage detection from EEG/PSG."""

    def analyze(self, data: np.ndarray, srate: float) -> SleepScore:
        """Analyze sleep stage from single-channel EEG."""
        # Extract band powers
        delta = band_power(data, srate, 0.5, 4)
        theta = band_power(data, srate, 4, 8)
        alpha = band_power(data, srate, 8, 13)
        beta = band_power(data, srate, 13, 30)

        total = delta + theta + alpha + beta + 1e-10

        delta_rel = delta / total
        theta_rel = theta / total
        beta_rel = beta / total

        # Classification based on validated thresholds
        if delta_rel > 0.88 and theta_rel < 0.08:
            stage = 3  # N3 (Deep sleep)
            confidence = min(1.0, delta_rel)
        elif theta_rel > 0.16:
            if beta_rel > 0.045:
                stage = 1  # N1
            else:
                stage = 4  # REM
            confidence = min(1.0, theta_rel * 2)
        elif beta_rel > 0.028:
            stage = 0  # Wake
            confidence = min(1.0, beta_rel * 10)
        else:
            stage = 2  # N2
            confidence = 0.6

        # Sleep quality (higher delta in N3 = better)
        sleep_quality = delta_rel if stage == 3 else 0.5

        return SleepScore(
            stage=stage,
            confidence=confidence,
            delta_power=delta_rel,
            sleep_quality=sleep_quality
        )


class MeditationSentinel:
    """Proof of Focus - Meditation detection from EEG."""

    def __init__(self):
        self.baseline_index = 1.0
        self.calibrated = False

    def calibrate(self, baseline_data: np.ndarray, srate: float):
        """Calibrate with resting baseline."""
        features = self._extract_features(baseline_data, srate)
        self.baseline_index = features['meditation_index']
        self.calibrated = True

    def _extract_features(self, data: np.ndarray, srate: float) -> Dict:
        """Extract meditation features."""
        theta = band_power(data, srate, 4, 8)
        alpha = band_power(data, srate, 8, 13)
        beta = band_power(data, srate, 13, 30)
        gamma = band_power(data, srate, 30, 45)

        total = theta + alpha + beta + gamma + 1e-10

        return {
            'theta_rel': theta / total,
            'alpha_rel': alpha / total,
            'beta_rel': beta / total,
            'meditation_index': (alpha + theta) / (beta + 1e-10),
        }

    def analyze(self, data: np.ndarray, srate: float) -> MeditationScore:
        """Analyze meditation state from EEG."""
        features = self._extract_features(data, srate)

        # Depth based on meditation index relative to baseline
        index_ratio = features['meditation_index'] / (self.baseline_index + 1e-10)
        depth = np.tanh((index_ratio - 1) * 0.5) * 0.5 + 0.5  # 0-1 scale

        # Stability would require multiple epochs - use alpha consistency
        stability = min(1.0, features['alpha_rel'] * 2)

        return MeditationScore(
            depth=float(depth),
            stability=float(stability),
            alpha_power=float(features['alpha_rel']),
            theta_power=float(features['theta_rel']),
            meditation_index=float(features['meditation_index'])
        )


class ConsciousnessEngine:
    """Unified Proof of Consciousness engine."""

    def __init__(self):
        self.emotion = EmotionSentinel()
        self.sleep = SleepSentinel()
        self.meditation = MeditationSentinel()

    def analyze(self, data: np.ndarray, srate: float,
                mode: str = "auto",
                left_channels: List[int] = [0],
                right_channels: List[int] = [1]) -> ProofOfConsciousness:
        """
        Analyze consciousness state from EEG data.

        Args:
            data: EEG data (channels x samples) or (samples,) for single channel
            srate: Sampling rate in Hz
            mode: "auto", "emotion", "sleep", or "meditation"
            left_channels: Indices for left hemisphere
            right_channels: Indices for right hemisphere

        Returns:
            ProofOfConsciousness with all scores
        """
        import time
        timestamp = time.time()

        # Ensure 2D array
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Get individual scores
        emotion_score = None
        sleep_score = None
        meditation_score = None

        if mode in ["auto", "emotion"]:
            try:
                emotion_score = self.emotion.analyze(data, srate, left_channels, right_channels)
            except Exception:
                pass

        if mode in ["auto", "sleep"]:
            try:
                sleep_score = self.sleep.analyze(data[0], srate)
            except Exception:
                pass

        if mode in ["auto", "meditation"]:
            try:
                meditation_score = self.meditation.analyze(data[0], srate)
            except Exception:
                pass

        # Determine overall state
        state, consciousness_level, wellbeing = self._compute_overall(
            emotion_score, sleep_score, meditation_score
        )

        return ProofOfConsciousness(
            emotion=emotion_score,
            sleep=sleep_score,
            meditation=meditation_score,
            timestamp=timestamp,
            overall_state=state,
            consciousness_level=consciousness_level,
            wellbeing_score=wellbeing
        )

    def _compute_overall(self, emotion: Optional[EmotionScore],
                         sleep: Optional[SleepScore],
                         meditation: Optional[MeditationScore]
                         ) -> Tuple[ConsciousnessState, float, float]:
        """Compute overall consciousness state and scores."""

        # Default values
        consciousness_level = 0.5
        wellbeing = 0.5
        state = ConsciousnessState.RELAXED

        # Sleep state takes priority (if detected)
        if sleep and sleep.stage > 0:
            if sleep.stage == 3:
                state = ConsciousnessState.DEEP_SLEEP
                consciousness_level = 0.1
                wellbeing = 0.6 + 0.4 * sleep.sleep_quality
            elif sleep.stage == 4:
                state = ConsciousnessState.REM
                consciousness_level = 0.4
                wellbeing = 0.5
            else:  # N1, N2
                state = ConsciousnessState.LIGHT_SLEEP
                consciousness_level = 0.3
                wellbeing = 0.4 + 0.2 * sleep.sleep_quality
            return state, consciousness_level, wellbeing

        # Awake states
        consciousness_level = 0.7  # Base awake level

        # Meditation influence
        if meditation:
            if meditation.depth > 0.6:
                state = ConsciousnessState.MEDITATIVE
                consciousness_level = 0.8 + 0.2 * meditation.depth
                wellbeing = 0.7 + 0.3 * meditation.stability
            elif meditation.depth > 0.4:
                state = ConsciousnessState.FOCUSED
                consciousness_level = 0.75
                wellbeing = 0.65

        # Emotion influence
        if emotion:
            # Positive valence increases wellbeing
            wellbeing = 0.5 + 0.3 * emotion.valence + 0.2 * (1 - emotion.arousal)
            wellbeing = max(0, min(1, wellbeing))

            # High arousal + negative valence = stressed
            if emotion.arousal > 0.6 and emotion.valence < -0.3:
                state = ConsciousnessState.STRESSED
                consciousness_level = 0.9  # High but not optimal
                wellbeing = 0.3

            # Low arousal = drowsy or relaxed
            if emotion.arousal < 0.3:
                if state != ConsciousnessState.MEDITATIVE:
                    state = ConsciousnessState.RELAXED
                consciousness_level = 0.6

        # Flow state: high consciousness + high wellbeing + focused/meditative
        if consciousness_level > 0.8 and wellbeing > 0.7:
            if state in [ConsciousnessState.FOCUSED, ConsciousnessState.MEDITATIVE]:
                state = ConsciousnessState.FLOW

        return state, consciousness_level, wellbeing


# Convenience function
def analyze_consciousness(data: np.ndarray, srate: float,
                          mode: str = "auto") -> ProofOfConsciousness:
    """
    Analyze consciousness state from EEG data.

    This is the main entry point for Proof of Consciousness analysis.

    Args:
        data: EEG data array
        srate: Sampling rate in Hz
        mode: "auto" (all), "emotion", "sleep", or "meditation"

    Returns:
        ProofOfConsciousness object with all scores and overall state

    Example:
        >>> data = np.random.randn(256 * 30)  # 30 seconds at 256 Hz
        >>> poc = analyze_consciousness(data, 256)
        >>> print(f"State: {poc.overall_state.value}")
        >>> print(f"Consciousness: {poc.consciousness_level:.2f}")
        >>> print(f"Wellbeing: {poc.wellbeing_score:.2f}")
    """
    engine = ConsciousnessEngine()
    return engine.analyze(data, srate, mode=mode)


if __name__ == "__main__":
    # Demo with synthetic data
    print("=" * 60)
    print("Proof of Consciousness Demo".center(60))
    print("=" * 60)
    print()

    # Generate synthetic EEG-like data
    srate = 256
    duration = 30
    t = np.linspace(0, duration, srate * duration)

    # Mix of alpha, theta, and beta
    alpha = 0.5 * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
    theta = 0.3 * np.sin(2 * np.pi * 6 * t)   # 6 Hz theta
    beta = 0.2 * np.sin(2 * np.pi * 20 * t)   # 20 Hz beta
    noise = 0.1 * np.random.randn(len(t))

    data = alpha + theta + beta + noise

    # Analyze
    poc = analyze_consciousness(data, srate)

    print(f"Overall State: {poc.overall_state.value}")
    print(f"Consciousness Level: {poc.consciousness_level:.2f}")
    print(f"Wellbeing Score: {poc.wellbeing_score:.2f}")
    print()

    if poc.emotion:
        print("Emotion:")
        print(f"  Valence: {poc.emotion.valence:+.2f}")
        print(f"  Arousal: {poc.emotion.arousal:.2f}")
        print(f"  Quadrant: {poc.emotion.quadrant}")

    if poc.sleep:
        print(f"Sleep Stage: {poc.sleep.stage_name}")
        print(f"  Delta Power: {poc.sleep.delta_power*100:.1f}%")

    if poc.meditation:
        print("Meditation:")
        print(f"  Depth: {poc.meditation.depth:.2f} ({poc.meditation.state})")
        print(f"  Index: {poc.meditation.meditation_index:.2f}")
