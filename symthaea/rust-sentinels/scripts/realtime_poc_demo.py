#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Real-Time Proof of Consciousness Demo

Simulates live EEG streaming with real-time consciousness state visualization.
Demonstrates the complete Consciousness Trilogy (Emotion, Sleep, Meditation)
plus the Extended Proofs (Attention, Flow, Engagement).

Usage:
    python scripts/realtime_poc_demo.py [--duration SECONDS] [--scenario SCENARIO]

Scenarios:
    - meditation: Transition from alert to deep meditation
    - sleep: Progression through sleep stages
    - focus: Attention and flow state variations
    - mixed: Natural consciousness fluctuations
"""

import sys
import time
import argparse
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum, auto

# Terminal colors for visualization
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Consciousness states
    AWAKE = "\033[38;5;226m"      # Yellow
    RELAXED = "\033[38;5;120m"    # Light green
    FOCUSED = "\033[38;5;39m"     # Blue
    MEDITATIVE = "\033[38;5;141m" # Purple
    DROWSY = "\033[38;5;208m"     # Orange
    SLEEP = "\033[38;5;27m"       # Deep blue
    FLOW = "\033[38;5;199m"       # Magenta

    # Emotions
    JOY = "\033[38;5;220m"        # Gold
    CALM = "\033[38;5;87m"        # Cyan
    STRESS = "\033[38;5;196m"     # Red

    # Bar colors
    BAR_HIGH = "\033[38;5;46m"    # Bright green
    BAR_MED = "\033[38;5;226m"    # Yellow
    BAR_LOW = "\033[38;5;196m"    # Red


class ConsciousnessState(Enum):
    DEEP_SLEEP = auto()
    LIGHT_SLEEP = auto()
    REM = auto()
    DROWSY = auto()
    RELAXED = auto()
    ALERT = auto()
    FOCUSED = auto()
    MEDITATIVE = auto()
    FLOW = auto()
    STRESSED = auto()


class EmotionQuadrant(Enum):
    HAPPY_CALM = "Happy-Calm"
    HAPPY_EXCITED = "Happy-Excited"
    SAD_CALM = "Sad-Calm"
    SAD_EXCITED = "Anxious"


@dataclass
class EmotionScore:
    valence: float      # -1 (negative) to +1 (positive)
    arousal: float      # 0 (calm) to 1 (excited)

    @property
    def quadrant(self) -> EmotionQuadrant:
        if self.valence >= 0:
            return EmotionQuadrant.HAPPY_EXCITED if self.arousal > 0.5 else EmotionQuadrant.HAPPY_CALM
        else:
            return EmotionQuadrant.SAD_EXCITED if self.arousal > 0.5 else EmotionQuadrant.SAD_CALM


@dataclass
class SleepScore:
    stage: str          # W, N1, N2, N3, REM
    delta_power: float  # 0-1
    theta_power: float  # 0-1
    confidence: float   # 0-1


@dataclass
class MeditationScore:
    depth: float            # 0-1
    stability: float        # 0-1
    meditation_index: float # typically 1-10
    state: str              # Light, Moderate, Deep


@dataclass
class AttentionScore:
    sustained: float        # 0-1
    selective: float        # 0-1
    alertness: float        # 0-1
    attention_index: float  # 0-3


@dataclass
class FlowScore:
    immersion: float        # 0-1
    effortlessness: float   # 0-1
    flow_index: float       # 0-1


@dataclass
class EngagementScore:
    cognitive: float        # 0-1
    emotional: float        # 0-1
    engagement_index: float # 0-3


@dataclass
class ProofOfConsciousness:
    """Complete consciousness state at a moment in time."""
    timestamp: float

    # Core state
    state: ConsciousnessState
    consciousness_level: float  # 0-1
    wellbeing_score: float      # 0-1

    # Trilogy
    emotion: EmotionScore
    sleep: SleepScore
    meditation: MeditationScore

    # Extended
    attention: AttentionScore
    flow: FlowScore
    engagement: EngagementScore

    # Band powers
    delta: float
    theta: float
    alpha: float
    beta: float
    gamma: float


class SignalGenerator:
    """Generates synthetic EEG-like signals for different consciousness states."""

    def __init__(self, sample_rate: float = 256.0):
        self.sample_rate = sample_rate
        self.t = 0.0

    def generate_epoch(self, duration: float, state_params: dict) -> List[float]:
        """Generate an epoch of EEG data with specified characteristics."""
        n_samples = int(self.sample_rate * duration)
        signal = []

        for i in range(n_samples):
            t = self.t + i / self.sample_rate
            sample = self._generate_sample(t, state_params)
            signal.append(sample)

        self.t += duration
        return signal

    def _generate_sample(self, t: float, params: dict) -> float:
        """Generate a single sample with mixed frequencies."""
        pi2 = 2 * math.pi

        # Band weights from params
        delta_w = params.get('delta', 0.1)
        theta_w = params.get('theta', 0.2)
        alpha_w = params.get('alpha', 0.3)
        beta_w = params.get('beta', 0.2)
        gamma_w = params.get('gamma', 0.1)
        noise_w = params.get('noise', 0.1)

        # Generate signal components
        delta = delta_w * math.sin(pi2 * 2.0 * t)    # 2 Hz
        theta = theta_w * math.sin(pi2 * 6.0 * t)    # 6 Hz
        alpha = alpha_w * math.sin(pi2 * 10.0 * t)   # 10 Hz
        beta = beta_w * math.sin(pi2 * 20.0 * t)     # 20 Hz
        gamma = gamma_w * math.sin(pi2 * 40.0 * t)   # 40 Hz

        # Add noise
        noise = noise_w * (2 * (hash(str(t)) % 1000) / 1000 - 1)

        return delta + theta + alpha + beta + gamma + noise


class ConsciousnessAnalyzer:
    """Analyzes EEG signals to produce Proof of Consciousness."""

    def __init__(self):
        self.baseline_alpha = 0.3
        self.baseline_theta = 0.2

    def analyze(self, signal: List[float], sample_rate: float,
                timestamp: float) -> ProofOfConsciousness:
        """Analyze signal and return consciousness state."""

        # Extract band powers (simplified for demo)
        powers = self._extract_powers(signal, sample_rate)

        # Compute scores
        emotion = self._analyze_emotion(powers)
        sleep = self._analyze_sleep(powers)
        meditation = self._analyze_meditation(powers)
        attention = self._analyze_attention(powers)
        flow = self._analyze_flow(powers)
        engagement = self._analyze_engagement(powers)

        # Determine overall state
        state = self._classify_state(powers, meditation, attention, flow)

        # Compute composite scores
        consciousness_level = self._compute_consciousness_level(state, powers)
        wellbeing_score = self._compute_wellbeing(emotion, meditation, flow)

        return ProofOfConsciousness(
            timestamp=timestamp,
            state=state,
            consciousness_level=consciousness_level,
            wellbeing_score=wellbeing_score,
            emotion=emotion,
            sleep=sleep,
            meditation=meditation,
            attention=attention,
            flow=flow,
            engagement=engagement,
            delta=powers['delta'],
            theta=powers['theta'],
            alpha=powers['alpha'],
            beta=powers['beta'],
            gamma=powers['gamma'],
        )

    def _extract_powers(self, signal: List[float], sample_rate: float) -> dict:
        """Extract band powers using simple RMS approximation."""
        n = len(signal)
        if n == 0:
            return {'delta': 0, 'theta': 0, 'alpha': 0, 'beta': 0, 'gamma': 0}

        # Simple power estimate (RMS)
        rms = math.sqrt(sum(x*x for x in signal) / n)

        # In real implementation, we'd use Welch's method
        # For demo, we derive from signal characteristics
        total = rms + 0.001

        # Estimate from signal variance at different frequencies
        # (simplified - real implementation uses FFT)
        delta = max(0, rms * 0.3 + 0.1 * signal[0] if signal else 0)
        theta = max(0, rms * 0.25)
        alpha = max(0, rms * 0.3)
        beta = max(0, rms * 0.1)
        gamma = max(0, rms * 0.05)

        # Normalize
        total_power = delta + theta + alpha + beta + gamma + 0.001

        return {
            'delta': delta / total_power,
            'theta': theta / total_power,
            'alpha': alpha / total_power,
            'beta': beta / total_power,
            'gamma': gamma / total_power,
        }

    def _analyze_emotion(self, powers: dict) -> EmotionScore:
        """Analyze emotional state from band powers."""
        # Frontal alpha asymmetry approximation
        alpha = powers['alpha']
        beta = powers['beta']

        # Higher alpha = more positive valence
        valence = (alpha - 0.3) * 3  # Scale to [-1, 1]
        valence = max(-1, min(1, valence))

        # Higher beta = higher arousal
        arousal = beta * 3
        arousal = max(0, min(1, arousal))

        return EmotionScore(valence=valence, arousal=arousal)

    def _analyze_sleep(self, powers: dict) -> SleepScore:
        """Analyze sleep stage from band powers."""
        delta = powers['delta']
        theta = powers['theta']
        beta = powers['beta']

        # Sleep stage classification
        if delta > 0.5:
            stage = "N3"
            confidence = min(1.0, delta * 1.5)
        elif theta > 0.3 and beta < 0.1:
            stage = "N2"
            confidence = 0.7
        elif theta > 0.25:
            stage = "N1" if beta > 0.05 else "REM"
            confidence = 0.6
        else:
            stage = "W"
            confidence = 0.8

        return SleepScore(
            stage=stage,
            delta_power=delta,
            theta_power=theta,
            confidence=confidence
        )

    def _analyze_meditation(self, powers: dict) -> MeditationScore:
        """Analyze meditation state from band powers."""
        alpha = powers['alpha']
        theta = powers['theta']
        beta = powers['beta']

        # Meditation index based on alpha/theta dominance
        meditation_index = (alpha + theta * 0.5) / (beta + 0.1) * 2
        meditation_index = max(0, min(10, meditation_index))

        # Depth from alpha power
        depth = min(1.0, alpha * 2)

        # Stability (inverse of beta)
        stability = 1.0 - min(1.0, beta * 3)

        # State classification
        if depth > 0.7:
            state = "Deep"
        elif depth > 0.4:
            state = "Moderate"
        else:
            state = "Light"

        return MeditationScore(
            depth=depth,
            stability=stability,
            meditation_index=meditation_index,
            state=state
        )

    def _analyze_attention(self, powers: dict) -> AttentionScore:
        """Analyze attention from band powers."""
        theta = powers['theta']
        alpha = powers['alpha']
        beta = powers['beta']

        # Frontal theta indicates sustained attention
        sustained = min(1.0, theta * 2.5)

        # Alpha suppression indicates selective attention
        selective = 1.0 - min(1.0, alpha * 2)

        # Beta indicates alertness
        alertness = min(1.0, beta * 4)

        # Attention index
        attention_index = (sustained + selective + alertness) / 1.5

        return AttentionScore(
            sustained=sustained,
            selective=selective,
            alertness=alertness,
            attention_index=attention_index
        )

    def _analyze_flow(self, powers: dict) -> FlowScore:
        """Analyze flow state from band powers."""
        alpha = powers['alpha']
        beta = powers['beta']
        theta = powers['theta']

        # Flow requires moderate alpha (not too high, not too low)
        alpha_optimal = 1.0 - abs(alpha - 0.35) * 3
        alpha_optimal = max(0, alpha_optimal)

        # Low frontal beta (less self-monitoring)
        effortlessness = 1.0 - min(1.0, beta * 3)

        # Immersion from theta/alpha balance
        immersion = min(1.0, (alpha + theta * 0.5) * 1.5)

        # Flow index
        flow_index = (alpha_optimal * effortlessness * immersion) ** 0.5

        return FlowScore(
            immersion=immersion,
            effortlessness=effortlessness,
            flow_index=flow_index
        )

    def _analyze_engagement(self, powers: dict) -> EngagementScore:
        """Analyze engagement from band powers."""
        alpha = powers['alpha']
        theta = powers['theta']
        beta = powers['beta']
        gamma = powers['gamma']

        # Classic engagement index: beta / (alpha + theta)
        engagement_index = beta / (alpha + theta + 0.1)

        # Cognitive engagement from beta + gamma
        cognitive = min(1.0, (beta + gamma) * 3)

        # Emotional engagement (approximation)
        emotional = min(1.0, gamma * 5)

        return EngagementScore(
            cognitive=cognitive,
            emotional=emotional,
            engagement_index=engagement_index
        )

    def _classify_state(self, powers: dict, meditation: MeditationScore,
                        attention: AttentionScore, flow: FlowScore) -> ConsciousnessState:
        """Classify overall consciousness state."""
        delta = powers['delta']
        theta = powers['theta']
        alpha = powers['alpha']
        beta = powers['beta']

        # Sleep states
        if delta > 0.5:
            return ConsciousnessState.DEEP_SLEEP
        if theta > 0.35 and beta < 0.1:
            return ConsciousnessState.LIGHT_SLEEP
        if theta > 0.3 and alpha > 0.3:
            return ConsciousnessState.REM

        # Drowsy
        if theta > 0.25 and alpha > 0.25 and beta < 0.15:
            return ConsciousnessState.DROWSY

        # Flow state (special)
        if flow.flow_index > 0.6:
            return ConsciousnessState.FLOW

        # Meditation
        if meditation.depth > 0.5 and beta < 0.2:
            return ConsciousnessState.MEDITATIVE

        # Stressed
        if beta > 0.35:
            return ConsciousnessState.STRESSED

        # Focused
        if attention.attention_index > 1.5:
            return ConsciousnessState.FOCUSED

        # Relaxed
        if alpha > 0.3:
            return ConsciousnessState.RELAXED

        return ConsciousnessState.ALERT

    def _compute_consciousness_level(self, state: ConsciousnessState,
                                      powers: dict) -> float:
        """Compute overall consciousness level (0-1)."""
        state_levels = {
            ConsciousnessState.DEEP_SLEEP: 0.1,
            ConsciousnessState.LIGHT_SLEEP: 0.2,
            ConsciousnessState.REM: 0.3,
            ConsciousnessState.DROWSY: 0.4,
            ConsciousnessState.RELAXED: 0.6,
            ConsciousnessState.ALERT: 0.7,
            ConsciousnessState.FOCUSED: 0.8,
            ConsciousnessState.MEDITATIVE: 0.85,
            ConsciousnessState.FLOW: 0.95,
            ConsciousnessState.STRESSED: 0.7,
        }

        base = state_levels.get(state, 0.5)

        # Modulate by band powers
        alpha_boost = powers['alpha'] * 0.1
        beta_boost = powers['beta'] * 0.05

        return min(1.0, base + alpha_boost + beta_boost)

    def _compute_wellbeing(self, emotion: EmotionScore,
                           meditation: MeditationScore,
                           flow: FlowScore) -> float:
        """Compute overall wellbeing score (0-1)."""
        # Positive valence contributes
        valence_contrib = (emotion.valence + 1) / 2  # Normalize to 0-1

        # Optimal arousal (not too high, not too low)
        arousal_optimal = 1.0 - abs(emotion.arousal - 0.4)

        # Meditation depth
        meditation_contrib = meditation.depth

        # Flow state
        flow_contrib = flow.flow_index

        # Weighted average
        wellbeing = (
            valence_contrib * 0.3 +
            arousal_optimal * 0.2 +
            meditation_contrib * 0.25 +
            flow_contrib * 0.25
        )

        return min(1.0, max(0.0, wellbeing))


class Display:
    """Terminal-based visualization of consciousness states."""

    def __init__(self, width: int = 60):
        self.width = width

    def clear_screen(self):
        """Clear the terminal screen."""
        print("\033[2J\033[H", end="")

    def render(self, poc: ProofOfConsciousness, epoch: int):
        """Render the current consciousness state."""
        self.clear_screen()

        print(f"{Colors.BOLD}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}")
        print(f"{Colors.BOLD}║     🧠  REAL-TIME PROOF OF CONSCIOUSNESS MONITOR  🧠         ║{Colors.RESET}")
        print(f"{Colors.BOLD}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}")
        print()

        # Time and epoch
        mins = int(poc.timestamp // 60)
        secs = int(poc.timestamp % 60)
        print(f"  ⏱️  Time: {mins:02d}:{secs:02d}    Epoch: {epoch}")
        print()

        # Main state with color
        state_color = self._get_state_color(poc.state)
        print(f"  {Colors.BOLD}Consciousness State:{Colors.RESET} {state_color}{poc.state.name}{Colors.RESET}")
        print()

        # Composite scores
        print(f"  {Colors.BOLD}━━━ Composite Scores ━━━{Colors.RESET}")
        self._render_bar("  Consciousness Level", poc.consciousness_level)
        self._render_bar("  Wellbeing Score    ", poc.wellbeing_score)
        print()

        # Trilogy
        print(f"  {Colors.BOLD}━━━ Consciousness Trilogy ━━━{Colors.RESET}")

        # Emotion
        emotion_color = Colors.JOY if poc.emotion.valence > 0 else Colors.STRESS
        print(f"  {emotion_color}♥ Emotion:{Colors.RESET} {poc.emotion.quadrant.value}")
        print(f"      Valence: {poc.emotion.valence:+.2f}  Arousal: {poc.emotion.arousal:.2f}")

        # Sleep
        print(f"  {Colors.SLEEP}☾ Sleep:{Colors.RESET} Stage {poc.sleep.stage} (conf: {poc.sleep.confidence:.0%})")

        # Meditation
        med_color = Colors.MEDITATIVE if poc.meditation.depth > 0.5 else Colors.DIM
        print(f"  {med_color}🕉 Meditation:{Colors.RESET} {poc.meditation.state} (depth: {poc.meditation.depth:.2f})")
        print()

        # Extended Proofs
        print(f"  {Colors.BOLD}━━━ Extended Proofs ━━━{Colors.RESET}")
        self._render_bar("  Attention Index ", poc.attention.attention_index / 3)
        self._render_bar("  Flow Index      ", poc.flow.flow_index)
        self._render_bar("  Engagement Index", poc.engagement.engagement_index / 3)
        print()

        # Band Powers visualization
        print(f"  {Colors.BOLD}━━━ EEG Band Powers ━━━{Colors.RESET}")
        bands = [
            ("δ Delta ", poc.delta, Colors.SLEEP),
            ("θ Theta ", poc.theta, Colors.DROWSY),
            ("α Alpha ", poc.alpha, Colors.RELAXED),
            ("β Beta  ", poc.beta, Colors.FOCUSED),
            ("γ Gamma ", poc.gamma, Colors.FLOW),
        ]
        for name, power, color in bands:
            bar = self._make_bar(power, 30)
            print(f"  {color}{name}{Colors.RESET} {bar} {power:.1%}")

        print()
        print(f"  {Colors.DIM}Press Ctrl+C to stop{Colors.RESET}")

    def _get_state_color(self, state: ConsciousnessState) -> str:
        colors = {
            ConsciousnessState.DEEP_SLEEP: Colors.SLEEP,
            ConsciousnessState.LIGHT_SLEEP: Colors.SLEEP,
            ConsciousnessState.REM: Colors.SLEEP,
            ConsciousnessState.DROWSY: Colors.DROWSY,
            ConsciousnessState.RELAXED: Colors.RELAXED,
            ConsciousnessState.ALERT: Colors.AWAKE,
            ConsciousnessState.FOCUSED: Colors.FOCUSED,
            ConsciousnessState.MEDITATIVE: Colors.MEDITATIVE,
            ConsciousnessState.FLOW: Colors.FLOW,
            ConsciousnessState.STRESSED: Colors.STRESS,
        }
        return colors.get(state, Colors.RESET)

    def _render_bar(self, label: str, value: float):
        bar = self._make_bar(value, 30)
        if value > 0.7:
            color = Colors.BAR_HIGH
        elif value > 0.4:
            color = Colors.BAR_MED
        else:
            color = Colors.BAR_LOW
        print(f"{label}: {color}{bar}{Colors.RESET} {value:.1%}")

    def _make_bar(self, value: float, width: int) -> str:
        filled = int(value * width)
        empty = width - filled
        return "█" * filled + "░" * empty


class ScenarioManager:
    """Manages different consciousness scenarios for demonstration."""

    @staticmethod
    def get_params(scenario: str, progress: float) -> dict:
        """Get signal parameters for a scenario at given progress (0-1)."""

        if scenario == "meditation":
            return ScenarioManager._meditation_scenario(progress)
        elif scenario == "sleep":
            return ScenarioManager._sleep_scenario(progress)
        elif scenario == "focus":
            return ScenarioManager._focus_scenario(progress)
        else:  # mixed
            return ScenarioManager._mixed_scenario(progress)

    @staticmethod
    def _meditation_scenario(progress: float) -> dict:
        """Transition from alert to deep meditation."""
        # Start alert, gradually increase alpha and theta
        alpha = 0.2 + progress * 0.4
        theta = 0.15 + progress * 0.2
        beta = 0.25 - progress * 0.15
        delta = 0.1
        gamma = 0.05

        return {
            'delta': delta, 'theta': theta, 'alpha': alpha,
            'beta': beta, 'gamma': gamma, 'noise': 0.05
        }

    @staticmethod
    def _sleep_scenario(progress: float) -> dict:
        """Progression through sleep stages."""
        if progress < 0.2:
            # Awake -> drowsy
            return {'delta': 0.1, 'theta': 0.2, 'alpha': 0.35,
                    'beta': 0.2, 'gamma': 0.05, 'noise': 0.1}
        elif progress < 0.4:
            # N1
            return {'delta': 0.15, 'theta': 0.3, 'alpha': 0.25,
                    'beta': 0.1, 'gamma': 0.02, 'noise': 0.08}
        elif progress < 0.6:
            # N2
            return {'delta': 0.3, 'theta': 0.25, 'alpha': 0.15,
                    'beta': 0.05, 'gamma': 0.01, 'noise': 0.06}
        elif progress < 0.8:
            # N3 (deep sleep)
            return {'delta': 0.6, 'theta': 0.15, 'alpha': 0.05,
                    'beta': 0.02, 'gamma': 0.01, 'noise': 0.04}
        else:
            # REM
            return {'delta': 0.1, 'theta': 0.35, 'alpha': 0.2,
                    'beta': 0.08, 'gamma': 0.03, 'noise': 0.08}

    @staticmethod
    def _focus_scenario(progress: float) -> dict:
        """Attention and flow state variations."""
        # Oscillate between focused and flow states
        phase = math.sin(progress * math.pi * 4)  # Two full cycles

        if phase > 0.5:
            # Flow state
            return {'delta': 0.05, 'theta': 0.25, 'alpha': 0.35,
                    'beta': 0.15, 'gamma': 0.05, 'noise': 0.03}
        elif phase > 0:
            # Focused
            return {'delta': 0.05, 'theta': 0.2, 'alpha': 0.25,
                    'beta': 0.3, 'gamma': 0.08, 'noise': 0.05}
        else:
            # Brief relaxation
            return {'delta': 0.1, 'theta': 0.2, 'alpha': 0.4,
                    'beta': 0.15, 'gamma': 0.03, 'noise': 0.05}

    @staticmethod
    def _mixed_scenario(progress: float) -> dict:
        """Natural consciousness fluctuations."""
        # Combine multiple sine waves for natural variation
        alpha = 0.25 + 0.15 * math.sin(progress * math.pi * 3)
        theta = 0.2 + 0.1 * math.sin(progress * math.pi * 2 + 1)
        beta = 0.2 + 0.1 * math.sin(progress * math.pi * 4 + 2)
        delta = 0.1 + 0.05 * math.sin(progress * math.pi * 1)
        gamma = 0.05 + 0.03 * math.sin(progress * math.pi * 5)

        return {
            'delta': max(0.05, delta),
            'theta': max(0.1, theta),
            'alpha': max(0.1, alpha),
            'beta': max(0.1, beta),
            'gamma': max(0.02, gamma),
            'noise': 0.05
        }


def run_demo(duration: float = 60.0, scenario: str = "mixed",
             epoch_duration: float = 2.0):
    """Run the real-time demo."""

    print(f"Starting {scenario} scenario for {duration} seconds...")
    print("Initializing sensors...")
    time.sleep(1)

    generator = SignalGenerator()
    analyzer = ConsciousnessAnalyzer()
    display = Display()

    n_epochs = int(duration / epoch_duration)

    try:
        for epoch in range(n_epochs):
            progress = epoch / n_epochs
            timestamp = epoch * epoch_duration

            # Get scenario-specific parameters
            params = ScenarioManager.get_params(scenario, progress)

            # Generate signal
            signal = generator.generate_epoch(epoch_duration, params)

            # Analyze
            poc = analyzer.analyze(signal, generator.sample_rate, timestamp)

            # Display
            display.render(poc, epoch + 1)

            # Wait (simulate real-time)
            time.sleep(epoch_duration)

        print("\n\n  Demo complete!")

    except KeyboardInterrupt:
        print("\n\n  Demo stopped by user.")


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Proof of Consciousness Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scenarios:
  meditation  Transition from alert to deep meditation state
  sleep       Progression through sleep stages (W → N1 → N2 → N3 → REM)
  focus       Oscillation between focused attention and flow states
  mixed       Natural consciousness fluctuations (default)

Examples:
  python realtime_poc_demo.py --duration 120 --scenario meditation
  python realtime_poc_demo.py --scenario sleep
        """
    )

    parser.add_argument('--duration', type=float, default=60.0,
                        help='Duration in seconds (default: 60)')
    parser.add_argument('--scenario', type=str, default='mixed',
                        choices=['meditation', 'sleep', 'focus', 'mixed'],
                        help='Consciousness scenario to simulate (default: mixed)')
    parser.add_argument('--epoch', type=float, default=2.0,
                        help='Epoch duration in seconds (default: 2.0)')

    args = parser.parse_args()

    run_demo(
        duration=args.duration,
        scenario=args.scenario,
        epoch_duration=args.epoch
    )


if __name__ == "__main__":
    main()
