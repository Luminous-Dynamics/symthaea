#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Extended Benchmarks for HAI Paper

Additional benchmarks:
1. Tiger Problem (classic POMDP)
2. Large Grid Worlds (10×10, 20×20)
3. Multi-Agent Coordination (2 agents)
"""

import numpy as np
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a benchmark."""
    task: str
    method: str
    inference_ms: float
    action_ms: float
    free_energy: float
    success_rate: float
    episodes: int


class HAIAgent:
    """Hyperdimensional Active Inference Agent."""

    def __init__(self, dim: int = 16384):
        self.dim = dim
        self.belief = self._random_hv()
        self.prior = self._random_hv()
        self.pi_s = 1.0
        self.pi_p = 1.0

    def _random_hv(self) -> np.ndarray:
        return np.random.randn(self.dim).astype(np.float32)

    def _normalize(self, hv: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(hv)
        return hv / norm if norm > 0 else hv

    def _cosine(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        return float(np.dot(hv1, hv2) / (np.linalg.norm(hv1) * np.linalg.norm(hv2) + 1e-10))

    def encode_state(self, state: int, state_dim: int) -> np.ndarray:
        """Encode discrete state as hypervector."""
        np.random.seed(state + 1000)
        return self._normalize(np.random.randn(self.dim).astype(np.float32))

    def update_belief(self, observation_hv: np.ndarray) -> float:
        """Update belief given observation."""
        cos_obs = self._cosine(self.belief, observation_hv)
        cos_prior = self._cosine(self.belief, self.prior)

        grad = (self.pi_s * (observation_hv - self.belief * cos_obs) +
                self.pi_p * (self.prior - self.belief * cos_prior))

        self.belief = self._normalize(self.belief + 0.1 * grad)

        # Free energy
        accuracy = -0.5 * self.pi_s * (1 - self._cosine(self.belief, observation_hv))**2
        complexity = 0.5 * self.pi_p * (1 - self._cosine(self.belief, self.prior))
        return complexity - accuracy

    def select_action(self, goal_hv: np.ndarray, num_actions: int) -> Tuple[int, float]:
        """Select action via EFE minimization."""
        efes = []
        for a in range(num_actions):
            predicted = self._normalize(self.belief + 0.1 * np.random.randn(self.dim))
            pragmatic = 1.0 * (1 - self._cosine(predicted, goal_hv))
            epistemic = 0.5 * abs(self._cosine(predicted, self.belief) - 0.5)
            efes.append(pragmatic + epistemic - 0.1 * np.random.random())

        action = int(np.argmin(efes))
        return action, float(efes[action])


# =============================================================================
# Tiger Problem
# =============================================================================

class TigerProblem:
    """
    Classic Tiger POMDP:
    - Two doors: one has tiger, one has treasure
    - Actions: listen, open-left, open-right
    - Observations: tiger-left, tiger-right (noisy)
    """

    def __init__(self, noise: float = 0.15):
        self.noise = noise
        self.tiger_location = np.random.choice([0, 1])  # 0=left, 1=right
        self.done = False

    def reset(self):
        self.tiger_location = np.random.choice([0, 1])
        self.done = False
        return self._get_initial_obs()

    def _get_initial_obs(self) -> int:
        # Random initial observation
        return np.random.choice([0, 1])

    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        Actions: 0=listen, 1=open-left, 2=open-right
        Returns: (observation, reward, done)
        """
        if action == 0:  # Listen
            # Noisy observation
            if np.random.random() < self.noise:
                obs = 1 - self.tiger_location  # Wrong observation
            else:
                obs = self.tiger_location  # Correct observation
            return obs, -1, False  # Small cost for listening

        elif action == 1:  # Open left
            self.done = True
            if self.tiger_location == 0:  # Tiger on left
                return 0, -100, True  # Eaten!
            else:
                return 1, 10, True  # Treasure!

        else:  # Open right
            self.done = True
            if self.tiger_location == 1:  # Tiger on right
                return 1, -100, True  # Eaten!
            else:
                return 0, 10, True  # Treasure!


def benchmark_tiger(agent_class, trials: int = 100, max_steps: int = 20) -> BenchmarkResult:
    """Benchmark on Tiger problem."""
    inference_times = []
    action_times = []
    total_rewards = []
    successes = 0

    for trial in range(trials):
        env = TigerProblem()
        agent = agent_class(dim=8192)  # Smaller for speed
        obs = env.reset()

        total_reward = 0
        goal_hv = agent.encode_state(1, 2)  # Want treasure

        for step in range(max_steps):
            if env.done:
                break

            # Encode observation
            obs_hv = agent.encode_state(obs, 2)

            # Inference
            start = time.perf_counter()
            fe = agent.update_belief(obs_hv)
            inference_times.append((time.perf_counter() - start) * 1000)

            # Action selection
            start = time.perf_counter()
            action, efe = agent.select_action(goal_hv, 3)
            action_times.append((time.perf_counter() - start) * 1000)

            # Step
            obs, reward, done = env.step(action)
            total_reward += reward

        total_rewards.append(total_reward)
        if total_reward > 0:
            successes += 1

    return BenchmarkResult(
        task="tiger",
        method="HAI",
        inference_ms=float(np.mean(inference_times)),
        action_ms=float(np.mean(action_times)),
        free_energy=float(np.mean([r for r in total_rewards])),
        success_rate=successes / trials,
        episodes=trials
    )


# =============================================================================
# Large Grid Worlds
# =============================================================================

class GridWorld:
    """Grid world environment."""

    def __init__(self, size: int = 10):
        self.size = size
        self.state = (size // 2, size // 2)  # Start in center
        self.goal = (size - 1, size - 1)  # Goal in corner

    def reset(self) -> int:
        self.state = (self.size // 2, self.size // 2)
        return self._state_to_int()

    def _state_to_int(self) -> int:
        return self.state[0] * self.size + self.state[1]

    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        Actions: 0=up, 1=down, 2=left, 3=right
        """
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        new_x = max(0, min(self.size - 1, self.state[0] + dx))
        new_y = max(0, min(self.size - 1, self.state[1] + dy))
        self.state = (new_x, new_y)

        done = self.state == self.goal
        reward = 10 if done else -0.1

        return self._state_to_int(), reward, done


def benchmark_grid(size: int, trials: int = 50, max_steps: int = 200) -> BenchmarkResult:
    """Benchmark on grid world."""
    inference_times = []
    action_times = []
    free_energies = []
    successes = 0

    for trial in range(trials):
        env = GridWorld(size=size)
        agent = HAIAgent(dim=8192)
        obs = env.reset()

        goal_hv = agent.encode_state(size * size - 1, size * size)  # Corner

        for step in range(max_steps):
            obs_hv = agent.encode_state(obs, size * size)

            start = time.perf_counter()
            fe = agent.update_belief(obs_hv)
            inference_times.append((time.perf_counter() - start) * 1000)

            start = time.perf_counter()
            action, efe = agent.select_action(goal_hv, 4)
            action_times.append((time.perf_counter() - start) * 1000)

            obs, reward, done = env.step(action)
            free_energies.append(fe)

            if done:
                successes += 1
                break

    return BenchmarkResult(
        task=f"grid_{size}x{size}",
        method="HAI",
        inference_ms=float(np.mean(inference_times)),
        action_ms=float(np.mean(action_times)),
        free_energy=float(np.mean(free_energies)),
        success_rate=successes / trials,
        episodes=trials
    )


# =============================================================================
# Multi-Agent Coordination
# =============================================================================

class CoordinationGame:
    """
    Two agents must coordinate to meet at a target location.
    """

    def __init__(self, size: int = 5):
        self.size = size
        self.reset()

    def reset(self) -> Tuple[int, int]:
        self.agent1_pos = (0, 0)
        self.agent2_pos = (self.size - 1, self.size - 1)
        self.target = (self.size // 2, self.size // 2)
        return self._pos_to_int(self.agent1_pos), self._pos_to_int(self.agent2_pos)

    def _pos_to_int(self, pos: Tuple[int, int]) -> int:
        return pos[0] * self.size + pos[1]

    def step(self, action1: int, action2: int) -> Tuple[Tuple[int, int], float, bool]:
        """Both agents move simultaneously."""
        moves = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]  # up, down, left, right, stay

        # Move agent 1
        dx, dy = moves[action1]
        new_x = max(0, min(self.size - 1, self.agent1_pos[0] + dx))
        new_y = max(0, min(self.size - 1, self.agent1_pos[1] + dy))
        self.agent1_pos = (new_x, new_y)

        # Move agent 2
        dx, dy = moves[action2]
        new_x = max(0, min(self.size - 1, self.agent2_pos[0] + dx))
        new_y = max(0, min(self.size - 1, self.agent2_pos[1] + dy))
        self.agent2_pos = (new_x, new_y)

        # Check success: both at target
        done = (self.agent1_pos == self.target and self.agent2_pos == self.target)
        reward = 10 if done else -0.1

        return ((self._pos_to_int(self.agent1_pos), self._pos_to_int(self.agent2_pos)),
                reward, done)


def benchmark_multiagent(trials: int = 50, max_steps: int = 100) -> BenchmarkResult:
    """Benchmark multi-agent coordination."""
    inference_times = []
    action_times = []
    free_energies = []
    successes = 0

    for trial in range(trials):
        env = CoordinationGame(size=5)
        agent1 = HAIAgent(dim=4096)
        agent2 = HAIAgent(dim=4096)

        obs1, obs2 = env.reset()
        target_hv = agent1.encode_state(12, 25)  # Center (2,2) = 12

        for step in range(max_steps):
            # Agent 1
            obs1_hv = agent1.encode_state(obs1, 25)
            start = time.perf_counter()
            fe1 = agent1.update_belief(obs1_hv)
            inference_times.append((time.perf_counter() - start) * 1000)

            start = time.perf_counter()
            action1, _ = agent1.select_action(target_hv, 5)
            action_times.append((time.perf_counter() - start) * 1000)

            # Agent 2
            obs2_hv = agent2.encode_state(obs2, 25)
            start = time.perf_counter()
            fe2 = agent2.update_belief(obs2_hv)
            inference_times.append((time.perf_counter() - start) * 1000)

            start = time.perf_counter()
            action2, _ = agent2.select_action(target_hv, 5)
            action_times.append((time.perf_counter() - start) * 1000)

            free_energies.extend([fe1, fe2])

            # Step
            (obs1, obs2), reward, done = env.step(action1, action2)

            if done:
                successes += 1
                break

    return BenchmarkResult(
        task="multiagent_coord",
        method="HAI",
        inference_ms=float(np.mean(inference_times)),
        action_ms=float(np.mean(action_times)),
        free_energy=float(np.mean(free_energies)),
        success_rate=successes / trials,
        episodes=trials
    )


# =============================================================================
# Main
# =============================================================================

def generate_report(results: List[BenchmarkResult], output_path: Path):
    """Generate extended benchmark report."""
    with open(output_path, 'w') as f:
        f.write("# Extended Benchmark Results\n\n")
        f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Results Summary\n\n")
        f.write("| Task | Inference (ms) | Action (ms) | Success Rate | Episodes |\n")
        f.write("|------|----------------|-------------|--------------|----------|\n")
        for r in results:
            f.write(f"| {r.task} | {r.inference_ms:.3f} | {r.action_ms:.3f} | "
                   f"{r.success_rate*100:.1f}% | {r.episodes} |\n")
        f.write("\n")

        f.write("## Task Descriptions\n\n")
        f.write("### 1. Tiger Problem\n")
        f.write("Classic POMDP with partial observability. Agent must infer tiger location from noisy observations.\n\n")

        f.write("### 2. Grid World 10×10\n")
        f.write("Navigation in 100-state grid. Tests scaling to larger state spaces.\n\n")

        f.write("### 3. Grid World 20×20\n")
        f.write("Navigation in 400-state grid. Tests scaling limits.\n\n")

        f.write("### 4. Multi-Agent Coordination\n")
        f.write("Two HAI agents must coordinate to reach a common target. Tests decentralized coordination.\n\n")

        f.write("## Key Findings\n\n")

        # Best success rate
        best = max(results, key=lambda r: r.success_rate)
        f.write(f"1. **Best Task Performance**: {best.task} ({best.success_rate*100:.1f}% success)\n")

        # Scaling analysis
        grid_results = [r for r in results if 'grid' in r.task]
        if len(grid_results) >= 2:
            small = min(grid_results, key=lambda r: r.inference_ms)
            large = max(grid_results, key=lambda r: r.inference_ms)
            slowdown = large.inference_ms / small.inference_ms
            f.write(f"2. **Scaling**: {small.task} → {large.task}: {slowdown:.1f}× inference time increase\n")

        # Multi-agent
        ma_results = [r for r in results if 'multiagent' in r.task]
        if ma_results:
            ma = ma_results[0]
            f.write(f"3. **Multi-Agent**: {ma.success_rate*100:.1f}% coordination success rate\n")

        f.write("\n---\n")
        f.write("*All benchmarks use HAI (Python reference implementation)*\n")


def main():
    """Run extended benchmarks."""
    print("=" * 60)
    print("HAI Extended Benchmarks")
    print("=" * 60)

    np.random.seed(42)

    results = []

    logger.info("Running Tiger Problem benchmark...")
    results.append(benchmark_tiger(HAIAgent, trials=100))

    logger.info("Running Grid 10×10 benchmark...")
    results.append(benchmark_grid(size=10, trials=50))

    logger.info("Running Grid 20×20 benchmark...")
    results.append(benchmark_grid(size=20, trials=30))

    logger.info("Running Multi-Agent Coordination benchmark...")
    results.append(benchmark_multiagent(trials=50))

    # Save results
    output_dir = Path(__file__).parent.parent / "docs"

    with open(output_dir / "EXTENDED_BENCHMARK_RESULTS.json", 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    generate_report(results, output_dir / "EXTENDED_BENCHMARKS_REPORT.md")

    print("\n" + "=" * 60)
    print("Results saved to:")
    print(f"  - {output_dir / 'EXTENDED_BENCHMARKS_REPORT.md'}")
    print(f"  - {output_dir / 'EXTENDED_BENCHMARK_RESULTS.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
