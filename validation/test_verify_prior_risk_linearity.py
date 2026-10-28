import random
import unittest

from verify_prior_risk_linearity import expected_terminal_cross_entropy, mixture


def draw_simplex(rng, n):
    raw = [rng.random() + 0.01 for _ in range(n)]
    s = sum(raw)
    return tuple(x / s for x in raw)


def make_model(rng, n_states, horizon, n_actions, n_observations):
    model = []
    for _z in range(n_states):
        state_steps = []
        for _t in range(horizon):
            action_rows = []
            for _a in range(n_actions):
                action_rows.append(draw_simplex(rng, n_observations))
            state_steps.append(action_rows)
        model.append(state_steps)
    return model


def make_policy(rng, n_actions):
    table = {}
    def policy(history):
        if history not in table:
            table[history] = rng.randrange(n_actions)
        return table[history]
    return policy


class PriorRiskLinearityTests(unittest.TestCase):
    def test_linearity_over_300_seeded_finite_games(self):
        rng = random.Random(20261024)
        for _ in range(300):
            n_states = rng.randint(2, 7)
            horizon = rng.randint(1, 4)
            n_actions = rng.randint(1, 3)
            n_observations = rng.randint(2, 3)
            model = make_model(rng, n_states, horizon, n_actions, n_observations)
            learner = draw_simplex(rng, n_states)
            learner = tuple(0.9 * x + 0.1 / n_states for x in learner)
            q1, q2 = draw_simplex(rng, n_states), draw_simplex(rng, n_states)
            alpha = rng.random()
            class_count = rng.randint(1, min(n_states, 4))
            classes = [rng.randrange(class_count) for _ in range(n_states)]
            policy = make_policy(rng, n_actions)
            r1 = expected_terminal_cross_entropy(q1, learner, model, policy, classes, horizon)
            r2 = expected_terminal_cross_entropy(q2, learner, model, policy, classes, horizon)
            qm = mixture(q1, q2, alpha)
            rm = expected_terminal_cross_entropy(qm, learner, model, policy, classes, horizon)
            expected = alpha * r1 + (1.0 - alpha) * r2
            self.assertAlmostEqual(rm, expected, delta=2e-10)

    def test_max_linear_risk_over_a_polytope_is_at_a_vertex(self):
        rng = random.Random(8286)
        for _ in range(100):
            n_states = rng.randint(2, 6)
            horizon = rng.randint(1, 3)
            model = make_model(rng, n_states, horizon, 2, 2)
            learner = tuple(0.8 * x + 0.2 / n_states for x in draw_simplex(rng, n_states))
            classes = [i % max(1, n_states // 2) for i in range(n_states)]
            policy = make_policy(rng, 2)
            vertices = [draw_simplex(rng, n_states) for _ in range(rng.randint(2, 5))]
            weights = draw_simplex(rng, len(vertices))
            q_mix = tuple(sum(weights[j] * vertices[j][i] for j in range(len(vertices))) for i in range(n_states))
            scores = [expected_terminal_cross_entropy(q, learner, model, policy, classes, horizon) for q in vertices]
            mixed_score = expected_terminal_cross_entropy(q_mix, learner, model, policy, classes, horizon)
            self.assertLessEqual(mixed_score, max(scores) + 2e-10)

    def test_probability_normalization_fails_closed(self):
        bad_model = [[[[0.2, 0.2]]], [[[0.5, 0.5]]]]
        with self.assertRaises(ValueError):
            expected_terminal_cross_entropy((0.5, 0.5), (0.5, 0.5), bad_model, lambda _h: 0, (0, 1), 1)

    def test_rejects_prior_mixture_outside_unit_interval(self):
        with self.assertRaises(ValueError):
            mixture((0.5, 0.5), (0.2, 0.8), 1.1)


if __name__ == "__main__":
    unittest.main()
