#!/usr/bin/env node
/**
 * Dependency-free exact conditional power analysis for a paired binary SFEB
 * primary endpoint (A4 vs A0), using the two-sided exact McNemar test.
 *
 * delta = P(A4 succeeds, A0 fails) - P(A0 succeeds, A4 fails)
 * q = P(the paired outcomes are discordant)
 *
 * Conditional on D discordant pairs, improvements follow
 * Binomial(D, (q + delta) / (2q)). Unconditional power sums over
 * D ~ Binomial(n, q), using exact binomial-tail rejection rules.
 */
import assert from 'node:assert/strict';

const ALPHA = 0.05;
const DELTA = 0.05;
const DISCORDANCE_RATES = [0.05, 0.10, 0.20, 0.30];
const TARGET_POWERS = [0.80, 0.90, 0.95];
const ALPHA_ONE_SIDED = 0.05;
const MAX_N = 2500;
const FACTORIAL_LOGS = new Float64Array(MAX_N + 1);
for (let i = 2; i <= MAX_N; i++) {
  FACTORIAL_LOGS[i] = FACTORIAL_LOGS[i - 1] + Math.log(i);
}

function logBinomialPmf(k, n, p) {
  if (k < 0 || k > n) return -Infinity;
  if (p === 0) return k === 0 ? 0 : -Infinity;
  if (p === 1) return k === n ? 0 : -Infinity;
  return FACTORIAL_LOGS[n] - FACTORIAL_LOGS[k] - FACTORIAL_LOGS[n - k]
    + k * Math.log(p) + (n - k) * Math.log1p(-p);
}

function binomialPmf(k, n, p) {
  const logP = logBinomialPmf(k, n, p);
  return logP === -Infinity ? 0 : Math.exp(logP);
}

/** Stable binomial CDF using the nearer tail and adjacent-PMF recurrence. */
function binomialCdf(k, n, p) {
  if (k < 0) return 0;
  if (k >= n) return 1;
  if (p === 0) return 1;
  if (p === 1) return 0;
  const mode = Math.floor((n + 1) * p);
  if (k >= mode) return 1 - binomialCdf(n - k - 1, n, 1 - p);
  let mass = binomialPmf(mode, n, p);
  let sum = 0;
  for (let x = mode; x >= 0; x--) {
    if (x <= k) sum += mass;
    if (x === 0) break;
    mass *= (x / (n - x + 1)) * ((1 - p) / p);
  }
  return Math.min(1, Math.max(0, sum));
}

/** Largest lower-tail count k with exact two-sided null p-value <= alpha. */
function exactCriticalLower(d, alpha = ALPHA) {
  if (d === 0) return -1;
  let k = Math.floor((d + 1) / 2);
  let mass = binomialPmf(k, d, 0.5);
  let lower = binomialCdf(k, d, 0.5);
  while (k >= 0) {
    if (2 * lower <= alpha + 1e-14) return k;
    lower = Math.max(0, lower - mass);
    if (k === 0) break;
    mass *= k / (d - k + 1);
    k--;
  }
  return -1;
}

function exactPowerCurve(q, maxN, delta = DELTA, alpha = ALPHA) {
  assert(q >= delta && q <= 1, "discordance q must be >= absolute paired effect");
  const pImprove = (q + delta) / (2 * q);
  const conditionalPower = new Float64Array(maxN + 1);
  for (let d = 1; d <= maxN; d++) {
    const critical = exactCriticalLower(d, alpha);
    if (critical >= 0) {
      conditionalPower[d] =
        binomialCdf(critical, d, pImprove)
        + binomialCdf(critical, d, 1 - pImprove);
    }
  }

  // Grow D ~ Binomial(n,q) one task at a time; no Monte Carlo noise.
  let discordantDistribution = new Float64Array([1]);
  const curve = new Float64Array(maxN + 1);
  for (let n = 1; n <= maxN; n++) {
    const next = new Float64Array(n + 1);
    next[0] = discordantDistribution[0] * (1 - q);
    for (let d = 1; d < n; d++) {
      next[d] = discordantDistribution[d] * (1 - q)
        + discordantDistribution[d - 1] * q;
    }
    next[n] = discordantDistribution[n - 1] * q;
    let power = 0;
    for (let d = 1; d <= n; d++) {
      power += next[d] * conditionalPower[d];
    }
    curve[n] = Math.min(1, Math.max(0, power));
    discordantDistribution = next;
  }
  return curve;
}

function normalApproxN(q, targetPower, delta = DELTA) {
  const zAlpha = 1.959963984540054;
  const zPower = targetPower === 0.80 ? 0.8416212335729143
    : targetPower === 0.90 ? 1.2815515655446004
      : 1.6448536269514722;
  return Math.ceil(((zAlpha + zPower) ** 2 * (q - delta ** 2)) / delta ** 2);
}

// Check CDF recurrence against direct finite sums for small cases.
for (let n = 1; n <= 30; n++) {
  for (const p of [0.1, 0.25, 0.5, 0.75, 0.9]) {
    let direct = 0;
    for (let k = 0; k <= n; k++) {
      direct += binomialPmf(k, n, p);
      assert.ok(
        Math.abs(binomialCdf(k, n, p) - direct) < 1e-10,
        "binomial CDF mismatch n=" + n + " k=" + k + " p=" + p,
      );
    }
  }
}
assert.equal(exactCriticalLower(5), -1);
assert.equal(exactCriticalLower(6), 0);
assert.ok(exactCriticalLower(9) >= 1);

const rows = [];
for (const q of DISCORDANCE_RATES) {
  const curve = exactPowerCurve(q, MAX_N);
  const targets = TARGET_POWERS.map((target) => {
    const exactN = curve.findIndex((power, n) => n > 0 && power >= target);
    assert.ok(exactN > 0, "target power not reached for q=" + q);
    return {
      target_power: target,
      exact_minimum_task_pairs: exactN,
      exact_power_at_n: curve[exactN],
      normal_approx_n: normalApproxN(q, target),
    };
  });
  rows.push({
    discordant_pair_rate: q,
    effect_pp: DELTA * 100,
    improvement_only_probability: (q + DELTA) / 2,
    regression_only_probability: (q - DELTA) / 2,
    improvement_fraction_of_discordant: (q + DELTA) / (2 * q),
    n_30_exact_power: curve[30],
    n_150_exact_power: curve[150],
    n_500_exact_power: curve[500],
    targets,
  });
}

// Verify null type-I control and power monotonicity.
const nullCurve = exactPowerCurve(0.20, MAX_N, 0);
assert.ok(nullCurve[150] <= ALPHA + 1e-12, "exact test must control type-I error");
for (const row of rows) {
  assert.ok(row.n_30_exact_power <= row.n_150_exact_power);
  assert.ok(row.n_150_exact_power <= row.targets[0].exact_power_at_n);
}

const zeroEventUpper95 = (n) => 1 - ALPHA_ONE_SIDED ** (1 / n);
const mutantSamples = [20, 60, 100, 299];
const mutantsForOnePercent = Math.ceil(
  Math.log(ALPHA_ONE_SIDED) / Math.log(0.99),
);
assert.equal(mutantsForOnePercent, 299);

console.log(JSON.stringify({
  schema: "sfeb-paired-mcnemar-power-v1",
  method: "exact two-sided McNemar power; conditional binomial test summed over Binomial(n,q) discordant counts",
  alpha_two_sided: ALPHA,
  paired_success_difference: DELTA,
  interpretation: "q is P(A4 and A0 outcomes disagree); the sign of the difference is assumed to favor A4.",
  rows,
  zero_miss_mutation_sensitivity: {
    interval: "one-sided exact 95% Clopper-Pearson upper bound after zero accepted mutants",
    upper_bound_by_sample: mutantSamples.map((n) => ({
      independent_mutants: n,
      upper_95_false_acceptance_rate: zeroEventUpper95(n),
    })),
    independent_mutants_needed_for_upper_bound_below_1_percent: mutantsForOnePercent,
    caveat: "This binomial bound applies only to an independently sampled representative mutant population; a hand-curated mutation suite is a diagnostic challenge set, not a random sample of production defects.",
  },
  limitations: [
    "Power is conditional on the declared discordance rate and +5 percentage-point paired effect.",
    "The 150-task figure is treated as 150 independent task-level pairs after trial aggregation, not 450 repeated task-trial rows.",
    "This does not account for task-family sampling, unequal cluster weights, missing/inconclusive outcomes, multiplicity beyond the single primary contrast, or drift in model/tool versions.",
    "The final sample size must be locked from a blinded pilot estimate of discordance and task-level variability before opening the holdout."
  ]
}, null, 2));
