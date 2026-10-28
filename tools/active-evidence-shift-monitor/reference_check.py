#!/usr/bin/env python3
"""Independent exact reference checker for the shift-monitor Rust DP."""
import csv
import math
import sys

ALPHA = 0.05
CELLS = 12.0
PER_CELL_ALPHA = ALPHA / CELLS
P0 = 0.5
P1 = 0.8
CHANGE_AFTER = 100
HORIZONS = (100, 250, 500, 1000)
BOUNDARIES = (
    "fixed_time_final_only",
    "fixed_time_reused_each_look",
    "time_uniform_alpha_spending",
    "beta_mixture_e_process",
)
SCENARIOS = (
    "stationary_null",
    "step_down_010", "step_down_020", "step_down_030", "step_down_040", "step_down_045",
    "step_up_055", "step_up_060", "step_up_070", "step_up_080", "step_up_090",
    "selection_bias_no_environment_shift",
    "stable_baseline_rate_045", "stable_baseline_rate_048", "stable_baseline_rate_049",
    "stable_baseline_rate_051", "stable_baseline_rate_052", "stable_baseline_rate_055",
    "within_band_transition_049_to_051",
)
SHIFT_PROBABILITIES = {
    "stationary_null": None,
    "step_down_010": 0.10,
    "step_down_020": 0.20,
    "step_down_030": 0.30,
    "step_down_040": 0.40,
    "step_down_045": 0.45,
    "step_up_055": 0.55,
    "step_up_060": 0.60,
    "step_up_070": 0.70,
    "step_up_080": 0.80,
    "step_up_090": 0.90,
    "selection_bias_no_environment_shift": None,
    "stable_baseline_rate_045": None,
    "stable_baseline_rate_048": None,
    "stable_baseline_rate_049": None,
    "stable_baseline_rate_051": None,
    "stable_baseline_rate_052": None,
    "stable_baseline_rate_055": None,
    "within_band_transition_049_to_051": 0.51,
}
PRE_SHIFT_PROBABILITIES = {scenario: P0 for scenario in SCENARIOS}
PRE_SHIFT_PROBABILITIES["selection_bias_no_environment_shift"] = 0.74
PRE_SHIFT_PROBABILITIES["stable_baseline_rate_045"] = 0.45
PRE_SHIFT_PROBABILITIES["stable_baseline_rate_048"] = 0.48
PRE_SHIFT_PROBABILITIES["stable_baseline_rate_049"] = 0.49
PRE_SHIFT_PROBABILITIES["stable_baseline_rate_051"] = 0.51
PRE_SHIFT_PROBABILITIES["stable_baseline_rate_052"] = 0.52
PRE_SHIFT_PROBABILITIES["stable_baseline_rate_055"] = 0.55
PRE_SHIFT_PROBABILITIES["within_band_transition_049_to_051"] = 0.49
FIELDS = (
    "boundary", "scenario", "scenario_class", "horizon", "change_after", "p0", "stream_pre_p", "p1",
    "scenario_or_pre_shift_alarm_probability", "post_shift_detection_probability",
    "mean_post_shift_detection_delay", "per_cell_alpha", "survival_probability",
)
EPS = 1e-10
TOTAL_LABEL_BUDGET = 1000
SHIFT_AFTER_TOTAL_LABELS = 100
SENTINEL_FRACTIONS = (0.0, 0.05, 0.10, 0.20, 0.25, 0.50, 1.0)
BUDGET_SHIFT_PROBABILITIES = (0.55, 0.60, 0.70, 0.80, 0.90)
BUDGET_FIELDS = (
    "sentinel_fraction", "post_shift_p", "total_label_budget", "sentinel_labels", "adaptive_labels",
    "change_after_sentinel_labels", "no_target_shift_alarm_probability",
    "pre_change_false_alarm_probability", "true_shift_detection_probability",
    "mean_delay_sentinel_labels", "mean_delay_total_label_equivalent", "per_cell_alpha",
)
EPROCESS_FAMILY_PRIORS = {
    "beta_0.5_0.5": (0.5, 0.5),
    "beta_1_1": (1.0, 1.0),
    "beta_5_5": (5.0, 5.0),
    "beta_20_20": (20.0, 20.0),
}
COMPOSITE_NULL_LOWER = 0.48
COMPOSITE_NULL_UPPER = 0.52
COMPOSITE_BETTING_FRACTIONS = (0.05, 0.1, 0.2, 0.4, 0.8)
COMPOSITE_FAMILY = "composite_band_betting_0.48_0.52"
EPROCESS_FAMILY_ORDER = (
    "beta_0.5_0.5",
    "beta_1_1",
    "beta_5_5",
    "beta_20_20",
    "equal_portfolio_4_priors",
    COMPOSITE_FAMILY,
)
EPROCESS_FAMILY_SCENARIOS = (
    "stationary_null",
    "step_down_010", "step_down_020", "step_down_030", "step_down_040", "step_down_045",
    "step_up_055", "step_up_060", "step_up_070", "step_up_080", "step_up_090",
    "selection_bias_no_environment_shift",
    "stable_baseline_rate_045", "stable_baseline_rate_048", "stable_baseline_rate_049",
    "stable_baseline_rate_051", "stable_baseline_rate_052", "stable_baseline_rate_055",
    "within_band_transition_049_to_051",
)
EPROCESS_FAMILY_FIELDS = (
    "family", "scenario", "scenario_class", "horizon", "change_after", "p0", "stream_pre_p", "p1",
    "scenario_or_pre_shift_alarm_probability", "post_shift_detection_probability",
    "mean_post_shift_detection_delay", "per_cell_alpha", "survival_probability", "log_e_threshold",
)
CUSUM_SCALE = 25
CUSUM_UPPER_INCREMENT = 12
CUSUM_LOWER_INCREMENT = 13
CUSUM_FIELDS = (
    "direction", "scenario", "scenario_class", "horizon", "change_after", "p0", "stream_pre_p", "p1",
    "null_lower", "null_upper", "threshold_scaled", "threshold_score",
    "matched_reference_alarm_probability", "calibration_boundary_alarm_probability",
    "family_worst_case_union_bound", "scenario_or_pre_shift_alarm_probability",
    "post_shift_detection_probability", "mean_post_shift_detection_delay", "per_cell_alpha",
    "survival_probability",
)
BETA_A = 0.5
BETA_B = 0.5


def scenario_class(scenario):
    if scenario == "stationary_null":
        return "stationary_null"
    if scenario.startswith("step_"):
        return "target_shift"
    if scenario == "selection_bias_no_environment_shift":
        return "selection_bias_control"
    if scenario.startswith("stable_baseline_rate_"):
        return "baseline_misspecification"
    if scenario == "within_band_transition_049_to_051":
        return "within_band_transition_control"
    raise ValueError(f"unclassified benchmark scenario: {scenario}")


def reporting_metrics(scenario, metrics):
    """Separate accepted-null alarm risk from real target-shift detection metrics."""
    false_alarm, detection, delay, survival = metrics
    kind = scenario_class(scenario)
    null_alarm = false_alarm + (detection or 0.0) if kind == "within_band_transition_control" else false_alarm
    is_target_shift = kind == "target_shift"
    return kind, null_alarm, detection if is_target_shift else None, delay if is_target_shift else None, survival

def radius(boundary, n):
    if boundary == "time_uniform_alpha_spending":
        log_term = math.log(2.0 * CELLS * n * (n + 1) / ALPHA)
    else:
        log_term = math.log(2.0 * CELLS / ALPHA)
    return math.sqrt(log_term / (2.0 * n))


def log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def beta_mixture_log_e_value(n, successes, p0):
    if n <= 0 or not (0 <= successes <= n) or not (0.0 < p0 < 1.0):
        raise ValueError("invalid Bernoulli e-process parameters")
    log_marginal = (
        log_beta(successes + BETA_A, n - successes + BETA_B)
        - log_beta(BETA_A, BETA_B)
    )
    log_null_likelihood = (
        successes * math.log(p0) + (n - successes) * math.log1p(-p0)
    )
    return log_marginal - log_null_likelihood


def should_alarm(boundary, n, successes, horizon):
    if boundary == "fixed_time_final_only" and n != horizon:
        return False
    if boundary == "beta_mixture_e_process":
        return beta_mixture_log_e_value(n, successes, P0) >= math.log(1.0 / PER_CELL_ALPHA)
    width = radius(boundary, n)
    estimate = successes / n
    lower = max(0.0, estimate - width)
    upper = min(1.0, estimate + width)
    return P0 < lower or P0 > upper


def simulate(boundary, scenario, horizon, change_after_override=None, post_shift_override=None):
    post_shift_p = post_shift_override if post_shift_override is not None else SHIFT_PROBABILITIES[scenario]
    change_after = CHANGE_AFTER if post_shift_p is not None else 0
    if change_after_override is not None:
        change_after = change_after_override
    active = [0.0] * (horizon + 1)
    active[0] = 1.0
    first_alarm = [0.0] * (horizon + 1)

    for n in range(1, horizon + 1):
        probability = (
            PRE_SHIFT_PROBABILITIES[scenario]
            if post_shift_p is None or n <= change_after
            else post_shift_p
        )
        next_active = [0.0] * (horizon + 1)
        for successes, mass in enumerate(active[:n]):
            if mass == 0.0:
                continue
            branches = (
                (successes, mass * (1.0 - probability)),
                (successes + 1, mass * probability),
            )
            for next_successes, branch_mass in branches:
                if should_alarm(boundary, n, next_successes, horizon):
                    first_alarm[n] += branch_mass
                else:
                    next_active[next_successes] += branch_mass
        active = next_active

    if post_shift_p is None:
        # No target-environment change occurred. For selection_bias_no_environment_shift,
        # any alarm is spurious relative to the unchanged target population, even though the
        # selected observation stream violates the detector's iid(p0) null assumption.
        false_alarm = sum(first_alarm[1:])
        detection = None
        delay = None
    else:
        false_alarm = sum(first_alarm[1:min(change_after, horizon) + 1])
        detection = sum(first_alarm[change_after + 1:])
        delay_mass = sum(
            (n - change_after) * first_alarm[n]
            for n in range(change_after + 1, horizon + 1)
        )
        delay = delay_mass / detection if detection > 0.0 else None

    survival = sum(active)
    return false_alarm, detection, delay, survival


def as_text(value):
    return "NA" if value is None else f"{value:.12f}"


def reference_rows():
    rows = []
    for boundary in BOUNDARIES:
        for horizon in HORIZONS:
            for scenario in SCENARIOS:
                metrics = simulate(boundary, scenario, horizon)
                kind, null_alarm, detection, delay, survival = reporting_metrics(scenario, metrics)
                rows.append({
                    "boundary": boundary,
                    "scenario": scenario,
                    "scenario_class": kind,
                    "horizon": str(horizon),
                    "change_after": str(CHANGE_AFTER) if SHIFT_PROBABILITIES[scenario] is not None else "NA",
                    "p0": f"{P0:.12f}",
                    "stream_pre_p": f"{PRE_SHIFT_PROBABILITIES[scenario]:.12f}",
                    "p1": f"{SHIFT_PROBABILITIES[scenario]:.12f}" if SHIFT_PROBABILITIES[scenario] is not None else "NA",
                    "scenario_or_pre_shift_alarm_probability": f"{null_alarm:.12f}",
                    "post_shift_detection_probability": as_text(detection),
                    "mean_post_shift_detection_delay": as_text(delay),
                    "per_cell_alpha": f"{PER_CELL_ALPHA:.12f}",
                    "survival_probability": f"{survival:.12f}",
                })
    return rows


def read_csv(path):
    try:
        with open(path, newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError(f"header mismatch: {reader.fieldnames}")
            return list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"cannot read output CSV: {exc}") from exc


def check_csv(path):
    expected = reference_rows()
    actual = read_csv(path)
    if len(actual) != len(expected):
        raise ValueError(f"row count mismatch: expected {len(expected)}, got {len(actual)}")
    key_fields = ("boundary", "scenario", "horizon")
    text_fields = ("boundary", "scenario", "scenario_class", "horizon", "change_after", "p1",
                   "post_shift_detection_probability", "mean_post_shift_detection_delay")
    numeric_fields = ("p0", "stream_pre_p", "scenario_or_pre_shift_alarm_probability", "per_cell_alpha", "survival_probability")
    for line, (got, want) in enumerate(zip(actual, expected), 2):
        if tuple(got[k] for k in key_fields) != tuple(want[k] for k in key_fields):
            raise ValueError(f"row key mismatch on line {line}")
        for field in text_fields:
            if got[field] != want[field]:
                raise ValueError(f"{field} mismatch on line {line}: {got[field]} != {want[field]}")
        for field in numeric_fields:
            if want[field] == "NA":
                if got[field] != "NA":
                    raise ValueError(f"{field} should be NA on line {line}")
                continue
            try:
                value = float(got[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid numeric {field} on line {line}") from exc
            target = float(want[field])
            if not math.isfinite(value) or abs(value - target) > 5e-10:
                raise ValueError(f"{field} mismatch on line {line}: {value} != {target}")
    print(f"PASS: independently recomputed all {len(expected)} exact-DP rows")



def sentinel_budget_rows():
    """Exact frontier at fixed total labels over sentinel shares and shift sizes."""
    rows = []
    for fraction in SENTINEL_FRACTIONS:
        sentinel_labels = int(round(fraction * TOTAL_LABEL_BUDGET))
        adaptive_labels = TOTAL_LABEL_BUDGET - sentinel_labels
        if sentinel_labels == 0:
            no_shift = simulate(
                "beta_mixture_e_process",
                "selection_bias_no_environment_shift",
                TOTAL_LABEL_BUDGET,
            )[0]
            rows.append({
                "sentinel_fraction": f"{fraction:.4f}",
                "post_shift_p": "NA",
                "total_label_budget": str(TOTAL_LABEL_BUDGET),
                "sentinel_labels": "0",
                "adaptive_labels": str(TOTAL_LABEL_BUDGET),
                "change_after_sentinel_labels": "NA",
                "no_target_shift_alarm_probability": f"{no_shift:.12f}",
                "pre_change_false_alarm_probability": "NA",
                "true_shift_detection_probability": "NA",
                "mean_delay_sentinel_labels": "NA",
                "mean_delay_total_label_equivalent": "NA",
                "per_cell_alpha": f"{PER_CELL_ALPHA:.12f}",
            })
            continue

        change_after = int(round(fraction * SHIFT_AFTER_TOTAL_LABELS))
        stable_alarm = simulate(
            "beta_mixture_e_process", "stationary_null", sentinel_labels
        )[0]
        for post_shift_p in BUDGET_SHIFT_PROBABILITIES:
            pre_change_false_alarm, detection, delay, _survival = simulate(
                "beta_mixture_e_process",
                "step_up_080",
                sentinel_labels,
                change_after_override=change_after,
                post_shift_override=post_shift_p,
            )
            delay_total = delay / fraction if delay is not None else None
            rows.append({
                "sentinel_fraction": f"{fraction:.4f}",
                "post_shift_p": f"{post_shift_p:.2f}",
                "total_label_budget": str(TOTAL_LABEL_BUDGET),
                "sentinel_labels": str(sentinel_labels),
                "adaptive_labels": str(adaptive_labels),
                "change_after_sentinel_labels": str(change_after),
                "no_target_shift_alarm_probability": f"{stable_alarm:.12f}",
                "pre_change_false_alarm_probability": f"{pre_change_false_alarm:.12f}",
                "true_shift_detection_probability": as_text(detection),
                "mean_delay_sentinel_labels": as_text(delay),
                "mean_delay_total_label_equivalent": as_text(delay_total),
                "per_cell_alpha": f"{PER_CELL_ALPHA:.12f}",
            })
    return rows


def check_sentinel_budget_csv(path):
    expected = sentinel_budget_rows()
    try:
        with open(path, newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != BUDGET_FIELDS:
                raise ValueError(f"sentinel-budget header mismatch: {reader.fieldnames}")
            actual = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"cannot read sentinel-budget output: {exc}") from exc
    if len(actual) != len(expected):
        raise ValueError(f"sentinel-budget row count mismatch: expected {len(expected)}, got {len(actual)}")

    text_fields = (
        "sentinel_fraction", "post_shift_p", "total_label_budget", "sentinel_labels",
        "adaptive_labels", "change_after_sentinel_labels",
    )
    numeric_fields = (
        "no_target_shift_alarm_probability", "pre_change_false_alarm_probability",
        "true_shift_detection_probability", "mean_delay_sentinel_labels",
        "mean_delay_total_label_equivalent", "per_cell_alpha",
    )
    for line, (got, want) in enumerate(zip(actual, expected), 2):
        for field in text_fields:
            if got[field] != want[field]:
                raise ValueError(f"{field} mismatch on line {line}: {got[field]} != {want[field]}")
        if int(got["sentinel_labels"]) + int(got["adaptive_labels"]) != TOTAL_LABEL_BUDGET:
            raise ValueError(f"label budget does not conserve on line {line}")
        for field in numeric_fields:
            if want[field] == "NA":
                if got[field] != "NA":
                    raise ValueError(f"{field} should be NA on line {line}")
                continue
            try:
                value = float(got[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid numeric {field} on line {line}") from exc
            target = float(want[field])
            if not math.isfinite(value) or abs(value - target) > 5e-10:
                raise ValueError(f"{field} mismatch on line {line}: {value} != {target}")
    print(f"PASS: independently recomputed all {len(expected)} sentinel-budget rows")


def sentinel_budget_self_test():
    rows = sentinel_budget_rows()
    expected_count = 1 + (len(SENTINEL_FRACTIONS) - 1) * len(BUDGET_SHIFT_PROBABILITIES)
    if len(rows) != expected_count:
        raise ValueError(f"sentinel-budget row count mismatch: expected {expected_count}, got {len(rows)}")
    by_key = {(row["sentinel_fraction"], row["post_shift_p"]): row for row in rows}
    no_audit = by_key[("0.0000", "NA")]
    five_percent = by_key[("0.0500", "0.80")]
    ten_percent_mild = by_key[("0.1000", "0.60")]
    ten_percent_moderate = by_key[("0.1000", "0.70")]
    ten_percent_large = by_key[("0.1000", "0.80")]
    twenty_percent_moderate = by_key[("0.2000", "0.70")]
    twenty_percent_large = by_key[("0.2000", "0.80")]
    if float(no_audit["no_target_shift_alarm_probability"]) < 0.99:
        raise ValueError("zero-sentinel negative control did not expose selection-induced alarms")
    if float(five_percent["no_target_shift_alarm_probability"]) > PER_CELL_ALPHA + EPS:
        raise ValueError("5% randomized-sentinel null alarm exceeded alpha")
    if float(ten_percent_mild["true_shift_detection_probability"]) >= 0.05:
        raise ValueError("10% sentinel budget unexpectedly has strong power for the mild shift")
    if not 0.40 < float(ten_percent_moderate["true_shift_detection_probability"]) < 0.50:
        raise ValueError("10% sentinel budget moderate-shift power left expected fixture band")
    if float(ten_percent_large["true_shift_detection_probability"]) < 0.95:
        raise ValueError("10% sentinel budget missed large-shift detection gate")
    if float(twenty_percent_moderate["true_shift_detection_probability"]) < 0.85:
        raise ValueError("20% sentinel budget missed moderate-shift detection gate")
    if float(twenty_percent_large["true_shift_detection_probability"]) < 0.99:
        raise ValueError("20% sentinel budget missed large-shift detection gate")
    for row in rows:
        if int(row["sentinel_labels"]) + int(row["adaptive_labels"]) != TOTAL_LABEL_BUDGET:
            raise ValueError("sentinel budget frontier does not conserve labels")
        if row["sentinel_labels"] != "0":
            if float(row["no_target_shift_alarm_probability"]) > PER_CELL_ALPHA + EPS:
                raise ValueError("stable sentinel alarm exceeded the allocated per-cell alpha")
            if float(row["pre_change_false_alarm_probability"]) > PER_CELL_ALPHA + EPS:
                raise ValueError("pre-change alarm exceeded the allocated per-cell alpha")
    print("PASS: audit-budget conservation, zero-sentinel negative control, all sentinel null and pre-change alpha checks, and shift-magnitude power gates")


def _beta_log_e_from_parameters(n, successes, p0, a, b, prior_log_beta=None):
    if prior_log_beta is None:
        prior_log_beta = log_beta(a, b)
    return (
        log_beta(successes + a, n - successes + b)
        - prior_log_beta
        - successes * math.log(p0)
        - (n - successes) * math.log1p(-p0)
    )


def composite_band_log_e_value(n, successes, lower=COMPOSITE_NULL_LOWER,
                              upper=COMPOSITE_NULL_UPPER):
    """Log e-value from a fixed mixture of directional bets valid on p in [lower, upper]."""
    if n <= 0 or not (0 <= successes <= n):
        raise ValueError("invalid composite-null e-process state")
    if not (0.0 <= lower < upper <= 1.0):
        raise ValueError("invalid composite-null rate interval")

    failures = n - successes
    upper_logs = []
    lower_logs = []
    for lam in COMPOSITE_BETTING_FRACTIONS:
        if not (lam >= 0.0 and lam * upper <= 1.0 and lam * (1.0 - lower) <= 1.0):
            raise ValueError("betting fraction yields a negative factor")
        upper_logs.append(
            successes * math.log1p(lam * (1.0 - upper))
            + failures * math.log1p(-lam * upper)
        )
        lower_logs.append(
            successes * math.log1p(-lam * (1.0 - lower))
            + failures * math.log1p(lam * lower)
        )

    def log_mean_exp(values):
        maximum = max(values)
        return maximum + math.log(sum(math.exp(value - maximum) for value in values)
                                  / len(values))

    upper_log_e = log_mean_exp(upper_logs)
    lower_log_e = log_mean_exp(lower_logs)
    maximum = max(upper_log_e, lower_log_e)
    return maximum + math.log(
        0.5 * math.exp(upper_log_e - maximum)
        + 0.5 * math.exp(lower_log_e - maximum)
    )


def eprocess_family_alarm_tables(horizon=1000):
    """Precompute every family boundary once for exact DP/reference parity."""
    threshold = math.log(1.0 / PER_CELL_ALPHA)
    prior_logs = {name: [None] * (horizon + 1) for name in EPROCESS_FAMILY_PRIORS}
    normalizers = {
        name: log_beta(a, b)
        for name, (a, b) in EPROCESS_FAMILY_PRIORS.items()
    }
    for n in range(1, horizon + 1):
        for name, (a, b) in EPROCESS_FAMILY_PRIORS.items():
            normalizer = normalizers[name]
            denominator = math.lgamma(n + a + b)
            rows = []
            for successes in range(n + 1):
                log_marginal = (
                    math.lgamma(successes + a)
                    + math.lgamma(n - successes + b)
                    - denominator
                )
                log_null = (
                    successes * math.log(P0)
                    + (n - successes) * math.log1p(-P0)
                )
                rows.append(log_marginal - normalizer - log_null)
            prior_logs[name][n] = rows

    log_values = {name: [None] * (horizon + 1) for name in EPROCESS_FAMILY_ORDER}
    for name in EPROCESS_FAMILY_PRIORS:
        log_values[name] = prior_logs[name]
    for n in range(1, horizon + 1):
        for successes in range(n + 1):
            components = [
                prior_logs[name][n][successes]
                for name in EPROCESS_FAMILY_PRIORS
            ]
            maximum = max(components)
            log_values["equal_portfolio_4_priors"][n] = (
                log_values["equal_portfolio_4_priors"][n] or [None] * (n + 1)
            )
            log_values["equal_portfolio_4_priors"][n][successes] = (
                maximum
                + math.log(
                    sum(0.25 * math.exp(value - maximum) for value in components)
                )
            )
            log_values[COMPOSITE_FAMILY][n] = (
                log_values[COMPOSITE_FAMILY][n] or [None] * (n + 1)
            )
            log_values[COMPOSITE_FAMILY][n][successes] = composite_band_log_e_value(
                n, successes, COMPOSITE_NULL_LOWER, COMPOSITE_NULL_UPPER
            )

    return {
        family: [
            None if n == 0 else [value >= threshold for value in log_values[family][n]]
            for n in range(horizon + 1)
        ]
        for family in EPROCESS_FAMILY_ORDER
    }


def simulate_from_alarm_table(alarm_table, scenario, horizon=1000):
    post_shift_p = SHIFT_PROBABILITIES[scenario]
    change_after = CHANGE_AFTER if post_shift_p is not None else 0
    active = [0.0] * (horizon + 1)
    active[0] = 1.0
    first_alarm = [0.0] * (horizon + 1)

    for n in range(1, horizon + 1):
        probability = (
            PRE_SHIFT_PROBABILITIES[scenario]
            if post_shift_p is None or n <= change_after
            else post_shift_p
        )
        next_active = [0.0] * (horizon + 1)
        boundary = alarm_table[n]
        for successes, mass in enumerate(active[:n]):
            if mass == 0.0:
                continue
            for next_successes, branch_mass in (
                (successes, mass * (1.0 - probability)),
                (successes + 1, mass * probability),
            ):
                if boundary[next_successes]:
                    first_alarm[n] += branch_mass
                else:
                    next_active[next_successes] += branch_mass
        active = next_active

    if post_shift_p is None:
        false_alarm = sum(first_alarm[1:])
        detection = None
        delay = None
    else:
        false_alarm = sum(first_alarm[1:min(change_after, horizon) + 1])
        detection = sum(first_alarm[change_after + 1:])
        delay_mass = sum(
            (n - change_after) * first_alarm[n]
            for n in range(change_after + 1, horizon + 1)
        )
        delay = delay_mass / detection if detection > 0.0 else None
    survival = sum(active)
    return false_alarm, detection, delay, survival


def simulate_null_schedule(alarm_table, horizon, probability_at):
    """Exact DP under a predictable Bernoulli-rate schedule entirely inside the null band."""
    active = [0.0] * (horizon + 1)
    active[0] = 1.0
    first_alarm = [0.0] * (horizon + 1)

    for n in range(1, horizon + 1):
        probability = probability_at(n)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("scheduled Bernoulli probability is outside [0, 1]")
        next_active = [0.0] * (horizon + 1)
        boundary = alarm_table[n]
        for successes, mass in enumerate(active[:n]):
            if mass == 0.0:
                continue
            for next_successes, branch_mass in (
                (successes, mass * (1.0 - probability)),
                (successes + 1, mass * probability),
            ):
                if boundary[next_successes]:
                    first_alarm[n] += branch_mass
                else:
                    next_active[next_successes] += branch_mass
        active = next_active

    false_alarm = sum(first_alarm[1:])
    survival = sum(active)
    if abs(false_alarm + survival - 1.0) > 1e-9:
        raise ValueError("probability mass does not conserve in scheduled null DP")
    return false_alarm, survival


def eprocess_family_rows(tables=None):
    horizon = 1000
    if tables is None:
        tables = eprocess_family_alarm_tables(horizon)
    rows = []
    for family in EPROCESS_FAMILY_ORDER:
        for scenario in EPROCESS_FAMILY_SCENARIOS:
            metrics = simulate_from_alarm_table(tables[family], scenario, horizon)
            post_shift_p = SHIFT_PROBABILITIES[scenario]
            kind, null_alarm, detection, delay, survival = reporting_metrics(scenario, metrics)
            rows.append({
                "family": family,
                "scenario": scenario,
                "scenario_class": kind,
                "horizon": str(horizon),
                "change_after": str(CHANGE_AFTER) if post_shift_p is not None else "NA",
                "p0": f"{P0:.12f}",
                "stream_pre_p": f"{PRE_SHIFT_PROBABILITIES[scenario]:.12f}",
                "p1": f"{post_shift_p:.12f}" if post_shift_p is not None else "NA",
                "scenario_or_pre_shift_alarm_probability": f"{null_alarm:.12f}",
                "post_shift_detection_probability": as_text(detection),
                "mean_post_shift_detection_delay": as_text(delay),
                "per_cell_alpha": f"{PER_CELL_ALPHA:.12f}",
                "survival_probability": f"{survival:.12f}",
                "log_e_threshold": f"{math.log(1.0 / PER_CELL_ALPHA):.12f}",
            })
    return rows


def check_eprocess_family_csv(path):
    expected = eprocess_family_rows()
    try:
        with open(path, newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != EPROCESS_FAMILY_FIELDS:
                raise ValueError(f"e-process family header mismatch: {reader.fieldnames}")
            actual = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"cannot read e-process family CSV: {exc}") from exc
    if len(actual) != len(expected):
        raise ValueError(f"e-process family row count mismatch: expected {len(expected)}, got {len(actual)}")
    key_fields = ("family", "scenario", "horizon")
    text_fields = (
        "family", "scenario", "scenario_class", "horizon", "change_after", "p0", "stream_pre_p", "p1",
        "post_shift_detection_probability", "mean_post_shift_detection_delay",
    )
    numeric_fields = (
        "scenario_or_pre_shift_alarm_probability", "per_cell_alpha", "survival_probability", "log_e_threshold",
    )
    for line, (got, want) in enumerate(zip(actual, expected), 2):
        if tuple(got[k] for k in key_fields) != tuple(want[k] for k in key_fields):
            raise ValueError(f"e-process family row key mismatch on line {line}")
        for field in text_fields:
            if got[field] != want[field]:
                raise ValueError(f"{field} mismatch on line {line}: {got[field]} != {want[field]}")
        for field in numeric_fields:
            if want[field] == "NA":
                if got[field] != "NA":
                    raise ValueError(f"{field} should be NA on line {line}")
                continue
            try:
                value = float(got[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid numeric {field} on line {line}") from exc
            target = float(want[field])
            if not math.isfinite(value) or abs(value - target) > 5e-10:
                raise ValueError(f"{field} mismatch on line {line}: {value} != {target}")
    print(f"PASS: independently recomputed all {len(expected)} e-process family rows")


def eprocess_family_self_test():
    tables = eprocess_family_alarm_tables()
    rows = eprocess_family_rows(tables)
    expected_count = len(EPROCESS_FAMILY_ORDER) * len(EPROCESS_FAMILY_SCENARIOS)
    if len(rows) != expected_count:
        raise ValueError(f"e-process family row count mismatch: {len(rows)} != {expected_count}")
    indexed = {(row["family"], row["scenario"]): row for row in rows}
    for family in EPROCESS_FAMILY_ORDER:
        null = indexed[(family, "stationary_null")]
        selected = indexed[(family, "selection_bias_no_environment_shift")]
        if float(null["scenario_or_pre_shift_alarm_probability"]) > PER_CELL_ALPHA + EPS:
            raise ValueError(f"{family} exceeds the anytime null alarm budget")
        if float(selected["scenario_or_pre_shift_alarm_probability"]) < 0.99:
            raise ValueError(f"{family} failed the selection-bias negative control")
        misspecified = indexed[(family, "stable_baseline_rate_048")]
        if family == COMPOSITE_FAMILY:
            if float(misspecified["scenario_or_pre_shift_alarm_probability"]) > PER_CELL_ALPHA + EPS:
                raise ValueError("composite-null e-process exceeded alpha at p=0.48 within its null interval")
            upper_edge = indexed[(family, "stable_baseline_rate_052")]
            if float(upper_edge["scenario_or_pre_shift_alarm_probability"]) > PER_CELL_ALPHA + EPS:
                raise ValueError("composite-null e-process exceeded alpha at p=0.52 within its null interval")
            outside_band = indexed[(family, "stable_baseline_rate_045")]
            if float(outside_band["scenario_or_pre_shift_alarm_probability"]) <= 0.04:
                raise ValueError("composite-null outside-band negative control did not expose baseline drift")
            within_band_transition = indexed[(family, "within_band_transition_049_to_051")]
            transition_alarm = float(within_band_transition["scenario_or_pre_shift_alarm_probability"])
            if transition_alarm > PER_CELL_ALPHA + EPS:
                raise ValueError("composite-null e-process exceeded alpha when conditional rates moved within its accepted band")
        elif float(misspecified["scenario_or_pre_shift_alarm_probability"]) <= PER_CELL_ALPHA:
            raise ValueError(f"{family} did not expose nominal-rate misspecification")
        for scenario in EPROCESS_FAMILY_SCENARIOS:
            row = indexed[(family, scenario)]
            if scenario.startswith("step_"):
                total = (
                    float(row["scenario_or_pre_shift_alarm_probability"])
                    + float(row["post_shift_detection_probability"])
                    + float(row["survival_probability"])
                )
                if abs(total - 1.0) > 1e-9:
                    raise ValueError(f"{family} probability mass does not conserve for {scenario}")
        if family == COMPOSITE_FAMILY:
            within_band = indexed[(family, "within_band_transition_049_to_051")]
            if float(within_band["scenario_or_pre_shift_alarm_probability"]) > PER_CELL_ALPHA + EPS:
                raise ValueError("composite-null process exceeded alpha on a predictable within-band schedule")

    alternating_alarm, alternating_survival = simulate_null_schedule(
        tables[COMPOSITE_FAMILY],
        1000,
        lambda n: 0.52 if n % 2 == 0 else 0.48,
    )
    if alternating_alarm > PER_CELL_ALPHA + EPS:
        raise ValueError("composite-null e-process exceeded alpha for alternating in-band rates")
    if not 0.00015 < alternating_alarm < 0.00030:
        raise ValueError("alternating in-band endpoint power/alarm witness changed unexpectedly")
    if abs(alternating_alarm + alternating_survival - 1.0) > 1e-9:
        raise ValueError("alternating in-band schedule did not conserve probability mass")

    jeffreys = float(indexed[("beta_0.5_0.5", "step_up_055")]["post_shift_detection_probability"])
    portfolio = float(indexed[("equal_portfolio_4_priors", "step_up_055")]["post_shift_detection_probability"])
    if portfolio <= jeffreys + 0.04:
        raise ValueError("fixed prior portfolio did not improve weak-shift power in this fixture")
    composite_mild = float(indexed[(COMPOSITE_FAMILY, "step_up_060")]["post_shift_detection_probability"])
    composite_large = float(indexed[(COMPOSITE_FAMILY, "step_up_080")]["post_shift_detection_probability"])
    if composite_mild < 0.70 or composite_large < 0.99:
        raise ValueError("composite-null power controls were outside their preregistered fixture gates")
    print("PASS: family row count, point/composite null controls, selection bias, within-band step and alternating schedules, mass conservation and portfolio/power gates")


def self_test():
    fixed_final = simulate("fixed_time_final_only", "stationary_null", 1000)[0]
    fixed_reused = simulate("fixed_time_reused_each_look", "stationary_null", 1000)[0]
    anytime = simulate("time_uniform_alpha_spending", "stationary_null", 1000)[0]
    beta_null = simulate("beta_mixture_e_process", "stationary_null", 1000)[0]
    if fixed_final > PER_CELL_ALPHA + EPS:
        raise ValueError(f"single final look exceeds per-cell alpha: {fixed_final}")
    if anytime > PER_CELL_ALPHA + EPS:
        raise ValueError(f"time-uniform false-alarm probability exceeds alpha: {anytime}")
    if beta_null > PER_CELL_ALPHA + EPS:
        raise ValueError(f"Beta-mixture e-process false-alarm probability exceeds alpha: {beta_null}")
    if fixed_reused <= PER_CELL_ALPHA:
        raise ValueError(f"negative control failed to expose repeated-look inflation: {fixed_reused}")

    random_sentinel = simulate(
        "beta_mixture_e_process", "stationary_null", 1000
    )[0]
    selection_biased = simulate(
        "beta_mixture_e_process", "selection_bias_no_environment_shift", 1000
    )[0]
    if random_sentinel > PER_CELL_ALPHA + EPS:
        raise ValueError(f"randomized sentinel exceeded null alpha: {random_sentinel}")
    if selection_biased < 0.50:
        raise ValueError(
            f"selection-bias negative control did not expose spurious alarms: {selection_biased}"
        )

    shifted = simulate("time_uniform_alpha_spending", "step_up_080", 500)
    beta_shifted = simulate("beta_mixture_e_process", "step_up_080", 500)
    if shifted[1] < 0.95 or shifted[2] is None:
        raise ValueError(f"time-uniform shift detection failed expected pilot gate: {shifted}")
    if beta_shifted[1] < 0.95 or beta_shifted[2] is None:
        raise ValueError(f"Beta-mixture shift detection failed expected pilot gate: {beta_shifted}")
    if beta_shifted[2] >= shifted[2]:
        raise ValueError(
            f"Beta-mixture sequence should detect earlier in this fixture: beta={beta_shifted}, hoeffding={shifted}"
        )
    if abs(math.exp(beta_mixture_log_e_value(2, 1, 0.5)) - 0.5) > 1e-10:
        raise ValueError("Beta-mixture e-value formula failed its n=2, S=1 reference")

    for boundary in BOUNDARIES:
        for horizon in HORIZONS:
            for scenario in SCENARIOS:
                false_alarm, detection, _delay, survival = simulate(boundary, scenario, horizon)
                total = false_alarm + (detection or 0.0) + survival
                if abs(total - 1.0) > 1e-9:
                    raise ValueError(f"probability mass does not sum to one: {boundary}, {scenario}, {horizon}")
    print("PASS: optional-stopping negative control, null controls, shift detection, selection-bias counterexample, formula witness, and mass conservation")


def as_text_precise(value):
    return "NA" if value is None else f"{value:.12e}"


def cusum_next_state(state, event):
    return state + CUSUM_UPPER_INCREMENT if event else max(0, state - CUSUM_LOWER_INCREMENT)


def cusum_alarm_probability(horizon, threshold, event_probability):
    """Independent exact DP for one-sided reflected CUSUM under constant event rate."""
    if threshold <= 0:
        return 1.0
    active = [0.0] * threshold
    active[0] = 1.0
    alarm = 0.0
    for n in range(1, horizon + 1):
        next_active = [0.0] * threshold
        reachable_states = min(threshold, (n - 1) * CUSUM_UPPER_INCREMENT + 1)
        for state, mass in enumerate(active[:reachable_states]):
            if mass == 0.0:
                continue
            for event, branch_mass in (
                (True, mass * event_probability),
                (False, mass * (1.0 - event_probability)),
            ):
                next_state = cusum_next_state(state, event)
                if next_state >= threshold:
                    alarm += branch_mass
                else:
                    next_active[next_state] += branch_mass
        active = next_active
    return alarm


def calibrate_cusum_threshold(horizon, alpha):
    """Smallest integer threshold whose exact p=.52 boundary alarm probability meets alpha."""
    low, high = 1, horizon * CUSUM_UPPER_INCREMENT + 1
    while low < high:
        middle = low + (high - low) // 2
        if cusum_alarm_probability(horizon, middle, COMPOSITE_NULL_UPPER) <= alpha:
            high = middle
        else:
            low = middle + 1
    return low


def simulate_cusum_scenario(direction, scenario, threshold):
    """Run one directional CUSUM exactly and snapshot the preregistered horizons."""
    horizon_max = HORIZONS[-1]
    active = [0.0] * threshold
    active[0] = 1.0
    first_alarm = [0.0] * (horizon_max + 1)
    survival_at = {}
    checkpoint_index = 0

    for n in range(1, horizon_max + 1):
        post_shift_p = SHIFT_PROBABILITIES[scenario]
        observed_p = (
            SHIFT_PROBABILITIES[scenario]
            if post_shift_p is not None and n > CHANGE_AFTER
            else PRE_SHIFT_PROBABILITIES[scenario]
        )
        event_p = observed_p if direction == "upper_cusum" else 1.0 - observed_p
        next_active = [0.0] * threshold
        reachable_states = min(threshold, (n - 1) * CUSUM_UPPER_INCREMENT + 1)
        for state, mass in enumerate(active[:reachable_states]):
            if mass == 0.0:
                continue
            for event, branch_mass in (
                (True, mass * event_p),
                (False, mass * (1.0 - event_p)),
            ):
                next_state = cusum_next_state(state, event)
                if next_state >= threshold:
                    first_alarm[n] += branch_mass
                else:
                    next_active[next_state] += branch_mass
        active = next_active

        if checkpoint_index < len(HORIZONS) and n == HORIZONS[checkpoint_index]:
            survival_at[n] = sum(active)
            checkpoint_index += 1

    rows = []
    is_target_shift = scenario_class(scenario) == "target_shift"
    change_after = CHANGE_AFTER if post_shift_p is not None else 0
    for horizon in HORIZONS:
        if is_target_shift:
            pre_alarm = sum(first_alarm[1:min(change_after, horizon) + 1])
            detection = sum(first_alarm[change_after + 1:horizon + 1])
            delay = (
                sum((n - change_after) * first_alarm[n]
                    for n in range(change_after + 1, horizon + 1)) / detection
                if detection > 0.0 else None
            )
            scenario_alarm = pre_alarm
        else:
            scenario_alarm = sum(first_alarm[1:horizon + 1])
            detection = None
            delay = None
        rows.append({
            "horizon": str(horizon),
            "scenario_or_pre_shift_alarm_probability": f"{scenario_alarm:.12f}",
            "post_shift_detection_probability": as_text_precise(detection),
            "mean_post_shift_detection_delay": as_text_precise(delay),
            "survival_probability": f"{survival_at[horizon]:.12f}",
        })
    return rows


def composite_boundary_reference_alarm_probability(horizon=1000, tables=None):
    if tables is None:
        tables = eprocess_family_alarm_tables(horizon)
    return simulate_from_alarm_table(
        tables[COMPOSITE_FAMILY], "stable_baseline_rate_052", horizon
    )[0]


def cusum_reference_rows():
    horizon = HORIZONS[-1]
    tables = eprocess_family_alarm_tables(horizon)
    matched_reference_alarm = composite_boundary_reference_alarm_probability(horizon, tables)
    threshold = calibrate_cusum_threshold(horizon, matched_reference_alarm)
    calibration_alarm = cusum_alarm_probability(horizon, threshold, COMPOSITE_NULL_UPPER)
    family_union_bound = 2.0 * calibration_alarm
    rows = []
    for direction in ("upper_cusum", "lower_cusum"):
        for scenario in SCENARIOS:
            for metrics in simulate_cusum_scenario(direction, scenario, threshold):
                post_shift_p = SHIFT_PROBABILITIES[scenario]
                rows.append({
                    "direction": direction,
                    "scenario": scenario,
                    "scenario_class": scenario_class(scenario),
                    "horizon": metrics["horizon"],
                    "change_after": str(CHANGE_AFTER) if post_shift_p is not None else "NA",
                    "p0": f"{P0:.12f}",
                    "stream_pre_p": f"{PRE_SHIFT_PROBABILITIES[scenario]:.12f}",
                    "p1": f"{post_shift_p:.12f}" if post_shift_p is not None else "NA",
                    "null_lower": f"{COMPOSITE_NULL_LOWER:.12f}",
                    "null_upper": f"{COMPOSITE_NULL_UPPER:.12f}",
                    "threshold_scaled": str(threshold),
                    "threshold_score": f"{threshold / CUSUM_SCALE:.12f}",
                    "matched_reference_alarm_probability": f"{matched_reference_alarm:.12f}",
                    "calibration_boundary_alarm_probability": f"{calibration_alarm:.12f}",
                    "family_worst_case_union_bound": f"{family_union_bound:.12f}",
                    "scenario_or_pre_shift_alarm_probability": metrics["scenario_or_pre_shift_alarm_probability"],
                    "post_shift_detection_probability": metrics["post_shift_detection_probability"],
                    "mean_post_shift_detection_delay": metrics["mean_post_shift_detection_delay"],
                    "per_cell_alpha": f"{PER_CELL_ALPHA:.12f}",
                    "survival_probability": metrics["survival_probability"],
                })
    return rows


def check_cusum_csv(path):
    expected = cusum_reference_rows()
    try:
        with open(path, newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != CUSUM_FIELDS:
                raise ValueError(f"CUSUM header mismatch: {reader.fieldnames}")
            actual = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"cannot read CUSUM output CSV: {exc}") from exc
    if len(actual) != len(expected):
        raise ValueError(f"CUSUM row count mismatch: expected {len(expected)}, got {len(actual)}")
    key_fields = ("direction", "scenario", "horizon")
    text_fields = (
        "direction", "scenario", "scenario_class", "horizon", "change_after", "p1", "threshold_scaled",
        "post_shift_detection_probability", "mean_post_shift_detection_delay",
    )
    numeric_fields = (
        "p0", "stream_pre_p", "null_lower", "null_upper", "threshold_score",
        "matched_reference_alarm_probability", "calibration_boundary_alarm_probability",
        "family_worst_case_union_bound", "scenario_or_pre_shift_alarm_probability",
        "per_cell_alpha", "survival_probability",
    )
    for line, (got, want) in enumerate(zip(actual, expected), 2):
        if tuple(got[k] for k in key_fields) != tuple(want[k] for k in key_fields):
            raise ValueError(f"CUSUM row key mismatch on line {line}")
        for field in text_fields:
            if got[field] != want[field]:
                raise ValueError(f"CUSUM {field} mismatch on line {line}: {got[field]} != {want[field]}")
        for field in numeric_fields:
            try:
                value = float(got[field])
                target = float(want[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid CUSUM numeric {field} on line {line}") from exc
            if not math.isfinite(value) or abs(value - target) > 5e-10:
                raise ValueError(f"CUSUM {field} mismatch on line {line}: {value} != {target}")

        is_target_shift = got["scenario_class"] == "target_shift"
        detection_text = got["post_shift_detection_probability"]
        delay_text = got["mean_post_shift_detection_delay"]
        if is_target_shift and detection_text == "NA":
            raise ValueError(f"CUSUM target-shift row lacks detection metric on line {line}")
        if not is_target_shift and (detection_text != "NA" or delay_text != "NA"):
            raise ValueError(f"CUSUM non-target scenario was mislabeled as a target detection on line {line}")
        alarm = float(got["scenario_or_pre_shift_alarm_probability"])
        detection = 0.0 if detection_text == "NA" else float(detection_text)
        survival = float(got["survival_probability"])
        if abs(alarm + detection + survival - 1.0) > 1e-9:
            raise ValueError(f"CUSUM probability mass does not conserve on line {line}")
        stream_p = float(got["stream_pre_p"])
        class_name = got["scenario_class"]
        if (
            class_name in ("stationary_null", "within_band_transition_control")
            or (class_name == "baseline_misspecification"
                and COMPOSITE_NULL_LOWER <= stream_p <= COMPOSITE_NULL_UPPER)
        ) and alarm > CUSUM_SIDE_ALPHA + EPS and not is_target_shift:
            raise ValueError(f"CUSUM exceeded per-side alpha on an accepted-null scenario on line {line}")
        if is_target_shift and alarm > CUSUM_SIDE_ALPHA + EPS:
            raise ValueError(f"CUSUM pre-shift alarm exceeded per-side alpha on line {line}")
    print(f"PASS: independently recomputed and semantically checked all {len(expected)} exact-DP CUSUM rows")


def cusum_self_test():
    horizon = HORIZONS[-1]
    tables = eprocess_family_alarm_tables(horizon)
    matched_reference_alarm = composite_boundary_reference_alarm_probability(horizon, tables)
    threshold = calibrate_cusum_threshold(horizon, matched_reference_alarm)
    boundary_alarm = cusum_alarm_probability(horizon, threshold, COMPOSITE_NULL_UPPER)
    previous_alarm = cusum_alarm_probability(horizon, threshold - 1, COMPOSITE_NULL_UPPER)
    if boundary_alarm > matched_reference_alarm + EPS or previous_alarm <= matched_reference_alarm - EPS:
        raise ValueError("CUSUM threshold is not the minimum threshold at the matched reference FPR")
    if 2.0 * boundary_alarm > PER_CELL_ALPHA + EPS:
        raise ValueError("CUSUM two-direction union bound exceeds the per-cell alpha allocation")
    symmetric_lower_alarm = cusum_alarm_probability(
        HORIZONS[-1], threshold, 1.0 - COMPOSITE_NULL_LOWER
    )
    if abs(boundary_alarm - symmetric_lower_alarm) > EPS:
        raise ValueError("symmetric upper/lower CUSUM calibration diverged")

    controls = (
        ("upper_cusum", "stationary_null"),
        ("lower_cusum", "stationary_null"),
        ("upper_cusum", "stable_baseline_rate_048"),
        ("upper_cusum", "stable_baseline_rate_052"),
        ("lower_cusum", "stable_baseline_rate_048"),
        ("lower_cusum", "stable_baseline_rate_052"),
        ("upper_cusum", "within_band_transition_049_to_051"),
        ("lower_cusum", "within_band_transition_049_to_051"),
    )
    for direction, scenario in controls:
        metrics = simulate_cusum_scenario(direction, scenario, threshold)[-1]
        alarm = float(metrics["scenario_or_pre_shift_alarm_probability"])
        survival = float(metrics["survival_probability"])
        if alarm > matched_reference_alarm + EPS:
            raise ValueError(f"{direction} exceeded its finite-horizon null allocation on {scenario}")
        if abs(alarm + survival - 1.0) > 1e-9:
            raise ValueError(f"{direction} did not conserve mass on {scenario}")

    up_power = float(simulate_cusum_scenario("upper_cusum", "step_up_080", threshold)[-1]["post_shift_detection_probability"])
    down_power = float(simulate_cusum_scenario("lower_cusum", "step_down_010", threshold)[-1]["post_shift_detection_probability"])
    selection_alarm = float(simulate_cusum_scenario("upper_cusum", "selection_bias_no_environment_shift", threshold)[-1]["scenario_or_pre_shift_alarm_probability"])
    if up_power < 0.99 or down_power < 0.99:
        raise ValueError(f"CUSUM large-shift power controls failed: up={up_power}, down={down_power}")
    if selection_alarm < 0.99:
        raise ValueError("CUSUM failed the selection-induced alarm negative control")
    print("PASS: exact minimum threshold, symmetric directions, within-band control, mass conservation, large-shift power, and selection-bias negative control")


def main(argv):
    try:
        if len(argv) == 2 and argv[1] == "--self-test":
            self_test()
            return 0
        if len(argv) == 2 and argv[1] == "--sentinel-budget-self-test":
            sentinel_budget_self_test()
            return 0
        if len(argv) == 3 and argv[1] == "--sentinel-budget":
            check_sentinel_budget_csv(argv[2])
            return 0
        if len(argv) == 2 and argv[1] == "--eprocess-family-self-test":
            eprocess_family_self_test()
            return 0
        if len(argv) == 3 and argv[1] == "--eprocess-family":
            check_eprocess_family_csv(argv[2])
            return 0
        if len(argv) == 2 and argv[1] == "--cusum-self-test":
            cusum_self_test()
            return 0
        if len(argv) == 3 and argv[1] == "--cusum":
            check_cusum_csv(argv[2])
            return 0
        if len(argv) == 2 and argv[1] not in ("-h", "--help"):
            check_csv(argv[1])
            return 0
        print("Usage: reference_check.py OUTPUT.csv")
        print("       reference_check.py --self-test")
        print("       reference_check.py --sentinel-budget OUTPUT.csv")
        print("       reference_check.py --sentinel-budget-self-test")
        print("       reference_check.py --eprocess-family OUTPUT.csv")
        print("       reference_check.py --eprocess-family-self-test")
        print("       reference_check.py --cusum OUTPUT.csv")
        print("       reference_check.py --cusum-self-test")
        return 2
    except (OSError, UnicodeError, csv.Error, ValueError, AssertionError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
