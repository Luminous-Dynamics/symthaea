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
}
PRE_SHIFT_PROBABILITIES = {scenario: P0 for scenario in SCENARIOS}
PRE_SHIFT_PROBABILITIES["selection_bias_no_environment_shift"] = 0.74
FIELDS = (
    "boundary", "scenario", "horizon", "change_after", "p0", "stream_pre_p", "p1",
    "false_alarm_probability", "detection_probability", "mean_detection_delay",
    "per_cell_alpha", "survival_probability",
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
EPROCESS_FAMILY_ORDER = (
    "beta_0.5_0.5",
    "beta_1_1",
    "beta_5_5",
    "beta_20_20",
    "equal_portfolio_4_priors",
)
EPROCESS_FAMILY_SCENARIOS = (
    "stationary_null",
    "step_up_055", "step_up_060", "step_up_070", "step_up_080", "step_up_090",
    "selection_bias_no_environment_shift",
)
EPROCESS_FAMILY_FIELDS = (
    "family", "scenario", "horizon", "change_after", "p0", "stream_pre_p", "p1",
    "false_alarm_probability", "detection_probability", "mean_detection_delay",
    "per_cell_alpha", "survival_probability", "log_e_threshold",
)
BETA_A = 0.5
BETA_B = 0.5


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
                false_alarm, detection, delay, survival = simulate(boundary, scenario, horizon)
                rows.append({
                    "boundary": boundary,
                    "scenario": scenario,
                    "horizon": str(horizon),
                    "change_after": str(CHANGE_AFTER) if SHIFT_PROBABILITIES[scenario] is not None else "NA",
                    "p0": f"{P0:.12f}",
                    "stream_pre_p": f"{PRE_SHIFT_PROBABILITIES[scenario]:.12f}",
                    "p1": f"{SHIFT_PROBABILITIES[scenario]:.12f}" if SHIFT_PROBABILITIES[scenario] is not None else "NA",
                    "false_alarm_probability": f"{false_alarm:.12f}",
                    "detection_probability": as_text(detection),
                    "mean_detection_delay": as_text(delay),
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
    text_fields = ("boundary", "scenario", "horizon", "change_after", "p1",
                   "detection_probability", "mean_detection_delay")
    numeric_fields = ("p0", "stream_pre_p", "false_alarm_probability", "per_cell_alpha", "survival_probability")
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


def eprocess_family_rows():
    horizon = 1000
    tables = eprocess_family_alarm_tables(horizon)
    rows = []
    for family in EPROCESS_FAMILY_ORDER:
        for scenario in EPROCESS_FAMILY_SCENARIOS:
            false_alarm, detection, delay, survival = simulate_from_alarm_table(
                tables[family], scenario, horizon
            )
            post_shift_p = SHIFT_PROBABILITIES[scenario]
            rows.append({
                "family": family,
                "scenario": scenario,
                "horizon": str(horizon),
                "change_after": str(CHANGE_AFTER) if post_shift_p is not None else "NA",
                "p0": f"{P0:.12f}",
                "stream_pre_p": f"{PRE_SHIFT_PROBABILITIES[scenario]:.12f}",
                "p1": f"{post_shift_p:.12f}" if post_shift_p is not None else "NA",
                "false_alarm_probability": f"{false_alarm:.12f}",
                "detection_probability": as_text(detection),
                "mean_detection_delay": as_text(delay),
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
        "family", "scenario", "horizon", "change_after", "p0", "stream_pre_p", "p1",
    )
    numeric_fields = (
        "false_alarm_probability", "detection_probability", "mean_detection_delay",
        "per_cell_alpha", "survival_probability", "log_e_threshold",
    )
    for line, (got, want) in enumerate(zip(actual, expected), 2):
        if tuple(got[k] for k in key_fields) != tuple(want[k] for k in key_fields):
            raise ValueError(f"e-process family row key mismatch on line {line}")
        for field in text_fields:
            if got[field] != want[field]:
                raise ValueError(f"{field} mismatch on line {line}: {got[field]} != {want[field]}")
        for field in numeric_fields:
            try:
                value = float(got[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid numeric {field} on line {line}") from exc
            target = float(want[field])
            if not math.isfinite(value) or abs(value - target) > 5e-10:
                raise ValueError(f"{field} mismatch on line {line}: {value} != {target}")
    print(f"PASS: independently recomputed all {len(expected)} e-process family rows")


def eprocess_family_self_test():
    rows = eprocess_family_rows()
    expected_count = len(EPROCESS_FAMILY_ORDER) * len(EPROCESS_FAMILY_SCENARIOS)
    if len(rows) != expected_count:
        raise ValueError(f"e-process family row count mismatch: {len(rows)} != {expected_count}")
    indexed = {(row["family"], row["scenario"]): row for row in rows}
    for family in EPROCESS_FAMILY_ORDER:
        null = indexed[(family, "stationary_null")]
        selected = indexed[(family, "selection_bias_no_environment_shift")]
        if float(null["false_alarm_probability"]) > PER_CELL_ALPHA + EPS:
            raise ValueError(f"{family} exceeds the anytime null alarm budget")
        if float(selected["false_alarm_probability"]) < 0.99:
            raise ValueError(f"{family} failed the selection-bias negative control")
        for scenario in EPROCESS_FAMILY_SCENARIOS:
            row = indexed[(family, scenario)]
            if scenario.startswith("step_up_"):
                total = (
                    float(row["false_alarm_probability"])
                    + float(row["detection_probability"])
                    + float(row["survival_probability"])
                )
                if abs(total - 1.0) > 1e-9:
                    raise ValueError(f"{family} probability mass does not conserve for {scenario}")
    jeffreys = float(indexed[("beta_0.5_0.5", "step_up_055")]["detection_probability"])
    portfolio = float(indexed[("equal_portfolio_4_priors", "step_up_055")]["detection_probability"])
    if portfolio <= jeffreys + 0.04:
        raise ValueError("fixed prior portfolio did not improve weak-shift power in this fixture")
    print("PASS: family row count, anytime null budget, selection-bias negative control, mass conservation and weak-shift portfolio gate")


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
        if len(argv) == 2 and argv[1] not in ("-h", "--help"):
            check_csv(argv[1])
            return 0
        print("Usage: reference_check.py OUTPUT.csv")
        print("       reference_check.py --self-test")
        print("       reference_check.py --sentinel-budget OUTPUT.csv")
        print("       reference_check.py --sentinel-budget-self-test")
        print("       reference_check.py --eprocess-family OUTPUT.csv")
        print("       reference_check.py --eprocess-family-self-test")
        return 2
    except (OSError, UnicodeError, csv.Error, ValueError, AssertionError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
