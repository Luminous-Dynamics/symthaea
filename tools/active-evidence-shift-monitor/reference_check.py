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
BUDGET_FIELDS = (
    "sentinel_fraction", "total_label_budget", "sentinel_labels", "adaptive_labels",
    "change_after_sentinel_labels", "no_target_shift_alarm_probability",
    "pre_change_false_alarm_probability", "true_shift_detection_probability",
    "mean_delay_sentinel_labels", "mean_delay_total_label_equivalent", "per_cell_alpha",
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


def simulate(boundary, scenario, horizon, change_after_override=None):
    post_shift_p = SHIFT_PROBABILITIES[scenario]
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
            try:
                value = float(got[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid numeric {field} on line {line}") from exc
            target = float(want[field])
            if not math.isfinite(value) or abs(value - target) > 5e-10:
                raise ValueError(f"{field} mismatch on line {line}: {value} != {target}")
    print(f"PASS: independently recomputed all {len(expected)} exact-DP rows")



def sentinel_budget_rows():
    """Exact frontier at fixed total labels, using scheduled randomized target audits."""
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
        pre_change_false_alarm, detection, delay, _survival = simulate(
            "beta_mixture_e_process",
            "step_up_080",
            sentinel_labels,
            change_after_override=change_after,
        )
        delay_total = delay / fraction if delay is not None else None
        rows.append({
            "sentinel_fraction": f"{fraction:.4f}",
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
        "sentinel_fraction", "total_label_budget", "sentinel_labels",
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
    if len(rows) != len(SENTINEL_FRACTIONS):
        raise ValueError("sentinel-budget frontier row count mismatch")
    no_audit = rows[0]
    five_percent = rows[1]
    ten_percent = rows[2]
    twenty_percent = rows[3]
    if float(no_audit["no_target_shift_alarm_probability"]) < 0.99:
        raise ValueError("zero-sentinel negative control did not expose selection-induced alarms")
    if float(five_percent["no_target_shift_alarm_probability"]) > PER_CELL_ALPHA + EPS:
        raise ValueError("5% randomized-sentinel null alarm exceeded alpha")
    if float(ten_percent["true_shift_detection_probability"]) < 0.95:
        raise ValueError("10% sentinel budget missed target shift detection gate")
    if float(twenty_percent["true_shift_detection_probability"]) < 0.99:
        raise ValueError("20% sentinel budget missed target shift detection gate")
    for row in rows:
        if int(row["sentinel_labels"]) + int(row["adaptive_labels"]) != TOTAL_LABEL_BUDGET:
            raise ValueError("sentinel budget frontier does not conserve labels")
    print("PASS: audit-budget conservation, zero-sentinel negative control, sentinel null and shift-power gates")


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
        if len(argv) == 2 and argv[1] not in ("-h", "--help"):
            check_csv(argv[1])
            return 0
        print("Usage: reference_check.py OUTPUT.csv")
        print("       reference_check.py --self-test")
        print("       reference_check.py --sentinel-budget OUTPUT.csv")
        print("       reference_check.py --sentinel-budget-self-test")
        return 2
    except (OSError, UnicodeError, csv.Error, ValueError, AssertionError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
