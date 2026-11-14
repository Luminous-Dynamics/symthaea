from __future__ import annotations
import json
from pathlib import Path
from intervention_censoring import (
    from_dict, classify_untreated_event_time, censor_method_admissible,
    intention_to_treat_summary, cohort_identifiability,
)

ROOT = Path(__file__).resolve().parent
raw = json.loads((ROOT / "fixtures.json").read_text())
episodes = [from_dict(x) for x in raw["episodes"]]
trials = [from_dict(x) for x in raw["randomized_policy_trials"]]
classes = {e.episode_id: classify_untreated_event_time(e).value for e in episodes}
censor_gates = {
    e.episode_id: {"admissible": censor_method_admissible(e)[0], "reason": censor_method_admissible(e)[1]}
    for e in episodes
    if classify_untreated_event_time(e).value == "AdministrativeRightCensorBeforeIntervention"
}
report = {
    "schema_version": "mcr2-intervention-censoring-report-v0.1",
    "fixture_count": len(episodes),
    "classifications": classes,
    "censor_method_gates": censor_gates,
    "cohort_identifiability": cohort_identifiability(episodes),
    "randomized_policy_itt": intention_to_treat_summary(trials, "reactive_safe", "predictive_safe"),
    "claim_boundary": "synthetic_protocol_reference_only",
}
(ROOT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
