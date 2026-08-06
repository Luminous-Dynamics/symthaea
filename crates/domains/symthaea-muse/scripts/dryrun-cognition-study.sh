#!/usr/bin/env bash
# dryrun-cognition-study.sh — run the cognition-study pipeline end to end on
# synthetic responses, and check it can tell signal from noise.
#
# WHY THIS EXISTS
#
# The cognition-study apparatus is ~26K lines across 51 modules with a
# 120-subcommand CLI (src/bin/cognitive_study.rs). Until 2026-07-29 none of it
# had ever been executed: data/cognition-study/ holds only *.example.json
# templates, and nothing has ever been committed under it.
#
# The existing checks do not exercise it either:
#   * scripts/check-cognition-v*.sh are `test -f` and `grep -q` assertions —
#     they verify source files exist and contain certain strings;
#   * scripts/verify_cognition_study_v*.py reimplement the sealing/digest logic
#     in Python and self-test that reimplementation. Verified 2026-07-29: zero
#     of the five ever invoke the `cognitive_study` binary.
#
# So this is the first thing that runs the real chain. It is a MECHANISM check,
# not evidence about music: every response here is synthetic.
#
# WHAT IT CHECKS
#
# Two arms over identical fixtures, schedule, structural evidence, and analysis
# plan — only the listener preferences differ:
#
#   EFFECT  preference ordered Symthaea > Heuristic > RandomValid > Fixed,
#           with 20% of listeners randomised, so it is not noise-free.
#   NULL    preference independent of arm.
#
# The pipeline must ACCEPT the first and REJECT the second. A run where both
# pass (or both fail) means the apparatus is not discriminating, and this script
# exits non-zero — it fails closed rather than printing a plausible summary.
#
# USE
#   cargo build -p symthaea-muse --features theory --bin cognitive_study
#   scripts/dryrun-cognition-study.sh [path/to/cognitive_study]
set -euo pipefail
cd "$(dirname "$0")/.."  # -> symthaea-muse/

CS="${1:-${CARGO_TARGET_DIR:-../../../target}/debug/cognitive_study}"
if [[ ! -x "$CS" ]]; then
  echo "cognitive_study binary not found at: $CS" >&2
  echo "build it first: cargo build -p symthaea-muse --features theory --bin cognitive_study" >&2
  exit 2
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# ── Frozen inputs ────────────────────────────────────────────────────────────
# Mirrors the fixture shape blinded_study.rs's own tests build: the minimum
# 4 pilot + 24 confirmatory fixtures, 4 arms each = 112 presentations.
python3 - "$WORK" <<'PY'
import json, hashlib, sys
work = sys.argv[1]
D = 'a' * 64
secret = bytes([0x42] * 32)
ARMS = ['Fixed', 'RandomValid', 'Heuristic', 'Symthaea']
MIN_PILOT, MIN_CONF = 4, 24
fixtures, artifacts = [], []
for i in range(MIN_PILOT + MIN_CONF):
    key = {'fixture_id': f'fixture-{i:03d}', 'seed': i + 1}
    fixtures.append({
        'key': key, 'family_id': f'family-{i}',
        'split': 'Pilot' if i < MIN_PILOT else 'Confirmatory',
        'frozen_input_sha256': format(i + 1, '064x'), 'subject_sha256': D,
        'renderer_sha256': D, 'soundfont_sha256': D,
        'theory_constraints_sha256': D, 'tonic': 'C', 'meter': '4/4',
        'orchestration': 'piano',
    })
    artifacts += [{'key': key, 'arm': a, 'audio_sha256': D, 'recipe_sha256': D}
                  for a in ARMS]
json.dump({
    'manifest_version': 'symthaea-muse-cognition-study-v1',
    'preregistration_sha256': D, 'analysis_plan_sha256': D,
    'randomization_commitment_sha256': hashlib.sha256(secret).hexdigest(),
    'policy_versions': {a: 'policy-v1' for a in ARMS},
    'primary_endpoints': ['Preference'], 'alpha': 0.05, 'fixtures': fixtures,
}, open(f'{work}/manifest.json', 'w'), indent=1)
json.dump(artifacts, open(f'{work}/artifacts.json', 'w'), indent=1)
open(f'{work}/secret.key', 'w').write('42' * 32)
PY

echo "── schedule ──"
"$CS" validate-manifest "$WORK/manifest.json" >/dev/null
"$CS" build-schedule "$WORK/manifest.json" "$WORK/artifacts.json" "$WORK/secret.key" \
      "$WORK/schedule.json" "$WORK/codebook.json"
"$CS" validate-schedule "$WORK/manifest.json" "$WORK/schedule.json" "$WORK/codebook.json" >/dev/null
SCHED_SHA="$("$CS" digest-json "$WORK/schedule.json")"
echo "  112 presentations, blinded; schedule sha256 ${SCHED_SHA:0:16}…"

# ── Analysis plan ────────────────────────────────────────────────────────────
# The minimums below are ENFORCED by analyze (a first attempt at 2,000
# replicates / 8 listeners was rejected with TooFewRandomizationReplicates and
# TooFewListenersPerFixture) — they are the apparatus's preregistered floors,
# not tuning knobs.
python3 - "$WORK" "$("$CS" digest-json "$WORK/manifest.json")" "$SCHED_SHA" \
    "$("$CS" digest-json "$WORK/codebook.json")" <<'PY'
import json, sys
work, m, s, c = sys.argv[1:5]
json.dump({
    'analysis_version': 'symthaea-muse-confirmatory-analysis-v1',
    'manifest_sha256': m, 'schedule_sha256': s, 'codebook_sha256': c,
    'analysis_spec_sha256': 'a' * 64, 'alpha': 0.05,
    'min_confirmatory_pairs': 24, 'bootstrap_replicates': 10000,
    'randomization_replicates': 10000, 'rng_seed': 20260729,
    'minimum_listeners_per_fixture': 12, 'primary_endpoints': ['Preference'],
    'minimum_primary_endpoints_passing': 1,
}, open(f'{work}/plan.json', 'w'), indent=1)
PY

# ── Both arms ────────────────────────────────────────────────────────────────
python3 - "$WORK" "$SCHED_SHA" <<'PY'
import json, random, sys
work, sched_sha = sys.argv[1], sys.argv[2]
sched = json.load(open(f'{work}/schedule.json'))
arm_of = {e['presentation_id']: e['arm']
          for e in json.load(open(f'{work}/codebook.json'))['entries']}
byfix = {}
for p in sched['presentations']:
    byfix.setdefault(p['key']['fixture_id'], []).append(p)
structural = [{'presentation_id': p['presentation_id'], 'outcome': {
    'hard_constraints_valid': True, 'obligations_total': 4,
    'obligations_fulfilled': 4, 'voice_leading_violations': 0,
    'motif_return_similarity': 0.98, 'tonic_returned': True}}
    for p in sched['presentations']]
RANK = {'Symthaea': 0, 'Heuristic': 1, 'RandomValid': 2, 'Fixed': 3}

def blocks(effect, seed):
    random.seed(seed)
    out = []
    for li in range(12):
        for fid, pres in byfix.items():
            if effect:
                order = sorted(pres, key=lambda p: RANK[arm_of[p['presentation_id']]])
                if random.random() < 0.2:      # 20% inattentive listeners
                    random.shuffle(order)
            else:
                order = pres[:]
                random.shuffle(order)          # preference independent of arm
            out.append({
                'block_id': f'{"eff" if effect else "null"}-{li}-{fid}',
                'listener_id': f'listener-{li:02d}', 'key': pres[0]['key'],
                'status': 'Included',
                'responses': [{
                    'presentation_id': p['presentation_id'],
                    'return_recognized': r <= 2, 'development_instability': 0.1 * r,
                    'earned_recapitulation': 1.0 - 0.2 * r, 'preference_rank': r,
                    'playback_completed': True, 'attention_check_passed': True,
                    'elapsed_ms': 30000 + r * 500,
                } for r, p in enumerate(order, start=1)],
            })
    return out

for tag, effect, seed in (('effect', True, 7), ('null', False, 99)):
    json.dump({'manifest_sha256': sched['manifest_sha256'],
               'schedule_sha256': sched_sha, 'raw_evidence_sha256': '',
               'structural': structural, 'listener_blocks': blocks(effect, seed),
               'workflow_blocks': []},
              open(f'{work}/{tag}_draft.json', 'w'), indent=1)
PY

for tag in effect null; do
  echo "── $tag arm ──"
  "$CS" seal-evidence "$WORK/${tag}_draft.json" "$WORK/${tag}_evidence.json"
  "$CS" compile-evidence "$WORK/manifest.json" "$WORK/schedule.json" \
        "$WORK/codebook.json" "$WORK/${tag}_evidence.json" "$WORK/${tag}_dataset.json"
  # analyze exits non-zero only on VALIDATION issues; a legitimate negative
  # result is a successful analysis, so `|| true` here is correct.
  "$CS" analyze "$WORK/manifest.json" "$WORK/${tag}_dataset.json" \
        "$WORK/plan.json" "$WORK/${tag}_report.json" || true
done

# ── Manipulation check ───────────────────────────────────────────────────────
python3 - "$WORK" <<'PY'
import json, sys
work = sys.argv[1]
eff = json.load(open(f'{work}/effect_report.json'))
null = json.load(open(f'{work}/null_report.json'))
for tag, r in (('EFFECT', eff), ('NULL', null)):
    print(f'  {tag:6} success={r["success"]!s:5} '
          f'analysis_gate={r["analysis_gate_passed"]!s:5} '
          f'passing={r["passing_primary_endpoints"]}')
    for c in r['endpoint_conclusions'][0]['comparisons']:
        print(f'         {c["comparator"]:26} effect={c["mean_effect"]:+.4f} '
              f'CI=[{c["confidence_interval_95"][0]:+.4f},{c["confidence_interval_95"][1]:+.4f}] '
              f'holm_p={c["holm_adjusted_p"]:.5f}')
if not (eff['success'] and not null['success']):
    print('\nMANIPULATION CHECK FAILED: the pipeline did not separate '
          'an injected preference ordering from random preference.', file=sys.stderr)
    sys.exit(1)
print('\nmanipulation check passed: injected effect accepted, null rejected.')
PY
