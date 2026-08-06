# MA-001A-delta run logs — preserved raw output

Per `ALIFE_MA001A_DELTA_RERUN_PLAN_2026-07-26.md` (user-directed "Continue into MA-001A" after
MA-001R-delta reached a closed, resolved state) and the `../ma001-runs/`/`../ma001r-runs/`/
`../ma001l-runs/`/`../ma001r-delta-runs/` non-erasure precedent.

- **`ma001a-delta-confirmatory-2026-07-26.txt`** — ecological viability gate, calibration (seed
  9999), 10 confirmatory seeds (Bound/Shuffled/NoHistory), swap intervention, all under the
  validated raw-observation delta rule (`Ma001Run::new_with_delta_rule`/`run_with_delta_rule`) in
  place of each organism's default Hebbian+TD pathway. Run via
  `cargo run -p symthaea-alife --example ma001a_delta_run --release`. Build noise excluded.

**Result: NULL / architecture limitation, essentially unchanged from MA-001A's original
default-learning result.** Viability gate passes (100/100/100 alive across all 3 calibration
conditions — even better survival than the original's 98/98/100, likely because the delta rule's
own transition-matrix predictions differ from default learning's in ways that happen to favor
survival slightly, not investigated further). Freshly calibrated margin: 0.0000 (vs. the original
0.0001 — both near the floor of measurement precision). **0/10 confirmatory seeds pass** — the
same result as the original MA-001A run. Divergence means across Bound/Shuffled/NoHistory
(0.0024–0.0025) are **nearly identical in magnitude to the original default-learning run's own
numbers** (0.0023–0.0025) — swapping the learning mechanism entirely left the population-level
divergence metric essentially unchanged. Swap intervention: 27 history-following vs. 15
identity-following among 245 classifiable (of 1000 organism-instances; 755 ineligible) — a similar
lack of a clear directional pattern to the original.

**Reading**: MA-001L and MA-001R-delta proved the delta rule *can* learn the underlying
social→physical relationship, both on prerecorded data and on a real, believing `Organism`. That
capacity does not translate into population-scale policy differentiation under free action
selection. The most likely explanation, following the interpretation ladder both MA-001A's and
MA-001R's own plans set out in advance: this may be a **policy gap, not a representation gap** —
`Action::Transfer`'s pragmatic (EFE) value may rarely be favored over `Forage`/`Rest` regardless of
what the transition model has learned about Transfer's social-context-dependent outcome, meaning
the learned coupling exists in the model but is rarely, if ever, exercised by real action
selection. MA-001R's own plan explicitly deferred exactly this check (§9, "does model sensitivity
reach action selection at all") as a secondary step, never run in this research arc.

- **`ma001a-delta-action-frequency-2026-07-26.txt`** (§10) — direct test of the policy-gap
  hypothesis above, reusing `Ma001Run::analysis_counts` from a fresh calibration-seed run.

**Result: the specific hypothesis is REFUTED, not confirmed.** Transfer is selected 33.30% of the
time — essentially uniform with Forage (34.17%) and Rest (32.53%), not rare at all. Per-organism
Transfer rate is tightly clustered (0.2844–0.3656) and within-organism, across-partner spread is
negligible (max 10.67 percentage points across the whole population; only 1/100 organisms exceeds
even that). The near-exact 1/3-1/3-1/3 split is itself a disclosed finding — consistent with action
selection being close to *state-insensitive altogether* at `action_temperature: 1.0` (a softmax
whose EFE differences are small relative to temperature would produce exactly this pattern), a
more general reframing than "Transfer specifically loses."

- **`ma001a-delta-efe-probability-check-2026-07-26.txt`** (§11) — direct measurement of the actual
  post-softmax `action_probabilities` (newly exposed `OrganismTick::action_probabilities`), for
  organisms at the end of a real run, including a same-organism rich-vs-blank-context comparison.

**Result: the reframed hypothesis is CONFIRMED.** Every sampled organism's probabilities sit within
~2 percentage points of perfectly uniform (spread 0.0166–0.0192). Presenting the *same* organism
with a rich social history vs. a completely blank one shifts probabilities by only ~0.003–0.005 —
direct evidence the delta rule's proven-real learned coupling essentially never reaches the
action-selection softmax at `action_temperature: 1.0`. This fully explains the null with a precise,
mechanistic, code-verified answer: the learning mechanism works; the policy layer downstream of it
is close to state-insensitive at this configuration, so learning barely translates into behavior.
This line is now closed with a positive explanation, not an unexplained negative result.
