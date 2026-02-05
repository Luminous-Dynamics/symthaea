# Voice Cortex Telemetry - Observability & Tuning Guide

**Purpose**: Turn telemetry events into actionable insights about Voice Cortex behavior
**Philosophy**: "The voice watches itself to improve itself"
**Status**: Active observability framework (v0.3+)

---

## 🎯 Core Insight

We're no longer in "build more features" territory. We're in **"observe behavior, tune policy"** territory.

The Voice Cortex now has:
- Constitutional grounding (Mycelix + charters)
- Decision substrate (SwarmAdvisor)
- Resonant layer (templates + UserState + K-Index)
- Meta layer (flat mode + telemetry)

This is **paper-worthy architecture**. Now we need to *watch it live*.

---

## 📊 The Five Core Questions

### 1. Mode vs Load Sanity Check

**Question**: How often do we speak in `Coach + Macro` when cognitive load is high or context is urgent?

**Why it matters**: Macro reflection during emergencies is tone-deaf. We should almost never do this.

**Query** (conceptual):
```sql
SELECT COUNT(*)
FROM resonant_events
WHERE temporal_frame = 'Macro'
  AND (
    cognitive_load = 'High'
    OR context_kind = 'ErrorHandling'
  )
```

**Interpretation**:
- **Ideal**: <5% of all Macro events
- **Warning**: 5-15% (thresholds need tightening)
- **Red flag**: >15% (routing logic broken)

**Tuning actions**:
- Increase cognitive load threshold for Macro frame
- Add explicit "no Macro under urgency" rule
- Review UserStateInference urgency detection

---

### 2. Macro Under Urgency

**Question**: Count events where frame is Macro AND decision is AutoApply AND context was urgent.

**Why it matters**: Macro framing should *accompany* reflection, not *lead* action in urgent moments.

**Query**:
```sql
SELECT COUNT(*)
FROM resonant_events
WHERE temporal_frame = 'Macro'
  AND suggestion_decision = 'AutoApply'
  AND urgency_hint IS NOT NULL
```

**Interpretation**:
- **Ideal**: 0 events (macro never auto-applies under urgency)
- **Acceptable**: <2% of Macro events (rare edge cases)
- **Problematic**: >5% (macro is overstepping)

**Statistical note** (for N ≥ 50 Macro events):
```python
# Binomial test for violation rate
from scipy.stats import binomtest

n_macro = 100
n_violations = 3  # 3% observed

# Test H0: rate ≤ 2% vs H1: rate > 2%
result = binomtest(n_violations, n_macro, 0.02, alternative='greater')
print(f"p-value: {result.pvalue:.3f}")
# If p < 0.05, rate is significantly above threshold
```

**Tuning actions**:
- Add "no AutoApply in Macro+Urgent" constraint
- Switch to AskUser for all urgent Macro contexts
- Review SwarmAdvisor decision logic

---

### 3. Flat Mode Usage Analysis

**Question**: How often is flat mode triggered, and why?

**Breakdown**:
- Automatic (trust < 0.3)
- Explicit (`/mode flat` command)

**Query**:
```sql
SELECT
  CASE
    WHEN flat_mode = true AND trust_in_sophia < 0.3 THEN 'Auto (low trust)'
    WHEN flat_mode = true AND trust_in_sophia >= 0.3 THEN 'Explicit command'
    ELSE 'Normal mode'
  END as mode_trigger,
  COUNT(*) as event_count,
  AVG(utterance_length) as avg_length
FROM resonant_events
GROUP BY mode_trigger
```

**Interpretation** (with statistical rigor):

**Auto flat mode (trust < 0.3)**:
- **Good**: 5-10% of events (system appropriately cautious)
- **Warning**: >20% (trust model too sensitive OR actual trust issues)
- **Concern**: <1% (might be missing low-trust scenarios)

**Statistical note** (for N ≥ 100 events):
```python
# Binomial confidence interval for flat mode rate
from scipy.stats import binom

n_events = 200
n_flat_auto = 10  # 5% observed

# 95% CI
ci_lower, ci_upper = binom.interval(0.95, n_events, n_flat_auto / n_events)
# Result: [3.5%, 7.8%] - consistent with "good" range

# If N < 100, use wider margins or wait for more data
```

**Explicit flat mode**:
- Tells you: "Just the facts" user preference
- **High usage** (>15%): Consider making flat mode the default for this user
- **Low usage** (<2%): Resonant speech is well-calibrated

**Per-user calibration**:
```sql
-- Identify users who prefer flat mode
SELECT
  user_id,
  AVG(flat_mode::int) as flat_rate,
  COUNT(*) as n_events
FROM resonant_events
GROUP BY user_id
HAVING flat_rate > 0.15 AND n_events > 20
-- Recommend: Set flat as default for these users
```

**Per-cohort analysis** (cross-user patterns):
```sql
-- Compare new vs experienced users
WITH user_experience AS (
  SELECT
    user_id,
    COUNT(*) as total_events,
    CASE
      WHEN COUNT(*) < 50 THEN 'new'
      WHEN COUNT(*) < 200 THEN 'intermediate'
      ELSE 'experienced'
    END as cohort
  FROM resonant_events
  GROUP BY user_id
)
SELECT
  ue.cohort,
  AVG(re.flat_mode::int) as avg_flat_rate,
  AVG(re.trust_in_sophia) as avg_trust,
  COUNT(*) as n_events
FROM resonant_events re
JOIN user_experience ue ON re.user_id = ue.user_id
GROUP BY ue.cohort
ORDER BY ue.cohort;

-- Expected: New users may have higher flat_mode rates until trust builds
```

**Tuning actions**:
- Adjust trust threshold (currently 0.3) based on aggregate stats
- Add per-user preference for default mode
- Review suggestion rejection patterns (what's causing low trust?)
- Consider different trust thresholds for new vs experienced cohorts

---

### 4. Template Mix Distribution

**Question**: Which templates are actually being used, and is the distribution healthy?

**Query**:
```sql
SELECT
  tags,
  relationship_mode,
  temporal_frame,
  COUNT(*) as usage_count,
  AVG(utterance_length) as avg_length
FROM resonant_events
WHERE flat_mode = false
GROUP BY tags, relationship_mode, temporal_frame
ORDER BY usage_count DESC
```

**Healthy distribution** (approximate):
- **Technician + Micro + High**: 20-30% (urgent/error handling)
- **CoAuthor + Meso + Medium**: 30-40% (main work mode)
- **Coach + Macro + Low**: 10-20% (reflection)
- **Controversial**: <10% (edge cases)
- **Flat mode**: 5-15% (safety valve)

**Warning signs**:

**Over-dominance of TechnicianMicroHigh** (>40%):
- Too many things classified as emergencies
- Cognitive load threshold too low
- User might be in chronic crisis mode (need different intervention)

**Underuse of CoachMacroLow** (<5%):
- People not being invited into reflection enough
- Macro frame trigger too conservative
- K-Index integration not activating

**High controversial rate** (>15%):
- SwarmAdvisor uncertainty too common
- Need better epistemic claim validation
- Or: actually lots of genuinely contentious decisions (good!)

**Tuning actions**:
- Adjust cognitive load inference thresholds
- Review temporal frame selection logic
- Consider user-specific template preferences

---

### 5. K-Index References Frequency

**Question**: How often does the Voice Cortex use K-Index data, and do users find it helpful?

**Query**:
```sql
SELECT
  k_deltas_count > 0 as has_k_index,
  COUNT(*) as events,
  AVG(utterance_length) as avg_length
FROM resonant_events
WHERE temporal_frame = 'Macro'
GROUP BY has_k_index
```

**Also track** (requires user feedback integration):
```sql
SELECT
  k_deltas_count > 0 as has_k_index,
  AVG(user_helpful_rating) as avg_rating
FROM resonant_events e
JOIN user_feedback f ON e.event_id = f.event_id
WHERE temporal_frame = 'Macro'
GROUP BY has_k_index
```

**Interpretation**:
- **K-Index activation rate**: Should be 50-80% of Macro events (when enabled)
- **User helpfulness**: K-Index reflections should rate ≥4/5
- **Low activation**: K-Index backend not responding OR macro_enabled = false

**Tuning actions**:
- Verify K-Index HTTP client connectivity
- Adjust Macro frame frequency
- Tune K-Index delta significance thresholds
- Add more dimensions (Skills, Relationships, Ethics)

---

## 🎨 Dashboard Ideas

### Real-Time Voice Cortex Monitor

```
┌─ Voice Cortex Status ─────────────────────────┐
│ Events last hour: 47                          │
│ Flat mode: 6 (13%)                            │
│ Low trust triggers: 2                         │
│ Macro reflections: 5 (with K-Index: 4)       │
└───────────────────────────────────────────────┘

┌─ Template Distribution (Last 24h) ────────────┐
│ ████████████░░░░░░░░ Technician+Micro    35% │
│ ███████████████░░░░░ CoAuthor+Meso       42% │
│ ████░░░░░░░░░░░░░░░░ Coach+Macro         12% │
│ ██░░░░░░░░░░░░░░░░░░ Controversial        6% │
│ ██░░░░░░░░░░░░░░░░░░ Flat Mode            5% │
└───────────────────────────────────────────────┘

┌─ Trust & Load Metrics ────────────────────────┐
│ Avg trust: 0.78 ████████░░                    │
│ Avg load:  Medium ███████░░░                  │
│ Mode mismatches: 3 ⚠                          │
└───────────────────────────────────────────────┘
```

### Weekly Voice Cortex Report

```markdown
# Voice Cortex Weekly Report
**Period**: Dec 1-7, 2025
**Total events**: 324

## Highlights
✅ Flat mode usage healthy (8%)
✅ K-Index integration working (72% of Macro)
⚠️  Technician mode overused (41% - investigate)
⚠️  3 Macro events during high load

## Template Mix
- Technician + Micro: 132 (41%)
- CoAuthor + Meso: 115 (35%)
- Coach + Macro: 42 (13%)
- Controversial: 18 (6%)
- Flat mode: 17 (5%)

## Trust Analysis
- Mean trust: 0.76
- Auto flat triggers: 5
- Explicit flat: 12
- Recommendation: Trust model well-calibrated

## K-Index Usage
- Macro events with K-Index: 30/42 (71%)
- Most common dimension: Knowledge (18)
- Avg delta magnitude: +0.17
- Recommendation: Add Governance dimension tracking
```

---

## 🔧 Telemetry Collection Setup

### 1. Event Storage

**SQLite schema** (for local analysis):
```sql
CREATE TABLE resonant_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    relationship_mode TEXT NOT NULL,
    temporal_frame TEXT NOT NULL,
    cognitive_load TEXT NOT NULL,
    trust_in_sophia REAL NOT NULL,
    flat_mode BOOLEAN NOT NULL,
    suggestion_decision TEXT NOT NULL,
    arc_name TEXT,
    arc_delta REAL,
    k_deltas_count INTEGER NOT NULL,
    tags TEXT, -- JSON array
    utterance_length INTEGER NOT NULL,

    -- Optional context
    context_kind TEXT,
    urgency_hint TEXT,
    macro_enabled BOOLEAN,
    user_id TEXT,  -- For per-user/cohort analysis

    -- Indexes
    INDEX idx_timestamp (timestamp),
    INDEX idx_flat_mode (flat_mode),
    INDEX idx_frame (temporal_frame),
    INDEX idx_user_id (user_id)
);

-- Optional: Utterance text audit table (for governance queries)
-- Stored separately to respect privacy default (not logged by default)
CREATE TABLE utterance_audit (
    event_id INTEGER PRIMARY KEY,
    utterance_text TEXT NOT NULL,
    FOREIGN KEY (event_id) REFERENCES resonant_events(event_id)
);
```

**Privacy Note**:
- `utterance_text` is **NOT** logged by default in `resonant_events`
- Only `utterance_length` is captured for telemetry
- If governance/audit queries need text (e.g., checking for "should" language), store in separate `utterance_audit` table
- User must opt-in to utterance text logging

### 2. Event Collection Hook

```rust
// In your main Sophia loop
use sophia_hlb::resonant_telemetry::{compose_utterance_with_event, ResonantEvent};

let (utterance, event) = compose_utterance_with_event(&resonant_ctx, &engine);

// Store event
telemetry_db.insert_event(&event)?;

// Return utterance to user
println!("{}", utterance.text);
```

### 3. Analysis Scripts

**Python analysis** (example):
```python
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

conn = sqlite3.connect('resonant_events.db')

# Load events
events = pd.read_sql_query("SELECT * FROM resonant_events", conn)

# Template distribution
template_dist = events.groupby(['relationship_mode', 'temporal_frame']).size()
print(template_dist)

# Flat mode analysis
flat_analysis = events[events['flat_mode'] == True].groupby(
    events['trust_in_sophia'] < 0.3
).size()
print("Auto vs explicit flat mode:", flat_analysis)

# K-Index effectiveness
macro_events = events[events['temporal_frame'] == 'Macro']
print(f"K-Index usage: {(macro_events['k_deltas_count'] > 0).mean():.1%}")
```

---

## 🎯 Governance & Audit Integration

### Template Audit Pass

For each template, document:

| Template | Epistemic Alignment | Commons Alignment | Approval Status |
|----------|-------------------|------------------|----------------|
| TechnicianMicroHigh | ✅ Clear certainty | ✅ Decomposable | Approved |
| CoAuthorMesoMed | ✅ "Sound good?" | ✅ Reversible noted | Approved |
| CoachMacroLow | ⚠️  Review drivers | ✅ "Does this feel..." | Needs Review |
| Controversial | ✅ "Low certainty" | ✅ Alternatives offered | Approved |
| Flat Mode | ✅ Facts only | ✅ No narrative | Approved |

**Red flags to watch for**:
- Guilt-inducing language ("you should have...")
- FOMO patterns ("everyone else is...")
- Certainty inflation ("definitely", "obviously")
- Hidden assumptions
- Manipulation through urgency

### Charter Compliance Metrics

```sql
-- Templates using "should" language (potential guilt)
SELECT event_id, tags, utterance_length
FROM resonant_events
WHERE utterance_text LIKE '%should%'
   OR utterance_text LIKE '%ought%';

-- High certainty in low-trust scenarios
SELECT COUNT(*)
FROM resonant_events
WHERE trust_in_sophia < 0.5
  AND suggestion_decision = 'AutoApply';
```

---

## 🧪 Experience-Level Testing Protocols

### Scenario 1: Incident Drill (Technician Panic)

**Setup**:
- Simulate `ContextKind::ErrorHandling`
- High urgency hints ("urgent: fix now!")
- Multiple rapid queries

**Observe**:
- [ ] Voice stays in Technician/Micro throughout
- [ ] No macro reflections unless explicitly invited
- [ ] Flat mode never auto-triggers (trust should stay reasonable)
- [ ] Utterances are concise (<150 chars)
- [ ] Each utterance includes clear "what/why/certainty/tradeoffs"

**Expected telemetry signature**:
```
relationship_mode: Technician
temporal_frame: Micro
cognitive_load: High
trust_in_sophia: ~0.7-0.9 (not in crisis)
flat_mode: false (unless user requests)
```

### Scenario 2: Deep Work Sprint (CoAuthor Meso)

**Setup**:
- 2-3 hour work session on actual project
- Medium cognitive load
- Sequential task flow

**Observe**:
- [ ] Voice helps sequence the arc ("next step is...")
- [ ] Utterances feel like partnership, not verbose completion
- [ ] Appropriate task suggestions based on context
- [ ] No unwanted interruptions or reflections
- [ ] Tradeoffs help with prioritization

**Expected telemetry signature**:
```
relationship_mode: CoAuthor
temporal_frame: Meso
cognitive_load: Medium
trust_in_sophia: ~0.75-0.85
arc_delta: Small positive movements
```

### Scenario 3: Weekly Reflection (Coach Macro + K-Index)

**Setup**:
- Week of real logs (or simulated with K-Index data)
- Low cognitive load (dedicated reflection time)
- Coach mode

**Observe**:
- [ ] Reflections feel grounded (not hand-wavey)
- [ ] K-Index deltas are meaningful (not overfitted to noise)
- [ ] NOT guilt-inducing ("you should do more...")
- [ ] Helps choose next week's focus
- [ ] Drivers make sense ("O/R paper", "refactor X")

**Expected telemetry signature**:
```
relationship_mode: Coach
temporal_frame: Macro
cognitive_load: Low
k_deltas_count: 2-3
arc_name: Meaningful dimension
flat_mode: false (reflection wanted)
```

---

## 📋 Tuning Decision Matrix

| Observation | Likely Cause | Tuning Action |
|------------|--------------|---------------|
| Flat mode >20% | Trust too sensitive | Increase trust threshold to 0.25 |
| Macro during urgency | Frame logic broken | Add explicit urgency check |
| TechnicianMicro >40% | Load threshold too low | Increase High load threshold |
| K-Index never used | Backend issues | Check HTTP client, add retries |
| Coach <5% | Macro too rare | Lower Macro frame barrier |
| Controversial >15% | Swarm uncertainty | Review epistemic claim quality |

---

## 🚀 Next Steps

### Week 1: Baseline Collection
- [ ] Deploy telemetry to all Voice Cortex usage
- [ ] Collect 100+ events across different scenarios
- [ ] Run initial queries to establish baselines

### Week 2: Analysis & Tuning
- [ ] Generate first weekly report
- [ ] Identify top 3 anomalies
- [ ] Tune thresholds based on findings

### Week 3: Governance Review
- [ ] Template audit against charters
- [ ] Flag any manipulative patterns
- [ ] Document approval status

### Week 4: Experience Testing
- [ ] Run 3 scenario drills
- [ ] Collect qualitative feedback
- [ ] Refine templates based on UX

---

## 📈 Meta Metrics: Closing the Goodhart Loop

**Question**: Are we optimizing the wrong things? Are the telemetry metrics themselves being gamed?

**Why it matters**: Once you measure something, people (or systems) optimize for the metric, not the goal. This is Goodhart's Law, and the Voice Cortex is not immune.

### Meta-Metric 1: Disagreement Index

**Tracks**: How often does the Voice Cortex produce utterances that users *don't* follow?

```sql
-- Calculate disagreement rate per template
SELECT
  relationship_mode,
  temporal_frame,
  COUNT(*) FILTER (WHERE user_followed = false) AS disagreements,
  COUNT(*) as total,
  CAST(COUNT(*) FILTER (WHERE user_followed = false) AS FLOAT) / COUNT(*) as disagreement_rate
FROM resonant_events re
LEFT JOIN user_feedback uf ON re.event_id = uf.event_id
GROUP BY relationship_mode, temporal_frame
ORDER BY disagreement_rate DESC;

-- Expected: Some disagreement is healthy (5-15%)
-- Red flag: <2% (might be too agreeable / not offering real alternatives)
-- Red flag: >30% (suggestions are off-target)
```

**Interpretation**:
- **Too low** (<2%): Voice Cortex is just echoing user's existing intent (not adding value)
- **Healthy** (5-15%): Offering genuine alternatives, users sometimes choose differently
- **Too high** (>30%): Mis-calibrated suggestions, users reject most advice

### Meta-Metric 2: Anomaly Detection (Statistical Process Control)

**Tracks**: Are metrics suddenly changing in suspicious ways?

```sql
-- Detect sudden shifts in template usage (weekly moving average)
WITH weekly_stats AS (
  SELECT
    DATE(timestamp, 'unixepoch', 'weekday 0', '-6 days') as week_start,
    relationship_mode,
    COUNT(*) as usage_count
  FROM resonant_events
  GROUP BY week_start, relationship_mode
),
stats_with_baseline AS (
  SELECT
    *,
    AVG(usage_count) OVER (
      PARTITION BY relationship_mode
      ORDER BY week_start
      ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
    ) as baseline_avg,
    STDEV(usage_count) OVER (
      PARTITION BY relationship_mode
      ORDER BY week_start
      ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
    ) as baseline_std
  FROM weekly_stats
)
SELECT
  week_start,
  relationship_mode,
  usage_count,
  baseline_avg,
  (usage_count - baseline_avg) / NULLIF(baseline_std, 0) as z_score
FROM stats_with_baseline
WHERE ABS((usage_count - baseline_avg) / NULLIF(baseline_std, 0)) > 2.0
ORDER BY week_start DESC;

-- If z_score > 2.0: Investigate sudden change
```

**What to look for**:
- Sudden spike in Technician mode (are errors increasing, or is the classifier broken?)
- Drop in Macro usage (K-Index backend down? Macro threshold too high?)
- Increase in flat mode without trust drop (users discovered workaround?)

### Meta-Metric 3: Voice Goodhart Score

**Tracks**: Composite score combining multiple gaming signals

```python
def calculate_voice_goodhart_score(events_df):
    """
    Returns 0.0 (no gaming) to 1.0 (maximum gaming detected)
    """
    signals = []

    # Signal 1: Template mix variance collapse
    # Healthy: diverse template usage; Gaming: all one template
    template_variance = events_df.groupby('relationship_mode').size().var()
    expected_variance = 1000  # Tune based on baseline
    signals.append(max(0, 1 - template_variance / expected_variance))

    # Signal 2: Trust inflation without behavior change
    # Healthy: trust correlates with acceptance; Gaming: trust rises but rejections don't drop
    trust_acceptance_corr = events_df[['trust_in_sophia', 'user_followed']].corr().iloc[0, 1]
    signals.append(max(0, 1 - abs(trust_acceptance_corr)))  # Low correlation = suspicious

    # Signal 3: K-Index delta inflation
    # Healthy: deltas vary; Gaming: all deltas positive and similar
    if 'arc_delta' in events_df.columns:
        delta_skewness = events_df['arc_delta'].skew()
        # Skewness near 0 is suspicious (too symmetric)
        signals.append(1 - min(1, abs(delta_skewness)))

    # Signal 4: Flat mode avoidance despite low trust
    # Healthy: flat mode activates when trust < 0.3; Gaming: flat mode suppressed
    low_trust_events = events_df[events_df['trust_in_sophia'] < 0.3]
    if len(low_trust_events) > 0:
        flat_rate_when_low_trust = low_trust_events['flat_mode'].mean()
        signals.append(1 - flat_rate_when_low_trust)  # Should be high

    return sum(signals) / len(signals)

# Usage
goodhart_score = calculate_voice_goodhart_score(events_df)
if goodhart_score > 0.5:
    print("⚠️  WARNING: Possible metric gaming detected!")
```

### Meta-Metric 4: Human Review Samples

**Process**: Randomly sample 1% of events for manual review

```sql
-- Select random sample for human review (1% of events)
SELECT *
FROM resonant_events
WHERE RANDOM() % 100 = 0
LIMIT 100;
```

**Review checklist**:
- [ ] Does the template choice match the context?
- [ ] Are tradeoffs meaningful or generic?
- [ ] Is certainty calibrated (no inflation)?
- [ ] Would this utterance manipulate you?
- [ ] Is K-Index data actually relevant?

**Action**: If >10% of samples fail review, trigger full audit

### Closing the Loop

**Quarterly Goodhart Review**:
1. Run all meta-metrics
2. Review Voice Goodhart Score
3. Sample 100 events for human review
4. Check for gaming patterns (template mix collapse, trust inflation, etc.)
5. **If gaming detected**: Retrain classifiers, adjust thresholds, update charters
6. Document findings and adjustments

**The Goal**: Telemetry should reveal gaming, not enable it. These meta-metrics watch the watchers.

---

## 🎓 Meta-Level Insight

**You're now doing system science, not just software engineering.**

The Voice Cortex is a **governed, epistemic, arc-aware voice** that:
- Knows when to speak softly
- Knows when to shut up and show numbers
- Sees how long arcs are actually moving
- **Watches itself to improve itself**

This telemetry framework operationalizes that last point. You're not just building a voice - you're building a voice that **learns from its own behavior**.

---

**Status**: Active observability framework for Voice Cortex v0.3+

**Philosophy**: *"The voice that watches itself is the voice that improves itself."*

**Next**: Use these queries and dashboards to tune the constitutional AI voice into something that's not just helpful, but **trustworthy**.
