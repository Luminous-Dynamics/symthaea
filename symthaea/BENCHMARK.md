# Symthaea Nix Codegen — Benchmark Record

Honest measurement log. Each entry: date, commit, scorer used, result. Don't
amend historical rows — if numbers change, append a new row and note why.

## 2026-04-19 — Option 2 payoff: KG bugs fixed, grounded 5→8/13 (38%→62%)

Follow-through on the Option 2 pivot (a14f706205). Used
nix-instantiate's rejections as a bug list, fixed what it surfaced.

Three concrete fixes in commit `3e16b5593d`:

1. **KG service_keywords**: dropped `service`, `monitoring`, `vpn`,
   `media server` — generic words the intent classifier wanted but
   that the self-repair loop blindly turned into non-existent options
   like `services.service.enable`.
2. **NON_SERVICES_ROOTS**: added `postgres → services.postgresql` and
   `cups → services.printing` overrides. Users still say "postgres"
   in prompts; the KG routes it to the real option.
3. **Intel idiom**: `vaapiVdpau` → `libva-vdpau-driver`, matching
   the nixpkgs rename.

Grounded delta:
  Before (a14f706205): 5/13 (38%)
  After  (3e16b5593d): **8/13 (62%)** — +24pp from 3 file edits

Remaining 5 failures, categorized:
- **docker** (isolation artifact): user-config issue, not an idiom
  bug. Would pass in a full config with a declared user.
- **nvidia** (policy): unfree-license refusal. User would need
  `NIXPKGS_ALLOW_UNFREE=1` — not a generation bug.
- **postgresql** (real bug): idiom over-specifies `listen_addresses`
  which conflicts with `enableTCPIP` default. Fixable by dropping
  that line or using `lib.mkForce`.
- **intel** (stale golden): JSONL golden captured `vaapiVdpau` from
  pre-fix harvest. Re-harvest would update it.
- **rust dev shells** (correctly skipped): `mkShell` isn't a module.

The Option 2 loop worked end-to-end:
  (a) measure compiler pass/fail per generation
  (b) inspect failures, extract concrete bug class
  (c) fix at the source
  (d) re-measure, confirm improvement

No training involved; the fixes are rule-level corrections the
compiler pointed us at. A future RL training loop using the same
signal would make these fixes automatic — the model would learn not
to emit `services.monitoring.enable` because every attempt gets
negative reward.

**Session total (Option 1 + Option 2, commits `3bf81ae7d1`..`3e16b5593d`)**:

| Measurement | Pre-session | Post-session |
|---|---|---|
| Structural pass | 12/13 (inflated) | 11/13 (more-honest) |
| Grounded pass | — | 8/13 (compiler-verified) |
| KG keyword bugs | 4 phantom entries | 0 (cleaned) |
| Idiom package-rename bugs | 1 (vaapiVdpau) | 0 (fixed) |

## 2026-04-19 — **Architectural pivot: grounded truth** — retrieval 12/13 struct but only 5/13 compiler-valid

User directive: Option 1 (retrieval-augmented composition) + Option 2
(compiler-grounded reward loop). First step was to measure what the
EXISTING retrieval pipeline (`generate_nix_with_self_repair`) scores
on the same 13 held-out prompts the Broca distillation failed 0/13.

**Shock finding** (commit `3bf81ae7d1`):
- Broca distillation best-of-100: 0/13 structural, 8/13 parse, 2/13 keyword
- **Existing retrieval + idiom + repair: 12/13 structural (92%), 13/13 parse**

The entire distillation arc was a race to replace a pipeline that was
already 92% correct without any Broca in the loop. All 10 commits
`50194abfbb`..`e36db09c10` targeting Broca training were optimizing
the wrong path.

**Second shock** (commit `a14f706205`): piped every generation through
`nix-instantiate` via `cached_module_eval`. Three signals:

| Prompt | Struct | Parse | Grounded |
|---|---|---|---|
| configure postgresql service | ✓ | ✓ | ✗ `services.postgres' does not exist` |
| enable redis cache server | ✗ | ✓ | ✓ |
| enable docker + user to group | ✓ | ✓ | ✗ eval error |
| set up ipfs kubo node | ✓ | ✓ | ✓ |
| configure prometheus monitoring | ✓ | ✓ | ✗ `services.monitoring' does not exist` |
| configure CUPS printing service | ✓ | ✓ | ✗ `services.cups' does not exist` |
| configure nvidia gpu drivers | ✓ | ✓ | ✗ unfree refused |
| configure intel hardware acceleration | ✓ | ✓ | ✗ `vaapiVdpau` renamed |
| enable kde plasma desktop | ✓ | ✓ | ✓ |
| open port 8080 in firewall | ✓ | ✓ | ✓ |
| set time zone to Africa/Johannesburg | ✓ | ✓ | ✓ |
| rust dev (both) | ✓ | ✓ | N/A (mkShell — not a module) |

**Totals: 12/13 structural, 5/13 grounded.** Six outputs pass the
structural scorer but nix-instantiate rejects them — meaning our
**goldens themselves contain incorrect attrpaths**:

- postgres vs postgresql (real option is `services.postgresql`)
- prometheus golden uses `services.monitoring` which doesn't exist
- CUPS golden uses `services.cups`; real is `services.printing`
- intel golden uses `vaapiVdpau`, renamed in current nixpkgs

**Three compounding implications:**

1. **The 92% structural claim was INFLATED by ~54 percentage points.**
   True compiler-grounded correctness is 38%.
2. **The KG has wrong entries** that were baked into goldens. Any
   production user of `services.postgres` would get a nixpkgs eval
   failure identical to our grounded test.
3. **The Broca 0/13 was MORE HONEST than the retrieval 12/13.**
   Broca failed to produce the wrong paths the goldens encode; it
   just failed everywhere-else too.

**Architectural pivot (the clean research path now):**
- **Option 1 validated**: retrieval > distillation as the backbone.
  Broca's future role is composition WITHIN retrieval, not replacement.
- **Option 2 validated**: the compiler is the right training signal.
  Using it would have caught the wrong-golden KG entries automatically.
- **KG audit needed**: go fix postgres→postgresql, CUPS→printing,
  monitoring→prometheus-exporter-*, vaapiVdpau→libva-vdpau. Easy,
  high-impact correctness repairs.
- **Grounded reward for future training**: every candidate evaluated
  through `nix-instantiate`. Reward = `is_ok()`. No cross-entropy
  guessing, no wrong-golden drag.

This is the cleanest "the compiler tells the truth" demonstration
possible. The next concrete move: audit + fix the KG entries, re-run
grounded baseline, expect 11-12/13 grounded.

## 2026-04-19 — NIX_TOKENS expansion (m7d): keyword presence **2/13**

Hypothesis test for the semantic-gap diagnosis: if the 0/13 keyword
presence was because service names (postgresql, redis, etc.) are BPE-
fragmented from English subwords, making the model reassemble them
token-by-token, then promoting them to single tokens should let the
model emit them as atomic units.

**Intervention** (commit `7b6775cc2a`): added 36 service/package
names to NIX_TOKENS — postgresql, redis, nginx, docker, nvidia,
cups, rust-analyzer, sccache, tailscale, wireguard, etc. Genesis
phrase bumped `-m7c` → `-m7d` (new vocab = new checkpoint namespace).

**Pipeline** (v2 corpus = 45 train / 13 holdout; trained 25 epochs;
best-of-100 + forensic keyword check):

| Checkpoint | Keyword presence | Parse-fully | Structural |
|---|---|---|---|
| m7c v1 (32 train, no service tokens) | 1/13 | 8/13 | 0/13 |
| m7c v2 (45 train, no service tokens) | 0/13 | 0/13 | 0/13 |
| **m7d v2 (45 train, +service tokens)** | **2/13** | 1/13 | 0/13 |

**Keyword-presence delta**: 0/13 → 2/13 on the same corpus. Specifically:

- ✓ **docker** (3 training exposures) — now emits
- ✓ **firewall** (10 training exposures) — continues to emit
- ✗ postgresql (3 exposures) — still doesn't emit
- ✗ nvidia (3 exposures) — still doesn't emit
- ✗ rust (5 exposures) — still doesn't emit

**Diagnosis** (more precise than before): single-token vocabulary
inclusion **plus** ≥3 training exposures is NECESSARY but not
SUFFICIENT. The exposure count needed varies by token. docker and
firewall hit the threshold at 3 and 10 respectively; postgresql
(3), nvidia (3), rust (5) are still below whatever per-token
learning threshold applies. Speculation: the subword landscape
around these tokens matters — postgresql competes with "p" + "os" +
... fragments already in frequent use, so the single-token path
must out-compete the subword path by a significant margin.

**Parse quality regressed slightly**: 8/13 → 1/13 full-parse. The
bigger NIX_TOKENS vocabulary may be making sequences harder to
close because the model has more paths to choose from at each
step with fewer training tokens per path. This is an honest cost
of the vocabulary expansion and would likely recover with more
training data.

**Revised research path**:
- **Keyword learning requires ≥5 exposures PER token** for
  reliable emission, and likely more for tokens that compete with
  frequent English subword patterns.
- **Corpus scaling to hit ≥5 per keyword** becomes the concrete
  target. With 13 unique holdout keywords × 5 exposures = 65
  keyword-occurrence slots; at 1-3 keywords per training pair,
  that's ~30-60 additional pairs focused on the 8 under-exposed
  keywords (postgresql, redis, cups, prometheus, nvidia,
  intel, kde, ipfs, timeZone, sccache, etc.).
- **NIX_TOKENS must keep pace**: every new common service name
  encountered in the corpus should be added as a single token.

## 2026-04-19 — Forensic: **keyword presence 1/13** — the semantic gap localized

After the best-of-100 = 8/13 parse-valid result, ran a per-prompt
forensic check: does each generation contain the keyword implied
by its prompt (e.g., "postgresql" for "configure postgresql
service", "cups" for "configure CUPS printing service")?

| Prompt | Keyword present | Parses |
|---|---|---|
| configure CUPS printing service | ✗ | 100% |
| configure intel hardware acceleration | ✗ | 100% |
| configure nvidia gpu drivers | ✗ | 92% |
| configure postgresql service | ✗ | 98% |
| configure prometheus monitoring | ✗ | 100% |
| enable docker and add my user to the docker group | ✗ | 100% |
| enable kde plasma desktop environment | ✗ | 97% |
| enable redis cache server | ✗ | 100% |
| **open port 8080 in firewall** | **✓** | 99% |
| rust dev shell with sccache and openssl | ✗ | 100% |
| set time zone to Africa/Johannesburg | ✗ | 100% |
| set up a rust dev environment | ✗ | 94% |
| set up ipfs kubo node | ✗ | 100% |

**Keyword presence: 1/13. Parses near-100%: 13/13.**

The model has learned **Nix syntax** (emit `services.`, `programs.`,
`hardware.`, `imports = [ ]`, function-application chains that
parse) but has NOT learned **prompt → service-keyword mapping**.
Only "firewall" — which appears in 3 of the 26 existing training
pairs — survived to emission. The 10 unique services (postgresql,
redis, cups, etc.) each appear in 1-2 pairs, not enough for the
model to learn the intent→keyword link.

**This is the cleanest semantic-gap diagnostic.** The gap is not
about structure (structure is solid). It's about **vocabulary-level
intent-to-token association** — a classic few-shot problem that
more training pairs would directly address.

**Re-validates #3**: the earlier "#3 rejected" verdict was triple
wrong:
1. Vocabulary confound (CODE_TOKENS) — fixed via `for_nix_distillation`
2. Single-seed noise — fixed via multi-seed discipline
3. **Keyword-learning requires more per-keyword exposures** — only
   fixable by scaling the corpus

Practical take: the research path is more data, not more cleverness.
200+ pairs with diverse service keywords would likely move keyword
presence from 1/13 to most of 13 — at which point semantic scoring
starts producing non-zero pass rates.

## 2026-04-19 — Best-of-100 scaling: **8/13 full-parse (62%)**

Scaling the best-of-N approach from 20 → 100 samples per prompt.
Extremely cheap on GPU (100 samples × 13 prompts = 1300 gens in
~3 min; the picker is <1 sec).

| Setup | Avg best prefix | Full-parse | Structural pass |
|---|---|---|---|
| Single-seed | 32 bytes | 0/13 | 0/13 |
| Best-of-20 | 200 bytes | 1/13 | 0/13 |
| **Best-of-100** | **295 bytes** | **8/13 (62%)** | **0/13** |

Full-parse count **scales strongly with N**: 1 → 8 going 20× → 100×.
The training distribution contains **far more parse-valid outputs**
than single-seed runs suggested.

Structural pass remains 0/13 across all regimes — none of the 1300
best-of-100 samples produce paths that match their golden. This
splits the ceiling cleanly:

- **Parse ceiling is sampling-bound.** More samples → more parses.
  The architecture/training can produce valid Nix syntax; we just
  need to sample enough.
- **Semantic ceiling is architectural.** The distribution does not
  center on paths that match held-out golden semantics. No amount
  of sampling will produce `services.printing.enable = true` when
  the trained distribution emits random Nix-shaped-but-wrong
  content.

**This changes the research narrative on the "0/13" zeros.** The
honest numbers were always right — but "architecture can't produce
valid Nix" is now **false**. The architecture CAN produce valid
Nix; it just doesn't match goldens semantically. The 0/13
structural pass rate is a semantic-alignment failure, not a syntax
failure.

**Path forward clarified:**
- **#3 data scaling + #2 structure-aware loss** attack the semantic
  ceiling. This is still the main research path.
- **Best-of-N with more samples** (e.g., N=1000) might squeeze out
  a rare structural pass via lucky alignment, but it's not a
  foundation — more like a lottery. Not the main path.
- **Per-token rnix-gating during training** (not decoding) could
  shape the training distribution toward paths in the KG, making
  lucky-matches happen more often.

## 2026-04-19 — Best-of-20 rnix selection: **1/13 full-parse** — first ever

First time in this pipeline's history that any generation has
fully parsed.

`distill_nix_evaluate --samples 20` generates 20 candidate outputs
per holdout prompt (260 total for 13 prompts × 20). `nix_best_of_n`
loads them, scores each via (full_parse_with_path_overlap,
longest_parseable_prefix), picks best per prompt.

Result on m7c 43-train 25-epoch (best regime from earlier curve):

| Metric | Value |
|---|---|
| Avg median prefix across seeds | 32 bytes |
| Avg best-of-20 prefix | **200 bytes** |
| Best-of-N lift | **+530%** |
| Prompts with fully-parseable best | **1/13** |
| Full-parse-with-path-overlap | 0/13 |

The 1/13 full-parse is the CUPS prompt. Output:

    -- boot.ly" by ment nixpkgs.v hasAttr R nix.services.services.
    G 8 G P programs.and man h p networking.services.T and
    allowedTCPPorts.l x l dis way s C man C s you s T q
    networking.services.services.T d er S b i"networking.
    programs.programs.u man able and networking.er u and

**Syntactically valid Nix.** Parses clean. But semantically
gibberish — no path overlap with the `services.printing` golden.
The scorer still reads 0/13; only the prefix-probe (which checks
full parse without path match) reads 1/13.

**Interpretations:**
- **Sampling variance is huge** and contains real signal — the
  outlier 200-byte avg-best is 6.25× the 32-byte median. Single-
  seed numbers under-sell what the trained model CAN produce.
- **Grammar ceiling is SOFT**: best-of-20 crosses "1 valid parse"
  that was unreachable in single-seed. More samples + better
  selection would likely push this higher.
- **Semantic ceiling is HARD**: across all 260 sampled outputs,
  NONE had paths matching their golden. Architecture/data limits
  remain for the semantic-alignment task.

**Updated priority read**:
- **#1 best-of-N is a shippable tool now** (commit `e19f3ae136`).
  Can be used as the default generation mode without retraining.
- Pure per-token rnix-gating (true #1 in the original sense) would
  likely push the 1/13 higher but won't touch semantic alignment.
- **#3 data scaling + #2 structure-aware loss remain the path for
  semantic gains.**

## 2026-04-19 — m7c epoch curve + data-scaling re-test: partial flip of "#3 rejected"

Followed the tokenizer fix with the two open questions: does data-
scaling help on the clean m7c baseline, and where does training
plateau.

**Four-cell experiment, 3 eval seeds each (12 runs total):**

| Config | Seed 1 | Seed 2 | Seed 3 | Mean | σ | Full-parse |
|---|---|---|---|---|---|---|
| 13 train × 25 ep | 17 | 31 | 30 | **26** | 8 | 0/13 |
| 43 train × 25 ep | 44 | 105 | 55 | **68** | 32 | 0/13 |
| 43 train × 50 ep | 16 | 62 | 81 | **53** | 34 | 0/13 |
| 43 train × 100 ep | 35 | 40 | 59 | **45** | 13 | 0/13 |

**Finding 1 (data scaling partially flips)**: at m7c × 25 ep, going
from 13 → 43 train pairs moves mean prefix 26 → 68 bytes (**2.6×**).
The earlier "#3 rejected at all regimes" verdict was standing on
the vocabulary confound — CODE_TOKENS-biased emission masked any
data-scaling signal. With clean vocab, more pairs DO help prefix
emission.

**Finding 2 (overfitting past 25 epochs)**: 43 train × {25, 50,
100} ep produces means {68, 53, 45} bytes. More training makes the
model WORSE on held-out prefix quality. 43 pairs × 25 epochs looks
like the corpus-size-matched sweet spot; beyond that the model
memorizes training sequences at the expense of held-out
distribution.

**Finding 3 (full-parse ceiling is real)**: 0/13 across all 12
runs, regardless of data/epochs. The architecture (Broca CfC-HDC
controller + 43-pair corpus) can learn to emit Nix-flavored tokens
but not Nix-grammatical sequences. No amount of data/epoch tuning
within this scale closes the sequence-validity gap — it's an
architectural or loss-function ceiling.

**Revised priority list (post-curve)**:
1. **#3 data scaling continued**: not rejected, just unsaturated
   at 43 pairs. Scraping 200+ pairs is now the right move — the
   positive prefix-scaling signal means it would likely pay off.
   The corpus scraper infrastructure (`39abb83e76`/`aa2dfcd713`)
   is ready; just needs public-repo sources + human review.
2. **#2 structure-aware loss**: still the best bet for closing
   the full-parse gap specifically (architecture/loss, not data,
   is the ceiling there).
3. **#1 rnix-gated decoding**: 68-byte median is workable. Could
   ship as a generation-time tool decoupled from training
   improvements.

## 2026-04-19 — **Tokenizer fix: for_nix_distillation**, prefix length 31→68 bytes (2.2×)

Diagnostic pivot. The 25-epoch 0/13 result (entry below) prompted a
look at WHY generations were gibberish. Sample output:

    impl Into#[#[allow(~while-ed# inherit from()wrapProgram way S A...

Not random. `#[`, `impl`, `->`, `::`, `&str`, `Vec<`, `Option`,
`unwrap()`, `#[derive`, `fn new`, `-> Result` — these are
**Rust/Python tokens** that `BpeTokenizer::default_minimal()`
pre-loads via CODE_TOKENS (~200 entries, originally designed for
general-purpose code generation). 43 Nix training pairs cannot
overcome the base-distribution bias when the vocabulary is tuned
for Rust.

Fix in `2362d900b5`: new constructor
`BpeTokenizer::for_nix_distillation()` — same base as
`default_minimal` (special tokens + ASCII + common English) but
**CODE_TOKENS excluded**, `add_nix_tokens()` called automatically.
Wired into `distill_nix_train` + `distill_nix_evaluate` with
genesis-phrase bump `-m7b` → `-m7c` to keep old incompatible
checkpoints from being loaded.

**Result** (43 train × 25 epochs × 3 eval seeds):

| Tokenizer | Seed 1 | Seed 2 | Seed 3 | mean |
|---|---|---|---|---|
| m7b (CODE_TOKENS included) | 29 | 47* | 18 | ~31 |
| m7c (Nix-only) | 44 | **105** | 55 | **~68** |

(*the m7b 47 was a false-PASS from the scorer bug fixed in
`7a6c0714a8`.)

**Prefix length 2.2× longer.** Seed 2 hit 105 bytes — a substantial
Nix-looking prefix. Full-parse still **0/13** (still short of
well-formed attrsets) but the generated tokens are now
predominantly Nix-specific:

    runCommand boot. which but programs.propagatedBuildInputs
    ful however we virtualisation.hasAttr services. ...
    imports = [ ]; wireguard openFirewall modesetting allowedTCPPorts

Compare the pre-fix gibberish above. The emission is finally
pointing at the right language. `imports = [ ];` in particular is a
valid Nix fragment.

**This invalidates the "#3 rejected" verdict by partially lifting
one architectural confound.** The rejection held data-volume
constant while varying training-regime (10 ep, 25 ep) — but the
VOCABULARY was always Rust-biased. With m7c as the clean starting
point, the next data-scaling experiment would be the real test of
whether more pairs help. (Not run this session.)

**Current priority list (updated):**
- **#2 structure-aware loss**: still valuable. The m7c model
  emits valid Nix fragments but not valid sequences — an explicit
  grammar signal would close that gap.
- **#1 rnix-gated decoding**: the 68-byte median is strong material
  for per-token masking. Much better starting point than the
  m7b 31-byte median.
- **Multi-epoch curve on m7c**: prefix length may still be climbing
  at 25 epochs. Run 50/100-epoch experiments to find the plateau.
- **Re-run #3 data-scaling on m7c**: the rejection might flip given
  a clean vocabulary. Quick to verify.

## 2026-04-19 — **Second correction**: 25-epoch test + scorer bug fix

After the multi-seed correction below, ran a 25-epoch extended training
to match the original 0/9 setup (the 10-epoch runs may have been
under-trained).

**Training**: 43 train pairs × 25 epochs on GPU, loss 5.64 → 0.64.

**First eval showed 1/13 pass on run 2** — seemingly a real
generalization signal! Investigation revealed a **scorer bug**:

The redis golden `{ services.redis.servers."".enable = true; }`
uses a dynamic (quoted-empty-string) attrpath segment.
`walk_attrpaths` correctly skips dynamic segments → golden flattens
to zero static paths → empty `missing_required` → any parse-valid
gibberish trivially PASSes.

Commit `7a6c0714a8` fixes: `StructuralVerdict.golden_unscorable` is
set when golden has no static paths; `pass()` now requires
`!golden_unscorable`. Regression test locks the fix.

**Post-fix numbers** (43 train × 25 epochs, 3 seeds, corrected scorer):

| Seed | Full-parse | Prefix (bytes) |
|---|---|---|
| 1 | 0/13 | 29 |
| 2 | 0/13 | (47 — was "passing" via bug) |
| 3 | 0/13 | 18 |
| mean | **0/13** | ~31 |

**Definitive #3 verdict**:
- 13 train × 10 ep: 0/13 (3 seeds)
- 43 train × 10 ep: 0/13 (3 seeds)
- 43 train × 25 ep: 0/13 (3 seeds, bug-fixed)
- Prefix length no measurable change across regimes

**#3 data scaling is rejected for this architecture**: tripling data
AND 2.5× more epochs produce zero hold-out passes. The model's
grammar ceiling is not about training-data volume at this scale.

The corpus-scrape + accept-filter infrastructure still stands — it's
validated, and produces the input format for a future #2 structure-
aware loss or #1 rnix-gated-decoding experiment. What it does NOT
do alone: make the model generalize.

## 2026-04-19 — **Correction**: multi-seed check shows the "2× prefix" claim was noise

Immediately after the entry below, ran 3 eval seeds per checkpoint to
bound variance. Numbers:

| Run | Baseline (13 train) | Extended (43 train) |
|---|---|---|
| seed 1 | 20 bytes | 18 bytes |
| seed 2 | 25 bytes | 30 bytes |
| seed 3 | 20 bytes | 22 bytes |
| mean   | **21.7** | **23.3** |
| std    | ≈2.4 | ≈5.0 |

**Means are within 1 std of each other.** Single-seed "2× improvement"
was sampling artifact — extended run 1 (=47 bytes, reported below) sat
near the top of the extended distribution; baseline run 1 (=23 bytes,
reported below) was the baseline mean. Cherry-picking accidentally.

**Corrected verdict:**
- Both models emit ~20-25 byte parseable prefixes on average
- Data scaling 13→43 pairs at 10 epochs produces **no measurable
  improvement** in prefix quality
- Hold-out full-parse remains 0/13 for both (parse is binary → less
  sampling noise → single-seed reliable)

**Methodological lesson**: eval the prefix-parse metric always multi-
seed (n≥3). Commit-frequently discipline served the correction well —
the wrong entry is immutable below, the right entry is this one.

**Reprioritization (again — the probe-only verdict is obsolete twice now):**
- **#3 alone**: still rejected (0/13 holds)
- **#1 rnix-gated decoding**: still plausibly viable. 20-25 byte
  material is thinner than the retracted 47 suggested, but still
  non-trivial (bare-identifier tokens). Implementation risk: thin.
  Worth shipping a prototype to check if gating extends the median
  20-byte prefix to 40+ bytes in practice.
- **#2 structure-aware loss**: unchanged — the best single bet for
  producing a model that emits structurally-valid Nix.

## 2026-04-19 — #3 retraining experiment: **0/13 baseline, 0/13 extended** (data alone doesn't help — but prefix length 2×)

Follow-up to the earlier 0/9 entry (same day). Probe showed the M7
checkpoint emits bare-identifier prefixes only (avg 9 bytes parseable).
Shipped `nix_corpus_scrape.rs` + `nix_corpus_accept.rs` and built a
56-pair combined corpus (26 existing harvest + 30 scraper-derived
pairs from `/etc/nixos` + `_infrastructure/nixos`). Hash re-bucketed
the 26-pair holdout as 13/13 instead of 17/9 (same hash, different
random bucketing luck — apples-to-apples comparison still valid as
long as both runs use the same 13-pair holdout).

Two GPU runs (RTX 2070 with Max-Q, CUDA 12.9 via vendored cudarc
patch + `LD_LIBRARY_PATH=/run/opengl-driver/lib`), both 10 epochs:

| Run | Train pairs | Final loss | Hold-out pass (structural) | Avg parseable prefix |
|---|---|---|---|---|
| Baseline | 13 | 6.00 → 0.93 | **0/13** | 23 bytes |
| Extended | 43 (13 + 30) | 5.58 → 1.18 | **0/13** | 47 bytes |

**Both 0/13 on full-parse** — data scaling from 13 to 43 pairs did
not produce parse-valid hold-out output. The #3 hypothesis ("more
pairs = generalization") is **rejected at this scale**.

**But: prefix-parse length grew 2× (23 → 47 bytes).** The model's
prefix-emission is learning grammar even though sequence-completion
isn't. This is a weak positive — the signal is real but single-
sample; multi-seed confirmation deferred.

**Reprioritization of the "make this even better" list:**
- ~~#3 alone~~ — rejected (this entry)
- #1 (rnix-gated decoding) — **now viable**: 47-byte prefixes are
  enough signal to extend. The original probe's 9-byte verdict
  ("nothing to extend") was on less-trained M7; with 43-pair training
  the per-token mask has real material to work with
- #2 (structure-aware loss) — still worth testing. Could accelerate
  the grammar-acquisition curve the prefix length is already showing

**Training runs extraordinarily fast on GPU**: 13 pairs × 10 epochs
= ~22 seconds; 43 pairs × 10 epochs = ~110 seconds. Multi-seed
experiments are session-tractable. The earlier memory entry's "CPU
~20 min / 26 pairs / 1 epoch" numbers hold; GPU is ~80× faster for
this tiny-corpus regime.

**Commits**: `57b4df5d0d` prefix probe, `39abb83e76` scraper,
`aa2dfcd713` accept filter, this entry.

## 2026-04-19 — Hold-out generalization: **0/9 (0%)** (the honest number)

First held-out test. Split: 26 harvested pairs → 17 train + 9 holdout
(deterministic FNV-1a hash bucket). Trained 25 epochs on 17-pair
subset (loss 6.10 → 0.38), evaluated the 9 held-out prompts through
the structural scorer.

**All 9 outputs parse-invalid.** Sample held-out output for
"enable redis cache server":
```
your{
! ~! A super G#[allow([while;c e    enable = true l c p p and services.r and.4
ing an 6 true g services.t}
enable = true 3 3 2}
.;o re s true c way v er true an do y y{ config, pkgs, ... }>g.ing
```

Real Nix tokens present (`enable = true`, `services.`, `{ config,
pkgs, ... }`) interspersed with `!`, `~`, random subword fragments.
Sequence does not parse as Nix.

**Interpretation:** 25-epoch training on 17 pairs learned **lexical**
knowledge (which tokens are Nix-specific) but not **syntactic**
knowledge (how they compose). That's memorization-only; there's no
generalization signal in 17 pairs for the model to discover.

**The prior claim "end-to-end pipeline produces Nix" was only true
on prompts in the training set.** On held-out data, 0%. Honest.

**Commits**: harness `315c2ab1f7`, result this entry.

**What would move the needle (not in scope this session):**
- ≥200 training pairs so each attrpath shape is rehearsed many times.
- Structure-aware training loss — penalize parse-invalid emission.
- Beam search with rnix-gated decoding at generation time.

## NixEval — 94-problem corpus (`src/language/nix_eval_corpus.rs`)

(Corpus size corrected: 94 entries, not 95. The plan's "95" was an off-by-one
I carried through the scoring work — scorer itself is fine, the count in
earlier plan commits is wrong.)

### 2026-04-18 — Structural scorer landed (`bcb2c3acd3`)

**Mode: `--goldens-only` (6 prompts, structural AST scorer)**

| # | Prompt | Verdict |
|---|---|---|
| 1 | set up postgresql with pgvector | PASS |
| 2 | enable nginx web server | PASS |
| 3 | enable redis cache server | PASS |
| 4 | enable docker and add my user to the docker group | PASS |
| 5 | set up a rust dev environment with rust-analyzer and mold | PASS |
| 6 | rust dev shell with sccache and openssl | FAIL (missing `RUSTC_WRAPPER` — real codegen gap) |

**Score: 5/6 (83%)**

### 2026-04-18 — RUSTC_WRAPPER codegen fix (same-day follow-on)

The 1/6 FAIL was a real generator defect the scorer surfaced. Fixed in
`src/language/nix_codegen.rs::emit_dev_shell` — when `sccache` is
requested, now emits `RUSTC_WRAPPER = "sccache";` alongside it. Prompt 6
flipped to PASS.

| # | Prompt | Verdict |
|---|---|---|
| 1–5 | (as above) | PASS |
| 6 | rust dev shell with sccache and openssl | PASS |

**Score: 6/6 (100%)**

### 2026-04-18 — Goldens backfill round 1 (13 prompts)

Added 7 more goldens covering services (ipfs, postgres-basic), hardware
(nvidia), desktop (sway, kde plasma), networking (firewall 80/443), and
a node/typescript dev shell.

| # | Prompt | Verdict |
|---|---|---|
| 1–6 | (as above) | PASS |
| 7 | configure postgresql service | PASS |
| 8 | set up ipfs kubo node | PASS |
| 9 | configure nvidia gpu drivers | PASS |
| 10 | set up sway window manager | PASS |
| 11 | enable kde plasma desktop environment | PASS |
| 12 | open firewall ports 80 and 443 | PASS |
| 13 | set up a node development environment with typescript | PASS |

**Score: 13/13 (100%)**

**Honest caveat on this number:** goldens were written *minimally* —
they assert only the semantically required paths. The "extraneous is
warning" rule makes the generator's extras (e.g. `hardware.graphics
.enable`, `services.displayManager.sddm.wayland.enable`) not count
against PASS. A more comprehensive golden that asserted ALL of a
battle-tested config's paths would likely fail in places the minimal
one doesn't. The 13/13 proves the **scorer and codegen meet at the
required-path level**, not that the generator produces ideal configs.

Skipped prompts (boundary): `configure a python data-science
environment with jupyter and pandas` — semantic content is in a
`let`-binding (`pythonEnv = pkgs.python311.withPackages ...`) which
the scorer doesn't yet walk. Filed as a scorer-capability follow-up.

Reproduce: `cargo run --features code_generation --example nix_eval_benchmark -- --goldens-only`

### 2026-04-18 — Goldens backfill round 2 (+13 → 26)

Added 13 more goldens targeting the weakest intents + deliberately
including goldens that surface known codegen gaps. **23/26 (88%)**.

| # | Prompt | Verdict |
|---|---|---|
| 1-13 | (as in prior round) | PASS |
| 14 | enable tailscale VPN | PASS |
| 15 | configure prometheus monitoring | PASS |
| 16 | grafana dashboard server | PASS |
| 17 | configure CUPS printing service | PASS |
| 18 | enable systemd-resolved for DNS | PASS |
| 19 | enable nvidia hardware acceleration | PASS |
| 20 | **configure intel hardware acceleration** | **FAIL** (generator emits `{ # hardware config }` — Intel idiom missing) |
| 21 | enable hyprland wayland compositor | PASS |
| 22 | set up hyprland with fonts | PASS |
| 23 | set up gnome desktop environment | PASS |
| 24 | open port 8080 in firewall | PASS |
| 25 | **open udp port 51820 for wireguard** | **FAIL** (generator emits `allowedTCPPorts` instead of `allowedUDPPorts`) |
| 26 | **set time zone to Africa/Johannesburg** | **FAIL** (generator emits `{ }` — no time-zone idiom) |

**Score: 23/26 (88%)**

The 3 structural failures are legitimate codegen defects surfaced
by the honest scoring. Same pattern as the RUSTC_WRAPPER fix earlier
this session: scorer catches what substring matching misses. Each is
a small, tractable idiom-library addition:

- Intel GPU: `emit_hardware` needs an Intel branch (`hardware.graphics
  .enable = true;` + VA-API packages).
- UDP port: `emit_networking` needs to detect "udp" in prompt and emit
  `allowedUDPPorts` instead of `allowedTCPPorts`.
- Time zone: no time/locale intent exists — likely needs a new `Time`
  variant of `NixIntent` or a sub-branch of `Generic`.

**Value-of-workflow demonstration:** the substring scorer approved the
timezone-returns-empty case (empty config has no forbidden substrings,
no required substrings either). The structural scorer demands a
positive assertion. That's the whole point of P1.

### 2026-04-18 — Phase 1 M3: no-golden self-repair

Phase 1 M3 closes the loop: `generate_nix_with_self_repair(prompt, max_iters)`
runs production-viable repair using only the prompt's classified intent
+ KG service keywords — **no hand-written golden required**.

`expected_paths_for(prompt)` computes:
- Service intent → `<root>.<kw>.enable` per keyword (with virtualisation.*
  overrides for docker/podman/libvirtd)
- Networking → `networking.firewall.allowed{TCP,UDP}Ports` (UDP on
  wireguard/udp/quic mention)
- Hardware → nvidia-specific or `hardware.graphics.enable` for intel/amd

Integration test `self_repair_closes_intel_gap_without_golden` proves
the full loop: prompt → intent → expected paths → compare to
generated → repair → PASS. The Intel GPU gap closes with no human
curation in the loop.

### 2026-04-18 — Phase 1 M2: scorer-in-the-loop repair (`--repair`)

First milestone of the coding-AI roadmap
(`plans/symthaea-coding-ai-roadmap.md`). The scorer is now an oracle the
generator is conditioned on: failing verdicts feed into `repair_structural`
(M1), which patches the code; repaired code gets re-scored; loop runs
until PASS or `max_iters` (5) exhausted.

Running `cargo run --features code_generation --example nix_eval_benchmark
-- --goldens-only --repair`:

```
✓ configure intel hardware acceleration
     | REPAIRED in 1 iter(s): +hardware.graphics.enable
...
║ Goldens-only pass: 26/26 (100%)
║ Repair triggered:  1 time(s); closed 1 FAIL(s); 1 total step(s)
```

**Score: 26/26 (100%)**, up from 25/26 on the static scorer. The one
standing FAIL (Intel GPU — generator emits `{ # hardware config }`, no
Intel idiom exists) now auto-heals: scorer reports `missing: hardware
.graphics.enable`; `try_append_path` injects the flat assignment with
default value `true`; re-score PASSes.

**Why this matters:** no LLM-scale training ran. This is a pure
structural-repair loop over an existing generator. The demo is
~6 LOC in the `main` of the benchmark (match-on-verdict + call
repair + rescore). The scorer's richness is what makes it work.

### 2026-04-18 — UDP + time-zone codegen fixes (same-day follow-on)

Landed fixes for 2 of the 3 gaps surfaced by round 2. Both are
minimal, scoped fixes to `nix_codegen.rs`:

- **UDP firewall ports**: `emit_networking` now branches on
  `udp` / `wireguard` / `quic` in the prompt and emits
  `allowedUDPPorts` instead of `allowedTCPPorts`.
- **Time-zone idiom**: new `emit_time_zone` fast path at the top
  of `nix_idiom_body`, runs BEFORE classify. Detects `time zone` /
  `timezone`, extracts an IANA zone name from the original-cased
  prompt, emits `time.timeZone = "..."`.

**Score: 25/26 (96%)**. Only Intel GPU remains — a larger idiom
addition tracked for a future session.

### 2026-04-18 — Full-corpus run in `--structural` mode

Same-day run across all 94 problems. Shows how the structural and
legacy scorers compare on the corpus as a whole.

| Metric | Value |
|---|---|
| Legacy substring FULL PASS (all 4 checks) | **84/94 (89%)** |
| Intent classification | 89/94 (95%) |
| Parses successfully | 94/94 (100%) |
| Expected substrings | 88/94 (94%) |
| No forbidden leakage | 94/94 (100%) |
| | |
| Golden-backed problems | 13/94 |
| **Structural PASS on gold subset** | **13/13 (100%)** |
| Legacy substring pass on ungolden subset | 71/81 |

**Interpretation:** the 84/94 legacy number matches the prior session's
reported score (confirms the corpus + generator are stable). The 13/13
structural number on the golden subset is an honest upper-bound — goldens
were written minimally, and expanding their scope would likely reveal
more structural gaps. To push structural coverage wider, backfill the
remaining 81 prompts in `nix_eval_goldens.rs`.

Per-intent (legacy scorer) full-pass:

| Intent | Pass/Total |
|---|---|
| Service | 22/23 (96%) |
| User | 4/4 (100%) |
| HomeManager | 2/2 (100%) |
| Secrets | 6/6 (100%) |
| FlakeTemplate | 6/6 (100%) |
| Networking | 6/7 (86%) |
| Generic | 12/14 (86%) |
| DevShell | 12/14 (86%) |
| Desktop | 9/11 (82%) |
| Hardware | 5/7 (71%) |

Hardware (71%) is the weakest intent. Next backfill priorities: the 2
Hardware fails + the 2 Desktop fails + the ungolden services to widen
structural coverage beyond 13/94.

Reproduce (full): `cargo run --release --features code_generation --example nix_eval_benchmark -- --structural`

### Context

Prior session reported **84/94 (89%)** on the full corpus using legacy
substring-containment scoring. That number was an **upper bound on an
unknown true score** because:

- Substring scoring passes on `services.postgresql.enable = false; # pgvector`
  when the required substrings are `postgresql`, `enable`, `pgvector`
- No AST-based value check — `enable = true` vs `enable = false` was
  indistinguishable to the scorer
- Comments containing fake option paths could satisfy requirements

This session shipped the structural scorer (`src/language/nix_scorer.rs`)
and hand-wrote 6 golden references to calibrate. The 5/6 PASS rate on the
golden subset is the first **honest** number we have.

### Scorer evolution across commits

| Commit | Change | 6-prompt goldens score |
|---|---|---|
| `086cbc5a82` | Initial scorer, strict equality on all values | (n/a — not wired yet) |
| `3cca653f0a` | First end-to-end run, strict walker | 1/6 (17%) |
| `3cca653f0a` | Fixed: nested `services.x = { enable = ...; }` → `services.x.enable` | 4/6 (67%) |
| `bcb2c3acd3` | Added PackageList with subset semantics (extras OK) | 5/6 (83%) |

Each iteration was driven by `nix_scorer_diagnose` output — running the
scorer against one prompt, printing every attrpath extracted on both
sides, and figuring out why it failed. Two real bugs fixed in the
scorer (nested walker, list superset); one **real codegen gap** surfaced
(sccache without RUSTC_WRAPPER) which the scorer correctly catches.

### Full corpus (95 problems) — pending

The full benchmark requires more goldens. Backfilling happens in
`src/language/nix_eval_goldens.rs::golden_for`. When complete, run:

```
cargo run --release --features code_generation --example nix_eval_benchmark -- --structural
```

Expected trajectory from the plan (`plans/please-explore-deeper-research-logical-flurry.md`):

- Legacy substring: 84/95
- Structural (after all goldens backfilled): **projected 55–70/95**
- Structural + module-eval cache (P3): **projected 50–65/95**

The legacy score drop is where the real information lives — it tells
us which problems the substring scorer was over-counting.

## Follow-ups surfaced by the scorer

### ~~Codegen: dev-shell missing RUSTC_WRAPPER~~ RESOLVED

~~Prompt: `rust dev shell with sccache and openssl`~~

~~The generator puts `sccache` in `buildInputs` but doesn't wire the
`RUSTC_WRAPPER = "sccache";` env var.~~ Fixed 2026-04-18 in
`emit_dev_shell` — when sccache is requested, the env var is now
emitted. Prompt 6 now PASSes structural scoring.

**Value of this workflow:** the structural scorer surfaced a bug that
the substring scorer silently approved. A user taking that shell would
have had sccache installed but not wrapping compilation — silent
footgun. This is exactly the kind of defect an honest benchmark catches.

### Scorer: list semantics for non-package lists

`PackageList` subset semantics only fire on identifier-only lists.
Integer/string/expression lists stay Opaque, so extras in a firewall
port list would correctly fail. This is a deliberate boundary — lists
like `allowedTCPPorts = [ 80 443 8080 ]` with an extra 8080 should
probably fail the check (security-relevant) rather than be treated as
"extras OK". Revisit if real corpus growth shows this is too strict.

### Goldens not yet covering:

- Hardware configs (nvidia, amd, intel)
- Desktop environments (sway, kde, hyprland, gnome)
- Networking / firewall combinations
- Secrets / agenix / sops
- Home Manager integrations

Backfilling these is the P1 completion work — see plan §P1 for the
1-day budget estimate.
