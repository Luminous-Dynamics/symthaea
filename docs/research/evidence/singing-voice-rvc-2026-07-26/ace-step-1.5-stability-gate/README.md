# ACE-Step 1.5 stability gate: isolate and attempt to fix the float16 NaN bug (2026-07-27)

Per user direction: before letting the 5/15 NaN failures from the
controlled v1-vs-1.5 replication contaminate the upcoming cover-mechanism
comparison, reproduce the failure in isolation, trace where it first
appears, and try a bounded fix.

## Step 1: is it seed-specific? No -- correction of an earlier wrong claim

The controlled-replication write-up initially claimed "seed 222 failed
on all 3 phrases, a consistently unstable seed." **This was wrong.**
`trace_nan.py` reran the exact (seed=222, "I love the summer breeze
tonight") pair in a fresh process and it **succeeded**. `repeat_seed222.py`
then ran the identical pair 6 times in one process: **5/6 succeeded, 1/6
failed** with the same NaN error. The failure is genuinely
**non-deterministic** -- consistent with GPU-kernel-execution numerical
variance occasionally pushing marginal float16 values over the overflow
edge, not a fixed property of any seed. (This correction has been
propagated back into `controlled-v1-vs-15-replication/README.md` and
persistent memory rather than left standing.)

## Step 2: localize with per-module NaN-detection hooks

`trace_nan_loop.py` registers a forward hook on every leaf submodule of
the DiT model, checking each module's output for NaN/Inf, and retries the
same (seed=222, phrase) pair until a failure is actually caught (since
most attempts succeed). Caught on attempt 2 of the loop:

```
FIRST BAD MODULE: decoder.layers.21.self_attn_norm (call #2396 of 13328)
  shape=[2, 188, 2048], dtype=torch.float16, nan=4, inf=0
  input_had_nan=False
```

`decoder.layers.21.self_attn_norm` is a `Qwen3RMSNorm` (confirmed via the
checkpoint's `modeling_acestep_v15_base.py`) -- layer 21's post/pre
self-attention normalization. Only 4 NaN values out of the full tensor,
consistent with a narrow numerical edge case rather than wholesale
corruption. **Caveat on `input_had_nan=False`**: the hook only checked
`torch.isnan()` on the input, not `torch.isinf()` -- so an Inf (not NaN)
could have already been present in the input without this check catching
it, which matters for the root-cause reasoning below.

## Step 3: hypothesized fix, tested, did NOT work -- an honest negative result

`Qwen3RMSNorm.forward()` (from `transformers`) already upcasts to
float32 for the variance/rsqrt computation -- this is not a "someone
forgot to upcast" bug. But it casts back down to float16 **before** the
final multiply by `self.weight`:

```python
# original (transformers library)
return self.weight * hidden_states.to(input_dtype)
```

Hypothesis: if the normalized value is legitimately large before scaling,
downcasting to float16 first can overflow to Inf, and `Inf * weight` can
produce NaN depending on sign/zero -- a known-fragile ordering. The more
robust pattern defers the downcast until after the weight multiply:

```python
# patched (repeat_seed222_patched.py)
return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)
```

Monkeypatched `Qwen3RMSNorm.forward` globally and reran the identical
6-trial test:

| Condition | Trials | Succeeded | Rate |
|---|---|---|---|
| Unpatched | 6 | 5 | 83% |
| Patched | 6 | 3 | 50% |

**The patch did not fix it, and the small sample even looks worse** --
though given failure rates observed across different small batches so
far range from 17% to 50% (5/15 in the original controlled batch ≈ 33%,
1/6 here ≈ 17%, 3/6 patched ≈ 50%), this specific comparison is likely
within sampling noise for an underlying rate somewhere in the 20-35%
range, not strong evidence the patch actively made things worse. What it
IS strong evidence for: **this specific fix does not reliably resolve
the issue**, and the root-cause hypothesis (early downcast before the
weight multiply) is likely incomplete or wrong.

## Where this leaves the root cause (not solved, disclosed as open)

The model uses **eager attention** ("Attempting to load model with
attention implementation: eager" -- forced on this pre-Ampere GPU per
`init_service_loader.py`'s own float16-numerical-stability logic), i.e. a
manual matmul + softmax + matmul computation rather than a fused/
optimized SDPA kernel. Manual float16 softmax is a classic source of
numerical instability (attention-score overflow before an unstabilized
exp(), or precision loss in the max-subtraction step) -- a more likely
locus for the actual poison value than the norm layer itself, which
merely inherits and (per the `input_had_nan` caveat above) does not
rule out an Inf arriving from the immediately-preceding self-attention
computation at layer 21 specifically. **Not investigated further in this
pass** -- tracing into the eager-attention implementation itself would be
the natural next step if this bug becomes load-bearing for a future
decision, but per the bounded-task framing, this stability gate stops
here with a clean negative result rather than continuing to chase it.

## What this means for the cover-mechanism comparison

Per the user's explicit instruction: **do not silently remove seed 222
(or any seed) from evaluation**. Since the failure is confirmed
non-deterministic rather than seed-specific, there is no principled seed
to exclude anyway -- any seed could hit it on a given run. The upcoming
cover-mechanism comparison should simply **record valid-render rate
directly** (retry failed renders once, log both attempts) rather than
assume any seed is safe or unsafe. **This is a documented, disclosed,
real reliability cost of ACE-Step 1.5 on this hardware class** -- not
eliminated, carried forward into the next gate's methodology and into
the audit's overall verdict.

## Filed-issue-ready summary (not yet filed upstream)

- Repo/commit: `ace-step/ACE-Step-1.5`, cloned 2026-07-27 (see this
  bundle's env.sh-adjacent setup notes in `ace-step-1.5-verification/README.md`).
- GPU/CUDA: NVIDIA GeForce RTX 2070 with Max-Q Design (Turing, sm_75,
  7.6GB reported), driver 595.84, torch 2.10.0+cu128.
- Command/params: `acestep-v15-base`, `inference_steps=32`,
  `guidance_scale=7.0`, `seed=222`, lyrics "I love the summer breeze
  tonight", 15s duration -- non-deterministic, ~20-35% failure rate
  across repeated identical calls.
- Stack trace / error: `Generation produced NaN or Inf latents
  (shape=[1, 375, 64], dtype=torch.float16, ..., nan=24000, inf=0)`.
- First-bad-module trace: `decoder.layers.21.self_attn_norm`
  (Qwen3RMSNorm), 4 NaN values in a `[2, 188, 2048]` float16 output,
  clean (non-NaN, Inf unchecked) input.
- Note: the error message's own suggested fix, `ACESTEP_DTYPE=float32`,
  does not exist anywhere in the codebase (verified via grep of the
  cloned repo) -- a real bug in the error-handling text itself.
- Attempted fix (RMSNorm downcast-ordering patch) did not resolve it.

## Files

- `trace_nan.py` -- single-attempt hook-based trace (first run happened
  to succeed, demonstrating non-determinism before the loop version was written).
- `repeat_seed222.py` -- 6x repeat of the identical (seed, phrase) pair, unpatched.
- `trace_nan_loop.py` -- retries with hooks active until a failure is caught.
- `repeat_seed222_patched.py` -- the RMSNorm-ordering patch + 6x repeat.
- `*_results.log` -- raw output for each script.
