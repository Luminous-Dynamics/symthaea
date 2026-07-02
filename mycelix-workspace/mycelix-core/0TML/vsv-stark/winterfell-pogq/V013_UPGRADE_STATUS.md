# Winterfell v0.13 Upgrade Status

**Date**: 2025-11-09
**Status**: In Progress - Architectural Change Needed

## Summary

Upgraded from Winterfell v0.10 to v0.13.1. API simplifications are good news, but the core challenge remains: **conditional logic requires selector columns**, not direct comparisons.

## Progress Made ✅

### Dependency Upgrade
- ✅ winterfell 0.10 → 0.13.1
- ✅ winter-math 0.10 → 0.13.1
- ✅ winter-crypto 0.10 → 0.13.1
- ✅ winter-utils 0.10 → 0.13.1
- ✅ criterion 0.5 → 0.7

### API Fixes Completed
- ✅ Removed GkrProof/GkrVerifier types (v0.13 simplified this!)
- ✅ Fixed type bounds with `where E: From<Self::BaseField>`
- ✅ Fixed field element conversions

### Error Reduction
- **v0.10**: 18+ compilation errors (many unfixable)
- **v0.13**: 18 errors (but all related to same architectural issue)

## Current Blockers ❌

**Single Root Cause**: **Conditional logic in polynomial constraints**

### The Problem

PoGQ has branching logic that doesn't translate to polynomial equations:

```rust
// ❌ Can't do this in AIR constraints:
let is_viol = if x_t < threshold { 1 } else { 0 };
viol_next = if is_viol { viol_t + 1 } else { 0 };
quar_next = if warmup { 0 } else if release { 0 } else if enter { 1 } else { quar_t };
```

**Why it fails**: `.as_int()` method doesn't exist on `E: FieldElement` - you can't extract integers from field elements in constraints.

### The Solution: Selector Columns (4-6 hours of work)

**Pattern**: Pre-compute booleans in trace builder, use them to gate constraints

**Step 1 - Expand trace to 11 columns**:
```
Current 8 columns:
0: ema_t, 1: viol_t, 2: clear_t, 3: quar_t,
4: x_t, 5: threshold, 6: beta, 7: round

Add 3 selector columns:
8: is_violation  (x_t < threshold → 1, else 0)
9: is_warmup     (round ≤ w → 1, else 0)
10: is_release   (quar==1 && clear >= m → 1, else 0)
```

**Step 2 - Enforce boolean selectors**:
```rust
// Constraint: s*(s-1) = 0  forces s ∈ {0,1}
result[5] = is_violation * (is_violation - one);
result[6] = is_warmup * (is_warmup - one);
result[7] = is_release * (is_release - one);
```

**Step 3 - Gate updates with selectors**:
```rust
// Instead of: viol_next = if is_viol { viol_t + 1 } else { 0 }
// Do: viol_next = is_violation * (viol_t + 1)
result[1] = viol_next - (is_violation * (viol_t + one));

// Instead of: clear_next = if !is_viol { clear_t + 1 } else { 0 }
// Do: clear_next = (1 - is_violation) * (clear_t + 1)
let not_violation = one - is_violation;
result[2] = clear_next - (not_violation * (clear_t + one));
```

**Step 4 - Quarantine with cascading selectors**:
```rust
// quar_next = warmup ? 0 : (release ? 0 : (enter ? 1 : quar_t))
let quar_expected =
    is_warmup * zero +                    // If warmup → 0
    (one - is_warmup) * (                 // Else:
        is_release * zero +               //   If release → 0
        (one - is_release) * (            //   Else:
            is_entering * one +           //     If entering → 1
            (one - is_entering) * quar_t  //     Else → maintain
        )
    );
result[3] = quar_next - quar_expected;
```

## Comparison: v0.10 vs v0.13

| Aspect | v0.10 | v0.13 |
|--------|-------|-------|
| **API Complexity** | More complex | ✅ Simpler (removed GKR) |
| **Type System** | Harder bounds | ✅ Clearer (where clause) |
| **Documentation** | Sparse | ✅ Better examples |
| **Our Errors** | 18 (mixed issues) | 18 (single root cause) |
| **Fix Difficulty** | High (API mismatch) | Medium (architectural) |

**Verdict**: v0.13 is the right choice, but needs selector column refactor.

## Estimated Effort

**Selector Column Implementation**: 4-6 hours

1. **Trace builder update** (1-2 hours)
   - Expand to 11 columns
   - Compute selectors in trace generation
   - Add tests for selector correctness

2. **AIR constraint rewrite** (2-3 hours)
   - Remove all `.as_int()` calls
   - Implement selector gating
   - Add boolean enforcement constraints

3. **Testing & debugging** (1 hour)
   - Run 7 integration tests
   - Fix off-by-one errors in selector logic
   - Verify outputs match vsv_core reference

## Decision: Defer or Continue?

### Option A: Continue Now (4-6 hours)
**Pros**:
- Get 3-10× speedup for paper
- Table VII-bis with dual-backend comparison
- Better technical story

**Cons**:
- 4-6 hour investment while experiments run
- Risk of unexpected issues
- Delays other paper tasks

**Recommendation**: ✅ **DO THIS**
- Experiments run for ~11 more hours
- Selector refactor is straightforward (proven pattern)
- Parallelizes perfectly with experiment time
- If done in 4 hours → test & benchmark
- If blocked → still have RISC Zero fallback

### Option B: Ship RISC Zero Only
**Pros**:
- Zero risk
- Focus on paper content
- 46.6s is publishable

**Cons**:
- Misses 3-10× optimization story
- Less impressive demo
- Paper becomes single-backend

**Recommendation**: ❌ Only if Option A hits unexpected blocker

## Recommended Action

**Next 4 hours**: Implement selector columns

**Hour 1-2**: Trace builder
```rust
// trace.rs additions
let is_violation = if x_t < public.threshold { 1 } else { 0 };
let is_warmup = if round <= public.w { 1 } else { 0 };
let is_release = if quar == 1 && clear >= public.m { 1 } else { 0 };

trace[8][i] = BaseElement::from(is_violation);
trace[9][i] = BaseElement::from(is_warmup);
trace[10][i] = BaseElement::from(is_release);
```

**Hour 3-4**: AIR constraints
```rust
// air.rs refactor
const TRACE_WIDTH: usize = 11;  // Was 8

// In evaluate_transition:
let is_violation = current[8];
let is_warmup = current[9];
let is_release = current[10];

// Boolean constraints
result[5] = is_violation * (is_violation - one);
result[6] = is_warmup * (is_warmup - one);
result[7] = is_release * (is_release - one);

// Gated updates (replace .as_int() logic)
result[1] = viol_next - (is_violation * (viol_t + one));
// etc...
```

**Hour 5-6** (if needed): Debug & test

## Success Criteria

✅ **Compilation**: `cargo build --release` succeeds
✅ **Tests**: All 7 integration tests pass
✅ **Correctness**: Outputs match vsv_core for 50 test cases
✅ **Performance**: Proving < 15s (vs 46.6s RISC Zero baseline)
✅ **Proof size**: ~200-300KB

## Rollback Plan

If blocked after 4 hours:
1. Git stash selector changes
2. Use RISC Zero data for Table VII
3. Note Winterfell as "future work" in paper
4. Resume selector work post-submission

## Files Modified So Far

```
winterfell-pogq/
├── Cargo.toml          ✅ Upgraded to v0.13.1
├── src/
│   ├── air.rs          ⚠️  Type fixes done, need selectors
│   ├── trace.rs        ⚠️  Need to add 3 selector columns
│   ├── prover.rs       ❌ Not touched yet (will need updates)
│   └── tests.rs        ✅ No changes needed
```

## Next Immediate Steps

1. **Save current state**:
   ```bash
   cd winterfell-pogq && git diff > /tmp/v013-progress.patch
   ```

2. **Update trace builder** (src/trace.rs):
   - Change `TRACE_WIDTH` 8 → 11
   - Add selector computation logic
   - Test trace generation

3. **Update AIR** (src/air.rs):
   - Replace `.as_int()` comparisons with selector gates
   - Add boolean enforcement constraints
   - Update `TRACE_WIDTH` constant

4. **Test**:
   ```bash
   cargo test -p winterfell-pogq --target-dir /tmp/wf13
   ```

5. **Benchmark** (if tests pass):
   ```bash
   ./bin/winterfell-prover prove --public test.json --witness witness.json --output proof.bin
   ```

---

**Status**: Ready to implement selector columns (estimated 4-6 hours)
**Risk Level**: Low (proven pattern, clear approach)
**Recommendation**: ✅ Proceed with selector refactor while experiments run
**Fallback**: RISC Zero data already captured and ready
