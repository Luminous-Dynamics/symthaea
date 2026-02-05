# PyPhi Build Strategy - Priority 1.2

**Date**: December 28, 2025
**Status**: 🔄 **READY TO EXECUTE** (waiting for system resources)
**Blocker**: Multiple test processes currently consuming memory

---

## 🎯 Objective

Successfully build the PyPhi validation example that previously failed with OOM (exit code 137).

---

## 📊 Current System Status

### Memory Situation (as of 19:46)
```
Total RAM:   31 GiB
Used RAM:    29 GiB
Available:    1.4 GiB
Total Swap:  63 GiB
Used Swap:   56 GiB
Available:    7.7 GiB
```

### Running Processes
- **14+ test processes** - Dimensional sweep and topology validation (consuming 15-20 GB total)
- **3 rustc processes** - Current release build in progress (~3 GB)
- **Total memory pressure**: Very high - system heavily into swap

---

## 🎯 Build Strategy: Try Solutions A→B→C→D

### Option A: Reduce Parallel Build Jobs (RECOMMENDED)

**Command**:
```bash
export CARGO_BUILD_JOBS=1
cargo build --example pyphi_validation --features pyphi --release
```

**Rationale**:
- Simplest solution
- Reduces peak memory usage during linking
- No code changes required
- Build takes 20-30 minutes instead of 10 minutes

**Pros**:
- ✅ Simple to implement
- ✅ No code changes
- ✅ Works with current swap configuration

**Cons**:
- ⏱️ Slower build time (2-3x)
- ⚠️ May still OOM if multiple rustc processes run simultaneously

**When to use**: When system memory is available (< 25 GB used)

---

### Option B: Build in Debug Mode

**Command**:
```bash
cargo build --example pyphi_validation --features pyphi
cargo run --example pyphi_validation --features pyphi
```

**Rationale**:
- Debug builds use much less memory (~2-3 GB vs 8-10 GB)
- No optimizations = faster compilation, less memory

**Pros**:
- ✅ Much lower memory usage
- ✅ Guaranteed to succeed even with limited RAM
- ✅ Fast compilation

**Cons**:
- ❌ **~10x slower execution** (80-150 hours instead of 8-15 hours)
- ❌ Not suitable for production validation

**When to use**: If Option A fails or for quick testing

---

### Option C: Incremental Compilation

**Command**:
```bash
# Step 1: Build library first
cargo build --lib --features pyphi --release

# Step 2: Build example (reuses compiled lib)
cargo build --example pyphi_validation --features pyphi --release
```

**Rationale**:
- Split build into stages
- Incremental builds reuse previously compiled code
- Reduces peak memory usage

**Pros**:
- ✅ Divides memory load across two stages
- ✅ Library compilation reusable for other examples

**Cons**:
- ⚠️ May still OOM during example linking
- ⏱️ Slightly longer total build time

**When to use**: If Option A fails due to linking stage OOM

---

### Option D: Monitor and Kill Other Processes

**Command**:
```bash
# Check current memory usage
free -h

# Identify heavy processes
ps aux --sort=-%mem | head -20

# Kill specific test processes if needed
kill <PID>

# Then attempt Option A
export CARGO_BUILD_JOBS=1
cargo build --example pyphi_validation --features pyphi --release
```

**Rationale**:
- Free up memory by stopping non-critical processes
- System currently running 14+ test processes

**Pros**:
- ✅ Frees up 10-15 GB of RAM
- ✅ Makes Option A much more likely to succeed

**Cons**:
- ❌ Stops ongoing dimensional sweep tests
- ❌ May need to restart tests later

**When to use**: As last resort if other options fail

---

## 📋 Execution Checklist

### Pre-Build Checks
- [ ] Verify system has < 25 GB RAM used
- [ ] Check if test processes can be paused
- [ ] Ensure at least 8 GB available RAM + swap

### Build Attempt (Option A)
- [ ] Set `CARGO_BUILD_JOBS=1`
- [ ] Run `cargo build --example pyphi_validation --features pyphi --release`
- [ ] Monitor memory usage with `watch -n 1 free -h`
- [ ] If OOM (exit 137), proceed to Option B

### Build Attempt (Option B - Fallback)
- [ ] Run `cargo build --example pyphi_validation --features pyphi` (debug)
- [ ] Verify build succeeds
- [ ] Document debug mode execution time tradeoff

### Build Attempt (Option C - If B not acceptable)
- [ ] Build library: `cargo build --lib --features pyphi --release`
- [ ] Build example: `cargo build --example pyphi_validation --features pyphi --release`
- [ ] If linking OOMs, proceed to Option D

### Build Attempt (Option D - Last Resort)
- [ ] Identify test processes to kill
- [ ] Kill non-critical processes
- [ ] Retry Option A with freed memory

---

## 🔍 Previous Failure Analysis

### Original Error
```
Error: Process exited with code 137 (Out of Memory)
Command: cargo build --example pyphi_validation --features pyphi --release
```

### Root Cause
- **Parallel rustc invocations** - Multiple compilation units building simultaneously
- **Release optimization** - Heavy LTO (Link-Time Optimization) consumes 8-10 GB
- **Large dependency tree** - PyPhi + pyo3 + numpy + complex dependencies
- **System memory pressure** - Other processes consuming available RAM

### Why Option A Should Work
- Single build job = single rustc process = ~4-5 GB peak memory
- System has 7.7 GB swap available
- Total available resources: 1.4 GB RAM + 7.7 GB swap = 9.1 GB
- Build should succeed with 4-5 GB requirement

---

## ⏱️ Expected Timeline

| Option | Build Time | Execution Time | Total Time |
|--------|------------|----------------|------------|
| **A (release, serial)** | 20-30 min | 8-15 hours | 8-15.5 hours |
| **B (debug, serial)** | 5-10 min | 80-150 hours | 80-150 hours |
| **C (release, incremental)** | 25-35 min | 8-15 hours | 8-15.5 hours |

**Recommendation**: Try Option A first (best time/memory tradeoff)

---

## 🚀 Next Steps

1. **Wait for current release build to complete** (~5-10 min)
2. **Check available memory** - ideally < 25 GB used
3. **Attempt Option A** when resources available
4. **Monitor memory** during build to detect OOM early
5. **Fall back to Option B** if A fails (accept 10x slower execution)

---

## 📝 Success Criteria

- [ ] `cargo build --example pyphi_validation --features pyphi` exits with code 0
- [ ] Binary created at `target/{debug|release}/examples/pyphi_validation`
- [ ] Test run: `./target/release/examples/pyphi_validation --help` works
- [ ] Ready to execute full validation suite (Priority 1.3)

---

## 🎯 Impact on Project Timeline

### If Option A Succeeds
- ✅ Can start 160-comparison validation immediately
- ✅ Results in 8-15 hours
- ✅ On track for 2-3 week publication timeline

### If Only Option B Works (Debug Mode)
- ⚠️ Validation takes 80-150 hours (3-6 days continuous)
- ⚠️ May need to run subset of comparisons
- ⚠️ Publication timeline extends by 1-2 weeks

---

**Status**: Strategy documented, ready to execute when system resources permit.
**Next Action**: Attempt Option A when RAM usage < 25 GB.
**Fallback**: Option B (debug mode) if memory constraints persist.

