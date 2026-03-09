# Standalone CI Triage (2026-03-09)

## Status: 93/108 jobs passing

### Green (passing)
- Format Check
- Security Audit
- Test (default features) — 4128 tests
- 21/21 single-feature tests (except voice-tts, mesh, lancedb-backend, code_generation)
- 40/40 sub-crate tests (except psych-bench, stt)
- 8/8 genesis benchmarks
- Butlin Validation (14/14)
- Integration Tests

### Red — Fixable

| Job | Root Cause | Fix |
|-----|-----------|-----|
| Clippy | `unexpected_cfgs: parallel` (3x), `f32->f32` cast (1x) | Add `[lints.rust]` check-cfg allow; remove redundant cast |
| cargo-deny | ~~`unmaintained = "warn"` invalid~~ | FIXED in latest deny.toml |

### Red — Blocked (system deps on CI)

| Job | Missing Dep | Notes |
|-----|------------|-------|
| voice-tts | libclang + espeak-rs-sys | ort 2.0 API migration done, but build needs C libs |
| vocal-tract + voice-tts | Same | Feature interaction inherits voice-tts failure |
| vision-manifold-camera | v4l2-sys-mit (libclang) | Bindgen needs LIBCLANG_PATH |
| ssm_language | libclang | mamba-ssm bindgen |
| lancedb-backend | libclang (lancedb native) | Blocked on lancedb FFI build |

### Red — Pre-existing / Known

| Job | Issue | Priority |
|-----|-------|----------|
| mesh | Missing mesh-encryption dep | Low — mesh feature rarely used |
| code_generation | Compile error in code_generator.rs | Medium — investigate |
| pathology_resilience | Test expects specific supervisor behavior | Low — behavioral test |
| web_research | Missing school_learning interaction | Low |
| psych-bench (subcrate) | Test assertion mismatch | Medium |
| stt (subcrate) | espeak-rs-sys build | Same as voice-tts |

### Action Items

1. **Quick win**: Add `[lints.rust]` for check-cfg → Clippy green
2. **Medium**: Install libclang in CI for voice-tts, vision-camera, ssm_language
3. **Low**: Fix code_generation compile error, psych-bench assertion
4. **Won't fix**: mesh-encryption (niche feature), pathology (behavioral)
