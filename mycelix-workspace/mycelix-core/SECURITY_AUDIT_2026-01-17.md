# Security Triage Report - Mycelix Ecosystem

**Date**: 2026-01-17
**Auditor**: Claude Code (automated)
**Scope**: Credential exposure, CI/CD security, key management

---

## Executive Summary

| Category | Status | Risk Level |
|----------|--------|------------|
| Hardcoded Credentials | **RESOLVED** | ✅ Low |
| CI/CD Security Bypasses | **NEEDS FIX** | ⚠️ Medium |
| Key Management | **ACCEPTABLE** | ✅ Low |
| Git History | **CLEAN** | ✅ Low |

---

## Detailed Findings

### 1. Credential Files - RESOLVED ✅

**`.env.local`** and **`.env.development`** have been properly secured:

```bash
# Before (VULNERABLE):
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1...

# After (SECURE):
NEXT_PUBLIC_SUPABASE_URL=${SUPABASE_URL:-}
NEXT_PUBLIC_SUPABASE_ANON_KEY=${SUPABASE_ANON_KEY:-}
```

**Evidence**: Files use environment variable references with proper documentation pointing to secrets managers.

**`.gitignore`** properly excludes sensitive files:
```
.env
.env.*
!.env.example
*secret*
```

### 2. CI/CD Security Bypasses - NEEDS FIX ⚠️

Multiple workflows use `continue-on-error: true` on security-critical steps:

| File | Line | Step | Risk |
|------|------|------|------|
| `mycelix-ci.yml` | 548 | `npm audit --audit-level=high` | **HIGH** |
| `mycelix-ci.yml` | 555 | `cargo audit` | **HIGH** |
| `python-ml.yml` | 277 | `pip-audit` | **HIGH** |
| `ci.yml` | 149-166 | Linting/Type checks | Medium |
| `contracts.yml` | 272 | Slither SARIF upload | Medium |

**Recommendation**: Remove `continue-on-error: true` from security audit steps. These should block PRs from merging when vulnerabilities are found.

**Proposed Fix for `mycelix-ci.yml:545-556`**:
```yaml
      - name: Audit npm packages
        run: |
          cd mycelix-workspace/sdk-ts
          npm audit --audit-level=high
        # REMOVED: continue-on-error: true

      - name: Audit Rust packages
        run: |
          cd mycelix-workspace/sdk
          cargo audit
        # REMOVED: continue-on-error: true
```

### 3. Key Management - ACCEPTABLE ✅

**Age Identity Key** (`~/.mycelix/governance_keys/.age-identity`):
- Permissions: `-rw-------` (600) - **CORRECT**
- Location: User home directory, not in repository
- Purpose: Encrypting governance signing keys

**Encrypted Keys**:
- `governance_signing_key.pem.age` - Properly age-encrypted
- `governance_verify_key.pem.age` - Properly age-encrypted

**Recommendation**: For production, migrate to hardware security module (HSM) or cloud KMS.

### 4. No Exposed Secrets in Git

Searched for JWT tokens (`eyJ...` pattern) - found only in binary files (audio test data).

---

## Action Items

### Immediate (P0)
- [ ] Remove `continue-on-error: true` from security audit steps in CI workflows

### Short-term (P1)
- [ ] Add branch protection rules requiring security checks to pass
- [ ] Configure Dependabot for automatic vulnerability alerts

### Medium-term (P2)
- [ ] Migrate governance keys to HSM or Vault
- [ ] Add SBOM generation to release workflow
- [ ] Implement Sigstore code signing

---

## Verification Commands

```bash
# Verify no hardcoded secrets in .env files
grep -rn "eyJ" /srv/luminous-dynamics/.env* 2>/dev/null | wc -l  # Should be 0

# Check .gitignore coverage
git check-ignore /srv/luminous-dynamics/.env.local  # Should print path

# Find remaining continue-on-error in security steps
grep -n "continue-on-error.*true" /srv/luminous-dynamics/.github/workflows/*.yml | grep -i audit
```

---

## Compliance Notes

- OWASP Top 10: A07:2021 - Identification and Authentication Failures - **MITIGATED**
- CIS Controls: 3.11 - Encrypt Sensitive Data at Rest - **COMPLIANT** (age encryption)
- SOC 2: CC6.1 - Logical and Physical Access Controls - **PARTIAL** (needs HSM for Type 2)
