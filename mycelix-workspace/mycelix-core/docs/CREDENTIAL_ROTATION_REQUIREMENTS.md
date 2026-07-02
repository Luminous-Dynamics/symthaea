# Credential Rotation Requirements

**Date**: 2026-01-19
**Priority**: CRITICAL
**Status**: Action Required

---

## Summary

During the security review, several credentials were identified that require immediate rotation. This document outlines the affected credentials, rotation procedures, and preventive measures.

---

## Critical Credentials Requiring Rotation

### 1. Supabase Keys
**Location**: `/srv/luminous-dynamics/.env.local`
**Risk**: Database access, authentication bypass
**Action Required**: IMMEDIATE

**Rotation Steps**:
1. Log into Supabase dashboard
2. Navigate to Project Settings > API
3. Regenerate both `anon` and `service_role` keys
4. Update all deployed services with new keys
5. Verify services are functioning with new keys
6. Delete old `.env.local` and create new one with rotated keys

### 2. JWT Secret
**Location**: `/srv/luminous-dynamics/.env.development`
**Risk**: Token forgery, session hijacking
**Action Required**: IMMEDIATE

**Rotation Steps**:
1. Generate new JWT secret: `openssl rand -base64 64`
2. Update JWT_SECRET in secrets manager (not in files)
3. Deploy updated secret to all environments
4. Monitor for authentication failures (old tokens will be invalidated)

### 3. Age Identity Key
**Location**: `~/.mycelix/governance_keys/.age-identity`
**Risk**: Governance key exposure, unauthorized actions
**Action Required**: HIGH

**Rotation Steps**:
1. Generate new age identity: `age-keygen -o new-identity.txt`
2. Re-encrypt all governance documents with new key
3. Update CI/CD pipelines with new public key
4. Archive (do not delete) old encrypted files
5. Securely delete old identity key

---

## Preventive Measures

### 1. Never Commit Secrets to Git

**Pre-commit Hook** (add to `.git/hooks/pre-commit`):
```bash
#!/bin/bash
# Prevent committing files with potential secrets
PATTERNS=(
    "eyJ"           # JWT tokens
    "sk_live_"      # Stripe keys
    "AKIA"          # AWS keys
    "supabase"      # Supabase URLs with keys
    "password.*="   # Hardcoded passwords
)

for pattern in "${PATTERNS[@]}"; do
    if git diff --cached | grep -qE "$pattern"; then
        echo "ERROR: Potential secret detected matching '$pattern'"
        echo "Remove secrets before committing"
        exit 1
    fi
done
```

### 2. Use Secrets Management

**Recommended**: HashiCorp Vault, AWS Secrets Manager, or similar

```yaml
# Example: Reference secrets from vault
DATABASE_URL: ${vault:mycelix/database/url}
JWT_SECRET: ${vault:mycelix/auth/jwt_secret}
```

### 3. Environment Variable Best Practices

```bash
# .env.example (safe to commit)
DATABASE_URL=your_database_url_here
JWT_SECRET=your_jwt_secret_here

# .env.local (NEVER commit)
# Contains actual secrets, gitignored
```

### 4. Git History Cleanup

If secrets were committed, clean the git history:

```bash
# WARNING: This rewrites history - coordinate with team
# Install BFG Repo-Cleaner first

# Remove files containing secrets
bfg --delete-files .env.local
bfg --delete-files .env.development

# Alternative: Remove specific patterns
bfg --replace-text patterns.txt

# Force garbage collection
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Force push (requires team coordination)
git push --force
```

---

## Monitoring After Rotation

### Signs of Compromised Credentials

1. **Unusual API activity**
   - High volume of requests
   - Requests from unexpected IPs
   - Access patterns outside normal hours

2. **Authentication anomalies**
   - Failed logins with old tokens
   - New sessions from unknown locations
   - Multiple simultaneous sessions

3. **Database activity**
   - Unusual queries
   - Data exports
   - Schema modifications

### Alert Configuration

Set up monitoring for:
- [ ] Failed authentication attempts (threshold: 10/minute)
- [ ] API rate limit hits
- [ ] Database connection from new IPs
- [ ] Governance key usage

---

## Checklist

### Immediate (Day 1)
- [ ] Rotate Supabase keys
- [ ] Rotate JWT secret
- [ ] Rotate age identity key
- [ ] Update all deployments

### Short-term (Week 1)
- [ ] Audit git history for exposed secrets
- [ ] Implement pre-commit hooks
- [ ] Set up secrets manager
- [ ] Configure monitoring alerts

### Long-term
- [ ] Regular credential rotation schedule (quarterly)
- [ ] Automated secret scanning in CI/CD
- [ ] Security training for team

---

## Contacts

- **Security Lead**: [security@luminous-dynamics.org]
- **DevOps**: [devops@luminous-dynamics.org]
- **Incident Response**: [incident@luminous-dynamics.org]

---

## Revision History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-19 | Security Review | Initial document |

