# Credential Rotation Runbook

**Phase 0 Security Remediation**
**Created**: 2026-01-18
**Priority**: CRITICAL

This runbook provides step-by-step procedures for rotating all exposed credentials identified in the security audit.

---

## Pre-Rotation Checklist

- [ ] Schedule maintenance window (recommended: low-traffic period)
- [ ] Notify team members of planned rotation
- [ ] Ensure access to all credential management systems
- [ ] Have rollback plan ready
- [ ] Test environment access verified

---

## 1. Supabase Credentials

### Affected Files
- `/srv/luminous-dynamics/.env.local`
- `/srv/luminous-dynamics/.env.development`

### Variables to Rotate
- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `NEXT_PUBLIC_SUPABASE_URL`
- `NEXT_PUBLIC_SUPABASE_ANON_KEY`

### Rotation Procedure

#### Step 1: Generate New API Keys in Supabase Dashboard
1. Log in to [Supabase Dashboard](https://app.supabase.com)
2. Navigate to your project → Settings → API
3. Click "Generate new anon key" (this will invalidate the old key)
4. Copy the new `anon` key and `URL`

#### Step 2: Update Local Configuration
```bash
# Backup current env files
cp /srv/luminous-dynamics/.env.local /srv/luminous-dynamics/.env.local.backup.$(date +%Y%m%d)
cp /srv/luminous-dynamics/.env.development /srv/luminous-dynamics/.env.development.backup.$(date +%Y%m%d)

# Edit files and replace credentials
# DO NOT commit these files to git
```

#### Step 3: Update Deployed Environments
```bash
# For production (example using fly.io secrets)
fly secrets set SUPABASE_URL="new-url" SUPABASE_ANON_KEY="new-key" -a mycelix-prod

# For staging
fly secrets set SUPABASE_URL="new-url" SUPABASE_ANON_KEY="new-key" -a mycelix-staging
```

#### Step 4: Verify
```bash
# Test Supabase connection
curl -H "apikey: YOUR_NEW_ANON_KEY" \
     -H "Authorization: Bearer YOUR_NEW_ANON_KEY" \
     "YOUR_SUPABASE_URL/rest/v1/"
```

### Rollback
If issues occur, restore from backup:
```bash
cp /srv/luminous-dynamics/.env.local.backup.YYYYMMDD /srv/luminous-dynamics/.env.local
```

---

## 2. JWT Secret

### Affected Files
- `/srv/luminous-dynamics/.env.development`

### Variables to Rotate
- `JWT_SECRET`

### Rotation Procedure

#### Step 1: Generate New Secret
```bash
# Generate a cryptographically secure 256-bit secret
openssl rand -base64 32
```

#### Step 2: Update Configuration
```bash
# Update .env.development with new JWT_SECRET
# Note: This will invalidate all existing JWT tokens
```

#### Step 3: Update All Environments
```bash
# Production
fly secrets set JWT_SECRET="new-secret-here" -a mycelix-prod

# Staging
fly secrets set JWT_SECRET="new-secret-here" -a mycelix-staging
```

#### Step 4: Verify
- Users will need to re-authenticate after JWT secret rotation
- Monitor authentication errors in logs
- Verify new tokens are being issued correctly

### Impact
- **All existing sessions will be invalidated**
- Users will need to log in again
- Schedule during low-traffic period

---

## 3. Age Identity Key (Governance)

### Affected Files
- `~/.mycelix/governance_keys/.age-identity`

### Rotation Procedure

#### Step 1: Generate New Age Identity
```bash
# Backup existing key (store securely offline)
cp ~/.mycelix/governance_keys/.age-identity ~/.mycelix/governance_keys/.age-identity.backup.$(date +%Y%m%d)

# Generate new age identity
age-keygen -o ~/.mycelix/governance_keys/.age-identity.new

# Extract public key
age-keygen -y ~/.mycelix/governance_keys/.age-identity.new
```

#### Step 2: Re-encrypt Governance Keys
```bash
cd ~/.mycelix/governance_keys

# Decrypt existing keys with old identity
age -d -i .age-identity governance_signing_key.pem.age > governance_signing_key.pem
age -d -i .age-identity governance_verify_key.pem.age > governance_verify_key.pem

# Re-encrypt with new identity
AGE_PUBLIC_KEY=$(age-keygen -y .age-identity.new)
age -r "$AGE_PUBLIC_KEY" governance_signing_key.pem > governance_signing_key.pem.age.new
age -r "$AGE_PUBLIC_KEY" governance_verify_key.pem > governance_verify_key.pem.age.new

# Securely delete plaintext keys
shred -u governance_signing_key.pem governance_verify_key.pem

# Replace old encrypted files
mv governance_signing_key.pem.age.new governance_signing_key.pem.age
mv governance_verify_key.pem.age.new governance_verify_key.pem.age
mv .age-identity.new .age-identity
```

#### Step 3: Update decrypt-key.sh
Verify the decrypt script still works:
```bash
./decrypt-key.sh
```

#### Step 4: Secure Storage
- Store backup of old identity in secure offline storage (hardware wallet, safe deposit box)
- Update any key escrow/recovery procedures
- Document new public key in team password manager

---

## 4. Git History Scrubbing

### Purpose
Remove any accidentally committed secrets from git history.

### Procedure

#### Step 1: Install BFG Repo Cleaner
```bash
# On NixOS
nix-shell -p bfg-repo-cleaner

# Or download directly
wget https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar
```

#### Step 2: Create Backup
```bash
# Full backup before scrubbing
git clone --mirror /srv/luminous-dynamics /srv/luminous-dynamics-backup-$(date +%Y%m%d)
```

#### Step 3: Scrub Secrets
```bash
cd /srv/luminous-dynamics

# Remove .env files from history
bfg --delete-files .env.local
bfg --delete-files .env.development
bfg --delete-files .env.production

# Remove files containing specific patterns
bfg --replace-text patterns.txt  # Create file with patterns to redact

# Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

#### Step 4: Force Push (CAUTION)
```bash
# Only after verification and team notification
git push --force --all
git push --force --tags
```

### patterns.txt Example
```
eyJ==>***REDACTED_JWT***
sb-==>***REDACTED_SUPABASE***
AGE-SECRET-KEY-==>***REDACTED_AGE***
```

---

## 5. Post-Rotation Verification

### Checklist
- [ ] All services start without errors
- [ ] Authentication works (test login flow)
- [ ] Supabase queries succeed
- [ ] Governance key decryption works
- [ ] CI/CD pipelines pass
- [ ] No secrets in git history (`git log -p | grep -i "secret\|key\|password"`)

### Monitoring
Monitor for 24 hours after rotation:
- Authentication error rates
- API error rates
- Service health metrics

---

## 6. Secrets Management Migration (Recommended)

### Current State
Secrets stored in `.env` files on disk.

### Recommended State
Use a secrets manager:

#### Option A: HashiCorp Vault
```bash
# Store secrets
vault kv put secret/mycelix/supabase \
  url="https://..." \
  anon_key="..."

# Retrieve in application
vault kv get -field=anon_key secret/mycelix/supabase
```

#### Option B: AWS Secrets Manager
```bash
aws secretsmanager create-secret \
  --name mycelix/supabase \
  --secret-string '{"url":"...","anon_key":"..."}'
```

#### Option C: SOPS + Age (GitOps friendly)
```bash
# Encrypt secrets file
sops --encrypt --age $(age-keygen -y key.txt) secrets.yaml > secrets.enc.yaml

# Decrypt at runtime
sops --decrypt secrets.enc.yaml
```

---

## Emergency Contacts

| Role | Contact | Responsibility |
|------|---------|----------------|
| Security Lead | [REDACTED] | Incident coordination |
| DevOps Lead | [REDACTED] | Infrastructure access |
| Supabase Admin | [REDACTED] | Database credentials |

---

## Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-18 | 1.0 | Initial creation |
