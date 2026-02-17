# Security Guide

## Security Features

Mycelix-Mail implements multiple layers of security to protect user data and prevent common vulnerabilities.

## Authentication & Authorization

### Password Security

- **Hashing Algorithm**: bcrypt with 12 rounds
- **Minimum Length**: 8 characters
- **Storage**: Never stored in plain text
- **Transmission**: Only sent over HTTPS

### JWT Tokens

- **Algorithm**: HS256 (HMAC-SHA256)
- **Secret**: Environment-specific, minimum 32 characters
- **Expiration**: Configurable (default: 7 days)
- **Storage**: LocalStorage (frontend), not in cookies by default
- **Transmission**: Bearer token in Authorization header

### Session Management

- Stateless authentication with JWT
- Token refresh mechanism (optional)
- Automatic logout on token expiration
- Single sign-out capability

## Data Encryption

### Email Credentials Encryption

Email account passwords are encrypted before storage:

- **Algorithm**: AES-256-GCM
- **Key Derivation**: PBKDF2 (100,000 iterations)
- **Salt**: 64 bytes, randomly generated per encryption
- **IV**: 16 bytes, randomly generated per encryption
- **Auth Tag**: 16 bytes for integrity verification
- **Encoding**: Base64 for storage

### Encryption Flow

```typescript
Plain Password
    ↓
Random Salt (64 bytes)
    ↓
PBKDF2 Key Derivation
    ↓
Random IV (16 bytes)
    ↓
AES-256-GCM Encryption
    ↓
Auth Tag Generation
    ↓
Base64 Encoding
    ↓
Stored in Database
```

### Decryption Flow

```typescript
Encrypted Data (Base64)
    ↓
Base64 Decoding
    ↓
Extract Salt, IV, Tag, Ciphertext
    ↓
PBKDF2 Key Derivation (with salt)
    ↓
Verify Auth Tag
    ↓
AES-256-GCM Decryption
    ↓
Plain Password
```

## API Security

### Rate Limiting

Protect against brute force and DDoS attacks:

- **General Endpoints**: 100 requests per 15 minutes
- **Auth Endpoints**: 5 requests per 15 minutes
- **Implementation**: express-rate-limit
- **Storage**: In-memory (configurable to Redis)

### Input Validation

All inputs are validated using Joi:

```typescript
// Example validation schema
{
  email: Joi.string().email().required(),
  password: Joi.string().min(8).required()
}
```

### CORS Configuration

- Configurable allowed origins
- Credentials support enabled
- Preflight caching

### Security Headers (Helmet.js)

- **X-DNS-Prefetch-Control**: off
- **X-Frame-Options**: SAMEORIGIN
- **X-Content-Type-Options**: nosniff
- **X-XSS-Protection**: 1; mode=block
- **Content-Security-Policy**: Configured
- **Strict-Transport-Security**: max-age=31536000

## Database Security

### SQL Injection Prevention

- **ORM**: Prisma (parameterized queries)
- **No raw SQL**: Avoided unless necessary
- **Input sanitization**: All user inputs validated

### Access Control

- Row-level security checks
- User-owned resources only
- Database user with minimal privileges

### Connection Security

- TLS/SSL connections in production
- Connection pooling with limits
- Secure connection strings in environment variables

## Network Security

### HTTPS/TLS

**Production Requirements:**
- TLS 1.2 or higher
- Strong cipher suites
- Valid SSL certificates

### WebSocket Security

- WSS (WebSocket Secure) in production
- JWT authentication for connections
- Origin validation

## Vulnerability Prevention

### XSS (Cross-Site Scripting)

- **Frontend**: React's built-in escaping
- **Backend**: No HTML rendering server-side
- **CSP Headers**: Content Security Policy
- **Input sanitization**: All user inputs sanitized

### CSRF (Cross-Site Request Forgery)

- Stateless JWT authentication
- SameSite cookie attribute (if using cookies)
- CORS configuration

### Injection Attacks

- **SQL Injection**: Prevented by Prisma ORM
- **Command Injection**: No shell commands from user input
- **Email Injection**: Input validation for email headers

### Path Traversal

- No direct file path access from user input
- Whitelist approach for allowed files
- Sandboxed file operations

## Secrets Management

### Environment Variables

**Never commit:**
- Database credentials
- JWT secrets
- Encryption keys
- API keys
- Email passwords

**Use .env files:**
```env
# Bad - committed to git
JWT_SECRET=my-secret

# Good - in .env (gitignored)
JWT_SECRET=complex-random-string-generated-securely
```

### Generating Secure Secrets

```bash
# Generate JWT secret (32 characters)
openssl rand -base64 32

# Generate encryption key (32 characters)
openssl rand -hex 32
```

## Security Best Practices

### For Developers

1. **Never log sensitive data**
   ```typescript
   // Bad
   console.log('User password:', password);

   // Good
   console.log('User authenticated');
   ```

2. **Validate all inputs**
   ```typescript
   // Always validate
   const schema = Joi.object({
     email: Joi.string().email().required()
   });
   ```

3. **Use parameterized queries**
   ```typescript
   // Good (Prisma)
   await prisma.user.findUnique({ where: { email } });

   // Bad (raw SQL)
   await db.query(`SELECT * FROM users WHERE email = '${email}'`);
   ```

4. **Handle errors securely**
   ```typescript
   // Don't expose stack traces in production
   if (process.env.NODE_ENV === 'production') {
     return res.status(500).json({ message: 'Internal server error' });
   }
   ```

### For Deployment

1. **Use HTTPS everywhere**
2. **Keep dependencies updated**
3. **Enable firewall rules**
4. **Use strong database passwords**
5. **Regular security audits**
6. **Monitor for suspicious activity**
7. **Implement backup strategies**
8. **Use separate environments (dev/staging/prod)**

## Dependency Security

### Regular Updates

```bash
# Check for vulnerabilities
npm audit

# Fix vulnerabilities
npm audit fix

# Check for outdated packages
npm outdated
```

### Automated Scanning

- Dependabot (GitHub)
- Snyk
- npm audit in CI/CD

## Incident Response

### If a Security Issue is Found

1. **Do not** disclose publicly immediately
2. Email security@mycelix.com (or open a private security advisory)
3. Provide details:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### Response Timeline

- **Acknowledgment**: Within 24 hours
- **Assessment**: Within 72 hours
- **Fix**: Depends on severity
  - Critical: Within 24 hours
  - High: Within 1 week
  - Medium: Within 2 weeks
  - Low: Next release

## Security Checklist

### Before Deployment

- [ ] All secrets in environment variables
- [ ] HTTPS enabled
- [ ] Strong JWT secret configured
- [ ] Strong encryption key configured
- [ ] Rate limiting enabled
- [ ] CORS properly configured
- [ ] Database over TLS
- [ ] Error messages don't leak sensitive info
- [ ] Security headers configured
- [ ] No console.logs with sensitive data
- [ ] Dependencies updated
- [ ] npm audit shows no vulnerabilities

### Regular Maintenance

- [ ] Weekly dependency updates
- [ ] Monthly security audit
- [ ] Review access logs
- [ ] Monitor error rates
- [ ] Check for suspicious activity
- [ ] Backup verification
- [ ] SSL certificate renewal

## Security Headers Example

```typescript
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      scriptSrc: ["'self'"],
      imgSrc: ["'self'", "data:", "https:"],
    },
  },
  hsts: {
    maxAge: 31536000,
    includeSubDomains: true,
    preload: true,
  },
}));
```

## OWASP Top 10 Mitigation

| Vulnerability | Mitigation |
|--------------|------------|
| Broken Access Control | JWT auth + userId checks |
| Cryptographic Failures | AES-256-GCM + bcrypt |
| Injection | Prisma ORM + Joi validation |
| Insecure Design | Security-first architecture |
| Security Misconfiguration | Helmet.js + secure defaults |
| Vulnerable Components | npm audit + Dependabot |
| Authentication Failures | bcrypt + JWT + rate limiting |
| Data Integrity Failures | Auth tags + checksums |
| Logging Failures | Winston + sanitized logs |
| SSRF | No user-controlled URLs |

## Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [OWASP Cheat Sheets](https://cheatsheetseries.owasp.org/)
- [Node.js Security Best Practices](https://nodejs.org/en/docs/guides/security/)
- [Prisma Security](https://www.prisma.io/docs/concepts/components/prisma-client/security)

## Contact

For security issues, please email: security@mycelix.com

Or open a private security advisory on GitHub.
