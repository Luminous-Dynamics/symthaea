# Privacy Policy Template

**Mycelix Federated Learning Platform**
**Template Version**: 1.0
**Last Updated**: 2026-01-18

---

> **INSTRUCTIONS**: This template should be reviewed by legal counsel and customized for your specific jurisdiction and use case. Replace all [BRACKETED TEXT] with appropriate values.

---

# Privacy Policy

**Effective Date**: [DATE]
**Last Updated**: [DATE]

## 1. Introduction

[ORGANIZATION NAME] ("we," "us," or "our") operates the Mycelix federated learning platform ("Service"). This Privacy Policy explains how we collect, use, disclose, and safeguard your information when you participate in our decentralized federated learning network.

We are committed to protecting your privacy and complying with the General Data Protection Regulation (GDPR), the California Consumer Privacy Act (CCPA), and other applicable privacy laws.

## 2. Data Controller

**Data Controller**: [ORGANIZATION NAME]
**Address**: [ADDRESS]
**Email**: [PRIVACY EMAIL]
**Data Protection Officer**: [DPO NAME AND CONTACT]

## 3. Information We Collect

### 3.1 Information You Provide

| Data Type | Purpose | Legal Basis |
|-----------|---------|-------------|
| Account Information | Platform access | Contract performance |
| Agent Public Key | Identity verification | Contract performance |
| Contact Information | Communication | Legitimate interest |

### 3.2 Information Collected Automatically

| Data Type | Purpose | Legal Basis |
|-----------|---------|-------------|
| Model Gradients | Federated learning participation | Legitimate interest |
| Trust Scores | Quality assurance | Legitimate interest |
| Contribution Metadata | Audit and compliance | Legal obligation |
| Usage Analytics | Service improvement | Consent |

### 3.3 Information We Do NOT Collect

- **Raw Training Data**: Your local training data never leaves your device
- **Unencrypted Model Weights**: All contributions are encrypted
- **Real Identity**: Participation is pseudonymous via public keys

## 4. How We Use Your Information

We use collected information to:

1. **Provide the Service**: Coordinate federated learning rounds, aggregate model updates
2. **Ensure Security**: Detect Byzantine behavior, verify contribution quality
3. **Comply with Law**: Respond to legal requests, maintain audit trails
4. **Improve Service**: Analyze usage patterns, optimize performance

## 5. Information Sharing

We may share your information with:

| Recipient | Purpose | Data Shared |
|-----------|---------|-------------|
| Network Participants | Distributed consensus | Pseudonymous contributions |
| Ethereum Network | Proof anchoring | Cryptographic hashes only |
| Law Enforcement | Legal compliance | As required by law |
| Service Providers | Infrastructure | Under data processing agreements |

We do NOT sell your personal information.

## 6. Data Security

We implement industry-standard security measures:

- **Encryption**: ChaCha20-Poly1305 for data at rest, TLS 1.3 in transit
- **Access Control**: Per-user encryption keys
- **Byzantine Tolerance**: System resists up to 45% malicious participants
- **Zero-Knowledge Proofs**: Verify contributions without revealing data

## 7. Your Rights

### 7.1 GDPR Rights (EU/EEA Residents)

You have the right to:

| Right | Description | How to Exercise |
|-------|-------------|-----------------|
| **Access** | Obtain a copy of your data | Dashboard or API export |
| **Rectification** | Correct inaccurate data | Submit corrected contribution |
| **Erasure** | Delete your data | Request via dashboard or [ERASURE EMAIL] |
| **Restriction** | Limit processing | Pause participation |
| **Portability** | Receive data in machine-readable format | JSON export via API |
| **Object** | Opt-out of processing | Withdraw from network |
| **Withdraw Consent** | Revoke previously given consent | Account settings |

### 7.2 CCPA Rights (California Residents)

You have the right to:
- Know what personal information is collected
- Delete personal information
- Opt-out of the sale of personal information (we do not sell)
- Non-discrimination for exercising rights

To exercise CCPA rights, contact: [CCPA EMAIL]

### 7.3 Right to Erasure Implementation

We implement the right to erasure through **cryptographic key deletion**:

1. Your data is encrypted with a unique key
2. Upon erasure request, we permanently delete this key
3. Your encrypted data becomes mathematically unrecoverable
4. You receive a verifiable `ErasureReceipt` as proof

**Response Time**: Within 30 days of verified request

## 8. Data Retention

| Data Type | Retention Period | Reason |
|-----------|------------------|--------|
| Account Information | Until account deletion | Service provision |
| Model Gradients | 30 days rolling | Aggregation window |
| Trust Scores | Until erasure request | Reputation continuity |
| Contribution Metadata | 90 days | Audit requirements |
| ZK Proofs | Permanent | No personal data; integrity verification |

After retention periods, data is automatically deleted or anonymized.

## 9. International Data Transfers

Your information may be transferred to and processed in countries outside your residence. We ensure appropriate safeguards:

- Standard Contractual Clauses (SCCs)
- Adequacy decisions (where applicable)
- Binding Corporate Rules (if adopted)

## 10. Children's Privacy

The Service is not intended for individuals under [16/18] years of age. We do not knowingly collect personal information from children.

## 11. Updates to This Policy

We may update this Privacy Policy periodically. We will notify you of material changes by:
- Posting notice on the platform
- Sending email to registered users
- Updating the "Last Updated" date

Continued use after changes constitutes acceptance.

## 12. Contact Us

For privacy-related inquiries:

**Email**: [PRIVACY EMAIL]
**Address**: [ADDRESS]
**DPO**: [DPO EMAIL]

For erasure requests specifically:
**Email**: [ERASURE EMAIL]

## 13. Supervisory Authority

If you believe we have violated your privacy rights, you have the right to lodge a complaint with your local data protection authority:

- **EU**: [RELEVANT DPA]
- **UK**: Information Commissioner's Office (ICO)
- **California**: California Attorney General

---

## Appendix: Technical Privacy Measures

### A.1 Federated Learning Privacy

```
Your Device              Network                   Global Model
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│ Raw Data     │         │ Aggregator   │         │ Model Weights│
│ (stays here) │         │              │         │              │
│      ↓       │         │              │         │              │
│ Local Model  │         │              │         │              │
│      ↓       │   DP    │              │         │              │
│ Gradient    ─┼────────>│  Aggregate  ─┼────────>│ Updated      │
│ + Noise      │ encrypt │              │ anchor  │              │
└──────────────┘         └──────────────┘         └──────────────┘
```

### A.2 Cryptographic Erasure

When you request erasure:
1. Your unique encryption key is permanently deleted
2. All your encrypted contributions become unreadable
3. A cryptographic receipt proves the erasure occurred

### A.3 Zero-Knowledge Proofs

We use zero-knowledge proofs to verify:
- Contribution validity
- Trust score calculations
- Protocol compliance

Without revealing the underlying data.

---

*This Privacy Policy template is provided for informational purposes. Consult legal counsel for your specific requirements.*
