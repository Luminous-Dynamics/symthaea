# Privacy Policy for Mycelix Knowledge Browser Extension

**Last Updated**: January 2024

## Overview

The Mycelix Knowledge browser extension ("Extension") is committed to protecting your privacy. This policy explains what data we collect, how we use it, and your rights.

## Data Collection

### What We Collect

**Text You Choose to Fact-Check**
- When you select text and click "Fact Check", that text is sent to the Mycelix Knowledge network
- This is necessary to provide the fact-checking service
- We do not store this text beyond the immediate query

**User Preferences**
- Extension settings (Holochain URL, theme, etc.)
- Stored locally in your browser's storage
- Never sent to any server

### What We Do NOT Collect

- Browsing history
- Personal information (name, email, etc.)
- Cookies from other sites
- Any data without your explicit action

## Data Storage

### Local Storage Only
All user preferences are stored locally using the browser's `storage.local` API. This data:
- Never leaves your device
- Is not accessible by websites you visit
- Can be cleared through browser settings

### No Cloud Storage
We do not maintain any cloud servers that store user data.

## Data Transmission

### Holochain Network
When you fact-check text:
1. Text is sent to your local Holochain node
2. Query is distributed through the Holochain DHT
3. Results are returned from network participants
4. No central server processes or stores your query

### Decentralized Architecture
- No single point collects all queries
- Your node participates in the distributed hash table
- Standard Holochain privacy protections apply

## Third-Party Services

This extension does not use:
- Analytics services
- Advertising networks
- Social media trackers
- Any third-party data collection

## Your Rights

You can:
- **Clear data**: Remove all local storage via browser settings
- **Disable**: Remove the extension at any time
- **Inspect**: View all stored data in browser developer tools
- **Control**: Choose what text to fact-check (nothing is automatic unless enabled)

## Security

We protect your data through:
- Local-only storage for preferences
- Holochain's cryptographic security for network queries
- No external API calls to centralized servers
- Open source code for transparency

## Changes to This Policy

We may update this policy. Changes will be:
- Noted in the extension's changelog
- Effective upon the next extension update
- Available at this URL

## Contact

For privacy questions:
- Email: privacy@luminousdynamics.org
- GitHub: https://github.com/Luminous-Dynamics/mycelix-knowledge/issues

## Consent

By using this extension, you consent to:
- Sending selected text to Holochain network for fact-checking
- Local storage of your preferences
- The practices described in this policy

---

**Summary**: We collect only what you explicitly choose to fact-check. Preferences stay on your device. We use decentralized Holochain, not central servers. Your privacy is protected by design.
