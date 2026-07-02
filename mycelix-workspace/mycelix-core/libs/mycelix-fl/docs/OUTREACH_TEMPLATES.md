# Mycelix FL -- Customer Outreach Templates

## Outreach Strategy

- **First wave**: 5 healthcare systems, 5 banks/fintech companies, 10 research labs
- **Personalization**: Reference the recipient's published work, recent ML initiatives, or relevant press coverage in the opening line
- **Follow-up cadence**: 1 week after initial send if no response; second follow-up at 3 weeks with a case-relevant technical brief attached
- **Evaluation offer**: Free 90-day evaluation license (Professional tier) for qualified leads -- no commitment, no credit card
- **Tracking**: Log all outreach in CRM with vertical tag (healthcare/finance/research) and response status

---

## Email 1: Healthcare (Hospital Networks)

**Subject:** Your multi-hospital ML models may be vulnerable to data poisoning

**To:** CTO, CISO, Clinical Informatics Director at hospital networks

---

Hi [First Name],

Federated learning lets hospitals train ML models collaboratively without sharing patient data. That is the promise. The problem: every major FL framework -- TensorFlow Federated, NVIDIA Clara, PySyft -- has zero protection against Byzantine attacks. A single compromised or misconfigured node can silently poison the shared model.

Mycelix FL is the first federated learning platform to break the 33% Byzantine fault tolerance barrier, achieving 45%. That means even if nearly half your participating nodes are compromised, the global model remains safe. Malicious participants are detected and quarantined within 2-4 rounds with 99%+ accuracy.

For compliance teams: every detection decision generates a ZK-STARK proof -- cryptographically verifiable, audit-ready evidence for HIPAA reviews.

Deployment is a single Rust binary via Docker. No Python environment management, no GPU requirements for the coordinator.

We offer a free 90-day evaluation. Reply to this email and I will set up a 30-minute walkthrough tailored to your infrastructure.

Best regards,
Tristan Stoltz
Luminous Dynamics
tristan.stoltz@evolvingresonantcocreationism.com
https://luminousdynamics.org

---

## Email 2: Financial Fraud Detection (Banks/Fintech)

**Subject:** Cross-institutional fraud detection without sharing transaction data

**To:** Head of AI/ML, Fintech CTO, Fraud Detection Team Lead

---

Hi [First Name],

Banks and fintechs want to share fraud patterns across institutions to improve detection models. Regulations and competitive concerns make sharing raw transaction data impossible. Federated learning solves the data-sharing problem -- but introduces a new one: model poisoning. A malicious or compromised participant can submit poisoned gradients that degrade your fraud model for everyone.

Mycelix FL detects and quarantines malicious participants within 2-4 rounds at 99%+ accuracy across 35 known attack types. Our PoGQ v4.1 algorithm achieves 45% Byzantine fault tolerance, well beyond the classical 33% limit. Differential privacy with configurable epsilon/delta satisfies regulatory requirements out of the box.

Performance matters for daily model updates: 7ms aggregation latency for 50 nodes with 10K parameters. Single Rust binary, Docker deployment, no vendor lock-in.

I would welcome 30 minutes to walk through how this fits your fraud detection pipeline. Reply here or grab time on my calendar: [link].

Best regards,
Tristan Stoltz
Luminous Dynamics
tristan.stoltz@evolvingresonantcocreationism.com
https://luminousdynamics.org

---

## Email 3: Research Institutions

**Subject:** Open-source Byzantine-resilient federated learning for collaborative research

**To:** ML Researchers, Lab Directors, Research Computing Teams

---

Hi [First Name],

If your group works on federated learning security, Mycelix FL may be a useful foundation.

We have open-sourced (AGPL-3.0) the first FL platform to break the 33% Byzantine fault tolerance barrier, reaching 45% validated tolerance across 35 attack types. The core is implemented in Rust -- not Python -- giving reproducible, deterministic benchmarks without floating-point nondeterminism from framework differences.

What is inside: 13 defense algorithms (FedAvg, Krum, Bulyan, FLTrust, Coordinate Median, Trimmed Mean, and 7 more) plus our novel PoGQ v4.1 with Mondrian conformal detection and hysteresis quarantine. Detection decisions are backed by ZK-STARK proofs for verifiable experiment provenance.

The platform is free for research. Extend the defense stack, benchmark against your own attacks, or use it as infrastructure for FL security papers. A standalone Rust crate is coming to crates.io shortly.

Try it now:

```bash
docker compose up coordinator
```

Happy to discuss collaboration. Reply here or open an issue on GitHub.

Best regards,
Tristan Stoltz
Luminous Dynamics
tristan.stoltz@evolvingresonantcocreationism.com
https://github.com/Luminous-Dynamics/mycelix
