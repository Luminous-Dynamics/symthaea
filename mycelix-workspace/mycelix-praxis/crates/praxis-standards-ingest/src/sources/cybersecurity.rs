// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cybersecurity curriculum — NIST NICE Framework + CompTIA pathway.

use super::{CurriculumSource, SourceEntry, SourceError};
use crate::converter::{
    AcademicStandardRef, CurriculumDocument, CurriculumEdge, CurriculumMetadata, CurriculumNode,
};

pub struct CybersecuritySource;
impl CybersecuritySource { pub fn new() -> Self { Self } }

impl CurriculumSource for CybersecuritySource {
    fn name(&self) -> &str { "Cybersecurity (NICE Framework)" }
    fn list_available(&self) -> Result<Vec<SourceEntry>, SourceError> {
        Ok(vec![
            SourceEntry { id: "cyber-all".into(), title: "Cybersecurity — Full Pathway".into(), subject: "Cybersecurity".into(), level: "Beginner-Expert".into(), description: "NICE Framework foundations + CompTIA pathway + work roles (65 nodes)".into() },
        ])
    }
    fn fetch(&self, id: &str) -> Result<CurriculumDocument, SourceError> {
        if id != "cyber-all" { return Err(SourceError::NotFound(id.into())); }
        Ok(build_cyber_curriculum())
    }
}

fn c(code: &str, title: &str, desc: &str, bloom: &str, diff: &str, tags: &[&str], prereqs: &[&str], hours: u32) -> (String, CurriculumNode, Vec<CurriculumEdge>) {
    let id = format!("CYBER.{}", code);
    let node = CurriculumNode {
        id: id.clone(), title: title.into(), description: desc.into(),
        node_type: if bloom == "Remember" || bloom == "Understand" { "Concept" } else { "Skill" }.into(),
        difficulty: diff.into(), domain: "Cybersecurity".into(), subdomain: code.split('.').next().unwrap_or("General").into(),
        tags: tags.iter().map(|t| t.to_string()).chain(["cybersecurity".to_string()]).collect(),
        estimated_hours: hours,
        grade_levels: vec![match diff { "Beginner" => "Undergraduate", "Intermediate" => "Undergraduate", "Advanced" => "Graduate", _ => "Undergraduate" }.into()],
        bloom_level: bloom.into(), subject_area: "Cybersecurity".into(),
        academic_standards: vec![AcademicStandardRef { framework: "NIST NICE SP 800-181r1".into(), code: id.clone(), description: desc.into(), grade_level: diff.into() }],
        credit_hours: None, course_level: None, cip_code: Some("11.1003".into()), program_id: None, corequisites: vec![], supplementary_resources: vec![], lab_rubric: None, exam_weight: None,
    };
    let edges = prereqs.iter().map(|p| CurriculumEdge { from: format!("CYBER.{}", p), to: id.clone(), edge_type: "Requires".into(), strength_permille: 850, rationale: format!("{} is prerequisite for {}", p, code) }).collect();
    (id, node, edges)
}

fn build_cyber_curriculum() -> CurriculumDocument {
    let items = vec![
        // Foundations
        c("FND.CIA", "CIA Triad & Security Principles", "Confidentiality, Integrity, Availability. Defence in depth. Least privilege. Zero trust architecture.", "Understand", "Beginner", &["cia-triad", "security-principles", "zero-trust"], &[], 8),
        c("FND.RISK", "Risk Management", "Risk assessment, threat modelling, vulnerability analysis, risk treatment (avoid, mitigate, transfer, accept). NIST RMF.", "Analyze", "Beginner", &["risk-management", "threat-modelling", "nist-rmf"], &["FND.CIA"], 10),
        c("FND.CRYPTO", "Cryptography Fundamentals", "Symmetric (AES) vs asymmetric (RSA, ECC). Hashing (SHA-256). Digital signatures. PKI. TLS/SSL.", "Apply", "Beginner", &["cryptography", "encryption", "hashing", "pki", "tls"], &["FND.CIA"], 12),
        c("FND.NET", "Network Security Basics", "TCP/IP, DNS, firewalls, IDS/IPS, VPNs, network segmentation, DMZ architecture.", "Apply", "Beginner", &["network-security", "firewalls", "ids", "vpn", "dmz"], &["FND.CIA"], 10),
        c("FND.ACCESS", "Access Control & Identity", "Authentication factors (MFA), authorisation models (RBAC, ABAC), IAM, SSO, OAuth, SAML.", "Apply", "Beginner", &["access-control", "authentication", "mfa", "iam", "rbac"], &["FND.CIA"], 8),
        c("FND.OPS", "Security Operations Fundamentals", "SOC operations, SIEM, log analysis, incident detection, security monitoring, alert triage.", "Apply", "Intermediate", &["soc", "siem", "monitoring", "incident-detection"], &["FND.NET", "FND.ACCESS"], 10),
        c("FND.LEGAL", "Legal, Ethics & Compliance", "GDPR, POPIA (SA), HIPAA, PCI-DSS, computer crime laws, ethical hacking rules of engagement.", "Understand", "Beginner", &["gdpr", "popia", "compliance", "ethics", "legal"], &["FND.CIA"], 6),
        c("FND.ACAD", "Academic Security Research", "Responsible disclosure, CVE process, security research methodology, peer review.", "Evaluate", "Advanced", &["research", "cve", "disclosure", "methodology"], &["FND.LEGAL"], 6),

        // CompTIA Security+
        c("SEC.THREATS", "Threats, Attacks & Vulnerabilities", "Malware types, social engineering, application attacks, network attacks, threat actors, vulnerability types.", "Analyze", "Intermediate", &["malware", "social-engineering", "attacks", "vulnerability"], &["FND.NET", "FND.CRYPTO"], 15),
        c("SEC.ARCH", "Security Architecture & Design", "Secure network design, cloud security, virtualisation security, IoT security, embedded systems.", "Apply", "Intermediate", &["architecture", "cloud-security", "iot-security"], &["FND.NET", "SEC.THREATS"], 12),
        c("SEC.IMPL", "Security Implementation", "Secure protocols, endpoint protection, PKI implementation, wireless security, mobile security.", "Apply", "Intermediate", &["implementation", "endpoint", "wireless", "mobile-security"], &["SEC.ARCH", "FND.CRYPTO"], 12),
        c("SEC.OPS", "Security Operations & IR", "Incident response lifecycle, digital forensics basics, disaster recovery, BCP, security automation.", "Apply", "Intermediate", &["incident-response", "forensics", "disaster-recovery", "bcp"], &["FND.OPS", "SEC.THREATS"], 15),
        c("SEC.GRC", "Governance, Risk & Compliance", "Security frameworks (NIST CSF, ISO 27001), policies, procedures, security awareness training.", "Analyze", "Intermediate", &["governance", "nist-csf", "iso-27001", "policies"], &["FND.RISK", "FND.LEGAL"], 10),

        // Advanced — CySA+
        c("CYSA.THREAT", "Threat Intelligence & Management", "Threat intelligence sources, IOC analysis, MITRE ATT&CK, threat hunting, intelligence sharing.", "Analyze", "Advanced", &["threat-intelligence", "mitre-attack", "threat-hunting", "ioc"], &["SEC.THREATS", "SEC.OPS"], 15),
        c("CYSA.VULN", "Vulnerability Management", "Vulnerability scanning (Nessus, OpenVAS), CVSS scoring, remediation prioritisation, patch management.", "Apply", "Advanced", &["vulnerability-scanning", "cvss", "patch-management", "nessus"], &["SEC.THREATS"], 12),
        c("CYSA.SECOPS", "Advanced Security Operations", "SIEM tuning, EDR/XDR, SOAR, threat detection engineering, playbook development.", "Apply", "Advanced", &["siem", "edr", "xdr", "soar", "detection-engineering"], &["FND.OPS", "CYSA.THREAT"], 15),

        // PenTest+
        c("PEN.PLAN", "Penetration Testing Planning & Scoping", "Rules of engagement, scope definition, legal considerations, methodology selection (OWASP, PTES).", "Apply", "Advanced", &["pentest", "scoping", "owasp", "ptes", "methodology"], &["SEC.THREATS", "FND.LEGAL"], 10),
        c("PEN.RECON", "Information Gathering & Reconnaissance", "OSINT, passive/active recon, network scanning (Nmap), service enumeration, social engineering recon.", "Apply", "Advanced", &["osint", "recon", "nmap", "enumeration", "scanning"], &["PEN.PLAN", "FND.NET"], 12),
        c("PEN.ATTACK", "Exploitation & Attacks", "Web app attacks (OWASP Top 10), network exploitation, privilege escalation, lateral movement, post-exploitation.", "Apply", "Advanced", &["exploitation", "owasp-top10", "privilege-escalation", "lateral-movement"], &["PEN.RECON", "FND.CRYPTO"], 18),
        c("PEN.REPORT", "Reporting & Remediation", "Technical and executive reporting, risk rating, remediation guidance, retesting, client communication.", "Evaluate", "Advanced", &["reporting", "remediation", "risk-rating", "communication"], &["PEN.ATTACK"], 8),

        // Work Roles
        c("ROLE.SOC", "SOC Analyst", "Monitor security alerts, analyse events, escalate incidents, maintain documentation, shift operations.", "Apply", "Intermediate", &["soc-analyst", "monitoring", "triage"], &["FND.OPS"], 20),
        c("ROLE.IR", "Incident Responder", "Lead incident response, contain threats, perform forensic analysis, document findings, improve processes.", "Analyze", "Advanced", &["incident-responder", "forensics", "containment"], &["SEC.OPS", "CYSA.SECOPS"], 25),
        c("ROLE.PENTEST", "Penetration Tester", "Conduct authorised attack simulations, identify vulnerabilities, write detailed reports, recommend fixes.", "Apply", "Advanced", &["pentester", "ethical-hacking", "red-team"], &["PEN.ATTACK", "PEN.REPORT"], 30),
        c("ROLE.ARCH", "Security Architect", "Design secure systems, evaluate architectures, create security standards, threat model designs.", "Create", "Expert", &["security-architect", "design", "standards"], &["SEC.ARCH", "SEC.GRC"], 40),
        c("ROLE.CISO", "Chief Information Security Officer", "Strategic security leadership, risk management, board communication, security program management.", "Evaluate", "Expert", &["ciso", "leadership", "strategy", "governance"], &["SEC.GRC", "ROLE.ARCH"], 50),
    ];

    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    for (_, node, es) in items {
        nodes.push(node);
        edges.extend(es);
    }

    CurriculumDocument {
        metadata: CurriculumMetadata {
            title: "Cybersecurity — NICE Framework & CompTIA Pathway".into(),
            framework: "NIST NICE SP 800-181r1".into(),
            source: "https://www.nist.gov/itl/applied-cybersecurity/nice".into(),
            grade_level: "Undergraduate".into(), subject_area: "Cybersecurity".into(),
            domain: "Cybersecurity".into(), version: "2025".into(),
            total_standards: nodes.len(),
            domains: vec!["Foundations".into(), "CompTIA Security+".into(), "CySA+".into(), "PenTest+".into(), "Work Roles".into()],
            created_at: chrono::Local::now().format("%Y-%m-%d").to_string(),
            notes: "Curated from NIST NICE Framework (SP 800-181r1) and CompTIA certification pathway.".into(),
            academic_level: Some("Undergraduate-Graduate".into()), institution: None, cip_code: Some("11.1003".into()), total_credits: None, duration_semesters: None,
        },
        nodes, edges,
    }
}
