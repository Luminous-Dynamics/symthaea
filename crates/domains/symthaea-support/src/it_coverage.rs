// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Machine-readable inventory of Symthaea's IT implementation footprint.
//!
//! This registry answers "what already exists, and where?" without conflating
//! repository presence with demonstrated competence.
//!
//! ```text
//! source-code mention != implementation
//! implementation != expert knowledge
//! benchmark case != passing result
//! passing result != domain qualification
//! active PR != merged capability
//! ```

use crate::it_qualification::ItDomainV1;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

pub const IT_COVERAGE_INVENTORY_SCHEMA_V1: &str = "symthaea-it-coverage-inventory-v1";

pub const ALL_IT_DOMAINS_V1: [ItDomainV1; 22] = [
    ItDomainV1::ComputerArchitecture,
    ItDomainV1::HardwareDatacenter,
    ItDomainV1::OperatingSystems,
    ItDomainV1::LinuxUnix,
    ItDomainV1::Windows,
    ItDomainV1::IdentityAccess,
    ItDomainV1::Networking,
    ItDomainV1::WirelessTelecom,
    ItDomainV1::Storage,
    ItDomainV1::Virtualization,
    ItDomainV1::Containers,
    ItDomainV1::Orchestration,
    ItDomainV1::Cloud,
    ItDomainV1::DistributedSystems,
    ItDomainV1::DatabasesData,
    ItDomainV1::ApplicationsProtocols,
    ItDomainV1::Cybersecurity,
    ItDomainV1::DevOpsPlatform,
    ItDomainV1::ObservabilitySre,
    ItDomainV1::EnterpriseOperations,
    ItDomainV1::OperationalTechnology,
    ItDomainV1::LegacyComputing,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ItCoverageSignalKindV1 {
    DedicatedSubsystem,
    CoreReasoning,
    EvidenceIngestion,
    KnowledgeSource,
    OperationalTooling,
    Benchmark,
    ArchitecturePlan,
    VocabularyOnly,
}

impl ItCoverageSignalKindV1 {
    fn is_concrete_implementation(self) -> bool {
        matches!(
            self,
            Self::DedicatedSubsystem
                | Self::CoreReasoning
                | Self::EvidenceIngestion
                | Self::KnowledgeSource
                | Self::OperationalTooling
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ItCoverageSourceV1 {
    RepositoryPath(String),
    PullRequest(u64),
    Issue(u64),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ItCoverageSignalV1 {
    pub kind: ItCoverageSignalKindV1,
    pub source: ItCoverageSourceV1,
    pub note: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ItImplementationFootprintV1 {
    Unmapped,
    ReferenceOnly,
    FragmentedImplementation,
    DedicatedImplementation,
    MultiSubsystemImplementation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ItDomainCoverageV1 {
    pub domain: ItDomainV1,
    #[serde(default)]
    pub signals: Vec<ItCoverageSignalV1>,
    /// Stable machine-readable gap tags, not prose claims of impossibility.
    #[serde(default)]
    pub gap_tags: BTreeSet<String>,
}

impl ItDomainCoverageV1 {
    pub fn footprint(&self) -> ItImplementationFootprintV1 {
        if self.signals.is_empty() {
            return ItImplementationFootprintV1::Unmapped;
        }
        let concrete_kinds: BTreeSet<_> = self
            .signals
            .iter()
            .filter(|signal| signal.kind.is_concrete_implementation())
            .map(|signal| signal.kind)
            .collect();
        if concrete_kinds.is_empty() {
            return ItImplementationFootprintV1::ReferenceOnly;
        }
        let dedicated = concrete_kinds.contains(&ItCoverageSignalKindV1::DedicatedSubsystem);
        if dedicated && concrete_kinds.len() >= 3 {
            ItImplementationFootprintV1::MultiSubsystemImplementation
        } else if dedicated {
            ItImplementationFootprintV1::DedicatedImplementation
        } else {
            ItImplementationFootprintV1::FragmentedImplementation
        }
    }

    pub fn has_benchmark_signal(&self) -> bool {
        self.signals
            .iter()
            .any(|signal| signal.kind == ItCoverageSignalKindV1::Benchmark)
    }

    pub fn has_operational_tooling(&self) -> bool {
        self.signals
            .iter()
            .any(|signal| signal.kind == ItCoverageSignalKindV1::OperationalTooling)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ItCoverageInventoryV1 {
    pub schema_version: String,
    pub audited_repository_ref: String,
    pub audited_at_unix_ms: u64,
    pub domains: Vec<ItDomainCoverageV1>,
}

impl ItCoverageInventoryV1 {
    pub fn validate(&self) -> Result<(), ItCoverageErrorV1> {
        if self.schema_version != IT_COVERAGE_INVENTORY_SCHEMA_V1 {
            return Err(ItCoverageErrorV1::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        require_nonempty(&self.audited_repository_ref, "audited repository ref")?;
        let expected: BTreeSet<_> = ALL_IT_DOMAINS_V1.into_iter().collect();
        let mut seen = BTreeSet::new();
        for domain in &self.domains {
            if !seen.insert(domain.domain) {
                return Err(ItCoverageErrorV1::DuplicateDomain(domain.domain));
            }
            for signal in &domain.signals {
                validate_signal(signal)?;
            }
            for tag in &domain.gap_tags {
                require_nonempty(tag, "coverage gap tag")?;
                if tag.chars().any(|ch| ch.is_ascii_uppercase() || ch.is_whitespace()) {
                    return Err(ItCoverageErrorV1::InvalidGapTag(tag.clone()));
                }
            }
        }
        if seen != expected {
            let missing = expected.difference(&seen).copied().collect();
            let unexpected = seen.difference(&expected).copied().collect();
            return Err(ItCoverageErrorV1::DomainSetMismatch { missing, unexpected });
        }
        Ok(())
    }

    pub fn domain(&self, domain: ItDomainV1) -> Option<&ItDomainCoverageV1> {
        self.domains.iter().find(|entry| entry.domain == domain)
    }

    pub fn footprint_counts(&self) -> BTreeMap<ItImplementationFootprintV1, usize> {
        let mut counts = BTreeMap::new();
        for domain in &self.domains {
            *counts.entry(domain.footprint()).or_default() += 1;
        }
        counts
    }

    pub fn unmapped_domains(&self) -> Vec<ItDomainV1> {
        self.domains
            .iter()
            .filter(|domain| domain.footprint() == ItImplementationFootprintV1::Unmapped)
            .map(|domain| domain.domain)
            .collect()
    }

    pub fn domains_without_benchmarks(&self) -> Vec<ItDomainV1> {
        self.domains
            .iter()
            .filter(|domain| !domain.has_benchmark_signal())
            .map(|domain| domain.domain)
            .collect()
    }
}

/// Audited seed inventory from the repository plus active IT-system-intelligence
/// PR/issue lines. Active PR signals are intentionally distinguishable from repository
/// paths and therefore must not be interpreted as merged capability.
pub fn seed_it_coverage_inventory_v1(
    audited_repository_ref: impl Into<String>,
    audited_at_unix_ms: u64,
) -> Result<ItCoverageInventoryV1, ItCoverageErrorV1> {
    let mut domains = Vec::new();

    domains.push(entry(
        ItDomainV1::ComputerArchitecture,
        vec![
            path(ItCoverageSignalKindV1::EvidenceIngestion, "crates/domains/symthaea-spore/src/hardware_probe.rs", "CPU/GPU/memory/platform hardware probing"),
            path(ItCoverageSignalKindV1::VocabularyOnly, "crates/core/symthaea-app-db/src/semantic_search.rs", "hardware/tool semantic vocabulary includes architecture-adjacent terms"),
        ],
        &["cpu-memory-hierarchy", "numa-pcie-ecc", "uefi-firmware"],
    ));
    domains.push(entry(
        ItDomainV1::HardwareDatacenter,
        vec![
            path(ItCoverageSignalKindV1::EvidenceIngestion, "crates/domains/symthaea-spore/src/hardware_probe.rs", "machine hardware inventory"),
            issue(ItCoverageSignalKindV1::ArchitecturePlan, 1100, "Redfish/BMC and network-device continuity plan"),
        ],
        &["bmc-redfish-runtime", "power-ups-pdu", "racks-cabling-cooling"],
    ));
    domains.push(entry(
        ItDomainV1::OperatingSystems,
        vec![
            path(ItCoverageSignalKindV1::CoreReasoning, "src/language/nixos_plugin.rs", "domain-routed NixOS language capability"),
            path(ItCoverageSignalKindV1::CoreReasoning, "src/intelligence/nixos_causal.rs", "NixOS causal analysis"),
            path(ItCoverageSignalKindV1::DedicatedSubsystem, "crates/core/nixward", "Nix/NixOS policy, parsing and causal tooling"),
            path(ItCoverageSignalKindV1::OperationalTooling, "crates/domains/symthaea-spore", "installer/migration and system probing"),
        ],
        &["cross-os-kernel-model", "bsd-macos-depth"],
    ));
    domains.push(entry(
        ItDomainV1::LinuxUnix,
        vec![
            path(ItCoverageSignalKindV1::DedicatedSubsystem, "crates/core/nixward", "deep NixOS/Linux configuration and policy tooling"),
            path(ItCoverageSignalKindV1::OperationalTooling, "crates/domains/symthaea-spore", "Linux/NixOS installation and migration tooling"),
            path(ItCoverageSignalKindV1::KnowledgeSource, "docs/guides/03-troubleshooting.md", "Linux/NixOS troubleshooting guidance"),
        ],
        &["non-nixos-distro-depth", "kernel-internals", "unix-enterprise-aix-solaris"],
    ));
    domains.push(entry(
        ItDomainV1::Windows,
        vec![
            path(ItCoverageSignalKindV1::EvidenceIngestion, "crates/core/symthaea-logparse/src/evtx_source.rs", "offline EVTX ingestion"),
            path(ItCoverageSignalKindV1::EvidenceIngestion, "crates/core/symthaea-logparse/src/otrf_source.rs", "Windows security-event corpus ingestion"),
            path(ItCoverageSignalKindV1::Benchmark, "crates/core/symthaea-logparse/PHASE1_RESULTS.md", "Windows-event ATT&CK classification experiment"),
            pr(ItCoverageSignalKindV1::Benchmark, 1500, "Kerberos/currentness multi-domain qualification cases"),
        ],
        &["windows-admin-state", "active-directory-admin", "gpo-powershell", "live-etw-wmi"],
    ));
    domains.push(entry(
        ItDomainV1::IdentityAccess,
        vec![
            path(ItCoverageSignalKindV1::EvidenceIngestion, "crates/core/symthaea-logparse/src/fixtures.rs", "Kerberos/authentication event semantics"),
            pr(ItCoverageSignalKindV1::Benchmark, 1500, "cloud IAM and identity/time qualification case"),
            pr(ItCoverageSignalKindV1::KnowledgeSource, 1498, "versioned cyber ontology mappings"),
        ],
        &["directory-service-model", "iam-policy-evaluator", "pki-pam-federation"],
    ));
    domains.push(entry(
        ItDomainV1::Networking,
        vec![
            path(ItCoverageSignalKindV1::CoreReasoning, "crates/domains/symthaea-support/src/diagnostics.rs", "network/DNS diagnostic planning"),
            path(ItCoverageSignalKindV1::OperationalTooling, "crates/domains/symthaea-support/src/triage.rs", "network issue triage"),
            pr(ItCoverageSignalKindV1::EvidenceIngestion, 1172, "typed TCP/DNS/TLS/QUIC/ICMP protocol evidence"),
            pr(ItCoverageSignalKindV1::DedicatedSubsystem, 1494, "network continuity witness domain"),
            issue(ItCoverageSignalKindV1::ArchitecturePlan, 1100, "OpenConfig/Batfish/device continuity program"),
            pr(ItCoverageSignalKindV1::Benchmark, 1500, "multi-fault networking qualification cases"),
        ],
        &["routing-protocol-live-adapters", "pcap-decoder", "carrier-network-depth"],
    ));
    domains.push(entry(
        ItDomainV1::WirelessTelecom,
        vec![
            path(ItCoverageSignalKindV1::CoreReasoning, "src/swarm/mesh/dual_layer.rs", "802.11s/LoRa mesh selection model"),
            path(ItCoverageSignalKindV1::KnowledgeSource, "docs/sovereign-inoculation-research.md", "Wi-Fi configuration and installer research"),
            path(ItCoverageSignalKindV1::OperationalTooling, "nix/modules/installer-iso-module.nix", "NetworkManager/Wi-Fi installer support"),
        ],
        &["rf-link-budget", "wifi-protocol-depth", "cellular-5g", "satellite-lpwan"],
    ));
    domains.push(entry(
        ItDomainV1::Storage,
        vec![
            path(ItCoverageSignalKindV1::ArchitecturePlan, "docs/architecture/STORAGE_ARCHITECTURE.md", "Symthaea storage backend architecture"),
            path(ItCoverageSignalKindV1::OperationalTooling, "nix/disko-vps.nix", "disk layout and virtual-disk provisioning"),
            path(ItCoverageSignalKindV1::OperationalTooling, "crates/domains/symthaea-spore", "installation, disk probing and migration"),
        ],
        &["san-fc-iscsi-nvmeof", "ceph-object-storage", "backup-restore-qualification", "zfs-btrfs-deep-diagnostics"],
    ));
    domains.push(entry(
        ItDomainV1::Virtualization,
        vec![
            path(ItCoverageSignalKindV1::OperationalTooling, "src/language/nix_codegen.rs", "libvirt/QEMU configuration generation"),
            path(ItCoverageSignalKindV1::Benchmark, "scripts/automated-demo.sh", "QEMU end-to-end environment"),
            path(ItCoverageSignalKindV1::KnowledgeSource, "crates/core/symthaea-app-db/src/lib.rs", "KVM/QEMU virtualization mappings"),
        ],
        &["hypervisor-runtime-observation", "vmware-hyperv-proxmox", "live-migration-ha"],
    ));
    domains.push(entry(
        ItDomainV1::Containers,
        vec![
            path(ItCoverageSignalKindV1::CoreReasoning, "src/language/compose_codegen.rs", "Compose stack generation"),
            path(ItCoverageSignalKindV1::OperationalTooling, "crates/domains/symthaea-broca-tools/src/iac_repair.rs", "Docker Compose/IaC validation and repair"),
            path(ItCoverageSignalKindV1::EvidenceIngestion, "crates/domains/symthaea-spore/src/bin/ssh_relay.rs", "container presence/risk probing"),
        ],
        &["container-runtime-internals", "cgroups-namespaces", "oci-supply-chain"],
    ));
    domains.push(entry(
        ItDomainV1::Orchestration,
        vec![
            path(ItCoverageSignalKindV1::OperationalTooling, "crates/domains/symthaea-broca-tools/src/iac_repair.rs", "Kubernetes validation/repair path"),
            path(ItCoverageSignalKindV1::CoreReasoning, "crates/domains/symthaea-broca/src/language_gates.rs", "Kubernetes language gate"),
            pr(ItCoverageSignalKindV1::Benchmark, 1500, "Kubernetes CNI/DNS multi-fault case"),
        ],
        &["k8s-control-plane-state", "cni-csi-runtime", "scheduler-operator-depth"],
    ));
    domains.push(entry(
        ItDomainV1::Cloud,
        vec![
            path(ItCoverageSignalKindV1::OperationalTooling, "crates/domains/symthaea-broca-tools/src/iac_repair.rs", "Terraform/CloudFormation/Pulumi validation"),
            path(ItCoverageSignalKindV1::KnowledgeSource, "crates/core/nixward/src/encoding/package_aliases.rs", "cloud/devops package taxonomy"),
            pr(ItCoverageSignalKindV1::Benchmark, 1500, "cloud IAM/time multi-fault case"),
        ],
        &["aws-azure-gcp-live-state", "cloud-networking-runtime", "serverless-managed-services", "finops"],
    ));
    domains.push(entry(
        ItDomainV1::DistributedSystems,
        vec![
            path(ItCoverageSignalKindV1::DedicatedSubsystem, "crates/domains/symthaea-swarm", "gossip/direct transport and distributed swarm state"),
            path(ItCoverageSignalKindV1::CoreReasoning, "crates/domains/symthaea-fabrication-kernel/src/witness_gossip.rs", "signed witness gossip/equivocation concepts"),
            path(ItCoverageSignalKindV1::ArchitecturePlan, "docs/planning/MYCELIX_SYMTHAEA_INTEGRATION_ROADMAP.md", "DHT/eventual-consistency bridge planning"),
        ],
        &["consensus-taxonomy", "crdt-library", "distributed-clocks-leases", "failure-injection"],
    ));
    domains.push(entry(
        ItDomainV1::DatabasesData,
        vec![
            path(ItCoverageSignalKindV1::DedicatedSubsystem, "crates/domains/symthaea-wisdom", "PostgreSQL-backed wisdom/runtime work"),
            path(ItCoverageSignalKindV1::CoreReasoning, "src/language/compose_codegen.rs", "database-aware service composition"),
            pr(ItCoverageSignalKindV1::Benchmark, 1500, "database TLS/replication multi-fault case"),
        ],
        &["query-planner-diagnostics", "replication-ha-runtime", "mysql-sqlserver-oracle", "streaming-data-systems"],
    ));
    domains.push(entry(
        ItDomainV1::ApplicationsProtocols,
        vec![
            path(ItCoverageSignalKindV1::CoreReasoning, "src/language/compose_codegen.rs", "application/service stack generation"),
            path(ItCoverageSignalKindV1::CoreReasoning, "src/language/programming_plugin.rs", "programming/platform domain routing"),
            pr(ItCoverageSignalKindV1::EvidenceIngestion, 1172, "typed DNS/TLS/QUIC/TCP protocol evidence"),
        ],
        &["mail-smtp-imap-enterprise", "http-proxy-loadbalancer-depth", "sip-voip", "protocol-state-machines"],
    ));
    domains.push(entry(
        ItDomainV1::Cybersecurity,
        vec![
            path(ItCoverageSignalKindV1::DedicatedSubsystem, "crates/domains/symthaea-audit", "security/audit domain"),
            path(ItCoverageSignalKindV1::EvidenceIngestion, "crates/core/symthaea-logparse", "Windows/syslog security-event ingestion and clustering"),
            path(ItCoverageSignalKindV1::CoreReasoning, "crates/core/nixward", "configuration/policy security reasoning"),
            pr(ItCoverageSignalKindV1::KnowledgeSource, 1498, "versioned ATT&CK/D3FEND ontology bridge"),
            pr(ItCoverageSignalKindV1::Benchmark, 1500, "TLS/security adversarial cases"),
        ],
        &["live-edr-siem-adapters", "forensics-depth", "vulnerability-management", "security-control-effectiveness"],
    ));
    domains.push(entry(
        ItDomainV1::DevOpsPlatform,
        vec![
            path(ItCoverageSignalKindV1::DedicatedSubsystem, "crates/domains/symthaea-broca-tools", "IaC harvesting, validation and repair"),
            path(ItCoverageSignalKindV1::DedicatedSubsystem, "crates/core/nixward", "Nix configuration/platform policy"),
            path(ItCoverageSignalKindV1::CoreReasoning, "src/language/programming_plugin.rs", "toolchain/platform language routing"),
        ],
        &["ci-provider-live-adapters", "gitops-release-evidence", "sbom-slsa-integration"],
    ));
    domains.push(entry(
        ItDomainV1::ObservabilitySre,
        vec![
            path(ItCoverageSignalKindV1::DedicatedSubsystem, "crates/domains/symthaea-observability", "cognitive/causal observability"),
            path(ItCoverageSignalKindV1::EvidenceIngestion, "crates/core/symthaea-logparse", "logs/security-event normalization"),
            pr(ItCoverageSignalKindV1::EvidenceIngestion, 1158, "external telemetry normalization into live state"),
            pr(ItCoverageSignalKindV1::CoreReasoning, 1136, "causal diagnostic beliefs and true EIG"),
        ],
        &["otel-sdk-ingestion", "slo-error-budget", "production-incident-correlation", "capacity-saturation-models"],
    ));
    domains.push(entry(
        ItDomainV1::EnterpriseOperations,
        vec![
            path(ItCoverageSignalKindV1::DedicatedSubsystem, "crates/domains/symthaea-support", "triage, diagnostics, knowledge, privacy and support actions"),
            path(ItCoverageSignalKindV1::EvidenceIngestion, "crates/core/symthaea-logparse", "Windows/syslog/SNMP-shaped event ingestion"),
            pr(ItCoverageSignalKindV1::CoreReasoning, 1160, "dependency/blast-radius analysis"),
            pr(ItCoverageSignalKindV1::Benchmark, 1500, "multi-domain operational incidents"),
        ],
        &["cmdb-asset-lifecycle", "change-problem-management", "backup-dr-bcp", "m365-print-voip-vdi"],
    ));
    domains.push(entry(
        ItDomainV1::OperationalTechnology,
        vec![
            path(ItCoverageSignalKindV1::DedicatedSubsystem, "crates/domains/symthaea-fabrication-kernel", "manufacturing/fabrication execution and evidence"),
            path(ItCoverageSignalKindV1::ArchitecturePlan, "papers/applications/manufacturing-consciousness/manufacturing_consciousness.tex", "OPC-UA/MTConnect manufacturing integration proposal"),
        ],
        &["opcua-runtime-adapter", "modbus-dnp3", "plc-scada-model", "ot-safety-zones"],
    ));
    domains.push(entry(
        ItDomainV1::LegacyComputing,
        Vec::new(),
        &["mainframe-zos", "ibm-i", "aix-solaris-hpux", "legacy-network-protocols", "legacy-database-ops"],
    ));

    let inventory = ItCoverageInventoryV1 {
        schema_version: IT_COVERAGE_INVENTORY_SCHEMA_V1.into(),
        audited_repository_ref: audited_repository_ref.into(),
        audited_at_unix_ms,
        domains,
    };
    inventory.validate()?;
    Ok(inventory)
}

fn entry(
    domain: ItDomainV1,
    signals: Vec<ItCoverageSignalV1>,
    gaps: &[&str],
) -> ItDomainCoverageV1 {
    ItDomainCoverageV1 {
        domain,
        signals,
        gap_tags: gaps.iter().map(|value| (*value).into()).collect(),
    }
}

fn path(kind: ItCoverageSignalKindV1, value: &str, note: &str) -> ItCoverageSignalV1 {
    ItCoverageSignalV1 {
        kind,
        source: ItCoverageSourceV1::RepositoryPath(value.into()),
        note: note.into(),
    }
}

fn pr(kind: ItCoverageSignalKindV1, number: u64, note: &str) -> ItCoverageSignalV1 {
    ItCoverageSignalV1 {
        kind,
        source: ItCoverageSourceV1::PullRequest(number),
        note: note.into(),
    }
}

fn issue(kind: ItCoverageSignalKindV1, number: u64, note: &str) -> ItCoverageSignalV1 {
    ItCoverageSignalV1 {
        kind,
        source: ItCoverageSourceV1::Issue(number),
        note: note.into(),
    }
}

fn validate_signal(signal: &ItCoverageSignalV1) -> Result<(), ItCoverageErrorV1> {
    require_nonempty(&signal.note, "coverage signal note")?;
    match &signal.source {
        ItCoverageSourceV1::RepositoryPath(path) => {
            require_nonempty(path, "coverage repository path")?;
            if path.starts_with('/') || path.contains("..") {
                return Err(ItCoverageErrorV1::InvalidRepositoryPath(path.clone()));
            }
        }
        ItCoverageSourceV1::PullRequest(number) => {
            if *number == 0 {
                return Err(ItCoverageErrorV1::InvalidReference("pull request 0".into()));
            }
        }
        ItCoverageSourceV1::Issue(number) => {
            if *number == 0 {
                return Err(ItCoverageErrorV1::InvalidReference("issue 0".into()));
            }
        }
    }
    Ok(())
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), ItCoverageErrorV1> {
    if value.trim().is_empty() {
        Err(ItCoverageErrorV1::EmptyField(field))
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ItCoverageErrorV1 {
    UnsupportedSchema(String),
    EmptyField(&'static str),
    DuplicateDomain(ItDomainV1),
    DomainSetMismatch {
        missing: Vec<ItDomainV1>,
        unexpected: Vec<ItDomainV1>,
    },
    InvalidGapTag(String),
    InvalidRepositoryPath(String),
    InvalidReference(String),
}

impl fmt::Display for ItCoverageErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchema(value) => write!(f, "unsupported IT coverage schema {value}"),
            Self::EmptyField(field) => write!(f, "empty IT coverage field {field}"),
            Self::DuplicateDomain(domain) => write!(f, "duplicate IT coverage domain {domain:?}"),
            Self::DomainSetMismatch { missing, unexpected } => write!(
                f,
                "IT coverage domain set mismatch; missing={missing:?}, unexpected={unexpected:?}"
            ),
            Self::InvalidGapTag(tag) => write!(f, "invalid IT coverage gap tag {tag}"),
            Self::InvalidRepositoryPath(path) => write!(f, "invalid IT coverage repository path {path}"),
            Self::InvalidReference(value) => write!(f, "invalid IT coverage reference {value}"),
        }
    }
}

impl Error for ItCoverageErrorV1 {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_inventory_covers_every_qualification_domain_exactly_once() {
        let inventory = seed_it_coverage_inventory_v1("main@audit", 1).unwrap();
        assert_eq!(inventory.domains.len(), ALL_IT_DOMAINS_V1.len());
        inventory.validate().unwrap();
    }

    #[test]
    fn legacy_computing_remains_explicitly_unmapped_not_silently_omitted() {
        let inventory = seed_it_coverage_inventory_v1("main@audit", 1).unwrap();
        let legacy = inventory.domain(ItDomainV1::LegacyComputing).unwrap();
        assert_eq!(legacy.footprint(), ItImplementationFootprintV1::Unmapped);
        assert!(!legacy.gap_tags.is_empty());
    }

    #[test]
    fn windows_evidence_does_not_become_windows_admin_qualification() {
        let inventory = seed_it_coverage_inventory_v1("main@audit", 1).unwrap();
        let windows = inventory.domain(ItDomainV1::Windows).unwrap();
        assert_eq!(
            windows.footprint(),
            ItImplementationFootprintV1::FragmentedImplementation
        );
        assert!(windows.gap_tags.contains("active-directory-admin"));
        assert!(!windows.has_operational_tooling());
    }

    #[test]
    fn networking_records_active_programs_without_claiming_merge_or_qualification() {
        let inventory = seed_it_coverage_inventory_v1("main@audit", 1).unwrap();
        let networking = inventory.domain(ItDomainV1::Networking).unwrap();
        assert!(networking.signals.iter().any(|signal| matches!(
            signal.source,
            ItCoverageSourceV1::Issue(1100)
        )));
        assert!(networking.signals.iter().any(|signal| matches!(
            signal.source,
            ItCoverageSourceV1::PullRequest(1494)
        )));
        assert!(networking.has_benchmark_signal());
    }

    #[test]
    fn inventory_surfaces_domains_without_benchmark_evidence() {
        let inventory = seed_it_coverage_inventory_v1("main@audit", 1).unwrap();
        let missing = inventory.domains_without_benchmarks();
        assert!(missing.contains(&ItDomainV1::LegacyComputing));
        assert!(missing.contains(&ItDomainV1::WirelessTelecom));
    }
}
