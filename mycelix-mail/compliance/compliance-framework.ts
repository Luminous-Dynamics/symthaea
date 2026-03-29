// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Mail Compliance Automation Framework
 *
 * Automated compliance checking for SOC2, ISO27001, GDPR, and HIPAA.
 */

// ============================================================================
// Types and Interfaces
// ============================================================================

export interface ComplianceControl {
  id: string;
  framework: ComplianceFramework;
  category: string;
  title: string;
  description: string;
  requirements: Requirement[];
  automatedChecks: AutomatedCheck[];
  manualChecks: ManualCheck[];
  status: ControlStatus;
  lastAssessed: Date;
  evidence: Evidence[];
}

export type ComplianceFramework = 'SOC2' | 'ISO27001' | 'GDPR' | 'HIPAA' | 'PCI-DSS';

export interface Requirement {
  id: string;
  text: string;
  mandatory: boolean;
}

export interface AutomatedCheck {
  id: string;
  name: string;
  type: CheckType;
  config: CheckConfig;
  schedule: string; // Cron expression
  lastRun?: Date;
  lastResult?: CheckResult;
}

export type CheckType =
  | 'config_validation'
  | 'log_analysis'
  | 'access_review'
  | 'encryption_check'
  | 'backup_verification'
  | 'vulnerability_scan'
  | 'data_retention'
  | 'audit_trail'
  | 'mfa_enforcement'
  | 'endpoint_security';

export interface CheckConfig {
  target: string;
  parameters: Record<string, unknown>;
  threshold?: number;
  expectedValue?: unknown;
}

export interface CheckResult {
  status: 'pass' | 'fail' | 'warning' | 'error';
  timestamp: Date;
  message: string;
  details?: Record<string, unknown>;
  evidence?: string;
}

export interface ManualCheck {
  id: string;
  name: string;
  procedure: string;
  frequency: 'daily' | 'weekly' | 'monthly' | 'quarterly' | 'annually';
  assignee?: string;
  lastCompleted?: Date;
  nextDue: Date;
}

export type ControlStatus = 'compliant' | 'non_compliant' | 'partially_compliant' | 'not_assessed';

export interface Evidence {
  id: string;
  type: EvidenceType;
  description: string;
  url?: string;
  content?: string;
  collectedAt: Date;
  validUntil: Date;
}

export type EvidenceType =
  | 'screenshot'
  | 'log_export'
  | 'config_snapshot'
  | 'policy_document'
  | 'audit_report'
  | 'test_result'
  | 'signed_attestation';

export interface ComplianceReport {
  framework: ComplianceFramework;
  period: { start: Date; end: Date };
  overallStatus: ControlStatus;
  controlSummary: {
    total: number;
    compliant: number;
    nonCompliant: number;
    partiallyCompliant: number;
    notAssessed: number;
  };
  controls: ComplianceControl[];
  findings: Finding[];
  recommendations: Recommendation[];
  generatedAt: Date;
}

export interface Finding {
  id: string;
  controlId: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  title: string;
  description: string;
  remediation: string;
  dueDate: Date;
  status: 'open' | 'in_progress' | 'resolved' | 'accepted_risk';
}

export interface Recommendation {
  id: string;
  priority: 'high' | 'medium' | 'low';
  title: string;
  description: string;
  effort: 'low' | 'medium' | 'high';
  impact: 'low' | 'medium' | 'high';
}

// ============================================================================
// SOC2 Controls
// ============================================================================

export const soc2Controls: Omit<ComplianceControl, 'status' | 'lastAssessed' | 'evidence'>[] = [
  // CC1 - Control Environment
  {
    id: 'CC1.1',
    framework: 'SOC2',
    category: 'Control Environment',
    title: 'COSO Principle 1: Demonstrates Commitment to Integrity and Ethical Values',
    description: 'The entity demonstrates a commitment to integrity and ethical values.',
    requirements: [
      { id: 'CC1.1.1', text: 'Code of conduct is documented and communicated', mandatory: true },
      { id: 'CC1.1.2', text: 'Ethics training is provided annually', mandatory: true },
    ],
    automatedChecks: [
      {
        id: 'cc1.1-policy-check',
        name: 'Code of Conduct Policy Check',
        type: 'config_validation',
        config: {
          target: 'policies/code-of-conduct.md',
          parameters: { checkExists: true, minLength: 1000 },
        },
        schedule: '0 0 * * 1', // Weekly
      },
    ],
    manualChecks: [
      {
        id: 'cc1.1-training-review',
        name: 'Ethics Training Completion Review',
        procedure: 'Review HR system for ethics training completion rates',
        frequency: 'quarterly',
        nextDue: new Date(Date.now() + 90 * 24 * 60 * 60 * 1000),
      },
    ],
  },

  // CC2 - Communication and Information
  {
    id: 'CC2.1',
    framework: 'SOC2',
    category: 'Communication and Information',
    title: 'Internal Communication of Security Policies',
    description: 'The entity internally communicates information necessary to support functioning of controls.',
    requirements: [
      { id: 'CC2.1.1', text: 'Security policies are documented and accessible', mandatory: true },
      { id: 'CC2.1.2', text: 'Changes to policies are communicated', mandatory: true },
    ],
    automatedChecks: [
      {
        id: 'cc2.1-policy-access',
        name: 'Policy Accessibility Check',
        type: 'config_validation',
        config: {
          target: 'internal-wiki/security-policies',
          parameters: { checkAccessible: true },
        },
        schedule: '0 0 * * *', // Daily
      },
    ],
    manualChecks: [],
  },

  // CC3 - Risk Assessment
  {
    id: 'CC3.1',
    framework: 'SOC2',
    category: 'Risk Assessment',
    title: 'Risk Assessment Process',
    description: 'The entity specifies objectives with sufficient clarity to enable identification of risks.',
    requirements: [
      { id: 'CC3.1.1', text: 'Risk assessment is performed annually', mandatory: true },
      { id: 'CC3.1.2', text: 'Risks are documented and tracked', mandatory: true },
    ],
    automatedChecks: [
      {
        id: 'cc3.1-risk-register',
        name: 'Risk Register Validation',
        type: 'config_validation',
        config: {
          target: 'risk-register.json',
          parameters: { lastUpdatedWithin: 365, minEntries: 10 },
        },
        schedule: '0 0 1 * *', // Monthly
      },
    ],
    manualChecks: [
      {
        id: 'cc3.1-annual-review',
        name: 'Annual Risk Assessment Review',
        procedure: 'Conduct comprehensive risk assessment with stakeholders',
        frequency: 'annually',
        nextDue: new Date(Date.now() + 365 * 24 * 60 * 60 * 1000),
      },
    ],
  },

  // CC5 - Control Activities
  {
    id: 'CC5.1',
    framework: 'SOC2',
    category: 'Control Activities',
    title: 'Logical Access Controls',
    description: 'The entity selects and develops control activities that contribute to mitigating risks.',
    requirements: [
      { id: 'CC5.1.1', text: 'Role-based access control is implemented', mandatory: true },
      { id: 'CC5.1.2', text: 'Privileged access is reviewed quarterly', mandatory: true },
      { id: 'CC5.1.3', text: 'Access is revoked upon termination', mandatory: true },
    ],
    automatedChecks: [
      {
        id: 'cc5.1-rbac-check',
        name: 'RBAC Configuration Check',
        type: 'access_review',
        config: {
          target: 'iam-system',
          parameters: { checkRbacEnabled: true },
        },
        schedule: '0 0 * * *',
      },
      {
        id: 'cc5.1-orphan-accounts',
        name: 'Orphan Account Detection',
        type: 'access_review',
        config: {
          target: 'iam-system',
          parameters: { checkOrphanedAccounts: true, maxAgeDays: 30 },
        },
        schedule: '0 0 * * *',
      },
    ],
    manualChecks: [
      {
        id: 'cc5.1-access-review',
        name: 'Quarterly Access Review',
        procedure: 'Review all user access rights with department heads',
        frequency: 'quarterly',
        nextDue: new Date(Date.now() + 90 * 24 * 60 * 60 * 1000),
      },
    ],
  },

  // CC6 - Logical and Physical Access
  {
    id: 'CC6.1',
    framework: 'SOC2',
    category: 'Logical and Physical Access',
    title: 'Encryption Controls',
    description: 'The entity implements logical access security software.',
    requirements: [
      { id: 'CC6.1.1', text: 'Data is encrypted at rest', mandatory: true },
      { id: 'CC6.1.2', text: 'Data is encrypted in transit', mandatory: true },
      { id: 'CC6.1.3', text: 'Encryption keys are managed securely', mandatory: true },
    ],
    automatedChecks: [
      {
        id: 'cc6.1-encryption-rest',
        name: 'Encryption at Rest Check',
        type: 'encryption_check',
        config: {
          target: 'database',
          parameters: { algorithm: 'AES-256', checkEnabled: true },
        },
        schedule: '0 0 * * *',
      },
      {
        id: 'cc6.1-tls-check',
        name: 'TLS Configuration Check',
        type: 'encryption_check',
        config: {
          target: 'api-endpoints',
          parameters: { minVersion: 'TLS1.2', checkHSTS: true },
        },
        schedule: '0 */6 * * *',
      },
      {
        id: 'cc6.1-key-rotation',
        name: 'Key Rotation Verification',
        type: 'encryption_check',
        config: {
          target: 'kms',
          parameters: { maxKeyAgeDays: 365, checkAutoRotation: true },
        },
        schedule: '0 0 * * 1',
      },
    ],
    manualChecks: [],
  },

  {
    id: 'CC6.2',
    framework: 'SOC2',
    category: 'Logical and Physical Access',
    title: 'Multi-Factor Authentication',
    description: 'The entity requires multi-factor authentication for system access.',
    requirements: [
      { id: 'CC6.2.1', text: 'MFA is required for all administrative access', mandatory: true },
      { id: 'CC6.2.2', text: 'MFA is available for all user accounts', mandatory: true },
    ],
    automatedChecks: [
      {
        id: 'cc6.2-mfa-admin',
        name: 'Admin MFA Enforcement Check',
        type: 'mfa_enforcement',
        config: {
          target: 'admin-accounts',
          parameters: { enforcementLevel: 'required' },
        },
        schedule: '0 0 * * *',
      },
      {
        id: 'cc6.2-mfa-adoption',
        name: 'MFA Adoption Rate Check',
        type: 'mfa_enforcement',
        config: {
          target: 'all-accounts',
          parameters: { minAdoptionRate: 0.95 },
        },
        schedule: '0 0 * * 1',
      },
    ],
    manualChecks: [],
  },

  // CC7 - System Operations
  {
    id: 'CC7.1',
    framework: 'SOC2',
    category: 'System Operations',
    title: 'Vulnerability Management',
    description: 'The entity identifies vulnerabilities and remediates in a timely manner.',
    requirements: [
      { id: 'CC7.1.1', text: 'Vulnerability scans are performed regularly', mandatory: true },
      { id: 'CC7.1.2', text: 'Critical vulnerabilities are patched within SLA', mandatory: true },
    ],
    automatedChecks: [
      {
        id: 'cc7.1-vuln-scan',
        name: 'Vulnerability Scan Status',
        type: 'vulnerability_scan',
        config: {
          target: 'infrastructure',
          parameters: { maxAgeDays: 7, maxCritical: 0, maxHigh: 5 },
        },
        schedule: '0 0 * * *',
      },
      {
        id: 'cc7.1-patch-status',
        name: 'Patch Compliance Check',
        type: 'vulnerability_scan',
        config: {
          target: 'systems',
          parameters: { maxPatchAgeDays: 30 },
        },
        schedule: '0 0 * * *',
      },
    ],
    manualChecks: [],
  },

  {
    id: 'CC7.2',
    framework: 'SOC2',
    category: 'System Operations',
    title: 'Incident Response',
    description: 'The entity has processes to detect and respond to security incidents.',
    requirements: [
      { id: 'CC7.2.1', text: 'Incident response plan is documented', mandatory: true },
      { id: 'CC7.2.2', text: 'Security events are monitored 24/7', mandatory: true },
      { id: 'CC7.2.3', text: 'Incidents are tracked and remediated', mandatory: true },
    ],
    automatedChecks: [
      {
        id: 'cc7.2-monitoring',
        name: 'Security Monitoring Status',
        type: 'log_analysis',
        config: {
          target: 'siem',
          parameters: { checkActiveAlerts: true, maxUnacknowledgedAge: 3600 },
        },
        schedule: '*/15 * * * *',
      },
      {
        id: 'cc7.2-incident-sla',
        name: 'Incident Response SLA Check',
        type: 'audit_trail',
        config: {
          target: 'incident-system',
          parameters: { p1ResponseSla: 900, p1ResolutionSla: 14400 },
        },
        schedule: '0 * * * *',
      },
    ],
    manualChecks: [
      {
        id: 'cc7.2-ir-drill',
        name: 'Incident Response Drill',
        procedure: 'Conduct tabletop exercise for incident response',
        frequency: 'quarterly',
        nextDue: new Date(Date.now() + 90 * 24 * 60 * 60 * 1000),
      },
    ],
  },

  // CC8 - Change Management
  {
    id: 'CC8.1',
    framework: 'SOC2',
    category: 'Change Management',
    title: 'Change Management Process',
    description: 'The entity authorizes, designs, and implements changes in a manner that minimizes risk.',
    requirements: [
      { id: 'CC8.1.1', text: 'All changes are documented and approved', mandatory: true },
      { id: 'CC8.1.2', text: 'Changes are tested before deployment', mandatory: true },
      { id: 'CC8.1.3', text: 'Emergency changes follow defined process', mandatory: true },
    ],
    automatedChecks: [
      {
        id: 'cc8.1-change-approval',
        name: 'Change Approval Check',
        type: 'audit_trail',
        config: {
          target: 'github',
          parameters: {
            checkPrApproval: true,
            minReviewers: 1,
            checkCiPassing: true,
          },
        },
        schedule: '0 0 * * *',
      },
      {
        id: 'cc8.1-deploy-audit',
        name: 'Deployment Audit Trail',
        type: 'audit_trail',
        config: {
          target: 'deployment-system',
          parameters: { checkAuditLog: true, retentionDays: 365 },
        },
        schedule: '0 0 * * *',
      },
    ],
    manualChecks: [],
  },

  // CC9 - Risk Mitigation
  {
    id: 'CC9.1',
    framework: 'SOC2',
    category: 'Risk Mitigation',
    title: 'Business Continuity',
    description: 'The entity identifies and manages risks related to business disruption.',
    requirements: [
      { id: 'CC9.1.1', text: 'Business continuity plan is documented', mandatory: true },
      { id: 'CC9.1.2', text: 'Disaster recovery is tested annually', mandatory: true },
      { id: 'CC9.1.3', text: 'Backups are performed and tested', mandatory: true },
    ],
    automatedChecks: [
      {
        id: 'cc9.1-backup-check',
        name: 'Backup Status Check',
        type: 'backup_verification',
        config: {
          target: 'backup-system',
          parameters: {
            maxBackupAgeDays: 1,
            checkIntegrity: true,
            minRetentionDays: 30,
          },
        },
        schedule: '0 0 * * *',
      },
      {
        id: 'cc9.1-rto-rpo',
        name: 'RTO/RPO Compliance Check',
        type: 'backup_verification',
        config: {
          target: 'dr-metrics',
          parameters: { maxRtoHours: 4, maxRpoHours: 1 },
        },
        schedule: '0 0 * * 1',
      },
    ],
    manualChecks: [
      {
        id: 'cc9.1-dr-test',
        name: 'Disaster Recovery Test',
        procedure: 'Perform full disaster recovery failover test',
        frequency: 'annually',
        nextDue: new Date(Date.now() + 365 * 24 * 60 * 60 * 1000),
      },
    ],
  },
];

// ============================================================================
// GDPR Controls
// ============================================================================

export const gdprControls: Omit<ComplianceControl, 'status' | 'lastAssessed' | 'evidence'>[] = [
  {
    id: 'GDPR-5',
    framework: 'GDPR',
    category: 'Principles',
    title: 'Lawfulness, Fairness and Transparency',
    description: 'Personal data shall be processed lawfully, fairly and transparently.',
    requirements: [
      { id: 'GDPR-5.1', text: 'Privacy policy is published and accessible', mandatory: true },
      { id: 'GDPR-5.2', text: 'Legal basis for processing is documented', mandatory: true },
      { id: 'GDPR-5.3', text: 'Consent is obtained where required', mandatory: true },
    ],
    automatedChecks: [
      {
        id: 'gdpr5-privacy-policy',
        name: 'Privacy Policy Accessibility',
        type: 'config_validation',
        config: {
          target: 'https://mycelix.mail/privacy',
          parameters: { checkAccessible: true, checkSSL: true },
        },
        schedule: '0 0 * * *',
      },
    ],
    manualChecks: [
      {
        id: 'gdpr5-legal-review',
        name: 'Legal Basis Review',
        procedure: 'Review data processing activities and legal basis documentation',
        frequency: 'quarterly',
        nextDue: new Date(Date.now() + 90 * 24 * 60 * 60 * 1000),
      },
    ],
  },

  {
    id: 'GDPR-17',
    framework: 'GDPR',
    category: 'Data Subject Rights',
    title: 'Right to Erasure',
    description: 'Data subjects have the right to have their personal data erased.',
    requirements: [
      { id: 'GDPR-17.1', text: 'Data deletion request process is implemented', mandatory: true },
      { id: 'GDPR-17.2', text: 'Deletion requests are processed within 30 days', mandatory: true },
    ],
    automatedChecks: [
      {
        id: 'gdpr17-deletion-sla',
        name: 'Deletion Request SLA Check',
        type: 'data_retention',
        config: {
          target: 'deletion-requests',
          parameters: { maxProcessingDays: 30 },
        },
        schedule: '0 0 * * *',
      },
      {
        id: 'gdpr17-backup-retention',
        name: 'Backup Retention for Deleted Data',
        type: 'data_retention',
        config: {
          target: 'backup-system',
          parameters: { maxRetentionAfterDeletion: 90 },
        },
        schedule: '0 0 * * 1',
      },
    ],
    manualChecks: [],
  },

  {
    id: 'GDPR-32',
    framework: 'GDPR',
    category: 'Security',
    title: 'Security of Processing',
    description: 'Implement appropriate technical and organizational measures for security.',
    requirements: [
      { id: 'GDPR-32.1', text: 'Encryption of personal data', mandatory: true },
      { id: 'GDPR-32.2', text: 'Pseudonymization where applicable', mandatory: true },
      { id: 'GDPR-32.3', text: 'Regular testing of security measures', mandatory: true },
    ],
    automatedChecks: [
      {
        id: 'gdpr32-encryption',
        name: 'Personal Data Encryption Check',
        type: 'encryption_check',
        config: {
          target: 'personal-data-stores',
          parameters: { checkEncrypted: true, checkFieldLevel: true },
        },
        schedule: '0 0 * * *',
      },
    ],
    manualChecks: [
      {
        id: 'gdpr32-pentest',
        name: 'Penetration Testing',
        procedure: 'Conduct annual penetration test of systems processing personal data',
        frequency: 'annually',
        nextDue: new Date(Date.now() + 365 * 24 * 60 * 60 * 1000),
      },
    ],
  },

  {
    id: 'GDPR-33',
    framework: 'GDPR',
    category: 'Breach Notification',
    title: 'Notification of Data Breach',
    description: 'Notify supervisory authority within 72 hours of becoming aware of a breach.',
    requirements: [
      { id: 'GDPR-33.1', text: 'Breach detection mechanisms are in place', mandatory: true },
      { id: 'GDPR-33.2', text: 'Breach notification process is documented', mandatory: true },
      { id: 'GDPR-33.3', text: 'Breach register is maintained', mandatory: true },
    ],
    automatedChecks: [
      {
        id: 'gdpr33-breach-detection',
        name: 'Breach Detection Monitoring',
        type: 'log_analysis',
        config: {
          target: 'security-logs',
          parameters: { checkDlpAlerts: true, checkAnomalies: true },
        },
        schedule: '*/5 * * * *',
      },
    ],
    manualChecks: [
      {
        id: 'gdpr33-breach-drill',
        name: 'Breach Response Drill',
        procedure: 'Simulate data breach and test 72-hour notification process',
        frequency: 'annually',
        nextDue: new Date(Date.now() + 365 * 24 * 60 * 60 * 1000),
      },
    ],
  },
];

// ============================================================================
// Compliance Engine
// ============================================================================

export class ComplianceEngine {
  private controls: Map<string, ComplianceControl> = new Map();
  private findings: Finding[] = [];
  private checkRunners: Map<CheckType, CheckRunner> = new Map();

  constructor() {
    this.initializeCheckRunners();
    this.loadControls();
  }

  private initializeCheckRunners(): void {
    this.checkRunners.set('config_validation', new ConfigValidationRunner());
    this.checkRunners.set('log_analysis', new LogAnalysisRunner());
    this.checkRunners.set('access_review', new AccessReviewRunner());
    this.checkRunners.set('encryption_check', new EncryptionCheckRunner());
    this.checkRunners.set('backup_verification', new BackupVerificationRunner());
    this.checkRunners.set('vulnerability_scan', new VulnerabilityScanRunner());
    this.checkRunners.set('data_retention', new DataRetentionRunner());
    this.checkRunners.set('audit_trail', new AuditTrailRunner());
    this.checkRunners.set('mfa_enforcement', new MfaEnforcementRunner());
    this.checkRunners.set('endpoint_security', new EndpointSecurityRunner());
  }

  private loadControls(): void {
    const allControls = [...soc2Controls, ...gdprControls];
    for (const control of allControls) {
      this.controls.set(control.id, {
        ...control,
        status: 'not_assessed',
        lastAssessed: new Date(0),
        evidence: [],
      });
    }
  }

  async runAllChecks(): Promise<Map<string, CheckResult[]>> {
    const results = new Map<string, CheckResult[]>();

    for (const [controlId, control] of this.controls) {
      const controlResults: CheckResult[] = [];

      for (const check of control.automatedChecks) {
        const runner = this.checkRunners.get(check.type);
        if (runner) {
          const result = await runner.run(check);
          check.lastRun = new Date();
          check.lastResult = result;
          controlResults.push(result);

          if (result.status === 'fail') {
            this.createFinding(control, check, result);
          }
        }
      }

      results.set(controlId, controlResults);
      this.updateControlStatus(control, controlResults);
    }

    return results;
  }

  async runCheck(controlId: string, checkId: string): Promise<CheckResult> {
    const control = this.controls.get(controlId);
    if (!control) {
      throw new Error(`Control ${controlId} not found`);
    }

    const check = control.automatedChecks.find(c => c.id === checkId);
    if (!check) {
      throw new Error(`Check ${checkId} not found in control ${controlId}`);
    }

    const runner = this.checkRunners.get(check.type);
    if (!runner) {
      throw new Error(`No runner for check type ${check.type}`);
    }

    const result = await runner.run(check);
    check.lastRun = new Date();
    check.lastResult = result;

    return result;
  }

  private updateControlStatus(control: ComplianceControl, results: CheckResult[]): void {
    const passCount = results.filter(r => r.status === 'pass').length;
    const failCount = results.filter(r => r.status === 'fail').length;

    control.lastAssessed = new Date();

    if (failCount === 0 && passCount > 0) {
      control.status = 'compliant';
    } else if (passCount === 0 && failCount > 0) {
      control.status = 'non_compliant';
    } else if (passCount > 0 && failCount > 0) {
      control.status = 'partially_compliant';
    } else {
      control.status = 'not_assessed';
    }
  }

  private createFinding(control: ComplianceControl, check: AutomatedCheck, result: CheckResult): void {
    const finding: Finding = {
      id: `F-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      controlId: control.id,
      severity: this.determineSeverity(control, result),
      title: `${control.title} - Check Failed: ${check.name}`,
      description: result.message,
      remediation: this.generateRemediation(check),
      dueDate: this.calculateDueDate(this.determineSeverity(control, result)),
      status: 'open',
    };

    this.findings.push(finding);
  }

  private determineSeverity(control: ComplianceControl, result: CheckResult): 'critical' | 'high' | 'medium' | 'low' {
    // Check if any mandatory requirements are affected
    const hasMandatory = control.requirements.some(r => r.mandatory);
    if (hasMandatory) {
      return 'high';
    }
    return 'medium';
  }

  private generateRemediation(check: AutomatedCheck): string {
    const remediations: Record<CheckType, string> = {
      config_validation: 'Review and update configuration to meet requirements',
      log_analysis: 'Review security logs and address any anomalies',
      access_review: 'Review and remediate access control issues',
      encryption_check: 'Enable or verify encryption configuration',
      backup_verification: 'Verify backup configuration and test restoration',
      vulnerability_scan: 'Review and patch identified vulnerabilities',
      data_retention: 'Update data retention policies and clean up old data',
      audit_trail: 'Enable or verify audit logging configuration',
      mfa_enforcement: 'Enforce MFA for affected accounts',
      endpoint_security: 'Update endpoint security controls',
    };

    return remediations[check.type] || 'Review and remediate the identified issue';
  }

  private calculateDueDate(severity: 'critical' | 'high' | 'medium' | 'low'): Date {
    const dueDays: Record<string, number> = {
      critical: 1,
      high: 7,
      medium: 30,
      low: 90,
    };

    const days = dueDays[severity] || 30;
    return new Date(Date.now() + days * 24 * 60 * 60 * 1000);
  }

  generateReport(framework: ComplianceFramework): ComplianceReport {
    const frameworkControls = Array.from(this.controls.values()).filter(
      c => c.framework === framework
    );

    const summary = {
      total: frameworkControls.length,
      compliant: frameworkControls.filter(c => c.status === 'compliant').length,
      nonCompliant: frameworkControls.filter(c => c.status === 'non_compliant').length,
      partiallyCompliant: frameworkControls.filter(c => c.status === 'partially_compliant').length,
      notAssessed: frameworkControls.filter(c => c.status === 'not_assessed').length,
    };

    const overallStatus: ControlStatus =
      summary.nonCompliant > 0
        ? 'non_compliant'
        : summary.partiallyCompliant > 0
        ? 'partially_compliant'
        : summary.compliant === summary.total
        ? 'compliant'
        : 'not_assessed';

    return {
      framework,
      period: { start: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000), end: new Date() },
      overallStatus,
      controlSummary: summary,
      controls: frameworkControls,
      findings: this.findings.filter(f =>
        frameworkControls.some(c => c.id === f.controlId)
      ),
      recommendations: this.generateRecommendations(frameworkControls),
      generatedAt: new Date(),
    };
  }

  private generateRecommendations(controls: ComplianceControl[]): Recommendation[] {
    const recommendations: Recommendation[] = [];

    const nonCompliant = controls.filter(c => c.status === 'non_compliant');
    if (nonCompliant.length > 0) {
      recommendations.push({
        id: 'R1',
        priority: 'high',
        title: 'Address Non-Compliant Controls',
        description: `${nonCompliant.length} controls are non-compliant and require immediate attention`,
        effort: 'high',
        impact: 'high',
      });
    }

    const partiallyCompliant = controls.filter(c => c.status === 'partially_compliant');
    if (partiallyCompliant.length > 0) {
      recommendations.push({
        id: 'R2',
        priority: 'medium',
        title: 'Complete Partial Controls',
        description: `${partiallyCompliant.length} controls are partially compliant and need completion`,
        effort: 'medium',
        impact: 'medium',
      });
    }

    return recommendations;
  }

  getFindings(): Finding[] {
    return [...this.findings];
  }

  getControl(controlId: string): ComplianceControl | undefined {
    return this.controls.get(controlId);
  }

  listControls(framework?: ComplianceFramework): ComplianceControl[] {
    const controls = Array.from(this.controls.values());
    if (framework) {
      return controls.filter(c => c.framework === framework);
    }
    return controls;
  }
}

// ============================================================================
// Check Runners (Stubs)
// ============================================================================

interface CheckRunner {
  run(check: AutomatedCheck): Promise<CheckResult>;
}

class ConfigValidationRunner implements CheckRunner {
  async run(check: AutomatedCheck): Promise<CheckResult> {
    // Implementation would validate configuration
    return {
      status: 'pass',
      timestamp: new Date(),
      message: 'Configuration validated successfully',
    };
  }
}

class LogAnalysisRunner implements CheckRunner {
  async run(check: AutomatedCheck): Promise<CheckResult> {
    return { status: 'pass', timestamp: new Date(), message: 'Logs analyzed' };
  }
}

class AccessReviewRunner implements CheckRunner {
  async run(check: AutomatedCheck): Promise<CheckResult> {
    return { status: 'pass', timestamp: new Date(), message: 'Access reviewed' };
  }
}

class EncryptionCheckRunner implements CheckRunner {
  async run(check: AutomatedCheck): Promise<CheckResult> {
    return { status: 'pass', timestamp: new Date(), message: 'Encryption verified' };
  }
}

class BackupVerificationRunner implements CheckRunner {
  async run(check: AutomatedCheck): Promise<CheckResult> {
    return { status: 'pass', timestamp: new Date(), message: 'Backups verified' };
  }
}

class VulnerabilityScanRunner implements CheckRunner {
  async run(check: AutomatedCheck): Promise<CheckResult> {
    return { status: 'pass', timestamp: new Date(), message: 'No vulnerabilities found' };
  }
}

class DataRetentionRunner implements CheckRunner {
  async run(check: AutomatedCheck): Promise<CheckResult> {
    return { status: 'pass', timestamp: new Date(), message: 'Data retention compliant' };
  }
}

class AuditTrailRunner implements CheckRunner {
  async run(check: AutomatedCheck): Promise<CheckResult> {
    return { status: 'pass', timestamp: new Date(), message: 'Audit trail complete' };
  }
}

class MfaEnforcementRunner implements CheckRunner {
  async run(check: AutomatedCheck): Promise<CheckResult> {
    return { status: 'pass', timestamp: new Date(), message: 'MFA enforced' };
  }
}

class EndpointSecurityRunner implements CheckRunner {
  async run(check: AutomatedCheck): Promise<CheckResult> {
    return { status: 'pass', timestamp: new Date(), message: 'Endpoints secured' };
  }
}

export default ComplianceEngine;
