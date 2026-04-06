// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Praxis Integration
 *
 * hApp-specific adapter for Mycelix-Praxis providing:
 * - Educational credential issuance and verification
 * - Course completion certificates with epistemic claims
 * - Skill certification with proficiency levels
 * - Learner profile tracking with trust metrics
 * - Cross-hApp reputation integration via Bridge
 *
 * @packageDocumentation
 * @module integrations/praxis
 * @see {@link PraxisCredentialService} - Main service class
 * @see {@link getPraxisService} - Singleton accessor
 *
 * @example Issuing course completion certificate
 * ```typescript
 * import { getPraxisService } from '@mycelix/sdk/integrations/praxis';
 *
 * const praxis = getPraxisService();
 *
 * // Issue certificate for course completion
 * const credential = praxis.issueCertificate({
 *   studentId: 'student-alice',
 *   courseId: 'nix-fundamentals',
 *   courseName: 'NixOS Fundamentals',
 *   grade: 95,
 * });
 *
 * console.log(`Certificate ID: ${credential.id}`);
 * console.log(`Expires: ${new Date(credential.expiresAt!)}`);
 * ```
 *
 * @example Skill certification
 * ```typescript
 * const skillCert = praxis.issueSkillCertification({
 *   holderId: 'developer-bob',
 *   skillId: 'typescript-advanced',
 *   skillName: 'Advanced TypeScript',
 *   level: 88,
 * });
 *
 * // Verify the credential
 * const verification = praxis.verifyCredential(skillCert.id);
 * if (verification.valid) {
 *   console.log('Credential is valid');
 * }
 * ```
 */

import { LocalBridge, createReputationQuery } from '../../bridge/index.js';
import {
  claim,
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  type EpistemicClaim,
} from '../../epistemic/index.js';
import {
  createReputation,
  recordPositive,
  reputationValue,
  type ReputationScore,
} from '../../matl/index.js';


// ============================================================================
// Praxis-Specific Types
// ============================================================================

/**
 * Course completion status lifecycle
 *
 * @remarks
 * Status progression: `in_progress` → `completed` → `certified` → `expired`
 * - `in_progress`: Student actively enrolled
 * - `completed`: All requirements met, awaiting certification
 * - `certified`: Official credential issued
 * - `expired`: Credential past expiration date
 */
export type CompletionStatus = 'in_progress' | 'completed' | 'certified' | 'expired';

/** Course record */
export interface Course {
  id: string;
  title: string;
  description: string;
  provider: string;
  providerDid: string;
  duration: number;
  level: 'beginner' | 'intermediate' | 'advanced' | 'expert';
  prerequisites: string[];
  skills: string[];
}

/** Student course completion */
export interface CourseCompletion {
  studentId: string;
  courseId: string;
  startedAt: number;
  completedAt?: number;
  grade: number;
  status: CompletionStatus;
  projectsCompleted: number;
  timeSpent: number;
}

/** Educational credential */
export interface EducationalCredential {
  id: string;
  type: 'course_completion' | 'skill_certification' | 'degree' | 'badge';
  issuerId: string;
  issuerDid: string;
  holderId: string;
  holderDid: string;
  courseId?: string;
  skillId?: string;
  grade?: number;
  issuedAt: number;
  expiresAt?: number;
  claim: EpistemicClaim;
  verified: boolean;
}

/** Skill assessment result */
export interface SkillAssessment {
  skillId: string;
  skillName: string;
  level: number;
  confidence: number;
  evidenceCount: number;
  lastAssessed: number;
}

/** Learner profile */
export interface LearnerProfile {
  learnerId: string;
  reputation: ReputationScore;
  coursesCompleted: number;
  certificationsEarned: number;
  skills: SkillAssessment[];
  learningStreak: number;
  totalLearningHours: number;
  trustScore: number;
}

// ============================================================================
// Praxis Credential Service
// ============================================================================

/**
 * PraxisCredentialService - Educational credential and trust management
 *
 * @example
 * ```typescript
 * const praxis = new PraxisCredentialService();
 *
 * // Issue a course completion certificate
 * const credential = praxis.issueCertificate({
 *   studentId: 'student-123',
 *   courseId: 'nix-fundamentals',
 *   courseName: 'NixOS Fundamentals',
 *   grade: 95,
 * });
 *
 * // Verify a credential
 * const verified = praxis.verifyCredential(credential.id);
 * ```
 */
export class PraxisCredentialService {
  private learnerReputations: Map<string, ReputationScore> = new Map();
  private credentials: Map<string, EducationalCredential> = new Map();
  private completionCounts: Map<string, number> = new Map();
  private bridge: LocalBridge;

  constructor() {
    this.bridge = new LocalBridge();
    this.bridge.registerHapp('praxis');
  }

  /**
   * Issue a course completion certificate with epistemic claim
   *
   * @param params - Certificate parameters
   * @param params.studentId - Unique student identifier
   * @param params.courseId - Course identifier
   * @param params.courseName - Human-readable course name
   * @param params.grade - Grade achieved (0-100)
   * @param params.issuerDid - Optional custom issuer DID
   * @returns Issued educational credential with epistemic claim
   *
   * @throws Error if grade is not between 0 and 100
   *
   * @remarks
   * - Creates an E3 (Cryptographic) level epistemic claim
   * - Certificate expires after 1 year by default
   * - Automatically updates learner's reputation score
   * - Stores reputation in Bridge for cross-hApp queries
   *
   * @example
   * ```typescript
   * const cert = service.issueCertificate({
   *   studentId: 'alice-123',
   *   courseId: 'rust-programming',
   *   courseName: 'Rust Programming',
   *   grade: 92,
   * });
   *
   * console.log(cert.claim.content); // "Completed Rust Programming with grade 92%"
   * ```
   */
  issueCertificate(params: {
    studentId: string;
    courseId: string;
    courseName: string;
    grade: number;
    issuerDid?: string;
  }): EducationalCredential {
    const { studentId, courseId, courseName, grade, issuerDid } = params;

    if (grade < 0 || grade > 100) {
      throw new Error('Grade must be between 0 and 100');
    }

    // Create epistemic claim for the credential
    const certificateClaim = claim(`Completed ${courseName} with grade ${grade}%`)
      .withEmpirical(EmpiricalLevel.E3_Cryptographic)
      .withNormative(NormativeLevel.N2_Network)
      .withMateriality(MaterialityLevel.M2_Persistent)
      .build();

    // Create credential
    const credential: EducationalCredential = {
      id: `cred-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      type: 'course_completion',
      issuerId: 'praxis-issuer',
      issuerDid: issuerDid || 'did:mycelix:praxis',
      holderId: studentId,
      holderDid: `did:mycelix:${studentId}`,
      courseId,
      grade,
      issuedAt: Date.now(),
      expiresAt: Date.now() + 365 * 24 * 60 * 60 * 1000, // 1 year
      claim: certificateClaim,
      verified: true,
    };

    this.credentials.set(credential.id, credential);

    // Update learner reputation
    let learnerRep = this.learnerReputations.get(studentId) || createReputation(studentId);
    learnerRep = recordPositive(learnerRep);
    this.learnerReputations.set(studentId, learnerRep);

    // Update completion count
    const count = (this.completionCounts.get(studentId) || 0) + 1;
    this.completionCounts.set(studentId, count);

    // Store reputation in bridge for cross-hApp queries
    this.bridge.setReputation('praxis', studentId, learnerRep);

    return credential;
  }

  /**
   * Issue a skill proficiency certification
   *
   * @param params - Skill certification parameters
   * @param params.holderId - Person receiving the certification
   * @param params.skillId - Unique skill identifier
   * @param params.skillName - Human-readable skill name
   * @param params.level - Proficiency level (0-100)
   * @returns Skill certification credential
   *
   * @throws Error if level is not between 0 and 100
   *
   * @remarks
   * Skill certifications differ from course completions:
   * - Valid for 2 years (vs 1 year for courses)
   * - Can be issued independently of course enrollment
   * - Represent demonstrated ability, not just completion
   *
   * @example
   * ```typescript
   * const skillCert = service.issueSkillCertification({
   *   holderId: 'dev-alice',
   *   skillId: 'kubernetes',
   *   skillName: 'Kubernetes Administration',
   *   level: 85,
   * });
   * ```
   */
  issueSkillCertification(params: {
    holderId: string;
    skillId: string;
    skillName: string;
    level: number;
  }): EducationalCredential {
    const { holderId, skillId, skillName, level } = params;

    if (level < 0 || level > 100) {
      throw new Error('Level must be between 0 and 100');
    }

    const certificationClaim = claim(`Certified in ${skillName} at level ${level}%`)
      .withEmpirical(EmpiricalLevel.E3_Cryptographic)
      .withNormative(NormativeLevel.N2_Network)
      .withMateriality(MaterialityLevel.M2_Persistent)
      .build();

    const credential: EducationalCredential = {
      id: `skill-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      type: 'skill_certification',
      issuerId: 'praxis-skill-issuer',
      issuerDid: 'did:mycelix:praxis:skills',
      holderId,
      holderDid: `did:mycelix:${holderId}`,
      skillId,
      issuedAt: Date.now(),
      expiresAt: Date.now() + 2 * 365 * 24 * 60 * 60 * 1000, // 2 years
      claim: certificationClaim,
      verified: true,
    };

    this.credentials.set(credential.id, credential);

    return credential;
  }

  /**
   * Verify an educational credential's validity
   *
   * @param credentialId - The credential ID to verify
   * @returns Verification result with credential details if found
   *
   * @remarks
   * Verification checks:
   * 1. Credential exists in the system
   * 2. Credential has not expired
   *
   * @example
   * ```typescript
   * const result = service.verifyCredential('cred-abc123');
   *
   * if (result.valid) {
   *   console.log(`Holder: ${result.credential!.holderId}`);
   *   console.log(`Grade: ${result.credential!.grade}`);
   * } else {
   *   console.log(`Invalid: ${result.reason}`);
   * }
   * ```
   */
  verifyCredential(credentialId: string): {
    valid: boolean;
    credential: EducationalCredential | null;
    reason?: string;
  } {
    const credential = this.credentials.get(credentialId);

    if (!credential) {
      return { valid: false, credential: null, reason: 'Credential not found' };
    }

    // Check expiration
    if (credential.expiresAt && credential.expiresAt < Date.now()) {
      return { valid: false, credential, reason: 'Credential expired' };
    }

    return { valid: true, credential };
  }

  /**
   * Get comprehensive learner profile with trust metrics
   *
   * @param learnerId - Unique learner identifier
   * @returns Complete learner profile including credentials and trust score
   *
   * @remarks
   * The trust score is calculated from the learner's reputation,
   * which increases with each credential earned. Unknown learners
   * receive a default score of 0.5 (neutral).
   *
   * @example
   * ```typescript
   * const profile = service.getLearnerProfile('student-123');
   *
   * console.log(`Courses: ${profile.coursesCompleted}`);
   * console.log(`Certifications: ${profile.certificationsEarned}`);
   * console.log(`Trust: ${(profile.trustScore * 100).toFixed(1)}%`);
   * ```
   */
  getLearnerProfile(learnerId: string): LearnerProfile {
    const reputation = this.learnerReputations.get(learnerId) || createReputation(learnerId);

    // Count credentials for this learner
    let coursesCompleted = 0;
    let certificationsEarned = 0;

    for (const cred of this.credentials.values()) {
      if (cred.holderId === learnerId) {
        if (cred.type === 'course_completion') coursesCompleted++;
        if (cred.type === 'skill_certification') certificationsEarned++;
      }
    }

    return {
      learnerId,
      reputation,
      coursesCompleted,
      certificationsEarned,
      skills: [],
      learningStreak: 0,
      totalLearningHours: 0,
      trustScore: reputationValue(reputation),
    };
  }

  /**
   * Get all credentials for a holder
   */
  getCredentialsForHolder(holderId: string): EducationalCredential[] {
    return Array.from(this.credentials.values()).filter(
      (cred) => cred.holderId === holderId
    );
  }

  /**
   * Query learner reputation from other hApps
   */
  queryExternalReputation(learnerId: string): void {
    const query = createReputationQuery('praxis', learnerId);
    this.bridge.send('marketplace', query);
    this.bridge.send('mail', query);
  }

  /**
   * Check if learner is trusted
   */
  isLearnerTrusted(learnerId: string, minimumScore = 0.5): boolean {
    const reputation = this.learnerReputations.get(learnerId);
    if (!reputation) return false;
    return reputationValue(reputation) >= minimumScore;
  }
}

// ============================================================================
// Bridge Client (Holochain Zome Calls)
// ============================================================================

import { type MycelixClient } from '../../client/index.js';

const EDUNET_ROLE = 'praxis';

/**
 * PraxisBridgeClient - Direct Holochain zome calls for educational operations
 *
 * Provides the same capabilities as PraxisCredentialService but backed by the
 * Holochain conductor instead of in-memory Maps.
 */
export class PraxisBridgeClient {
  constructor(private client: MycelixClient) {}

  // --- learning_zome ---

  async createCourse(course: {
    title: string;
    description: string;
    provider: string;
    provider_did: string;
    duration: number;
    level: string;
    prerequisites: string[];
    skills: string[];
  }): Promise<Uint8Array> {
    return this.client.callZome({
      role_name: EDUNET_ROLE,
      zome_name: 'learning_zome',
      fn_name: 'create_course',
      payload: course,
    });
  }

  async getCourse(actionHash: Uint8Array): Promise<Record<string, unknown> | null> {
    return this.client.callZome({
      role_name: EDUNET_ROLE,
      zome_name: 'learning_zome',
      fn_name: 'get_course',
      payload: actionHash,
    });
  }

  async listCourses(): Promise<Record<string, unknown>[]> {
    return this.client.callZome({
      role_name: EDUNET_ROLE,
      zome_name: 'learning_zome',
      fn_name: 'list_courses',
      payload: null,
    });
  }

  async updateCourse(input: {
    original_action_hash: Uint8Array;
    updated_course: {
      title: string;
      description: string;
      provider: string;
      provider_did: string;
      duration: number;
      level: string;
      prerequisites: string[];
      skills: string[];
    };
  }): Promise<Uint8Array> {
    return this.client.callZome({
      role_name: EDUNET_ROLE,
      zome_name: 'learning_zome',
      fn_name: 'update_course',
      payload: input,
    });
  }

  async deleteCourse(actionHash: Uint8Array): Promise<Uint8Array> {
    return this.client.callZome({
      role_name: EDUNET_ROLE,
      zome_name: 'learning_zome',
      fn_name: 'delete_course',
      payload: actionHash,
    });
  }

  async enroll(courseActionHash: Uint8Array): Promise<void> {
    return this.client.callZome({
      role_name: EDUNET_ROLE,
      zome_name: 'learning_zome',
      fn_name: 'enroll',
      payload: courseActionHash,
    });
  }

  async getEnrolledCourses(): Promise<Record<string, unknown>[]> {
    return this.client.callZome({
      role_name: EDUNET_ROLE,
      zome_name: 'learning_zome',
      fn_name: 'get_enrolled_courses',
      payload: null,
    });
  }

  async getCourseEnrollments(courseActionHash: Uint8Array): Promise<Uint8Array[]> {
    return this.client.callZome({
      role_name: EDUNET_ROLE,
      zome_name: 'learning_zome',
      fn_name: 'get_course_enrollments',
      payload: courseActionHash,
    });
  }

  async updateProgress(progress: {
    course_hash: Uint8Array;
    completion_percentage: number;
    current_module?: string;
  }): Promise<Uint8Array> {
    return this.client.callZome({
      role_name: EDUNET_ROLE,
      zome_name: 'learning_zome',
      fn_name: 'update_progress',
      payload: progress,
    });
  }

  async getProgress(actionHash: Uint8Array): Promise<Record<string, unknown> | null> {
    return this.client.callZome({
      role_name: EDUNET_ROLE,
      zome_name: 'learning_zome',
      fn_name: 'get_progress',
      payload: actionHash,
    });
  }

  async recordActivity(activity: {
    course_hash: Uint8Array;
    activity_type: string;
    duration_minutes: number;
    metadata?: string;
  }): Promise<Uint8Array> {
    return this.client.callZome({
      role_name: EDUNET_ROLE,
      zome_name: 'learning_zome',
      fn_name: 'record_activity',
      payload: activity,
    });
  }

  // --- credential_zome ---

  async issueCredential(input: {
    learner: Uint8Array;
    course_id: string;
    credential_type: string;
    grade?: number;
    skills?: string[];
    metadata?: string;
  }): Promise<Uint8Array> {
    return this.client.callZome({
      role_name: EDUNET_ROLE,
      zome_name: 'credential_zome',
      fn_name: 'issue_credential',
      payload: input,
    });
  }

  async verifyCredential(actionHash: Uint8Array): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: EDUNET_ROLE,
      zome_name: 'credential_zome',
      fn_name: 'verify_credential',
      payload: actionHash,
    });
  }

  async getLearnerCredentials(learnerAgent: Uint8Array): Promise<Record<string, unknown>[]> {
    return this.client.callZome({
      role_name: EDUNET_ROLE,
      zome_name: 'credential_zome',
      fn_name: 'get_learner_credentials',
      payload: learnerAgent,
    });
  }

  async getCourseCredentials(courseId: string): Promise<Record<string, unknown>[]> {
    return this.client.callZome({
      role_name: EDUNET_ROLE,
      zome_name: 'credential_zome',
      fn_name: 'get_course_credentials',
      payload: courseId,
    });
  }

  async getIssuerCredentials(issuer: Uint8Array): Promise<Record<string, unknown>[]> {
    return this.client.callZome({
      role_name: EDUNET_ROLE,
      zome_name: 'credential_zome',
      fn_name: 'get_issuer_credentials',
      payload: issuer,
    });
  }

  async revokeCredential(input: {
    credential_hash: Uint8Array;
    reason: string;
  }): Promise<void> {
    return this.client.callZome({
      role_name: EDUNET_ROLE,
      zome_name: 'credential_zome',
      fn_name: 'revoke_credential',
      payload: input,
    });
  }

  async verifyEpistemicRequirements(input: {
    credential_hash: Uint8Array;
    required_level: string;
  }): Promise<boolean> {
    return this.client.callZome({
      role_name: EDUNET_ROLE,
      zome_name: 'credential_zome',
      fn_name: 'verify_epistemic_requirements',
      payload: input,
    });
  }

  async getEpistemicClassification(credentialHash: Uint8Array): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: EDUNET_ROLE,
      zome_name: 'credential_zome',
      fn_name: 'get_epistemic_classification',
      payload: credentialHash,
    });
  }
}

// Bridge client singleton
let praxisBridgeInstance: PraxisBridgeClient | null = null;

export function getPraxisBridgeClient(client: MycelixClient): PraxisBridgeClient {
  if (!praxisBridgeInstance) praxisBridgeInstance = new PraxisBridgeClient(client);
  return praxisBridgeInstance;
}

export function resetPraxisBridgeClient(): void {
  praxisBridgeInstance = null;
}

// ============================================================================
// Exports
// ============================================================================

export {
  claim,
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  createReputation,
  recordPositive,
  reputationValue,
};

// Default service instance
let defaultService: PraxisCredentialService | null = null;

/**
 * Get the default Praxis credential service instance
 */
export function getPraxisService(): PraxisCredentialService {
  if (!defaultService) {
    defaultService = new PraxisCredentialService();
  }
  return defaultService;
}
