/**
 * @mycelix/sdk EduNet Integration Example
 *
 * Demonstrates educational credential issuance, verification, and learner profiles.
 *
 * Run with: npx tsx examples/09-edunet-integration.ts
 */

import {
  getEduNetService,
  EduNetCredentialService,
  type EducationalCredential,
  type LearnerProfile,
} from '../src/integrations/edunet/index.js';

console.log('=== EduNet Credential Integration ===\n');

const edunet = getEduNetService();

// === Issuing Course Completion Certificates ===

console.log('--- Issuing Course Certificates ---');

const students = {
  alice: 'student-alice-001',
  bob: 'student-bob-002',
  carol: 'student-carol-003',
};

// Alice completes multiple courses with high grades
const aliceCourses = [
  { courseId: 'nix-101', courseName: 'NixOS Fundamentals', grade: 95 },
  { courseId: 'nix-201', courseName: 'Advanced Nix Expressions', grade: 92 },
  { courseId: 'flake-101', courseName: 'Nix Flakes Workshop', grade: 98 },
];

console.log(`\nIssuing certificates for ${students.alice}:`);
const aliceCreds: EducationalCredential[] = [];
for (const course of aliceCourses) {
  const credential = edunet.issueCertificate({
    studentId: students.alice,
    courseId: course.courseId,
    courseName: course.courseName,
    grade: course.grade,
  });
  aliceCreds.push(credential);
  console.log(`  ✅ ${course.courseName}: Grade ${course.grade}%`);
  console.log(`     Credential ID: ${credential.id}`);
  console.log(`     Claim: ${credential.claim.content}`);
}

// Bob completes one course
console.log(`\nIssuing certificate for ${students.bob}:`);
const bobCred = edunet.issueCertificate({
  studentId: students.bob,
  courseId: 'nix-101',
  courseName: 'NixOS Fundamentals',
  grade: 78,
});
console.log(`  ✅ NixOS Fundamentals: Grade 78%`);

// === Skill Certifications ===

console.log('\n--- Issuing Skill Certifications ---');

// Alice gets skill certifications
const aliceSkills = [
  { skillId: 'typescript-advanced', skillName: 'Advanced TypeScript', level: 88 },
  { skillId: 'holochain-dev', skillName: 'Holochain Development', level: 75 },
];

console.log(`\nSkill certifications for ${students.alice}:`);
for (const skill of aliceSkills) {
  const skillCert = edunet.issueSkillCertification({
    holderId: students.alice,
    skillId: skill.skillId,
    skillName: skill.skillName,
    level: skill.level,
  });
  console.log(`  🎯 ${skill.skillName}: Level ${skill.level}%`);
  console.log(`     Valid for: 2 years`);
  console.log(`     Credential ID: ${skillCert.id}`);
}

// === Credential Verification ===

console.log('\n--- Credential Verification ---');

// Verify Alice's first credential
const verification1 = edunet.verifyCredential(aliceCreds[0].id);
console.log(`\nVerifying: ${aliceCreds[0].id}`);
console.log(`  Valid: ${verification1.valid ? '✅ Yes' : '❌ No'}`);
if (verification1.credential) {
  console.log(`  Course: ${verification1.credential.courseId}`);
  console.log(`  Grade: ${verification1.credential.grade}%`);
  console.log(`  Expires: ${new Date(verification1.credential.expiresAt!).toLocaleDateString()}`);
}

// Try to verify non-existent credential
const verification2 = edunet.verifyCredential('fake-credential-xyz');
console.log(`\nVerifying: fake-credential-xyz`);
console.log(`  Valid: ${verification2.valid ? '✅ Yes' : '❌ No'}`);
console.log(`  Reason: ${verification2.reason}`);

// === Learner Profiles ===

console.log('\n--- Learner Profiles ---');

for (const [name, learnerId] of Object.entries(students)) {
  const profile = edunet.getLearnerProfile(learnerId);
  console.log(`\n${name} (${learnerId}):`);
  console.log(`  Courses Completed: ${profile.coursesCompleted}`);
  console.log(`  Certifications Earned: ${profile.certificationsEarned}`);
  console.log(`  Trust Score: ${profile.trustScore.toFixed(3)}`);
  console.log(`  Learning Hours: ${profile.totalLearningHours}`);
  console.log(`  Learning Streak: ${profile.learningStreak} days`);
}

// === Trust Checks ===

console.log('\n--- Learner Trust Checks ---');
console.log(`Alice trusted (0.5): ${edunet.isLearnerTrusted(students.alice, 0.5)}`);
console.log(`Alice trusted (0.7): ${edunet.isLearnerTrusted(students.alice, 0.7)}`);
console.log(`Bob trusted (0.5): ${edunet.isLearnerTrusted(students.bob, 0.5)}`);
console.log(`Carol trusted (0.5): ${edunet.isLearnerTrusted(students.carol, 0.5)}`);

// === Get All Credentials for a Holder ===

console.log('\n--- All Credentials for Alice ---');
const allAliceCreds = edunet.getCredentialsForHolder(students.alice);
console.log(`Total credentials: ${allAliceCreds.length}`);
for (const cred of allAliceCreds) {
  const type = cred.type === 'course_completion' ? '📚' : '🎯';
  const identifier = cred.courseId || cred.skillId;
  console.log(`  ${type} ${cred.type}: ${identifier}`);
}

// === Real-World Pattern: Hiring Verification ===

console.log('\n--- Hiring Verification Example ---');

interface JobRequirement {
  type: 'course' | 'skill';
  id: string;
  minGrade?: number;
  minLevel?: number;
}

const jobRequirements: JobRequirement[] = [
  { type: 'course', id: 'nix-101', minGrade: 70 },
  { type: 'skill', id: 'typescript-advanced', minLevel: 80 },
];

function checkCandidateQualifications(
  candidateId: string,
  requirements: JobRequirement[]
): { qualified: boolean; missing: string[] } {
  const credentials = edunet.getCredentialsForHolder(candidateId);
  const missing: string[] = [];

  for (const req of requirements) {
    const matchingCred = credentials.find((c) => {
      if (req.type === 'course') {
        return c.type === 'course_completion' && c.courseId === req.id;
      } else {
        return c.type === 'skill_certification' && c.skillId === req.id;
      }
    });

    if (!matchingCred) {
      missing.push(`${req.type}: ${req.id}`);
      continue;
    }

    // Verify credential is still valid
    const verification = edunet.verifyCredential(matchingCred.id);
    if (!verification.valid) {
      missing.push(`${req.type}: ${req.id} (expired)`);
      continue;
    }

    // Check minimum requirements
    if (req.minGrade && matchingCred.grade && matchingCred.grade < req.minGrade) {
      missing.push(`${req.type}: ${req.id} (grade ${matchingCred.grade}% < required ${req.minGrade}%)`);
    }
  }

  return { qualified: missing.length === 0, missing };
}

for (const [name, candidateId] of Object.entries(students)) {
  const result = checkCandidateQualifications(candidateId, jobRequirements);
  const icon = result.qualified ? '✅' : '❌';
  console.log(`${icon} ${name}: ${result.qualified ? 'Qualified' : 'Not Qualified'}`);
  if (result.missing.length > 0) {
    console.log(`   Missing: ${result.missing.join(', ')}`);
  }
}

console.log('\n=== EduNet Integration Complete ===');
