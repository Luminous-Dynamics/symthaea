// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
'use strict';

const fs = require('fs');
const path = require('path');

const EXPECTED = Object.freeze([
  Object.freeze({ role: 'ContactAuthorityD4', sourceSha: 'c00fedffffd9d0240f5a5be44cac293bac6cef44', qualifierSha: '4210a5e51127f6c8a6c663766dc6d130bf4224df', qualifierPr: 4113, workflowName: 'HUM-WRENCH-001D4 Contact Authority', workflowId: 361892536, workflowPath: '.github/workflows/hum-wrench-001d4-contact-authority.yml', runId: 35426272913, jobName: 'exact-head-contact-authority-qualification', artifactName: 'hum-wrench-001d4-4210a5e51127f6c8a6c663766dc6d130bf4224df' }),
  Object.freeze({ role: 'VerifiedSurfaceSupportD5A', sourceSha: 'b64e62287cfb49797d280369545672cff87b48dd', qualifierSha: '32b49ed10987e7b3a48ed54dd8dd7b551d1731e8', qualifierPr: 4182, workflowName: 'HUM-WRENCH-001D5A Verified Surface Support', workflowId: 361896937, workflowPath: '.github/workflows/hum-wrench-001d5a-verified-support.yml', runId: 35431016942, jobName: 'exact-head-verified-surface-support-qualification', artifactName: 'hum-wrench-001d5a-32b49ed10987e7b3a48ed54dd8dd7b551d1731e8' }),
  Object.freeze({ role: 'MujocoContactBiasDyn002C', sourceSha: '67107f337ee454d4d9da859b7fd4dea8628cedf2', qualifierSha: '0b222328a6123a87716c8a535cb21179dc8e6774', qualifierPr: 4184, workflowName: 'HUM-DYN-002C MuJoCo Contact Bias', workflowId: 361532488, workflowPath: '.github/workflows/hum-dyn-002c-mujoco-contact-bias.yml', runId: 35431329409, jobName: 'exact-head-mujoco-qualification', artifactName: 'hum-dyn-002c-0b222328a6123a87716c8a535cb21179dc8e6774' }),
  Object.freeze({ role: 'CompleteContactEvidenceD5B', sourceSha: '156ed01855b2c6fd89118b5ac4063282added0cb', qualifierSha: '3d69c5f7006fc300ae538effde87ebec9b43486e', qualifierPr: 4186, workflowName: 'HUM-WRENCH-001D5B Complete Contact Evidence', workflowId: 361945663, workflowPath: '.github/workflows/hum-wrench-001d5b-complete-contact-evidence.yml', runId: 35431653533, jobName: 'exact-head-complete-contact-evidence-qualification', artifactName: 'hum-wrench-001d5b-3d69c5f7006fc300ae538effde87ebec9b43486e' }),
  Object.freeze({ role: 'IndependentCompleteVerifierD5B2', sourceSha: '47f6e0339a36ddfcef025e6277adbab1b91b6163', qualifierSha: '75e91066db405daf49d1b78f9630d68e4ff72eeb', qualifierPr: 4193, workflowName: 'HUM-WRENCH-001D5B2 Independent Complete Contact Verifier', workflowId: 361971294, workflowPath: '.github/workflows/hum-wrench-001d5b2-independent-verifier.yml', runId: 35434249956, jobName: 'exact-head-independent-complete-contact-verifier', artifactName: 'hum-wrench-001d5b2-75e91066db405daf49d1b78f9630d68e4ff72eeb' }),
  Object.freeze({ role: 'PreparedAuthorityMappingD5B3', sourceSha: '44a0cdf8b3c939d02b8dad89a531d18b50d4979a', qualifierSha: '66343eb1f9d57fb9d22458bb22252c2059cbe469', qualifierPr: 4196, workflowName: 'HUM-WRENCH-001D5B3 Prepared Contact Authority Evidence', workflowId: 361973645, workflowPath: '.github/workflows/hum-wrench-001d5b3-prepared-authority.yml', runId: 35434645590, jobName: 'exact-head-prepared-authority-mapping-qualification', artifactName: 'hum-wrench-001d5b3-66343eb1f9d57fb9d22458bb22252c2059cbe469' }),
  Object.freeze({ role: 'IndependentPreparedVerifierD5B4', sourceSha: '9e8248fbdc92af1179ff3a29fed9f84d8d52fa99', qualifierSha: 'bb4f2029e13308067660e4f0f0b59bdb6c77a984', qualifierPr: 4213, workflowName: 'HUM-WRENCH-001D5B4 Independent Prepared Authority Verifier', workflowId: 361999685, workflowPath: '.github/workflows/hum-wrench-001d5b4-prepared-verifier.yml', runId: 35437190217, jobName: 'exact-head-independent-prepared-authority-verifier', artifactName: 'hum-wrench-001d5b4-bb4f2029e13308067660e4f0f0b59bdb6c77a984' }),
  Object.freeze({ role: 'SealedPreAuthorityCandidateD5B5', sourceSha: '792ead8e08565995b0bfc5e3f6ab5ebbde6aa8df', qualifierSha: '015f82ff0df8746a0c0c06f14a00ba6b25e70ed6', qualifierPr: 4223, workflowName: 'HUM-WRENCH-001D5B5 Sealed Pre-Authority Surface Candidate', workflowId: 362002235, workflowPath: '.github/workflows/hum-wrench-001d5b5-surface-candidate.yml', runId: 35437475927, jobName: 'exact-head-sealed-pre-authority-candidate', artifactName: 'hum-wrench-001d5b5-015f82ff0df8746a0c0c06f14a00ba6b25e70ed6' }),
]);

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

async function collect({ github, context }) {
  const { owner, repo } = context.repo;
  const fullName = `${owner}/${repo}`;
  const { data: repository } = await github.rest.repos.get({ owner, repo });
  const expectedRef = `refs/heads/${repository.default_branch}`;

  requireCondition(context.eventName === 'workflow_dispatch', `unexpected event: ${context.eventName}`);
  requireCondition(context.ref === expectedRef, `QUAL-PROMO-002 may only be dispatched from the default branch: expected ${expectedRef}, got ${context.ref}`);

  const records = [];
  const snapshots = [];
  const digestPattern = /^sha256:([0-9a-f]{64})$/;

  for (const item of EXPECTED) {
    const { data: currentPr } = await github.rest.pulls.get({ owner, repo, pull_number: item.qualifierPr });
    requireCondition(currentPr.head?.sha === item.qualifierSha, `current PR head mismatch for ${item.role}`);
    requireCondition(currentPr.base?.sha === item.sourceSha, `current PR base mismatch for ${item.role}`);
    requireCondition(currentPr.head?.repo?.id === repository.id, `current PR head repository mismatch for ${item.role}`);
    requireCondition(currentPr.base?.repo?.id === repository.id, `current PR base repository mismatch for ${item.role}`);

    const { data: run } = await github.rest.actions.getWorkflowRun({ owner, repo, run_id: item.runId });
    const assertions = [
      [run.id === item.runId, `run id mismatch for ${item.role}`],
      [run.workflow_id === item.workflowId, `workflow id mismatch for ${item.role}`],
      [run.name === item.workflowName, `workflow name mismatch for ${item.role}`],
      [run.path === item.workflowPath, `workflow path mismatch for ${item.role}`],
      [run.head_sha === item.qualifierSha, `head sha mismatch for ${item.role}`],
      [run.event === 'pull_request', `event is not pull_request for ${item.role}`],
      [run.status === 'completed', `run not completed for ${item.role}: ${run.status}`],
      [run.conclusion === 'success', `run not successful for ${item.role}: ${run.conclusion}`],
      [run.run_attempt === 1, `QUAL-PROMO-002 v1 requires run attempt 1 for ${item.role}: ${run.run_attempt}`],
      [run.repository?.id === repository.id, `repository mismatch for ${item.role}`],
      [run.repository?.full_name === fullName, `repository name mismatch for ${item.role}`],
      [run.head_repository?.id === repository.id, `head repository mismatch for ${item.role}`],
      [run.head_repository?.full_name === fullName, `head repository name mismatch for ${item.role}`],
    ];
    for (const [ok, message] of assertions) requireCondition(ok, message);

    const prMatches = (run.pull_requests ?? []).filter((pr) =>
      pr.number === item.qualifierPr && pr.head?.sha === item.qualifierSha && pr.base?.sha === item.sourceSha &&
      pr.head?.repo?.id === repository.id && pr.base?.repo?.id === repository.id
    );
    requireCondition(prMatches.length === 1, `expected exactly one exact qualifier PR association for ${item.role}, found ${prMatches.length}`);

    const jobs = await github.paginate(github.rest.actions.listJobsForWorkflowRunAttempt, {
      owner, repo, run_id: item.runId, attempt_number: 1, per_page: 100,
    });
    requireCondition(jobs.length === 1, `expected exactly one focused job for ${item.role}, found ${jobs.length}`);
    const job = jobs[0];
    requireCondition(job.name === item.jobName, `focused job name mismatch for ${item.role}: ${job.name}`);
    requireCondition(job.head_sha === item.qualifierSha, `job head sha mismatch for ${item.role}: ${job.name}`);
    requireCondition(job.status === 'completed' && job.conclusion === 'success', `job did not execute successfully for ${item.role}: ${job.name} status=${job.status} conclusion=${job.conclusion}`);
    requireCondition(job.started_at && job.completed_at, `job lacks execution timestamps for ${item.role}: ${job.name}`);

    const steps = job.steps ?? [];
    requireCondition(steps.length > 0, `job has no executable steps for ${item.role}: ${job.name}`);
    for (const step of steps) {
      requireCondition(step.status === 'completed' && step.conclusion === 'success', `step did not execute successfully for ${item.role}: ${job.name} / ${step.name} status=${step.status} conclusion=${step.conclusion}`);
      requireCondition(step.started_at && step.completed_at, `step lacks execution timestamps for ${item.role}: ${job.name} / ${step.name}`);
    }

    const artifacts = await github.paginate(github.rest.actions.listWorkflowRunArtifacts, { owner, repo, run_id: item.runId, per_page: 100 });
    const matches = artifacts.filter((artifact) => artifact.name === item.artifactName);
    requireCondition(matches.length === 1, `expected exactly one evidence artifact for ${item.role}, found ${matches.length}`);
    const artifact = matches[0];
    requireCondition(!artifact.expired, `evidence artifact expired for ${item.role}`);
    requireCondition(artifact.workflow_run?.id === item.runId, `artifact run id mismatch for ${item.role}`);
    requireCondition(artifact.workflow_run?.head_sha === item.qualifierSha, `artifact head sha mismatch for ${item.role}`);
    requireCondition(artifact.workflow_run?.repository_id === repository.id, `artifact repository mismatch for ${item.role}`);
    requireCondition(artifact.workflow_run?.head_repository_id === repository.id, `artifact head repository mismatch for ${item.role}`);
    const digestMatch = digestPattern.exec(artifact.digest ?? '');
    requireCondition(digestMatch && !/^0{64}$/.test(digestMatch[1]), `artifact lacks non-placeholder canonical sha256 digest for ${item.role}: ${artifact.digest}`);

    records.push({
      role: item.role,
      source_sha: item.sourceSha,
      qualifier_sha: item.qualifierSha,
      workflow_name: item.workflowName,
      workflow_id: item.workflowId,
      run_id: item.runId,
      run_attempt: 1,
      execution_state: 'ExecutablePassed',
      evidence_artifact_id: artifact.id,
      evidence_artifact_name: artifact.name,
      evidence_artifact_sha256: digestMatch[1],
    });

    snapshots.push({
      role: item.role,
      qualifier_pr: { number: currentPr.number, state: currentPr.state, merged: currentPr.merged, head_sha: currentPr.head?.sha, base_sha: currentPr.base?.sha, head_repository_id: currentPr.head?.repo?.id, base_repository_id: currentPr.base?.repo?.id },
      run: { id: run.id, workflow_id: run.workflow_id, name: run.name, path: run.path, event: run.event, status: run.status, conclusion: run.conclusion, head_sha: run.head_sha, head_branch: run.head_branch, run_attempt: run.run_attempt, created_at: run.created_at, run_started_at: run.run_started_at, updated_at: run.updated_at },
      job: { id: job.id, name: job.name, head_sha: job.head_sha, status: job.status, conclusion: job.conclusion, started_at: job.started_at, completed_at: job.completed_at, steps: steps.map((step) => ({ number: step.number, name: step.name, status: step.status, conclusion: step.conclusion, started_at: step.started_at, completed_at: step.completed_at })) },
      artifact: { id: artifact.id, name: artifact.name, digest: artifact.digest, expired: artifact.expired, created_at: artifact.created_at, expires_at: artifact.expires_at, workflow_run: artifact.workflow_run },
    });
  }

  const outDir = path.join(process.env.GITHUB_WORKSPACE, 'qual-promo-002-evidence');
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(outDir, 'records.json'), `${JSON.stringify(records, null, 2)}\n`);
  fs.writeFileSync(path.join(outDir, 'github-api-evidence.json'), `${JSON.stringify({
    schema: 'symthaea.qual-promo-002.github-actions-evidence.v1',
    repository: fullName,
    repository_id: repository.id,
    collector_head_sha: context.sha,
    collector_ref: context.ref,
    collected_at: new Date().toISOString(),
    snapshots,
  }, null, 2)}\n`);
}

module.exports = { collect, EXPECTED };
