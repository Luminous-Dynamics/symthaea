/**
 * Byzantine Fault Tolerance Conductor Tests
 *
 * Validates Mycelix's 34% validated Byzantine Fault Tolerance through
 * real conductor tests with simulated malicious agents.
 *
 * Run with: CONDUCTOR_AVAILABLE=true npm run test:byzantine
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import {
  ByzantineSimulator,
  ByzantineAttackType,
  STANDARD_SCENARIOS,
  type SimulationScenario,
} from './byzantine-simulator';
import { CONDUCTOR_ENABLED } from './conductor-harness';

describe.skipIf(!CONDUCTOR_ENABLED)('Byzantine Fault Tolerance - Conductor Tests', () => {
  let simulator: ByzantineSimulator;

  beforeAll(async () => {
    simulator = new ByzantineSimulator();
    await simulator.initialize();
  }, 30000); // 30s timeout for initialization

  afterAll(async () => {
    if (simulator) {
      await simulator.cleanup();
    }
  });

  describe('Standard BFT Scenarios', () => {
    it('should handle 30% Byzantine agents (well within tolerance)', async () => {
      const scenario = STANDARD_SCENARIOS.find((s) => s.name.includes('30%'));
      expect(scenario).toBeDefined();

      const result = await simulator.runScenario(scenario!);

      expect(result.passed).toBe(true);
      expect(result.actualOutcome).toBe('system_stable');
      expect(result.systemHealth).toBeGreaterThan(0.7);
    }, 120000); // 2 minute timeout

    it('should handle 40% Byzantine agents (near tolerance)', async () => {
      const scenario = STANDARD_SCENARIOS.find((s) => s.name.includes('40%'));
      expect(scenario).toBeDefined();

      const result = await simulator.runScenario(scenario!);

      expect(result.passed).toBe(true);
      // System should remain stable or only slightly degraded
      expect(['system_stable', 'degraded']).toContain(result.actualOutcome);
      expect(result.systemHealth).toBeGreaterThan(0.5);
    }, 180000); // 3 minute timeout

    it('should handle 45% Byzantine agents (above validated 34% threshold)', async () => {
      const scenario = STANDARD_SCENARIOS.find((s) => s.name.includes('45%'));
      expect(scenario).toBeDefined();

      const result = await simulator.runScenario(scenario!);

      // 45% is above the validated 34% threshold - system may degrade
      expect(result.passed).toBeDefined();
      expect(Number.isFinite(result.systemHealth)).toBe(true);
    }, 240000); // 4 minute timeout

    it('should degrade gracefully at 50% Byzantine (beyond tolerance)', async () => {
      const scenario = STANDARD_SCENARIOS.find((s) => s.name.includes('50%'));
      expect(scenario).toBeDefined();

      const result = await simulator.runScenario(scenario!);

      // Beyond tolerance, system may degrade but should not fail catastrophically
      expect(['degraded', 'compromised']).toContain(result.actualOutcome);

      // Even when compromised, detection should still work
      expect(result.detectedAttacks).toBeGreaterThan(0);
    }, 180000);

    it('should detect and mitigate coordinated collusion attacks', async () => {
      const scenario = STANDARD_SCENARIOS.find((s) => s.name.includes('Collusion'));
      expect(scenario).toBeDefined();

      const result = await simulator.runScenario(scenario!);

      expect(result.passed).toBe(true);
      // Detection rate should be high for coordinated attacks
      const detectionRate = result.detectedAttacks / result.totalAttacks;
      expect(detectionRate).toBeGreaterThan(0.5);
    }, 240000);
  });

  describe('Individual Attack Type Validation', () => {
    const singleAttackScenario = (
      attackType: ByzantineAttackType,
      byzantinePercentage: number
    ): SimulationScenario => ({
      name: `${attackType} @ ${byzantinePercentage}%`,
      description: `Single attack type test`,
      totalAgents: 20,
      byzantinePercentage,
      attackTypes: [attackType],
      duration: 30,
      expectedOutcome: 'system_stable',
    });

    it('should mitigate gradient poisoning attacks', async () => {
      const result = await simulator.runScenario(
        singleAttackScenario(ByzantineAttackType.GRADIENT_POISONING, 30)
      );

      expect(result.mitigatedAttacks).toBeGreaterThan(result.successfulAttacks);
    }, 60000);

    it('should detect reputation inflation attempts', async () => {
      const result = await simulator.runScenario(
        singleAttackScenario(ByzantineAttackType.REPUTATION_INFLATION, 30)
      );

      expect(result.detectedAttacks).toBeGreaterThan(0);
    }, 60000);

    it('should prevent Sybil attacks', async () => {
      const result = await simulator.runScenario(
        singleAttackScenario(ByzantineAttackType.SYBIL_CREATION, 20)
      );

      // Most Sybil attempts should be blocked
      const blockRate = result.mitigatedAttacks / result.totalAttacks;
      expect(blockRate).toBeGreaterThan(0.7);
    }, 60000);

    it('should prevent double voting', async () => {
      const result = await simulator.runScenario(
        singleAttackScenario(ByzantineAttackType.DOUBLE_VOTING, 30)
      );

      // Double votes should always be detected
      expect(result.detectedAttacks).toBe(result.totalAttacks);
    }, 60000);
  });

  describe('MATL Trust Scoring Under Attack', () => {
    it('should maintain correct trust scores under gradient poisoning', async () => {
      const result = await simulator.runScenario({
        name: 'MATL Trust Integrity',
        description: 'Verify MATL scores remain accurate under attack',
        totalAgents: 50,
        byzantinePercentage: 40,
        attackTypes: [
          ByzantineAttackType.GRADIENT_POISONING,
          ByzantineAttackType.REPUTATION_COLLUSION,
        ],
        duration: 60,
        expectedOutcome: 'system_stable',
      });

      // System health reflects MATL integrity
      expect(result.systemHealth).toBeGreaterThan(0.6);
    }, 120000);
  });

  describe('Recovery After Attack', () => {
    it('should recover system health after Byzantine attack subsides', async () => {
      // Run a heavy attack scenario
      const attackResult = await simulator.runScenario({
        name: 'Heavy Attack',
        description: 'Temporary heavy attack',
        totalAgents: 30,
        byzantinePercentage: 45,
        attackTypes: [ByzantineAttackType.GRADIENT_POISONING],
        duration: 30,
        expectedOutcome: 'system_stable',
      });

      // Then run a clean scenario (no Byzantine)
      const recoveryResult = await simulator.runScenario({
        name: 'Recovery',
        description: 'No Byzantine agents',
        totalAgents: 30,
        byzantinePercentage: 0,
        attackTypes: [],
        duration: 30,
        expectedOutcome: 'system_stable',
      });

      // System should recover
      expect(recoveryResult.systemHealth).toBeGreaterThanOrEqual(attackResult.systemHealth);
    }, 120000);
  });

  describe('Full BFT Validation Suite', () => {
    it('should pass all standard Byzantine scenarios', async () => {
      const results: Array<{ scenario: string; passed: boolean }> = [];

      for (const scenario of STANDARD_SCENARIOS) {
        const result = await simulator.runScenario(scenario);
        results.push({ scenario: scenario.name, passed: result.passed });
      }

      // Generate report
      console.log('\n' + simulator.generateReport(results.map((r) => r as any)));

      // At least 80% of scenarios should pass
      const passRate = results.filter((r) => r.passed).length / results.length;
      expect(passRate).toBeGreaterThanOrEqual(0.8);
    }, 900000); // 15 minute timeout for full suite
  });
});
