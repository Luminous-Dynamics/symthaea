/**
 * Finance Cluster E2E Tests
 *
 * Tests the 3-currency system (SAP/TEND/MYCEL) via Tryorama.
 * Requires: mycelix-finance.happ bundle built via `just build-finance`
 */

import { describe, it, expect } from 'vitest';
import { Scenario, runScenario, dhtSync, type PlayerApp } from '@holochain/tryorama';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { existsSync } from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));

const FINANCE_HAPP_PATH = join(__dirname, '../../../../mycelix-finance/mycelix-finance.happ');

function skipIfNoBuild() {
  if (!existsSync(FINANCE_HAPP_PATH)) {
    console.warn(`Skipping: ${FINANCE_HAPP_PATH} not found. Run 'just build-finance' first.`);
    return true;
  }
  return false;
}

async function callZome<T>(
  player: PlayerApp,
  zome: string,
  fnName: string,
  payload: unknown = null,
): Promise<T> {
  return await player.appWs.callZome({
    role_name: 'finance',
    zome_name: zome,
    fn_name: fnName,
    payload,
  }) as T;
}

describe('Finance Cluster E2E', () => {
  if (skipIfNoBuild()) return;

  describe('SAP Balance Init', () => {
    it('should initialize a SAP balance for a new member', async () => {
      await runScenario(async (scenario: Scenario) => {
        const [alice] = await scenario.addPlayersWithApps([
          { appBundleSource: { path: FINANCE_HAPP_PATH } },
        ]);

        const memberDid = `did:mycelix:${alice.agentPubKey}`;

        // Initialize SAP balance (zero balance entry)
        const record = await callZome(alice, 'payments', 'initialize_sap_balance', memberDid);
        expect(record).toBeDefined();

        // Query the balance back
        const balance = await callZome(alice, 'payments', 'get_sap_balance', memberDid);
        expect(balance).toBeDefined();
      });
    });
  });

  describe('TEND Exchange Flow', () => {
    it('should record a time exchange between two agents', async () => {
      await runScenario(async (scenario: Scenario) => {
        const [alice, bob] = await scenario.addPlayersWithApps([
          { appBundleSource: { path: FINANCE_HAPP_PATH } },
          { appBundleSource: { path: FINANCE_HAPP_PATH } },
        ]);

        const bobDid = `did:mycelix:${bob.agentPubKey}`;

        // Alice records a time exchange where she provided service to Bob
        const exchange = await callZome(alice, 'tend', 'record_exchange', {
          receiver_did: bobDid,
          hours: 1.0,
          service_description: 'Community garden help',
          service_category: 'GeneralAssistance',
          cultural_alias: null,
          dao_did: 'did:mycelix:test-dao',
          service_date: null,
        });
        expect(exchange).toBeDefined();

        await dhtSync([alice, bob], alice.cells[0].cell_id[0]);
      });
    });
  });

  describe('Treasury Operations', () => {
    it('should create and query a treasury', async () => {
      await runScenario(async (scenario: Scenario) => {
        const [alice] = await scenario.addPlayersWithApps([
          { appBundleSource: { path: FINANCE_HAPP_PATH } },
        ]);

        const aliceDid = `did:mycelix:${alice.agentPubKey}`;

        // Create a treasury
        const treasury = await callZome(alice, 'treasury', 'create_treasury', {
          name: 'Community Infrastructure Fund',
          description: 'Shared pool for infrastructure improvements',
          currency: 'SAP',
          reserve_ratio: 0.25,
          managers: [aliceDid],
        });
        expect(treasury).toBeDefined();
      });
    });
  });

  describe('Staking Flow', () => {
    it('should create a staking position', async () => {
      await runScenario(async (scenario: Scenario) => {
        const [alice] = await scenario.addPlayersWithApps([
          { appBundleSource: { path: FINANCE_HAPP_PATH } },
        ]);

        const aliceDid = `did:mycelix:${alice.agentPubKey}`;

        // Create collateral stake with SAP amount
        const stake = await callZome(alice, 'staking', 'create_stake', {
          staker_did: aliceDid,
          sap_amount: 500,
        });
        expect(stake).toBeDefined();
      });
    });
  });

  describe('Recognition Flow', () => {
    it('should initialize a member MYCEL state', async () => {
      await runScenario(async (scenario: Scenario) => {
        const [alice] = await scenario.addPlayersWithApps([
          { appBundleSource: { path: FINANCE_HAPP_PATH } },
        ]);

        const aliceDid = `did:mycelix:${alice.agentPubKey}`;

        // Initialize Alice as a full member (not apprentice)
        const record = await callZome(alice, 'recognition', 'initialize_member', {
          member_did: aliceDid,
          is_apprentice: false,
          mentor_did: null,
        });
        expect(record).toBeDefined();

        // Query MYCEL score
        const mycelState = await callZome(alice, 'recognition', 'get_mycel_score', aliceDid);
        expect(mycelState).toBeDefined();
      });
    });
  });

  describe('Finance Bridge Health', () => {
    it('should return bridge health status', async () => {
      await runScenario(async (scenario: Scenario) => {
        const [alice] = await scenario.addPlayersWithApps([
          { appBundleSource: { path: FINANCE_HAPP_PATH } },
        ]);

        const health = await callZome(alice, 'finance_bridge', 'health_check', null);
        expect(health).toBeDefined();
        expect(health.healthy).toBe(true);
        expect(health.zomes).toContain('payments');
      });
    });
  });
});
