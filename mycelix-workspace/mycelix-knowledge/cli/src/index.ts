#!/usr/bin/env node

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

/**
 * Mycelix Knowledge CLI
 *
 * Command-line interface for the Knowledge hApp.
 *
 * Usage:
 *   mk claim create "Statement here" --domain climate --e 0.8
 *   mk claim get <id>
 *   mk search "query" --min-e 0.5
 *   mk factcheck "Is climate changing?"
 *   mk author <did> --reputation
 *   mk graph visualize <claim-id>
 */

import { Command } from 'commander';
import chalk from 'chalk';
import ora from 'ora';
import Table from 'cli-table3';
import boxen from 'boxen';
import inquirer from 'inquirer';
import { AppWebsocket } from '@holochain/client';

// Types (inline to avoid import issues)
interface KnowledgeClient {
  claims: any;
  query: any;
  factcheck: any;
  inference: any;
  graph: any;
  marketsIntegration: any;
}

// ============================================================================
// Configuration
// ============================================================================

interface Config {
  wsUrl: string;
  roleName: string;
}

const DEFAULT_CONFIG: Config = {
  wsUrl: process.env.KNOWLEDGE_WS_URL || 'ws://localhost:8888',
  roleName: process.env.KNOWLEDGE_ROLE_NAME || 'knowledge',
};

let client: KnowledgeClient | null = null;

// ============================================================================
// Connection
// ============================================================================

async function connect(config: Config = DEFAULT_CONFIG): Promise<KnowledgeClient> {
  if (client) return client;

  const spinner = ora('Connecting to Knowledge hApp...').start();

  try {
    const appClient = await AppWebsocket.connect(config.wsUrl);
    // Note: In real implementation, would import KnowledgeClient from SDK
    // For CLI, we create a lightweight wrapper
    client = createLightweightClient(appClient, config.roleName);
    spinner.succeed('Connected to Knowledge hApp');
    return client;
  } catch (error) {
    spinner.fail('Failed to connect');
    throw error;
  }
}

function createLightweightClient(appClient: any, roleName: string): KnowledgeClient {
  const callZome = async (zomeName: string, fnName: string, payload: any) => {
    return appClient.callZome({
      role_name: roleName,
      zome_name: zomeName,
      fn_name: fnName,
      payload,
    });
  };

  return {
    claims: {
      createClaim: (input: any) => callZome('claims', 'create_claim', input),
      getClaim: (hash: string) => callZome('claims', 'get_claim', hash),
      updateClaim: (input: any) => callZome('claims', 'update_claim', input),
      deleteClaim: (hash: string) => callZome('claims', 'delete_claim', hash),
      listClaimsByDomain: (domain: string) => callZome('claims', 'list_claims_by_domain', domain),
      listClaimsByAuthor: (author: string) => callZome('claims', 'list_claims_by_author', author),
    },
    query: {
      search: (query: string, options?: any) => callZome('query', 'search', { query, options }),
      queryByEpistemic: (minE: number, maxE: number, limit: number) =>
        callZome('query', 'query_by_epistemic', { min_e: minE, max_e: maxE, limit }),
      findContradictions: (claimId: string) => callZome('query', 'find_contradictions', claimId),
    },
    factcheck: {
      factCheck: (input: any) => callZome('factcheck', 'fact_check', input),
      batchFactCheck: (inputs: any[]) => callZome('factcheck', 'batch_fact_check', inputs),
    },
    inference: {
      calculateEnhancedCredibility: (entityId: string, entityType: string) =>
        callZome('inference', 'calculate_enhanced_credibility', { entity_id: entityId, entity_type: entityType }),
      getAuthorReputation: (did: string) => callZome('inference', 'get_author_reputation', did),
      batchCredibilityAssessment: (ids: string[]) =>
        callZome('inference', 'batch_credibility_assessment', ids),
    },
    graph: {
      getRelationships: (claimId: string) => callZome('graph', 'get_relationships', claimId),
      createRelationship: (input: any) => callZome('graph', 'create_relationship', input),
      propagateBelief: (claimId: string) => callZome('graph', 'propagate_belief', claimId),
      getDependencyTree: (claimId: string, maxDepth: number) =>
        callZome('graph', 'get_dependency_tree', { claim_id: claimId, max_depth: maxDepth }),
      rankByInformationValue: (limit: number) =>
        callZome('graph', 'rank_by_information_value', limit),
    },
    marketsIntegration: {
      getClaimMarkets: (claimId: string) =>
        callZome('markets_integration', 'get_claim_markets', claimId),
      calculateMarketValue: (claimId: string) =>
        callZome('markets_integration', 'calculate_market_value', claimId),
    },
  };
}

// ============================================================================
// Formatters
// ============================================================================

function formatEpistemic(e: number, n: number, m: number): string {
  const eBar = '█'.repeat(Math.round(e * 10)).padEnd(10, '░');
  const nBar = '█'.repeat(Math.round(n * 10)).padEnd(10, '░');
  const mBar = '█'.repeat(Math.round(m * 10)).padEnd(10, '░');

  return [
    `${chalk.cyan('E')} ${eBar} ${(e * 100).toFixed(0)}%`,
    `${chalk.yellow('N')} ${nBar} ${(n * 100).toFixed(0)}%`,
    `${chalk.magenta('M')} ${mBar} ${(m * 100).toFixed(0)}%`,
  ].join('\n');
}

function formatVerdict(verdict: string): string {
  const colors: Record<string, (s: string) => string> = {
    True: chalk.green,
    MostlyTrue: chalk.greenBright,
    Mixed: chalk.yellow,
    MostlyFalse: chalk.redBright,
    False: chalk.red,
    Unverifiable: chalk.gray,
    InsufficientEvidence: chalk.gray,
  };
  return (colors[verdict] || chalk.white)(verdict);
}

function formatCredibility(score: number): string {
  if (score >= 0.8) return chalk.green(`${(score * 100).toFixed(0)}%`);
  if (score >= 0.6) return chalk.greenBright(`${(score * 100).toFixed(0)}%`);
  if (score >= 0.4) return chalk.yellow(`${(score * 100).toFixed(0)}%`);
  if (score >= 0.2) return chalk.redBright(`${(score * 100).toFixed(0)}%`);
  return chalk.red(`${(score * 100).toFixed(0)}%`);
}

// ============================================================================
// Commands
// ============================================================================

const program = new Command();

program
  .name('mycelix-knowledge')
  .alias('mk')
  .description('CLI for Mycelix Knowledge - Decentralized Knowledge Graph')
  .version('0.1.0');

// ============================================================================
// Claim Commands
// ============================================================================

const claimCmd = program.command('claim').description('Manage claims');

claimCmd
  .command('create <content>')
  .description('Create a new claim')
  .option('-d, --domain <domain>', 'Knowledge domain', 'general')
  .option('-e, --empirical <number>', 'Empirical level (0-1)', '0.5')
  .option('-n, --normative <number>', 'Normative level (0-1)', '0.2')
  .option('-m, --mythic <number>', 'Mythic level (0-1)', '0.3')
  .option('-t, --topics <topics>', 'Comma-separated topics', '')
  .option('-i, --interactive', 'Interactive mode')
  .action(async (content, options) => {
    try {
      const knowledgeClient = await connect();

      let input: any = {
        content,
        classification: {
          empirical: parseFloat(options.empirical),
          normative: parseFloat(options.normative),
          mythic: parseFloat(options.mythic),
        },
        domain: options.domain,
        topics: options.topics ? options.topics.split(',').map((t: string) => t.trim()) : [],
        evidence: [],
      };

      if (options.interactive) {
        const answers = await inquirer.prompt([
          {
            type: 'input',
            name: 'domain',
            message: 'Domain:',
            default: input.domain,
          },
          {
            type: 'number',
            name: 'empirical',
            message: 'Empirical level (0-1):',
            default: input.classification.empirical,
          },
          {
            type: 'number',
            name: 'normative',
            message: 'Normative level (0-1):',
            default: input.classification.normative,
          },
          {
            type: 'number',
            name: 'mythic',
            message: 'Mythic level (0-1):',
            default: input.classification.mythic,
          },
          {
            type: 'input',
            name: 'topics',
            message: 'Topics (comma-separated):',
            default: input.topics.join(', '),
          },
        ]);

        input = {
          ...input,
          domain: answers.domain,
          classification: {
            empirical: answers.empirical,
            normative: answers.normative,
            mythic: answers.mythic,
          },
          topics: answers.topics ? answers.topics.split(',').map((t: string) => t.trim()) : [],
        };
      }

      const spinner = ora('Creating claim...').start();
      const hash = await knowledgeClient.claims.createClaim(input);
      spinner.succeed('Claim created!');

      console.log(boxen(
        [
          chalk.bold('Claim Created'),
          '',
          chalk.gray('Hash: ') + chalk.cyan(hash),
          '',
          chalk.gray('Content: ') + content.substring(0, 60) + (content.length > 60 ? '...' : ''),
          '',
          formatEpistemic(
            input.classification.empirical,
            input.classification.normative,
            input.classification.mythic
          ),
        ].join('\n'),
        { padding: 1, borderColor: 'green' }
      ));
    } catch (error) {
      console.error(chalk.red('Error:'), error);
      process.exit(1);
    }
  });

claimCmd
  .command('get <id>')
  .description('Get a claim by ID')
  .action(async (id) => {
    try {
      const knowledgeClient = await connect();
      const spinner = ora('Fetching claim...').start();

      const record = await knowledgeClient.claims.getClaim(id);
      spinner.stop();

      if (!record) {
        console.log(chalk.yellow('Claim not found'));
        return;
      }

      const claim = record.entry;

      console.log(boxen(
        [
          chalk.bold(claim.content),
          '',
          chalk.gray('ID: ') + chalk.cyan(id),
          chalk.gray('Domain: ') + claim.domain,
          chalk.gray('Status: ') + claim.status,
          chalk.gray('Topics: ') + (claim.topics?.join(', ') || 'none'),
          '',
          formatEpistemic(
            claim.classification.empirical,
            claim.classification.normative,
            claim.classification.mythic
          ),
        ].join('\n'),
        { padding: 1, borderColor: 'cyan' }
      ));
    } catch (error) {
      console.error(chalk.red('Error:'), error);
      process.exit(1);
    }
  });

claimCmd
  .command('list')
  .description('List claims')
  .option('-d, --domain <domain>', 'Filter by domain')
  .option('-l, --limit <number>', 'Limit results', '20')
  .action(async (options) => {
    try {
      const knowledgeClient = await connect();
      const spinner = ora('Fetching claims...').start();

      let claims;
      if (options.domain) {
        claims = await knowledgeClient.claims.listClaimsByDomain(options.domain);
      } else {
        claims = await knowledgeClient.query.search('', { limit: parseInt(options.limit) });
      }
      spinner.stop();

      const table = new Table({
        head: [
          chalk.cyan('ID'),
          chalk.cyan('Content'),
          chalk.cyan('E'),
          chalk.cyan('Domain'),
          chalk.cyan('Status'),
        ],
        colWidths: [15, 50, 8, 15, 12],
      });

      for (const claim of claims.slice(0, parseInt(options.limit))) {
        table.push([
          claim.id?.substring(0, 12) + '...' || '-',
          claim.content?.substring(0, 45) + (claim.content?.length > 45 ? '...' : '') || '-',
          (claim.classification?.empirical * 100).toFixed(0) + '%' || '-',
          claim.domain || '-',
          claim.status || '-',
        ]);
      }

      console.log(table.toString());
      console.log(chalk.gray(`\nShowing ${Math.min(claims.length, parseInt(options.limit))} of ${claims.length} claims`));
    } catch (error) {
      console.error(chalk.red('Error:'), error);
      process.exit(1);
    }
  });

// ============================================================================
// Search Command
// ============================================================================

program
  .command('search <query>')
  .description('Search for claims')
  .option('--min-e <number>', 'Minimum empirical level', '0')
  .option('--max-e <number>', 'Maximum empirical level', '1')
  .option('-d, --domain <domain>', 'Filter by domain')
  .option('-l, --limit <number>', 'Limit results', '20')
  .action(async (query, options) => {
    try {
      const knowledgeClient = await connect();
      const spinner = ora(`Searching for "${query}"...`).start();

      const results = await knowledgeClient.query.search(query, {
        min_e: parseFloat(options.minE),
        max_e: parseFloat(options.maxE),
        domain: options.domain,
        limit: parseInt(options.limit),
      });
      spinner.stop();

      if (results.length === 0) {
        console.log(chalk.yellow('No results found'));
        return;
      }

      const table = new Table({
        head: [
          chalk.cyan('ID'),
          chalk.cyan('Content'),
          chalk.cyan('E'),
          chalk.cyan('Domain'),
        ],
        colWidths: [15, 55, 8, 15],
      });

      for (const result of results) {
        table.push([
          result.id?.substring(0, 12) + '...' || '-',
          result.content?.substring(0, 50) + (result.content?.length > 50 ? '...' : '') || '-',
          (result.classification?.empirical * 100).toFixed(0) + '%' || '-',
          result.domain || '-',
        ]);
      }

      console.log(table.toString());
      console.log(chalk.gray(`\nFound ${results.length} results`));
    } catch (error) {
      console.error(chalk.red('Error:'), error);
      process.exit(1);
    }
  });

// ============================================================================
// Marketplace/Media Helpers
// ============================================================================

program
  .command('marketplace-claims')
  .description('Inspect claims linked to a marketplace listing')
  .argument('<listing-id>', 'Listing hash or local ID')
  .option('-l, --limit <number>', 'Limit results', '5')
  .action(async (listingId, options) => {
    try {
      const knowledgeClient = await connect();
      const uri = `happ://marketplace/listing/${listingId}`;
      const spinner = ora(`Searching claims for ${uri}...`).start();

      const claims = await knowledgeClient.query.search(uri, {
        limit: parseInt(options.limit),
      });
      spinner.stop();

      if (!claims.length) {
        console.log(chalk.yellow('No claims found for this listing.'));
        return;
      }

      const table = new Table({
        head: [
          chalk.cyan('ID'),
          chalk.cyan('Content'),
          chalk.cyan('E'),
          chalk.cyan('Domain'),
          chalk.cyan('Status'),
        ],
        colWidths: [15, 40, 8, 15, 12],
      });

      for (const claim of claims) {
        table.push([
          claim.id?.substring(0, 12) + '...' || '-',
          claim.content?.substring(0, 35) + (claim.content?.length > 35 ? '...' : '') || '-',
          (claim.classification?.empirical * 100).toFixed(0) + '%' || '-',
          claim.domain || '-',
          claim.status || '-',
        ]);
      }

      console.log(table.toString());
    } catch (error) {
      console.error(chalk.red('Error:'), error);
      process.exit(1);
    }
  });

program
  .command('media-claims')
  .description('Inspect claims linked to a media publication')
  .argument('<publication-id>', 'Publication identifier used by Mycelix Media')
  .option('-l, --limit <number>', 'Limit results', '5')
  .action(async (publicationId, options) => {
    try {
      const knowledgeClient = await connect();
      const uri = `happ://media/publication/${publicationId}`;
      const spinner = ora(`Searching claims for ${uri}...`).start();

      const claims = await knowledgeClient.query.search(uri, {
        limit: parseInt(options.limit),
      });
      spinner.stop();

      if (!claims.length) {
        console.log(chalk.yellow('No claims found for this publication.'));
        return;
      }

      const table = new Table({
        head: [
          chalk.cyan('ID'),
          chalk.cyan('Content'),
          chalk.cyan('E'),
          chalk.cyan('Domain'),
          chalk.cyan('Status'),
        ],
        colWidths: [15, 40, 8, 15, 12],
      });

      for (const claim of claims) {
        table.push([
          claim.id?.substring(0, 12) + '...' || '-',
          claim.content?.substring(0, 35) + (claim.content?.length > 35 ? '...' : '') || '-',
          (claim.classification?.empirical * 100).toFixed(0) + '%' || '-',
          claim.domain || '-',
          claim.status || '-',
        ]);
      }

      console.log(table.toString());
    } catch (error) {
      console.error(chalk.red('Error:'), error);
      process.exit(1);
    }
  });

// ============================================================================
// Fact-Check Command
// ============================================================================

program
  .command('factcheck <statement>')
  .description('Fact-check a statement')
  .option('--min-e <number>', 'Minimum empirical level for evidence', '0.6')
  .option('-c, --context <text>', 'Additional context')
  .action(async (statement, options) => {
    try {
      const knowledgeClient = await connect();
      const spinner = ora('Fact-checking...').start();

      const result = await knowledgeClient.factcheck.factCheck({
        statement,
        context: options.context,
        min_e: parseFloat(options.minE),
      });
      spinner.stop();

      console.log(boxen(
        [
          chalk.bold('Fact-Check Result'),
          '',
          chalk.gray('Statement: ') + statement,
          '',
          chalk.bold('Verdict: ') + formatVerdict(result.verdict),
          chalk.bold('Confidence: ') + formatCredibility(result.confidence),
          '',
          chalk.gray('Supporting claims: ') + result.supporting_claims?.length || 0,
          chalk.gray('Contradicting claims: ') + result.contradicting_claims?.length || 0,
          '',
          chalk.italic(result.explanation?.substring(0, 100) || ''),
        ].join('\n'),
        { padding: 1, borderColor: result.verdict === 'True' ? 'green' : result.verdict === 'False' ? 'red' : 'yellow' }
      ));

      if (result.supporting_claims?.length > 0) {
        console.log(chalk.green('\nSupporting Evidence:'));
        for (const claim of result.supporting_claims.slice(0, 3)) {
          console.log(`  • ${claim.excerpt || claim.claim_id}`);
        }
      }

      if (result.contradicting_claims?.length > 0) {
        console.log(chalk.red('\nContradicting Evidence:'));
        for (const claim of result.contradicting_claims.slice(0, 3)) {
          console.log(`  • ${claim.excerpt || claim.claim_id}`);
        }
      }
    } catch (error) {
      console.error(chalk.red('Error:'), error);
      process.exit(1);
    }
  });

// ============================================================================
// Author Command
// ============================================================================

program
  .command('author <did>')
  .description('Get author information')
  .option('-r, --reputation', 'Show detailed reputation')
  .action(async (did, options) => {
    try {
      const knowledgeClient = await connect();
      const spinner = ora('Fetching author info...').start();

      const reputation = await knowledgeClient.inference.getAuthorReputation(did);
      spinner.stop();

      console.log(boxen(
        [
          chalk.bold('Author Profile'),
          '',
          chalk.gray('DID: ') + chalk.cyan(did),
          '',
          chalk.bold('Overall Score: ') + formatCredibility(reputation.overall_score),
          chalk.bold('MATL Trust: ') + formatCredibility(reputation.matl_trust),
          '',
          chalk.gray('Claims authored: ') + reputation.claims_authored,
          chalk.gray('Verified true: ') + chalk.green(reputation.claims_verified_true),
          chalk.gray('Verified false: ') + chalk.red(reputation.claims_verified_false),
          chalk.gray('Historical accuracy: ') + formatCredibility(reputation.historical_accuracy),
        ].join('\n'),
        { padding: 1, borderColor: 'blue' }
      ));

      if (options.reputation && reputation.domain_scores?.length > 0) {
        console.log(chalk.bold('\nDomain Expertise:'));
        const table = new Table({
          head: [chalk.cyan('Domain'), chalk.cyan('Score'), chalk.cyan('Claims')],
        });

        for (const ds of reputation.domain_scores) {
          table.push([ds.domain, formatCredibility(ds.score), ds.claims_in_domain]);
        }

        console.log(table.toString());
      }
    } catch (error) {
      console.error(chalk.red('Error:'), error);
      process.exit(1);
    }
  });

// ============================================================================
// Graph Command
// ============================================================================

const graphCmd = program.command('graph').description('Knowledge graph operations');

graphCmd
  .command('relationships <claimId>')
  .description('Show relationships for a claim')
  .action(async (claimId) => {
    try {
      const knowledgeClient = await connect();
      const spinner = ora('Fetching relationships...').start();

      const relationships = await knowledgeClient.graph.getRelationships(claimId);
      spinner.stop();

      if (relationships.length === 0) {
        console.log(chalk.yellow('No relationships found'));
        return;
      }

      const table = new Table({
        head: [chalk.cyan('Type'), chalk.cyan('Target'), chalk.cyan('Weight')],
      });

      for (const rel of relationships) {
        table.push([
          rel.relationship_type,
          rel.target_id?.substring(0, 20) + '...' || '-',
          (rel.weight * 100).toFixed(0) + '%',
        ]);
      }

      console.log(table.toString());
    } catch (error) {
      console.error(chalk.red('Error:'), error);
      process.exit(1);
    }
  });

graphCmd
  .command('propagate <claimId>')
  .description('Propagate belief updates from a claim')
  .action(async (claimId) => {
    try {
      const knowledgeClient = await connect();
      const spinner = ora('Propagating beliefs...').start();

      const result = await knowledgeClient.graph.propagateBelief(claimId);
      spinner.succeed('Belief propagation complete');

      console.log(boxen(
        [
          chalk.bold('Propagation Result'),
          '',
          chalk.gray('Nodes updated: ') + result.nodes_updated,
          chalk.gray('Iterations: ') + result.iterations,
          chalk.gray('Converged: ') + (result.converged ? chalk.green('Yes') : chalk.yellow('No')),
          chalk.gray('Max change: ') + (result.max_change * 100).toFixed(2) + '%',
        ].join('\n'),
        { padding: 1, borderColor: 'magenta' }
      ));
    } catch (error) {
      console.error(chalk.red('Error:'), error);
      process.exit(1);
    }
  });

graphCmd
  .command('iv')
  .description('Show claims ranked by information value')
  .option('-l, --limit <number>', 'Limit results', '10')
  .action(async (options) => {
    try {
      const knowledgeClient = await connect();
      const spinner = ora('Calculating information values...').start();

      const rankings = await knowledgeClient.graph.rankByInformationValue(parseInt(options.limit));
      spinner.stop();

      console.log(chalk.bold('\nClaims by Information Value (verification priority):\n'));

      const table = new Table({
        head: [
          chalk.cyan('#'),
          chalk.cyan('Claim ID'),
          chalk.cyan('IV'),
          chalk.cyan('Uncertainty'),
          chalk.cyan('Connectivity'),
        ],
      });

      rankings.forEach((item: any, index: number) => {
        table.push([
          index + 1,
          item.claim_id?.substring(0, 15) + '...' || '-',
          item.value?.toFixed(3) || '-',
          item.components?.uncertainty?.toFixed(2) || '-',
          item.components?.connectivity?.toFixed(2) || '-',
        ]);
      });

      console.log(table.toString());
    } catch (error) {
      console.error(chalk.red('Error:'), error);
      process.exit(1);
    }
  });

// ============================================================================
// Market Command
// ============================================================================

program
  .command('market <claimId>')
  .description('Show verification markets for a claim')
  .action(async (claimId) => {
    try {
      const knowledgeClient = await connect();
      const spinner = ora('Fetching markets...').start();

      const markets = await knowledgeClient.marketsIntegration.getClaimMarkets(claimId);
      spinner.stop();

      if (markets.length === 0) {
        console.log(chalk.yellow('No verification markets found for this claim'));

        // Suggest creating one
        const assessment = await knowledgeClient.marketsIntegration.calculateMarketValue(claimId);
        if (assessment.expected_value > 0.3) {
          console.log(chalk.green('\n✓ This claim is a good candidate for verification!'));
          console.log(chalk.gray(`  Expected value: ${(assessment.expected_value * 100).toFixed(0)}%`));
          console.log(chalk.gray(`  Recommended subsidy: $${assessment.recommended_subsidy}`));
        }
        return;
      }

      const table = new Table({
        head: [
          chalk.cyan('Market ID'),
          chalk.cyan('Status'),
          chalk.cyan('Target E'),
          chalk.cyan('Price'),
          chalk.cyan('Participants'),
        ],
      });

      for (const market of markets) {
        table.push([
          market.market_id?.substring(0, 15) + '...' || '-',
          market.status,
          (market.target_e * 100).toFixed(0) + '%',
          (market.current_price * 100).toFixed(0) + '%',
          market.participant_count || 0,
        ]);
      }

      console.log(table.toString());
    } catch (error) {
      console.error(chalk.red('Error:'), error);
      process.exit(1);
    }
  });

// ============================================================================
// Config Command
// ============================================================================

program
  .command('config')
  .description('Show/set configuration')
  .option('--ws-url <url>', 'WebSocket URL')
  .option('--role <name>', 'Role name')
  .action((options) => {
    if (options.wsUrl) {
      DEFAULT_CONFIG.wsUrl = options.wsUrl;
      console.log(chalk.green(`WebSocket URL set to: ${options.wsUrl}`));
    }
    if (options.role) {
      DEFAULT_CONFIG.roleName = options.role;
      console.log(chalk.green(`Role name set to: ${options.role}`));
    }

    if (!options.wsUrl && !options.role) {
      console.log(boxen(
        [
          chalk.bold('Current Configuration'),
          '',
          chalk.gray('WebSocket URL: ') + DEFAULT_CONFIG.wsUrl,
          chalk.gray('Role name: ') + DEFAULT_CONFIG.roleName,
          '',
          chalk.italic('Set via environment variables:'),
          chalk.gray('  KNOWLEDGE_WS_URL'),
          chalk.gray('  KNOWLEDGE_ROLE_NAME'),
        ].join('\n'),
        { padding: 1, borderColor: 'gray' }
      ));
    }
  });

// ============================================================================
// Run
// ============================================================================

program.parse();
