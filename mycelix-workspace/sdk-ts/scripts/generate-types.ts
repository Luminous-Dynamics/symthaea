#!/usr/bin/env node
/**
 * TypeScript Type Generation Pipeline
 *
 * This script generates TypeScript types from Rust structs for all Mycelix hApps.
 * It orchestrates the ts-rs binding generation from Rust and organizes the output
 * into domain-specific modules.
 *
 * Usage:
 *   npx ts-node scripts/generate-types.ts
 *   pnpm generate:types
 *
 * @module @mycelix/sdk/scripts/generate-types
 */

import * as fs from 'fs';
import * as path from 'path';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';

// ES Module compatibility
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// =============================================================================
// Configuration
// =============================================================================

interface TypeGenerationConfig {
  /** Root of the mycelix workspace */
  workspaceRoot: string;
  /** Path to Rust SDK */
  rustSdkPath: string;
  /** Path to TypeScript SDK */
  tsSdkPath: string;
  /** Output directory for generated types */
  outputDir: string;
  /** Rust bindings source directories */
  bindingsSourceDirs: string[];
}

const CONFIG: TypeGenerationConfig = {
  workspaceRoot: path.resolve(__dirname, '../..'),
  rustSdkPath: path.resolve(__dirname, '../../sdk'),
  tsSdkPath: path.resolve(__dirname, '..'),
  outputDir: path.resolve(__dirname, '../src/generated'),
  bindingsSourceDirs: [
    path.resolve(__dirname, '../../sdk/bindings/bindings'),
    path.resolve(__dirname, '../../sdk/bindings/matl'),
    path.resolve(__dirname, '../../sdk/bindings/epistemic'),
    path.resolve(__dirname, '../../sdk/bindings/dkg'),
  ],
};

// =============================================================================
// Type Definitions for Generation
// =============================================================================

interface TypeModule {
  name: string;
  description: string;
  types: TypeDefinition[];
  sourceDir?: string;
}

interface TypeDefinition {
  name: string;
  description?: string;
  isEnum?: boolean;
  fields?: TypeField[];
  variants?: string[];
}

interface TypeField {
  name: string;
  type: string;
  optional?: boolean;
  description?: string;
}

// =============================================================================
// Type Modules Configuration
// =============================================================================

const TYPE_MODULES: TypeModule[] = [
  {
    name: 'common',
    description: 'Common Holochain types and utilities',
    types: [
      {
        name: 'AgentPubKey',
        description: 'Holochain agent public key (base64 encoded)',
      },
      {
        name: 'ActionHash',
        description: 'Hash of a Holochain action',
      },
      {
        name: 'EntryHash',
        description: 'Hash of a Holochain entry',
      },
      {
        name: 'DnaHash',
        description: 'Hash of a Holochain DNA',
      },
      {
        name: 'Timestamp',
        description: 'Unix timestamp in microseconds',
      },
    ],
  },
  {
    name: 'identity',
    description: 'Identity types (DID, VerifiableCredential, etc.)',
    types: [
      {
        name: 'DID',
        description: 'Decentralized Identifier',
        fields: [
          { name: 'id', type: 'string', description: 'The DID string (did:mycelix:...)' },
          { name: 'controller', type: 'string', description: 'Controller agent key' },
          { name: 'created', type: 'number', description: 'Creation timestamp' },
          { name: 'updated', type: 'number', description: 'Last update timestamp' },
        ],
      },
      {
        name: 'VerifiableCredential',
        description: 'W3C Verifiable Credential',
        fields: [
          { name: 'context', type: 'string[]', description: 'JSON-LD context' },
          { name: 'id', type: 'string', description: 'Credential ID' },
          { name: 'type', type: 'string[]', description: 'Credential types' },
          { name: 'issuer', type: 'string', description: 'Issuer DID' },
          { name: 'issuanceDate', type: 'string', description: 'ISO 8601 issuance date' },
          { name: 'expirationDate', type: 'string', optional: true, description: 'ISO 8601 expiration date' },
          { name: 'credentialSubject', type: 'CredentialSubject', description: 'Credential claims' },
          { name: 'proof', type: 'CredentialProof', optional: true, description: 'Cryptographic proof' },
        ],
      },
      {
        name: 'CredentialSubject',
        description: 'Subject of a verifiable credential',
        fields: [
          { name: 'id', type: 'string', description: 'Subject DID' },
        ],
      },
      {
        name: 'CredentialProof',
        description: 'Cryptographic proof for a credential',
        fields: [
          { name: 'type', type: 'string', description: 'Proof type' },
          { name: 'created', type: 'string', description: 'Creation timestamp' },
          { name: 'verificationMethod', type: 'string', description: 'Verification method ID' },
          { name: 'proofPurpose', type: 'string', description: 'Purpose of proof' },
          { name: 'proofValue', type: 'string', description: 'Proof signature value' },
        ],
      },
      {
        name: 'VerificationMethod',
        description: 'Public key verification method',
        fields: [
          { name: 'id', type: 'string', description: 'Method ID' },
          { name: 'type', type: 'string', description: 'Key type' },
          { name: 'controller', type: 'string', description: 'Controller DID' },
          { name: 'publicKeyMultibase', type: 'string', description: 'Multibase-encoded public key' },
        ],
      },
      {
        name: 'ServiceEndpoint',
        description: 'DID service endpoint',
        fields: [
          { name: 'id', type: 'string', description: 'Service ID' },
          { name: 'type', type: 'string', description: 'Service type' },
          { name: 'serviceEndpoint', type: 'string', description: 'Endpoint URL' },
        ],
      },
    ],
  },
  {
    name: 'governance',
    description: 'Governance types (Proposal, Vote, DAO, etc.)',
    types: [
      {
        name: 'Proposal',
        description: 'A DAO governance proposal',
        fields: [
          { name: 'id', type: 'string', description: 'Proposal ID' },
          { name: 'daoId', type: 'string', description: 'DAO this proposal belongs to' },
          { name: 'title', type: 'string', description: 'Proposal title' },
          { name: 'description', type: 'string', description: 'Detailed description' },
          { name: 'proposer', type: 'string', description: 'Proposer DID' },
          { name: 'proposalType', type: 'ProposalType', description: 'Type of proposal' },
          { name: 'status', type: 'ProposalStatus', description: 'Current status' },
          { name: 'votingPeriodHours', type: 'number', description: 'Voting period in hours' },
          { name: 'quorum', type: 'number', description: 'Required quorum (0.0-1.0)' },
          { name: 'threshold', type: 'number', description: 'Approval threshold (0.0-1.0)' },
          { name: 'approveWeight', type: 'number', description: 'Total approve vote weight' },
          { name: 'rejectWeight', type: 'number', description: 'Total reject vote weight' },
          { name: 'abstainWeight', type: 'number', description: 'Total abstain vote weight' },
          { name: 'votingEnds', type: 'number', description: 'Voting end timestamp' },
          { name: 'created', type: 'number', description: 'Creation timestamp' },
        ],
      },
      {
        name: 'ProposalType',
        description: 'Type of governance proposal',
        isEnum: true,
        variants: ['Standard', 'Emergency', 'Constitutional'],
      },
      {
        name: 'ProposalStatus',
        description: 'Status of a governance proposal',
        isEnum: true,
        variants: ['Draft', 'Active', 'Passed', 'Rejected', 'Executed', 'Expired', 'Cancelled'],
      },
      {
        name: 'GovernanceVote',
        description: 'A vote on a governance proposal',
        fields: [
          { name: 'id', type: 'string', description: 'Vote ID' },
          { name: 'proposalId', type: 'string', description: 'Proposal being voted on' },
          { name: 'voter', type: 'string', description: 'Voter DID' },
          { name: 'choice', type: 'VoteChoice', description: 'Vote choice' },
          { name: 'weight', type: 'number', description: 'Vote weight' },
          { name: 'reason', type: 'string', optional: true, description: 'Reason for vote' },
          { name: 'votedAt', type: 'number', description: 'Vote timestamp' },
        ],
      },
      {
        name: 'VoteChoice',
        description: 'Vote choice options',
        isEnum: true,
        variants: ['Approve', 'Reject', 'Abstain'],
      },
      {
        name: 'DAO',
        description: 'Decentralized Autonomous Organization',
        fields: [
          { name: 'id', type: 'string', description: 'DAO ID' },
          { name: 'name', type: 'string', description: 'DAO name' },
          { name: 'description', type: 'string', description: 'DAO description' },
          { name: 'creator', type: 'string', description: 'Creator DID' },
          { name: 'charterHash', type: 'string', optional: true, description: 'Charter document hash' },
          { name: 'defaultVotingPeriod', type: 'number', description: 'Default voting period in hours' },
          { name: 'defaultQuorum', type: 'number', description: 'Default quorum' },
          { name: 'defaultThreshold', type: 'number', description: 'Default threshold' },
          { name: 'memberCount', type: 'number', description: 'Number of members' },
          { name: 'totalVotingPower', type: 'number', description: 'Total voting power' },
          { name: 'created', type: 'number', description: 'Creation timestamp' },
        ],
      },
      {
        name: 'Delegation',
        description: 'Voting power delegation',
        fields: [
          { name: 'id', type: 'string', description: 'Delegation ID' },
          { name: 'delegator', type: 'string', description: 'Delegator DID' },
          { name: 'delegate', type: 'string', description: 'Delegate DID' },
          { name: 'daoId', type: 'string', description: 'DAO ID' },
          { name: 'power', type: 'number', description: 'Delegated voting power' },
          { name: 'expiresAt', type: 'number', optional: true, description: 'Expiration timestamp' },
          { name: 'created', type: 'number', description: 'Creation timestamp' },
        ],
      },
    ],
  },
  {
    name: 'finance',
    description: 'Finance types (Loan, Payment, Wallet, etc.)',
    types: [
      {
        name: 'Wallet',
        description: 'A financial wallet',
        fields: [
          { name: 'id', type: 'string', description: 'Wallet ID' },
          { name: 'owner', type: 'string', description: 'Owner DID' },
          { name: 'walletType', type: 'WalletType', description: 'Type of wallet' },
          { name: 'name', type: 'string', description: 'Wallet name' },
          { name: 'balances', type: 'Record<string, number>', description: 'Balances by currency' },
          { name: 'createdAt', type: 'number', description: 'Creation timestamp' },
          { name: 'updatedAt', type: 'number', description: 'Last update timestamp' },
        ],
      },
      {
        name: 'WalletType',
        description: 'Type of wallet',
        isEnum: true,
        variants: ['Personal', 'Business', 'DAO', 'Escrow'],
      },
      {
        name: 'Transaction',
        description: 'A financial transaction',
        fields: [
          { name: 'id', type: 'string', description: 'Transaction ID' },
          { name: 'fromWallet', type: 'string', optional: true, description: 'Source wallet' },
          { name: 'toWallet', type: 'string', optional: true, description: 'Destination wallet' },
          { name: 'amount', type: 'number', description: 'Transaction amount' },
          { name: 'currency', type: 'Currency', description: 'Currency' },
          { name: 'transactionType', type: 'TransactionType', description: 'Type of transaction' },
          { name: 'status', type: 'TransactionStatus', description: 'Transaction status' },
          { name: 'memo', type: 'string', optional: true, description: 'Transaction memo' },
          { name: 'createdAt', type: 'number', description: 'Creation timestamp' },
        ],
      },
      {
        name: 'TransactionType',
        description: 'Type of transaction',
        isEnum: true,
        variants: ['Transfer', 'Deposit', 'Withdrawal', 'LoanDisbursement', 'LoanRepayment', 'Fee'],
      },
      {
        name: 'TransactionStatus',
        description: 'Status of a transaction',
        isEnum: true,
        variants: ['Pending', 'Confirmed', 'Failed', 'Reversed'],
      },
      {
        name: 'Currency',
        description: 'Supported currencies',
        isEnum: true,
        variants: ['MYC', 'USD', 'EUR', 'BTC', 'ETH', 'ENERGY'],
      },
      {
        name: 'Loan',
        description: 'A loan record',
        fields: [
          { name: 'id', type: 'string', description: 'Loan ID' },
          { name: 'borrower', type: 'string', description: 'Borrower DID' },
          { name: 'lender', type: 'string', description: 'Lender DID' },
          { name: 'principal', type: 'number', description: 'Principal amount' },
          { name: 'currency', type: 'Currency', description: 'Currency' },
          { name: 'interestRate', type: 'number', description: 'Interest rate (annual %)' },
          { name: 'termDays', type: 'number', description: 'Term in days' },
          { name: 'loanType', type: 'LoanType', description: 'Type of loan' },
          { name: 'status', type: 'LoanStatus', description: 'Loan status' },
          { name: 'amountRepaid', type: 'number', description: 'Amount repaid' },
          { name: 'createdAt', type: 'number', description: 'Creation timestamp' },
          { name: 'dueAt', type: 'number', optional: true, description: 'Due date timestamp' },
        ],
      },
      {
        name: 'LoanType',
        description: 'Type of loan',
        isEnum: true,
        variants: ['Personal', 'Business', 'Microfinance', 'Collateralized'],
      },
      {
        name: 'LoanStatus',
        description: 'Status of a loan',
        isEnum: true,
        variants: ['Pending', 'Active', 'Repaid', 'Defaulted', 'Cancelled'],
      },
      {
        name: 'CreditScore',
        description: 'MATL-based credit score',
        fields: [
          { name: 'did', type: 'string', description: 'Subject DID' },
          { name: 'score', type: 'number', description: 'Numeric score (0-1000)' },
          { name: 'tier', type: 'CreditTier', description: 'Credit tier' },
          { name: 'matlFactor', type: 'number', description: 'MATL contribution (0-1)' },
          { name: 'paymentFactor', type: 'number', description: 'Payment history factor' },
          { name: 'utilizationFactor', type: 'number', description: 'Utilization factor' },
          { name: 'historyFactor', type: 'number', description: 'History length factor' },
          { name: 'totalLimit', type: 'number', description: 'Total credit limit' },
          { name: 'currentUtilization', type: 'number', description: 'Current utilization' },
          { name: 'updatedAt', type: 'number', description: 'Last update timestamp' },
        ],
      },
      {
        name: 'CreditTier',
        description: 'Credit score tier',
        isEnum: true,
        variants: ['Unrated', 'Bronze', 'Silver', 'Gold', 'Platinum'],
      },
    ],
  },
];

// =============================================================================
// Code Generation Functions
// =============================================================================

/**
 * Generate TypeScript type definition
 */
function generateTypeDefinition(typeDef: TypeDefinition): string {
  const lines: string[] = [];

  // JSDoc comment
  if (typeDef.description) {
    lines.push(`/**`);
    lines.push(` * ${typeDef.description}`);
    lines.push(` * @generated`);
    lines.push(` */`);
  }

  if (typeDef.isEnum && typeDef.variants) {
    // Generate union type for enum
    const variants = typeDef.variants.map(v => `'${v}'`).join(' | ');
    lines.push(`export type ${typeDef.name} = ${variants};`);
  } else if (typeDef.fields) {
    // Generate interface
    lines.push(`export interface ${typeDef.name} {`);
    for (const field of typeDef.fields) {
      if (field.description) {
        lines.push(`  /** ${field.description} */`);
      }
      const optional = field.optional ? '?' : '';
      lines.push(`  ${field.name}${optional}: ${field.type};`);
    }
    lines.push(`}`);
  } else {
    // Simple type alias
    lines.push(`export type ${typeDef.name} = string;`);
  }

  return lines.join('\n');
}

/**
 * Generate module file content
 */
function generateModuleFile(module: TypeModule): string {
  const lines: string[] = [
    `/**`,
    ` * ${module.description}`,
    ` *`,
    ` * Auto-generated TypeScript types from Rust SDK.`,
    ` * DO NOT EDIT MANUALLY - regenerate with: pnpm generate:types`,
    ` *`,
    ` * @module @mycelix/sdk/generated/${module.name}`,
    ` * @generated`,
    ` */`,
    ``,
  ];

  for (const typeDef of module.types) {
    lines.push(generateTypeDefinition(typeDef));
    lines.push('');
  }

  return lines.join('\n');
}

/**
 * Copy ts-rs generated files from Rust SDK bindings
 */
function copyRustGeneratedTypes(sourceDir: string, targetDir: string): number {
  let count = 0;

  if (!fs.existsSync(sourceDir)) {
    console.log(`  Source directory not found: ${sourceDir}`);
    return 0;
  }

  const items = fs.readdirSync(sourceDir, { withFileTypes: true });

  for (const item of items) {
    const sourcePath = path.join(sourceDir, item.name);
    const targetPath = path.join(targetDir, item.name);

    if (item.isDirectory()) {
      // Copy subdirectory
      if (!fs.existsSync(targetPath)) {
        fs.mkdirSync(targetPath, { recursive: true });
      }
      count += copyRustGeneratedTypes(sourcePath, targetPath);
    } else if (item.name.endsWith('.ts')) {
      // Copy TypeScript file
      fs.copyFileSync(sourcePath, targetPath);
      count++;
    }
  }

  return count;
}

/**
 * Generate index file that re-exports all types
 * Handles duplicates by prioritizing ts-rs generated modules over bindings wrapper
 */
function generateIndexFile(tsRsModules: string[], additionalModules: string[]): string {
  const lines: string[] = [
    `/**`,
    ` * Auto-generated TypeScript types from Rust SDK`,
    ` *`,
    ` * DO NOT EDIT MANUALLY - regenerate with: pnpm generate:types`,
    ` *`,
    ` * Generated: ${new Date().toISOString()}`,
    ` *`,
    ` * @module @mycelix/sdk/generated`,
    ` * @generated`,
    ` */`,
    ``,
    `// Core generated types from ts-rs (Rust SDK)`,
    `// These are the canonical types generated directly from Rust structs`,
  ];

  // Export ts-rs generated modules (excluding bindings wrapper to avoid duplicates)
  for (const mod of tsRsModules.filter(m => m !== 'bindings')) {
    lines.push(`export * from './${mod}';`);
  }

  lines.push('');
  lines.push('// Additional TypeScript-native types for domain modules');

  // Export additional modules
  for (const mod of additionalModules) {
    lines.push(`export * from './${mod}';`);
  }

  return lines.join('\n');
}

// =============================================================================
// Main Generation Pipeline
// =============================================================================

async function main(): Promise<void> {
  console.log('=== Mycelix TypeScript Type Generation Pipeline ===\n');

  // Step 1: Create output directory
  console.log('[1/5] Creating output directory...');
  if (!fs.existsSync(CONFIG.outputDir)) {
    fs.mkdirSync(CONFIG.outputDir, { recursive: true });
  }

  // Step 2: Try to run Rust ts-rs generation
  console.log('[2/5] Running Rust type generation (ts-rs)...');
  try {
    execSync('cargo test export_bindings --features ts-export -- --nocapture 2>/dev/null', {
      cwd: CONFIG.rustSdkPath,
      stdio: 'pipe',
    });
    console.log('  Rust type generation completed');
  } catch {
    console.log('  Rust type generation skipped (may need cargo build first)');
  }

  // Step 3: Copy ts-rs generated files
  console.log('[3/5] Copying ts-rs generated types...');
  let copiedCount = 0;
  for (const sourceDir of CONFIG.bindingsSourceDirs) {
    const relativePath = path.relative(path.dirname(CONFIG.bindingsSourceDirs[0]), sourceDir);
    const targetDir = relativePath === '.'
      ? CONFIG.outputDir
      : path.join(CONFIG.outputDir, path.basename(sourceDir));

    if (!fs.existsSync(targetDir)) {
      fs.mkdirSync(targetDir, { recursive: true });
    }

    const count = copyRustGeneratedTypes(sourceDir, targetDir);
    if (count > 0) {
      console.log(`  Copied ${count} files from ${path.basename(sourceDir)}/`);
    }
    copiedCount += count;
  }

  // Step 4: Generate additional type modules
  console.log('[4/5] Generating additional type modules...');
  const generatedModules: string[] = [];
  let generatedCount = 0;

  for (const module of TYPE_MODULES) {
    const content = generateModuleFile(module);
    const filePath = path.join(CONFIG.outputDir, `${module.name}.ts`);
    fs.writeFileSync(filePath, content);
    generatedModules.push(module.name);
    generatedCount += module.types.length;
    console.log(`  Generated ${module.name}.ts (${module.types.length} types)`);
  }

  // Step 5: Generate index file
  console.log('[5/5] Generating index file...');
  const tsRsModules: string[] = [];

  // Find all existing module directories (from ts-rs)
  const items = fs.readdirSync(CONFIG.outputDir, { withFileTypes: true });
  for (const item of items) {
    if (item.isDirectory()) {
      // Check if directory has an index.ts
      const indexPath = path.join(CONFIG.outputDir, item.name, 'index.ts');
      if (fs.existsSync(indexPath)) {
        tsRsModules.push(item.name);
      }
    }
  }

  // Generate index avoiding duplicates (bindings wrapper re-exports same types)
  const indexContent = generateIndexFile(tsRsModules, generatedModules);
  fs.writeFileSync(path.join(CONFIG.outputDir, 'index.ts'), indexContent);
  const totalModules = tsRsModules.filter(m => m !== 'bindings').length + generatedModules.length;
  console.log(`  Generated index.ts with ${totalModules} module exports`);

  // Summary
  const allModules = [...tsRsModules.filter(m => m !== 'bindings'), ...generatedModules];
  console.log('\n=== Type Generation Complete ===\n');
  console.log(`Summary:`);
  console.log(`  - Copied ${copiedCount} ts-rs generated files`);
  console.log(`  - Generated ${generatedCount} additional types in ${generatedModules.length} modules`);
  console.log(`  - Total modules exported: ${totalModules}`);
  console.log(`\nOutput directory: ${CONFIG.outputDir}`);
  console.log(`\nModules:`);
  for (const mod of allModules.sort()) {
    console.log(`  - ${mod}`);
  }
  console.log(`\nNext steps:`);
  console.log(`  1. Run 'pnpm typecheck' to validate types`);
  console.log(`  2. Import types from '@mycelix/sdk/generated'`);
}

// Run the pipeline
main().catch((error) => {
  console.error('Type generation failed:', error);
  process.exit(1);
});
