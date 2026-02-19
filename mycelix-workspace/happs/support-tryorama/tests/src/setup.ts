import { Scenario, Player, dhtSync } from '@holochain/tryorama';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Path to the packed hApp bundle (built by Step 1)
export const HAPP_PATH = path.resolve(__dirname, '../../../../mycelix-commons/mycelix-commons.happ');

// Mirror types matching support zome entry definitions
export interface SupportKnowledgeArticle {
  title: string;
  content: string;
  category: string;
  tags: string[];
  source: string;
  difficulty_level: string;
  upvotes: number;
  verified: boolean;
  deprecated: boolean;
  deprecation_reason: string | null;
  version: number;
}

export interface SupportTicket {
  title: string;
  description: string;
  category: string;
  priority: string;
  status: string;
  created_at: number;
  closed_at: number | null;
}

export interface TicketComment {
  ticket_hash: Uint8Array;
  author: Uint8Array;
  content: string;
  created_at: number;
}

export interface DiagnosticResult {
  diagnostic_type: string;
  findings: string[];
  severity: string;
  recommendations: string[];
  ticket_hash: Uint8Array | null;
  scrubbed: boolean;
  created_at: number;
}

export interface PrivacyPreference {
  sharing_tier: string;
  consent_verified: boolean;
  updated_at: number;
}

export interface CognitiveUpdate {
  category: string;
  encoding: number[];
  phi: number;
  resolution_pattern: string;
  published_at: number;
}

export async function setupScenario(): Promise<Scenario> {
  return new Scenario();
}
