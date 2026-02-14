/**
 * React Hooks for Mycelix Energy Module
 */

import type { QueryState, MutationState } from './index.js';
import type {
  EnergyProject,
  RegisterProjectInput,
  EnergyParticipant,
  RegisterParticipantInput,
  EnergyTrade,
  TradeInput,
  EnergyCredit,
  Investment,
  InvestInput,
  EnergySource,
  ParticipantType,
  ProjectStatus,
  HolochainRecord,
} from '../energy/index.js';

// Project Hooks
export function useEnergyProject(
  _projectId: string
): QueryState<HolochainRecord<EnergyProject> | null> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useProjectsBySource(
  _source: EnergySource
): QueryState<HolochainRecord<EnergyProject>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useProjectsByStatus(
  _status: ProjectStatus
): QueryState<HolochainRecord<EnergyProject>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useSearchProjects(_query: {
  source?: EnergySource;
  status?: ProjectStatus;
  minCapacity?: number;
}): QueryState<HolochainRecord<EnergyProject>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useRegisterEnergyProject(): MutationState<
  HolochainRecord<EnergyProject>,
  RegisterProjectInput
> {
  throw new Error('React hooks require React. This is a type stub.');
}

// Participant Hooks
export function useEnergyParticipant(
  _did: string
): QueryState<HolochainRecord<EnergyParticipant> | null> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useParticipantsByType(
  _type: ParticipantType
): QueryState<HolochainRecord<EnergyParticipant>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useRegisterEnergyParticipantMutation(): MutationState<
  HolochainRecord<EnergyParticipant>,
  RegisterParticipantInput
> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useUpdateEnergyCapacity(): MutationState<
  HolochainRecord<EnergyParticipant>,
  number
> {
  throw new Error('React hooks require React. This is a type stub.');
}

// Trading Hooks
export function useTradesByParticipant(_did: string): QueryState<HolochainRecord<EnergyTrade>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useOpenTrades(_source?: EnergySource): QueryState<HolochainRecord<EnergyTrade>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useCreateEnergyTrade(): MutationState<HolochainRecord<EnergyTrade>, TradeInput> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useAcceptEnergyTrade(): MutationState<HolochainRecord<EnergyTrade>, string> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useConfirmEnergyDelivery(): MutationState<HolochainRecord<EnergyTrade>, string> {
  throw new Error('React hooks require React. This is a type stub.');
}

// Credits Hooks
export function useEnergyCreditsByParticipant(
  _participantDid: string
): QueryState<HolochainRecord<EnergyCredit>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useIssueEnergyCredit(): MutationState<
  HolochainRecord<EnergyCredit>,
  { participantDid: string; amountKwh: number; source: EnergySource }
> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useUseEnergyCredit(): MutationState<HolochainRecord<EnergyCredit>, string> {
  throw new Error('React hooks require React. This is a type stub.');
}

// Investment Hooks
export function useInvestmentsByProject(
  _projectId: string
): QueryState<HolochainRecord<Investment>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useInvestmentsByInvestor(
  _investorDid: string
): QueryState<HolochainRecord<Investment>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useInvestInProject(): MutationState<HolochainRecord<Investment>, InvestInput> {
  throw new Error('React hooks require React. This is a type stub.');
}
