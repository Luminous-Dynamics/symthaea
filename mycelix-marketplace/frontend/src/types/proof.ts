/**
 * Proof request status for trust/attestation flows
 */
export type ProofRequestStatus = 'pending' | 'requested' | 'fulfilled' | 'denied';

export interface ProofRequest {
  key: string;
  agent_id: string;
  listing_hash?: string;
  status: ProofRequestStatus;
  message?: string;
  updated_at: number;
  usingMock?: boolean;
}
