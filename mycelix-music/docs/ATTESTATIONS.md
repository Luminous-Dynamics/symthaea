# Strategy Attestations (On-Chain)

## Goal
Link published strategy configs to on-chain attestations so clients can verify integrity without trusting the API.

## Proposed Flow
1) **Hash** the config payload (already in place) and sign with `ADMIN_SIGNER_KEY`.
2) **Publish** to an attestations/registry contract:
   - Fields: `configHash`, `adminSignature`, `publisher`, `publishedAt`.
   - Emit event `StrategyPublished(configHash, publisher)`.
3) **Verify endpoint** (`/api/strategy-configs/:id/verify-onchain`):
   - Fetch config by id, read registry event/state, confirm hash match and signature.
   - Return `{ onchain: true|false, txHash?, verifiedBy? }`.
4) **Client UX**:
   - Show “On-chain verified” badge in Lab/preview and on publish confirmation.
   - Link to explorer (e.g., `https://explorer.block/tx/{txHash}`).

## Contract Sketch (Solidity)
```solidity
event StrategyPublished(bytes32 indexed configHash, address indexed publisher, string uri);

function publish(bytes32 configHash, string calldata uri, bytes calldata adminSig) external {
    // Optional: verify adminSig recovers ADMIN_SIGNER_PUBLIC_KEY
    emit StrategyPublished(configHash, msg.sender, uri);
}
```

## API Changes
- Add `verify-onchain` endpoint.
- Store `publish_tx` in DB when available (optional).

## Deployment
- Deploy registry to target networks (local/testnet/mainnet).
- Add `STRATEGY_REGISTRY_ADDRESS` env and surface in Next.js env for client links.
