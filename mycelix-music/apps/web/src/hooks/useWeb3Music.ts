// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Web3 Music Hook
 *
 * Blockchain and decentralized features:
 * - NFT minting for tracks and stems
 * - IPFS decentralized storage
 * - Smart contract royalty splits
 * - Token-gated content
 */

import { useState, useCallback, useRef, useEffect } from 'react';

// Types
export interface NFTMetadata {
  name: string;
  description: string;
  image: string;
  animation_url?: string;
  external_url?: string;
  attributes: NFTAttribute[];
  properties: {
    audio: {
      uri: string;
      mimeType: string;
      duration: number;
    };
    stems?: {
      uri: string;
      type: string;
    }[];
    visualizer?: string;
  };
}

export interface NFTAttribute {
  trait_type: string;
  value: string | number;
  display_type?: 'number' | 'date' | 'boost_percentage';
}

export interface MusicNFT {
  tokenId: string;
  contractAddress: string;
  chain: Chain;
  metadata: NFTMetadata;
  owner: string;
  creator: string;
  royaltyPercentage: number;
  editions: {
    current: number;
    max: number;
  };
  mintedAt: Date;
  price?: {
    amount: string;
    currency: string;
  };
  listed: boolean;
}

export type Chain = 'ethereum' | 'polygon' | 'optimism' | 'arbitrum' | 'base' | 'zora';

export interface RoyaltySplit {
  address: string;
  percentage: number;
  role: 'creator' | 'producer' | 'writer' | 'performer' | 'label' | 'other';
  name?: string;
}

export interface IPFSUploadResult {
  cid: string;
  uri: string;
  size: number;
  pinned: boolean;
}

export interface TokenGate {
  id: string;
  name: string;
  type: 'nft' | 'token' | 'poap';
  contractAddress: string;
  chain: Chain;
  minBalance?: string;
  tokenIds?: string[];
  content: {
    type: 'track' | 'album' | 'stems' | 'exclusive';
    id: string;
  };
}

export interface SmartContractConfig {
  royaltySplits: RoyaltySplit[];
  primarySalePercentage: number;
  secondarySalePercentage: number;
  mintPrice: string;
  maxSupply: number;
  reservedAmount: number;
}

export interface Web3MusicState {
  isConnected: boolean;
  address: string | null;
  chain: Chain | null;
  myNFTs: MusicNFT[];
  pendingMints: string[];
  ipfsGateway: string;
  isLoading: boolean;
  error: string | null;
}

// IPFS gateways
const IPFS_GATEWAYS = [
  'https://ipfs.io/ipfs/',
  'https://cloudflare-ipfs.com/ipfs/',
  'https://gateway.pinata.cloud/ipfs/',
  'https://dweb.link/ipfs/',
];

// Supported chains
const CHAIN_CONFIG: Record<Chain, { name: string; chainId: number; rpc: string }> = {
  ethereum: { name: 'Ethereum', chainId: 1, rpc: 'https://eth.llamarpc.com' },
  polygon: { name: 'Polygon', chainId: 137, rpc: 'https://polygon-rpc.com' },
  optimism: { name: 'Optimism', chainId: 10, rpc: 'https://mainnet.optimism.io' },
  arbitrum: { name: 'Arbitrum', chainId: 42161, rpc: 'https://arb1.arbitrum.io/rpc' },
  base: { name: 'Base', chainId: 8453, rpc: 'https://mainnet.base.org' },
  zora: { name: 'Zora', chainId: 7777777, rpc: 'https://rpc.zora.energy' },
};

export function useWeb3Music() {
  const [state, setState] = useState<Web3MusicState>({
    isConnected: false,
    address: null,
    chain: null,
    myNFTs: [],
    pendingMints: [],
    ipfsGateway: IPFS_GATEWAYS[0],
    isLoading: false,
    error: null,
  });

  const providerRef = useRef<any>(null);

  /**
   * Connect wallet
   */
  const connect = useCallback(async (): Promise<boolean> => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      if (typeof window === 'undefined' || !window.ethereum) {
        throw new Error('No wallet found. Please install MetaMask or another Web3 wallet.');
      }

      // Request account access
      const accounts = await window.ethereum.request({
        method: 'eth_requestAccounts',
      });

      if (accounts.length === 0) {
        throw new Error('No accounts found');
      }

      // Get chain
      const chainId = await window.ethereum.request({ method: 'eth_chainId' });
      const chain = getChainFromId(parseInt(chainId, 16));

      providerRef.current = window.ethereum;

      setState(prev => ({
        ...prev,
        isConnected: true,
        address: accounts[0],
        chain,
        isLoading: false,
      }));

      // Listen for account changes
      window.ethereum.on('accountsChanged', handleAccountsChanged);
      window.ethereum.on('chainChanged', handleChainChanged);

      // Load user's NFTs
      await loadMyNFTs(accounts[0]);

      return true;
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to connect wallet',
      }));
      return false;
    }
  }, []);

  /**
   * Handle account changes
   */
  const handleAccountsChanged = useCallback((accounts: string[]) => {
    if (accounts.length === 0) {
      disconnect();
    } else {
      setState(prev => ({ ...prev, address: accounts[0] }));
      loadMyNFTs(accounts[0]);
    }
  }, []);

  /**
   * Handle chain changes
   */
  const handleChainChanged = useCallback((chainId: string) => {
    const chain = getChainFromId(parseInt(chainId, 16));
    setState(prev => ({ ...prev, chain }));
  }, []);

  /**
   * Disconnect wallet
   */
  const disconnect = useCallback(() => {
    if (window.ethereum) {
      window.ethereum.removeListener('accountsChanged', handleAccountsChanged);
      window.ethereum.removeListener('chainChanged', handleChainChanged);
    }

    setState(prev => ({
      ...prev,
      isConnected: false,
      address: null,
      chain: null,
      myNFTs: [],
    }));
  }, [handleAccountsChanged, handleChainChanged]);

  /**
   * Switch chain
   */
  const switchChain = useCallback(async (chain: Chain): Promise<boolean> => {
    if (!window.ethereum) return false;

    const chainConfig = CHAIN_CONFIG[chain];
    const chainIdHex = `0x${chainConfig.chainId.toString(16)}`;

    try {
      await window.ethereum.request({
        method: 'wallet_switchEthereumChain',
        params: [{ chainId: chainIdHex }],
      });
      return true;
    } catch (error: any) {
      // Chain not added, try to add it
      if (error.code === 4902) {
        try {
          await window.ethereum.request({
            method: 'wallet_addEthereumChain',
            params: [{
              chainId: chainIdHex,
              chainName: chainConfig.name,
              rpcUrls: [chainConfig.rpc],
            }],
          });
          return true;
        } catch {
          return false;
        }
      }
      return false;
    }
  }, []);

  /**
   * Upload to IPFS
   */
  const uploadToIPFS = useCallback(async (
    file: File | Blob,
    options: { pin?: boolean; name?: string } = {}
  ): Promise<IPFSUploadResult> => {
    setState(prev => ({ ...prev, isLoading: true }));

    try {
      // Using Pinata or similar service
      const formData = new FormData();
      formData.append('file', file, options.name);

      const response = await fetch('/api/ipfs/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to upload to IPFS');
      }

      const result = await response.json();

      setState(prev => ({ ...prev, isLoading: false }));

      return {
        cid: result.cid,
        uri: `ipfs://${result.cid}`,
        size: file.size,
        pinned: options.pin ?? true,
      };
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'IPFS upload failed',
      }));
      throw error;
    }
  }, []);

  /**
   * Upload audio and metadata to IPFS
   */
  const uploadMusicToIPFS = useCallback(async (
    audioFile: File,
    coverImage: File,
    metadata: Omit<NFTMetadata, 'image' | 'properties'>
  ): Promise<{ audioCid: string; imageCid: string; metadataCid: string }> => {
    setState(prev => ({ ...prev, isLoading: true }));

    try {
      // Upload audio
      const audioResult = await uploadToIPFS(audioFile, { name: 'audio.mp3' });

      // Upload cover image
      const imageResult = await uploadToIPFS(coverImage, { name: 'cover.png' });

      // Create full metadata
      const fullMetadata: NFTMetadata = {
        ...metadata,
        image: `ipfs://${imageResult.cid}`,
        animation_url: `ipfs://${audioResult.cid}`,
        properties: {
          audio: {
            uri: `ipfs://${audioResult.cid}`,
            mimeType: audioFile.type,
            duration: 0, // Would extract from audio
          },
        },
      };

      // Upload metadata
      const metadataBlob = new Blob([JSON.stringify(fullMetadata)], {
        type: 'application/json',
      });
      const metadataResult = await uploadToIPFS(metadataBlob, { name: 'metadata.json' });

      setState(prev => ({ ...prev, isLoading: false }));

      return {
        audioCid: audioResult.cid,
        imageCid: imageResult.cid,
        metadataCid: metadataResult.cid,
      };
    } catch (error) {
      setState(prev => ({ ...prev, isLoading: false }));
      throw error;
    }
  }, [uploadToIPFS]);

  /**
   * Mint music NFT
   */
  const mintNFT = useCallback(async (
    metadataUri: string,
    config: SmartContractConfig,
    chain: Chain = 'polygon'
  ): Promise<MusicNFT | null> => {
    if (!state.isConnected || !state.address) {
      throw new Error('Wallet not connected');
    }

    setState(prev => ({ ...prev, isLoading: true }));

    try {
      // Switch to correct chain if needed
      if (state.chain !== chain) {
        await switchChain(chain);
      }

      // Deploy or use existing contract
      const contractAddress = await deployOrGetContract(config, chain);

      // Mint NFT
      const mintTx = await mintOnContract(contractAddress, metadataUri, config);

      // Wait for confirmation
      const receipt = await waitForTransaction(mintTx.hash);
      const tokenId = extractTokenId(receipt);

      // Fetch metadata
      const metadataResponse = await fetch(resolveIPFSUri(metadataUri));
      const metadata = await metadataResponse.json();

      const nft: MusicNFT = {
        tokenId,
        contractAddress,
        chain,
        metadata,
        owner: state.address,
        creator: state.address,
        royaltyPercentage: config.secondarySalePercentage,
        editions: {
          current: 1,
          max: config.maxSupply,
        },
        mintedAt: new Date(),
        listed: false,
      };

      setState(prev => ({
        ...prev,
        isLoading: false,
        myNFTs: [...prev.myNFTs, nft],
      }));

      return nft;
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Minting failed',
      }));
      return null;
    }
  }, [state.isConnected, state.address, state.chain, switchChain]);

  /**
   * Create royalty split contract
   */
  const createRoyaltySplit = useCallback(async (
    splits: RoyaltySplit[]
  ): Promise<string | null> => {
    if (!state.isConnected) {
      throw new Error('Wallet not connected');
    }

    setState(prev => ({ ...prev, isLoading: true }));

    try {
      // Validate splits total 100%
      const total = splits.reduce((sum, s) => sum + s.percentage, 0);
      if (Math.abs(total - 100) > 0.01) {
        throw new Error('Royalty splits must total 100%');
      }

      // Deploy split contract
      const contractAddress = await deploySplitContract(splits);

      setState(prev => ({ ...prev, isLoading: false }));
      return contractAddress;
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to create split',
      }));
      return null;
    }
  }, [state.isConnected]);

  /**
   * Create token gate
   */
  const createTokenGate = useCallback(async (
    gate: Omit<TokenGate, 'id'>
  ): Promise<TokenGate | null> => {
    setState(prev => ({ ...prev, isLoading: true }));

    try {
      // Store gate configuration
      const response = await fetch('/api/token-gates', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(gate),
      });

      if (!response.ok) {
        throw new Error('Failed to create token gate');
      }

      const result = await response.json();

      setState(prev => ({ ...prev, isLoading: false }));
      return result;
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to create gate',
      }));
      return null;
    }
  }, []);

  /**
   * Verify token gate access
   */
  const verifyTokenGateAccess = useCallback(async (
    gateId: string
  ): Promise<boolean> => {
    if (!state.isConnected || !state.address) {
      return false;
    }

    try {
      const response = await fetch(`/api/token-gates/${gateId}/verify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ address: state.address }),
      });

      const result = await response.json();
      return result.hasAccess;
    } catch {
      return false;
    }
  }, [state.isConnected, state.address]);

  /**
   * Load user's NFTs
   */
  const loadMyNFTs = useCallback(async (address: string) => {
    try {
      // Would query indexer or contract
      const nfts: MusicNFT[] = [];
      setState(prev => ({ ...prev, myNFTs: nfts }));
    } catch (error) {
      console.error('Failed to load NFTs:', error);
    }
  }, []);

  /**
   * List NFT for sale
   */
  const listForSale = useCallback(async (
    tokenId: string,
    price: string,
    currency: string = 'ETH'
  ): Promise<boolean> => {
    setState(prev => ({ ...prev, isLoading: true }));

    try {
      // Would interact with marketplace contract
      await new Promise(resolve => setTimeout(resolve, 1000));

      setState(prev => ({
        ...prev,
        isLoading: false,
        myNFTs: prev.myNFTs.map(nft =>
          nft.tokenId === tokenId
            ? { ...nft, listed: true, price: { amount: price, currency } }
            : nft
        ),
      }));

      return true;
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to list NFT',
      }));
      return false;
    }
  }, []);

  /**
   * Resolve IPFS URI to HTTP URL
   */
  const resolveIPFSUri = useCallback((uri: string): string => {
    if (uri.startsWith('ipfs://')) {
      const cid = uri.replace('ipfs://', '');
      return `${state.ipfsGateway}${cid}`;
    }
    return uri;
  }, [state.ipfsGateway]);

  return {
    ...state,
    chainConfig: CHAIN_CONFIG,
    supportedChains: Object.keys(CHAIN_CONFIG) as Chain[],
    connect,
    disconnect,
    switchChain,
    uploadToIPFS,
    uploadMusicToIPFS,
    mintNFT,
    createRoyaltySplit,
    createTokenGate,
    verifyTokenGateAccess,
    listForSale,
    resolveIPFSUri,
  };
}

// ============================================================================
// Helper Functions
// ============================================================================

function getChainFromId(chainId: number): Chain | null {
  for (const [chain, config] of Object.entries(CHAIN_CONFIG)) {
    if (config.chainId === chainId) {
      return chain as Chain;
    }
  }
  return null;
}

async function deployOrGetContract(
  config: SmartContractConfig,
  chain: Chain
): Promise<string> {
  // Would deploy or return existing contract
  return '0x1234567890abcdef1234567890abcdef12345678';
}

async function mintOnContract(
  contractAddress: string,
  metadataUri: string,
  config: SmartContractConfig
): Promise<{ hash: string }> {
  // Would call contract mint function
  return { hash: '0xabcdef...' };
}

async function waitForTransaction(hash: string): Promise<any> {
  // Would wait for tx confirmation
  await new Promise(resolve => setTimeout(resolve, 2000));
  return { logs: [] };
}

function extractTokenId(receipt: any): string {
  // Would extract from logs
  return '1';
}

async function deploySplitContract(splits: RoyaltySplit[]): Promise<string> {
  // Would deploy 0xSplits or similar
  return '0xsplit...';
}

// Type augmentation for window.ethereum
declare global {
  interface Window {
    ethereum?: {
      request: (args: { method: string; params?: any[] }) => Promise<any>;
      on: (event: string, callback: (...args: any[]) => void) => void;
      removeListener: (event: string, callback: (...args: any[]) => void) => void;
    };
  }
}

export default useWeb3Music;
