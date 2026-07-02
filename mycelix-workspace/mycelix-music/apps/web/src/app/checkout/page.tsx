// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState, useEffect } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { useAccount, useWriteContract, useWaitForTransactionReceipt } from 'wagmi';
import { parseEther, formatEther } from 'viem';

type CheckoutType = 'subscription' | 'nft' | 'tip';

interface CheckoutItem {
  type: CheckoutType;
  id: string;
  name: string;
  description: string;
  price: string;
  priceUSD: string;
  image?: string;
  metadata?: Record<string, unknown>;
}

interface SubscriptionTier {
  id: string;
  name: string;
  price: string;
  priceUSD: string;
  features: string[];
  period: 'monthly' | 'yearly';
}

const SUBSCRIPTION_TIERS: SubscriptionTier[] = [
  {
    id: 'basic',
    name: 'Basic',
    price: '0.005',
    priceUSD: '$9.99',
    period: 'monthly',
    features: [
      'Ad-free listening',
      'High quality audio (256kbps)',
      'Offline downloads (10 songs)',
      'Basic recommendations',
    ],
  },
  {
    id: 'premium',
    name: 'Premium',
    price: '0.01',
    priceUSD: '$19.99',
    period: 'monthly',
    features: [
      'Everything in Basic',
      'Lossless audio quality',
      'Unlimited downloads',
      'Early access to releases',
      'Exclusive content',
      'Artist meet & greets',
    ],
  },
  {
    id: 'artist',
    name: 'Artist Pro',
    price: '0.025',
    priceUSD: '$49.99',
    period: 'monthly',
    features: [
      'Everything in Premium',
      'Upload unlimited tracks',
      'Advanced analytics',
      'Priority support',
      'Promotional tools',
      'Revenue dashboard',
    ],
  },
];

export default function CheckoutPage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const { address, isConnected } = useAccount();

  const [item, setItem] = useState<CheckoutItem | null>(null);
  const [loading, setLoading] = useState(true);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [txHash, setTxHash] = useState<string | null>(null);

  const type = searchParams.get('type') as CheckoutType;
  const itemId = searchParams.get('id');

  const { writeContract, data: hash, isPending } = useWriteContract();
  const { isLoading: isConfirming, isSuccess } = useWaitForTransactionReceipt({
    hash,
  });

  useEffect(() => {
    if (type && itemId) {
      loadCheckoutItem();
    }
  }, [type, itemId]);

  useEffect(() => {
    if (hash) {
      setTxHash(hash);
    }
  }, [hash]);

  useEffect(() => {
    if (isSuccess) {
      handlePaymentSuccess();
    }
  }, [isSuccess]);

  const loadCheckoutItem = async () => {
    setLoading(true);
    try {
      if (type === 'subscription') {
        const tier = SUBSCRIPTION_TIERS.find(t => t.id === itemId);
        if (tier) {
          setItem({
            type: 'subscription',
            id: tier.id,
            name: `${tier.name} Subscription`,
            description: tier.features.join(', '),
            price: tier.price,
            priceUSD: tier.period === 'monthly' ? `${tier.priceUSD}/mo` : `${tier.priceUSD}/yr`,
            metadata: { period: tier.period, features: tier.features },
          });
        }
      } else if (type === 'nft' || type === 'tip') {
        const response = await fetch(`/api/checkout/item?type=${type}&id=${itemId}`);
        if (response.ok) {
          const data = await response.json();
          setItem(data);
        }
      }
    } catch (err) {
      setError('Failed to load checkout item');
    } finally {
      setLoading(false);
    }
  };

  const handlePayment = async () => {
    if (!isConnected || !address || !item) {
      setError('Please connect your wallet');
      return;
    }

    setProcessing(true);
    setError(null);

    try {
      const contractAddress = process.env.NEXT_PUBLIC_PAYMENT_CONTRACT as `0x${string}`;

      if (type === 'subscription') {
        writeContract({
          address: contractAddress,
          abi: SUBSCRIPTION_ABI,
          functionName: 'subscribe',
          args: [item.id],
          value: parseEther(item.price),
        });
      } else if (type === 'nft') {
        writeContract({
          address: contractAddress,
          abi: NFT_ABI,
          functionName: 'mint',
          args: [item.id],
          value: parseEther(item.price),
        });
      } else if (type === 'tip') {
        writeContract({
          address: contractAddress,
          abi: TIP_ABI,
          functionName: 'tip',
          args: [item.metadata?.artistAddress as `0x${string}`],
          value: parseEther(item.price),
        });
      }
    } catch (err) {
      setError('Transaction failed. Please try again.');
      setProcessing(false);
    }
  };

  const handlePaymentSuccess = async () => {
    try {
      await fetch('/api/checkout/confirm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type,
          itemId: item?.id,
          txHash,
          address,
        }),
      });

      router.push(`/checkout/success?tx=${txHash}`);
    } catch (err) {
      console.error('Failed to confirm payment:', err);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-purple-500" />
      </div>
    );
  }

  if (!item) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center text-white">
        <div className="text-center">
          <h1 className="text-2xl font-bold mb-4">Item Not Found</h1>
          <button
            onClick={() => router.back()}
            className="px-6 py-3 bg-purple-600 rounded-lg"
          >
            Go Back
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="max-w-2xl mx-auto px-4 py-12">
        <h1 className="text-3xl font-bold mb-8">Checkout</h1>

        {/* Order Summary */}
        <div className="bg-gray-800 rounded-xl p-6 mb-6">
          <h2 className="text-lg font-semibold mb-4">Order Summary</h2>

          <div className="flex gap-4 mb-4">
            {item.image && (
              <img
                src={item.image}
                alt={item.name}
                className="w-20 h-20 rounded-lg object-cover"
              />
            )}
            <div className="flex-1">
              <h3 className="font-semibold">{item.name}</h3>
              <p className="text-gray-400 text-sm mt-1">{item.description}</p>
            </div>
          </div>

          {type === 'subscription' && item.metadata?.features && (
            <div className="border-t border-gray-700 pt-4 mt-4">
              <p className="text-sm text-gray-400 mb-2">Includes:</p>
              <ul className="space-y-1">
                {(item.metadata.features as string[]).map((feature, i) => (
                  <li key={i} className="text-sm flex items-center gap-2">
                    <span className="text-green-400">✓</span>
                    {feature}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="border-t border-gray-700 pt-4 mt-4">
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Total</span>
              <div className="text-right">
                <p className="text-2xl font-bold">{item.price} ETH</p>
                <p className="text-gray-400 text-sm">{item.priceUSD}</p>
              </div>
            </div>
          </div>
        </div>

        {/* Payment Section */}
        <div className="bg-gray-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold mb-4">Payment</h2>

          {!isConnected ? (
            <div className="text-center py-6">
              <p className="text-gray-400 mb-4">Connect your wallet to continue</p>
              <w3m-button />
            </div>
          ) : (
            <div>
              <div className="bg-gray-700 rounded-lg p-4 mb-4">
                <p className="text-sm text-gray-400">Connected Wallet</p>
                <p className="font-mono">{address?.slice(0, 6)}...{address?.slice(-4)}</p>
              </div>

              {error && (
                <div className="bg-red-500/20 border border-red-500 rounded-lg p-4 mb-4">
                  <p className="text-red-400">{error}</p>
                </div>
              )}

              {txHash && !isSuccess && (
                <div className="bg-yellow-500/20 border border-yellow-500 rounded-lg p-4 mb-4">
                  <p className="text-yellow-400">Transaction submitted. Waiting for confirmation...</p>
                  <a
                    href={`https://etherscan.io/tx/${txHash}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-yellow-300 underline"
                  >
                    View on Etherscan
                  </a>
                </div>
              )}

              <button
                onClick={handlePayment}
                disabled={isPending || isConfirming}
                className="w-full py-4 bg-gradient-to-r from-purple-600 to-pink-600 rounded-xl font-semibold text-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all hover:scale-[1.02]"
              >
                {isPending || isConfirming ? (
                  <span className="flex items-center justify-center gap-2">
                    <span className="animate-spin rounded-full h-5 w-5 border-t-2 border-white" />
                    {isConfirming ? 'Confirming...' : 'Processing...'}
                  </span>
                ) : (
                  `Pay ${item.price} ETH`
                )}
              </button>

              <p className="text-center text-gray-500 text-sm mt-4">
                By completing this purchase, you agree to our Terms of Service
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Contract ABIs (simplified)
const SUBSCRIPTION_ABI = [
  {
    name: 'subscribe',
    type: 'function',
    inputs: [{ name: 'tierId', type: 'string' }],
    outputs: [],
    stateMutability: 'payable',
  },
] as const;

const NFT_ABI = [
  {
    name: 'mint',
    type: 'function',
    inputs: [{ name: 'tokenId', type: 'string' }],
    outputs: [],
    stateMutability: 'payable',
  },
] as const;

const TIP_ABI = [
  {
    name: 'tip',
    type: 'function',
    inputs: [{ name: 'artist', type: 'address' }],
    outputs: [],
    stateMutability: 'payable',
  },
] as const;
