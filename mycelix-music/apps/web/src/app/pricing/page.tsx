// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useAccount } from 'wagmi';

interface PricingTier {
  id: string;
  name: string;
  price: string;
  priceETH: string;
  period: string;
  description: string;
  features: string[];
  highlighted?: boolean;
  buttonText: string;
}

const PRICING_TIERS: PricingTier[] = [
  {
    id: 'free',
    name: 'Free',
    price: '$0',
    priceETH: '0',
    period: 'forever',
    description: 'Get started with the basics',
    features: [
      'Stream unlimited music',
      'Ad-supported listening',
      'Standard audio quality',
      'Basic recommendations',
      'Create up to 5 playlists',
    ],
    buttonText: 'Get Started',
  },
  {
    id: 'basic',
    name: 'Basic',
    price: '$9.99',
    priceETH: '0.005',
    period: '/month',
    description: 'For casual music lovers',
    features: [
      'Everything in Free',
      'Ad-free listening',
      'High quality audio (256kbps)',
      'Offline downloads (10 songs)',
      'Skip unlimited tracks',
    ],
    buttonText: 'Subscribe',
  },
  {
    id: 'premium',
    name: 'Premium',
    price: '$19.99',
    priceETH: '0.01',
    period: '/month',
    description: 'The complete experience',
    features: [
      'Everything in Basic',
      'Lossless audio quality',
      'Unlimited offline downloads',
      'Early access to releases',
      'Exclusive content & remixes',
      'Artist meet & greets',
      'Priority customer support',
    ],
    highlighted: true,
    buttonText: 'Go Premium',
  },
  {
    id: 'artist',
    name: 'Artist Pro',
    price: '$49.99',
    priceETH: '0.025',
    period: '/month',
    description: 'For creators and artists',
    features: [
      'Everything in Premium',
      'Upload unlimited tracks',
      'Advanced analytics dashboard',
      'Promotional tools',
      'Priority listing in search',
      'Revenue & royalty dashboard',
      'Direct fan messaging',
      'NFT minting tools',
    ],
    buttonText: 'Start Creating',
  },
];

export default function PricingPage() {
  const [billingPeriod, setBillingPeriod] = useState<'monthly' | 'yearly'>('monthly');
  const { isConnected } = useAccount();

  const getAdjustedPrice = (tier: PricingTier) => {
    if (tier.id === 'free') return tier;
    if (billingPeriod === 'yearly') {
      const monthlyPrice = parseFloat(tier.price.replace('$', ''));
      const yearlyPrice = (monthlyPrice * 10).toFixed(2); // 2 months free
      const yearlyETH = (parseFloat(tier.priceETH) * 10).toFixed(4);
      return {
        ...tier,
        price: `$${yearlyPrice}`,
        priceETH: yearlyETH,
        period: '/year',
      };
    }
    return tier;
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="max-w-7xl mx-auto px-4 py-16">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            Choose Your Plan
          </h1>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Unlock the full potential of decentralized music streaming.
            Pay with crypto, own your experience.
          </p>
        </div>

        {/* Billing Toggle */}
        <div className="flex items-center justify-center gap-4 mb-12">
          <span className={billingPeriod === 'monthly' ? 'text-white' : 'text-gray-500'}>
            Monthly
          </span>
          <button
            onClick={() => setBillingPeriod(p => p === 'monthly' ? 'yearly' : 'monthly')}
            className="relative w-14 h-8 bg-gray-700 rounded-full transition-colors"
          >
            <div
              className={`absolute top-1 w-6 h-6 bg-purple-500 rounded-full transition-all ${
                billingPeriod === 'yearly' ? 'left-7' : 'left-1'
              }`}
            />
          </button>
          <span className={billingPeriod === 'yearly' ? 'text-white' : 'text-gray-500'}>
            Yearly
            <span className="ml-2 text-green-400 text-sm">Save 17%</span>
          </span>
        </div>

        {/* Pricing Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {PRICING_TIERS.map((tier) => {
            const adjustedTier = getAdjustedPrice(tier);
            return (
              <div
                key={tier.id}
                className={`relative rounded-2xl p-6 ${
                  tier.highlighted
                    ? 'bg-gradient-to-b from-purple-600/20 to-pink-600/20 border-2 border-purple-500'
                    : 'bg-gray-800 border border-gray-700'
                }`}
              >
                {tier.highlighted && (
                  <div className="absolute -top-4 left-1/2 -translate-x-1/2 bg-gradient-to-r from-purple-500 to-pink-500 px-4 py-1 rounded-full text-sm font-semibold">
                    Most Popular
                  </div>
                )}

                <h3 className="text-xl font-bold mb-2">{tier.name}</h3>
                <p className="text-gray-400 text-sm mb-4">{tier.description}</p>

                <div className="mb-6">
                  <span className="text-4xl font-bold">{adjustedTier.price}</span>
                  <span className="text-gray-400">{adjustedTier.period}</span>
                  {tier.id !== 'free' && (
                    <p className="text-sm text-gray-500 mt-1">
                      ≈ {adjustedTier.priceETH} ETH
                    </p>
                  )}
                </div>

                <ul className="space-y-3 mb-6">
                  {tier.features.map((feature, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm">
                      <span className="text-green-400 mt-0.5">✓</span>
                      <span className="text-gray-300">{feature}</span>
                    </li>
                  ))}
                </ul>

                {tier.id === 'free' ? (
                  <Link
                    href="/signup"
                    className="block w-full py-3 text-center bg-gray-700 hover:bg-gray-600 rounded-xl font-semibold transition-colors"
                  >
                    {tier.buttonText}
                  </Link>
                ) : (
                  <Link
                    href={`/checkout?type=subscription&id=${tier.id}&period=${billingPeriod}`}
                    className={`block w-full py-3 text-center rounded-xl font-semibold transition-all hover:scale-[1.02] ${
                      tier.highlighted
                        ? 'bg-gradient-to-r from-purple-600 to-pink-600'
                        : 'bg-purple-600 hover:bg-purple-700'
                    }`}
                  >
                    {tier.buttonText}
                  </Link>
                )}
              </div>
            );
          })}
        </div>

        {/* FAQ Section */}
        <div className="mt-20">
          <h2 className="text-3xl font-bold text-center mb-10">
            Frequently Asked Questions
          </h2>
          <div className="max-w-3xl mx-auto space-y-4">
            <FAQ
              question="How does payment with crypto work?"
              answer="We accept ETH payments directly from your wallet. Connect your wallet, select a plan, and confirm the transaction. Your subscription activates immediately upon blockchain confirmation."
            />
            <FAQ
              question="Can I cancel anytime?"
              answer="Yes! You can cancel your subscription at any time. You'll continue to have access until the end of your billing period."
            />
            <FAQ
              question="What happens to my royalties as an artist?"
              answer="Artist royalties are distributed automatically via smart contracts. You can track and withdraw your earnings from your dashboard at any time."
            />
            <FAQ
              question="Is there a free trial?"
              answer="New users can try Premium features for 7 days free. No payment required until you decide to continue."
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function FAQ({ question, answer }: { question: string; answer: string }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="bg-gray-800 rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full px-6 py-4 flex items-center justify-between text-left"
      >
        <span className="font-semibold">{question}</span>
        <span className={`transition-transform ${open ? 'rotate-180' : ''}`}>
          ▼
        </span>
      </button>
      {open && (
        <div className="px-6 pb-4 text-gray-400">
          {answer}
        </div>
      )}
    </div>
  );
}
