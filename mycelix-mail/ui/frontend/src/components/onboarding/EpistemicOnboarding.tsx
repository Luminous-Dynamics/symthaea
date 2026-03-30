// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Epistemic Onboarding Wizard
 *
 * Guides new users through the epistemic mail features:
 * 1. Introduction to trust-based email
 * 2. Understanding epistemic tiers (T0-T4)
 * 3. Setting up identity verification
 * 4. Building your trust network
 * 5. Configuring AI insights
 *
 * Can be triggered on first use or from settings.
 */

import { useState, useEffect } from 'react';
import { useMutation } from '@tanstack/react-query';

interface EpistemicOnboardingProps {
  onComplete: () => void;
  onSkip?: () => void;
}

interface OnboardingStep {
  id: string;
  title: string;
  description: string;
  icon: string;
  content: React.ReactNode;
  action?: {
    label: string;
    onClick: () => void | Promise<void>;
  };
}

// Step 1: Introduction
function IntroductionStep() {
  return (
    <div className="text-center py-8">
      <div className="w-24 h-24 mx-auto mb-6 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl flex items-center justify-center shadow-lg">
        <span className="text-4xl">🔐</span>
      </div>
      <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-4">
        Welcome to Epistemic Mail
      </h2>
      <p className="text-gray-600 dark:text-gray-400 max-w-md mx-auto leading-relaxed">
        Mycelix Mail uses <strong>verifiable trust</strong> to help you know who you're really
        communicating with. No more phishing, impersonation, or spam from unknown senders.
      </p>
      <div className="mt-8 grid grid-cols-3 gap-4 max-w-lg mx-auto">
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
          <span className="text-2xl block mb-2">🛡️</span>
          <span className="text-xs text-blue-700 dark:text-blue-300 font-medium">Verified Identity</span>
        </div>
        <div className="p-4 bg-emerald-50 dark:bg-emerald-900/20 rounded-lg">
          <span className="text-2xl block mb-2">🌐</span>
          <span className="text-xs text-emerald-700 dark:text-emerald-300 font-medium">Trust Network</span>
        </div>
        <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
          <span className="text-2xl block mb-2">🤖</span>
          <span className="text-xs text-purple-700 dark:text-purple-300 font-medium">AI Insights</span>
        </div>
      </div>
    </div>
  );
}

// Step 2: Epistemic Tiers
function TiersStep() {
  const tiers = [
    {
      tier: 'T0',
      name: 'Unverifiable',
      color: 'gray',
      description: 'No verification available',
      example: 'Random email with no proof',
    },
    {
      tier: 'T1',
      name: 'Sender Known',
      color: 'blue',
      description: 'Email verified (DKIM/SPF)',
      example: 'Newsletter from verified domain',
    },
    {
      tier: 'T2',
      name: 'Identity Verified',
      color: 'indigo',
      description: 'Real identity attestation',
      example: 'Gitcoin Passport verified',
    },
    {
      tier: 'T3',
      name: 'Trust Connected',
      color: 'purple',
      description: 'In your trust network',
      example: 'Friend of a trusted contact',
    },
    {
      tier: 'T4',
      name: 'Fully Attested',
      color: 'emerald',
      description: 'Multiple verified claims',
      example: 'Colleague with org credentials',
    },
  ];

  return (
    <div className="py-4">
      <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-2 text-center">
        Understanding Epistemic Tiers
      </h2>
      <p className="text-gray-600 dark:text-gray-400 text-center mb-6 text-sm">
        Every email is classified by how verifiable its sender is
      </p>

      <div className="space-y-3">
        {tiers.map((tier) => (
          <div
            key={tier.tier}
            className={`flex items-center gap-4 p-3 rounded-lg bg-${tier.color}-50 dark:bg-${tier.color}-900/20 border border-${tier.color}-200 dark:border-${tier.color}-800`}
          >
            <div className={`w-12 h-12 rounded-lg bg-${tier.color}-100 dark:bg-${tier.color}-900/40 flex items-center justify-center`}>
              <span className={`text-lg font-bold text-${tier.color}-600 dark:text-${tier.color}-400`}>
                {tier.tier}
              </span>
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span className={`font-medium text-${tier.color}-700 dark:text-${tier.color}-300`}>
                  {tier.name}
                </span>
              </div>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                {tier.description}
              </p>
            </div>
            <div className="text-xs text-gray-400 dark:text-gray-500 max-w-[120px] text-right">
              {tier.example}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// Step 3: Identity Verification
function VerificationStep({ onVerify }: { onVerify: () => void }) {
  const [verifying, setVerifying] = useState(false);

  const handleVerify = async () => {
    setVerifying(true);
    // Simulate verification
    await new Promise((resolve) => setTimeout(resolve, 2000));
    setVerifying(false);
    onVerify();
  };

  return (
    <div className="py-4">
      <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-2 text-center">
        Verify Your Identity
      </h2>
      <p className="text-gray-600 dark:text-gray-400 text-center mb-6 text-sm">
        Get verified so others can trust emails from you
      </p>

      <div className="space-y-4">
        {/* Email Verification - Already done */}
        <div className="p-4 rounded-lg border border-emerald-200 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-900/20">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="text-2xl">📧</span>
              <div>
                <p className="font-medium text-gray-900 dark:text-gray-100">Email Verified</p>
                <p className="text-xs text-gray-500 dark:text-gray-400">E1 Level • $100 attack cost</p>
              </div>
            </div>
            <span className="text-emerald-500">✓</span>
          </div>
        </div>

        {/* Gitcoin Passport */}
        <div className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="text-2xl">🛂</span>
              <div>
                <p className="font-medium text-gray-900 dark:text-gray-100">Gitcoin Passport</p>
                <p className="text-xs text-gray-500 dark:text-gray-400">E2 Level • $1,000 attack cost</p>
              </div>
            </div>
            <button
              onClick={handleVerify}
              disabled={verifying}
              className="px-3 py-1.5 text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 rounded-lg disabled:opacity-50"
            >
              {verifying ? 'Connecting...' : 'Connect'}
            </button>
          </div>
        </div>

        {/* GitHub */}
        <div className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="text-2xl">🐙</span>
              <div>
                <p className="font-medium text-gray-900 dark:text-gray-100">GitHub Account</p>
                <p className="text-xs text-gray-500 dark:text-gray-400">Developer identity proof</p>
              </div>
            </div>
            <button className="px-3 py-1.5 text-sm font-medium text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700 rounded-lg">
              Link
            </button>
          </div>
        </div>

        {/* More options */}
        <button className="w-full py-2 text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300">
          View more verification options →
        </button>
      </div>
    </div>
  );
}

// Step 4: Build Trust Network
function TrustNetworkStep() {
  const [inviteSent, setInviteSent] = useState(false);

  return (
    <div className="py-4">
      <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-2 text-center">
        Build Your Trust Network
      </h2>
      <p className="text-gray-600 dark:text-gray-400 text-center mb-6 text-sm">
        Connect with people you know and trust to strengthen your network
      </p>

      {/* Trust visualization */}
      <div className="relative h-48 mb-6">
        <div className="absolute inset-0 flex items-center justify-center">
          {/* Central node (you) */}
          <div className="w-16 h-16 rounded-full bg-blue-500 flex items-center justify-center text-white font-bold z-10">
            You
          </div>

          {/* Connection lines */}
          <svg className="absolute inset-0 w-full h-full" style={{ zIndex: 0 }}>
            <line x1="50%" y1="50%" x2="25%" y2="25%" stroke="#10b981" strokeWidth="2" strokeDasharray="4" />
            <line x1="50%" y1="50%" x2="75%" y2="25%" stroke="#10b981" strokeWidth="2" strokeDasharray="4" />
            <line x1="50%" y1="50%" x2="20%" y2="70%" stroke="#6366f1" strokeWidth="2" strokeDasharray="4" />
            <line x1="50%" y1="50%" x2="80%" y2="70%" stroke="#6366f1" strokeWidth="2" strokeDasharray="4" />
          </svg>

          {/* Contact nodes */}
          <div className="absolute top-4 left-1/4 w-10 h-10 rounded-full bg-emerald-500 flex items-center justify-center text-white text-sm">
            A
          </div>
          <div className="absolute top-4 right-1/4 w-10 h-10 rounded-full bg-emerald-500 flex items-center justify-center text-white text-sm">
            B
          </div>
          <div className="absolute bottom-4 left-1/5 w-10 h-10 rounded-full bg-indigo-500 flex items-center justify-center text-white text-sm">
            ?
          </div>
          <div className="absolute bottom-4 right-1/5 w-10 h-10 rounded-full bg-indigo-500 flex items-center justify-center text-white text-sm">
            ?
          </div>
        </div>
      </div>

      {/* Invite contacts */}
      <div className="space-y-4">
        <div className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
          <p className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-3">
            Invite trusted contacts
          </p>
          <div className="flex gap-2">
            <input
              type="email"
              placeholder="friend@example.com"
              className="flex-1 px-3 py-2 text-sm border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800"
            />
            <button
              onClick={() => setInviteSent(true)}
              className="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg"
            >
              Invite
            </button>
          </div>
          {inviteSent && (
            <p className="text-xs text-emerald-600 mt-2">
              ✓ Invitation sent! They'll be added to your trust network once they join.
            </p>
          )}
        </div>

        <p className="text-xs text-gray-500 dark:text-gray-400 text-center">
          Your contacts will help validate unknown senders through trust paths
        </p>
      </div>
    </div>
  );
}

// Step 5: AI Insights
function AIInsightsStep() {
  const [enabled, setEnabled] = useState({
    intentDetection: true,
    prioritization: true,
    replySuggestions: false,
    summarization: true,
  });

  return (
    <div className="py-4">
      <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-2 text-center">
        Configure AI Insights
      </h2>
      <p className="text-gray-600 dark:text-gray-400 text-center mb-6 text-sm">
        Local AI helps you understand and respond to emails better
      </p>

      <div className="space-y-3">
        {[
          {
            key: 'intentDetection',
            icon: '🎯',
            title: 'Intent Detection',
            description: 'Identifies questions, requests, and action items',
          },
          {
            key: 'prioritization',
            icon: '📊',
            title: 'Smart Prioritization',
            description: 'Highlights urgent and important emails',
          },
          {
            key: 'summarization',
            icon: '📝',
            title: 'Thread Summaries',
            description: 'Summarizes long conversations',
          },
          {
            key: 'replySuggestions',
            icon: '💬',
            title: 'Reply Suggestions',
            description: 'Suggests responses based on context',
          },
        ].map((feature) => (
          <label
            key={feature.key}
            className="flex items-center justify-between p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700"
          >
            <div className="flex items-center gap-3">
              <span className="text-2xl">{feature.icon}</span>
              <div>
                <p className="font-medium text-gray-900 dark:text-gray-100">{feature.title}</p>
                <p className="text-xs text-gray-500 dark:text-gray-400">{feature.description}</p>
              </div>
            </div>
            <div className="relative">
              <input
                type="checkbox"
                checked={enabled[feature.key as keyof typeof enabled]}
                onChange={(e) => setEnabled({ ...enabled, [feature.key]: e.target.checked })}
                className="sr-only"
              />
              <div
                className={`w-10 h-5 rounded-full transition-colors ${
                  enabled[feature.key as keyof typeof enabled]
                    ? 'bg-blue-500'
                    : 'bg-gray-300 dark:bg-gray-600'
                }`}
              >
                <div
                  className={`w-4 h-4 rounded-full bg-white shadow transform transition-transform ${
                    enabled[feature.key as keyof typeof enabled]
                      ? 'translate-x-5'
                      : 'translate-x-0.5'
                  } mt-0.5`}
                />
              </div>
            </div>
          </label>
        ))}
      </div>

      <p className="text-xs text-gray-500 dark:text-gray-400 text-center mt-4">
        🔒 All AI processing happens locally on your device
      </p>
    </div>
  );
}

// Step 6: Complete
function CompleteStep() {
  return (
    <div className="text-center py-8">
      <div className="w-24 h-24 mx-auto mb-6 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-full flex items-center justify-center shadow-lg">
        <span className="text-4xl">🎉</span>
      </div>
      <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-4">
        You're All Set!
      </h2>
      <p className="text-gray-600 dark:text-gray-400 max-w-md mx-auto leading-relaxed mb-8">
        Your epistemic mail environment is configured. You'll now see trust indicators on every email
        and can manage your trust network from the dashboard.
      </p>

      <div className="grid grid-cols-2 gap-4 max-w-sm mx-auto">
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg text-left">
          <p className="text-sm font-medium text-blue-700 dark:text-blue-300">Quick Tips</p>
          <ul className="text-xs text-blue-600 dark:text-blue-400 mt-2 space-y-1">
            <li>• Alt+I opens insights</li>
            <li>• Alt+C shows contact</li>
            <li>• Alt+T for thread summary</li>
          </ul>
        </div>
        <div className="p-4 bg-emerald-50 dark:bg-emerald-900/20 rounded-lg text-left">
          <p className="text-sm font-medium text-emerald-700 dark:text-emerald-300">Next Steps</p>
          <ul className="text-xs text-emerald-600 dark:text-emerald-400 mt-2 space-y-1">
            <li>• Invite more contacts</li>
            <li>• Add credentials</li>
            <li>• Explore trust graph</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

// Main Onboarding Component
export default function EpistemicOnboarding({ onComplete, onSkip }: EpistemicOnboardingProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [verified, setVerified] = useState(false);

  const steps: OnboardingStep[] = [
    {
      id: 'intro',
      title: 'Welcome',
      description: 'Introduction to epistemic mail',
      icon: '👋',
      content: <IntroductionStep />,
    },
    {
      id: 'tiers',
      title: 'Tiers',
      description: 'Understanding trust levels',
      icon: '📊',
      content: <TiersStep />,
    },
    {
      id: 'verify',
      title: 'Verify',
      description: 'Set up identity verification',
      icon: '🔐',
      content: <VerificationStep onVerify={() => setVerified(true)} />,
    },
    {
      id: 'network',
      title: 'Network',
      description: 'Build your trust network',
      icon: '🌐',
      content: <TrustNetworkStep />,
    },
    {
      id: 'ai',
      title: 'AI',
      description: 'Configure AI insights',
      icon: '🤖',
      content: <AIInsightsStep />,
    },
    {
      id: 'complete',
      title: 'Done',
      description: 'Setup complete',
      icon: '✅',
      content: <CompleteStep />,
    },
  ];

  const currentStepData = steps[currentStep];
  const isLastStep = currentStep === steps.length - 1;
  const isFirstStep = currentStep === 0;

  const handleNext = () => {
    if (isLastStep) {
      onComplete();
    } else {
      setCurrentStep((s) => s + 1);
    }
  };

  const handlePrevious = () => {
    if (!isFirstStep) {
      setCurrentStep((s) => s - 1);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-white dark:bg-gray-900 rounded-2xl shadow-2xl max-w-lg w-full mx-4 overflow-hidden">
        {/* Progress */}
        <div className="px-6 pt-6">
          <div className="flex items-center justify-between mb-4">
            {steps.map((step, i) => (
              <div
                key={step.id}
                className={`flex items-center ${i < steps.length - 1 ? 'flex-1' : ''}`}
              >
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                    i < currentStep
                      ? 'bg-emerald-500 text-white'
                      : i === currentStep
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-200 dark:bg-gray-700 text-gray-500 dark:text-gray-400'
                  }`}
                >
                  {i < currentStep ? '✓' : step.icon}
                </div>
                {i < steps.length - 1 && (
                  <div
                    className={`flex-1 h-0.5 mx-2 ${
                      i < currentStep
                        ? 'bg-emerald-500'
                        : 'bg-gray-200 dark:bg-gray-700'
                    }`}
                  />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="px-6 min-h-[400px]">
          {currentStepData.content}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-gray-200 dark:border-gray-700 flex items-center justify-between">
          <div>
            {!isFirstStep && (
              <button
                onClick={handlePrevious}
                className="px-4 py-2 text-sm font-medium text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100"
              >
                ← Back
              </button>
            )}
          </div>

          <div className="flex items-center gap-3">
            {onSkip && !isLastStep && (
              <button
                onClick={onSkip}
                className="px-4 py-2 text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
              >
                Skip for now
              </button>
            )}
            <button
              onClick={handleNext}
              className="px-6 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg"
            >
              {isLastStep ? 'Get Started' : 'Continue'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
