// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Economic Strategy Configuration Wizard
 *
 * Guides artists through choosing and customizing their economic model
 */

import React, { useState } from 'react';
import { EconomicStrategySDK, PaymentModel, PRESET_STRATEGIES, EconomicConfig, Split } from '@/lib';
import { useWallet } from '@/hooks/useWallet';

// ========== TYPES ==========

interface WizardStep {
  title: string;
  description: string;
  component: React.ComponentType<any>;
}

interface PresetCard {
  id: string;
  name: string;
  description: string;
  icon: string;
  bestFor: string[];
  config: Partial<EconomicConfig>;
}

// ========== MAIN COMPONENT ==========

export function EconomicStrategyWizard({ songId, onComplete, isLoading }: {
  songId: string;
  onComplete: (config: EconomicConfig) => void;
  isLoading?: boolean;
}) {
  const [currentStep, setCurrentStep] = useState(0);
  const [config, setConfig] = useState<Partial<EconomicConfig>>({});
  const wallet = useWallet();

  const steps: WizardStep[] = [
    {
      title: 'Choose Your Economic Model',
      description: 'Start with a preset or build custom',
      component: PresetSelector,
    },
    {
      title: 'Configure Payment Model',
      description: 'How should listeners pay?',
      component: PaymentModelConfig,
    },
    {
      title: 'Set Revenue Splits',
      description: 'How should earnings be distributed?',
      component: RevenueSplitConfig,
    },
    {
      title: 'Add Incentives',
      description: 'Reward your community',
      component: IncentivesConfig,
    },
    {
      title: 'Review & Deploy',
      description: 'Confirm your economic operating system',
      component: ReviewAndDeploy,
    },
  ];

  const CurrentStepComponent = steps[currentStep].component;

  async function handleComplete() {
    if (!wallet.connected) {
      alert('Please connect wallet');
      return;
    }

    // Validate config is complete
    if (!config.strategyId || !config.paymentModel || !config.distributionSplits) {
      alert('Please complete all configuration steps');
      return;
    }

    // Pass config to parent - parent handles blockchain registration
    // This allows the upload page to handle the full flow including IPFS upload
    onComplete(config as EconomicConfig);
  }

  return (
    <div className="max-w-4xl mx-auto p-8">
      {/* Progress Bar */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-2">
          {steps.map((step, index) => (
            <div
              key={index}
              className={`flex-1 ${
                index === currentStep
                  ? 'text-blue-600 font-semibold'
                  : index < currentStep
                  ? 'text-green-600'
                  : 'text-gray-400'
              }`}
            >
              <div className="text-sm">{step.title}</div>
            </div>
          ))}
        </div>
        <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-600 transition-all"
            style={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
          />
        </div>
      </div>

      {/* Current Step Content */}
      <div className="bg-white rounded-lg shadow-lg p-8">
        <h2 className="text-2xl font-bold mb-2">{steps[currentStep].title}</h2>
        <p className="text-gray-600 mb-6">{steps[currentStep].description}</p>

        <CurrentStepComponent config={config} setConfig={setConfig} />
      </div>

      {/* Navigation */}
      <div className="flex justify-between mt-6">
        <button
          onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
          disabled={currentStep === 0}
          className="px-6 py-2 border rounded disabled:opacity-50"
        >
          Back
        </button>

        {currentStep < steps.length - 1 ? (
          <button
            onClick={() => setCurrentStep(currentStep + 1)}
            className="px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Next
          </button>
        ) : (
          <button
            onClick={handleComplete}
            className="px-6 py-2 bg-green-600 text-white rounded hover:bg-green-700"
          >
            Deploy Strategy
          </button>
        )}
      </div>
    </div>
  );
}

// ========== STEP 1: PRESET SELECTOR ==========

function PresetSelector({ config, setConfig }: {
  config: Partial<EconomicConfig>;
  setConfig: (config: Partial<EconomicConfig>) => void;
}) {
  const presets: PresetCard[] = [
    {
      id: 'independent',
      name: 'Independent Artist',
      description: 'Pure market model - you keep what you earn',
      icon: '🎸',
      bestFor: ['Solo artists', 'Viral potential', 'Established fanbase'],
      config: PRESET_STRATEGIES.independentArtist,
    },
    {
      id: 'collective',
      name: 'Community Collective',
      description: 'Free music supported by gifts and patronage',
      icon: '🤝',
      bestFor: ['Community-focused', 'Experimental music', 'Local scenes'],
      config: PRESET_STRATEGIES.communityCollective,
    },
    {
      id: 'collaborative',
      name: 'Collaborative Split',
      description: 'Multi-artist project with custom revenue shares',
      icon: '👥',
      bestFor: ['Bands', 'Producer collabs', 'Featured artists'],
      config: PRESET_STRATEGIES.collaborativeSplit,
    },
  ];

  return (
    <div className="space-y-4">
      {presets.map((preset) => (
        <button
          key={preset.id}
          onClick={() => setConfig(preset.config)}
          className={`w-full p-6 border-2 rounded-lg text-left hover:border-blue-500 transition ${
            config.strategyId === preset.config.strategyId
              ? 'border-blue-500 bg-blue-50'
              : 'border-gray-200'
          }`}
        >
          <div className="flex items-start gap-4">
            <div className="text-4xl">{preset.icon}</div>
            <div className="flex-1">
              <h3 className="text-xl font-semibold mb-1">{preset.name}</h3>
              <p className="text-gray-600 mb-3">{preset.description}</p>
              <div className="flex flex-wrap gap-2">
                {preset.bestFor.map((tag) => (
                  <span
                    key={tag}
                    className="px-3 py-1 bg-gray-100 text-sm rounded-full"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </button>
      ))}

      <button
        onClick={() => setConfig({ strategyId: 'custom' })}
        className="w-full p-6 border-2 border-dashed border-gray-300 rounded-lg hover:border-blue-500 transition"
      >
        <div className="text-center">
          <div className="text-4xl mb-2">⚙️</div>
          <h3 className="text-xl font-semibold mb-1">Start from Scratch</h3>
          <p className="text-gray-600">Build a custom economic model</p>
        </div>
      </button>
    </div>
  );
}

// ========== STEP 2: PAYMENT MODEL CONFIG ==========

function PaymentModelConfig({ config, setConfig }: {
  config: Partial<EconomicConfig>;
  setConfig: (config: Partial<EconomicConfig>) => void;
}) {
  const models = [
    {
      value: PaymentModel.PAY_PER_STREAM,
      name: 'Pay Per Stream',
      description: 'Listeners pay $0.01 each time they play your song',
      icon: '▶️',
      example: '1000 streams = $10 earned',
    },
    {
      value: PaymentModel.GIFT_ECONOMY,
      name: 'Gift Economy',
      description: 'Listening is free, but listeners can gift you CGC tokens',
      icon: '🎁',
      example: 'Build reputation and community support',
    },
    {
      value: PaymentModel.PATRONAGE,
      name: 'Patronage',
      description: 'Fans subscribe monthly to support you',
      icon: '💝',
      example: '100 patrons × $5/month = $500/month',
    },
    {
      value: PaymentModel.PAY_WHAT_YOU_WANT,
      name: 'Pay What You Want',
      description: 'Listener chooses the amount (minimum optional)',
      icon: '🎯',
      example: 'Let fans decide your worth',
    },
  ];

  return (
    <div className="space-y-4">
      {models.map((model) => (
        <div
          key={model.value}
          className={`p-4 border-2 rounded-lg cursor-pointer transition ${
            config.paymentModel === model.value
              ? 'border-blue-500 bg-blue-50'
              : 'border-gray-200 hover:border-blue-300'
          }`}
          onClick={() => setConfig({ ...config, paymentModel: model.value })}
        >
          <div className="flex items-start gap-3">
            <div className="text-3xl">{model.icon}</div>
            <div className="flex-1">
              <h4 className="font-semibold mb-1">{model.name}</h4>
              <p className="text-gray-600 text-sm mb-2">{model.description}</p>
              <p className="text-blue-600 text-sm font-medium">{model.example}</p>
            </div>
            {config.paymentModel === model.value && (
              <div className="text-blue-600">✓</div>
            )}
          </div>
        </div>
      ))}

      {/* Additional Config for Selected Model */}
      {config.paymentModel === PaymentModel.PAY_PER_STREAM && (
        <div className="mt-4 p-4 bg-gray-50 rounded-lg">
          <label className="block text-sm font-medium mb-2">
            Price per stream (FLOW)
          </label>
          <input
            type="number"
            step="0.001"
            defaultValue="0.01"
            className="w-full px-3 py-2 border rounded"
            onChange={(e) =>
              setConfig({
                ...config,
                minimumPayment: parseFloat(e.target.value),
              })
            }
          />
        </div>
      )}

      {config.paymentModel === PaymentModel.GIFT_ECONOMY && (
        <div className="mt-4 p-4 bg-gray-50 rounded-lg">
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              defaultChecked
              onChange={(e) =>
                setConfig({ ...config, acceptsGifts: e.target.checked })
              }
            />
            <span className="text-sm">Accept voluntary tips (recommended)</span>
          </label>
        </div>
      )}
    </div>
  );
}

// ========== STEP 3: REVENUE SPLIT CONFIG ==========

function RevenueSplitConfig({ config, setConfig }: {
  config: Partial<EconomicConfig>;
  setConfig: (config: Partial<EconomicConfig>) => void;
}) {
  const [splits, setSplits] = useState<Split[]>(
    config.distributionSplits || [
      { recipient: '', basisPoints: 9500, role: 'artist' },
      { recipient: '', basisPoints: 500, role: 'protocol' },
    ]
  );

  const totalBasisPoints = splits.reduce((sum, s) => sum + s.basisPoints, 0);
  const isValid = totalBasisPoints === 10000;

  function addSplit() {
    setSplits([...splits, { recipient: '', basisPoints: 0, role: '' }]);
  }

  function removeSplit(index: number) {
    setSplits(splits.filter((_, i) => i !== index));
  }

  function updateSplit(index: number, field: keyof Split, value: any) {
    const newSplits = [...splits];
    newSplits[index] = { ...newSplits[index], [field]: value };
    setSplits(newSplits);
    setConfig({ ...config, distributionSplits: newSplits });
  }

  return (
    <div className="space-y-4">
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
        <p className="text-sm text-blue-800">
          <strong>Tip:</strong> Splits must add up to 100%. The protocol fee (0.5-1%)
          helps maintain the network and is required.
        </p>
      </div>

      {splits.map((split, index) => (
        <div key={index} className="flex gap-2 items-start p-4 border rounded-lg">
          <div className="flex-1 grid grid-cols-3 gap-2">
            <input
              type="text"
              placeholder="Wallet address or DID"
              value={split.recipient}
              onChange={(e) => updateSplit(index, 'recipient', e.target.value)}
              className="px-3 py-2 border rounded text-sm"
            />
            <input
              type="text"
              placeholder="Role (artist, producer...)"
              value={split.role}
              onChange={(e) => updateSplit(index, 'role', e.target.value)}
              className="px-3 py-2 border rounded text-sm"
            />
            <div className="flex gap-1 items-center">
              <input
                type="number"
                placeholder="Percentage"
                value={split.basisPoints / 100}
                onChange={(e) =>
                  updateSplit(index, 'basisPoints', parseFloat(e.target.value) * 100)
                }
                className="flex-1 px-3 py-2 border rounded text-sm"
              />
              <span className="text-gray-500 text-sm">%</span>
            </div>
          </div>
          {index > 0 && (
            <button
              onClick={() => removeSplit(index)}
              className="px-3 py-2 text-red-600 hover:bg-red-50 rounded"
            >
              ✕
            </button>
          )}
        </div>
      ))}

      <button
        onClick={addSplit}
        className="w-full py-2 border-2 border-dashed border-gray-300 rounded-lg hover:border-blue-500 transition"
      >
        + Add Another Recipient
      </button>

      {/* Validation */}
      <div className={`p-4 rounded-lg ${isValid ? 'bg-green-50' : 'bg-red-50'}`}>
        <div className="flex items-center justify-between">
          <span className="font-medium">
            Total: {(totalBasisPoints / 100).toFixed(1)}%
          </span>
          {isValid ? (
            <span className="text-green-600">✓ Valid</span>
          ) : (
            <span className="text-red-600">Must equal 100%</span>
          )}
        </div>
      </div>

      {/* Visualization */}
      <div className="mt-6">
        <h4 className="font-medium mb-2">Preview: If you earn $1000</h4>
        <div className="space-y-2">
          {splits.map((split, index) => (
            <div key={index} className="flex items-center gap-2">
              <div
                className="h-8 bg-blue-500 rounded flex items-center justify-center text-white text-sm"
                style={{ width: `${split.basisPoints / 100}%` }}
              >
                ${((1000 * split.basisPoints) / 10000).toFixed(2)}
              </div>
              <span className="text-sm text-gray-600">{split.role}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ========== STEP 4: INCENTIVES CONFIG ==========

function IncentivesConfig({ config, setConfig }: {
  config: Partial<EconomicConfig>;
  setConfig: (config: Partial<EconomicConfig>) => void;
}) {
  return (
    <div className="space-y-6">
      <div>
        <h4 className="font-semibold mb-3">Listener Rewards</h4>
        <div className="space-y-2">
          <label className="flex items-start gap-3 p-3 border rounded hover:bg-gray-50">
            <input type="checkbox" className="mt-1" />
            <div className="flex-1">
              <div className="font-medium">Early Listener Bonus</div>
              <div className="text-sm text-gray-600">
                First 100 listeners get 10 CGC tokens
              </div>
            </div>
          </label>

          <label className="flex items-start gap-3 p-3 border rounded hover:bg-gray-50">
            <input type="checkbox" className="mt-1" />
            <div className="flex-1">
              <div className="font-medium">Share Bonus</div>
              <div className="text-sm text-gray-600">
                Earn 0.001 FLOW for each new listener you bring
              </div>
            </div>
          </label>

          <label className="flex items-start gap-3 p-3 border rounded hover:bg-gray-50">
            <input type="checkbox" className="mt-1" />
            <div className="flex-1">
              <div className="font-medium">Repeat Listener Bonus</div>
              <div className="text-sm text-gray-600">
                Loyal fans earn 1.5x CGC for repeat plays
              </div>
            </div>
          </label>
        </div>
      </div>

      <div>
        <h4 className="font-semibold mb-3">Creator Incentives</h4>
        <div className="space-y-2">
          <label className="flex items-start gap-3 p-3 border rounded hover:bg-gray-50">
            <input type="checkbox" className="mt-1" />
            <div className="flex-1">
              <div className="font-medium">Quality Bonus</div>
              <div className="text-sm text-gray-600">
                Songs rated 4.5+ stars earn 10% more
              </div>
            </div>
          </label>

          <label className="flex items-start gap-3 p-3 border rounded hover:bg-gray-50">
            <input type="checkbox" className="mt-1" />
            <div className="flex-1">
              <div className="font-medium">Collaboration Bonus</div>
              <div className="text-sm text-gray-600">
                Multi-artist songs earn 20% more
              </div>
            </div>
          </label>
        </div>
      </div>

      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
        <p className="text-sm text-yellow-800">
          <strong>Note:</strong> Incentives are funded by the DAO treasury and may
          change based on network economics. You can enable/disable these later via
          governance.
        </p>
      </div>
    </div>
  );
}

// ========== STEP 5: REVIEW AND DEPLOY ==========

function ReviewAndDeploy({ config }: { config: Partial<EconomicConfig> }) {
  return (
    <div className="space-y-6">
      <div className="bg-gray-50 rounded-lg p-6 space-y-4">
        <div>
          <h4 className="font-medium text-gray-700 mb-1">Payment Model</h4>
          <p className="text-lg">{config.paymentModel?.replace(/_/g, ' ').toUpperCase()}</p>
        </div>

        <div>
          <h4 className="font-medium text-gray-700 mb-1">Revenue Distribution</h4>
          <div className="space-y-1">
            {config.distributionSplits?.map((split, i) => (
              <div key={i} className="flex justify-between text-sm">
                <span>{split.role}</span>
                <span className="font-medium">{split.basisPoints / 100}%</span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h4 className="font-medium text-gray-700 mb-1">Strategy ID</h4>
          <code className="text-sm bg-white px-2 py-1 rounded">{config.strategyId}</code>
        </div>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h4 className="font-medium text-blue-800 mb-2">What happens next?</h4>
        <ol className="text-sm text-blue-700 space-y-1 list-decimal list-inside">
          <li>Your economic strategy will be deployed on-chain</li>
          <li>Listeners can immediately start paying according to your model</li>
          <li>You'll receive instant payments to your wallet</li>
          <li>You can update this strategy later via governance</li>
        </ol>
      </div>

      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
        <p className="text-sm text-yellow-800">
          <strong>Important:</strong> This will cost a small gas fee (~$0.50 on Gnosis Chain).
          Make sure you have enough XDAI in your wallet.
        </p>
      </div>
    </div>
  );
}
