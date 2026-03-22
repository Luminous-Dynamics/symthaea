'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Shield, 
  Lock, 
  Award, 
  Users, 
  Globe, 
  CheckCircle2, 
  AlertCircle,
  TrendingUp,
  Activity,
  Eye,
  FileCheck,
  Zap,
  RefreshCw,
  ExternalLink,
  Info
} from 'lucide-react';

interface TrustMetric {
  id: string;
  label: string;
  value: number | string;
  status: 'excellent' | 'good' | 'warning' | 'critical';
  trend: 'up' | 'down' | 'stable';
  description: string;
  details?: string[];
}

interface AuditRecord {
  date: string;
  auditor: string;
  result: 'passed' | 'passed_with_notes' | 'failed';
  report_url: string;
}

export default function TrustIndicatorsDashboard() {
  const [activeCategory, setActiveCategory] = useState<'security' | 'transparency' | 'compliance' | 'community'>('security');
  const [realTimeMetrics, setRealTimeMetrics] = useState({
    uptime: 99.97,
    responseTime: 127,
    activeUsers: 14283,
    transactionVolume: 2847392,
    lastUpdated: new Date()
  });

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setRealTimeMetrics(prev => ({
        uptime: prev.uptime + (Math.random() - 0.5) * 0.01,
        responseTime: Math.max(50, prev.responseTime + (Math.random() - 0.5) * 20),
        activeUsers: Math.floor(prev.activeUsers + (Math.random() - 0.5) * 100),
        transactionVolume: prev.transactionVolume + Math.floor(Math.random() * 10000),
        lastUpdated: new Date()
      }));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const trustMetrics: Record<string, TrustMetric[]> = {
    security: [
      {
        id: 'encryption',
        label: 'Encryption Status',
        value: 'AES-256',
        status: 'excellent',
        trend: 'stable',
        description: 'Military-grade encryption for all data',
        details: [
          'End-to-end encryption for all transactions',
          'Zero-knowledge architecture for sensitive data',
          'Quantum-resistant algorithms implemented'
        ]
      },
      {
        id: 'uptime',
        label: 'Platform Uptime',
        value: `${realTimeMetrics.uptime.toFixed(3)}%`,
        status: realTimeMetrics.uptime > 99.95 ? 'excellent' : 'good',
        trend: 'stable',
        description: 'Last 90 days availability',
        details: [
          'Zero security breaches in 24 months',
          '< 5 minute average recovery time',
          'Redundant infrastructure across 3 regions'
        ]
      },
      {
        id: 'audits',
        label: 'Security Audits',
        value: '12/12 Passed',
        status: 'excellent',
        trend: 'up',
        description: 'External security audits this year',
        details: [
          'Monthly penetration testing',
          'Quarterly code audits by CertiK',
          'Annual SOC 2 Type II certification'
        ]
      },
      {
        id: 'insurance',
        label: 'Insurance Coverage',
        value: '$100M',
        status: 'excellent',
        trend: 'up',
        description: 'Comprehensive cyber insurance',
        details: [
          'Lloyd\'s of London underwriter',
          'Covers all user funds',
          'Additional $50M excess coverage'
        ]
      }
    ],
    transparency: [
      {
        id: 'blockchain',
        label: 'On-Chain Transparency',
        value: '100%',
        status: 'excellent',
        trend: 'stable',
        description: 'All transactions on public blockchain',
        details: [
          'Ethereum mainnet for all transfers',
          'IPFS for document storage',
          'Real-time transaction explorer'
        ]
      },
      {
        id: 'reporting',
        label: 'Financial Reports',
        value: 'Quarterly',
        status: 'excellent',
        trend: 'stable',
        description: 'Published financial statements',
        details: [
          'Audited by PwC Switzerland',
          'Full balance sheet disclosure',
          'Community fund allocation reports'
        ]
      },
      {
        id: 'api_access',
        label: 'Data Access',
        value: 'Open API',
        status: 'excellent',
        trend: 'up',
        description: 'Public API for all platform data',
        details: [
          '1000+ API endpoints available',
          'Real-time WebSocket streams',
          'No rate limiting for public data'
        ]
      },
      {
        id: 'governance',
        label: 'Governance Votes',
        value: '47 Public',
        status: 'excellent',
        trend: 'up',
        description: 'Community governance decisions',
        details: [
          'All votes recorded on-chain',
          'Proposal discussions public',
          '72% average participation rate'
        ]
      }
    ],
    compliance: [
      {
        id: 'regulatory',
        label: 'Regulatory Status',
        value: 'Compliant',
        status: 'excellent',
        trend: 'stable',
        description: 'Full regulatory compliance',
        details: [
          'SEC registered (Form D)',
          'Swiss FINMA approved',
          'EU MiFID II compliant'
        ]
      },
      {
        id: 'kyc',
        label: 'KYC/AML',
        value: '100%',
        status: 'excellent',
        trend: 'stable',
        description: 'Identity verification rate',
        details: [
          'Automated KYC via Jumio',
          'Enhanced due diligence for large investments',
          'Continuous monitoring via Chainalysis'
        ]
      },
      {
        id: 'gdpr',
        label: 'Data Protection',
        value: 'GDPR Certified',
        status: 'excellent',
        trend: 'stable',
        description: 'Privacy compliance certification',
        details: [
          'Right to erasure implemented',
          'Data portability available',
          'Privacy by design architecture'
        ]
      },
      {
        id: 'licenses',
        label: 'Operating Licenses',
        value: '12 Active',
        status: 'excellent',
        trend: 'up',
        description: 'Regulatory licenses worldwide',
        details: [
          'US: Money transmitter licenses',
          'EU: E-money institution',
          'Asia: Payment services licenses'
        ]
      }
    ],
    community: [
      {
        id: 'trust_score',
        label: 'Community Trust',
        value: '94%',
        status: 'excellent',
        trend: 'up',
        description: 'User trust rating',
        details: [
          'Based on 12,847 reviews',
          '4.8/5.0 average rating',
          '89% would recommend'
        ]
      },
      {
        id: 'response_time',
        label: 'Support Response',
        value: '< 2 hours',
        status: 'excellent',
        trend: 'stable',
        description: 'Average first response time',
        details: [
          '24/7 support coverage',
          '15 languages supported',
          '97% resolution rate'
        ]
      },
      {
        id: 'community_fund',
        label: 'Community Fund',
        value: '$47.3M',
        status: 'excellent',
        trend: 'up',
        description: 'Reserved for community projects',
        details: [
          '3% of all revenue allocated',
          'Community votes on allocation',
          '23 projects funded to date'
        ]
      },
      {
        id: 'ownership',
        label: 'Path to Ownership',
        value: 'Year 3 of 7',
        status: 'good',
        trend: 'up',
        description: 'Community ownership transition',
        details: [
          '28% community ownership achieved',
          'On track for 100% by year 7',
          'Governance tokens being distributed'
        ]
      }
    ]
  };

  const auditHistory: AuditRecord[] = [
    {
      date: '2025-01-15',
      auditor: 'CertiK',
      result: 'passed',
      report_url: '#'
    },
    {
      date: '2024-12-10',
      auditor: 'Trail of Bits',
      result: 'passed_with_notes',
      report_url: '#'
    },
    {
      date: '2024-11-05',
      auditor: 'Quantstamp',
      result: 'passed',
      report_url: '#'
    }
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'excellent':
        return 'text-emerald-600 bg-emerald-50 border-emerald-200';
      case 'good':
        return 'text-blue-600 bg-blue-50 border-blue-200';
      case 'warning':
        return 'text-amber-600 bg-amber-50 border-amber-200';
      case 'critical':
        return 'text-red-600 bg-red-50 border-red-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up':
        return <TrendingUp className="w-4 h-4 text-emerald-500" />;
      case 'down':
        return <TrendingUp className="w-4 h-4 text-red-500 rotate-180" />;
      default:
        return <Activity className="w-4 h-4 text-gray-400" />;
    }
  };

  const categories = [
    { id: 'security', label: 'Security', icon: Shield },
    { id: 'transparency', label: 'Transparency', icon: Eye },
    { id: 'compliance', label: 'Compliance', icon: FileCheck },
    { id: 'community', label: 'Community', icon: Users }
  ];

  return (
    <div className="w-full max-w-7xl mx-auto p-6 space-y-6">
      {/* Header with Real-time Status */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-r from-emerald-600 to-teal-600 rounded-3xl p-8 text-white"
      >
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-3xl font-bold mb-2">Trust & Security Dashboard</h2>
            <p className="text-emerald-100">Real-time platform health and transparency metrics</p>
          </div>
          <div className="flex items-center gap-2 px-4 py-2 bg-white/20 rounded-xl backdrop-blur">
            <Zap className="w-5 h-5 text-yellow-300 animate-pulse" />
            <span className="font-semibold">Live</span>
          </div>
        </div>

        {/* Real-time Metrics Bar */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-white/10 backdrop-blur rounded-xl p-4">
            <div className="flex items-center gap-2 mb-2">
              <Activity className="w-4 h-4 text-emerald-300" />
              <span className="text-sm text-emerald-200">Uptime</span>
            </div>
            <div className="text-2xl font-bold">{realTimeMetrics.uptime.toFixed(3)}%</div>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-xl p-4">
            <div className="flex items-center gap-2 mb-2">
              <Zap className="w-4 h-4 text-emerald-300" />
              <span className="text-sm text-emerald-200">Response</span>
            </div>
            <div className="text-2xl font-bold">{realTimeMetrics.responseTime}ms</div>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-xl p-4">
            <div className="flex items-center gap-2 mb-2">
              <Users className="w-4 h-4 text-emerald-300" />
              <span className="text-sm text-emerald-200">Active Users</span>
            </div>
            <div className="text-2xl font-bold">{realTimeMetrics.activeUsers.toLocaleString()}</div>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-xl p-4">
            <div className="flex items-center gap-2 mb-2">
              <Globe className="w-4 h-4 text-emerald-300" />
              <span className="text-sm text-emerald-200">24h Volume</span>
            </div>
            <div className="text-2xl font-bold">${(realTimeMetrics.transactionVolume / 1000000).toFixed(1)}M</div>
          </div>
        </div>

        <div className="mt-4 text-xs text-emerald-200">
          Last updated: {realTimeMetrics.lastUpdated.toLocaleTimeString()}
        </div>
      </motion.div>

      {/* Category Tabs */}
      <div className="flex gap-2 overflow-x-auto">
        {categories.map((category) => {
          const Icon = category.icon;
          return (
            <button
              key={category.id}
              onClick={() => setActiveCategory(category.id as any)}
              className={`
                flex items-center gap-2 px-6 py-3 rounded-xl font-semibold transition-all whitespace-nowrap
                ${activeCategory === category.id
                  ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-600/20'
                  : 'bg-white text-gray-600 hover:bg-gray-50 border border-gray-200'
                }
              `}
            >
              <Icon className="w-5 h-5" />
              <span>{category.label}</span>
            </button>
          );
        })}
      </div>

      {/* Metrics Grid */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeCategory}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.3 }}
          className="grid md:grid-cols-2 gap-6"
        >
          {trustMetrics[activeCategory].map((metric, index) => (
            <motion.div
              key={metric.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="bg-white rounded-2xl border border-gray-200 shadow-lg overflow-hidden"
            >
              <div className={`p-6 border-l-4 ${getStatusColor(metric.status)}`}>
                <div className="flex items-start justify-between mb-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <h3 className="text-lg font-bold text-gray-900">{metric.label}</h3>
                      {getTrendIcon(metric.trend)}
                    </div>
                    <div className="text-2xl font-bold text-gray-900 mb-2">{metric.value}</div>
                    <p className="text-sm text-gray-600">{metric.description}</p>
                  </div>
                  <div className={`p-2 rounded-lg ${getStatusColor(metric.status)}`}>
                    {metric.status === 'excellent' && <CheckCircle2 className="w-5 h-5" />}
                    {metric.status === 'good' && <CheckCircle2 className="w-5 h-5" />}
                    {metric.status === 'warning' && <AlertCircle className="w-5 h-5" />}
                    {metric.status === 'critical' && <AlertCircle className="w-5 h-5" />}
                  </div>
                </div>

                {metric.details && (
                  <div className="pt-4 border-t border-gray-200">
                    <ul className="space-y-2">
                      {metric.details.map((detail, idx) => (
                        <li key={idx} className="flex gap-2 text-sm text-gray-600">
                          <CheckCircle2 className="w-4 h-4 text-emerald-500 flex-shrink-0 mt-0.5" />
                          <span>{detail}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </motion.div>
      </AnimatePresence>

      {/* Audit History */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="bg-white rounded-2xl border border-gray-200 shadow-lg p-6"
      >
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-xl font-bold text-gray-900">Recent Security Audits</h3>
          <Award className="w-6 h-6 text-emerald-600" />
        </div>

        <div className="space-y-3">
          {auditHistory.map((audit, index) => (
            <div
              key={index}
              className="flex items-center justify-between p-4 bg-gray-50 rounded-xl hover:bg-gray-100 transition-colors"
            >
              <div className="flex items-center gap-4">
                <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                  audit.result === 'passed' ? 'bg-emerald-100' :
                  audit.result === 'passed_with_notes' ? 'bg-blue-100' : 'bg-red-100'
                }`}>
                  {audit.result === 'passed' && <CheckCircle2 className="w-5 h-5 text-emerald-600" />}
                  {audit.result === 'passed_with_notes' && <Info className="w-5 h-5 text-blue-600" />}
                  {audit.result === 'failed' && <AlertCircle className="w-5 h-5 text-red-600" />}
                </div>
                <div>
                  <div className="font-semibold text-gray-900">{audit.auditor}</div>
                  <div className="text-sm text-gray-600">{new Date(audit.date).toLocaleDateString()}</div>
                </div>
              </div>
              <a
                href={audit.report_url}
                className="flex items-center gap-2 px-4 py-2 text-emerald-600 hover:bg-emerald-50 rounded-lg transition-colors"
              >
                <span className="text-sm font-semibold">View Report</span>
                <ExternalLink className="w-4 h-4" />
              </a>
            </div>
          ))}
        </div>

        <button className="mt-6 w-full py-3 bg-emerald-50 text-emerald-600 rounded-xl font-semibold hover:bg-emerald-100 transition-colors">
          View All Audit History →
        </button>
      </motion.div>

      {/* Trust Score Summary */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="bg-gradient-to-r from-emerald-50 to-teal-50 rounded-2xl border border-emerald-200 p-8"
      >
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-2xl font-bold text-gray-900 mb-2">Overall Trust Score</h3>
            <p className="text-gray-600">
              Based on 47 metrics across security, transparency, compliance, and community engagement
            </p>
          </div>
          <div className="text-center">
            <div className="text-5xl font-bold text-emerald-600 mb-2">96/100</div>
            <div className="flex items-center gap-1">
              {[...Array(5)].map((_, i) => (
                <Award
                  key={i}
                  className={`w-5 h-5 ${i < 5 ? 'text-amber-400 fill-amber-400' : 'text-gray-300'}`}
                />
              ))}
            </div>
          </div>
        </div>

        <div className="mt-6 flex gap-4">
          <button className="flex-1 py-3 bg-emerald-600 text-white rounded-xl font-semibold hover:bg-emerald-700 transition-colors">
            Download Trust Report
          </button>
          <button className="flex-1 py-3 bg-white text-emerald-600 border border-emerald-300 rounded-xl font-semibold hover:bg-emerald-50 transition-colors">
            Subscribe to Updates
          </button>
        </div>
      </motion.div>
    </div>
  );
}