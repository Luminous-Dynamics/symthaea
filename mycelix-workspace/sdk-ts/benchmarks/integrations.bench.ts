/**
 * @mycelix/sdk Integration Module Benchmarks
 *
 * Performance benchmarks for hApp-specific integration adapters
 */

import { bench, describe } from 'vitest';
import { MailTrustService } from '../src/integrations/mail/index.js';
import { MarketplaceReputationService } from '../src/integrations/marketplace/index.js';
import { EduNetCredentialService } from '../src/integrations/edunet/index.js';
import { SupplyChainProvenanceService } from '../src/integrations/supplychain/index.js';

// ============================================================================
// Mail Integration Benchmarks
// ============================================================================

describe('Mail Integration Performance', () => {
  const mailService = new MailTrustService();

  bench('recordInteraction - single', () => {
    mailService.recordInteraction(`sender-${Math.random()}@example.com`, true);
  });

  bench('recordInteraction - existing sender update', () => {
    mailService.recordInteraction('recurring-sender@example.com', Math.random() > 0.5);
  });

  bench('getSenderTrust - existing sender', () => {
    // Pre-record interactions
    const sender = 'benchmark-sender@example.com';
    for (let i = 0; i < 5; i++) {
      mailService.recordInteraction(sender, true);
    }
    return () => {
      mailService.getSenderTrust(sender);
    };
  });

  bench('getSenderTrust - unknown sender', () => {
    mailService.getSenderTrust(`unknown-${Math.random()}@example.com`);
  });

  bench('createEmailClaim - with DKIM verification', () => {
    mailService.createEmailClaim({
      id: `email-${Math.random()}`,
      subject: 'Benchmark Test Email',
      from: 'sender@example.com',
      to: ['recipient@example.com'],
      body: 'Test body content',
      timestamp: Date.now(),
      verification: {
        dkimVerified: true,
        spfPassed: true,
        dmarcPassed: true,
      },
    });
  });

  bench('isTrusted - check', () => {
    mailService.isTrusted('benchmark-sender@example.com');
  });
});

// ============================================================================
// Marketplace Integration Benchmarks
// ============================================================================

describe('Marketplace Integration Performance', () => {
  const marketplaceService = new MarketplaceReputationService();

  bench('recordTransaction - successful purchase', () => {
    marketplaceService.recordTransaction({
      id: `tx-${Math.random()}`,
      type: 'purchase',
      buyerId: 'buyer-1',
      sellerId: 'seller-1',
      amount: 100,
      currency: 'USD',
      itemId: 'item-1',
      timestamp: Date.now(),
      success: true,
    });
  });

  bench('recordTransaction - failed transaction', () => {
    marketplaceService.recordTransaction({
      id: `tx-${Math.random()}`,
      type: 'purchase',
      buyerId: 'buyer-2',
      sellerId: 'seller-2',
      amount: 50,
      currency: 'USD',
      itemId: 'item-2',
      timestamp: Date.now(),
      success: false,
    });
  });

  bench('getSellerProfile - existing seller', () => {
    // Pre-record transactions
    const sellerId = 'benchmark-seller';
    for (let i = 0; i < 10; i++) {
      marketplaceService.recordTransaction({
        id: `tx-setup-${i}`,
        type: 'sale',
        buyerId: `buyer-${i}`,
        sellerId,
        amount: 100,
        currency: 'USD',
        itemId: `item-${i}`,
        timestamp: Date.now(),
        success: true,
      });
    }
    return () => {
      marketplaceService.getSellerProfile(sellerId);
    };
  });

  bench('getBuyerProfile - existing buyer', () => {
    marketplaceService.getBuyerProfile('buyer-1');
  });

  bench('verifyListing - trusted seller', () => {
    marketplaceService.verifyListing('listing-1', 'benchmark-seller');
  });

  bench('isSellerTrustworthy - check', () => {
    marketplaceService.isSellerTrustworthy('benchmark-seller');
  });
});

// ============================================================================
// EduNet Integration Benchmarks
// ============================================================================

describe('EduNet Integration Performance', () => {
  const eduNetService = new EduNetCredentialService();

  bench('issueCertificate - course completion', () => {
    eduNetService.issueCertificate({
      studentId: `student-${Math.random()}`,
      courseId: 'nix-101',
      courseName: 'NixOS Fundamentals',
      grade: 90,
    });
  });

  bench('issueSkillCertification - skill cert', () => {
    eduNetService.issueSkillCertification({
      holderId: `dev-${Math.random()}`,
      skillId: 'typescript',
      skillName: 'TypeScript',
      level: 85,
    });
  });

  bench('verifyCredential - valid credential', () => {
    // Pre-issue credential
    const cred = eduNetService.issueCertificate({
      studentId: 'benchmark-student',
      courseId: 'benchmark-course',
      courseName: 'Benchmark Course',
      grade: 95,
    });
    return () => {
      eduNetService.verifyCredential(cred.id);
    };
  });

  bench('getLearnerProfile - existing learner', () => {
    eduNetService.getLearnerProfile('benchmark-student');
  });

  bench('getCredentialsForHolder - multiple credentials', () => {
    // Pre-issue credentials
    const holderId = 'multi-cred-holder';
    for (let i = 0; i < 5; i++) {
      eduNetService.issueCertificate({
        studentId: holderId,
        courseId: `course-${i}`,
        courseName: `Course ${i}`,
        grade: 80 + i,
      });
    }
    return () => {
      eduNetService.getCredentialsForHolder(holderId);
    };
  });

  bench('isLearnerTrusted - check', () => {
    eduNetService.isLearnerTrusted('benchmark-student');
  });
});

// ============================================================================
// SupplyChain Integration Benchmarks
// ============================================================================

describe('SupplyChain Integration Performance', () => {
  const supplyChainService = new SupplyChainProvenanceService();

  bench('recordCheckpoint - basic', () => {
    supplyChainService.recordCheckpoint({
      productId: `product-${Math.random()}`,
      location: 'Warehouse A',
      handler: 'handler-1',
      action: 'received',
      evidence: [
        { type: 'manual_entry', data: {}, timestamp: Date.now(), verified: true },
      ],
    });
  });

  bench('recordCheckpoint - with IoT evidence', () => {
    supplyChainService.recordCheckpoint({
      productId: `product-iot-${Math.random()}`,
      location: 'Processing Facility',
      coordinates: { lat: 37.7749, lng: -122.4194 },
      handler: 'handler-2',
      action: 'processed',
      evidence: [
        { type: 'iot_sensor', data: { temp: 4.5, humidity: 45 }, timestamp: Date.now(), verified: true },
        { type: 'gps', data: { lat: 37.7749, lng: -122.4194 }, timestamp: Date.now(), verified: true },
        { type: 'photo', data: { url: 'photo.jpg' }, timestamp: Date.now(), verified: true },
      ],
    });
  });

  bench('recordCheckpoint - chain continuation', () => {
    // Record to same product (chain grows)
    supplyChainService.recordCheckpoint({
      productId: 'benchmark-product',
      location: `Location-${Math.random()}`,
      handler: 'handler-chain',
      action: 'shipped',
      evidence: [],
    });
  });

  bench('getProvenanceChain - existing product', () => {
    // Pre-record checkpoints
    const productId = 'traced-benchmark';
    for (let i = 0; i < 5; i++) {
      supplyChainService.recordCheckpoint({
        productId,
        location: `Location ${i}`,
        handler: `handler-${i}`,
        action: 'received',
        evidence: [{ type: 'iot_sensor', data: {}, timestamp: Date.now(), verified: true }],
      });
    }
    return () => {
      supplyChainService.getProvenanceChain(productId);
    };
  });

  bench('verifyChain - complete verification', () => {
    supplyChainService.verifyChain('traced-benchmark');
  });

  bench('getHandlerProfile - existing handler', () => {
    supplyChainService.getHandlerProfile('handler-chain');
  });

  bench('isHandlerTrusted - check', () => {
    supplyChainService.isHandlerTrusted('handler-chain');
  });
});

// ============================================================================
// Cross-Service Benchmarks
// ============================================================================

describe('Cross-Service Integration Performance', () => {
  const mailService = new MailTrustService();
  const marketplaceService = new MarketplaceReputationService();
  const supplyChainService = new SupplyChainProvenanceService();

  bench('multi-service reputation update', () => {
    const agentId = `agent-${Math.random()}`;

    // Update reputation across multiple services
    mailService.recordInteraction(agentId, true);
    marketplaceService.recordTransaction({
      id: `tx-${Math.random()}`,
      type: 'purchase',
      buyerId: agentId,
      sellerId: 'seller-multi',
      amount: 50,
      currency: 'USD',
      itemId: 'item-multi',
      timestamp: Date.now(),
      success: true,
    });
    supplyChainService.recordCheckpoint({
      productId: `product-${Math.random()}`,
      location: 'Location',
      handler: agentId,
      action: 'received',
      evidence: [],
    });
  });

  bench('multi-service trust query', () => {
    const agentId = 'multi-trusted-agent';

    // Pre-populate
    mailService.recordInteraction(agentId, true);
    marketplaceService.recordTransaction({
      id: 'tx-setup',
      type: 'sale',
      buyerId: 'buyer',
      sellerId: agentId,
      amount: 100,
      currency: 'USD',
      itemId: 'item',
      timestamp: Date.now(),
      success: true,
    });

    return () => {
      mailService.getSenderTrust(agentId);
      marketplaceService.getSellerProfile(agentId);
      supplyChainService.getHandlerProfile(agentId);
    };
  });

  bench('complete supply chain with marketplace listing', () => {
    const productId = `product-complete-${Math.random()}`;
    const sellerId = 'full-chain-seller';

    // Record provenance
    supplyChainService.recordCheckpoint({
      productId,
      location: 'Origin',
      handler: sellerId,
      action: 'shipped',
      evidence: [{ type: 'blockchain', data: {}, timestamp: Date.now(), verified: true }],
    });

    // Verify listing
    marketplaceService.verifyListing(productId, sellerId);
  });
});
