// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { AppClient } from "@holochain/client";
import {
  KnowledgeService,
  CreateClaimInput,
  EpistemicPosition,
  ClaimSubmissionResult,
} from "@mycelix/knowledge-client";

/**
 * Input for creating a media-backed knowledge claim.
 */
export interface MediaClaimInput {
  /** Publication identifier in the media hApp (e.g., entry hash or slug). */
  publicationId: string;
  /** Human-readable title of the article or content. */
  title: string;
  /** Core claim text or headline to be tracked in the knowledge graph. */
  claimText: string;
  /** Author DID (from mycelix-identity). */
  authorDid: string;
  /** Optional domain/category (e.g., "journalism", "climate", "governance"). */
  domain?: string;
  /** Optional topics/tags describing the claim. */
  topics?: string[];
  /**
   * Optional explicit epistemic position. If omitted, a conservative default
   * is used that treats media publications as moderately empirical, with
   * network-level normative weight and persistent mythic value.
   */
  classification?: EpistemicPosition;
}

/**
 * Result of linking a media publication to the knowledge graph.
 */
export interface MediaClaimSubmissionResult extends ClaimSubmissionResult {
  /** Publication identifier the claim is associated with. */
  publicationId: string;
}

/**
 * Bridge between Mycelix Media and Mycelix Knowledge.
 *
 * This helper uses the KnowledgeService to create epistemic claims for
 * media publications, so that fact-checking and belief propagation can
 * treat published content as first-class knowledge graph nodes.
 */
export class MediaKnowledgeBridge {
  private knowledge: KnowledgeService;
  private mediaRoleName: string;

  /**
   * Create a new bridge instance.
   *
   * @param appClient - Shared Holochain AppClient connected to the conductor.
   * @param options.knowledgeRoleName - Role name for the knowledge DNA (default: "knowledge").
   * @param options.mediaRoleName - Role name for the media DNA (default: "media").
   */
  constructor(
    appClient: AppClient,
    options: {
      knowledgeRoleName?: string;
      mediaRoleName?: string;
    } = {},
  ) {
    const knowledgeRoleName = options.knowledgeRoleName ?? "knowledge";
    this.mediaRoleName = options.mediaRoleName ?? "media";
    this.knowledge = new KnowledgeService(appClient, knowledgeRoleName);
  }

  /**
   * Submit a claim for a media publication into the Mycelix Knowledge Graph.
   *
   * This:
   *  - Builds a CreateClaimInput from publication metadata.
   *  - Calls KnowledgeService.submitAndAnalyzeClaim.
   *  - Returns the result alongside the originating publicationId.
   */
  async submitPublicationClaim(
    input: MediaClaimInput,
  ): Promise<MediaClaimSubmissionResult> {
    const classification: EpistemicPosition =
      input.classification ??
      this.defaultClassification();

    const claimInput: CreateClaimInput = {
      content: input.claimText,
      classification,
      domain: input.domain ?? "media",
      topics: input.topics ?? [],
      sources: [
        {
          uri: `happ://${this.mediaRoleName}/publication/${input.publicationId}`,
          title: input.title,
          author: input.authorDid,
          publishedAt: Date.now(),
          // Initial reliability is neutral; can be updated via fact-checks.
          reliability: 0.5,
        },
      ],
      evidence: [],
    };

    const result = await this.knowledge.submitAndAnalyzeClaim(claimInput);

    return {
      publicationId: input.publicationId,
      ...result,
    };
  }

  /**
   * Conservative default epistemic classification for media claims.
   */
  private defaultClassification(): EpistemicPosition {
    return {
      empirical: 0.6,
      normative: 0.4,
      mythic: 0.3,
    };
  }
}

