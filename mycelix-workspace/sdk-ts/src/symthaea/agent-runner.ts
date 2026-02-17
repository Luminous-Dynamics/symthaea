/**
 * Symthaea Agent Runner
 *
 * Core runtime for civic AI agents. Handles conversation flow,
 * knowledge retrieval, intent classification, and human escalation.
 *
 * Philosophy: AI should empower, not replace. When in doubt, connect
 * citizens with human experts. Transparency over automation.
 *
 * @module symthaea/agent-runner
 */

import type {
  AgentConfig,
  AgentResponse,
  AgentStats,
  ConversationContext,
  ConversationTurn,
  CivicAgentDomain,
  EscalationRequest,
  ExtractedEntity,
  Intent,
  KnowledgeSource,
  SuggestedAction,
} from './types.js';

/**
 * Default agent configuration
 */
export const DEFAULT_AGENT_CONFIG: Omit<AgentConfig, 'id' | 'name' | 'domain'> = {
  capabilityLevel: 'navigational',
  model: {
    provider: 'ollama',
    model: 'llama3.2:3b',
    temperature: 0.3,
    maxTokens: 500,
    systemPrompt: '', // Set per-domain
  },
  knowledgeBase: {
    enabled: true,
    sources: ['agency_faqs', 'forms_library'],
    maxSourcesPerResponse: 3,
    minRelevance: 0.6,
    dkgEnabled: false,
  },
  escalation: {
    enabled: true,
    confidenceThreshold: 0.5,
    alwaysEscalateTopics: ['legal_advice', 'discrimination', 'emergency', 'fraud'],
    maxTurnsBeforeEscalation: 10,
    channels: [
      {
        type: 'phone',
        value: '311',
        availability: '24/7',
        avgWaitTime: '5 minutes',
      },
    ],
  },
  personality: {
    tone: 'friendly',
    readingLevel: 'simple',
    useEmojis: false,
    languages: ['en'],
    greetingStyle: 'brief',
  },
  rateLimit: {
    requestsPerMinute: 10,
    requestsPerHour: 100,
    maxConversationLength: 50,
  },
};

/**
 * Agent Runner
 *
 * Executes agent conversations with knowledge retrieval and escalation.
 */
export class AgentRunner {
  private config: AgentConfig;
  private conversations: Map<string, ConversationContext> = new Map();
  private escalationQueue: EscalationRequest[] = [];
  private stats: AgentStats;
  private knowledgeRetriever: KnowledgeRetriever | null = null;
  private modelClient: ModelClient | null = null;

  constructor(config: AgentConfig) {
    this.config = config;
    this.stats = this.initStats();
  }

  /**
   * Initialize or get a conversation
   */
  getOrCreateConversation(
    conversationId: string,
    channel: ConversationContext['channel'],
    citizenDid?: string
  ): ConversationContext {
    let context = this.conversations.get(conversationId);
    if (!context) {
      context = {
        conversationId,
        citizenDid,
        channel,
        history: [],
        entities: [],
        domain: this.config.domain,
        metadata: {},
        startedAt: new Date(),
        lastActivityAt: new Date(),
      };
      this.conversations.set(conversationId, context);
      this.stats.totalConversations++;
    }
    return context;
  }

  /**
   * Process a citizen message and generate a response
   */
  async processMessage(
    conversationId: string,
    message: string,
    channel: ConversationContext['channel'],
    citizenDid?: string
  ): Promise<AgentResponse> {
    const startTime = Date.now();
    const context = this.getOrCreateConversation(conversationId, channel, citizenDid);

    // Check rate limits
    if (context.history.length >= this.config.rateLimit.maxConversationLength) {
      return this.createEscalationResponse(
        context,
        'Conversation length limit reached. Let me connect you with a human.'
      );
    }

    // Add citizen turn
    const citizenTurn: ConversationTurn = {
      id: `turn-${Date.now()}`,
      role: 'citizen',
      content: message,
      timestamp: new Date(),
    };
    context.history.push(citizenTurn);
    context.lastActivityAt = new Date();

    // Extract entities from message
    const newEntities = await this.extractEntities(message, citizenTurn.id);
    context.entities.push(...newEntities);

    // Classify intent
    const intent = await this.classifyIntent(context);
    context.currentIntent = intent;

    // Check for escalation triggers
    const escalationCheck = this.checkEscalationTriggers(context, message);
    if (escalationCheck.shouldEscalate) {
      return this.createEscalationResponse(context, escalationCheck.reason!);
    }

    // Retrieve relevant knowledge
    const knowledge = this.config.knowledgeBase.enabled
      ? await this.retrieveKnowledge(message, context)
      : [];

    // Generate response
    const response = await this.generateResponse(context, knowledge);

    // Check confidence for escalation
    if (response.confidence < this.config.escalation.confidenceThreshold) {
      response.escalateToHuman = true;
      response.escalationReason = 'I want to make sure you get accurate information.';
    }

    // Add agent turn
    const agentTurn: ConversationTurn = {
      id: `turn-${Date.now() + 1}`,
      role: 'agent',
      content: response.text,
      timestamp: new Date(),
      confidence: response.confidence,
      sources: response.sources,
      suggestedActions: response.suggestedActions,
    };
    context.history.push(agentTurn);

    // Update stats
    const latency = Date.now() - startTime;
    this.updateStats(response, latency);

    // Set response metadata
    response.metadata = {
      model: this.config.model.model,
      latencyMs: latency,
      domain: this.config.domain,
      intentMatched: intent?.name,
      kbQueries: knowledge.length > 0 ? 1 : 0,
      cached: false,
    };

    return response;
  }

  /**
   * Handle escalation to human
   */
  async escalateToHuman(
    conversationId: string,
    reason: string,
    priority: EscalationRequest['priority'] = 'normal'
  ): Promise<EscalationRequest> {
    const context = this.conversations.get(conversationId);
    if (!context) {
      throw new Error(`Conversation ${conversationId} not found`);
    }

    const request: EscalationRequest = {
      id: `esc-${Date.now()}`,
      conversationId,
      citizenDid: context.citizenDid,
      reason,
      priority,
      domain: this.config.domain,
      contextSummary: this.summarizeContext(context),
      entities: context.entities,
      requestedAt: new Date(),
      status: 'pending',
    };

    this.escalationQueue.push(request);
    this.stats.escalationRate = this.calculateEscalationRate();

    return request;
  }

  /**
   * Get pending escalations
   */
  getPendingEscalations(): EscalationRequest[] {
    return this.escalationQueue.filter((e) => e.status === 'pending');
  }

  /**
   * Resolve an escalation
   */
  resolveEscalation(escalationId: string, assignedTo: string): void {
    const request = this.escalationQueue.find((e) => e.id === escalationId);
    if (request) {
      request.status = 'resolved';
      request.assignedTo = assignedTo;
    }
  }

  /**
   * Get agent statistics
   */
  getStats(): AgentStats {
    return { ...this.stats };
  }

  /**
   * Set knowledge retriever
   */
  setKnowledgeRetriever(retriever: KnowledgeRetriever): void {
    this.knowledgeRetriever = retriever;
  }

  /**
   * Set model client
   */
  setModelClient(client: ModelClient): void {
    this.modelClient = client;
  }

  /**
   * Get conversation history
   */
  getConversationHistory(conversationId: string): ConversationTurn[] {
    return this.conversations.get(conversationId)?.history || [];
  }

  /**
   * End a conversation
   */
  endConversation(conversationId: string): void {
    const context = this.conversations.get(conversationId);
    if (context) {
      this.stats.avgConversationLength =
        (this.stats.avgConversationLength * (this.stats.totalConversations - 1) +
          context.history.length) /
        this.stats.totalConversations;
    }
    this.conversations.delete(conversationId);
  }

  // Private methods

  private async extractEntities(
    message: string,
    turnId: string
  ): Promise<ExtractedEntity[]> {
    const entities: ExtractedEntity[] = [];

    // Zip code
    const zipMatch = message.match(/\b\d{5}(?:-\d{4})?\b/);
    if (zipMatch) {
      entities.push({
        type: 'zip_code',
        value: zipMatch[0],
        confidence: 0.95,
        turnId,
      });
    }

    // Income (various formats)
    const incomeMatch = message.match(/\$?([\d,]+)(?:\s*(?:per|a|\/)\s*(?:year|month|week))?/i);
    if (incomeMatch && parseInt(incomeMatch[1].replace(',', '')) > 100) {
      entities.push({
        type: 'income',
        value: incomeMatch[1].replace(',', ''),
        confidence: 0.8,
        turnId,
      });
    }

    // Household size
    const householdMatch = message.match(/(\d+)\s*(?:people|person|family members?|in (?:my|our) (?:household|family))/i);
    if (householdMatch) {
      entities.push({
        type: 'household_size',
        value: householdMatch[1],
        confidence: 0.9,
        turnId,
      });
    }

    // Case number
    const caseMatch = message.match(/(?:case|reference|confirmation)\s*#?\s*([A-Z0-9-]+)/i);
    if (caseMatch) {
      entities.push({
        type: 'case_number',
        value: caseMatch[1],
        confidence: 0.85,
        turnId,
      });
    }

    // Phone number
    const phoneMatch = message.match(/(?:\+1)?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}/);
    if (phoneMatch) {
      entities.push({
        type: 'phone_number',
        value: phoneMatch[0].replace(/[^0-9]/g, ''),
        confidence: 0.9,
        turnId,
      });
    }

    // Email
    const emailMatch = message.match(/[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/);
    if (emailMatch) {
      entities.push({
        type: 'email',
        value: emailMatch[0],
        confidence: 0.95,
        turnId,
      });
    }

    return entities;
  }

  private async classifyIntent(context: ConversationContext): Promise<Intent | undefined> {
    const lastMessage = context.history[context.history.length - 1]?.content.toLowerCase() || '';

    // Intent patterns for civic domain
    const intentPatterns: Array<{
      name: string;
      patterns: RegExp[];
      requiredEntities: ExtractedEntity['type'][];
    }> = [
      {
        name: 'check_eligibility',
        patterns: [
          /am i eligible/i,
          /do i qualify/i,
          /can i get/i,
          /what benefits/i,
        ],
        requiredEntities: ['income', 'household_size', 'zip_code'],
      },
      {
        name: 'check_status',
        patterns: [
          /status of/i,
          /where is my/i,
          /check on/i,
          /what happened to/i,
        ],
        requiredEntities: ['case_number'],
      },
      {
        name: 'find_resources',
        patterns: [
          /where can i/i,
          /find a/i,
          /looking for/i,
          /need help with/i,
        ],
        requiredEntities: ['zip_code'],
      },
      {
        name: 'apply',
        patterns: [
          /how do i apply/i,
          /want to apply/i,
          /sign up for/i,
          /enroll in/i,
        ],
        requiredEntities: [],
      },
      {
        name: 'appeal',
        patterns: [
          /appeal/i,
          /disagree with/i,
          /unfair/i,
          /denied/i,
        ],
        requiredEntities: ['case_number'],
      },
      {
        name: 'general_question',
        patterns: [
          /what is/i,
          /how does/i,
          /tell me about/i,
          /explain/i,
        ],
        requiredEntities: [],
      },
    ];

    for (const intentDef of intentPatterns) {
      for (const pattern of intentDef.patterns) {
        if (pattern.test(lastMessage)) {
          const providedEntities = context.entities.map((e) => e.type);
          const hasRequired = intentDef.requiredEntities.every((req) =>
            providedEntities.includes(req)
          );

          return {
            name: intentDef.name,
            confidence: hasRequired ? 0.85 : 0.7,
            requiredEntities: intentDef.requiredEntities,
            providedEntities: providedEntities,
            complete: hasRequired,
          };
        }
      }
    }

    return undefined;
  }

  private checkEscalationTriggers(
    context: ConversationContext,
    message: string
  ): { shouldEscalate: boolean; reason?: string } {
    const lower = message.toLowerCase();

    // Check for always-escalate topics
    for (const topic of this.config.escalation.alwaysEscalateTopics) {
      if (lower.includes(topic.replace('_', ' '))) {
        return {
          shouldEscalate: true,
          reason: `This involves ${topic.replace('_', ' ')} - let me connect you with an expert.`,
        };
      }
    }

    // Check for explicit escalation requests
    if (
      lower.includes('speak to') ||
      lower.includes('talk to a human') ||
      lower.includes('real person') ||
      lower.includes('supervisor') ||
      lower.includes('agent')
    ) {
      return { shouldEscalate: true, reason: 'You asked to speak with a human.' };
    }

    // Check for frustration indicators
    const frustrationIndicators = ['this is ridiculous', 'not helping', 'useless', 'waste of time'];
    if (frustrationIndicators.some((ind) => lower.includes(ind))) {
      return {
        shouldEscalate: true,
        reason: 'I want to make sure you get the help you need.',
      };
    }

    // Check turn limit
    if (context.history.length >= this.config.escalation.maxTurnsBeforeEscalation) {
      return {
        shouldEscalate: true,
        reason: 'This seems complex - a human can help more efficiently.',
      };
    }

    return { shouldEscalate: false };
  }

  private async retrieveKnowledge(
    query: string,
    context: ConversationContext
  ): Promise<KnowledgeSource[]> {
    if (this.knowledgeRetriever) {
      return this.knowledgeRetriever.retrieve(query, {
        domain: context.domain,
        maxResults: this.config.knowledgeBase.maxSourcesPerResponse,
        minRelevance: this.config.knowledgeBase.minRelevance,
      });
    }

    // Default: return empty (no knowledge base configured)
    return [];
  }

  private async generateResponse(
    context: ConversationContext,
    knowledge: KnowledgeSource[]
  ): Promise<AgentResponse> {
    // Build prompt
    const systemPrompt = this.buildSystemPrompt();
    const conversationHistory = this.formatConversationHistory(context);
    const knowledgeContext = this.formatKnowledge(knowledge);

    // If model client is configured, use it
    if (this.modelClient) {
      const modelResponse = await this.modelClient.generate({
        systemPrompt,
        conversationHistory,
        knowledgeContext,
        maxTokens: this.config.model.maxTokens,
        temperature: this.config.model.temperature,
      });

      return {
        text: modelResponse.text,
        confidence: modelResponse.confidence,
        sources: knowledge,
        suggestedActions: this.generateSuggestedActions(context, modelResponse.text),
        needsMoreInfo: this.checkNeedsMoreInfo(context),
        followUpQuestions: this.generateFollowUpQuestions(context),
        escalateToHuman: false,
        metadata: {} as AgentResponse['metadata'],
      };
    }

    // Fallback: Generate rule-based response
    return this.generateRuleBasedResponse(context, knowledge);
  }

  private buildSystemPrompt(): string {
    const base = this.config.model.systemPrompt || this.getDefaultSystemPrompt();
    const personality = this.config.personality;

    let prompt = base;
    prompt += `\n\nTone: ${personality.tone}`;
    prompt += `\nReading Level: ${personality.readingLevel}`;
    if (personality.useEmojis) {
      prompt += '\nYou may use emojis sparingly to be friendly.';
    }

    return prompt;
  }

  private getDefaultSystemPrompt(): string {
    const domainPrompts: Record<CivicAgentDomain, string> = {
      benefits: `You are a friendly benefits navigator helping citizens understand and access government assistance programs like SNAP, Medicaid, and housing assistance. Be compassionate and non-judgmental. Never shame anyone for needing help.`,
      permits: `You are a permits assistant helping citizens understand permit requirements and application processes for businesses, buildings, and other activities. Be clear about requirements but helpful in navigating them.`,
      tax: `You are a tax assistance guide helping citizens understand their tax obligations and available credits/deductions. Be accurate and clear that you cannot provide legal tax advice - recommend a professional for complex situations.`,
      voting: `You are a voter services assistant helping citizens with registration, polling locations, and ballot information. Be strictly non-partisan and focus only on helping people exercise their right to vote.`,
      justice: `You are a justice system navigator helping citizens understand court processes, find legal aid, and access expungement services. Be compassionate and never assume guilt. Always recommend proper legal counsel for serious matters.`,
      housing: `You are a housing assistance guide helping citizens find affordable housing, understand tenant rights, and access rental assistance programs. Be empathetic about housing challenges.`,
      employment: `You are an employment services guide helping citizens find job training, understand unemployment benefits, and navigate workers' compensation. Be encouraging and practical.`,
      education: `You are an education services guide helping families with school enrollment, financial aid, and educational resources. Be supportive and helpful.`,
      health: `You are a health services navigator helping citizens find healthcare coverage, locate clinics, and access mental health resources. Be compassionate and non-judgmental about health needs.`,
      emergency: `You are an emergency services guide helping citizens in crisis find immediate assistance. Prioritize safety and clarity. Always provide crisis hotline numbers when relevant.`,
      general: `You are a civic services assistant helping citizens navigate government services. Be helpful, clear, and always direct people to the right resources.`,
    };

    return domainPrompts[this.config.domain] + `

Important guidelines:
- Be honest when you don't know something - say so and offer to connect them with someone who does
- Keep responses concise but complete
- Always provide actionable next steps when possible
- Never make promises about eligibility or outcomes
- Respect privacy - don't ask for sensitive information unnecessarily
- If someone seems to be in crisis, prioritize connecting them with help`;
  }

  private formatConversationHistory(context: ConversationContext): string {
    return context.history
      .map((turn) => `${turn.role === 'citizen' ? 'Citizen' : 'Agent'}: ${turn.content}`)
      .join('\n');
  }

  private formatKnowledge(knowledge: KnowledgeSource[]): string {
    if (knowledge.length === 0) return '';

    return `Relevant information:\n${knowledge
      .map((k) => `[${k.type}] ${k.title}: ${k.id}`)
      .join('\n')}`;
  }

  private generateRuleBasedResponse(
    context: ConversationContext,
    knowledge: KnowledgeSource[]
  ): AgentResponse {
    const lastMessage = context.history[context.history.length - 1]?.content || '';
    const intent = context.currentIntent;

    // Generate response based on intent
    let text = '';
    let confidence = 0.7;

    if (intent) {
      switch (intent.name) {
        case 'check_eligibility':
          if (!intent.complete) {
            text = this.askForMissingEntities(intent, context);
            confidence = 0.8;
          } else {
            text = 'Based on what you\'ve shared, you may be eligible for several programs. I recommend using our online screening tool or calling 311 for a full eligibility check.';
            confidence = 0.75;
          }
          break;

        case 'check_status':
          if (!intent.complete) {
            text = 'I\'d be happy to help you check on that. Could you provide your case or reference number?';
          } else {
            const caseNumber = context.entities.find((e) => e.type === 'case_number')?.value;
            text = `Let me look up case ${caseNumber} for you. One moment... I can see your case is in our system. For the most current status, please check our online portal or call 311.`;
          }
          confidence = 0.8;
          break;

        case 'find_resources':
          const zip = context.entities.find((e) => e.type === 'zip_code')?.value;
          if (zip) {
            text = `I can help you find resources near ${zip}. What type of help are you looking for? For example: food assistance, housing help, healthcare, or job training.`;
          } else {
            text = 'I can help you find local resources. What\'s your zip code so I can find options near you?';
          }
          confidence = 0.85;
          break;

        case 'apply':
          text = 'Great! I can help you get started with an application. The process usually involves: 1) Gathering required documents (ID, proof of income, proof of address), 2) Completing the application online or in person, and 3) Attending an interview if required. What program are you interested in applying for?';
          confidence = 0.8;
          break;

        case 'appeal':
          text = 'You have the right to appeal any decision you disagree with. The appeal process typically involves: 1) Filing a written request within 90 days of the decision, 2) Providing any new information or documentation, 3) Attending a hearing. Would you like me to explain the specific appeal process for your situation?';
          confidence = 0.75;
          break;

        default:
          text = 'I\'m here to help! Could you tell me more about what you\'re looking for? I can assist with benefits, permits, finding local resources, and more.';
          confidence = 0.6;
      }
    } else {
      // No intent matched
      if (lastMessage.includes('hello') || lastMessage.includes('hi')) {
        text = this.getGreeting();
        confidence = 0.9;
      } else if (lastMessage.includes('thank')) {
        text = 'You\'re welcome! Is there anything else I can help you with?';
        confidence = 0.9;
      } else {
        text = 'I\'m not sure I understood that correctly. Could you rephrase or tell me what kind of help you\'re looking for? I can assist with benefits eligibility, permit applications, finding local resources, and more.';
        confidence = 0.4;
      }
    }

    return {
      text,
      confidence,
      sources: knowledge,
      suggestedActions: this.generateSuggestedActions(context, text),
      needsMoreInfo: !intent?.complete,
      followUpQuestions: this.generateFollowUpQuestions(context),
      escalateToHuman: confidence < this.config.escalation.confidenceThreshold,
      escalationReason: confidence < this.config.escalation.confidenceThreshold
        ? 'I want to make sure you get accurate information.'
        : undefined,
      metadata: {} as AgentResponse['metadata'],
    };
  }

  private askForMissingEntities(intent: Intent, context: ConversationContext): string {
    const provided = context.entities.map((e) => e.type);
    const missing = intent.requiredEntities.filter((e) => !provided.includes(e));

    if (missing.includes('income')) {
      return 'To check your eligibility, I\'ll need to know your approximate household income. What is your total monthly or yearly income before taxes?';
    }
    if (missing.includes('household_size')) {
      return 'How many people are in your household, including yourself?';
    }
    if (missing.includes('zip_code')) {
      return 'What\'s your zip code? This helps me find programs available in your area.';
    }
    if (missing.includes('case_number')) {
      return 'Could you provide your case or reference number? You can find this on any correspondence you\'ve received.';
    }

    return 'I need a bit more information to help you. Could you tell me more about your situation?';
  }

  private getGreeting(): string {
    const greetings: Record<PersonalityConfig['greetingStyle'], string> = {
      full: `Hello! I'm here to help you navigate government services. I can assist with checking benefit eligibility, understanding permit requirements, finding local resources, and more. How can I help you today?`,
      brief: 'Hello! How can I help you today?',
      none: '',
    };
    return greetings[this.config.personality.greetingStyle];
  }

  private generateSuggestedActions(
    context: ConversationContext,
    _responseText: string
  ): SuggestedAction[] {
    const actions: SuggestedAction[] = [];

    // Always offer 311 as a secondary option
    actions.push({
      type: 'call',
      label: 'Call 311',
      value: '311',
      priority: 'secondary',
    });

    // Add domain-specific actions
    if (context.domain === 'benefits') {
      actions.unshift({
        type: 'link',
        label: 'Benefits Screener',
        value: '/benefits/screener',
        priority: 'primary',
      });
    }

    if (context.domain === 'permits') {
      actions.unshift({
        type: 'link',
        label: 'Apply Online',
        value: '/permits/apply',
        priority: 'primary',
      });
    }

    return actions;
  }

  private generateFollowUpQuestions(context: ConversationContext): string[] {
    if (!context.currentIntent || context.currentIntent.complete) {
      return [];
    }

    const missing = context.currentIntent.requiredEntities.filter(
      (e) => !context.entities.map((ent) => ent.type).includes(e)
    );

    return missing.slice(0, 2).map((entity) => {
      switch (entity) {
        case 'income':
          return 'What is your household income?';
        case 'household_size':
          return 'How many people are in your household?';
        case 'zip_code':
          return 'What is your zip code?';
        default:
          return `Could you provide your ${entity.replace('_', ' ')}?`;
      }
    });
  }

  private checkNeedsMoreInfo(context: ConversationContext): boolean {
    return context.currentIntent ? !context.currentIntent.complete : true;
  }

  private createEscalationResponse(
    context: ConversationContext,
    reason: string
  ): AgentResponse {
    const channels = this.config.escalation.channels;

    let text = reason + '\n\n';
    text += 'You can reach a human representative:\n';
    for (const channel of channels) {
      text += `- ${channel.type === 'phone' ? 'Call' : channel.type}: ${channel.value}`;
      if (channel.availability) text += ` (${channel.availability})`;
      if (channel.avgWaitTime) text += ` - typically ${channel.avgWaitTime} wait`;
      text += '\n';
    }

    // Create escalation request
    this.escalateToHuman(context.conversationId, reason);

    return {
      text,
      confidence: 1.0,
      sources: [],
      suggestedActions: channels.map((ch) => ({
        type: ch.type as SuggestedAction['type'],
        label: `${ch.type === 'phone' ? 'Call' : ch.type} ${ch.value}`,
        value: ch.value,
        priority: 'primary' as const,
      })),
      needsMoreInfo: false,
      escalateToHuman: true,
      escalationReason: reason,
      metadata: {} as AgentResponse['metadata'],
    };
  }

  private summarizeContext(context: ConversationContext): string {
    const summary: string[] = [];

    summary.push(`Domain: ${context.domain}`);
    summary.push(`Channel: ${context.channel}`);
    summary.push(`Turns: ${context.history.length}`);

    if (context.currentIntent) {
      summary.push(`Intent: ${context.currentIntent.name}`);
    }

    if (context.entities.length > 0) {
      summary.push(`Entities: ${context.entities.map((e) => `${e.type}=${e.value}`).join(', ')}`);
    }

    summary.push(`Last message: "${context.history[context.history.length - 1]?.content || 'N/A'}"`);

    return summary.join('\n');
  }

  private initStats(): AgentStats {
    return {
      totalConversations: 0,
      avgConversationLength: 0,
      resolutionRate: 1.0,
      escalationRate: 0,
      avgConfidence: 0.8,
      avgLatencyMs: 0,
      byDomain: {} as Record<CivicAgentDomain, number>,
      topIntents: [],
    };
  }

  private updateStats(response: AgentResponse, latency: number): void {
    // Running average for latency
    const n = this.stats.totalConversations;
    this.stats.avgLatencyMs = (this.stats.avgLatencyMs * (n - 1) + latency) / n;

    // Running average for confidence
    this.stats.avgConfidence =
      (this.stats.avgConfidence * (n - 1) + response.confidence) / n;

    // Update domain counts
    this.stats.byDomain[this.config.domain] =
      (this.stats.byDomain[this.config.domain] || 0) + 1;
  }

  private calculateEscalationRate(): number {
    const total = this.stats.totalConversations;
    const escalated = this.escalationQueue.length;
    return total > 0 ? escalated / total : 0;
  }
}

/**
 * Interface for knowledge retrieval
 */
export interface KnowledgeRetriever {
  retrieve(
    query: string,
    options: {
      domain: CivicAgentDomain;
      maxResults: number;
      minRelevance: number;
    }
  ): Promise<KnowledgeSource[]>;
}

/**
 * Interface for model client
 */
export interface ModelClient {
  generate(options: {
    systemPrompt: string;
    conversationHistory: string;
    knowledgeContext: string;
    maxTokens: number;
    temperature: number;
  }): Promise<{ text: string; confidence: number }>;
}

/**
 * Create an agent runner
 */
export function createAgentRunner(config: AgentConfig): AgentRunner {
  return new AgentRunner(config);
}

// Type alias for PersonalityConfig import
type PersonalityConfig = import('./types.js').PersonalityConfig;
