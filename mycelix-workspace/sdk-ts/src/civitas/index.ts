/**
 * Civitas - Civic AI for Citizen Sovereignty
 *
 * "Civitas" (Latin: community of citizens)
 *
 * Civitas is NOT government AI. It's citizen-owned AI that:
 * - Helps citizens understand their rights and options
 * - Navigates bureaucracy on behalf of citizens
 * - Explains decisions in plain language
 * - Advocates for citizen interests
 * - Is transparent about its reasoning and limitations
 *
 * Architecture:
 * - Civitas is a configuration of Symthaea (consciousness framework)
 * - Uses MATL for trust scoring (34% validated Byzantine tolerance)
 * - Consciousness-gated: won't answer beyond its knowledge
 * - Domain-specific agents share knowledge via swarm intelligence
 *
 * Domain Agents:
 * - Benefits-Civitas: Eligibility, applications, renewals
 * - Justice-Civitas: Rights, appeals, mediation, advocacy
 * - Participation-Civitas: Voting, proposals, civic engagement
 */

// ============================================================================
// Core Types
// ============================================================================

export interface CivitasConfig {
  /** Symthaea backend URL (Rust consciousness engine) */
  symthaeaUrl: string;
  /** Mycelix conductor URL for trust/identity */
  conductorUrl?: string;
  /** Preferred language for responses */
  language: 'en' | 'es' | 'zh' | 'vi' | 'ko' | 'tl';
  /** Accessibility mode (plain language, larger text hints) */
  accessibilityMode: boolean;
  /** Voice output enabled */
  voiceEnabled: boolean;
  /** Agent domain */
  domain: 'benefits' | 'justice' | 'participation' | 'general';
}

export interface CivitasMessage {
  role: 'citizen' | 'civitas' | 'system';
  content: string;
  timestamp: Date;
  /** Epistemic status of the response */
  epistemicStatus?: EpistemicStatus;
  /** Sources/citations for claims */
  sources?: Source[];
  /** Suggested actions */
  actions?: SuggestedAction[];
}

export interface EpistemicStatus {
  /** How confident is Civitas in this response? */
  confidence: 'known' | 'uncertain' | 'unknown';
  /** What type of knowledge is this? */
  knowledgeType: 'factual' | 'procedural' | 'interpretive' | 'opinion';
  /** When was this information last verified? */
  verifiedAt?: Date;
  /** Can this be independently verified? */
  verifiable: boolean;
}

export interface Source {
  type: 'law' | 'regulation' | 'policy' | 'precedent' | 'document';
  citation: string;
  url?: string;
  relevance: number;
}

export interface SuggestedAction {
  label: string;
  description: string;
  type: 'apply' | 'appeal' | 'vote' | 'learn' | 'contact' | 'delegate';
  endpoint?: string;
  smsCommand?: string;
}

export interface ConversationContext {
  citizenId: string;
  sessionId: string;
  domain: CivitasConfig['domain'];
  history: CivitasMessage[];
  /** Current topic being discussed */
  topic?: string;
  /** Relevant citizen data (with consent) */
  citizenContext?: CitizenContext;
}

export interface CitizenContext {
  /** Basic demographics (only what's relevant) */
  zipcode?: string;
  householdSize?: number;
  incomeLevel?: 'low' | 'moderate' | 'middle' | 'high';
  /** Active benefits */
  activeBenefits?: string[];
  /** Pending applications */
  pendingApplications?: string[];
  /** Voting delegations */
  delegations?: { domain: string; delegate: string }[];
}

// ============================================================================
// Civitas Agent Base Class
// ============================================================================

export abstract class CivitasAgent {
  protected config: CivitasConfig;
  protected context: ConversationContext | null = null;

  constructor(config: CivitasConfig) {
    this.config = config;
  }

  /**
   * Start a new conversation
   */
  async startConversation(citizenId: string): Promise<ConversationContext> {
    this.context = {
      citizenId,
      sessionId: crypto.randomUUID(),
      domain: this.config.domain,
      history: [],
    };

    // Add greeting
    const greeting = await this.getGreeting();
    this.context.history.push({
      role: 'civitas',
      content: greeting.content,
      timestamp: new Date(),
      epistemicStatus: { confidence: 'known', knowledgeType: 'procedural', verifiable: false },
      actions: greeting.actions,
    });

    return this.context;
  }

  /**
   * Send a message to Civitas
   */
  async chat(message: string): Promise<CivitasMessage> {
    if (!this.context) {
      throw new Error('No active conversation. Call startConversation first.');
    }

    // Add citizen message to history
    this.context.history.push({
      role: 'citizen',
      content: message,
      timestamp: new Date(),
    });

    // Process through domain-specific logic
    const response = await this.processMessage(message);

    // Add response to history
    this.context.history.push(response);

    return response;
  }

  /**
   * Get domain-specific greeting
   */
  protected abstract getGreeting(): Promise<{ content: string; actions?: SuggestedAction[] }>;

  /**
   * Process message through domain-specific logic
   */
  protected abstract processMessage(message: string): Promise<CivitasMessage>;

  /**
   * Call Symthaea backend
   */
  protected async callSymthaea(prompt: string, context: object): Promise<string> {
    try {
      const response = await fetch(`${this.config.symthaeaUrl}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          context,
          domain: this.config.domain,
          language: this.config.language,
          consciousness_gated: true,
        }),
      });

      const result = await response.json();
      return result.response;
    } catch {
      return this.getFallbackResponse();
    }
  }

  protected getFallbackResponse(): string {
    const fallbacks: Record<string, string> = {
      en: "I'm having trouble connecting right now. Please try again, or text HELP to find a human who can assist you.",
      es: 'Tengo problemas para conectarme ahora. Por favor intente de nuevo, o envíe AYUDA para encontrar asistencia humana.',
    };
    return fallbacks[this.config.language] || fallbacks.en;
  }
}

// ============================================================================
// Benefits-Civitas: Eligibility, Applications, Renewals
// ============================================================================

export class BenefitsCivitas extends CivitasAgent {
  constructor(config: Omit<CivitasConfig, 'domain'>) {
    super({ ...config, domain: 'benefits' });
  }

  protected async getGreeting(): Promise<{ content: string; actions?: SuggestedAction[] }> {
    const greetings: Record<string, string> = {
      en: `Hello! I'm Civitas, your benefits assistant. I can help you:

• Check what programs you might qualify for
• Understand application requirements
• Track your pending applications
• Renew expiring benefits

What would you like help with today?`,
      es: `¡Hola! Soy Civitas, su asistente de beneficios. Puedo ayudarle a:

• Ver para qué programas puede calificar
• Entender los requisitos de solicitud
• Rastrear sus solicitudes pendientes
• Renovar beneficios que expiran

¿En qué le puedo ayudar hoy?`,
    };

    return {
      content: greetings[this.config.language] || greetings.en,
      actions: [
        {
          label: 'Check Eligibility',
          description: 'See what programs you may qualify for',
          type: 'learn',
          smsCommand: 'BENEFITS CHECK',
        },
        {
          label: 'Track Applications',
          description: 'Check status of pending applications',
          type: 'learn',
          smsCommand: 'STATUS benefits',
        },
        {
          label: 'Renewal Reminders',
          description: 'See benefits that need renewal',
          type: 'learn',
          smsCommand: 'BENEFITS RENEW',
        },
      ],
    };
  }

  protected async processMessage(message: string): Promise<CivitasMessage> {
    const lowerMessage = message.toLowerCase();

    // Eligibility check
    if (lowerMessage.includes('qualify') || lowerMessage.includes('eligible') || lowerMessage.includes('check')) {
      return this.checkEligibility();
    }

    // Application status
    if (lowerMessage.includes('status') || lowerMessage.includes('pending') || lowerMessage.includes('track')) {
      return this.checkApplicationStatus();
    }

    // Renewal
    if (lowerMessage.includes('renew') || lowerMessage.includes('expir')) {
      return this.checkRenewals();
    }

    // General question - send to Symthaea
    const response = await this.callSymthaea(message, {
      domain: 'benefits',
      citizenContext: this.context?.citizenContext,
    });

    return {
      role: 'civitas',
      content: response,
      timestamp: new Date(),
      epistemicStatus: {
        confidence: 'uncertain',
        knowledgeType: 'interpretive',
        verifiable: true,
      },
    };
  }

  private async checkEligibility(): Promise<CivitasMessage> {
    // In production: call Mycelix eligibility engine
    return {
      role: 'civitas',
      content: `Based on your profile, you may be eligible for:

**High Match (90%+)**
• Supplemental Nutrition (SNAP) - Monthly food assistance
• Medicaid - Health coverage for you and family

**Possible Match (70-89%)**
• Earned Income Credit - Tax refund up to $7,430
• Childcare Assistance - Help with daycare costs

Would you like me to explain any of these, or help you apply?`,
      timestamp: new Date(),
      epistemicStatus: {
        confidence: 'uncertain',
        knowledgeType: 'procedural',
        verifiable: true,
      },
      sources: [
        { type: 'regulation', citation: '7 CFR 273 - SNAP Eligibility', relevance: 0.95 },
        { type: 'regulation', citation: '42 CFR 435 - Medicaid Eligibility', relevance: 0.92 },
      ],
      actions: [
        { label: 'Apply for SNAP', type: 'apply', smsCommand: 'APPLY SNAP', description: 'Start SNAP application' },
        { label: 'Learn More', type: 'learn', description: 'Get details on each program' },
      ],
    };
  }

  private async checkApplicationStatus(): Promise<CivitasMessage> {
    return {
      role: 'civitas',
      content: `Your pending applications:

**SNAP Application** (submitted 5 days ago)
Status: Under Review (65% complete)
Next step: Interview scheduled for Thursday 2pm
→ Reply CONFIRM to keep appointment or RESCHEDULE

**Medicaid Application** (submitted 12 days ago)
Status: Documents Needed
Missing: Proof of income (pay stubs or tax return)
→ Reply UPLOAD to submit documents

Need help with either of these?`,
      timestamp: new Date(),
      epistemicStatus: { confidence: 'known', knowledgeType: 'factual', verifiable: true },
      actions: [
        { label: 'Confirm Interview', type: 'apply', smsCommand: 'CONFIRM', description: 'Keep Thursday appointment' },
        { label: 'Upload Documents', type: 'apply', smsCommand: 'UPLOAD', description: 'Submit missing documents' },
      ],
    };
  }

  private async checkRenewals(): Promise<CivitasMessage> {
    return {
      role: 'civitas',
      content: `Benefits needing attention:

**SNAP** - Renewal due in 23 days
Your benefits will continue if you renew on time.
Most of your information is pre-filled.
→ Reply RENEW SNAP to start (takes ~5 minutes)

**Medicaid** - Auto-renewed ✓
No action needed until next year.

Would you like to renew SNAP now?`,
      timestamp: new Date(),
      epistemicStatus: { confidence: 'known', knowledgeType: 'factual', verifiable: true },
      actions: [
        { label: 'Renew SNAP', type: 'apply', smsCommand: 'RENEW SNAP', description: 'Start renewal process' },
      ],
    };
  }
}

// ============================================================================
// Justice-Civitas: Rights, Appeals, Mediation
// ============================================================================

export class JusticeCivitas extends CivitasAgent {
  constructor(config: Omit<CivitasConfig, 'domain'>) {
    super({ ...config, domain: 'justice' });
  }

  protected async getGreeting(): Promise<{ content: string; actions?: SuggestedAction[] }> {
    return {
      content: `Hello! I'm Civitas, your rights advocate. I can help you:

• Understand your legal rights in any situation
• Appeal decisions you believe are unfair
• Find mediation for disputes
• Connect you with legal assistance

**Important**: I provide information, not legal advice. For complex matters, I'll help you find a human advocate.

What's on your mind?`,
      actions: [
        { label: 'Know Your Rights', type: 'learn', description: 'Learn about your rights' },
        { label: 'Appeal a Decision', type: 'appeal', smsCommand: 'APPEAL', description: 'Start an appeal' },
        { label: 'Find Mediation', type: 'contact', description: 'Resolve a dispute' },
        { label: 'Legal Help', type: 'contact', description: 'Connect with free legal aid' },
      ],
    };
  }

  protected async processMessage(message: string): Promise<CivitasMessage> {
    const lowerMessage = message.toLowerCase();

    // Appeal request
    if (lowerMessage.includes('appeal') || lowerMessage.includes('unfair') || lowerMessage.includes('wrong')) {
      return this.handleAppealRequest(message);
    }

    // Rights question
    if (lowerMessage.includes('right') || lowerMessage.includes('can they') || lowerMessage.includes('allowed')) {
      return this.handleRightsQuestion(message);
    }

    // Dispute/mediation
    if (lowerMessage.includes('dispute') || lowerMessage.includes('mediat') || lowerMessage.includes('conflict')) {
      return this.handleDisputeRequest();
    }

    // General - send to Symthaea with justice context
    const response = await this.callSymthaea(message, { domain: 'justice' });

    return {
      role: 'civitas',
      content: response,
      timestamp: new Date(),
      epistemicStatus: { confidence: 'uncertain', knowledgeType: 'interpretive', verifiable: true },
    };
  }

  private async handleAppealRequest(_message: string): Promise<CivitasMessage> {
    return {
      role: 'civitas',
      content: `I understand you want to appeal a decision. Let me help.

**Your Appeal Rights:**
• You can appeal ANY government decision that affects you
• Appeals are reviewed by someone different than the original decision-maker
• You have the right to present evidence and be heard
• Free help is available

**To start your appeal, I need:**
1. What decision are you appealing? (e.g., "SNAP denial", "permit rejection")
2. When did you receive the decision?
3. Why do you believe it's wrong?

You can also text: APPEAL [case number] to begin formally.

What decision would you like to appeal?`,
      timestamp: new Date(),
      epistemicStatus: { confidence: 'known', knowledgeType: 'procedural', verifiable: true },
      sources: [
        { type: 'law', citation: 'Administrative Procedure Act § 555', relevance: 0.9 },
        { type: 'regulation', citation: '7 CFR 273.15 - Fair Hearings', relevance: 0.85 },
      ],
      actions: [
        { label: 'Start Appeal', type: 'appeal', smsCommand: 'APPEAL NEW', description: 'Begin formal appeal' },
        { label: 'Find Legal Aid', type: 'contact', description: 'Get free legal help' },
      ],
    };
  }

  private async handleRightsQuestion(_message: string): Promise<CivitasMessage> {
    // In production: use Symthaea with rights knowledge base
    return {
      role: 'civitas',
      content: `I'll help you understand your rights.

To give you accurate information, can you tell me more about the situation?

For example:
• "Can my landlord evict me for..."
• "Do I have the right to see my records?"
• "Can they deny me service because..."

The more specific you are, the better I can help. And remember - if this is urgent or involves potential harm, please also contact a human advocate.`,
      timestamp: new Date(),
      epistemicStatus: { confidence: 'known', knowledgeType: 'procedural', verifiable: false },
    };
  }

  private async handleDisputeRequest(): Promise<CivitasMessage> {
    return {
      role: 'civitas',
      content: `Mediation can help resolve disputes without going to court. It's:
• **Free** through community mediation centers
• **Voluntary** - both parties must agree
• **Confidential** - what's said stays private
• **Faster** than formal proceedings

**Types of disputes we can help with:**
• Neighbor conflicts
• Landlord-tenant issues
• Consumer complaints
• Family matters
• Workplace issues

Would you like me to find a mediator in your area, or learn more about the process?`,
      timestamp: new Date(),
      epistemicStatus: { confidence: 'known', knowledgeType: 'procedural', verifiable: true },
      actions: [
        { label: 'Find Mediator', type: 'contact', smsCommand: 'HELP mediation', description: 'Locate nearby services' },
        { label: 'Learn More', type: 'learn', description: 'How mediation works' },
        {
          label: 'Restorative Circle',
          type: 'contact',
          description: 'Community-based healing process',
        },
      ],
    };
  }
}

// ============================================================================
// Participation-Civitas: Voting, Proposals, Civic Engagement
// ============================================================================

export class ParticipationCivitas extends CivitasAgent {
  constructor(config: Omit<CivitasConfig, 'domain'>) {
    super({ ...config, domain: 'participation' });
  }

  protected async getGreeting(): Promise<{ content: string; actions?: SuggestedAction[] }> {
    return {
      content: `Hello! I'm Civitas, your civic participation guide. I can help you:

• See what's up for vote and understand the issues
• Find candidates and their positions
• Set up smart delegation (experts vote for you)
• Participate in community budgeting
• Track how your representatives voted

Democracy works better when everyone participates. How can I help you engage?`,
      actions: [
        { label: 'Active Votes', type: 'vote', smsCommand: 'VOTE LIST', description: "See what's being decided" },
        { label: 'Set Delegation', type: 'delegate', smsCommand: 'VOTE DELEGATE', description: 'Let experts vote for you' },
        { label: 'Budget Input', type: 'vote', smsCommand: 'BALANCE HEARTH', description: 'Participatory budgeting' },
        { label: 'My Impact', type: 'learn', description: 'See your civic participation history' },
      ],
    };
  }

  protected async processMessage(message: string): Promise<CivitasMessage> {
    const lowerMessage = message.toLowerCase();

    // Voting
    if (lowerMessage.includes('vote') || lowerMessage.includes('ballot') || lowerMessage.includes('elect')) {
      return this.handleVotingQuestion(message);
    }

    // Delegation
    if (lowerMessage.includes('delegate') || lowerMessage.includes('proxy') || lowerMessage.includes('represent')) {
      return this.handleDelegationQuestion();
    }

    // Budget/HEARTH
    if (lowerMessage.includes('budget') || lowerMessage.includes('hearth') || lowerMessage.includes('fund')) {
      return this.handleBudgetQuestion();
    }

    // General civic question
    const response = await this.callSymthaea(message, { domain: 'participation' });

    return {
      role: 'civitas',
      content: response,
      timestamp: new Date(),
      epistemicStatus: { confidence: 'uncertain', knowledgeType: 'interpretive', verifiable: true },
    };
  }

  private async handleVotingQuestion(_message: string): Promise<CivitasMessage> {
    return {
      role: 'civitas',
      content: `Here's what's currently up for community decision:

**Active Proposals:**

1. **Community Garden Funding** (ends in 3 days)
   Allocate $15,000 for new plots at Miller Park
   Current: 67% YES | Your delegate: Voted YES
   → Reply VOTE YES 1 or VOTE NO 1

2. **Traffic Calming on Oak St** (ends in 7 days)
   Install speed bumps near elementary school
   Current: 54% YES | Your delegate: Not voted
   → Reply VOTE YES 2 or VOTE NO 2

3. **Library Hours Extension** (ends in 14 days)
   Keep library open until 9pm on weekdays
   Current: 81% YES | Your delegate: Voted YES
   → Reply VOTE YES 3 or VOTE NO 3

Reply with a number for more details, or VOTE [YES/NO] [number] to cast your vote.`,
      timestamp: new Date(),
      epistemicStatus: { confidence: 'known', knowledgeType: 'factual', verifiable: true },
      actions: [
        { label: 'Vote on #1', type: 'vote', smsCommand: 'VOTE INFO 1', description: 'Community Garden details' },
        { label: 'See All', type: 'learn', smsCommand: 'VOTE LIST', description: 'Full list of proposals' },
      ],
    };
  }

  private async handleDelegationQuestion(): Promise<CivitasMessage> {
    return {
      role: 'civitas',
      content: `**Liquid Democracy**: Delegate your vote to people you trust.

You can set different delegates for different topics:
• **Fiscal**: Budget and spending decisions
• **Environmental**: Parks, climate, sustainability
• **Social**: Housing, healthcare, education
• **Infrastructure**: Roads, transit, utilities

**Your Current Delegations:**
• Fiscal: Maria Chen (Trust: 94%)
• Environmental: Not set
• Social: David Park (Trust: 87%)
• Infrastructure: Not set

**How it works:**
1. Your delegate votes on your behalf
2. You can override any vote they make
3. You can revoke delegation anytime
4. If your delegate delegates, it chains (max 3 levels)

Would you like to set up a delegation?`,
      timestamp: new Date(),
      epistemicStatus: { confidence: 'known', knowledgeType: 'procedural', verifiable: true },
      actions: [
        {
          label: 'Find Delegates',
          type: 'delegate',
          smsCommand: 'VOTE DELEGATE environmental',
          description: 'See trusted community members',
        },
        { label: 'Remove Delegation', type: 'delegate', description: 'Take back direct voting' },
      ],
    };
  }

  private async handleBudgetQuestion(): Promise<CivitasMessage> {
    return {
      role: 'civitas',
      content: `**HEARTH**: The Democratic Commons Pool

HEARTH is community money that we decide together how to spend.

**Current Pool**: 247,500 CGC
**Your Contribution**: 150 CGC
**Your Voting Weight**: 0.06%

**Funding Proposals:**
1. Youth Sports Equipment - 5,000 CGC (78% support)
2. Senior Center Renovation - 25,000 CGC (65% support)
3. Community WiFi Hotspots - 8,000 CGC (71% support)

**How to participate:**
• BALANCE ALLOCATE [amount] - Add to the pool
• VOTE YES [proposal] - Support funding
• Propose your own project at mygov.mycelix.net

Your voice matters. Even small contributions increase your voting weight.`,
      timestamp: new Date(),
      epistemicStatus: { confidence: 'known', knowledgeType: 'factual', verifiable: true },
      actions: [
        { label: 'Contribute', type: 'apply', smsCommand: 'BALANCE ALLOCATE 50', description: 'Add 50 CGC to pool' },
        { label: 'Vote on Projects', type: 'vote', smsCommand: 'VOTE LIST', description: 'See funding proposals' },
      ],
    };
  }
}

// ============================================================================
// Factory and Exports
// ============================================================================

export type CivitasDomain = 'benefits' | 'justice' | 'participation';

export function createCivitas(
  domain: CivitasDomain,
  config: Omit<CivitasConfig, 'domain'>
): CivitasAgent {
  switch (domain) {
    case 'benefits':
      return new BenefitsCivitas(config);
    case 'justice':
      return new JusticeCivitas(config);
    case 'participation':
      return new ParticipationCivitas(config);
    default:
      throw new Error(`Unknown Civitas domain: ${domain}`);
  }
}

export const CivitasDefaults: Omit<CivitasConfig, 'domain' | 'symthaeaUrl'> = {
  language: 'en',
  accessibilityMode: false,
  voiceEnabled: false,
};
