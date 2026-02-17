# Mycelix Conflict Transformation & Restorative Justice Framework

## Philosophy: Conflict as Growth Catalyst

> "Conflict is not the problem. Conflict is the soil in which transformation grows. The question is whether we have the practices to compost our pain into wisdom."

This framework provides comprehensive protocols for transforming conflict into community strengthening, moving beyond punitive justice toward healing and growth.

---

## Part 1: Foundational Principles

### 1.1 The Conflict Transformation Paradigm

```
Traditional Justice          vs.         Restorative/Transformative Justice
─────────────────────────────────────────────────────────────────────────────
What rule was broken?                    Who was harmed?
Who did it?                              What are their needs?
What punishment?                         Whose obligation to address?
                                         How do we prevent this?
                                         How do we heal together?
```

### 1.2 Core Principles

```typescript
interface TransformativeJusticePrinciples {
  // Harm-centered (not rule-centered)
  harmCentered: {
    focusOnImpact: true;           // What harm occurred?
    needsIdentification: true;      // What do affected parties need?
    healingOriented: true;          // How do we repair?
  };

  // Relationship-focused
  relationshipFocused: {
    preserveConnection: true;       // Maintain community bonds
    restoreRelationship: true;      // Heal between parties
    strengthenCommunity: true;      // Community grows from process
  };

  // Accountability without punishment
  accountabilityModel: {
    acknowledgment: true;           // Recognize harm caused
    understanding: true;            // Grasp impact on others
    amends: true;                   // Take action to repair
    transformation: true;           // Change behavior/conditions
    noPunishment: true;             // No retribution
  };

  // Systemic awareness
  systemicAwareness: {
    rootCauses: true;               // What conditions enabled harm?
    powerDynamics: true;            // How does power play into this?
    preventionFocus: true;          // How do we prevent recurrence?
    collectiveResponsibility: true; // Community's role
  };

  // Voluntary participation
  voluntaryParticipation: {
    consentBased: true;             // All parties choose to participate
    alternativesAvailable: true;    // Other options if unwilling
    paceRespected: true;            // Healing takes its own time
  };
}
```

### 1.3 Developmental Stage Considerations

```typescript
interface StageAdaptedConflict {
  // Traditional/Amber stage
  traditional: {
    conflictView: "violation-of-sacred-order";
    resolution: "restoration-of-proper-relations";
    authority: "elders-and-tradition";
    adaptations: {
      honorRituals: true;
      elderInvolvement: true;
      traditionalCeremony: true;
      clearRolesRestored: true;
    };
  };

  // Modern/Orange stage
  modern: {
    conflictView: "breach-of-agreement";
    resolution: "fair-outcome-for-individuals";
    authority: "neutral-process";
    adaptations: {
      clearProcedures: true;
      documentedAgreements: true;
      measurableOutcomes: true;
      individualRights: true;
    };
  };

  // Postmodern/Green stage
  postmodern: {
    conflictView: "expression-of-unmet-needs";
    resolution: "healing-relationships";
    authority: "consensus-and-equality";
    adaptations: {
      feelingsHonored: true;
      everyoneHeard: true;
      processOverOutcome: true;
      powerDynamicsNamed: true;
    };
  };

  // Integral/Teal stage
  integral: {
    conflictView: "growth-opportunity";
    resolution: "transformation-and-evolution";
    authority: "context-appropriate";
    adaptations: {
      multiplePerspectives: true;
      systemicAnalysis: true;
      shadowWork: true;
      adaptiveProcess: true;
    };
  };
}
```

---

## Part 2: Conflict Types & Pathways

### 2.1 Conflict Classification

```typescript
interface ConflictClassification {
  // By severity
  severity: {
    tier1_minor: {
      description: "Misunderstandings, small slights, communication issues";
      examples: ["missed-commitment", "misread-tone", "scheduling-conflict"];
      pathway: "direct-dialogue";
      timeframe: "hours-to-days";
    };

    tier2_moderate: {
      description: "Repeated issues, resource disputes, role conflicts";
      examples: ["ongoing-tension", "resource-allocation", "boundary-violations"];
      pathway: "facilitated-dialogue";
      timeframe: "days-to-weeks";
    };

    tier3_significant: {
      description: "Trust breaches, harm caused, community impact";
      examples: ["broken-agreements", "financial-harm", "exclusionary-behavior"];
      pathway: "restorative-circle";
      timeframe: "weeks-to-months";
    };

    tier4_severe: {
      description: "Serious harm, safety concerns, pattern of harm";
      examples: ["harassment", "theft", "abuse-of-power", "safety-threats"];
      pathway: "full-transformative-process";
      timeframe: "months";
    };

    tier5_crisis: {
      description: "Immediate safety risk, community-threatening";
      examples: ["violence", "imminent-danger", "existential-threat"];
      pathway: "safety-first-then-process";
      timeframe: "immediate-response";
    };
  };

  // By type
  type: {
    interpersonal: "between-individuals";
    group: "between-groups-or-factions";
    structural: "with-community-systems";
    values: "about-fundamental-differences";
    resource: "about-allocation-distribution";
    power: "about-authority-influence";
    boundary: "about-limits-consent";
  };
}
```

### 2.2 Conflict Pathway Selection

```typescript
interface PathwaySelection {
  // Decision tree
  pathway: (conflict: Conflict) => ConflictPathway = {
    // Step 1: Safety check
    if (conflict.immediateSafetyConcern) {
      return 'safety-protocol';
    }

    // Step 2: Willingness check
    if (!conflict.allPartiesWilling) {
      return 'parallel-process';  // Work with willing parties
    }

    // Step 3: Complexity assessment
    if (conflict.severity <= 'tier1') {
      return 'direct-dialogue';
    }

    if (conflict.severity === 'tier2') {
      return conflict.relationshipStrength > 0.5
        ? 'peer-mediation'
        : 'facilitated-dialogue';
    }

    if (conflict.severity === 'tier3') {
      return 'restorative-circle';
    }

    if (conflict.severity >= 'tier4') {
      return 'full-transformative-process';
    }
  };
}
```

---

## Part 3: Restorative Processes

### 3.1 Direct Dialogue Protocol

For Tier 1 conflicts between willing parties.

```typescript
interface DirectDialogue {
  // Preparation
  preparation: {
    selfReflection: {
      questions: [
        "What happened from my perspective?",
        "How am I feeling about this?",
        "What do I need?",
        "What might their perspective be?",
        "What outcome would feel good?"
      ];
      duration: "30-minutes-minimum";
    };

    requestConversation: {
      template: "I'd like to talk about [situation]. Would you be open to a conversation? I'm hoping we can understand each other better.";
      timing: "when-both-calm";
      setting: "private-comfortable";
    };
  };

  // Process
  process: {
    opening: {
      acknowledgment: "Thank you for being willing to talk";
      intention: "I want to understand and be understood";
      agreement: "Can we agree to listen fully before responding?";
    };

    sharing: {
      speakerA: {
        observation: "When [specific behavior]...";
        feeling: "I felt [emotion]...";
        need: "Because I need [need]...";
        request: "Would you be willing to [request]?";
      };

      speakerB: {
        reflection: "What I hear you saying is...";
        response: "[Their perspective using same format]";
      };
    };

    resolution: {
      mutualUnderstanding: "So we both...";
      agreements: "Going forward, we agree to...";
      checkIn: "Let's check in about this in [timeframe]";
    };
  };

  // Follow-up
  followUp: {
    scheduledCheckIn: true;
    documentAgreements: 'optional';
    escalationPath: 'facilitated-dialogue';
  };
}
```

### 3.2 Facilitated Dialogue Protocol

For Tier 2 conflicts requiring neutral support.

```typescript
interface FacilitatedDialogue {
  // Facilitator role
  facilitator: {
    qualities: [
      "trusted-by-both-parties",
      "skilled-in-nonviolent-communication",
      "able-to-remain-neutral",
      "culturally-competent"
    ];

    responsibilities: [
      "create-safe-container",
      "ensure-equal-voice",
      "reflect-and-clarify",
      "guide-toward-resolution",
      "not-judge-or-advise"
    ];
  };

  // Pre-session
  preSession: {
    // Individual meetings with each party
    individualMeetings: {
      purpose: [
        "build-rapport",
        "understand-perspective",
        "assess-readiness",
        "explain-process",
        "identify-needs"
      ];
      duration: "45-60-minutes-each";
    };

    // Readiness assessment
    readiness: {
      emotional: "can-participate-without-flooding";
      willingness: "genuinely-wants-resolution";
      safety: "no-power-imbalance-preventing-honesty";
    };
  };

  // Session structure
  session: {
    opening: {
      welcomeAndGratitude: true;
      groundRules: [
        "speak-from-I",
        "listen-to-understand",
        "one-person-speaks",
        "confidentiality",
        "take-breaks-as-needed"
      ];
      intention: "We're here to understand each other and find a way forward";
    };

    storytelling: {
      // Each person shares uninterrupted
      format: "what-happened-how-felt-what-need";
      facilitation: "reflection-and-clarification";
      duration: "10-15-minutes-each";
    };

    dialogue: {
      // Facilitated exchange
      exploration: "What would help you understand each other better?";
      needsIdentification: "What does each person need?";
      brainstorming: "What might address those needs?";
    };

    agreements: {
      specific: true;          // Clear actions
      mutual: true;            // Both contribute
      timebound: true;         // When by
      followUp: true;          // Check-in scheduled
    };

    closing: {
      appreciation: "What do you appreciate about this conversation?";
      commitment: "Verbal commitment to agreements";
      documentation: "Written summary if desired";
    };
  };

  // Post-session
  postSession: {
    checkIn: "1-2-weeks-after";
    assessment: "Are agreements working?";
    adjustment: "What needs to change?";
    celebration: "If resolved, acknowledge healing";
  };
}
```

### 3.3 Restorative Circle Protocol

For Tier 3 conflicts requiring community involvement.

```typescript
interface RestorativeCircle {
  // Circle philosophy
  philosophy: {
    equality: "Everyone sits as equals in circle";
    wholeness: "The whole person is present";
    connection: "We are all connected";
    storytelling: "Truth emerges through stories";
    consensus: "We seek shared understanding";
  };

  // Participants
  participants: {
    core: {
      harmedParty: "Person(s) who experienced harm";
      responsibleParty: "Person(s) who caused harm";
      facilitator: "Trained circle keeper";
    };

    support: {
      harmedSupport: "1-2 supporters for harmed party";
      responsibleSupport: "1-2 supporters for responsible party";
      communityWitnesses: "2-4 community members";
      eldersOrAdvisors: "Optional wisdom holders";
    };
  };

  // Pre-circle preparation
  preparation: {
    // Individual meetings
    individualPrep: {
      withHarmed: {
        assess: "readiness-and-needs";
        explain: "process-and-options";
        identify: "what-they-need-from-circle";
        prepare: "what-they-want-to-say";
        support: "who-to-bring";
      };

      withResponsible: {
        assess: "readiness-and-accountability";
        explain: "process-and-expectations";
        explore: "understanding-of-impact";
        prepare: "what-they-want-to-say";
        support: "who-to-bring";
      };
    };

    // Logistics
    logistics: {
      space: "comfortable-circular-seating";
      time: "2-4-hours-uninterrupted";
      centerpiece: "meaningful-object";
      talkingPiece: "passed-object-grants-speaking";
      refreshments: "available-for-breaks";
    };
  };

  // Circle process
  process: {
    // Round 1: Opening
    round1_opening: {
      welcomeAndAcknowledgment: "We gather because something happened that affected us";
      introductions: "Name and connection to situation";
      guidelines: "Speak from heart, listen from heart, speak spontaneously, speak leanly";
      openingQuestion: "What brings you here today? What do you hope for?";
    };

    // Round 2: Storytelling
    round2_storytelling: {
      harmedPartyFirst: {
        prompt: "Please share what happened and how it affected you";
        fullListening: "No interruptions, full attention";
        reflection: "Facilitator reflects key points";
      };

      responsibleParty: {
        prompt: "What do you want us to understand about what happened?";
        fullListening: true;
        reflection: true;
      };

      communityWitnesses: {
        prompt: "How has this affected the community?";
      };
    };

    // Round 3: Impact exploration
    round3_impact: {
      toHarmed: "What has been the hardest part?";
      toResponsible: "What do you think about what you've heard?";
      toCommunity: "What needs to happen for the community to heal?";
    };

    // Round 4: Needs identification
    round4_needs: {
      harmedNeeds: "What do you need to move forward?";
      responsibleNeeds: "What do you need to make things right?";
      communityNeeds: "What does the community need?";
    };

    // Round 5: Agreements
    round5_agreements: {
      brainstorm: "What actions might address these needs?";
      specifics: "Who does what by when?";
      consensus: "Does everyone agree this is fair and achievable?";
      documentation: "Written agreement signed by all";
    };

    // Round 6: Closing
    round6_closing: {
      closingRound: "One word or phrase about how you're leaving";
      gratitude: "Thank everyone for courage and commitment";
      ceremony: "Closing ritual appropriate to community";
    };
  };

  // Follow-up
  followUp: {
    checkInSchedule: ["2-weeks", "1-month", "3-months"];
    agreementMonitoring: "Is responsible party following through?";
    healingAssessment: "How is harmed party doing?";
    communityReintegration: "Is community healing?";
    celebrationOfCompletion: "When agreements fulfilled";
  };
}
```

### 3.4 Full Transformative Justice Process

For Tier 4 severe harm requiring comprehensive response.

```typescript
interface TransformativeJusticeProcess {
  // Philosophy
  philosophy: {
    noPoliceNoPrisons: "We don't rely on state violence";
    communityCentered: "Community takes responsibility";
    survivorCentered: "Survivor's needs guide process";
    accountabilityFocused: "Responsible party must transform";
    systemsChange: "Address conditions that enabled harm";
  };

  // Phase 1: Immediate response
  phase1_immediate: {
    safetyFirst: {
      survivorSafety: "Ensure survivor is safe";
      separation: "Create space between parties if needed";
      supportNetwork: "Activate support for survivor";
      crisisResources: "Professional support if needed";
    };

    initialAssessment: {
      harmAssessment: "What happened? What is the impact?";
      safetyAssessment: "Is there ongoing danger?";
      willingnessAssessment: "Who is willing to participate?";
      resourceAssessment: "What support is available?";
    };
  };

  // Phase 2: Support team formation
  phase2_teams: {
    survivorSupportTeam: {
      role: "Support survivor throughout process";
      members: ["trusted-friends", "trained-advocates", "professional-support"];
      responsibilities: [
        "check-in-regularly",
        "accompany-to-meetings",
        "advocate-for-needs",
        "help-with-healing"
      ];
    };

    accountabilityTeam: {
      role: "Support responsible party in accountability";
      members: ["trusted-friends", "mentors", "trained-facilitators"];
      responsibilities: [
        "regular-check-ins",
        "challenge-denial",
        "support-transformation",
        "monitor-agreements"
      ];
    };

    coordinationTeam: {
      role: "Manage overall process";
      members: ["trained-facilitators", "community-leaders"];
      responsibilities: [
        "facilitate-communication",
        "maintain-boundaries",
        "document-process",
        "ensure-accountability"
      ];
    };
  };

  // Phase 3: Survivor-centered process
  phase3_survivorCentered: {
    survivorNeeds: {
      identify: "What does survivor need to heal?";
      categories: [
        "safety",
        "acknowledgment",
        "understanding",
        "amends",
        "assurance-of-no-repeat",
        "community-support"
      ];
      timeline: "survivor-determines-pace";
    };

    survivorChoices: {
      contactWithResponsible: "Does survivor want any contact?";
      communityInvolvement: "How public should process be?";
      desiredOutcomes: "What would resolution look like?";
      participationLevel: "How involved does survivor want to be?";
    };
  };

  // Phase 4: Accountability process
  phase4_accountability: {
    acknowledgment: {
      harmRecognition: "Fully recognize what was done";
      impactUnderstanding: "Understand effects on survivor and community";
      noExcuses: "No minimizing, justifying, or blaming";
      publicOrPrivate: "Based on survivor preference";
    };

    rootCauseWork: {
      personalWork: {
        therapy: "Professional support";
        education: "Learn about harm caused";
        shadowWork: "Explore patterns and wounds";
        skillBuilding: "Develop healthier behaviors";
      };

      communityEducation: {
        sharelearning: "What are you learning?";
        preventionWork: "Help prevent similar harm";
      };
    };

    amends: {
      directAmends: "What can be done for survivor?";
      communityAmends: "What can be done for community?";
      systemicAmends: "What can change conditions?";
      ongoingCommitments: "Long-term accountability";
    };

    consequences: {
      notPunishment: "Consequences are not punishment";
      natural: "Natural results of actions";
      protective: "Protecting community if needed";
      timebound: "Clear duration and path back";

      examples: [
        "temporary-role-restrictions",
        "supervised-participation",
        "loss-of-trust-positions",
        "required-education",
        "community-service",
        "in-extreme-cases-removal"
      ];
    };
  };

  // Phase 5: Community healing
  phase5_communityHealing: {
    communityProcess: {
      acknowledgment: "Community acknowledges what happened";
      collectiveResponsibility: "How did community enable this?";
      systemsChange: "What needs to change?";
      healingRituals: "Community healing practices";
    };

    prevention: {
      policyChanges: "New policies to prevent harm";
      educationPrograms: "Community education";
      earlyWarning: "How to catch problems earlier";
      supportStructures: "Better support systems";
    };
  };

  // Phase 6: Reintegration (if appropriate)
  phase6_reintegration: {
    conditions: {
      survivorConsent: "Survivor agrees to reintegration";
      demonstratedChange: "Clear evidence of transformation";
      communityReadiness: "Community ready to receive";
      ongoingAccountability: "Continued accountability structures";
    };

    process: {
      gradual: "Phased return to community";
      supervised: "Accountability team continues";
      celebrated: "Acknowledge transformation";
      monitored: "Ongoing check-ins";
    };
  };

  // Timeline
  timeline: {
    phase1: "immediate-to-1-week";
    phase2: "week-1-to-2";
    phase3: "ongoing";
    phase4: "3-12-months";
    phase5: "parallel-to-phase4";
    phase6: "when-conditions-met";
  };
}
```

---

## Part 4: Specialized Protocols

### 4.1 Power-Based Harm Protocol

When harm involves power imbalance (leaders, elders, those with authority).

```typescript
interface PowerBasedHarmProtocol {
  // Elevated response
  elevatedResponse: {
    whyDifferent: "Power imbalance means normal processes insufficient";
    additionalSafeguards: [
      "external-facilitators",
      "independent-investigation",
      "enhanced-survivor-support",
      "community-transparency"
    ];
  };

  // Process modifications
  modifications: {
    // Investigation
    investigation: {
      independent: "Facilitators from outside power structure";
      thorough: "Examine patterns, not just single incident";
      confidential: "Protect survivors who come forward";
    };

    // Accountability
    accountability: {
      immediateStepDown: "Remove from power during process";
      publicAcknowledgment: "Community has right to know";
      higherBar: "More required for reintegration";
    };

    // Systems change
    systemsChange: {
      structuralReview: "How did power enable harm?";
      checksAndBalances: "New accountability structures";
      powerDistribution: "Reduce concentration of power";
    };
  };
}
```

### 4.2 Group Conflict Protocol

When conflict is between groups or factions.

```typescript
interface GroupConflictProtocol {
  // Assessment
  assessment: {
    groupIdentification: "Who are the groups?";
    issueMapping: "What are the issues?";
    historyUnderstanding: "What's the history?";
    leaderIdentification: "Who can speak for groups?";
  };

  // Process
  process: {
    // Phase 1: Within-group work
    withinGroup: {
      internalDialogue: "Each group clarifies their perspective";
      needsIdentification: "What does the group need?";
      representativeSelection: "Who will represent in dialogue?";
      mandateClarity: "What can representatives agree to?";
    };

    // Phase 2: Representative dialogue
    representativeDialogue: {
      format: "facilitated-dialogue-between-representatives";
      goal: "mutual-understanding-and-framework";
      reporting: "representatives-report-back";
    };

    // Phase 3: Broader dialogue
    broaderDialogue: {
      format: "fishbowl-or-world-cafe";
      participation: "all-group-members";
      goal: "build-shared-understanding";
    };

    // Phase 4: Agreement building
    agreementBuilding: {
      proposal: "representatives-draft-agreement";
      feedback: "groups-review-and-modify";
      ratification: "both-groups-formally-agree";
    };
  };

  // Reconciliation
  reconciliation: {
    sharedRitual: "Joint ceremony or celebration";
    ongoingDialogue: "Regular cross-group conversations";
    jointProjects: "Collaborative work to rebuild trust";
  };
}
```

### 4.3 Values Conflict Protocol

When conflict is about fundamental values, not just behavior.

```typescript
interface ValuesConflictProtocol {
  // Recognition
  recognition: {
    valuesNotBehavior: "This is about deeply held values";
    noResolution: "May not be resolvable";
    coexistence: "Goal may be peaceful coexistence";
  };

  // Process
  process: {
    // Deep listening
    deepListening: {
      format: "extended-empathic-listening";
      goal: "understand-why-values-matter";
      duration: "multiple-sessions";
      noDebate: "not-trying-to-convince";
    };

    // Common ground exploration
    commonGround: {
      sharedValues: "What values do we share?";
      sharedGoals: "What outcomes do we both want?";
      sharedHistory: "What connects us?";
    };

    // Boundary setting
    boundaries: {
      mustHaves: "What can't be compromised?";
      negotiables: "What can flex?";
      coexistenceTerms: "How do we live together?";
    };
  };

  // Outcomes
  possibleOutcomes: {
    synthesis: "New understanding that includes both";
    coexistence: "Agree to disagree respectfully";
    separation: "Amicable parting if needed";
    ongoingDialogue: "Continue the conversation";
  };
}
```

---

## Part 5: Mycelix Technical Integration

### 5.1 Arbiter hApp Enhancement

```typescript
// Enhanced Arbiter data structures
interface ConflictCase {
  case_id: string;
  case_type: ConflictType;
  severity_tier: SeverityTier;
  status: CaseStatus;

  // Parties
  parties: {
    harmed: PartyInfo[];
    responsible: PartyInfo[];
    supporters: Map<AgentPubKey, SupportRole>;
    facilitators: AgentPubKey[];
    community_witnesses: AgentPubKey[];
  };

  // Process tracking
  process: {
    pathway: ConflictPathway;
    current_phase: ProcessPhase;
    sessions: Session[];
    agreements: Agreement[];
    follow_ups: FollowUp[];
  };

  // Privacy
  privacy: {
    visibility: 'parties-only' | 'support-team' | 'community' | 'public';
    anonymized_learning: boolean;  // Share lessons without identifying
  };

  // Outcome
  outcome?: {
    resolution_type: ResolutionType;
    agreements_fulfilled: boolean;
    healing_assessment: HealingMetrics;
    lessons_extracted: string[];
  };
}

// Facilitator matching
interface FacilitatorMatch {
  match_score: number;
  factors: {
    cultural_competence: number;
    language_match: number;
    availability: boolean;
    no_conflict_of_interest: boolean;
    skill_match: number;
    party_preferences: number;
  };
}

// Agreement tracking
interface Agreement {
  agreement_id: string;
  created_in_session: string;
  parties_committed: AgentPubKey[];

  // Specifics
  action: string;
  responsible_party: AgentPubKey;
  deadline: Timestamp;
  verification_method: VerificationMethod;

  // Status
  status: 'pending' | 'in-progress' | 'completed' | 'modified' | 'not-fulfilled';
  completion_evidence?: string;
  verified_by?: AgentPubKey[];
}
```

### 5.2 Process Automation

```typescript
interface ConflictWorkflow {
  // Intake
  intake: {
    report_submission: (report: ConflictReport) => CaseId;
    initial_assessment: (case_id: CaseId) => AssessmentResult;
    pathway_recommendation: (assessment: AssessmentResult) => PathwayRecommendation;
    facilitator_matching: (case_id: CaseId) => FacilitatorMatch[];
  };

  // Scheduling
  scheduling: {
    find_available_times: (participants: AgentPubKey[]) => TimeSlot[];
    send_invitations: (session: Session) => void;
    send_reminders: (session: Session) => void;
    reschedule: (session: Session, new_time: TimeSlot) => void;
  };

  // Documentation
  documentation: {
    session_notes: (session: Session, notes: Notes) => void;
    agreement_creation: (session: Session, agreements: Agreement[]) => void;
    outcome_recording: (case_id: CaseId, outcome: Outcome) => void;
  };

  // Follow-up
  followUp: {
    schedule_check_ins: (case_id: CaseId, schedule: CheckInSchedule) => void;
    agreement_reminders: (agreement: Agreement) => void;
    completion_verification: (agreement: Agreement) => void;
    case_closure: (case_id: CaseId) => void;
  };

  // Learning
  learning: {
    extract_patterns: (cases: CaseId[]) => Pattern[];
    anonymize_for_learning: (case_id: CaseId) => AnonymizedCase;
    community_report: (period: TimePeriod) => CommunityReport;
  };
}
```

### 5.3 Safety Integrations

```typescript
interface SafetyIntegrations {
  // Sanctuary integration (mental health support)
  sanctuary: {
    crisis_referral: (party: AgentPubKey) => void;
    support_coordination: (case_id: CaseId) => void;
    trauma_informed_resources: () => Resource[];
  };

  // Beacon integration (emergency response)
  beacon: {
    safety_alert: (situation: SafetyRisk) => void;
    emergency_contacts: (party: AgentPubKey) => Contact[];
    safe_space_coordination: () => void;
  };

  // Kinship integration (family support)
  kinship: {
    family_notification: (party: AgentPubKey, level: NotificationLevel) => void;
    family_support_coordination: () => void;
  };
}
```

---

## Part 6: Community Capacity Building

### 6.1 Facilitator Training Pathway

```typescript
interface FacilitatorTraining {
  // Levels
  levels: {
    peer_supporter: {
      training: "20-hours";
      skills: ["active-listening", "empathy", "basic-NVC", "knowing-limits"];
      scope: "tier-1-support-only";
    };

    dialogue_facilitator: {
      training: "40-hours";
      skills: ["facilitation", "conflict-dynamics", "NVC-intermediate", "power-awareness"];
      scope: "tier-1-and-2";
      mentorship: "5-co-facilitations";
    };

    circle_keeper: {
      training: "80-hours";
      skills: ["circle-process", "trauma-informed", "NVC-advanced", "cultural-competence"];
      scope: "tier-1-2-3";
      mentorship: "10-circles-observed-5-co-kept";
    };

    transformative_justice_practitioner: {
      training: "200-hours";
      skills: ["full-TJ-process", "systems-analysis", "team-coordination", "crisis-response"];
      scope: "all-tiers";
      experience: "2-years-circle-keeping";
      mentorship: "ongoing-supervision";
    };
  };

  // Training components
  components: {
    theory: ["conflict-transformation-theory", "restorative-justice-history", "trauma-theory"];
    skills: ["listening", "facilitation", "de-escalation", "cultural-humility"];
    practice: ["role-plays", "simulations", "observed-practice", "mentored-practice"];
    selfWork: ["own-conflict-patterns", "shadow-work", "bias-examination"];
  };
}
```

### 6.2 Community Culture Building

```typescript
interface ConflictCulture {
  // Norms to cultivate
  communityNorms: {
    conflictNormalization: "Conflict is normal and growthful";
    directCommunication: "Address issues directly with care";
    earlyIntervention: "Don't let things fester";
    askForHelp: "It's strength to seek support";
    accountability: "We hold each other accountable with love";
    transformation: "People can change and grow";
  };

  // Practices
  regularPractices: {
    clearingCircles: {
      frequency: "monthly";
      purpose: "Address small tensions before they grow";
      format: "structured-sharing-and-listening";
    };

    appreciationPractices: {
      frequency: "weekly";
      purpose: "Build relational foundation";
      format: "gratitude-sharing";
    };

    conflictEducation: {
      frequency: "quarterly";
      purpose: "Build skills and awareness";
      format: "workshops-and-practice";
    };
  };

  // Early warning response
  earlyWarning: {
    tensionSensing: "Trained members notice early signs";
    checkInOffers: "Proactive outreach when tension sensed";
    lightweightIntervention: "Casual facilitation before escalation";
  };
}
```

---

## Part 7: Special Considerations

### 7.1 Online Conflict Specifics

```typescript
interface OnlineConflict {
  // Unique challenges
  challenges: {
    misinterpretation: "Text lacks tone and context";
    escalationSpeed: "Things can spiral quickly";
    publicNature: "Conflicts often visible to many";
    asyncTiming: "Delayed responses increase tension";
    screenshot_permanence: "Everything can be captured";
  };

  // Adaptations
  adaptations: {
    pauseNorm: "24-hour pause before responding to conflict";
    privateFirst: "Move to private channels quickly";
    videoOption: "Offer video call for nuanced conversation";
    moderatorIntervention: "Trained moderators step in early";
    coolDownChannels: "Dedicated spaces for processing";
  };

  // Platform features
  features: {
    conflictFlag: "Flag conversation for facilitator attention";
    pauseThread: "Temporarily pause heated threads";
    summaryRequest: "Ask AI to summarize perspectives";
    moveToPrivate: "One-click move to private channel";
    requestFacilitation: "Easy request for help";
  };
}
```

### 7.2 Cross-Cultural Conflict

```typescript
interface CrossCulturalConflict {
  // Additional considerations
  considerations: {
    culturalContext: "Understand cultural conflict norms";
    languageNuance: "Translation may miss nuance";
    powerDynamics: "Colonial and cultural power at play";
    differentNorms: "What's offensive varies by culture";
  };

  // Process adaptations
  adaptations: {
    culturalBridge: "Include cultural interpreters";
    multipleFormats: "Offer culturally appropriate formats";
    extendedTime: "Allow more time for understanding";
    educationComponent: "Mutual cultural education";
  };
}
```

---

## Appendix: Quick Reference Cards

### Conflict Response Flowchart

```
Conflict Arises
      │
      ▼
┌─────────────────┐
│ Safety concern? │
└────────┬────────┘
         │
    Yes  │  No
    ▼    │   ▼
Safety   └──► Tier Assessment
Protocol      │
              ▼
         ┌────┴────┐
         │ Tier 1? │──Yes──► Direct Dialogue
         └────┬────┘
              │ No
              ▼
         ┌────┴────┐
         │ Tier 2? │──Yes──► Facilitated Dialogue
         └────┬────┘
              │ No
              ▼
         ┌────┴────┐
         │ Tier 3? │──Yes──► Restorative Circle
         └────┬────┘
              │ No
              ▼
         Transformative
         Justice Process
```

### Facilitator Checklist

**Before Session:**
- [ ] Met individually with all parties
- [ ] Assessed readiness
- [ ] Explained process and options
- [ ] Identified support people
- [ ] Prepared space
- [ ] Reviewed cultural considerations

**During Session:**
- [ ] Created safe container
- [ ] Established agreements
- [ ] Ensured equal voice
- [ ] Stayed neutral
- [ ] Reflected and clarified
- [ ] Guided toward resolution
- [ ] Documented agreements

**After Session:**
- [ ] Sent summary to parties
- [ ] Scheduled follow-up
- [ ] Debriefed (if needed)
- [ ] Self-care

### NVC Quick Reference

**Observation:** "When I see/hear [specific behavior]..."
**Feeling:** "I feel [emotion]..."
**Need:** "Because I need [need]..."
**Request:** "Would you be willing to [specific request]?"

---

*"Every conflict is an invitation. An invitation to understand more deeply, to love more fully, to grow more completely. May we have the courage to accept these invitations."*
