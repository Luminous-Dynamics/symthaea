# Mycelix Crisis Response Network Protocol

## Philosophy: The Network as Immune System

> "When one node suffers, the whole network feels it. When the network responds together, even the greatest challenges can be met. We are each other's resilience."

This protocol defines how the Mycelix network responds to crises that exceed single community capacity—from natural disasters to economic shocks to coordinated threats.

---

## Architecture Alignment

> **Reference**: See [ARCHITECTURE_ALIGNMENT_GUIDE.md](../ARCHITECTURE_ALIGNMENT_GUIDE.md) for complete alignment standards.

### Epistemic Classification (LEM v2.0)

| Data Type | E-Tier | N-Tier | M-Tier | Description |
|-----------|--------|--------|--------|-------------|
| Crisis Alert | E2 (Privately Verifiable) | N1-N2 (Communal/Network) | M2 (Persistent) | Emergency notifications |
| Situation Report | E1-E2 | N1 (Communal) | M2 (Persistent) | Crisis status updates |
| Resource Request | E3 (Cryptographically Proven) | N1 (Communal) | M2 (Persistent) | Mutual aid requests |
| Safety Check-In | E2 (Privately Verifiable) | N0 (Personal) | M1 (Temporal) | Member safety status |
| After-Action Report | E1-E2 | N1-N2 | M2-M3 (Persistent/Foundational) | Post-crisis learnings |

### Token System Integration

- **CIV (Civic Standing)**: Determines eligibility for Crisis Coordinator roles; minimum thresholds required
- **CGC (Civic Gifting Credit)**: Recognize crisis responders and volunteers; express community gratitude
- **FLOW**: Emergency fund disbursements, resource allocation during crises

### Governance Tier Mapping

| Severity Level | Primary DAO Tier | Decision Authority |
|----------------|-----------------|-------------------|
| Level 1-2 (Watch/Alert) | Local DAO | Community Crisis Coordinator |
| Level 3 (Emergency) | Regional DAO | Regional Crisis Command |
| Level 4 (Major) | Global DAO | Federation Crisis Council |
| Level 5 (Catastrophe) | Global DAO + Foundation | Emergency Unity Council |

### MIP Categories for Crisis Evolution

- **MIP-T (Technical)**: Beacon hApp improvements, alert system upgrades
- **MIP-G (Governance)**: Crisis protocol amendments, authority adjustments
- **MIP-S (Social)**: Community response practices, volunteer coordination

### Key Institution References

- **Member Redress Council**: Appeals for crisis-related disputes
- **Foundation**: Golden Veto authority for Level 5 catastrophes; external liaison
- **Audit Guild**: Emergency fund accountability; post-crisis financial review

---

## Part 1: Crisis Classification

### 1.1 Crisis Types

```typescript
interface CrisisTypes {
  // Natural disasters
  natural: {
    acute: {
      examples: ["earthquake", "hurricane", "flood", "wildfire", "tsunami"];
      characteristics: ["sudden-onset", "geographic", "physical-damage"];
      response: "emergency-mobilization";
    };
    chronic: {
      examples: ["drought", "sea-level-rise", "desertification"];
      characteristics: ["slow-onset", "prolonged", "systemic"];
      response: "adaptation-and-migration-support";
    };
  };

  // Social crises
  social: {
    conflict: {
      examples: ["inter-community-conflict", "civil-unrest", "violence"];
      characteristics: ["human-caused", "relational", "escalating"];
      response: "mediation-and-protection";
    };
    displacement: {
      examples: ["refugee-crisis", "mass-migration", "forced-relocation"];
      characteristics: ["population-movement", "resource-strain"];
      response: "reception-and-integration";
    };
    persecution: {
      examples: ["targeted-community", "political-repression", "discrimination"];
      characteristics: ["power-imbalance", "systemic"];
      response: "protection-and-solidarity";
    };
  };

  // Economic crises
  economic: {
    localShock: {
      examples: ["major-employer-closure", "crop-failure", "market-collapse"];
      characteristics: ["economic-disruption", "livelihood-threat"];
      response: "economic-support-and-transition";
    };
    systemicCrisis: {
      examples: ["currency-crisis", "banking-collapse", "supply-chain-breakdown"];
      characteristics: ["widespread", "cascading"];
      response: "alternative-systems-activation";
    };
  };

  // Technical crises
  technical: {
    platformCrisis: {
      examples: ["major-outage", "data-breach", "coordinated-attack"];
      characteristics: ["infrastructure-threat", "network-wide"];
      response: "technical-emergency-response";
    };
    informational: {
      examples: ["disinformation-campaign", "coordinated-manipulation"];
      characteristics: ["trust-threat", "information-integrity"];
      response: "information-integrity-response";
    };
  };

  // Health crises
  health: {
    epidemic: {
      examples: ["disease-outbreak", "pandemic"];
      characteristics: ["health-threat", "contagion-risk"];
      response: "health-emergency-protocol";
    };
    mentalHealth: {
      examples: ["mass-trauma", "collective-grief", "crisis-of-meaning"];
      characteristics: ["psychological", "collective"];
      response: "collective-care-response";
    };
  };

  // Existential crises
  existential: {
    examples: ["community-collapse-risk", "fundamental-purpose-crisis"];
    characteristics: ["identity-threatening", "foundational"];
    response: "deep-renewal-process";
  };
}
```

### 1.2 Crisis Severity Levels

```typescript
interface SeverityLevels {
  // Level 1: Watch
  level1_watch: {
    description: "Potential crisis developing";
    indicators: [
      "early-warning-signs",
      "elevated-risk-indicators",
      "concerning-patterns"
    ];
    response: {
      monitoring: "Enhanced monitoring";
      preparation: "Review response plans";
      communication: "Internal awareness";
    };
    authority: "Crisis sensing team";
  };

  // Level 2: Alert
  level2_alert: {
    description: "Crisis likely or beginning";
    indicators: [
      "clear-threat-emerging",
      "early-impacts-visible",
      "escalation-trajectory"
    ];
    response: {
      activation: "Activate response teams";
      preparation: "Stage resources";
      communication: "Network-wide awareness";
    };
    authority: "Community crisis coordinators";
  };

  // Level 3: Emergency
  level3_emergency: {
    description: "Active crisis, single community or small region";
    indicators: [
      "significant-impacts",
      "community-capacity-strained",
      "urgent-needs"
    ];
    response: {
      mobilization: "Regional response activation";
      resources: "Resource flow to affected area";
      coordination: "Active coordination hub";
    };
    authority: "Regional crisis command";
  };

  // Level 4: Major emergency
  level4_major: {
    description: "Severe crisis, multiple communities affected";
    indicators: [
      "severe-widespread-impacts",
      "multiple-communities-affected",
      "regional-capacity-strained"
    ];
    response: {
      mobilization: "Network-wide response";
      resources: "Major resource mobilization";
      coordination: "Federation crisis command";
    };
    authority: "Federation crisis council";
  };

  // Level 5: Catastrophe
  level5_catastrophe: {
    description: "Existential threat to network or major portion";
    indicators: [
      "catastrophic-impacts",
      "network-survival-threatened",
      "unprecedented-scale"
    ];
    response: {
      mobilization: "All-network response";
      resources: "Everything available";
      coordination: "Emergency unified command";
      protocols: "Extraordinary measures activated";
    };
    authority: "Emergency unity council";
  };
}
```

---

## Part 2: Network Response Architecture

### 2.1 Response Structure

```
                    ┌─────────────────────────┐
                    │   GLOBAL COORDINATION   │
                    │   (Level 5 only)        │
                    └───────────┬─────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
   ┌────────▼────────┐ ┌───────▼────────┐ ┌───────▼────────┐
   │ CONTINENTAL     │ │ CONTINENTAL    │ │ CONTINENTAL    │
   │ COORDINATION    │ │ COORDINATION   │ │ COORDINATION   │
   │ (Level 4+)      │ │                │ │                │
   └────────┬────────┘ └───────┬────────┘ └───────┬────────┘
            │                  │                   │
   ┌────────▼────────┐        │          ┌───────▼────────┐
   │ BIOREGIONAL     │        │          │ BIOREGIONAL    │
   │ COORDINATION    │        │          │ COORDINATION   │
   │ (Level 3+)      │        │          │                │
   └────────┬────────┘        │          └───────┬────────┘
            │                 │                   │
   ┌────────▼────────┐        │          ┌───────▼────────┐
   │ COMMUNITY       │        │          │ COMMUNITY      │
   │ RESPONSE        │        │          │ RESPONSE       │
   │ (All levels)    │        │          │                │
   └─────────────────┘        │          └────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   AFFECTED        │
                    │   COMMUNITIES     │
                    └───────────────────┘
```

### 2.2 Crisis Response Roles

```typescript
interface CrisisRoles {
  // Community level
  communityLevel: {
    crisisCoordinator: {
      responsibility: "Coordinate community crisis response";
      activation: "Automatic at Level 2+";
      authority: "Emergency decision-making for community";
    };

    resourceSteward: {
      responsibility: "Manage community resources for crisis";
      activation: "As needed";
      authority: "Emergency resource allocation";
    };

    communicationsLead: {
      responsibility: "Manage crisis communications";
      activation: "At any level";
      authority: "Official community communications";
    };

    wellnessCoordinator: {
      responsibility: "Support member wellbeing during crisis";
      activation: "At any level";
      authority: "Wellness resource deployment";
    };
  };

  // Regional/Federation level
  regionLevel: {
    regionalCoordinator: {
      responsibility: "Coordinate across affected communities";
      activation: "Level 3+";
      authority: "Regional resource allocation";
    };

    mutualAidCoordinator: {
      responsibility: "Coordinate resource flows";
      activation: "Level 3+";
      authority: "Direct mutual aid resources";
    };

    externalLiaison: {
      responsibility: "Interface with external organizations";
      activation: "As needed";
      authority: "Represent network externally";
    };
  };

  // Network level
  networkLevel: {
    networkCrisisCouncil: {
      responsibility: "Coordinate network-wide response";
      activation: "Level 4+";
      composition: "Representatives from major regions";
      authority: "Network-level decisions";
    };

    emergencyUnityCouncil: {
      responsibility: "Guide existential crisis response";
      activation: "Level 5 only";
      composition: "Senior leaders + experts";
      authority: "Extraordinary measures";
    };
  };
}
```

### 2.3 Communication Protocols

```typescript
interface CrisisCommunication {
  // Internal communication
  internal: {
    // Alert channels
    alertChannels: {
      immediate: "Push notifications to all relevant members";
      broadcast: "Community-wide announcements";
      coordination: "Response team channels";
    };

    // Information flow
    informationFlow: {
      upward: "Situation reports from affected to coordination";
      downward: "Guidance from coordination to communities";
      lateral: "Peer-to-peer between communities";
    };

    // Formats
    formats: {
      situationReport: {
        frequency: "Every 4-6 hours in active crisis";
        content: [
          "current-situation",
          "impacts",
          "response-actions",
          "needs",
          "next-update"
        ];
      };

      resourceRequest: {
        content: [
          "requesting-community",
          "resource-type",
          "quantity",
          "urgency",
          "delivery-requirements"
        ];
      };

      actionAlert: {
        content: [
          "action-required",
          "who",
          "by-when",
          "how"
        ];
      };
    };
  };

  // External communication
  external: {
    // Spokesperson designation
    spokesperson: "Designated by crisis coordinator";

    // Message approval
    approval: "Crisis coordinator or delegate";

    // Key messages
    keyMessages: {
      template: [
        "what-is-happening",
        "what-we-are-doing",
        "what-people-can-do",
        "how-to-help",
        "where-to-get-updates"
      ];
    };

    // Media relations
    mediaRelations: {
      single_point: "One spokesperson for consistency";
      coordination: "Coordinate with affected communities";
      truthfulness: "Honest, even when difficult";
    };
  };

  // Silence protocols
  silenceProtocols: {
    when: [
      "security-sensitive-situations",
      "ongoing-negotiations",
      "protecting-vulnerable-members"
    ];
    how: "Coordinate silence across network";
    duration: "Minimum necessary";
  };
}
```

---

## Part 3: Resource Mobilization

### 3.1 Mutual Aid Activation

```typescript
interface MutualAidActivation {
  // Standing agreements
  standingAgreements: {
    mutualAidPledges: {
      description: "Communities pre-commit to mutual aid";
      content: [
        "resource-types-available",
        "capacity-levels",
        "activation-triggers",
        "delivery-capabilities"
      ];
      maintenance: "Annual review and update";
    };

    resourceRegistries: {
      description: "Inventories of available resources";
      types: [
        "material-resources",
        "skills-and-expertise",
        "infrastructure",
        "financial-capacity"
      ];
      updates: "Regular updates + crisis updates";
    };
  };

  // Activation process
  activation: {
    // Level 2-3: Regional mutual aid
    regional: {
      trigger: "Community request + coordinator approval";
      scope: "Bioregional federation";
      process: [
        "request-submitted",
        "coordinator-validates",
        "resources-matched",
        "delivery-coordinated"
      ];
    };

    // Level 4: Network-wide mutual aid
    networkWide: {
      trigger: "Regional request + network council approval";
      scope: "Entire Mycelix network";
      process: [
        "regional-request-submitted",
        "network-council-approves",
        "network-wide-call",
        "resources-flow"
      ];
    };
  };

  // Resource types
  resourceTypes: {
    material: {
      examples: ["food", "water", "shelter", "medicine", "equipment"];
      coordination: "Logistics coordinator";
      tracking: "Resource tracking system";
    };

    financial: {
      examples: ["emergency-funds", "mutual-credit-extension", "solidarity-payments"];
      coordination: "Financial coordinator";
      tracking: "Treasury integration";
    };

    human: {
      examples: ["volunteers", "specialists", "coordinators"];
      coordination: "Volunteer coordinator";
      tracking: "Skills matching system";
    };

    informational: {
      examples: ["expertise", "connections", "documentation"];
      coordination: "Information coordinator";
      tracking: "Knowledge management";
    };

    emotional: {
      examples: ["support", "presence", "care"];
      coordination: "Wellness coordinator";
      tracking: "Care coordination";
    };
  };
}
```

### 3.2 Emergency Funds

```typescript
interface EmergencyFunds {
  // Fund types
  fundTypes: {
    // Community emergency fund
    communityFund: {
      source: "Community contributions";
      governance: "Community governance";
      uses: "Community-level emergencies";
      target: "3-6 months operating reserves";
    };

    // Regional emergency fund
    regionalFund: {
      source: "Federation member contributions";
      governance: "Federation council";
      uses: "Multi-community emergencies";
      target: "Adequate for regional disasters";
    };

    // Network emergency fund
    networkFund: {
      source: "All federation contributions";
      governance: "Network crisis council";
      uses: "Major disasters, network-level crises";
      target: "Major disaster response capacity";
    };

    // Solidarity fund
    solidarityFund: {
      source: "Voluntary surplus contributions";
      governance: "Solidarity council";
      uses: "Support for struggling communities";
      philosophy: "From each according to ability, to each according to need";
    };
  };

  // Disbursement
  disbursement: {
    // Emergency grants
    emergencyGrants: {
      trigger: "Crisis declaration";
      approval: "Appropriate level coordinator";
      speed: "Within 24-48 hours";
      accountability: "Post-crisis reporting";
    };

    // Loans
    emergencyLoans: {
      trigger: "Request from community";
      approval: "Fund governance body";
      terms: "Low/no interest, flexible repayment";
      purpose: "Larger or longer-term needs";
    };
  };
}
```

### 3.3 Volunteer Coordination

```typescript
interface VolunteerCoordination {
  // Volunteer registry
  volunteerRegistry: {
    standingRegistry: {
      content: [
        "skills-and-expertise",
        "availability",
        "geographic-flexibility",
        "past-experience"
      ];
      updates: "Quarterly confirmation";
    };

    crisisSignup: {
      content: [
        "availability-for-this-crisis",
        "skills-to-offer",
        "constraints"
      ];
      activation: "Crisis-specific call";
    };
  };

  // Matching and deployment
  matching: {
    needsAssessment: "What volunteers are needed?";
    skillsMatching: "Match needs to available volunteers";
    logisticsPlanning: "How do volunteers get there/engage?";
    onboarding: "Crisis-specific orientation";
  };

  // Volunteer support
  support: {
    orientation: "Crisis context and role";
    logistics: "Travel, housing, meals";
    safety: "Safety briefing and equipment";
    emotional: "Debrief and support";
    recognition: "Appreciation and acknowledgment";
  };

  // Remote volunteering
  remote: {
    types: [
      "communications-support",
      "coordination-assistance",
      "research-and-information",
      "technical-support",
      "emotional-support"
    ];
    coordination: "Remote volunteer coordinator";
    tools: "Crisis collaboration platform";
  };
}
```

---

## Part 4: Response Protocols by Crisis Type

### 4.1 Natural Disaster Response

```typescript
interface NaturalDisasterResponse {
  // Immediate response (0-72 hours)
  immediate: {
    safetyFirst: {
      actions: [
        "account-for-all-members",
        "evacuate-if-needed",
        "provide-emergency-shelter",
        "address-immediate-medical-needs"
      ];
      coordination: "Community crisis coordinator";
    };

    communicationEstablishment: {
      actions: [
        "establish-communication-with-affected",
        "notify-network",
        "set-up-information-hub"
      ];
      tools: ["mesh-networks", "satellite", "radio", "runners"];
    };

    immediateNeeds: {
      assessment: "Rapid needs assessment";
      priorities: ["water", "food", "shelter", "medical", "safety"];
      resourceFlow: "Begin mutual aid flow";
    };
  };

  // Short-term response (72 hours - 2 weeks)
  shortTerm: {
    stabilization: {
      actions: [
        "establish-temporary-services",
        "continue-meeting-basic-needs",
        "begin-damage-assessment"
      ];
    };

    resourceMobilization: {
      actions: [
        "full-mutual-aid-activation",
        "volunteer-deployment",
        "external-resource-coordination"
      ];
    };

    documentation: {
      actions: [
        "document-damage",
        "track-resources",
        "record-decisions"
      ];
    };
  };

  // Medium-term response (2 weeks - 3 months)
  mediumTerm: {
    recovery: {
      actions: [
        "begin-rebuilding",
        "restore-services",
        "support-livelihoods"
      ];
    };

    communitySupport: {
      actions: [
        "grief-and-trauma-processing",
        "maintain-community-connection",
        "celebrate-progress"
      ];
    };
  };

  // Long-term response (3+ months)
  longTerm: {
    rebuilding: {
      actions: [
        "full-reconstruction",
        "systems-improvement",
        "resilience-building"
      ];
    };

    learning: {
      actions: [
        "after-action-review",
        "documentation-of-lessons",
        "policy-updates"
      ];
    };

    prevention: {
      actions: [
        "risk-reduction-measures",
        "early-warning-improvement",
        "preparedness-enhancement"
      ];
    };
  };
}
```

### 4.2 Economic Crisis Response

```typescript
interface EconomicCrisisResponse {
  // Immediate stabilization
  immediateStabilization: {
    assessment: {
      actions: [
        "assess-economic-impact",
        "identify-most-affected",
        "map-resource-availability"
      ];
    };

    emergencyMeasures: {
      actions: [
        "activate-emergency-funds",
        "extend-mutual-credit-limits",
        "pause-non-essential-obligations",
        "mobilize-solidarity-support"
      ];
    };

    basicNeeds: {
      actions: [
        "ensure-food-security",
        "protect-housing",
        "maintain-essential-services"
      ];
    };
  };

  // Alternative systems activation
  alternativeSystems: {
    mutualCreditExpansion: {
      actions: [
        "expand-credit-limits",
        "activate-backup-currencies",
        "facilitate-barter-networks"
      ];
    };

    collectivePurchasing: {
      actions: [
        "bulk-buying-coordination",
        "supplier-relationships",
        "cost-sharing"
      ];
    };

    skillSharing: {
      actions: [
        "activate-skill-banks",
        "coordinate-mutual-aid",
        "reduce-cash-dependence"
      ];
    };
  };

  // Economic recovery
  recovery: {
    livelihoodSupport: {
      actions: [
        "job-creation-initiatives",
        "retraining-programs",
        "cooperative-development"
      ];
    };

    economicDiversification: {
      actions: [
        "reduce-dependence-on-failed-sectors",
        "develop-local-production",
        "strengthen-internal-economy"
      ];
    };

    resilience building: {
      actions: [
        "build-reserves",
        "diversify-income-sources",
        "strengthen-mutual-aid-capacity"
      ];
    };
  };
}
```

### 4.3 Conflict Response

```typescript
interface ConflictResponse {
  // Internal conflict (within/between communities)
  internalConflict: {
    containment: {
      actions: [
        "prevent-escalation",
        "establish-communication-channels",
        "protect-vulnerable-parties"
      ];
    };

    mediation: {
      actions: [
        "activate-arbiter-processes",
        "bring-in-external-mediators-if-needed",
        "facilitate-dialogue"
      ];
    };

    resolution: {
      actions: [
        "work-toward-resolution",
        "repair-relationships",
        "address-root-causes"
      ];
    };
  };

  // External conflict (threat to communities)
  externalConflict: {
    protection: {
      actions: [
        "protect-members",
        "secure-resources",
        "establish-safe-zones"
      ];
    };

    solidarity: {
      actions: [
        "network-wide-solidarity",
        "external-advocacy",
        "resource-support"
      ];
    };

    documentation: {
      actions: [
        "document-incidents",
        "preserve-evidence",
        "support-accountability"
      ];
    };

    resilience: {
      actions: [
        "maintain-community-function",
        "psychological-support",
        "long-term-planning"
      ];
    };
  };
}
```

### 4.4 Health Crisis Response

```typescript
interface HealthCrisisResponse {
  // Epidemic response
  epidemicResponse: {
    containment: {
      actions: [
        "implement-health-protocols",
        "coordinate-with-health-authorities",
        "protect-vulnerable-members"
      ];
    };

    careCoordination: {
      actions: [
        "coordinate-medical-care",
        "provide-isolation-support",
        "ensure-basic-needs-met"
      ];
    };

    communitySupport: {
      actions: [
        "maintain-social-connection-safely",
        "provide-mental-health-support",
        "support-caregivers"
      ];
    };
  };

  // Mental health crisis response
  mentalHealthResponse: {
    immediateSupport: {
      actions: [
        "activate-crisis-support",
        "ensure-safety",
        "provide-immediate-care"
      ];
    };

    collectiveCare: {
      actions: [
        "normalize-struggle",
        "create-support-spaces",
        "coordinate-professional-support"
      ];
    };

    longTermHealing: {
      actions: [
        "ongoing-support-structures",
        "trauma-processing-opportunities",
        "resilience-building"
      ];
    };
  };
}
```

### 4.5 Technical Crisis Response

```typescript
interface TechnicalCrisisResponse {
  // Platform crisis
  platformCrisis: {
    immediateResponse: {
      actions: [
        "assess-scope-and-impact",
        "implement-containment",
        "activate-backup-systems"
      ];
    };

    recovery: {
      actions: [
        "restore-services",
        "verify-integrity",
        "communicate-status"
      ];
    };

    postCrisis: {
      actions: [
        "root-cause-analysis",
        "implement-fixes",
        "update-prevention-measures"
      ];
    };
  };

  // Information crisis
  informationCrisis: {
    detection: {
      actions: [
        "identify-disinformation",
        "assess-spread-and-impact",
        "determine-source"
      ];
    };

    response: {
      actions: [
        "provide-accurate-information",
        "counter-disinformation",
        "support-affected-members"
      ];
    };

    prevention: {
      actions: [
        "strengthen-information-literacy",
        "improve-verification-processes",
        "build-trust-in-community-sources"
      ];
    };
  };
}
```

---

## Part 5: Beacon hApp Crisis Features

### 5.1 Enhanced Beacon for Network Crisis

```typescript
interface BeaconCrisisFeatures {
  // Alert system
  alertSystem: {
    // Alert levels
    levels: {
      watch: "Elevated awareness";
      alert: "Prepare for response";
      emergency: "Active response";
      majorEmergency: "Network mobilization";
      catastrophe: "All-hands response";
    };

    // Alert propagation
    propagation: {
      local: "Community-only";
      regional: "Bioregional federation";
      networkWide: "Entire network";
    };

    // Alert channels
    channels: {
      push: "Immediate notification";
      sms: "SMS fallback";
      email: "Email for details";
      radio: "Emergency broadcast";
      physical: "Runner networks";
    };
  };

  // Resource coordination
  resourceCoordination: {
    // Needs tracking
    needsTracking: {
      submission: "Affected communities submit needs";
      categorization: "Categorize by type and urgency";
      matching: "Match with available resources";
      tracking: "Track fulfillment";
    };

    // Resource flow
    resourceFlow: {
      offers: "Communities offer resources";
      matching: "Match offers to needs";
      logistics: "Coordinate delivery";
      confirmation: "Confirm receipt";
    };
  };

  // Coordination hub
  coordinationHub: {
    // Situation dashboard
    dashboard: {
      currentSituation: "Real-time situation summary";
      resourceStatus: "Resource needs and availability";
      responseStatus: "Response activities underway";
      communications: "Recent updates";
    };

    // Coordination tools
    tools: {
      taskManagement: "Assign and track tasks";
      communication: "Team communication channels";
      documentation: "Record decisions and actions";
      mapping: "Geographic coordination";
    };
  };

  // Safety check-in
  safetyCheckIn: {
    trigger: "Crisis activation";
    process: {
      request: "System requests check-in";
      response: "Members mark safe/need-help/unresponsive";
      followUp: "Follow up with non-responders";
      reporting: "Report to coordinators";
    };
  };
}
```

### 5.2 Integration with Other hApps

```typescript
interface CrisisIntegrations {
  // Diplomat integration
  diplomat: {
    mutualAidAgreements: "Activate mutual aid";
    resourceSharing: "Coordinate resource flow";
    communicationBridges: "Connect affected to helpers";
  };

  // Treasury integration
  treasury: {
    emergencyFunds: "Activate emergency funds";
    resourceTracking: "Track resource flows";
    donationCoordination: "Coordinate donations";
  };

  // Commons integration
  commons: {
    sharedResources: "Mobilize shared resources";
    equipmentSharing: "Coordinate equipment";
    spaceSharing: "Coordinate spaces";
  };

  // Kinship integration
  kinship: {
    vulnerableMembers: "Identify vulnerable";
    careCoordination: "Coordinate care";
    familyNotification: "Family communication";
  };

  // Sanctuary integration
  sanctuary: {
    crisisSupport: "Activate crisis support";
    traumaResponse: "Coordinate trauma response";
    ongoingCare: "Transition to ongoing care";
  };
}
```

---

## Part 6: Preparedness

### 6.1 Community Preparedness

```typescript
interface CommunityPreparedness {
  // Risk assessment
  riskAssessment: {
    frequency: "Annual comprehensive + ongoing monitoring";
    areas: [
      "natural-hazards",
      "social-risks",
      "economic-vulnerabilities",
      "technical-risks",
      "health-risks"
    ];
    output: "Community risk profile";
  };

  // Response planning
  responsePlanning: {
    plans: {
      general: "Overall crisis response plan";
      specific: "Plans for identified risks";
    };
    content: [
      "roles-and-responsibilities",
      "communication-protocols",
      "resource-inventories",
      "evacuation-routes",
      "assembly-points"
    ];
    updates: "Annual review";
  };

  // Resource preparation
  resourcePreparation: {
    emergencyFund: "Community emergency reserves";
    supplies: "Emergency supplies if appropriate";
    relationships: "Mutual aid agreements in place";
    skills: "Trained crisis responders";
  };

  // Training and exercises
  training: {
    training: {
      topics: [
        "crisis-response-roles",
        "first-aid",
        "communication-protocols",
        "psychological-first-aid"
      ];
      frequency: "Ongoing";
    };
    exercises: {
      types: ["tabletop", "functional", "full-scale"];
      frequency: "Annual tabletop, periodic others";
    };
  };
}
```

### 6.2 Network Preparedness

```typescript
interface NetworkPreparedness {
  // Federation-level preparation
  federation: {
    agreements: "Mutual aid agreements between communities";
    funds: "Regional emergency funds";
    coordination: "Crisis coordination structures";
    communication: "Communication protocols and systems";
  };

  // Network-level preparation
  network: {
    structures: "Global crisis coordination";
    funds: "Network emergency fund";
    protocols: "Network-wide protocols";
    relationships: "External partnerships";
  };

  // Cross-network learning
  learning: {
    patternLibrary: "Crisis response patterns";
    afterActionReports: "Learn from past crises";
    exerciseCoordination: "Multi-community exercises";
    bestPracticeSharing: "Share what works";
  };
}
```

---

## Part 7: Recovery and Learning

### 7.1 Recovery Process

```typescript
interface RecoveryProcess {
  // Phases
  phases: {
    relief: {
      duration: "Immediate - 1 month";
      focus: "Meeting immediate needs";
      activities: ["emergency-services", "basic-needs", "safety"];
    };

    recovery: {
      duration: "1-6 months";
      focus: "Restoring function";
      activities: ["service-restoration", "livelihood-support", "community-healing"];
    };

    rebuilding: {
      duration: "6 months - 2 years";
      focus: "Full restoration and improvement";
      activities: ["infrastructure", "economic-recovery", "resilience-building"];
    };

    renewal: {
      duration: "Ongoing";
      focus: "Learning and growing stronger";
      activities: ["lesson-integration", "capacity-building", "culture-renewal"];
    };
  };

  // Support structures
  supportStructures: {
    ongoingCoordination: "Maintain coordination through recovery";
    resourceFlow: "Continued resource support";
    emotionalSupport: "Ongoing wellness support";
    documentation: "Continue documenting lessons";
  };
}
```

### 7.2 After-Action Review

```typescript
interface AfterActionReview {
  // Timing
  timing: {
    hotWashup: "Within 48 hours of crisis stabilization";
    fullReview: "Within 1 month of crisis end";
    deepAnalysis: "Within 3 months for major crises";
  };

  // Participants
  participants: {
    hotWashup: "Core response team";
    fullReview: "All response participants + affected members";
    deepAnalysis: "Include external perspectives";
  };

  // Questions
  questions: {
    whatHappened: "What actually occurred?";
    whatWorked: "What went well?";
    whatFailed: "What didn't work?";
    whyDifference: "Why the difference between plan and reality?";
    lessonsLearned: "What did we learn?";
    recommendations: "What should we do differently?";
  };

  // Outputs
  outputs: {
    report: "After-action report";
    lessonsDocument: "Documented lessons";
    planUpdates: "Updated plans and protocols";
    trainingUpdates: "Updated training needs";
    patternCapture: "Patterns for network library";
  };
}
```

---

## Appendix: Quick Reference

### Crisis Activation Checklist

**Level 2 (Alert):**
- [ ] Assess situation
- [ ] Notify crisis coordinator
- [ ] Activate response team
- [ ] Begin monitoring
- [ ] Prepare resources
- [ ] Notify network (awareness)

**Level 3 (Emergency):**
- [ ] Activate full community response
- [ ] Notify regional coordinator
- [ ] Begin resource mobilization
- [ ] Establish communication hub
- [ ] Begin situation reporting
- [ ] Request mutual aid if needed

**Level 4 (Major):**
- [ ] Activate regional coordination
- [ ] Notify network crisis council
- [ ] Network-wide resource call
- [ ] Establish coordination hub
- [ ] Deploy volunteers
- [ ] Continuous situation updates

**Level 5 (Catastrophe):**
- [ ] Activate all-network response
- [ ] Emergency unity council
- [ ] Extraordinary measures
- [ ] All available resources
- [ ] Maximum coordination
- [ ] External partnerships activated

### Key Contacts Template

| Role | Name | Primary Contact | Backup Contact |
|------|------|-----------------|----------------|
| Community Crisis Coordinator | | | |
| Resource Steward | | | |
| Communications Lead | | | |
| Wellness Coordinator | | | |
| Regional Coordinator | | | |
| Network Liaison | | | |

### Situation Report Template

```
SITUATION REPORT
================
Report #: ___  Date/Time: _______  Reporter: _______

SITUATION:
- Current status:
- Affected area/people:
- Key developments since last report:

IMPACTS:
- Casualties/injuries:
- Displacement:
- Infrastructure:
- Services:

RESPONSE:
- Actions taken:
- Resources deployed:
- External coordination:

NEEDS:
- Immediate needs:
- Upcoming needs:
- Resources requested:

NEXT STEPS:
- Planned actions:
- Decision points:

NEXT REPORT: _______________
```

---

*"In crisis, we discover who we truly are—not separate individuals struggling alone, but nodes in a living network, capable of extraordinary coordination when we remember our connection. The mycelium responds as one."*
