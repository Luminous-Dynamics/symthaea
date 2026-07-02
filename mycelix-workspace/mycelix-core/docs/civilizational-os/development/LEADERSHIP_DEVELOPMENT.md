# Mycelix Facilitation & Leadership Development Curriculum

## Philosophy: Growing the Humans Who Grow the System

> "The most elegant systems design will fail without capable humans to steward them. The most capable humans will be limited without systems that support their wisdom. We must grow both together."

This curriculum develops the human capacities needed to make Mycelix communities thrive—facilitation, leadership, and the inner development that enables both.

---

## Architecture Alignment

> **Reference**: See [ARCHITECTURE_ALIGNMENT_GUIDE.md](../ARCHITECTURE_ALIGNMENT_GUIDE.md) for complete alignment standards.

### Epistemic Classification (LEM v2.0)

| Data Type | E-Tier | N-Tier | M-Tier | Description |
|-----------|--------|--------|--------|-------------|
| Learning Credential | E3 (Cryptographically Proven) | N2 (Network) | M2 (Persistent) | Verified competency attestation |
| Self-Assessment | E1 (Testimonial) | N0 (Personal) | M1-M2 | Personal reflection data |
| Peer Feedback | E2 (Privately Verifiable) | N1 (Communal) | M2 (Persistent) | Peer attestations |
| Mentor Attestation | E3 (Cryptographically Proven) | N1-N2 | M2 (Persistent) | Mentor-signed competency claims |
| Curriculum Content | E3-E4 | N2 (Network) | M3 (Foundational) | Canonical training materials |

### Token System Integration

- **CIV (Civic Standing)**: Earned through curriculum completion; higher tiers require minimum CIV thresholds
- **CGC (Civic Gifting Credit)**: Recognize teachers, mentors, and facilitators; fund learning circles
- **FLOW**: Optional compensation for professional curriculum development

### Governance Tier Mapping

| Curriculum Tier | Primary DAO Tier | Credential Authority |
|-----------------|-----------------|---------------------|
| Tier 1: Foundations | Local DAO | Community learning coordinator |
| Tier 2: Facilitation | Local/Sector DAO | Praxis credential system |
| Tier 3: Leadership | Regional DAO | Cross-community validation |
| Tier 4: Mastery | Global DAO | Network-wide recognition |

### MATL Integration (Mycelix Adaptive Trust Layer)

Leadership credentials integrate with MATL trust scoring:
- Facilitation credential → Enhanced trust for meeting facilitation contexts
- Conflict credential → Enhanced trust for mediation contexts
- Leadership credential → Enhanced trust for coordination contexts

### Key Institution References

- **Knowledge Council**: Oversees curriculum epistemic quality; validates Tier 4 credentials
- **Praxis Integration**: All credentials issued through Praxis hApp
- **Spiral hApp**: Developmental stage assessment for stage-appropriate learning paths

---

## Part 1: Leadership in Mycelix

### 1.1 Distributed Leadership Model

```typescript
interface DistributedLeadership {
  // Philosophy
  philosophy: {
    notHierarchical: "Leadership is not command and control";
    notLeaderless: "Communities need leadership, not leaders";
    distributed: "Leadership capacity is cultivated in many";
    situational: "Different contexts call for different leadership";
    rotational: "Roles rotate to develop capacity and prevent concentration";
  };

  // Leadership as function, not position
  leadershipFunctions: {
    vision: "Holding and articulating shared vision";
    facilitation: "Enabling group processes";
    coordination: "Orchestrating collective action";
    caregiving: "Tending to community wellbeing";
    bridging: "Connecting across differences";
    challenging: "Holding community accountable";
    grounding: "Anchoring in values and wisdom";
    pioneering: "Exploring new possibilities";
  };

  // Anyone can lead
  accessibleLeadership: {
    training: "Leadership development for all who want it";
    opportunity: "Rotation creates opportunities";
    support: "Support for emerging leaders";
    mentorship: "Experienced leaders mentor emerging";
  };
}
```

### 1.2 Developmental Approach to Leadership

```typescript
interface DevelopmentalLeadership {
  // Stage-appropriate leadership
  stageAppropriate: {
    traditional: {
      leadershipStyle: "Clear hierarchy, respected roles";
      development: "Mentorship within established structures";
      challenge: "Expanding beyond single authority";
    };

    modern: {
      leadershipStyle: "Meritocratic, goal-oriented";
      development: "Skills training, measurable growth";
      challenge: "Including multiple perspectives";
    };

    postmodern: {
      leadershipStyle: "Facilitative, consensus-oriented";
      development: "Sensitivity training, process skills";
      challenge: "Making decisions and taking action";
    };

    integral: {
      leadershipStyle: "Adaptive, context-appropriate";
      development: "Multi-perspective capacity building";
      challenge: "Integrating all prior stages";
    };
  };

  // Growing through stages
  growthSupport: {
    recognition: "Recognize current developmental stage";
    stretching: "Provide appropriate challenges";
    support: "Support through discomfort of growth";
    integration: "Help integrate new capacities";
  };
}
```

---

## Part 2: Core Curriculum

### 2.1 Curriculum Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     LEADERSHIP DEVELOPMENT                       │
│                         CURRICULUM                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TIER 1: FOUNDATIONS (All Members)                              │
│  ├── Self-Awareness & Personal Mastery                          │
│  ├── Communication & Listening                                  │
│  ├── Collaboration Basics                                        │
│  └── Mycelix Citizenship                                        │
│                                                                  │
│  TIER 2: FACILITATION (Circle/Role Holders)                     │
│  ├── Meeting Facilitation                                       │
│  ├── Decision Facilitation                                      │
│  ├── Conflict Navigation                                        │
│  └── Process Design                                             │
│                                                                  │
│  TIER 3: LEADERSHIP (Coordinators & Leaders)                    │
│  ├── Adaptive Leadership                                        │
│  ├── Systems Thinking                                           │
│  ├── Power & Responsibility                                     │
│  └── Community Stewarding                                       │
│                                                                  │
│  TIER 4: MASTERY (Senior Leaders & Mentors)                     │
│  ├── Leadership of Leaders                                      │
│  ├── Cultural Architecture                                      │
│  ├── Network Leadership                                         │
│  └── Legacy & Succession                                        │
│                                                                  │
│  ONGOING: INNER DEVELOPMENT                                      │
│  ├── Shadow Work                                                │
│  ├── Developmental Growth                                       │
│  └── Contemplative Practice                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Tier 1: Foundations

For all community members.

```typescript
interface FoundationsTier {
  // Module 1: Self-Awareness & Personal Mastery
  selfAwareness: {
    objectives: [
      "Understand own patterns, triggers, and tendencies",
      "Develop emotional intelligence",
      "Cultivate self-regulation",
      "Build personal practices for presence"
    ];

    content: {
      selfAssessment: "Personality, values, strengths assessments";
      emotionalIntelligence: "Recognizing and managing emotions";
      triggers: "Identifying and working with triggers";
      practices: "Meditation, journaling, reflection";
    };

    format: "4 sessions + ongoing practice";
    assessment: "Self-reflection journal";
  };

  // Module 2: Communication & Listening
  communication: {
    objectives: [
      "Listen deeply and empathically",
      "Speak clearly and authentically",
      "Navigate difficult conversations",
      "Give and receive feedback"
    ];

    content: {
      deepListening: "Active and empathic listening";
      nvc: "Nonviolent Communication basics";
      difficultConversations: "Staying present in tension";
      feedback: "Giving and receiving feedback";
    };

    format: "4 sessions + practice pairs";
    assessment: "Practice demonstration";
  };

  // Module 3: Collaboration Basics
  collaboration: {
    objectives: [
      "Participate effectively in groups",
      "Contribute to collective intelligence",
      "Navigate group dynamics",
      "Support group process"
    ];

    content: {
      groupDynamics: "Understanding how groups work";
      roles: "Playing different roles effectively";
      contribution: "Making valuable contributions";
      support: "Supporting the group process";
    };

    format: "3 sessions + group practice";
    assessment: "Peer feedback";
  };

  // Module 4: Mycelix Citizenship
  citizenship: {
    objectives: [
      "Understand Mycelix philosophy and values",
      "Navigate community systems effectively",
      "Exercise rights and responsibilities",
      "Contribute to community health"
    ];

    content: {
      philosophy: "Mycelix vision and principles";
      systems: "How community systems work";
      participation: "Effective participation";
      contribution: "Ways to contribute";
    };

    format: "3 sessions (part of onboarding)";
    assessment: "Participation in community";
  };
}
```

### 2.3 Tier 2: Facilitation

For those holding circle roles or facilitating processes.

```typescript
interface FacilitationTier {
  // Module 1: Meeting Facilitation
  meetingFacilitation: {
    objectives: [
      "Design and facilitate effective meetings",
      "Hold space for diverse participation",
      "Manage time and energy",
      "Navigate common challenges"
    ];

    content: {
      design: "Meeting design principles";
      opening: "Creating good beginnings";
      process: "Guiding discussion and decision";
      challenges: "Handling disruptions, conflicts, stuck points";
      closing: "Effective closures";
    };

    format: "6 sessions + observed practice";
    assessment: "Facilitate a real meeting with feedback";
  };

  // Module 2: Decision Facilitation
  decisionFacilitation: {
    objectives: [
      "Facilitate various decision-making processes",
      "Ensure inclusive participation",
      "Navigate to clear decisions",
      "Handle dissent and concerns"
    ];

    content: {
      decisionTypes: "When to use which process";
      consensus: "Facilitating consensus";
      consent: "Facilitating consent";
      voting: "Facilitating voting processes";
      concerns: "Working with objections";
    };

    format: "6 sessions + observed practice";
    assessment: "Facilitate decision process with feedback";
  };

  // Module 3: Conflict Navigation
  conflictNavigation: {
    objectives: [
      "Recognize and respond to conflict early",
      "Facilitate difficult conversations",
      "Support conflict transformation",
      "Know when to escalate"
    ];

    content: {
      conflictTheory: "Understanding conflict";
      earlyIntervention: "Catching conflict early";
      facilitatingDialogue: "Facilitating between parties";
      transformation: "Moving from conflict to growth";
      escalation: "When and how to escalate";
    };

    format: "8 sessions + role-play practice";
    assessment: "Navigate simulated conflict";
  };

  // Module 4: Process Design
  processDesign: {
    objectives: [
      "Design participatory processes",
      "Match process to purpose",
      "Adapt processes to context",
      "Evaluate and improve processes"
    ];

    content: {
      principles: "Process design principles";
      methods: "Range of participatory methods";
      adaptation: "Adapting to context";
      evaluation: "Assessing process effectiveness";
    };

    format: "6 sessions + design project";
    assessment: "Design and facilitate a process";
  };
}
```

### 2.4 Tier 3: Leadership

For community coordinators and leaders.

```typescript
interface LeadershipTier {
  // Module 1: Adaptive Leadership
  adaptiveLeadership: {
    objectives: [
      "Diagnose adaptive vs. technical challenges",
      "Lead when there are no easy answers",
      "Mobilize others to tackle challenges",
      "Manage self in leadership pressure"
    ];

    content: {
      diagnosis: "Distinguishing challenge types";
      mobilizing: "Getting others engaged";
      selfManagement: "Managing self under pressure";
      experimentation: "Leading through experimentation";
    };

    format: "8 sessions + real challenge application";
    assessment: "Lead an adaptive challenge";
  };

  // Module 2: Systems Thinking
  systemsThinking: {
    objectives: [
      "See systems and their dynamics",
      "Understand feedback loops and leverage points",
      "Navigate complex systems",
      "Lead systemic change"
    ];

    content: {
      systemsBasics: "Systems thinking fundamentals";
      dynamics: "Understanding system dynamics";
      leverage: "Finding leverage points";
      change: "Leading systemic change";
    };

    format: "6 sessions + systems mapping project";
    assessment: "Systems analysis and intervention design";
  };

  // Module 3: Power & Responsibility
  powerResponsibility: {
    objectives: [
      "Understand power dynamics",
      "Use power responsibly",
      "Distribute and share power",
      "Navigate the ethics of leadership"
    ];

    content: {
      powerAnalysis: "Understanding power";
      responsibleUse: "Using power well";
      distribution: "Distributing power";
      ethics: "Ethical leadership";
    };

    format: "6 sessions + reflection project";
    assessment: "Power autobiography and commitment";
  };

  // Module 4: Community Stewarding
  communityStewardship: {
    objectives: [
      "Hold long-term community wellbeing",
      "Navigate community lifecycle",
      "Build community resilience",
      "Develop next generation of leaders"
    ];

    content: {
      stewardship: "What it means to steward community";
      lifecycle: "Community development stages";
      resilience: "Building community resilience";
      succession: "Developing future leaders";
    };

    format: "6 sessions + stewardship project";
    assessment: "Stewardship plan and practice";
  };
}
```

### 2.5 Tier 4: Mastery

For senior leaders and mentors.

```typescript
interface MasteryTier {
  // Module 1: Leadership of Leaders
  leadershipOfLeaders: {
    objectives: [
      "Develop and mentor other leaders",
      "Create conditions for leadership to emerge",
      "Build leadership culture",
      "Let go while staying engaged"
    ];

    content: {
      mentoring: "Effective mentorship";
      conditions: "Creating generative conditions";
      culture: "Building leadership culture";
      lettingGo: "Stepping back skillfully";
    };

    format: "6 sessions + mentoring practice";
    assessment: "Mentor an emerging leader";
  };

  // Module 2: Cultural Architecture
  culturalArchitecture: {
    objectives: [
      "Understand culture as living system",
      "Shape culture intentionally",
      "Navigate cultural change",
      "Preserve and evolve cultural wisdom"
    ];

    content: {
      cultureUnderstanding: "How culture works";
      shapingCulture: "Intentional culture design";
      culturalChange: "Leading cultural change";
      preservation: "Preserving what matters";
    };

    format: "6 sessions + cultural project";
    assessment: "Cultural intervention with reflection";
  };

  // Module 3: Network Leadership
  networkLeadership: {
    objectives: [
      "Lead in federated contexts",
      "Navigate multi-community coordination",
      "Build network-level capacity",
      "Represent community in network"
    ];

    content: {
      networkDynamics: "How networks work";
      multiCommunity: "Cross-community leadership";
      networkCapacity: "Building network capacity";
      representation: "Representing community well";
    };

    format: "6 sessions + network engagement";
    assessment: "Network leadership project";
  };

  // Module 4: Legacy & Succession
  legacySuccession: {
    objectives: [
      "Plan for leadership transitions",
      "Prepare community for own departure",
      "Leave healthy legacy",
      "Support community through transition"
    ];

    content: {
      transitionPlanning: "Planning leadership transitions";
      preparation: "Preparing community and successors";
      legacy: "Creating positive legacy";
      support: "Supporting healthy transitions";
    };

    format: "6 sessions + succession planning";
    assessment: "Complete succession plan";
  };
}
```

---

## Part 3: Inner Development Track

### 3.1 Shadow Work

```typescript
interface ShadowWork {
  // What is shadow work
  definition: {
    shadow: "Parts of ourselves we've rejected or hidden";
    why: "Unexamined shadows distort our leadership";
    goal: "Integrate shadow for more whole leadership";
  };

  // Program
  program: {
    // Level 1: Shadow Awareness
    level1: {
      focus: "Recognize your shadow patterns";
      content: [
        "What is shadow?",
        "Common leader shadows",
        "Identifying your shadows",
        "Triggers as shadow doorways"
      ];
      practices: [
        "trigger-tracking",
        "projection-work",
        "feedback-invitation"
      ];
      format: "4 sessions + ongoing practice";
    };

    // Level 2: Shadow Dialogue
    level2: {
      focus: "Engage with shadow parts";
      content: [
        "Dialoguing with shadow",
        "Understanding shadow gifts",
        "Shadow in relationship",
        "Shadow in leadership"
      ];
      practices: [
        "voice-dialogue",
        "journaling",
        "body-work"
      ];
      format: "4 sessions + ongoing practice";
    };

    // Level 3: Shadow Integration
    level3: {
      focus: "Integrate shadow into whole self";
      content: [
        "Integration practices",
        "Shadow as teacher",
        "Ongoing shadow work",
        "Supporting others' shadow work"
      ];
      practices: [
        "integration-rituals",
        "ongoing-awareness",
        "peer-support"
      ];
      format: "4 sessions + ongoing practice";
    };
  };
}
```

### 3.2 Developmental Growth

Integrated with Spiral hApp.

```typescript
interface DevelopmentalGrowth {
  // Developmental awareness
  awareness: {
    objectives: [
      "Understand developmental stages",
      "Recognize own stage and growth edge",
      "Understand others' developmental perspectives",
      "Lead across stages"
    ];

    content: {
      stages: "Understanding developmental stages";
      selfAssessment: "Identifying own stage and edge";
      empathy: "Seeing from other stages";
      leadership: "Leading across stages";
    };
  };

  // Growth support
  growthSupport: {
    edgeIdentification: "Identify current growth edge";
    challenges: "Appropriate challenges for growth";
    support: "Support through developmental transitions";
    integration: "Integrate new capacities";
  };

  // Integration with Spiral hApp
  spiralIntegration: {
    assessment: "Stage assessment through Spiral";
    tracking: "Track developmental progress";
    resources: "Stage-appropriate resources";
    community: "Connect with developmental peers";
  };
}
```

### 3.3 Contemplative Practice

```typescript
interface ContemplativePractice {
  // Why contemplative practice
  why: {
    presence: "Cultivate capacity for presence";
    clarity: "Develop clarity of perception";
    equanimity: "Build capacity for difficult situations";
    compassion: "Deepen compassion and empathy";
    wisdom: "Access deeper wisdom";
  };

  // Practice offerings
  practices: {
    meditation: {
      types: ["concentration", "awareness", "loving-kindness", "body-based"];
      format: "Ongoing, individual and group";
      instruction: "Available through Praxis";
    };

    contemplativeDialogue: {
      description: "Dialogue with contemplative dimension";
      examples: ["Bohm-dialogue", "council", "collective-inquiry"];
      format: "Regular community offerings";
    };

    retreats: {
      description: "Extended contemplative time";
      types: ["silent-retreat", "nature-retreat", "leadership-retreat"];
      format: "Periodic community offerings";
    };
  };

  // Integration with leadership
  leadershipIntegration: {
    beforeMeetings: "Brief centering practice";
    inChallenges: "Grounding in difficult moments";
    ongoing: "Regular practice supports leadership";
  };
}
```

---

## Part 4: Learning Methods

### 4.1 Learning Principles

```typescript
interface LearningPrinciples {
  // Adult learning principles
  adultLearning: {
    experiential: "Learn by doing, not just hearing";
    relevant: "Connect to real challenges and needs";
    selfdirected: "Learners drive their own development";
    reflective: "Integrate through reflection";
    social: "Learn with and from others";
    developmental: "Meet learners where they are";
  };

  // Embodied learning
  embodiedLearning: {
    notJustCognitive: "Engage whole person";
    practice: "Skills developed through practice";
    somatic: "Body as site of learning";
    emotional: "Engage emotions";
  };

  // Transformative learning
  transformativeLearning: {
    disorientation: "Challenges to existing views";
    reflection: "Deep reflection on assumptions";
    dialogue: "Dialogue with others";
    action: "Testing new perspectives in action";
    integration: "Integrating new ways of being";
  };
}
```

### 4.2 Learning Formats

```typescript
interface LearningFormats {
  // Cohort learning
  cohortLearning: {
    description: "Learn with a peer group";
    benefits: [
      "peer-support",
      "accountability",
      "diverse-perspectives",
      "relationship-building"
    ];
    structure: "Regular sessions over weeks/months";
  };

  // Mentorship
  mentorship: {
    description: "One-on-one developmental relationship";
    elements: [
      "regular-meetings",
      "feedback-and-guidance",
      "challenge-support",
      "role-modeling"
    ];
    matching: "Match based on goals and chemistry";
  };

  // Apprenticeship
  apprenticeship: {
    description: "Learn by doing alongside experienced practitioner";
    elements: [
      "observation",
      "guided-practice",
      "increasing-responsibility",
      "reflection"
    ];
    domains: ["facilitation", "conflict-resolution", "leadership"];
  };

  // Action learning
  actionLearning: {
    description: "Learn through working on real challenges";
    elements: [
      "real-challenge",
      "action-steps",
      "reflection",
      "peer-support"
    ];
    structure: "Learning sets + real projects";
  };

  // Self-directed learning
  selfDirected: {
    description: "Individual learning with support";
    elements: [
      "learning-plan",
      "curated-resources",
      "practice-opportunities",
      "reflection"
    ];
    support: "Learning coach or mentor";
  };

  // Community of practice
  communityOfPractice: {
    description: "Learn with practitioners of same domain";
    elements: [
      "regular-gatherings",
      "case-discussions",
      "skill-sharing",
      "peer-support"
    ];
    examples: [
      "facilitators-circle",
      "leaders-circle",
      "conflict-practitioners"
    ];
  };
}
```

### 4.3 Assessment and Recognition

```typescript
interface AssessmentRecognition {
  // Assessment approaches
  assessment: {
    notGrades: "Assessment for learning, not ranking";
    selfAssessment: "Learners assess own progress";
    peerFeedback: "Feedback from learning peers";
    mentorFeedback: "Feedback from mentors/teachers";
    demonstration: "Demonstrate capability in practice";
    reflection: "Reflective integration";
  };

  // Recognition
  recognition: {
    credentials: {
      tier1: "Foundation skills credential";
      tier2: "Facilitation credential";
      tier3: "Leadership credential";
      tier4: "Master facilitator/leader credential";
    };

    credentialIntegration: {
      hApp: "Praxis credentials";
      requirements: "Demonstrated competence + peer/mentor attestation";
      display: "Visible in community profile";
    };
  };

  // Ongoing development
  ongoingDevelopment: {
    continuingEducation: "Ongoing learning expected";
    practiceGroups: "Regular skill practice";
    supervision: "Ongoing supervision for practitioners";
    renewal: "Periodic credential renewal";
  };
}
```

---

## Part 5: Leadership Pathways

### 5.1 Facilitation Pathway

```typescript
interface FacilitationPathway {
  // Stages
  stages: {
    // Stage 1: Apprentice Facilitator
    apprentice: {
      focus: "Learn basics through observation and assisted practice";
      activities: [
        "complete-tier-1-foundations",
        "observe-experienced-facilitators",
        "co-facilitate-with-mentor",
        "begin-facilitation-curriculum"
      ];
      duration: "6-12 months";
      recognition: "Apprentice facilitator designation";
    };

    // Stage 2: Practicing Facilitator
    practicing: {
      focus: "Develop skills through regular practice";
      activities: [
        "complete-tier-2-facilitation",
        "facilitate-regular-meetings",
        "receive-ongoing-feedback",
        "participate-in-facilitators-circle"
      ];
      duration: "1-2 years";
      recognition: "Facilitation credential";
    };

    // Stage 3: Skilled Facilitator
    skilled: {
      focus: "Handle complex facilitations, begin mentoring";
      activities: [
        "facilitate-complex-processes",
        "design-custom-processes",
        "mentor-apprentice-facilitators",
        "ongoing-learning"
      ];
      duration: "Ongoing";
      recognition: "Senior facilitator designation";
    };

    // Stage 4: Master Facilitator
    master: {
      focus: "Facilitation mastery, develop other facilitators";
      activities: [
        "handle-highest-complexity",
        "innovate-facilitation-methods",
        "train-facilitators",
        "contribute-to-facilitation-field"
      ];
      duration: "Ongoing";
      recognition: "Master facilitator credential";
    };
  };
}
```

### 5.2 Community Leadership Pathway

```typescript
interface LeadershipPathway {
  // Stages
  stages: {
    // Stage 1: Engaged Member
    engagedMember: {
      focus: "Active participation and contribution";
      activities: [
        "complete-tier-1-foundations",
        "participate-actively",
        "take-on-small-responsibilities",
        "develop-relationships"
      ];
      recognition: "Engaged member status";
    };

    // Stage 2: Role Holder
    roleHolder: {
      focus: "Hold defined roles effectively";
      activities: [
        "hold-circle-role",
        "begin-tier-2-facilitation",
        "receive-role-mentorship",
        "participate-in-governance"
      ];
      recognition: "Role holder status";
    };

    // Stage 3: Community Coordinator
    coordinator: {
      focus: "Coordinate community functions";
      activities: [
        "complete-tier-2-and-begin-tier-3",
        "coordinate-community-area",
        "participate-in-leadership-circle",
        "mentor-role-holders"
      ];
      recognition: "Coordinator status";
    };

    // Stage 4: Community Steward
    steward: {
      focus: "Steward overall community wellbeing";
      activities: [
        "complete-tier-3-and-begin-tier-4",
        "hold-senior-leadership",
        "guide-community-direction",
        "develop-next-generation"
      ];
      recognition: "Steward status";
    };

    // Stage 5: Elder/Mentor
    elder: {
      focus: "Wisdom-holder and mentor";
      activities: [
        "complete-tier-4",
        "mentor-leaders",
        "hold-community-wisdom",
        "advise-on-significant-matters"
      ];
      recognition: "Elder/mentor status";
    };
  };
}
```

### 5.3 Specialized Pathways

```typescript
interface SpecializedPathways {
  // Conflict practitioner
  conflictPractitioner: {
    path: [
      "foundation-training",
      "conflict-navigation-tier-2",
      "advanced-conflict-transformation-training",
      "supervised-practice",
      "conflict-practitioner-credential",
      "ongoing-supervision-and-development"
    ];
    recognition: "Conflict practitioner credential";
  };

  // Health steward
  healthSteward: {
    path: [
      "foundation-training",
      "community-health-training",
      "health-circle-participation",
      "health-steward-designation"
    ];
    recognition: "Health steward designation";
  };

  // Knowledge steward
  knowledgeSteward: {
    path: [
      "foundation-training",
      "knowledge-management-training",
      "knowledge-circle-participation",
      "knowledge-steward-designation"
    ];
    recognition: "Knowledge steward designation";
  };

  // Crisis responder
  crisisResponder: {
    path: [
      "foundation-training",
      "crisis-response-training",
      "psychological-first-aid",
      "crisis-exercises",
      "crisis-responder-credential"
    ];
    recognition: "Crisis responder credential";
  };
}
```

---

## Part 6: Program Administration

### 6.1 Learning Infrastructure

```typescript
interface LearningInfrastructure {
  // Praxis integration
  eduNet: {
    courses: "Curriculum delivered through Praxis";
    credentials: "Credentials issued through Praxis";
    tracking: "Progress tracked in Praxis";
    resources: "Learning resources in Praxis";
  };

  // Learning support roles
  roles: {
    learningCoordinator: {
      responsibility: "Coordinate learning programs";
      activities: [
        "schedule-offerings",
        "recruit-teachers",
        "support-learners",
        "track-progress"
      ];
    };

    teacherFacilitator: {
      responsibility: "Deliver learning experiences";
      requirements: "Expertise + teaching skills";
      development: "Train-the-trainer programs";
    };

    mentor: {
      responsibility: "Support individual development";
      requirements: "Experience + mentoring skills";
      matching: "Match with mentees";
    };
  };

  // Learning communities
  learningCommunities: {
    cohorts: "Peer learning groups";
    practiceCircles: "Domain-specific practice groups";
    alumni: "Graduated learner networks";
  };
}
```

### 6.2 Quality Assurance

```typescript
interface LearningQuality {
  // Curriculum review
  curriculumReview: {
    frequency: "Annual";
    reviewers: "Teachers + learners + leaders";
    criteria: [
      "learning-outcomes-achieved",
      "relevance-to-community-needs",
      "pedagogical-effectiveness",
      "accessibility-and-inclusion"
    ];
    output: "Updated curriculum";
  };

  // Teacher development
  teacherDevelopment: {
    training: "Train-the-trainer programs";
    feedback: "Regular feedback on teaching";
    community: "Teacher community of practice";
    support: "Ongoing support and development";
  };

  // Learner feedback
  learnerFeedback: {
    endOfModule: "Feedback at end of each module";
    endOfProgram: "Comprehensive feedback at end";
    followUp: "Follow-up after application";
  };
}
```

---

## Part 7: Cross-Community Learning

### 7.1 Network-Level Programs

```typescript
interface NetworkPrograms {
  // Train-the-trainer
  trainTheTrainer: {
    purpose: "Develop teachers for each community";
    content: "Teaching skills + curriculum content";
    delivery: "Network-level training events";
    outcome: "Certified teachers in each community";
  };

  // Advanced programs
  advancedPrograms: {
    purpose: "Advanced development beyond single community";
    examples: [
      "master-facilitator-program",
      "transformative-justice-intensive",
      "leadership-retreat"
    ];
    delivery: "Network-wide offerings";
  };

  // Exchange programs
  exchangePrograms: {
    purpose: "Learn from other communities";
    format: "Spend time learning in another community";
    areas: ["facilitation", "governance", "economics", "conflict"];
    coordination: "Diplomat hApp";
  };
}
```

### 7.2 Sharing & Collaboration

```typescript
interface LearningSharing {
  // Curriculum sharing
  curriculumSharing: {
    sharedCurriculum: "Core curriculum shared across network";
    localAdaptation: "Communities adapt to local context";
    contributions: "Communities contribute innovations";
    coordination: "Network learning circle";
  };

  // Teacher network
  teacherNetwork: {
    purpose: "Connect teachers across communities";
    activities: [
      "peer-learning",
      "resource-sharing",
      "problem-solving",
      "innovation"
    ];
    platform: "Network community of practice";
  };

  // Research & innovation
  researchInnovation: {
    purpose: "Advance learning practices";
    activities: [
      "action-research",
      "pilot-new-approaches",
      "document-learnings",
      "share-innovations"
    ];
    coordination: "Network learning circle";
  };
}
```

---

## Appendix: Quick Reference

### Curriculum Summary

| Tier | Audience | Modules | Duration |
|------|----------|---------|----------|
| 1: Foundations | All members | Self-Awareness, Communication, Collaboration, Citizenship | 14 sessions |
| 2: Facilitation | Role holders | Meeting, Decision, Conflict, Process | 26 sessions |
| 3: Leadership | Coordinators | Adaptive, Systems, Power, Stewardship | 26 sessions |
| 4: Mastery | Senior leaders | Leadership of Leaders, Culture, Network, Legacy | 24 sessions |
| Inner Development | All (optional) | Shadow Work, Developmental Growth, Contemplative Practice | Ongoing |

### Credential Requirements

| Credential | Requirements |
|------------|--------------|
| Tier 1 Complete | All Tier 1 modules + reflection |
| Facilitation Credential | Tier 1 + Tier 2 + 10 facilitated meetings + mentor attestation |
| Leadership Credential | Facilitation + Tier 3 + leadership experience + mentor attestation |
| Master Credential | Leadership + Tier 4 + significant contribution + peer attestation |

### Mentor Matching Template

```markdown
# Mentor-Mentee Matching

## Mentee Information
- Name:
- Current role:
- Learning goals:
- Preferred meeting frequency:
- Learning style:

## Mentor Preferences
- Domain expertise sought:
- Personality preferences:
- Availability requirements:

## Matching Criteria
- Goal alignment
- Style compatibility
- Availability match
- Relationship chemistry
```

---

*"The greatest gift a community can give its members is the opportunity to grow into their full potential. The greatest gift leaders can give their community is to develop the leaders who will come after them. In Mycelix, we grow people as carefully as we grow systems."*
