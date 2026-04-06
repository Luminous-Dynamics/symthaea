# Praxis Universal Taxonomy Design

## Design Principles

1. **Graph-first, not tree-first** — prerequisite DAG like Khan Academy, not rigid hierarchies like edX
2. **Standards-anchored** — many-to-many alignment to external frameworks like IXL
3. **Competency-layered** — CTDL-inspired competency framework as the cross-cutting layer
4. **Multi-scale** — works from a single exercise to a PhD program
5. **Internationally portable** — ISCED/EQF levels, not US-centric grade numbers
6. **Machine-readable** — JSON-LD compatible, SPARQL-queryable

## The Five-Layer Model

```
Layer 5: PATHWAYS        — Ordered sequences toward credentials (degree, cert, career)
Layer 4: CREDENTIALS     — Verifiable achievements (W3C VC, badges, diplomas, degrees)
Layer 3: COMPETENCIES    — What you can DO (skill + knowledge + context)
Layer 2: CONTENT         — Learning materials (lessons, courses, assessments, projects)
Layer 1: STANDARDS       — External anchor points (CCSS, NGSS, ISCED, CIP, ESCO, O*NET)
```

Each layer references the layers below it. Content nodes teach competencies. Competencies align to standards. Credentials require competencies. Pathways sequence content toward credentials.

## Layer 1: Standards (External Anchors)

### Supported Frameworks

| Framework | Scope | Levels | Machine-Readable |
|-----------|-------|--------|-----------------|
| **ISCED** (UNESCO) | Global education levels | 0-8 (Early childhood → Doctoral) | Yes (Wikidata P3984) |
| **ISCED-F** (UNESCO) | Fields of education | 11 broad → 29 narrow → 80 detailed | Yes (CSV) |
| **CCSS** | US K-12 Math + ELA | K-12 | Yes (CSP API) |
| **NGSS** | US K-12 Science | K-12 | Yes (CSP API) |
| **C3 Framework** | US K-12 Social Studies | K-12 | Partial |
| **ISTE** | Technology standards | K-12 + Educator | Yes (PDF) |
| **National Core Arts** | Arts standards | PreK-12 | Partial |
| **SHAPE America** | PE/Health standards | K-12 | Partial |
| **ACM CS2013/CC2020** | Computing curricula | Undergraduate | Yes (structured) |
| **ABET** | Engineering accreditation | Undergraduate | Partial |
| **CIP** (NCES) | US program classification | All levels | Yes (CSV) |
| **EQF** (EU) | European qualifications | 1-8 | Yes |
| **CEFR** | Language proficiency | A1-C2 | Yes |
| **ESCO** (EU) | Skills/occupations | All | Yes (API) |
| **O*NET** (US DOL) | Occupations/skills | All | Yes (API) |
| **SOC** (US BLS) | Occupation classification | All | Yes |
| **DDC/LCC** | Library classification | All | Yes |
| **IB** | International Baccalaureate | MYP + DP | Partial |
| **Cambridge** | Cambridge International | IGCSE + A-Level | Partial |

### ISCED Level Mapping (Universal)

| ISCED | Level Name | Our GradeLevel | Age Range | Bloom Typical |
|-------|-----------|----------------|-----------|---------------|
| 0 | Early childhood | PreK | 3-5 | Remember |
| 1 | Primary | Grade1-Grade5 | 6-11 | Understand |
| 2 | Lower secondary | Grade6-Grade8 | 12-14 | Apply |
| 3 | Upper secondary | Grade9-Grade12 | 15-18 | Analyze |
| 4 | Post-secondary non-tertiary | Certificate | 18+ | Apply |
| 5 | Short-cycle tertiary | Associate | 18-20 | Apply |
| 6 | Bachelor's | Undergraduate | 18-22 | Analyze |
| 7 | Master's | Graduate | 22-26 | Evaluate |
| 8 | Doctoral | Doctoral | 22-30+ | Create |

### ISCED-F Subject Mapping

| ISCED-F Code | Broad Field | Our Subject Areas |
|-------------|-------------|-------------------|
| 01 | Education | Education |
| 02 | Arts and Humanities | Literature, Philosophy, History, Arts |
| 03 | Social Sciences, Journalism | Political Science, Sociology, Economics |
| 04 | Business, Administration, Law | Business, Law |
| 05 | Natural Sciences, Mathematics | Mathematics, Physics, Chemistry, Biology |
| 06 | Information and Communication Technologies | Computer Science |
| 07 | Engineering, Manufacturing, Construction | Engineering |
| 08 | Agriculture, Forestry, Fisheries | Agriculture |
| 09 | Health and Welfare | Medicine, Psychology, Public Health |
| 10 | Services | Hospitality, Sports |

## Layer 2: Content (Learning Materials)

### Node Types (expanded)

| Type | Description | Examples |
|------|------------|---------|
| **Concept** | A foundational idea | "Variable", "Natural selection", "Supply and demand" |
| **Skill** | A learnable ability | "Solve quadratic equations", "Write a thesis statement" |
| **Topic** | A knowledge area | "Machine Learning", "Renaissance Art" |
| **Course** | A structured learning experience | "MIT 6.006", "AP Calculus BC" |
| **Module** | A unit within a course | "Week 3: Sorting Algorithms" |
| **Lesson** | A single learning session | "Merge Sort Implementation" |
| **Exercise** | A practice problem | "Sort this array using merge sort" |
| **Assessment** | A formal evaluation | "Midterm Exam", "Qualifying Exam" |
| **Project** | An applied creation | "Build a web scraper", "Dissertation" |
| **Resource** | A reference material | Textbook chapter, research paper, video |
| **Milestone** | A program checkpoint | "Dissertation Proposal Defense" |

### Edge Types (expanded)

| Type | Semantics | Strength | Example |
|------|----------|----------|---------|
| **Requires** | Hard prerequisite | 800-1000 | "Calculus I" requires "Pre-calculus" |
| **Recommends** | Soft prerequisite | 500-700 | "Machine Learning" recommends "Linear Algebra" |
| **LeadsTo** | Temporal sequence | 600-900 | Grade 5 Math leads to Grade 6 Math |
| **PartOf** | Composition | 1000 | "Module 3" is part of "CS101" |
| **Specializes** | Deeper treatment | 700 | "Organic Chemistry" specializes "Chemistry" |
| **AppliedIn** | Practical application | 600 | "Statistics" is applied in "Data Science" |
| **RelatedTo** | Conceptual link | 300-500 | "Physics" related to "Mathematics" |
| **AlternativeTo** | Interchangeable | 500 | "Python" alternative to "Java" (for intro CS) |
| **Corequisite** | Must be taken together | 800 | "Physics Lab" corequisite with "Physics I" |
| **CrossDisciplinary** | Cross-field bridge | 400-600 | "Bioinformatics" bridges "Biology" and "CS" |

## Layer 3: Competencies

### Structure (CTDL-inspired)

```
CompetencyFramework
  └── CompetencyDomain
       └── Competency
            ├── knowledgeStatements[]    — what you know
            ├── skillStatements[]        — what you can do
            ├── abilityStatements[]      — how well you can do it
            ├── bloomLevel              — cognitive complexity
            ├── webbDOK                 — depth of knowledge (1-4)
            ├── dreyfusLevel            — novice → expert (1-5)
            ├── standardAlignments[]    — CCSS/NGSS/ESCO/O*NET codes
            └── assessmentCriteria[]    — how to verify mastery
```

### Webb's Depth of Knowledge (DOK) — complements Bloom

| DOK | Level | Description | Examples |
|-----|-------|-------------|---------|
| 1 | Recall | Facts, definitions, simple procedures | "What is the formula for area of a circle?" |
| 2 | Skill/Concept | Requires some mental processing | "Classify these triangles by their angles" |
| 3 | Strategic Thinking | Reasoning, planning, evidence | "Design an experiment to test this hypothesis" |
| 4 | Extended Thinking | Complex, multi-step, novel | "Research and write a paper on climate change solutions" |

### Dreyfus Model (skill acquisition stages)

| Stage | Description | Praxis Mapping |
|-------|-------------|----------------|
| 1. Novice | Follows rules rigidly | Beginner, Remember/Understand |
| 2. Advanced Beginner | Recognizes patterns from experience | Intermediate, Apply |
| 3. Competent | Plans deliberately, manages complexity | Advanced, Analyze |
| 4. Proficient | Sees the big picture, intuitive decisions | Expert, Evaluate |
| 5. Expert | Transcends rules, deep intuitive mastery | Expert, Create |

## Layer 4: Credentials

### Types (from CTDL taxonomy)

| Category | Types | ISCED Level |
|----------|-------|-------------|
| **K-12** | High School Diploma, GED | 2-3 |
| **Certificates** | Professional Certificate, Micro-credential, Digital Badge | 4-5 |
| **Degrees** | Associate, Bachelor's, Master's, Doctorate | 5-8 |
| **Professional** | License (PE, CPA, MD), Certification (AWS, CompTIA) | 6-8 |
| **Informal** | Course Completion, Badge, Skill Verification | Any |

### Credential Requirements (ConditionProfile)

A credential can require any combination of:
- Other credentials (prerequisite degrees)
- Competencies (demonstrated skills)
- Assessments (passed exams)
- Learning opportunities (completed courses)
- Experience (years of work)

## Layer 5: Pathways

### Pathway Types

| Type | Description | Duration | Example |
|------|------------|----------|---------|
| **Grade Progression** | K-12 year-by-year | 13 years | "K-12 Mathematics" |
| **Degree Program** | University curriculum | 2-6 years | "BS Computer Science" |
| **Career Path** | Occupation preparation | Variable | "Software Engineer" |
| **Certification Track** | Professional credential | 3-12 months | "AWS Solutions Architect" |
| **Research Path** | PhD progression | 4-7 years | "PhD in Physics" |
| **Skill Path** | Focused skill building | 1-6 months | "Web Development Fundamentals" |
| **Cross-Disciplinary** | Spanning multiple fields | Variable | "Computational Biology" |
| **Lifelong** | Birth to death | Lifetime | "K → PhD → PostDoc → Faculty" |

## Subject Taxonomy (ISCED-F aligned, extended)

### Tier 1: Broad Fields (11)

```
01 Education
02 Arts & Humanities
03 Social Sciences
04 Business & Law
05 Natural Sciences & Mathematics
06 Information & Communication Technologies
07 Engineering & Manufacturing
08 Agriculture & Environment
09 Health & Welfare
10 Services & Recreation
11 Meta-Learning (Symthaea/Mycelix — our addition)
```

### Tier 2: Narrow Fields (~40)

```
01 Education
  01.1 Teaching & Curriculum
  01.2 Educational Technology
  01.3 Special Education

02 Arts & Humanities
  02.1 Arts (Visual, Performing, Music)
  02.2 Literature & Writing
  02.3 Philosophy & Ethics
  02.4 History & Archaeology
  02.5 Languages & Linguistics
  02.6 Religion & Theology

03 Social Sciences
  03.1 Economics
  03.2 Political Science & Governance
  03.3 Sociology & Anthropology
  03.4 Psychology
  03.5 Geography
  03.6 Communication & Media

04 Business & Law
  04.1 Business & Management
  04.2 Finance & Accounting
  04.3 Law & Legal Studies
  04.4 Public Administration

05 Natural Sciences & Mathematics
  05.1 Mathematics & Statistics
  05.2 Physics & Astronomy
  05.3 Chemistry
  05.4 Biology & Life Sciences
  05.5 Earth & Environmental Sciences

06 Information & Communication Technologies
  06.1 Computer Science (Theory)
  06.2 Software Engineering
  06.3 Data Science & AI
  06.4 Cybersecurity
  06.5 Information Systems

07 Engineering & Manufacturing
  07.1 Mechanical Engineering
  07.2 Electrical Engineering
  07.3 Civil Engineering
  07.4 Chemical Engineering
  07.5 Biomedical Engineering
  07.6 Aerospace Engineering

08 Agriculture & Environment
  08.1 Agriculture & Farming
  08.2 Environmental Science
  08.3 Forestry & Marine Science

09 Health & Welfare
  09.1 Medicine
  09.2 Nursing & Allied Health
  09.3 Public Health
  09.4 Pharmacy
  09.5 Social Work

10 Services & Recreation
  10.1 Physical Education & Sports
  10.2 Hospitality & Tourism
  10.3 Culinary Arts

11 Meta-Learning (Luminous Dynamics)
  11.1 Consciousness Computing (Symthaea)
  11.2 Decentralized Civic Infrastructure (Mycelix)
  11.3 Holographic Computing (HDC)
  11.4 Ethical AI & Value Alignment
```

## Data Sources Roadmap

### Currently Implemented (6 sources)

| Source | Coverage | Status |
|--------|----------|--------|
| CSP API | K-12 CCSS Math, ELA; NGSS Science | 1,527 nodes |
| CIP Taxonomy | 46 program families | Embedded |
| ACM CS2013 | 9 knowledge areas, 28 units | Embedded |
| MIT OCW | 200 courses, 8 departments | Live API |
| PhD Templates | 15 disciplines, 159 milestones | Embedded |
| Bridge Generator | Cross-level edges | Computed |

### Priority Additions

| Source | Coverage | Effort | Impact |
|--------|----------|--------|--------|
| **C3 Framework** | K-12 Social Studies | 1 day | Fills SocialStudies gap |
| **ISTE Standards** | K-12 Technology | 1 day | Fills Technology gap |
| **National Core Arts** | PreK-12 Arts | 1 day | Fills Arts gap |
| **SHAPE America** | K-12 PE/Health | 1 day | Fills PE gap |
| **CEFR Language** | A1-C2 proficiency | 1 day | Fills ForeignLanguage gap |
| **ESCO API** | EU skills/occupations | 2 days | Career pathway anchoring |
| **O*NET API** | US occupations | 2 days | Career pathway anchoring |
| **Wikidata SPARQL** | Academic concept ontology | 2 days | Cross-disciplinary links |
| **Credential Engine** | 1.4M credentials | 3 days | Credential layer |
| **More OCW depts** | 19 remaining MIT depts | 1 day | Fills undergrad gaps |

### Symthaea & Mycelix Courses (Field 11)

Self-referential curriculum about the system itself:

| Course | Level | Prerequisites | Nodes |
|--------|-------|--------------|-------|
| Consciousness Computing 101 | Undergraduate | None | ~20 |
| HDC Fundamentals | Undergraduate | Linear Algebra | ~15 |
| IIT/Phi Theory | Graduate | Information Theory | ~12 |
| Liquid Neural Networks | Graduate | Differential Equations | ~12 |
| Active Inference | Graduate | Bayesian Statistics | ~10 |
| Moral Algebra | Graduate | Ethics, HDC | ~15 |
| Symthaea Architecture | Graduate | All above | ~20 |
| Holochain Fundamentals | Undergraduate | Distributed Systems | ~15 |
| Mycelix Governance | Undergraduate | Political Science | ~12 |
| Consciousness Gating | Graduate | IIT/Phi, Governance | ~10 |
| Decentralized Education | Graduate | Holochain, Pedagogy | ~12 |

## Implementation Notes

### Node ID Convention

```
{source}:{framework}:{level}:{code}

Examples:
  csp:ccss:grade3:CCSS.Math.Content.3.OA.A.1
  ocw:mit:undergrad:6.006
  acm:cs2013:undergrad:AL/BasicAnalysis
  phd:template:doctoral:phd-cs/qualifying-exam
  ld:symthaea:graduate:consciousness-101/hdc-basics
  esco:skill:all:S1.1.1
```

### Competency ID Convention

```
comp:{framework}:{domain}:{code}

Examples:
  comp:ccss:math:3.OA.A
  comp:acm:cs:al-basic-analysis
  comp:esco:ict:S1.1.1.1
  comp:bloom:cognitive:analyze
  comp:dok:depth:3
```

### Edge Confidence Encoding

```
strength_permille:
  1000 = Absolute prerequisite (cannot proceed without)
  800-900 = Strong prerequisite (should complete first)
  600-700 = Recommended (helpful but not required)
  400-500 = Related (conceptually connected)
  200-300 = Tangential (might be interesting)
```
