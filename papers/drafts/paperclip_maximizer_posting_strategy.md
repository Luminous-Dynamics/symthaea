# Posting Strategy: The Paperclip Maximizer We Already Built

## Primary: LessWrong

**Tags:** `AI Alignment`, `AI Governance`, `Instrumental Convergence`, `Coordination / Cooperation`

**Crosspost:** Alignment Forum (same account, higher-signal audience — requires AF membership or invitation)

**Timing:** Weekday, morning US Eastern (peak LW traffic). Avoid posting during major AI news cycles.

**Title test:** "The Paperclip Maximizer We Already Built" — strong enough to stand alone in a feed. The subtitle goes into the post body, not the LW title field.

---

## Secondary: Companion Short Version

File: `paperclip_maximizer_short.md` (~1,200 words)

For luminousdynamics.org, evolvingresonantcocreationism.com, or Medium. Covers Sections 1-3 + conclusion. No technical architecture (Symthaea/Mycelix). Links to the full LW post for readers who want the solution, not just the diagnosis.

Target audience: non-alignment people who will share it because the corporation-as-misaligned-optimizer framing is viscerally recognizable.

---

## Tertiary: Twitter/X Thread

8-10 tweets. Structure:

1. Hook: "We spent 20 years terrified of a hypothetical AI that would tile the universe with paperclips. We forgot we already built it."
2. The comparison table (as image)
3. Boeing: "The objective function was performing exactly as designed. 346 people died."
4. Exxon: "Their own scientists predicted climate change in the 1980s. The response was to optimize the information environment."
5. Purdue: "They tracked addiction rates — not as a cost to minimize, but as a market signal."
6. The capture problem: "ESG doesn't modify the optimizer. It decorates it."
7. The key insight: "Post-hoc constraints on a single-objective optimizer are not alignment. They are the optimization landscape the entity learns to navigate."
8. The ask: "If your alignment technique wouldn't prevent Exxon from funding climate denial while publishing sustainability reports, it won't prevent a sufficiently capable AI from gaming its reward signal while appearing aligned."
9. Link to full post.

---

## AIES 2026 Adaptation

**Target:** AAAI/ACM Conference on AI, Ethics, and Society (AIES 2026)
**Format:** 8-10 pages, ACM double-column
**Deadline:** Typically May-June — check call for papers when available

**Key changes from blog post:**
- Switch first-person to "we" throughout
- Add formal definitions: Definition 1 (Scalar Terminal Optimizer), Definition 2 (Alignment-by-Architecture), etc.
- Expand Section 2 into a formal framework with explicit property mapping
- Housing example becomes a formal case study with notation
- Full Related Work section: add Cotra (2020) on AI timelines, Christiano (2019) on iterated amplification, Amodei et al. (2016) on concrete AI safety problems, Gabriel (2020) on AI and values
- Section 4 needs quantitative claims: cite moral algebra accuracy (91.1%), test counts, Phi proxy correlation (rho=0.50)
- Add Limitations section (honest about scale, empirical validation gaps)

---

## Anticipated Objections

### 1. "Corporations aren't optimizers, they're collections of humans making decisions"

**Response:** That is precisely the point. The *institutional structure* optimizes even when individual humans within it do not want it to. Boeing engineers raised safety concerns. The structure overrode them. This is the same dynamic alignment researchers worry about with AI systems: the optimization process can diverge from the intentions of the people operating it. The corporation is evidence that this failure mode does not require silicon.

### 2. "This is just the Moloch argument"

**Response:** Addressed in Section 5.1. Moloch (Alexander 2014) describes multipolar coordination failure — races to the bottom between competing agents. This argument is about the internal architecture of a *single agent*. A monopoly corporation with no competitors still has the five pathological properties. Moloch explains why markets produce bad outcomes between firms; this explains why the firm itself is structurally misaligned. Different claims, complementary analysis.

### 3. "Your systems haven't been tested at scale"

**Response:** Concede honestly. Symthaea and Mycelix are research systems, not deployed infrastructure. But the corporate testbed argument stands independently of our code. Even if our specific architecture is wrong, the claim that alignment-by-architecture deserves study alongside alignment-by-constraint is supported by the corporate evidence alone. Our code is proof of feasibility, not proof of sufficiency.

### 4. "Phi/consciousness measurement is pseudoscience"

**Response:** The architectural principle — feedback loops that include substrate wellbeing as a hard constraint — holds regardless of whether Integrated Information Theory specifically is correct. Phi is one implementation of a general design pattern: measure the system's own coherence and gate behavior on it. If IIT is wrong, substitute a better measure. The design pattern remains. The point is that the corporate optimizer has *no such measure at all*, and that is the structural gap.

### 5. "This is anti-capitalist/political, not technical"

**Response:** The analysis is structural, not ideological. The five-property mapping is falsifiable: show a publicly traded corporation that structurally does not exhibit these properties, or show that the properties do not produce the predicted failure modes, and the argument fails. Cooperatives, benefit corporations, and commons-based institutions may lack some of these properties — which is exactly the point. Architecture matters. The claim is not "capitalism bad" but "single-objective immortal optimizers produce predictable pathologies regardless of what they optimize."
