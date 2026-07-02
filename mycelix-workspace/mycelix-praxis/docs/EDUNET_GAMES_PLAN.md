# Praxis Educational Games Plan

## Evidence Base
- Game-based STEM learning: effect size g=0.624 (medium-large) over conventional instruction
- Algebra and ratio topics show greatest gains from game mechanics
- PhET simulations effective "regardless of gender or GPA, across diverse populations"
- DragonBox: in-game progress predicts later algebraic performance

## Key Insight
The learning IS the gameplay. Not gamification (points on top of boring content).
Interactive parameter manipulation with instant visual feedback is the gold standard.

## Technology Decision
- **SVG + Leptos signals** for 11 of 14 games (reactive DOM, no Canvas needed)
- **Canvas 2D** only for 3 games needing continuous animation (projectile, energy, Doppler)
- **No Bevy/WebGL** — binary size would go from 2.4MB to 44MB for features we don't need

## Top 5 Games (Ordered by Impact)

| Game | Topic | Marks | Technology | Days |
|------|-------|-------|------------|------|
| Parabola Explorer | Functions | 25 | SVG | 3-4 |
| Tangent Line Explorer | Calculus | 35 | SVG | 3-4 |
| Unit Circle Explorer | Trigonometry | 40 | SVG | 2-3 |
| Circuit Builder | Electricity | 35 | SVG+drag | 5-7 |
| Equilibrium Simulator | Chemistry | 25 | SVG | 3-4 |

## Architecture
- Games appear as "Explore" tab on the study page (alongside Learn/Examples/Practice/Pitfalls)
- Static game registry maps node IDs to game components (no schema changes)
- GraphRenderer is the keystone shared component (SVG viewBox, axes, grid, labels)
- Game completion feeds into existing ProgressStore mastery tracking

## Phases
- Phase 0: Shared infrastructure (GraphRenderer, SliderControl, DragHandle) — 5-7 days
- Phase 1: First 3 math games (parabola, tangent, unit circle) — 8-12 days
- Phase 2: Physics + more math (circuit, analytical geom, stats, equilibrium, projectile) — 15-20 days
- Phase 3: Remaining games (molecule builder, proof builder, equation balance) — 10-15 days

## CAPS Exam Context
- Mathematics pass rate fell from 69% to 64% (2024→2025)
- Physical Sciences distinctions dropped from 6,398 to 5,620
- Most common errors: parameter effects on graphs, derivative concepts, circuit analysis
- These are exactly the topics where interactive manipulation has the strongest evidence

## Full specification
See the comprehensive plan in memory for detailed specs on the Parabola Explorer
(SVG rendering, Leptos signal architecture, challenge mode, mobile touch interaction).
