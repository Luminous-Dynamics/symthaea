//! Deterministic resource acquisition and crafting fixture (GameBench F5).
//!
//! The environment models scarce gatherable piles, travel, a crafting station,
//! recipe dependencies, and a bounded action budget. Exact planning remains a
//! crate-private evaluator diagnostic.

use std::collections::{HashMap, HashSet, VecDeque};
use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::{Action, Position, TerminalStatus};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ResourceKind {
    Wood,
    Stone,
    Plank,
    Tool,
}

impl ResourceKind {
    pub const fn index(self) -> usize {
        match self {
            Self::Wood => 0,
            Self::Stone => 1,
            Self::Plank => 2,
            Self::Tool => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourcePile {
    pub kind: ResourceKind,
    pub position: Position,
    pub amount: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ingredient {
    pub kind: ResourceKind,
    pub amount: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecipeSpec {
    pub output: ResourceKind,
    pub output_amount: u32,
    pub inputs: Vec<Ingredient>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CraftingSpecError {
    ZeroDimension,
    ZeroActionLimit,
    ZeroTargetAmount,
    PositionOutOfBounds(&'static str),
    DuplicateWall,
    DuplicatePilePosition,
    ZeroPileAmount,
    WallOverlapsObject(&'static str),
    WorkbenchOverlapsPile,
    ResourceAmountOverflow,
    DuplicateRecipeOutput,
    ZeroRecipeOutput,
    EmptyRecipe,
    ZeroIngredientAmount,
    DuplicateIngredient,
    RecipeCycle,
}

impl Display for CraftingSpecError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "crafting map dimensions must be nonzero"),
            Self::ZeroActionLimit => write!(f, "crafting action limit must be nonzero"),
            Self::ZeroTargetAmount => write!(f, "crafting target amount must be nonzero"),
            Self::PositionOutOfBounds(name) => write!(f, "{name} is outside the map"),
            Self::DuplicateWall => write!(f, "crafting wall list contains duplicates"),
            Self::DuplicatePilePosition => write!(f, "resource pile positions must be unique"),
            Self::ZeroPileAmount => write!(f, "resource pile amount must be nonzero"),
            Self::WallOverlapsObject(name) => write!(f, "wall overlaps {name}"),
            Self::WorkbenchOverlapsPile => write!(f, "workbench cannot overlap a resource pile"),
            Self::ResourceAmountOverflow => write!(f, "total resource amount exceeds inventory representation"),
            Self::DuplicateRecipeOutput => write!(f, "recipe outputs must be unique"),
            Self::ZeroRecipeOutput => write!(f, "recipe output amount must be nonzero"),
            Self::EmptyRecipe => write!(f, "recipe must have at least one input"),
            Self::ZeroIngredientAmount => write!(f, "recipe ingredient amount must be nonzero"),
            Self::DuplicateIngredient => write!(f, "recipe ingredient kinds must be unique"),
            Self::RecipeCycle => write!(f, "recipe dependency graph must be acyclic"),
        }
    }
}

impl Error for CraftingSpecError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CraftingSpec {
    width: u16,
    height: u16,
    walls: HashSet<Position>,
    piles: Vec<ResourcePile>,
    workbench: Position,
    recipes: HashMap<ResourceKind, RecipeSpec>,
    start: Position,
    target: ResourceKind,
    target_amount: u32,
    max_actions: u32,
}

impl CraftingSpec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        width: u16,
        height: u16,
        walls: Vec<Position>,
        piles: Vec<ResourcePile>,
        workbench: Position,
        recipes: Vec<RecipeSpec>,
        start: Position,
        target: ResourceKind,
        target_amount: u32,
        max_actions: u32,
    ) -> Result<Self, CraftingSpecError> {
        if width == 0 || height == 0 {
            return Err(CraftingSpecError::ZeroDimension);
        }
        if max_actions == 0 {
            return Err(CraftingSpecError::ZeroActionLimit);
        }
        if target_amount == 0 {
            return Err(CraftingSpecError::ZeroTargetAmount);
        }

        let walls_set: HashSet<_> = walls.iter().copied().collect();
        if walls_set.len() != walls.len() {
            return Err(CraftingSpecError::DuplicateWall);
        }

        for (name, position) in [
            ("start", start),
            ("workbench", workbench),
        ] {
            if position.x >= width || position.y >= height {
                return Err(CraftingSpecError::PositionOutOfBounds(name));
            }
            if walls_set.contains(&position) {
                return Err(CraftingSpecError::WallOverlapsObject(name));
            }
        }

        let mut pile_positions = HashSet::new();
        let mut total_pile_amounts = [0_u64; 4];
        for pile in &piles {
            if pile.position.x >= width || pile.position.y >= height {
                return Err(CraftingSpecError::PositionOutOfBounds("resource pile"));
            }
            if pile.amount == 0 {
                return Err(CraftingSpecError::ZeroPileAmount);
            }
            let total = &mut total_pile_amounts[pile.kind.index()];
            *total = total
                .checked_add(u64::from(pile.amount))
                .ok_or(CraftingSpecError::ResourceAmountOverflow)?;
            if walls_set.contains(&pile.position) {
                return Err(CraftingSpecError::WallOverlapsObject("resource pile"));
            }
            if !pile_positions.insert(pile.position) {
                return Err(CraftingSpecError::DuplicatePilePosition);
            }
        }
        if pile_positions.contains(&workbench) {
            return Err(CraftingSpecError::WorkbenchOverlapsPile);
        }

        let mut recipe_map = HashMap::new();
        for recipe in recipes {
            if recipe.output_amount == 0 {
                return Err(CraftingSpecError::ZeroRecipeOutput);
            }
            if recipe.inputs.is_empty() {
                return Err(CraftingSpecError::EmptyRecipe);
            }
            let mut input_kinds = HashSet::new();
            for ingredient in &recipe.inputs {
                if ingredient.amount == 0 {
                    return Err(CraftingSpecError::ZeroIngredientAmount);
                }
                if !input_kinds.insert(ingredient.kind) {
                    return Err(CraftingSpecError::DuplicateIngredient);
                }
            }
            if recipe_map.insert(recipe.output, recipe).is_some() {
                return Err(CraftingSpecError::DuplicateRecipeOutput);
            }
        }

        if !recipes_are_acyclic(&recipe_map) {
            return Err(CraftingSpecError::RecipeCycle);
        }

        Ok(Self {
            width,
            height,
            walls: walls_set,
            piles,
            workbench,
            recipes: recipe_map,
            start,
            target,
            target_amount,
            max_actions,
        })
    }
}

fn recipes_are_acyclic(recipes: &HashMap<ResourceKind, RecipeSpec>) -> bool {
    fn visit(
        resource: ResourceKind,
        recipes: &HashMap<ResourceKind, RecipeSpec>,
        visiting: &mut HashSet<ResourceKind>,
        visited: &mut HashSet<ResourceKind>,
    ) -> bool {
        if visited.contains(&resource) {
            return true;
        }
        if !visiting.insert(resource) {
            return false;
        }
        if let Some(recipe) = recipes.get(&resource) {
            for ingredient in &recipe.inputs {
                if recipes.contains_key(&ingredient.kind)
                    && !visit(ingredient.kind, recipes, visiting, visited)
                {
                    return false;
                }
            }
        }
        visiting.remove(&resource);
        visited.insert(resource);
        true
    }

    let mut visiting = HashSet::new();
    let mut visited = HashSet::new();
    for resource in recipes.keys().copied() {
        if !visit(resource, recipes, &mut visiting, &mut visited) {
            return false;
        }
    }
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CraftingState {
    pub position: Position,
    pub inventory: [u64; 4],
    pub elapsed_actions: u32,
}

impl CraftingState {
    pub fn quantity(&self, kind: ResourceKind) -> u64 {
        self.inventory[kind.index()]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CraftingAction {
    Move(Action),
    Gather,
    Craft(ResourceKind),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CraftingTransition {
    pub state: CraftingState,
    pub moved: bool,
    pub gathered: Option<ResourceKind>,
    pub crafted: Option<ResourceKind>,
    pub status: TerminalStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CraftingStepError {
    AlreadyTerminal(TerminalStatus),
}

impl Display for CraftingStepError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyTerminal(status) => {
                write!(f, "cannot step crafting environment after terminal status {status:?}")
            }
        }
    }
}

impl Error for CraftingStepError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CraftingGame {
    spec: CraftingSpec,
    state: CraftingState,
    remaining_piles: Vec<u32>,
    status: TerminalStatus,
}

impl CraftingGame {
    pub fn new(spec: CraftingSpec) -> Self {
        let state = CraftingState {
            position: spec.start,
            inventory: [0; 4],
            elapsed_actions: 0,
        };
        let remaining_piles = spec.piles.iter().map(|pile| pile.amount).collect();
        let status = TerminalStatus::Running;
        Self {
            spec,
            state,
            remaining_piles,
            status,
        }
    }

    pub const fn state(&self) -> CraftingState {
        self.state
    }

    pub const fn status(&self) -> TerminalStatus {
        self.status
    }

    pub fn step(
        &mut self,
        action: CraftingAction,
    ) -> Result<CraftingTransition, CraftingStepError> {
        if self.status != TerminalStatus::Running {
            return Err(CraftingStepError::AlreadyTerminal(self.status));
        }

        let transition = simulate(&self.spec, self.state, &self.remaining_piles, action);
        self.state = transition.0.state;
        self.remaining_piles = transition.1;
        self.status = transition.0.status;
        Ok(transition.0)
    }
}

fn adjacent(position: Position, action: Action) -> Option<Position> {
    match action {
        Action::North => Some(Position::new(position.x, position.y.checked_sub(1)?)),
        Action::East => Some(Position::new(position.x.checked_add(1)?, position.y)),
        Action::South => Some(Position::new(position.x, position.y.checked_add(1)?)),
        Action::West => Some(Position::new(position.x.checked_sub(1)?, position.y)),
    }
}

fn simulate(
    spec: &CraftingSpec,
    state: CraftingState,
    remaining_piles: &[u32],
    action: CraftingAction,
) -> (CraftingTransition, Vec<u32>) {
    let mut next = state;
    let mut piles = remaining_piles.to_vec();
    let mut moved = false;
    let mut gathered = None;
    let mut crafted = None;

    match action {
        CraftingAction::Move(direction) => {
            if let Some(destination) = adjacent(state.position, direction) {
                let in_bounds = destination.x < spec.width && destination.y < spec.height;
                if in_bounds && !spec.walls.contains(&destination) {
                    next.position = destination;
                    moved = true;
                }
            }
        }
        CraftingAction::Gather => {
            if let Some(index) = spec.piles.iter().enumerate().position(|(index, pile)| {
                pile.position == state.position && piles.get(index).copied().unwrap_or(0) > 0
            }) {
                let kind = spec.piles[index].kind;
                if let (Some(remaining), Some(quantity)) = (
                    piles.get_mut(index),
                    next.inventory[kind.index()].checked_add(1),
                ) {
                    if *remaining > 0 {
                        *remaining -= 1;
                        next.inventory[kind.index()] = quantity;
                        gathered = Some(kind);
                    }
                }
            }
        }
        CraftingAction::Craft(output) => {
            if state.position == spec.workbench {
                if let Some(recipe) = spec.recipes.get(&output) {
                    let enough = recipe.inputs.iter().all(|ingredient| {
                        next.inventory[ingredient.kind.index()] >= u64::from(ingredient.amount)
                    });
                    let next_amount = next.inventory[output.index()]
                        .checked_add(u64::from(recipe.output_amount));
                    if enough {
                        if let Some(next_amount) = next_amount {
                            for ingredient in &recipe.inputs {
                                next.inventory[ingredient.kind.index()] -=
                                    u64::from(ingredient.amount);
                            }
                            next.inventory[output.index()] = next_amount;
                            crafted = Some(output);
                        }
                    }
                }
            }
        }
    }

    next.elapsed_actions += 1;
    let status = if next.inventory[spec.target.index()] >= u64::from(spec.target_amount) {
        TerminalStatus::Success
    } else if next.elapsed_actions >= spec.max_actions {
        TerminalStatus::ActionLimit
    } else {
        TerminalStatus::Running
    };

    (
        CraftingTransition {
            state: next,
            moved,
            gathered,
            crafted,
            status,
        },
        piles,
    )
}

/// Exact BFS reference for evaluator use on small crafting fixtures only.
#[allow(dead_code)]
pub(crate) fn optimal_crafting_plan(spec: &CraftingSpec) -> Option<Vec<CraftingAction>> {
    let start = CraftingState {
        position: spec.start,
        inventory: [0; 4],
        elapsed_actions: 0,
    };
    if start.inventory[spec.target.index()] >= u64::from(spec.target_amount) {
        return Some(Vec::new());
    }

    let mut queue = VecDeque::from([(start, spec.piles.iter().map(|pile| pile.amount).collect::<Vec<_>>())]);
    let mut seen = HashSet::from([(start, spec.piles.iter().map(|pile| pile.amount).collect::<Vec<_>>())]);
    let mut parents: HashMap<(CraftingState, Vec<u32>), ((CraftingState, Vec<u32>), CraftingAction)> =
        HashMap::new();
    let mut goal = None;

    const MOVES: [Action; 4] = [Action::North, Action::East, Action::South, Action::West];
    const CRAFTS: [ResourceKind; 4] = [
        ResourceKind::Wood,
        ResourceKind::Stone,
        ResourceKind::Plank,
        ResourceKind::Tool,
    ];

    while let Some((state, piles)) = queue.pop_front() {
        if state.elapsed_actions >= spec.max_actions {
            continue;
        }

        let mut actions: Vec<CraftingAction> = MOVES.into_iter().map(CraftingAction::Move).collect();
        actions.push(CraftingAction::Gather);
        actions.extend(CRAFTS.into_iter().map(CraftingAction::Craft));

        for action in actions {
            let (transition, next_piles) = simulate(spec, state, &piles, action);
            let next_state = transition.state;
            if !transition.moved && transition.gathered.is_none() && transition.crafted.is_none() {
                continue;
            }
            let key = (next_state, next_piles.clone());
            if !seen.insert(key.clone()) {
                continue;
            }

            parents.insert(key.clone(), ((state, piles.clone()), action));
            if transition.status == TerminalStatus::Success {
                goal = Some(key);
                break;
            }
            if transition.status == TerminalStatus::Running {
                queue.push_back(key);
            }
        }
        if goal.is_some() {
            break;
        }
    }

    let mut cursor = goal?;
    let start_key = (start, spec.piles.iter().map(|pile| pile.amount).collect::<Vec<_>>());
    let mut path = Vec::new();
    while cursor != start_key {
        let (previous, action) = parents.get(&cursor)?.clone();
        path.push(action);
        cursor = previous;
    }
    path.reverse();
    Some(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(max_actions: u32) -> CraftingSpec {
        // Start -> wood x1 -> stone x3 -> workbench x4.
        // Two wood craft one plank; one plank + one stone craft one tool.
        CraftingSpec::new(
            5,
            1,
            Vec::new(),
            vec![
                ResourcePile {
                    kind: ResourceKind::Wood,
                    position: Position::new(1, 0),
                    amount: 2,
                },
                ResourcePile {
                    kind: ResourceKind::Stone,
                    position: Position::new(3, 0),
                    amount: 1,
                },
            ],
            Position::new(4, 0),
            vec![
                RecipeSpec {
                    output: ResourceKind::Plank,
                    output_amount: 1,
                    inputs: vec![Ingredient {
                        kind: ResourceKind::Wood,
                        amount: 2,
                    }],
                },
                RecipeSpec {
                    output: ResourceKind::Tool,
                    output_amount: 1,
                    inputs: vec![
                        Ingredient {
                            kind: ResourceKind::Plank,
                            amount: 1,
                        },
                        Ingredient {
                            kind: ResourceKind::Stone,
                            amount: 1,
                        },
                    ],
                },
            ],
            Position::new(0, 0),
            ResourceKind::Tool,
            1,
            max_actions,
        )
        .expect("valid crafting fixture")
    }

    #[test]
    fn gather_craft_dependency_chain_reaches_target() {
        let mut game = CraftingGame::new(fixture(20));
        let actions = [
            CraftingAction::Move(Action::East),
            CraftingAction::Gather,
            CraftingAction::Gather,
            CraftingAction::Move(Action::East),
            CraftingAction::Move(Action::East),
            CraftingAction::Gather,
            CraftingAction::Move(Action::East),
            CraftingAction::Craft(ResourceKind::Plank),
            CraftingAction::Craft(ResourceKind::Tool),
        ];
        for action in actions {
            game.step(action).expect("valid action");
        }
        assert_eq!(game.status(), TerminalStatus::Success);
        assert_eq!(game.state().quantity(ResourceKind::Tool), 1);
    }

    #[test]
    fn depleted_resource_pile_cannot_be_gathered_again() {
        let mut game = CraftingGame::new(fixture(20));
        game.step(CraftingAction::Move(Action::East)).expect("move to wood");
        game.step(CraftingAction::Gather).expect("first wood");
        game.step(CraftingAction::Gather).expect("second wood");
        let empty = game.step(CraftingAction::Gather).expect("empty gather is legal attempt");
        assert_eq!(empty.gathered, None);
        assert_eq!(empty.state.quantity(ResourceKind::Wood), 2);
    }

    #[test]
    fn crafting_requires_station_and_inputs() {
        let mut game = CraftingGame::new(fixture(10));
        let result = game
            .step(CraftingAction::Craft(ResourceKind::Plank))
            .expect("missing prereq is an unsuccessful legal attempt");
        assert_eq!(result.crafted, None);
        assert_eq!(result.state.quantity(ResourceKind::Plank), 0);
        assert_eq!(result.state.elapsed_actions, 1);
    }

    #[test]
    fn exact_oracle_finds_complete_resource_chain() {
        let plan = optimal_crafting_plan(&fixture(20)).expect("target is reachable");
        assert_eq!(plan.len(), 9);
        assert_eq!(plan.last(), Some(&CraftingAction::Craft(ResourceKind::Tool)));
    }

    #[test]
    fn rejects_recipe_cycles_and_zero_piles() {
        let cycle = CraftingSpec::new(
            3,
            1,
            Vec::new(),
            Vec::new(),
            Position::new(1, 0),
            vec![
                RecipeSpec {
                    output: ResourceKind::Plank,
                    output_amount: 1,
                    inputs: vec![Ingredient {
                        kind: ResourceKind::Tool,
                        amount: 1,
                    }],
                },
                RecipeSpec {
                    output: ResourceKind::Tool,
                    output_amount: 1,
                    inputs: vec![Ingredient {
                        kind: ResourceKind::Plank,
                        amount: 1,
                    }],
                },
            ],
            Position::new(0, 0),
            ResourceKind::Tool,
            1,
            10,
        );
        assert_eq!(cycle, Err(CraftingSpecError::RecipeCycle));

        let zero_pile = CraftingSpec::new(
            3,
            1,
            Vec::new(),
            vec![ResourcePile {
                kind: ResourceKind::Wood,
                position: Position::new(1, 0),
                amount: 0,
            }],
            Position::new(2, 0),
            Vec::new(),
            Position::new(0, 0),
            ResourceKind::Wood,
            1,
            5,
        );
        assert_eq!(zero_pile, Err(CraftingSpecError::ZeroPileAmount));
    }
}
