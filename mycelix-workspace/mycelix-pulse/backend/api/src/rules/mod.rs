// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Email Rules Engine Module
//!
//! Advanced email filtering, organization, and automation rules.

pub mod engine;

pub use engine::{
    ActionType, ConditionField, ConditionOperator, ConditionValue,
    EmailContext, LogicalOperator, Rule, RuleAction, RuleBuilder,
    RuleCondition, RuleConditionGroup, RuleEvaluationResult, RulesEngine,
};
