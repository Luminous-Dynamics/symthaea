//! Email Rules Engine Module
//!
//! Advanced email filtering, organization, and automation rules.

pub mod engine;

pub use engine::{
    ActionType, ConditionField, ConditionOperator, ConditionValue,
    EmailContext, LogicalOperator, Rule, RuleAction, RuleBuilder,
    RuleCondition, RuleConditionGroup, RuleEvaluationResult, RulesEngine,
};
