// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Workflow Automation Module
//!
//! Visual workflow builder, conditional routing, and automated actions

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use sqlx::PgPool;
use uuid::Uuid;
use std::collections::HashMap;

// ============================================================================
// Workflow Engine
// ============================================================================

pub struct WorkflowEngine {
    pool: PgPool,
    action_handlers: HashMap<String, Box<dyn ActionHandler>>,
}

impl WorkflowEngine {
    pub fn new(pool: PgPool) -> Self {
        let mut engine = Self {
            pool,
            action_handlers: HashMap::new(),
        };
        engine.register_default_handlers();
        engine
    }

    fn register_default_handlers(&mut self) {
        self.action_handlers.insert(
            "move_to_folder".to_string(),
            Box::new(MoveToFolderHandler),
        );
        self.action_handlers.insert(
            "apply_label".to_string(),
            Box::new(ApplyLabelHandler),
        );
        self.action_handlers.insert(
            "mark_read".to_string(),
            Box::new(MarkReadHandler),
        );
        self.action_handlers.insert(
            "forward".to_string(),
            Box::new(ForwardHandler),
        );
        self.action_handlers.insert(
            "auto_reply".to_string(),
            Box::new(AutoReplyHandler),
        );
        self.action_handlers.insert(
            "notify".to_string(),
            Box::new(NotifyHandler),
        );
        self.action_handlers.insert(
            "delay".to_string(),
            Box::new(DelayHandler),
        );
        self.action_handlers.insert(
            "webhook".to_string(),
            Box::new(WebhookHandler),
        );
    }

    /// Create a new workflow
    pub async fn create_workflow(
        &self,
        user_id: Uuid,
        definition: WorkflowDefinition,
    ) -> Result<Workflow, WorkflowError> {
        // Validate the workflow
        self.validate_workflow(&definition)?;

        let workflow_id = Uuid::new_v4();

        sqlx::query(
            r#"
            INSERT INTO workflows (id, user_id, name, description, trigger_type,
                                   trigger_config, nodes, enabled, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            "#,
        )
        .bind(workflow_id)
        .bind(user_id)
        .bind(&definition.name)
        .bind(&definition.description)
        .bind(definition.trigger.trigger_type.to_string())
        .bind(serde_json::to_value(&definition.trigger).unwrap())
        .bind(serde_json::to_value(&definition.nodes).unwrap())
        .bind(true)
        .bind(Utc::now())
        .execute(&self.pool)
        .await
        .map_err(|e| WorkflowError::Database(e.to_string()))?;

        Ok(Workflow {
            id: workflow_id,
            user_id,
            name: definition.name,
            description: definition.description,
            trigger: definition.trigger,
            nodes: definition.nodes,
            enabled: true,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            run_count: 0,
            last_run_at: None,
        })
    }

    /// Process an email against all active workflows
    pub async fn process_email(
        &self,
        user_id: Uuid,
        email: &EmailContext,
    ) -> Result<Vec<WorkflowExecution>, WorkflowError> {
        // Get all enabled workflows for this user with email triggers
        let workflows: Vec<WorkflowRow> = sqlx::query_as(
            r#"
            SELECT id, trigger_config, nodes FROM workflows
            WHERE user_id = $1 AND enabled = true
            AND trigger_type IN ('email_received', 'email_sent')
            "#,
        )
        .bind(user_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| WorkflowError::Database(e.to_string()))?;

        let mut executions = Vec::new();

        for workflow in workflows {
            let trigger: WorkflowTrigger = serde_json::from_value(workflow.trigger_config)
                .map_err(|e| WorkflowError::InvalidDefinition(e.to_string()))?;

            // Check if trigger conditions match
            if self.evaluate_trigger(&trigger, email) {
                let nodes: Vec<WorkflowNode> = serde_json::from_value(workflow.nodes)
                    .map_err(|e| WorkflowError::InvalidDefinition(e.to_string()))?;

                let execution = self.execute_workflow(workflow.id, &nodes, email).await?;
                executions.push(execution);
            }
        }

        Ok(executions)
    }

    fn evaluate_trigger(&self, trigger: &WorkflowTrigger, email: &EmailContext) -> bool {
        for condition in &trigger.conditions {
            if !self.evaluate_condition(condition, email) {
                return false;
            }
        }
        true
    }

    fn evaluate_condition(&self, condition: &Condition, email: &EmailContext) -> bool {
        let value = match condition.field.as_str() {
            "from" => &email.from,
            "to" => email.to.first().map(|s| s.as_str()).unwrap_or(""),
            "subject" => &email.subject,
            "body" => &email.body,
            _ => return false,
        };

        match condition.operator {
            ConditionOperator::Contains => {
                value.to_lowercase().contains(&condition.value.to_lowercase())
            }
            ConditionOperator::NotContains => {
                !value.to_lowercase().contains(&condition.value.to_lowercase())
            }
            ConditionOperator::Equals => value.to_lowercase() == condition.value.to_lowercase(),
            ConditionOperator::NotEquals => value.to_lowercase() != condition.value.to_lowercase(),
            ConditionOperator::StartsWith => {
                value.to_lowercase().starts_with(&condition.value.to_lowercase())
            }
            ConditionOperator::EndsWith => {
                value.to_lowercase().ends_with(&condition.value.to_lowercase())
            }
            ConditionOperator::Matches => {
                regex::Regex::new(&condition.value)
                    .map(|re| re.is_match(value))
                    .unwrap_or(false)
            }
            ConditionOperator::IsEmpty => value.is_empty(),
            ConditionOperator::IsNotEmpty => !value.is_empty(),
        }
    }

    async fn execute_workflow(
        &self,
        workflow_id: Uuid,
        nodes: &[WorkflowNode],
        email: &EmailContext,
    ) -> Result<WorkflowExecution, WorkflowError> {
        let execution_id = Uuid::new_v4();
        let mut context = ExecutionContext {
            email: email.clone(),
            variables: HashMap::new(),
            current_node_index: 0,
        };

        let mut executed_actions = Vec::new();
        let mut status = ExecutionStatus::Success;

        for (index, node) in nodes.iter().enumerate() {
            context.current_node_index = index;

            match &node.node_type {
                NodeType::Condition { condition, then_branch, else_branch } => {
                    let result = self.evaluate_condition(condition, &context.email);
                    let branch = if result { then_branch } else { else_branch };
                    if let Some(branch_nodes) = branch {
                        // Execute branch (simplified - in real impl would be recursive)
                    }
                }
                NodeType::Action { action_type, config } => {
                    if let Some(handler) = self.action_handlers.get(action_type) {
                        match handler.execute(&context, config).await {
                            Ok(result) => {
                                executed_actions.push(ExecutedAction {
                                    node_id: node.id.clone(),
                                    action_type: action_type.clone(),
                                    success: true,
                                    result: Some(result),
                                    error: None,
                                });
                            }
                            Err(e) => {
                                executed_actions.push(ExecutedAction {
                                    node_id: node.id.clone(),
                                    action_type: action_type.clone(),
                                    success: false,
                                    result: None,
                                    error: Some(e.to_string()),
                                });
                                if !node.continue_on_error {
                                    status = ExecutionStatus::Failed;
                                    break;
                                }
                            }
                        }
                    }
                }
                NodeType::Delay { duration_seconds } => {
                    // In real impl, would schedule continuation
                    tokio::time::sleep(std::time::Duration::from_secs(*duration_seconds as u64)).await;
                }
                NodeType::Split { branches } => {
                    // Execute multiple branches in parallel (simplified)
                }
            }
        }

        // Record execution
        let execution = WorkflowExecution {
            id: execution_id,
            workflow_id,
            email_id: email.email_id,
            status,
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
            actions_executed: executed_actions,
        };

        sqlx::query(
            r#"
            INSERT INTO workflow_executions (id, workflow_id, email_id, status,
                                             started_at, completed_at, actions_executed)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            "#,
        )
        .bind(execution.id)
        .bind(execution.workflow_id)
        .bind(execution.email_id)
        .bind(execution.status.to_string())
        .bind(execution.started_at)
        .bind(execution.completed_at)
        .bind(serde_json::to_value(&execution.actions_executed).unwrap())
        .execute(&self.pool)
        .await
        .map_err(|e| WorkflowError::Database(e.to_string()))?;

        // Update workflow stats
        sqlx::query(
            "UPDATE workflows SET run_count = run_count + 1, last_run_at = $1 WHERE id = $2",
        )
        .bind(Utc::now())
        .bind(workflow_id)
        .execute(&self.pool)
        .await
        .ok();

        Ok(execution)
    }

    fn validate_workflow(&self, definition: &WorkflowDefinition) -> Result<(), WorkflowError> {
        if definition.name.is_empty() {
            return Err(WorkflowError::InvalidDefinition("Name is required".into()));
        }

        if definition.nodes.is_empty() {
            return Err(WorkflowError::InvalidDefinition(
                "At least one node is required".into(),
            ));
        }

        // Validate each node
        for node in &definition.nodes {
            if let NodeType::Action { action_type, .. } = &node.node_type {
                if !self.action_handlers.contains_key(action_type) {
                    return Err(WorkflowError::InvalidDefinition(format!(
                        "Unknown action type: {}",
                        action_type
                    )));
                }
            }
        }

        Ok(())
    }

    /// Get workflow templates
    pub fn get_templates(&self) -> Vec<WorkflowTemplate> {
        vec![
            WorkflowTemplate {
                id: "auto_archive_newsletters".to_string(),
                name: "Auto-archive Newsletters".to_string(),
                description: "Automatically archive emails from newsletters".to_string(),
                category: "Organization".to_string(),
                definition: WorkflowDefinition {
                    name: "Auto-archive Newsletters".to_string(),
                    description: Some("Archive newsletter emails automatically".to_string()),
                    trigger: WorkflowTrigger {
                        trigger_type: TriggerType::EmailReceived,
                        conditions: vec![
                            Condition {
                                field: "from".to_string(),
                                operator: ConditionOperator::Contains,
                                value: "newsletter".to_string(),
                            },
                        ],
                    },
                    nodes: vec![
                        WorkflowNode {
                            id: "1".to_string(),
                            node_type: NodeType::Action {
                                action_type: "apply_label".to_string(),
                                config: serde_json::json!({"label": "Newsletter"}),
                            },
                            continue_on_error: true,
                        },
                        WorkflowNode {
                            id: "2".to_string(),
                            node_type: NodeType::Action {
                                action_type: "move_to_folder".to_string(),
                                config: serde_json::json!({"folder": "Archive"}),
                            },
                            continue_on_error: false,
                        },
                    ],
                },
            },
            WorkflowTemplate {
                id: "urgent_notification".to_string(),
                name: "Urgent Email Notification".to_string(),
                description: "Get notified immediately for urgent emails".to_string(),
                category: "Notifications".to_string(),
                definition: WorkflowDefinition {
                    name: "Urgent Email Notification".to_string(),
                    description: Some("Push notification for urgent emails".to_string()),
                    trigger: WorkflowTrigger {
                        trigger_type: TriggerType::EmailReceived,
                        conditions: vec![
                            Condition {
                                field: "subject".to_string(),
                                operator: ConditionOperator::Contains,
                                value: "urgent".to_string(),
                            },
                        ],
                    },
                    nodes: vec![
                        WorkflowNode {
                            id: "1".to_string(),
                            node_type: NodeType::Action {
                                action_type: "notify".to_string(),
                                config: serde_json::json!({
                                    "title": "Urgent Email",
                                    "priority": "high"
                                }),
                            },
                            continue_on_error: false,
                        },
                    ],
                },
            },
            WorkflowTemplate {
                id: "auto_reply_vacation".to_string(),
                name: "Vacation Auto-Reply".to_string(),
                description: "Automatically reply when on vacation".to_string(),
                category: "Auto-Reply".to_string(),
                definition: WorkflowDefinition {
                    name: "Vacation Auto-Reply".to_string(),
                    description: Some("Send automatic replies during vacation".to_string()),
                    trigger: WorkflowTrigger {
                        trigger_type: TriggerType::EmailReceived,
                        conditions: vec![],
                    },
                    nodes: vec![
                        WorkflowNode {
                            id: "1".to_string(),
                            node_type: NodeType::Action {
                                action_type: "auto_reply".to_string(),
                                config: serde_json::json!({
                                    "subject": "Out of Office",
                                    "body": "Thank you for your email. I am currently out of office and will respond when I return.",
                                    "once_per_sender": true
                                }),
                            },
                            continue_on_error: false,
                        },
                    ],
                },
            },
        ]
    }
}

// ============================================================================
// Action Handlers
// ============================================================================

#[async_trait]
trait ActionHandler: Send + Sync {
    async fn execute(
        &self,
        context: &ExecutionContext,
        config: &JsonValue,
    ) -> Result<String, WorkflowError>;
}

struct MoveToFolderHandler;
#[async_trait]
impl ActionHandler for MoveToFolderHandler {
    async fn execute(&self, context: &ExecutionContext, config: &JsonValue) -> Result<String, WorkflowError> {
        let folder = config["folder"].as_str().ok_or(WorkflowError::InvalidConfig("folder required".into()))?;
        Ok(format!("Moved to folder: {}", folder))
    }
}

struct ApplyLabelHandler;
#[async_trait]
impl ActionHandler for ApplyLabelHandler {
    async fn execute(&self, context: &ExecutionContext, config: &JsonValue) -> Result<String, WorkflowError> {
        let label = config["label"].as_str().ok_or(WorkflowError::InvalidConfig("label required".into()))?;
        Ok(format!("Applied label: {}", label))
    }
}

struct MarkReadHandler;
#[async_trait]
impl ActionHandler for MarkReadHandler {
    async fn execute(&self, context: &ExecutionContext, config: &JsonValue) -> Result<String, WorkflowError> {
        Ok("Marked as read".to_string())
    }
}

struct ForwardHandler;
#[async_trait]
impl ActionHandler for ForwardHandler {
    async fn execute(&self, context: &ExecutionContext, config: &JsonValue) -> Result<String, WorkflowError> {
        let to = config["to"].as_str().ok_or(WorkflowError::InvalidConfig("to required".into()))?;
        Ok(format!("Forwarded to: {}", to))
    }
}

struct AutoReplyHandler;
#[async_trait]
impl ActionHandler for AutoReplyHandler {
    async fn execute(&self, context: &ExecutionContext, config: &JsonValue) -> Result<String, WorkflowError> {
        let subject = config["subject"].as_str().unwrap_or("Re: Auto-Reply");
        Ok(format!("Auto-reply sent with subject: {}", subject))
    }
}

struct NotifyHandler;
#[async_trait]
impl ActionHandler for NotifyHandler {
    async fn execute(&self, context: &ExecutionContext, config: &JsonValue) -> Result<String, WorkflowError> {
        let title = config["title"].as_str().unwrap_or("New Email");
        Ok(format!("Notification sent: {}", title))
    }
}

struct DelayHandler;
#[async_trait]
impl ActionHandler for DelayHandler {
    async fn execute(&self, context: &ExecutionContext, config: &JsonValue) -> Result<String, WorkflowError> {
        let seconds = config["seconds"].as_i64().unwrap_or(60);
        Ok(format!("Delayed for {} seconds", seconds))
    }
}

struct WebhookHandler;
#[async_trait]
impl ActionHandler for WebhookHandler {
    async fn execute(&self, context: &ExecutionContext, config: &JsonValue) -> Result<String, WorkflowError> {
        let url = config["url"].as_str().ok_or(WorkflowError::InvalidConfig("url required".into()))?;
        // Would make HTTP request
        Ok(format!("Webhook called: {}", url))
    }
}

// ============================================================================
// Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    pub id: Uuid,
    pub user_id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub trigger: WorkflowTrigger,
    pub nodes: Vec<WorkflowNode>,
    pub enabled: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub run_count: i64,
    pub last_run_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowDefinition {
    pub name: String,
    pub description: Option<String>,
    pub trigger: WorkflowTrigger,
    pub nodes: Vec<WorkflowNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowTrigger {
    pub trigger_type: TriggerType,
    pub conditions: Vec<Condition>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TriggerType {
    EmailReceived,
    EmailSent,
    Scheduled,
    Manual,
    Webhook,
}

impl ToString for TriggerType {
    fn to_string(&self) -> String {
        match self {
            TriggerType::EmailReceived => "email_received".to_string(),
            TriggerType::EmailSent => "email_sent".to_string(),
            TriggerType::Scheduled => "scheduled".to_string(),
            TriggerType::Manual => "manual".to_string(),
            TriggerType::Webhook => "webhook".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Condition {
    pub field: String,
    pub operator: ConditionOperator,
    pub value: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ConditionOperator {
    Contains,
    NotContains,
    Equals,
    NotEquals,
    StartsWith,
    EndsWith,
    Matches,
    IsEmpty,
    IsNotEmpty,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowNode {
    pub id: String,
    pub node_type: NodeType,
    #[serde(default)]
    pub continue_on_error: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeType {
    Condition {
        condition: Condition,
        then_branch: Option<Vec<WorkflowNode>>,
        else_branch: Option<Vec<WorkflowNode>>,
    },
    Action {
        action_type: String,
        config: JsonValue,
    },
    Delay {
        duration_seconds: i32,
    },
    Split {
        branches: Vec<Vec<WorkflowNode>>,
    },
}

#[derive(Debug, Clone)]
pub struct ExecutionContext {
    pub email: EmailContext,
    pub variables: HashMap<String, String>,
    pub current_node_index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailContext {
    pub email_id: Uuid,
    pub from: String,
    pub to: Vec<String>,
    pub subject: String,
    pub body: String,
    pub received_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowExecution {
    pub id: Uuid,
    pub workflow_id: Uuid,
    pub email_id: Uuid,
    pub status: ExecutionStatus,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub actions_executed: Vec<ExecutedAction>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ExecutionStatus {
    Running,
    Success,
    Failed,
    Cancelled,
}

impl ToString for ExecutionStatus {
    fn to_string(&self) -> String {
        match self {
            ExecutionStatus::Running => "running".to_string(),
            ExecutionStatus::Success => "success".to_string(),
            ExecutionStatus::Failed => "failed".to_string(),
            ExecutionStatus::Cancelled => "cancelled".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutedAction {
    pub node_id: String,
    pub action_type: String,
    pub success: bool,
    pub result: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowTemplate {
    pub id: String,
    pub name: String,
    pub description: String,
    pub category: String,
    pub definition: WorkflowDefinition,
}

#[derive(Debug, sqlx::FromRow)]
struct WorkflowRow {
    id: Uuid,
    trigger_config: JsonValue,
    nodes: JsonValue,
}

#[derive(Debug, thiserror::Error)]
pub enum WorkflowError {
    #[error("Database error: {0}")]
    Database(String),
    #[error("Invalid workflow definition: {0}")]
    InvalidDefinition(String),
    #[error("Invalid action config: {0}")]
    InvalidConfig(String),
    #[error("Workflow not found")]
    NotFound,
    #[error("Execution error: {0}")]
    ExecutionError(String),
}
