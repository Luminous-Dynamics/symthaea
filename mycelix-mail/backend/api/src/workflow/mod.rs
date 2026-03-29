// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Track V: Workflow Automation Engine
//!
//! Visual workflow builder, conditional email routing, scheduled actions,
//! email sequences, and cross-app integrations.

use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;
use std::collections::HashMap;

// ============================================================================
// Workflow Definition
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    pub id: Uuid,
    pub user_id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub trigger: WorkflowTrigger,
    pub nodes: Vec<WorkflowNode>,
    pub connections: Vec<NodeConnection>,
    pub enabled: bool,
    pub run_count: u64,
    pub last_run: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowNode {
    pub id: String,
    pub node_type: NodeType,
    pub position: Position,
    pub config: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConnection {
    pub from_node: String,
    pub from_port: String,
    pub to_node: String,
    pub to_port: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkflowTrigger {
    EmailReceived {
        filters: Vec<TriggerFilter>,
    },
    EmailSent,
    Schedule {
        cron: String,
    },
    Manual,
    Webhook {
        secret: String,
    },
    LabelAdded {
        label: String,
    },
    ThreadUpdated {
        thread_id: Option<Uuid>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerFilter {
    pub field: TriggerField,
    pub operator: FilterOperator,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriggerField {
    From,
    To,
    Subject,
    Body,
    HasAttachment,
    Label,
    Folder,
    Domain,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterOperator {
    Equals,
    NotEquals,
    Contains,
    NotContains,
    StartsWith,
    EndsWith,
    Matches,  // Regex
    GreaterThan,
    LessThan,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeType {
    // Conditions
    Condition(ConditionNode),
    Switch(SwitchNode),

    // Actions
    MoveTo(MoveToNode),
    AddLabel(AddLabelNode),
    RemoveLabel(RemoveLabelNode),
    MarkRead(MarkReadNode),
    Star(StarNode),
    Archive(ArchiveNode),
    Delete(DeleteNode),
    Forward(ForwardNode),
    Reply(ReplyNode),
    SendEmail(SendEmailNode),

    // Timing
    Delay(DelayNode),
    ScheduleSend(ScheduleSendNode),
    WaitFor(WaitForNode),

    // Integrations
    Webhook(WebhookNode),
    SlackNotify(SlackNotifyNode),
    MatrixNotify(MatrixNotifyNode),
    CreateTask(CreateTaskNode),
    UpdateCRM(UpdateCRMNode),

    // Utilities
    SetVariable(SetVariableNode),
    Transform(TransformNode),
    Filter(FilterNode),
    Merge(MergeNode),
    Split(SplitNode),
}

// ============================================================================
// Node Configurations
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionNode {
    pub conditions: Vec<TriggerFilter>,
    pub logic: ConditionLogic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConditionLogic {
    And,
    Or,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwitchNode {
    pub cases: Vec<SwitchCase>,
    pub default_output: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwitchCase {
    pub conditions: Vec<TriggerFilter>,
    pub output_port: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoveToNode {
    pub folder: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddLabelNode {
    pub label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoveLabelNode {
    pub label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkReadNode {
    pub read: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarNode {
    pub starred: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveNode {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteNode {
    pub permanent: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardNode {
    pub to: Vec<String>,
    pub include_attachments: bool,
    pub add_note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplyNode {
    pub template_id: Option<Uuid>,
    pub body: Option<String>,
    pub reply_all: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SendEmailNode {
    pub to: Vec<String>,
    pub cc: Option<Vec<String>>,
    pub bcc: Option<Vec<String>>,
    pub subject: String,
    pub body: String,
    pub template_id: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelayNode {
    pub duration: DelayDuration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DelayDuration {
    Minutes(u32),
    Hours(u32),
    Days(u32),
    Until { time: String, timezone: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleSendNode {
    pub send_at: ScheduleTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScheduleTime {
    Absolute(DateTime<Utc>),
    Relative { duration: DelayDuration },
    RecipientTimezone { time: String },
    OptimalTime,  // AI-determined best time
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaitForNode {
    pub wait_for: WaitCondition,
    pub timeout: Option<DelayDuration>,
    pub timeout_action: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WaitCondition {
    Reply,
    Read,
    Click { link_pattern: Option<String> },
    Forward,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookNode {
    pub url: String,
    pub method: HttpMethod,
    pub headers: HashMap<String, String>,
    pub body_template: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Patch,
    Delete,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlackNotifyNode {
    pub channel: String,
    pub message: String,
    pub include_email_preview: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixNotifyNode {
    pub room_id: String,
    pub message: String,
    pub formatted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateTaskNode {
    pub integration: TaskIntegration,
    pub title_template: String,
    pub description_template: Option<String>,
    pub due_date: Option<TaskDueDate>,
    pub priority: Option<String>,
    pub project: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskIntegration {
    Vikunja { api_url: String },
    OpenProject { api_url: String },
    NextcloudTasks { api_url: String },
    Todoist,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskDueDate {
    FromEmail,  // Extract from email content
    RelativeDays(u32),
    Fixed(DateTime<Utc>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateCRMNode {
    pub crm: CRMIntegration,
    pub action: CRMAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CRMIntegration {
    SuiteCRM { api_url: String },
    EspoCRM { api_url: String },
    Monica { api_url: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CRMAction {
    LogEmail,
    UpdateContact { fields: HashMap<String, String> },
    CreateNote,
    CreateActivity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetVariableNode {
    pub variable_name: String,
    pub value: VariableValue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VariableValue {
    Static(String),
    FromEmail(String),  // JSONPath to extract from email
    Expression(String),  // Template expression
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformNode {
    pub transformations: Vec<Transformation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transformation {
    pub field: String,
    pub operation: TransformOperation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformOperation {
    Lowercase,
    Uppercase,
    Trim,
    Replace { from: String, to: String },
    Extract { pattern: String },
    Append { value: String },
    Prepend { value: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterNode {
    pub conditions: Vec<TriggerFilter>,
    pub logic: ConditionLogic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeNode {
    pub wait_for_all: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitNode {
    pub split_by: SplitBy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SplitBy {
    Recipients,
    Attachments,
    Labels,
}

// ============================================================================
// Workflow Execution
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowExecution {
    pub id: Uuid,
    pub workflow_id: Uuid,
    pub trigger_data: serde_json::Value,
    pub status: ExecutionStatus,
    pub current_node: Option<String>,
    pub variables: HashMap<String, serde_json::Value>,
    pub logs: Vec<ExecutionLog>,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionStatus {
    Running,
    Waiting,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionLog {
    pub timestamp: DateTime<Utc>,
    pub node_id: String,
    pub action: String,
    pub result: LogResult,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogResult {
    Success { output: serde_json::Value },
    Skipped { reason: String },
    Failed { error: String },
}

// ============================================================================
// Services
// ============================================================================

pub struct WorkflowService {
    pool: PgPool,
}

impl WorkflowService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn create(&self, user_id: Uuid, workflow: Workflow) -> Result<Workflow> {
        sqlx::query!(
            r#"
            INSERT INTO workflows (id, user_id, name, description, trigger, nodes, connections, enabled, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW(), NOW())
            "#,
            workflow.id,
            user_id,
            workflow.name,
            workflow.description,
            serde_json::to_value(&workflow.trigger)?,
            serde_json::to_value(&workflow.nodes)?,
            serde_json::to_value(&workflow.connections)?,
            workflow.enabled
        )
        .execute(&self.pool)
        .await?;

        Ok(workflow)
    }

    pub async fn get(&self, workflow_id: Uuid) -> Result<Workflow> {
        let row = sqlx::query_as!(
            WorkflowRow,
            r#"
            SELECT id, user_id, name, description, trigger, nodes, connections,
                   enabled, run_count, last_run, created_at, updated_at
            FROM workflows
            WHERE id = $1
            "#,
            workflow_id
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(row.into())
    }

    pub async fn list(&self, user_id: Uuid) -> Result<Vec<Workflow>> {
        let rows = sqlx::query_as!(
            WorkflowRow,
            r#"
            SELECT id, user_id, name, description, trigger, nodes, connections,
                   enabled, run_count, last_run, created_at, updated_at
            FROM workflows
            WHERE user_id = $1
            ORDER BY updated_at DESC
            "#,
            user_id
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().map(Into::into).collect())
    }

    pub async fn update(&self, workflow: &Workflow) -> Result<()> {
        sqlx::query!(
            r#"
            UPDATE workflows SET
                name = $2,
                description = $3,
                trigger = $4,
                nodes = $5,
                connections = $6,
                enabled = $7,
                updated_at = NOW()
            WHERE id = $1
            "#,
            workflow.id,
            workflow.name,
            workflow.description,
            serde_json::to_value(&workflow.trigger)?,
            serde_json::to_value(&workflow.nodes)?,
            serde_json::to_value(&workflow.connections)?,
            workflow.enabled
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn delete(&self, workflow_id: Uuid) -> Result<()> {
        sqlx::query!("DELETE FROM workflows WHERE id = $1", workflow_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn toggle(&self, workflow_id: Uuid, enabled: bool) -> Result<()> {
        sqlx::query!(
            "UPDATE workflows SET enabled = $2, updated_at = NOW() WHERE id = $1",
            workflow_id,
            enabled
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn get_triggered_workflows(&self, user_id: Uuid, email: &EmailContext) -> Result<Vec<Workflow>> {
        let all_workflows = self.list(user_id).await?;

        let mut matching = Vec::new();
        for workflow in all_workflows {
            if !workflow.enabled {
                continue;
            }

            if self.matches_trigger(&workflow.trigger, email) {
                matching.push(workflow);
            }
        }

        Ok(matching)
    }

    fn matches_trigger(&self, trigger: &WorkflowTrigger, email: &EmailContext) -> bool {
        match trigger {
            WorkflowTrigger::EmailReceived { filters } => {
                for filter in filters {
                    if !self.matches_filter(filter, email) {
                        return false;
                    }
                }
                true
            }
            WorkflowTrigger::LabelAdded { label } => {
                email.labels.contains(label)
            }
            _ => false,
        }
    }

    fn matches_filter(&self, filter: &TriggerFilter, email: &EmailContext) -> bool {
        let field_value = match filter.field {
            TriggerField::From => &email.from,
            TriggerField::Subject => &email.subject,
            TriggerField::Body => &email.body,
            TriggerField::Folder => &email.folder,
            TriggerField::To => email.to.join(",").as_str(),
            TriggerField::Domain => email.from.split('@').last().unwrap_or(""),
            TriggerField::HasAttachment => if email.has_attachments { "true" } else { "false" },
            TriggerField::Label => email.labels.join(",").as_str(),
        };

        match filter.operator {
            FilterOperator::Equals => field_value == filter.value,
            FilterOperator::NotEquals => field_value != filter.value,
            FilterOperator::Contains => field_value.contains(&filter.value),
            FilterOperator::NotContains => !field_value.contains(&filter.value),
            FilterOperator::StartsWith => field_value.starts_with(&filter.value),
            FilterOperator::EndsWith => field_value.ends_with(&filter.value),
            FilterOperator::Matches => {
                regex::Regex::new(&filter.value)
                    .map(|re| re.is_match(field_value))
                    .unwrap_or(false)
            }
            _ => false,
        }
    }
}

#[derive(Debug)]
pub struct EmailContext {
    pub id: Uuid,
    pub from: String,
    pub to: Vec<String>,
    pub subject: String,
    pub body: String,
    pub folder: String,
    pub labels: Vec<String>,
    pub has_attachments: bool,
    pub received_at: DateTime<Utc>,
}

#[derive(Debug)]
struct WorkflowRow {
    id: Uuid,
    user_id: Uuid,
    name: String,
    description: Option<String>,
    trigger: serde_json::Value,
    nodes: serde_json::Value,
    connections: serde_json::Value,
    enabled: bool,
    run_count: i64,
    last_run: Option<DateTime<Utc>>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

impl From<WorkflowRow> for Workflow {
    fn from(row: WorkflowRow) -> Self {
        Workflow {
            id: row.id,
            user_id: row.user_id,
            name: row.name,
            description: row.description,
            trigger: serde_json::from_value(row.trigger).unwrap_or(WorkflowTrigger::Manual),
            nodes: serde_json::from_value(row.nodes).unwrap_or_default(),
            connections: serde_json::from_value(row.connections).unwrap_or_default(),
            enabled: row.enabled,
            run_count: row.run_count as u64,
            last_run: row.last_run,
            created_at: row.created_at,
            updated_at: row.updated_at,
        }
    }
}

// ============================================================================
// Workflow Executor
// ============================================================================

pub struct WorkflowExecutor {
    pool: PgPool,
}

impl WorkflowExecutor {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn execute(&self, workflow: &Workflow, context: &EmailContext) -> Result<WorkflowExecution> {
        let execution = WorkflowExecution {
            id: Uuid::new_v4(),
            workflow_id: workflow.id,
            trigger_data: serde_json::to_value(context)?,
            status: ExecutionStatus::Running,
            current_node: None,
            variables: self.init_variables(context),
            logs: Vec::new(),
            started_at: Utc::now(),
            completed_at: None,
            error: None,
        };

        // Store execution start
        self.save_execution(&execution).await?;

        // Find start node (first node after trigger)
        let start_node = self.find_start_node(workflow);

        if let Some(node_id) = start_node {
            self.execute_node(workflow, &node_id, context, &execution).await?;
        }

        // Update workflow stats
        sqlx::query!(
            "UPDATE workflows SET run_count = run_count + 1, last_run = NOW() WHERE id = $1",
            workflow.id
        )
        .execute(&self.pool)
        .await?;

        Ok(execution)
    }

    fn init_variables(&self, context: &EmailContext) -> HashMap<String, serde_json::Value> {
        let mut vars = HashMap::new();
        vars.insert("email.id".to_string(), serde_json::json!(context.id));
        vars.insert("email.from".to_string(), serde_json::json!(context.from));
        vars.insert("email.to".to_string(), serde_json::json!(context.to));
        vars.insert("email.subject".to_string(), serde_json::json!(context.subject));
        vars.insert("email.body".to_string(), serde_json::json!(context.body));
        vars.insert("email.folder".to_string(), serde_json::json!(context.folder));
        vars.insert("email.labels".to_string(), serde_json::json!(context.labels));
        vars
    }

    fn find_start_node(&self, workflow: &Workflow) -> Option<String> {
        // Find node that is connected from trigger (has no incoming connections)
        let connected_to: std::collections::HashSet<_> = workflow.connections.iter()
            .map(|c| c.to_node.clone())
            .collect();

        workflow.nodes.iter()
            .find(|n| !connected_to.contains(&n.id))
            .map(|n| n.id.clone())
    }

    async fn execute_node(
        &self,
        workflow: &Workflow,
        node_id: &str,
        context: &EmailContext,
        execution: &WorkflowExecution,
    ) -> Result<()> {
        let node = workflow.nodes.iter()
            .find(|n| n.id == node_id)
            .ok_or_else(|| anyhow::anyhow!("Node not found: {}", node_id))?;

        let start = std::time::Instant::now();

        match &node.node_type {
            NodeType::AddLabel(config) => {
                sqlx::query!(
                    "UPDATE emails SET labels = array_append(labels, $1) WHERE id = $2",
                    config.label,
                    context.id
                )
                .execute(&self.pool)
                .await?;
            }
            NodeType::RemoveLabel(config) => {
                sqlx::query!(
                    "UPDATE emails SET labels = array_remove(labels, $1) WHERE id = $2",
                    config.label,
                    context.id
                )
                .execute(&self.pool)
                .await?;
            }
            NodeType::MoveTo(config) => {
                sqlx::query!(
                    "UPDATE emails SET folder = $1 WHERE id = $2",
                    config.folder,
                    context.id
                )
                .execute(&self.pool)
                .await?;
            }
            NodeType::MarkRead(config) => {
                sqlx::query!(
                    "UPDATE emails SET is_read = $1 WHERE id = $2",
                    config.read,
                    context.id
                )
                .execute(&self.pool)
                .await?;
            }
            NodeType::Star(config) => {
                sqlx::query!(
                    "UPDATE emails SET is_starred = $1 WHERE id = $2",
                    config.starred,
                    context.id
                )
                .execute(&self.pool)
                .await?;
            }
            NodeType::Archive(_) => {
                sqlx::query!(
                    "UPDATE emails SET folder = 'Archive' WHERE id = $1",
                    context.id
                )
                .execute(&self.pool)
                .await?;
            }
            NodeType::Delete(config) => {
                if config.permanent {
                    sqlx::query!("DELETE FROM emails WHERE id = $1", context.id)
                        .execute(&self.pool)
                        .await?;
                } else {
                    sqlx::query!(
                        "UPDATE emails SET folder = 'Trash' WHERE id = $1",
                        context.id
                    )
                    .execute(&self.pool)
                    .await?;
                }
            }
            NodeType::Delay(config) => {
                // Schedule continuation
                let resume_at = match &config.duration {
                    DelayDuration::Minutes(m) => Utc::now() + Duration::minutes(*m as i64),
                    DelayDuration::Hours(h) => Utc::now() + Duration::hours(*h as i64),
                    DelayDuration::Days(d) => Utc::now() + Duration::days(*d as i64),
                    DelayDuration::Until { time, .. } => {
                        // Parse time and calculate next occurrence
                        Utc::now() + Duration::hours(1)  // Placeholder
                    }
                };

                sqlx::query!(
                    r#"
                    INSERT INTO workflow_delays (execution_id, node_id, resume_at)
                    VALUES ($1, $2, $3)
                    "#,
                    execution.id,
                    node_id,
                    resume_at
                )
                .execute(&self.pool)
                .await?;

                return Ok(());  // Stop execution, will resume later
            }
            NodeType::Webhook(config) => {
                let client = reqwest::Client::new();
                let mut request = match config.method {
                    HttpMethod::Get => client.get(&config.url),
                    HttpMethod::Post => client.post(&config.url),
                    HttpMethod::Put => client.put(&config.url),
                    HttpMethod::Patch => client.patch(&config.url),
                    HttpMethod::Delete => client.delete(&config.url),
                };

                for (key, value) in &config.headers {
                    request = request.header(key.as_str(), value.as_str());
                }

                if let Some(body_template) = &config.body_template {
                    let body = self.render_template(body_template, &execution.variables);
                    request = request.body(body);
                }

                let _ = request.send().await;
            }
            NodeType::Condition(config) => {
                let matches = match config.logic {
                    ConditionLogic::And => config.conditions.iter().all(|f| self.matches_filter_exec(f, context)),
                    ConditionLogic::Or => config.conditions.iter().any(|f| self.matches_filter_exec(f, context)),
                };

                // Get next node based on condition
                let output_port = if matches { "true" } else { "false" };
                let next_node = workflow.connections.iter()
                    .find(|c| c.from_node == node_id && c.from_port == output_port)
                    .map(|c| c.to_node.clone());

                if let Some(next) = next_node {
                    return self.execute_node(workflow, &next, context, execution).await;
                }
                return Ok(());
            }
            NodeType::SlackNotify(config) => {
                self.send_slack_notification(config, context).await?;
            }
            NodeType::MatrixNotify(config) => {
                self.send_matrix_notification(config, context).await?;
            }
            NodeType::CreateTask(config) => {
                self.create_task(config, context).await?;
            }
            _ => {}
        }

        // Find and execute next node
        let next_node = workflow.connections.iter()
            .find(|c| c.from_node == node_id)
            .map(|c| c.to_node.clone());

        if let Some(next) = next_node {
            Box::pin(self.execute_node(workflow, &next, context, execution)).await?;
        }

        Ok(())
    }

    fn matches_filter_exec(&self, filter: &TriggerFilter, email: &EmailContext) -> bool {
        let field_value = match filter.field {
            TriggerField::From => email.from.clone(),
            TriggerField::Subject => email.subject.clone(),
            TriggerField::Body => email.body.clone(),
            _ => String::new(),
        };

        match filter.operator {
            FilterOperator::Contains => field_value.contains(&filter.value),
            FilterOperator::Equals => field_value == filter.value,
            _ => false,
        }
    }

    fn render_template(&self, template: &str, variables: &HashMap<String, serde_json::Value>) -> String {
        let mut result = template.to_string();
        for (key, value) in variables {
            let placeholder = format!("{{{{{}}}}}", key);
            let value_str = match value {
                serde_json::Value::String(s) => s.clone(),
                _ => value.to_string(),
            };
            result = result.replace(&placeholder, &value_str);
        }
        result
    }

    async fn save_execution(&self, execution: &WorkflowExecution) -> Result<()> {
        sqlx::query!(
            r#"
            INSERT INTO workflow_executions (id, workflow_id, trigger_data, status, variables, logs, started_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            "#,
            execution.id,
            execution.workflow_id,
            execution.trigger_data,
            serde_json::to_string(&execution.status)?,
            serde_json::to_value(&execution.variables)?,
            serde_json::to_value(&execution.logs)?,
            execution.started_at
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    async fn send_slack_notification(&self, config: &SlackNotifyNode, context: &EmailContext) -> Result<()> {
        // Would integrate with Slack API
        Ok(())
    }

    async fn send_matrix_notification(&self, config: &MatrixNotifyNode, context: &EmailContext) -> Result<()> {
        // Would integrate with Matrix API
        Ok(())
    }

    async fn create_task(&self, config: &CreateTaskNode, context: &EmailContext) -> Result<()> {
        // Would integrate with task management systems
        Ok(())
    }
}

// ============================================================================
// Email Sequences
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailSequence {
    pub id: Uuid,
    pub user_id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub steps: Vec<SequenceStep>,
    pub stop_conditions: Vec<StopCondition>,
    pub active: bool,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceStep {
    pub step_number: u32,
    pub delay: DelayDuration,
    pub email: SequenceEmail,
    pub conditions: Option<Vec<TriggerFilter>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceEmail {
    pub subject: String,
    pub body: String,
    pub template_id: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StopCondition {
    RecipientReplied,
    LinkClicked,
    EmailOpened,
    ManualStop,
    MaxStepsReached,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceEnrollment {
    pub id: Uuid,
    pub sequence_id: Uuid,
    pub recipient_email: String,
    pub current_step: u32,
    pub status: EnrollmentStatus,
    pub next_send_at: Option<DateTime<Utc>>,
    pub enrolled_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnrollmentStatus {
    Active,
    Paused,
    Completed,
    Stopped,
    Bounced,
}

pub struct SequenceService {
    pool: PgPool,
}

impl SequenceService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn create(&self, user_id: Uuid, sequence: EmailSequence) -> Result<EmailSequence> {
        sqlx::query!(
            r#"
            INSERT INTO email_sequences (id, user_id, name, description, steps, stop_conditions, active, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
            "#,
            sequence.id,
            user_id,
            sequence.name,
            sequence.description,
            serde_json::to_value(&sequence.steps)?,
            serde_json::to_value(&sequence.stop_conditions)?,
            sequence.active
        )
        .execute(&self.pool)
        .await?;

        Ok(sequence)
    }

    pub async fn enroll(&self, sequence_id: Uuid, email: &str) -> Result<SequenceEnrollment> {
        let sequence = self.get(sequence_id).await?;

        let first_delay = sequence.steps.first()
            .map(|s| self.calculate_delay(&s.delay))
            .unwrap_or_else(|| Duration::hours(1));

        let enrollment = SequenceEnrollment {
            id: Uuid::new_v4(),
            sequence_id,
            recipient_email: email.to_string(),
            current_step: 0,
            status: EnrollmentStatus::Active,
            next_send_at: Some(Utc::now() + first_delay),
            enrolled_at: Utc::now(),
            completed_at: None,
        };

        sqlx::query!(
            r#"
            INSERT INTO sequence_enrollments (id, sequence_id, recipient_email, current_step, status, next_send_at, enrolled_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            "#,
            enrollment.id,
            sequence_id,
            email,
            0i32,
            serde_json::to_string(&enrollment.status)?,
            enrollment.next_send_at,
            enrollment.enrolled_at
        )
        .execute(&self.pool)
        .await?;

        Ok(enrollment)
    }

    pub async fn get(&self, sequence_id: Uuid) -> Result<EmailSequence> {
        let row = sqlx::query!(
            "SELECT * FROM email_sequences WHERE id = $1",
            sequence_id
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(EmailSequence {
            id: row.id,
            user_id: row.user_id,
            name: row.name,
            description: row.description,
            steps: serde_json::from_value(row.steps)?,
            stop_conditions: serde_json::from_value(row.stop_conditions)?,
            active: row.active,
            created_at: row.created_at,
        })
    }

    pub async fn process_due_enrollments(&self) -> Result<Vec<Uuid>> {
        let due = sqlx::query!(
            r#"
            SELECT e.id, e.sequence_id, e.recipient_email, e.current_step
            FROM sequence_enrollments e
            JOIN email_sequences s ON e.sequence_id = s.id
            WHERE e.status = 'Active'
              AND e.next_send_at <= NOW()
              AND s.active = true
            LIMIT 100
            "#
        )
        .fetch_all(&self.pool)
        .await?;

        let mut processed = Vec::new();

        for enrollment in due {
            let sequence = self.get(enrollment.sequence_id).await?;
            let step_idx = enrollment.current_step as usize;

            if step_idx < sequence.steps.len() {
                let step = &sequence.steps[step_idx];

                // Send email
                self.send_sequence_email(
                    &enrollment.recipient_email,
                    &step.email,
                    &sequence
                ).await?;

                // Update enrollment
                let next_step = step_idx + 1;
                if next_step < sequence.steps.len() {
                    let next_delay = self.calculate_delay(&sequence.steps[next_step].delay);
                    sqlx::query!(
                        r#"
                        UPDATE sequence_enrollments
                        SET current_step = $1, next_send_at = $2
                        WHERE id = $3
                        "#,
                        next_step as i32,
                        Utc::now() + next_delay,
                        enrollment.id
                    )
                    .execute(&self.pool)
                    .await?;
                } else {
                    // Sequence completed
                    sqlx::query!(
                        r#"
                        UPDATE sequence_enrollments
                        SET status = 'Completed', completed_at = NOW(), next_send_at = NULL
                        WHERE id = $1
                        "#,
                        enrollment.id
                    )
                    .execute(&self.pool)
                    .await?;
                }

                processed.push(enrollment.id);
            }
        }

        Ok(processed)
    }

    fn calculate_delay(&self, duration: &DelayDuration) -> Duration {
        match duration {
            DelayDuration::Minutes(m) => Duration::minutes(*m as i64),
            DelayDuration::Hours(h) => Duration::hours(*h as i64),
            DelayDuration::Days(d) => Duration::days(*d as i64),
            DelayDuration::Until { .. } => Duration::hours(24),  // Placeholder
        }
    }

    async fn send_sequence_email(&self, to: &str, email: &SequenceEmail, _sequence: &EmailSequence) -> Result<()> {
        // Would integrate with email sending service
        Ok(())
    }
}

// ============================================================================
// Approval Workflows
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalWorkflow {
    pub id: Uuid,
    pub user_id: Uuid,
    pub name: String,
    pub conditions: Vec<TriggerFilter>,
    pub approvers: Vec<Uuid>,
    pub require_all: bool,
    pub timeout_hours: Option<u32>,
    pub timeout_action: ApprovalTimeoutAction,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApprovalTimeoutAction {
    Approve,
    Reject,
    Escalate { to: Uuid },
    Notify,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequest {
    pub id: Uuid,
    pub workflow_id: Uuid,
    pub email_id: Uuid,
    pub status: ApprovalStatus,
    pub approvals: Vec<ApprovalVote>,
    pub created_at: DateTime<Utc>,
    pub resolved_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalVote {
    pub approver_id: Uuid,
    pub decision: ApprovalDecision,
    pub comment: Option<String>,
    pub voted_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApprovalStatus {
    Pending,
    Approved,
    Rejected,
    Expired,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApprovalDecision {
    Approve,
    Reject,
    Abstain,
}

pub struct ApprovalService {
    pool: PgPool,
}

impl ApprovalService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn create_request(&self, workflow_id: Uuid, email_id: Uuid) -> Result<ApprovalRequest> {
        let request = ApprovalRequest {
            id: Uuid::new_v4(),
            workflow_id,
            email_id,
            status: ApprovalStatus::Pending,
            approvals: Vec::new(),
            created_at: Utc::now(),
            resolved_at: None,
        };

        sqlx::query!(
            r#"
            INSERT INTO approval_requests (id, workflow_id, email_id, status, approvals, created_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
            "#,
            request.id,
            workflow_id,
            email_id,
            serde_json::to_string(&request.status)?,
            serde_json::to_value(&request.approvals)?
        )
        .execute(&self.pool)
        .await?;

        // Notify approvers
        let workflow = sqlx::query!(
            "SELECT approvers FROM approval_workflows WHERE id = $1",
            workflow_id
        )
        .fetch_one(&self.pool)
        .await?;

        let approvers: Vec<Uuid> = serde_json::from_value(workflow.approvers)?;
        for approver_id in approvers {
            self.notify_approver(approver_id, &request).await?;
        }

        Ok(request)
    }

    pub async fn vote(&self, request_id: Uuid, approver_id: Uuid, decision: ApprovalDecision, comment: Option<String>) -> Result<ApprovalRequest> {
        let vote = ApprovalVote {
            approver_id,
            decision,
            comment,
            voted_at: Utc::now(),
        };

        sqlx::query!(
            r#"
            UPDATE approval_requests
            SET approvals = approvals || $1::jsonb
            WHERE id = $2
            "#,
            serde_json::to_value(&vote)?,
            request_id
        )
        .execute(&self.pool)
        .await?;

        // Check if request should be resolved
        self.check_and_resolve(request_id).await
    }

    async fn check_and_resolve(&self, request_id: Uuid) -> Result<ApprovalRequest> {
        let request = sqlx::query!(
            "SELECT * FROM approval_requests WHERE id = $1",
            request_id
        )
        .fetch_one(&self.pool)
        .await?;

        let workflow = sqlx::query!(
            "SELECT * FROM approval_workflows WHERE id = $1",
            request.workflow_id
        )
        .fetch_one(&self.pool)
        .await?;

        let approvals: Vec<ApprovalVote> = serde_json::from_value(request.approvals)?;
        let approvers: Vec<Uuid> = serde_json::from_value(workflow.approvers)?;

        let approved_count = approvals.iter()
            .filter(|v| matches!(v.decision, ApprovalDecision::Approve))
            .count();
        let rejected_count = approvals.iter()
            .filter(|v| matches!(v.decision, ApprovalDecision::Reject))
            .count();

        let new_status = if workflow.require_all {
            if approved_count == approvers.len() {
                Some(ApprovalStatus::Approved)
            } else if rejected_count > 0 {
                Some(ApprovalStatus::Rejected)
            } else {
                None
            }
        } else {
            if approved_count > approvers.len() / 2 {
                Some(ApprovalStatus::Approved)
            } else if rejected_count > approvers.len() / 2 {
                Some(ApprovalStatus::Rejected)
            } else {
                None
            }
        };

        if let Some(status) = new_status {
            sqlx::query!(
                "UPDATE approval_requests SET status = $1, resolved_at = NOW() WHERE id = $2",
                serde_json::to_string(&status)?,
                request_id
            )
            .execute(&self.pool)
            .await?;

            // Execute post-approval action
            if matches!(status, ApprovalStatus::Approved) {
                sqlx::query!(
                    "UPDATE emails SET folder = 'Sent' WHERE id = $1",
                    request.email_id
                )
                .execute(&self.pool)
                .await?;
            }
        }

        // Return updated request
        Ok(ApprovalRequest {
            id: request.id,
            workflow_id: request.workflow_id,
            email_id: request.email_id,
            status: new_status.unwrap_or(ApprovalStatus::Pending),
            approvals,
            created_at: request.created_at,
            resolved_at: request.resolved_at,
        })
    }

    async fn notify_approver(&self, approver_id: Uuid, request: &ApprovalRequest) -> Result<()> {
        sqlx::query!(
            r#"
            INSERT INTO notifications (id, user_id, type, title, body, data, created_at)
            VALUES ($1, $2, 'approval_request', 'Approval Required', 'An email requires your approval', $3, NOW())
            "#,
            Uuid::new_v4(),
            approver_id,
            serde_json::json!({
                "request_id": request.id,
                "email_id": request.email_id,
            })
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }
}
